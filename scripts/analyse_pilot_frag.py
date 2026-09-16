"""Experiment E7 of docs/adaptations-proposal.md: can frag be computed from a pilot?

The paper's predictor ``frag(sigma_e)`` (Equation eq:frag; ``selection_fragility``
in ``boba_benchmarks.py``) is the expected loss from ONE greedy pick among
m = 256 scrambled-Sobol candidates when every candidate is observed with
N(0, sigma_e^2) error, computed on the exact standardised objective::

    frag(sigma_e) = E_eps[ max_i g_i - g_{argmax_i (g_i + eps_i)} ]

A practitioner does not have ``g``. This script asks whether a surrogate fitted
to a short clean pilot is good enough: for every landscape of the main sweep and
every seed it takes the clean ``baseline_exact`` run of the ``logei``
acquisition, fits the study's own GP (BoTorch ``SingleTaskGP``, inputs
normalised to the unit cube from the landscape box, outputs standardised by
BoTorch's ``Standardize`` and mapped back, ``ExactMarginalLogLikelihood`` fitted
by ``fit_gpytorch_mll``) to its first k points, and evaluates Equation eq:frag
with the GP posterior MEAN in place of ``g`` on the SAME 256-point Sobol design,
with the SAME 4000 N(0, sigma_e^2) draws (common random numbers: the generator
is seeded exactly as in ``selection_fragility`` and the six noise levels are
drawn in the same order, so ``frag_k`` and the exact ``frag`` differ only
through the surrogate).

Scale. The runs record ``objective_true`` = ``SyntheticOracle.predict`` =
``(f(x) - mean) / std`` with the ``mean``/``std`` of ``boba_landscape_stats.json``,
which is exactly the ``g`` of Equation eq:frag; the clean runs have
``objective_observed == objective_true``. Both facts are re-checked on every
run that is loaded (``max |g(x) - objective_true|`` is printed) rather than
assumed.

Outputs (in ``<output-dir>/analysis``):

* ``pilot_frag.csv`` -- one row per landscape x sigma_e: the exact frag, the
  measured cost, and the seed mean / SD of ``frag_k`` for each k.
* ``pilot_frag_per_seed.csv`` -- every (landscape, seed, k, sigma_e) value,
  with a per-fit diagnostic (Spearman correlation of the GP mean with the
  exact objective over the 256 candidates).
* ``pilot_frag_summary.csv`` -- per k and sigma_e, the Spearman and Pearson
  correlation across landscapes of ``frag_k`` (seed mean) with (a) the exact
  frag and (b) the measured cost, plus the exact frag's own correlation with
  the cost as the ceiling. ``sigma_e = pooled`` rows use every landscape x
  magnitude cell at once.

The measured cost is the landscape x magnitude mean of ``fragility``
(``excess_sd / opt_z``, the fraction of the achievable gain destroyed) at the
early onset (``jitter_iteration == 0``) in ``analysis/cell_means.csv``,
averaged over the model-based acquisitions (``random`` and ``sobol`` excluded)
and the four error processes.

Runtime: 20 landscapes x 10 seeds x 3 values of k = 600 GP fits on at most 50
points, about three to four minutes on this machine with the full 4000 draws.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, qmc, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import boba_benchmarks as bb  # noqa: E402

from botorch.exceptions.warnings import OptimizationWarning  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.transforms import Normalize, Standardize  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

# ``selection_fragility`` draws its noise for these six levels, in this order,
# from ONE generator seeded with STATS_SEED + 11. Reproducing the draws for the
# four reported levels therefore requires drawing all six in the same order.
NOISE_LEVELS_ALL: tuple[float, ...] = (0.05, 0.25, 0.5, 1.0, 2.0, 5.0)
REPORT_LEVELS: tuple[float, ...] = (0.05, 0.25, 1.0, 5.0)
K_VALUES: tuple[int, ...] = (10, 20, 50)
SEEDS: tuple[int, ...] = tuple(range(7, 17))
MODEL_FREE: tuple[str, ...] = ("random", "sobol")
DEFAULT_ACQUISITION = "logei"
N_CANDIDATES = 256
N_DRAWS = 4000


# ---------------------------------------------------------------------------
# Equation eq:frag, factored so that the objective can be swapped for a GP mean
# ---------------------------------------------------------------------------


def frag_candidates(name: str, n_candidates: int = N_CANDIDATES, seed: int = bb.STATS_SEED) -> np.ndarray:
    """The 256-point scrambled-Sobol design ``selection_fragility`` scores on."""
    spec = bb.BENCHMARKS[name]
    engine = qmc.Sobol(d=spec.dim, scramble=True, seed=seed + 7)
    return spec.lo + engine.random(n_candidates) * (spec.hi - spec.lo)


def frag_from_values(
    g: np.ndarray,
    noise_levels: tuple[float, ...] = NOISE_LEVELS_ALL,
    n_draws: int = N_DRAWS,
    seed: int = bb.STATS_SEED,
) -> dict[float, float]:
    """Equation eq:frag on an arbitrary value vector ``g`` over the candidates.

    A line-for-line replica of the loop in ``boba_benchmarks.selection_fragility``
    (same generator seed, same draw shapes, same level order), so that with the
    exact standardised objective it returns the ``frag_*`` fields of
    ``boba_landscape_stats.json`` bit for bit, and with a GP posterior mean it
    returns the pilot estimate under common random numbers.
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    best = float(np.max(g))
    rng = np.random.default_rng(seed + 11)
    out: dict[float, float] = {}
    for c in noise_levels:
        if c <= 0:
            out[c] = 0.0
            continue
        noise = rng.normal(0.0, c, size=(n_draws, g.size))
        picked = np.argmax(g[None, :] + noise, axis=1)
        out[c] = float(np.mean(best - g[picked]))
    return out


def exact_frag(name: str, stats: dict[str, dict[str, float]], **kwargs) -> dict[float, float]:
    """The exact frag, from ``selection_fragility`` with the study's scaling."""
    entry = stats[name]
    bb.landscape_stats.__dict__.setdefault("_scaling", {})[name] = (entry["mean"], entry["std"])
    raw = bb.selection_fragility(name, **kwargs)
    return {c: raw[f"frag_{c:g}"] for c in kwargs.get("noise_levels", NOISE_LEVELS_ALL)}


# ---------------------------------------------------------------------------
# The pilot surrogate: the sweep's own GP configuration
# ---------------------------------------------------------------------------


def fit_gp(X: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, seed: int = 0) -> SingleTaskGP:
    """``SingleTaskGP`` + ``Normalize(bounds)`` + ``Standardize`` fitted by MLL.

    Identical to the model ``bo_sensor_error_simulation.run_simulation`` fits at
    every iteration of the sweep (inferred homoscedastic noise, default
    Matern-5/2 ARD kernel and priors).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    train_X = torch.tensor(X, dtype=torch.double)
    train_Y = torch.tensor(y, dtype=torch.double)
    bounds = torch.tensor(np.vstack([np.asarray(lo, float), np.asarray(hi, float)]), dtype=torch.double)
    torch.manual_seed(seed)
    gp = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=Normalize(d=X.shape[1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    return gp


def posterior_mean(gp: SingleTaskGP, X: np.ndarray) -> np.ndarray:
    """Posterior mean on the recorded (standardised-landscape) scale."""
    with torch.no_grad():
        mu = gp.posterior(torch.tensor(np.asarray(X, dtype=float), dtype=torch.double)).mean
    return mu.reshape(-1).cpu().numpy()


def pilot_frag(
    X_pilot: np.ndarray,
    y_pilot: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    X_cand: np.ndarray,
    noise_levels: tuple[float, ...] = NOISE_LEVELS_ALL,
    n_draws: int = N_DRAWS,
    seed: int = bb.STATS_SEED,
    fit_seed: int = 0,
) -> tuple[dict[float, float], np.ndarray, int]:
    """frag with the GP mean of a pilot in place of the exact objective.

    Returns ``(frag_k by level, GP mean on the candidates, number of optimiser
    warnings raised during the fit)``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", OptimizationWarning)
        gp = fit_gp(X_pilot, y_pilot, lo, hi, seed=fit_seed)
    n_warn = sum(issubclass(w.category, OptimizationWarning) for w in caught)
    mu = posterior_mean(gp, X_cand)
    return frag_from_values(mu, noise_levels=noise_levels, n_draws=n_draws, seed=seed), mu, n_warn


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def clean_run_path(output_dir: Path, name: str, acquisition: str, seed: int) -> Path:
    return output_dir / name / f"bo_sensor_error_{name}_value_{acquisition}_seed{seed}_baseline_exact.csv"


def load_clean_run(path: Path, name: str, stats: dict[str, dict[str, float]]) -> tuple[np.ndarray, np.ndarray, float]:
    """The visited designs and recorded objective of one clean run, in order.

    Checks, rather than assumes, that the run is clean (observed == true) and
    that the recorded objective is the standardised exact objective. Returns
    ``(X, y_observed, max |g(x) - objective_true|)``.
    """
    df = pd.read_csv(path).sort_values("iteration")
    spec = bb.BENCHMARKS[name]
    X = df[spec.param_columns].to_numpy(dtype=float)
    y_obs = df["objective_observed"].to_numpy(dtype=float)
    y_true = df["objective_true"].to_numpy(dtype=float)
    if not np.allclose(y_obs, y_true, rtol=0, atol=1e-12):
        raise ValueError(f"{path} is not a clean run: objective_observed != objective_true")
    entry = stats[name]
    g = (bb.evaluate(name, X) - entry["mean"]) / entry["std"]
    scale_gap = float(np.max(np.abs(g - y_true)))
    if scale_gap > 1e-6:
        raise ValueError(
            f"{path}: recorded objective is not the standardised exact objective "
            f"(max gap {scale_gap:.3g})"
        )
    return X, y_obs, scale_gap


def measured_cost(cell_means: pd.DataFrame) -> pd.DataFrame:
    """Landscape x magnitude mean of the early-onset fraction destroyed.

    Averages ``fragility`` over the four error processes and the model-based
    acquisitions (``random`` and ``sobol`` excluded) at ``jitter_iteration == 0``.
    """
    early = cell_means[(cell_means["jitter_iteration"] == 0) & (~cell_means["acquisition"].isin(MODEL_FREE))]
    cost = (
        early.groupby(["dataset", "jitter_std"])
        .agg(cost_measured=("fragility", "mean"),
             cost_cells=("fragility", "size"),
             cost_acquisitions=("acquisition", "nunique"),
             cost_error_models=("error_model", "nunique"))
        .reset_index()
        .rename(columns={"jitter_std": "sigma_e"})
    )
    return cost


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------


def _corr(x: pd.Series, y: pd.Series) -> dict[str, float]:
    mask = x.notna() & y.notna()
    x, y = x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)
    out = {"n": int(mask.sum()), "spearman": np.nan, "spearman_p": np.nan, "pearson": np.nan, "pearson_p": np.nan}
    if out["n"] < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rs, ps = spearmanr(x, y)
        rp, pp = pearsonr(x, y)
    out.update(spearman=float(rs), spearman_p=float(ps), pearson=float(rp), pearson_p=float(pp))
    return out


def correlation_summary(table: pd.DataFrame, k_values: tuple[int, ...]) -> pd.DataFrame:
    """Per k and sigma_e: correlations of frag_k (and the exact frag) with the targets."""
    predictors = [(f"{k}", f"frag_k{k}_mean") for k in k_values] + [("exact", "frag_exact")]
    targets = [("frag_exact", "frag_exact"), ("cost_measured", "cost_measured")]
    scopes = [(f"{c:g}", table[table["sigma_e"] == c]) for c in sorted(table["sigma_e"].unique())]
    scopes.append(("pooled", table))
    rows = []
    for k_label, pred in predictors:
        for sigma_label, block in scopes:
            for target_label, target in targets:
                stats = _corr(block[pred], block[target])
                row = {"k": k_label, "sigma_e": sigma_label, "target": target_label, **stats}
                if target == "frag_exact":
                    diff = (block[pred] - block[target]).dropna()
                    row["bias"] = float(diff.mean()) if len(diff) else np.nan
                    row["mae"] = float(diff.abs().mean()) if len(diff) else np.nan
                else:
                    row["bias"] = np.nan
                    row["mae"] = np.nan
                rows.append(row)
    cols = ["k", "sigma_e", "target", "n", "spearman", "spearman_p", "pearson", "pearson_p", "bias", "mae"]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    output_dir: Path,
    analysis_dir: Path,
    stats_path: Path,
    landscapes: list[str] | None,
    seeds: tuple[int, ...],
    k_values: tuple[int, ...],
    acquisition: str,
    n_draws: int,
    quiet: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    stats = bb.load_stats(stats_path)
    cell_means = pd.read_csv(analysis_dir / "cell_means.csv")
    cost = measured_cost(cell_means)
    if landscapes is None:
        landscapes = [n for n in bb.DEFAULT_SUITE if n in set(cell_means["dataset"])]
    missing = [n for n in landscapes if n not in stats]
    if missing:
        raise KeyError(f"No landscape statistics for {missing}")
    k_values = tuple(sorted(k_values))
    largest_k = max(k_values)

    per_seed_rows: list[dict] = []
    exact_rows: list[dict] = []
    scale_gap_max = 0.0
    replica_gap_max = 0.0
    json_gap_max = 0.0
    n_fits = 0
    n_warn_total = 0
    for name in landscapes:
        spec = bb.BENCHMARKS[name]
        entry = stats[name]
        X_cand = frag_candidates(name)
        g_exact = (bb.evaluate(name, X_cand) - entry["mean"]) / entry["std"]
        frag_exact_dict = exact_frag(name, stats, n_draws=n_draws) if n_draws == N_DRAWS else \
            frag_from_values(g_exact, n_draws=n_draws)
        # The replica must reproduce selection_fragility exactly, and both must
        # match the stored JSON at the full draw count.
        replica = frag_from_values(g_exact, n_draws=n_draws)
        replica_gap_max = max(replica_gap_max, max(abs(replica[c] - frag_exact_dict[c]) for c in NOISE_LEVELS_ALL))
        if n_draws == N_DRAWS:
            json_gap_max = max(json_gap_max, max(abs(entry[f"frag_{c:g}"] - frag_exact_dict[c]) for c in NOISE_LEVELS_ALL))
        for c in NOISE_LEVELS_ALL:
            exact_rows.append({"dataset": name, "dim": spec.dim, "opt_z": entry["opt_z"],
                               "sigma_e": c, "frag_exact": frag_exact_dict[c]})

        for seed in seeds:
            path = clean_run_path(output_dir, name, acquisition, seed)
            if not path.exists():
                warnings.warn(f"missing clean run {path}; skipped", stacklevel=1)
                continue
            X, y, gap = load_clean_run(path, name, stats)
            scale_gap_max = max(scale_gap_max, gap)
            if X.shape[0] < largest_k:
                warnings.warn(f"{path} has {X.shape[0]} < {largest_k} rows", stacklevel=1)
            for k in k_values:
                if X.shape[0] < k:
                    continue
                t_fit = time.perf_counter()
                frag_k, mu, n_warn = pilot_frag(
                    X[:k], y[:k], spec.bounds_low, spec.bounds_high, X_cand,
                    n_draws=n_draws, fit_seed=seed * 1000 + k,
                )
                fit_seconds = time.perf_counter() - t_fit
                n_fits += 1
                n_warn_total += n_warn
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean_rank_corr = float(spearmanr(mu, g_exact)[0]) if np.ptp(mu) > 0 else np.nan
                for c in NOISE_LEVELS_ALL:
                    per_seed_rows.append({
                        "dataset": name, "dim": spec.dim, "seed": seed, "k": k, "sigma_e": c,
                        "frag_k": frag_k[c], "frag_exact": frag_exact_dict[c],
                        "gp_mean_rank_corr": mean_rank_corr, "gp_mean_max": float(np.max(mu)),
                        "exact_max": float(np.max(g_exact)), "fit_warnings": n_warn,
                        "fit_seconds": fit_seconds,
                    })
        if not quiet:
            done = len({r["dataset"] for r in per_seed_rows})
            print(f"  [{done:2d}/{len(landscapes)}] {name:<20s} {n_fits:4d} fits  "
                  f"{time.perf_counter() - t0:6.1f}s", flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    exact = pd.DataFrame(exact_rows)

    # Per-landscape table: seed mean / SD of frag_k per k, the exact frag, the cost.
    agg = (
        per_seed.groupby(["dataset", "k", "sigma_e"])["frag_k"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    wide = agg.pivot(index=["dataset", "sigma_e"], columns="k", values=["mean", "std", "count"])
    wide.columns = [f"frag_k{k}_{'mean' if s == 'mean' else 'sd' if s == 'std' else 'n'}" for s, k in wide.columns]
    wide = wide.reset_index()
    n_cols = [c for c in wide.columns if c.endswith("_n")]
    wide["n_seeds"] = wide[n_cols].min(axis=1).astype(int)
    wide = wide.drop(columns=n_cols)
    table = exact.merge(wide, on=["dataset", "sigma_e"], how="left")
    table = table.merge(cost, on=["dataset", "sigma_e"], how="left")
    table = table[table["sigma_e"].isin(REPORT_LEVELS)].copy()
    order = {n: i for i, n in enumerate(landscapes)}
    table = table.sort_values(["sigma_e", "dataset"], key=lambda s: s.map(order) if s.name == "dataset" else s)
    table = table.reset_index(drop=True)

    summary = correlation_summary(table, k_values)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(analysis_dir / "pilot_frag.csv", index=False)
    per_seed.to_csv(analysis_dir / "pilot_frag_per_seed.csv", index=False)
    summary.to_csv(analysis_dir / "pilot_frag_summary.csv", index=False)

    if not quiet:
        print()
        print("Scale check: the recorded objective is the standardised exact objective, "
              f"max |g(x) - objective_true| over all loaded runs = {scale_gap_max:.2e}; "
              "clean runs have objective_observed == objective_true.")
        print(f"Replica check: frag_from_values vs selection_fragility, max gap = {replica_gap_max:.2e}"
              + (f"; vs boba_landscape_stats.json, max gap = {json_gap_max:.2e}" if n_draws == N_DRAWS else
                 f" (n_draws = {n_draws}, not compared with the JSON's 4000-draw values)"))
        cost_cells = cost.loc[cost["dataset"].isin(landscapes)]
        print(f"Measured cost: {len(cost_cells)} landscape x magnitude cells, each the mean of "
              f"{int(cost_cells['cost_cells'].min())}-{int(cost_cells['cost_cells'].max())} cells "
              f"({int(cost_cells['cost_acquisitions'].min())} model-based acquisitions x "
              f"{int(cost_cells['cost_error_models'].min())} error processes), early onset only.")
        print(f"GP fits: {n_fits} ({n_warn_total} optimiser warnings), acquisition = {acquisition}, "
              f"seeds = {list(seeds)}, k = {list(k_values)}, n_draws = {n_draws}, "
              f"elapsed {time.perf_counter() - t0:.1f}s")
        print()
        print("Per-landscape table (analysis/pilot_frag.csv):")
        show_cols = ["dataset", "dim", "sigma_e", "frag_exact", "cost_measured"] + \
            [c for k in k_values for c in (f"frag_k{k}_mean", f"frag_k{k}_sd")] + ["n_seeds"]
        with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_columns", 30):
            print(table[show_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()
        print("Correlation summary across landscapes (analysis/pilot_frag_summary.csv):")
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table, per_seed, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output-boba",
                        help="sweep output root holding <landscape>/ run CSVs (default: output-boba)")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="where cell_means.csv lives and the results go (default: <output-dir>/analysis)")
    parser.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    parser.add_argument("--landscapes", type=str, default=None,
                        help="comma-separated subset (default: every DEFAULT_SUITE landscape in cell_means.csv)")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--k", type=str, default=",".join(str(k) for k in K_VALUES),
                        help="pilot sizes, comma-separated (default: 10,20,50)")
    parser.add_argument("--acquisition", type=str, default=DEFAULT_ACQUISITION)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS,
                        help="Monte Carlo draws per noise level (default: 4000, as in selection_fragility)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    analysis_dir = args.analysis_dir if args.analysis_dir is not None else args.output_dir / "analysis"
    landscapes = [s.strip() for s in args.landscapes.split(",")] if args.landscapes else None
    run(
        output_dir=args.output_dir,
        analysis_dir=analysis_dir,
        stats_path=args.stats_path,
        landscapes=landscapes,
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        k_values=tuple(int(k) for k in args.k.split(",")),
        acquisition=args.acquisition,
        n_draws=args.n_draws,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
