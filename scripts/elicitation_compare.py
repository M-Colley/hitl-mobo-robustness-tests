"""Does asking for a COMPARISON instead of a RATING buy immunity to rater faults?

The faults this project measures fall into two classes, and the split is exact
rather than a matter of degree.

A fault is SHARED within a sitting when it acts on every design the rater judges
at that moment by the same monotone map g: a constant bias, a slow drift, a
per-rater offset, a saturating scale. A comparison only asks which of two designs
is better, so for strictly increasing g,

    g(f(a)) > g(f(b))   <=>   f(a) > f(b)

and the comparison is EXACTLY what it would have been with no fault at all. A
rating is not: g(f(a)) is not f(a). A saturating scale is the interesting edge,
being monotone but not strictly so -- two designs above the cap map to the same
value, and the comparison degrades to a coin flip rather than staying exact.

A fault is IDIOSYNCRATIC when it is drawn afresh for each judgement: ordinary
rating noise, a slip of attention, a gross fault. Nothing cancels; the comparison
inherits the noise of both designs, so its effective noise is sqrt(2) times the
rating's. On these faults a comparison should do no better than a rating, and on
a per-judgement budget it should do slightly worse.

The paper currently cites this trade-off from the preference-learning literature
without measuring it. This script measures it: two loops with the SAME number of
human judgements, on the same landscapes, seeds and faults.

    rating    a SingleTaskGP on noisy ratings, LogEI, the standard process.
    pairwise  a PairwiseGP (Laplace-approximate, BoTorch) on comparisons, LogEI on
              its latent posterior. Each trial proposes one design and asks the
              rater to compare it with the incumbent, so one judgement buys one
              comparison, exactly as one judgement buys one rating.

Budget parity. The rating loop rates n0 random designs, then rates one new design
per trial: T judgements, T designs. The pairwise loop compares n0+1 random
designs in a chain (n0 judgements), then one new design against the incumbent per
trial: T judgements, T+1 designs. One extra design out of fifty, and no extra
human effort, which is the budget that matters.

Ship rules, since a comparison loop has no rating to rank by:
    rating    the best observed rating (the standard process)
    pairwise  the design with the highest latent posterior mean

Both are scored by the true objective at the design they ship, against the
identically seeded CLEAN run of the same loop, so each elicitation is its own
control and the comparison is of excess, never of levels. That own-twin share
(removed_share) is what this arm has always reported. The comparison loop also
changes the clean run, so the summary carries the standard-process estimand of
AGENTS.md beside it (analyse_boba_adaptations.summarise): cost = rating noisy -
rating clean, gain = rating noisy - comparison noisy, price = comparison clean -
rating clean, each a landscape mean in units of opt_z with a landscape bootstrap.

Determinism. Each run seeds the global numpy and torch generators from its task
seed (_seed_globals). BoTorch's PairwiseGP perturbs the start of every Laplace
MAP search with numpy's global generator, and a failed hyperparameter fit
retries from values drawn with torch's; before the seed was set the global numpy
generator was seeded from the operating system, so the comparison loop gave a
different result for the same task on every execution. The log in
output-elicitation predates the seed and cannot be regenerated;
output-elicitation-rerun is the deterministic arm.

Ties. The rating loop ships the EARLIEST design with the best observed rating
(np.argmax), the standard rule of the paper. Under the noise-free cap many
designs tie at the cap, so each run also records the exact expectation over a
uniformly random choice among the tied designs (shipped_true_uniform_ties), and
the summary gives both. A comparison between two looks that tie (both above the
cap) goes to the first design of the pair, the new proposal in the loop.

``--functions`` takes a comma-separated list, ``suite`` for the twenty
landscapes of the paper (boba_benchmarks.DEFAULT_SUITE), or the default ``all``
for every landscape in the stats file, which also holds the manipulation
families. The paper's arm, with the twenty named explicitly so that a reader of
the command (and scripts/check_provenance.py) sees which landscapes it covers:

    python scripts/elicitation_compare.py --functions ackley,branin,eggholder,griewank,hartmann_3,hartmann_6,hicks_law,levy_10,michalewicz,moving_peaks,powell,power_law_practice,rastrigin,rosenbrock,schwefel,shekel,steering_law,stevens,weber_fechner,yerkes_dodson --error-models bias,drift,ceiling,gaussian --magnitudes 1 --iterations 20 --workers 12 --output-dir output-elicitation-rerun
    python scripts/elicitation_compare.py --summary-only --output-dir output-elicitation-rerun
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260916
OUTPUT_NAME = "elicitation"

# Faults that act on every design of one sitting through the same monotone map.
SHARED_FAULTS = ("bias", "drift", "ceiling")
# Faults drawn afresh for each judgement.
IDIOSYNCRATIC_FAULTS = ("gaussian", "spike")
ALL_FAULTS = SHARED_FAULTS + IDIOSYNCRATIC_FAULTS


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("output-elicitation"))
    p.add_argument("--functions", type=str, default="all",
                   help="'all' (every landscape in the stats file), 'suite' (the paper's twenty, "
                        "boba_benchmarks.DEFAULT_SUITE) or a comma-separated list")
    p.add_argument("--error-models", type=str, default=",".join(ALL_FAULTS))
    p.add_argument("--magnitudes", type=str, default="1,5")
    p.add_argument("--seeds", type=str, default="7,8,9")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--initial-samples", type=int, default=5)
    p.add_argument("--candidate-pool", type=int, default=512)
    p.add_argument("--spike-prob", type=float, default=0.15)
    p.add_argument("--spike-sd", type=float, default=5.0)
    p.add_argument("--ceiling-quantile", type=float, default=0.9)
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# The rater
# ---------------------------------------------------------------------------


def shared_offset(error_model: str, magnitude: float, iteration: int, T: int,
                  rng: np.random.Generator) -> float:
    """The part of the fault every design of one sitting receives alike."""
    if error_model == "bias":
        return magnitude
    if error_model == "drift":
        # A ramp that reaches `magnitude` at the final trial, as in the sweep.
        return magnitude * (iteration / max(1, T))
    return 0.0


def idiosyncratic_draw(error_model: str, magnitude: float, size: int, spike_prob: float,
                       spike_sd: float, rng: np.random.Generator) -> np.ndarray:
    """The part drawn afresh for every judgement."""
    if error_model == "gaussian":
        return rng.normal(0.0, magnitude, size=size)
    if error_model == "spike":
        hit = rng.random(size) < spike_prob
        return np.where(hit, rng.normal(0.0, magnitude * spike_sd, size=size), 0.0)
    return np.zeros(size)


def perceive(values: np.ndarray, error_model: str, magnitude: float, iteration: int, T: int,
             cap: float | None, spike_prob: float, spike_sd: float,
             rng: np.random.Generator) -> np.ndarray:
    """What the rater perceives for the designs shown together in one sitting.

    Shared parts are added once; idiosyncratic parts are drawn per design. A
    ceiling is applied last, because it acts on what the rater would have said.
    """
    out = np.asarray(values, dtype=float).copy()
    out = out + shared_offset(error_model, magnitude, iteration, T, rng)
    out = out + idiosyncratic_draw(error_model, magnitude, len(out), spike_prob, spike_sd, rng)
    if error_model == "ceiling" and cap is not None:
        out = np.minimum(out, cap)
    return out


# ---------------------------------------------------------------------------
# The two loops
# ---------------------------------------------------------------------------


def _values(oracle, X: np.ndarray) -> np.ndarray:
    """The exact objective at each row, always as a flat float array.

    predict_many returns a column for a single-objective oracle, and a column
    kept where a row is expected turns every scalar comparison below into an
    array comparison.
    """
    return np.asarray(oracle.predict_many(np.atleast_2d(X)), dtype=float).ravel()


def _fit_rating_gp(X: np.ndarray, y: np.ndarray, bounds: torch.Tensor):
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    train_X = torch.tensor(X, dtype=torch.double)
    train_Y = torch.tensor(y.reshape(-1, 1), dtype=torch.double)
    gp = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
                      outcome_transform=Standardize(m=1))
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp, False
    except Exception:  # noqa: BLE001 - counted per run, never silent
        return gp, True


def _fit_pairwise_gp(X: np.ndarray, comparisons: np.ndarray, bounds: torch.Tensor):
    """A PairwiseGP, with its prior hyperparameters if the Laplace fit will not converge.

    The Laplace approximation fails on a minority of runs -- typically when many
    comparisons agree and the latent scale runs away. Falling back to the
    unfitted model keeps that run in the design instead of deleting it, which
    would drop exactly the hardest runs and flatter the comparison loop. The
    fallback is counted and reported, never silent.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
    from botorch.models.transforms import Normalize
    train_X = torch.tensor(X, dtype=torch.double)
    comps = torch.tensor(comparisons, dtype=torch.long)
    gp = PairwiseGP(train_X, comps,
                    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds))
    try:
        fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(gp.likelihood, gp))
        return gp, False
    except Exception:  # noqa: BLE001 - counted per run, as the loop counts acq failures
        return gp, True


def _propose(gp, bounds: torch.Tensor, best_f: float, pool: np.ndarray) -> int:
    """Index into `pool` of the LogEI-best candidate.

    A fixed random pool rather than gradient optimisation: the two loops must
    differ in their observation model and nothing else, and a discrete pool is
    the one proposal mechanism that means the same thing for both surrogates.
    """
    from botorch.acquisition.analytic import LogExpectedImprovement
    acq = LogExpectedImprovement(model=gp, best_f=best_f)
    with torch.no_grad():
        values = acq(torch.tensor(pool, dtype=torch.double).unsqueeze(1))
    return int(torch.argmax(values).item())


TIE_TOL = 1e-12


def _ship(score: np.ndarray, truth: np.ndarray) -> dict:
    """What the loop ships from ``score``, under both tie conventions.

    shipped_true is the earliest design with the best score (np.argmax, the
    rule every number of this arm has used). shipped_true_uniform_ties is the
    exact expectation over a uniformly random choice among the designs whose
    score is within TIE_TOL of the best, and n_tied_top counts them. Computed
    after the loop, so it draws no random number and cannot change the run.
    """
    score = np.asarray(score, dtype=float)
    tied = np.flatnonzero(score >= score.max() - TIE_TOL)
    return {"shipped_true": float(truth[int(np.argmax(score))]),
            "shipped_true_uniform_ties": float(np.mean(truth[tied])),
            "n_tied_top": int(len(tied))}


def _seed_globals(seed: int) -> None:
    """Start the run's GLOBAL numpy and torch streams from its own seed.

    The loops draw their own numbers from a local generator, but the libraries
    draw from the global ones: every PairwiseGP starts its Laplace MAP search
    from the win counts plus 0.05 of a standard normal drawn with
    np.random.standard_normal (botorch.models.pairwise_gp), and a failed
    hyperparameter fit retries from values sampled from the priors with torch's
    generator (botorch.fit). Without this, numpy's global generator was seeded
    from the operating system in every worker, so the comparison loop gave a
    different result for the same task on every execution, and its clean twin
    differed between error-process cells. Seeding both at the start of every run
    makes each run a function of its task alone.
    """
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def run_rating(oracle, bounds_low, bounds_high, T: int, n0: int, pool_size: int,
               error_model: str, magnitude: float, apply_error: bool, seed: int,
               cap: float | None, spike_prob: float, spike_sd: float) -> dict:
    _seed_globals(seed)
    rng = np.random.default_rng(seed)
    fit_failures = 0
    n_fits = 0
    bounds = torch.tensor(np.stack([bounds_low, bounds_high]), dtype=torch.double)
    d = len(bounds_low)
    X = rng.uniform(bounds_low, bounds_high, size=(n0, d))
    truth = _values(oracle, X)
    observed = perceive(truth, error_model if apply_error else "none", magnitude, 0, T,
                        cap, spike_prob, spike_sd, rng)
    for t in range(n0, T):
        pool = rng.uniform(bounds_low, bounds_high, size=(pool_size, d))
        gp, failed = _fit_rating_gp(X, observed, bounds)
        fit_failures += int(failed)
        n_fits += 1
        idx = _propose(gp, bounds, float(observed.max()), pool)
        x_new = pool[idx : idx + 1]
        f_new = _values(oracle, x_new)
        y_new = perceive(f_new, error_model if apply_error else "none", magnitude, t, T,
                         cap, spike_prob, spike_sd, rng)
        X = np.vstack([X, x_new])
        truth = np.concatenate([truth, f_new])
        observed = np.concatenate([observed, y_new])
    return {**_ship(observed, truth), "best_visited_true": float(truth.max()),
            "n_designs": len(X), "fit_failures": fit_failures, "n_fits": n_fits}


def run_pairwise(oracle, bounds_low, bounds_high, T: int, n0: int, pool_size: int,
                 error_model: str, magnitude: float, apply_error: bool, seed: int,
                 cap: float | None, spike_prob: float, spike_sd: float) -> dict:
    _seed_globals(seed)
    rng = np.random.default_rng(seed)
    fit_failures = 0
    n_fits = 0
    bounds = torch.tensor(np.stack([bounds_low, bounds_high]), dtype=torch.double)
    d = len(bounds_low)
    model = error_model if apply_error else "none"

    # n0 judgements over n0+1 random designs, compared down a chain.
    X = rng.uniform(bounds_low, bounds_high, size=(n0 + 1, d))
    truth = _values(oracle, X)
    comparisons = []
    for t in range(n0):
        pair = (t, t + 1)
        looks = perceive(truth[list(pair)], model, magnitude, t, T, cap, spike_prob, spike_sd, rng)
        win, lose = (pair[0], pair[1]) if looks[0] >= looks[1] else (pair[1], pair[0])
        comparisons.append([win, lose])

    for t in range(n0, T):
        gp, failed = _fit_pairwise_gp(X, np.asarray(comparisons), bounds)
        fit_failures += int(failed)
        n_fits += 1
        with torch.no_grad():
            mean = gp.posterior(torch.tensor(X, dtype=torch.double)).mean.squeeze(-1).numpy()
        incumbent = int(np.argmax(mean))
        pool = rng.uniform(bounds_low, bounds_high, size=(pool_size, d))
        idx = _propose(gp, bounds, float(mean.max()), pool)
        x_new = pool[idx : idx + 1]
        f_new = _values(oracle, x_new)
        X = np.vstack([X, x_new])
        truth = np.concatenate([truth, f_new])
        new = len(X) - 1
        looks = perceive(truth[[new, incumbent]], model, magnitude, t, T, cap,
                         spike_prob, spike_sd, rng)
        win, lose = (new, incumbent) if looks[0] >= looks[1] else (incumbent, new)
        comparisons.append([win, lose])

    gp, failed = _fit_pairwise_gp(X, np.asarray(comparisons), bounds)
    fit_failures += int(failed)
    n_fits += 1
    with torch.no_grad():
        mean = gp.posterior(torch.tensor(X, dtype=torch.double)).mean.squeeze(-1).numpy()
    return {**_ship(mean, truth), "best_visited_true": float(truth.max()),
            "n_designs": len(X), "fit_failures": fit_failures, "n_fits": n_fits}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _init_worker() -> None:
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def run_cell(task: dict) -> dict:
    stats = bb.load_stats(Path(task["stats_path"]))
    oracle = bb.SyntheticOracle.from_stats(task["dataset"], stats)
    entry = stats[task["dataset"]]
    low = np.asarray(oracle.bounds_low, dtype=float)
    high = np.asarray(oracle.bounds_high, dtype=float)
    cap = None
    if task["error_model"] == "ceiling":
        # The cap sits at a quantile of the landscape the rater would see, so
        # "saturating" means the same thing on every benchmark.
        grid = np.random.default_rng(0).uniform(low, high, size=(4096, len(low)))
        cap = float(np.quantile(_values(oracle, grid), task["ceiling_quantile"]))
    fn = run_rating if task["elicitation"] == "rating" else run_pairwise
    t0 = time.time()
    out = fn(oracle, low, high, task["iterations"], task["initial_samples"], task["candidate_pool"],
             task["error_model"], task["magnitude"], task["apply_error"], task["seed"], cap,
             task["spike_prob"], task["spike_sd"])
    out.update({k: task[k] for k in ("dataset", "elicitation", "error_model", "magnitude",
                                     "seed", "apply_error")})
    out["y_opt"] = float(oracle.y_opt)
    out["seconds"] = time.time() - t0
    return out


def build_tasks(args: argparse.Namespace, names: list[str]) -> list[dict]:
    seeds = [int(s) for s in args.seeds.split(",")]
    mags = [float(m) for m in args.magnitudes.split(",")]
    models = [m.strip() for m in args.error_models.split(",")]
    tasks = []
    for dataset in names:
        for seed in seeds:
            for elicitation in ("rating", "pairwise"):
                for model in models:
                    for mag in mags:
                        for apply_error in (True, False):
                            tasks.append({
                                "dataset": dataset, "elicitation": elicitation,
                                "error_model": model, "magnitude": mag, "seed": seed,
                                "apply_error": apply_error,
                                "iterations": args.iterations,
                                "initial_samples": args.initial_samples,
                                "candidate_pool": args.candidate_pool,
                                "spike_prob": args.spike_prob, "spike_sd": args.spike_sd,
                                "ceiling_quantile": args.ceiling_quantile,
                                "stats_path": str(args.stats_path),
                            })
    return tasks


def _standard_process(regret: pd.DataFrame, column: str) -> dict:
    """The comparison loop scored against the STANDARD process (AGENTS.md).

    ``regret`` holds one row per (dataset, seed) with the rating loop's noisy and
    clean regret and the comparison loop's, all in units of opt_z. The estimand
    and its landscape bootstrap are analyse_boba_adaptations.summarise, imported
    rather than copied; the suppression rule for a near-zero or sign-changing
    cost is its share_flags.
    """
    import analyse_boba_adaptations as aba  # a heavy import, kept out of the workers
    block = pd.DataFrame({
        "dataset": regret["dataset"],
        "ref_noisy": regret[f"rating_noisy{column}"], "ref_clean": regret[f"rating_clean{column}"],
        "trt_noisy": regret["pairwise_noisy"], "trt_clean": regret["pairwise_clean"],
    })
    out = aba.summarise(block, np.random.default_rng(aba.BOOTSTRAP_SEED))
    flags = aba.share_flags(block)
    keep = ("cost", "gain", "price", "recovered", "recovered_lo", "recovered_hi", "wilcoxon_p")
    res = {f"std_{k}{column}": out[k] for k in keep}
    res[f"std_recovered_suppressed{column}"] = flags["recovered_suppressed"]
    return res


def summarise(runs: pd.DataFrame, opt_z: dict[str, float]) -> pd.DataFrame:
    """Excess regret of each elicitation over its OWN clean twin, and more.

    The first seven columns are the arm's original own-twin estimand, computed
    exactly as before (same bootstrap stream). The rest: the number of runs, how
    many noisy runs ship a different design than their clean twin, the fits and
    fallbacks, the same share under a uniformly random tie-break of the rating
    loop's ship rule (same landscape resamples), and the standard-process
    estimand (std_*) under both tie conventions.
    """
    runs = runs.copy()
    missing = sorted(set(runs["dataset"]) - set(opt_z))
    if missing:
        raise KeyError(f"no opt_z for {missing}; regrets from different landscapes would mix units")
    z = runs["dataset"].map(opt_z)
    runs["regret"] = (runs["y_opt"] - runs["shipped_true"]) / z
    uniform = "shipped_true_uniform_ties" in runs.columns
    if uniform:
        runs["regret_u"] = (runs["y_opt"] - runs["shipped_true_uniform_ties"]) / z
    keys = ["dataset", "elicitation", "error_model", "magnitude", "seed"]
    noisy_runs = runs[runs["apply_error"]].set_index(keys)
    clean_runs = runs[~runs["apply_error"]].set_index(keys)
    noisy = noisy_runs["regret"]
    clean = clean_runs["regret"]
    excess = (noisy - clean).rename("excess").reset_index()
    moved = ((noisy_runs["shipped_true"] - clean_runs["shipped_true"]).abs() > TIE_TOL).rename("moved")
    excess = excess.merge(moved.reset_index(), on=keys, how="left")
    if uniform:
        # The clean twin has no cap, so its uniform value is its first-index value
        # unless two designs tie exactly; both sides use the same convention.
        excess_u = (noisy_runs["regret_u"] - clean_runs["regret_u"]).rename("excess_u").reset_index()
        excess = excess.merge(excess_u, on=keys, how="left")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (model, mag), group in excess.groupby(["error_model", "magnitude"]):
        per = group.groupby(["dataset", "elicitation"])["excess"].mean().unstack("elicitation")
        if not {"rating", "pairwise"} <= set(per.columns):
            continue
        per = per.dropna()
        diff = (per["rating"] - per["pairwise"]).to_numpy()
        base = per["rating"].to_numpy()
        if uniform:
            per_u = (group.groupby(["dataset", "elicitation"])["excess_u"].mean()
                     .unstack("elicitation").loc[per.index])
            diff_u = (per_u["rating"] - per_u["pairwise"]).to_numpy()
            base_u = per_u["rating"].to_numpy()
        n = len(diff)
        draws, draws_u = [], []
        for _ in range(BOOTSTRAP_REPS):
            idx = rng.integers(0, n, n)
            b = base[idx].mean()
            if b != 0:
                draws.append(diff[idx].mean() / b)
            if uniform and base_u[idx].mean() != 0:
                draws_u.append(diff_u[idx].mean() / base_u[idx].mean())
        share = float(diff.mean() / base.mean()) if base.mean() != 0 else float("nan")
        row = {
            "error_model": model, "magnitude": mag, "class":
                "shared" if model in SHARED_FAULTS else "idiosyncratic",
            "n_landscapes": n,
            "rating_excess": float(per["rating"].mean()),
            "pairwise_excess": float(per["pairwise"].mean()),
            "removed_share": share,
            "removed_lo": float(np.percentile(draws, 2.5)) if draws else float("nan"),
            "removed_hi": float(np.percentile(draws, 97.5)) if draws else float("nan"),
        }
        for elic in ("rating", "pairwise"):
            g = group[group["elicitation"] == elic]
            row[f"{elic}_runs"] = int(len(g))
            row[f"{elic}_runs_moved"] = int(g["moved"].sum())
        cell = runs[(runs["error_model"] == model) & (runs["magnitude"] == mag)]
        for elic in ("rating", "pairwise"):
            c = cell[cell["elicitation"] == elic]
            row[f"{elic}_fit_fallbacks"] = int(c["fit_failures"].sum())
            if "n_fits" in c.columns:
                row[f"{elic}_fits"] = int(c["n_fits"].sum())
        if uniform:
            nr = noisy_runs.reset_index()
            nr = nr[(nr["error_model"] == model) & (nr["magnitude"] == mag) & (nr["elicitation"] == "rating")]
            row["rating_runs_tied_top"] = int((nr["n_tied_top"] > 1).sum())
            row["rating_runs_tie_matters"] = int(
                ((nr["shipped_true"] - nr["shipped_true_uniform_ties"]).abs() > TIE_TOL).sum())
            row["rating_excess_uniform_ties"] = float(per_u["rating"].mean())
            row["removed_share_uniform_ties"] = (float(diff_u.mean() / base_u.mean())
                                                 if base_u.mean() != 0 else float("nan"))
            row["removed_lo_uniform_ties"] = float(np.percentile(draws_u, 2.5)) if draws_u else float("nan")
            row["removed_hi_uniform_ties"] = float(np.percentile(draws_u, 97.5)) if draws_u else float("nan")
        # The standard-process estimand, one row per (dataset, seed).
        wide = {}
        for elic in ("rating", "pairwise"):
            for flag, frame in (("noisy", noisy_runs), ("clean", clean_runs)):
                f = frame.reset_index()
                f = f[(f["error_model"] == model) & (f["magnitude"] == mag) & (f["elicitation"] == elic)]
                wide[f"{elic}_{flag}"] = f.set_index(["dataset", "seed"])["regret"]
                if uniform and elic == "rating":
                    wide[f"rating_{flag}_u"] = f.set_index(["dataset", "seed"])["regret_u"]
        regret = pd.DataFrame(wide).dropna(subset=["rating_noisy", "rating_clean",
                                                    "pairwise_noisy", "pairwise_clean"]).reset_index()
        row.update(_standard_process(regret, ""))
        if uniform:
            row.update(_standard_process(regret, "_u"))
        rows.append(row)
    return pd.DataFrame(rows)


def landscape_names(functions: str, stats: dict) -> list[str]:
    """The landscapes ``--functions`` names: 'all', 'suite' or a comma-separated list.

    A name the stats file does not hold is an error rather than silently dropped.
    """
    if functions == "all":
        return sorted(stats)
    if functions == "suite":
        names = sorted(bb.DEFAULT_SUITE)
    else:
        names = [f.strip() for f in functions.split(",") if f.strip()]
    unknown = [n for n in names if n not in stats]
    if unknown:
        raise SystemExit(f"--functions names landscapes the stats file does not hold: {unknown}")
    return names


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / f"{OUTPUT_NAME}_runs.csv"

    stats = bb.load_stats(args.stats_path)
    names = landscape_names(args.functions, stats)

    if args.summary_only and runs_path.is_file():
        runs = pd.read_csv(runs_path)
    else:
        tasks = build_tasks(args, names)
        print(f"{len(tasks):,} runs ({len(names)} landscapes) with {args.workers} worker(s)", flush=True)
        records = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(args.workers, os.cpu_count() or 1),
                                 initializer=_init_worker) as pool:
            for i, rec in enumerate(pool.map(run_cell, tasks, chunksize=2), 1):
                records.append(rec)
                if i % 50 == 0:
                    rate = i / (time.time() - t0)
                    print(f"  {i:,}/{len(tasks):,}  {rate:.2f}/s  "
                          f"eta {(len(tasks) - i) / max(rate, 1e-9) / 60:.1f} min", flush=True)
        runs = pd.DataFrame(records)
        runs.to_csv(runs_path, index=False)

    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    summary = summarise(runs, opt_z)
    summary.to_csv(args.output_dir / f"{OUTPUT_NAME}_summary.csv", index=False)

    print("\nExcess regret of the shipped design over each loop's own clean twin,")
    print("and the share of the rating loop's excess that comparisons remove:\n")
    for klass in ("shared", "idiosyncratic"):
        block = summary[summary["class"] == klass]
        if block.empty:
            continue
        print(f"  {klass.upper()} faults" + (" (a monotone map on the whole sitting)" if klass == "shared"
                                             else " (drawn afresh per judgement)"))
        for _, r in block.sort_values(["error_model", "magnitude"]).iterrows():
            print(f"    {r['error_model']:<9} {r['magnitude']:>4}sd  rating {r['rating_excess']:+.4f}  "
                  f"pairwise {r['pairwise_excess']:+.4f}  removed {r['removed_share'] * 100:+6.1f}% "
                  f"[{r['removed_lo'] * 100:+.0f},{r['removed_hi'] * 100:+.0f}]  "
                  f"n={int(r['n_landscapes'])}")
        print()
    print(report(summary))
    print(f"Wrote {runs_path} and {args.output_dir / (OUTPUT_NAME + '_summary.csv')}")
    return summary


def report(summary: pd.DataFrame) -> str:
    """The run counts, fallbacks, tie sensitivity and standard-process estimand.

    Every number is a landscape mean of the shipped design's regret in units of
    opt_z (the deployed design at the final judgement), with a landscape bootstrap.
    """
    def pct(v: float) -> str:
        return "  nan" if pd.isna(v) else f"{v * 100:+.1f}%"

    lines = ["Per cell: noisy runs whose shipped design differs from the clean twin's, and fit fallbacks",
             "(a failed hyperparameter fit, kept with the prior hyperparameters):"]
    for _, r in summary.sort_values(["error_model", "magnitude"]).iterrows():
        fits = {e: (int(r[f"{e}_fits"]) if f"{e}_fits" in r and pd.notna(r[f"{e}_fits"]) else None)
                for e in ("rating", "pairwise")}
        lines.append(
            f"  {r['error_model']:<9} {r['magnitude']:>4}sd  moved: rating {int(r['rating_runs_moved'])}/"
            f"{int(r['rating_runs'])}, pairwise {int(r['pairwise_runs_moved'])}/{int(r['pairwise_runs'])}; "
            f"fallbacks: rating {int(r['rating_fit_fallbacks'])}"
            + (f"/{fits['rating']} fits" if fits["rating"] is not None else "")
            + f", pairwise {int(r['pairwise_fit_fallbacks'])}"
            + (f"/{fits['pairwise']} fits" if fits["pairwise"] is not None else ""))
    total = {e: int(summary[f"{e}_fit_fallbacks"].sum()) for e in ("rating", "pairwise")}
    lines.append(f"  all cells: fallbacks rating {total['rating']}, pairwise {total['pairwise']}"
                 + (f"; fits rating {int(summary['rating_fits'].sum())}, pairwise "
                    f"{int(summary['pairwise_fits'].sum())}" if "rating_fits" in summary else "")
                 + f"; runs {int(summary['rating_runs'].sum() + summary['pairwise_runs'].sum()) * 2}"
                 " (noisy and clean)")
    if "removed_share_uniform_ties" in summary:
        lines += ["", "Ties: the rating loop ships the earliest design with the best rating (first index);",
                  "under a uniformly random choice among tied designs (exact expectation) instead:"]
        for _, r in summary.sort_values(["error_model", "magnitude"]).iterrows():
            lines.append(
                f"  {r['error_model']:<9} {r['magnitude']:>4}sd  noisy rating runs with a tied top "
                f"{int(r['rating_runs_tied_top'])}, tie matters {int(r['rating_runs_tie_matters'])}; "
                f"rating excess {r['rating_excess']:+.4f} -> {r['rating_excess_uniform_ties']:+.4f}; "
                f"removed {pct(r['removed_share'])} -> {pct(r['removed_share_uniform_ties'])} "
                f"[{pct(r['removed_lo_uniform_ties'])}, {pct(r['removed_hi_uniform_ties'])}]")
    lines += ["", "Against the standard process (AGENTS.md): cost = rating noisy - rating clean, gain = rating",
              "noisy - comparison noisy, price = comparison clean - rating clean; landscape means, opt_z units:"]
    for suffix, label in (("", "first-index ties"), ("_u", "uniform ties")):
        if f"std_cost{suffix}" not in summary:
            continue
        lines.append(f"  rating loop's ship rule with {label}:")
        for _, r in summary.sort_values(["error_model", "magnitude"]).iterrows():
            sup = r.get(f"std_recovered_suppressed{suffix}", "")
            share = (f"share {pct(r[f'std_recovered{suffix}'])} [{pct(r[f'std_recovered_lo{suffix}'])}, "
                     f"{pct(r[f'std_recovered_hi{suffix}'])}]") if not (isinstance(sup, str) and sup) \
                else f"share not reported ({sup})"
            lines.append(
                f"    {r['error_model']:<9} {r['magnitude']:>4}sd  cost {r[f'std_cost{suffix}']:+.4f}  "
                f"gain {r[f'std_gain{suffix}']:+.4f}  price {r[f'std_price{suffix}']:+.4f}  {share}  "
                f"Wilcoxon p {r[f'std_wilcoxon_p{suffix}']:.3g}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
