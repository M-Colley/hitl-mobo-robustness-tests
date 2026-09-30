"""The oracle-isolation comparison on the paper's headline unit, the share of the achievable improvement.

The floor-gap fraction of oracle_isolation.py divides by the optimum less a
same-budget model-free floor. That normaliser depends on how hard the objective
is for random search as well as on its scale. On Branin, Powell and Rosenbrock
random search nearly reaches the optimum, so the exact arm's floor gap is 1-2%
of opt_z there and those three landscapes carry most of a ratio of landscape
means; and on a tree-ensemble oracle random search leaves more of the
improvement unreached than on the exact landscape, because the oracle's
estimated optimum lies a median 17% of its achievable improvement above the
best value any clean run reaches. This check puts both arms on the normaliser of
every headline number of the paper, which cannot approach zero:

  exact arm   excess / opt_z, opt_z the optimum less the landscape mean in
              landscape-SD units (boba_benchmarks stats; the exact objective is
              standardised, so its mean is 0 and y_opt = opt_z, which is checked);
  fitted arm  each seed's oracle has its own optimum and mean, so the achievable
              improvement is formed within each seed, A_s, and a landscape's
              fraction is the ratio of seed means, mean_s(excess_s) / mean_s(A_s).

Fitted-arm estimators of A_s, each written to its own column and summary file
(the exact arm keeps opt_z):

  oracle      (primary) y_opt_s - mean_f,s: y_opt_s the logged optimum of seed
              s's oracle (the simulator's estimate over the data box, fixed
              before any run) and mean_f,s its mean over 20,000 uniform points
              in the data box, the box the runs search
              (output-oracle-iso/oracle_sigmaf_seeds.csv,
              scripts/review_checks/oracle_sigmaf_seeds.py). The analogue of
              opt_z. Each family sets its own A_s, and the families set it
              differently (A_s over sigma_f,s * opt_z below).
  best_clean  (sensitivity) the best value any clean run of seed s reached in
              place of y_opt_s; attainable by construction, it shrinks A_s.
  empirical   (check of mean_f) y_opt_s less the mean of the oracle over the 100
              designs the clean random and Sobol floor runs of seed s visited,
              read from the run logs, no refit.
  visited     (check of y_opt) y_opt_s raised to the best value any run of seed s
              visited, clean or noisy, where one beat it (oracle_isolation's
              "visited" optimum), less mean_f,s. Runs beat the random-search
              y_opt on most GP and MLP landscape-seeds.
  sigma_f     (sensitivity) A_s = sigma_f,s * opt_z: the landscape's own achievable
              improvement in units of the SD of seed s's oracle, so no family's
              optimum or mean estimate enters. The error is scaled by sigma_f too,
              so this keeps the noise-to-normaliser ratio of the exact arm.
  sigma_f_seed7  (check) the same with the seed-7 oracle's sigma_f, the one that
              scales every seed's error grid (oracle_isolation.calibrate), so the
              nominal noise-to-normaliser ratio equals the exact arm's exactly.

and one estimator that changes both arms, written to *_both_best_clean.csv:

  both_best_clean  (sensitivity) each arm's optimum replaced by the best value any
              clean run of the seed reached, the fitted arm as in best_clean and
              the exact arm as mean_s(best clean_s) - 0 over the same six
              acquisitions (LogEI, qNEI, UCB, EI, random, Sobol) and seeds, so the
              two optima are estimated the same way.

The excess is the response of oracle_isolation.py (the trajectory's post-onset
per-iteration excess search regret, and the deployed design's excess at the
final trial), read from its per-landscape files, which are balanced across
seeds. Per family this writes

  <root>/oracle_isolation_per_landscape_achievable{,_deployed}.csv
  <root>/oracle_isolation_summary_achievable{,_deployed}{,_best_clean,_empirical,_visited,
        _sigma_f,_sigma_f_seed7,_both_best_clean}.csv

(the summaries from oracle_isolation.summarise: ratio of landscape means with the
landscape-bootstrap interval, median per-landscape ratio, landscapes lower with
the Wilcoxon p, Spearman of the per-landscape costs, the ratio without the three
small-gap landscapes) and prints the mean_f check, how each family's A_s compares
with sigma_f,s * opt_z, the exact arm's best clean value against opt_z, and the
floor-share decomposition: floor share = floor gap / achievable improvement, the
part of the improvement over a random design that a same-budget random or Sobol
run does not reach. Per landscape, the fitted/exact ratio in floor-gap units is
the ratio in achievable units times (exact share / fitted share). The family
comparison under every estimator is printed by
scripts/review_checks/oracle_families.py. The landscapes share their random
numbers by design; the bootstrap treats them as independent.

    python scripts/review_checks/oracle_sigmaf_seeds.py   # writes the per-seed means
    python scripts/review_checks/oracle_achievable.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import boba_benchmarks as bb  # noqa: E402
import oracle_isolation as iso  # noqa: E402

ROOTS = {"tree": REPO / "output-oracle-iso", "GP": REPO / "output-oracle-iso-gaussian_process",
         "MLP": REPO / "output-oracle-iso-mlp"}
SIGMAF = {"tree": "tree", "GP": "gaussian_process", "MLP": "mlp"}  # family labels in oracle_sigmaf_seeds.csv
METRICS = (("", "trajectory"), ("_deployed", "deployed"))
UNIT = "_achievable"  # infix of the files this writes
# Fitted-arm estimators of A_s, primary first, and the suffix of the column and
# summary each writes; new ones are appended so the files of the old ones keep their names.
ESTIMATORS = {"oracle": "", "best_clean": "_best_clean", "empirical": "_empirical", "visited": "_visited",
              "sigma_f": "_sigma_f", "sigma_f_seed7": "_sigma_f_seed7"}
# Estimators that re-estimate the exact arm's normaliser as well:
# name -> (fitted column, exact column); the summary suffix is "_" + name.
PAIRED = {"both_best_clean": ("frac_fitted_best_clean", "frac_exact_best_clean")}


def seed_achievable(runs: pd.DataFrame, mean_f: pd.Series, how: str = "oracle") -> pd.Series:
    """A_s = optimum_s - mean_f,s for every seed of one landscape's fitted runs.

    ``how`` picks the optimum as oracle_isolation.seed_optima does ("oracle" is the
    logged y_opt, "best_clean" the best clean value, "visited" y_opt raised to the
    best value any run visited); ``mean_f`` is indexed by seed.
    """
    optimum = iso.seed_optima(runs, how)
    mean_f = pd.Series(mean_f).reindex(optimum.index)
    if mean_f.isna().any():
        raise ValueError(f"no oracle mean for seeds {list(mean_f.index[mean_f.isna()])}")
    gap = optimum - mean_f
    if (gap <= 0).any():
        raise ValueError(f"optimum below the oracle mean on seeds {list(gap.index[gap <= 0])}")
    return gap


def sigma_f_achievable(runs: pd.DataFrame, sigma_f: pd.Series, opt_z: float) -> pd.Series:
    """A_s = sigma_f,s * opt_z for every seed of one landscape's fitted runs: the
    landscape's achievable improvement (opt_z, in landscape-SD units) in units of
    the SD of seed s's oracle. ``sigma_f`` is indexed by seed."""
    seeds = pd.Index(sorted(runs["seed"].unique()), name="seed")
    sigma_f = pd.Series(sigma_f).reindex(seeds)
    if sigma_f.isna().any():
        raise ValueError(f"no sigma_f for seeds {list(sigma_f.index[sigma_f.isna()])}")
    if (sigma_f <= 0).any() or opt_z <= 0:
        raise ValueError("sigma_f and opt_z must be positive")
    return sigma_f * float(opt_z)


def exact_best_clean(exact: pd.DataFrame, opt_z: float) -> float:
    """The exact arm's achievable improvement with each seed's best clean value as
    the optimum: mean over seeds of the best final value any clean run of the
    fitted arm's six acquisitions reached, less the landscape mean (0: the exact
    objective is standardised). Checks that the logged y_opt is opt_z."""
    runs = exact[exact.acquisition.isin(iso.ACQS + iso.FLOORS)]
    y_opt = runs[iso.BASELINE_BEST] + runs[iso.BASELINE_REGRET]
    if not np.allclose(y_opt, opt_z, rtol=1e-6, atol=1e-6):
        raise ValueError(f"the exact arm's logged y_opt is not opt_z ({y_opt.min():g} to {y_opt.max():g} "
                         f"against {opt_z:g})")
    missing = set(iso.ACQS + iso.FLOORS) - set(runs.acquisition)
    if missing:
        raise ValueError(f"no clean runs of {sorted(missing)}")
    best = runs.groupby("seed")[iso.BASELINE_BEST].max()
    if (best <= 0).any():
        raise ValueError("a seed's best clean value is below the landscape mean")
    return float(best.mean())


def empirical_means(run_dir: Path, name: str) -> pd.Series:
    """The oracle's mean per seed over the designs the clean random and Sobol runs visited."""
    out = {}
    for seed in iso.SEEDS:
        values = []
        for acq in iso.FLOORS:
            files = sorted(run_dir.glob(f"bo_sensor_error_iso_{name}_composite_{acq}_seed{seed}_baseline_*.csv"))
            if len(files) != 1:
                raise SystemExit(f"{run_dir}: expected one clean {acq} run of seed {seed}, found {len(files)}")
            values.append(pd.read_csv(files[0], usecols=["objective_true"]).objective_true.to_numpy())
        out[seed] = float(np.mean(np.concatenate(values)))
    return pd.Series(out)


def achievable(root: Path, family: str, seeds_table: pd.DataFrame, opt_z: pd.Series,
               exact_bc: pd.Series) -> pd.DataFrame:
    """Per landscape: mean_s(A_s) under each estimator, in the oracle's units, and
    the exact arm's best-clean achievable improvement."""
    rows = {}
    manifest = pd.read_csv(root / "manifest.csv")
    for name in manifest.landscape:
        runs = iso._paired(root / "runs" / f"iso_{name}")
        runs = runs[(runs.jitter_iteration == 0) & runs.seed.isin(iso.SEEDS)]
        table = seeds_table[(seeds_table.family == SIGMAF[family]) & (seeds_table.landscape == name)].set_index("seed")
        mean_f = table["mean_f"]
        empirical = empirical_means(root / "runs" / f"iso_{name}", name)
        seed7 = pd.Series(float(table.loc[7, "sigma_f"]), index=table.index)
        rows[name] = {
            "achievable_fitted": float(seed_achievable(runs, mean_f, "oracle").mean()),
            "achievable_fitted_best_clean": float(seed_achievable(runs, mean_f, "best_clean").mean()),
            "achievable_fitted_empirical": float(seed_achievable(runs, empirical, "oracle").mean()),
            "achievable_fitted_visited": float(seed_achievable(runs, mean_f, "visited").mean()),
            "achievable_fitted_sigma_f": float(sigma_f_achievable(runs, table["sigma_f"], opt_z[name]).mean()),
            "achievable_fitted_sigma_f_seed7": float(sigma_f_achievable(runs, seed7, opt_z[name]).mean()),
            "achievable_exact_best_clean": float(exact_bc[name]),
            "mean_f_minus_empirical_over_sigma_f": float(
                ((mean_f - empirical.reindex(mean_f.index)) / table["sigma_f"]).abs().max()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def per_landscape(floor_gap_per: pd.DataFrame, A: pd.DataFrame, opt_z: pd.Series) -> pd.DataFrame:
    """The floor-gap per-landscape frame re-expressed as shares of the achievable improvement."""
    p = floor_gap_per.sort_values(["sigma_multiple", "landscape"]).reset_index(drop=True)
    a = A.reindex(p.landscape)
    if a.isna().any().any():
        raise ValueError("a landscape has no achievable improvement")
    out = p[["landscape", "sigma_multiple", "n_fitted", "n_exact", "oracle_model", "corr_oracle_truth",
             "excess_fitted", "excess_exact", "gap_fitted", "gap_fitted_best_clean", "gap_exact",
             "gap_exact_over_opt_z"]].copy()
    for how, osuffix in ESTIMATORS.items():
        out[f"achievable_fitted{osuffix}"] = a[f"achievable_fitted{osuffix}"].to_numpy()
        out[f"frac_fitted{osuffix}"] = out.excess_fitted / out[f"achievable_fitted{osuffix}"]
    out["opt_z"] = p.landscape.map(opt_z).to_numpy()
    out["frac_exact"] = out.excess_exact / out.opt_z
    out["achievable_exact_best_clean"] = a["achievable_exact_best_clean"].to_numpy()
    out["frac_exact_best_clean"] = out.excess_exact / out.achievable_exact_best_clean
    out["floor_share_fitted"] = out.gap_fitted / out.achievable_fitted
    out["floor_share_fitted_best_clean"] = out.gap_fitted_best_clean / out.achievable_fitted_best_clean
    out["floor_share_exact"] = out.gap_exact / out.opt_z
    # how this family's own achievable improvement compares with the landscape's in its sigma_f units
    out["achievable_fitted_over_sigma_f_opt_z"] = out.achievable_fitted / out.achievable_fitted_sigma_f
    return out


def _range(x: pd.Series) -> str:
    return f"median {x.median():.3f}, range [{x.min():.3f}, {x.max():.3f}]"


def main() -> None:
    table = pd.read_csv(REPO / "output-oracle-iso" / "oracle_sigmaf_seeds.csv")
    if "mean_f" not in table:
        raise SystemExit("oracle_sigmaf_seeds.csv has no mean_f; rerun scripts/review_checks/oracle_sigmaf_seeds.py")
    stats = bb.load_stats()
    opt_z = pd.Series({k: float(v["opt_z"]) for k, v in stats.items()})
    names = sorted(pd.read_csv(ROOTS["tree"] / "manifest.csv").landscape)
    exact_bc = pd.Series({n: exact_best_clean(iso.exact_runs(n, REPO / "output-boba"), opt_z[n]) for n in names})
    print("Share of the achievable improvement: fitted mean_s(excess_s) / mean_s(y_opt_s - mean_f,s) "
          "against exact excess / opt_z.")
    q = exact_bc / opt_z.reindex(exact_bc.index)
    print(f"\n== exact arm: best clean value over opt_z (mean over seeds 7-11, the six acquisitions): {_range(q)}; "
          f"below 0.9 on {', '.join(f'{k} {v:.3f}' for k, v in q[q < 0.9].sort_values().items())}")
    for family, root in ROOTS.items():
        A = achievable(root, family, table, opt_z, exact_bc)
        print(f"\n== {family}: achievable improvement of the fitted oracle (oracle units), mean over seeds: "
              f"median {A.achievable_fitted.median():.2f}, range [{A.achievable_fitted.min():.2f}, "
              f"{A.achievable_fitted.max():.2f}]; opt_z of the same landscapes median "
              f"{opt_z.reindex(A.index).median():.2f}, range [{opt_z.reindex(A.index).min():.2f}, "
              f"{opt_z.reindex(A.index).max():.2f}]")
        print(f"   mean_f check: |mean_f - mean over the 100 clean floor designs| at most "
              f"{A.mean_f_minus_empirical_over_sigma_f.max():.3f} sigma_f of the seed; A from the refit mean over A "
              f"from the floor designs in [{(A.achievable_fitted / A.achievable_fitted_empirical).min():.3f}, "
              f"{(A.achievable_fitted / A.achievable_fitted_empirical).max():.3f}]; best-clean A over y_opt A "
              f"{_range(A.achievable_fitted_best_clean / A.achievable_fitted)}")
        print(f"   the family's own A over sigma_f * opt_z (per-seed sigma_f): "
              f"{_range(A.achievable_fitted / A.achievable_fitted_sigma_f)}; above 1 on "
              f"{int((A.achievable_fitted > A.achievable_fitted_sigma_f).sum())}/{len(A)}; with the seed-7 sigma_f "
              f"{_range(A.achievable_fitted / A.achievable_fitted_sigma_f_seed7)}; best-clean A over sigma_f * opt_z "
              f"{_range(A.achievable_fitted_best_clean / A.achievable_fitted_sigma_f)}")
        for suffix, label in METRICS:
            fg = pd.read_csv(root / f"oracle_isolation_per_landscape{suffix}.csv")
            per = per_landscape(fg, A, opt_z)
            per.to_csv(root / f"oracle_isolation_per_landscape{UNIT}{suffix}.csv", index=False)
            jobs = [(how, f"frac_fitted{osuffix}", "frac_exact", osuffix) for how, osuffix in ESTIMATORS.items()]
            jobs += [(how, fit, ex, f"_{how}") for how, (fit, ex) in PAIRED.items()]
            for how, fit, ex, osuffix in jobs:
                summary = iso.summarise(per, fit, ex)
                summary.to_csv(root / f"oracle_isolation_summary{UNIT}{suffix}{osuffix}.csv", index=False)
                cells = "; ".join(
                    f"{r.sigma_multiple:g}s {r.ratio_fitted_over_exact:.3f} [{r.ratio_lo:.3f}, {r.ratio_hi:.3f}] "
                    f"median {r.median_ratio:.3f} lower {int(r.n_lower)}/{int(r.n_landscapes)} p {r.wilcoxon_p:.2g}"
                    for r in summary.itertuples())
                print(f"   {label:10s} {how:15s} {cells}")
            if suffix == "":
                b = per[per.sigma_multiple == 1.0]
                print(f"   floor share (floor gap / achievable improvement): fitted median "
                      f"{b.floor_share_fitted.median():.3f} [{b.floor_share_fitted.min():.3f}, "
                      f"{b.floor_share_fitted.max():.3f}], with the best-clean optimum "
                      f"{b.floor_share_fitted_best_clean.median():.3f}; "
                      f"exact median {b.floor_share_exact.median():.3f} [{b.floor_share_exact.min():.3f}, "
                      f"{b.floor_share_exact.max():.3f}]; fitted share above the exact on "
                      f"{int((b.floor_share_fitted > b.floor_share_exact).sum())}/{len(b)} landscapes")
    print("\nThe family comparison under every estimator: scripts/review_checks/oracle_families.py")


if __name__ == "__main__":
    main()
