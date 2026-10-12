"""Which landscapes are ceiling-limited for the late onset, and what the late-onset numbers are without them.

A late-onset pair (error from trial 21) shares every design up to trial s = 21,
so the noisy run's post-onset search excess can never exceed what its clean twin
still gains after s (the running-maximum bound, scripts/onset_bound.py). A clean
run already at the optimum at s forces that excess to exactly zero; one that
gains nothing after s forces it to be at most zero. The headroom screen of
analyse_boba_robustness.py does not see this: it compares BO's clean trajectory
with random search's, not with the optimum.

This check

1. measures, per landscape and over the ten model-based learners' clean twins,
   the share at the optimum (search loss <= 1e-9 landscape SDs; regret at the
   optimum is exactly 0.0 here, and the smallest positive clean regret is
   4.5e-8, so any tolerance below that gives the same shares) at trial 5 (the
   last design an early-onset pair shares), at trial 20 (the last clean rating
   before the late onset), at trial 21 (the last design a late-onset pair
   shares) and at the final trial, and the share whose clean twin gains nothing
   after trial 21. The per-trial regrets come from onset_bound.py's per-run cache
   (r_c(s) = rc_s, r_c(T) = rc_T, r_c(20) = Bt_final + rc_T), and the shares at
   trial 21 and T and the no-gain share must equal analyse_boba_robustness's
   ceiling_screen.csv, which derives them from the paired metrics alone;

2. recomputes the late-onset search-loss numbers the paper reports with all
   twenty landscapes and without the ceiling-limited ones (a majority of clean
   runs at the optimum at trial 21): the trajectory rows of tab:dose and the
   onset ratio (cell_means.csv, as make_boba_paper_tables does), the final-trial
   search loss (decompose_regret on ship_rules_per_run.csv), and the onset-bound
   quantities of Section 4 with onset_bound.py's own functions (10,000 paired
   landscape resamples): the late and early normalised losses, their ratio, the
   bound ratio and the share of late-onset runs that make no progress after
   trial 21 while their clean twin does. The deployed excess from trial 21, which
   the bound does not constrain, is given beside them.

The landscapes share their random numbers by design, so the landscape bootstrap
treats correlated clusters as independent.

    python scripts/checks/ceiling_clean_runs.py > output-boba/analysis/review/register_checks/ceiling_clean_runs.txt

Writes output-boba/analysis/review/ceiling_clean_runs.csv (part 1, per landscape)
and output-boba/analysis/review/ceiling_sensitivity.csv (part 2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
import boba_benchmarks as bb  # noqa: E402
import decompose_regret as dr  # noqa: E402
import onset_bound as ob  # noqa: E402

TOL = 1e-9
LATE = 20
KEY = ["dataset", "acquisition", "seed"]
MODEL_FREE = ("random", "sobol")
ANALYSIS = REPO / "output-boba" / "analysis"
REVIEW = ANALYSIS / "review"


def clean_shares(runs: pd.DataFrame) -> pd.DataFrame:
    """Per landscape: clean twins at the optimum at trials 20, 21 and T, and gaining nothing after 21."""
    late = runs[(runs["arm"] == "main") & (runs["onset"] == LATE)]
    if set(late["s"]) != {LATE + 1}:
        raise ValueError(f"the late onset's last shared design is not trial {LATE + 1}: {sorted(set(late['s']))}")
    cols = ["rc_s", "rc_T", "Bt_final"]
    spread = late.groupby(KEY)[cols].agg(lambda s: float(np.ptp(s.to_numpy())))
    if float(spread.to_numpy().max()) > 1e-12:
        raise ValueError("a clean twin's regrets differ between its noisy runs")
    twins = late.groupby(KEY)[cols].first().reset_index()
    twins["rc_20"] = twins["Bt_final"] + twins["rc_T"]
    twins = twins.assign(at_t20=twins["rc_20"] <= TOL, at_t21=twins["rc_s"] <= TOL, at_T=twins["rc_T"] <= TOL,
                         no_gain=(twins["rc_s"] - twins["rc_T"]) <= TOL)
    positive = twins[["rc_20", "rc_s", "rc_T"]].to_numpy()
    print(f"clean twins: {len(twins)}; regrets exactly 0.0 at trial 21: {int((twins['rc_s'] == 0).sum())}, "
          f"<= {TOL:g}: {int(twins['at_t21'].sum())}; smallest positive clean regret at trials 20, 21, T: "
          f"{positive[positive > 0].min():.2e}")
    # Error from the first rating shares only the five initial designs (s = 5).
    early = runs[(runs["arm"] == "main") & (runs["onset"] == 0)]
    if set(early["s"]) != {5}:
        raise ValueError(f"the early onset's last shared design is not trial 5: {sorted(set(early['s']))}")
    at5 = (early.groupby(KEY)["rc_s"].first() <= TOL).groupby(level="dataset").mean()
    per = twins.groupby("dataset").agg(n_clean_runs=("at_t21", "size"),
                                       share_clean_at_optimum_t20=("at_t20", "mean"),
                                       share_clean_at_optimum_t21=("at_t21", "mean"),
                                       share_clean_at_optimum_T=("at_T", "mean"),
                                       share_clean_gains_nothing_after_t21=("no_gain", "mean"))
    per.insert(1, "share_clean_at_optimum_t5", per.index.map(at5))
    return per.sort_values(["share_clean_at_optimum_t21"], ascending=False)


def cross_check(per: pd.DataFrame) -> str:
    path = ANALYSIS / "ceiling_screen.csv"
    if not path.is_file():
        return f"cross-check skipped: {path.relative_to(REPO)} not written yet (run analyse_boba_robustness.py)"
    screen = pd.read_csv(path).set_index("dataset")
    pairs = (("share_clean_at_optimum_t21", "share_clean_at_optimum_t21"),
             ("share_clean_at_optimum_T", "share_clean_at_optimum_T"),
             ("share_clean_gains_nothing_after_t21", "share_late_runs_clean_gains_nothing_after_t21"))
    worst = max(float((per[a] - screen.loc[per.index, b]).abs().max()) for a, b in pairs)
    if worst > 1e-12:
        raise SystemExit(f"ceiling_screen.csv disagrees with the per-trial regrets by {worst:.3e}")
    return (f"cross-check: ceiling_screen.csv (paired metrics only) equals the per-trial regrets of the "
            f"onset-bound cache on all {len(per)} landscapes (largest difference {worst:.1e})")


def dose_rows(cells: pd.DataFrame, opt_z: dict, exclude: list[str]) -> list[dict]:
    """tab:dose's trajectory and deployed rows from cell_means.csv, and the onset ratio."""
    cells = cells[~cells["acquisition"].isin(MODEL_FREE) & ~cells["dataset"].isin(exclude)]
    cells = cells.assign(deployed=cells["inference_excess"] / cells["dataset"].map(opt_z))
    out = []
    for value, label in (("fragility", "trajectory search loss"), ("deployed", "deployed excess")):
        grid = cells.pivot_table(index="jitter_iteration", columns="jitter_std", values=value)
        for std in grid.columns:
            out.append({"quantity": f"{label}, from trial {LATE + 1}", "jitter_std": float(std),
                        "estimate": float(grid.loc[LATE, std]), "lo": np.nan, "hi": np.nan})
            out.append({"quantity": f"{label}, onset ratio (trial 1 / trial {LATE + 1})", "jitter_std": float(std),
                        "estimate": float(grid.loc[0, std] / grid.loc[LATE, std]), "lo": np.nan, "hi": np.nan})
    return out


def final_search_rows(noisy: pd.DataFrame, clean: pd.DataFrame, exclude: list[str]) -> list[dict]:
    """decompose_regret's final-trial search excess from trial 21, pooled over the four processes."""
    out = []
    keep = noisy[~noisy["dataset"].isin(exclude)]
    for std, block in keep[keep["jitter_iteration"] == LATE].groupby("jitter_std"):
        row = dr.summarise(block, clean, np.random.default_rng(dr.BOOTSTRAP_SEED))
        out.append({"quantity": f"final search loss, from trial {LATE + 1}", "jitter_std": float(std),
                    "estimate": row["mean_excess_search"], "lo": np.nan, "hi": np.nan})
    return out


def bound_rows(runs: pd.DataFrame, exclude: list[str]) -> list[dict]:
    """onset_bound.py's pooled cells, with its own functions and resamples."""
    main = runs[(runs["arm"] == "main") & ~runs["dataset"].isin(exclude)]
    out = []
    for std in sorted(main["jitter_std"].unique()):
        cell = main[main["jitter_std"] == std]
        early, late = cell[cell["onset"] == 0], cell[cell["onset"] == LATE]
        late_q = {q: (e, lo, hi) for q, e, lo, hi in ob.cell_quantities(late)}
        early_q = {q: (e, lo, hi) for q, e, lo, hi in ob.cell_quantities(early)}
        ratio_q = {q: (e, lo, hi) for q, e, lo, hi in ob.ratio_quantities(early, late)}
        for label, (e, lo, hi) in (
                (f"normalised search loss (excess / bound), from trial {LATE + 1}", late_q["normalised"]),
                ("normalised search loss (excess / bound), from trial 1", early_q["normalised"]),
                ("normalised ratio (trial 1 / trial 21)", ratio_q["normalised_ratio"]),
                ("bound ratio (trial 1 / trial 21)", ratio_q["bound_ratio"]),
                ("raw trajectory ratio (trial 1 / trial 21)", ratio_q["raw_ratio"]),
                (f"share of late-onset runs: no progress after trial {LATE + 1}, clean twin progresses",
                 late_q["share_runs_noisy_no_gain_clean_gain"])):
            out.append({"quantity": label, "jitter_std": float(std), "estimate": e, "lo": lo, "hi": hi})
    return out


def main() -> None:
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    # onset_bound.py's per-run cache, in fractions of opt_z as its own aggregation uses them.
    runs = ob.add_fractions(pd.read_parquet(REVIEW / "onset_bound_runs.parquet"), opt_z)
    per = clean_shares(runs)
    limited = sorted(per.index[per["share_clean_at_optimum_t21"] >= 0.5])

    print("\nCLEAN RUNS AT THE OPTIMUM, ten model-based learners x ten seeds per landscape")
    print(f"(search loss <= {TOL:g} landscape SDs; 'no gain': the clean twin's search loss at trial 21 equals its "
          f"final one, so the late-onset excess is at most zero)")
    print(f"  {'landscape':<20s} {'trial 5':>8s} {'trial 20':>9s} {'trial 21':>9s} {'trial T':>8s} {'no gain':>8s}")
    for name, r in per.iterrows():
        mark = "  <- ceiling-limited" if name in limited else ""
        print(f"  {name:<20s} {r['share_clean_at_optimum_t5']:>8.2f} {r['share_clean_at_optimum_t20']:>9.2f} "
              f"{r['share_clean_at_optimum_t21']:>9.2f} {r['share_clean_at_optimum_T']:>8.2f} "
              f"{r['share_clean_gains_nothing_after_t21']:>8.2f}{mark}")
    print("  (trial 5: the last design an early-onset pair shares; no landscape is ceiling-limited there)"
          if not (per["share_clean_at_optimum_t5"] >= 0.5).any() else
          "  (trial 5: some landscape is ceiling-limited even for error from the first rating)")
    print(cross_check(per))
    REVIEW.mkdir(parents=True, exist_ok=True)
    per.assign(ceiling_limited_late=per.index.isin(limited)).to_csv(REVIEW / "ceiling_clean_runs.csv")

    cells = pd.read_csv(ANALYSIS / "cell_means.csv")
    table = dr.decompose(dr.load_runs(ANALYSIS / "ship_rules_per_run.csv", None), opt_z)
    table = table[~table["acquisition"].isin(MODEL_FREE)]
    clean, noisy = table[table["baseline"]], table[~table["baseline"]]
    rows = []
    for scope, exclude in (("all 20 landscapes", []), (f"without the {len(limited)} ceiling-limited", limited)):
        for r in dose_rows(cells, opt_z, exclude) + final_search_rows(noisy, clean, exclude) + bound_rows(runs, exclude):
            rows.append({"scope": scope, "n_landscapes": 20 - len(exclude), **r})
    sens = pd.DataFrame(rows)
    sens.to_csv(REVIEW / "ceiling_sensitivity.csv", index=False)

    print(f"\nLATE-ONSET NUMBERS WITH AND WITHOUT THE CEILING-LIMITED LANDSCAPES ({', '.join(limited)})")
    print("fractions of opt_z (or ratios), pooled over the four error processes and the ten model-based learners;")
    print("brackets: 95% landscape bootstrap, 10,000 paired resamples (onset_bound.py)")
    wide = sens.pivot_table(index=["quantity", "jitter_std"], columns="scope", values=["estimate", "lo", "hi"],
                            sort=False)
    scopes = list(dict.fromkeys(sens["scope"]))
    for (quantity, std), r in wide.iterrows():
        parts = []
        for scope in scopes:
            e, lo, hi = r[("estimate", scope)], r[("lo", scope)], r[("hi", scope)]
            parts.append(f"{e:8.4f}" + (f" [{lo:.4f}, {hi:.4f}]" if np.isfinite(lo) else " " * 18))
        print(f"  {quantity:<78s} {std:>5g}  " + "   ".join(parts))
    print(f"  (columns: {' | '.join(scopes)})")
    print(f"\nwrote {(REVIEW / 'ceiling_clean_runs.csv').relative_to(REPO)} and "
          f"{(REVIEW / 'ceiling_sensitivity.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
