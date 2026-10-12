"""The model-free floors' deployed design against each acquisition's, in absolute terms.

Section 6 says that with error from the first rating the floors (random and
Sobol, averaged) ship a design at 1 sigma almost as good as PI's and Greedy's,
and at 5 sigma a better one than four of the ten acquisitions. That sentence is
about the ABSOLUTE regret of the shipped design, which tab:acq does not show:
the table's deployed column is the EXCESS over the clean twin, on which the floor
(whose clean twin visits the same designs) looks better than every acquisition.

Metric: ``final_inference_simple_regret_true_jitter / opt_z``, the true regret
of the design with the best observed rating after the last trial of the noisy
run, as a fraction of the achievable improvement (0 ships the optimum, 1 ships a
design no better than the landscape's mean). Onset 0 (error from the first
rating), the four response-error processes (gaussian, bias, drift, AR(1)), the
main sweep's seeds. Each landscape's value is the mean over processes and seeds,
and the summary is the mean of the twenty landscape means. The floor is the mean
of random and Sobol within each landscape. Beside each acquisition the paired
difference floor - acquisition (positive: the floor ships the worse design) is
given with a percentile bootstrap over landscapes (100,000 resamples; one set of
resamples is shared by the ten acquisitions at a magnitude). The landscapes
share their random numbers by design, so that bootstrap treats correlated
clusters as independent.

Ten acquisitions are compared with the floor at each magnitude, so each
comparison carries three two-sided tests, each Holm-adjusted over the ten
acquisitions at that magnitude and onset (the Holm family): the bootstrap p
(twice the smaller tail mass of the resampled mean at zero, floored at
1/resamples, the definition used by analyse_boba_robustness._cluster_bootstrap;
``p_bootstrap``, ``p_holm``), a one-sample t test of the twenty landscape
differences (``t_p``, ``t_p_holm``; with twenty clusters a t interval is the one
that covers about 95%, scripts/bootstrap_coverage.py) and a Wilcoxon
signed-rank test (``wilcoxon_p``, ``wilcoxon_p_holm``). Until 2026-10-09 the
bootstrap used 2000 resamples, which floors p at 0.0005 and leaves a Holm p near
0.01 to the luck of one draw: LogPI's at 5 sigma printed 0.010, its median over
bootstrap seeds was 0.030, and with 100,000 resamples it is 0.033 (0.031 with
200,000; its t-test and signed-rank Holm p are 0.10 and 0.12). A comparison
is called detectable below only when all three Holm-adjusted p values are below
0.05. The clean twin's absolute deployed regret and the deployed excess are
printed too, so the three quantities can be told apart. Onset 20 is repeated at
the end for completeness.

The deployed excess is also split by the paper's decomposition into its search
part (``final_simple_regret_excess_true / opt_z``, the excess regret of the best
design visited by the final trial) and its selection part (deployed excess minus
search excess). Under response error the floors visit the same designs as their
clean twins, so their search excess is exactly zero and their deployed excess is
selection loss alone; that, not a smaller selection loss, is why they lead
tab:acq's deployed-excess column while shipping a poor design in absolute terms.

Reads the per-landscape evaluation outputs
(output-boba/<landscape>/evaluation/paired_excess_metrics.csv) through
analyse_boba_robustness.load_paired, and, when present, cross-checks the noisy
runs' values against output-boba/bo_synthetic_error_summary.csv.

    python scripts/checks/floor_deployed.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
import analyse_boba_robustness as ab  # noqa: E402
import boba_benchmarks as bb  # noqa: E402

PROCESSES = ("gaussian", "bias", "drift", "ar1")
FLOORS = ("random", "sobol")
# 100,000 resamples: a bootstrap p is floored at 1/REPS, and a Holm adjustment
# over ten multiplies a raw p near 0.003 by up to ten, so 2000 resamples left
# the adjusted p's second digit to chance.
REPS = 100_000
SEED = 20260930
TESTS = {"bootstrap": ("p_bootstrap", "p_holm"), "t": ("t_p", "t_p_holm"),
         "wilcoxon": ("wilcoxon_p", "wilcoxon_p_holm")}
ALPHA = 0.05
NOISY = "final_inference_simple_regret_true_jitter"
CLEAN = "final_inference_simple_regret_true_baseline"
EXCESS = "final_inference_simple_regret_excess_true"
SEARCH = "final_simple_regret_excess_true"
VALUES = ("deployed", "deployed_clean", "deployed_excess", "search_excess", "selection_excess")


def landscape_means(paired: pd.DataFrame, opt_z: dict[str, float], onset: int) -> pd.DataFrame:
    """Per (acquisition, magnitude, landscape): mean over processes and seeds, in opt_z units."""
    sub = paired[(paired["jitter_iteration"] == onset) & paired["error_model"].isin(PROCESSES)].copy()
    scale = sub["dataset"].map(opt_z)
    if scale.isna().any():
        raise ValueError(f"no opt_z for {sorted(sub.loc[scale.isna(), 'dataset'].unique())}")
    sub["deployed"] = sub[NOISY] / scale
    sub["deployed_clean"] = sub[CLEAN] / scale
    sub["deployed_excess"] = sub[EXCESS] / scale
    # The decomposition of the deployed excess: search (best visited) and selection.
    sub["search_excess"] = sub[SEARCH] / scale
    sub["selection_excess"] = sub["deployed_excess"] - sub["search_excess"]
    return (sub.groupby(["acquisition", "jitter_std", "dataset"])
            [list(VALUES)].mean().reset_index())


def floor_table(cells: pd.DataFrame, reps: int = REPS, seed: int = SEED) -> pd.DataFrame:
    """Mean of landscape means for the floor and each acquisition, with paired differences."""
    rows = []
    rng = np.random.default_rng(seed)
    for std, block in cells.groupby("jitter_std"):
        wide = block.pivot_table(index="dataset", columns="acquisition", values="deployed")
        # The other value columns (clean twin, excess and its search/selection split) are
        # carried as landscape means; the decomposition columns are optional so that a
        # frame without them still yields the deployed comparison.
        extra = {v: block.pivot_table(index="dataset", columns="acquisition", values=v)
                 for v in VALUES[1:] if v in block.columns}
        present = [f for f in FLOORS if f in wide.columns]
        if not present:
            continue
        floor = wide[present].mean(axis=1)
        n = len(floor)
        picks = rng.integers(0, n, size=(reps, n))
        rows.append({"jitter_std": float(std), "acquisition": "floor", "deployed": float(floor.mean()),
                     **{v: float(t[present].mean(axis=1).mean()) for v, t in extra.items()},
                     "floor_minus_acq": 0.0, "ci_low": np.nan, "ci_high": np.nan,
                     **{col: np.nan for pair in TESTS.values() for col in pair},
                     "landscapes_floor_better": np.nan, "n_landscapes": n})
        arms = []
        for acq in sorted(c for c in wide.columns if c not in FLOORS):
            diff = (floor - wide[acq]).to_numpy()
            draws = diff[picks].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            p_boot = 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean()))
            p_boot = float(np.clip(p_boot, 1.0 / reps, 1.0))
            nonzero = bool(np.any(diff != 0))
            p_t = float(ttest_1samp(diff, 0.0).pvalue) if nonzero else 1.0
            p_wil = float(wilcoxon(diff).pvalue) if nonzero else 1.0
            arms.append({"jitter_std": float(std), "acquisition": acq,
                         "deployed": float(wide[acq].mean()),
                         **{v: float(t[acq].mean()) for v, t in extra.items()},
                         "floor_minus_acq": float(diff.mean()), "ci_low": float(lo),
                         "ci_high": float(hi), "p_bootstrap": p_boot, "t_p": p_t, "wilcoxon_p": p_wil,
                         "landscapes_floor_better": int((diff < 0).sum()), "n_landscapes": n})
        if arms:
            # Holm over the acquisitions compared with the floor at this magnitude
            # and onset, separately for each test.
            for raw, label in TESTS.values():
                adjusted = multipletests([a[raw] for a in arms], alpha=ALPHA, method="holm")[1]
                for a, value in zip(arms, adjusted):
                    a[label] = float(value)
        rows.extend(arms)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    arms_only = table[table["acquisition"] != "floor"]
    worse = (arms_only.assign(floor_better=lambda t: t["floor_minus_acq"] < 0)
             .groupby("jitter_std")["floor_better"].sum())
    table["acquisitions_worse_than_floor"] = table["jitter_std"].map(worse).astype(int)
    # Detectable after Holm over the ten: the arm ships a better (floor-arm > 0) or a
    # worse (floor-arm < 0) design than the floor, by each test and by all three.
    holm = {name: arms_only[label] < ALPHA for name, (_, label) in TESTS.items()}
    holm["all"] = holm["bootstrap"] & holm["t"] & holm["wilcoxon"]
    for name, hit in holm.items():
        suffix = "" if name == "bootstrap" else f"_{name}"
        for direction, sign in (("better", arms_only["floor_minus_acq"] > 0),
                                ("worse", arms_only["floor_minus_acq"] < 0)):
            count = (hit & sign).groupby(arms_only["jitter_std"]).sum()
            table[f"acquisitions_{direction}_than_floor_holm{suffix}"] = (
                table["jitter_std"].map(count).fillna(0).astype(int))
    return table


def cross_check(paired: pd.DataFrame, summary_path: Path, onset: int) -> str:
    """Largest |difference| between the paired file and the sweep summary on the noisy runs."""
    if not summary_path.is_file():
        return "summary cross-check skipped: no bo_synthetic_error_summary.csv"
    cols = ["dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed",
            "baseline", "final_inference_simple_regret_true"]
    summary = pd.read_csv(summary_path, usecols=cols)
    summary = summary[~summary["baseline"].astype(bool)]
    keys = ["dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed"]
    sub = paired[(paired["jitter_iteration"] == onset) & paired["error_model"].isin(PROCESSES)]
    merged = sub[keys + [NOISY]].merge(summary, on=keys, how="left", validate="one_to_one")
    missing = int(merged["final_inference_simple_regret_true"].isna().sum())
    gap = float(np.nanmax(np.abs(merged[NOISY] - merged["final_inference_simple_regret_true"])))
    return (f"summary cross-check: {len(merged):,} noisy runs matched, {missing} missing, "
            f"largest |difference| {gap:.3e}")


def format_block(table: pd.DataFrame, onset: int) -> list[str]:
    out = [f"onset {onset}: absolute deployed regret / opt_z, mean of landscape means",
           f"  {'sigma':>6s} {'arm':<8s} {'deployed':>9s} {'clean':>7s} {'excess':>7s} "
           f"{'floor-arm':>10s} {'95% CI':>18s} {'p boot':>8s} {'Holm':>7s} {'p t':>8s} {'Holm':>7s} "
           f"{'p Wilc.':>8s} {'Holm':>7s} {'floor better on':>16s}"]
    for std, block in table.groupby("jitter_std"):
        block = block.assign(order=lambda t: np.where(t["acquisition"] == "floor", -1.0,
                                                      t["deployed"]))
        for _, r in block.sort_values("order").iterrows():
            if r["acquisition"] == "floor":
                out.append(f"  {std:>6g} {'floor':<8s} {r['deployed']:>9.3f} {r['deployed_clean']:>7.3f} "
                           f"{r['deployed_excess']:>7.3f}")
            else:
                out.append(f"  {std:>6g} {r['acquisition']:<8s} {r['deployed']:>9.3f} "
                           f"{r['deployed_clean']:>7.3f} {r['deployed_excess']:>7.3f} "
                           f"{r['floor_minus_acq']:>+10.3f} [{r['ci_low']:>+7.3f}, {r['ci_high']:>+7.3f}] "
                           f"{r['p_bootstrap']:>8.5f} {r['p_holm']:>7.4f} "
                           f"{r['t_p']:>8.5f} {r['t_p_holm']:>7.4f} "
                           f"{r['wilcoxon_p']:>8.5f} {r['wilcoxon_p_holm']:>7.4f} "
                           f"{int(r['landscapes_floor_better']):>7d} of {int(r['n_landscapes'])}")
        n_arms = int((block["acquisition"] != "floor").sum())
        first = block.iloc[0]
        tests = "; ".join(
            f"{label}: {int(first[f'acquisitions_better_than_floor_holm{suffix}'])} better, "
            f"{int(first[f'acquisitions_worse_than_floor_holm{suffix}'])} worse"
            for label, suffix in (("bootstrap p", ""), ("t test", "_t"), ("signed-rank", "_wilcoxon"),
                                  ("all three", "_all")))
        out.append(f"  {std:>6g} acquisitions shipping a worse design than the floor: "
                   f"{int(first['acquisitions_worse_than_floor'])} of {n_arms} (point estimates); "
                   f"detectably, after Holm over the {n_arms}: {tests}")
        out.append("")
    return out


def format_split(table: pd.DataFrame, onset: int) -> list[str]:
    """The deployed excess split into search and selection, per arm and magnitude."""
    if not {"search_excess", "selection_excess"} <= set(table.columns):
        return []
    out = [f"onset {onset}: deployed excess / opt_z = search excess + selection excess "
           f"(final trial, mean of landscape means)",
           f"  {'sigma':>6s} {'arm':<8s} {'excess':>7s} {'search':>7s} {'select.':>7s}"]
    for std, block in table.groupby("jitter_std"):
        for _, r in block.sort_values("deployed_excess").iterrows():
            out.append(f"  {std:>6g} {r['acquisition']:<8s} {r['deployed_excess']:>7.3f} "
                       f"{r['search_excess']:>7.3f} {r['selection_excess']:>7.3f}")
        arms = block[block["acquisition"] != "floor"]
        floor = block[block["acquisition"] == "floor"].iloc[0]
        out.append(f"  {std:>6g} floor search excess {floor['search_excess']:.3e}; acquisitions' "
                   f"search {arms['search_excess'].min():.3f} to {arms['search_excess'].max():.3f}, "
                   f"selection {arms['selection_excess'].min():.3f} to "
                   f"{arms['selection_excess'].max():.3f} (floor {floor['selection_excess']:.3f})")
        out.append("")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=REPO / "output-boba")
    p.add_argument("--csv", type=Path, default=REPO / "output-boba" / "analysis" / "review" / "floor_deployed.csv")
    args = p.parse_args(argv)
    paired = ab.load_paired(args.input_dir)
    opt_z = {name: float(entry["opt_z"]) for name, entry in bb.load_stats().items()}
    lines = ["FLOOR vs ACQUISITIONS ON THE DEPLOYED DESIGN (absolute regret, fraction of opt_z)",
             "=" * 80,
             "metric: final_inference_simple_regret_true of the noisy run / opt_z; four response",
             "processes (gaussian, bias, drift, ar1); floor = mean of random and sobol per landscape;",
             f"floor-arm is the paired floor - acquisition difference, landscape bootstrap ({REPS:,} resamples,",
             f"so a bootstrap p is floored at {1 / REPS:.0e});",
             "p boot is its two-sided bootstrap p, p t a one-sample t test and p Wilc. a Wilcoxon signed-rank",
             "test over the twenty landscape differences; each 'Holm' column is the p to its left, Holm-adjusted",
             "over the ten acquisitions at that magnitude and onset (one family per test, magnitude and onset);",
             "'clean' is the clean twin's absolute deployed regret, 'excess' the deployed excess.",
             "Each onset's second block splits that excess into search (final_simple_regret_excess_true",
             "/ opt_z, the best design visited) and selection (the rest).",
             ""]
    frames = []
    for onset in (0, 20):
        table = floor_table(landscape_means(paired, opt_z, onset))
        if table.empty:
            continue
        table.insert(0, "jitter_iteration", onset)
        frames.append(table)
        lines += format_block(table, onset)
        lines.append(cross_check(paired, args.input_dir / "bo_synthetic_error_summary.csv", onset))
        lines.append("")
        lines += format_split(table, onset)
    result = pd.concat(frames, ignore_index=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.csv, index=False)
    lines.append(f"wrote {args.csv.relative_to(REPO) if args.csv.is_relative_to(REPO) else args.csv}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
