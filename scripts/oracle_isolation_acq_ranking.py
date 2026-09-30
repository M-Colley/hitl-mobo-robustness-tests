"""Does the fitted-oracle design preserve the ranking of acquisitions?

Companion to ``oracle_isolation.py``. Within every (landscape, magnitude) cell
the four acquisitions run in both arms (EI, LogEI, UCB, qNEI) are ranked by
their loss, once in the fitted-oracle arm (``output-oracle-iso``, seeds 7-11)
and once in the exact arm (GAUSSIAN error from the first rating). Ranks are
taken within a cell, so the normaliser of either metric cancels. The script
reports the mean rank of each acquisition per arm, the Spearman correlation of
the two mean-rank orders (Pearson on ranks, four items, descriptive only), and
how often the best acquisition of a landscape agrees between the arms.

Two metrics: the trajectory (the run's ``auc_simple_regret_excess_true``; in the
exact arm ``cell_means.csv``'s fragility, the post-onset per-iteration excess
over opt_z, which ranks identically from the first rating) and the deployed
design (``final_inference_simple_regret_excess_true``; in the exact arm
``cell_means.csv``'s inference_excess). The exact arm of the committed
comparison is ``output-boba/analysis/cell_means.csv``, seeds 7-16.

A RELIABILITY CEILING. With four acquisitions and twenty landscapes a low
agreement between the arms can mean that the fitted oracle ranks them
differently or that no ranking is stable at this sample size. The exact arm
against itself on disjoint seed halves (7-11 against 12-16, read from
``output-boba/*/evaluation/paired_excess_metrics.csv``) is the ceiling any
comparison can reach. The reliability file gives, per metric and magnitude and
pooled, the best-acquisition agreement with its chance level (one in four per
cell) and upper-tail binomial p, and the Spearman correlation of the mean-rank
orders, for three comparisons: fitted against exact seeds 7-16 (the committed
one), fitted against exact seeds 7-11 (the seeds the isolation comparison
uses), and exact seeds 7-11 against exact seeds 12-16.

    python scripts/oracle_isolation_acq_ranking.py
    python scripts/oracle_isolation_acq_ranking.py --iso output-oracle-iso-gaussian_process \
        --out output-oracle-iso-gaussian_process/oracle_isolation_acq_ranking.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from scipy.stats import binom

ACQS = ["ei", "logei", "ucb", "qnei"]
MULTIPLES = [0.25, 1.0, 5.0]
# metric -> (the fitted run's column, the exact arm's cell_means column, the exact run's column)
METRICS = {
    "trajectory": ("auc_simple_regret_excess_true", "fragility", "auc_simple_regret_excess_true"),
    "deployed": ("final_inference_simple_regret_excess_true", "inference_excess",
                 "final_inference_simple_regret_excess_true"),
}
HALVES = ((7, 8, 9, 10, 11), (12, 13, 14, 15, 16))


def _multiples(iso_dir: Path) -> dict:
    manifest = pd.read_csv(iso_dir / "manifest.csv")
    lookup = {}
    for _, row in manifest.iterrows():
        stds = [float(x) for x in str(row["jitter_stds"]).split(",")]
        for std, mult in zip(stds, MULTIPLES):
            lookup[(row["landscape"], round(std, 6))] = mult
    return lookup


def _rank(frame: pd.DataFrame, value: str, name: str) -> pd.DataFrame:
    out = frame.groupby(["landscape", "mult", "acquisition"], as_index=False)[value].mean()
    out[name] = out.groupby(["landscape", "mult"])[value].rank()
    return out


def load_fitted(iso_dir: Path, metric: str = "trajectory") -> pd.DataFrame:
    """The fitted arm's ranks. The trajectory reads the per-run excess summaries
    (the committed comparison); the deployed design, which they do not carry,
    reads the evaluated pairs."""
    column = METRICS[metric][0]
    pattern = ("bo_sensor_error_excess_summary.csv" if metric == "trajectory"
               else str(Path("evaluation") / "paired_excess_metrics.csv"))
    files = sorted(glob.glob(str(iso_dir / "runs" / "iso_*" / pattern)))
    if not files:
        raise SystemExit(f"no {pattern} under {iso_dir}/runs")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d["acquisition"].isin(ACQS) & (d["jitter_iteration"] == 0)].copy()
    d["landscape"] = d["dataset"].str.replace("iso_", "", regex=False)
    lookup = _multiples(iso_dir)
    d["mult"] = [lookup[(l, round(s, 6))] for l, s in zip(d["landscape"], d["jitter_std"])]
    return _rank(d, column, "rank_fitted")


def load_exact(cell_means: Path, metric: str = "trajectory") -> pd.DataFrame:
    """The exact arm's ranks from cell_means.csv (seeds 7-16)."""
    column = METRICS[metric][1]
    e = pd.read_csv(cell_means)
    e = e[(e["error_model"] == "gaussian") & (e["jitter_iteration"] == 0)
          & (e["acquisition"].isin(ACQS)) & (e["jitter_std"].isin(MULTIPLES))].copy()
    e = e.rename(columns={"dataset": "landscape", "jitter_std": "mult"})
    e["rank_exact"] = e.groupby(["landscape", "mult"])[column].rank()
    return e[["landscape", "mult", "acquisition", "rank_exact"]]


def load_exact_runs(boba: Path) -> pd.DataFrame:
    """The exact arm's evaluated pairs: gaussian, from the first rating, the four acquisitions."""
    files = sorted(glob.glob(str(boba / "*" / "evaluation" / "paired_excess_metrics.csv")))
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration"] + \
        sorted({v[2] for v in METRICS.values()})
    d = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    d = d[(d["error_model"] == "gaussian") & (d["jitter_iteration"] == 0) & d["acquisition"].isin(ACQS)
          & d["jitter_std"].round(2).isin(MULTIPLES)].copy()
    return d.rename(columns={"dataset": "landscape"}).assign(mult=lambda x: x["jitter_std"].round(2))


def exact_ranks(runs: pd.DataFrame, seeds, metric: str, name: str) -> pd.DataFrame:
    sub = runs[runs["seed"].isin(seeds)]
    per = sub.groupby(["landscape", "mult"])["seed"].nunique()
    if per.min() != len(seeds):
        raise SystemExit(f"exact arm: seeds {sorted(seeds)} incomplete in {int((per < len(seeds)).sum())} cells")
    return _rank(sub, METRICS[metric][2], name)[["landscape", "mult", "acquisition", name]]


def spearman_of_means(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def compare(merged: pd.DataFrame, a: str, b: str, with_means: bool = True) -> pd.DataFrame:
    """Per magnitude and pooled: mean-rank Spearman and best-acquisition agreement of rank columns a and b."""
    rows = []
    for label, sub in [(f"{m:g}sigma", merged[merged["mult"] == m]) for m in MULTIPLES] + [("pooled", merged)]:
        means = sub.groupby("acquisition")[[a, b]].mean()
        best_a = sub.loc[sub.groupby(["landscape", "mult"])[a].idxmin()].set_index(["landscape", "mult"])["acquisition"]
        best_b = sub.loc[sub.groupby(["landscape", "mult"])[b].idxmin()].set_index(["landscape", "mult"])["acquisition"]
        agree = int((best_a == best_b.reindex(best_a.index)).sum())
        n = int(sub.groupby(["landscape", "mult"]).ngroups)
        row = {"magnitude": label, "n_cells": n,
               "spearman_mean_ranks": round(spearman_of_means(means[a], means[b]), 3),
               "best_acquisition_agrees": agree}
        if with_means:
            for acq in ACQS:
                row[f"mean_rank_{a.replace('rank_', '')}_{acq}"] = round(float(means.loc[acq, a]), 2)
                row[f"mean_rank_{b.replace('rank_', '')}_{acq}"] = round(float(means.loc[acq, b]), 2)
        else:
            row["expected_by_chance"] = n / len(ACQS)
            row["p_binomial_at_least"] = round(float(binom.sf(agree - 1, n, 1 / len(ACQS))), 4)
            row["order_a"] = " > ".join(means[a].sort_values().index)   # best first
            row["order_b"] = " > ".join(means[b].sort_values().index)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=Path, default=Path("output-oracle-iso"))
    parser.add_argument("--cell-means", type=Path, default=Path("output-boba/analysis/cell_means.csv"))
    parser.add_argument("--boba", type=Path, default=Path("output-boba"))
    parser.add_argument("--out", type=Path, default=Path("output-oracle-iso/oracle_isolation_acq_ranking.csv"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    runs = load_exact_runs(args.boba)
    reliability = []
    for metric in METRICS:
        fitted = load_fitted(args.iso, metric)
        # The committed comparison, fitted (seeds 7-11) against cell_means (seeds 7-16).
        merged = fitted.merge(load_exact(args.cell_means, metric), on=["landscape", "mult", "acquisition"])
        table = compare(merged, "rank_fitted", "rank_exact")
        out = args.out if metric == "trajectory" else args.out.with_name(f"{args.out.stem}_{metric}{args.out.suffix}")
        table.to_csv(out, index=False)
        print(f"\n== {metric}: fitted (seeds 7-11) against exact (cell_means, seeds 7-16) -> {out}")
        print(table.to_string(index=False))

        first = exact_ranks(runs, HALVES[0], metric, "rank_exact_7_11")
        second = exact_ranks(runs, HALVES[1], metric, "rank_exact_12_16")
        keys = ["landscape", "mult", "acquisition"]
        pairs = {
            "fitted vs exact seeds 7-16 (cell_means)": merged.rename(columns={"rank_exact": "rank_b"}),
            "fitted vs exact seeds 7-11": fitted.merge(first, on=keys).rename(columns={"rank_exact_7_11": "rank_b"}),
            "exact seeds 7-11 vs exact seeds 12-16": first.merge(second, on=keys).rename(
                columns={"rank_exact_7_11": "rank_fitted", "rank_exact_12_16": "rank_b"}),
        }
        for label, frame in pairs.items():
            block = compare(frame, "rank_fitted", "rank_b", with_means=False)
            block.insert(0, "comparison", label)
            block.insert(0, "metric", metric)
            reliability.append(block)
    reliability = pd.concat(reliability, ignore_index=True)
    rel_out = args.out.with_name(args.out.name.replace("acq_ranking", "acq_reliability"))
    reliability.to_csv(rel_out, index=False)
    print(f"\n== agreement against its reliability ceiling -> {rel_out}")
    print(reliability.to_string(index=False))


if __name__ == "__main__":
    main()
