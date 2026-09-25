"""Does the fitted-oracle design preserve the ranking of acquisitions?

Companion to ``oracle_isolation.py``. Within every (landscape, magnitude) cell
the four acquisitions run in both arms (EI, LogEI, UCB, qNEI) are ranked by
their trajectory loss, once in the fitted-oracle arm (``output-oracle-iso``,
seeds 7-11, the run's own ``auc_simple_regret_excess_true``) and once in the
exact arm (``output-boba/analysis/cell_means.csv``, GAUSSIAN error from the
first rating, seeds 7-16). Ranks are taken within a cell, so the normaliser of
either metric cancels. The script reports the mean rank of each acquisition per
arm, the Spearman correlation of the two mean-rank orders (Pearson on ranks,
four items, descriptive only), and how often the best acquisition of a
landscape agrees between the arms. Needs only pandas.

    python scripts/oracle_isolation_acq_ranking.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

ACQS = ["ei", "logei", "ucb", "qnei"]
MULTIPLES = [0.25, 1.0, 5.0]


def load_fitted(iso_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(iso_dir / "runs" / "iso_*" / "bo_sensor_error_excess_summary.csv")))
    if not files:
        raise SystemExit(f"no excess summaries under {iso_dir}/runs")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d["acquisition"].isin(ACQS)].copy()
    d["landscape"] = d["dataset"].str.replace("iso_", "", regex=False)
    manifest = pd.read_csv(iso_dir / "manifest.csv")
    lookup = {}
    for _, row in manifest.iterrows():
        stds = [float(x) for x in str(row["jitter_stds"]).split(",")]
        for std, mult in zip(stds, MULTIPLES):
            lookup[(row["landscape"], round(std, 6))] = mult
    d["mult"] = [lookup[(l, round(s, 6))] for l, s in zip(d["landscape"], d["jitter_std"])]
    fit = d.groupby(["landscape", "mult", "acquisition"], as_index=False)["auc_simple_regret_excess_true"].mean()
    fit["rank_fitted"] = fit.groupby(["landscape", "mult"])["auc_simple_regret_excess_true"].rank()
    return fit


def load_exact(cell_means: Path) -> pd.DataFrame:
    e = pd.read_csv(cell_means)
    e = e[(e["error_model"] == "gaussian") & (e["jitter_iteration"] == 0)
          & (e["acquisition"].isin(ACQS)) & (e["jitter_std"].isin(MULTIPLES))].copy()
    e = e.rename(columns={"dataset": "landscape", "jitter_std": "mult"})
    e["rank_exact"] = e.groupby(["landscape", "mult"])["fragility"].rank()
    return e[["landscape", "mult", "acquisition", "rank_exact"]]


def spearman_of_means(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=Path, default=Path("output-oracle-iso"))
    parser.add_argument("--cell-means", type=Path, default=Path("output-boba/analysis/cell_means.csv"))
    parser.add_argument("--out", type=Path, default=Path("output-oracle-iso/oracle_isolation_acq_ranking.csv"))
    args = parser.parse_args()

    merged = load_fitted(args.iso).merge(load_exact(args.cell_means), on=["landscape", "mult", "acquisition"])
    rows = []
    for label, sub in [(f"{m:g}sigma", merged[merged["mult"] == m]) for m in MULTIPLES] + [("pooled", merged)]:
        means = sub.groupby("acquisition")[["rank_fitted", "rank_exact"]].mean()
        best_fit = sub.loc[sub.groupby(["landscape", "mult"])["rank_fitted"].idxmin()].set_index(["landscape", "mult"])["acquisition"]
        best_exact = sub.loc[sub.groupby(["landscape", "mult"])["rank_exact"].idxmin()].set_index(["landscape", "mult"])["acquisition"]
        agree = int((best_fit == best_exact.reindex(best_fit.index)).sum())
        row = {"magnitude": label, "n_cells": int(sub.groupby(["landscape", "mult"]).ngroups),
               "spearman_mean_ranks": round(spearman_of_means(means["rank_fitted"], means["rank_exact"]), 3),
               "best_acquisition_agrees": agree}
        for acq in ACQS:
            row[f"mean_rank_fitted_{acq}"] = round(float(means.loc[acq, "rank_fitted"]), 2)
            row[f"mean_rank_exact_{acq}"] = round(float(means.loc[acq, "rank_exact"]), 2)
        rows.append(row)
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
