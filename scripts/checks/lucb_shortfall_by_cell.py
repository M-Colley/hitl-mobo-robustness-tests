"""Where the best-arm (LUCB) sitting falls behind one look each: its shortfall per magnitude and onset.

Appendix app:remedies says the corrected best-arm sitting of eight (full idiosyncratic SD) trails one look each
to the top eight by 0.014 of the achievable improvement, "most of it at 5 sigma from trial 21". This splits that
paired shortfall (LUCB's deployed regret less the one-look sitting's, per run, in units of opt_z; landscape
means, then the mean over landscapes) by cell, from replay_hitl_remedies' main-sweep per-run table (LogEI and
qNEI, four processes, seeds 7-16).

    python scripts/checks/lucb_shortfall_by_cell.py > output-boba/analysis/review/register_checks/lucb_shortfall_by_cell.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import pandas as pd  # noqa: E402

import analyse_boba_adaptations as aba  # noqa: E402

PER_RUN = REPO / "output-boba" / "analysis" / "hitl_remedies" / "hitl_remedies_per_run.csv.gz"


def main() -> None:
    opt_z = aba.dataset_scale("opt_z")
    p = pd.read_csv(PER_RUN, low_memory=False)
    print("The best-arm sitting's shortfall against one look each (LUCB regret - one-look regret, per run / opt_z;")
    print("landscape means, then over landscapes; positive = the best-arm sitting ships worse); main sweep, LogEI and qNEI,")
    print("four processes, seeds 7-16")
    for k, rho in ((8, "1"), (8, "0.5"), (4, "1"), (4, "0.5")):
        a = p[p.procedure == f"lucb_k{k}_rho{rho}"].set_index("file")
        b = p[p.procedure == f"tournament_k{k}_lcb_rho{rho}_look"].set_index("file")
        if len(a) != len(b) or not a.index.sort_values().equals(b.index.sort_values()):
            raise ValueError(f"k={k}, rho={rho}: the two procedures do not cover the same runs")
        b = b.loc[a.index]
        f = a[["dataset", "jitter_std", "jitter_iteration"]].assign(
            d=(a.regret_noisy - b.regret_noisy) / a.dataset.map(opt_z))
        cell = f.groupby(["jitter_std", "jitter_iteration", "dataset"]).d.mean().groupby(
            ["jitter_std", "jitter_iteration"]).mean()
        pooled = f.groupby("dataset").d.mean().mean()
        print(f"-- k = {k}, rho = {rho}: pooled {pooled:+.4f}; share of the summed cell shortfall at 5 sigma from "
              f"trial 21 {cell.loc[(5.0, 20)] / cell.sum():.2f}")
        for (s, o), v in cell.items():
            print(f"   {s:g} sigma from trial {int(o) + 1:<3d} {v:+.4f}")


if __name__ == "__main__":
    main()
