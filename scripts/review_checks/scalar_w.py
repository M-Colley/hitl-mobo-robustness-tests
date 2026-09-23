# D9: Kendall's W over condition-averaged losses, scalar suite.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""D9: the scalar suite's Kendall W 'computed the same way' as analyse_boba_mo.acquisition_agreement."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import analyse_boba_robustness as ar, boba_benchmarks as bb

def w_of(frac, col):
    wide = frac.pivot_table(index="dataset", columns="acquisition", values=col, aggfunc="mean")
    ranks = wide.rank(axis=1); n, k = ranks.shape
    spread = ((ranks.sum(axis=0) - ranks.sum(axis=0).mean()) ** 2).sum()
    return 12 * spread / (n ** 2 * (k ** 3 - k)), n, k

df = ar.attach_landscape(ar.load_paired(REPO / "output-boba"), bb.load_stats(bb.DEFAULT_STATS_PATH))
L = df[~df.acquisition.isin(ar.MODEL_FREE)]
cond = ["error_model", "jitter_std", "jitter_iteration"]
for label, sub in [("all 32 conditions", L), ("gaussian only", L[L.error_model == "gaussian"])]:
    cell = sub.groupby(cond + ["dataset", "acquisition"])[["excess_sd", "fragility"]].mean().reset_index()
    for col in ("excess_sd", "fragility"):
        print(label, col, "W=%.4f n=%d k=%d" % w_of(cell, col))
    # the four hypervolume analogue: restrict to 4 acquisitions? report the EI family subset too
    for acqs in (["logei", "ei", "qei", "qnei"], ["logei", "qnei", "ucb", "ei"]):
        c2 = cell[cell.acquisition.isin(acqs)]
        print(label, "fragility", acqs, "W=%.4f n=%d k=%d" % w_of(c2, "fragility"))
