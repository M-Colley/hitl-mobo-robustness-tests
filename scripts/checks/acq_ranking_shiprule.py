"""Does the acquisition ranking on the deployed design depend on the ship rule? (Section 6; review round 2026-11-07)

tab:acq ranks the ten model-based acquisitions on the deployed design shipped by the best observed rating. A
mock reviewer asked whether the ranking survives a noise-aware ship rule. This re-ranks them, exactly as
make_boba_paper_tables._deployed_ranks does (rank within each condition and landscape, then the mean rank over
both), on the cost of error under each rule: the rule's noisy deployed regret less the same rule's regret on the
acquisition's own clean twin, as a fraction of opt_z, from rescore_ship_rules' per-run table (seeds 7-16, the
four processes, every magnitude and onset). It prints the mean ranks under the best observed rating (which must
reproduce tab:acq's deployed column), the posterior-mean maximiser and LCB2, and the Spearman correlations
between the three rankings and with the trajectory ranking of overall_acquisition_rankings.csv.

    python scripts/checks/acq_ranking_shiprule.py > output-boba/analysis/review/register_checks/acq_ranking_shiprule.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import boba_benchmarks as bb  # noqa: E402
import make_boba_paper_tables as mbt  # noqa: E402

ANALYSIS = REPO / "output-boba" / "analysis"
RULES = {"best observed rating": "best_observed", "posterior-mean maximiser": "pm", "LCB2": "lcb2"}
KEYS = ["error_model", "jitter_std", "jitter_iteration", "dataset"]


def main() -> None:
    s = pd.read_csv(ANALYSIS / "ship_rules_per_run.csv", low_memory=False)
    s["baseline"] = s["baseline"].astype(str).str.lower().isin(("true", "1"))
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    regret = [f"regret_{r}" for r in RULES.values()]
    clean = s[s["baseline"]][["dataset", "acquisition", "seed"] + regret]
    noisy = s[~s["baseline"]].merge(clean, on=["dataset", "acquisition", "seed"], suffixes=("", "_clean"),
                                    validate="many_to_one")
    z = noisy["dataset"].map(opt_z)
    ranks = {}
    for label, r in RULES.items():
        noisy[f"excess_{r}"] = (noisy[f"regret_{r}"] - noisy[f"regret_{r}_clean"]) / z
        cells = noisy.groupby(KEYS + ["acquisition"], as_index=False)[f"excess_{r}"].mean()
        cells["rank"] = cells.groupby(KEYS)[f"excess_{r}"].rank()
        ranks[label] = cells.groupby("acquisition").agg(rank=("rank", "mean"), excess=(f"excess_{r}", "mean"))
    # The best-observed ranking must be tab:acq's deployed column.
    published = mbt._deployed_ranks(ANALYSIS)
    gap = float((ranks["best observed rating"]["rank"] - published["deployed_rank"]).abs().max())
    if gap > 1e-9:
        raise ValueError(f"the best-observed ranking does not reproduce tab:acq's deployed ranks ({gap:g})")
    traj = pd.read_csv(ANALYSIS / "overall_acquisition_rankings.csv").set_index("acquisition")["mean_rank"]
    table = pd.DataFrame({label: ranks[label]["rank"] for label in RULES}).assign(trajectory=traj)
    table = table.sort_values("LCB2")
    print("Mean rank of the ten acquisitions on the deployed design's cost of error, by ship rule (rank within each")
    print("condition and landscape, then the mean; seeds 7-16, four processes, every magnitude and onset; lower is")
    print("more robust); the best-observed column reproduces tab:acq's deployed ranks.")
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.to_string())
    print()
    cols = list(table.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = spearmanr(table[a], table[b]).statistic
            print(f"Spearman({a}, {b}) = {rho:+.2f}")
    print()
    for label in RULES:
        e = ranks[label]["excess"]
        print(f"{label}: mean cost of error over acquisitions {e.mean():.3f} of opt_z (range {e.min():.3f} to {e.max():.3f})")


if __name__ == "__main__":
    main()
