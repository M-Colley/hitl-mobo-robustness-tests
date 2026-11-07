"""Does the selection share of the deployed loss depend on the height of the optimum? (Section 5, Contributions)

Reads the per-landscape headline cell (1 sigma from the first rating, four processes and ten model-based
acquisitions pooled, seeds 7-16) that smooth_subset_and_gp_diagnostics.py writes for the per-landscape figure,
and prints the Spearman rank correlation over the twenty landscapes between opt_z (the optimum's height above the
landscape mean in landscape SDs) and (a) the selection share of the deployed excess, (b) the deployed excess as a
fraction of opt_z. Proposed by a mock reviewer (2026-11-07) as the transferable form of the selection-share result.

    python scripts/checks/selection_share_optz.py > output-boba/analysis/review/register_checks/selection_share_optz.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
PER_LANDSCAPE = REPO / "output-boba" / "analysis" / "review" / "per_landscape_headline.csv"


def headline(per: pd.DataFrame) -> pd.DataFrame:
    flag = per["headline_cell"].astype(str).str.lower() == "true"
    h = per[flag]
    if len(h) != 20 or set(h["error_model"]) != {"pooled"} or set(h["jitter_std"]) != {1.0} \
            or set(h["jitter_iteration"]) != {0}:
        raise ValueError(f"{PER_LANDSCAPE.name}: expected the 1-sigma, onset-0 pooled cell on twenty landscapes")
    return h.sort_values("opt_z")


def main() -> None:
    h = headline(pd.read_csv(PER_LANDSCAPE))
    share = spearmanr(h["opt_z"], h["selection_share"])
    deployed = spearmanr(h["opt_z"], h["deployed"])
    print("Selection share against opt_z, 1 sigma from the first rating (four processes, ten acquisitions, seeds 7-16),")
    print("per landscape, ordered by opt_z:")
    for r in h.itertuples(index=False):
        print(f"  {r.landscape:<20s} opt_z {r.opt_z:7.2f}  deployed {r.deployed:.3f}  selection share {r.selection_share:.3f}")
    print(f"Spearman(opt_z, selection share) = {share.statistic:+.2f} (p = {share.pvalue:.1g}, n = {len(h)})")
    print(f"Spearman(opt_z, deployed excess / opt_z) = {deployed.statistic:+.2f} (p = {deployed.pvalue:.2f})")


if __name__ == "__main__":
    main()
