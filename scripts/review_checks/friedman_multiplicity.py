"""Multiplicity correction of the 32 Friedman tests behind the acquisition ranking.

Section 6 says the Friedman test over landscapes reaches p < 0.05 in 28 of 32
conditions (4 error processes x 4 magnitudes x 2 onsets), the same 28 after a
Benjamini-Hochberg correction over the 32, and 25 after Holm's. The raw p-values
are the ``__friedman__`` rows of output-boba/analysis/acquisition_tests.csv,
written by analyse_boba_robustness.acquisition_rankings: per condition, the ten
model-based acquisitions ranked on the trajectory loss (post-onset per-iteration
excess, as a fraction of opt_z) within each of the twenty landscapes. That file's
``wilcoxon_p_fdr`` column on those rows is the raw p (the pairwise BH there is
within a condition, not across the 32), so the correction over conditions is
done here with statsmodels' multipletests (BH, Holm and Bonferroni).

    python scripts/review_checks/friedman_multiplicity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
ALPHA = 0.05
METHODS = (("raw", None), ("BH", "fdr_bh"), ("Holm", "holm"), ("Bonferroni", "bonferroni"))


def corrected(tests: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """The Friedman rows with each correction's adjusted p and rejection flag."""
    rows = tests[tests["compared_with"] == "__friedman__"].copy()
    rows = rows.dropna(subset=["wilcoxon_p"]).rename(columns={"wilcoxon_p": "p_raw"})
    keep = ["error_model", "jitter_std", "jitter_iteration", "p_raw", "kendall_w", "n_benchmarks"]
    rows = rows[[c for c in keep if c in rows.columns]].reset_index(drop=True)
    for label, method in METHODS:
        if method is None:
            rows["reject_raw"] = rows["p_raw"] < alpha
            continue
        reject, p_adj, _, _ = multipletests(rows["p_raw"].to_numpy(), alpha=alpha, method=method)
        rows[f"p_{label.lower()}"] = p_adj
        rows[f"reject_{label.lower()}"] = reject
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", type=Path, default=REPO / "output-boba" / "analysis" / "acquisition_tests.csv")
    p.add_argument("--csv", type=Path,
                   default=REPO / "output-boba" / "analysis" / "review" / "friedman_multiplicity.csv")
    args = p.parse_args(argv)
    rows = corrected(pd.read_csv(args.tests))
    n = len(rows)
    out = [f"FRIEDMAN TESTS OVER LANDSCAPES, {n} CONDITIONS (trajectory loss, ten acquisitions)",
           "=" * 78,
           f"source: {args.tests.relative_to(REPO) if args.tests.is_relative_to(REPO) else args.tests}",
           f"alpha = {ALPHA}; corrections over the {n} conditions (statsmodels multipletests)", ""]
    for label, _ in METHODS:
        out.append(f"  {label:<11s} significant in {int(rows[f'reject_{label.lower()}'].sum()):>2d} of {n}")
    if "kendall_w" in rows:
        out.append(f"  median Kendall's W over the {n} conditions: {rows['kendall_w'].median():.3f}")
    out.append("")
    out.append(f"  {'process':<9s} {'sigma':>6s} {'onset':>5s} {'p raw':>10s} {'p BH':>10s} "
               f"{'p Holm':>10s} {'W':>6s}")
    for _, r in rows.sort_values(["jitter_iteration", "error_model", "jitter_std"]).iterrows():
        flag = "" if r["reject_holm"] else ("  <- lost under Holm" if r["reject_bh"] else
                                             "  <- not significant")
        out.append(f"  {r['error_model']:<9s} {r['jitter_std']:>6g} {int(r['jitter_iteration']):>5d} "
                   f"{r['p_raw']:>10.2e} {r['p_bh']:>10.2e} {r['p_holm']:>10.2e} "
                   f"{r['kendall_w']:>6.3f}{flag}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.csv, index=False)
    out.append("")
    out.append(f"wrote {args.csv.relative_to(REPO) if args.csv.is_relative_to(REPO) else args.csv}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
