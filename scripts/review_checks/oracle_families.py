"""The oracle-isolation result by oracle family.

The pipeline's model selection picks a tree ensemble on all twenty synthetic
archival datasets (output-oracle-iso). To test whether the understatement is a
property of piecewise-constant oracles, the same datasets were rerun against a
forced Gaussian process and a forced two-layer MLP, without jitter augmentation
(output-oracle-iso-gaussian_process, output-oracle-iso-mlp; see
scripts/oracle_isolation.py --family). For each family this prints the oracle's
fidelity (median correlation with the true landscape at fresh points, and its
range), its grouped cross-validated R2, and, per magnitude and response, the
ratio of landscape means of the fitted over the exact floor-gap fraction with
the analysis's landscape-bootstrap interval, the landscapes on which the fitted
design reports less, the Wilcoxon p, and the Spearman correlation of fidelity
with the per-landscape ratio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
ROOTS = {
    "tree ensemble (selected)": REPO / "output-oracle-iso",
    "gaussian_process": REPO / "output-oracle-iso-gaussian_process",
    "mlp": REPO / "output-oracle-iso-mlp",
}


def cv_r2(root: Path) -> list[float]:
    values = []
    for path in sorted((root / "selection").glob("iso_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for dataset in payload["datasets"]:
            entry = dataset["objectives"]["composite"]
            values.append(float(entry["scores"][entry["best_model"]]))
    return values


def main() -> None:
    for family, root in ROOTS.items():
        manifest = root / "manifest.csv"
        if not manifest.is_file():
            print(f"== {family}: no manifest yet")
            continue
        m = pd.read_csv(manifest)
        r2 = cv_r2(root)
        print(f"== {family}: {len(m)} landscapes; models {m.oracle_model.value_counts().to_dict()}")
        print(f"   fidelity median {m.corr_oracle_truth.median():.3f} range [{m.corr_oracle_truth.min():.3f}, "
              f"{m.corr_oracle_truth.max():.3f}]; sigma_f range [{m.sigma_f_fitted.min():.2f}, {m.sigma_f_fitted.max():.2f}]"
              + (f"; grouped CV R2 median {np.median(r2):.3f} range [{min(r2):.3f}, {max(r2):.3f}]" if r2 else ""))
        for suffix, label in (("", "trajectory"), ("_deployed", "deployed")):
            summary_path = root / f"oracle_isolation_summary{suffix}.csv"
            per_path = root / f"oracle_isolation_per_landscape{suffix}.csv"
            if not summary_path.is_file() or not per_path.is_file():
                print(f"   {label}: not analysed yet")
                continue
            summary = pd.read_csv(summary_path).set_index("sigma_multiple")
            per = pd.read_csv(per_path)
            for s, block in per.groupby("sigma_multiple"):
                row = summary.loc[s]
                lower = int((block.frac_fitted < block.frac_exact).sum())
                ratio = block.frac_fitted / block.frac_exact
                rho = spearmanr(block.corr_oracle_truth, ratio).statistic
                print(f"   {label:10s} {s:>4}sigma  fitted {row.frac_fitted:7.4f}  exact {row.frac_exact:7.4f}  "
                      f"ratio {row.ratio_fitted_over_exact:.3f} [{row.ratio_lo:.3f}, {row.ratio_hi:.3f}]  "
                      f"lower {lower}/{len(block)}  p {row.wilcoxon_p:.2g}  spearman(fidelity, ratio) {rho:+.2f}")


if __name__ == "__main__":
    main()
