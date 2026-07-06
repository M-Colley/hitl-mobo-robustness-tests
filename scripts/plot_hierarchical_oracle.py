"""Summary figure for the hierarchical-oracle additional test.

Reads the JSON produced by hierarchical_oracle_test.py and renders a 4-panel
figure:

  A. cold point fidelity (R^2) per dataset x model
  B. cold calibration (cov90) per dataset x model, with the 0.90 target line
  C. between-user variance fraction (ICC) per dataset -- the heterogeneity story
  D. warm per-user updating: pooled tree vs tree_hier_warm (RMSE), where available

Usage:
  python scripts/plot_hierarchical_oracle.py \
      --input output/hierarchical_oracle_test.json \
      --output output/figures/hierarchical_oracle.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["extra_trees", "tree_hier", "hbm_linear", "hbm_rff"]
MODEL_COLORS = {
    "extra_trees": "#1f77b4",
    "tree_hier": "#2ca02c",
    "hbm_linear": "#ff7f0e",
    "hbm_rff": "#d62728",
}


def _collect(payload: dict, objective: str = "composite"):
    rows = {}
    for ds in payload["datasets"]:
        obj = ds["objectives"].get(objective)
        if obj is None:
            continue
        rows[ds["name"]] = obj
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("output") / "hierarchical_oracle_test.json")
    ap.add_argument("--objective", type=str, default="composite")
    ap.add_argument("--output", type=Path, default=Path("output") / "figures" / "hierarchical_oracle.png")
    args = ap.parse_args()

    payload = json.loads(args.input.read_text())
    rows = _collect(payload, args.objective)
    datasets = list(rows.keys())
    present_models = [m for m in MODEL_ORDER if any(m in rows[d]["models"] for d in datasets)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Hierarchical Bayesian oracle vs extra_trees — cold held-out-user protocol "
        f"({args.objective})", fontsize=13, fontweight="bold",
    )
    x = np.arange(len(datasets))
    width = 0.8 / max(len(present_models), 1)

    # Panel A: R^2
    axA = axes[0, 0]
    for i, m in enumerate(present_models):
        vals = [rows[d]["models"].get(m, {}).get("r2", np.nan) for d in datasets]
        axA.bar(x + i * width, vals, width, label=m, color=MODEL_COLORS.get(m))
    axA.axhline(0.0, color="gray", lw=0.8)
    axA.set_title("A. Point fidelity — cold R²  (higher = better)")
    axA.set_ylabel("R²")
    axA.set_xticks(x + width * (len(present_models) - 1) / 2)
    axA.set_xticklabels(datasets, rotation=15)
    axA.legend(fontsize=8, ncol=2)

    # Panel B: coverage90
    axB = axes[0, 1]
    for i, m in enumerate(present_models):
        vals = [rows[d]["models"].get(m, {}).get("coverage90", np.nan) for d in datasets]
        axB.bar(x + i * width, vals, width, label=m, color=MODEL_COLORS.get(m))
    axB.axhline(0.90, color="black", ls="--", lw=1.2, label="0.90 target")
    axB.set_ylim(0.5, 1.0)
    axB.set_title("B. Calibration — 90% coverage  (near 0.90 = honest)")
    axB.set_ylabel("empirical coverage @ 90%")
    axB.set_xticks(x + width * (len(present_models) - 1) / 2)
    axB.set_xticklabels(datasets, rotation=15)
    axB.legend(fontsize=8, ncol=2)

    # Panel C: ICC between-user
    axC = axes[1, 0]
    iccs = [rows[d]["variance_decomposition"].get("icc_between_user") for d in datasets]
    iccs = [np.nan if v is None else v for v in iccs]
    bars = axC.bar(x, iccs, 0.6, color="#6a3d9a")
    for xi, v in zip(x, iccs):
        if not np.isnan(v):
            axC.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axC.set_title("C. Between-user variance fraction (ICC)\nHBM's native heterogeneity estimate")
    axC.set_ylabel("ICC = τ² / (τ² + σ²)")
    axC.set_ylim(0, max([v for v in iccs if not np.isnan(v)] + [0.1]) * 1.25)
    axC.set_xticks(x)
    axC.set_xticklabels(datasets, rotation=15)

    # Panel D: warm updating (RMSE pooled vs warm)
    axD = axes[1, 1]
    warm_ds, pooled_rmse, warm_rmse = [], [], []
    for d in datasets:
        w = rows[d].get("warm")
        if isinstance(w, dict) and "tree_hier_warm" in w:
            warm_ds.append(d)
            pooled_rmse.append(w["pooled_tree"]["rmse"])
            warm_rmse.append(w["tree_hier_warm"]["rmse"])
    if warm_ds:
        xd = np.arange(len(warm_ds))
        axD.bar(xd - 0.2, pooled_rmse, 0.4, label="pooled tree (ignores context)", color="#1f77b4")
        axD.bar(xd + 0.2, warm_rmse, 0.4, label="tree_hier warm (uses k ratings)", color="#2ca02c")
        k = next(iter(rows[warm_ds[0]]["warm"].values())) if warm_ds else None
        kctx = rows[warm_ds[0]]["warm"]["warm_context"]
        axD.set_title(f"D. Warm per-user updating — RMSE ↓\n(reveal k={kctx} ratings of a held-out user)")
        axD.set_ylabel("RMSE on remaining ratings")
        axD.set_xticks(xd)
        axD.set_xticklabels(warm_ds, rotation=15)
        axD.legend(fontsize=8)
    else:
        axD.axis("off")
        axD.text(0.5, 0.5, "No warm results\n(run with --warm)", ha="center", va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
