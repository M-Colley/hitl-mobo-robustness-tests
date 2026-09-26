"""The headline in matched units: the deployed cost on the landscapes whose
opt_z lies within the range of the three human studies' fitted objectives.

The currency test rejects the landscape SD as a neutral unit (the cost grows
with opt_z), so the paper also reports the deployed cost of error from the
first rating on the analytic landscapes whose opt_z matches the human studies',
interpolated log-linearly between the grid magnitudes to the three measured
noise levels, with a landscape bootstrap. It also prints the clean twin's own
deployed regret, the "13% short of the optimum" of Section 4.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import boba_benchmarks as bb  # noqa: E402

GRID = np.array([0.05, 0.25, 1.0, 5.0])
BOOT = 4000


def interpolate(row: pd.Series, sigma: float) -> float:
    values = row[GRID].to_numpy(dtype=float)
    j = int(np.searchsorted(GRID, sigma))
    lo, hi = GRID[j - 1], GRID[j]
    w = (np.log(sigma) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return float(values[j - 1] + w * (values[j] - values[j - 1]))


def main() -> None:
    stats = bb.load_stats()
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if isinstance(v, dict) and "opt_z" in v}
    cells = pd.read_csv(REPO / "output-boba" / "analysis" / "cell_means.csv")
    cells = cells[~cells.acquisition.isin(["random", "sobol"])]
    cells = cells.assign(deployed=cells.inference_excess / cells.dataset.map(opt_z))
    per = cells[cells.jitter_iteration == 0].groupby(["dataset", "jitter_std"])["deployed"].mean().unstack()

    anchor = pd.read_csv(REPO / "output" / "noise_anchor.csv").set_index("dataset")
    levels = {name: float(anchor.loc[name, "noise_in_landscape_sd"]) for name in anchor.index}
    human = [float(anchor.loc[name, "opt_z"]) for name in anchor.index]
    lo, hi = min(human), max(human)
    matched = [d for d in per.index if lo <= opt_z[d] <= hi]
    print(f"human opt_z {dict(zip(anchor.index, np.round(human, 3)))}; matched range [{lo:.3f}, {hi:.3f}]")
    print(f"matched landscapes ({len(matched)}): {', '.join(sorted(matched))}")

    rng = np.random.default_rng(20260925)
    for label, subset in (("all", list(per.index)), ("matched", matched)):
        block = per.loc[subset]
        for name, sigma in [("1 sigma", 1.0)] + [(n, s) for n, s in sorted(levels.items(), key=lambda kv: kv[1])]:
            values = np.array([interpolate(block.loc[d], sigma) for d in block.index])
            boot = values[rng.integers(0, len(values), (BOOT, len(values)))].mean(axis=1)
            print(f"{label:8s} {name:11s} sigma={sigma:.3f}  mean={values.mean():.4f} "
                  f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]  "
                  f"min={values.min():.4f} max={values.max():.4f} n={len(values)}")

    rows = []
    for path in glob.glob(str(REPO / "output-boba" / "*" / "evaluation" / "paired_excess_metrics.csv")):
        rows.append(pd.read_csv(path, usecols=lambda c: c in (
            "dataset", "acquisition", "seed", "jitter_iteration", "final_inference_simple_regret_true_baseline")))
    runs = pd.concat(rows, ignore_index=True)
    runs = runs[~runs.acquisition.isin(["random", "sobol"]) & runs.dataset.isin(list(per.index))]
    clean = runs[runs.jitter_iteration == 0].drop_duplicates(["dataset", "acquisition", "seed"])
    clean = clean.assign(share=clean.final_inference_simple_regret_true_baseline / clean.dataset.map(opt_z))
    by_landscape = clean.groupby("dataset")["share"].mean()
    print(f"clean twin's deployed regret, share of the achievable improvement: mean over landscapes "
          f"{by_landscape.mean():.4f} ({len(clean)} clean runs, {len(by_landscape)} landscapes)")


if __name__ == "__main__":
    main()
