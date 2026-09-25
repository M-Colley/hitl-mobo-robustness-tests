"""Main-effect variance shares of the trajectory loss, for the factor sentence of Section 4.

Reads ``output-boba/analysis/cell_means.csv`` (one row per error process,
magnitude, onset, landscape and acquisition, the mean over seeds), drops the
model-free floors, and reports for each factor the share of the variance of the
opt_z-normalised trajectory loss (``fragility``) that its one-way means explain.
The design is balanced (200 cells per condition), so these are the additive
main-effect shares; what they leave is interaction. The same is repeated inside
the 1 sigma, first-rating cell. Needs only pandas.

    python scripts/factor_variance_shares.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FACTORS = ["jitter_std", "jitter_iteration", "dataset", "acquisition", "error_model"]
NAMES = {"jitter_std": "magnitude", "jitter_iteration": "onset", "dataset": "landscape",
         "acquisition": "acquisition", "error_model": "error process"}


def shares(frame: pd.DataFrame, factors: list[str]) -> dict[str, float]:
    y = frame["fragility"]
    sst = float(((y - y.mean()) ** 2).sum())
    out = {}
    for col in factors:
        g = frame.groupby(col)["fragility"].transform("mean")
        out[NAMES[col]] = float(((g - y.mean()) ** 2).sum() / sst)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-means", type=Path, default=Path("output-boba/analysis/cell_means.csv"))
    parser.add_argument("--out", type=Path, default=Path("output-boba/analysis/factor_variance_shares.csv"))
    args = parser.parse_args()
    d = pd.read_csv(args.cell_means)
    d = d[~d["acquisition"].isin(["random", "sobol"])]
    rows = []
    for name, s in shares(d, FACTORS).items():
        rows.append({"scope": "all cells", "factor": name, "variance_share": round(s, 3)})
    cell = d[(d["jitter_std"] == 1.0) & (d["jitter_iteration"] == 0)]
    for name, s in shares(cell, ["dataset", "acquisition", "error_model"]).items():
        rows.append({"scope": "1 sigma, first rating", "factor": name, "variance_share": round(s, 3)})
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
