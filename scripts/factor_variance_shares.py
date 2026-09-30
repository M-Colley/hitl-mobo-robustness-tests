"""Main-effect variance shares of the trajectory loss, for the factor sentence of Section 4.

Reads ``output-boba/analysis/cell_means.csv`` (one row per error process,
magnitude, onset, landscape and acquisition, the mean over seeds), drops the
model-free floors, and reports for each factor the share of the variance of the
trajectory loss that its one-way means explain. The design is balanced (200
cells per condition), so these are the additive main-effect shares; what they
leave is interaction. The same is repeated inside the 1 sigma, first-rating
cell.

The shares are reported on two responses, named in the ``response`` column,
because the ordering of the factors depends on the unit:

* ``fragility`` -- the post-onset per-iteration excess as a fraction of opt_z
  (the paper's "cost"). Dividing by opt_z removes most of the between-landscape
  scale, so on this response the landscape factor is what is LEFT after that
  division. analyse_boba_robustness.py forbids this response in a regression
  that has opt_z, or anything collinear with it, on the right-hand side, and
  landscape indicators determine opt_z. A one-way share is a description, not a
  test, so it is kept here, but the landscape's share on it must not be read as
  "how much the landscape matters".
* ``excess_sd`` -- the same excess in landscape SDs (fragility x opt_z), the
  response of every regression in analyse_boba_robustness.py.

opt_z is read directly from boba_landscape_stats.json (``--stats-path``), the
file boba_benchmarks.load_stats reads, so the script needs pandas only.

    python scripts/factor_variance_shares.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FACTORS = ["jitter_std", "jitter_iteration", "dataset", "acquisition", "error_model"]
NAMES = {"jitter_std": "magnitude", "jitter_iteration": "onset", "dataset": "landscape",
         "acquisition": "acquisition", "error_model": "error process"}
RESPONSES = ("fragility", "excess_sd")


def shares(frame: pd.DataFrame, factors: list[str], response: str = "fragility") -> dict[str, float]:
    """Share of the total sum of squares of ``response`` explained by each factor's one-way means."""
    y = frame[response]
    sst = float(((y - y.mean()) ** 2).sum())
    out = {}
    for col in factors:
        g = frame.groupby(col)[response].transform("mean")
        out[NAMES[col]] = float(((g - y.mean()) ** 2).sum() / sst)
    return out


def load_opt_z(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: float(entry["opt_z"]) for name, entry in payload["functions"].items()}


def share_table(d: pd.DataFrame) -> pd.DataFrame:
    """Rows of (scope, factor, variance_share, response) for both responses and both scopes."""
    d = d[~d["acquisition"].isin(["random", "sobol"])]
    rows = []
    for response in RESPONSES:
        for name, s in shares(d, FACTORS, response).items():
            rows.append({"scope": "all cells", "factor": name, "variance_share": round(s, 3),
                         "response": response})
        cell = d[(d["jitter_std"] == 1.0) & (d["jitter_iteration"] == 0)]
        for name, s in shares(cell, ["dataset", "acquisition", "error_model"], response).items():
            rows.append({"scope": "1 sigma, first rating", "factor": name,
                         "variance_share": round(s, 3), "response": response})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-means", type=Path, default=Path("output-boba/analysis/cell_means.csv"))
    parser.add_argument("--stats-path", type=Path, default=REPO / "boba_landscape_stats.json")
    parser.add_argument("--out", type=Path, default=Path("output-boba/analysis/factor_variance_shares.csv"))
    args = parser.parse_args()
    d = pd.read_csv(args.cell_means)
    opt_z = load_opt_z(args.stats_path)
    scale = d["dataset"].map(opt_z)
    if scale.isna().any():
        raise ValueError(f"no opt_z for {sorted(d.loc[scale.isna(), 'dataset'].unique())}")
    # cell_means.csv holds the seed mean of excess_sd / opt_z; opt_z is constant
    # within a landscape, so multiplying back gives the seed mean of excess_sd.
    d["excess_sd"] = d["fragility"] * scale
    table = share_table(d)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
