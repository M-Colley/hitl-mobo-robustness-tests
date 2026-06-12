"""Estimate within-rater rating noise from the source datasets.

Purpose: ground the simulation's --jitter-stds sweep in empirical human
rating variability instead of arbitrary magnitudes. Two estimators:

1. Exact re-presentations (test-retest): within one rater (= one observation
   file), groups of evaluations with identical parameter vectors. Groups
   whose objective values are also bit-identical are treated as logging
   artifacts and excluded; the remaining groups are genuine repeated ratings
   of the same design, and their pooled within-group SD estimates the
   intra-rater noise directly.

2. Nearest-neighbor nugget: within one rater, each evaluation is paired with
   its nearest other evaluation in [0,1]-normalized parameter space. For
   close pairs the true objective difference is ~0, so SD(rating diff)/sqrt(2)
   estimates the noise SD (a semivariogram-nugget style estimator). Reported
   for "close" pairs (distance <= 0.05 * sqrt(d)) and for all NN pairs
   (upper bound: includes real design effects).

Outputs a per-dataset table (composite and per-construct), and a recommended
--jitter-stds grid expressed as {0.5, 1, 2, 4} x the composite noise SD.

Example:
  python scripts/calibrate_noise_from_data.py --output-path output/noise_calibration.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bo_sensor_error_simulation as bo_sim  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=Path(".dataset_cache"),
        help="Local cache directory for remote dataset repositories.",
    )
    parser.add_argument(
        "--close-pair-fraction",
        type=float,
        default=0.05,
        help="NN pairs closer than this fraction of the maximal normalized "
        "distance sqrt(d) count as 'close' (near-replicates).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output") / "noise_calibration.csv",
    )
    return parser.parse_args()


def pooled_repeat_sd(values_by_group: list[np.ndarray]) -> tuple[float, int]:
    """Pooled SD over repeat groups: sqrt(sum((n_i-1) var_i) / sum(n_i-1))."""
    ss = 0.0
    dof = 0
    for values in values_by_group:
        n = len(values)
        if n < 2:
            continue
        ss += float(np.var(values, ddof=1)) * (n - 1)
        dof += n - 1
    if dof == 0:
        return float("nan"), 0
    return float(np.sqrt(ss / dof)), dof


def exact_repeat_groups(
    rater_df: pd.DataFrame,
    param_columns: list[str],
    value_columns: list[str],
) -> tuple[list[pd.DataFrame], int]:
    """Genuine re-presentations: identical params, non-identical objectives.

    Returns (groups, n_artifact_groups). Groups where every objective value
    is bit-identical across rows are counted as logging artifacts.
    """
    groups: list[pd.DataFrame] = []
    artifacts = 0
    for _, group in rater_df.groupby(param_columns, sort=False):
        if len(group) < 2:
            continue
        if group[value_columns].round(12).drop_duplicates().shape[0] == 1:
            artifacts += 1
            continue
        groups.append(group)
    return groups, artifacts


def nn_pairs(
    rater_df: pd.DataFrame,
    param_columns: list[str],
    bounds: bo_sim.Bounds,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique nearest-neighbor pairs within one rater.

    Returns (pair_indices [m, 2], distances [m]) in normalized parameter space.
    """
    X = rater_df[param_columns].to_numpy(dtype=float)
    span = np.where(bounds.high > bounds.low, bounds.high - bounds.low, 1.0)
    Xn = (X - bounds.low) / span
    n = len(Xn)
    if n < 2:
        return np.empty((0, 2), dtype=int), np.empty(0)
    dists = np.linalg.norm(Xn[:, None, :] - Xn[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argmin(dists, axis=1)
    pairs = {tuple(sorted((i, int(nn_idx[i])))) for i in range(n)}
    pair_arr = np.array(sorted(pairs), dtype=int)
    return pair_arr, dists[pair_arr[:, 0], pair_arr[:, 1]]


def drop_full_duplicates(
    df: pd.DataFrame, param_columns: list[str], value_columns: list[str]
) -> pd.DataFrame:
    """Drop rows that are exact (rater, params, objectives) duplicates.

    These are logging artifacts; keeping them would deflate the NN-nugget
    estimate with artificial zero-difference pairs.
    """
    key = [bo_sim.OBSERVATION_SOURCE_COLUMN] + param_columns + value_columns
    return df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def calibrate_dataset(
    dataset: bo_sim.DatasetConfig,
    close_pair_fraction: float,
) -> list[dict[str, object]]:
    objective_columns = dataset.objective_map["composite"]
    df = bo_sim.load_observations(dataset, "composite")
    base_columns = [
        bo_sim._objective_base_column(column) for column in objective_columns
    ]
    df = drop_full_duplicates(df, dataset.param_columns, base_columns)
    bounds = bo_sim.bounds_from_data(df, dataset.param_columns)

    # Value columns analyzed: the unweighted composite (what the simulation
    # perturbs for objective=composite) plus each raw construct.
    df = df.copy()
    df["__composite"] = bo_sim.compute_objective(df, objective_columns, False, None)
    targets = [("composite", "__composite")] + [
        (base, base) for base in base_columns
    ]

    close_threshold = close_pair_fraction * np.sqrt(len(dataset.param_columns))
    rows: list[dict[str, object]] = []

    for label, column in targets:
        repeat_values: list[np.ndarray] = []
        artifact_groups = 0
        diffs_close: list[float] = []
        diffs_all: list[float] = []
        nn_distances: list[float] = []

        for _, rater_df in df.groupby(bo_sim.OBSERVATION_SOURCE_COLUMN, sort=False):
            groups, artifacts = exact_repeat_groups(
                rater_df, dataset.param_columns, [column]
            )
            artifact_groups += artifacts
            repeat_values.extend(
                group[column].to_numpy(dtype=float) for group in groups
            )

            pair_arr, dists = nn_pairs(rater_df, dataset.param_columns, bounds)
            if len(pair_arr) == 0:
                continue
            values = rater_df[column].to_numpy(dtype=float)
            pair_diffs = values[pair_arr[:, 0]] - values[pair_arr[:, 1]]
            nn_distances.extend(dists.tolist())
            diffs_all.extend(pair_diffs.tolist())
            diffs_close.extend(pair_diffs[dists <= close_threshold].tolist())

        sd_repeat, repeat_dof = pooled_repeat_sd(repeat_values)
        sd_nn_close = (
            float(np.std(diffs_close, ddof=1) / np.sqrt(2.0))
            if len(diffs_close) >= 2
            else float("nan")
        )
        sd_nn_all = (
            float(np.std(diffs_all, ddof=1) / np.sqrt(2.0))
            if len(diffs_all) >= 2
            else float("nan")
        )
        values_all = df[column].to_numpy(dtype=float)
        value_range = float(np.nanmax(values_all) - np.nanmin(values_all))

        rows.append(
            {
                "dataset": dataset.name,
                "objective": label,
                "n_rows": int(len(df)),
                "n_raters": int(df[bo_sim.OBSERVATION_SOURCE_COLUMN].nunique()),
                "n_repeat_groups": int(len(repeat_values)),
                "n_artifact_duplicate_groups": int(artifact_groups),
                "repeat_dof": int(repeat_dof),
                "sd_repeat": sd_repeat,
                "n_close_pairs": int(len(diffs_close)),
                "sd_nn_close": sd_nn_close,
                "n_nn_pairs": int(len(diffs_all)),
                "sd_nn_all": sd_nn_all,
                "median_nn_distance": float(np.median(nn_distances))
                if nn_distances
                else float("nan"),
                "close_pair_threshold": float(close_threshold),
                "value_range": value_range,
            }
        )
    return rows


def preferred_sd(row: pd.Series) -> tuple[float, str]:
    """Pick the most trustworthy estimator available for a dataset."""
    if row["repeat_dof"] >= 10 and np.isfinite(row["sd_repeat"]):
        return float(row["sd_repeat"]), "exact re-presentations"
    if row["n_close_pairs"] >= 30 and np.isfinite(row["sd_nn_close"]):
        return float(row["sd_nn_close"]), "close NN pairs"
    return float(row["sd_nn_all"]), "all NN pairs (upper bound)"


def main() -> None:
    args = parse_args()
    datasets = bo_sim.parse_dataset_configs(
        None, args.dataset_config, args.dataset_cache_dir
    )

    all_rows: list[dict[str, object]] = []
    for dataset in datasets:
        all_rows.extend(calibrate_dataset(dataset, args.close_pair_fraction))

    table = pd.DataFrame(all_rows)
    table["sd_as_pct_of_range"] = 100.0 * table.apply(
        lambda r: preferred_sd(r)[0], axis=1
    ) / table["value_range"]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_path, index=False)

    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("Recommended --jitter-stds grids (0.5x, 1x, 2x, 4x the composite noise SD):")
    for _, row in table[table["objective"] == "composite"].iterrows():
        sd_hat, source = preferred_sd(row)
        grid = ",".join(f"{m * sd_hat:.2g}" for m in (0.5, 1.0, 2.0, 4.0))
        print(
            f"  {row['dataset']}: sd_hat={sd_hat:.3f} ({source}; "
            f"{100 * sd_hat / row['value_range']:.0f}% of observed composite range) "
            f"--jitter-stds {grid}"
        )
    print(f"\nSaved: {args.output_path}")


if __name__ == "__main__":
    main()
