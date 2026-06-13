"""Fidelity + noise evaluation for the converted nisqa_sim dataset.

1. Loads external_datasets/nisqa_sim through the REAL pipeline loader
   (bo_sim.load_observations with a hand-built DatasetConfig) to prove the
   conversion is loadable.
2. evaluate_cold (GroupKFold over Group_ID = source speakers) with the
   deployed oracle family (ExtraTrees, n_jobs=1) for:
   - composite (mean of mos/noi/col/dis/loud)
   - mos only
   Plus a secondary cold run grouped by source corpus (4 groups).
3. Noise estimates (all INTER-rater; the corpus has no rater ids or repeats
   for SIM, so intra-rater noise is NOT identifiable):
   - per-file crowd SD (mos_std etc.) and the SEM of the ~5-vote file mean
   - NISQA_TEST_LIVETALK raw rating file: per-condition across-rater SD
   - NN-nugget (calibrate_noise_from_data.nn_pairs) on the pooled per-file
     means in normalized param space, as an upper-bound residual estimate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "sandbox" / "oracle_experiments"))

import bo_sensor_error_simulation as bo_sim  # noqa: E402
import calibrate_noise_from_data as cal  # noqa: E402
import harness  # noqa: E402

from nisqa_convert import OBJECTIVE_COLUMNS, PARAM_COLUMNS  # noqa: E402

DATA_DIR = REPO / "external_datasets" / "nisqa_sim"
RAW = HERE / "raw" / "nisqa"

OBJECTIVE_MAP = {
    "composite": OBJECTIVE_COLUMNS,
    "multi_objective": OBJECTIVE_COLUMNS,
    "mos": ["mos"],
}


def make_bundle(objective: str, group_col: str = "Group_ID") -> harness.DatasetBundle:
    ds = bo_sim.DatasetConfig(
        name="nisqa_sim",
        data_dirs=[DATA_DIR],
        param_columns=PARAM_COLUMNS,
        objective_map=OBJECTIVE_MAP,
    )
    df = bo_sim.load_observations(ds, objective)
    cols = ds.objective_map[objective]
    y = bo_sim.compute_objective(df, cols, False, None).to_numpy(dtype=float)
    if group_col == "Group_ID":
        groups, source = bo_sim.infer_oracle_groups(df)
        assert source == "Group_ID", source
    else:
        groups, source = df[group_col].to_numpy(), group_col
    return harness.DatasetBundle(
        name=f"nisqa_sim[{objective},groups={source}]",
        df=df,
        X=df[ds.param_columns].to_numpy(dtype=float),
        y=y,
        groups=np.asarray(groups).astype(str),
        param_columns=list(ds.param_columns),
        objective_columns=list(cols),
        group_source=source,
        construct_Y=bo_sim._extract_objective_values(df, cols),
    )


def _nn_pairs_large(df: pd.DataFrame, param_columns: list[str], bounds) -> tuple[np.ndarray, np.ndarray]:
    """Memory-efficient version of calibrate_noise_from_data.nn_pairs
    (identical estimator; cal.nn_pairs builds an n^2 distance matrix which
    needs 33 GB at n=12500)."""
    from sklearn.neighbors import NearestNeighbors

    X = df[param_columns].to_numpy(dtype=float)
    span = np.where(bounds.high > bounds.low, bounds.high - bounds.low, 1.0)
    Xn = (X - bounds.low) / span
    nn = NearestNeighbors(n_neighbors=2, n_jobs=1).fit(Xn)
    dist, idx = nn.kneighbors(Xn)
    pairs = {tuple(sorted((i, int(idx[i, 1])))) for i in range(len(Xn))}
    pair_arr = np.array(sorted(pairs), dtype=int)
    d = np.linalg.norm(Xn[pair_arr[:, 0]] - Xn[pair_arr[:, 1]], axis=1)
    return pair_arr, d


def nn_nugget(df: pd.DataFrame, value_col: str) -> dict:
    """NN-nugget on the pooled per-file means (single pseudo-rater)."""
    bounds = bo_sim.bounds_from_data(df, PARAM_COLUMNS)
    pair_arr, dists = _nn_pairs_large(df, PARAM_COLUMNS, bounds)
    values = df[value_col].to_numpy(dtype=float)
    diffs = values[pair_arr[:, 0]] - values[pair_arr[:, 1]]
    thr = 0.05 * np.sqrt(len(PARAM_COLUMNS))
    close = dists <= thr
    return {
        "value_col": value_col,
        "n_pairs": int(len(diffs)),
        "n_close": int(close.sum()),
        "sd_nn_all": float(np.std(diffs, ddof=1) / np.sqrt(2)),
        "sd_nn_close": float(np.std(diffs[close], ddof=1) / np.sqrt(2)) if close.sum() >= 2 else float("nan"),
        "median_nn_dist": float(np.median(dists)),
        "close_thr": float(thr),
    }


def main() -> None:
    rows = []

    for objective in ["composite", "mos"]:
        bundle = make_bundle(objective)
        res = harness.evaluate_cold(bundle, harness.baseline_factory)
        res["dataset"] = bundle.name
        res["y_range"] = float(bundle.y.max() - bundle.y.min())
        rows.append(res)
        print(bundle.name, res)

    bundle = make_bundle("composite", group_col="source")
    res = harness.evaluate_cold(bundle, harness.baseline_factory, n_splits=4)
    res["dataset"] = bundle.name
    res["y_range"] = float(bundle.y.max() - bundle.y.min())
    rows.append(res)
    print(bundle.name, res)

    # ---------------- noise ----------------
    df = pd.read_csv(DATA_DIR / "ObservationsPerEvaluation.csv", sep=";")
    print("\n--- INTER-rater noise, SIM subsets (per-file crowd SD over ~5 votes) ---")
    noise = {}
    for c in ["mos_std", "noi_std", "col_std", "dis_std", "loud_std"]:
        # pooled single-vote SD: sqrt(mean variance) weighted by votes-1
        w = df["votes"] - 1
        pooled = float(np.sqrt(np.sum(w * df[c] ** 2) / np.sum(w)))
        noise[c] = {
            "mean_sd": float(df[c].mean()),
            "pooled_sd": pooled,
            "sem_of_file_mean": float((df[c] / np.sqrt(df["votes"])).mean()),
        }
        print(c, noise[c])

    print("\n--- INTER-rater noise, LIVETALK raw per-rater ratings (24 raters) ---")
    lt = pd.read_csv(
        RAW / "NISQA_TEST_LIVETALK" / "NISQA_TEST_LIVETALK_listening_test_ratings.csv",
        sep=";",
    )
    lt_noise = {}
    for c in ["QOE", "NOI", "DIS", "COL", "LOU"]:
        per_cond_sd = lt.groupby("Condition")[c].std(ddof=1)
        lt_noise[c] = {
            "mean_per_condition_sd": float(per_cond_sd.mean()),
            "n_conditions": int(per_cond_sd.size),
        }
        print(c, lt_noise[c])

    print("\n--- NN-nugget on pooled per-file means (content+panel residual, upper bound) ---")
    comp = df[OBJECTIVE_COLUMNS].mean(axis=1)
    df2 = df.copy()
    df2["__composite"] = comp
    for col in ["mos", "__composite"]:
        print(nn_nugget(df2, col))

    harness.print_results("nisqa_sim_fidelity", rows)


if __name__ == "__main__":
    main()
