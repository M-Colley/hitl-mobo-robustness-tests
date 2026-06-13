"""Fidelity + noise evaluation for the converted ASHRAE DB-II dataset.

1. Loads the converted CSV THROUGH the pipeline loader (temp config; proves
   datasets.json compatibility without editing it).
2. Cold protocol (harness.evaluate_cold, baseline_factory) with
   a) User_ID grouping (unseen subjects) and b) Group_ID grouping (unseen
   buildings).
3. Warm per-user protocol (train_frac=0.5, min_rows=4) -- caveat: most
   subjects voted once; only the repeat-subset participates.
4. Intra-rater noise via the NN-nugget estimators of
   calibrate_noise_from_data.py, rater = composite subject key, for
   ThermalComfort (converted data) and ThermalSensation (larger longitudinal
   subset rebuilt from raw).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\Mark\Desktop\hitl-mobo-robustness-tests")
for p in (REPO / "scripts", REPO / "sandbox" / "oracle_experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bo_sensor_error_simulation as bo_sim  # noqa: E402
import calibrate_noise_from_data as cal  # noqa: E402
import harness  # noqa: E402

PARAMS = [
    "AirTemperature",
    "RadiantTemperature",
    "RelativeHumidity",
    "AirVelocity",
    "MetabolicRate",
    "ClothingInsulation",
]
EVAL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EVAL_DIR / "ashrae_dataset_config.json"

DRAFT_CONFIG = [
    {
        "name": "ashrae",
        "data_dir": str(REPO / "external_datasets" / "ashrae"),
        "param_columns": PARAMS,
        "objective_map": {
            "composite": ["ThermalComfort"],
            "multi_objective": ["ThermalComfort", "-ThermalSensationAbs"],
            "comfort": ["ThermalComfort"],
            "sensation_neutrality": ["-ThermalSensationAbs"],
        },
    }
]


def load_via_pipeline() -> tuple[bo_sim.DatasetConfig, pd.DataFrame]:
    CONFIG_PATH.write_text(json.dumps(DRAFT_CONFIG, indent=2))
    datasets = bo_sim.parse_dataset_configs(None, CONFIG_PATH, REPO / ".dataset_cache")
    ds = datasets[0]
    df = bo_sim.load_observations(ds, "composite")
    groups, source = bo_sim.infer_oracle_groups(df)
    print(f"pipeline load OK: {len(df)} rows, group source={source}, "
          f"{pd.Series(groups).nunique()} groups")
    return ds, df


def make_bundle(ds, df, group_col: str) -> harness.DatasetBundle:
    cols = ds.objective_map["composite"]
    y = bo_sim.compute_objective(df, cols, False, None).to_numpy(dtype=float)
    return harness.DatasetBundle(
        name=f"ashrae[{group_col}]",
        df=df,
        X=df[ds.param_columns].to_numpy(dtype=float),
        y=y,
        groups=df[group_col].astype(str).to_numpy(),
        param_columns=list(ds.param_columns),
        objective_columns=list(cols),
        group_source=group_col,
        construct_Y=bo_sim._extract_objective_values(df, cols),
    )


def noise_rows(df: pd.DataFrame, value_col: str, label: str,
               close_pair_fraction: float = 0.05) -> dict:
    """NN-nugget + exact-repeat noise, rater = OBSERVATION_SOURCE_COLUMN."""
    df = cal.drop_full_duplicates(df, PARAMS, [value_col])
    bounds = bo_sim.bounds_from_data(df, PARAMS)
    close_threshold = close_pair_fraction * np.sqrt(len(PARAMS))

    repeat_values, artifacts = [], 0
    diffs_close, diffs_all, nn_distances = [], [], []
    for _, rater_df in df.groupby(bo_sim.OBSERVATION_SOURCE_COLUMN, sort=False):
        groups, art = cal.exact_repeat_groups(rater_df, PARAMS, [value_col])
        artifacts += art
        repeat_values.extend(g[value_col].to_numpy(dtype=float) for g in groups)
        pair_arr, dists = cal.nn_pairs(rater_df, PARAMS, bounds)
        if len(pair_arr) == 0:
            continue
        values = rater_df[value_col].to_numpy(dtype=float)
        pair_diffs = values[pair_arr[:, 0]] - values[pair_arr[:, 1]]
        nn_distances.extend(dists.tolist())
        diffs_all.extend(pair_diffs.tolist())
        diffs_close.extend(pair_diffs[dists <= close_threshold].tolist())

    sd_repeat, repeat_dof = cal.pooled_repeat_sd(repeat_values)
    row = {
        "objective": label,
        "n_rows": int(len(df)),
        "n_raters": int(df[bo_sim.OBSERVATION_SOURCE_COLUMN].nunique()),
        "n_repeat_groups": len(repeat_values),
        "n_artifact_duplicate_groups": artifacts,
        "repeat_dof": repeat_dof,
        "sd_repeat": sd_repeat,
        "n_close_pairs": len(diffs_close),
        "sd_nn_close": float(np.std(diffs_close, ddof=1) / np.sqrt(2.0))
        if len(diffs_close) >= 2 else float("nan"),
        "n_nn_pairs": len(diffs_all),
        "sd_nn_all": float(np.std(diffs_all, ddof=1) / np.sqrt(2.0))
        if len(diffs_all) >= 2 else float("nan"),
        "median_nn_distance": float(np.median(nn_distances)) if nn_distances else float("nan"),
        "close_pair_threshold": float(close_threshold),
        "value_range": float(np.nanmax(df[value_col]) - np.nanmin(df[value_col])),
    }
    sd_hat, src = cal.preferred_sd(pd.Series(row))
    row["preferred_sd"] = sd_hat
    row["preferred_source"] = src
    return row


def main() -> None:
    ds, df = load_via_pipeline()

    rows = []
    for group_col in ["User_ID", "Group_ID"]:
        bundle = make_bundle(ds, df, group_col)
        res = harness.evaluate_cold(bundle, harness.baseline_factory)
        res["grouping"] = group_col
        rows.append(res)
        print(group_col, res)

    # Warm: chronological within-subject order was preserved at conversion.
    bundle_u = make_bundle(ds, df, "User_ID")
    try:
        warm = harness.evaluate_warm_per_user(
            bundle_u, harness.baseline_factory, train_frac=0.5, min_rows=4
        )
        warm["grouping"] = "User_ID"
        rows.append(warm)
        print("warm", warm)
    except ValueError as exc:
        print("warm failed:", exc)

    # Noise: rater = composite subject key (User_ID in converted data).
    noise = []
    dfn = df.copy()
    dfn[bo_sim.OBSERVATION_SOURCE_COLUMN] = dfn["User_ID"].astype(str)
    noise.append(noise_rows(dfn, "ThermalComfort", "ThermalComfort(converted)"))

    # Larger longitudinal sensation subset straight from raw (subject-keyed).
    raw = pd.read_csv(
        REPO / "external_datasets" / "ashrae" / "raw" / "db_measurements_v2.1.0.csv.gz",
        low_memory=False,
    )
    raw_cols = ["ta", "tr", "rh", "vel", "met", "clo"]
    m = (
        raw["subject_id"].notna()
        & raw[raw_cols].notna().all(axis=1)
        & raw["thermal_sensation"].notna()
    )
    sen = raw.loc[m, raw_cols + ["thermal_sensation", "building_id", "subject_id"]].copy()
    sen.columns = PARAMS + ["ThermalSensation", "building_id", "subject_id"]
    sen[bo_sim.OBSERVATION_SOURCE_COLUMN] = (
        sen["building_id"].astype(int).astype(str) + "_" + sen["subject_id"].astype(str)
    )
    sen = sen.reset_index(drop=True)
    noise.append(noise_rows(sen, "ThermalSensation", "ThermalSensation(raw longitudinal)"))

    print()
    print(pd.DataFrame(noise).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    harness.print_results("ashrae_fidelity", rows + noise)


if __name__ == "__main__":
    main()
