"""Independent verification of the claimed ASHRAE conversion.

Builds the DatasetConfig from the CLAIMED config entry verbatim (relative
data_dir, default observation_glob) via parse_dataset_configs, loads through
load_observations, checks NaNs and infer_oracle_groups. Run from repo root.
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(r"C:\Users\Mark\Desktop\hitl-mobo-robustness-tests")
sys.path.insert(0, str(REPO / "scripts"))
import bo_sensor_error_simulation as bo_sim  # noqa: E402

CLAIMED_ENTRY = {
    "name": "ashrae",
    "data_dir": "external_datasets/ashrae",
    "observation_glob": "ObservationsPerEvaluation.csv",
    "param_columns": [
        "AirTemperature",
        "RadiantTemperature",
        "RelativeHumidity",
        "AirVelocity",
        "MetabolicRate",
        "ClothingInsulation",
    ],
    "objective_map": {
        "composite": ["ThermalComfort"],
        "multi_objective": ["ThermalComfort", "-ThermalSensationAbs"],
        "comfort": ["ThermalComfort"],
        "sensation_neutrality": ["-ThermalSensationAbs"],
    },
}


def main() -> None:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, dir=str(REPO / "sandbox" / "external_datasets_eval")
    ) as fh:
        json.dump([CLAIMED_ENTRY], fh)
        cfg_path = Path(fh.name)
    try:
        datasets = bo_sim.parse_dataset_configs(None, cfg_path, REPO / ".dataset_cache")
        ds = datasets[0]
        print("DatasetConfig:", ds.name, "| dirs:", [str(d) for d in ds.data_dirs])
        for objective in ["composite", "multi_objective", "comfort", "sensation_neutrality"]:
            df = bo_sim.load_observations(ds, objective)
            req = ds.param_columns + [
                c[1:] if c.startswith("-") else c for c in ds.objective_map[objective]
            ]
            nan_count = int(df[req].isna().sum().sum())
            groups, source = bo_sim.infer_oracle_groups(df)
            n_groups = pd.Series(groups).nunique() if groups is not None else 0
            y = bo_sim.compute_objective(df, ds.objective_map[objective], False, None)
            print(
                f"objective={objective:22s} rows={len(df):5d} NaNs(required cols)={nan_count} "
                f"group_source={source} n_groups={n_groups} "
                f"y[min={y.min():.2f},max={y.max():.2f},sd={y.std():.3f}]"
            )
            assert nan_count == 0, "NaN explosion in required columns"
            assert n_groups >= 10, "fewer than 10 groups"
        print("LOAD VERIFICATION PASSED")
    finally:
        cfg_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
