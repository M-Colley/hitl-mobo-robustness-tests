"""Independent verifier for the claimed nisqa_sim conversion.

Builds DatasetConfig verbatim from the claimed config_entry and pushes the
data through the real pipeline loader (load_observations), then checks
infer_oracle_groups. Prints a JSON block for the verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import bo_sensor_error_simulation as bo_sim  # noqa: E402

CONFIG_ENTRY = {
    "name": "nisqa_sim",
    "data_dir": "external_datasets/nisqa_sim",
    "observation_glob": "ObservationsPerEvaluation.csv",
    "param_columns": [
        "bp_low", "bp_high", "arb_filter_on", "tc_fer", "tc_nburst",
        "wbgn_snr", "bgn_snr", "p50_q", "cl_th",
        "asl_in_on", "asl_in_level", "asl_out_on", "asl_out_level",
        "codec1_id", "bMode1", "FER1", "plc1_random", "plc1_bursty",
        "codec2_id", "bMode2", "FER2", "plc2_random", "plc2_bursty",
        "codec3_id", "bMode3", "FER3", "plc3_random", "plc3_bursty",
    ],
    "objective_map": {
        "composite": ["mos", "noi", "col", "dis", "loud"],
        "multi_objective": ["mos", "noi", "col", "dis", "loud"],
        "mos": ["mos"],
        "noisiness": ["noi"],
        "coloration": ["col"],
        "discontinuity": ["dis"],
        "loudness": ["loud"],
    },
}


def main() -> None:
    ds = bo_sim.DatasetConfig(
        name=CONFIG_ENTRY["name"],
        data_dirs=[REPO / CONFIG_ENTRY["data_dir"]],
        param_columns=CONFIG_ENTRY["param_columns"],
        objective_map=CONFIG_ENTRY["objective_map"],
        observation_glob=CONFIG_ENTRY["observation_glob"],
    )
    out = {}
    for objective in CONFIG_ENTRY["objective_map"]:
        df = bo_sim.load_observations(ds, objective)
        cols = ds.objective_map[objective]
        params = df[ds.param_columns]
        objs = df[[c.lstrip("-") for c in cols]]
        groups, source = bo_sim.infer_oracle_groups(df)
        y = bo_sim.compute_objective(df, cols, False, None)
        out[objective] = {
            "n_rows": int(len(df)),
            "n_param_cols": int(params.shape[1]),
            "param_nans": int(params.isna().sum().sum()),
            "objective_nans": int(objs.isna().sum().sum()),
            "group_source": source,
            "n_groups": int(pd.Series(groups).nunique()) if groups is not None else 0,
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "y_std": float(y.std()),
            "param_constant_cols": [c for c in ds.param_columns if params[c].nunique() <= 1],
        }
    print("VERIFY_JSON_BEGIN")
    print(json.dumps(out, indent=2))
    print("VERIFY_JSON_END")


if __name__ == "__main__":
    main()
