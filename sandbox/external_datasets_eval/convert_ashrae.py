"""Convert ASHRAE Global Thermal Comfort Database II (v2.1.0) to pipeline format.

Source: https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II (v2.1.0)
        mirrored on Dryad doi:10.6078/D1F671, license CC0-1.0.
Raw file: external_datasets/ashrae/raw/db_measurements_v2.1.0.csv.gz

Output: external_datasets/ashrae/ObservationsPerEvaluation.csv

Conversion decisions
--------------------
- Parameter space (6 continuous columns, the classical PMV inputs):
    AirTemperature      <- ta   [deg C]
    RadiantTemperature  <- tr   [deg C]
    RelativeHumidity    <- rh   [%]
    AirVelocity         <- vel  [m/s]
    MetabolicRate       <- met  [met]
    ClothingInsulation  <- clo  [clo]
  ta/tr/rh/vel are HVAC-controllable; met/clo are occupant covariates that a
  stricter BO framing would treat as context, but they are standard model
  inputs for comfort prediction.
- Objectives:
    ThermalComfort        <- thermal_comfort   [1 very uncomfortable .. 6 very comfortable]
    ThermalSensation      <- thermal_sensation [-3 cold .. +3 hot] (reference only)
    ThermalSensationAbs   <- |thermal_sensation| (use with '-' prefix: maximize
                             neutrality; precomputed because the pipeline has
                             no |.| transform)
- Rows: subject_id present AND all 6 params AND thermal_comfort present.
  (thermal_sensation may be NaN in <10 rows; the loader drops those only when
  that objective is requested.)
- Grouping: subject_id in the raw DB is NOT globally unique (ids are reused
  across buildings with inconsistent gender/age), so the rater key is the
  composite (building_id, subject_id), factorized to a numeric User_ID.
  Group_ID = building_id is also written (coarser, leakage-safe alternative).
- Row order: sorted by (building_id, subject key, timestamp) so within-user
  order is chronological for warm splits.
"""
from pathlib import Path

import pandas as pd

REPO = Path(r"C:\Users\Mark\Desktop\hitl-mobo-robustness-tests")
RAW = REPO / "external_datasets" / "ashrae" / "raw" / "db_measurements_v2.1.0.csv.gz"
OUT = REPO / "external_datasets" / "ashrae" / "ObservationsPerEvaluation.csv"

PARAM_MAP = {
    "ta": "AirTemperature",
    "tr": "RadiantTemperature",
    "rh": "RelativeHumidity",
    "vel": "AirVelocity",
    "met": "MetabolicRate",
    "clo": "ClothingInsulation",
}


def main() -> None:
    df = pd.read_csv(RAW, low_memory=False)
    raw_cols = list(PARAM_MAP)
    mask = (
        df["subject_id"].notna()
        & df[raw_cols].notna().all(axis=1)
        & df["thermal_comfort"].notna()
    )
    sub = df.loc[mask].copy()

    sub["__subject_key"] = (
        sub["building_id"].astype(int).astype(str) + "_" + sub["subject_id"].astype(str)
    )
    sub = sub.sort_values(
        ["building_id", "__subject_key", "timestamp"], kind="stable"
    ).reset_index(drop=True)

    out = sub[raw_cols].rename(columns=PARAM_MAP)
    out["ThermalComfort"] = sub["thermal_comfort"].astype(float)
    out["ThermalSensation"] = sub["thermal_sensation"].astype(float)
    out["ThermalSensationAbs"] = sub["thermal_sensation"].abs()
    out["User_ID"] = pd.factorize(sub["__subject_key"])[0] + 1
    out["Group_ID"] = sub["building_id"].astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {out.shape[0]} rows, {out['User_ID'].nunique()} subjects, "
          f"{out['Group_ID'].nunique()} buildings")
    print(out.describe().round(3).T)


if __name__ == "__main__":
    main()
