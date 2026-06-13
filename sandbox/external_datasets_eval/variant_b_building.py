"""Variant B: ALL comfort-complete rows (no subject_id requirement),
grouping = building_id. Wider environmental range (incl. naturally
ventilated buildings) -> test whether cold signal improves."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\Mark\Desktop\hitl-mobo-robustness-tests")
for p in (REPO / "scripts", REPO / "sandbox" / "oracle_experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import harness  # noqa: E402

raw = pd.read_csv(
    REPO / "external_datasets" / "ashrae" / "raw" / "db_measurements_v2.1.0.csv.gz",
    low_memory=False,
)
core5 = ["ta", "rh", "vel", "met", "clo"]

for params, tag in [(core5, "5param"), (core5 + ["tr"], "6param")]:
    m = raw[params].notna().all(axis=1) & raw["thermal_comfort"].notna()
    sub = raw.loc[m].reset_index(drop=True)
    print(f"\n{tag}: rows={len(sub)} buildings={sub['building_id'].nunique()} "
          f"ta range={sub['ta'].min():.1f}-{sub['ta'].max():.1f} "
          f"ta std={sub['ta'].std():.2f}")
    bundle = harness.DatasetBundle(
        name=f"ashrae_b_{tag}",
        df=sub,
        X=sub[params].to_numpy(dtype=float),
        y=sub["thermal_comfort"].to_numpy(dtype=float),
        groups=sub["building_id"].astype(str).to_numpy(),
        param_columns=params,
        objective_columns=["thermal_comfort"],
        group_source="building_id",
    )
    res = harness.evaluate_cold(bundle, harness.baseline_factory)
    print(tag, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})

# Also: thermal_sensation as target (predictability check; objective use
# would need the precomputed -|TSV| column). Subsample to ~10k stratified
# by building to keep runtime sane (>20k rows rule).
m = raw[core5].notna().all(axis=1) & raw["thermal_sensation"].notna()
sub = raw.loc[m].reset_index(drop=True)
rng = np.random.default_rng(7)
if len(sub) > 10000:
    frac = 10000 / len(sub)
    sub = (
        sub.groupby("building_id", group_keys=False)
        .apply(lambda g: g.sample(max(1, int(round(frac * len(g)))), random_state=7))
        .reset_index(drop=True)
    )
print(f"\nsensation: rows={len(sub)} buildings={sub['building_id'].nunique()}")
bundle = harness.DatasetBundle(
    name="ashrae_sensation",
    df=sub,
    X=sub[core5].to_numpy(dtype=float),
    y=sub["thermal_sensation"].to_numpy(dtype=float),
    groups=sub["building_id"].astype(str).to_numpy(),
    param_columns=core5,
    objective_columns=["thermal_sensation"],
    group_source="building_id",
)
res = harness.evaluate_cold(bundle, harness.baseline_factory)
print("sensation", {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})

# -|TSV| target (the actual BO objective form for neutrality)
bundle.y = -np.abs(bundle.y)
res = harness.evaluate_cold(bundle, harness.baseline_factory)
print("neg_abs_sensation", {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
