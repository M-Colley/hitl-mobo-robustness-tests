"""Profile ASHRAE DB-II measurements: missingness, grouping options, objectives."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\Mark\Desktop\hitl-mobo-robustness-tests\external_datasets\ashrae\raw")

df = pd.read_csv(RAW / "db_measurements_v2.1.0.csv.gz", low_memory=False)
n = len(df)
print(f"rows={n}")

cand_params = ["ta", "top", "tr", "rh", "vel", "met", "clo", "t_out_isd", "rh_out_isd"]
objectives = ["thermal_comfort", "thermal_sensation", "thermal_acceptability", "thermal_preference"]
groupers = ["subject_id", "building_id"]

print("\n-- missingness (% non-null) --")
for c in cand_params + objectives + groupers:
    nn = df[c].notna().sum()
    print(f"{c:22s} {100*nn/n:6.1f}%  dtype={df[c].dtype}")

print("\n-- objective distributions --")
print("thermal_comfort:", df["thermal_comfort"].dropna().describe().to_dict())
print(df["thermal_comfort"].value_counts(dropna=False).sort_index().head(20))
print("thermal_sensation:", df["thermal_sensation"].dropna().describe().to_dict())

print("\n-- grouping granularity --")
for g in groupers:
    sub = df[df[g].notna()]
    counts = sub.groupby(g).size()
    print(f"{g}: {sub.shape[0]} rows non-null, {counts.shape[0]} groups, "
          f"rows/group median={counts.median():.0f} mean={counts.mean():.1f} max={counts.max()}")

# subject_id + comfort joint availability
core = ["ta", "rh", "vel", "met", "clo"]
for obj in ["thermal_comfort", "thermal_sensation"]:
    for g in groupers:
        m = df[core + [obj, g]].notna().all(axis=1)
        sub = df[m]
        gc = sub.groupby(g).size()
        print(f"complete rows core+{obj}+{g}: {m.sum()}  groups={gc.shape[0]} "
              f"median rows/group={gc.median() if len(gc) else float('nan'):.0f}")

# repeated samples per subject (longitudinal)?
m = df["subject_id"].notna()
sub = df[m]
counts = sub.groupby("subject_id").size()
print("\nsubject_id repeat structure:", counts.describe().to_dict())
print("subjects with >=2 rows:", (counts >= 2).sum(), "with >=10 rows:", (counts >= 10).sum())

# do subject_ids span buildings? check uniqueness scope
nb = sub.groupby("subject_id")["building_id"].nunique()
print("subjects spanning >1 building:", (nb > 1).sum())

# which buildings have subject_id?
print("\nbuildings with any subject_id:", sub["building_id"].nunique(), "of", df["building_id"].nunique())
