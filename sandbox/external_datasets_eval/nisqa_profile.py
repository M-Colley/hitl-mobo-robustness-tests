"""Profile NISQA_TRAIN_SIM / NISQA_VAL_SIM per-file metadata: which columns
are usable as numeric design parameters, missingness, value ranges, and the
inter-rater rating SD distribution."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "raw" / "nisqa"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)

frames = []
for sub in ["NISQA_TRAIN_SIM", "NISQA_VAL_SIM"]:
    df = pd.read_csv(RAW / sub / f"{sub}_file.csv")
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
print("rows:", len(df))
print("columns:", list(df.columns))
print()

for c in df.columns:
    s = df[c]
    nun = s.nunique(dropna=True)
    nmiss = int(s.isna().sum())
    num = pd.to_numeric(s, errors="coerce")
    n_numeric = int(num.notna().sum())
    if n_numeric >= (s.notna().sum()) and n_numeric > 0:
        print(
            f"{c:>15} num  nuniq={nun:5d} miss={nmiss:5d} "
            f"min={num.min():.3f} max={num.max():.3f} mean={num.mean():.3f}"
        )
    else:
        vals = s.dropna().unique()
        vc = s.value_counts(dropna=False)
        print(
            f"{c:>15} obj  nuniq={nun:5d} miss={nmiss:5d} numeric_ok={n_numeric:5d} "
            f"top={dict(vc.head(8))}"
        )

print()
print("votes distribution:", df["votes"].describe().to_dict())
print()
print("mos_std stats:", df["mos_std"].describe().to_dict())
for c in ["noi_std", "col_std", "dis_std", "loud_std"]:
    print(c, "mean:", df[c].mean())
print()
print("source counts:", df["source"].value_counts().to_dict())
