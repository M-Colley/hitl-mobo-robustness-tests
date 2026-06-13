"""Check chronology of row order within merged User_ID groups (D1 core)."""
import numpy as np
import pandas as pd

import harness

data = harness.load_dataset("provoice")
df = data.df
t = pd.to_datetime(df["Timestamp"], errors="coerce")
print(f"unparseable timestamps: {int(t.isna().sum())}/{len(t)}")

src = df["__source_file"].astype(str).to_numpy()
groups = data.groups

# Within each merged user group: is row order chronological?
chrono = 0
first_file_started_later = 0
for u in np.unique(groups):
    idx = np.where(groups == u)[0]
    tt = t.iloc[idx]
    chrono += int(tt.is_monotonic_increasing)
    files = pd.unique(src[idx])
    if len(files) == 2:
        t_first_file = t.iloc[idx[src[idx] == files[0]]].min()
        t_second_file = t.iloc[idx[src[idx] == files[1]]].min()
        if t_first_file > t_second_file:
            first_file_started_later += 1
print(f"merged user groups with chronological row order: {chrono}/{len(np.unique(groups))}")
print(f"user groups where the FIRST-globbed file actually STARTED LATER: {first_file_started_later}/19")

# within each session file: chronological?
chrono_f = sum(int(t.iloc[np.where(src == f)[0]].is_monotonic_increasing) for f in np.unique(src))
print(f"session files with chronological row order: {chrono_f}/{len(np.unique(src))}")

# Which condition label appears first in glob order, per user; and which ran first by time
rows = []
for u in np.unique(groups):
    idx = np.where(groups == u)[0]
    files = pd.unique(src[idx])
    if len(files) != 2:
        rows.append((u, "single-file", None, None))
        continue
    f0, f1 = files
    c0 = df["Condition_ID"].iloc[idx[src[idx] == f0]].iloc[0]
    t0 = t.iloc[idx[src[idx] == f0]].min()
    t1 = t.iloc[idx[src[idx] == f1]].min()
    rows.append((u, f"glob-first cond={c0}", "glob-first ran first" if t0 < t1 else "glob-first ran LATER", str(t0 < t1)))
for r in rows:
    print(r)
