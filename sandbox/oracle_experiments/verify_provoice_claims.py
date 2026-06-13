"""Adversarial verification of data-level claims in exp_provoice.py (read-only checks)."""
import numpy as np
import pandas as pd

import harness

data = harness.load_dataset("provoice")
df = data.df
y = data.y
n = len(y)
print(f"n_rows={n}, n_default_groups={len(np.unique(data.groups))}, group_source={data.group_source}")
print(f"param_columns={data.param_columns}")
print(f"objective_columns={data.objective_columns}")

# Claim: 38 source files, 19 users x 2 conditions
src = df["__source_file"].astype(str)
print(f"n_source_files={src.nunique()}")

# Claim: rglob orders 'Condition 2' before 'Condition 1' within each merged user group;
# warm test rows under User_ID grouping are 0% condition 2.
cond = df["Condition_ID"].to_numpy()
splits = harness._warm_split_indices(data, 0.7, 10)
test_idx = np.concatenate([t for _, _, t in splits])
print(f"user-grouped warm: n_users={len(splits)}, n_test_rows={len(test_idx)}, "
      f"share_cond2_in_test={float((cond[test_idx]==2).mean()):.3f}")
# check row order within a user group: which condition comes first?
first_cond = []
for u in np.unique(data.groups):
    idx = np.where(data.groups == u)[0]
    first_cond.append(cond[idx[0]])
print(f"per-user first-row condition counts: {pd.Series(first_cond).value_counts().to_dict()}")

# Timestamps: condition 1 ran first?
ts_col = [c for c in df.columns if "time" in c.lower() or "stamp" in c.lower() or "date" in c.lower()]
print(f"timestamp-ish columns: {ts_col}")
if ts_col:
    tcol = ts_col[0]
    t = pd.to_datetime(df[tcol], errors="coerce")
    agg = df.assign(_t=t).groupby(["User_ID", "Condition_ID"])["_t"].min().unstack()
    if agg.shape[1] == 2:
        c1_first = (agg.iloc[:, 0] < agg.iloc[:, 1]).sum()
        print(f"users where cond {agg.columns[0]} starts before cond {agg.columns[1]}: {c1_first}/{len(agg)}")

# Claim: P17 'Condition 2' file has Condition_ID==1 -> user x condition gives 37 not 38
uc = df.groupby(["User_ID", "Condition_ID"]).ngroup()
print(f"user x condition_id groups: {uc.nunique()}")
p17 = df[df["User_ID"].astype(str).str.contains("17")]
for f, sub in p17.groupby("__source_file"):
    print(f"  P17 file={f.split(chr(92))[-1]} cond_ids={sorted(sub['Condition_ID'].unique())} rows={len(sub)}")

# Claim: Mental Demand spans 1-17, Pred/Useful 1-5; signed variances 8.24/1.08/0.89;
# corr(-MD, composite)=0.965; -MD share of composite variance 80.7%
P = df["Predictability"].to_numpy(float)
U = df["Percieved Usefulness"].to_numpy(float)
MD = df["Mental Demand"].to_numpy(float)
print(f"Pred range [{P.min()},{P.max()}], Useful range [{U.min()},{U.max()}], MD range [{MD.min()},{MD.max()}]")
print(f"signed variances: -MD={np.var(-MD):.2f} Pred={np.var(P):.2f} Useful={np.var(U):.2f}")
print(f"corr(-MD, composite)={np.corrcoef(-MD, y)[0,1]:.3f}")
tot = np.var(-MD/3) + np.var(P/3) + np.var(U/3)  # not exact decomposition; use covariance share instead
cov_share = np.cov(-MD/3, y)[0,1] / np.var(y)
print(f"cov(-MD/3, composite)/var(composite) = {cov_share:.3f}")

# Claim: 304/532 rows are a shared sampling design; 64.8% of variance within identical-X rows
Xdf = pd.DataFrame(data.X, columns=data.param_columns)
key = Xdf.round(10).apply(tuple, axis=1)
counts = key.value_counts()
shared = counts[counts > 1]
n_shared_rows = int(shared.sum())
print(f"rows with duplicated X: {n_shared_rows}/{n} ; top config counts: {shared.head(10).tolist()}")
grand = np.var(y)
within = 0.0
for k, sub in pd.Series(y).groupby(key):
    if len(sub) > 1:
        within += len(sub) * np.var(sub)
within /= n
print(f"within-identical-X variance share: {within/grand:.3f} (params-only R2 ceiling ~{1-within/grand:.3f})")

# Claim: 17.3% exact duplicate rows (params + all 3 constructs)
full = pd.concat([Xdf, pd.DataFrame({"P": P, "U": U, "MD": MD})], axis=1)
dup_share = full.duplicated(keep="first").mean()
print(f"exact duplicate rows (params+constructs, keep=first): {dup_share:.3f}")

# Claim: iteration drift corr 0.33 mean per-session
it = df["Iteration"].to_numpy(float)
sess = src.to_numpy()
corrs = []
for s in np.unique(sess):
    idx = np.where(sess == s)[0]
    if len(idx) > 3 and np.std(y[idx]) > 0 and np.std(it[idx]) > 0:
        corrs.append(np.corrcoef(it[idx], y[idx])[0, 1])
print(f"mean per-session corr(iteration, y) = {np.mean(corrs):.3f} over {len(corrs)} sessions")

# Claim: iteration is monotone within file row order (row order = chronological)?
mono = 0
for s in np.unique(sess):
    idx = np.where(sess == s)[0]
    mono += int(np.all(np.diff(it[idx]) >= 0))
print(f"sessions where Iteration is non-decreasing in row order: {mono}/{src.nunique()}")

# LevelOfAutonomy continuous claim
if "LevelOfAutonomy" in df.columns:
    loa = df["LevelOfAutonomy"].to_numpy(float)
    print(f"LevelOfAutonomy: unique={len(np.unique(loa))}, range=[{loa.min():.3f},{loa.max():.3f}]")

# Sanity-check the noise floor comparison: y std and scale
print(f"composite std={np.std(y):.3f}, y range [{y.min():.2f},{y.max():.2f}]")
