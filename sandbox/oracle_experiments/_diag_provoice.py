"""Diagnostics for provoice (no harness evaluations here, just pandas)."""
import numpy as np
import pandas as pd

import harness

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

data = harness.load_dataset("provoice")
df = data.df

print("=== BASIC ===")
print("rows:", len(df), "group_source:", data.group_source)
print("columns:", list(df.columns))
print("n unique groups:", len(np.unique(data.groups)))
print("groups:", sorted(np.unique(data.groups), key=lambda s: (len(s), s)))

print("\n=== GROUP STRUCTURE ===")
src = df["observation_source"] if "observation_source" in df.columns else None
for c in ["User_ID", "UserID", "ConditionID", "Condition_ID", "GroupID", "Group_ID"]:
    if c in df.columns:
        print(c, "unique:", sorted(df[c].dropna().unique().tolist()))
# files per group
src_col = [c for c in df.columns if "source" in c.lower()]
print("source-ish columns:", src_col)
if src_col:
    s = df[src_col[0]]
    print("n unique source files:", s.nunique())
    per_group_files = df.groupby(data.groups)[src_col[0]].nunique()
    print("files per group:\n", per_group_files.to_string())

print("\n=== CONDITION x USER crosstab (rows per cell) ===")
if "ConditionID" in df.columns:
    print(pd.crosstab(df["User_ID"] if "User_ID" in df.columns else data.groups, df["ConditionID"]))

print("\n=== CONSTRUCT SCALES ===")
print("objective_columns:", data.objective_columns)
cons = pd.DataFrame(data.construct_Y, columns=data.objective_columns)
print(cons.describe().T)
print("\nvariance of signed constructs:")
print(cons.var().to_string())
print("\ncomposite y: mean %.3f std %.3f min %.3f max %.3f" % (data.y.mean(), data.y.std(), data.y.min(), data.y.max()))
print("\ncorrelation of each signed construct with composite y:")
for c in cons.columns:
    print(f"  {c}: r={np.corrcoef(cons[c], data.y)[0,1]:.3f}")
print("\nvariance share: Var(c)/3^2 relative contributions (composite = mean of 3)")
print((cons.var() / cons.var().sum()).to_string())

print("\n=== RAW construct unique value counts ===")
for c in ["Predictability", "Percieved Usefulness", "Mental Demand"]:
    u = np.sort(df[c].unique())
    print(f"{c}: n_unique={len(u)} range=[{u.min()},{u.max()}] uniques={u[:25]}")

print("\n=== PARAM DIAGNOSTICS ===")
Xdf = pd.DataFrame(data.X, columns=data.param_columns)
print(Xdf.describe().T)
for c in data.param_columns:
    u = np.sort(Xdf[c].unique())
    print(f"{c}: n_unique={len(u)}", "uniques:" if len(u) <= 15 else "first 10:", u[:15])

print("\n=== DUPLICATES ===")
dup_X = Xdf.duplicated(keep=False)
print("rows with duplicated X (params only): %d (%.1f%%)" % (dup_X.sum(), 100 * dup_X.mean()))
full = Xdf.copy()
full["y"] = data.y
dup_Xy = full.duplicated(keep=False)
print("rows with duplicated X+y: %d (%.1f%%)" % (dup_Xy.sum(), 100 * dup_Xy.mean()))
# Within same X, how variable is y? (irreducible noise given X)
g = full.groupby(data.param_columns)["y"]
sizes = g.size()
multi = sizes[sizes > 1]
print("n distinct X configs:", len(sizes), "| configs appearing >1x:", len(multi), "| rows in them:", multi.sum())
within_sd = g.std().dropna()
print("within-identical-X composite SD: mean=%.3f median=%.3f" % (within_sd.mean(), within_sd.median()))
# variance decomposition: how much of total variance is within identical-X cells
total_var = full["y"].var(ddof=0)
within_var = (g.transform("mean") - full["y"]).pow(2).mean()
print("total var=%.3f within-identical-X var=%.3f (%.1f%% irreducible w/o extra features)" % (total_var, within_var, 100 * within_var / total_var))

print("\n=== Same X across conditions (sampling phase shared Sobol?) ===")
if "ConditionID" in df.columns and "Phase" in df.columns:
    print("Phase counts:", df["Phase"].value_counts().to_dict())
    key = df.groupby(data.param_columns + ["User_ID"])["ConditionID"].nunique()
    print("X+user combos seen in both conditions:", (key > 1).sum(), "of", len(key))

print("\n=== X variation within session (group) ===")
for c in data.param_columns:
    nun = df.groupby(data.groups)[c].nunique()
    print(f"{c}: median unique values per group={nun.median():.0f} min={nun.min()}")

print("\n=== RATING DRIFT over iteration ===")
if "Iteration" in df.columns:
    it = df["Iteration"].astype(float)
    print("Iteration range:", it.min(), it.max())
    print("corr(iteration, composite) overall: %.3f" % np.corrcoef(it, data.y)[0, 1])
    per_user = []
    for u in np.unique(data.groups):
        m = data.groups == u
        if m.sum() > 5 and np.var(it[m]) > 0 and np.var(data.y[m]) > 0:
            per_user.append(np.corrcoef(it[m], data.y[m])[0, 1])
    print("per-group corr(iter, y): mean=%.3f median=%.3f, n=%d" % (np.mean(per_user), np.median(per_user), len(per_user)))

print("\n=== Condition effect on y ===")
if "ConditionID" in df.columns:
    print(df.groupby("ConditionID")[df.columns.intersection(["Predictability", "Percieved Usefulness", "Mental Demand"])].mean())
    comp = pd.Series(data.y)
    print("composite mean by condition:")
    print(comp.groupby(df["ConditionID"].values).agg(["mean", "std", "count"]))

print("\n=== Per-construct correlation with params ===")
for j, c in enumerate(data.objective_columns):
    rs = [np.corrcoef(data.X[:, k], data.construct_Y[:, j])[0, 1] for k in range(data.X.shape[1])]
    print(c, " ".join(f"{p}={r:+.3f}" for p, r in zip(data.param_columns, rs)))
print("composite:", " ".join(f"{p}={np.corrcoef(data.X[:, k], data.y)[0,1]:+.3f}" for k, p in enumerate(data.param_columns)))
