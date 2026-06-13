import numpy as np
import pandas as pd

import harness

pd.set_option("display.width", 200)
data = harness.load_dataset("provoice")
df = data.df

print("=== rows per file / ordering ===")
sf = df["__source_file"].str.replace(r".*ProVoiceData", "", regex=True)
print("rows per file: min=%d max=%d" % (df.groupby("__source_file").size().min(), df.groupby("__source_file").size().max()))
print(df.groupby("__source_file").size().value_counts().to_string())
# is df ordered cond1-file before cond2-file within each user?
order_ok = True
for u in df["User_ID"].unique():
    sub = df[df["User_ID"] == u]
    conds = sub["Condition_ID"].to_numpy()
    if not (np.all(np.diff(np.where(np.diff(conds) != 0)[0]) >= 0) and conds[0] == conds.min()):
        pass
    # simpler: condition sequence should be non-decreasing
    if np.any(np.diff(conds) < 0):
        order_ok = False
        print("user", u, "condition order NOT monotone:", conds)
print("condition order monotone within all users:", order_ok)
# iteration monotone within file?
it_ok = all(np.all(np.diff(g["Iteration"].to_numpy()) > 0) for _, g in df.groupby("__source_file"))
print("iteration strictly increasing within each file:", it_ok)

print("\n=== Phase ===")
print(df["Phase"].value_counts().to_string())
print("iterations by phase:", df.groupby("Phase")["Iteration"].agg(["min", "max"]).to_string())

print("\n=== shared initial design? ===")
key = df.groupby(data.param_columns)["__source_file"].nunique().sort_values(ascending=False)
print("top X-configs by #files:")
print(key.head(8).to_string())
samp = df[df["Phase"] == "sampling"] if "sampling" in set(df["Phase"]) else None
if samp is not None:
    print("sampling rows:", len(samp), "distinct sampling X:", samp.groupby(data.param_columns).ngroups)

print("\n=== exact full-row duplicates (params + 3 constructs) ===")
cols = data.param_columns + ["Predictability", "Percieved Usefulness", "Mental Demand"]
print("dup keep=first: %d (%.1f%%)" % (df.duplicated(subset=cols).sum(), 100 * df.duplicated(subset=cols).mean()))
print("dup keep=False: %d (%.1f%%)" % (df.duplicated(subset=cols, keep=False).sum(), 100 * df.duplicated(subset=cols, keep=False).mean()))

print("\n=== condition effect ===")
print(df.groupby("Condition_ID")[["Predictability", "Percieved Usefulness", "Mental Demand"]].agg(["mean", "std"]))
comp = pd.Series(data.y, index=df.index)
print("composite by condition:")
print(comp.groupby(df["Condition_ID"]).agg(["mean", "std", "count"]).to_string())
# per-user condition gap
gap = comp.groupby([df["User_ID"], df["Condition_ID"]]).mean().unstack()
gap["diff_c2_minus_c1"] = gap[2] - gap[1]
print("per-user composite mean by condition (head) and gap stats:")
print(gap["diff_c2_minus_c1"].describe().to_string())

print("\n=== between-user variance ===")
user_means = comp.groupby(df["User_ID"]).mean()
print("var of user means: %.3f (total var %.3f) -> between-user share %.1f%%"
      % (user_means.var(ddof=0), comp.var(ddof=0), 100 * comp.groupby(df["User_ID"]).transform("mean").var(ddof=0) / comp.var(ddof=0)))
sess_means = comp.groupby(df["__source_file"]).transform("mean")
print("between-session (user x cond) share: %.1f%%" % (100 * sess_means.var(ddof=0) / comp.var(ddof=0)))

print("\n=== warm split composition under current merged groups ===")
splits = harness._warm_split_indices(data, 0.7, 10)
cond = df["Condition_ID"].to_numpy()
tr_c2 = np.mean([np.mean(cond[t] == 2) for _, t, _ in splits])
te_c2 = np.mean([np.mean(cond[te] == 2) for _, _, te in splits])
print("n warm users:", len(splits), "| mean frac condition-2 in train: %.2f | in test: %.2f" % (tr_c2, te_c2))
sizes = [(len(t), len(te)) for _, t, te in splits]
print("train/test sizes per user:", sizes[:5], "...")
