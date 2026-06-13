"""Final calibration numbers from the OFFICIAL KonIQ-10k scores file.

koniq10k_scores_and_distributions.csv: c1..c5 = raw vote counts on the 5-point
ACR scale (after reliability screening of workers), c_total = sum, MOS = mean
on 1..5, SD = sample SD on 1..5, MOS_zscore = per-user z-scored then rescaled
to ~[1,100].
"""
import numpy as np
import pandas as pd

CSV = r"C:\Users\Mark\Desktop\hitl-mobo-robustness-tests\sandbox\external_datasets_eval\raw\koniq10k\koniq10k_scores_and_distributions.csv"
df = pd.read_csv(CSV)
print("rows:", len(df), "| columns:", list(df.columns))

C = df[["c1", "c2", "c3", "c4", "c5"]].to_numpy(float)
n = C.sum(axis=1)
assert np.allclose(n, df["c_total"]), "c_total mismatch"
levels = np.arange(1, 6, dtype=float)
mos5 = (C * levels).sum(axis=1) / n
var_pop = (C * (levels[None, :] - mos5[:, None]) ** 2).sum(axis=1) / n
sd5 = np.sqrt(var_pop * n / (n - 1))

print("MOS column == recomputed mean:", np.abs(mos5 - df["MOS"]).max())
print("SD column == recomputed sample SD:", np.abs(sd5 - df["SD"]).max())

def describe(x, label):
    qs = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"{label}: mean={x.mean():.3f} median={qs[2]:.3f} "
          f"p5={qs[0]:.3f} q25={qs[1]:.3f} q75={qs[3]:.3f} p95={qs[4]:.3f}")

print("\nTotal ratings:", int(n.sum()), "| ratings/image:", round(n.mean(), 1))
print("\n-- Per-image inter-rater SD, 5-point ACR scale --")
describe(sd5, "SD(1..5)")
print("fraction of 4-unit range: mean", round(sd5.mean() / 4, 4),
      "median", round(np.median(sd5) / 4, 4))

print("\n-- MOS (1..5) --")
describe(mos5, "MOS")
sig_sd = mos5.std()
noise_sd = np.sqrt((sd5 ** 2).mean())
print(f"\nsignal SD (between-image): {sig_sd:.4f}")
print(f"noise SD (RMS within-image): {noise_sd:.4f}")
print(f"noise/signal SD ratio: {noise_sd/sig_sd:.3f}")
print(f"single-rating reliability sig^2/(sig^2+noise^2): "
      f"{sig_sd**2/(sig_sd**2+noise_sd**2):.4f}")

print("\n-- MOS_zscore scale (~1..100, worker-bias removed by z-scoring) --")
mz = df["MOS_zscore"].to_numpy(float)
describe(mz, "MOS_zscore")
# Linear map between MOS and MOS_zscore to express SD on the zscore scale
A = np.polyfit(mos5, mz, 1)
resid = mz - np.polyval(A, mos5)
print(f"linear fit MOS_zscore ~ {A[0]:.2f}*MOS + {A[1]:.2f}; resid SD {resid.std():.2f}"
      f" (R2={1-resid.var()/mz.var():.4f})")
print(f"per-image SD mapped to zscore scale: mean ~ {A[0]*sd5.mean():.2f} "
      f"(of ~{mz.max()-mz.min():.0f}-unit observed range)")

print("\n-- Heteroscedasticity: mean SD by MOS bin --")
tab = pd.DataFrame({"mos": mos5, "sd": sd5}).groupby(
    pd.cut(mos5, np.arange(1, 5.51, 0.5)), observed=True)["sd"].agg(["size", "mean", "median"])
print(tab.round(3).to_string())

print("\n-- Pooled vote shares --")
print(pd.Series(C.sum(axis=0) / C.sum(), index=["1", "2", "3", "4", "5"]).round(4).to_string())
