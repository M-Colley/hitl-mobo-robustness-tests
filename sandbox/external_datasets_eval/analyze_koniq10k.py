"""Analyze KonIQ-10k score distributions for noise-calibration value.

KonIQ-10k provides per-image rating DISTRIBUTIONS (proportions of 5-point ACR
votes c1..c5 after worker filtering, plus number of ratings c_total), a
rescaled MOS, and a per-image SD column. No per-worker raw ratings are public,
so we characterize INTER-rater dispersion per image (which upper-bounds
intra-rater noise) and verify which scale each column is on by recomputing
moments from the categorical distribution.
"""
import numpy as np
import pandas as pd

CSV = r"C:\Users\Mark\Desktop\hitl-mobo-robustness-tests\sandbox\external_datasets_eval\raw\koniq10k\koniq10k_distributions_sets.csv"

df = pd.read_csv(CSV)
print("rows:", len(df))
print("columns:", list(df.columns))
print(df.head(3).to_string())

C = df[["c1", "c2", "c3", "c4", "c5"]].to_numpy(float)
print("\nrow sums of c1..c5 (should be ~1 if proportions):",
      np.round(C.sum(axis=1), 6).min(), np.round(C.sum(axis=1), 6).max())

n = df["c_total"].to_numpy(float)
levels = np.arange(1, 6, dtype=float)

# Recompute MOS and SD on the native 5-point scale from the distribution
mos5 = (C * levels).sum(axis=1)
var5_pop = (C * (levels[None, :] - mos5[:, None]) ** 2).sum(axis=1)
# sample SD with Bessel correction using c_total
sd5_sample = np.sqrt(var5_pop * n / np.maximum(n - 1, 1))
sd5_pop = np.sqrt(var5_pop)

# Compare with the file's MOS and SD columns
mos_file = df["MOS"].to_numpy(float)
sd_file = df["SD"].to_numpy(float)

# Hypothesis A: file MOS is 100-point rescale  ((mos5-1)/4*100 ... or zscore-based)
lin_mos = (mos5 - 1.0) / 4.0 * 99.0 + 1.0
print("\ncorr(file MOS, recomputed mos5):", np.corrcoef(mos_file, mos5)[0, 1])
print("file MOS range:", mos_file.min(), mos_file.max())
print("recomputed mos5 range:", mos5.min(), mos5.max())
print("max |file MOS - linear rescale of mos5|:", np.abs(mos_file - lin_mos).max())

print("\nfile SD vs recomputed 5-pt SD:")
for name, cand in [("sample SD (Bessel)", sd5_sample), ("population SD", sd5_pop)]:
    print(f"  corr with {name}: {np.corrcoef(sd_file, cand)[0,1]:.6f}; "
          f"max abs diff {np.abs(sd_file - cand).max():.6f}; "
          f"median abs diff {np.median(np.abs(sd_file - cand)):.6f}")

# ---- headline numbers ----
def describe(x, label, scale=""):
    qs = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"{label}{scale}: mean={x.mean():.3f} sd={x.std():.3f} "
          f"p5={qs[0]:.3f} q25={qs[1]:.3f} median={qs[2]:.3f} q75={qs[3]:.3f} p95={qs[4]:.3f}")

print("\n=== Ratings per image ===")
describe(n, "c_total")

print("\n=== Per-image inter-rater SD, native 5-point ACR scale ===")
describe(sd5_sample, "SD(5pt, sample)")
describe(sd5_pop, "SD(5pt, population)")
print("As fraction of 4-unit scale range: mean", round(sd5_sample.mean() / 4, 4))

print("\n=== Per-image SD on the 0-100-style rescaled MOS scale (x25) ===")
describe(sd5_sample * 24.75, "SD(100pt approx)")

print("\n=== SEM of per-image MOS (SD/sqrt(n)) on 5-pt scale ===")
describe(sd5_sample / np.sqrt(n), "SEM(5pt)")

print("\n=== Heteroscedasticity: SD by MOS bin (5-pt scale) ===")
bins = pd.cut(mos5, bins=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
tab = pd.DataFrame({"bin": bins, "sd": sd5_sample, "mos": mos5}).groupby("bin", observed=True).agg(
    n_img=("sd", "size"), mean_sd=("sd", "mean"), median_sd=("sd", "median"))
print(tab.to_string())

print("\n=== MOS distribution (5-pt) ===")
describe(mos5, "MOS(5pt)")

# Inter-image (signal) variance vs within-image (noise) variance:
sig_var = mos5.var()
noise_var = (sd5_sample ** 2).mean()
print("\n=== Signal vs noise on 5-pt scale ===")
print(f"between-image MOS variance (signal): {sig_var:.4f} (SD {np.sqrt(sig_var):.4f})")
print(f"mean within-image rating variance (noise): {noise_var:.4f} (SD {np.sqrt(noise_var):.4f})")
print(f"single-rating reliability (ICC(1,1)-style signal/(signal+noise)): "
      f"{sig_var/(sig_var+noise_var):.4f}")

# discrete vote-share overview
print("\n=== Average vote shares ===")
print(pd.Series((C * n[:, None]).sum(axis=0) / (C * n[:, None]).sum(),
                index=["c1", "c2", "c3", "c4", "c5"]).round(4).to_string())
print("total ratings represented:", int(n.sum()))
print("\nset split counts:")
print(df["set"].value_counts().to_string())
