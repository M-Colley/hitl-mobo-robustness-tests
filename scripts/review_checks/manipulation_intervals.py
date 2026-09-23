# I9/E6: seed-bootstrap intervals for the manipulated-landscape contrasts.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""I9/E6: intervals for Table manipulation cells and the two contrasts the text
rests on (narrow-spike amplitude 4 -> 16; fixed-opt_z ladder d = 7 -> 11).
Same frame as make_boba_paper_tables.table_manipulation. Bootstrap resamples
seeds (the replication unit inside a landscape), keeping every acquisition of a
drawn seed; contrasts are paired on (acquisition, seed)."""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
metric = "auc_simple_regret_excess_true_postonset_per_iter"
files = sorted(glob.glob(str(REPO / "output-boba-extensions" / "*" / "evaluation" / "paired_excess_metrics.csv")))
frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
top = pd.read_csv(REPO / "output-boba" / "levy_10" / "evaluation" / "paired_excess_metrics.csv")
frame = pd.concat([frame, top[top["seed"] <= frame["seed"].max()]], ignore_index=True)
fixed = sorted(glob.glob(str(REPO / "output-boba-ladder" / "*" / "evaluation" / "paired_excess_metrics.csv")))
frame = pd.concat([frame] + [pd.read_csv(f) for f in fixed], ignore_index=True)
frame = frame[(~frame["acquisition"].isin(["random", "sobol"])) & (frame["error_model"] == "gaussian")
              & (frame["jitter_iteration"] == 0)]
print(frame.groupby("dataset")["seed"].agg(lambda s: sorted(s.unique())).to_string())
print(frame.groupby("dataset")["acquisition"].nunique().to_string())

rng = np.random.default_rng(20260923)
REPS = 4000


def cell_ci(ds, mag):
    c = frame[(frame.dataset == ds) & (frame.jitter_std == mag)]
    seeds = np.sort(c.seed.unique())
    per = c.groupby("seed")[metric].mean().reindex(seeds).to_numpy()
    draws = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(REPS)]
    return c[metric].mean(), *np.percentile(draws, [2.5, 97.5]), len(seeds)


def contrast(a, b, mag, ratio=False):
    ca = frame[(frame.dataset == a) & (frame.jitter_std == mag)].set_index(["acquisition", "seed"])[metric]
    cb = frame[(frame.dataset == b) & (frame.jitter_std == mag)].set_index(["acquisition", "seed"])[metric]
    j = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, join="inner").reset_index()
    seeds = np.sort(j.seed.unique())
    ga = j.groupby("seed")["a"].mean().reindex(seeds).to_numpy()
    gb = j.groupby("seed")["b"].mean().reindex(seeds).to_numpy()
    stat = (lambda x, y: y.mean() / x.mean()) if ratio else (lambda x, y: y.mean() - x.mean())
    draws = []
    for _ in range(REPS):
        i = rng.integers(0, len(seeds), len(seeds))
        draws.append(stat(ga[i], gb[i]))
    return stat(ga, gb), *np.percentile(draws, [2.5, 97.5]), len(seeds), len(j)


print("\ncells: mean [seed-bootstrap 95%] (n seeds)")
for ds in ["bump_a4_w0.05", "bump_a16_w0.05", "bump_a4_w0.15", "bump_a16_w0.15", "levy_4d", "levy_7d",
           "levy_10", "bump_d4", "bump_d7", "bump_d11"]:
    parts = []
    for mag in [0.05, 0.25, 1.0, 5.0]:
        m, lo, hi, n = cell_ci(ds, mag)
        parts.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"{ds:16s} n={n}: " + " | ".join(parts))

print("\ncontrasts (paired on acquisition x seed, seed bootstrap)")
for mag in [0.05, 0.25, 1.0, 5.0]:
    d, lo, hi, n, npair = contrast("bump_a4_w0.05", "bump_a16_w0.05", mag)
    print(f"narrow spike a4->a16 at {mag:g}: diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (seeds {n}, pairs {npair})")
for mag in [0.05, 0.25, 1.0, 5.0]:
    d, lo, hi, n, npair = contrast("bump_a4_w0.15", "bump_a16_w0.15", mag)
    print(f"wide spike a4->a16 at {mag:g}: diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (seeds {n}, pairs {npair})")
for mag in [0.25, 1.0, 5.0]:
    r, lo, hi, n, npair = contrast("bump_d7", "bump_d11", mag, ratio=True)
    print(f"fixed-optz ladder d7->d11 at {mag:g}: ratio {r:.2f} [{lo:.2f}, {hi:.2f}] (seeds {n}, pairs {npair})")
    r, lo, hi, n, npair = contrast("bump_d4", "bump_d11", mag, ratio=True)
    print(f"fixed-optz ladder d4->d11 at {mag:g}: ratio {r:.2f} [{lo:.2f}, {hi:.2f}] (seeds {n}, pairs {npair})")
    r, lo, hi, n, npair = contrast("levy_7d", "levy_10", mag, ratio=True)
    print(f"levy d7->d11 at {mag:g}: ratio {r:.2f} [{lo:.2f}, {hi:.2f}] (seeds {n}, pairs {npair})")
