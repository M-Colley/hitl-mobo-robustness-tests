# I8/D4: the augmentation contrast per archival dataset.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""I8/D4: the augmentation contrast per dataset, with the fraction destroyed at
each magnitude in both arms. Reuses analyse_fitted_companion's own functions."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))
import analyse_fitted_companion as fc  # noqa: E402
import analyse_boba_robustness as ab  # noqa: E402

fitted = fc.relative_magnitude(ab.load_paired(REPO / "output-fitted"))
fitted = fc.floor_fraction(fitted, fc.fitted_optimum(fitted))
noaug = fc.relative_magnitude(ab.load_paired(REPO / "output-fitted-noaug"))
noaug = fc.floor_fraction(noaug, fc.fitted_optimum(noaug))
keys = ["dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed"]
m = fitted.merge(noaug, on=keys, suffixes=("_aug", "_noaug"))
m["delta"] = m["frac_noaug"] - m["frac_aug"]
print(f"paired cells: {len(m)}; pooled mean delta {m.delta.mean():+.4f} median {m.delta.median():+.4f}; "
      f"noaug worse in {100 * (m.delta > 0).mean():.1f}%")
rng = np.random.default_rng(20260923)
for ds, block in m.groupby("dataset"):
    seeds = np.sort(block.seed.unique())
    per = block.groupby("seed")["delta"].mean().reindex(seeds).to_numpy()
    draws = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(4000)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"{ds:12s} n={len(block):5d} mean delta {block.delta.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] (seed bootstrap, "
          f"{len(seeds)} seeds) median {block.delta.median():+.4f} worse {100 * (block.delta > 0).mean():.1f}%")
    for (mag, onset), c in block.groupby(["sigma_multiple_aug", "jitter_iteration"]):
        print(f"    {mag:5g}sigma onset {onset:2d}: aug {c.frac_aug.mean():+.3f}  noaug {c.frac_noaug.mean():+.3f}  "
              f"delta {c.delta.mean():+.3f}")
print(sorted(m.acquisition.unique()), sorted(m.seed.unique()), sorted(m.error_model.unique()))
