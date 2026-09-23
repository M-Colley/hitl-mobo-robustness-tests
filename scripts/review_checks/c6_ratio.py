# C6: two-sample bootstrap of the fitted-oracle companion ratio.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""C6: the fitted-oracle companion's ratio of known-function to fitted cost at 1 sigma from the first
rating, with an interval from resampling the landscapes and the datasets independently."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))
import analyse_fitted_companion as fc, analyse_boba_robustness as ab
fitted = fc.relative_magnitude(ab.load_paired(REPO / "output-fitted"))
fitted = fc.floor_fraction(fitted, fc.fitted_optimum(fitted))
synth = ab.load_paired(REPO / "output-boba"); synth = synth[synth.error_model == "gaussian"]
meta = json.loads((REPO / "output-boba" / "run_metadata.json").read_text(encoding="utf-8"))
optz = pd.Series({k: float(v["opt_z"]) for k, v in meta["landscape_stats"].items()})
synth = fc.floor_fraction(synth, optz); synth["sigma_multiple"] = synth["jitter_std"].round(2)
f = fitted[(fitted.sigma_multiple == 1.0) & (fitted.jitter_iteration == 0)].groupby("dataset")["frac"].mean()
s = synth[(synth.jitter_std == 1.0) & (synth.jitter_iteration == 0)].groupby("dataset")["frac"].mean()
print("fitted per dataset:", f.round(3).to_dict()); print("synthetic mean", s.mean().round(3), "fitted mean", f.mean().round(3), "ratio", (s.mean() / f.mean()).round(2))
for seed in (20260910, 20260923):
    rng = np.random.default_rng(seed)
    fv, sv = f.to_numpy(), s.to_numpy()
    r = np.array([sv[rng.integers(0, len(sv), len(sv))].mean() / fv[rng.integers(0, len(fv), len(fv))].mean() for _ in range(2000)])
    print(seed, "percentile 95%:", np.percentile(r, [2.5, 97.5]).round(2), "share of draws with fitted mean <= 0:", "n/a")
