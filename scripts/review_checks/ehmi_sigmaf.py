# B3/D5: ehmi's sigma_f on per-design means against individual ratings.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""D5: ehmi's sigma_f on the 'mean' surface (paper) vs the 'individual' surface the other two use."""
import dataclasses, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))
import bo_sensor_error_simulation as sim
cfg = Path(sys.argv[1])
sel = sim.load_oracle_selection(Path("output/best_oracle_models.json"))
for ds in sim.parse_dataset_configs(None, cfg, Path(".dataset_cache")):
    if ds.name != "ehmi":
        continue
    model = sel[(ds.name, "composite")]["best_model"]
    for target in ("mean", "individual"):
        d = dataclasses.replace(ds, oracle_target=target)
        frame = sim.load_observations(d, "composite", None, None)
        oracle = sim.build_oracle(df=frame, objective="composite", objective_columns=d.objective_map["composite"],
                                  param_columns=d.param_columns, seed=10_007, normalize=False, weights=None,
                                  oracle_model=model, oracle_augmentation="jitter", oracle_augment_repeats=2,
                                  oracle_augment_std=0.02, oracle_fast=False, oracle_target=target)
        b = sim.bounds_from_data(frame, d.param_columns)
        X = np.random.default_rng(10_007).uniform(b.low, b.high, size=(100_000, len(d.param_columns)))
        y = oracle.predict_many(X).reshape(-1)
        print(ds.name, model, target, "sigma_f=%.3f" % np.std(y, ddof=1), "nugget/sigma_f=%.2f" % (0.2692177 / np.std(y, ddof=1)))
