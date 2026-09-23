# B3: sensitivity of the close-pair nugget to the closeness radius.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""B3: sensitivity of the close-pair nugget to the closeness threshold (0.05*sqrt(d) in the paper)."""
import contextlib, io, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))
import calibrate_noise_from_data as cn, bo_sensor_error_simulation as sim
cfg = sys.argv[1] if len(sys.argv) > 1 else "datasets.json"
ds = sim.parse_dataset_configs(None, Path(cfg), Path(".dataset_cache"))
for d in ds:
    out = []
    for f in (0.025, 0.05, 0.10, 0.20):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rows = cn.calibrate_dataset(d, f, normalize_objective=False)
        r = next(r for r in rows if r["objective"] == "composite")
        out.append(f"f={f:.3f}: sd_close={r['sd_nn_close']:.3f} (n={r['n_close_pairs']}, thr={r['close_pair_threshold']:.2f})")
    print(d.name, f"d={len(d.param_columns)}", "sd_all_nn=%.3f sd_repeat=%.3f n_repeat=%d median_nn_dist=%.2f" % (r["sd_nn_all"], r["sd_repeat"], r["n_repeat_groups"], r["median_nn_distance"]), "|", " | ".join(out))
