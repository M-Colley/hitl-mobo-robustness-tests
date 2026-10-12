"""The current simulator reproduces a logged main-sweep run, trial for trial.

One configuration of the main sweep (output-boba/): branin, LogEI, seed 7,
gaussian error at 1 landscape SD from trial 21, and its clean twin. Both are
rerun with the current code and compared column by column with the logged
per-trial CSVs (run_id and the wall-clock fit time excluded). The comparison is
exact: the runs are fully determined by their seeds, and a code change that
moves any logged number, however little, should fail here and be explained.

output-boba/ is git-ignored and lives only on the simulation machine, so the
test skips elsewhere. It takes about 25 s (two 50-trial runs on a 2-d landscape).
scripts/check_provenance.py runs the same comparison for one run of every arm.

Exactness also needs the numerical stack the run was logged with: under torch
2.14.1 a rerun first differs at trial 8 by about 1e-6, and later designs can
diverge. When the installed torch, BoTorch, gpytorch, numpy or scipy differs from
the versions recorded in output-boba/run_metadata.json, the test checks only
what no library version can move (row count, columns, the initial random
designs) and then skips the exact comparison;
scripts/checks/version_sensitivity.py measures what such a change does to the
reported numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

ARM = REPO / "output-boba"
FUNCTION, ACQ, SEED = "branin", "logei", 7
ERROR_MODEL, STD, ONSET = "gaussian", 1.0, 20
NOISY = f"bo_sensor_error_{FUNCTION}_value_{ACQ}_seed{SEED}_jittered_exact_{ERROR_MODEL}_jit{ONSET}_std{STD}.csv"
CLEAN = f"bo_sensor_error_{FUNCTION}_value_{ACQ}_seed{SEED}_baseline_exact.csv"

pytestmark = pytest.mark.skipif(
    not (ARM / "run_metadata.json").is_file() or not (ARM / FUNCTION / NOISY).is_file()
    or not (ARM / FUNCTION / CLEAN).is_file(),
    reason="the main sweep (output-boba/, git-ignored) is not on this machine",
)


@pytest.fixture(scope="module")
def rerun(tmp_path_factory):
    import json

    import bo_synthetic_error_simulation as synth
    import check_provenance as cp

    meta = json.loads((ARM / "run_metadata.json").read_text(encoding="utf-8"))
    parser, defaults = cp._capture_parser(lambda: synth.parse_args([]))
    args = cp.namespace_from_metadata(parser, defaults, meta["args"])
    args.resume = False
    args.baseline_run = True
    stats = synth.bb.load_stats(Path(args.stats_path))
    out = tmp_path_factory.mktemp("regression") / FUNCTION
    out.mkdir()
    synth.run_task(synth.Task(function=FUNCTION, seed=SEED), args, stats[FUNCTION], [ACQ],
                   [ERROR_MODEL], [STD], [ONSET], out, None)
    return out


NUMERICAL_STACK = ("torch", "botorch", "gpytorch", "numpy", "scipy")
INITIAL_SAMPLES = 5


def stack_mismatches() -> list[str]:
    """The numerically relevant packages whose installed version differs from the logged one."""
    import importlib.metadata as md
    import json

    recorded = json.loads((ARM / "run_metadata.json").read_text(encoding="utf-8")).get("package_versions") or {}
    out = []
    for name in NUMERICAL_STACK:
        if name not in recorded:
            continue
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            installed = "absent"
        if name == "torch":
            import torch
            installed = torch.__version__
        if installed != recorded[name]:
            out.append(f"{name} {installed} (logged with {recorded[name]})")
    return out


@pytest.mark.parametrize("name", [NOISY, CLEAN])
def test_the_current_code_reproduces_the_logged_run(rerun, name):
    import check_provenance as cp
    import pandas as pd

    result = cp.compare_runs(ARM / FUNCTION / name, rerun / name)
    assert result["n_rows_logged"] == result["n_rows_new"] == 50
    # every logged column is still written (new columns may be added)
    assert result["columns_only_logged"] == ""
    mismatches = stack_mismatches()
    if mismatches:
        # The initial random designs do not touch the GP, so no library version may move them.
        logged = pd.read_csv(ARM / FUNCTION / name, float_precision="round_trip")
        new = pd.read_csv(rerun / name, float_precision="round_trip")
        design = [c for c in logged.columns if c.startswith("x") and c[1:].isdigit()]
        assert design, "no design columns found"
        pd.testing.assert_frame_equal(logged[design].head(INITIAL_SAMPLES), new[design].head(INITIAL_SAMPLES))
        pytest.skip("exact reproduction needs the logged numerical stack; installed " + "; ".join(mismatches))
    assert result["status"] == "exact", result
