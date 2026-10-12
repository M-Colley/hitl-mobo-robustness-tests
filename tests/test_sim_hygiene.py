"""Guards added on 2026-10-09 after an audit of the simulators.

* --resume must not take a run log cut off in its last row for a complete one,
  and run logs are written atomically so a killed process leaves no such file.
* --hold-until is a fraction of the budget.
* The fitted-oracle driver refuses the input-error flags it would ignore.
* code_stamp masks the home directory in argv in every spelling.
* The synthetic driver does not mark a sweep complete when an acquisition fell back.
* The qLog(N)EHVI fused C++ kernel is switched off, so multi-objective runs take
  the pure-Python path that every logged run took.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bo_sensor_error_simulation as sim  # noqa: E402

T = 6


def _log(iterations: int = T) -> pd.DataFrame:
    rows = []
    for it in range(1, iterations + 1):
        rows.append({"iteration": it, "run_id": "r", "jitter_std": 1.0, "error_model": "gaussian",
                     "simple_regret_true": 1.0 / it, "inference_simple_regret_true": 1.5 / it,
                     "regret_cum_true": float(it), "regret_avg_true": 1.0, "best_true_so_far": float(it),
                     "acq_opt_failed": False, "dataset": "toy"})
    return pd.DataFrame(rows)


def test_a_complete_log_is_resumed(tmp_path):
    path = tmp_path / "run.csv"
    sim.write_csv_atomic(_log(), path)
    loaded = sim._load_resumable_run(path, T)
    assert loaded is not None and len(loaded) == T
    assert not (tmp_path / "run.csv.tmp").exists()


def test_a_log_cut_off_in_its_last_row_is_not_resumed(tmp_path):
    path = tmp_path / "run.csv"
    _log().to_csv(path, index=False)
    text = path.read_text(encoding="utf-8")
    # drop the last few fields of the last row, as a killed write would
    cut = text.rstrip("\n").rsplit(",", 6)[0] + "\n"
    path.write_text(cut, encoding="utf-8")
    assert sim._load_resumable_run(path, T) is None


def test_a_log_with_misnumbered_trials_is_not_resumed(tmp_path):
    path = tmp_path / "run.csv"
    frame = _log()
    frame.loc[2, "iteration"] = 7
    frame.to_csv(path, index=False)
    assert sim._load_resumable_run(path, T) is None


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_hold_until_must_be_a_fraction(value):
    with pytest.raises(ValueError):
        sim.hold_until_fraction(argparse.Namespace(hold_until=value))


def test_hold_until_accepts_zero_and_the_default():
    assert sim.hold_until_fraction(argparse.Namespace(hold_until=0.0)) == 0.0
    assert sim.hold_until_fraction(argparse.Namespace(hold_until=None)) == 0.6


def test_the_fitted_driver_refuses_input_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bo_sensor_error_simulation.py", "--input-error", "slip",
                                      "--input-error-scale", "0.1"])
    monkeypatch.setattr(sim, "validate_inputs", lambda args: None)
    with pytest.raises(SystemExit) as info:
        sim.main()
    assert "input error" in str(info.value)


def test_code_stamp_masks_every_spelling_of_home(monkeypatch, tmp_path):
    home = str(Path.home())
    spellings = [home, home.replace("\\", "/")]
    if len(home) > 2 and home[1] == ":":
        spellings.append("/" + home[0].lower() + home[2:].replace("\\", "/"))
    monkeypatch.setattr(sys, "argv", ["x.py"] + [s + "/data" for s in spellings])
    stamp = sim.code_stamp(REPO)
    for arg in stamp["argv"][1:]:
        assert arg == "~/data", arg


def test_the_fused_logehvi_kernel_is_switched_off():
    import botorch.acquisition.multi_objective.logei as mo

    assert mo._load_attempted is True and mo._C is None


def test_the_synthetic_driver_refuses_to_complete_a_sweep_with_fallbacks():
    import bo_synthetic_error_simulation as synth

    args = synth.parse_args([])
    assert args.allow_acq_fallback is False
    source = (REPO / "scripts" / "bo_synthetic_error_simulation.py").read_text(encoding="utf-8")
    i = source.index("if failed_acq and not args.allow_acq_fallback:")
    assert i < source.index('(output_dir / "SWEEP_COMPLETE").write_text(')
