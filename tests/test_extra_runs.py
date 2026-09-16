"""The extra-trials count (scripts/analyse_extra_runs.py).

Two things it must get right: how many extra trials a noisy run needs (measured
from where its clean twin first reached the target, so plateaus do not count as
a head start), and which clean run is the twin -- including when the clean run
carries a variant suffix, or lives in another arm's directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyse_extra_runs as aer  # noqa: E402

STEM = "bo_sensor_error_hartmann_3_value_logei_seed7"
CLEAN = [5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0]
NOISY = [5.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0]


def _write(root: Path, name: str, regret: list[float]) -> None:
    folder = root / "hartmann_3"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"iteration": np.arange(1, len(regret) + 1), "simple_regret_true": regret}).to_csv(
        folder / f"{name}.csv", index=False)


def test_extra_trials_counts_from_where_the_clean_run_got_there():
    # clean reaches 3.0 at trial 3; noisy at trial 5
    assert aer.extra_trials(np.array(CLEAN), np.array(NOISY), k=3) == (2.0, False)
    # a clean plateau is not a head start: at k = 7 the clean run has been at 1.0 since trial 5
    assert aer.extra_trials(np.array(CLEAN), np.array(NOISY), k=7) == (2.0, False)
    # an identical run scores zero
    assert aer.extra_trials(np.array(CLEAN), np.array(CLEAN), k=6) == (0.0, False)


def test_a_run_that_never_gets_there_is_censored():
    noisy = [5.0] * 8
    extra, censored = aer.extra_trials(np.array(CLEAN), np.array(noisy), k=3)
    assert censored and extra == 8 - 3


def test_clean_twin_with_a_variant_suffix_is_found(tmp_path):
    _write(tmp_path, f"{STEM}_baseline_exact_inc-observed_max", CLEAN)
    _write(tmp_path, f"{STEM}_jittered_exact_gaussian_jit0_std1.0_inc-observed_max", NOISY)
    runs = aer.per_run_table(tmp_path, [3], [0.0])
    assert len(runs) == 1
    assert runs.iloc[0]["extra"] == 2.0 and runs.iloc[0]["variant"] == "inc-observed_max"


def test_baseline_dir_supplies_the_clean_twin(tmp_path):
    standard, arm = tmp_path / "standard", tmp_path / "arm"
    _write(standard, f"{STEM}_baseline_exact", CLEAN)
    _write(arm, f"{STEM}_baseline_exact", NOISY)          # the arm's own, handicapped twin
    _write(arm, f"{STEM}_jittered_exact_gaussian_jit0_std1.0_rep10", NOISY)
    own = aer.per_run_table(arm, [3], [0.0])
    vs_standard = aer.per_run_table(arm, [3], [0.0], baseline_dir=standard)
    assert own.iloc[0]["extra"] == 0.0
    assert vs_standard.iloc[0]["extra"] == 2.0


def test_two_clean_runs_for_one_stem_is_an_error(tmp_path):
    _write(tmp_path, f"{STEM}_baseline_exact", CLEAN)
    _write(tmp_path, f"{STEM}_baseline_exact_inc-observed_max", CLEAN)
    with pytest.raises(ValueError):
        aer.index_runs(tmp_path)
