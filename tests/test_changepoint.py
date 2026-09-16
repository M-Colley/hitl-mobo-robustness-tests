"""The three mid-study change detectors, and the matched-false-alarm comparison.

Each detector maps a stream of standardised one-step-ahead residuals to a
statistic that should stay low while the rater behaves and rise once the rater's
variance inflates. The tests pin that shape on streams whose answer is known by
construction, pin that the tuned threshold really does hit its false-alarm
budget, and pin the accounting that decides whether an alarm was true: an alarm
before the onset is false, however loudly it fired.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import changepoint_compare as cp  # noqa: E402

N0 = 15
COLS = 35


def _steady(n: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(n, COLS))


def _changed(n: int = 200, at: int = 10, scale: float = 5.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=(n, COLS))
    z[:, at:] = rng.normal(0.0, scale, size=(n, COLS - at))
    return z


PATHS = {
    "cusum": lambda Z: cp.cusum_path(Z, 0.5),
    "glr": cp.glr_path,
    "bocpd": lambda Z: cp.bocpd_path(Z, 0.02, 5),
}


@pytest.mark.parametrize("name", sorted(PATHS))
def test_every_detector_separates_a_variance_jump_from_a_steady_stream(name):
    path = PATHS[name]
    steady, changed = path(_steady()), path(_changed())
    h = float(np.quantile(steady.max(axis=1), 0.90))
    fired_steady = (cp.first_alarm(steady, h, N0) > 0).mean()
    fired_changed = (cp.first_alarm(changed, h, N0) > 0).mean()
    assert fired_steady == pytest.approx(0.10, abs=0.03)
    assert fired_changed > 0.8, f"{name} detects only {fired_changed:.0%} of a 5x variance jump"


@pytest.mark.parametrize("name", sorted(PATHS))
def test_no_detector_alarms_on_a_stream_that_never_changes(name):
    """With the threshold above every steady peak, nothing fires."""
    steady = PATHS[name](_steady())
    h = float(steady.max()) + 1.0
    assert (cp.first_alarm(steady, h, N0) > 0).sum() == 0


def test_the_cusum_accumulates_only_above_its_slack():
    """z^2 - 1 - kappa below zero is clamped away, so a quiet stream stays at 0."""
    quiet = np.zeros((1, 5))
    assert cp.cusum_path(quiet, kappa=0.5).max() == 0.0
    loud = np.full((1, 5), 4.0)     # z^2 = 16, well above 1 + kappa
    assert cp.cusum_path(loud, kappa=0.5).max() > 0.0


def test_the_glr_is_zero_when_the_variance_is_not_inflated():
    """It is one-sided: a stream QUIETER than predicted is not a fault to detect."""
    quiet = np.full((1, 6), 0.1)
    assert cp.glr_path(quiet).max() == pytest.approx(0.0)


def test_the_glr_grows_with_the_length_of_the_inflated_stretch():
    short = np.concatenate([np.ones((1, 8)), np.full((1, 2), 4.0)], axis=1)
    long = np.concatenate([np.ones((1, 8)), np.full((1, 12), 4.0)], axis=1)
    assert cp.glr_path(long).max() > cp.glr_path(short).max()


def test_bocpd_says_nothing_before_its_run_length_support_reaches_rmax():
    """Every run is short at the start, so P(r <= rmax) is identically 1 there;
    reporting it would put the tuned threshold at 1 and silence the detector."""
    path = cp.bocpd_path(_steady(n=5), hazard=0.02, rmax=5)
    assert np.all(path[:, :5] == 0.0)
    assert np.any(path[:, 5:] > 0.0)


def test_bocpd_is_a_probability():
    path = cp.bocpd_path(_steady(n=20), hazard=0.02, rmax=5)
    assert path.min() >= 0.0 and path.max() <= 1.0


def test_first_alarm_reports_the_trial_not_the_column():
    path = np.array([[0.0, 0.0, 9.0, 0.0]])
    assert cp.first_alarm(path, h=1.0, n0=N0)[0] == N0 + 1 + 2


def test_first_alarm_is_zero_when_nothing_crosses():
    assert cp.first_alarm(np.zeros((3, 4)), h=1.0, n0=N0).tolist() == [0, 0, 0]


def test_an_infinite_threshold_never_alarms():
    assert cp.first_alarm(np.full((2, 4), 1e9), h=float("inf"), n0=N0).tolist() == [0, 0]


# ---------------------------------------------------------------------------
# The accounting
# ---------------------------------------------------------------------------


def _block(changed, onset):
    return pd.DataFrame({"changed": changed, "onset": onset})


def test_an_alarm_before_the_onset_is_false_not_early():
    """The rater has not changed yet, so the detector cannot be right about it."""
    block = _block([True], [21])
    got = cp.rates(np.array([18]), block)
    assert got["detection_rate"] == 0.0


def test_an_alarm_after_the_onset_is_a_detection_and_its_delay_is_measured():
    block = _block([True, True], [21, 21])
    got = cp.rates(np.array([24, 26]), block)
    assert got["detection_rate"] == 1.0
    assert got["median_delay"] == pytest.approx(4.0)


def test_any_alarm_on_a_steady_run_is_false():
    block = _block([False, False], [0, 0])
    got = cp.rates(np.array([20, 0]), block)
    assert got["false_alarm_rate"] == pytest.approx(0.5)


def test_the_tuned_threshold_hits_its_budget():
    """The quantile of the per-run running maximum is exactly the tightest
    threshold meeting a false-alarm budget, because a run alarms iff its maximum
    exceeds the threshold."""
    steady = PATHS["cusum"](_steady(n=400))
    block = _block([False] * 400, [0] * 400)
    for target in (0.05, 0.10, 0.20):
        h = cp.threshold_for(steady, block, target, N0)
        got = cp.rates(cp.first_alarm(steady, h, N0), block)
        assert got["false_alarm_rate"] == pytest.approx(target, abs=0.02)
