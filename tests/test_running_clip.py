"""The running response clip (--response-clip running<c>): the version of clipping a study can run.

It clips each rating after the initial design to [min, max] of the ratings recorded so far, widened by c robust
SDs (1.4826 MAD) of them. It needs no knowledge of the landscape's range, so unlike the 'sample' clip it acts
on the clean twin too and has a measured price. What must hold: the bounds are those of the record so far,
the clip waits for the initial design, it names itself in the file name and ties a directory's clean runs to
one margin, and a run without it is unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bo_sensor_error_simulation as sim  # noqa: E402
import bo_synthetic_error_simulation as synth  # noqa: E402


def test_the_bounds_are_the_record_widened_by_robust_sds():
    history = [0.0, 1.0, 2.0, 3.0, 4.0]
    s = 1.4826 * 1.0                                   # MAD of 0..4 about the median 2 is 1
    out = sim.running_clip(np.array([100.0]), history, 2.0)
    assert out[0] == pytest.approx(4.0 + 2.0 * s)
    out = sim.running_clip(np.array([-100.0]), history, 2.0)
    assert out[0] == pytest.approx(0.0 - 2.0 * s)
    inside = np.array([3.5])
    assert sim.running_clip(inside, history, 2.0)[0] == 3.5


def test_too_short_or_flat_a_record_clips_nothing_and_missing_ratings_are_ignored():
    assert sim.running_clip(np.array([50.0]), [1.0, 2.0], 2.0)[0] == 50.0
    assert sim.running_clip(np.array([50.0]), [1.0] * 6, 2.0)[0] == 50.0
    history = [0.0, np.nan, 1.0, 2.0, np.nan, 3.0, 4.0]
    assert sim.running_clip(np.array([100.0]), history, 0.0)[0] == pytest.approx(4.0)
    assert np.isnan(sim.running_clip(np.array([np.nan]), [0.0, 1.0, 2.0, 3.0], 1.0)[0])


def test_the_driver_parses_names_and_guards_the_running_clip():
    assert synth.running_margin("running2") == 2.0
    assert synth.running_margin("sample") is None and synth.running_margin("none") is None
    with pytest.raises(ValueError):
        synth.running_margin("runningx")
    with pytest.raises(ValueError):
        synth.running_margin("running-1")
    assert synth.resolve_clip("running2", {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "y_opt": 1.0}) == (None, None)
    args = synth.parse_args(["--response-clip", "running2"])
    assert synth.instrument_suffix_parts(args) == ["clip-run2"]
    assert synth._clean_run_settings(args)["response_clip_running"] == 2.0
    assert "response_clip_running" not in synth._clean_run_settings(synth.parse_args([]))


def test_a_short_run_clips_the_clean_twin_only_where_a_record_is_beaten_by_more_than_the_margin(tmp_path):
    """branin, LogEI, 10 trials: with c = 0 the clean twin can never record a rating above its running maximum
    after the initial design; without the clip the same run records its true values."""
    base = ["--functions", "branin", "--acq-list", "logei", "--seeds", "7", "--iterations", "10",
            "--initial-samples", "5", "--error-models", "gaussian", "--jitter-stds", "1.0",
            "--jitter-iterations", "0", "--n-jobs", "1"]
    import pandas as pd

    plain_dir, clip_dir = tmp_path / "plain", tmp_path / "clip"
    synth.main(base + ["--output-dir", str(plain_dir)])
    synth.main(base + ["--output-dir", str(clip_dir), "--response-clip", "running0"])
    plain = pd.read_csv(next(plain_dir.rglob("*_baseline_exact.csv")))
    clipped = pd.read_csv(next(clip_dir.rglob("*_baseline_exact.csv")))
    obs = clipped["objective_observed"].to_numpy()
    assert np.allclose(plain["objective_observed"].to_numpy()[:5], obs[:5])
    for t in range(5, len(obs)):
        assert obs[t] <= np.nanmax(obs[:t]) + 1e-12
    assert np.array_equal(clipped["objective_true"].to_numpy()[:6], plain["objective_true"].to_numpy()[:6])
    noisy = next(clip_dir.rglob("*_jittered_*.csv"))
    assert noisy.name.endswith("_clip-run0.csv")
