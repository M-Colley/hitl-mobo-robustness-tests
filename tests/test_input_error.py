"""The input-error arm: the person acts on the wrong design.

Every other error model corrupts the reported value -- a wrong value at the
right location. These tests cover the other corruption: the optimizer proposes
x, the person experiences x' != x and rates x' honestly, and the rating is filed
against x. A right value at the wrong location.

The properties that matter and are easy to get wrong:
  * the baseline run must NEVER slip, or the pairing measures nothing;
  * with the arm switched off the run must be bit-identical to the ordinary one;
  * the logged design and the evaluated design must be the ones the
    input_error_recorded setting says they are, because getting that backwards
    silently inverts the whole result.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bo_sensor_error_simulation as sim  # noqa: E402


def _config(**overrides):
    base = dict(
        iterations=10,
        jitter_iteration=0,
        jitter_std=0.0,
        single_error=False,
        initial_samples=3,
        candidate_pool=64,
        objective="composite",
        objective_columns=["value"],
        param_columns=["x0", "x1"],
        seed=7,
        error_model="none",
        error_bias=0.0,
        error_spike_prob=0.0,
        error_spike_std=0.0,
        dropout_strategy="hold_last",
        normalize_objective=False,
        objective_weights=None,
        acq_num_restarts=2,
        acq_raw_samples=16,
        acq_maxiter=20,
        acq_mc_samples=16,
        ref_point=None,
    )
    base.update(overrides)
    return sim.SimulationConfig(**base)


BOUNDS = sim.Bounds(low=np.array([-5.0, 0.0]), high=np.array([10.0, 15.0]))


# ---------------------------------------------------------------------------
# apply_input_error
# ---------------------------------------------------------------------------


def test_none_is_a_no_op():
    cfg = _config(input_error_model="none", input_error_scale=0.5)
    x = np.array([1.0, 2.0])
    out, slipped = sim.apply_input_error(x, 5, cfg, np.random.default_rng(0), BOUNDS)
    assert not slipped
    assert np.array_equal(out, x)


def test_zero_scale_is_a_no_op():
    """A swept grid that includes 0 must mean 'no error', not 'degenerate draw'."""
    cfg = _config(input_error_model="slip", input_error_scale=0.0)
    x = np.array([1.0, 2.0])
    out, slipped = sim.apply_input_error(x, 5, cfg, np.random.default_rng(0), BOUNDS)
    assert not slipped
    assert np.array_equal(out, x)


@pytest.mark.parametrize("model", ["slip", "misclick"])
def test_exact_up_to_and_including_the_onset(model):
    """Same convention as the response models: first corrupted trial is t0 + 1."""
    cfg = _config(input_error_model=model, input_error_scale=1.0, jitter_iteration=4)
    x = np.array([1.0, 2.0])
    for iteration in range(0, 5):
        out, slipped = sim.apply_input_error(
            x, iteration, cfg, np.random.default_rng(0), BOUNDS
        )
        assert not slipped, f"slipped at t={iteration}, at or before the onset"
        assert np.array_equal(out, x)
    _, slipped = sim.apply_input_error(x, 5, cfg, np.random.default_rng(0), BOUNDS)
    assert slipped


def test_single_error_slips_exactly_once():
    cfg = _config(
        input_error_model="slip", input_error_scale=0.5, jitter_iteration=2,
        single_error=True,
    )
    x = np.array([1.0, 2.0])
    fired = [
        sim.apply_input_error(x, t, cfg, np.random.default_rng(t), BOUNDS)[1]
        for t in range(0, 9)
    ]
    assert fired == [False, False, False, True, False, False, False, False, False]


@pytest.mark.parametrize("model", ["slip", "misclick"])
def test_stays_inside_the_box(model):
    """A slider that overshoots stops at its end; it does not leave the box."""
    cfg = _config(input_error_model=model, input_error_scale=0.9)
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = BOUNDS.low + rng.random(2) * (BOUNDS.high - BOUNDS.low)
        out, _ = sim.apply_input_error(x, 5, cfg, rng, BOUNDS)
        assert np.all(out >= BOUNDS.low - 1e-12)
        assert np.all(out <= BOUNDS.high + 1e-12)


def test_slip_scales_with_each_coordinate_range():
    """The scale is a FRACTION OF THE RANGE, so a wider axis slips further."""
    cfg = _config(input_error_model="slip", input_error_scale=0.05)
    rng = np.random.default_rng(2)
    centre = (BOUNDS.low + BOUNDS.high) / 2.0
    deltas = np.array(
        [sim.apply_input_error(centre, 5, cfg, rng, BOUNDS)[0] - centre for _ in range(4000)]
    )
    widths = BOUNDS.high - BOUNDS.low
    observed = deltas.std(axis=0)
    expected = 0.05 * widths
    assert np.allclose(observed, expected, rtol=0.12), (observed, expected)


def test_misclick_fires_at_about_its_probability():
    cfg = _config(input_error_model="misclick", input_error_scale=0.25)
    rng = np.random.default_rng(3)
    x = np.array([1.0, 2.0])
    fired = sum(sim.apply_input_error(x, 5, cfg, rng, BOUNDS)[1] for _ in range(4000))
    assert 0.22 < fired / 4000 < 0.28


def test_misclick_lands_somewhere_else_entirely():
    """Not a perturbation of x: a different design, drawn across the whole box."""
    cfg = _config(input_error_model="misclick", input_error_scale=1.0)
    rng = np.random.default_rng(4)
    x = BOUNDS.low.copy()
    landed = np.array([sim.apply_input_error(x, 5, cfg, rng, BOUNDS)[0] for _ in range(2000)])
    # Spread over the box, not clustered at the proposal.
    assert np.all(landed.std(axis=0) > 0.2 * (BOUNDS.high - BOUNDS.low))


def test_unknown_model_is_rejected():
    cfg = _config(input_error_model="typo", input_error_scale=0.1)
    with pytest.raises(ValueError, match="Unknown input error model"):
        sim.apply_input_error(np.array([1.0, 2.0]), 5, cfg, np.random.default_rng(0), BOUNDS)


# ---------------------------------------------------------------------------
# Interaction with the response channel
# ---------------------------------------------------------------------------


def test_response_error_none_leaves_the_rating_exact():
    """The input arm's premise: the person rates what they touched, honestly."""
    cfg = _config(error_model="none", jitter_std=3.0)
    true_value = np.array([1.25])
    observed, magnitude = sim.apply_sensor_error(
        true_value=true_value, iteration=5, config=cfg,
        rng=np.random.default_rng(0), previous_observed=true_value,
    )
    assert np.array_equal(observed, true_value)
    assert np.array_equal(magnitude, np.zeros_like(true_value))


def test_response_error_none_declares_no_observation_noise():
    cfg = _config(error_model="none", jitter_std=3.0, observation_noise="known")
    assert sim.known_noise_variance(5, cfg, apply_error=True) == pytest.approx(1e-6)


def test_none_consumes_no_randomness():
    """Otherwise the input channel's draws shift and the arm is not reproducible."""
    cfg = _config(error_model="none", jitter_std=3.0)
    rng = np.random.default_rng(11)
    sim.apply_sensor_error(
        true_value=np.array([1.0]), iteration=5, config=cfg, rng=rng,
        previous_observed=np.array([1.0]),
    )
    assert rng.bit_generator.state == np.random.default_rng(11).bit_generator.state
