"""Comparisons against ratings: the invariance, and its limits.

The claim under test is exact, not statistical. A fault that acts on every design
of one sitting through the same STRICTLY increasing map leaves every comparison
unchanged, so a comparison loop is bit-identical to its clean twin. A map that is
monotone but not strictly so (a saturating scale) does not, and a fault drawn
afresh per judgement does not. These tests pin all three, plus the budget parity
that makes the comparison fair and the fallback that keeps a failed fit in the
design instead of deleting the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402
import elicitation_compare as ec  # noqa: E402


def _cell(**over):
    task = dict(dataset="branin", elicitation="rating", error_model="gaussian", magnitude=1.0,
                seed=7, apply_error=True, iterations=12, initial_samples=5, candidate_pool=32,
                spike_prob=0.15, spike_sd=5.0, ceiling_quantile=0.9,
                stats_path=str(bb.DEFAULT_STATS_PATH))
    task.update(over)
    return ec.run_cell(task)


# ---------------------------------------------------------------------------
# The rater
# ---------------------------------------------------------------------------


def test_a_shared_fault_moves_every_design_of_a_sitting_alike():
    rng = np.random.default_rng(0)
    values = np.array([1.0, 2.0, 3.0])
    got = ec.perceive(values, "bias", 2.0, 0, 20, None, 0.15, 5.0, rng)
    assert np.allclose(got - values, 2.0)


def test_a_shared_fault_preserves_every_ordering():
    """The whole claim in one line: a strictly increasing map keeps the ranking."""
    rng = np.random.default_rng(0)
    values = np.array([1.0, 2.0, 3.0, 2.5])
    for model, mag in (("bias", 3.0), ("drift", 4.0)):
        got = ec.perceive(values, model, mag, 7, 20, None, 0.15, 5.0, rng)
        assert np.array_equal(np.argsort(got), np.argsort(values))


def test_drift_grows_with_the_trial_but_stays_shared_within_one():
    rng = np.random.default_rng(0)
    values = np.array([1.0, 2.0])
    early = ec.perceive(values, "drift", 4.0, 1, 20, None, 0.15, 5.0, rng)
    late = ec.perceive(values, "drift", 4.0, 19, 20, None, 0.15, 5.0, rng)
    assert np.allclose(early - values, early[0] - values[0])   # shared within the sitting
    assert late[0] - values[0] > early[0] - values[0]          # and larger later


def test_an_idiosyncratic_fault_does_not_move_designs_alike():
    rng = np.random.default_rng(0)
    values = np.zeros(200)
    got = ec.perceive(values, "gaussian", 1.0, 0, 20, None, 0.15, 5.0, rng)
    assert got.std() > 0.5


def test_a_ceiling_is_monotone_but_not_strictly():
    """Two designs above the cap become indistinguishable, which is what breaks
    the comparison exactly where a study needs it: near the optimum."""
    rng = np.random.default_rng(0)
    got = ec.perceive(np.array([5.0, 9.0]), "ceiling", 0.0, 0, 20, 3.0, 0.0, 0.0, rng)
    assert got[0] == got[1] == 3.0


def test_no_fault_leaves_the_values_alone():
    rng = np.random.default_rng(0)
    values = np.array([1.0, 2.0])
    assert np.array_equal(ec.perceive(values, "none", 5.0, 3, 20, None, 0.15, 5.0, rng), values)


# ---------------------------------------------------------------------------
# The loops
# ---------------------------------------------------------------------------


def test_the_two_loops_cost_the_same_number_of_judgements():
    """T judgements each; the comparison loop sees one extra DESIGN, which costs
    no human effort, and that is the budget that matters."""
    rating = _cell(elicitation="rating", iterations=12)
    pairwise = _cell(elicitation="pairwise", iterations=12)
    assert rating["n_designs"] == 12
    assert pairwise["n_designs"] == 13


@pytest.mark.parametrize("model", ["drift", "bias"])
def test_a_comparison_loop_is_bit_identical_to_its_clean_twin_under_a_shared_fault(model):
    """Not approximately: the comparisons are the same comparisons, so the run is
    the same run and it ships the same design."""
    noisy = _cell(elicitation="pairwise", error_model=model, apply_error=True)
    clean = _cell(elicitation="pairwise", error_model=model, apply_error=False)
    assert noisy["shipped_true"] == clean["shipped_true"]


def test_a_rating_loop_is_not_immune_to_drift():
    """The contrast that makes the previous test mean something."""
    noisy = _cell(elicitation="rating", error_model="drift", magnitude=5.0, apply_error=True)
    clean = _cell(elicitation="rating", error_model="drift", magnitude=5.0, apply_error=False)
    assert noisy["shipped_true"] != clean["shipped_true"]


def test_a_comparison_loop_is_not_immune_to_an_idiosyncratic_fault():
    noisy = _cell(elicitation="pairwise", error_model="gaussian", magnitude=5.0, apply_error=True)
    clean = _cell(elicitation="pairwise", error_model="gaussian", magnitude=5.0, apply_error=False)
    assert noisy["shipped_true"] != clean["shipped_true"]


def test_a_failed_fit_is_counted_not_dropped():
    """Deleting the runs whose fit fails would delete the hardest runs and
    flatter whichever loop fails more often."""
    out = _cell(elicitation="pairwise")
    assert "fit_failures" in out and out["fit_failures"] >= 0
