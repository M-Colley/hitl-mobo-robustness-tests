"""The rule that decides how many trials to spend identifying rather than searching.

The tests pin the two forces the identification term is built to trade off --
coverage (a larger candidate set can only contain a better design) against
discrimination (a larger candidate set gives a worse design more chances to win
the sitting on a lucky look) -- pin the search term's estimator, and pin that
the rule reads nothing but the log and the surrogate: the true objective enters
only when the result is scored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import budget_split as bs  # noqa: E402
import replay_end_of_study as eos  # noqa: E402


# ---------------------------------------------------------------------------
# The identification term
# ---------------------------------------------------------------------------


def test_a_noiseless_sitting_ships_the_best_candidate():
    rng = np.random.default_rng(0)
    mu, sd = np.array([1.0, 0.5, 0.0]), np.full(3, 1e-9)
    assert bs.expected_shipped_value(mu, sd, 1e-9, rng) == pytest.approx(1.0, abs=1e-6)


def test_an_uninformative_sitting_ships_at_random():
    """With look noise swamping every gap the winner is uniform over candidates."""
    rng = np.random.default_rng(0)
    mu, sd = np.array([1.0, 0.0]), np.full(2, 1e-9)
    assert bs.expected_shipped_value(mu, sd, 1e6, rng) == pytest.approx(0.5, abs=0.02)


def test_coverage_a_noiseless_sitting_never_loses_by_adding_a_candidate():
    rng = np.random.default_rng(1)
    mu = np.array([0.0, 1.0])          # the SECOND candidate is the better one
    sd = np.full(2, 1e-9)
    one = bs.expected_shipped_value(mu[:1], sd[:1], 1e-9, rng)
    two = bs.expected_shipped_value(mu, sd, 1e-9, rng)
    assert two > one


def test_discrimination_a_noisy_sitting_loses_by_adding_a_worse_candidate():
    """The force that stops k from growing without bound.

    The padding has to sit within reach of the look noise: a candidate five SDs
    below the leader never wins a sitting, so adding it changes nothing.
    """
    rng = np.random.default_rng(2)
    good = bs.expected_shipped_value(np.array([1.0, 0.9]), np.full(2, 1e-9), 1.0, rng)
    padded = bs.expected_shipped_value(np.array([1.0, 0.9, 0.2, 0.2]), np.full(4, 1e-9), 1.0, rng)
    assert padded < good


def test_a_single_candidate_needs_no_sitting():
    rng = np.random.default_rng(0)
    assert bs.expected_shipped_value(np.array([0.7]), np.array([0.1]), 1.0, rng) == pytest.approx(0.7)


def test_posterior_uncertainty_is_carried_not_ignored():
    """The candidates' latent values are drawn from the posterior, so a wide
    posterior makes the best MEAN less certain to be the best VALUE."""
    rng = np.random.default_rng(3)
    tight = bs.expected_shipped_value(np.array([1.0, 0.9]), np.full(2, 1e-9), 1e-9, rng)
    wide = bs.expected_shipped_value(np.array([1.0, 0.9]), np.full(2, 3.0), 1e-9, rng)
    assert wide > tight  # a wide posterior sometimes offers something better than mu_1


# ---------------------------------------------------------------------------
# The search term
# ---------------------------------------------------------------------------


def test_the_rate_is_the_recent_gain_per_trial():
    assert bs.recent_improvement_rate(np.array([0.0, 1.0, 2.0, 3.0]), 4, 10) == pytest.approx(1.0)


def test_the_rate_only_looks_at_the_window():
    # A big early gain, then flat: a short window must not see the early gain.
    obs = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    assert bs.recent_improvement_rate(obs, 6, window=3) == pytest.approx(0.0)
    assert bs.recent_improvement_rate(obs, 6, window=10) > 0.0


def test_the_rate_is_floored_at_zero():
    """A run whose best rating fell back (a noisy rating can do that) forgoes
    nothing by stopping, so the search term must never be negative."""
    assert bs.recent_improvement_rate(np.array([5.0, 4.0, 3.0]), 3, 10) == 0.0


def test_the_rate_of_a_run_with_no_history_is_zero():
    assert bs.recent_improvement_rate(np.array([1.0]), 1, 10) == 0.0


# ---------------------------------------------------------------------------
# The rule end to end
# ---------------------------------------------------------------------------


def _run(observed: np.ndarray, seed: int = 0) -> "eos.RunLog":
    rng = np.random.default_rng(seed)
    T = len(observed)
    X = rng.uniform(0.0, 1.0, size=(T, 2))
    return eos.RunLog(name="test", X=X, observed=observed, deployed=observed.copy(),
                      logged_inference=np.maximum.accumulate(observed), y_opt=float(observed.max()) + 1.0)


BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
GRID = (2, 3, 5, 8)


def test_a_search_still_improving_fast_keeps_its_trials():
    """When every extra trial is still worth a lot, the rule spends none on a sitting."""
    observed = np.linspace(0.0, 40.0, 40)          # a steep, unbroken climb
    out = bs.derive_k(_run(observed), BOUNDS, GRID, beta=1.0, window=10,
                      sitting_sd=0.05, rng=np.random.default_rng(0), T=40)
    assert out["k_hat"] in bs.NO_TRIAL_POLICIES


def test_a_plateaued_search_under_a_precise_sitting_buys_identification():
    """Nothing left to find and a sitting that can tell candidates apart: spend."""
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    observed += np.random.default_rng(4).normal(0.0, 0.05, size=40)
    out = bs.derive_k(_run(observed), BOUNDS, GRID, beta=1.0, window=10,
                      sitting_sd=1e-6, rng=np.random.default_rng(0), T=40)
    assert out["k_hat"] in GRID


def test_a_sitting_too_noisy_to_discriminate_is_not_bought():
    """Same plateaued run, but a sitting that cannot tell anything apart.

    A useless sitting ships a uniformly random candidate, whose expected value
    cannot beat the best posterior mean already in hand. Comparing only against
    the lcb pick made this case buy a tournament whenever lcb happened to rank
    below the posterior mean -- a reason to change the ship rule, not to spend
    trials -- which is why the rule weighs both no-trial policies.
    """
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    observed += np.random.default_rng(4).normal(0.0, 0.05, size=40)
    out = bs.derive_k(_run(observed), BOUNDS, GRID, beta=1.0, window=10,
                      sitting_sd=1e6, rng=np.random.default_rng(0), T=40)
    assert out["k_hat"] in bs.NO_TRIAL_POLICIES


def test_every_k_on_the_grid_is_scored():
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    out = bs.derive_k(_run(observed), BOUNDS, GRID, beta=1.0, window=10,
                      sitting_sd=0.1, rng=np.random.default_rng(0), T=40)
    assert set(out["scores"]) == set(bs.NO_TRIAL_POLICIES) | set(GRID)


def test_the_rule_never_reads_the_true_objective():
    """Two runs with identical ratings but different truths must choose the same k.

    This is the property that makes the rule usable: an experimenter has the
    ratings and the surrogate, never f.
    """
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    a, b = _run(observed), _run(observed)
    b.deployed = b.deployed * -3.0 + 7.0
    b.y_opt = 1e6
    ka = bs.derive_k(a, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)
    kb = bs.derive_k(b, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)
    assert ka["k_hat"] == kb["k_hat"]
