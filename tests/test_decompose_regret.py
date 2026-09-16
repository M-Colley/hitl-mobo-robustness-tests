"""The search/selection split of a deployed design's regret.

The identity under test is

    R(deployed) = [f(x*) - max_visited f] + [max_visited f - f(deployed)]
                =      search loss        +      selection loss

so the tests pin that the two terms add back to the deployed regret, that the
selection term is never negative (a rule that ships a VISITED design cannot beat
the oracle over visited designs), that a clean run of an exact objective has no
selection loss at all, and that the shares are ratios of landscape means rather
than means of per-run ratios.
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

import decompose_regret as dr  # noqa: E402

OPT_Z = {"ackley": 2.0, "branin": 4.0}


def _runs(**overrides) -> pd.DataFrame:
    base = pd.DataFrame({
        "dataset": ["ackley", "ackley", "branin", "branin"],
        "acquisition": ["logei"] * 4,
        "seed": [7, 8, 7, 8],
        "baseline": [False] * 4,
        "error_model": ["gaussian"] * 4,
        "jitter_std": [1.0] * 4,
        "jitter_iteration": [0.0] * 4,
        "regret_best_observed": [1.0, 2.0, 4.0, 8.0],
        "regret_best_visited": [0.5, 0.5, 2.0, 2.0],
        "regret_lcb2": [0.8, 1.6, 3.0, 6.0],
    })
    for key, value in overrides.items():
        base[key] = value
    return base


def test_the_two_terms_add_back_to_the_deployed_regret():
    out = dr.decompose(_runs(), OPT_Z)
    assert np.allclose(out["search"] + out["selection"], out["deployed"])


def test_regret_is_divided_by_the_landscapes_achievable_improvement():
    out = dr.decompose(_runs(), OPT_Z)
    ackley = out[out.dataset == "ackley"]
    branin = out[out.dataset == "branin"]
    # 1.0 / opt_z 2.0, and 4.0 / opt_z 4.0: the same normalised regret from very
    # different raw numbers, which is the point of dividing.
    assert ackley["deployed"].iloc[0] == pytest.approx(0.5)
    assert branin["deployed"].iloc[0] == pytest.approx(1.0)


def test_a_landscape_absent_from_the_stats_is_left_unscaled():
    out = dr.decompose(_runs(dataset=["zzz"] * 4), {})
    assert out["deployed"].iloc[0] == pytest.approx(1.0)


def test_selection_loss_cannot_be_negative():
    """The best-observed design IS a visited design, so the oracle over visited
    designs is at least as good. A violation means the two are being scored on
    different objectives, which would silently corrupt every share."""
    bad = _runs(regret_best_visited=[5.0, 0.5, 2.0, 2.0])  # worse than best_observed
    with pytest.raises(ValueError, match="negative selection loss"):
        dr.decompose(bad, OPT_Z)


def test_floating_point_noise_is_clipped_not_raised():
    tiny = _runs(regret_best_visited=[1.0 + 1e-12, 0.5, 2.0, 2.0])
    out = dr.decompose(tiny, OPT_Z)
    assert out["selection"].min() >= 0.0


def test_a_clean_run_has_no_selection_loss():
    clean = dr.decompose(_runs(regret_best_visited=[1.0, 2.0, 4.0, 8.0], baseline=True), OPT_Z)
    assert dr.assert_clean_runs_have_no_selection_loss(clean) == pytest.approx(0.0, abs=1e-9)


def test_a_clean_run_with_selection_loss_is_an_error():
    """An exact objective rates every design exactly, so a clean run that ships
    the wrong one means the baselines are not clean or are mispaired."""
    clean = dr.decompose(_runs(baseline=True), OPT_Z)
    with pytest.raises(ValueError, match="clean run shows selection loss"):
        dr.assert_clean_runs_have_no_selection_loss(clean)


def test_the_share_is_a_ratio_of_landscape_means():
    """Not a mean of per-run ratios: a landscape with tiny regret would otherwise
    weigh as much as one with large regret."""
    rng = np.random.default_rng(0)
    noisy = dr.decompose(_runs(), OPT_Z)
    row = dr.summarise(noisy, pd.DataFrame(), rng)
    per = noisy.groupby("dataset")[["deployed", "selection"]].mean()
    expected = per["selection"].mean() / per["deployed"].mean()
    assert row["selection_share"] == pytest.approx(expected)
    assert row["n_landscapes"] == 2


def test_the_excess_share_uses_the_identically_seeded_clean_twin():
    noisy = dr.decompose(_runs(), OPT_Z)
    clean = dr.decompose(_runs(regret_best_observed=[0.4, 0.4, 1.0, 1.0],
                               regret_best_visited=[0.4, 0.4, 1.0, 1.0],
                               baseline=True), OPT_Z)
    rng = np.random.default_rng(0)
    row = dr.summarise(noisy, clean, rng)
    # Clean deployed = clean search, so excess_deployed = deployed - clean and
    # the selection term carries over whole.
    assert row["n_paired"] == 4
    assert row["mean_excess_selection"] == pytest.approx(row["mean_selection"])
    assert 0.0 < row["excess_selection_share"] <= 1.0
