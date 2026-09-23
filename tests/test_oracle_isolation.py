"""oracle_isolation._fraction: the fraction-of-the-floor-gap metric on either response.

Since 2026-09-22 the oracle-isolation comparison is made on the trajectory
response (the companion arm's) and on the deployed design (the paper's primary
estimand), so _fraction takes the response column as an argument. The floor is
the mean clean best of the model-free floors, and each learner's response is
divided by (optimum - floor).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import oracle_isolation as iso

DEPLOYED = "final_inference_simple_regret_excess_true"


def _frame() -> pd.DataFrame:
    # Floors: clean bests 1, 2, 3 -> floor 2. Learners and the unlisted qkg carry
    # clean bests far from that, so pooling them into the floor would show.
    rows = [
        # acquisition  clean best  trajectory  deployed
        ("random",     1.0,        99.0,       99.0),
        ("logei",      100.0,      1.0,        0.4),
        ("sobol",      3.0,        99.0,       99.0),
        ("qnei",       100.0,      2.0,        0.8),
        ("random",     2.0,        99.0,       99.0),
        ("ucb",        100.0,      3.0,        1.2),
        ("qkg",        50.0,       7.0,        7.0),   # neither a learner nor a floor
        ("ei",         100.0,      4.0,        2.0),
    ]
    return pd.DataFrame(rows, columns=["acquisition", iso.BASELINE_BEST, iso.RESPONSE, DEPLOYED])


def test_the_two_responses_are_the_trajectory_and_the_deployed_design():
    assert iso.RESPONSES == {"": iso.RESPONSE, "_deployed": DEPLOYED}
    assert iso.RESPONSE == "auc_simple_regret_excess_true_postonset_per_iter"


def test_default_response_is_the_trajectory():
    frame = _frame()
    out = iso._fraction(frame, optimum=6.0)          # gap = 6 - mean(1, 3, 2) = 4
    assert out["acquisition"].tolist() == ["logei", "qnei", "ucb", "ei"]
    np.testing.assert_allclose(out["frac"].to_numpy(), [0.25, 0.5, 0.75, 1.0], rtol=0, atol=1e-15)
    pd.testing.assert_frame_equal(out, iso._fraction(frame, 6.0, iso.RESPONSE))


def test_deployed_response_uses_its_own_column_over_the_same_floor_gap():
    frame = _frame()
    trajectory = iso._fraction(frame, 6.0, iso.RESPONSE)
    deployed = iso._fraction(frame, 6.0, DEPLOYED)
    np.testing.assert_allclose(deployed["frac"].to_numpy(), [0.1, 0.2, 0.3, 0.5], rtol=0, atol=1e-15)
    # Same learners, same rows, only the numerator changes.
    assert deployed.index.tolist() == trajectory.index.tolist() == [1, 3, 5, 7]
    np.testing.assert_allclose(deployed["frac"] * 4.0, deployed[DEPLOYED], rtol=0, atol=1e-15)
    np.testing.assert_allclose(trajectory["frac"] * 4.0, trajectory[iso.RESPONSE], rtol=0, atol=1e-15)


def test_the_floor_is_the_floors_mean_and_the_input_is_not_modified():
    frame = _frame()
    before = frame.copy()
    out = iso._fraction(frame, optimum=10.0, response=DEPLOYED)   # gap = 10 - 2 = 8
    np.testing.assert_allclose(out["frac"].to_numpy(), np.array([0.4, 0.8, 1.2, 2.0]) / 8.0,
                               rtol=0, atol=1e-15)
    assert "frac" not in frame.columns
    pd.testing.assert_frame_equal(frame, before)


def test_an_unknown_response_column_is_an_error():
    with pytest.raises(KeyError):
        iso._fraction(_frame(), 6.0, "no_such_response")
