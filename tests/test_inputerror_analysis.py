"""The input-error analysis: the parts that could silently mislead.

The mechanism table is read across its columns -- floor against logged against
unnoticed -- so every column must describe the same work. Two versions broke
that on partial data without raising anything: the first let each column
average over whatever its own arm had finished; the second matched landscapes
and seeds but still let a half-finished task change which acquisitions a column
averaged over. The tests below build both situations deliberately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyse_boba_inputerror as ie  # noqa: E402

COND = ["jitter_std", "jitter_iteration"]


def _arm(datasets, acquisition, frac, seeds=(7,)):
    return pd.DataFrame([
        {"dataset": d, "seed": s, "acquisition": acquisition,
         "jitter_std": 0.05, "jitter_iteration": 0, "frac": frac}
        for d in datasets for s in seeds
    ])


def _row(table, arm):
    return table[table["arm"] == arm].iloc[0]


def test_every_mechanism_column_uses_the_shared_landscapes_only():
    """A landscape only the complete arm has must not drag its column."""
    slip = pd.concat([
        _arm(["a", "b", "c"], "logei", 0.10),
        _arm(["a", "b", "c"], "random", 0.02),
        _arm(["d"], "logei", 9.0),
    ])
    actual = _arm(["a"], "logei", 0.05)
    mechanism, _ = ie.mechanism_tables(slip, actual, COND)
    assert set(mechanism["n_landscapes"]) == {1}
    assert _row(mechanism, "proposed")["mean"] == pytest.approx(0.10)
    assert _row(mechanism, "actual")["mean"] == pytest.approx(0.05)
    assert _row(mechanism, "floor")["mean"] == pytest.approx(0.02)


def test_a_half_finished_task_cannot_change_a_columns_composition():
    """Landscape b has only logei so far in the logged arm, so b is not a whole task.

    Matched on (landscape, seed) alone -- the second broken version -- 'unnoticed'
    on b would average logei and qnei while 'logged' on b averaged logei only, and
    the gap between the columns would stop equalling the paired cost.
    """
    slip = pd.concat([
        _arm(["a", "b"], "logei", 0.30), _arm(["a", "b"], "qnei", 0.50),
        _arm(["a", "b"], "random", 0.01), _arm(["a", "b"], "sobol", 0.02),
    ])
    actual = pd.concat([_arm(["a", "b"], "logei", 0.20), _arm(["a"], "qnei", 0.25)])
    mechanism, mislabel = ie.mechanism_tables(slip, actual, COND)
    assert set(mechanism["n_landscapes"]) == {1}, "only landscape a is a whole task everywhere"
    gap = _row(mechanism, "proposed")["mean"] - _row(mechanism, "actual")["mean"]
    assert gap == pytest.approx(0.175)
    assert _row(mislabel, "proposed - actual")["mean"] == pytest.approx(gap)


def test_mislabelling_agrees_with_the_columns_it_is_read_against():
    slip = pd.concat([_arm(["a", "b"], "logei", 0.30), _arm(["a", "b"], "random", 0.01)])
    actual = _arm(["a", "b"], "logei", 0.20)
    mechanism, mislabel = ie.mechanism_tables(slip, actual, COND)
    gap = _row(mechanism, "proposed")["mean"] - _row(mechanism, "actual")["mean"]
    assert _row(mislabel, "proposed - actual")["mean"] == pytest.approx(gap)


def test_no_whole_task_anywhere_skips_rather_than_guesses():
    slip = pd.concat([_arm(["a"], "logei", 0.3), _arm(["a"], "qnei", 0.5),
                      _arm(["a"], "random", 0.0)])
    actual = _arm(["a"], "logei", 0.2)            # qnei missing: a is not whole
    actual_other = _arm(["b"], "qnei", 0.1)       # b exists only here
    mechanism, mislabel = ie.mechanism_tables(slip, pd.concat([actual, actual_other]), COND)
    assert mechanism.empty and mislabel.empty


def test_without_the_logged_arm_there_is_no_mislabelling_row():
    slip = pd.concat([_arm(["a"], "logei", 0.1), _arm(["a"], "random", 0.0)])
    mechanism, mislabel = ie.mechanism_tables(slip, None, COND)
    assert set(mechanism["arm"]) == {"floor", "proposed"}
    assert mislabel.empty


def test_no_slip_arm_means_no_mechanism_table():
    mechanism, mislabel = ie.mechanism_tables(None, None, COND)
    assert mechanism.empty and mislabel.empty


def test_an_unevaluated_arm_is_skipped_not_fatal(tmp_path, capsys):
    assert ie.load_arm(tmp_path, pd.Series(dtype=float), "misclick") is None
    assert "skipped" in capsys.readouterr().out
