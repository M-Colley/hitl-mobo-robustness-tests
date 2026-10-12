"""The smooth-subset report's brackets (scripts/smooth_subset_and_gp_diagnostics.py).

With three landscapes a percentile landscape bootstrap can only return the range
of the three per-landscape values: each extreme is the whole resample with
probability 1/27 > 2.5%. The report prints such a bracket as a range, labelled
'range', and computes it as that range; these tests pin both facts.
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

import smooth_subset_and_gp_diagnostics as ss  # noqa: E402


def test_a_three_landscape_bootstrap_returns_the_range():
    values = np.array([0.18, 0.27, 0.39])
    _, lo, hi = ss.boot_mean(values, np.random.default_rng(ss.BOOT_SEED))
    assert (lo, hi) == pytest.approx(ss.value_range(values))
    assert ss.interval_kind(3) == "range"
    assert ss.interval_kind(6) == "bootstrap95"


def test_the_range_of_a_ratio_of_means_is_the_range_of_the_per_landscape_ratios():
    num, den = np.array([0.02, 0.05, 0.10]), np.array([1.0, 0.5, 0.6])
    lo, hi = ss.ratio_range(num, den)
    assert lo == pytest.approx(0.02) and hi == pytest.approx(1 / 6)
    # A non-positive denominator has no ratio, so there is no range either.
    assert np.isnan(ss.ratio_range(num, np.array([1.0, -0.5, 0.6]))[0])


def test_a_range_is_labelled_as_one():
    assert ss._pct(0.092, 0.02, 0.17, 1, kind="range") == "+9.2% range [+2, +17]"
    assert ss._pct(0.092, 0.02, 0.17, 1) == "+9.2% [+2, +17]"
    assert ss._num(0.5, 0.4, 0.6, "{:.2f}", "range") == "0.50 range [0.40, 0.60]"


def test_a_suppressed_share_prints_as_a_dash_with_its_reason():
    assert ss._share(0.3, 0.2, 0.4, "reference negative on 1 of 20 landscapes") == \
        "-- (reference negative on 1 of 20 landscapes)"
    assert ss._share(0.3, 0.2, 0.4, "", 0) == "+30% [+20, +40]"
    assert ss._share(0.3, 0.2, 0.4, np.nan, 0) == "+30% [+20, +40]"


def test_the_excess_selection_range_pairs_as_decompose_regret_does():
    noisy = pd.DataFrame({"dataset": ["a", "a", "b", "c"], "acquisition": "ei", "seed": [7, 8, 7, 7],
                          "deployed": [0.5, 0.7, 0.4, 0.9], "selection": [0.2, 0.2, 0.1, 0.6]})
    clean = pd.DataFrame({"dataset": ["a", "a", "b", "c"], "acquisition": "ei", "seed": [7, 8, 7, 7],
                          "deployed": [0.1, 0.1, 0.2, 0.3]})
    lo, hi = ss.excess_selection_range(noisy, clean)
    # per landscape: a 0.2 / 0.5, b 0.1 / 0.2, c 0.6 / 0.6
    assert lo == pytest.approx(0.4) and hi == pytest.approx(1.0)
