"""The scorer of the 2026-11-07 review-round experiments (scripts/checks/review_experiments.py).

What must hold: it refuses to score once the frozen plan has changed, its t interval is the textbook one, its
two-way bootstrap collapses to a landscape bootstrap when there is one seed, its summary reports a share of
the cost only where analyse_boba_adaptations would, and the plan's control never passes with nothing compared.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "checks"))
sys.path.insert(0, str(REPO / "scripts"))

import review_experiments as rx  # noqa: E402


def test_the_plan_is_the_frozen_one_and_a_changed_plan_is_refused(tmp_path, monkeypatch):
    assert rx.verify_plan() == rx.PLAN_SHA256
    changed = tmp_path / "plan.json"
    changed.write_bytes(rx.PLAN.read_bytes() + b" ")
    monkeypatch.setattr(rx, "PLAN", changed)
    with pytest.raises(SystemExit, match="edited after it was frozen"):
        rx.verify_plan()


def test_the_control_refuses_a_checkout_without_run_logs(tmp_path, monkeypatch):
    # The run logs are git-ignored; an arm holding only its analysis files must stop the control, not pass 0 of 0.
    for d in rx.E1_DIRS.values():
        (tmp_path / d / "ackley" / "evaluation").mkdir(parents=True)
    monkeypatch.setattr(rx, "REPO", tmp_path)
    with pytest.raises(FileNotFoundError, match="no clean run logs"):
        rx.clean_twin_control()


def test_the_t_interval_is_the_textbook_one():
    x = np.array([0.1, 0.3, -0.2, 0.4, 0.0, 0.25])
    lo, hi = rx.t_interval(x)
    half = sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    assert lo == pytest.approx(x.mean() - half) and hi == pytest.approx(x.mean() + half)


def _frame(n_land: int, seeds: list[int], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(n_land):
        for s in seeds:
            rows.append({"dataset": f"l{i}", "seed": s, "gain": rng.normal(0.1, 0.05), "cost": 0.5})
    return pd.DataFrame(rows)


def test_the_two_way_bootstrap_is_a_landscape_bootstrap_with_one_seed():
    df = _frame(20, [7], np.random.default_rng(0))
    lo2, hi2 = rx.two_way(df, "gain", None, np.random.default_rng(3))
    v = df.groupby("dataset")["gain"].mean().to_numpy()
    rng = np.random.default_rng(3)
    draws = []
    for _ in range(rx.REPS):
        li = rng.integers(0, len(v), len(v))
        rng.integers(0, 1, 1)   # the seed draw the two-way bootstrap makes, here of one seed
        draws.append(v[li].mean())
    assert lo2 == pytest.approx(np.percentile(draws, 2.5)) and hi2 == pytest.approx(np.percentile(draws, 97.5))


def test_the_summary_reports_a_share_only_where_the_cost_allows():
    rng = np.random.default_rng(1)
    df = _frame(20, [7, 8], rng)
    s = rx.summarise(df, "gain", np.random.default_rng(2))
    assert s["share_suppressed"] == "" and s["share"] == pytest.approx(df["gain"].mean() / 0.5, rel=1e-9)
    df["cost"] = 0.001
    s = rx.summarise(df, "gain", np.random.default_rng(2))
    assert s["share_suppressed"] != "" and np.isnan(s["share"])
