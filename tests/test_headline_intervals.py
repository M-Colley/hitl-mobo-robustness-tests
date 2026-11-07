"""scripts/checks/headline_intervals.py on small synthetic tables.

The two-way (landscape x seed) bootstrap must reduce to the landscape bootstrap when there is one seed and must
have the pigeonhole bootstrap's closed-form variance; the t interval must be the textbook one (19 df, the
delta-method SE of a ratio of landscape means); point estimates must be ratios of landscape means, not ratios of
sums or means of per-run ratios; and the guards must stop on unbalanced tables and on estimates that do not
reproduce the published ones.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "checks"))

import bootstrap_coverage as bc  # noqa: E402
import headline_intervals as hi  # noqa: E402


def _ratio_table(n_land: int = 20, n_seed: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    den = rng.uniform(0.5, 2.0, (n_land, n_seed))
    num = 0.3 * den + rng.normal(0.0, 0.1, (n_land, n_seed))
    return num, den


def test_two_way_bootstrap_reduces_to_the_landscape_bootstrap_with_one_seed():
    num, den = _ratio_table(n_seed=1)
    li, si = hi.draw_indices(np.random.default_rng(1), 20, 1, 500)
    assert (si == 0).all()
    for d in (den, None):
        two, ok = hi.two_way_draws(num, d, li, si)
        land, _ = bc.estimate(num[:, 0][li], None if d is None else d[:, 0][li])
        assert ok.all()
        np.testing.assert_allclose(two, land, rtol=1e-13, atol=0.0)
        iv = hi.three_intervals(num, d, li, si)
        assert iv["two_way"] == pytest.approx(iv["percentile"], abs=1e-12)


def test_one_seed_reduction_holds_from_the_per_run_frame():
    rng = np.random.default_rng(2)
    rows = [(f"l{l:02d}", 7, rng.normal(0.3, 0.1), rng.uniform(0.5, 1.5)) for l in range(20) for _ in range(3)]
    runs = pd.DataFrame(rows, columns=["dataset", "seed", "num", "den"])
    num, den, lands = hi.landscape_seed_tables(runs, (7,))
    per = runs.groupby("dataset")[["num", "den"]].mean()
    assert lands == list(per.index)
    np.testing.assert_allclose(num[:, 0], per["num"].to_numpy(), rtol=1e-14)
    li, si = hi.draw_indices(np.random.default_rng(3), 20, 1, 400)
    iv = hi.three_intervals(num, den, li, si)
    assert iv["estimate"] == pytest.approx(hi.landscape_estimate(runs), abs=1e-14)
    assert iv["two_way"] == pytest.approx(iv["percentile"], abs=1e-12)


def test_t_interval_matches_a_hand_computation():
    rng = np.random.default_rng(4)
    li, si = hi.draw_indices(rng, 20, 1, 200)
    # a mean: xbar +/- t(19) s / sqrt(20)
    x = rng.normal(0.2, 0.1, 20)
    iv = hi.three_intervals(x[:, None], None, li, si)
    half = stats.t.ppf(0.975, 19) * x.std(ddof=1) / np.sqrt(20)
    assert iv["t"] == pytest.approx((x.mean() - half, x.mean() + half), abs=1e-12)
    # a ratio of landscape means: the linearisation u_i = (num_i - R den_i) / mean(den), SE^2 = sum u^2 / (n (n - 1))
    num, den = _ratio_table(seed=5)
    n_, d_ = num[:, 0], den[:, 0]
    r = n_.mean() / d_.mean()
    u = (n_ - r * d_) / d_.mean()
    se = np.sqrt((u ** 2).sum() / (20 * 19))
    half = stats.t.ppf(0.975, 19) * se
    iv = hi.three_intervals(num, den, li, si)
    assert iv["estimate"] == pytest.approx(r, abs=1e-14)
    assert iv["se_t"] == pytest.approx(se, rel=1e-12)
    assert iv["t"] == pytest.approx((r - half, r + half), abs=1e-12)


def test_point_estimates_are_ratios_of_landscape_means():
    # Landscape A has one run per seed, B three with a different ratio: the three candidate estimands differ.
    rows = []
    for seed in (7, 8):
        rows.append(("A", seed, 1.0, 1.0))
        rows += [("B", seed, v, 4.0) for v in (4.0, 6.0, 8.0)]
    runs = pd.DataFrame(rows, columns=["dataset", "seed", "num", "den"])
    expected = np.mean([1.0, 6.0]) / np.mean([1.0, 4.0])             # 1.4
    est = hi.landscape_estimate(runs)
    assert est == pytest.approx(expected, abs=1e-14)
    assert abs(est - runs["num"].sum() / runs["den"].sum()) > 0.05    # ratio of sums 38 / 26
    assert abs(est - (runs["num"] / runs["den"]).mean()) > 0.02      # mean of per-run ratios 1.375
    num, den, lands = hi.landscape_seed_tables(runs, (7, 8))
    assert lands == ["A", "B"]
    assert num.mean(axis=1).mean() / den.mean(axis=1).mean() == pytest.approx(expected, abs=1e-14)
    # a mean: the mean of landscape means, not of runs
    assert hi.landscape_estimate(runs[["dataset", "seed", "num"]]) == pytest.approx((1.0 + 6.0) / 2)


def test_tables_refuse_missing_cells_unequal_seed_counts_and_foreign_seeds():
    runs = pd.DataFrame({"dataset": ["A", "A", "B"], "seed": [7, 8, 7], "num": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="hold no run"):
        hi.landscape_seed_tables(runs, (7, 8))
    runs = pd.DataFrame({"dataset": ["A", "A", "A", "B", "B"], "seed": [7, 7, 8, 7, 8],
                         "num": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ValueError, match="unequal run counts"):
        hi.landscape_seed_tables(runs, (7, 8))
    with pytest.raises(ValueError, match="scope names"):
        hi.landscape_seed_tables(runs.assign(seed=7), (7, 8))


def test_two_way_variance_is_the_pigeonhole_variance_and_includes_the_seed_term():
    rng = np.random.default_rng(6)
    r_, c_ = 20, 6
    y = rng.normal(0, 1.0, (r_, 1)) + rng.normal(0, 0.7, (1, c_)) + rng.normal(0, 0.5, (r_, c_))
    li, si = hi.draw_indices(rng, r_, c_, 40000)
    draws, ok = hi.two_way_draws(y, None, li, si)
    assert ok.all()
    anova = hi.crossed_anova(y)
    assert draws.var() == pytest.approx(anova["var_pigeonhole"], rel=0.05)
    land, _ = bc.estimate(y.mean(axis=1)[li], None)
    assert land.var() < 0.8 * draws.var()          # resampling landscapes alone misses the seed term
    # the diagnostic's landscape-only variance is the t interval's SE squared, for a mean and for a ratio
    iv = hi.three_intervals(y, None, li[:200], si[:200])
    assert anova["var_landscape"] == pytest.approx(iv["se_t"] ** 2, rel=1e-12)
    num, den = _ratio_table(n_seed=4, seed=7)
    li4, si4 = hi.draw_indices(rng, 20, 4, 200)
    iv = hi.three_intervals(num, den, li4, si4)
    assert hi.crossed_anova(hi.linearised(num, den))["var_landscape"] == pytest.approx(iv["se_t"] ** 2, rel=1e-12)


def test_seed_effect_test_finds_a_shared_seed_shift_and_not_pure_noise():
    noise = np.random.default_rng(8).normal(0.0, 1.0, (20, 10))
    shift = np.linspace(-1.0, 1.0, 10)[None, :]
    assert hi.seed_effect_p(noise + shift, np.random.default_rng(9), 999) == pytest.approx(1 / 1000)
    assert hi.seed_effect_p(noise, np.random.default_rng(9), 999) > 0.05
    assert np.isnan(hi.seed_effect_p(noise[:, :1], np.random.default_rng(9), 99))


def test_conclusion_changes_name_only_flipped_verdicts():
    iv = {"percentile": (0.01, 0.20), "t": (-0.01, 0.21), "two_way": (0.02, 0.19)}
    assert hi.conclusion_changes(iv, (0.0,)) == ["t includes 0"]
    iv = {"percentile": (0.50, 0.90), "t": (0.48, 0.95), "two_way": (0.45, 1.02)}
    assert hi.conclusion_changes(iv, (0.0, 1.0)) == ["two-way includes 1"]
    iv = {"percentile": (-0.01, 0.20), "t": (-0.02, 0.21), "two_way": (0.001, 0.30)}
    mcse = {"percentile": (0.004, 0.01), "two_way": (0.002, 0.01)}
    assert hi.conclusion_changes(iv, (0.0,), mcse) == ["two-way excludes 0 (MC)"]
    assert hi.conclusion_changes({m: (0.1, 0.2) for m in ("percentile", "t", "two_way")}, (0.0, 1.0)) == []


def test_guards_stop_on_estimates_and_intervals_that_do_not_reproduce():
    q = hi.Quantity("x", "x", "", "gain", pd.DataFrame(), (7,), published=0.1234, paper="+0.123",
                    paper_value=0.123, paper_digits=3)
    hi.check_point(q, 0.12341)
    with pytest.raises(SystemExit):
        hi.check_point(q, 0.1236)                                   # off the published value in the 4th decimal
    with pytest.raises(SystemExit):
        hi.check_point(dataclasses.replace(q, paper_value=0.124), 0.1234)   # off the paper's rounding
    assert hi.check_interval("x", (0.10, 0.20), (0.01, 0.01), (0.11, 0.21), 2000, 2000) < 1.0
    with pytest.raises(SystemExit):
        hi.check_interval("x", (0.10, 0.20), (0.01, 0.01), (0.10, 0.30), 2000, 2000)


def test_source_constants_are_read_without_importing(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("import a_module_that_does_not_exist\nRULE = 'ordinal_lcb1'\n"
                 "def table(frame, opt_z, reps: int = 2000, seed: int = 3):\n    pass\n", encoding="utf-8")
    assert hi.source_constant(f, "RULE") == "ordinal_lcb1"
    assert hi.source_default(f, "table", "reps") == 2000
    assert hi.source_default(f, "table", "seed") == 3
    with pytest.raises(KeyError):
        hi.source_constant(f, "MISSING")


def test_score_runs_end_to_end_on_a_synthetic_share():
    rng = np.random.default_rng(10)
    rows = []
    for l in range(20):
        for s in range(12, 17):
            for _ in range(4):
                c = rng.uniform(0.5, 1.5)
                rows.append((f"l{l:02d}", s, 0.4 * c + rng.normal(0.0, 0.05), c))
    runs = pd.DataFrame(rows, columns=["dataset", "seed", "num", "den"])
    est = hi.landscape_estimate(runs)
    q = hi.Quantity("syn", "synthetic share", "test", "share", runs, (12, 13, 14, 15, 16), published=est)
    row = hi.score(q, boot=500, perms=200)
    assert row["estimate"] == pytest.approx(est, abs=1e-14)
    for m in ("pct", "t", "two_way"):
        assert row[f"{m}_lo"] < est < row[f"{m}_hi"]
    assert {"percentile_excludes_0", "percentile_excludes_1", "t_excludes_1", "two_way_excludes_0"} <= set(row)
    assert row["conclusion_changes"] == "none"
    assert 0.0 < row["seed_effect_p"] <= 1.0
