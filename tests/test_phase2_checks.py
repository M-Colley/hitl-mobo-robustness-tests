"""Producers of numbers the paper quotes that had none, or had a stale one.

The reach trial of the extra-trial appendix (analyse_extra_runs.py), the
instrument screen's seed set (design_rules_from_pilot.py), the rank rules
tested in the scope of their recoveries (heldout_remedies.py), the register
runner's list of checks, and the dataset scale the tie-break check passes to
analyse_boba_adaptations.paired_frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
for p in (SCRIPTS, SCRIPTS / "review_checks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analyse_extra_runs as aer  # noqa: E402
import run_review_checks as rrc  # noqa: E402

STEM = "bo_sensor_error_hartmann_3_value_logei_seed7"
CLEAN = [5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0]
NOISY = [5.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0]


def _write(root: Path, name: str, regret: list[float]) -> None:
    folder = root / "hartmann_3"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"iteration": np.arange(1, len(regret) + 1), "simple_regret_true": regret}).to_csv(
        folder / f"{name}.csv", index=False)


# ---------------------------------------------------------------------------
# analyse_extra_runs: the reach trial
# ---------------------------------------------------------------------------


def test_the_reach_trial_is_origin_plus_extra(tmp_path):
    _write(tmp_path, f"{STEM}_baseline_exact", CLEAN)
    _write(tmp_path, f"{STEM}_jittered_exact_gaussian_jit0_std1.0", NOISY)
    run = aer.per_run_table(tmp_path, [7], [0.0]).iloc[0]
    # the clean run reached its trial-7 regret (1.0) at trial 5; the noisy run at trial 7
    assert run["clean_origin"] == 5 and run["extra"] == 2.0 and run["reach_trial"] == 7
    assert run["reach_over_k"] == pytest.approx(1.0)
    assert run["multiplier"] == pytest.approx((7 + 2) / 7)    # kept: k + extra, not the reach


def test_a_censored_run_reaches_at_the_budget(tmp_path):
    _write(tmp_path, f"{STEM}_baseline_exact", CLEAN)
    _write(tmp_path, f"{STEM}_jittered_exact_gaussian_jit0_std1.0", [5.0] * 8)
    run = aer.per_run_table(tmp_path, [3], [0.0]).iloc[0]
    assert bool(run["censored"]) and run["reach_trial"] == 8 and run["clean_origin"] == 3


def test_the_summary_appends_the_medians_after_every_earlier_column(tmp_path):
    _write(tmp_path, f"{STEM}_baseline_exact", CLEAN)
    _write(tmp_path, f"{STEM}_jittered_exact_gaussian_jit0_std1.0", NOISY)
    s = aer.summarise(aer.per_run_table(tmp_path, [3, 7], [0.0]))
    assert list(s.columns[-3:]) == ["median_clean_origin", "median_reach_trial", "median_reach_over_k"]
    assert s.loc[s.k == 7, "median_reach_trial"].iloc[0] == 7


BUDGET100 = REPO / "output-boba-budget100" / "analysis" / "extra_runs.csv"


@pytest.mark.skipif(not BUDGET100.is_file(), reason="budget-100 analysis not present")
def test_app_b7_reach_trial_has_a_producer():
    e = pd.read_csv(BUDGET100)
    r = e[(e.error_model == "gaussian") & e.variant.isna() & (e.jitter_std == 1.0) & (e.jitter_iteration == 0)
          & (e.tolerance == 0.01) & (e.k == 25)].iloc[0]
    assert r["censored_fraction"] < 0.5                       # so the medians are exact
    assert r["median_extra"] == pytest.approx(40.5)
    assert r["median_reach_trial"] == pytest.approx(56.0)
    assert r["median_reach_over_k"] == pytest.approx(2.24)


# ---------------------------------------------------------------------------
# design_rules_from_pilot: the published instrument screen is seeds 7-11
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "output-boba-ceiling").is_dir()
                    or not (REPO / "output-boba" / "analysis" / "cell_means.csv").is_file(),
                    reason="capped-scale arm or main sweep not present")
def test_the_instrument_screen_reproduces_on_the_capped_arms_first_five_seeds():
    import os

    import boba_benchmarks as bb
    import design_rules_from_pilot as dr

    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        inst = dr.instrument(Path("output-boba-ceiling"), Path("output-boba"), stats, opt_z, 0.9,
                             np.random.default_rng(dr.SEED), {7, 8, 9, 10, 11})
    finally:
        os.chdir(cwd)
    fit = dr.screen_fit(inst).set_index("variant")
    published = pd.read_csv(REPO / "output-boba" / "analysis" / "design_rule_instrument_screen.csv").set_index("variant")
    cols = ["n_cells", "median_extra_cost", "rho_headroom", "extra_cost_low_headroom", "extra_cost_high_headroom"]
    pd.testing.assert_frame_equal(fit[cols], published.loc[fit.index, cols], check_exact=False, atol=1e-12)
    assert fit.loc["ceil0.9-fixed", "rho_headroom"] == pytest.approx(0.65, abs=0.005)


# ---------------------------------------------------------------------------
# heldout_remedies: the rank rules in the scope of their recoveries
# ---------------------------------------------------------------------------

HELDOUT = REPO / "output-boba" / "analysis" / "review" / "heldout_remedies.csv"


@pytest.mark.skipif(not HELDOUT.is_file(), reason="held-out remedies output not present")
def test_the_scope_matched_rank_rule_tests_survive_holm():
    h = pd.read_csv(HELDOUT)
    s = h[h.section == "4b_holm_rank_scope"].set_index(["problem", "candidate"])
    spike = s.loc[("output-boba-spike", "ordinal_lcb1")]
    cap = s.loc[("output-boba-ceiling", "ordinal_lcb1")]
    assert (spike["landscapes_gaining_test"], cap["landscapes_gaining_test"]) == (19, 20)
    assert spike["p_test"] == pytest.approx(5.7e-6, rel=0.05) and cap["p_test"] == pytest.approx(1.9e-6, rel=0.05)
    assert spike["test_value"] == pytest.approx(0.488, abs=0.001) and cap["test_value"] == pytest.approx(0.379, abs=0.001)
    assert spike["p_holm_core_replaced_test"] < 0.05 and cap["p_holm_core_replaced_test"] < 0.05
    # Section 4 is untouched: its core family still has the main-sweep rank-rule tests.
    core = h[(h.section == "4_holm") & (h.family == "core")]
    assert set(core.loc[core.group == "rank rule", "candidate"]) == {"ordinal_lcb1", "ordinal_pm"}


# ---------------------------------------------------------------------------
# run_review_checks
# ---------------------------------------------------------------------------


def test_every_check_is_a_script_and_listed_once():
    assert len(rrc.CHECKS) == len(set(rrc.CHECKS))
    for name in rrc.CHECKS:
        assert (SCRIPTS / "review_checks" / f"{name}.py").is_file(), name
    assert set(rrc.ARGS) <= set(rrc.CHECKS)
    for name in ("oracle_sigmaf_seeds", "oracle_achievable", "oracle_companion_estimators", "tie_break",
                 "sitting_vs_shiprule", "sitting_sequential", "floor_deployed", "friedman_multiplicity",
                 "budget_split_by_onset", "instrument_scale", "mo_onset_bound", "extra_reach"):
        assert name in rrc.CHECKS, name


def test_the_checks_run_after_the_ones_they_read():
    order = {n: i for i, n in enumerate(rrc.CHECKS)}
    assert order["oracle_sigmaf_seeds"] < order["oracle_achievable"] < order["oracle_families"]
    assert order["fresh_seed_replication"] < order["sitting_vs_shiprule"]
    assert rrc.ARGS["tie_break"] == ["rescore"]


def test_an_unknown_check_is_refused():
    with pytest.raises(SystemExit, match="unknown"):
        rrc.main(["--only", "not_a_check"])


# ---------------------------------------------------------------------------
# tie_break: the arm's own dataset scale, once per kind
# ---------------------------------------------------------------------------


def test_tie_break_asks_analyse_boba_adaptations_for_the_scale_once(monkeypatch):
    import analyse_boba_adaptations as aba
    import tie_break as tb

    calls = []
    monkeypatch.setattr(aba, "dataset_scale", lambda kind: calls.append(kind) or {"a": 2.0})
    monkeypatch.setattr(tb, "_SCALES", {})
    assert tb._dataset_scale("fitted_achievable") == {"a": 2.0}
    assert tb._dataset_scale("fitted_achievable") == {"a": 2.0}
    assert calls == ["fitted_achievable"]
