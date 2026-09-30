"""The first-index versus uniform tie-break of the standard ship rule.

scripts/review_checks/tie_break.py re-scores the logs: the standard rule ships
the FIRST design with the maximum rating (np.argmax), and the sensitivity
replaces it with the exact expectation under a uniformly random choice among the
tied designs. These fixtures pin the arithmetic, the tie tolerance, the handling
of lost ratings and slipped designs, and the join onto an evaluation frame that
feeds analyse_boba_adaptations.paired_frame unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "review_checks"))

import tie_break as tb  # noqa: E402
import analyse_boba_adaptations as aba  # noqa: E402


def test_tied_maximum_expected_regret_is_the_mean_over_the_tied_designs():
    ratings = np.array([1.0, 3.0, 3.0, 2.0, 3.0])
    truth = np.array([0.5, 1.0, 2.0, 5.0, 4.0])
    r = tb.tie_regrets(ratings, truth, y_opt=6.0)
    assert r["n_tied"] == 3 and r["n_tied_distinct_true"] == 3
    assert r["first_idx"] == int(np.argmax(ratings)) == 1
    assert r["regret_first"] == pytest.approx(5.0)
    assert r["regret_uniform"] == pytest.approx(6.0 - (1.0 + 2.0 + 4.0) / 3.0)
    assert r["regret_best_tie"] == pytest.approx(2.0)
    assert r["regret_worst_tie"] == pytest.approx(5.0)
    assert r["regret_last_tie"] == pytest.approx(2.0)


def test_uniform_expectation_matches_a_random_tie_break_on_average():
    ratings = np.array([0.2, 0.9, 0.9, 0.1, 0.9, 0.9])
    truth = np.array([0.0, 0.3, 0.7, 2.0, 0.1, 0.9])
    exact = tb.tie_regrets(ratings, truth, y_opt=1.0)["regret_uniform"]
    rng = np.random.default_rng(0)
    tied = np.flatnonzero(ratings == ratings.max())
    draws = 1.0 - truth[rng.choice(tied, size=200_000)]
    assert draws.mean() == pytest.approx(exact, abs=3e-3)


def test_a_unique_maximum_or_tied_equal_designs_change_nothing():
    unique = tb.tie_regrets(np.array([1.0, 2.0, 1.5]), np.array([3.0, 1.0, 9.0]), y_opt=10.0)
    assert unique["n_tied"] == 1
    assert unique["regret_uniform"] == unique["regret_first"] == pytest.approx(9.0)
    # A clean run that revisits the same design, or a piecewise-constant oracle:
    # the tie is real but every tied design has the same true value.
    same = tb.tie_regrets(np.array([2.0, 1.0, 2.0]), np.array([2.0, 1.0, 2.0]), y_opt=3.0)
    assert same["n_tied"] == 2 and same["n_tied_distinct_true"] == 1
    assert same["regret_uniform"] == same["regret_first"] == pytest.approx(1.0)


def test_tie_tolerance_is_absolute_1e_12():
    base = 0.37
    tied = tb.tied_indices(np.array([base, base - 5e-13, base - 1e-9]))
    assert tied.tolist() == [0, 1]


def test_float_noise_between_tied_designs_does_not_make_the_tie_matter():
    # A fitted oracle returned the same value to 8e-13 at four designs.
    r = tb.tie_regrets(np.array([2.0, 2.0, 1.0]), np.array([2.0, 2.0 + 8e-13, 1.0]), y_opt=3.0)
    assert r["n_tied"] == 2 and r["n_tied_distinct_true"] == 1
    runs = pd.DataFrame([r])
    assert not bool(tb.tie_matters(runs).iloc[0])
    real = pd.DataFrame([tb.tie_regrets(np.array([2.0, 2.0]), np.array([1.0, 1.5]), y_opt=3.0)])
    assert bool(tb.tie_matters(real).iloc[0])


def test_lost_ratings_are_never_deployed_and_an_empty_log_ships_trial_one():
    r = tb.tie_regrets(np.array([np.nan, 2.0, 2.0]), np.array([9.0, 1.0, 3.0]), y_opt=10.0)
    assert r["first_idx"] == 1 and r["n_tied"] == 2
    assert r["regret_uniform"] == pytest.approx(10.0 - 2.0)
    empty = tb.tie_regrets(np.array([np.nan, np.nan]), np.array([4.0, 5.0]), y_opt=6.0)
    assert empty["first_idx"] == 0 and empty["regret_first"] == pytest.approx(2.0)


def test_tie_share_counts_ties_consequential_ties_and_where_the_first_is_worse():
    recs = []
    cases = (  # ratings, truth: a consequential tie with the first worse, one with the first better,
        ([1.0, 1.0, 0.5], [0.2, 0.8, 0.1]),  # a tie between equal designs, and no tie at all
        ([1.0, 1.0, 0.5], [0.9, 0.3, 0.1]),
        ([1.0, 1.0, 0.5], [0.4, 0.4, 0.1]),
        ([1.0, 0.7, 0.5], [0.4, 0.9, 0.1]),
    )
    for i, (ratings, truth) in enumerate(cases):
        rec = tb.tie_regrets(np.array(ratings), np.array(truth), y_opt=1.0)
        rec.update(arm="toy", variant="", baseline=False, file=f"run{i}.csv", is_multi=False, match_first=True,
                   error_model="gaussian", jitter_std=0.25, jitter_iteration=0)
        recs.append(rec)
    share = tb.tie_share(pd.DataFrame(recs)).iloc[0]
    assert share["n_runs"] == 4
    assert share["share_tied"] == pytest.approx(3 / 4)
    assert share["share_tie_matters"] == pytest.approx(2 / 4)
    assert share["share_first_worse_than_tied_mean"] == pytest.approx(1 / 2)
    assert share["mean_regret_first"] == pytest.approx(np.mean([0.8, 0.1, 0.6, 0.6]))
    assert share["mean_regret_uniform"] == pytest.approx(np.mean([0.5, 0.4, 0.6, 0.6]))


def _write_run(path: Path, observed, true, y_opt, error_model="gaussian", deployed=None, logged=None,
               dataset="toy", acquisition="logei", seed=7, std=0.25, onset=0):
    n = len(observed)
    first = int(np.nanargmax(observed))
    target = deployed if deployed is not None else true
    frame = pd.DataFrame({
        "iteration": np.arange(1, n + 1), "x0": np.linspace(0, 1, n),
        "objective_true": true, "objective_observed": observed,
        "acquisition": acquisition, "seed": seed, "error_model": error_model,
        "jitter_std": 0.0 if error_model == "none" else std, "jitter_iteration": onset,
        "oracle_model": "exact", "objective": "value", "y_opt": y_opt,
        "inference_simple_regret_true": y_opt - np.asarray(target)[first] if logged is None else logged,
        "dataset": dataset,
    })
    if deployed is not None:
        frame["objective_true_deployed"] = deployed
    frame.to_csv(path, index=False)


def test_run_record_reproduces_the_logged_rule_and_scores_the_deployed_design(tmp_path):
    noisy = tmp_path / "bo_sensor_error_toy_value_logei_seed7_jittered_exact_gaussian_jit0_std0.25_ceil0.9-fixed.csv"
    # The first tied design was a slip: the person saw x', the log says x, and
    # the deployed design is scored where the log says.
    _write_run(noisy, observed=[0.1, 0.8, 0.8, 0.8], true=[0.0, 0.5, 0.6, 0.9], y_opt=1.0,
               deployed=[0.0, 0.2, 0.6, 0.9])
    rec = tb.run_record(noisy)
    assert rec["variant"] == "ceil0.9-fixed" and not rec["baseline"]
    assert rec["match_first"] is True
    assert rec["n_tied"] == 3
    assert rec["regret_first"] == pytest.approx(0.8)
    assert rec["regret_uniform"] == pytest.approx(1.0 - (0.2 + 0.6 + 0.9) / 3)

    wrong = tmp_path / "bo_sensor_error_toy_value_logei_seed8_jittered_exact_gaussian_jit0_std0.25.csv"
    _write_run(wrong, observed=[0.1, 0.8, 0.8], true=[0.0, 0.5, 0.6], y_opt=1.0, logged=0.4, seed=8)
    assert tb.run_record(wrong)["match_first"] is False


def test_scan_writes_the_run_table_and_the_tie_shares(tmp_path, monkeypatch):
    import argparse

    arm = tmp_path / "output-boba-toy"
    (arm / "toy").mkdir(parents=True)
    (arm / "analysis").mkdir()
    _write_run(arm / "toy" / "bo_sensor_error_toy_value_logei_seed7_baseline_exact.csv",
               observed=[0.2, 0.9, 0.4], true=[0.2, 0.9, 0.4], y_opt=1.0, error_model="none")
    _write_run(arm / "toy" / "bo_sensor_error_toy_value_logei_seed7_jittered_exact_gaussian_jit0_std0.25.csv",
               observed=[0.7, 0.7, 0.3], true=[0.1, 0.9, 0.3], y_opt=1.0)
    # a copy under analysis/ is not a run and must be skipped
    _write_run(arm / "analysis" / "bo_sensor_error_toy_value_logei_seed8_baseline_exact.csv",
               observed=[0.1], true=[0.1], y_opt=1.0, error_model="none", seed=8)
    out = tmp_path / "out"
    monkeypatch.setattr(tb, "OUT_DIR", out)
    monkeypatch.setattr(tb, "arm_directories", lambda: [arm])
    tb.cmd_scan(argparse.Namespace(arms=None, workers=1))
    runs = pd.read_parquet(out / "tie_runs.parquet")
    assert len(runs) == 2 and runs["match_first"].all()
    noisy = runs[~runs["baseline"]].iloc[0]
    assert noisy["n_tied"] == 2 and noisy["regret_uniform"] == pytest.approx(0.5)
    share = pd.read_csv(out / "tie_share_by_arm.csv")
    row = share[share["baseline"] == False].iloc[0]  # noqa: E712
    assert row["share_tie_matters"] == 1.0 and row["share_first_worse_than_tied_mean"] == 1.0
    assert (out / "tie_share_by_condition.csv").is_file()


def test_attach_uniform_feeds_paired_frame_with_the_estimand_unchanged(tmp_path):
    runs = []
    # clean twin: exact ratings, unique maximum
    clean = tmp_path / "bo_sensor_error_toy_value_logei_seed7_baseline_exact.csv"
    _write_run(clean, observed=[0.2, 0.9, 0.4], true=[0.2, 0.9, 0.4], y_opt=1.0, error_model="none")
    noisy = tmp_path / "bo_sensor_error_toy_value_logei_seed7_jittered_exact_gaussian_jit0_std0.25.csv"
    _write_run(noisy, observed=[0.7, 0.7, 0.3], true=[0.1, 0.9, 0.3], y_opt=1.0)
    for p in (clean, noisy):
        rec = tb.run_record(p)
        rec["arm"] = "toy-arm"
        runs.append(rec)
    runs = pd.DataFrame(runs)
    evaluation = pd.DataFrame([{
        "dataset": "toy", "acquisition": "logei", "seed": 7, "error_model": "gaussian",
        "jitter_std": 0.25, "jitter_iteration": 0, "variant": "", "oracle_model": "exact",
        "final_inference_simple_regret_true_jitter": 0.9,     # first-index: design 0
        "final_inference_simple_regret_true_baseline": 0.1,
    }])
    attached = tb.attach_uniform(evaluation, runs, "toy-arm")
    assert attached["final_inference_uniform_tie_jitter"].iloc[0] == pytest.approx(1.0 - 0.5)
    assert attached["final_inference_uniform_tie_baseline"].iloc[0] == pytest.approx(0.1)
    # The same frame as reference and treatment: cost is noisy minus clean, gain zero.
    paired = aba.paired_frame(attached, attached, "final_inference_uniform_tie", {"toy": 2.0}, pool=False)
    s = aba.summarise(paired, np.random.default_rng(0))
    assert s["cost"] == pytest.approx((0.5 - 0.1) / 2.0)
    assert s["gain"] == pytest.approx(0.0) and s["price"] == pytest.approx(0.0)

    # A first-index value that does not match the evaluation's is refused.
    bad = evaluation.assign(final_inference_simple_regret_true_jitter=0.5)
    with pytest.raises(ValueError, match="does not reproduce"):
        tb.attach_uniform(bad, runs, "toy-arm")


def test_tie_origin_tells_exact_pre_onset_ties_from_ties_that_matter(tmp_path):
    arm = tmp_path / "output-boba"
    (arm / "toy").mkdir(parents=True)
    # A revisited design rated exactly twice before a late onset (trial 3): a tie
    # between equal designs, which never changes what is deployed.
    late = "bo_sensor_error_toy_value_logei_seed7_jittered_exact_gaussian_jit2_std0.25.csv"
    _write_run(arm / "toy" / late, observed=[0.9, 0.9, 0.5, 0.4], true=[0.9, 0.9, 0.3, 0.6], y_opt=1.0,
               onset=2)
    # A clean twin with the same revisit.
    clean = "bo_sensor_error_toy_value_logei_seed7_baseline_exact.csv"
    _write_run(arm / "toy" / clean, observed=[0.9, 0.9, 0.3, 0.6], true=[0.9, 0.9, 0.3, 0.6], y_opt=1.0,
               error_model="none")
    # A noisy tie after the onset between designs of different true value.
    capped = "bo_sensor_error_toy_value_logei_seed8_jittered_exact_gaussian_jit0_std0.25.csv"
    _write_run(arm / "toy" / capped, observed=[0.2, 0.8, 0.8], true=[0.2, 0.5, 0.7], y_opt=1.0, seed=8)
    # No tie: not listed.
    single = "bo_sensor_error_toy_value_logei_seed9_jittered_exact_gaussian_jit0_std0.25.csv"
    _write_run(arm / "toy" / single, observed=[0.2, 0.8, 0.7], true=[0.2, 0.5, 0.7], y_opt=1.0, seed=9)
    recs = []
    for name in (late, clean, capped, single):
        rec = tb.run_record(arm / "toy" / name)
        rec["arm"] = "output-boba"
        recs.append(rec)
    runs = pd.DataFrame(recs).assign(not_a_run_log=np.nan)
    origin = tb.tie_origin(runs, repo=tmp_path).set_index("file")
    assert set(origin.index) == {late, clean, capped}
    assert origin.loc[late, "all_tied_exact"] and origin.loc[late, "all_tied_before_onset"] is True
    assert (origin.loc[late, "first_tied_idx"], origin.loc[late, "last_tied_idx"]) == (0, 1)
    assert not origin.loc[late, "tie_matters"]
    assert origin.loc[clean, "all_tied_exact"] and pd.isna(origin.loc[clean, "all_tied_before_onset"])
    assert not origin.loc[capped, "all_tied_exact"]
    assert origin.loc[capped, "all_tied_before_onset"] is False and origin.loc[capped, "tie_matters"]
