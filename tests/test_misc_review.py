"""The 2026-09-28 register's miscellaneous checks.

Three scripts back sentences of the paper:

* changepoint_compare.py scores the freeze rule's own CUSUM (kappa, h) beside
  the re-tuned detectors, re-tunes every detector to its false-alarm rate, and
  checks the fixed point against rule A's logged alarms;
* review_checks/instrument_scale.py measures each archival instrument's step and
  span on its composite's own grid;
* review_checks/mo_onset_bound.py applies onset_bound.py's running-maximum bound
  to the multi-objective arm, whose logs onset_bound.discover now groups through
  replaceable file-name patterns.

A last data test pins what App. B.7 says about the budget-100 arm's extra trials:
analyse_extra_runs counts them from the trial at which the clean twin itself
first came within the tolerance of its k-trial regret, so the noisy run's reach
trial is that origin plus the extra, not k plus the extra.

The synthetic tests pin the arithmetic; the data tests (skipped when the outputs
are absent) pin the published files against the identities they must satisfy.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import changepoint_compare as cp  # noqa: E402
import onset_bound as ob  # noqa: E402

N0 = 15
COLS = 35


def _load(name: str):
    path = SCRIPTS / "review_checks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"review_checks_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# changepoint_compare: the freeze rule's operating point
# ---------------------------------------------------------------------------


def _frame(n_steady: int = 300, n_changed: int = 300, seed: int = 0):
    """Residual streams: steady ones (half clean, half error from trial 1) and
    ones whose variance jumps 4x at trial 21 (column 5)."""
    rng = np.random.default_rng(seed)
    z_steady = rng.normal(0.0, 1.0, size=(n_steady, COLS))
    z_changed = rng.normal(0.0, 1.0, size=(n_changed, COLS))
    z_changed[:, 5:] *= 4.0
    Z = np.vstack([z_steady, z_changed])
    clean = np.r_[np.arange(n_steady) < n_steady // 2, np.zeros(n_changed, bool)]
    frame = pd.DataFrame({
        "file": [f"run{i}.csv" for i in range(len(Z))],
        "z": list(Z),
        "error_model": np.where(clean, "none", "gaussian"),
        "jitter_iteration": np.r_[np.zeros(n_steady), np.full(n_changed, 20)],
        "jitter_std": np.r_[np.where(clean[:n_steady], 0.0, 1.0), np.full(n_changed, 1.0)],
    })
    return cp.label_runs(frame), Z


def test_steady_split_separates_clean_and_onset0_false_alarms():
    block = pd.DataFrame({"clean": [True, True, False, False, False], "changed": [False, False, False, True, True],
                          "onset": [0, 0, 0, 20, 20]})
    got = cp.steady_split(np.array([17, 0, 0, 19, 25]), block)
    assert got["false_alarm_clean"] == pytest.approx(0.5)
    assert got["false_alarm_onset0"] == pytest.approx(0.0)
    assert got["pre_onset_alarm_late"] == pytest.approx(0.5)     # the alarm at 19 precedes onset 20
    assert (got["n_clean"], got["n_onset0"]) == (2, 1)


def test_the_fixed_operating_point_is_not_retuned_and_sets_the_matched_budget():
    tune, Zt = _frame(seed=1)
    score, Zs = _frame(seed=2)
    kappa = 4.0
    h = float(np.quantile(cp.cusum_path(Zt[~tune["changed"].to_numpy()], kappa).max(axis=1), 0.9))
    freeze = {"kappa": kappa, "h": h, "w": 3}
    table = cp.freeze_point_table(tune, score, Zt, Zs, freeze, [0.25, 1.0], [0.02], [0.05, 0.10], 5, N0)

    fixed = table[table["comparison"] == "freeze_rule_fixed"].set_index("split")
    assert fixed.loc["tune", "threshold"] == h and fixed.loc["score", "threshold"] == h
    alarms = cp.first_alarm(cp.cusum_path(Zs, kappa), h, N0)
    assert fixed.loc["score", "detection_rate"] == pytest.approx(cp.rates(alarms, score)["detection_rate"])
    assert json.loads(fixed.loc["score", "params"]) == {"kappa": kappa, "h": h, "w": 3}

    matched = table[table["comparison"] == "freeze_rule_budget"]
    budget = fixed.loc["tune", "false_alarm_rate"]
    assert np.allclose(matched["target_false_alarm"], budget)
    assert set(matched["detector"]) == {"cusum", "cusum_freeze_kappa", "glr", "bocpd"}
    pooled = matched[matched["jitter_std"].astype(str) == "all"]
    assert (pooled["tune_false_alarm"] <= budget + 1e-12).all()
    # The CUSUM at the freeze kappa, re-thresholded to the freeze rule's own
    # false-alarm rate, fires on the same steady tuning runs; its threshold moves
    # only between the same two steady peaks, so detection barely moves.
    same = pooled[pooled["detector"] == "cusum_freeze_kappa"].iloc[0]
    assert same["tune_false_alarm"] == pytest.approx(budget)
    assert abs(same["tune_detection"] - fixed.loc["tune", "detection_rate"]) < 0.05
    # The grid family includes the freeze kappa, so it can only detect as much or more on the tuning seeds.
    grid = pooled[pooled["detector"] == "cusum"].iloc[0]
    assert grid["tune_detection"] >= same["tune_detection"] - 1e-12


def test_the_budget_block_repeats_the_main_comparison_for_the_grid_cusum_and_the_glr():
    tune, Zt = _frame(seed=3)
    score, Zs = _frame(seed=4)
    freeze = {"kappa": 16.0, "h": 50.0, "w": 3}
    table = cp.freeze_point_table(tune, score, Zt, Zs, freeze, [0.25, 1.0], [0.02], [0.10], 5, N0)
    row = table[(table["comparison"] == "budget") & (table["detector"] == "glr")].iloc[0]
    p_tune = cp.glr_path(Zt)
    h = cp.threshold_for(p_tune, tune, 0.10, N0)
    assert row["threshold"] == pytest.approx(h)
    expect = cp.rates(cp.first_alarm(cp.glr_path(Zs), h, N0), score)
    assert row["detection_rate"] == pytest.approx(expect["detection_rate"])


def test_the_per_run_check_counts_mismatched_alarms(tmp_path):
    frame, Z = _frame(n_steady=20, n_changed=20, seed=5)
    freeze = {"kappa": 1.0, "h": 5.0, "w": 3}
    alarms = cp.first_alarm(cp.cusum_path(Z, 1.0), 5.0, N0)
    per = pd.DataFrame({"file": frame["file"], "rule": "A", "stop_t": np.where(alarms > 0, alarms, np.nan),
                        "h": 5.0, "kappa": 1.0, "w": 3})
    path = tmp_path / "stopping_per_run.csv"
    per.to_csv(path, index=False)
    got = cp.check_against_per_run(path, frame, freeze, N0)
    assert got["alarm_mismatches"] == 0 and got["streams_missing_from_per_run"] == 0
    per.loc[0, "stop_t"] = 49.0 if alarms[0] != 49 else 48.0
    per.to_csv(path, index=False)
    assert cp.check_against_per_run(path, frame, freeze, N0)["alarm_mismatches"] == 1
    per.assign(kappa=4.0).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        cp.check_against_per_run(path, frame, freeze, N0)


def test_freeze_params_none_skips_and_inf_is_refused(tmp_path):
    assert cp.load_freeze_params("none") is None
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"h": float("inf"), "kappa": 0.0, "w": 0}), encoding="utf-8")
    with pytest.raises(SystemExit):
        cp.load_freeze_params(str(path))


CP_DIR = REPO / "output-boba" / "analysis" / "changepoint"


@pytest.mark.skipif(not (CP_DIR / "changepoint_freeze_point.csv").is_file(), reason="no changepoint outputs")
def test_published_freeze_point_matches_the_freeze_rule_and_its_log():
    meta = json.loads((CP_DIR / "changepoint_metadata.json").read_text(encoding="utf-8"))
    tuned = json.loads((REPO / "output-boba/analysis/stopping/stopping_tuned_A.json").read_text(encoding="utf-8"))
    assert meta["freeze_params"]["kappa"] == tuned["kappa"] and meta["freeze_params"]["h"] == tuned["h"]
    assert meta["per_run_check"]["alarm_mismatches"] == 0
    assert meta["per_run_check"]["streams_checked"] == meta["n_streams"] == 13200
    t = pd.read_csv(CP_DIR / "changepoint_freeze_point.csv")
    fixed = t[t["comparison"] == "freeze_rule_fixed"].set_index("split")
    # stopping_tuned_A.json's tuning row is the same detector on the same runs.
    assert fixed.loc["tune", "detection_rate"] == pytest.approx(tuned["tune_row"]["detection_rate_late"])
    assert fixed.loc["tune", "false_alarm_clean"] == pytest.approx(tuned["tune_row"]["false_alarm_clean"])
    assert fixed.loc["tune", "false_alarm_onset0"] == pytest.approx(tuned["tune_row"]["false_alarm_onset0"])
    # The budget block reproduces changepoint_detectors.csv for the grid CUSUM and the GLR.
    main = pd.read_csv(CP_DIR / "changepoint_detectors.csv")
    budget = t[t["comparison"] == "budget"]
    for det in ("cusum", "glr"):
        a = main[main["detector"] == det].set_index("target_false_alarm")["detection_rate"]
        b = budget[budget["detector"] == det].set_index("target_false_alarm")["detection_rate"]
        assert np.allclose(a.sort_index().to_numpy(), b.sort_index().to_numpy())


# ---------------------------------------------------------------------------
# instrument_scale
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inst():
    return _load("instrument_scale")


def test_grid_step_finds_the_common_step(inst):
    assert inst.grid_step(np.array([1.0, 1.5, 2.25, 3.0])) == pytest.approx(0.25)
    assert inst.grid_step(np.array([-1.0, -11 / 12, 1.0])) == pytest.approx(1 / 12)
    assert inst.grid_step(np.array([2.0, 2.0])) == 0.0
    assert inst.fraction_gcd([Fraction(1, 10), Fraction(1, 20), Fraction(1, 5)]) == Fraction(1, 20)


def test_a_value_off_any_rational_grid_is_refused(inst):
    # 1/2 and 5000/9999 are its nearest fractions with denominator <= 10,000, both > 1e-6 away.
    with pytest.raises(ValueError):
        inst.as_fraction(0.50003)


def _arm_meta(path: Path, clip: str = "-8,8", step: float = 0.55) -> Path:
    stats = {"flat": {"opt_z": 1.0, "min": -3.0, "mean": 0.0, "std": 1.0},
             "tall": {"opt_z": 20.0, "min": -1.0, "mean": 0.0, "std": 1.0},
             "edge": {"opt_z": 8.0, "min": -20.0, "mean": 2.0, "std": 2.0}}     # min is -11 sd: below -8
    path.write_text(json.dumps({"args": {"response_clip": clip, "response_round": step},
                                "landscape_stats": stats}), encoding="utf-8")
    return path


def test_clip_reach_counts_where_the_clip_meets_the_landscape(inst, tmp_path):
    got = inst.clip_reach(_arm_meta(tmp_path / "meta.json"))
    assert got["n_landscapes"] == 3
    assert got["optimum_above_clip"] == ["tall"]          # opt_z = 8 exactly is not above the clip
    assert got["minimum_below_clip"] == ["edge"]
    with pytest.raises(SystemExit):                        # a different arm is refused
        inst.clip_reach(_arm_meta(tmp_path / "other.json", clip="-6,6"))
    with pytest.raises(SystemExit):
        inst.clip_reach(_arm_meta(tmp_path / "other2.json", step=1.1))


def test_end_counts_reports_the_logged_ends_and_their_ratings(inst):
    assert inst.end_counts(np.array([1.0, 1.0, 3.0, 17.0, 1.0])) == (1.0, 17.0, 3, 1)
    assert inst.end_counts(np.array([-0.75, 1.0, 1.0])) == (-0.75, 1.0, 1, 2)


@pytest.mark.skipif(not (REPO / "output-boba-instrument" / "run_metadata.json").is_file(),
                    reason="no instrument arm metadata")
def test_the_instrument_arm_clip_lies_below_the_optimum_on_three_landscapes(inst):
    got = inst.clip_reach()
    assert got["n_landscapes"] == 20
    assert got["optimum_above_clip"] == ["michalewicz", "ackley", "shekel"]


INST = REPO / "output-boba" / "analysis" / "review" / "instrument_scale.csv"


@pytest.mark.skipif(not INST.is_file(), reason="instrument_scale.py has not been run")
def test_published_instrument_scale_is_consistent():
    t = pd.read_csv(INST).set_index("dataset")
    anchor = pd.read_csv(REPO / "output" / "noise_anchor.csv").set_index("dataset")
    for name, r in t.iterrows():
        assert r["sigma_f"] == pytest.approx(anchor.loc[name, "sigma_f"])
        assert r["composite_step_raw"] == pytest.approx(r["composite_step_implied_by_items"])
        assert r["composite_step_sigma"] == pytest.approx(r["composite_step_raw"] / r["sigma_f"])
        assert r["span_items_raw"] >= r["span_logged_raw"] - 1e-12
        assert r["ceiling_above_mean_sigma"] + r["floor_below_mean_sigma"] == pytest.approx(r["span_items_sigma"])
        assert r["composite_step_raw"] <= r["coarsest_item_step_raw"] + 1e-12
    assert t.loc["ehmi", "composite_step_raw"] == pytest.approx(0.05)
    assert t.loc["provoice", "composite_step_raw"] == pytest.approx(1 / 6)
    assert t.loc["opticarvis", "span_items_raw"] == pytest.approx(2.0)      # column ranges [-1, 1]


# ---------------------------------------------------------------------------
# mo_onset_bound and onset_bound.discover
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mob():
    return _load("mo_onset_bound")


def test_mo_patterns_parse_the_arm_s_file_names(mob):
    m = mob.MO_BASE_RE.match("bo_sensor_error_carsideimpact_multi_objective_qlognehvi_seed10_baseline_exact.csv")
    assert (m["ds"], m["acq"], m["seed"]) == ("carsideimpact", "qlognehvi", "10")
    m = mob.MO_NOISY_RE.match(
        "bo_sensor_error_zdt1_multi_objective_qehvi_seed7_jittered_exact_gaussian_jit20_std0.25.csv")
    assert (m["ds"], m["acq"], m["em"], m["onset"], m["std"]) == ("zdt1", "qehvi", "gaussian", "20", "0.25")
    assert ob.BASE_RE.match("bo_sensor_error_zdt1_multi_objective_qehvi_seed7_baseline_exact.csv") is None


def _run(xs: np.ndarray, hv: np.ndarray, y_opt: float) -> pd.DataFrame:
    frame = pd.DataFrame({f"x{j}": xs[:, j] for j in range(xs.shape[1])})
    frame.insert(0, "iteration", np.arange(1, len(hv) + 1))
    frame["objective_true"] = hv
    frame["objective_observed"] = hv
    frame["simple_regret_true"] = y_opt - hv
    frame["inference_simple_regret_true"] = y_opt - hv
    frame["y_opt"] = y_opt
    return frame


def test_the_bound_holds_for_a_running_hypervolume():
    """A running hypervolume never falls, so after the shared prefix the noisy
    run's regret is at most the clean run's at s, and excess <= B."""
    rng = np.random.default_rng(0)
    T, k = 30, 10
    xs_c = rng.uniform(size=(T, 2))
    xs_n = xs_c.copy()
    xs_n[k + 1:] = rng.uniform(size=(T - k - 1, 2))        # diverges after the design at trial k + 1
    hv_c = np.maximum.accumulate(rng.uniform(0, 1, T).cumsum())
    hv_n = hv_c.copy()
    hv_n[k + 1:] = np.maximum.accumulate(hv_c[k] + rng.uniform(0, 0.5, T - k - 1).cumsum() * 0.1)
    base, noisy = _run(xs_c, hv_c, 40.0), _run(xs_n, hv_n, 40.0)
    noisy.loc[noisy["iteration"] > k, "objective_observed"] += 1.0     # observations differ from the onset
    got = ob.summarise_pair(base, noisy, k, 5)
    assert got["prefix_ok"]
    assert got["max_viol"] <= 1e-12
    assert got["exc_win"] <= got["B_win"] + 1e-12


def test_discover_takes_replacement_patterns(tmp_path, mob):
    root = tmp_path / "arm"
    ds = root / "zdt1"
    ds.mkdir(parents=True)
    (root / "run_metadata.json").write_text(json.dumps({"args": {"initial_samples": 5}}), encoding="utf-8")
    for name in ("bo_sensor_error_zdt1_multi_objective_qehvi_seed7_baseline_exact.csv",
                 "bo_sensor_error_zdt1_multi_objective_qehvi_seed7_jittered_exact_gaussian_jit0_std1.0.csv",
                 "bo_sensor_error_zdt1_multi_objective_random_seed7_baseline_exact.csv"):
        (ds / name).write_text("iteration\n1\n", encoding="utf-8")
    tasks, _ = ob.discover("mo", root, None, None)
    assert tasks == []                                     # the scalar patterns see nothing
    tasks, _ = ob.discover("mo", root, None, None, base_re=mob.MO_BASE_RE, noisy_re=mob.MO_NOISY_RE)
    assert len(tasks) == 1
    arm, n_init, runs = tasks[0]
    assert (arm, n_init) == ("mo", 5)
    key, _, noisy = runs[0]
    assert key == {"dataset": "zdt1", "acquisition": "qehvi", "seed": 7}
    assert noisy[0][0] == {"error_model": "gaussian", "onset": 0, "jitter_std": 1.0, "variant": ""}


def test_discover_defaults_still_group_the_scalar_arm(tmp_path):
    root = tmp_path / "scalar"
    ds = root / "branin"
    ds.mkdir(parents=True)
    (root / "run_metadata.json").write_text(json.dumps({"args": {"initial_samples": 5}}), encoding="utf-8")
    for name in ("bo_sensor_error_branin_value_logei_seed7_baseline_exact.csv",
                 "bo_sensor_error_branin_value_logei_seed7_jittered_exact_bias_jit20_std5.0.csv"):
        (ds / name).write_text("iteration\n1\n", encoding="utf-8")
    tasks, _ = ob.discover("main", root, None, None)
    assert len(tasks) == 1
    key, _, noisy = tasks[0][2][0]
    assert key == {"dataset": "branin", "acquisition": "logei", "seed": 7}
    assert noisy[0][0] == {"error_model": "bias", "onset": 20, "jitter_std": 5.0, "variant": ""}


MOB = REPO / "output-boba" / "analysis" / "review" / "mo_onset_bound.csv"


@pytest.mark.skipif(not MOB.is_file(), reason="mo_onset_bound.py has not been run")
def test_published_mo_bound_factorises_and_reproduces_the_cross_arm_table():
    t = pd.read_csv(MOB)
    r = t[t["onset"] == "early/late"].pivot_table(index=["arm", "jitter_std"], columns="quantity", values="estimate")
    # raw = bound x normalised, exactly, in every row.
    assert np.allclose(r["raw_ratio"], r["bound_ratio"] * r["normalised_ratio"])
    cells = t[(t["onset"] != "early/late")]
    assert (cells[cells["quantity"] == "n_bound_violations"]["estimate"] == 0).all()
    assert (cells[cells["quantity"] == "share_prefix_identical"]["estimate"] == 1).all()
    pub_path = REPO / "output-boba-mo" / "analysis" / "mo_vs_scalar.csv"
    if pub_path.is_file():
        pub = pd.read_csv(pub_path).assign(onset=lambda d: d["jitter_iteration"].astype(int).astype(str))
        mine = cells[cells["quantity"] == "excess"].merge(pub, on=["arm", "jitter_std", "onset"])
        assert len(mine) == len(pub)
        assert np.allclose(mine["estimate"], mine["mean"])


def _ratio_table(raw_mo, bound_mo, raw_sc, bound_sc, std=1.0):
    rows = []
    for arm, raw, bound in (("multi-objective", raw_mo, bound_mo), ("scalar", raw_sc, bound_sc)):
        for q, v in (("raw_ratio", raw), ("bound_ratio", bound), ("normalised_ratio", raw / bound)):
            rows.append({"arm": arm, "error_model": "gaussian", "jitter_std": std, "onset": "early/late",
                         "quantity": q, "estimate": v, "lo": np.nan, "hi": np.nan})
    return pd.DataFrame(rows)


def test_cross_arm_log_share_splits_the_raw_gap_exactly(mob):
    # The bound alone halves the ratio and the raw ratio falls by a quarter: the bound over-explains.
    g = mob.cross_arm_log_share(_ratio_table(6.0, 5.0, 8.0, 10.0)).iloc[0]
    assert g["log_raw_gap"] == pytest.approx(g["log_bound_gap"] + g["log_normalised_gap"])
    assert g["bound_share"] == pytest.approx(np.log(0.5) / np.log(0.75))
    # Same normalised ratio in both arms: the bound accounts for all of the gap.
    g = mob.cross_arm_log_share(_ratio_table(4.0, 8.0, 6.0, 12.0)).iloc[0]
    assert g["bound_share"] == pytest.approx(1.0)
    assert g["log_normalised_gap"] == pytest.approx(0.0)


@pytest.mark.skipif(not MOB.is_file(), reason="mo_onset_bound.py has not been run")
def test_published_cross_arm_split(mob):
    g = mob.cross_arm_log_share(pd.read_csv(MOB))
    assert np.allclose(g["log_raw_gap"], g["log_bound_gap"] + g["log_normalised_gap"])
    # The bound accounts for about three quarters of the gap at 1 sigma and all of it at 5 sigma.
    assert g.loc[1.0, "bound_share"] == pytest.approx(0.743, abs=0.005)
    assert g.loc[5.0, "bound_share"] > 1.0


# ---------------------------------------------------------------------------
# App. B.7: where the extra trials are counted from
# ---------------------------------------------------------------------------

BUDGET100 = REPO / "output-boba-budget100"


# The analysis file is committable, the run logs and run_metadata.json are not:
# guard on what the body reads, so a fresh checkout skips instead of failing.
@pytest.mark.skipif(not (BUDGET100 / "analysis" / "extra_runs_per_run.csv").is_file()
                    or not (BUDGET100 / "run_metadata.json").is_file()
                    or not any(BUDGET100.glob("*/bo_sensor_error_*_baseline_*.csv")),
                    reason="the budget-100 run logs (git-ignored) are not on this machine")
def test_extra_trials_count_from_the_clean_twins_own_reach_trial():
    import analyse_extra_runs as aer

    baselines, jittered = aer.index_runs(BUDGET100)
    opt_z = aer.load_opt_z(BUDGET100)
    k, tol, rows = 25, 0.01, []
    for run in jittered:
        if (run["acquisition"] in aer.MODEL_FREE or run["error_model"] != "gaussian" or run["variant"]
                or run["jitter_iteration"] != 0 or run["jitter_std"] != 1.0):
            continue
        clean, noisy = aer.regret_curve(baselines[run["stem"]]), aer.regret_curve(run["path"])
        T = min(len(clean), len(noisy))
        clean, noisy = clean[:T], noisy[:T]
        slack = tol * opt_z[run["dataset"]]
        extra, censored = aer.extra_trials(clean, noisy, k, slack)
        origin = int(np.flatnonzero(clean <= clean[k - 1] + 1e-12 + slack)[0]) + 1
        rows.append({"dataset": run["dataset"], "acquisition": run["acquisition"], "seed": run["seed"],
                     "extra": extra, "censored": censored, "origin": origin, "reach": origin + extra})
    mine = pd.DataFrame(rows)
    pub = pd.read_csv(BUDGET100 / "analysis" / "extra_runs_per_run.csv")
    pub = pub[(pub["error_model"] == "gaussian") & (pub["tolerance"] == tol) & (pub["k"] == k)
              & (pub["jitter_std"] == 1.0) & (pub["jitter_iteration"] == 0) & (pub["variant"].fillna("") == "")]
    m = mine.merge(pub, on=["dataset", "acquisition", "seed"], suffixes=("", "_pub"), validate="one_to_one")
    assert len(m) == len(pub) == 600
    assert (m["extra"] == m["extra_pub"]).all() and (m["censored"] == m["censored_pub"]).all()
    assert m["censored"].mean() < 0.5                      # so every median below is exact
    assert (m["origin"] <= k).all() and (m["origin"] < k).mean() > 0.9
    # App. B.7: a median of 41 extra trials (40.5), reached at a median trial of 56, 2.2 times 25.
    assert m["extra"].median() == pytest.approx(40.5)
    assert m["reach"].median() == pytest.approx(56.0)
    assert np.median(m["reach"] / k) == pytest.approx(2.24)
    # The per-run 'multiplier' column is (k + extra) / k, which is not the reach trial over k.
    assert np.allclose(m["multiplier"], (k + m["extra"]) / k)
    assert m["multiplier"].median() == pytest.approx(2.62)
