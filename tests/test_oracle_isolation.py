"""oracle_isolation._fraction: the fraction-of-the-floor-gap metric on either response.

Since 2026-09-22 the oracle-isolation comparison is made on the trajectory
response (the companion arm's) and on the deployed design (the paper's primary
estimand), so _fraction takes the response column as an argument. The floor is
the mean clean best of the model-free floors, and each learner's response is
divided by (optimum - floor). Since 2026-09-28 a per-seed optimum forms the gap
within each seed's oracle (every run seed refits the oracle), and the same
holds for analyse_fitted_companion.py; the tests below pin both, on a fixture
where the pooled and the per-seed gaps differ. Since 2026-09-30 the comparison
is also made on the achievable improvement (scripts/review_checks/
oracle_achievable.py, analyse_fitted_companion.share_fraction), pinned on the
same fixture.
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


# ---------------------------------------------------------------------------
# One oracle per run seed (2026-09-28). The simulator refits the oracle for
# every seed, so the floor gap is formed within each seed's oracle and a
# landscape's fraction is mean_s(excess_s) / mean_s(gap_s).
# ---------------------------------------------------------------------------


def _two_oracles() -> pd.DataFrame:
    """Seed 7's oracle peaks at 10, its floors reach 4 and 6, its learners 8 and
    lose 2. Seed 8's oracle peaks at 4, its floors reach 1, its learners 3 and
    lose 0.6. Pooled over both: max clean best 8, floor mean 3, gap 5. Per seed:
    y_opt gaps 5 and 3 (mean 4), best-clean gaps 3 and 2 (mean 2.5). One noisy
    run of seed 8 visits 4.5, above that oracle's logged y_opt, so the visited
    optimum is 10 and 4.5 (gaps 5 and 3.5, mean 4.25)."""
    rows = []
    for seed, y_opt, floors, best, excess in ((7, 10.0, (4.0, 6.0), 8.0, 2.0), (8, 4.0, (1.0, 1.0), 3.0, 0.6)):
        for acq, clean in zip(iso.FLOORS, floors):
            rows.append((seed, acq, clean, y_opt - clean, clean, 99.0))
        for i, acq in enumerate(iso.ACQS):
            noisy = 4.5 if (seed == 8 and i == 0) else best - 1.0
            rows.append((seed, acq, best, y_opt - best, noisy, excess))
    return pd.DataFrame(rows, columns=["seed", "acquisition", iso.BASELINE_BEST, iso.BASELINE_REGRET,
                                       iso.NOISY_BEST, iso.RESPONSE])


def test_seed_optima_read_the_logged_y_opt_and_the_best_clean_value_per_seed():
    frame = _two_oracles()
    pd.testing.assert_series_equal(iso.seed_optima(frame, "oracle"),
                                   pd.Series([10.0, 4.0], index=pd.Index([7, 8], name="seed")), check_names=False)
    pd.testing.assert_series_equal(iso.seed_optima(frame, "best_clean"),
                                   pd.Series([8.0, 3.0], index=pd.Index([7, 8], name="seed")), check_names=False)
    pd.testing.assert_series_equal(iso.seed_optima(frame, "visited"),
                                   pd.Series([10.0, 4.5], index=pd.Index([7, 8], name="seed")), check_names=False)
    assert list(iso.OPTIMA) == ["oracle", "best_clean", "visited"]  # the first is primary
    with pytest.raises(ValueError):
        iso.seed_optima(frame, "pooled")


def test_y_opt_must_be_constant_within_a_seed():
    frame = _two_oracles()
    frame.loc[0, iso.BASELINE_REGRET] += 0.5
    with pytest.raises(ValueError, match="y_opt differs"):
        iso.seed_optima(frame, "oracle")


def test_the_gap_is_formed_within_each_seed_and_differs_from_the_pooled_gap():
    frame = _two_oracles()
    pooled = iso.floor_gap(frame, float(frame[iso.BASELINE_BEST].max()))   # the estimator used until 2026-09-28
    assert pooled == pytest.approx(8.0 - 3.0)
    assert iso.floor_gap(frame, iso.seed_optima(frame, "oracle")) == pytest.approx(((10 - 5) + (4 - 1)) / 2)
    assert iso.floor_gap(frame, iso.seed_optima(frame, "best_clean")) == pytest.approx(((8 - 5) + (3 - 1)) / 2)
    assert iso.floor_gap(frame, iso.seed_optima(frame, "visited")) == pytest.approx(((10 - 5) + (4.5 - 1)) / 2)


def test_the_landscape_fraction_is_the_ratio_of_seed_means():
    frame = _two_oracles()
    out = iso._fraction(frame, iso.seed_optima(frame, "oracle"))
    mean_excess = (2.0 + 0.6) / 2
    assert out["frac"].mean() == pytest.approx(mean_excess / 4.0)                   # 0.325
    assert out["frac"].mean() != pytest.approx((2.0 / 5 + 0.6 / 3) / 2)             # not the mean of seed ratios
    assert iso._fraction(frame, iso.seed_optima(frame, "best_clean"))["frac"].mean() == pytest.approx(mean_excess / 2.5)
    assert iso._fraction(frame, float(frame[iso.BASELINE_BEST].max()))["frac"].mean() == pytest.approx(mean_excess / 5.0)


def test_a_seed_without_floor_runs_or_without_an_optimum_is_an_error():
    frame = _two_oracles()
    with pytest.raises(ValueError, match="no floor runs"):
        iso.floor_gap(frame[~((frame.seed == 8) & frame.acquisition.isin(iso.FLOORS))], iso.seed_optima(frame))
    with pytest.raises(ValueError, match="no optimum"):
        iso.floor_gap(frame, pd.Series({7: 10.0}))


def test_an_unbalanced_cell_is_refused():
    frame = _two_oracles()
    iso._check_balanced(frame, "balanced")
    with pytest.raises(ValueError, match="runs per seed differ"):
        iso._check_balanced(frame.iloc[1:], "unbalanced")


def _per_landscape() -> pd.DataFrame:
    rows = []
    for s in iso.SIGMA_GRID:
        for i, name in enumerate(["a", "b", "c", "d", "e", "f"]):
            exact = 1.0 + i
            gap = 0.01 if name == "a" else 0.5          # landscape a: random search nearly reaches the optimum
            fitted = exact * (0.1 if name == "a" else 0.8 + 0.1 * i)
            rows.append({"landscape": name, "sigma_multiple": s, "frac_fitted": fitted, "frac_exact": exact,
                         "frac_fitted_best_clean": exact, "gap_exact_over_opt_z": gap})
    return pd.DataFrame(rows)


def test_summarise_reports_the_ratio_of_means_the_median_and_the_ratio_without_small_gaps():
    per = _per_landscape()
    out = iso.summarise(per).set_index("sigma_multiple")
    b = per[per.sigma_multiple == 1.0]
    row = out.loc[1.0]
    assert row.ratio_fitted_over_exact == pytest.approx(b.frac_fitted.mean() / b.frac_exact.mean())
    assert row.median_ratio == pytest.approx(np.median(b.frac_fitted / b.frac_exact))
    assert row.n_lower == int((b.frac_fitted < b.frac_exact).sum())
    keep = b.gap_exact_over_opt_z >= iso.SMALL_GAP
    assert row.small_gap_landscapes == "a" and row.n_wo_small_gap == 5
    assert row.ratio_wo_small_gap == pytest.approx(b.frac_fitted[keep].mean() / b.frac_exact[keep].mean())
    assert row.ratio_lo <= row.ratio_fitted_over_exact <= row.ratio_hi
    same = iso.summarise(per, "frac_fitted_best_clean").set_index("sigma_multiple")
    assert same.loc[1.0].ratio_fitted_over_exact == pytest.approx(1.0) and same.loc[1.0].n_lower == 0


def test_summarise_can_re_estimate_the_exact_normaliser_too():
    # both arms on their best clean value: the exact fraction is replaced as well,
    # the output keeps the name frac_exact, and the default is unchanged
    per = _per_landscape()
    per["frac_exact_best_clean"] = per.frac_exact * 2.0
    out = iso.summarise(per, "frac_fitted", "frac_exact_best_clean").set_index("sigma_multiple").loc[1.0]
    b = per[per.sigma_multiple == 1.0]
    assert out.frac_exact == pytest.approx(b.frac_exact_best_clean.mean())
    assert out.ratio_fitted_over_exact == pytest.approx(b.frac_fitted.mean() / b.frac_exact_best_clean.mean())
    assert out.median_ratio == pytest.approx(np.median(b.frac_fitted / b.frac_exact_best_clean))
    pd.testing.assert_frame_equal(iso.summarise(per), iso.summarise(per, "frac_fitted", "frac_exact"))


def test_the_bootstrap_resamples_are_shared_so_family_contrasts_are_paired():
    a, b = iso.bootstrap_indices([20, 20, 20]), iso.bootstrap_indices([20, 20, 20])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert a[0].shape == (iso.BOOT, 20) and not np.array_equal(a[0], a[1])
    # the committed sequence: one generator seeded DATA_SEED, one draw per magnitude
    rng = np.random.default_rng(iso.DATA_SEED)
    assert np.array_equal(a[1], [rng.integers(0, 20, (iso.BOOT, 20)) for _ in range(2)][1])


# ---------------------------------------------------------------------------
# The companion arm (analyse_fitted_companion.py) gets the same fix.
# ---------------------------------------------------------------------------


def test_the_companion_forms_its_gap_within_each_seed():
    import analyse_fitted_companion as fc
    frame = _two_oracles().assign(dataset="ds")
    frame["acquisition"] = frame["acquisition"].replace({"logei": "pi"})     # the companion's learners are any non-floor
    frame[fc.RESPONSE] = frame[iso.RESPONSE]
    opt = fc.fitted_optimum(frame)
    assert opt.index.names == ["dataset", "seed"] and opt.to_dict() == {("ds", 7): 10.0, ("ds", 8): 4.0}
    assert fc.fitted_optimum(frame, "best_clean").to_dict() == {("ds", 7): 8.0, ("ds", 8): 3.0}
    assert fc.floor_gap(frame, opt).loc["ds"] == pytest.approx(4.0)
    out = fc.floor_fraction(frame, opt)
    assert out["frac"].mean() == pytest.approx(1.3 / 4.0)
    # one function per dataset (the synthetic arm): optimum less the floor pooled over seeds
    pooled = fc.floor_fraction(frame, pd.Series({"ds": 8.0}))
    assert pooled["gain"].iloc[0] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        fc.fitted_optimum(frame, "pooled")


# ---------------------------------------------------------------------------
# The achievable improvement, the paper's headline unit (2026-09-30): the
# exact arm divides by opt_z (optimum less the landscape mean), the fitted arm
# by A_s = optimum_s less the mean of seed s's oracle over its search box,
# formed within each seed; a landscape's fraction is mean_s(excess_s) / mean_s(A_s).
# ---------------------------------------------------------------------------


def _oracle_achievable():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "review_checks"))
    import oracle_achievable
    return oracle_achievable


def test_the_achievable_improvement_is_formed_within_each_seed():
    oa = _oracle_achievable()
    frame = _two_oracles()
    mean_f = pd.Series({7: 2.0, 8: 0.0})
    idx = pd.Index([7, 8], name="seed")
    pd.testing.assert_series_equal(oa.seed_achievable(frame, mean_f), pd.Series([8.0, 4.0], index=idx), check_names=False)
    pd.testing.assert_series_equal(oa.seed_achievable(frame, mean_f, "best_clean"), pd.Series([6.0, 3.0], index=idx),
                                   check_names=False)
    # pooling the two oracles (max y_opt less the mean of the means) would give 10 - 1 = 9, not the seed mean 6
    assert oa.seed_achievable(frame, mean_f).mean() == pytest.approx(6.0)
    # one noisy run of seed 8 visits 4.5, above that oracle's y_opt of 4
    pd.testing.assert_series_equal(oa.seed_achievable(frame, mean_f, "visited"), pd.Series([8.0, 4.5], index=idx),
                                   check_names=False)
    # the first is primary; new estimators are appended so older files keep their names
    assert list(oa.ESTIMATORS) == ["oracle", "best_clean", "empirical", "visited", "sigma_f", "sigma_f_seed7"]
    with pytest.raises(ValueError, match="no oracle mean"):
        oa.seed_achievable(frame, pd.Series({7: 2.0}))
    with pytest.raises(ValueError, match="below the oracle mean"):
        oa.seed_achievable(frame, pd.Series({7: 2.0, 8: 5.0}))


def test_the_sigma_f_normaliser_is_opt_z_in_each_seeds_oracle_units():
    # the landscape's own achievable improvement in the SD units of each seed's
    # oracle: no family's optimum or mean estimate enters
    oa = _oracle_achievable()
    frame = _two_oracles()
    A = oa.sigma_f_achievable(frame, pd.Series({7: 0.5, 8: 2.0, 9: 7.0}), opt_z=3.0)
    pd.testing.assert_series_equal(A, pd.Series([1.5, 6.0], index=pd.Index([7, 8], name="seed")), check_names=False)
    assert A.mean() == pytest.approx(3.0 * (0.5 + 2.0) / 2)
    with pytest.raises(ValueError, match="no sigma_f"):
        oa.sigma_f_achievable(frame, pd.Series({7: 0.5}), opt_z=3.0)
    with pytest.raises(ValueError, match="positive"):
        oa.sigma_f_achievable(frame, pd.Series({7: 0.5, 8: 1.0}), opt_z=0.0)


def test_the_exact_best_clean_normaliser_uses_the_same_six_acquisitions():
    oa = _oracle_achievable()
    rows = []
    for seed, best in ((7, (2.5, 2.9, 1.0)), (8, (2.0, 2.2, 2.7))):
        for acq, b in zip(("logei", "random", "turbo"), best):   # turbo is not one of the fitted arm's six
            rows.append({"seed": seed, "acquisition": acq, iso.BASELINE_BEST: b, iso.BASELINE_REGRET: 3.0 - b})
    exact = pd.DataFrame(rows)
    exact = pd.concat([exact] + [exact[exact.acquisition == "logei"].assign(acquisition=a)
                                 for a in iso.ACQS[1:] + iso.FLOORS[1:]], ignore_index=True)
    assert oa.exact_best_clean(exact, opt_z=3.0) == pytest.approx((2.9 + 2.2) / 2)   # turbo's 2.7 is ignored
    with pytest.raises(ValueError, match="not opt_z"):
        oa.exact_best_clean(exact, opt_z=3.5)
    with pytest.raises(ValueError, match="no clean runs"):
        oa.exact_best_clean(exact[exact.acquisition != "sobol"], opt_z=3.0)


def test_the_achievable_frame_divides_the_same_excess_by_the_other_normaliser():
    oa = _oracle_achievable()
    floor_gap_per = pd.DataFrame({
        "landscape": ["a", "b"], "sigma_multiple": [1.0, 1.0], "n_fitted": [20, 20], "n_exact": [20, 20],
        "oracle_model": ["m", "m"], "corr_oracle_truth": [0.9, 0.8],
        "excess_fitted": [1.3, 0.5], "excess_exact": [0.2, 2.0],
        "gap_fitted": [4.0, 1.0], "gap_fitted_best_clean": [2.5, 0.8], "gap_exact": [0.01, 1.0],
        "gap_exact_over_opt_z": [0.01, 0.5]})
    A = pd.DataFrame({"achievable_fitted": [6.0, 2.0], "achievable_fitted_best_clean": [4.5, 1.8],
                      "achievable_fitted_empirical": [6.1, 2.0], "achievable_fitted_visited": [6.0, 2.2],
                      "achievable_fitted_sigma_f": [4.0, 2.5], "achievable_fitted_sigma_f_seed7": [4.2, 2.4],
                      "achievable_exact_best_clean": [0.9, 1.6]},
                     index=["a", "b"])
    opt_z = pd.Series({"a": 1.0, "b": 2.0, "unused": 9.0})
    out = oa.per_landscape(floor_gap_per, A, opt_z).set_index("landscape")
    assert out.loc["a", "frac_fitted"] == pytest.approx(1.3 / 6.0)
    assert out.loc["a", "frac_fitted_best_clean"] == pytest.approx(1.3 / 4.5)
    assert out.loc["b", "frac_exact"] == pytest.approx(2.0 / 2.0)
    assert out.loc["a", "frac_fitted_sigma_f"] == pytest.approx(1.3 / 4.0)
    assert out.loc["b", "frac_fitted_sigma_f_seed7"] == pytest.approx(0.5 / 2.4)
    assert out.loc["b", "frac_exact_best_clean"] == pytest.approx(2.0 / 1.6)
    # a family's own achievable improvement against the landscape's in its sigma_f units
    assert out.loc["a", "achievable_fitted_over_sigma_f_opt_z"] == pytest.approx(6.0 / 4.0)
    # the floor share: the part of the achievable improvement a same-budget random run leaves
    assert out.loc["a", "floor_share_fitted"] == pytest.approx(4.0 / 6.0)
    assert out.loc["a", "floor_share_exact"] == pytest.approx(0.01 / 1.0)
    # per landscape the floor-gap ratio is the achievable ratio times exact share / fitted share
    fg = (1.3 / 4.0) / (0.2 / 0.01)
    ach = out.loc["a", "frac_fitted"] / out.loc["a", "frac_exact"]
    assert fg == pytest.approx(ach * out.loc["a", "floor_share_exact"] / out.loc["a", "floor_share_fitted"])
    with pytest.raises(ValueError):
        oa.per_landscape(floor_gap_per, A.drop(index="b"), opt_z)


def test_the_companion_achievable_improvement_uses_the_floor_runs_designs(tmp_path):
    import analyse_fitted_companion as fc
    frame = _two_oracles().assign(dataset="ds")
    frame[fc.RESPONSE] = frame[iso.RESPONSE]
    for seed, values in ((7, ([1.0, 3.0], [2.0, 2.0])), (8, ([0.0, 0.0], [0.5, -0.5]))):
        for acq, v in zip(fc.MODEL_FREE, values):
            (tmp_path / "ds").mkdir(exist_ok=True)
            pd.DataFrame({"objective_true": v}).to_csv(
                tmp_path / "ds" / f"bo_sensor_error_ds_composite_{acq}_seed{seed}_baseline_m.csv", index=False)
    means = fc.box_means(tmp_path, frame)
    assert means.to_dict() == {("ds", 7): 2.0, ("ds", 8): 0.0}
    A = fc.fitted_achievable(frame, means)
    assert A.loc["ds"] == pytest.approx(((10 - 2) + (4 - 0)) / 2)
    assert fc.fitted_achievable(frame, means, "best_clean").loc["ds"] == pytest.approx(((8 - 2) + (3 - 0)) / 2)
    out = fc.share_fraction(frame, A)
    assert out["frac"].mean() == pytest.approx(1.3 / 6.0)       # the ratio of seed means
    # the synthetic arm: excess over opt_z per landscape
    assert fc.share_fraction(frame, pd.Series({"ds": 2.0}))["frac"].mean() == pytest.approx(1.3 / 2.0)
    with pytest.raises(ValueError, match="no normaliser"):
        fc.share_fraction(frame, pd.Series({"other": 1.0}))
    with pytest.raises(ValueError, match="expected one clean"):
        fc.box_means(tmp_path / "missing", frame)


def test_acquisition_agreement_and_its_chance_level():
    import oracle_isolation_acq_ranking as ar
    rows = []
    for land in ("p", "q"):
        for m in ar.MULTIPLES:
            for i, acq in enumerate(ar.ACQS):
                a = float(i)                                       # ei best everywhere in arm a
                b = float(i) if land == "p" else float(-i)         # arm b agrees on p, reverses on q
                rows.append({"landscape": land, "mult": m, "acquisition": acq, "rank_fitted": a + 1, "rank_b": b})
    frame = pd.DataFrame(rows)
    frame["rank_b"] = frame.groupby(["landscape", "mult"])["rank_b"].rank()
    out = ar.compare(frame, "rank_fitted", "rank_b", with_means=False).set_index("magnitude")
    assert out.loc["pooled", "n_cells"] == 6 and out.loc["pooled", "best_acquisition_agrees"] == 3
    assert out.loc["pooled", "expected_by_chance"] == pytest.approx(1.5)
    from scipy.stats import binom
    assert out.loc["pooled", "p_binomial_at_least"] == pytest.approx(round(float(binom.sf(2, 6, 0.25)), 4))
    assert out.loc["1sigma", "best_acquisition_agrees"] == 1
    # one landscape agrees and one reverses, so the mean ranks tie and the order carries no information
    assert out.loc["pooled", "order_a"].startswith("ei")


def test_an_oracle_fit_leaves_no_catboost_logs_in_the_working_directory(tmp_path, monkeypatch):
    # CatBoost writes catboost_info/ into the working directory, which at the
    # repository root is a tracked folder; oracle_stats fits inside this guard.
    catboost = pytest.importorskip("catboost")
    monkeypatch.chdir(tmp_path)
    X = np.random.default_rng(0).uniform(size=(40, 2))
    with iso.fit_outside_repo() as scratch:
        assert scratch.resolve() != tmp_path.resolve()
        assert scratch.resolve() == type(scratch)(".").resolve()
        model = catboost.CatBoostRegressor(iterations=5, verbose=False, thread_count=1, random_seed=0)
        model.fit(X, X.sum(axis=1))
        assert (scratch / "catboost_info").is_dir()   # the guard is what keeps it out
    assert type(tmp_path).cwd().resolve() == tmp_path.resolve()
    assert not (tmp_path / "catboost_info").exists()
    assert model.predict(X).shape == (40,)            # the fitted model outlives the folder
