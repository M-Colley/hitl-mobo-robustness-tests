"""Error-process extensions: effort schedules, relevance pursuit, missing ratings,
relay raters and a rating-scale ceiling.

None of them adds a trial. The tests pin that no existing name moved (seeds and
choices are built from list POSITIONS), that each process draws what it should
from the stream it should, that each remedy matches a hand computation, that
the processes leave the clean run alone where they claim to, and that every
flag is off at its default and reaches the filename when on.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402
import boba_multiobjective as mob  # noqa: E402
import bo_sensor_error_simulation as sim  # noqa: E402
import bo_synthetic_error_simulation as syn  # noqa: E402

# The simulator before this build (2026-09-14). The index test compares against
# that module itself where the copy exists, and against its name lists, vendored
# in tests/fixtures/bo_sim_backup_2026-09-14.json, elsewhere (register item D1).
import _reference_fixtures as ref  # noqa: E402

BRANIN = bb.BENCHMARKS["branin"]
COLS = BRANIN.param_columns
BOUNDS = sim.Bounds(low=BRANIN.bounds_low, high=BRANIN.bounds_high)
NEW_COLUMNS = {"noise_effort", "missing", "imputed_value", "rater_id", "ceiling_value"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> "sim.SimulationConfig":
    fields = dict(
        iterations=8, jitter_iteration=0, jitter_std=0.0, single_error=False,
        initial_samples=5, candidate_pool=32, objective="value", objective_columns=["value"],
        param_columns=COLS, seed=7, error_model="none", error_bias=0.25,
        error_spike_prob=0.1, error_spike_std=0.5, dropout_strategy="hold_last",
        normalize_objective=False, objective_weights=None, acq_num_restarts=2,
        acq_raw_samples=16, acq_maxiter=30, acq_mc_samples=16, ref_point=None,
    )
    fields.update(overrides)
    return sim.SimulationConfig(**fields)


def _oracle() -> "bb.SyntheticOracle":
    return bb.SyntheticOracle.from_stats("branin", bb.load_stats())


def _run(acq: str = "logei", apply_error: bool = True, seed: int = 7, **overrides) -> pd.DataFrame:
    """A short branin run. apply_error=True defaults to gaussian noise of 0.25 SD from trial 1."""
    stats = bb.load_stats()
    if apply_error:
        overrides.setdefault("error_model", "gaussian")
        overrides.setdefault("jitter_std", 0.25)
    config = _config(seed=seed, **overrides)
    y_opt = (stats["branin"]["y_opt"] - stats["branin"]["mean"]) / stats["branin"]["std"]
    torch.manual_seed(seed)
    return sim.run_simulation(
        oracle=_oracle(), bounds=BOUNDS, config=config, acq=sim.AcquisitionConfig(name=acq),
        rng=np.random.default_rng(seed),
        jitter_rng=np.random.default_rng(99) if apply_error else None,
        run_id="t", apply_error=apply_error, oracle_model="exact", y_opt=y_opt,
    )


def _missing_run(handling: str = "drop", rate: float = 0.5, process: str = "missing_mcar", **overrides):
    """The input-error arm's shape: no response error, the rate drives the loss."""
    return _run("logei", error_model="none", input_error_model=process, input_error_scale=rate,
                missing_handling=handling, **overrides)


def _without_timing(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns="fit_time_sec")


@pytest.fixture(scope="module")
def standard_clean() -> pd.DataFrame:
    return _run("logei", apply_error=False)


class _RecordingGP(sim.SingleTaskGP):
    """SingleTaskGP that records the targets it is trained on."""

    targets: list[list[float]] = []

    def __init__(self, train_X, train_Y, *args, **kwargs):
        type(self).targets.append(train_Y.reshape(-1).tolist())
        super().__init__(train_X, train_Y, *args, **kwargs)


def _expected_training_targets(frame: pd.DataFrame, initial: int = 5) -> list[list[float]]:
    """What each model-based trial must train on, rebuilt from the log alone."""
    observed = frame["objective_observed"].to_numpy(dtype=float)
    imputed = (frame["imputed_value"].to_numpy(dtype=float) if "imputed_value" in frame.columns
               else np.full(len(frame), np.nan))
    expected = []
    for t in range(initial, len(frame)):  # 0-based index of each model-based trial
        rows, targets = sim._missing_training_rows(list(observed[:t]), list(imputed[:t]))
        if len(rows) >= 2:
            expected.append(targets)
    return expected


# ---------------------------------------------------------------------------
# names and indices
# ---------------------------------------------------------------------------


def test_new_names_are_appended():
    assert sim.INPUT_ERROR_CHOICES == ["none", "slip", "misclick", "missing_mcar", "missing_low"]
    assert sim.LIKELIHOOD_CHOICES == ["gaussian", "student_t", "relevance_pursuit", "student_t_rbf"]
    assert sim.ERROR_MODEL_CHOICES == ["gaussian", "bias", "dropout", "spike", "drift", "ar1"]
    assert sim.ACQUISITION_CHOICES[-3:] == ["ts", "aei", "shiplcb"]


def test_pre_existing_names_match_the_backup_module():
    backup = ref.backup_simulator("bo_sim_backup_errext_2026_09_14")
    for attr in ("ACQUISITION_CHOICES", "ERROR_MODEL_CHOICES", "INPUT_ERROR_CHOICES", "LIKELIHOOD_CHOICES",
                 "INCUMBENT_CHOICES", "INPUT_NOISE_MODEL_CHOICES", "INFERENCE_RULE_CHOICES",
                 "OBSERVATION_NOISE_CHOICES", "INPUT_ERROR_RECORDED_CHOICES"):
        old, new = list(getattr(backup, attr)), list(getattr(sim, attr))
        assert new[: len(old)] == old, attr
        for name in old:
            assert new.index(name) == old.index(name), (attr, name)
    assert list(sim.DEFAULT_ACQUISITION_CHOICES) == list(backup.DEFAULT_ACQUISITION_CHOICES)
    assert list(sim.parse_error_models(None, "all")) == list(backup.parse_error_models(None, "all"))


# ---------------------------------------------------------------------------
# (1) the effort schedule
# ---------------------------------------------------------------------------


def test_presets_at_the_standard_budget():
    front10 = (2.0,) * 10 + (0.75,) * 40
    assert sim.noise_schedule_efforts("front10", 50) == front10
    assert sim.noise_schedule_efforts("front20", 50) == (1.5,) * 20 + (0.6667,) * 30
    assert sim.noise_schedule_efforts("U", 50) == (1.6667,) * 20 + (0.3333,) * 20 + (1.0,) * 10
    assert sim.noise_schedule_efforts("back10", 50) == (0.75,) * 40 + (2.0,) * 10
    assert sim.noise_schedule_efforts("1-10:2, 11-:0.75", 50) == front10
    assert sim.noise_schedule_efforts("none", 4) == (1.0,) * 4
    for preset in ("front10", "front20", "U", "back10"):
        assert abs(np.mean(sim.noise_schedule_efforts(preset, 50)) - 1.0) <= 1e-3


@pytest.mark.parametrize(
    "spec, iterations, match",
    [
        ("front10", 8, "outside"),        # the presets are written for T = 50
        ("back10", 10, "more than 10"),
        ("back10", 60, "mean effort"),    # 50 x 0.75 + 10 x 2 averages 0.958 at T = 60
        ("1-4:2", 8, "no effort"),
        ("1-5:1,5-8:1", 8, "two efforts"),
        ("1-4:2,5-:1", 8, "mean effort"),
        ("1-4:x,5-:1", 8, "not a number"),
        ("1-4:-1,5-:3", 8, "positive"),
        ("first:2", 8, "is not"),
    ],
)
def test_malformed_or_unbalanced_schedules_fail_clearly(spec, iterations, match):
    with pytest.raises(ValueError, match=match):
        sim.noise_schedule_efforts(spec, iterations)


def test_schedule_names_are_filename_safe():
    assert sim.noise_schedule_name("front10") == "front10"
    assert sim.noise_schedule_name("U") == "U"
    assert sim.noise_schedule_name("1-10:2, 11-:0.75") == "1-10@2+11-@0.75"


def test_schedule_rescales_the_same_gaussian_draw():
    config = _config(error_model="gaussian", jitter_std=0.5, noise_schedule="1-4:1.5,5-8:0.5")
    true = np.array([1.0])
    for t, effort in ((2, 1.5), (6, 0.5)):
        _, error = sim.apply_sensor_error(true, t, config, np.random.default_rng(3), true)
        draw = np.random.default_rng(3).normal(0.0, 0.5, size=1)
        np.testing.assert_allclose(error, draw / math.sqrt(effort), rtol=1e-12)
        assert sim.known_noise_variance(t, config, True) == pytest.approx(0.25 / effort, rel=1e-12)
    assert sim.known_noise_variance(6, dataclasses.replace(config, noise_schedule="none"), True) == 0.25


def test_flat_schedule_reproduces_the_standard_noisy_run_exactly():
    standard = _run("logei")
    flat = _run("logei", noise_schedule="1-:1")
    assert not standard["acq_opt_failed"].any()
    assert (flat["noise_effort"] == 1.0).all()
    assert "noise_effort" not in standard.columns
    pd.testing.assert_frame_equal(
        _without_timing(standard), _without_timing(flat).drop(columns="noise_effort"), check_exact=True
    )


def test_schedule_moves_error_between_trials_on_common_random_numbers():
    standard = _run("logei")
    scheduled = _run("logei", noise_schedule="1-4:1.5,5-8:0.5")
    np.testing.assert_array_equal(scheduled["noise_effort"], [1.5] * 4 + [0.5] * 4)
    # The initial design is shared, so its errors are the standard draws rescaled.
    efforts = np.array([1.5] * 4 + [0.5])
    np.testing.assert_allclose(
        scheduled["error_magnitude"].head(5), standard["error_magnitude"].head(5) / np.sqrt(efforts),
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(scheduled["known_noise_var"].head(5), 0.0625 / efforts, rtol=1e-12)


# ---------------------------------------------------------------------------
# (2) relevance pursuit
# ---------------------------------------------------------------------------


def test_relevance_pursuit_surrogate_flags_a_planted_outlier():
    from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
    from botorch.models.transforms import Normalize, Standardize

    bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]], dtype=torch.double)
    torch.manual_seed(0)
    X = bounds[0] + torch.rand(12, 2) * (bounds[1] - bounds[0])
    Y = torch.sin(X[:, :1] / 3.0) * 20 + X[:, 1:]
    Y[1] += 60.0
    gp, _ = sim.fit_relevance_pursuit_gp(X, Y, bounds)
    assert isinstance(gp, RobustRelevancePursuitSingleTaskGP)
    assert isinstance(gp.input_transform, Normalize) and isinstance(gp.outcome_transform, Standardize)
    rho = gp.likelihood.noise_covar.rho.detach()
    assert int(torch.argmax(rho)) == 1 and int((rho > 1e-6).sum()) == 1
    with torch.no_grad():
        mean = gp.posterior(X).mean.reshape(-1)
    # the posterior ignores the outlier instead of bending through it
    assert abs(float(mean[1] - (Y[1, 0] - 60.0))) < 5.0
    # aei reads the base (inlier) noise, in output units
    base = float(gp.likelihood.noise_covar.base_noise.noise.reshape(-1)[0])
    stdv = float(gp.outcome_transform.stdvs.reshape(-1)[0])
    assert sim.observation_noise_sd(gp) == pytest.approx(math.sqrt(base) * stdv, rel=1e-10)


def test_relevance_pursuit_is_the_surrogate_of_every_model_based_trial(monkeypatch):
    sizes: list[int] = []
    original = sim.fit_relevance_pursuit_gp

    def spy(train_X, train_Y, bounds_tensor):
        sizes.append(int(train_X.shape[0]))
        return original(train_X, train_Y, bounds_tensor)

    monkeypatch.setattr(sim, "fit_relevance_pursuit_gp", spy)
    frame = _run("aei", likelihood="relevance_pursuit")
    assert not frame["acq_opt_failed"].any()
    assert sizes == [5, 6, 7]
    # in the clean run too: this remedy changes the clean twin
    sizes.clear()
    clean = _run("logei", apply_error=False, likelihood="relevance_pursuit")
    assert not clean["acq_opt_failed"].any()
    assert sizes == [5, 6, 7]


# ---------------------------------------------------------------------------
# (3) missing ratings
# ---------------------------------------------------------------------------


def test_landscape_quantile_is_fixed_cached_and_leaves_torch_alone():
    oracle = _oracle()
    state = torch.get_rng_state()
    values = sim.landscape_values(oracle, BOUNDS)
    assert torch.equal(state, torch.get_rng_state())
    assert values.shape == (4096,) and np.all(np.diff(values) >= 0)
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=sim.LANDSCAPE_SAMPLE_SEED)
    X = BRANIN.bounds_low + engine.draw(4096).to(torch.double).numpy() * (BRANIN.bounds_high - BRANIN.bounds_low)
    np.testing.assert_allclose(values, np.sort(oracle.predict_many(X)[:, 0]), rtol=1e-12)
    assert sim.landscape_values(_oracle(), BOUNDS) is values  # another object, same landscape
    assert sim.landscape_quantile(oracle, BOUNDS, 0.4) == pytest.approx(np.quantile(values, 0.4))

    class Plane:
        def predict_many(self, X):
            return (X[:, :1] + X[:, 1:]).reshape(-1, 1)

    plane = Plane()
    unit = sim.Bounds(low=np.zeros(2), high=np.ones(2))
    assert sim.landscape_quantile(plane, unit, 0.5) == pytest.approx(1.0, abs=0.02)
    assert "_landscape_values" in plane.__dict__


def test_missing_draws_one_uniform_per_post_onset_trial():
    config = _config(input_error_model="missing_mcar", input_error_scale=0.5, jitter_iteration=2)
    rng = np.random.default_rng(5)
    true = np.array([0.0])
    verdicts = [sim.draw_missing_rating(true, t, config, rng, None) for t in range(1, 9)]
    uniforms = np.random.default_rng(5).random(6)
    assert verdicts == [False, False] + [bool(u < 0.5) for u in uniforms]
    # missing_low: the same uniforms, a loss only below the threshold
    low = dataclasses.replace(config, input_error_model="missing_low")
    rng = np.random.default_rng(5)
    below = [sim.draw_missing_rating(np.array([-1.0 if t % 2 else 1.0]), t, low, rng, 0.0) for t in range(1, 9)]
    assert below == [False, False] + [bool(u < 0.5) and t % 2 == 1 for t, u in zip(range(3, 9), uniforms)]
    # nothing is drawn before the onset, or for a process that loses nothing
    rng = np.random.default_rng(5)
    sim.draw_missing_rating(true, 2, config, rng, None)
    sim.draw_missing_rating(true, 5, _config(input_error_model="slip", input_error_scale=0.5), rng, None)
    assert rng.random() == np.random.default_rng(5).random()


def test_imputation_and_nan_aware_picks_by_hand():
    assert sim.impute_low_value([]) is None
    assert sim.impute_low_value([np.nan, np.nan]) is None
    assert sim.impute_low_value([3.0, np.nan, 1.0, 2.0]) == 1.0  # fewer than five: the minimum
    assert sim.impute_low_value([5.0, 1.0, np.nan, 4.0, 2.0, 3.0]) == pytest.approx(1.8)  # 20th percentile
    assert sim._nan_argmax([np.nan, np.nan]) == 0
    assert sim._nan_argmax([np.nan, 2.0, 3.0, np.nan]) == 2
    rows, targets = sim._missing_training_rows([1.0, np.nan, np.nan, 4.0], [np.nan, np.nan, 0.5, np.nan])
    assert rows == [0, 2, 3] and targets == [1.0, 0.5, 4.0]
    X = [np.array([0.1]), np.array([0.9]), np.array([0.1])]
    assert sim._best_mean_index(X, [np.nan, 1.0, np.nan]) == 1
    assert sim._best_mean_index(X, [np.nan, np.nan, np.nan]) == 0


def test_drop_trains_and_deploys_without_the_lost_ratings(monkeypatch):
    _RecordingGP.targets = []
    monkeypatch.setattr(sim, "SingleTaskGP", _RecordingGP)
    frame = _missing_run("drop", rate=0.5, iterations=10)
    missing = frame["missing"].to_numpy(dtype=bool)
    assert missing.any() and not missing.all()
    observed = frame["objective_observed"].to_numpy(dtype=float)
    true = frame["objective_true"].to_numpy(dtype=float)
    assert np.isnan(observed[missing]).all() and np.isnan(frame["error_magnitude"].to_numpy()[missing]).all()
    np.testing.assert_array_equal(observed[~missing], true[~missing])  # no response error in this arm
    assert np.isfinite(true).all() and not frame["acq_opt_failed"].any()
    assert "imputed_value" not in frame.columns
    assert frame["error_model"].iloc[0] == "missing_mcar"
    assert _RecordingGP.targets == _expected_training_targets(frame)
    for t in range(len(frame)):
        seen = observed[: t + 1]
        index = 0 if np.isnan(seen).all() else int(np.nanargmax(seen))
        assert frame["inference_value_true"].iloc[t] == true[index]


def test_impute_low_trains_on_the_imputed_value_and_never_deploys_it(monkeypatch):
    _RecordingGP.targets = []
    monkeypatch.setattr(sim, "SingleTaskGP", _RecordingGP)
    frame = _missing_run("impute_low", rate=0.5, iterations=10)
    observed = frame["objective_observed"].to_numpy(dtype=float)
    imputed = frame["imputed_value"].to_numpy(dtype=float)
    missing = frame["missing"].to_numpy(dtype=bool)
    assert missing.any() and np.isnan(imputed[~missing]).all() and np.isnan(observed[missing]).all()
    for t in np.flatnonzero(missing):
        expected = sim.impute_low_value(list(observed[:t]))
        assert (np.isnan(imputed[t]) if expected is None else imputed[t] == expected)
    assert np.isfinite(imputed[missing]).any()
    assert not frame["acq_opt_failed"].any()
    assert _RecordingGP.targets == _expected_training_targets(frame)
    true = frame["objective_true"].to_numpy(dtype=float)
    for t in range(len(frame)):
        seen = observed[: t + 1]
        index = 0 if np.isnan(seen).all() else int(np.nanargmax(seen))
        assert frame["inference_value_true"].iloc[t] == true[index]


def test_every_rating_lost_keeps_sampling_like_the_initial_design():
    frame = _missing_run("drop", rate=1.0)
    assert frame["missing"].all() and not frame["acq_opt_failed"].any()
    assert (frame["inference_value_true"] == frame["objective_true"].iloc[0]).all()
    rng = np.random.default_rng(7)
    expected = np.vstack([sim.sample_uniform(BOUNDS, rng, size=1) for _ in range(8)])
    np.testing.assert_array_equal(frame[COLS].to_numpy(), expected)


def test_missing_low_loses_only_designs_below_the_40th_percentile():
    frame = _missing_run("drop", rate=1.0, process="missing_low")
    threshold = sim.landscape_quantile(_oracle(), BOUNDS, 0.4)
    np.testing.assert_array_equal(
        frame["missing"].to_numpy(dtype=bool), frame["objective_true"].to_numpy(dtype=float) < threshold
    )
    assert frame["error_model"].iloc[0] == "missing_low"


# ---------------------------------------------------------------------------
# (4) relay raters
# ---------------------------------------------------------------------------


def test_rater_assignment_parsing_and_indices():
    assert sim.parse_rater_assign("none") == ("none", 0)
    assert sim.parse_rater_assign(" block:5 ") == ("block", 5)
    assert sim.parse_rater_assign("roundrobin:3") == ("roundrobin", 3)
    for bad in ("block", "block:0", "block:x", "relay:2", "roundrobin:"):
        with pytest.raises(ValueError, match="--rater-assign"):
            sim.parse_rater_assign(bad)
    assert [sim.rater_index(t, "block", 3) for t in range(1, 8)] == [0, 0, 0, 1, 1, 1, 2]
    assert [sim.rater_index(t, "roundrobin", 3) for t in range(1, 8)] == [0, 1, 2, 0, 1, 2, 0]
    assert sim.rater_suffix("block:5", 0.5) == "rater-block5-tau0.5"


def test_rater_offsets_come_from_a_stream_of_their_own():
    offsets = sim.draw_rater_offsets(7, "block:3", 0.5, 0.2, 10)
    assert offsets.shape == (4,)  # ceil(10 / 3) raters
    z = np.random.default_rng(np.random.SeedSequence([7, sim.RATER_SEED_TAG, 1, 3, 500_000])).standard_normal(4)
    np.testing.assert_allclose(offsets, z * 0.1, rtol=1e-12)
    # the same standard normals at every error magnitude
    np.testing.assert_allclose(sim.draw_rater_offsets(7, "block:3", 0.5, 1.0, 10), z * 0.5, rtol=1e-12)
    assert sim.draw_rater_offsets(7, "roundrobin:2", 0.5, 1.0, 10).shape == (2,)
    assert not np.allclose(sim.draw_rater_offsets(8, "block:3", 0.5, 1.0, 10), z * 0.5)


def test_rater_offsets_ride_on_the_standard_gaussian_stream():
    standard = _run("logei", jitter_iteration=2)
    relay = _run("logei", jitter_iteration=2, rater_assign="block:2", rater_offset_ratio=2.0)
    assert not relay["acq_opt_failed"].any()
    ids = relay["rater_id"].to_numpy()
    assert ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    offsets = sim.draw_rater_offsets(7, "block:2", 2.0, 0.25, 8)
    # The initial design is shared: after the onset the errors differ by exactly
    # the rater's offset, before it not at all.
    diff = relay["error_magnitude"].head(5).to_numpy() - standard["error_magnitude"].head(5).to_numpy()
    np.testing.assert_allclose(diff, np.where(np.arange(1, 6) > 2, offsets[ids[:5]], 0.0), atol=1e-12)


def test_backfit_matches_a_hand_computation_with_a_flat_surrogate():
    class Flat:
        def posterior(self, X):
            return types.SimpleNamespace(mean=torch.zeros(X.shape[0], 1, dtype=torch.double))

    fits: list[torch.Tensor] = []

    def fit_gp(X, Y):
        fits.append(Y.clone())
        return Flat(), None

    X = torch.zeros(5, 2, dtype=torch.double)
    Y = torch.tensor([[1.0], [3.0], [2.0], [-2.0], [0.0]], dtype=torch.double)
    ids = [0, 0, 1, 1, 2]
    _, _, estimates = sim.fit_backfit_gp(X, Y, ids, fit_gp)
    counts = np.array([2.0, 2.0, 1.0])
    raw = np.array([4.0, 0.0, 0.0]) / (counts + 1.0)  # sum of (y - mu) / (n_r + 1), mu = 0
    expected = raw - np.sum(counts * raw) / counts.sum()
    got = np.array([estimates[0], estimates[1], estimates[2]])
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert abs(float(np.sum(counts * got))) < 1e-12
    assert len(fits) == sim.BACKFIT_ROUNDS + 1
    torch.testing.assert_close(fits[0], Y)
    torch.testing.assert_close(fits[-1], Y - torch.tensor([[estimates[r]] for r in ids], dtype=torch.double))


def test_backfit_recovers_planted_offsets():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    generator = torch.Generator().manual_seed(0)
    X = torch.rand(24, 2, dtype=torch.double, generator=generator)
    ids = [i % 3 for i in range(24)]
    planted = np.array([1.0, -0.5, -0.5])  # already n-weighted mean zero
    Y = (torch.sin(3.0 * X[:, :1]) + X[:, 1:] + torch.as_tensor(planted[ids]).reshape(-1, 1)
         + 0.01 * torch.randn(24, 1, dtype=torch.double, generator=generator))
    _, _, estimates = sim.fit_backfit_gp(X, Y, ids, lambda X_, Y_: sim._fit_gaussian_gp(X_, Y_, None, bounds))
    got = np.array([estimates[0], estimates[1], estimates[2]])
    assert got[0] > 0 > got[1] and got[0] > 0 > got[2]
    assert np.corrcoef(got, planted)[0, 1] > 0.9


def test_backfit_fits_four_times_per_model_based_trial(monkeypatch):
    calls: list[int] = []
    original = sim._fit_gaussian_gp

    def spy(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(sim, "_fit_gaussian_gp", spy)
    frame = _run("ucb", rater_assign="block:3", rater_offset_ratio=1.0, rater_model="backfit")
    assert not frame["acq_opt_failed"].any()
    assert len(calls) == 3 * (sim.BACKFIT_ROUNDS + 1)


def test_backfit_deploys_on_the_corrected_ratings_in_the_clean_run(monkeypatch):
    original = sim.fit_backfit_gp

    def forced(train_X, train_Y, rater_ids, fit_gp):
        model, mll, _ = original(train_X, train_Y, rater_ids, fit_gp)
        # rater 1 declared harsh by 100: its ratings count 100 higher
        return model, mll, {r: (-100.0 if r == 1 else 0.0) for r in set(rater_ids)}

    monkeypatch.setattr(sim, "fit_backfit_gp", forced)
    frame = _run("logei", apply_error=False, rater_assign="roundrobin:2", rater_model="backfit")
    assert not frame["acq_opt_failed"].any()
    ids = frame["rater_id"].to_numpy()
    assert ids.tolist() == [0, 1] * 4  # logged in the clean run, which the backfit changes
    true = frame["objective_true"].to_numpy(dtype=float)  # clean: rating == true value
    for t in range(5):  # no fit yet, so no estimates: the plain best rating
        assert frame["inference_value_true"].iloc[t] == true[: t + 1].max()
    for t in range(5, 8):
        corrected = true[: t + 1] + np.where(ids[: t + 1] == 1, 100.0, 0.0)
        assert frame["inference_value_true"].iloc[t] == true[int(np.argmax(corrected))]


# ---------------------------------------------------------------------------
# (5) the rating-scale ceiling
# ---------------------------------------------------------------------------


def test_fixed_ceiling_caps_every_noisy_rating_from_the_first_trial():
    standard = _run("logei", jitter_iteration=3)
    capped = _run("logei", jitter_iteration=3, response_ceiling=0.6)
    cap = sim.landscape_quantile(_oracle(), BOUNDS, 0.6)
    assert (capped["ceiling_value"] == cap).all()
    assert (capped["objective_observed"] <= cap).all()
    # no randomness consumed: the shared initial design is the standard rating, capped
    np.testing.assert_array_equal(
        capped["objective_observed"].head(5), np.minimum(standard["objective_observed"].head(5), cap)
    )
    pre = capped.head(3)  # before the onset the rating is exact, and still capped
    np.testing.assert_array_equal(pre["objective_observed"], np.minimum(pre["objective_true"], cap))
    np.testing.assert_allclose(
        capped["error_magnitude"], capped["objective_observed"] - capped["objective_true"], atol=1e-12
    )


def test_anchored_ceiling_follows_the_best_rated_design():
    frame = _run("logei", jitter_std=0.5, response_ceiling=0.3, ceiling_mode="anchored")
    base = sim.landscape_quantile(_oracle(), BOUNDS, 0.3)
    observed = frame["objective_observed"].to_numpy(dtype=float)
    true = frame["objective_true"].to_numpy(dtype=float)
    caps = frame["ceiling_value"].to_numpy(dtype=float)
    for t in range(len(frame)):
        expected = base if t == 0 else max(base, true[int(np.argmax(observed[:t]))] + 0.5)
        assert caps[t] == expected
    assert (observed <= caps).all() and (caps > base).any()


# ---------------------------------------------------------------------------
# the clean twin, and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        dict(noise_schedule="1-4:1.5,5-8:0.5"),
        dict(input_error_model="missing_mcar", input_error_scale=0.5, missing_handling="impute_low"),
        dict(input_error_model="missing_low", input_error_scale=0.5),
        dict(rater_assign="block:2", rater_offset_ratio=2.0),
        dict(response_ceiling=0.6, ceiling_mode="anchored"),
    ],
)
def test_noisy_process_settings_leave_the_clean_run_alone(standard_clean, overrides):
    clean = _run("logei", apply_error=False, **overrides)
    assert not NEW_COLUMNS & set(clean.columns)
    pd.testing.assert_frame_equal(
        _without_timing(clean[standard_clean.columns]), _without_timing(standard_clean), check_exact=True
    )


@pytest.mark.parametrize(
    "overrides, acq, noisy, match",
    [
        (dict(noise_schedule="1-:1", error_model="bias"), "logei", True, "gaussian"),
        (dict(noise_schedule="front10"), "logei", False, "outside"),
        (dict(likelihood="relevance_pursuit", observation_noise="known"), "logei", True, "relevance_pursuit"),
        (dict(likelihood="relevance_pursuit", input_noise_model="nigp"), "logei", True, "relevance_pursuit"),
        (dict(input_error_model="missing_mcar", replicate_first=2), "logei", True, "replication"),
        (dict(input_error_model="missing_mcar", final_rerate_top=1), "logei", True, "re-rating"),
        (dict(input_error_model="missing_low"), "replei", True, "replei"),
        (dict(input_error_model="missing_mcar", input_noise_model="nigp"), "logei", True, "slip"),
        (dict(input_error_model="missing_mcar", input_error_recorded="actual"), "logei", True, "actual"),
        (dict(input_error_model="missing_mcar", missing_handling="impute_low", observation_noise="known"),
         "logei", True, "imputed"),
        (dict(input_error_model="missing_mcar", input_error_scale=1.5), "logei", True, "probability"),
        (dict(missing_handling="impute_low"), "logei", True, "lost rating"),
        (dict(missing_handling="mean"), "logei", True, "--missing-handling"),
        (dict(rater_offset_ratio=0.5), "logei", True, "needs --rater-assign"),
        (dict(rater_model="backfit"), "logei", True, "needs --rater-assign"),
        (dict(rater_assign="block:3"), "logei", True, "do nothing"),
        (dict(rater_assign="block:3", rater_offset_ratio=-1.0), "logei", True, "non-negative"),
        (dict(rater_assign="block:3", rater_offset_ratio=0.5, error_model="ar1"), "logei", True, "gaussian"),
        (dict(rater_assign="block:3", rater_model="backfit", likelihood="student_t"), "logei", True, "backfit"),
        (dict(response_ceiling=1.0), "logei", True, r"\(0, 1\)"),
        (dict(ceiling_mode="anchored"), "logei", True, "needs --response-ceiling"),
    ],
)
def test_incompatible_combinations_fail_clearly(overrides, acq, noisy, match):
    settings = {**vars(_config(error_model="gaussian", jitter_std=0.25, input_error_scale=0.5)), **overrides}
    with pytest.raises(ValueError, match=match):
        sim.validate_error_extensions(acq, settings, is_multi=False, noisy=noisy)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(input_error_model="missing_mcar"),
        dict(rater_assign="block:2", rater_offset_ratio=1.0),
        dict(response_ceiling=0.5),
        dict(likelihood="relevance_pursuit"),
    ],
)
def test_single_objective_only(overrides):
    settings = {**vars(_config(error_model="gaussian", jitter_std=0.25)), **overrides}
    with pytest.raises(ValueError, match="single-objective"):
        sim.validate_error_extensions("qlogehvi", settings, is_multi=True)


def test_run_simulation_applies_the_same_checks():
    with pytest.raises(ValueError, match="gaussian"):
        _run("logei", error_model="bias", noise_schedule="1-:1")
    with pytest.raises(ValueError, match="replication"):
        _missing_run("drop", replicate_first=1)
    # a missing rate is a probability, so the input-uncertain wrapper cannot borrow it
    with pytest.raises(ValueError, match="probability"):
        sim.validate_acquisition_extensions(
            "logei", is_multi=False, input_error_model="missing_mcar", input_uncertain_acq=4)


# ---------------------------------------------------------------------------
# drivers and analyses
# ---------------------------------------------------------------------------


def test_flags_are_off_by_default_and_reach_the_filename():
    default = syn.parse_args([])
    assert syn._variant_suffix(default, "gaussian", 0.2, 0.5) == ""
    fields = sim.adaptation_fields(default)
    assert {key: fields[key] for key in (
        "noise_schedule", "missing_handling", "rater_assign", "rater_offset_ratio", "rater_model",
        "response_ceiling", "ceiling_mode")} == {
        "noise_schedule": "none", "missing_handling": "drop", "rater_assign": "none",
        "rater_offset_ratio": 0.0, "rater_model": "none", "response_ceiling": 0.0, "ceiling_mode": "fixed"}
    cases = [
        (["--noise-schedule", "front10"], "_sched-front10"),
        (["--noise-schedule", "1-10:2,11-:0.75"], "_sched-1-10@2+11-@0.75"),
        (["--input-error", "missing_mcar", "--input-error-from-sweep"], "_miss-drop"),
        (["--input-error", "missing_low", "--input-error-from-sweep", "--missing-handling", "impute_low"],
         "_miss-impute_low"),
        (["--input-error", "missing_mcar", "--input-error-scale", "0.2"], "_ie0.2_miss-drop"),
        (["--rater-assign", "block:5", "--rater-offset-ratio", "0.5"], "_rater-block5-tau0.5"),
        (["--rater-assign", "roundrobin:3", "--rater-model", "backfit"], "_rater-roundrobin3-tau0_raterfit-backfit"),
        (["--response-ceiling", "0.9", "--ceiling-mode", "anchored"], "_ceil0.9-anchored"),
        (["--likelihood", "relevance_pursuit"], "_relevancepursuit"),
    ]
    for argv, suffix in cases:
        assert syn._variant_suffix(syn.parse_args(argv), "gaussian", 0.2, 0.5) == suffix, argv


def test_a_namespace_without_the_flags_keeps_them_off():
    """The fitted-oracle driver has none of the new flags."""
    namespace = argparse.Namespace(replicate_first=0, final_rerate="0,0", input_noise_model="none",
                                   inference_rule=None)
    config = _config(**sim.adaptation_fields(namespace))
    assert (config.noise_schedule, config.missing_handling, config.rater_assign, config.rater_offset_ratio,
            config.rater_model, config.response_ceiling, config.ceiling_mode) == (
        "none", "drop", "none", 0.0, "none", 0.0, "fixed")
    assert config.likelihood == "gaussian"


@pytest.mark.parametrize(
    "argv, match",
    [
        (["--iterations", "8", "--noise-schedule", "front10", "--error-models", "gaussian"], "outside"),
        (["--noise-schedule", "front10", "--error-models", "gaussian,bias"], "gaussian"),
        (["--noise-schedule", "front10", "--error-models", "gaussian", "--input-error", "slip",
          "--input-error-from-sweep"], "gaussian"),
        (["--input-error", "missing_mcar", "--input-error-from-sweep", "--jitter-stds", "0.5,2"], "probability"),
        (["--input-error", "missing_mcar", "--input-error-scale", "0.3", "--replicate-first", "2"], "replication"),
        (["--rater-assign", "block:5"], "do nothing"),
        (["--rater-assign", "block:x", "--rater-offset-ratio", "1"], "--rater-assign"),
        (["--likelihood", "relevance_pursuit", "--observation-noise", "known"], "relevance_pursuit"),
        (["--response-ceiling", "1.5"], r"\(0, 1\)"),
    ],
)
def test_driver_rejects_bad_combinations_before_running(argv, match):
    with pytest.raises(ValueError, match=match):
        syn.main(["--functions", "branin", "--acq", "logei", "--dry-run", *argv])


def test_driver_accepts_a_valid_combination_and_rejects_the_multi_objective_suite(tmp_path):
    syn.main(["--functions", "branin", "--acq", "logei", "--dry-run", "--error-models", "gaussian",
              "--noise-schedule", "U", "--rater-assign", "roundrobin:4", "--rater-offset-ratio", "0.5",
              "--rater-model", "backfit", "--response-ceiling", "0.8", "--ceiling-mode", "anchored",
              "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="single-objective"):
        syn.main(["--multi-objective", "--functions", mob.MO_ORDER[0], "--response-ceiling", "0.5",
                  "--dry-run", "--output-dir", str(tmp_path)])


def test_clean_run_marker_tracks_the_remedies_that_change_the_clean_run(tmp_path):
    rp = syn.parse_args(["--likelihood", "relevance_pursuit"])
    syn._guard_clean_run_settings(tmp_path, rp)
    assert json.loads((tmp_path / syn.CLEAN_RUN_MARKER).read_text()) == {"likelihood": "relevance_pursuit"}
    backfit = syn.parse_args(["--rater-assign", "block:5", "--rater-model", "backfit"])
    with pytest.raises(ValueError, match="separate --output-dir"):
        syn._guard_clean_run_settings(tmp_path, backfit)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    syn._guard_clean_run_settings(fresh, backfit)
    assert json.loads((fresh / syn.CLEAN_RUN_MARKER).read_text()) == {"rater_model": ["backfit", "block:5"]}
    # the noisy-process settings leave the clean run alone and are not tracked
    other = tmp_path / "other"
    other.mkdir()
    for argv in (["--noise-schedule", "front10"], ["--rater-assign", "block:5", "--rater-offset-ratio", "1"],
                 ["--response-ceiling", "0.5"], ["--input-error", "missing_low", "--missing-handling", "impute_low"]):
        syn._guard_clean_run_settings(other, syn.parse_args(argv))
    assert not (other / syn.CLEAN_RUN_MARKER).exists()


def test_extra_runs_reads_the_missing_labels(tmp_path):
    import analyse_extra_runs as aer

    bench = tmp_path / "branin"
    bench.mkdir()
    stem = "bo_sensor_error_branin_value_logei_seed7"
    for name in (f"{stem}_baseline_exact.csv",
                 f"{stem}_jittered_exact_missing_mcar_jit0_std0.5_miss-drop.csv",
                 f"{stem}_jittered_exact_missing_low_jit20_std0.25_miss-impute_low.csv",
                 f"{stem}_jittered_exact_gaussian_jit0_std0.5_sched-front10.csv",
                 f"{stem}_jittered_extra_trees_gaussian_jit10_std0.5.csv"):
        (bench / name).write_text("iteration,simple_regret_true\n1,1.0\n")
    _, jittered = aer.index_runs(tmp_path)
    got = sorted((r["error_model"], r["oracle"], r["jitter_iteration"], r["variant"]) for r in jittered)
    assert got == [
        ("gaussian", "exact", 0, "sched-front10"),
        ("gaussian", "extra_trees", 10, ""),
        ("missing_low", "exact", 20, "miss-impute_low"),
        ("missing_mcar", "exact", 0, "miss-drop"),
    ]


SMOKE = ["--functions", "branin", "--seeds", "7", "--iterations", "8", "--initial-samples", "5",
         "--candidate-pool", "32", "--acq-num-restarts", "2", "--acq-raw-samples", "16",
         "--acq-mc-samples", "16", "--acq-maxiter", "20", "--n-jobs", "1", "--jitter-iterations", "0"]


def test_driver_end_to_end_missing_arm_pairs_and_analyses(tmp_path):
    syn.main([*SMOKE, "--acq-list", "logei", "--error-models", "gaussian", "--jitter-stds", "0.5",
              "--input-error", "missing_low", "--input-error-from-sweep", "--missing-handling", "impute_low",
              "--response-ceiling", "0.7", "--ceiling-mode", "anchored", "--output-dir", str(tmp_path)])
    bench = tmp_path / "branin"
    noisy = ("bo_sensor_error_branin_value_logei_seed7_jittered_exact_missing_low_jit0_std0.5"
             "_miss-impute_low_ceil0.7-anchored.csv")
    assert sorted(p.name for p in bench.glob("*.csv")) == [
        "bo_sensor_error_branin_value_logei_seed7_baseline_exact.csv", noisy]
    assert not (tmp_path / syn.CLEAN_RUN_MARKER).exists()
    frame = pd.read_csv(bench / noisy)
    assert {"missing", "imputed_value", "ceiling_value"} <= set(frame.columns)
    assert not frame["acq_opt_failed"].any()
    summary = pd.read_csv(tmp_path / "bo_synthetic_error_summary.csv")
    metrics = ["final_simple_regret_true", "auc_simple_regret_true", "final_inference_simple_regret_true",
               "auc_inference_simple_regret_true", "delta_l2_norm"]
    assert np.isfinite(summary[metrics].to_numpy(dtype=float)).all()

    import analyse_extra_runs as aer
    import evaluate_research_question as erq

    response = erq.build_response_table(erq.load_iteration_logs(bench))
    assert response["param_columns"].eq(",".join(COLS)).all()
    paired = erq.build_paired_table(response)
    assert len(paired) == 1 and paired["error_model"].iloc[0] == "missing_low"
    assert np.isfinite(paired[["auc_simple_regret_excess_true",
                               "final_inference_simple_regret_excess_true"]].to_numpy(dtype=float)).all()
    runs = aer.per_run_table(tmp_path, [3], [0.0])
    assert runs["error_model"].tolist() == ["missing_low"] and np.isfinite(runs["extra"]).all()


def test_driver_end_to_end_schedule_and_backfit_relay(tmp_path):
    syn.main([*SMOKE, "--acq-list", "ucb", "--error-models", "gaussian", "--jitter-stds", "0.25",
              "--noise-schedule", "1-4:1.5,5-8:0.5", "--rater-assign", "roundrobin:2",
              "--rater-offset-ratio", "1", "--rater-model", "backfit", "--output-dir", str(tmp_path)])
    bench = tmp_path / "branin"
    noisy = ("bo_sensor_error_branin_value_ucb_seed7_jittered_exact_gaussian_jit0_std0.25"
             "_sched-1-4@1.5+5-8@0.5_rater-roundrobin2-tau1_raterfit-backfit.csv")
    baseline = "bo_sensor_error_branin_value_ucb_seed7_baseline_exact.csv"
    assert sorted(p.name for p in bench.glob("*.csv")) == [baseline, noisy]
    frame = pd.read_csv(bench / noisy)
    assert frame["rater_id"].tolist() == [0, 1] * 4
    np.testing.assert_array_equal(frame["noise_effort"], [1.5] * 4 + [0.5] * 4)
    clean = pd.read_csv(bench / baseline)
    assert "rater_id" in clean.columns and "noise_effort" not in clean.columns
    assert json.loads((tmp_path / syn.CLEAN_RUN_MARKER).read_text()) == {
        "rater_model": ["backfit", "roundrobin:2"]}
