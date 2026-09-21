"""Multi-objective halo error: a rating factor shared across objectives, and its backfit.

Neither adds a trial. The tests pin that no existing name moved, that rho = 0
draws exactly the standard stream and rho > 0 adds ONE shared draw after it,
that the errors carry the SD and correlation they claim, that the backfit
matches a hand computation on a fitted ModelListGP, that a halo run refits on
and deploys the corrected ratings, that --error-cross-corr leaves the clean run
alone, and that both flags are off by default, reach the filename (and, for the
backfit, the clean-run marker) and fail clearly in bad combinations.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_multiobjective as mob  # noqa: E402
import bo_sensor_error_simulation as sim  # noqa: E402
import bo_synthetic_error_simulation as syn  # noqa: E402

# The simulator before this build (2026-09-14), where the copy exists.
BACKUP_SIM = Path(
    r"C:\Users\markc\AppData\Local\Temp\claude\C--Users-markc-Desktop-hitl-mobo-robustness-tests"
    r"\a133aa05-04f4-4678-851c-66290dbc1d23\scratchpad\backup_2026-09-14\scripts"
    r"\bo_sensor_error_simulation.py"
)
PROBLEM = "branincurrin"
SPEC = mob.MO_BENCHMARKS[PROBLEM]
OUTPUT_NAMES = [sim._objective_output_name(c) for c in SPEC.objective_columns]
HALO_COLUMNS = {"halo_rho_hat", *(f"halo_correction_{name}" for name in OUTPUT_NAMES)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stats() -> dict:
    return mob.load_mo_stats()[PROBLEM]


def _fields(**overrides) -> dict:
    fields = dict(
        iterations=6, jitter_iteration=0, jitter_std=0.0, single_error=False,
        initial_samples=5, candidate_pool=16, objective="multi_objective",
        objective_columns=SPEC.objective_columns, param_columns=SPEC.param_columns, seed=7,
        error_model="none", error_bias=0.25, error_spike_prob=0.1, error_spike_std=0.5,
        dropout_strategy="hold_last", normalize_objective=False, objective_weights=None,
        acq_num_restarts=1, acq_raw_samples=16, acq_maxiter=20, acq_mc_samples=16,
        ref_point=np.asarray(_stats()["ref_point"], dtype=float),
    )
    fields.update(overrides)
    return fields


def _config(**overrides) -> "sim.SimulationConfig":
    return sim.SimulationConfig(**_fields(**overrides))


def _run(module=sim, acq: str = "qlognehvi", apply_error: bool = True, seed: int = 7,
         **overrides) -> pd.DataFrame:
    """A short branincurrin run. apply_error=True defaults to gaussian noise of 0.5 SD from trial 1."""
    if apply_error:
        overrides.setdefault("error_model", "gaussian")
        overrides.setdefault("jitter_std", 0.5)
    config = module.SimulationConfig(**_fields(seed=seed, **overrides))
    oracle = mob.SyntheticMultiOracle.from_stats(PROBLEM, mob.load_mo_stats())
    torch.manual_seed(seed)
    return module.run_simulation(
        oracle=oracle, bounds=module.Bounds(low=oracle.bounds_low, high=oracle.bounds_high),
        config=config, acq=module.AcquisitionConfig(name=acq),
        rng=np.random.default_rng(seed),
        jitter_rng=np.random.default_rng(99) if apply_error else None,
        run_id="t", apply_error=apply_error, oracle_model="exact", y_opt=float(_stats()["max_hv"]),
    )


def _without_timing(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns="fit_time_sec")


@pytest.fixture(scope="module")
def standard_clean() -> pd.DataFrame:
    return _run(apply_error=False)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def test_no_acquisition_or_error_model_was_added():
    # The halo error is a flag on the gaussian model, not a new model, so the
    # seed indices of the jitter stream cannot have moved.
    assert sim.ERROR_MODEL_CHOICES == ["gaussian", "bias", "dropout", "spike", "drift", "ar1"]
    # shiplcb was appended on 2026-09-21. The guarantee this test exists for is
    # that nothing was INSERTED (the index seeds the jitter stream) and that
    # nothing new reached the multi-objective suite, which is checked below.
    assert sim.ACQUISITION_CHOICES[-3:] == ["ts", "aei", "shiplcb"]
    assert len(sim.ACQUISITION_CHOICES) == 21
    assert sim.MO_HALO_MODEL_CHOICES == ["none", "backfit"]


@pytest.fixture(scope="module")
def backup_module():
    if not BACKUP_SIM.is_file():
        pytest.skip("pre-build backup of the simulator not present")
    spec = importlib.util.spec_from_file_location("bo_sim_backup_halo_2026_09_14", BACKUP_SIM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pre_existing_names_keep_their_indices(backup_module):
    for attr in ("ACQUISITION_CHOICES", "ERROR_MODEL_CHOICES", "INPUT_ERROR_CHOICES", "LIKELIHOOD_CHOICES",
                 "INCUMBENT_CHOICES", "MULTI_ACQUISITION_CHOICES", "OBSERVATION_NOISE_CHOICES"):
        old, new = list(getattr(backup_module, attr)), list(getattr(sim, attr))
        assert new[: len(old)] == old, attr
        for name in old:
            assert new.index(name) == old.index(name), (attr, name)
    assert list(sim.DEFAULT_ACQUISITION_CHOICES) == list(backup_module.DEFAULT_ACQUISITION_CHOICES)


def test_a_default_multi_objective_run_reproduces_the_backup_simulator(backup_module):
    current = _run(sim, iterations=6)
    backup = _run(backup_module, iterations=6)
    assert not HALO_COLUMNS & set(current.columns)
    pd.testing.assert_frame_equal(_without_timing(current), _without_timing(backup), check_exact=True)


# ---------------------------------------------------------------------------
# (1) the cross-correlated error
# ---------------------------------------------------------------------------


def test_zero_cross_corr_draws_exactly_the_standard_stream():
    config = _config(error_model="gaussian", jitter_std=0.5, error_cross_corr=0.0)
    true = np.array([0.3, -1.2])
    rng, reference = np.random.default_rng(5), np.random.default_rng(5)
    observed, error = sim.apply_sensor_error(true, 3, config, rng, true)
    np.testing.assert_array_equal(observed, true + reference.normal(0.0, 0.5, size=2))
    np.testing.assert_array_equal(error, observed - true)
    assert rng.bit_generator.state == reference.bit_generator.state


@pytest.mark.parametrize("rho", [0.3, 1.0])
def test_cross_corr_adds_one_shared_draw_after_the_standard_one(rho):
    config = _config(error_model="gaussian", jitter_std=0.5, error_cross_corr=rho)
    true = np.array([0.3, -1.2, 2.0])
    rng, reference = np.random.default_rng(11), np.random.default_rng(11)
    observed, _ = sim.apply_sensor_error(true, 1, config, rng, true)
    eps = reference.normal(0.0, 0.5, size=3)
    z = reference.normal(0.0, 0.5)
    np.testing.assert_array_equal(observed, true + (math.sqrt(1.0 - rho) * eps + math.sqrt(rho) * z))
    assert rng.bit_generator.state == reference.bit_generator.state


def test_cross_corr_before_the_onset_draws_nothing():
    config = _config(error_model="gaussian", jitter_std=0.5, error_cross_corr=0.7, jitter_iteration=4)
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    observed, error = sim.apply_sensor_error(np.array([1.0, 2.0]), 4, config, rng, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(observed, [1.0, 2.0])
    assert not error.any() and rng.bit_generator.state == state


def test_cross_corr_errors_keep_their_sd_and_carry_the_correlation():
    rho, sd = 0.6, 0.5
    config = _config(error_model="gaussian", jitter_std=sd, error_cross_corr=rho)
    rng = np.random.default_rng(2026)
    true = np.zeros(3)
    errors = np.array([sim.apply_sensor_error(true, 1, config, rng, true)[1] for _ in range(4000)])
    np.testing.assert_allclose(errors.std(axis=0, ddof=1), sd, atol=0.03)
    corr = np.corrcoef(errors, rowvar=False)[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(corr, rho, atol=0.05)
    # known_noise_variance declares the per-objective variance, which rho leaves alone
    assert sim.known_noise_variance(1, config, True) == pytest.approx(sd**2)


def test_cross_corr_leaves_the_clean_run_alone(standard_clean):
    clean = _run(apply_error=False, error_cross_corr=0.7)
    pd.testing.assert_frame_equal(_without_timing(clean), _without_timing(standard_clean), check_exact=True)


# ---------------------------------------------------------------------------
# (2) the backfit
# ---------------------------------------------------------------------------


def test_shared_factor_by_hand():
    R = np.array([[1.0, 0.8, 1.2], [-0.5, -0.2, -0.9], [0.3, 0.6, 0.1], [-1.1, -0.7, -0.4]])
    rho_hat, z = sim.halo_shared_factor(R)
    corr = np.corrcoef(R, rowvar=False)
    expected_rho = float(np.clip(corr[~np.eye(3, dtype=bool)].mean(), 0.0, 0.95))
    assert rho_hat == pytest.approx(expected_rho, abs=1e-15) and 0 < rho_hat < 0.95
    np.testing.assert_allclose(z, math.sqrt(expected_rho) * 3 * R.mean(axis=1) / (1 + 2 * expected_rho),
                               rtol=1e-14)


def test_shared_factor_edge_cases():
    # identical columns: correlation 1, capped at 0.95
    column = np.array([[0.1], [0.9], [-0.4], [0.3]])
    rho_hat, z = sim.halo_shared_factor(np.hstack([column, column]))
    assert rho_hat == 0.95
    np.testing.assert_allclose(z, math.sqrt(0.95) * 2 * column[:, 0] / 1.95)
    # opposite columns: no shared factor, no correction
    rho_hat, z = sim.halo_shared_factor(np.hstack([column, -column]))
    assert rho_hat == 0.0 and not z.any()
    # two trials estimate nothing
    rho_hat, z = sim.halo_shared_factor(np.array([[1.0, 2.0], [-1.0, -2.0]]))
    assert rho_hat == 0.0 and not z.any()
    # a constant column has no correlation; the remaining pair decides
    rho_hat, _ = sim.halo_shared_factor(np.hstack([column, column, np.ones((4, 1))]))
    assert rho_hat == 0.95


def test_shared_factor_recovers_a_planted_halo():
    rng = np.random.default_rng(1)
    rho, n, m = 0.5, 4000, 3
    z = rng.standard_normal(n)
    R = math.sqrt(1 - rho) * rng.standard_normal((n, m)) + math.sqrt(rho) * z[:, None]
    rho_hat, z_hat = sim.halo_shared_factor(R)
    assert rho_hat == pytest.approx(rho, abs=0.03)
    # E[z | r] leaves posterior variance 1 - m rho / (1 + (m - 1) rho) = 0.25
    assert np.mean((z - z_hat) ** 2) == pytest.approx(0.25, abs=0.03)


def _halo_data(n: int = 20, rho: float = 0.9, sd: float = 0.7, seed: int = 4):
    rng = np.random.default_rng(seed)
    low, high = mob.bounds(PROBLEM)
    X = low + rng.random((n, SPEC.dim)) * (high - low)
    oracle = mob.SyntheticMultiOracle.from_stats(PROBLEM, mob.load_mo_stats())
    noise = sd * (math.sqrt(1 - rho) * rng.standard_normal((n, 2)) + math.sqrt(rho) * rng.standard_normal((n, 1)))
    Y = oracle.predict_many(X) + noise
    bounds_tensor = sim.Bounds(low=low, high=high).tensor
    train_X = torch.tensor(X, dtype=torch.double)
    train_Y_list = [torch.tensor(Y[:, [j]], dtype=torch.double) for j in range(2)]
    return train_X, train_Y_list, bounds_tensor


def test_backfit_matches_a_hand_computation_on_a_fitted_model_list():
    torch.manual_seed(0)
    train_X, train_Y_list, bounds_tensor = _halo_data()
    gp, _ = sim._fit_model_list_gp(train_X, train_Y_list, bounds_tensor)

    Y = torch.cat(train_Y_list, dim=1).numpy()
    with torch.no_grad():
        mu = np.column_stack([m.posterior(train_X).mean.reshape(-1).numpy() for m in gp.models])
    sigma = np.array([
        math.sqrt(float(m.likelihood.noise.detach().reshape(-1)[0])
                  * float(m.outcome_transform.stdvs.detach().reshape(-1)[0]) ** 2)
        for m in gp.models
    ])
    R = (Y - mu) / sigma
    corr = np.corrcoef(R, rowvar=False)[0, 1]
    rho = float(np.clip(corr, 0.0, 0.95))
    assert rho > 0, "the planted halo should leave positively correlated residuals"
    z = math.sqrt(rho) * 2 * R.mean(axis=1) / (1 + rho)
    expected = Y - sigma[None, :] * math.sqrt(rho) * z[:, None]

    refits = []

    def fit_models(X, Ys):
        refits.append(torch.cat(Ys, dim=1).numpy().copy())
        return sim._fit_model_list_gp(X, Ys, bounds_tensor)

    model, corrected, correction, rho_hat = sim.fit_mo_halo_backfit(gp, train_X, train_Y_list, fit_models)
    assert rho_hat == pytest.approx(rho, rel=1e-12)
    assert len(refits) == 1 and model is not gp and isinstance(model, sim.ModelListGP)
    np.testing.assert_allclose(refits[0], expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(torch.cat(corrected, dim=1).numpy(), expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(Y - correction, expected, rtol=0, atol=1e-10)


def test_backfit_keeps_the_fit_when_there_is_no_shared_factor(monkeypatch):
    torch.manual_seed(0)
    train_X, train_Y_list, bounds_tensor = _halo_data(n=8)
    gp, _ = sim._fit_model_list_gp(train_X, train_Y_list, bounds_tensor)
    monkeypatch.setattr(sim, "halo_shared_factor", lambda R: (0.0, np.zeros(len(R))))

    def refuse(*_):
        raise AssertionError("no refit without a correction")

    model, corrected, correction, rho_hat = sim.fit_mo_halo_backfit(gp, train_X, train_Y_list, refuse)
    assert model is gp and rho_hat == 0.0 and not correction.any()
    for before, after in zip(train_Y_list, corrected):
        assert torch.equal(before, after)


@pytest.fixture(scope="module")
def halo_run():
    """A noisy halo run with the backfit, recording every ModelListGP fit and acquisition call."""
    fits: list[tuple[np.ndarray, object]] = []
    calls: list[tuple[object, np.ndarray, int]] = []
    original_fit, original_candidate = sim._fit_model_list_gp, sim.get_botorch_candidate

    def recording_fit(X, Ys, bounds_tensor):
        model, mll = original_fit(X, Ys, bounds_tensor)
        fits.append((torch.cat(Ys, dim=1).numpy().copy(), model))
        return model, mll

    def recording_candidate(**kwargs):
        calls.append((kwargs["gp_model"], torch.cat(kwargs["train_Y"], dim=1).numpy().copy(), len(fits)))
        return original_candidate(**kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(sim, "_fit_model_list_gp", recording_fit)
    patch.setattr(sim, "get_botorch_candidate", recording_candidate)
    try:
        frame = _run(iterations=9, jitter_std=1.0, error_cross_corr=0.9, mo_halo_model="backfit")
    finally:
        patch.undo()
    return frame, fits, calls


def test_backfit_run_logs_its_estimates_and_refits_on_the_corrected_ratings(halo_run):
    frame, fits, calls = halo_run
    assert HALO_COLUMNS <= set(frame.columns)
    assert not frame["acq_opt_failed"].any()
    rho_hat = frame["halo_rho_hat"].to_numpy(dtype=float)
    assert np.isnan(rho_hat[:5]).all() and np.isfinite(rho_hat[5:]).all()
    assert ((rho_hat[5:] >= 0) & (rho_hat[5:] <= 0.95)).all()
    # one fit per model-based trial, plus the refit wherever a factor was found
    assert len(fits) == int(sum(1 + (r > 0) for r in rho_hat[5:]))
    assert len(calls) == 4
    for model, targets, fitted in calls:
        # the acquisition gets the LAST fit of its trial, and that fit's targets
        assert model is fits[fitted - 1][1]
        np.testing.assert_array_equal(targets, fits[fitted - 1][0])
    assert rho_hat[-1] > 0, "the planted halo should be found on the last trial"
    rows = fits[-1][0].shape[0]
    logged = frame[[f"halo_correction_{name}" for name in OUTPUT_NAMES]].to_numpy(dtype=float)
    np.testing.assert_allclose(fits[-2][0] - fits[-1][0], logged[:rows], rtol=0, atol=1e-12)
    assert not logged[rows:].any()
    # the first fit of each trial trains on the raw ratings
    raw = frame[[f"objective_observed_{name}" for name in OUTPUT_NAMES]].to_numpy(dtype=float)
    np.testing.assert_array_equal(fits[-2][0], raw[:rows])


def test_backfit_run_deploys_the_pareto_set_of_the_corrected_ratings(halo_run):
    frame, _, _ = halo_run
    observed = frame[[f"objective_observed_{name}" for name in OUTPUT_NAMES]].to_numpy(dtype=float)
    true = frame[[f"objective_true_{name}" for name in OUTPUT_NAMES]].to_numpy(dtype=float)
    logged = frame[[f"halo_correction_{name}" for name in OUTPUT_NAMES]].to_numpy(dtype=float)
    ref_point = np.asarray(_stats()["ref_point"], dtype=float)
    corrected = torch.tensor(observed - logged, dtype=torch.double)
    members = sim.is_non_dominated(corrected).numpy()
    replay = sim._compute_hypervolume(list(true[members]), ref_point)
    assert frame["inference_value_true"].iloc[-1] == pytest.approx(replay, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, acq, is_multi, noisy, match",
    [
        (dict(error_cross_corr=-0.1), "qlognehvi", True, True, r"\[0, 1\]"),
        (dict(error_cross_corr=1.5), "qlognehvi", True, True, r"\[0, 1\]"),
        (dict(error_cross_corr=float("nan")), "qlognehvi", True, True, r"\[0, 1\]"),
        (dict(error_cross_corr=0.5), "logei", False, True, "multi-objective only"),
        (dict(error_cross_corr=0.5), "logei", False, False, "multi-objective only"),
        (dict(error_cross_corr=0.5, error_model="bias"), "qlognehvi", True, True, "gaussian"),
        (dict(error_cross_corr=0.5, error_model="none"), "qlognehvi", True, True, "gaussian"),
        (dict(mo_halo_model="backfit"), "logei", False, True, "multi-objective only"),
        (dict(mo_halo_model="backfit"), "random", True, True, "fits no"),
        (dict(mo_halo_model="backfit"), "sobol", True, False, "fits no"),
        (dict(mo_halo_model="joint"), "qlognehvi", True, True, "--mo-halo-model"),
    ],
)
def test_incompatible_combinations_fail_clearly(overrides, acq, is_multi, noisy, match):
    settings = {**vars(_config(error_model="gaussian", jitter_std=0.25)), **overrides}
    with pytest.raises(ValueError, match=match):
        sim.validate_error_extensions(acq, settings, is_multi=is_multi, noisy=noisy)


@pytest.mark.parametrize(
    "overrides, acq, noisy",
    [
        (dict(error_cross_corr=1.0), "qlognehvi", True),
        (dict(error_cross_corr=0.5, error_model="bias"), "qlognehvi", False),  # the clean twin draws nothing
        (dict(error_cross_corr=0.5, mo_halo_model="backfit", noise_schedule="1-:1"), "qehvi", True),
        (dict(error_cross_corr=0.5), "random", True),
    ],
)
def test_valid_combinations_pass(overrides, acq, noisy):
    settings = {**vars(_config(error_model="gaussian", jitter_std=0.25)), **overrides}
    sim.validate_error_extensions(acq, settings, is_multi=True, noisy=noisy)


def test_run_simulation_applies_the_same_checks():
    with pytest.raises(ValueError, match="gaussian"):
        _run(error_model="bias", error_cross_corr=0.5)
    with pytest.raises(ValueError, match="fits no"):
        _run(acq="random", apply_error=False, mo_halo_model="backfit")


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def test_flags_are_off_by_default_and_reach_the_filename():
    default = syn.parse_args([])
    assert syn._variant_suffix(default, "gaussian", 0.2, 0.5) == ""
    fields = sim.adaptation_fields(default)
    assert (fields["error_cross_corr"], fields["mo_halo_model"]) == (0.0, "none")
    cases = [
        (["--error-cross-corr", "0.5"], "_xc0.5"),
        (["--mo-halo-model", "backfit"], "_halo-backfit"),
        (["--error-cross-corr", "0.25", "--mo-halo-model", "backfit"], "_xc0.25_halo-backfit"),
        (["--error-cross-corr", "0"], ""),
    ]
    for argv, suffix in cases:
        assert syn._variant_suffix(syn.parse_args(argv), "gaussian", 0.2, 0.5) == suffix, argv


def test_a_namespace_without_the_flags_keeps_them_off():
    """The fitted-oracle driver has neither flag."""
    namespace = argparse.Namespace(replicate_first=0, final_rerate="0,0", input_noise_model="none",
                                   inference_rule=None)
    config = _config(**sim.adaptation_fields(namespace))
    assert (config.error_cross_corr, config.mo_halo_model) == (0.0, "none")


MO = ["--multi-objective", "--functions", PROBLEM, "--dry-run"]


@pytest.mark.parametrize(
    "argv, match",
    [
        (["--functions", "branin", "--acq", "logei", "--dry-run", "--error-cross-corr", "0.5"],
         "multi-objective only"),
        (["--functions", "branin", "--acq", "logei", "--dry-run", "--mo-halo-model", "backfit"],
         "multi-objective only"),
        ([*MO, "--acq-list", "qlognehvi", "--error-models", "gaussian,bias", "--error-cross-corr", "0.5"],
         "gaussian"),
        ([*MO, "--acq-list", "qlognehvi", "--error-models", "gaussian", "--error-cross-corr", "1.5"],
         r"\[0, 1\]"),
        ([*MO, "--acq", "all", "--error-models", "gaussian", "--mo-halo-model", "backfit"], "fits no"),
    ],
)
def test_driver_rejects_bad_combinations_before_running(argv, match):
    with pytest.raises(ValueError, match=match):
        syn.main(argv)


def test_clean_run_marker_tracks_the_backfit_only(tmp_path):
    cross = tmp_path / "cross"
    cross.mkdir()
    syn._guard_clean_run_settings(cross, syn.parse_args(["--multi-objective", "--error-cross-corr", "0.5"]))
    assert not (cross / syn.CLEAN_RUN_MARKER).exists()
    backfit = syn.parse_args(["--multi-objective", "--mo-halo-model", "backfit"])
    syn._guard_clean_run_settings(tmp_path, backfit)
    assert json.loads((tmp_path / syn.CLEAN_RUN_MARKER).read_text()) == {"mo_halo_model": "backfit"}
    with pytest.raises(ValueError, match="separate --output-dir"):
        syn._guard_clean_run_settings(tmp_path, syn.parse_args(["--multi-objective"]))


def test_driver_end_to_end_halo_arm(tmp_path):
    syn.main(["--multi-objective", "--functions", PROBLEM, "--acq-list", "qlognehvi", "--seeds", "7",
              "--iterations", "8", "--initial-samples", "5", "--candidate-pool", "32",
              "--acq-num-restarts", "2", "--acq-raw-samples", "32", "--acq-mc-samples", "32",
              "--acq-maxiter", "20", "--n-jobs", "1", "--error-models", "gaussian", "--jitter-stds", "0.5",
              "--jitter-iterations", "0", "--error-cross-corr", "0.8", "--mo-halo-model", "backfit",
              "--output-dir", str(tmp_path)])
    bench = tmp_path / PROBLEM
    stem = f"bo_sensor_error_{PROBLEM}_multi_objective_qlognehvi_seed7"
    baseline = f"{stem}_baseline_exact.csv"
    noisy = f"{stem}_jittered_exact_gaussian_jit0_std0.5_xc0.8_halo-backfit.csv"
    assert sorted(p.name for p in bench.glob("*.csv")) == [baseline, noisy]
    assert json.loads((tmp_path / syn.CLEAN_RUN_MARKER).read_text()) == {"mo_halo_model": "backfit"}
    for name in (baseline, noisy):
        frame = pd.read_csv(bench / name)
        assert HALO_COLUMNS <= set(frame.columns), name
        assert not frame["acq_opt_failed"].any(), name
    summary = pd.read_csv(tmp_path / "bo_synthetic_error_summary.csv")
    metrics = ["final_simple_regret_true", "auc_simple_regret_true", "final_inference_simple_regret_true"]
    assert np.isfinite(summary[metrics].to_numpy(dtype=float)).all()
