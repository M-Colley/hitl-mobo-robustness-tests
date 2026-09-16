"""Acquisition-side follow-ups: lcb incumbent, ts, aei, input-uncertain wrapper, min-distance.

All five change which design is proposed and none changes how many are rated.
The tests pin three things: that no existing name moved (the noise seed is
built from list POSITIONS), the arithmetic of each piece against a hand
computation, and that every flag is off at its default and reaches the
filename when on.
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
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402
import boba_multiobjective as mob  # noqa: E402
import bo_sensor_error_simulation as sim  # noqa: E402
import bo_synthetic_error_simulation as syn  # noqa: E402

# The simulator as it stood before these features were added (2026-09-14).
# Where the copy exists, the index test compares against that module itself;
# the pinned lists were copied from it and keep the check alive elsewhere.
BACKUP_SIM = Path(
    r"C:\Users\markc\AppData\Local\Temp\claude\C--Users-markc-Desktop-hitl-mobo-robustness-tests"
    r"\a133aa05-04f4-4678-851c-66290dbc1d23\scratchpad\backup_2026-09-14\scripts"
    r"\bo_sensor_error_simulation.py"
)
PRE_EXISTING_ACQUISITIONS = [
    "logei", "logpi", "ei", "pi", "ucb", "qucb", "qei", "qpi", "qnei", "greedy",
    "qkg", "replei",
    "qehvi", "qnehvi", "qlogehvi", "qlognehvi",
    "random", "sobol",
]
PRE_EXISTING_ERROR_MODELS = ["gaussian", "bias", "dropout", "spike", "drift", "ar1"]
PRE_EXISTING_INCUMBENTS = ["posterior_mean", "observed_max"]
BRANIN = bb.BENCHMARKS["branin"]
COLS = BRANIN.param_columns


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> "sim.SimulationConfig":
    fields = dict(
        iterations=9, jitter_iteration=0, jitter_std=0.0, single_error=False,
        initial_samples=5, candidate_pool=32, objective="value", objective_columns=["value"],
        param_columns=COLS, seed=7, error_model="none", error_bias=0.25,
        error_spike_prob=0.1, error_spike_std=0.5, dropout_strategy="hold_last",
        normalize_objective=False, objective_weights=None, acq_num_restarts=2,
        acq_raw_samples=16, acq_maxiter=30, acq_mc_samples=16, ref_point=None,
    )
    fields.update(overrides)
    return sim.SimulationConfig(**fields)


def _run(acq: str = "logei", apply_error: bool = False, seed: int = 7, **overrides) -> pd.DataFrame:
    """A short branin run. apply_error=True defaults to gaussian noise of 0.25 SD."""
    stats = bb.load_stats()
    if apply_error:
        overrides.setdefault("error_model", "gaussian")
        overrides.setdefault("jitter_std", 0.25)
    config = _config(seed=seed, **overrides)
    y_opt = (stats["branin"]["y_opt"] - stats["branin"]["mean"]) / stats["branin"]["std"]
    torch.manual_seed(seed)
    return sim.run_simulation(
        oracle=bb.SyntheticOracle.from_stats("branin", stats),
        bounds=sim.Bounds(low=BRANIN.bounds_low, high=BRANIN.bounds_high),
        config=config, acq=sim.AcquisitionConfig(name=acq),
        rng=np.random.default_rng(seed),
        jitter_rng=np.random.default_rng(99) if apply_error else None,
        run_id="t", apply_error=apply_error, oracle_model="exact", y_opt=y_opt,
    )


def _data(scale: float = 40.0, seed: int = 0):
    torch.manual_seed(seed)
    bounds = torch.tensor([[0.0, -2.0], [1.0, 2.0]], dtype=torch.double)
    X = bounds[0] + torch.rand(10, 2, dtype=torch.double) * (bounds[1] - bounds[0])
    # A large output scale, so a missing standardisation step is off by ~20x.
    Y = scale * (torch.sin(3.0 * X[:, :1]) + 0.3 * X[:, 1:])
    Y = Y + 0.1 * scale * torch.randn(10, 1, dtype=torch.double)
    return X, Y, bounds


def _fitted_gp(train_Yvar: float | None = None):
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X, Y, bounds = _data()
    extra = {} if train_Yvar is None else {"train_Yvar": torch.full_like(Y, train_Yvar)}
    gp = SingleTaskGP(X, Y, input_transform=Normalize(d=2, bounds=bounds),
                      outcome_transform=Standardize(m=1), **extra)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    return gp, X, Y, bounds


def _test_points(bounds: torch.Tensor, n: int = 7, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return bounds[0] + torch.rand(n, 1, 2, dtype=torch.double, generator=generator) * (bounds[1] - bounds[0])


# ---------------------------------------------------------------------------
# names and indices
# ---------------------------------------------------------------------------


def test_pre_existing_names_keep_their_indices():
    for index, name in enumerate(PRE_EXISTING_ACQUISITIONS):
        assert sim.ACQUISITION_CHOICES.index(name) == index
    for index, name in enumerate(PRE_EXISTING_ERROR_MODELS):
        assert sim.ERROR_MODEL_CHOICES.index(name) == index
    assert sim.INCUMBENT_CHOICES[:2] == PRE_EXISTING_INCUMBENTS
    # the new names sit at the very end
    assert sim.ACQUISITION_CHOICES[len(PRE_EXISTING_ACQUISITIONS):] == ["ts", "aei"]
    assert sim.INCUMBENT_CHOICES[2:] == ["lcb"]


@pytest.mark.skipif(not BACKUP_SIM.is_file(), reason="pre-feature backup of the simulator not present")
def test_pre_existing_names_match_the_backup_module():
    spec = importlib.util.spec_from_file_location("bo_sim_backup_2026_09_14", BACKUP_SIM)
    backup = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(backup)
    assert backup.ACQUISITION_CHOICES == PRE_EXISTING_ACQUISITIONS
    for name in backup.ACQUISITION_CHOICES:
        assert sim.ACQUISITION_CHOICES.index(name) == backup.ACQUISITION_CHOICES.index(name)
    for name in backup.ERROR_MODEL_CHOICES:
        assert sim.ERROR_MODEL_CHOICES.index(name) == backup.ERROR_MODEL_CHOICES.index(name)
    for name in backup.INCUMBENT_CHOICES:
        assert sim.INCUMBENT_CHOICES.index(name) == backup.INCUMBENT_CHOICES.index(name)
    assert list(sim.DEFAULT_ACQUISITION_CHOICES) == list(backup.DEFAULT_ACQUISITION_CHOICES)
    assert list(sim.SINGLE_ACQUISITION_CHOICES) == list(backup.SINGLE_ACQUISITION_CHOICES)


def test_ts_and_aei_are_opt_in_and_single_objective():
    for name in ("ts", "aei"):
        assert name not in sim.DEFAULT_ACQUISITION_CHOICES
        assert name not in sim.SINGLE_ACQUISITION_CHOICES
        assert name not in syn.DEFAULT_ACQUISITIONS
        assert name in syn.SYNTHETIC_ACQUISITION_CHOICES
        assert name not in syn.MO_ACQUISITION_CHOICES
        with pytest.raises(ValueError, match="single-objective"):
            sim.validate_acquisition_extensions(name, is_multi=True)
    assert sim.parse_acquisition_list("all", None) == list(sim.DEFAULT_ACQUISITION_CHOICES)
    assert sim.parse_acquisition_list("ts,aei", None) == ["ts", "aei"]
    assert sim.filter_acquisitions_for_objective(["logei", "ts", "aei", "qehvi"], "composite") == [
        "logei", "ts", "aei"]
    assert sim.filter_acquisitions_for_objective(["ts", "aei", "qehvi"], "multi_objective") == ["qehvi"]


# ---------------------------------------------------------------------------
# (1) the lcb incumbent
# ---------------------------------------------------------------------------


def test_lcb_is_the_latent_mean_minus_one_sd():
    gp, X, _, _ = _fitted_gp()
    mean, lcb = sim._posterior_mean_and_lcb(gp, X)
    with torch.no_grad():
        latent = gp.posterior(X)
        predictive = gp.posterior(X, observation_noise=True)
    np.testing.assert_allclose(mean.numpy(), latent.mean.reshape(-1).numpy(), rtol=1e-12)
    np.testing.assert_allclose(
        lcb.numpy(), (latent.mean - latent.variance.sqrt()).reshape(-1).numpy(), rtol=1e-12)
    # latent, not predictive: including the observation noise would widen the band
    assert (predictive.variance.reshape(-1) > latent.variance.reshape(-1)).all()


def test_lcb_incumbent_changes_the_clean_run_after_the_initial_design():
    base = _run("logei")
    lcb = _run("logei", incumbent="lcb")
    assert not lcb["acq_opt_failed"].any()
    pd.testing.assert_frame_equal(base[COLS].head(5), lcb[COLS].head(5))
    assert not np.allclose(base[COLS].to_numpy(), lcb[COLS].to_numpy())


# ---------------------------------------------------------------------------
# (2) Thompson sampling
# ---------------------------------------------------------------------------


def test_thompson_path_is_seeded_by_the_run_rng_and_leaves_torch_alone():
    gp, _, _, bounds = _fitted_gp()
    test_X = _test_points(bounds, n=16)
    state = torch.get_rng_state()
    a = sim.build_thompson_sampling(gp, np.random.default_rng(3))
    assert torch.equal(state, torch.get_rng_state())
    b = sim.build_thompson_sampling(gp, np.random.default_rng(3))
    c = sim.build_thompson_sampling(gp, np.random.default_rng(4))
    with torch.no_grad():
        va, vb, vc = a(test_X), b(test_X), c(test_X)
    assert va.shape == (16,)
    assert torch.equal(va, vb)
    assert not torch.allclose(va, vc)
    # exactly one integer is taken from the run rng per path
    rng, reference = np.random.default_rng(3), np.random.default_rng(3)
    sim.build_thompson_sampling(gp, rng)
    reference.integers(0, 2**31 - 1)
    assert rng.integers(0, 10**9) == reference.integers(0, 10**9)


def test_thompson_paths_are_posterior_draws_in_output_units():
    gp, _, _, bounds = _fitted_gp()
    test_X = _test_points(bounds, n=5)
    draws = 200
    with torch.no_grad():
        values = torch.stack([
            sim.build_thompson_sampling(gp, np.random.default_rng(s))(test_X) for s in range(draws)])
        posterior = gp.posterior(test_X)
    mean = posterior.mean.reshape(-1)
    sd = posterior.variance.sqrt().reshape(-1)
    assert ((values.mean(0) - mean).abs() <= 6.0 * sd / math.sqrt(draws)).all()
    np.testing.assert_allclose(values.std(0).numpy(), sd.numpy(), rtol=0.25)


def test_matheron_fallback_draws_the_same_path(monkeypatch):
    if sim.PathwiseThompsonSampling is None:
        pytest.skip("PathwiseThompsonSampling absent; the fallback is already the one in use")
    gp, _, _, bounds = _fitted_gp()
    test_X = _test_points(bounds, n=9)
    primary = sim.build_thompson_sampling(gp, np.random.default_rng(11))
    monkeypatch.setattr(sim, "PathwiseThompsonSampling", None)
    fallback = sim.build_thompson_sampling(gp, np.random.default_rng(11))
    assert isinstance(fallback, sim._MatheronThompsonSampling)
    with torch.no_grad():
        torch.testing.assert_close(fallback(test_X), primary(test_X))


# ---------------------------------------------------------------------------
# (3) augmented expected improvement
# ---------------------------------------------------------------------------


def test_observation_noise_sd_is_converted_to_output_units():
    gp, _, Y, _ = _fitted_gp()
    standardised = float(gp.likelihood.noise)
    hand = math.sqrt(standardised * float(Y.std()) ** 2)
    assert sim.observation_noise_sd(gp) == pytest.approx(hand, rel=1e-10)
    # the standardised value is far away, so the conversion is doing real work
    assert math.sqrt(standardised) < 0.2 * hand
    # fixed (per-point) noise of variance 4 in output units is an SD of 2
    fixed, *_ = _fitted_gp(train_Yvar=4.0)
    assert sim.observation_noise_sd(fixed) == pytest.approx(2.0, rel=1e-8)


def test_observation_noise_sd_of_a_student_t_likelihood_is_its_variance():
    from robust_gp import build_robust_gp

    X, Y, bounds = _data()
    model = build_robust_gp(X, Y, bounds=bounds, fit=False)
    scale2 = float(model.likelihood.noise)
    nu = float(model.likelihood.deg_free)
    hand = math.sqrt(scale2 * nu / (nu - 2.0)) * float(Y.std())
    assert sim.observation_noise_sd(model) == pytest.approx(hand, rel=1e-10)


def test_aei_matches_a_hand_computation():
    gp, X, _, bounds = _fitted_gp()
    noise_sd = sim.observation_noise_sd(gp)
    acqf = sim.build_augmented_ei(gp, X)

    with torch.no_grad():
        at_train = gp.posterior(X)
    mean_train = at_train.mean.reshape(-1)
    sd_train = at_train.variance.sqrt().reshape(-1)
    threshold = float(mean_train[int(torch.argmax(mean_train - sd_train))])
    assert float(acqf.best_f) == pytest.approx(threshold, rel=1e-12)
    assert float(acqf.noise_sd) == pytest.approx(noise_sd, rel=1e-12)

    test_X = _test_points(bounds, n=7)
    with torch.no_grad():
        posterior = gp.posterior(test_X)
        got = acqf(test_X).numpy()
        log_ei = sim.LogExpectedImprovement(model=gp, best_f=threshold)(test_X).numpy()
    mu = posterior.mean.reshape(-1).numpy()
    s = posterior.variance.sqrt().reshape(-1).numpy()
    u = (mu - threshold) / s
    ei = s * (u * norm.cdf(u) + norm.pdf(u))
    factor = 1.0 - noise_sd / np.sqrt(s**2 + noise_sd**2)
    assert (ei > 1e-200).all()
    np.testing.assert_allclose(got, np.log(ei * factor), rtol=1e-8)
    # the discount alone, against LogEI at the same threshold
    np.testing.assert_allclose(np.exp(got - log_ei), factor, rtol=1e-9)


def test_aei_penalty_is_stable_where_the_model_is_sure():
    # 1 - s_n / sqrt(s^2 + s_n^2) ~ s^2 / (2 s_n^2) for s << s_n; the naive
    # difference loses all its digits at s = 1e-6, s_n = 1.
    sigma = torch.tensor([1e-6, 0.5, 3.0], dtype=torch.double)
    noise = torch.tensor(1.0, dtype=torch.double)
    got = sim.LogAugmentedExpectedImprovement.log_penalty(sigma, noise).numpy()
    assert got[0] == pytest.approx(math.log(0.5e-12), rel=1e-9)
    exact = np.log(1.0 - 1.0 / np.sqrt(sigma.numpy()[1:] ** 2 + 1.0))
    np.testing.assert_allclose(got[1:], exact, rtol=1e-12)
    assert float(sim.LogAugmentedExpectedImprovement.log_penalty(sigma, torch.tensor(0.0)).abs().max()) < 1e-12


def test_aei_under_known_noise_uses_the_injected_variance(monkeypatch):
    seen: list = []
    original = sim.build_augmented_ei

    def spy(model, train_X, noise_sd=None):
        seen.append(noise_sd)
        return original(model, train_X, noise_sd=noise_sd)

    monkeypatch.setattr(sim, "build_augmented_ei", spy)
    frame = _run("aei", apply_error=True, observation_noise="known")
    assert not frame["acq_opt_failed"].any()
    assert seen and all(value == pytest.approx(0.25) for value in seen)
    seen.clear()
    _run("aei", apply_error=True)
    assert seen and all(value is None for value in seen)


@pytest.mark.parametrize("acq", ["ts", "aei"])
def test_ts_and_aei_runs_are_deterministic(acq):
    first = _run(acq, apply_error=True)
    second = _run(acq, apply_error=True)
    assert not first["acq_opt_failed"].any()
    pd.testing.assert_frame_equal(first.drop(columns="fit_time_sec"), second.drop(columns="fit_time_sec"))


# ---------------------------------------------------------------------------
# (4) the input-uncertain wrapper
# ---------------------------------------------------------------------------


class _FirstCoordinate(torch.nn.Module):
    """A deterministic stand-in acquisition: the value is the first coordinate."""

    model = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., 0, 0]


def test_input_uncertain_wrapper_averages_clipped_perturbations():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    deltas = torch.tensor([[0.3, 0.0], [-0.1, 0.0], [0.05, 0.2]], dtype=torch.double)
    X = torch.tensor([[[0.9, 0.5]], [[0.2, 0.1]]], dtype=torch.double)
    # x0 = 0.9 -> 1.2 clips to 1.0, 0.8, 0.95;  x0 = 0.2 -> 0.5, 0.1, 0.25
    shifted = [[1.0, 0.8, 0.95], [0.5, 0.1, 0.25]]
    arithmetic = sim.InputUncertainAcquisition(_FirstCoordinate(), deltas, bounds, log_space=False)
    np.testing.assert_allclose(arithmetic(X).numpy(), [np.mean(v) for v in shifted], rtol=1e-12)
    log_space = sim.InputUncertainAcquisition(_FirstCoordinate(), deltas, bounds, log_space=True)
    np.testing.assert_allclose(
        log_space(X).numpy(), [math.log(np.mean(np.exp(v))) for v in shifted], rtol=1e-12)
    assert {"logei", "logpi", "aei", "replei"} <= set(sim.LOG_VALUED_ACQUISITIONS)
    assert not {"ei", "ucb", "ts", "qnei"} & set(sim.LOG_VALUED_ACQUISITIONS)


def test_perturbations_are_fixed_by_the_rng_and_scaled_by_the_range():
    bounds = sim.Bounds(low=np.array([-5.0, 0.0]), high=np.array([10.0, 5.0]))
    a = sim.draw_input_perturbations(np.random.default_rng(1), bounds, 512, 0.1)
    b = sim.draw_input_perturbations(np.random.default_rng(1), bounds, 512, 0.1)
    c = sim.draw_input_perturbations(np.random.default_rng(2), bounds, 512, 0.1)
    assert a.shape == (512, 2)
    assert torch.equal(a, b) and not torch.equal(a, c)
    np.testing.assert_allclose(a.std(dim=0).numpy(), [1.5, 0.5], rtol=0.05)
    np.testing.assert_allclose(a.mean(dim=0).numpy(), [0.0, 0.0], atol=0.05)


def test_effective_scale_explicit_applies_to_the_clean_run_and_borrowed_does_not():
    assert sim.input_uncertain_effective_scale(_config(), apply_error=True) == 0.0
    borrowed = _config(input_uncertain_acq=8, input_error_model="slip", input_error_scale=0.05)
    assert sim.input_uncertain_effective_scale(borrowed, apply_error=True) == 0.05
    assert sim.input_uncertain_effective_scale(borrowed, apply_error=False) == 0.0
    explicit = dataclasses.replace(borrowed, input_uncertain_scale=0.02)
    assert sim.input_uncertain_effective_scale(explicit, apply_error=False) == 0.02
    assert sim.input_uncertain_effective_scale(dataclasses.replace(explicit, input_uncertain_acq=0), True) == 0.0


def test_input_uncertain_wrapper_is_inert_in_the_clean_run_unless_scaled_explicitly():
    slip = dict(input_error_model="slip", input_error_scale=0.05)
    plain = _run("logei", **slip)
    borrowed = _run("logei", input_uncertain_acq=4, **slip)
    pd.testing.assert_frame_equal(plain.drop(columns="fit_time_sec"), borrowed.drop(columns="fit_time_sec"))
    explicit = _run("logei", input_uncertain_acq=4, input_uncertain_scale=0.05, **slip)
    assert not explicit["acq_opt_failed"].any()
    assert not np.allclose(plain[COLS].to_numpy(), explicit[COLS].to_numpy())
    noisy = _run("ucb", apply_error=True, error_model="none", input_uncertain_acq=4, **slip)
    assert not noisy["acq_opt_failed"].any()


def test_incompatible_combinations_fail_clearly():
    check = sim.validate_acquisition_extensions
    with pytest.raises(ValueError, match="probability"):
        check("logei", is_multi=False, input_error_model="misclick", input_uncertain_acq=4)
    check("logei", is_multi=False, input_error_model="misclick", input_uncertain_acq=4,
          input_uncertain_scale=0.05)
    with pytest.raises(ValueError, match="do nothing"):
        check("logei", is_multi=False, input_uncertain_acq=4)
    with pytest.raises(ValueError, match="qkg"):
        check("qkg", is_multi=False, input_uncertain_acq=4, input_uncertain_scale=0.1)
    with pytest.raises(ValueError, match="single-objective"):
        check("qlogehvi", is_multi=True, input_uncertain_acq=4, input_uncertain_scale=0.1)
    with pytest.raises(ValueError, match=">= 0"):
        check("logei", is_multi=False, input_uncertain_acq=-1)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        check("logei", is_multi=False, min_distance=1.0)
    with pytest.raises(ValueError, match="one-shot"):
        check("qkg", is_multi=False, min_distance=0.1)
    check("qlogehvi", is_multi=True, min_distance=0.1)
    # run_simulation applies the same checks before doing anything
    with pytest.raises(ValueError, match="do nothing"):
        _run("logei", input_uncertain_acq=4)


# ---------------------------------------------------------------------------
# (5) the min-distance redirect
# ---------------------------------------------------------------------------


def test_rms_unit_distance_by_hand():
    bounds = sim.Bounds(low=np.array([0.0, 0.0]), high=np.array([10.0, 10.0]))
    got = sim._rms_unit_distance(np.array([[0.0, 0.0]]), np.array([[10.0, 10.0], [5.0, 0.0]]), bounds)
    assert got[0] == pytest.approx(math.sqrt(0.5**2 / 2.0))


def test_min_distance_redirect_picks_the_best_far_pool_point():
    bounds = sim.Bounds(low=np.array([0.0, 0.0]), high=np.array([10.0, 10.0]))
    logged = np.array([[5.0, 5.0], [1.0, 1.0]])
    pool = torch.tensor([[5.1, 5.0], [8.0, 8.0], [2.0, 9.0], [1.2, 1.1]], dtype=torch.double)
    near = torch.tensor([[5.2, 5.1]], dtype=torch.double)
    far = torch.tensor([[9.0, 1.0]], dtype=torch.double)

    # the two best-valued pool points are near-repeats; of the far ones, 2.0 wins
    values = torch.tensor([9.0, 1.0, 2.0, 8.0], dtype=torch.double)
    chosen, redirected = sim._min_distance_redirect(near, pool, values, logged, bounds, 0.1)
    assert redirected
    torch.testing.assert_close(chosen, torch.tensor([[2.0, 9.0]], dtype=torch.double))
    # a NaN value never wins
    values_nan = torch.tensor([9.0, 5.0, float("nan"), 8.0], dtype=torch.double)
    chosen, _ = sim._min_distance_redirect(near, pool, values_nan, logged, bounds, 0.1)
    torch.testing.assert_close(chosen, torch.tensor([[8.0, 8.0]], dtype=torch.double))
    # a proposal that is already far enough is kept
    kept, redirected = sim._min_distance_redirect(far, pool, values, logged, bounds, 0.1)
    assert kept is far and not redirected
    # no pool point far enough: keep the proposal
    kept, redirected = sim._min_distance_redirect(near, pool, values, logged, bounds, 0.9)
    assert kept is near and not redirected


def test_screen_candidate_pool_returns_the_pool_without_changing_the_draw():
    class Sum:
        def __call__(self, X: torch.Tensor) -> torch.Tensor:
            return X.sum(dim=-1).squeeze(-1)

    bounds = sim.Bounds(low=np.zeros(2), high=np.ones(2))
    best, initial = sim.screen_candidate_pool(Sum(), bounds, np.random.default_rng(0), 16, 3)
    best2, initial2, pool, values = sim.screen_candidate_pool(
        Sum(), bounds, np.random.default_rng(0), 16, 3, return_pool=True)
    torch.testing.assert_close(best, best2)
    torch.testing.assert_close(initial, initial2)
    assert pool.shape == (16, 2)
    torch.testing.assert_close(values, pool.sum(dim=-1))
    torch.testing.assert_close(initial2[0, 0], pool[int(torch.argmax(values))])


def test_min_distance_logs_redirects_and_redirected_designs_are_far():
    default = _run("greedy")
    assert "min_distance_redirect" not in default.columns
    radius = 0.25
    frame = _run("greedy", min_distance=radius, iterations=10)
    assert not frame["acq_opt_failed"].any()
    flags = frame["min_distance_redirect"].to_numpy(dtype=bool)
    assert not flags[:5].any()
    assert flags.any()
    bounds = sim.Bounds(low=BRANIN.bounds_low, high=BRANIN.bounds_high)
    X = frame[COLS].to_numpy(dtype=float)
    for i in np.flatnonzero(flags):
        assert sim._rms_unit_distance(X[i:i + 1], X[:i], bounds)[0] >= radius


# ---------------------------------------------------------------------------
# drivers: defaults, filenames, validation, the clean-run marker
# ---------------------------------------------------------------------------


def test_flags_are_off_by_default_and_reach_the_filename():
    default = syn.parse_args([])
    assert syn._variant_suffix(default, "gaussian", 0.2, 0.5) == ""
    fields = sim.adaptation_fields(default)
    assert (fields["input_uncertain_acq"], fields["input_uncertain_scale"], fields["min_distance"]) == (0, -1.0, 0.0)
    both = syn.parse_args(["--input-uncertain-acq", "8", "--input-uncertain-scale", "0.05",
                           "--min-distance", "0.1"])
    assert syn._variant_suffix(both, "gaussian", 0.2, 0.5) == "_iu8-0.05_mind0.1"
    borrowed = syn.parse_args(["--input-uncertain-acq", "8", "--input-uncertain-scale", "-3"])
    assert syn._variant_suffix(borrowed, "gaussian", 0.2, 0.5) == "_iu8--1"
    lcb = syn.parse_args(["--incumbent", "lcb"])
    assert syn._variant_suffix(lcb, "gaussian", 0.2, 0.5) == "_inc-lcb"


def test_a_namespace_without_the_flags_keeps_them_off():
    """The fitted-oracle driver has none of the new flags."""
    namespace = argparse.Namespace(replicate_first=0, final_rerate="0,0", input_noise_model="none",
                                   inference_rule=None)
    config = _config(**sim.adaptation_fields(namespace))
    assert (config.input_uncertain_acq, config.input_uncertain_scale, config.min_distance) == (0, -1.0, 0.0)
    assert config.incumbent == "posterior_mean"


@pytest.mark.parametrize(
    "argv, match",
    [
        (["--acq", "logei", "--input-uncertain-acq", "4"], "do nothing"),
        (["--acq", "qkg", "--min-distance", "0.1"], "one-shot"),
        (["--acq", "logei", "--min-distance", "1.5"], r"\[0, 1\)"),
        (["--acq", "logei", "--input-error", "misclick", "--input-error-scale", "0.1",
          "--input-uncertain-acq", "4"], "probability"),
    ],
)
def test_driver_rejects_bad_combinations_before_running(argv, match):
    with pytest.raises(ValueError, match=match):
        syn.main(["--functions", "branin", "--dry-run", *argv])


def test_driver_rejects_ts_in_the_multi_objective_suite():
    with pytest.raises(ValueError, match="do not apply"):
        syn.main(["--multi-objective", "--functions", mob.MO_ORDER[0], "--acq-list", "ts", "--dry-run"])


def test_clean_run_marker_keeps_one_setting_per_directory(tmp_path):
    default = syn.parse_args([])
    mind = syn.parse_args(["--min-distance", "0.1"])
    other = syn.parse_args(["--min-distance", "0.2"])
    syn._guard_clean_run_settings(tmp_path, default)
    assert not (tmp_path / syn.CLEAN_RUN_MARKER).exists()
    syn._guard_clean_run_settings(tmp_path, mind)
    assert json.loads((tmp_path / syn.CLEAN_RUN_MARKER).read_text()) == {"min_distance": 0.1}
    syn._guard_clean_run_settings(tmp_path, mind)  # resuming the same setting is fine
    for args in (other, default):
        with pytest.raises(ValueError, match="separate --output-dir"):
            syn._guard_clean_run_settings(tmp_path, args)
    # a directory that already holds baselines refuses a clean-run-changing setting
    fresh = tmp_path / "fresh"
    (fresh / "branin").mkdir(parents=True)
    (fresh / "branin" / "bo_sensor_error_branin_value_logei_seed7_baseline_exact.csv").write_text("x\n")
    with pytest.raises(ValueError, match="fresh --output-dir"):
        syn._guard_clean_run_settings(fresh, mind)
    # a borrowed input-uncertain scale leaves the clean run alone, so it is not tracked
    syn._guard_clean_run_settings(fresh, syn.parse_args(["--input-uncertain-acq", "8", "--input-error", "slip"]))
    assert not (fresh / syn.CLEAN_RUN_MARKER).exists()


def test_driver_end_to_end_with_every_follow_up(tmp_path):
    syn.main([
        "--functions", "branin", "--acq-list", "ts,aei", "--incumbent", "lcb", "--seeds", "7",
        "--iterations", "7", "--initial-samples", "5", "--candidate-pool", "32",
        "--acq-num-restarts", "2", "--acq-raw-samples", "16", "--acq-mc-samples", "16",
        "--acq-maxiter", "20", "--n-jobs", "1", "--error-models", "gaussian",
        "--jitter-stds", "0.05", "--jitter-iterations", "0", "--input-error", "slip",
        "--input-error-from-sweep", "--input-uncertain-acq", "4", "--min-distance", "0.05",
        "--output-dir", str(tmp_path),
    ])
    files = sorted(p.name for p in (tmp_path / "branin").glob("*.csv"))
    assert "bo_sensor_error_branin_value_ts_seed7_baseline_exact_inc-lcb.csv" in files
    assert ("bo_sensor_error_branin_value_aei_seed7_jittered_exact_slip_jit0_std0.05"
            "_inc-lcb_iu4--1_mind0.05.csv") in files
    assert len(files) == 4
    for name in files:
        frame = pd.read_csv(tmp_path / "branin" / name)
        assert not frame["acq_opt_failed"].any()
        assert "min_distance_redirect" in frame.columns
    assert json.loads((tmp_path / syn.CLEAN_RUN_MARKER).read_text()) == {"min_distance": 0.05}
