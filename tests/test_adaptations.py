"""The process adaptations: early replication, re-evaluate-before-deploy, noisy-input GP.

Each option changes the trial schedule or the surrogate, and a slip in any of
them would show up as a plausible-looking curve rather than an error. These
tests pin what each one does to the sequence of designs and to the surrogate.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402
import bo_sensor_error_simulation as sim  # noqa: E402


def _run(iterations: int, initial: int, seed: int = 7, **overrides):
    stats = bb.load_stats()
    oracle = bb.SyntheticOracle.from_stats("hartmann_3", stats)
    spec = bb.BENCHMARKS["hartmann_3"]
    bounds = sim.Bounds(low=spec.bounds_low, high=spec.bounds_high)
    acq = sim.AcquisitionConfig(name="logei")
    config = sim.SimulationConfig(
        iterations=iterations, jitter_iteration=0, jitter_std=0.25, single_error=False,
        initial_samples=initial, candidate_pool=64, objective="value",
        objective_columns=["value"], param_columns=spec.param_columns, seed=seed,
        error_model="gaussian", error_bias=0.5, error_spike_prob=0.1,
        error_spike_std=0.5, dropout_strategy="hold_last", normalize_objective=False,
        objective_weights=None, acq_num_restarts=2, acq_raw_samples=32,
        acq_maxiter=50, acq_mc_samples=32, ref_point=None, **overrides,
    )
    torch.manual_seed(seed)
    frame = sim.run_simulation(
        oracle=oracle, bounds=bounds, config=config, acq=acq,
        rng=np.random.default_rng(seed), jitter_rng=np.random.default_rng(1234),
        run_id="t", apply_error=True, oracle_model="exact", y_opt=stats["hartmann_3"]["opt_z"],
    )
    return frame[spec.param_columns].to_numpy(dtype=float), frame


def test_replicate_first_rates_each_early_design_twice():
    X, frame = _run(iterations=11, initial=3, replicate_first=2, inference_rule="best_mean")
    assert not frame["acq_opt_failed"].any()
    # rows 4/5 and 6/7 (1-based) are two ratings of one design; then single ratings
    np.testing.assert_array_equal(X[3], X[4])
    np.testing.assert_array_equal(X[5], X[6])
    assert not np.array_equal(X[7], X[8])
    assert not np.array_equal(X[8], X[9])
    # the second rating is a fresh error draw on the same true value
    assert frame["objective_true"].iloc[3] == frame["objective_true"].iloc[4]
    assert frame["objective_observed"].iloc[3] != frame["objective_observed"].iloc[4]


def test_final_rerate_spends_the_last_trials_on_the_top_designs():
    X, frame = _run(iterations=12, initial=3, final_rerate_top=2, final_rerate_reps=2,
                    inference_rule="best_mean")
    tail = X[-4:]
    earlier = [sim._design_key(x) for x in X[:-4]]
    keys = [sim._design_key(x) for x in tail]
    assert len(set(keys)) == 2
    assert all(keys.count(k) == 2 for k in set(keys))
    assert all(k in earlier for k in keys)
    # round-robin: design A, design B, design A, design B
    assert keys[0] == keys[2] and keys[1] == keys[3]


def test_best_mean_index_prefers_the_design_with_the_best_mean_rating():
    X = [np.array([0.1]), np.array([0.9]), np.array([0.9]), np.array([0.1])]
    observed = [3.0, 2.5, 2.4, 0.0]   # design 0.1: mean 1.5; design 0.9: mean 2.45
    assert sim._best_mean_index(X, observed) == 1
    assert int(np.argmax(observed)) == 0


def test_rerate_schedule_ranks_by_mean_and_round_robins():
    X = [np.array([0.0]), np.array([0.5]), np.array([1.0]), np.array([0.5])]
    observed = [1.0, 2.0, 1.5, 3.0]   # means: 0.0 -> 1.0, 0.5 -> 2.5, 1.0 -> 1.5
    schedule = sim._rerate_schedule(X, observed, top=2, reps=2)
    keys = [sim._design_key(x) for x in schedule]
    assert keys == [sim._design_key(X[1]), sim._design_key(X[2])] * 2


def test_nigp_inflates_variance_where_the_posterior_mean_is_steep():
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    bounds = sim.Bounds(low=np.array([0.0]), high=np.array([1.0]))
    config = argparse.Namespace(input_error_scale=0.1)
    torch.manual_seed(0)
    X = torch.linspace(0.05, 0.95, 20, dtype=torch.double).reshape(-1, 1)

    def fixed_noise(slope: float) -> float:
        Y = slope * X + 0.01 * torch.randn_like(X)
        gp = SingleTaskGP(X, Y, input_transform=Normalize(d=1, bounds=bounds.tensor),
                          outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        refit, _ = sim._refit_with_input_noise(gp, X, Y, bounds, config)
        # FixedNoise likelihood, in standardised units: back to Y's units.
        return float((refit.likelihood.noise * refit.outcome_transform.stdvs.reshape(-1) ** 2).mean())

    steep, flat = fixed_noise(3.0), fixed_noise(0.1)
    # first order: inflation = (slope * s)^2 with s = 0.1 -> 0.09 against 1e-4
    assert steep > 5 * flat
    assert abs(steep - 0.09) < 0.04


def test_adaptation_fields_parse_and_default_the_inference_rule():
    args = argparse.Namespace(replicate_first=0, final_rerate="0,0", input_noise_model="none", inference_rule=None)
    assert sim.adaptation_fields(args)["inference_rule"] == "best_observed"
    args = argparse.Namespace(replicate_first=10, final_rerate="0,0", input_noise_model="none", inference_rule=None)
    assert sim.adaptation_fields(args)["inference_rule"] == "best_mean"
    args = argparse.Namespace(replicate_first=0, final_rerate="3,2", input_noise_model="nigp", inference_rule="best_observed")
    fields = sim.adaptation_fields(args)
    assert (fields["final_rerate_top"], fields["final_rerate_reps"], fields["inference_rule"]) == (3, 2, "best_observed")
    with pytest.raises(ValueError):
        sim.adaptation_fields(argparse.Namespace(replicate_first=0, final_rerate="3,0", input_noise_model="none", inference_rule=None))


def test_default_config_is_unchanged_for_published_runs():
    """Every published run used the defaults; they must still mean 'off'."""
    config = sim.SimulationConfig(
        iterations=5, jitter_iteration=0, jitter_std=0.0, single_error=False, initial_samples=2,
        candidate_pool=8, objective="value", objective_columns=["value"], param_columns=["x0"],
        seed=1, error_model="none", error_bias=0.0, error_spike_prob=0.0, error_spike_std=0.0,
        dropout_strategy="hold_last", normalize_objective=False, objective_weights=None,
        acq_num_restarts=1, acq_raw_samples=8, acq_maxiter=5, acq_mc_samples=8, ref_point=None,
    )
    assert (config.replicate_first, config.final_rerate_top, config.final_rerate_reps) == (0, 0, 0)
    assert config.input_noise_model == "none" and config.inference_rule == "best_observed"
