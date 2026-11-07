"""Two simulator options added for the review round of 2026-11-07.

drift_down is the drift ramp with its sign reversed, a rater who grows harsher; it shares drift's noise seed, so
the two directions are paired draw for draw. --gp-prior legacy replaces BoTorch's default surrogate (RBF with
dimension-scaled log-normal priors) by BoTorch's default before 0.12 (scaled Matern-5/2 with Gamma priors and a
Gamma(1.1, 0.05) noise prior), built by BoTorch's own helpers. What must hold: neither option moves any run that
does not ask for it, "--error-models all" keeps its meaning, the legacy prior reaches the loop and the ship-rule
refit, and the replay that cannot refit it refuses it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bo_sensor_error_simulation as sim  # noqa: E402
import bo_synthetic_error_simulation as synth  # noqa: E402


def _config(error_model: str) -> sim.SimulationConfig:
    # The same minimal configuration as tests/test_acquisition_extensions.py's.
    return sim.SimulationConfig(
        iterations=50, jitter_iteration=20, jitter_std=1.0, single_error=False,
        initial_samples=5, candidate_pool=32, objective="value", objective_columns=["value"],
        param_columns=["x0", "x1"], seed=7, error_model=error_model, error_bias=0.25,
        error_spike_prob=0.1, error_spike_std=0.5, dropout_strategy="hold_last",
        normalize_objective=False, objective_weights=None, acq_num_restarts=2,
        acq_raw_samples=16, acq_maxiter=30, acq_mc_samples=16, ref_point=None,
    )


def test_drift_down_mirrors_the_ramp_on_the_same_jitter():
    truth = np.array([0.3])
    for t in (21, 35, 50):
        up, _ = sim.apply_sensor_error(truth, t, _config("drift"), np.random.default_rng(5), truth)
        down, _ = sim.apply_sensor_error(truth, t, _config("drift_down"), np.random.default_rng(5), truth)
        ramp = 1.0 * (t - 20) / 30
        assert (up - truth)[0] - (down - truth)[0] == pytest.approx(2 * ramp)
        assert (up + down - 2 * truth)[0] == pytest.approx(2 * ((up - truth)[0] - ramp))
    before, _ = sim.apply_sensor_error(truth, 20, _config("drift_down"), np.random.default_rng(5), truth)
    assert before[0] == truth[0]


def test_the_seed_index_pairs_the_two_drifts_and_moves_no_existing_model():
    for i, name in enumerate(["gaussian", "bias", "dropout", "spike", "drift", "ar1"]):
        assert sim.error_model_seed_index(name) == i == sim.ERROR_MODEL_CHOICES.index(name)
    assert sim.error_model_seed_index("drift_down") == sim.ERROR_MODEL_CHOICES.index("drift")
    assert sim.ALL_ERROR_MODELS == ["gaussian", "bias", "dropout", "spike", "drift", "ar1"]
    assert sim.parse_error_models(None, "all") == sim.ALL_ERROR_MODELS
    assert sim.parse_error_models(None, "drift_down") == ["drift_down"]


def test_the_default_prior_adds_nothing_and_the_legacy_prior_is_botorchs():
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior

    assert sim.gp_prior_modules("default", 3) == {}
    legacy = sim.gp_prior_modules("legacy", 3)
    kernel, likelihood = legacy["covar_module"], legacy["likelihood"]
    assert isinstance(kernel, ScaleKernel) and isinstance(kernel.base_kernel, MaternKernel)
    assert kernel.base_kernel.nu == 2.5 and kernel.base_kernel.ard_num_dims == 3
    noise_prior = likelihood.noise_covar.noise_prior
    assert isinstance(noise_prior, GammaPrior)
    assert float(noise_prior.concentration) == pytest.approx(1.1) and float(noise_prior.rate) == pytest.approx(0.05)
    assert "likelihood" not in sim.gp_prior_modules("legacy", 3, fixed_noise=True)
    with pytest.raises(ValueError):
        sim.gp_prior_modules("matern", 3)


def test_the_driver_records_the_prior_and_refuses_it_with_another_surrogate(tmp_path):
    args = synth.parse_args(["--gp-prior", "legacy"])
    assert sim.adaptation_fields(args)["gp_prior"] == "legacy"
    assert synth._clean_run_settings(args)["gp_prior"] == "legacy"
    assert "gp_prior" not in synth._clean_run_settings(synth.parse_args([]))
    assert sim.adaptation_fields(synth.parse_args([]))["gp_prior"] == "default"
    with pytest.raises(ValueError, match="gp-prior"):
        synth.main(["--functions", "branin", "--acq-list", "logei", "--seeds", "7", "--iterations", "6",
                    "--initial-samples", "5", "--error-models", "gaussian", "--jitter-stds", "1.0",
                    "--jitter-iterations", "0", "--n-jobs", "1", "--gp-prior", "legacy",
                    "--likelihood", "student_t", "--output-dir", str(tmp_path / "bad")])


def test_a_short_legacy_run_differs_from_the_default_after_the_initial_design(tmp_path):
    base = ["--functions", "branin", "--acq-list", "logei", "--seeds", "7", "--iterations", "8",
            "--initial-samples", "5", "--error-models", "drift_down", "--jitter-stds", "1.0",
            "--jitter-iterations", "0", "--n-jobs", "1"]
    synth.main(base + ["--output-dir", str(tmp_path / "default")])
    synth.main(base + ["--output-dir", str(tmp_path / "legacy"), "--gp-prior", "legacy"])
    a = pd.read_csv(next((tmp_path / "default").rglob("*_baseline_exact.csv")))
    b = pd.read_csv(next((tmp_path / "legacy").rglob("*_baseline_exact.csv")))
    cols = str(a["param_columns"].iloc[0]).split(",")
    assert np.array_equal(a[cols].to_numpy()[:5], b[cols].to_numpy()[:5])      # the same initial design
    assert not np.array_equal(a[cols].to_numpy()[5:], b[cols].to_numpy()[5:])  # a different surrogate after it
    noisy = next((tmp_path / "legacy").rglob("*_jittered_*.csv"))
    assert "drift_down" in noisy.name
    df = pd.read_csv(noisy)
    err = (df["objective_observed"] - df["objective_true"]).to_numpy()
    assert np.isfinite(err).all()


def test_the_ship_rule_refit_uses_the_loops_prior_and_the_sitting_replay_refuses_it():
    import replay_end_of_study as eos
    import rescore_ship_rules as rs

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(8, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    gp, diag = rs.fit_loop_gp(X, y, np.zeros(2), np.ones(2), seed=1, gp_prior="legacy")
    assert diag["gp_kernel"] == "ScaleKernel"
    gp, diag = rs.fit_loop_gp(X, y, np.zeros(2), np.ones(2), seed=1)
    assert diag["gp_kernel"] != "ScaleKernel"
    assert eos.idiosyncratic_sd("drift_down", 2.0) == 2.0
    noise = eos.SittingNoise(base=1.0, onset=20, error_model="drift_down", jitter_std=1.0, iterations=50)
    assert noise.ramp(35) == pytest.approx(-0.5)
    assert eos.SittingNoise(base=1.0, onset=20, error_model="drift", jitter_std=1.0, iterations=50).ramp(35) == 0.5


def test_the_run_name_parser_keeps_drift_down_whole():
    import analyse_extra_runs as aer

    m = aer.JITTERED.match("exact_drift_down_jit0_std1.0")
    assert m and m["oracle"] == "exact" and m["model"] == "drift_down" and m["suffix"] == ""
    m = aer.JITTERED.match("exact_drift_jit20_std5.0_sp0.15-20")
    assert m and m["model"] == "drift" and m["suffix"] == "_sp0.15-20"
    m = aer.JITTERED.match("extra_trees_missing_mcar_jit0_std0.15")
    assert m and m["oracle"] == "extra_trees" and m["model"] == "missing_mcar"
