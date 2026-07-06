"""Tests for the NumPyro hierarchical oracle add-on (hierarchical_oracle_test.py).

These cover the deterministic pieces (feature maps, calibration scoring, the
honest cold-noise estimator, cold predictive assembly) without running MCMC, so
they stay fast. The scientific claims they lock in:

- the tree's honest cold-noise SD is far larger than its (near-zero) train
  residual SD -- this is the fairness fix that stops the calibration comparison
  being rigged in the HBM's favour;
- calibration scoring reports ~nominal coverage for a calibrated predictive and
  under-coverage for an over-confident one;
- the HBM cold predictive widens by sqrt(sigma^2 + tau_user^2) for unseen users.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load(module_name: str, filename: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


hier = _load("hierarchical_oracle_test_mod", "hierarchical_oracle_test.py")


# --------------------------------------------------------------------------- #
# Feature maps
# --------------------------------------------------------------------------- #
def test_linear_features_are_train_standardized() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(3.0, 2.0, size=(200, 4))
    X_test = rng.normal(3.0, 2.0, size=(50, 4))
    phi_tr, phi_te, info = hier.build_features(X_train, X_test, "linear", 0, rng)
    assert phi_tr.shape == (200, 4)
    assert phi_te.shape == (50, 4)
    assert np.allclose(phi_tr.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(phi_tr.std(axis=0), 1.0, atol=1e-6)
    assert info["lengthscale"] is None


def test_rff_features_shape_and_determinism() -> None:
    X_train = np.random.default_rng(1).normal(size=(120, 6))
    X_test = np.random.default_rng(2).normal(size=(30, 6))
    phi_tr_a, phi_te_a, info = hier.build_features(X_train, X_test, "rff", 64, np.random.default_rng(5))
    phi_tr_b, _, _ = hier.build_features(X_train, X_test, "rff", 64, np.random.default_rng(5))
    assert phi_tr_a.shape == (120, 64)
    assert phi_te_a.shape == (30, 64)
    assert info["lengthscale"] is not None and info["lengthscale"] > 0
    # same rng seed -> identical random Fourier projection
    assert np.allclose(phi_tr_a, phi_tr_b)


def test_unknown_mean_raises() -> None:
    X = np.zeros((3, 2))
    with pytest.raises(ValueError, match="Unknown mean"):
        hier.build_features(X, X, "quadratic", 0, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Honest cold-noise estimator (the fairness fix)
# --------------------------------------------------------------------------- #
def test_tree_cold_sigma_recovers_true_noise_and_beats_train_residual() -> None:
    """The honest cold estimate must recover the real unseen-user error (here the
    known total SD = sqrt(tau^2 + within^2) = sqrt(1.25) ~= 1.118), which the
    optimistic in-sample train-residual SD substantially underestimates. That gap
    is exactly why the naive plug-in produced a rigged over-confident tree."""
    rng = np.random.default_rng(7)
    n_users, per_user = 25, 12
    groups = np.repeat(np.arange(n_users), per_user)
    X = rng.normal(size=(n_users * per_user, 3))
    user_intercept = rng.normal(0.0, 1.0, size=n_users)[groups]     # tau = 1.0
    noise = rng.normal(0.0, 0.5, size=X.shape[0])                    # within = 0.5
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + user_intercept + noise

    tree = hier._fit_tree(X, y, seed=7, tree_scale=1.0)
    train_resid_sd = float(np.std(y - tree.predict(X)))
    cold_sigma = hier._tree_cold_sigma(X, y, groups, seed=7, tree_scale=1.0)

    true_total_sd = np.sqrt(1.0**2 + 0.5**2)                         # 1.118
    assert abs(cold_sigma - true_total_sd) < 0.25    # honest estimate ~ truth
    assert cold_sigma > 1.3 * train_resid_sd         # in-sample SD is optimistic


def test_extra_trees_predictive_shapes() -> None:
    rng = np.random.default_rng(3)
    X_train, y_train = rng.normal(size=(80, 3)), rng.normal(size=80)
    X_test = rng.normal(size=(20, 3))
    groups = np.repeat(np.arange(8), 10)
    point, samples = hier.extra_trees_predictive(
        X_train, y_train, X_test, groups, seed=1, tree_scale=0.5, draws=200
    )
    assert point.shape == (20,)
    assert samples.shape == (200, 20)


# --------------------------------------------------------------------------- #
# Calibration scoring
# --------------------------------------------------------------------------- #
def test_scoring_calibrated_predictive_hits_nominal_coverage() -> None:
    rng = np.random.default_rng(11)
    n, draws = 4000, 800
    y_true = rng.normal(0.0, 1.0, size=n)                      # truth ~ predictive
    samples = rng.normal(0.0, 1.0, size=(draws, n))            # predictive N(0,1)
    point = samples.mean(axis=0)
    s = hier.scoring(y_true, point, samples, seed=0)
    assert 0.86 <= s["coverage90"] <= 0.94
    assert 0.44 <= s["coverage50"] <= 0.56


def test_scoring_overconfident_predictive_undercovers() -> None:
    rng = np.random.default_rng(12)
    n, draws = 4000, 800
    y_true = rng.normal(0.0, 1.0, size=n)                      # true spread = 1
    samples = rng.normal(0.0, 0.3, size=(draws, n))            # claims spread 0.3
    point = samples.mean(axis=0)
    s = hier.scoring(y_true, point, samples, seed=0)
    assert s["coverage90"] < 0.80                             # over-confident
    assert s["nll"] > 1.0


def test_crps_rewards_accuracy() -> None:
    y_true = np.zeros(200)
    tight = np.random.default_rng(0).normal(0.0, 0.1, size=(300, 200))
    wide = np.random.default_rng(0).normal(0.0, 2.0, size=(300, 200))
    rng = np.random.default_rng(0)
    crps_tight = hier._crps_from_samples(y_true, tight, np.random.default_rng(1))
    crps_wide = hier._crps_from_samples(y_true, wide, np.random.default_rng(1))
    assert crps_tight < crps_wide


# --------------------------------------------------------------------------- #
# Cold predictive assembly
# --------------------------------------------------------------------------- #
def test_predict_hbm_cold_widens_by_between_user_variance() -> None:
    """For an unseen user the predictive SD must be sqrt(sigma^2 + tau^2) on the
    standardized scale, then rescaled by y_scale."""
    draws, p, n_test = 600, 2, 40
    post = {
        "intercept": np.zeros(draws),
        "w": np.zeros((draws, p)),          # zero mean function -> pure noise spread
        "sigma": np.full(draws, 0.4),
        "tau_user": np.full(draws, 0.3),
    }
    phi_test = np.random.default_rng(0).normal(size=(n_test, p))
    y_center, y_scale = 2.0, 5.0
    point, samples = hier.predict_hbm_cold(post, phi_test, y_center, y_scale, seed=0)
    assert point.shape == (n_test,)
    assert samples.shape == (draws, n_test)
    # point ~ y_center (mean function is zero)
    assert abs(point.mean() - y_center) < 0.3
    expected_sd = np.sqrt(0.4**2 + 0.3**2) * y_scale          # = 0.5 * 5 = 2.5
    assert abs(samples.std() - expected_sd) < 0.4


def test_predict_hbm_cold_without_tau() -> None:
    """Models with no groups have no tau_user; predictive uses sigma only."""
    draws, p, n_test = 400, 2, 30
    post = {
        "intercept": np.zeros(draws),
        "w": np.zeros((draws, p)),
        "sigma": np.full(draws, 0.5),
    }
    phi_test = np.random.default_rng(1).normal(size=(n_test, p))
    _, samples = hier.predict_hbm_cold(post, phi_test, 0.0, 1.0, seed=0)
    assert abs(samples.std() - 0.5) < 0.15


# --------------------------------------------------------------------------- #
# Hybrid: variance components + tree_hier decomposition
# --------------------------------------------------------------------------- #
def test_variance_components_recovers_known_split() -> None:
    """Residuals with known between-user (tau=0.8) and within-user (sigma=0.4)
    variance must be decomposed close to those values."""
    rng = np.random.default_rng(4)
    n_users, per_user = 60, 15
    groups = np.repeat(np.arange(n_users), per_user)
    user_effect = rng.normal(0.0, 0.8, size=n_users)[groups]
    within = rng.normal(0.0, 0.4, size=n_users * per_user)
    resid = user_effect + within
    tau2, sigma2, grand = hier.variance_components(resid, groups)
    assert abs(np.sqrt(tau2) - 0.8) < 0.15
    assert abs(np.sqrt(sigma2) - 0.4) < 0.08
    assert abs(grand) < 0.15


def test_variance_components_no_groups_is_all_within() -> None:
    resid = np.random.default_rng(0).normal(0.0, 1.0, size=500)
    tau2, sigma2, _ = hier.variance_components(resid, np.full(500, -1, dtype=object))
    assert tau2 == 0.0
    assert abs(np.sqrt(sigma2) - 1.0) < 0.1


def test_tree_hier_ties_and_reports_decomposition() -> None:
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(20), 12)
    X = rng.normal(size=(240, 3))
    y = np.sin(X[:, 0]) + rng.normal(0.0, 1.0, size=20)[groups] + rng.normal(0.0, 0.5, size=240)
    X_test = rng.normal(size=(40, 3))
    point, samples, decomp = hier.tree_hier_predictive(
        X, y, X_test, groups, seed=1, tree_scale=0.5, draws=200
    )
    assert point.shape == (40,)
    assert samples.shape == (200, 40)
    assert set(decomp) == {"tau", "sigma", "icc_resid"}
    assert 0.0 <= decomp["icc_resid"] <= 1.0
    assert decomp["tau"] > 0.0 and decomp["sigma"] > 0.0


def test_gauss_metrics_calibrated() -> None:
    rng = np.random.default_rng(5)
    n = 5000
    mu = np.zeros(n)
    sd = np.ones(n)
    y = rng.normal(0.0, 1.0, size=n)
    m = hier._gauss_metrics(y, mu, sd)
    assert 0.87 <= m["coverage90"] <= 0.93
    assert m["n"] == n
