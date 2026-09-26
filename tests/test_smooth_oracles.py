"""The two smooth oracle families of the oracle-isolation robustness arm.

The pipeline's model selection picks tree ensembles, which are piecewise
constant; the Gaussian-process and MLP oracles ask whether the isolation result
depends on that. These tests pin that both families are registered after the
existing ones, fit and predict on the original scale, are smooth where a tree
ensemble is not, and that oracle_isolation's --family switch sends a run to its
own directory without augmentation while reading the shared datasets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bo_sensor_error_simulation as sim
import oracle_isolation as iso


def _data(n=120, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, d))
    y = 5.0 + 3.0 * np.sin(X[:, 0]) + (X[:, 1] ** 2 if d > 1 else 0.0) + rng.normal(0.0, 0.1, size=n)
    return X, y


def test_the_families_are_appended_after_the_existing_choices():
    assert sim.ORACLE_MODEL_CHOICES[-2:] == ["gaussian_process", "mlp"]
    assert sim.ORACLE_MODEL_CHOICES[:8] == ["xgboost", "lightgbm", "catboost", "tabpfn", "random_forest",
                                            "extra_trees", "gradient_boosting", "hist_gradient_boosting"]


@pytest.mark.parametrize("family", ["gaussian_process", "mlp"])
def test_each_family_fits_and_predicts_on_the_original_scale(family):
    X, y = _data()
    model = sim._build_oracle_model(family, seed=7, tree_scale=1.0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    # the target is standardised inside and restored on the way out
    assert abs(pred.mean() - y.mean()) < 0.5
    assert np.corrcoef(pred, y)[0, 1] > 0.9
    assert model.score(X, y) > 0.8


def test_the_gp_oracle_is_smooth_where_a_tree_ensemble_steps():
    X, y = _data(n=200, d=1)
    grid = np.linspace(-1.5, 1.5, 400).reshape(-1, 1)
    gp = sim._build_oracle_model("gaussian_process", seed=7, tree_scale=1.0).fit(X, y)
    trees = sim._build_oracle_model("extra_trees", seed=7, tree_scale=1.0).fit(X, y)
    # a tree ensemble repeats values across neighbouring grid points; a GP does not
    assert len(np.unique(np.round(gp.predict(grid), 9))) == len(grid)
    assert np.max(np.abs(np.diff(gp.predict(grid)))) < np.max(np.abs(np.diff(trees.predict(grid))))


def test_the_smooth_oracle_is_deterministic_given_the_seed():
    X, y = _data()
    a = sim._build_oracle_model("mlp", seed=3, tree_scale=1.0).fit(X, y).predict(X)
    b = sim._build_oracle_model("mlp", seed=3, tree_scale=1.0).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b)


def test_family_switch_uses_its_own_root_and_no_augmentation(monkeypatch):
    calls = {}
    # main() sets these module globals; registering them first restores them afterwards
    for name in ("ROOT", "FAMILY", "AUGMENTATION"):
        monkeypatch.setattr(iso, name, getattr(iso, name))
    monkeypatch.setattr(iso, "select", lambda args: calls.setdefault("select", (iso.ROOT, iso.FAMILY, iso.AUGMENTATION)))
    iso.main(["select", "--family", "mlp"])
    root, family, augmentation = calls["select"]
    assert root == Path("output-oracle-iso-mlp")
    assert family == "mlp" and augmentation == "none"
    # the synthetic datasets and their configs are shared with the selected-oracle arm
    assert iso.DATA_ROOT == Path("output-oracle-iso")


def test_without_the_switch_the_selected_oracle_arm_is_unchanged(monkeypatch):
    calls = {}
    monkeypatch.setattr(iso, "analyse", lambda args: calls.setdefault("analyse", (iso.ROOT, iso.FAMILY, iso.AUGMENTATION)))
    iso.main(["analyse"])
    assert calls["analyse"] == (Path("output-oracle-iso"), None, "jitter")
