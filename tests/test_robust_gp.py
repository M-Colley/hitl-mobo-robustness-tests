"""The outlier-robust surrogate for the misclick arm (scripts/robust_gp.py).

A misclick files a rating against the wrong design, so relative to the true
function at the recorded design it is a gross outlier. The robust model is only
worth wiring into the sweep if it actually resists such points and still works
with the acquisitions the sweep uses; these tests check both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import robust_gp  # noqa: E402


def _truth(X: torch.Tensor) -> torch.Tensor:
    return torch.sin(3.0 * X[:, 0]) + torch.cos(2.0 * X[:, 1])


def _data_with_outliers(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(44, 2, generator=g, dtype=torch.double)
    f = _truth(X)
    Y = f + 0.05 * torch.randn(44, generator=g, dtype=torch.double)
    sd = float(f.std())
    outliers = torch.tensor([3, 11, 25, 37])
    Y[outliers] += 5.0 * sd * torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.double)
    return X, Y.unsqueeze(-1), f, outliers


def test_robust_model_resists_planted_outliers():
    X, Y, f, outliers = _data_with_outliers()
    plain = robust_gp.build_plain_gp(X, Y)
    robust = robust_gp.build_robust_gp(X, Y)
    with torch.no_grad():
        err_plain = (plain.posterior(X[outliers]).mean.squeeze(-1) - f[outliers]).abs().mean()
        err_robust = (robust.posterior(X[outliers]).mean.squeeze(-1) - f[outliers]).abs().mean()
    # The Gaussian model bends through the outliers; the Student-t model should not.
    assert err_robust < 0.5 * err_plain, (float(err_robust), float(err_plain))


def test_posterior_is_finite_in_batch_shape():
    X, Y, _, _ = _data_with_outliers(1)
    model = robust_gp.build_robust_gp(X, Y)
    q = torch.rand(16, 2, dtype=torch.double)
    with torch.no_grad():
        post = model.posterior(q)
    assert post.mean.shape == (16, 1)
    assert torch.isfinite(post.mean).all() and (post.variance > 0).all()


def test_raw_bounds_match_the_unit_cube_fit():
    """With bounds, the model normalises internally: same fit as on unit-cube data.

    Clean data on purpose. With planted outliers the Student-t fit is
    multimodal (which points are the outliers is itself being inferred), so two
    floating-point paths can settle in different optima; that is a property of
    the likelihood, not of the transform. On clean data the fits agree to a few
    thousandths, while a missing transform puts the raw model's RMSE near 0.75.
    """
    g = torch.Generator().manual_seed(3)
    X = torch.rand(44, 2, generator=g, dtype=torch.double)
    Y = (_truth(X) + 0.05 * torch.randn(44, generator=g, dtype=torch.double)).unsqueeze(-1)
    low = torch.tensor([-5.0, 10.0], dtype=torch.double)
    high = torch.tensor([5.0, 30.0], dtype=torch.double)
    unit_model = robust_gp.build_robust_gp(X, Y)
    raw_model = robust_gp.build_robust_gp(low + X * (high - low), Y, bounds=torch.stack([low, high]))
    q = torch.rand(50, 2, generator=g, dtype=torch.double)
    with torch.no_grad():
        a = unit_model.posterior(q).mean
        b = raw_model.posterior(low + q * (high - low)).mean
        unnormalised = raw_model.posterior(q).mean
    assert (a - b).abs().max() < 0.02, (a - b).abs().max()
    assert (a - unnormalised).abs().max() > 0.2


@pytest.mark.parametrize("which", ["ucb", "logei", "qnei"])
def test_acquisitions_evaluate_on_the_robust_model(which):
    from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

    X, Y, _, _ = _data_with_outliers(2)
    model = robust_gp.build_robust_gp(X, Y)
    if which == "ucb":
        acq = UpperConfidenceBound(model, beta=2.0)
    elif which == "logei":
        acq = LogExpectedImprovement(model, best_f=float(Y.max()))
    else:
        acq = qNoisyExpectedImprovement(model, X_baseline=X, prune_baseline=True)
    with torch.no_grad():
        value = acq(torch.rand(5, 1, 2, dtype=torch.double))
    assert value.shape == (5,)
    assert torch.isfinite(value).all()
