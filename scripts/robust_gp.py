"""Outlier-robust GP surrogates for BoTorch acquisitions (misclick arm).

Motivation
----------
In the misclick arm a fraction of ratings are attached to the wrong design:
the observation belongs to a uniformly random design, so relative to the true
function at the *recorded* design it is a gross outlier with no relation to
the local noise level. A ``SingleTaskGP`` with a Gaussian likelihood has to
explain such a point either by inflating the global noise (which blurs every
clean point) or by bending the posterior mean through it (which corrupts the
neighbourhood). Both hurt the acquisition. A heavy-tailed likelihood lets the
model treat the point as "probably wrong" without any explicit detection step.

What is delivered
-----------------
``build_robust_gp`` -- the PREFERRED implementation from the task brief:

* ``gpytorch.likelihoods.StudentTLikelihood`` (learned degrees of freedom
  ``nu > 2`` and scale ``sigma^2``), so the marginal likelihood of a point far
  from the latent function decays polynomially, not as ``exp(-r^2)``;
* ``botorch.models.approximate_gp.SingleTaskVariationalGP`` with the inducing
  points placed AT the training inputs (all of them, capped at
  ``max_inducing = 64`` via a greedy-variance-reduction subset beyond that) and
  ``learn_inducing_points=False``.  With ``M = n`` inducing points sitting on
  the data, the sparse variational GP is exactly the full variational GP for a
  non-Gaussian likelihood -- there is no sparsity approximation at the sizes
  used in this study (``n <= ~100``);
* Matern-5/2 ARD kernel with BoTorch's dimension-scaled log-normal
  lengthscale prior (``get_covar_module_with_dim_scaled_prior``), i.e. the same
  kernel family and prior as the ``SingleTaskGP`` used in
  ``bo_sensor_error_simulation.py``;
* ``Standardize(m=1)`` outcome transform, supported natively by
  ``SingleTaskVariationalGP`` (it standardises ``train_Y`` once at construction
  and un-transforms the posterior), so outputs are standardised the same way
  as in the clean arm.  BoTorch emits a ``UserInputWarning`` because the
  transform would misbehave under minibatching; we fit full-batch so it is
  correct here, and the warning is silenced;
* fitted with ``botorch.fit.fit_gpytorch_mll`` on a
  ``gpytorch.mlls.VariationalELBO`` (full-batch L-BFGS-B via BoTorch's
  approximate-MLL fallback; ``expected_log_prob`` of the Student-t likelihood
  is computed by Gauss-Hermite quadrature inside GPyTorch).

The returned object is a BoTorch ``Model`` whose ``posterior(X)`` returns a
``GPyTorchPosterior`` (latent-function posterior; ``observation_noise=False``
is the default, which is what BoTorch's acquisitions call), so
``UpperConfidenceBound``, ``LogExpectedImprovement`` and
``qNoisyExpectedImprovement`` work unchanged.

Why the variational path and not the reweighting fallback
--------------------------------------------------------
Measured on this machine (see ``report_fit_times`` / ``python
scripts/robust_gp.py --bench``; wall-clock per fit, double precision, CPU):
the Student-t VGP fit is well inside the ``~8x SingleTaskGP`` budget set in
the brief at both ``n = 20`` and ``n = 50``, ``d = 4`` (numbers in the module
docstring of the tests and in the task report).  The two-pass Huber
reweighting (``build_reweighted_gp``) is ALSO provided as a secondary option
because it is cheaper and stays entirely inside the exact-GP machinery, but it
is a heuristic detector (a point is down-weighted only if the FIRST fit already
finds it surprising, which fails when a cluster of outliers drags the first
fit towards itself) and is not the delivered default.

Usage
-----
    from robust_gp import build_robust_gp
    model = build_robust_gp(train_X, train_Y)          # X in unit cube, Y n x 1
    acq = UpperConfidenceBound(model, beta=2.0)
    candidate, _ = optimize_acqf(acq, bounds=unit_bounds, q=1, ...)
"""
from __future__ import annotations

import argparse
import contextlib
import time
import warnings
from typing import Iterable

import torch
from botorch.exceptions.warnings import UserInputWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
)
from botorch.models.utils.inducing_point_allocators import GreedyVarianceReduction
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.likelihoods import StudentTLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.priors import GammaPrior, LogNormalPrior
from torch import Tensor

__all__ = [
    "build_robust_gp",
    "build_reweighted_gp",
    "build_plain_gp",
    "report_fit_times",
]

DEFAULT_MAX_INDUCING = 64
# L-BFGS-B iteration cap for the ELBO.  Uncapped, BoTorch's scipy fallback runs
# ~3000 iterations (~30 s at n = 44) chasing the last 1e-6 of ELBO; profiling on
# this machine showed the outlier-location error is 0.016 SD after 100
# iterations vs 0.007 SD at convergence (plain GP: 0.28 SD), so the cap costs
# essentially nothing in accuracy and keeps the fit at ~5-6x SingleTaskGP.
DEFAULT_MAXITER = 100
# Degrees of freedom: > 2 keeps the variance finite (needed for the Gauss-
# Hermite expected log-prob to be well-behaved); the upper bound stops the fit
# from drifting to nu -> inf, where the Student-t collapses to a Gaussian and
# the robustness is lost on data sets too small to pin nu down.
DEG_FREE_BOUNDS = (2.01, 30.0)
DEG_FREE_INIT = 4.0
# Noise (scale^2 of the Student-t) in standardised units.  Same shape of prior
# as BoTorch's default GaussianLikelihood (LogNormal(-4, 1)) so the clean-arm
# and robust-arm noise beliefs are comparable; the lower bound matches
# MIN_INFERRED_NOISE_LEVEL.
NOISE_LOWER = 1e-4
NOISE_PRIOR = (-4.0, 1.0)


def _as_double_2d(train_X: Tensor, train_Y: Tensor) -> tuple[Tensor, Tensor]:
    """Validate shapes and cast to double (BoTorch's recommended dtype)."""
    if train_X.dim() != 2:
        raise ValueError(f"train_X must be n x d, got shape {tuple(train_X.shape)}")
    if train_Y.dim() == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.dim() != 2 or train_Y.shape[-1] != 1:
        raise ValueError(f"train_Y must be n x 1, got shape {tuple(train_Y.shape)}")
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError(
            f"train_X has {train_X.shape[0]} rows but train_Y has {train_Y.shape[0]}"
        )
    if train_X.shape[0] < 2:
        raise ValueError("need at least two observations to fit a GP")
    return train_X.to(dtype=torch.double), train_Y.to(dtype=torch.double)


def _student_t_likelihood() -> StudentTLikelihood:
    lo, hi = DEG_FREE_BOUNDS
    noise_prior = LogNormalPrior(loc=NOISE_PRIOR[0], scale=NOISE_PRIOR[1])
    lik = StudentTLikelihood(
        deg_free_constraint=Interval(lo, hi),
        # Weak prior pulling nu towards moderately heavy tails; Gamma(2, 0.2)
        # has mode 5 and puts ~5% mass above 24.
        deg_free_prior=GammaPrior(concentration=2.0, rate=0.2),
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(NOISE_LOWER),
    )
    lik.deg_free = torch.tensor(DEG_FREE_INIT, dtype=torch.double)
    # Start the scale at the prior mode (exp(-4 - 1) ~ 0.0067 in standardised
    # units): a small initial noise makes the first ELBO evaluations treat
    # far-away points as tail events rather than absorbing them into sigma.
    lik.noise = torch.tensor(float(noise_prior.mode), dtype=torch.double)
    return lik


def _select_inducing_points(
    train_X: Tensor, covar_module, max_inducing: int
) -> Tensor:
    """All training inputs as inducing points, or a greedy-variance subset."""
    n = train_X.shape[0]
    if n <= max_inducing:
        return train_X.clone()
    allocator = GreedyVarianceReduction()
    return allocator.allocate_inducing_points(
        inputs=train_X,
        covar_module=covar_module,
        num_inducing=max_inducing,
        input_batch_shape=torch.Size([]),
    )


def _warm_start_variational(model: SingleTaskVariationalGP) -> None:
    """Initialise the whitened variational distribution at the data.

    With the inducing points sitting ON the training inputs the natural
    starting point is ``q(u) = N(y_std, small)``.  GPyTorch's whitened
    strategy parametrises ``u = L v`` with ``K_uu = L L^T`` and ``q(v)``, so
    the whitened mean is ``L^{-1} (y_std - const_mean)``.  Starting there
    instead of at the prior ``N(0, I)`` saves the optimiser the first ~50
    L-BFGS-B iterations it would otherwise spend pulling the mean onto the
    data.  Only applied when every training point is an inducing point.
    """
    inner = model.model
    vs = inner.variational_strategy
    Z = vs.inducing_points
    y = inner.train_targets
    if Z.shape[-2] != y.shape[-1]:
        return  # subset of inducing points: leave GPyTorch's prior init
    with torch.no_grad():
        K = inner.covar_module(Z).add_jitter(1e-4).to_dense()
        L = torch.linalg.cholesky(K)
        resid = (y - inner.mean_module.constant).unsqueeze(-1)
        m_w = torch.linalg.solve_triangular(L, resid, upper=False).squeeze(-1)
        vd = vs._variational_distribution
        vd.variational_mean.copy_(m_w)
        # Posterior at observed inducing points is tighter than the whitened
        # prior N(0, I); start moderately tight rather than at identity.
        vd.chol_variational_covar.copy_(0.3 * torch.eye(Z.shape[-2], dtype=y.dtype))
        vs.variational_params_initialized.fill_(1)


def build_robust_gp(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    bounds: Tensor | None = None,
    max_inducing: int = DEFAULT_MAX_INDUCING,
    maxiter: int = DEFAULT_MAXITER,
    warm_start: bool = True,
    fit: bool = True,
    optimizer_kwargs: dict | None = None,
) -> SingleTaskVariationalGP:
    """Fit a Student-t-likelihood variational GP on ``(train_X, train_Y)``.

    Args:
        train_X: ``n x d`` inputs. In the unit cube when ``bounds`` is None;
            otherwise on the raw scale of ``bounds``.
        bounds: optional ``2 x d`` box. When given, a ``Normalize`` input
            transform maps inputs to the unit cube inside the model -- as the
            sweep's ``SingleTaskGP`` does -- so callers and acquisitions keep
            working on the raw scale; the inducing points are placed at the
            normalised training inputs.
        train_Y: ``n x 1`` observed objective values (raw scale; a
            ``Standardize`` outcome transform is fitted internally).
        max_inducing: cap on the number of inducing points.  Up to this many
            training inputs are used verbatim (no approximation); beyond it a
            greedy-variance-reduction subset of that size is used.
        maxiter: L-BFGS-B iteration cap for the ELBO (see ``DEFAULT_MAXITER``).
            Ignored if ``optimizer_kwargs`` carries its own ``options``.
        warm_start: initialise the variational distribution at the standardised
            targets (``_warm_start_variational``) instead of the prior.
        fit: fit the ELBO (``True``) or return the untrained model.
        optimizer_kwargs: forwarded to ``fit_gpytorch_mll`` (e.g.
            ``{"options": {"maxiter": 200}}``).

    Returns:
        A fitted ``SingleTaskVariationalGP`` (a BoTorch ``Model``) in eval mode
        whose ``posterior(X)`` returns the latent-function posterior on the
        original output scale.
    """
    train_X, train_Y = _as_double_2d(train_X, train_Y)
    n, d = train_X.shape

    covar_module = get_covar_module_with_dim_scaled_prior(
        ard_num_dims=d, use_rbf_kernel=False  # Matern-5/2 with ARD
    ).to(train_X)
    input_transform = None
    unit_X = train_X
    if bounds is not None:
        bounds = bounds.to(train_X)
        input_transform = Normalize(d=d, bounds=bounds)
        unit_X = (train_X - bounds[0]) / (bounds[1] - bounds[0])
    inducing = _select_inducing_points(unit_X, covar_module, max_inducing)

    with warnings.catch_warnings():
        # The transform warning is about minibatch training; we fit full-batch.
        warnings.simplefilter("ignore", category=UserInputWarning)
        model = SingleTaskVariationalGP(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=_student_t_likelihood(),
            learn_inducing_points=False,
            covar_module=covar_module,
            inducing_points=inducing,
            outcome_transform=Standardize(m=1),
            input_transform=input_transform,
        )
    model = model.to(train_X)
    if warm_start:
        _warm_start_variational(model)

    if fit:
        kwargs = dict(optimizer_kwargs or {})
        kwargs.setdefault("options", {"maxiter": int(maxiter)})
        mll = VariationalELBO(model.likelihood, model.model, num_data=n)
        fit_gpytorch_mll(mll, optimizer_kwargs=kwargs)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Secondary option: two-pass Huber-style reweighting of an exact GP
# --------------------------------------------------------------------------

def build_plain_gp(train_X: Tensor, train_Y: Tensor, *, fit: bool = True) -> SingleTaskGP:
    """The non-robust reference: ``SingleTaskGP`` + ``Standardize`` (as in the sweep)."""
    train_X, train_Y = _as_double_2d(train_X, train_Y)
    gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    if fit:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    return gp


MAD_TO_SD = 1.4826  # consistency factor: MAD -> SD for Gaussian residuals


def loo_residuals(gp: SingleTaskGP) -> tuple[Tensor, Tensor]:
    """Leave-one-out residuals of a fitted exact GP, in standardised units.

    Closed form (Rasmussen & Williams, sec. 5.4.2): with ``K_y = K + sigma^2 I``,
    ``P = K_y^{-1}`` and ``alpha = P (y - m)``, the LOO predictive mean and
    variance at point ``i`` are ``y_i - alpha_i / P_ii`` and ``1 / P_ii``, so the
    standardised LOO residual is ``alpha_i / sqrt(P_ii)``.  "Leave-in"
    residuals (posterior conditioned on the point itself) let an outlier
    explain itself away and never reach the threshold -- checked empirically:
    four 5-SD outliers among 40 clean points scored |r| < 3 with leave-in
    residuals, so LOO is used instead.

    Returns ``(r, raw_scale)`` where ``r`` is ``n``-vector of standardised LOO
    residuals and ``raw_scale`` the outcome transform's stdv (to convert
    standardised residual variances back to raw units).
    """
    gp.train()
    try:
        with torch.no_grad():
            (X_t,) = gp.train_inputs
            y_t = gp.train_targets
            prior = gp.forward(X_t)
            K_y = gp.likelihood(prior).covariance_matrix
            P = torch.linalg.inv(K_y)
            alpha = P @ (y_t - prior.mean)
            r = alpha / P.diagonal().clamp_min(1e-12).sqrt()
    finally:
        gp.eval()
    stdv = gp.outcome_transform.stdvs.reshape(-1)[0]
    return r, stdv


def build_reweighted_gp(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    threshold: float = 3.0,
    fit: bool = True,
) -> SingleTaskGP:
    """Two-pass Huber-style robustification of ``SingleTaskGP`` (secondary option).

    Pass 1 fits a plain ``SingleTaskGP``.  Its standardised leave-one-out
    residuals ``r`` (``loo_residuals``) are rescaled by their own MAD
    (``r' = r / (1.4826 MAD(r))``): when the first fit has inflated the noise
    to swallow outliers, every inlier residual shrinks and the outliers would
    otherwise slip under ``threshold``; the MAD rescaling undoes that
    (for a clean, calibrated fit MAD(r) ~ 1 and it is a no-op).  Pass 2 refits
    a fixed-noise ``SingleTaskGP`` with per-point ``train_Yvar``:

        var_i = base_var * max(1, (|r'_i| / threshold)^2)

    -- the Huber weight ``w = threshold / |r'|`` applied as a precision -- where
    ``base_var`` is the robust (MAD-based) residual variance in raw units, i.e.
    the noise an inlier would have if the outliers were absent.  Same kernel
    and transforms as the clean arm.

    NOT the delivered default (see module docstring): it is a detector, so a
    cluster of outliers that drags the pass-1 fit towards itself can survive.
    Kept because it is ~10x cheaper than the variational model and useful as
    an ablation.
    """
    train_X, train_Y = _as_double_2d(train_X, train_Y)
    first = build_plain_gp(train_X, train_Y, fit=fit)
    r, stdv = loo_residuals(first)
    with torch.no_grad():
        mad = (r - r.median()).abs().median() * MAD_TO_SD
        scale = mad.clamp_min(1e-6)
        r_robust = (r / scale).abs()
        # LOO predictive SD in standardised units for an inlier, robustly:
        # the residual scale in standardised units is `scale`; convert to raw.
        first.train()
        try:
            (X_t,) = first.train_inputs
            K_y = first.likelihood(first.forward(X_t)).covariance_matrix
            loo_var_std = 1.0 / torch.linalg.inv(K_y).diagonal().clamp_min(1e-12)
        finally:
            first.eval()
        base_var = (loo_var_std * scale**2 * stdv**2).clamp_min(1e-8)
        inflate = torch.where(
            r_robust > threshold, (r_robust / threshold) ** 2, torch.ones_like(r_robust)
        )
        train_Yvar = (base_var * inflate).reshape(-1, 1)
    gp = SingleTaskGP(
        train_X, train_Y, train_Yvar=train_Yvar, outcome_transform=Standardize(m=1)
    )
    if fit:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    return gp


# --------------------------------------------------------------------------
# Timing report
# --------------------------------------------------------------------------

def _synthetic_problem(n: int, d: int, seed: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, d, generator=g, dtype=torch.double)
    # Smooth, anisotropic test function with mild noise.
    f = torch.sin(3.0 * X[:, 0]) + 0.5 * torch.cos(2.0 * X[:, 1:].sum(-1)) - (X - 0.5).pow(2).sum(-1)
    Y = f + 0.05 * torch.randn(n, generator=g, dtype=torch.double)
    return X, Y.unsqueeze(-1)


def report_fit_times(
    ns: Iterable[int] = (20, 50),
    d: int = 4,
    repeats: int = 3,
    seed: int = 0,
) -> dict[tuple[str, int], float]:
    """Median wall-clock seconds per fit for each builder at each ``n``."""
    builders = {
        "SingleTaskGP": build_plain_gp,
        "robust_student_t_vgp": build_robust_gp,
        "reweighted_gp": build_reweighted_gp,
    }
    out: dict[tuple[str, int], float] = {}
    for n in ns:
        for name, builder in builders.items():
            times = []
            for k in range(repeats):
                X, Y = _synthetic_problem(n, d, seed + k)
                t0 = time.perf_counter()
                builder(X, Y)
                times.append(time.perf_counter() - t0)
            out[(name, n)] = float(sorted(times)[len(times) // 2])
    return out


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--bench", action="store_true", help="print per-fit timings")
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args(argv)
    if args.bench:
        res = report_fit_times(d=args.d, repeats=args.repeats)
        base = {n: res[("SingleTaskGP", n)] for (name, n) in res if name == "SingleTaskGP"}
        print(f"fit time per call, d={args.d}, median of {args.repeats} (seconds)")
        for (name, n), t in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            print(f"  n={n:3d}  {name:22s} {t:8.3f}s  ({t / base[n]:5.1f}x SingleTaskGP)")


if __name__ == "__main__":
    _main()
