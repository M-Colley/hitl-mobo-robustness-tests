"""Additional oracle test: a NumPyro hierarchical Bayesian "simulated human".

Motivation
----------
The deployed oracle (``select_best_oracle_model.py`` -> extra_trees /
gradient_boosting) is a *forward point regressor* design-parameters -> rating.
Its documented ceiling is between-user heterogeneity: for ehmi ~63% of composite
variance is per-user intercepts that no pooled point model can predict for an
unseen user (docs/oracle-fidelity-experiments-2026-06-12.md). A tree also emits
no predictive distribution, so the simulator has to bolt noise on afterwards.

This script fits a hierarchical Bayesian model that represents that structure
natively:

    y_ij = f(x_ij) + alpha_user[j] + eps,   alpha_user ~ N(0, tau_user),
                                             eps        ~ N(0, sigma)

with f either linear (``--mean linear``) or a random-Fourier-feature map that
approximates an RBF-kernel GP (``--mean rff``, nonlinear, competitive with the
trees on point fidelity). For a *held-out* user the model does not know their
intercept, so the cold predictive correctly widens to sqrt(sigma^2 + tau_user^2)
-- the honest "we have never met this person" uncertainty.

Every model is scored on the SAME cold protocol as the fidelity experiments
(individual rows, GroupKFold by inferred user/group, 5 folds), and the
extra_trees baseline is recomputed on the identical folds so the comparison is
apples-to-apples. Trees get a courtesy plug-in Gaussian N(pred, train-residual
sd) so calibration metrics (coverage / NLL / CRPS) are defined for them too --
they will be over-confident precisely because they cannot see tau_user.

This is a *diagnostic add-on*; it does not change the deployed oracle or any
sweep output. Results land in ``output/hierarchical_oracle_test.json`` plus a
printed table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bo_sensor_error_simulation import (  # noqa: E402
    compute_objective,
    fit_objective_normalization,
    infer_oracle_groups,
    load_observations,
    parse_dataset_configs,
    parse_objective_list,
    parse_objective_weights,
)

numpyro.set_platform("cpu")


# --------------------------------------------------------------------------- #
# Feature maps
# --------------------------------------------------------------------------- #
def _standardize(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    out = [(train - mean) / std]
    out.extend([(o - mean) / std for o in others])
    return out


def _median_lengthscale(x_std: np.ndarray, rng: np.random.Generator) -> float:
    """Median pairwise Euclidean distance (subsampled) -- RBF lengthscale heuristic."""
    n = x_std.shape[0]
    if n > 400:
        idx = rng.choice(n, size=400, replace=False)
        x_std = x_std[idx]
    diffs = x_std[:, None, :] - x_std[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=-1))
    iu = np.triu_indices_from(d, k=1)
    med = float(np.median(d[iu])) if iu[0].size else 1.0
    return med if med > 1e-6 else 1.0


def build_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mean: str,
    n_features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (phi_train, phi_test, info). X standardized on train statistics."""
    Xtr, Xte = _standardize(X_train, X_test)
    if mean == "linear":
        return Xtr, Xte, {"n_features": Xtr.shape[1], "lengthscale": None}
    if mean == "rff":
        d = Xtr.shape[1]
        ell = _median_lengthscale(Xtr, rng)
        omega = rng.normal(size=(d, n_features)) / ell
        b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
        scale = np.sqrt(2.0 / n_features)
        phi_tr = scale * np.cos(Xtr @ omega + b)
        phi_te = scale * np.cos(Xte @ omega + b)
        return phi_tr, phi_te, {"n_features": n_features, "lengthscale": ell}
    raise ValueError(f"Unknown mean function: {mean}")


# --------------------------------------------------------------------------- #
# Hierarchical model
# --------------------------------------------------------------------------- #
def _hier_model(phi, groups, n_groups, y=None):
    """y standardized. Non-centered per-group intercept + linear map in phi-space."""
    p = phi.shape[1]
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.0))
    w_scale = numpyro.sample("w_scale", dist.HalfNormal(1.0))
    with numpyro.plate("coefs", p):
        w = numpyro.sample("w", dist.Normal(0.0, w_scale))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    mu = intercept + phi @ w
    if n_groups > 0:
        tau_user = numpyro.sample("tau_user", dist.HalfNormal(1.0))
        with numpyro.plate("groups", n_groups):
            z_user = numpyro.sample("z_user", dist.Normal(0.0, 1.0))
        mu = mu + (tau_user * z_user)[groups]
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


def _intercept_only_model(groups, n_groups, y=None):
    """Variance-components model: grand mean + user random intercept + residual."""
    grand = numpyro.sample("intercept", dist.Normal(0.0, 1.0))
    tau_user = numpyro.sample("tau_user", dist.HalfNormal(1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    with numpyro.plate("groups", n_groups):
        z_user = numpyro.sample("z_user", dist.Normal(0.0, 1.0))
    mu = grand + (tau_user * z_user)[groups]
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


def _run_mcmc(model, rng_key, num_warmup, num_samples, chains, **kwargs) -> dict:
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    mcmc.run(rng_key, **kwargs)
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


def _run_svi(model, rng_key, steps, draws, **kwargs) -> dict:
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(5e-3), Trace_ELBO())
    result = svi.run(rng_key, steps, progress_bar=False, **kwargs)
    post = guide.sample_posterior(
        jax.random.fold_in(rng_key, 1), result.params, sample_shape=(draws,)
    )
    return {k: np.asarray(v) for k, v in post.items()}


def fit_hbm(
    phi_train, groups_train, n_groups, y_train_std, *, seed, infer, mcmc_cfg
):
    key = jax.random.PRNGKey(seed)
    model = _hier_model
    kwargs = dict(
        phi=jnp.asarray(phi_train),
        groups=jnp.asarray(groups_train),
        n_groups=int(n_groups),
        y=jnp.asarray(y_train_std),
    )
    if infer == "nuts":
        return _run_mcmc(model, key, mcmc_cfg["warmup"], mcmc_cfg["samples"],
                         mcmc_cfg["chains"], **kwargs)
    return _run_svi(model, key, mcmc_cfg["svi_steps"], mcmc_cfg["svi_draws"], **kwargs)


def predict_hbm_cold(post: dict, phi_test, y_center, y_scale, seed) -> tuple[np.ndarray, np.ndarray]:
    """Cold prediction for UNSEEN users.

    Point = E[mu] over posterior draws. Predictive samples marginalize the
    unknown new-user intercept: y* = mu + N(0, sqrt(sigma^2 + tau_user^2)),
    all mapped back to the raw objective scale.
    """
    intercept = post["intercept"][:, None]              # (S,1)
    w = post["w"]                                        # (S,P)
    sigma = post["sigma"][:, None]                       # (S,1)
    tau = post.get("tau_user")
    tau = (tau[:, None] if tau is not None else np.zeros_like(sigma))
    mu_std = intercept + w @ phi_test.T                  # (S, N_test)
    total_sd = np.sqrt(sigma ** 2 + tau ** 2)            # (S,1)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(mu_std.shape) * total_sd   # (S, N_test)
    samples_std = mu_std + eps
    point = mu_std.mean(axis=0) * y_scale + y_center
    samples = samples_std * y_scale + y_center           # (S, N_test)
    return point, samples


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _crps_from_samples(y_true: np.ndarray, samples: np.ndarray, rng, cap=200) -> float:
    """Sample CRPS = E|Y-y| - 0.5 E|Y-Y'|, averaged over test points."""
    s = samples
    if s.shape[0] > cap:
        idx = rng.choice(s.shape[0], size=cap, replace=False)
        s = s[idx]
    term1 = np.abs(s - y_true[None, :]).mean(axis=0)
    diff = np.abs(s[:, None, :] - s[None, :, :]).mean(axis=(0, 1))
    return float((term1 - 0.5 * diff).mean())


def scoring(y_true, point, samples, seed) -> dict:
    rng = np.random.default_rng(seed)
    mean = samples.mean(axis=0)
    sd = samples.std(axis=0)
    sd = np.where(sd < 1e-9, 1e-9, sd)
    lo90, hi90 = np.percentile(samples, [5.0, 95.0], axis=0)
    lo50, hi50 = np.percentile(samples, [25.0, 75.0], axis=0)
    nll = 0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * ((y_true - mean) / sd) ** 2
    return {
        "r2": float(r2_score(y_true, point)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, point))),
        "coverage50": float(np.mean((y_true >= lo50) & (y_true <= hi50))),
        "coverage90": float(np.mean((y_true >= lo90) & (y_true <= hi90))),
        "width90": float(np.mean(hi90 - lo90)),
        "nll": float(np.mean(nll)),
        "crps": _crps_from_samples(y_true, samples, rng),
    }


def _fit_tree(X, y, seed, tree_scale):
    return ExtraTreesRegressor(
        n_estimators=int(600 * tree_scale), min_samples_leaf=2,
        random_state=seed, n_jobs=1,
    ).fit(X, y)


def _tree_oof_residuals(X, y, groups, seed, tree_scale):
    """Out-of-fold residuals from an INNER GroupKFold on the training fold (never
    touches the outer test fold), with each residual's group label.

    A tree fits its own training data almost perfectly, so the naive train
    residual SD is ~0 and would make a plug-in Gaussian absurdly narrow. These
    inner *cold* residuals estimate the tree's true unseen-user error, which is
    what a "noisy simulated human" built from the tree should inject. They carry
    both within-rater noise and between-user heterogeneity (the outer test users
    are unseen)."""
    resid: list[np.ndarray] = []
    gout: list[np.ndarray] = []
    if groups is not None and np.unique(groups).size >= 3:
        splitter = GroupKFold(n_splits=min(3, int(np.unique(groups).size)))
        inner = splitter.split(X, groups=groups)
    else:
        inner = KFold(n_splits=min(3, len(X)), shuffle=True, random_state=seed).split(X)
    for tr, va in inner:
        m = _fit_tree(X[tr], y[tr], seed, tree_scale)
        resid.append(y[va] - m.predict(X[va]))
        gout.append(groups[va] if groups is not None else np.full(len(va), -1, object))
    return np.concatenate(resid), np.concatenate(gout)


def _tree_cold_sigma(X, y, groups, seed, tree_scale) -> float:
    resid, _ = _tree_oof_residuals(X, y, groups, seed, tree_scale)
    sd = float(np.std(resid)) if resid.size else float(np.std(y))
    return max(sd, 1e-6)


def variance_components(resid: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """Closed-form ANOVA random-effects split of residuals into between-group
    (tau^2) and within-group (sigma^2) variance, plus the residual grand mean.

    tau^2 = Var(group means) - sigma^2 / nbar (unbiased, floored at 0);
    sigma^2 = mean within-group variance. No MCMC -- fast and interpretable,
    which is the whole appeal of the tree+random-effects hybrid."""
    grand = float(np.mean(resid)) if resid.size else 0.0
    if groups is None:
        return 0.0, max(float(np.var(resid)), 1e-9), grand
    uniq = [g for g in pd.unique(groups) if g != -1]
    if len(uniq) < 2:
        return 0.0, max(float(np.var(resid)), 1e-9), grand
    within, means, ns = [], [], []
    for g in uniq:
        r = resid[groups == g]
        means.append(float(np.mean(r)))
        ns.append(len(r))
        if len(r) >= 2:
            within.append(float(np.var(r, ddof=1)))
    sigma2 = float(np.mean(within)) if within else max(float(np.var(resid)), 1e-9)
    nbar = float(np.mean(ns))
    tau2 = max(float(np.var(np.asarray(means), ddof=1)) - sigma2 / nbar, 0.0)
    return tau2, max(sigma2, 1e-9), grand


def extra_trees_predictive(X_train, y_train, X_test, groups_train, seed, tree_scale, draws):
    """Deployed extra_trees + honest plug-in Gaussian N(pred, cold-OOF residual sd)."""
    model = _fit_tree(X_train, y_train, seed, tree_scale)
    resid_sd = _tree_cold_sigma(X_train, y_train, groups_train, seed, tree_scale)
    point = np.asarray(model.predict(X_test), dtype=float)
    rng = np.random.default_rng(seed)
    samples = point[None, :] + rng.standard_normal((draws, point.size)) * resid_sd
    return point, samples


def tree_hier_predictive(X_train, y_train, X_test, groups_train, seed, tree_scale, draws):
    """Hybrid oracle: ExtraTrees mean + classical random-effects noise model.

    The tree supplies the (non-smooth) design->rating mean the HBM cannot match;
    the random-effects layer decomposes the honest cold residual into between-user
    tau and within-user sigma. Cold predictive = tree_mean + N(0, sqrt(tau^2 +
    sigma^2)) -- by construction this MATCHES the honest-noise tree on cold
    metrics; its added value is the decomposition (reported) and warm per-user
    updating (see warm_evaluation). Returns (point, samples, decomposition)."""
    model = _fit_tree(X_train, y_train, seed, tree_scale)
    resid, g_oof = _tree_oof_residuals(X_train, y_train, groups_train, seed, tree_scale)
    tau2, sigma2, grand = variance_components(resid, g_oof)
    total_sd = float(np.sqrt(tau2 + sigma2))
    point = np.asarray(model.predict(X_test), dtype=float) + grand
    rng = np.random.default_rng(seed)
    samples = point[None, :] + rng.standard_normal((draws, point.size)) * total_sd
    denom = tau2 + sigma2
    decomp = {"tau": float(np.sqrt(tau2)), "sigma": float(np.sqrt(sigma2)),
              "icc_resid": float(tau2 / denom) if denom > 0 else 0.0}
    return point, samples, decomp


# --------------------------------------------------------------------------- #
# Variance decomposition (native "% between-user" estimate)
# --------------------------------------------------------------------------- #
def variance_decomposition(y_std, groups, n_groups, seed, mcmc_cfg) -> dict:
    if n_groups < 2:
        return {"icc_between_user": None, "note": "fewer than 2 groups"}
    key = jax.random.PRNGKey(seed + 999)
    post = _run_mcmc(
        _intercept_only_model, key, min(mcmc_cfg["warmup"], 500),
        min(mcmc_cfg["samples"], 500), mcmc_cfg["chains"],
        groups=jnp.asarray(groups), n_groups=int(n_groups), y=jnp.asarray(y_std),
    )
    tau2 = post["tau_user"] ** 2
    sig2 = post["sigma"] ** 2
    icc = tau2 / (tau2 + sig2)
    return {
        "icc_between_user": float(np.mean(icc)),
        "icc_hdi90": [float(np.percentile(icc, 5)), float(np.percentile(icc, 95))],
        "tau_user": float(np.mean(post["tau_user"])),
        "sigma": float(np.mean(post["sigma"])),
    }


# --------------------------------------------------------------------------- #
# CV protocol (matches select_best_oracle_model / fidelity "cold")
# --------------------------------------------------------------------------- #
def build_cv_splits(df, seed, cv_folds):
    groups, source = infer_oracle_groups(df)
    if groups is not None and np.unique(groups).size >= 2:
        n = min(cv_folds, int(np.unique(groups).size))
        splits = list(GroupKFold(n_splits=n).split(df, groups=groups))
        return splits, {"strategy": "group_kfold", "group_source": source, "folds": n}
    n = min(cv_folds, len(df))
    splits = list(KFold(n_splits=n, shuffle=True, random_state=seed).split(df))
    return splits, {"strategy": "kfold", "group_source": None, "folds": n}


def _remap_groups(raw_groups: np.ndarray) -> tuple[np.ndarray, int]:
    uniq = {g: i for i, g in enumerate(pd.unique(raw_groups))}
    return np.asarray([uniq[g] for g in raw_groups], dtype=np.int64), len(uniq)


def evaluate_dataset_objective(df, objective, objective_columns, param_columns, args, mcmc_cfg):
    weights = (
        parse_objective_weights(args.objective_weights, objective, objective_columns)
        if objective != "multi_objective" else None
    )
    splits, info = build_cv_splits(df, args.seed, args.cv_folds)
    raw_groups, _ = infer_oracle_groups(df)

    model_names = ["extra_trees", "tree_hier"] + [f"hbm_{m}" for m in args.mean]
    per_model_folds: dict[str, list[dict]] = {m: [] for m in model_names}
    hybrid_decomp: list[dict] = []
    tree_scale = 1.0

    for fi, (tr_idx, te_idx) in enumerate(splits):
        train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
        normalization = fit_objective_normalization(train_df, objective_columns) if args.normalize_objective else None
        y_train = compute_objective(train_df, objective_columns, args.normalize_objective, weights, normalization=normalization).to_numpy(float)
        y_test = compute_objective(test_df, objective_columns, args.normalize_objective, weights, normalization=normalization).to_numpy(float)
        X_train = train_df[param_columns].to_numpy(float)
        X_test = test_df[param_columns].to_numpy(float)

        # extra_trees baseline (raw X, raw y); honest cold-noise plug-in
        tree_groups = raw_groups[tr_idx] if raw_groups is not None else None
        pt, sm = extra_trees_predictive(X_train, y_train, X_test, tree_groups, args.seed + fi, tree_scale, mcmc_cfg["pred_draws"])
        per_model_folds["extra_trees"].append(scoring(y_test, pt, sm, args.seed + fi))

        # tree_hier hybrid: tree mean + random-effects noise decomposition
        pt, sm, decomp = tree_hier_predictive(X_train, y_train, X_test, tree_groups, args.seed + fi, tree_scale, mcmc_cfg["pred_draws"])
        per_model_folds["tree_hier"].append(scoring(y_test, pt, sm, args.seed + fi))
        hybrid_decomp.append(decomp)

        # HBM models: standardize y on train
        y_center, y_scale = float(y_train.mean()), float(y_train.std() or 1.0)
        y_train_std = (y_train - y_center) / y_scale
        if raw_groups is not None:
            g_train, n_groups = _remap_groups(raw_groups[tr_idx])
        else:
            g_train, n_groups = np.zeros(len(tr_idx), np.int64), 0
        rng = np.random.default_rng(args.seed + fi)
        for mean in args.mean:
            phi_tr, phi_te, _finfo = build_features(X_train, X_test, mean, args.rff_features, rng)
            post = fit_hbm(phi_tr, g_train, n_groups, y_train_std,
                           seed=args.seed + fi, infer=args.infer, mcmc_cfg=mcmc_cfg)
            pt, sm = predict_hbm_cold(post, phi_te, y_center, y_scale, args.seed + fi)
            per_model_folds[f"hbm_{mean}"].append(scoring(y_test, pt, sm, args.seed + fi))

    def _agg(folds):
        keys = folds[0].keys()
        return {k: float(np.mean([f[k] for f in folds])) for k in keys}

    models = {name: _agg(folds) for name, folds in per_model_folds.items()}
    hybrid_resid_decomp = {
        k: float(np.mean([d[k] for d in hybrid_decomp])) for k in hybrid_decomp[0]
    } if hybrid_decomp else {}

    # Native variance decomposition on the full data (intercept-only)
    vdec = {"icc_between_user": None}
    if raw_groups is not None:
        y_full = compute_objective(df, objective_columns, args.normalize_objective, weights).to_numpy(float)
        y_full_std = (y_full - y_full.mean()) / (y_full.std() or 1.0)
        g_full, n_full = _remap_groups(raw_groups)
        vdec = variance_decomposition(y_full_std, g_full, n_full, args.seed, mcmc_cfg)

    result = {"validation": info, "models": models, "variance_decomposition": vdec,
              "hybrid_residual_decomp": hybrid_resid_decomp}
    if args.warm:
        result["warm"] = warm_evaluation(df, objective, objective_columns, param_columns, args)
    return result


def _gauss_metrics(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
    sd = np.maximum(sd, 1e-9)
    z90 = 1.6448536269514722  # two-sided 90% -> 5th/95th normal quantile
    return {
        "n": int(y.size),
        "r2": float(r2_score(y, mu)) if y.size > 1 else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(y, mu))),
        "coverage90": float(np.mean((y >= mu - z90 * sd) & (y <= mu + z90 * sd))),
        "nll": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * ((y - mu) / sd) ** 2)),
    }


def warm_evaluation(df, objective, objective_columns, param_columns, args) -> dict | None:
    """Warm per-user updating: the payoff the hybrid has over a pooled oracle.

    For each held-out user, reveal ``--warm-context`` of their ratings, update
    the user intercept via empirical-Bayes conjugate shrinkage using the
    training-fold (tau, sigma), and predict the user's REMAINING ratings. The
    pooled tree cannot use the context, so its warm prediction is just the cold
    mean + total noise. Both are scored on the same held-out query rows."""
    weights = (
        parse_objective_weights(args.objective_weights, objective, objective_columns)
        if objective != "multi_objective" else None
    )
    raw_groups, source = infer_oracle_groups(df)
    if raw_groups is None:
        return {"skipped": "no user/group structure"}
    splits, _ = build_cv_splits(df, args.seed, args.cv_folds)
    tree_scale = 1.0
    pooled_rows: list[tuple] = []
    warm_rows: list[tuple] = []
    n_users_used = 0

    for fi, (tr_idx, te_idx) in enumerate(splits):
        train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
        normalization = fit_objective_normalization(train_df, objective_columns) if args.normalize_objective else None
        y_train = compute_objective(train_df, objective_columns, args.normalize_objective, weights, normalization=normalization).to_numpy(float)
        y_test = compute_objective(test_df, objective_columns, args.normalize_objective, weights, normalization=normalization).to_numpy(float)
        X_train = train_df[param_columns].to_numpy(float)
        X_test = test_df[param_columns].to_numpy(float)
        g_test = raw_groups[te_idx]

        tree = _fit_tree(X_train, y_train, args.seed + fi, tree_scale)
        resid, g_oof = _tree_oof_residuals(X_train, y_train, raw_groups[tr_idx], args.seed + fi, tree_scale)
        tau2, sigma2, grand = variance_components(resid, g_oof)
        total_sd = float(np.sqrt(tau2 + sigma2))
        m_test = tree.predict(X_test) + grand
        rng = np.random.default_rng(args.seed + fi)

        for u in pd.unique(g_test):
            idx = np.where(g_test == u)[0]
            if idx.size < args.warm_context + 1:
                continue
            n_users_used += 1
            perm = rng.permutation(idx)
            ctx, qry = perm[:args.warm_context], perm[args.warm_context:]
            r_ctx = y_test[ctx] - m_test[ctx]
            n = ctx.size
            prior_prec = (1.0 / tau2) if tau2 > 0 else 0.0
            denom = n / sigma2 + prior_prec
            alpha = float((r_ctx.sum() / sigma2) / denom) if denom > 0 else 0.0
            alpha_var = float(1.0 / denom) if denom > 0 else tau2
            for qi in qry:
                pooled_rows.append((y_test[qi], m_test[qi], total_sd))
                warm_rows.append((y_test[qi], m_test[qi] + alpha, float(np.sqrt(sigma2 + alpha_var))))

    if not warm_rows:
        return {"skipped": f"no users with > {args.warm_context} ratings"}

    def _split(rows):
        arr = np.asarray(rows, dtype=float)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    return {
        "group_source": source,
        "warm_context": args.warm_context,
        "n_user_instances": n_users_used,
        "n_query_rows": len(warm_rows),
        "pooled_tree": _gauss_metrics(*_split(pooled_rows)),
        "tree_hier_warm": _gauss_metrics(*_split(warm_rows)),
    }


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--dataset-config", type=Path, default=Path("datasets.json"))
    p.add_argument("--dataset-cache-dir", type=Path, default=Path(".dataset_cache"))
    p.add_argument("--datasets", type=str, default=None, help="Comma list to filter dataset names.")
    p.add_argument("--objective", type=str, default=None)
    p.add_argument("--objectives", type=str, default="composite")
    p.add_argument("--objective-weights", type=str, default=None)
    p.add_argument("--normalize-objective", action="store_true", default=False)
    p.add_argument("--mean", type=str, default="linear,rff", help="Comma list of HBM mean functions: linear,rff.")
    p.add_argument("--rff-features", type=int, default=128)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--infer", type=str, default="nuts", choices=["nuts", "svi"])
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--svi-steps", type=int, default=4000)
    p.add_argument("--svi-draws", type=int, default=1000)
    p.add_argument("--pred-draws", type=int, default=1000)
    p.add_argument("--max-rows-per-dataset", type=int, default=None, help="Subsample large datasets for speed.")
    p.add_argument("--fast", action="store_true", default=False, help="Fast preset (fewer draws, subsample, rff=64).")
    p.add_argument("--warm", action="store_true", default=False,
                   help="Also run warm per-user updating (tree_hier warm vs pooled tree).")
    p.add_argument("--warm-context", type=int, default=3,
                   help="Ratings revealed per held-out user before predicting the rest.")
    p.add_argument("--output-path", type=Path, default=Path("output") / "hierarchical_oracle_test.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.mean = [m.strip() for m in args.mean.split(",") if m.strip()]
    if args.fast:
        args.num_warmup, args.num_samples = 300, 300
        args.svi_steps = 2500
        args.rff_features = min(args.rff_features, 64)
        if args.max_rows_per_dataset is None:
            args.max_rows_per_dataset = 1500

    mcmc_cfg = {
        "warmup": args.num_warmup, "samples": args.num_samples, "chains": args.chains,
        "svi_steps": args.svi_steps, "svi_draws": args.svi_draws, "pred_draws": args.pred_draws,
    }

    datasets = parse_dataset_configs(args.data_dir, args.dataset_config, args.dataset_cache_dir)
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        datasets = [d for d in datasets if d.name in keep]

    payload = {
        "protocol": "cold: individual rows, GroupKFold by inferred user/group",
        "infer": args.infer, "mean_functions": args.mean, "rff_features": args.rff_features,
        "cv_folds": args.cv_folds, "seed": args.seed, "normalize_objective": args.normalize_objective,
        "datasets": [],
    }

    rows_for_table: list[tuple] = []
    for ds in datasets:
        objective_names = parse_objective_list(args.objective, args.objectives, ds.objective_map)
        ds_entry = {"name": ds.name, "objectives": {}}
        for obj in objective_names:
            cols = ds.objective_map[obj]
            df = load_observations(ds, obj)
            if args.max_rows_per_dataset is not None and len(df) > args.max_rows_per_dataset:
                df = df.sample(n=args.max_rows_per_dataset, random_state=args.seed).reset_index(drop=True)
            print(f"[{ds.name}:{obj}] {len(df)} rows -> fitting {['extra_trees', 'tree_hier'] + ['hbm_' + m for m in args.mean]} ...", file=sys.stderr)
            res = evaluate_dataset_objective(df, obj, cols, ds.param_columns, args, mcmc_cfg)
            ds_entry["objectives"][obj] = res
            icc = res["variance_decomposition"].get("icc_between_user")
            for mname, m in res["models"].items():
                rows_for_table.append((ds.name, obj, mname, m["r2"], m["rmse"],
                                       m["coverage90"], m["nll"], m["crps"], icc))
        payload["datasets"].append(ds_entry)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2))

    # Printed comparison table
    print("\n" + "=" * 104)
    print(f"{'dataset':11s} {'objective':11s} {'model':12s} {'R2':>7s} {'RMSE':>7s} "
          f"{'cov90':>6s} {'NLL':>7s} {'CRPS':>7s} {'ICC_user':>8s}")
    print("-" * 104)
    last = None
    for r in rows_for_table:
        tag = (r[0], r[1])
        icc = f"{r[8]:.3f}" if r[8] is not None else "  -  "
        if tag != last and last is not None:
            print("-" * 104)
        print(f"{r[0]:11s} {r[1]:11s} {r[2]:12s} {r[3]:7.3f} {r[4]:7.3f} "
              f"{r[5]:6.2f} {r[6]:7.3f} {r[7]:7.3f} {icc:>8s}")
        last = tag
    print("=" * 104)
    print("cov90 target = 0.90 (higher-then-near-0.90 = calibrated; <0.90 = over-confident).")

    # Warm per-user updating table (hybrid's distinctive payoff)
    warm_present = any(
        "warm" in o and isinstance(o.get("warm"), dict) and "tree_hier_warm" in o["warm"]
        for ds in payload["datasets"] for o in ds["objectives"].values()
    )
    if warm_present:
        print("\n" + "=" * 92)
        print("WARM per-user updating (reveal k ratings of a held-out user, predict the rest)")
        print(f"{'dataset':11s} {'objective':11s} {'model':16s} {'k':>3s} {'n_qry':>6s} "
              f"{'R2':>7s} {'RMSE':>7s} {'cov90':>6s} {'NLL':>7s}")
        print("-" * 92)
        for ds in payload["datasets"]:
            for obj, o in ds["objectives"].items():
                w = o.get("warm")
                if not isinstance(w, dict) or "tree_hier_warm" not in w:
                    continue
                for label, key in [("pooled_tree", "pooled_tree"), ("tree_hier_warm", "tree_hier_warm")]:
                    m = w[key]
                    print(f"{ds['name']:11s} {obj:11s} {label:16s} {w['warm_context']:3d} "
                          f"{m['n']:6d} {m['r2']:7.3f} {m['rmse']:7.3f} {m['coverage90']:6.2f} {m['nll']:7.3f}")
        print("=" * 92)
        print("tree_hier_warm uses the revealed ratings; pooled_tree ignores them (its warm = cold).")

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
