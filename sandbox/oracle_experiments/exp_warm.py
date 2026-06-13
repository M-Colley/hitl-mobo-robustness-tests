"""exp_warm: best warm-protocol oracle per dataset.

Variants (all evaluated through harness.evaluate_warm_per_user /
harness.evaluate_warm_pooled, train_frac=0.7 by default):

A) per-user models (one fresh model per user, fit on that user's early rows):
   - ridge (StandardScaler + Ridge alpha=1.0)
   - GP (StandardScaler + ConstantKernel*RBF + WhiteKernel, normalize_y)
   - kNN k=3 / k=5 (scaled)
   - small ExtraTrees n_estimators=300, min_samples_leaf in {1,2,5}
   - baseline extra_trees (600, msl=2) for reference
B) pooled warm (one model on all users' early rows), user_onehot True/False:
   - extra_trees (deployed config), LightGBM (min_child_samples=20), ridge
C) hybrid shrinkage: pred = w * per_user_ridge + (1-w) * pooled_extratrees.
   The pooled ExtraTrees is pre-fit ONLY on the union of all users' warm-TRAIN
   rows (identical indices to evaluate_warm_pooled's train set) — no test
   contact. w in {0.3, 0.5, 0.7} fixed, plus an adaptive variant that picks w
   on a temporal holdout of each user's TRAIN slice only.

Finally: train_frac sensitivity (0.5 vs default 0.7) for the per-dataset best
variant by warm rmse_pooled.
"""
import numpy as np

import harness

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor

SEED = harness.SEED


# ---------------------------------------------------------------- factories
def make_ridge():
    return Pipeline([
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=1.0, random_state=SEED)),
    ])


def make_gp():
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-6, 1e2))
    )
    return Pipeline([
        ("sc", StandardScaler()),
        ("m", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=SEED)),
    ])


def make_knn(k):
    return Pipeline([
        ("sc", StandardScaler()),
        ("m", KNeighborsRegressor(n_neighbors=k)),
    ])


def make_small_et(msl):
    return ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=msl, random_state=SEED, n_jobs=1
    )


def make_lgbm():
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        min_child_samples=20,
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
    )


# ----------------------------------------------------------------- hybrid
def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


class HybridShrinkage:
    """pred = w * per_user_ridge + (1-w) * pre-fit pooled model.

    fit() only ever sees the user's TRAIN slice (the harness guarantees this);
    the pooled model was fit on the union of all users' train slices.
    w=None -> choose w from w_grid on a temporal holdout of the train slice
    (last ~30% of the train rows), then refit ridge on the full train slice.
    """

    def __init__(self, pooled_model, w=None, w_grid=(0.3, 0.5, 0.7)):
        self.pooled_model = pooled_model
        self.w = w
        self.w_grid = w_grid

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.w is not None:
            self.w_ = float(self.w)
        else:
            n = len(y)
            cut = int(np.ceil(0.7 * n))
            cut = min(max(cut, 2), n - 2)
            if cut >= 2 and (n - cut) >= 2:
                inner = make_ridge().fit(X[:cut], y[:cut])
                pr = np.asarray(inner.predict(X[cut:]), float).reshape(-1)
                pp = np.asarray(self.pooled_model.predict(X[cut:]), float).reshape(-1)
                best_w, best = self.w_grid[0], np.inf
                for w in self.w_grid:
                    err = _rmse(y[cut:], w * pr + (1.0 - w) * pp)
                    if err < best:
                        best, best_w = err, w
                self.w_ = float(best_w)
            else:
                self.w_ = 0.5
        self.ridge_ = make_ridge().fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pr = np.asarray(self.ridge_.predict(X), float).reshape(-1)
        pp = np.asarray(self.pooled_model.predict(X), float).reshape(-1)
        return self.w_ * pr + (1.0 - self.w_) * pp


def fit_pooled_et_on_warm_train(data, train_frac, min_rows=10):
    """Fit the deployed ExtraTrees config on the union of warm-TRAIN rows only."""
    splits = harness._warm_split_indices(data, train_frac, min_rows)
    train_idx = np.concatenate([t for _, t, _ in splits])
    model = ExtraTreesRegressor(
        n_estimators=600, min_samples_leaf=2, random_state=SEED, n_jobs=1
    )
    model.fit(data.X[train_idx], data.y[train_idx])
    return model


# ---------------------------------------------------------------- variants
# name -> runner(data, train_frac) -> metrics dict
def build_variants():
    variants = {}

    variants["et600_msl2_baseline"] = lambda d, tf: harness.evaluate_warm_per_user(
        d, harness.baseline_factory, train_frac=tf
    )
    variants["ridge_scaled"] = lambda d, tf: harness.evaluate_warm_per_user(
        d, make_ridge, train_frac=tf
    )
    variants["gp_rbf_white"] = lambda d, tf: harness.evaluate_warm_per_user(
        d, make_gp, train_frac=tf
    )
    for k in (3, 5):
        variants[f"knn{k}_scaled"] = (
            lambda d, tf, k=k: harness.evaluate_warm_per_user(
                d, lambda: make_knn(k), train_frac=tf
            )
        )
    for msl in (1, 2, 5):
        variants[f"et300_msl{msl}"] = (
            lambda d, tf, msl=msl: harness.evaluate_warm_per_user(
                d, lambda: make_small_et(msl), train_frac=tf
            )
        )

    for onehot in (True, False):
        tag = "onehot" if onehot else "noid"
        variants[f"pooled_et600_{tag}"] = (
            lambda d, tf, oh=onehot: harness.evaluate_warm_pooled(
                d, harness.baseline_factory, user_onehot=oh, train_frac=tf
            )
        )
        variants[f"pooled_lgbm_{tag}"] = (
            lambda d, tf, oh=onehot: harness.evaluate_warm_pooled(
                d, make_lgbm, user_onehot=oh, train_frac=tf
            )
        )
        variants[f"pooled_ridge_{tag}"] = (
            lambda d, tf, oh=onehot: harness.evaluate_warm_pooled(
                d, make_ridge, user_onehot=oh, train_frac=tf
            )
        )

    def hybrid_runner(d, tf, w):
        pooled = fit_pooled_et_on_warm_train(d, tf)
        return harness.evaluate_warm_per_user(
            d, lambda: HybridShrinkage(pooled, w=w), train_frac=tf
        )

    for w in (0.3, 0.5, 0.7):
        variants[f"hybrid_w{w}"] = lambda d, tf, w=w: hybrid_runner(d, tf, w)
    variants["hybrid_w_adaptive"] = lambda d, tf: hybrid_runner(d, tf, None)

    return variants


def main():
    variants = build_variants()
    rows = []
    best_by_dataset = {}

    for name in harness.DATASET_NAMES:
        data = harness.load_dataset(name)
        best_rmse, best_variant = np.inf, None
        for vname, runner in variants.items():
            try:
                metrics = runner(data, 0.7)
            except Exception as exc:  # report failures honestly
                rows.append({
                    "dataset": name, "model": vname, "protocol": "warm_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            rows.append({"dataset": name, "model": vname, **metrics})
            if metrics["rmse_pooled"] < best_rmse:
                best_rmse, best_variant = metrics["rmse_pooled"], vname
        best_by_dataset[name] = best_variant

        # train_frac sensitivity for the best variant (0.7 reported above)
        for tf in (0.5,):
            try:
                metrics = variants[best_variant](data, tf)
            except Exception as exc:
                rows.append({
                    "dataset": name, "model": f"{best_variant}_tf{tf}",
                    "protocol": "warm_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            rows.append({
                "dataset": name, "model": f"{best_variant}_tf{tf}", **metrics
            })

    rows.append({
        "dataset": "ALL", "model": "best_variant_per_dataset",
        "protocol": "meta", "r2_pooled": float("nan"),
        "best": best_by_dataset,
    })
    harness.print_results("warm_variants", rows)


if __name__ == "__main__":
    main()
