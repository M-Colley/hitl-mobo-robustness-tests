"""exp_tuned_trees: nested hyperparameter tuning for tree ensembles.

COLD protocol (ehmi, provoice; opticarvis ET-only since it is already at
ceiling): for each outer GroupKFold(5) fold the factory returns a
NestedGroupTuner whose fit() runs an inner GroupKFold(3) over the TRAINING
users only, picks the best params by mean inner R^2, then refits on the full
outer-training data. Groups reach the tuner via an extra column appended to X
(a pure row-wise label column) which is stripped before any model sees the
features — test rows/users are never touched during tuning.

WARM protocol (ehmi, provoice): shallow regularized ExtraTrees / RandomForest
variants per user (n_estimators=300, min_samples_leaf in {1,2,5}); each fixed
variant is reported as its own row (no selection on test data).
"""
import time
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid

import harness

SEED = harness.SEED
warnings.filterwarnings("ignore")


class NestedGroupTuner:
    """fit(X, y) where X[:, -1] is the group code (label only, never a feature).

    Runs inner GroupKFold over the training groups, selects params by mean
    inner R^2, refits the winner on all training rows.
    """

    def __init__(self, base_ctor, param_grid, chosen_log=None, n_inner=3):
        self.base_ctor = base_ctor
        self.param_grid = param_grid
        self.chosen_log = chosen_log
        self.n_inner = n_inner

    def fit(self, X, y):
        groups = X[:, -1]
        Xf = X[:, :-1]
        n_groups = len(np.unique(groups))
        gkf = GroupKFold(n_splits=min(self.n_inner, n_groups))
        inner_splits = list(gkf.split(Xf, y, groups=groups))
        best_score, best_params = -np.inf, None
        for params in ParameterGrid(self.param_grid):
            scores = []
            for tr, te in inner_splits:
                m = self.base_ctor(**params)
                m.fit(Xf[tr], y[tr])
                pred = np.asarray(m.predict(Xf[te]), dtype=float).reshape(-1)
                scores.append(r2_score(y[te], pred))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score, best_params = mean_score, params
        self.best_params_ = best_params
        self.best_inner_score_ = best_score
        if self.chosen_log is not None:
            self.chosen_log.append({k: str(v) for k, v in best_params.items()})
        self.model_ = self.base_ctor(**best_params)
        self.model_.fit(Xf, y)
        return self

    def predict(self, X):
        return self.model_.predict(X[:, :-1])


# ---- model families -------------------------------------------------------

def et_ctor(**p):
    return ExtraTreesRegressor(random_state=SEED, n_jobs=1, **p)


def rf_ctor(**p):
    return RandomForestRegressor(random_state=SEED, n_jobs=1, **p)


def lgbm_ctor(**p):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        random_state=SEED, n_jobs=1, verbose=-1,
        subsample=0.8, subsample_freq=1, **p,
    )


def xgb_ctor(**p):
    from xgboost import XGBRegressor

    return XGBRegressor(
        random_state=SEED, n_jobs=1, verbosity=0,
        subsample=0.8, tree_method="hist", **p,
    )


TREE_GRID = {
    "n_estimators": [600],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": [None, "sqrt", 0.5],
}
LGBM_GRID = {
    "n_estimators": [200, 800],
    "learning_rate": [0.02, 0.05],
    "num_leaves": [7, 15, 31],
    "min_child_samples": [5, 20, 50],
}
XGB_GRID = {
    "n_estimators": [200, 800],
    "learning_rate": [0.02, 0.05],
    "max_depth": [2, 3, 6],
    "min_child_weight": [1, 5, 10],
}

COLD_FAMILIES = {
    "ehmi": [
        ("extra_trees_tuned", et_ctor, TREE_GRID),
        ("random_forest_tuned", rf_ctor, TREE_GRID),
        ("lightgbm_tuned", lgbm_ctor, LGBM_GRID),
        ("xgboost_tuned", xgb_ctor, XGB_GRID),
    ],
    "provoice": [
        ("extra_trees_tuned", et_ctor, TREE_GRID),
        ("random_forest_tuned", rf_ctor, TREE_GRID),
        ("lightgbm_tuned", lgbm_ctor, LGBM_GRID),
        ("xgboost_tuned", xgb_ctor, XGB_GRID),
    ],
    # already at ceiling -> only the cheap deployed family, as a sanity check
    "opticarvis": [
        ("extra_trees_tuned", et_ctor, TREE_GRID),
    ],
}

WARM_DATASETS = ["ehmi", "provoice"]


def main() -> None:
    rows = []
    for name in ["ehmi", "provoice", "opticarvis"]:
        t0 = time.time()
        data = harness.load_dataset(name)
        # group code as an extra (stripped) column so the tuner can re-derive
        # groups for the inner split from training rows alone
        _, codes = np.unique(data.groups, return_inverse=True)
        Xg = np.hstack([data.X, codes[:, None].astype(float)])

        # baseline anchor (cheap, within-run reference)
        res = harness.evaluate_cold(data, harness.baseline_factory)
        rows.append({"dataset": name, "model": "extra_trees_baseline", **res})
        print(f"[{name}] baseline cold done ({time.time()-t0:.0f}s)", flush=True)

        for model_name, ctor, grid in COLD_FAMILIES[name]:
            t1 = time.time()
            chosen: list[dict] = []
            factory = lambda c=ctor, g=grid, log=chosen: NestedGroupTuner(c, g, chosen_log=log)
            res = harness.evaluate_cold(data, factory, X=Xg)
            rows.append({
                "dataset": name,
                "model": model_name,
                **res,
                "best_params_per_fold": chosen,
            })
            print(f"[{name}] {model_name} cold done ({time.time()-t1:.0f}s) "
                  f"r2={res['r2_mean_fold']:.3f} rmse={res['rmse_pooled']:.3f} "
                  f"params={chosen}", flush=True)

        if name in WARM_DATASETS:
            res = harness.evaluate_warm_per_user(data, harness.baseline_factory)
            rows.append({"dataset": name, "model": "extra_trees_baseline", **res})
            for label, ctor in [("extra_trees", et_ctor), ("random_forest", rf_ctor)]:
                for msl in [1, 2, 5]:
                    factory = lambda c=ctor, m=msl: c(n_estimators=300, min_samples_leaf=m)
                    res = harness.evaluate_warm_per_user(data, factory)
                    rows.append({
                        "dataset": name,
                        "model": f"{label}_n300_msl{msl}",
                        **res,
                    })
            print(f"[{name}] warm variants done", flush=True)
        print(f"[{name}] total {time.time()-t0:.0f}s", flush=True)

    harness.print_results("tuned_trees", rows)


if __name__ == "__main__":
    main()
