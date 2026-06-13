"""Model zoo: non-tree model families never tried by the oracle selection.

Models (all scaled inside a Pipeline so scaler state never leaks):
- ridge        : StandardScaler + RidgeCV (alphas logspace(-3,3,13), internal LOO)
- knn_k5/k10   : StandardScaler + KNeighborsRegressor(distance weights);
                 k is clipped to n_train at fit time for tiny warm splits
- svr_rbf_C1/C10 : StandardScaler + SVR(rbf)
- gp_rbf_white : StandardScaler + GaussianProcessRegressor(C*RBF + White)
- mlp_64_32    : StandardScaler + MLPRegressor((64,32), early_stopping)
- tabpfn_ref   : TabPFNRegressor cold-protocol fidelity reference only
                 (device='cpu', n_estimators=2; skipped if a fold exceeds 5 min)

Protocols: cold (evaluate_cold) and warm (evaluate_warm_per_user), all three
datasets. n_jobs=1 everywhere, random_state=harness.SEED.

Usage:
    python exp_zoo.py                 # everything (sklearn zoo + tabpfn)
    python exp_zoo.py sklearn         # sklearn zoo only, all datasets
    python exp_zoo.py tabpfn ehmi     # tabpfn cold reference, one dataset
    python exp_zoo.py mlp_64_32       # rerun a single sklearn model

Each finished row is also appended to zoo_rows.jsonl so partial runs are
never lost; the RESULTS_JSON block at the end holds the rows of this run.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import harness

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TABPFN_FOLD_BUDGET_S = 300.0


class ClippedKNN(BaseEstimator, RegressorMixin):
    """KNeighborsRegressor whose k is clipped to n_train at fit time.

    Needed for warm per-user splits where some users have fewer training
    rows than k. Clipping uses only the training rows handed to fit().
    """

    def __init__(self, n_neighbors=5, weights="distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        k = min(self.n_neighbors, len(np.asarray(y)))
        self.model_ = KNeighborsRegressor(
            n_neighbors=k, weights=self.weights, n_jobs=1
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def make_ridge():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])


def make_knn(k):
    def factory():
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", ClippedKNN(n_neighbors=k, weights="distance")),
        ])
    return factory


def make_svr(C):
    def factory():
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=C)),
        ])
    return factory


def make_gp():
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel()
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=harness.SEED,
        )),
    ])


class AdaptiveMLP(BaseEstimator, RegressorMixin):
    """MLP (64,32) with early stopping, except on tiny warm-protocol fits
    where the 10% validation split would have <2 rows (sklearn raises
    'validation set is too small'; R^2 scoring needs >=2 rows). The switch
    depends only on the number of training rows handed to fit()."""

    MIN_ROWS_FOR_EARLY_STOP = 20

    def fit(self, X, y):
        early = len(np.asarray(y)) >= self.MIN_ROWS_FOR_EARLY_STOP
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            early_stopping=early,
            max_iter=2000,
            random_state=harness.SEED,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def make_mlp():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", AdaptiveMLP()),
    ])


def make_tabpfn():
    import torch
    torch.set_num_threads(1)
    # bo_sensor_error_simulation (imported by harness) sets torch's default
    # dtype to float64, which breaks TabPFN's float32 weights ("mat1 and mat2
    # must have the same dtype"). TabPFN needs the stock float32 default.
    torch.set_default_dtype(torch.float32)
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        device="cpu",
        n_estimators=2,
        auto_scale_n_estimators=False,
        random_state=harness.SEED,
    )


def tabpfn_first_fold_seconds(data) -> float:
    """Time TabPFN fit+predict on the first cold fold (same splitter as
    evaluate_cold) to decide whether the full run fits the time budget."""
    X, y = data.X, data.y
    splits = min(5, len(np.unique(data.groups)))
    gkf = GroupKFold(n_splits=splits)
    train_idx, test_idx = next(iter(gkf.split(X, y, groups=data.groups)))
    t0 = time.time()
    model = make_tabpfn()
    model.fit(X[train_idx], y[train_idx])
    model.predict(X[test_idx])
    return time.time() - t0


ROWS_JSONL = Path(__file__).resolve().parent / "zoo_rows.jsonl"


def add_row(rows: list, row: dict) -> None:
    rows.append(row)
    with ROWS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


ZOO_MODELS = [
    ("ridge", make_ridge),
    ("knn_k5", make_knn(5)),
    ("knn_k10", make_knn(10)),
    ("svr_rbf_C1", make_svr(1.0)),
    ("svr_rbf_C10", make_svr(10.0)),
    ("gp_rbf_white", make_gp),
    ("mlp_64_32", make_mlp),
]


def run_sklearn_zoo(rows: list, dataset_names: list[str],
                    only_model: str | None = None) -> None:
    models = [(n, f) for n, f in ZOO_MODELS if only_model in (None, n)]
    for name in dataset_names:
        data = harness.load_dataset(name)
        for model_name, factory in models:
            for evaluator in (harness.evaluate_cold, harness.evaluate_warm_per_user):
                t0 = time.time()
                try:
                    res = evaluator(data, factory)
                    res["fit_seconds_total"] = round(time.time() - t0, 1)
                    add_row(rows, {"dataset": name, "model": model_name, **res})
                except Exception as exc:  # report failures honestly
                    add_row(rows, {
                        "dataset": name,
                        "model": model_name,
                        "protocol": "cold" if evaluator is harness.evaluate_cold
                        else "warm_per_user",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                print(
                    f"done {name} {model_name} "
                    f"{rows[-1]['protocol']} ({time.time() - t0:.1f}s)",
                    flush=True,
                )


def run_tabpfn(rows: list, dataset_names: list[str]) -> None:
    """TabPFN: cold-only fidelity reference with a per-fold time budget."""
    for name in dataset_names:
        data = harness.load_dataset(name)
        try:
            fold_s = tabpfn_first_fold_seconds(data)
            print(f"tabpfn first-fold timing {name}: {fold_s:.1f}s", flush=True)
            if fold_s > TABPFN_FOLD_BUDGET_S:
                add_row(rows, {
                    "dataset": name, "model": "tabpfn_ref", "protocol": "cold",
                    "skipped": f"first fold took {fold_s:.0f}s > "
                               f"{TABPFN_FOLD_BUDGET_S:.0f}s budget",
                })
            else:
                t0 = time.time()
                res = harness.evaluate_cold(data, make_tabpfn)
                res["fit_seconds_total"] = round(time.time() - t0, 1)
                add_row(rows, {"dataset": name, "model": "tabpfn_ref", **res})
                print(f"done {name} tabpfn_ref cold ({time.time() - t0:.1f}s)",
                      flush=True)
        except Exception as exc:
            add_row(rows, {
                "dataset": name, "model": "tabpfn_ref", "protocol": "cold",
                "error": f"{type(exc).__name__}: {exc}",
            })


def main(argv: list[str]) -> None:
    part = argv[0] if argv else "all"
    datasets = argv[1:] if len(argv) > 1 else harness.DATASET_NAMES
    rows: list = []
    if part in ("all", "sklearn"):
        run_sklearn_zoo(rows, datasets)
    elif part in {n for n, _ in ZOO_MODELS}:  # single sklearn model rerun
        run_sklearn_zoo(rows, datasets, only_model=part)
    if part in ("all", "tabpfn"):
        run_tabpfn(rows, datasets)
    harness.print_results("zoo", rows)


if __name__ == "__main__":
    main(sys.argv[1:])
