"""Shared evaluation harness for oracle-fidelity experiments.

Every experiment must evaluate through these functions so results are
comparable. Two protocols:

- COLD (matches select_best_oracle_model.py): pooled model, GroupKFold over
  users; measures how well the oracle predicts ratings of UNSEEN users.
  This is the published baseline: ehmi 0.142, opticarvis 0.518, provoice -0.040.

- WARM: within-user temporal split (first `train_frac` of each user's
  evaluations -> train, rest -> test). This matches the oracle's actual job
  in the simulation: predicting the SAME user's later responses during a
  session. Two variants: one model per user, or one pooled model with a
  user identifier feature.

Targets are the unweighted composite objective (normalize=False), identical
to the simulation. All randomness is seeded; estimators should use n_jobs=1
(a 16-worker sweep may be running concurrently).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bo_sensor_error_simulation as bo_sim  # noqa: E402

SEED = 7
DATASET_NAMES = ["ehmi", "opticarvis", "provoice"]
COLD_BASELINES = {"ehmi": 0.142, "opticarvis": 0.518, "provoice": -0.040}


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    param_columns: list[str]
    objective_columns: list[str]
    group_source: str
    construct_Y: np.ndarray = field(default=None)  # signed per-construct values


def load_dataset(name: str) -> DatasetBundle:
    datasets = {
        d.name: d
        for d in bo_sim.parse_dataset_configs(None, None, REPO_ROOT / ".dataset_cache")
    }
    ds = datasets[name]
    df = bo_sim.load_observations(ds, "composite")
    cols = ds.objective_map["composite"]
    y = bo_sim.compute_objective(df, cols, False, None).to_numpy(dtype=float)
    groups, source = bo_sim.infer_oracle_groups(df)
    if groups is None:
        raise ValueError(f"No grouping available for dataset {name}")
    return DatasetBundle(
        name=name,
        df=df,
        X=df[ds.param_columns].to_numpy(dtype=float),
        y=y,
        groups=np.asarray(groups).astype(str),
        param_columns=list(ds.param_columns),
        objective_columns=list(cols),
        group_source=source,
        construct_Y=bo_sim._extract_objective_values(df, cols),
    )


def evaluate_cold(
    data: DatasetBundle,
    factory,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    n_splits: int = 5,
) -> dict:
    """Pooled model, GroupKFold over users (the select_best_oracle_model protocol).

    factory() must return a fresh unfitted estimator with fit/predict.
    Pass X to evaluate engineered features; pass y to evaluate transformed targets.
    Returns mean per-fold R^2 (primary, matches the baseline) and pooled R^2.
    """
    X = data.X if X is None else X
    y = data.y if y is None else y
    unique_groups = np.unique(data.groups)
    splits = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=splits)
    fold_r2: list[float] = []
    pooled_pred = np.full(len(y), np.nan)
    for train_idx, test_idx in gkf.split(X, y, groups=data.groups):
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[test_idx]), dtype=float).reshape(-1)
        pooled_pred[test_idx] = pred
        fold_r2.append(float(r2_score(y[test_idx], pred)))
    return {
        "protocol": "cold",
        "r2_mean_fold": float(np.mean(fold_r2)),
        "r2_pooled": float(r2_score(y, pooled_pred)),
        "rmse_pooled": float(np.sqrt(np.mean((y - pooled_pred) ** 2))),
        "y_std": float(np.std(y)),
        "n_folds": splits,
        "n_rows": int(len(y)),
        "n_users": int(len(unique_groups)),
    }


def _warm_split_indices(
    data: DatasetBundle, train_frac: float, min_rows: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Per user: (user, train_idx, test_idx) using within-file row order
    (= chronological evaluation order in the source BO sessions)."""
    out = []
    order = np.arange(len(data.y))
    for user in np.unique(data.groups):
        idx = order[data.groups == user]
        if len(idx) < min_rows:
            continue
        cut = int(np.ceil(train_frac * len(idx)))
        if cut < 2 or len(idx) - cut < 2:
            continue
        out.append((user, idx[:cut], idx[cut:]))
    return out


def evaluate_warm_per_user(
    data: DatasetBundle,
    factory,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    train_frac: float = 0.7,
    min_rows: int = 10,
) -> dict:
    """One model per user, trained on that user's first train_frac of
    evaluations, tested on the rest. Pooled R^2 over all users' test rows
    (primary) plus the median of per-user R^2."""
    X = data.X if X is None else X
    y = data.y if y is None else y
    splits = _warm_split_indices(data, train_frac, min_rows)
    if not splits:
        raise ValueError("No users with enough rows for a warm split.")
    test_idx_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    per_user_r2: list[float] = []
    for _, train_idx, test_idx in splits:
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[test_idx]), dtype=float).reshape(-1)
        test_idx_all.append(test_idx)
        pred_all.append(pred)
        if np.var(y[test_idx]) > 0:
            per_user_r2.append(float(r2_score(y[test_idx], pred)))
    test_idx_cat = np.concatenate(test_idx_all)
    pred_cat = np.concatenate(pred_all)
    errors = y[test_idx_cat] - pred_cat
    # NOTE: warm R^2 is structurally pessimistic — late-session BO evaluations
    # concentrate near the optimum, so test variance collapses. Judge warm
    # fidelity primarily by rmse_pooled vs the intra-rater noise floor
    # (calibrate_noise_from_data.py: ehmi 0.27, opticarvis 1.25, provoice <=0.65).
    return {
        "protocol": "warm_per_user",
        "r2_pooled": float(r2_score(y[test_idx_cat], pred_cat)),
        "r2_median_user": float(np.median(per_user_r2)) if per_user_r2 else float("nan"),
        "rmse_pooled": float(np.sqrt(np.mean(errors**2))),
        "y_test_std": float(np.std(y[test_idx_cat])),
        "n_users": int(len(splits)),
        "n_test_rows": int(len(test_idx_cat)),
        "train_frac": train_frac,
    }


def evaluate_warm_pooled(
    data: DatasetBundle,
    factory,
    user_onehot: bool = True,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    train_frac: float = 0.7,
    min_rows: int = 10,
) -> dict:
    """One pooled model over all users' early evaluations, tested on all
    users' later evaluations; optionally with user one-hot features so the
    model can learn per-user offsets while borrowing strength across users."""
    X = data.X if X is None else X
    y = data.y if y is None else y
    splits = _warm_split_indices(data, train_frac, min_rows)
    if not splits:
        raise ValueError("No users with enough rows for a warm split.")
    train_idx = np.concatenate([t for _, t, _ in splits])
    test_idx = np.concatenate([t for _, _, t in splits])
    if user_onehot:
        users = np.unique(data.groups)
        onehot = (data.groups[:, None] == users[None, :]).astype(float)
        X = np.hstack([X, onehot])
    model = factory()
    model.fit(X[train_idx], y[train_idx])
    pred = np.asarray(model.predict(X[test_idx]), dtype=float).reshape(-1)
    errors = y[test_idx] - pred
    return {
        "protocol": "warm_pooled" + ("_onehot" if user_onehot else ""),
        "r2_pooled": float(r2_score(y[test_idx], pred)),
        "rmse_pooled": float(np.sqrt(np.mean(errors**2))),
        "y_test_std": float(np.std(y[test_idx])),
        "n_users": int(len(splits)),
        "n_test_rows": int(len(test_idx)),
        "train_frac": train_frac,
    }


def baseline_factory():
    """The currently deployed oracle family (cold baseline reference)."""
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(n_estimators=600, random_state=SEED, min_samples_leaf=2, n_jobs=1)


def print_results(experiment: str, rows: list[dict]) -> None:
    """Emit a machine-readable result block (agents parse this)."""
    print("RESULTS_JSON_BEGIN")
    print(json.dumps({"experiment": experiment, "rows": rows}, indent=2))
    print("RESULTS_JSON_END")
