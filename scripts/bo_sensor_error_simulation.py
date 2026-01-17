"""
Simulate sensor-error impacts in HITL Bayesian optimization using eHMI data.
(BoTorch Implementation + regret metrics)

Example:
  python scripts/bo_sensor_error_simulation.py \
    --iterations 50 \
    --jitter-iterations 20 \
    --jitter-stds 0.1 \
    --acq ei,ucb \
    --output-dir /tmp/botorch_output
"""
from __future__ import annotations


import os
import sys
import types

# Set thread limits BEFORE importing numpy/torch/sklearn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

import argparse
import dataclasses
import importlib.metadata
import json
import re
import subprocess
import time
import uuid 
import warnings
from pathlib import Path

# add near imports
import threading

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
#from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.sampling.normal import SobolQMCNormalSampler

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "eHMI-bo-participantdata"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_DATASET_NAME = "default"
if __name__ not in sys.modules:
    current_module = types.ModuleType(__name__)
    current_module.__dict__.update(globals())
    sys.modules[__name__] = current_module

PARAM_COLUMNS = [
    "verticalPosition",
    "verticalWidth",
    "horizontalWidth",
    "r",
    "g",
    "b",
    "a",
    "blinkFrequency",
    "volume",
]

OBJECTIVE_MAP = {
    "composite": ["Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance"],
    "multi_objective": ["Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance"],
    "trust": ["Trust"],
    "understanding": ["Understanding"],
    "perceived_safety": ["PerceivedSafety"],
    "aesthetics": ["Aesthetics"],
    "acceptance": ["Acceptance"],
}

SINGLE_ACQUISITION_CHOICES = [
    "logei",
    "logpi",
    "ei",
    "pi",
    "ucb",
    "qucb",
    "qei",
    "qpi",
    "qnei",
    "greedy",
]
MULTI_ACQUISITION_CHOICES = ["qehvi", "qnehvi"]
ACQUISITION_CHOICES = SINGLE_ACQUISITION_CHOICES + MULTI_ACQUISITION_CHOICES
ERROR_MODEL_CHOICES = ["gaussian", "bias", "dropout", "spike"]
ORACLE_MODEL_CHOICES = [
    "xgboost",
    "lightgbm",
    #"catboost",
    "tabpfn",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
]


@dataclasses.dataclass
class Bounds:
    low: np.ndarray
    high: np.ndarray

    @property
    def tensor(self) -> torch.Tensor:
        low_t = torch.from_numpy(self.low)
        high_t = torch.from_numpy(self.high)
        return torch.stack([low_t, high_t])


@dataclasses.dataclass
class AcquisitionConfig:
    name: str
    xi: float = 0.01
    kappa: float = 2.0


@dataclasses.dataclass
class SimulationConfig:
    iterations: int
    jitter_iteration: int
    jitter_std: float
    single_error: bool
    initial_samples: int
    candidate_pool: int
    objective: str
    objective_columns: list[str]
    param_columns: list[str]
    seed: int
    error_model: str
    error_bias: float
    error_spike_prob: float
    error_spike_std: float
    dropout_strategy: str
    normalize_objective: bool
    objective_weights: np.ndarray | None

    # BoTorch optimization controls
    acq_num_restarts: int
    acq_raw_samples: int
    acq_maxiter: int
    acq_mc_samples: int

    # Multi-objective settings
    ref_point: np.ndarray | None


@dataclasses.dataclass
class DatasetConfig:
    name: str
    data_dirs: list[Path]
    param_columns: list[str]
    objective_map: dict[str, list[str]]
    observation_glob: str = "ObservationsPerEvaluation.csv"


@dataclasses.dataclass
class OracleModel:
    model: object
    objective_name: str
    objective_columns: list[str]
    param_columns: list[str] | None = None

    def _prepare_features(self, X: np.ndarray, model: object) -> np.ndarray | pd.DataFrame:
        feature_names: list[str] | None = None
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        elif hasattr(model, "feature_name_"):
            feature_names = list(model.feature_name_)
        elif self.param_columns:
            feature_names = list(self.param_columns)
        if feature_names:
            return pd.DataFrame(X, columns=feature_names)
        return X

    def predict(self, x: np.ndarray) -> np.ndarray:
        X = x.reshape(1, -1)
        if isinstance(self.model, list):
            values = [
                float(m.predict(self._prepare_features(X, m))[0]) for m in self.model
            ]
            return np.asarray(values, dtype=float)
        return np.asarray([float(self.model.predict(self._prepare_features(X, self.model))[0])], dtype=float)

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.model, list):
            preds = [
                np.asarray(m.predict(self._prepare_features(X, m)), dtype=float)
                for m in self.model
            ]
            return np.stack(preds, axis=1)
        return np.asarray(
            self.model.predict(self._prepare_features(X, self.model)), dtype=float
        ).reshape(-1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)

    parser.add_argument("--jitter-iteration", type=int, default=20)
    parser.add_argument("--jitter-std", type=float, default=0.2)
    parser.add_argument(
        "--single-error",
        action="store_true",
        default=False,
        help="Apply sensor error only once at the first iteration after jitter-iteration.",
    )
    # TODO - add more
    parser.add_argument("--jitter-iterations", type=str, default="10,20,40")
    parser.add_argument("--jitter-stds", type=str, default="0.05,0.5,1,5")

    parser.add_argument("--initial-samples", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=1000)  # kept for compatibility

    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--objectives", type=str, default=None)

    parser.add_argument("--acq", type=str, default="all", choices=ACQUISITION_CHOICES + ["all"])
    parser.add_argument("--acq-list", type=str, default=None)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num-seeds", type=int, default=5)

    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Path to a JSON file describing one or more datasets to load.",
    )
    parser.add_argument(
        "--combine-datasets",
        action="store_true",
        default=False,
        help="Add a combined dataset when multiple datasets share objective/parameter columns.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=Path(".dataset_cache"),
        help="Local cache directory for remote dataset repositories.",
    )

    parser.add_argument("--baseline-run", action="store_true", default=True)
    parser.add_argument("--no-baseline-run", action="store_false", dest="baseline_run")

    parser.add_argument("--error-model", type=str, default="gaussian", choices=ERROR_MODEL_CHOICES + ["all"])
    parser.add_argument("--error-models", type=str, default="gaussian,bias")

    parser.add_argument("--error-bias", type=float, default=0.2)
    parser.add_argument("--error-spike-prob", type=float, default=0.1)
    parser.add_argument("--error-spike-std", type=float, default=0.5)
    parser.add_argument("--dropout-strategy", type=str, default="hold_last", choices=["hold_last"])

    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--group-id", type=str, default=None)

    parser.add_argument("--normalize-objective", action="store_true", default=False)
    parser.add_argument("--objective-weights", type=str, default=None)

    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=2.0)

    parser.add_argument("--oracle-model", type=str, default="extra_trees", choices=ORACLE_MODEL_CHOICES + ["all"])
    parser.add_argument("--oracle-models", type=str, default="extra_trees")
    parser.add_argument(
        "--oracle-augmentation",
        type=str,
        default="jitter",
        choices=["none", "jitter"],
        help="Optional data augmentation for oracle training.",
    )
    parser.add_argument("--oracle-augment-repeats", type=int, default=2)
    parser.add_argument("--oracle-augment-std", type=float, default=0.02)
    parser.add_argument(
        "--oracle-fast",
        action="store_true",
        default=False,
        help="Use reduced oracle model sizes for faster experimentation.",
    )

    # Oracle optimum approximation for regret
    parser.add_argument("--oracle-opt-samples", type=int, default=200_000)
    parser.add_argument("--oracle-opt-batch-size", type=int, default=50_000)
    parser.add_argument("--oracle-opt-seed", type=int, default=10_007)

    # BoTorch acquisition optimization controls
    parser.add_argument("--acq-num-restarts", type=int, default=10)
    parser.add_argument("--acq-raw-samples", type=int, default=512)
    parser.add_argument("--acq-maxiter", type=int, default=200)
    parser.add_argument("--acq-mc-samples", type=int, default=256)
    
    parser.add_argument("--parallel",action="store_true", default=False,
    help="Enable parallel processing (auto-enabled for multiple seeds)",)
    parser.add_argument("--n-jobs", type=int, default=-1,
    help="Number of parallel jobs (-1 = all cores, -2 = all but one)",)

    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_observation_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    last_df: pd.DataFrame | None = None
    for sep in (";", ","):
        df = pd.read_csv(path, sep=sep)
        df.columns = df.columns.str.strip()
        df = _normalize_qehvi_columns(df)
        last_df = df
        if all(col in df.columns for col in required_columns):
            return df

    available = ", ".join(
        [str(col) for col in (last_df.columns if last_df is not None else [])]
    )
    missing = (
        ", ".join(sorted(set(required_columns) - set(last_df.columns)))
        if last_df is not None
        else ", ".join(required_columns)
    )
    raise ValueError(
        f"Observation file '{path}' is missing required columns: {missing}. "
        f"Available columns: {available}"
    )


def _normalize_qehvi_columns(df: pd.DataFrame) -> pd.DataFrame:
    qehvi_suffix = re.compile(r"\s+QEHVI$", flags=re.IGNORECASE)
    existing = set(df.columns)
    rename_map: dict[str, str] = {}
    for col in df.columns:
        base = qehvi_suffix.sub("", col)
        if base != col and base not in existing:
            rename_map[col] = base
            existing.add(base)
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_observations(
    dataset: DatasetConfig,
    objective: str,
    user_id: str | None = None,
    group_id: str | None = None,
) -> pd.DataFrame:
    files: list[Path] = []
    for data_dir in dataset.data_dirs:
        files.extend(list(data_dir.rglob(dataset.observation_glob)))
    if not files:
        dirs = ", ".join(str(path) for path in dataset.data_dirs)
        raise FileNotFoundError(f"No observation files found in {dirs} using {dataset.observation_glob}")

    required_columns = dataset.param_columns + dataset.objective_map[objective]
    frames = [_read_observation_csv(path, required_columns) for path in files]
    df = pd.concat(frames, ignore_index=True)

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["User_ID", "Group_ID"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if user_id is not None and "User_ID" not in df.columns:
        raise ValueError("User_ID column missing from observations; cannot filter by --user-id.")
    if group_id is not None and "Group_ID" not in df.columns:
        raise ValueError("Group_ID column missing from observations; cannot filter by --group-id.")

    if user_id is not None:
        df = df[df["User_ID"] == float(user_id)]
    if group_id is not None:
        df = df[df["Group_ID"] == float(group_id)]

    df = df.dropna(subset=dataset.param_columns + dataset.objective_map[objective])
    if df.empty:
        raise ValueError("No data remaining after applying user/group filters.")
    return df.reset_index(drop=True)


def compute_objective(
    df: pd.DataFrame,
    objective_columns: list[str],
    normalize: bool,
    weights: np.ndarray | None,
) -> pd.Series:
    cols = objective_columns
    values = df[cols].to_numpy(dtype=float)

    if normalize:
        min_vals = np.nanmin(values, axis=0)
        max_vals = np.nanmax(values, axis=0)
        ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        values = (values - min_vals) / ranges

    if weights is None:
        return pd.Series(values.mean(axis=1), index=df.index)

    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return pd.Series(values @ weights, index=df.index)


def compute_objective_matrix(
    df: pd.DataFrame,
    objective_columns: list[str],
    normalize: bool,
) -> np.ndarray:
    cols = objective_columns
    values = df[cols].to_numpy(dtype=float)

    if normalize:
        min_vals = np.nanmin(values, axis=0)
        max_vals = np.nanmax(values, axis=0)
        ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        values = (values - min_vals) / ranges

    return values


def parse_objective_weights(
    weights_arg: str | None,
    objective: str,
    objective_columns: list[str],
) -> np.ndarray | None:
    if weights_arg is None:
        return None
    if objective == "multi_objective":
        raise ValueError("Objective weights are not supported for multi_objective.")
    values = [float(v.strip()) for v in weights_arg.split(",") if v.strip()]
    expected = len(objective_columns)
    if len(values) != expected:
        raise ValueError(f"Expected {expected} weights for objective={objective}, got {len(values)}.")
    return np.array(values, dtype=float)


def augment_oracle_data(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    repeats: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return X, y
    if repeats < 1:
        return X, y
    if mode != "jitter":
        raise ValueError(f"Unknown oracle augmentation mode: {mode}")

    augmented_X = [X]
    augmented_y = [y]
    for _ in range(repeats):
        noise = rng.normal(0.0, noise_std, size=X.shape)
        augmented_X.append(X + noise)
        augmented_y.append(y)

    return np.vstack(augmented_X), np.concatenate(augmented_y, axis=0)


def build_oracle(
    df: pd.DataFrame,
    objective: str,
    objective_columns: list[str],
    param_columns: list[str],
    seed: int,
    normalize: bool,
    weights: np.ndarray | None,
    oracle_model: str,
    oracle_augmentation: str,
    oracle_augment_repeats: int,
    oracle_augment_std: float,
    oracle_fast: bool,
) -> OracleModel:
    X = df[param_columns].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    if oracle_fast:
        tree_scale = 0.35
    else:
        tree_scale = 1.0

    if objective == "multi_objective":
        Y = compute_objective_matrix(df, objective_columns, normalize)
        X_aug, Y_aug = augment_oracle_data(
            X,
            Y,
            rng,
            oracle_augmentation,
            oracle_augment_repeats,
            oracle_augment_std,
        )
        models = []
        X_aug_df = pd.DataFrame(X_aug, columns=param_columns)
        for idx, target in enumerate(objective_columns):
            y = Y_aug[:, idx]
            models.append(
                _build_oracle_model(
                    oracle_model=oracle_model,
                    seed=seed,
                    tree_scale=tree_scale,
                )
            )
            models[-1].fit(X_aug_df, y)
            train_score = models[-1].score(X_aug_df, y)
            y_pred = models[-1].predict(X_aug_df)
            train_rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            print(f"Oracle ({oracle_model}) for {target} trained on {len(X_aug)} samples:")
            print(f"  R² score: {train_score:.4f}")
            print(f"  RMSE: {train_rmse:.4f}")

        return OracleModel(
            model=models,
            objective_name=objective,
            objective_columns=objective_columns,
            param_columns=param_columns,
        )

    y = compute_objective(df, objective_columns, normalize, weights).to_numpy(dtype=float)
    X_aug, y_aug = augment_oracle_data(
        X,
        y,
        rng,
        oracle_augmentation,
        oracle_augment_repeats,
        oracle_augment_std,
    )

    X_aug_df = pd.DataFrame(X_aug, columns=param_columns)

    model = _build_oracle_model(
        oracle_model=oracle_model,
        seed=seed,
        tree_scale=tree_scale,
    )

    model.fit(X_aug_df, y_aug)

    # Report oracle performance
    train_score = model.score(X_aug_df, y_aug)
    y_pred = model.predict(X_aug_df)
    train_rmse = np.sqrt(np.mean((y_aug - y_pred) ** 2))
    print(f"Oracle ({oracle_model}) trained on {len(X_aug)} samples:")
    print(f"  R² score: {train_score:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")

    return OracleModel(
        model=model,
        objective_name=objective,
        objective_columns=objective_columns,
        param_columns=param_columns,
    )


def _build_oracle_model(oracle_model: str, seed: int, tree_scale: float) -> object:
    if oracle_model == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(600 * tree_scale),
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=1,
        )
    if oracle_model == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(600 * tree_scale),
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=1,
        )
    if oracle_model == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(500 * tree_scale),
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        )
    if oracle_model == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=int(400 * tree_scale),
            learning_rate=0.05,
            max_depth=6,
            random_state=seed,
        )
    if oracle_model == "xgboost":
        return XGBRegressor(
            n_estimators=int(800 * tree_scale),
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
        )
    if oracle_model == "lightgbm":
        return LGBMRegressor(
            n_estimators=int(800 * tree_scale),
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            force_row_wise=True,
            verbosity=-1,
        )
    if oracle_model == "catboost":
        return CatBoostRegressor(
            iterations=int(800 * tree_scale),
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=seed,
            thread_count=1,
            verbose=False,
        )
    if oracle_model == "tabpfn":
        estimators = max(2, int(round(8 * tree_scale)))
        return TabPFNRegressor(
            n_estimators=estimators,
            device="cpu",
            n_preprocessing_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"Unknown oracle model: {oracle_model}")


def bounds_from_data(df: pd.DataFrame, param_columns: list[str]) -> Bounds:
    low = df[param_columns].min().to_numpy(dtype=float)
    high = df[param_columns].max().to_numpy(dtype=float)
    return Bounds(low=low, high=high)


def sample_uniform(bounds: Bounds, rng: np.random.Generator, size: int) -> np.ndarray:
    d = len(bounds.low)
    return rng.uniform(bounds.low, bounds.high, size=(size, d))


def estimate_oracle_optimum(
    oracle: OracleModel,
    bounds: Bounds,
    seed: int,
    n: int,
    batch_size: int,
) -> float:
    rng = np.random.default_rng(seed)
    best = -np.inf
    d = len(bounds.low)

    remaining = int(n)
    while remaining > 0:
        m = min(batch_size, remaining)
        X = rng.uniform(bounds.low, bounds.high, size=(m, d))
        y = oracle.predict_many(X)
        best = max(best, float(np.max(y)))
        remaining -= m

    return best


def estimate_oracle_hypervolume(
    oracle: OracleModel,
    bounds: Bounds,
    seed: int,
    n: int,
    batch_size: int,
    ref_point: np.ndarray,
) -> float:
    rng = np.random.default_rng(seed)
    d = len(bounds.low)
    collected: list[np.ndarray] = []

    remaining = int(n)
    while remaining > 0:
        m = min(batch_size, remaining)
        X = rng.uniform(bounds.low, bounds.high, size=(m, d))
        y = oracle.predict_many(X)
        collected.append(y)
        remaining -= m

    Y = np.vstack(collected)
    Y_t = torch.tensor(Y, dtype=torch.double)
    nd_mask = is_non_dominated(Y_t)
    pareto = Y_t[nd_mask]
    hv = Hypervolume(ref_point=torch.tensor(ref_point, dtype=torch.double))
    return float(hv.compute(pareto))


def parse_seed_list(seed_arg: str | None, seed: int, num_seeds: int | None) -> list[int]:
    if seed_arg:
        values = [int(v.strip()) for v in seed_arg.split(",") if v.strip()]
        if not values:
            raise ValueError("No valid seeds parsed from --seeds.")
        return values
    if num_seeds:
        return list(range(seed, seed + num_seeds))
    return [seed]


def parse_acquisition_list(acq_arg: str, acq_list: str | None) -> list[str]:
    raw = acq_list or acq_arg
    if raw == "all":
        return ACQUISITION_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one acquisition must be specified.")
    unknown = [v for v in values if v not in ACQUISITION_CHOICES]
    if unknown:
        raise ValueError(f"Unknown acquisition(s): {', '.join(unknown)}")
    return values


def filter_acquisitions_for_objective(acquisitions: list[str], objective: str) -> list[str]:
    if objective == "multi_objective":
        filtered = [a for a in acquisitions if a in MULTI_ACQUISITION_CHOICES]
        if not filtered:
            raise ValueError(
                "Multi-objective optimization requires one of "
                f"{', '.join(MULTI_ACQUISITION_CHOICES)}."
            )
        return filtered
    filtered = [a for a in acquisitions if a in SINGLE_ACQUISITION_CHOICES]
    if not filtered:
        raise ValueError(
            "Single-objective optimization requires one of "
            f"{', '.join(SINGLE_ACQUISITION_CHOICES)}."
        )
    return filtered


def parse_error_models(error_model: str, error_models: str | None) -> list[str]:
    raw = error_models or error_model
    if raw == "all":
        return ERROR_MODEL_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one error model must be specified.")
    unknown = [v for v in values if v not in ERROR_MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unknown error model(s): {', '.join(unknown)}")
    return values


def parse_oracle_models(oracle_model: str, oracle_models: str | None) -> list[str]:
    raw = oracle_models or oracle_model
    if raw == "all":
        return ORACLE_MODEL_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one oracle model must be specified.")
    unknown = [v for v in values if v not in ORACLE_MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unknown oracle model(s): {', '.join(unknown)}")
    return values


def is_remote_dataset_path(value: str) -> bool:
    return value.startswith(("http://", "https://", "git@")) or value.endswith(".git")


def sanitize_repo_name(value: str) -> str:
    trimmed = value.rstrip("/")
    if trimmed.endswith(".git"):
        trimmed = trimmed[:-4]
    return trimmed.split("/")[-1]


def fetch_remote_dataset(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_name = sanitize_repo_name(url)
    target_dir = cache_dir / repo_name
    if target_dir.exists():
        return target_dir
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target_dir)],
        check=True,
    )
    return target_dir


def resolve_data_dirs(raw_dirs: list[str], cache_dir: Path) -> list[Path]:
    resolved = []
    for entry in raw_dirs:
        if is_remote_dataset_path(entry):
            resolved.append(fetch_remote_dataset(entry, cache_dir))
        else:
            resolved.append(Path(entry))
    return resolved


def parse_dataset_configs(
    data_dir: Path,
    dataset_config_path: Path | None,
    cache_dir: Path,
) -> list[DatasetConfig]:
    if dataset_config_path is None:
        return [
            DatasetConfig(
                name=DEFAULT_DATASET_NAME,
                data_dirs=[data_dir],
                param_columns=list(PARAM_COLUMNS),
                objective_map={k: list(v) for k, v in OBJECTIVE_MAP.items()},
                observation_glob="ObservationsPerEvaluation.csv",
            )
        ]

    payload = json.loads(dataset_config_path.read_text())
    if isinstance(payload, dict) and "datasets" in payload:
        dataset_payloads = payload["datasets"]
    elif isinstance(payload, list):
        dataset_payloads = payload
    else:
        raise ValueError("Dataset config must be a list or a dict with a 'datasets' key.")

    datasets: list[DatasetConfig] = []
    for idx, entry in enumerate(dataset_payloads, start=1):
        if not isinstance(entry, dict):
            raise ValueError("Each dataset entry must be a JSON object.")
        name = entry.get("name") or f"dataset_{idx}"
        data_dirs = entry.get("data_dirs")
        data_dir = entry.get("data_dir")
        if data_dirs is None:
            if data_dir is None:
                raise ValueError(f"Dataset '{name}' must define 'data_dir' or 'data_dirs'.")
            data_dirs = [data_dir]
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]
        param_columns = entry.get("param_columns", PARAM_COLUMNS)
        objective_map = entry.get("objective_map", OBJECTIVE_MAP)
        observation_glob = entry.get("observation_glob", "ObservationsPerEvaluation.csv")

        if not isinstance(objective_map, dict):
            raise ValueError(f"Dataset '{name}' objective_map must be a dict.")
        cleaned_objective_map: dict[str, list[str]] = {}
        for key, value in objective_map.items():
            if not isinstance(value, list):
                raise ValueError(f"Objective '{key}' for dataset '{name}' must be a list of columns.")
            cleaned_objective_map[str(key)] = [str(col) for col in value]

        resolved_dirs = resolve_data_dirs([str(path) for path in data_dirs], cache_dir)
        datasets.append(
            DatasetConfig(
                name=str(name),
                data_dirs=resolved_dirs,
                param_columns=[str(col) for col in param_columns],
                objective_map=cleaned_objective_map,
                observation_glob=str(observation_glob),
            )
        )

    dataset_names = [dataset.name for dataset in datasets]
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Dataset names must be unique.")
    return datasets


def combine_dataset_configs(datasets: list[DatasetConfig], name: str = "combined") -> DatasetConfig | None:
    if len(datasets) < 2:
        return None
    first = datasets[0]
    if any(dataset.param_columns != first.param_columns for dataset in datasets[1:]):
        warnings.warn("Cannot combine datasets with different parameter columns.")
        return None
    if any(dataset.observation_glob != first.observation_glob for dataset in datasets[1:]):
        warnings.warn("Cannot combine datasets with different observation_glob patterns.")
        return None

    common_objectives = set(first.objective_map.keys())
    for dataset in datasets[1:]:
        common_objectives &= set(dataset.objective_map.keys())

    objective_map: dict[str, list[str]] = {}
    for objective in sorted(common_objectives):
        columns = first.objective_map[objective]
        if all(dataset.objective_map[objective] == columns for dataset in datasets[1:]):
            objective_map[objective] = columns

    if not objective_map:
        warnings.warn("No common objectives found to build a combined dataset.")
        return None

    combined_dirs: list[Path] = []
    for dataset in datasets:
        combined_dirs.extend(dataset.data_dirs)

    return DatasetConfig(
        name=name,
        data_dirs=combined_dirs,
        param_columns=first.param_columns,
        objective_map=objective_map,
        observation_glob=first.observation_glob,
    )


def parse_objective_list(
    objective_arg: str | None,
    objectives: str | None,
    objective_map: dict[str, list[str]],
) -> list[str]:
    if objectives is None and objective_arg is None:
        defaults = [name for name in ["composite", "multi_objective"] if name in objective_map]
        return defaults or list(objective_map.keys())
    raw = objectives or objective_arg
    if raw is None:
        defaults = [name for name in ["composite", "multi_objective"] if name in objective_map]
        return defaults or list(objective_map.keys())
    if raw == "all":
        return list(objective_map.keys())
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        defaults = [name for name in ["composite", "multi_objective"] if name in objective_map]
        return defaults or list(objective_map.keys())
    unknown = [v for v in values if v not in objective_map]
    if unknown:
        raise ValueError(f"Unknown objective(s): {', '.join(unknown)}")
    return values


def parse_float_list(value: str | None, default: float) -> list[float]:
    if value is None:
        return [default]
    values = [float(v.strip()) for v in value.split(",") if v.strip()]
    return values or [default]


def parse_int_list(value: str | None, default: int) -> list[int]:
    if value is None:
        return [default]
    values = [int(v.strip()) for v in value.split(",") if v.strip()]
    return values or [default]


def validate_sweeps(jitter_iterations: list[int], jitter_stds: list[float], iterations: int) -> None:
    for j in jitter_iterations:
        if j < 1 or j >= iterations:
            raise ValueError("Each jitter-iteration must be within [1, iterations - 1].")
    for s in jitter_stds:
        if s < 0:
            raise ValueError("Each jitter-std must be >= 0.")


def validate_inputs(args: argparse.Namespace) -> None:
    if args.iterations <= 1:
        raise ValueError("iterations must be greater than 1.")
    if args.initial_samples < 1 or args.initial_samples >= args.iterations:
        raise ValueError("initial-samples must be within [1, iterations - 1].")
    if args.error_spike_prob < 0 or args.error_spike_prob > 1:
        raise ValueError("error-spike-prob must be between 0 and 1.")
    if args.oracle_opt_samples < 10_000:
        raise ValueError("oracle-opt-samples should be reasonably large (>= 10000).")
    if args.oracle_opt_batch_size < 1:
        raise ValueError("oracle-opt-batch-size must be >= 1.")
    if args.acq_mc_samples < 1:
        raise ValueError("acq-mc-samples must be >= 1.")
    if args.oracle_augment_repeats < 0:
        raise ValueError("oracle-augment-repeats must be >= 0.")
    if args.oracle_augment_std < 0:
        raise ValueError("oracle-augment-std must be >= 0.")


def compute_reference_point(
    df: pd.DataFrame,
    objective: str,
    objective_columns: list[str],
) -> np.ndarray | None:
    if objective != "multi_objective":
        return None
    values = df[objective_columns].to_numpy(dtype=float)
    min_vals = np.nanmin(values, axis=0)
    max_vals = np.nanmax(values, axis=0)
    ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    return min_vals - 0.1 * ranges


def write_run_config(
    output_dir: Path,
    dataset_configs: list[DatasetConfig],
    objectives: dict[str, list[str]],
    acquisition_names: list[str],
    error_models: list[str],
    oracle_models: list[str],
    seeds: list[int],
    args: argparse.Namespace,
) -> None:
    dataset_lines = ["Datasets:"]
    for dataset in dataset_configs:
        dataset_lines.append(f"  - {dataset.name}: {', '.join(str(p) for p in dataset.data_dirs)}")

    lines = [
        "Sensor-error simulation configuration",
        "=" * 60,
        "",
        *dataset_lines,
        "",
        "Objectives by dataset:",
        *[f"  {name}: {', '.join(values)}" for name, values in objectives.items()],
        f"Acquisitions: {', '.join(acquisition_names)}",
        f"Error models: {', '.join(error_models)}",
        f"Oracle models: {', '.join(oracle_models)}",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "Core settings:",
        f"  iterations: {args.iterations}",
        f"  initial_samples: {args.initial_samples}",
        f"  jitter_iterations: {args.jitter_iterations}",
        f"  jitter_stds: {args.jitter_stds}",
        f"  single_error: {args.single_error}",
        f"  baseline_run: {args.baseline_run}",
        "",
        "Oracle settings:",
        f"  augmentation: {args.oracle_augmentation}",
        f"  augmentation_repeats: {args.oracle_augment_repeats}",
        f"  augmentation_std: {args.oracle_augment_std}",
        f"  oracle_fast: {args.oracle_fast}",
        "",
        "BO settings:",
        f"  xi: {args.xi}",
        f"  kappa: {args.kappa}",
        f"  acq_num_restarts: {args.acq_num_restarts}",
        f"  acq_raw_samples: {args.acq_raw_samples}",
        f"  acq_maxiter: {args.acq_maxiter}",
        f"  acq_mc_samples: {args.acq_mc_samples}",
        "",
        "Error model parameters:",
        f"  error_bias: {args.error_bias}",
        f"  error_spike_prob: {args.error_spike_prob}",
        f"  error_spike_std: {args.error_spike_std}",
        "",
        "Data filters:",
        f"  user_id: {args.user_id}",
        f"  group_id: {args.group_id}",
        "",
        "Normalization:",
        f"  normalize_objective: {args.normalize_objective}",
        f"  objective_weights: {args.objective_weights}",
    ]
    config_path = output_dir / "run_config.txt"
    config_path.write_text("\n".join(lines))


def collect_package_versions(packages: list[str]) -> dict[str, str]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def apply_sensor_error(
    true_value: np.ndarray,
    iteration: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    previous_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if iteration <= config.jitter_iteration:
        return true_value, np.zeros_like(true_value, dtype=float)
    if config.single_error and iteration != config.jitter_iteration + 1:
        return true_value, np.zeros_like(true_value, dtype=float)

    jitter = rng.normal(0.0, config.jitter_std, size=true_value.shape)

    if config.error_model == "gaussian":
        return true_value + jitter, jitter
    if config.error_model == "bias":
        bias = np.full_like(true_value, config.error_bias, dtype=float)
        combined = bias + jitter
        return true_value + combined, combined
    if config.error_model == "dropout":
        if config.dropout_strategy != "hold_last":
            raise ValueError(f"Unsupported dropout strategy: {config.dropout_strategy}")
        observed = previous_observed + jitter
        return observed, observed - true_value
    if config.error_model == "spike":
        spike = np.zeros_like(true_value, dtype=float)
        if rng.random() < config.error_spike_prob:
            spike = rng.normal(0.0, config.error_spike_std, size=true_value.shape)
        combined = spike + jitter
        return true_value + combined, combined

    raise ValueError(f"Unknown error model: {config.error_model}")


def get_botorch_candidate(
    gp_model: SingleTaskGP | ModelListGP,
    acq_config: AcquisitionConfig,
    bounds_tensor: torch.Tensor,
    best_f: float | None,
    num_restarts: int,
    raw_samples: int,
    maxiter: int,
    mc_samples: int,
    train_X: torch.Tensor,
    train_Y: torch.Tensor | list[torch.Tensor],
    ref_point: np.ndarray | None,
) -> torch.Tensor:
    if acq_config.name == "logei":
        if best_f is None:
            raise ValueError("best_f required for logei.")
        acqf = LogExpectedImprovement(model=gp_model, best_f=best_f + acq_config.xi)
    elif acq_config.name == "logpi":
        if best_f is None:
            raise ValueError("best_f required for logpi.")
        acqf = LogProbabilityOfImprovement(model=gp_model, best_f=best_f + acq_config.xi)
    elif acq_config.name == "ei":
        if best_f is None:
            raise ValueError("best_f required for ei.")
        acqf = ExpectedImprovement(model=gp_model, best_f=best_f + acq_config.xi)
    elif acq_config.name == "pi":
        if best_f is None:
            raise ValueError("best_f required for pi.")
        acqf = ProbabilityOfImprovement(model=gp_model, best_f=best_f + acq_config.xi)
    elif acq_config.name == "ucb":
        acqf = UpperConfidenceBound(model=gp_model, beta=float(acq_config.kappa**2))
    elif acq_config.name == "greedy":
        acqf = UpperConfidenceBound(model=gp_model, beta=0.0)
    elif acq_config.name == "qei":
        if best_f is None:
            raise ValueError("best_f required for qei.")
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qExpectedImprovement(model=gp_model, best_f=best_f + acq_config.xi, sampler=sampler)
    elif acq_config.name == "qpi":
        if best_f is None:
            raise ValueError("best_f required for qpi.")
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qProbabilityOfImprovement(model=gp_model, best_f=best_f + acq_config.xi, sampler=sampler)
    elif acq_config.name == "qucb":
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qUpperConfidenceBound(model=gp_model, beta=float(acq_config.kappa**2), sampler=sampler)
    elif acq_config.name == "qnei":
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qNoisyExpectedImprovement(
            model=gp_model,
            X_baseline=train_X,
            sampler=sampler,
        )
    elif acq_config.name == "qehvi":
        if ref_point is None:
            raise ValueError("ref_point required for qehvi.")
        if not isinstance(gp_model, ModelListGP):
            raise ValueError("qehvi requires a multi-objective ModelListGP.")
        if isinstance(train_Y, list):
            train_Y_stack = torch.cat(train_Y, dim=1)
        else:
            train_Y_stack = train_Y
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(ref_point, dtype=torch.double),
            Y=train_Y_stack,
        )
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qExpectedHypervolumeImprovement(
            model=gp_model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )
    elif acq_config.name == "qnehvi":
        if ref_point is None:
            raise ValueError("ref_point required for qnehvi.")
        if not isinstance(gp_model, ModelListGP):
            raise ValueError("qnehvi requires a multi-objective ModelListGP.")
        if isinstance(train_Y, list):
            train_Y_stack = torch.cat(train_Y, dim=1)
        else:
            train_Y_stack = train_Y
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(ref_point, dtype=torch.double),
            Y=train_Y_stack,
        )
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qNoisyExpectedHypervolumeImprovement(
            model=gp_model,
            ref_point=ref_point.tolist(),
            X_baseline=train_X,
            sampler=sampler,
            partitioning=partitioning,
        )
    else:
        raise ValueError(f"Unknown acquisition: {acq_config.name}")

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_tensor,
        q=1,
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
        options={"batch_limit": 5, "maxiter": int(maxiter)},
    )
    return candidate.detach()


def _compute_hypervolume(values: list[np.ndarray], ref_point: np.ndarray) -> float:
    if not values:
        return 0.0
    Y = torch.tensor(np.vstack(values), dtype=torch.double)
    nd_mask = is_non_dominated(Y)
    pareto = Y[nd_mask]
    hv = Hypervolume(ref_point=torch.tensor(ref_point, dtype=torch.double))
    return float(hv.compute(pareto))


def run_simulation(
    oracle: OracleModel,
    bounds: Bounds,
    config: SimulationConfig,
    acq: AcquisitionConfig,
    rng: np.random.Generator,
    jitter_rng: np.random.Generator | None,
    run_id: str,
    apply_error: bool,
    oracle_model: str,
    y_opt: float,
) -> pd.DataFrame:
    X_list: list[np.ndarray] = []
    y_observed_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []
    error_magnitudes: list[np.ndarray] = []
    fit_times: list[float] = []

    # Regret tracking (computed on true objective)
    best_true_so_far = -np.inf
    cum_regret = 0.0
    best_true_list: list[float] = []
    regret_inst_list: list[float] = []
    regret_cum_list: list[float] = []
    simple_regret_list: list[float] = []
    regret_avg_list: list[float] = []

    bounds_tensor = bounds.tensor
    previous_observed = None

    is_multi = config.objective == "multi_objective"
    objective_true_scalar: list[float] = []
    objective_observed_scalar: list[float] = []
    hv_ref_point = config.ref_point if config.ref_point is not None else None

    for iteration in range(1, config.iterations + 1):
        if iteration <= config.initial_samples:
            candidate_np = sample_uniform(bounds, rng, size=1)[0]
            fit_time = 0.0
        else:
            fit_start = time.perf_counter()

            train_X = torch.tensor(np.vstack(X_list), dtype=torch.double)
            if is_multi:
                train_Y_array = np.vstack(y_observed_list)
                train_Y_list = [
                    torch.tensor(train_Y_array[:, idx].reshape(-1, 1), dtype=torch.double)
                    for idx in range(train_Y_array.shape[1])
                ]
                gps = [
                    SingleTaskGP(
                        train_X,
                        train_Y_list[idx],
                        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
                        outcome_transform=Standardize(m=1),
                    )
                    for idx in range(train_Y_array.shape[1])
                ]
                gp = ModelListGP(*gps)
                mll = SumMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                best_f = None
                train_Y_for_acq: list[torch.Tensor] | torch.Tensor = train_Y_list
            else:
                train_Y = torch.tensor(
                    np.array(y_observed_list, dtype=float).reshape(-1, 1), dtype=torch.double
                )
                gp = SingleTaskGP(
                    train_X,
                    train_Y,
                    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
                    outcome_transform=Standardize(m=1),
                )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                best_f = train_Y.max().item()
                train_Y_for_acq = train_Y

            try:
                candidate_tensor = get_botorch_candidate(
                    gp_model=gp,
                    acq_config=acq,
                    bounds_tensor=bounds_tensor,
                    best_f=best_f,
                    num_restarts=config.acq_num_restarts,
                    raw_samples=config.acq_raw_samples,
                    maxiter=config.acq_maxiter,
                    mc_samples=config.acq_mc_samples,
                    train_X=train_X,
                    train_Y=train_Y_for_acq,
                    ref_point=hv_ref_point,
                )
                candidate_np = candidate_tensor.cpu().numpy().flatten()
            except Exception as e:
                warnings.warn(f"BoTorch optimization failed, falling back to random. Error: {e}")
                candidate_np = sample_uniform(bounds, rng, size=1)[0]

            fit_time = time.perf_counter() - fit_start

        true_value = oracle.predict(candidate_np)

        if previous_observed is None:
            previous_observed = true_value

        if apply_error:
            if jitter_rng is None:
                raise ValueError("jitter_rng must be provided when apply_error is True.")
            observed_value, error_magnitude = apply_sensor_error(
                true_value=true_value,
                iteration=iteration,
                config=config,
                rng=jitter_rng,
                previous_observed=previous_observed,
            )
        else:
            observed_value, error_magnitude = true_value, np.zeros_like(true_value, dtype=float)

        X_list.append(candidate_np)
        y_true_list.append(true_value)
        y_observed_list.append(observed_value)
        error_magnitudes.append(error_magnitude)
        fit_times.append(fit_time)
        previous_observed = observed_value

        if is_multi:
            if hv_ref_point is None:
                raise ValueError("ref_point required for multi_objective.")
            hv_true = _compute_hypervolume(y_true_list, hv_ref_point)
            hv_obs = _compute_hypervolume(y_observed_list, hv_ref_point)
            objective_true_scalar.append(hv_true)
            objective_observed_scalar.append(hv_obs)
            best_true_so_far = max(best_true_so_far, hv_true)
            r_t = max(0.0, y_opt - hv_true)
            cum_regret += r_t
            s_t = max(0.0, y_opt - best_true_so_far)
        else:
            scalar_true = float(true_value[0])
            scalar_obs = float(observed_value[0])
            objective_true_scalar.append(scalar_true)
            objective_observed_scalar.append(scalar_obs)
            best_true_so_far = max(best_true_so_far, scalar_true)
            r_t = max(0.0, y_opt - scalar_true)
        cum_regret += r_t
        s_t = max(0.0, y_opt - best_true_so_far)

        best_true_list.append(best_true_so_far)
        regret_inst_list.append(r_t)
        regret_cum_list.append(cum_regret)
        simple_regret_list.append(s_t)
        regret_avg_list.append(cum_regret / float(iteration))

    results = pd.DataFrame(X_list, columns=config.param_columns)
    results.insert(0, "iteration", np.arange(1, config.iterations + 1))

    results["objective_true"] = objective_true_scalar
    results["objective_observed"] = objective_observed_scalar
    if is_multi:
        for idx, column in enumerate(config.objective_columns):
            results[f"objective_true_{column}"] = [float(v[idx]) for v in y_true_list]
            results[f"objective_observed_{column}"] = [float(v[idx]) for v in y_observed_list]

    if apply_error:
        if config.single_error:
            results["error_applied"] = results["iteration"] == config.jitter_iteration + 1
        else:
            results["error_applied"] = results["iteration"] > config.jitter_iteration
    else:
        results["error_applied"] = False

    if is_multi:
        results["error_magnitude_l2"] = [float(np.linalg.norm(err)) for err in error_magnitudes]
        for idx, column in enumerate(config.objective_columns):
            results[f"error_magnitude_{column}"] = [float(err[idx]) for err in error_magnitudes]
    else:
        results["error_magnitude"] = [float(err[0]) for err in error_magnitudes]
    results["acquisition"] = acq.name
    results["fit_time_sec"] = fit_times
    results["seed"] = config.seed
    results["run_id"] = run_id
    results["error_model"] = config.error_model if apply_error else "none"
    results["jitter_std"] = config.jitter_std if apply_error else 0.0
    results["jitter_iteration"] = config.jitter_iteration
    results["oracle_model"] = oracle_model
    results["objective"] = config.objective

    # Regret columns
    results["y_opt"] = float(y_opt)
    results["best_true_so_far"] = best_true_list
    results["regret_inst_true"] = regret_inst_list
    results["regret_cum_true"] = regret_cum_list
    results["simple_regret_true"] = simple_regret_list
    results["regret_avg_true"] = regret_avg_list

    return results


def summarize_adjustment(
    results: pd.DataFrame,
    jitter_iteration: int,
    param_columns: list[str],
) -> pd.Series:
    max_iter = int(results["iteration"].max())
    if jitter_iteration < 1 or jitter_iteration >= max_iter:
        raise ValueError("jitter_iteration must be within [1, max_iteration - 1].")

    current = results.loc[results["iteration"] == jitter_iteration, param_columns].iloc[0]
    nxt = results.loc[results["iteration"] == jitter_iteration + 1, param_columns].iloc[0]
    delta = nxt - current
    l2_norm = float(np.linalg.norm(delta.to_numpy()))

    summary = {f"delta_{col}": float(delta[col]) for col in param_columns}
    summary["delta_l2_norm"] = l2_norm
    summary["iteration"] = jitter_iteration

    # Add run-level regret summaries (repeated per jitter_iteration row)
    summary["final_best_true"] = float(results["best_true_so_far"].iloc[-1])
    summary["final_simple_regret_true"] = float(results["simple_regret_true"].iloc[-1])
    summary["final_cum_regret_true"] = float(results["regret_cum_true"].iloc[-1])
    summary["final_avg_regret_true"] = float(results["regret_avg_true"].iloc[-1])

    sr = results["simple_regret_true"].to_numpy(dtype=float)
    summary["auc_simple_regret_true"] = float(np.trapezoid(sr, dx=1.0))

    return pd.Series(summary)


def run_single_seed(
    seed: int,
    dataset: DatasetConfig,
    objective: str,
    oracle_models: list[str],
    acquisitions: list[AcquisitionConfig],
    error_models: list[str],
    jitter_stds: list[float],
    jitter_iterations: list[int],
    df: pd.DataFrame,
    bounds: Bounds,
    args: argparse.Namespace,
    weights: np.ndarray | None,
    ref_point: np.ndarray | None,
    progress_q: object | None = None,          # multiprocessing.Manager().Queue() in parallel mode
    progress_update: callable | None = None,   # tqdm.update in sequential mode
) -> tuple[list[pd.Series], int]:
    """Run all simulations for a single seed."""
    summaries: list[pd.Series] = []
    run_count = 0

    def _tick() -> None:
        nonlocal run_count
        run_count += 1
        if progress_q is not None:
            try:
                progress_q.put(1)
            except Exception:
                pass
        if progress_update is not None:
            try:
                progress_update(1)
            except Exception:
                pass

    for oracle_model in oracle_models:
        objective_columns = dataset.objective_map[objective]
        oracle = build_oracle(
            df=df,
            objective=objective,
            objective_columns=objective_columns,
            param_columns=dataset.param_columns,
            seed=seed,  # use the actual seed for this run
            normalize=args.normalize_objective,
            weights=weights,
            oracle_model=oracle_model,
            oracle_augmentation=args.oracle_augmentation,
            oracle_augment_repeats=args.oracle_augment_repeats,
            oracle_augment_std=args.oracle_augment_std,
            oracle_fast=args.oracle_fast,
        )

        if objective == "multi_objective":
            if ref_point is None:
                raise ValueError("ref_point must be provided for multi_objective.")
            y_opt = estimate_oracle_hypervolume(
                oracle=oracle,
                bounds=bounds,
                seed=args.oracle_opt_seed,
                n=args.oracle_opt_samples,
                batch_size=args.oracle_opt_batch_size,
                ref_point=ref_point,
            )
        else:
            y_opt = estimate_oracle_optimum(
                oracle=oracle,
                bounds=bounds,
                seed=args.oracle_opt_seed,
                n=args.oracle_opt_samples,
                batch_size=args.oracle_opt_batch_size,
            )

        base_config = SimulationConfig(
            iterations=args.iterations,
            jitter_iteration=args.jitter_iteration,
            jitter_std=args.jitter_std,
            single_error=args.single_error,
            initial_samples=args.initial_samples,
            candidate_pool=args.candidate_pool,
            objective=objective,
            objective_columns=objective_columns,
            param_columns=dataset.param_columns,
            seed=seed,
            error_model=args.error_model,
            error_bias=args.error_bias,
            error_spike_prob=args.error_spike_prob,
            error_spike_std=args.error_spike_std,
            dropout_strategy=args.dropout_strategy,
            normalize_objective=args.normalize_objective,
            objective_weights=weights,
            acq_num_restarts=args.acq_num_restarts,
            acq_raw_samples=args.acq_raw_samples,
            acq_maxiter=args.acq_maxiter,
            acq_mc_samples=args.acq_mc_samples,
            ref_point=ref_point,
        )

        for acq in acquisitions:
            # Baseline run
            if args.baseline_run:
                baseline_run_id = str(uuid.uuid4())
                run_rng = np.random.default_rng(seed)
                torch.manual_seed(seed)

                config = dataclasses.replace(base_config, seed=seed)
                run_start = time.perf_counter()
                baseline_results = run_simulation(
                    oracle=oracle,
                    bounds=bounds,
                    config=config,
                    acq=acq,
                    rng=run_rng,
                    jitter_rng=None,
                    run_id=baseline_run_id,
                    apply_error=False,
                    oracle_model=oracle_model,
                    y_opt=y_opt,
                )
                baseline_results["dataset"] = dataset.name
                baseline_runtime = time.perf_counter() - run_start

                results_path = args.output_dir / (
                    f"bo_sensor_error_{dataset.name}_{objective}_{acq.name}_seed{seed}_baseline_{oracle_model}.csv"
                )
                baseline_results.to_csv(results_path, index=False)

                for jitter_iteration in jitter_iterations:
                    summary = summarize_adjustment(
                        baseline_results,
                        jitter_iteration,
                        dataset.param_columns,
                    )
                    summary["acquisition"] = acq.name
                    summary["objective"] = objective
                    summary["jitter_std"] = float(baseline_results["jitter_std"].iloc[0])
                    summary["jitter_iteration"] = int(jitter_iteration)
                    summary["iterations"] = int(args.iterations)
                    summary["seed"] = int(seed)
                    summary["run_id"] = baseline_run_id
                    summary["error_model"] = str(baseline_results["error_model"].iloc[0])
                    summary["oracle_model"] = oracle_model
                    summary["baseline"] = True
                    summary["xi"] = float(acq.xi)
                    summary["kappa"] = float(acq.kappa)
                    summary["runtime_sec"] = float(baseline_runtime)
                    summary["y_opt"] = float(y_opt)
                    summary["dataset"] = dataset.name
                    summary["param_columns"] = ",".join(dataset.param_columns)
                    summaries.append(summary)

                _tick()

            # Jittered runs
            for error_model in error_models:
                for jitter_std in jitter_stds:
                    for jitter_iteration in jitter_iterations:
                        run_id = str(uuid.uuid4())
                        run_rng = np.random.default_rng(seed)
                        torch.manual_seed(seed)

                        jitter_seed = np.random.SeedSequence(
                            [
                                seed,
                                ACQUISITION_CHOICES.index(acq.name),
                                int(jitter_iteration),
                                int(round(float(jitter_std) * 1_000_000)),
                                ERROR_MODEL_CHOICES.index(error_model),
                            ]
                        )
                        jitter_rng = np.random.default_rng(jitter_seed)

                        config = dataclasses.replace(
                            base_config,
                            seed=seed,
                            error_model=error_model,
                            jitter_std=float(jitter_std),
                            jitter_iteration=int(jitter_iteration),
                        )

                        run_start = time.perf_counter()
                        results = run_simulation(
                            oracle=oracle,
                            bounds=bounds,
                            config=config,
                            acq=acq,
                            rng=run_rng,
                            jitter_rng=jitter_rng,
                            run_id=run_id,
                            apply_error=True,
                            oracle_model=oracle_model,
                            y_opt=y_opt,
                        )
                        results["dataset"] = dataset.name
                        run_runtime = time.perf_counter() - run_start

                        results_path = args.output_dir / (
                            f"bo_sensor_error_{dataset.name}_{objective}_{acq.name}_seed{seed}_jittered_"
                            f"{oracle_model}_{error_model}_jit{jitter_iteration}_std{jitter_std}.csv"
                        )
                        results.to_csv(results_path, index=False)

                        summary = summarize_adjustment(
                            results,
                            int(jitter_iteration),
                            dataset.param_columns,
                        )
                        summary["acquisition"] = acq.name
                        summary["objective"] = objective
                        summary["jitter_std"] = float(results["jitter_std"].iloc[0])
                        summary["jitter_iteration"] = int(jitter_iteration)
                        summary["iterations"] = int(args.iterations)
                        summary["seed"] = int(seed)
                        summary["run_id"] = run_id
                        summary["error_model"] = str(results["error_model"].iloc[0])
                        summary["oracle_model"] = oracle_model
                        summary["baseline"] = False
                        summary["xi"] = float(acq.xi)
                        summary["kappa"] = float(acq.kappa)
                        summary["runtime_sec"] = float(run_runtime)
                        summary["y_opt"] = float(y_opt)
                        summary["dataset"] = dataset.name
                        summary["param_columns"] = ",".join(dataset.param_columns)
                        summaries.append(summary)

                        _tick()

    return summaries, run_count



def main() -> None:
    args = parse_args()
    validate_inputs(args)

    error_models = parse_error_models(args.error_model, args.error_models)
    jitter_stds = parse_float_list(args.jitter_stds, args.jitter_std)
    jitter_iterations = parse_int_list(args.jitter_iterations, args.jitter_iteration)
    validate_sweeps(jitter_iterations, jitter_stds, args.iterations)

    oracle_models = parse_oracle_models(args.oracle_model, args.oracle_models)
    seeds = parse_seed_list(args.seeds, args.seed, args.num_seeds)

    acquisition_names = parse_acquisition_list(args.acq, args.acq_list)

    dataset_configs = parse_dataset_configs(
        args.data_dir,
        args.dataset_config,
        args.dataset_cache_dir,
    )
    if args.combine_datasets:
        combined = combine_dataset_configs(dataset_configs)
        if combined is not None:
            dataset_configs.append(combined)

    output_dir = ensure_output_dir(args.output_dir)
    runtime_start = time.perf_counter()

    baseline_runs_per_objective = len(acquisition_names) * len(seeds) * len(oracle_models) if args.baseline_run else 0
    jittered_runs_per_objective = (
        len(acquisition_names)
        * len(seeds)
        * len(oracle_models)
        * len(error_models)
        * len(jitter_stds)
        * len(jitter_iterations)
    )
    dataset_objectives: dict[str, list[str]] = {}
    total_runs = 0
    for dataset in dataset_configs:
        objective_names = parse_objective_list(
            args.objective,
            args.objectives,
            dataset.objective_map,
        )
        dataset_objectives[dataset.name] = objective_names
        total_runs += (baseline_runs_per_objective + jittered_runs_per_objective) * len(objective_names)

    use_parallel = args.parallel or (len(seeds) > 1 and not args.parallel)

    if args.n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif args.n_jobs == -2:
        n_jobs = max(1, mp.cpu_count() - 1)
    elif args.n_jobs > 0:
        n_jobs = min(args.n_jobs, mp.cpu_count())
    else:
        n_jobs = 1
        use_parallel = False

    n_jobs = min(n_jobs, len(seeds))

    print(f"Running {len(seeds)} seed(s) with {total_runs} total simulation runs")
    if use_parallel and len(seeds) > 1:
        print(f"Using parallel processing with {n_jobs} worker(s)")
    else:
        print("Using sequential processing")

    summaries: list[pd.Series] = []

    progress = tqdm(total=total_runs, desc="Simulation runs", unit="run")

    for dataset in dataset_configs:
        objective_names = dataset_objectives[dataset.name]
        for objective_name in objective_names:
            objective_columns = dataset.objective_map[objective_name]
            weights = (
                parse_objective_weights(args.objective_weights, objective_name, objective_columns)
                if objective_name != "multi_objective"
                else None
            )
            df = load_observations(dataset, objective_name, args.user_id, args.group_id)
            bounds = bounds_from_data(df, dataset.param_columns)
            ref_point = compute_reference_point(df, objective_name, objective_columns)

            filtered_acq_names = filter_acquisitions_for_objective(acquisition_names, objective_name)
            acquisitions = [
                AcquisitionConfig(name=n, xi=args.xi, kappa=args.kappa) for n in filtered_acq_names
            ]

            if use_parallel and len(seeds) > 1:
                manager = mp.Manager()
                progress_q = manager.Queue()

                def _progress_monitor(q, pbar):
                    while True:
                        msg = q.get()
                        if msg is None:
                            break
                        try:
                            pbar.update(int(msg))
                        except Exception:
                            pass

                monitor = threading.Thread(target=_progress_monitor, args=(progress_q, progress), daemon=True)
                monitor.start()

                try:
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        futures = {
                            executor.submit(
                                run_single_seed,
                                seed,
                                dataset,
                                objective_name,
                                oracle_models,
                                acquisitions,
                                error_models,
                                jitter_stds,
                                jitter_iterations,
                                df,
                                bounds,
                                args,
                                weights,
                                ref_point,
                                progress_q,
                                None,
                            ): seed
                            for seed in seeds
                        }

                        for future in as_completed(futures):
                            seed = futures[future]
                            try:
                                seed_summaries, _ = future.result()
                                summaries.extend(seed_summaries)
                            except Exception as e:
                                print(f"\nError processing seed {seed}: {e}")
                                import traceback
                                traceback.print_exc()
                finally:
                    try:
                        progress_q.put(None)
                    except Exception:
                        pass
                    try:
                        monitor.join(timeout=10)
                    except Exception:
                        pass
                    try:
                        manager.shutdown()
                    except Exception:
                        pass

            else:
                try:
                    for seed in seeds:
                        seed_summaries, _ = run_single_seed(
                            seed,
                            dataset,
                            objective_name,
                            oracle_models,
                            acquisitions,
                            error_models,
                            jitter_stds,
                            jitter_iterations,
                            df,
                            bounds,
                            args,
                            weights,
                            ref_point,
                            progress_q=None,
                            progress_update=progress.update,
                        )
                        summaries.extend(seed_summaries)
                finally:
                    pass

    progress.close()

    if not summaries:
        print("No simulation runs completed.")
        return

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "bo_sensor_error_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if args.baseline_run:
        jittered = summary_df[summary_df["baseline"] == False]
        baseline = summary_df[summary_df["baseline"] == True]

        merged = jittered.merge(
            baseline,
            on=[
                "dataset",
                "acquisition",
                "objective",
                "iterations",
                "jitter_iteration",
                "seed",
                "oracle_model",
                "xi",
                "kappa",
                "param_columns",
            ],
            suffixes=("_jitter", "_baseline"),
        )

        excess_rows: list[pd.Series] = []
        for _, row in merged.iterrows():
            param_columns = row["param_columns"].split(",")
            excess_entry = {
                "dataset": row["dataset"],
                "objective": row["objective"],
                "acquisition": row["acquisition"],
                "oracle_model": row["oracle_model"],
                "error_model": row["error_model_jitter"],
                "jitter_std": row["jitter_std_jitter"],
                "jitter_iteration": row["jitter_iteration"],
                "seed": row["seed"],
                "param_columns": row["param_columns"],
            }
            for col in param_columns:
                excess_entry[f"delta_excess_{col}"] = (
                    row[f"delta_{col}_jitter"] - row[f"delta_{col}_baseline"]
                )
            excess_values = np.array(
                [excess_entry[f"delta_excess_{col}"] for col in param_columns],
                dtype=float,
            )
            excess_entry["delta_excess_l2_norm"] = float(np.linalg.norm(excess_values))
            excess_entry["final_simple_regret_excess_true"] = (
                row["final_simple_regret_true_jitter"] - row["final_simple_regret_true_baseline"]
            )
            excess_entry["final_cum_regret_excess_true"] = (
                row["final_cum_regret_true_jitter"] - row["final_cum_regret_true_baseline"]
            )
            excess_entry["final_avg_regret_excess_true"] = (
                row["final_avg_regret_true_jitter"] - row["final_avg_regret_true_baseline"]
            )
            excess_entry["auc_simple_regret_excess_true"] = (
                row["auc_simple_regret_true_jitter"] - row["auc_simple_regret_true_baseline"]
            )
            excess_rows.append(pd.Series(excess_entry))

        merged_excess = pd.DataFrame(excess_rows)

        merged_excess_path = output_dir / "bo_sensor_error_excess_summary.csv"
        merged_excess.to_csv(merged_excess_path, index=False)

        comparison_metrics = [
            "delta_excess_l2_norm",
            "final_simple_regret_excess_true",
            "final_cum_regret_excess_true",
            "final_avg_regret_excess_true",
            "auc_simple_regret_excess_true",
        ]
        dataset_stats = (
            merged_excess.groupby(
                [
                    "dataset",
                    "objective",
                    "acquisition",
                    "error_model",
                    "jitter_iteration",
                    "jitter_std",
                    "oracle_model",
                ]
            )
            .agg(
                **{f"{metric}_mean": (metric, "mean") for metric in comparison_metrics},
                **{f"{metric}_std": (metric, "std") for metric in comparison_metrics},
                runs=("delta_excess_l2_norm", "count"),
            )
            .reset_index()
        )
        overall_stats = (
            merged_excess.groupby(
                [
                    "objective",
                    "acquisition",
                    "error_model",
                    "jitter_iteration",
                    "jitter_std",
                    "oracle_model",
                ]
            )
            .agg(
                **{f"{metric}_mean": (metric, "mean") for metric in comparison_metrics},
                **{f"{metric}_std": (metric, "std") for metric in comparison_metrics},
                runs=("delta_excess_l2_norm", "count"),
            )
            .reset_index()
        )
        overall_stats.insert(0, "dataset", "all")
        dataset_comparison = pd.concat([dataset_stats, overall_stats], ignore_index=True)
        dataset_comparison_path = output_dir / "bo_sensor_error_dataset_effects.csv"
        dataset_comparison.to_csv(dataset_comparison_path, index=False)

    stats = (
        summary_df.groupby(
            [
                "dataset",
                "objective",
                "acquisition",
                "baseline",
                "error_model",
                "jitter_iteration",
                "jitter_std",
                "oracle_model",
            ]
        )
        .agg(
            delta_l2_mean=("delta_l2_norm", "mean"),
            delta_l2_std=("delta_l2_norm", "std"),
            final_simple_regret_mean=("final_simple_regret_true", "mean"),
            final_simple_regret_std=("final_simple_regret_true", "std"),
            final_cum_regret_mean=("final_cum_regret_true", "mean"),
            final_cum_regret_std=("final_cum_regret_true", "std"),
            final_avg_regret_mean=("final_avg_regret_true", "mean"),
            final_avg_regret_std=("final_avg_regret_true", "std"),
            auc_simple_regret_mean=("auc_simple_regret_true", "mean"),
            auc_simple_regret_std=("auc_simple_regret_true", "std"),
            runs=("delta_l2_norm", "count"),
        )
        .reset_index()
    )
    stats_path = output_dir / "bo_sensor_error_summary_stats.csv"
    stats.to_csv(stats_path, index=False)

    write_run_config(
        output_dir=output_dir,
        dataset_configs=dataset_configs,
        objectives=dataset_objectives,
        acquisition_names=acquisition_names,
        error_models=error_models,
        oracle_models=oracle_models,
        seeds=seeds,
        args=args,
    )

    metadata_payload = {
        "args": vars(args),
        "runtime_sec": float(time.perf_counter() - runtime_start),
        "parallel_execution": bool(use_parallel and len(seeds) > 1),
        "n_workers": int(n_jobs) if (use_parallel and len(seeds) > 1) else 1,
        "datasets": [
            {
                "name": dataset.name,
                "data_dirs": [str(path) for path in dataset.data_dirs],
                "param_columns": dataset.param_columns,
                "objective_map": dataset.objective_map,
                "observation_glob": dataset.observation_glob,
            }
            for dataset in dataset_configs
        ],
        "objectives": dataset_objectives,
        "error_models": error_models,
        "oracle_models": oracle_models,
        "acquisitions": acquisition_names,
        "seeds": seeds,
        "package_versions": collect_package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "matplotlib",
                "seaborn",
                "tqdm",
                "xgboost",
                "lightgbm",
                "statsmodels",
                "botorch",
                "torch",
                #"catboost",
                "tabpfn",
            ]
        ),
    }

    def json_fallback(obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, default=json_fallback))

    total_runtime = time.perf_counter() - runtime_start
    print("\nSimulation complete.")
    print(f"Total runtime: {total_runtime:.1f}s")
    if use_parallel and len(seeds) > 1:
        print(f"Parallel speedup with {n_jobs} workers")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    
    main()
