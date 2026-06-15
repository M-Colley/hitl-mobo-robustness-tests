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
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
try:
    from tabpfn import TabPFNRegressor
except ImportError:
    TabPFNRegressor = None

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
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
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
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "eHMI-bo-participantdata"
DEFAULT_DATASET_CONFIG_PATH = REPO_ROOT / "datasets.json"
DEFAULT_ORACLE_SELECTION_PATH = Path("output") / "best_oracle_models.json"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_DATASET_NAME = "default"
AUTO_ORACLE_MODEL = "auto"
OBSERVATION_SOURCE_COLUMN = "__source_file"
COLUMN_ALIASES = {
    "UserID": "User_ID",
    "GroupID": "Group_ID",
    "ConditionID": "Condition_ID",
}
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
# qlogehvi/qlognehvi are the numerically-stable log variants (Ament et al. 2023,
# arXiv:2310.20708). They share qEHVI/qNEHVI's box-decomposition memory cost, so
# at high objective counts they still need reduced sampling/concurrency, but they
# avoid the vanishing-gradient pathologies BoTorch warns about for the plain ones.
MULTI_ACQUISITION_CHOICES = ["qehvi", "qnehvi", "qlogehvi", "qlognehvi"]
# Model-free floors: candidates are independent of observations, so they bound
# what "no learning" achieves and anchor the robustness rankings.
BASELINE_ACQUISITION_CHOICES = ["random", "sobol"]
ACQUISITION_CHOICES = (
    SINGLE_ACQUISITION_CHOICES + MULTI_ACQUISITION_CHOICES + BASELINE_ACQUISITION_CHOICES
)
ERROR_MODEL_CHOICES = ["gaussian", "bias", "dropout", "spike", "drift", "ar1"]

DEFAULT_ERROR_MODELS = "gaussian,bias"
DEFAULT_JITTER_ITERATIONS = "10,20,40"
DEFAULT_JITTER_STDS = "0.05,0.5,1,5"
INCUMBENT_CHOICES = ["posterior_mean", "observed_max"]
ORACLE_MODEL_CHOICES = [
    "xgboost",
    "lightgbm",
    "catboost",
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

    # Human-plausible error-model extensions
    error_ar1_rho: float = 0.8
    response_clip_low: np.ndarray | None = None
    response_clip_high: np.ndarray | None = None
    response_round: float | None = None

    # Incumbent definition for improvement-based acquisitions.
    # "posterior_mean" avoids the noisy-max pitfall where a single positive
    # noise spike inflates best_f beyond any achievable value.
    incumbent: str = "posterior_mean"


@dataclasses.dataclass
class DatasetConfig:
    name: str
    data_dirs: list[Path]
    param_columns: list[str]
    objective_map: dict[str, list[str]]
    observation_glob: str = "ObservationsPerEvaluation.csv"


@dataclasses.dataclass(frozen=True)
class ObjectiveNormalization:
    min_vals: np.ndarray
    ranges: np.ndarray


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

    parser.add_argument(
        "--jitter-iteration",
        type=int,
        default=None,
        help="Single jitter onset iteration. Mutually exclusive with --jitter-iterations. "
        "Use 0 to apply noise to every observation after the first.",
    )
    parser.add_argument(
        "--jitter-std",
        type=float,
        default=None,
        help="Single jitter std. Mutually exclusive with --jitter-stds.",
    )
    parser.add_argument(
        "--single-error",
        action="store_true",
        default=False,
        help="Apply sensor error only once at the first iteration after jitter-iteration.",
    )
    parser.add_argument(
        "--jitter-iterations",
        type=str,
        default=None,
        help=f"Comma-separated jitter onset sweep (default: {DEFAULT_JITTER_ITERATIONS}).",
    )
    parser.add_argument(
        "--jitter-stds",
        type=str,
        default=None,
        help=f"Comma-separated jitter std sweep (default: {DEFAULT_JITTER_STDS}).",
    )

    parser.add_argument("--initial-samples", type=int, default=5)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=1000,
        help="Number of random candidates screened to seed acquisition optimization restarts.",
    )

    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--objectives", type=str, default=None)

    parser.add_argument("--acq", type=str, default="all", choices=ACQUISITION_CHOICES + ["all"])
    parser.add_argument("--acq-list", type=str, default=None)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num-seeds", type=int, default=5)

    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional local dataset directory or remote Git repository URL.",
    )
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

    parser.add_argument(
        "--error-model",
        type=str,
        default=None,
        choices=ERROR_MODEL_CHOICES + ["all"],
        help="Single error model. Mutually exclusive with --error-models.",
    )
    parser.add_argument(
        "--error-models",
        type=str,
        default=None,
        help=f"Comma-separated error models (default: {DEFAULT_ERROR_MODELS}).",
    )

    parser.add_argument("--error-bias", type=float, default=0.2)
    parser.add_argument("--error-spike-prob", type=float, default=0.1)
    parser.add_argument("--error-spike-std", type=float, default=0.5)
    parser.add_argument("--dropout-strategy", type=str, default="hold_last", choices=["hold_last"])
    parser.add_argument(
        "--error-ar1-rho",
        type=float,
        default=0.8,
        help="Autocorrelation coefficient for the ar1 error model.",
    )
    parser.add_argument(
        "--response-clip",
        type=str,
        default="none",
        help="Clip noisy observations to the response scale: 'none', 'auto' "
        "(use the data min/max per objective), or 'low,high' explicit bounds.",
    )
    parser.add_argument(
        "--response-round",
        type=float,
        default=None,
        help="Round noisy observations to this granularity (e.g. the rating-scale step).",
    )
    parser.add_argument(
        "--incumbent",
        type=str,
        default="posterior_mean",
        choices=INCUMBENT_CHOICES,
        help="Incumbent (best_f) definition for improvement-based acquisitions. "
        "'posterior_mean' is robust to noisy observations; 'observed_max' is the "
        "classic noise-naive choice.",
    )
    parser.add_argument(
        "--min-oracle-r2",
        type=float,
        default=None,
        help="Refuse to run when an auto-selected oracle's cross-validated R^2 is below "
        "this threshold. A warning is always printed below 0.3.",
    )

    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--group-id", type=str, default=None)

    parser.add_argument("--normalize-objective", action="store_true", default=False)
    parser.add_argument("--objective-weights", type=str, default=None)

    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=2.0)

    parser.add_argument(
        "--oracle-model",
        type=str,
        default="extra_trees",
        choices=ORACLE_MODEL_CHOICES + ["all", AUTO_ORACLE_MODEL],
    )
    parser.add_argument("--oracle-models", type=str, default=None)
    parser.add_argument(
        "--oracle-selection-path",
        type=Path,
        default=DEFAULT_ORACLE_SELECTION_PATH,
        help="JSON file produced by select_best_oracle_model.py for --oracle-model auto.",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Reuse per-run CSVs already present in --output-dir instead of "
        "re-simulating them. Safe because runs are fully seeded and "
        "deterministic; summaries are rebuilt from the loaded files. Use to "
        "continue a multi-day sweep after an interruption (e.g. reboot).",
    )

    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_observation_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    last_df: pd.DataFrame | None = None
    for sep in (";", ","):
        df = pd.read_csv(path, sep=sep)
        df.columns = df.columns.str.strip()
        df = _normalize_observation_columns(df)
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


def _normalize_observation_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_qehvi_columns(df)
    rename_map = {
        source: target
        for source, target in COLUMN_ALIASES.items()
        if source in df.columns and target not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _objective_base_column(column: str) -> str:
    return column[1:] if column.startswith("-") else column


def _objective_sign(column: str) -> float:
    return -1.0 if column.startswith("-") else 1.0


def _objective_required_columns(objective_columns: list[str]) -> list[str]:
    return [_objective_base_column(column) for column in objective_columns]


def _extract_objective_values(df: pd.DataFrame, objective_columns: list[str]) -> np.ndarray:
    base_columns = _objective_required_columns(objective_columns)
    values = df[base_columns].to_numpy(dtype=float)
    signs = np.asarray([_objective_sign(column) for column in objective_columns], dtype=float)
    return values * signs


def _objective_output_name(column: str) -> str:
    if column.startswith("-"):
        return f"neg_{_objective_base_column(column)}"
    return column


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

    objective_columns = dataset.objective_map[objective]
    required_columns = dataset.param_columns + _objective_required_columns(objective_columns)
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = _read_observation_csv(path, required_columns)
        frame[OBSERVATION_SOURCE_COLUMN] = str(path.resolve())
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    df = _normalize_observation_columns(df)

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["User_ID", "Group_ID"]:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            # Keep alphanumeric IDs (e.g. "0F") as strings: coercing them to
            # NaN would silently disable per-user grouping and filtering.
            if numeric.notna().sum() >= df[column].notna().sum():
                df[column] = numeric

    if user_id is not None and "User_ID" not in df.columns:
        raise ValueError("User_ID column missing from observations; cannot filter by --user-id.")
    if group_id is not None and "Group_ID" not in df.columns:
        raise ValueError("Group_ID column missing from observations; cannot filter by --group-id.")

    def _filter_by_id(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
        if pd.api.types.is_numeric_dtype(frame[column]):
            return frame[frame[column] == float(value)]
        return frame[frame[column].astype(str) == str(value)]

    if user_id is not None:
        df = _filter_by_id(df, "User_ID", user_id)
    if group_id is not None:
        df = _filter_by_id(df, "Group_ID", group_id)

    df = df.dropna(subset=dataset.param_columns + _objective_required_columns(objective_columns))
    if df.empty:
        raise ValueError("No data remaining after applying user/group filters.")
    return df.reset_index(drop=True)


def compute_objective(
    df: pd.DataFrame,
    objective_columns: list[str],
    normalize: bool,
    weights: np.ndarray | None,
    normalization: ObjectiveNormalization | None = None,
) -> pd.Series:
    values = _extract_objective_values(df, objective_columns)

    if normalize:
        stats = normalization or fit_objective_normalization(df, objective_columns)
        values = normalize_objective_values(values, stats)

    if weights is None:
        return pd.Series(values.mean(axis=1), index=df.index)

    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return pd.Series(values @ weights, index=df.index)


def compute_objective_matrix(
    df: pd.DataFrame,
    objective_columns: list[str],
    normalize: bool,
    normalization: ObjectiveNormalization | None = None,
) -> np.ndarray:
    values = _extract_objective_values(df, objective_columns)

    if normalize:
        stats = normalization or fit_objective_normalization(df, objective_columns)
        values = normalize_objective_values(values, stats)

    return values


def fit_objective_normalization(
    df: pd.DataFrame,
    objective_columns: list[str],
) -> ObjectiveNormalization:
    values = _extract_objective_values(df, objective_columns)
    min_vals = np.nanmin(values, axis=0)
    max_vals = np.nanmax(values, axis=0)
    ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    return ObjectiveNormalization(min_vals=min_vals, ranges=ranges)


def normalize_objective_values(
    values: np.ndarray,
    normalization: ObjectiveNormalization,
) -> np.ndarray:
    return (values - normalization.min_vals) / normalization.ranges


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
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return X, y
    if repeats < 1:
        return X, y
    if mode != "jitter":
        raise ValueError(f"Unknown oracle augmentation mode: {mode}")

    if low is None:
        low = np.min(X, axis=0)
    if high is None:
        high = np.max(X, axis=0)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    feature_scales = np.where(high > low, (high - low) * noise_std, 0.0)

    augmented_X = [X]
    augmented_y = [y]
    for _ in range(repeats):
        noise = rng.normal(0.0, feature_scales, size=X.shape)
        jittered = np.clip(X + noise, low, high)
        augmented_X.append(jittered)
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
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    if oracle_fast:
        # Keep in sync with select_best_oracle_model.py's fast mode so that
        # oracle-selection results transfer to the simulation.
        tree_scale = 0.7
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
            low=low,
            high=high,
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
            print(f"  R^2 score: {train_score:.4f}")
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
        low=low,
        high=high,
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
    print(f"  R^2 score: {train_score:.4f}")
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
        if XGBRegressor is None:
            raise ImportError("xgboost is required for oracle-model=xgboost. Install it via requirements.txt.")
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
        if LGBMRegressor is None:
            raise ImportError("lightgbm is required for oracle-model=lightgbm. Install it via requirements.txt.")
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
        if CatBoostRegressor is None:
            raise ImportError("catboost is required for oracle-model=catboost. Install it via requirements.txt.")
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
        if TabPFNRegressor is None:
            raise ImportError("tabpfn is required for oracle-model=tabpfn. Install it via requirements.txt.")
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
    X_known: np.ndarray | None = None,
) -> float:
    """Estimate max of the oracle by random search, optionally anchored on
    known points (e.g. the training data) so the estimate is never below the
    oracle's value at any observed design."""
    rng = np.random.default_rng(seed)
    best = -np.inf
    d = len(bounds.low)

    if X_known is not None and len(X_known) > 0:
        best = max(best, float(np.max(oracle.predict_many(np.asarray(X_known, dtype=float)))))

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
    X_known: np.ndarray | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    d = len(bounds.low)
    collected: list[np.ndarray] = []
    if X_known is not None and len(X_known) > 0:
        collected.append(oracle.predict_many(np.asarray(X_known, dtype=float)))

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
    # random/sobol are model-free and valid for every objective type.
    if objective == "multi_objective":
        allowed = MULTI_ACQUISITION_CHOICES + BASELINE_ACQUISITION_CHOICES
        filtered = [a for a in acquisitions if a in allowed]
        if not [a for a in filtered if a in MULTI_ACQUISITION_CHOICES] and not filtered:
            raise ValueError(
                "Multi-objective optimization requires one of "
                f"{', '.join(allowed)}."
            )
        return filtered
    allowed = SINGLE_ACQUISITION_CHOICES + BASELINE_ACQUISITION_CHOICES
    filtered = [a for a in acquisitions if a in allowed]
    if not filtered:
        raise ValueError(
            "Single-objective optimization requires one of "
            f"{', '.join(allowed)}."
        )
    return filtered


def parse_error_models(error_model: str | None, error_models: str | None) -> list[str]:
    if error_model is not None and error_models is not None:
        raise ValueError("Pass either --error-model or --error-models, not both.")
    raw = error_models if error_models is not None else error_model
    if raw is None:
        raw = DEFAULT_ERROR_MODELS
    if raw == "all":
        return ERROR_MODEL_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one error model must be specified.")
    unknown = [v for v in values if v not in ERROR_MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unknown error model(s): {', '.join(unknown)}")
    return values


def resolve_sweep_values(
    plural: str | None,
    singular: float | int | None,
    default: str,
    parse: callable,
    flag_names: tuple[str, str],
) -> list:
    """Resolve a sweep from plural/singular CLI flags, refusing ambiguous input."""
    if plural is not None and singular is not None:
        raise ValueError(f"Pass either {flag_names[0]} or {flag_names[1]}, not both.")
    if plural is not None:
        return parse(plural)
    if singular is not None:
        return [singular]
    return parse(default)


def parse_oracle_models(oracle_model: str, oracle_models: str | None) -> list[str]:
    raw = oracle_models if oracle_models is not None else oracle_model
    if raw == "all":
        return ORACLE_MODEL_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one oracle model must be specified.")
    if AUTO_ORACLE_MODEL in values:
        if len(values) != 1:
            raise ValueError("'auto' cannot be combined with explicit oracle models.")
        return [AUTO_ORACLE_MODEL]
    unknown = [v for v in values if v not in ORACLE_MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unknown oracle model(s): {', '.join(unknown)}")
    return values


def load_oracle_selection(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Oracle selection file not found: {path}. "
            "Run select_best_oracle_model.py first or pass --oracle-model(s) explicitly."
        )
    payload = json.loads(path.read_text())
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Oracle selection file must contain a top-level 'datasets' list.")

    selection: dict[tuple[str, str], dict[str, object]] = {}
    for dataset_entry in datasets:
        if not isinstance(dataset_entry, dict):
            raise ValueError("Each dataset entry in the oracle selection file must be an object.")
        dataset_name = str(dataset_entry.get("name"))
        objectives = dataset_entry.get("objectives")
        if not isinstance(objectives, dict):
            raise ValueError(f"Dataset '{dataset_name}' is missing an 'objectives' mapping.")
        for objective_name, objective_entry in objectives.items():
            if not isinstance(objective_entry, dict):
                raise ValueError(
                    f"Objective '{objective_name}' for dataset '{dataset_name}' must be an object."
                )
            best_model = objective_entry.get("best_model")
            if not isinstance(best_model, str):
                raise ValueError(
                    f"Objective '{objective_name}' for dataset '{dataset_name}' is missing 'best_model'."
                )
            selection[(dataset_name, str(objective_name))] = objective_entry
    return selection


def resolve_oracle_models_for_objective(
    requested_models: list[str],
    dataset_name: str,
    objective_name: str,
    oracle_selection: dict[tuple[str, str], dict[str, object]] | None = None,
) -> list[str]:
    if requested_models != [AUTO_ORACLE_MODEL]:
        return requested_models
    if oracle_selection is None:
        raise ValueError("Oracle selection data must be provided when using auto oracle mode.")
    selection_entry = oracle_selection.get((dataset_name, objective_name))
    if selection_entry is None:
        raise KeyError(
            f"No auto-selected oracle found for dataset='{dataset_name}', objective='{objective_name}'."
        )
    best_model = selection_entry.get("best_model")
    if not isinstance(best_model, str):
        raise ValueError(
            f"Auto-selection entry for dataset='{dataset_name}', objective='{objective_name}' is invalid."
        )
    return [best_model]


def infer_oracle_groups(df: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    if "User_ID" in df.columns:
        user_ids = df["User_ID"].dropna().astype(str)
        if user_ids.nunique() >= 2:
            return df["User_ID"].astype(str).to_numpy(), "User_ID"
    if "Group_ID" in df.columns:
        group_ids = df["Group_ID"].dropna().astype(str)
        if group_ids.nunique() >= 2:
            return df["Group_ID"].astype(str).to_numpy(), "Group_ID"
    if OBSERVATION_SOURCE_COLUMN in df.columns:
        source_ids = df[OBSERVATION_SOURCE_COLUMN].astype(str)
        if source_ids.nunique() >= 2:
            return source_ids.to_numpy(), OBSERVATION_SOURCE_COLUMN
    return None, "row"


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
            resolved.append(Path(entry).expanduser())
    return resolved


def _default_dataset_config() -> Path | None:
    if not DEFAULT_DATASET_CONFIG_PATH.is_file():
        return None
    return DEFAULT_DATASET_CONFIG_PATH


def _load_dataset_payloads(dataset_config_path: Path) -> list[dict]:
    payload = json.loads(dataset_config_path.read_text())
    if isinstance(payload, dict) and "datasets" in payload:
        dataset_payloads = payload["datasets"]
    elif isinstance(payload, list):
        dataset_payloads = payload
    else:
        raise ValueError("Dataset config must be a list or a dict with a 'datasets' key.")
    return dataset_payloads


def parse_dataset_configs(
    data_dir: str | Path | None,
    dataset_config_path: Path | None,
    cache_dir: Path,
) -> list[DatasetConfig]:
    if dataset_config_path is None:
        fallback_config_path = _default_dataset_config() if data_dir is None else None
        if fallback_config_path is not None:
            dataset_config_path = fallback_config_path
        else:
            raw_data_dir = str(data_dir) if data_dir is not None else str(DATA_DIR)
            resolved_dirs = resolve_data_dirs([raw_data_dir], cache_dir)
            return [
                DatasetConfig(
                    name=DEFAULT_DATASET_NAME,
                    data_dirs=resolved_dirs,
                    param_columns=list(PARAM_COLUMNS),
                    objective_map={k: list(v) for k, v in OBJECTIVE_MAP.items()},
                    observation_glob="ObservationsPerEvaluation.csv",
                )
            ]

    dataset_payloads = _load_dataset_payloads(dataset_config_path)
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
        print("Cannot combine datasets with different parameter columns.", file=sys.stderr)
        return None
    if any(dataset.observation_glob != first.observation_glob for dataset in datasets[1:]):
        print("Cannot combine datasets with different observation_glob patterns.", file=sys.stderr)
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
        print("No common objectives found to build a combined dataset.", file=sys.stderr)
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
        # 0 means: noise affects every observation after the first (the
        # human-plausible "noisy from the start" condition).
        if j < 0 or j >= iterations:
            raise ValueError("Each jitter-iteration must be within [0, iterations - 1].")
    for s in jitter_stds:
        if s < 0:
            raise ValueError("Each jitter-std must be >= 0.")


def parse_response_clip(
    response_clip: str,
    df: pd.DataFrame,
    objective: str,
    objective_columns: list[str],
    normalize: bool,
    weights: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve --response-clip into per-output low/high bounds.

    'auto' uses the observed data range of the objective (per column for
    multi_objective, of the scalarized objective otherwise) so that noisy
    observations stay on the instrument scale.
    """
    if response_clip == "none":
        return None, None
    if response_clip == "auto":
        if objective == "multi_objective":
            values = compute_objective_matrix(df, objective_columns, normalize)
            return np.nanmin(values, axis=0), np.nanmax(values, axis=0)
        values = compute_objective(df, objective_columns, normalize, weights).to_numpy(dtype=float)
        return np.array([np.nanmin(values)]), np.array([np.nanmax(values)])
    parts = [p.strip() for p in response_clip.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--response-clip must be 'none', 'auto', or 'low,high'.")
    low, high = float(parts[0]), float(parts[1])
    if low >= high:
        raise ValueError("--response-clip low must be smaller than high.")
    n_out = len(objective_columns) if objective == "multi_objective" else 1
    return np.full(n_out, low), np.full(n_out, high)


def validate_inputs(args: argparse.Namespace) -> None:
    if args.iterations <= 1:
        raise ValueError("iterations must be greater than 1.")
    if args.initial_samples < 1 or args.initial_samples >= args.iterations:
        raise ValueError("initial-samples must be within [1, iterations - 1].")
    if args.candidate_pool < 1:
        raise ValueError("candidate-pool must be >= 1.")
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
    normalize: bool = False,
) -> np.ndarray | None:
    if objective != "multi_objective":
        return None
    # The reference point must live in the same space as the oracle outputs:
    # when --normalize-objective is set the oracle predicts normalized values,
    # so the reference point is computed on normalized values as well.
    values = compute_objective_matrix(df, objective_columns, normalize)
    min_vals = np.nanmin(values, axis=0)
    max_vals = np.nanmax(values, axis=0)
    ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    return min_vals - 0.1 * ranges


def write_run_config(
    output_dir: Path,
    dataset_configs: list[DatasetConfig],
    objectives: dict[str, list[str]],
    acquisition_names: dict[tuple[str, str], list[str]],
    error_models: list[str],
    requested_oracle_models: list[str],
    resolved_oracle_models: dict[tuple[str, str], list[str]],
    seeds: list[int],
    args: argparse.Namespace,
    jitter_iterations: list[int] | None = None,
    jitter_stds: list[float] | None = None,
) -> None:
    dataset_lines = ["Datasets:"]
    for dataset in dataset_configs:
        dataset_lines.append(f"  - {dataset.name}: {', '.join(str(p) for p in dataset.data_dirs)}")

    objective_lines: list[str] = []
    for dataset_name, objective_names in objectives.items():
        objective_lines.append(f"  {dataset_name}: {', '.join(objective_names)}")
        for objective_name in objective_names:
            key = (dataset_name, objective_name)
            resolved = ", ".join(resolved_oracle_models[key])
            acquisitions = ", ".join(acquisition_names[key])
            objective_lines.append(f"    {objective_name}: oracles=[{resolved}] acquisitions=[{acquisitions}]")

    lines = [
        "Sensor-error simulation configuration",
        "=" * 60,
        "",
        *dataset_lines,
        "",
        "Objectives by dataset:",
        *objective_lines,
        f"Error models: {', '.join(error_models)}",
        f"Requested oracle models: {', '.join(requested_oracle_models)}",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "Core settings:",
        f"  iterations: {args.iterations}",
        f"  initial_samples: {args.initial_samples}",
        f"  candidate_pool: {args.candidate_pool}",
        f"  jitter_iterations: {jitter_iterations if jitter_iterations is not None else args.jitter_iterations}",
        f"  jitter_stds: {jitter_stds if jitter_stds is not None else args.jitter_stds}",
        f"  incumbent: {args.incumbent}",
        f"  response_clip: {args.response_clip}",
        f"  response_round: {args.response_round}",
        f"  single_error: {args.single_error}",
        f"  baseline_run: {args.baseline_run}",
        "",
        "Oracle settings:",
        f"  oracle_selection_path: {args.oracle_selection_path}",
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


def _postprocess_response(observed: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Map a noisy observation back onto the response instrument.

    Real raters produce bounded, discrete responses; without this step, large
    noise levels yield impossible ratings (e.g. -4.6 on a 1-7 scale).
    """
    if config.response_round is not None and config.response_round > 0:
        observed = np.round(observed / config.response_round) * config.response_round
    if config.response_clip_low is not None and config.response_clip_high is not None:
        observed = np.clip(observed, config.response_clip_low, config.response_clip_high)
    return observed


def apply_sensor_error(
    true_value: np.ndarray,
    iteration: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    previous_observed: np.ndarray,
    previous_error: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if iteration <= config.jitter_iteration:
        return true_value, np.zeros_like(true_value, dtype=float)
    if config.single_error and iteration != config.jitter_iteration + 1:
        return true_value, np.zeros_like(true_value, dtype=float)

    jitter = rng.normal(0.0, config.jitter_std, size=true_value.shape)

    if config.error_model == "gaussian":
        observed = true_value + jitter
    elif config.error_model == "bias":
        bias = np.full_like(true_value, config.error_bias, dtype=float)
        observed = true_value + bias + jitter
    elif config.error_model == "dropout":
        if config.dropout_strategy != "hold_last":
            raise ValueError(f"Unsupported dropout strategy: {config.dropout_strategy}")
        observed = previous_observed + jitter
    elif config.error_model == "spike":
        spike = np.zeros_like(true_value, dtype=float)
        if rng.random() < config.error_spike_prob:
            spike = rng.normal(0.0, config.error_spike_std, size=true_value.shape)
        observed = true_value + spike + jitter
    elif config.error_model == "drift":
        # Systematic drift/fatigue: bias ramps linearly from 0 at onset to
        # jitter_std at the final iteration (jitter_std doubles as the drift
        # magnitude so the sweep stays a single swept factor).
        span = max(1, config.iterations - config.jitter_iteration)
        ramp = config.jitter_std * (iteration - config.jitter_iteration) / span
        observed = true_value + np.full_like(true_value, ramp, dtype=float)
    elif config.error_model == "ar1":
        # Serially correlated rating error: e_t = rho * e_{t-1} + innovation,
        # with stationary SD equal to jitter_std.
        rho = float(config.error_ar1_rho)
        innovation = rng.normal(
            0.0, config.jitter_std * np.sqrt(max(0.0, 1.0 - rho**2)), size=true_value.shape
        )
        prev = (
            previous_error
            if previous_error is not None
            else np.zeros_like(true_value, dtype=float)
        )
        observed = true_value + rho * prev + innovation
    else:
        raise ValueError(f"Unknown error model: {config.error_model}")

    observed = _postprocess_response(observed, config)
    return observed, observed - true_value


def screen_candidate_pool(
    acqf: object,
    bounds: Bounds,
    rng: np.random.Generator,
    candidate_pool: int,
    num_restarts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be >= 1.")
    if num_restarts < 1:
        raise ValueError("num_restarts must be >= 1.")

    candidate_np = sample_uniform(bounds, rng, size=int(candidate_pool))
    candidate_tensor = torch.tensor(candidate_np, dtype=torch.double)

    with torch.no_grad():
        acq_values = acqf(candidate_tensor.unsqueeze(1)).reshape(-1)

    top_k = min(int(num_restarts), int(candidate_tensor.shape[0]))
    top_indices = torch.topk(acq_values, k=top_k).indices
    initial_conditions = candidate_tensor[top_indices].unsqueeze(1)
    best_candidate = initial_conditions[0].detach()
    return best_candidate, initial_conditions


def get_botorch_candidate(
    gp_model: SingleTaskGP | ModelListGP,
    acq_config: AcquisitionConfig,
    bounds: Bounds,
    bounds_tensor: torch.Tensor,
    best_f: float | None,
    num_restarts: int,
    raw_samples: int,
    maxiter: int,
    mc_samples: int,
    candidate_pool: int,
    rng: np.random.Generator,
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
    elif acq_config.name == "qlogehvi":
        if ref_point is None:
            raise ValueError("ref_point required for qlogehvi.")
        if not isinstance(gp_model, ModelListGP):
            raise ValueError("qlogehvi requires a multi-objective ModelListGP.")
        if isinstance(train_Y, list):
            train_Y_stack = torch.cat(train_Y, dim=1)
        else:
            train_Y_stack = train_Y
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(ref_point, dtype=torch.double),
            Y=train_Y_stack,
        )
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qLogExpectedHypervolumeImprovement(
            model=gp_model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )
    elif acq_config.name == "qlognehvi":
        if ref_point is None:
            raise ValueError("ref_point required for qlognehvi.")
        if not isinstance(gp_model, ModelListGP):
            raise ValueError("qlognehvi requires a multi-objective ModelListGP.")
        # qLogNEHVI builds its own partitioning from X_baseline; it does not
        # accept a precomputed `partitioning` argument.
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=gp_model,
            ref_point=ref_point.tolist(),
            X_baseline=train_X,
            sampler=sampler,
        )
    else:
        raise ValueError(f"Unknown acquisition: {acq_config.name}")

    fallback_candidate, batch_initial_conditions = screen_candidate_pool(
        acqf=acqf,
        bounds=bounds,
        rng=rng,
        candidate_pool=candidate_pool,
        num_restarts=num_restarts,
    )

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_tensor,
        q=1,
        num_restarts=int(batch_initial_conditions.shape[0]),
        raw_samples=int(raw_samples),
        options={"batch_limit": 5, "maxiter": int(maxiter)},
        batch_initial_conditions=batch_initial_conditions,
    )
    if candidate.numel() == 0:
        return fallback_candidate
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
    acq_failures: list[bool] = []

    # Regret tracking (computed on true objective)
    best_true_so_far = -np.inf
    cum_regret = 0.0
    best_true_list: list[float] = []
    regret_inst_list: list[float] = []
    regret_cum_list: list[float] = []
    simple_regret_list: list[float] = []
    regret_avg_list: list[float] = []

    # Inference regret: the incumbent the experimenter would actually pick
    # (best by OBSERVED value), scored by its TRUE value. Unlike
    # simple_regret_true this does not assume an omniscient final
    # recommendation, so it captures the identification cost of noise.
    inference_regret_list: list[float] = []
    inference_value_list: list[float] = []

    bounds_tensor = bounds.tensor
    previous_observed = None
    previous_error: np.ndarray | None = None

    is_multi = config.objective == "multi_objective"
    objective_true_scalar: list[float] = []
    objective_observed_scalar: list[float] = []
    hv_ref_point = config.ref_point if config.ref_point is not None else None

    sobol_engine = (
        torch.quasirandom.SobolEngine(dimension=len(bounds.low), scramble=True, seed=config.seed)
        if acq.name == "sobol"
        else None
    )

    for iteration in range(1, config.iterations + 1):
        acq_failed = False
        if iteration <= config.initial_samples:
            candidate_np = sample_uniform(bounds, rng, size=1)[0]
            fit_time = 0.0
        elif acq.name == "random":
            candidate_np = sample_uniform(bounds, rng, size=1)[0]
            fit_time = 0.0
        elif acq.name == "sobol":
            draw = sobol_engine.draw(1).to(torch.double).cpu().numpy()[0]
            candidate_np = bounds.low + draw * (bounds.high - bounds.low)
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
                if config.incumbent == "posterior_mean":
                    # Max posterior mean over visited points: robust to noisy
                    # observations (a single positive noise spike cannot
                    # inflate best_f beyond achievable values).
                    with torch.no_grad():
                        posterior_mean = gp.posterior(train_X).mean.reshape(-1)
                    best_f = posterior_mean.max().item()
                else:
                    best_f = train_Y.max().item()
                train_Y_for_acq = train_Y

            try:
                candidate_tensor = get_botorch_candidate(
                    gp_model=gp,
                    acq_config=acq,
                    bounds=bounds,
                    bounds_tensor=bounds_tensor,
                    best_f=best_f,
                    num_restarts=config.acq_num_restarts,
                    raw_samples=config.acq_raw_samples,
                    maxiter=config.acq_maxiter,
                    mc_samples=config.acq_mc_samples,
                    candidate_pool=config.candidate_pool,
                    rng=rng,
                    train_X=train_X,
                    train_Y=train_Y_for_acq,
                    ref_point=hv_ref_point,
                )
                candidate_np = candidate_tensor.cpu().numpy().flatten()
            except Exception as e:
                # Recorded per-iteration (acq_opt_failed column) so downstream
                # analysis can detect contaminated runs; never silently absorbed.
                acq_failed = True
                print(
                    f"[acq-fallback] run={run_id} acq={acq.name} iteration={iteration}: "
                    f"BoTorch optimization failed, falling back to random. Error: {e}",
                    file=sys.stderr,
                )
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
                previous_error=previous_error,
            )
        else:
            observed_value, error_magnitude = true_value, np.zeros_like(true_value, dtype=float)

        X_list.append(candidate_np)
        y_true_list.append(true_value)
        y_observed_list.append(observed_value)
        error_magnitudes.append(error_magnitude)
        fit_times.append(fit_time)
        acq_failures.append(acq_failed)
        previous_observed = observed_value
        previous_error = error_magnitude

        # Regret values are intentionally NOT clamped at zero: y_opt is a
        # sampling-based estimate that BO can legitimately exceed, and clamping
        # censors the metric distribution (observed in 22% of earlier runs).
        if is_multi:
            if hv_ref_point is None:
                raise ValueError("ref_point required for multi_objective.")
            hv_true = _compute_hypervolume(y_true_list, hv_ref_point)
            hv_obs = _compute_hypervolume(y_observed_list, hv_ref_point)
            objective_true_scalar.append(hv_true)
            objective_observed_scalar.append(hv_obs)
            best_true_so_far = max(best_true_so_far, hv_true)
            r_t = y_opt - hv_true
            # Inference incumbent: Pareto set as identified from OBSERVED
            # values, scored by the TRUE values of those same points.
            obs_t = torch.tensor(np.vstack(y_observed_list), dtype=torch.double)
            nd_mask = is_non_dominated(obs_t)
            inferred_true = [y_true_list[i] for i in range(len(y_true_list)) if bool(nd_mask[i])]
            inference_value = _compute_hypervolume(inferred_true, hv_ref_point)
        else:
            scalar_true = float(true_value[0])
            scalar_obs = float(observed_value[0])
            objective_true_scalar.append(scalar_true)
            objective_observed_scalar.append(scalar_obs)
            best_true_so_far = max(best_true_so_far, scalar_true)
            r_t = y_opt - scalar_true
            # Inference incumbent: the point the experimenter would pick
            # (highest OBSERVED value so far), scored by its TRUE value.
            best_obs_idx = int(np.argmax(objective_observed_scalar))
            inference_value = float(objective_true_scalar[best_obs_idx])
        cum_regret += r_t
        s_t = y_opt - best_true_so_far

        best_true_list.append(best_true_so_far)
        regret_inst_list.append(r_t)
        regret_cum_list.append(cum_regret)
        simple_regret_list.append(s_t)
        regret_avg_list.append(cum_regret / float(iteration))
        inference_value_list.append(inference_value)
        inference_regret_list.append(y_opt - inference_value)

    results = pd.DataFrame(X_list, columns=config.param_columns)
    results.insert(0, "iteration", np.arange(1, config.iterations + 1))

    results["objective_true"] = objective_true_scalar
    results["objective_observed"] = objective_observed_scalar
    if is_multi:
        for idx, column in enumerate(config.objective_columns):
            output_name = _objective_output_name(column)
            results[f"objective_true_{output_name}"] = [float(v[idx]) for v in y_true_list]
            results[f"objective_observed_{output_name}"] = [float(v[idx]) for v in y_observed_list]

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
            output_name = _objective_output_name(column)
            results[f"error_magnitude_{output_name}"] = [float(err[idx]) for err in error_magnitudes]
    else:
        results["error_magnitude"] = [float(err[0]) for err in error_magnitudes]
    results["acquisition"] = acq.name
    results["fit_time_sec"] = fit_times
    results["acq_opt_failed"] = acq_failures
    results["seed"] = config.seed
    results["run_id"] = run_id
    results["error_model"] = config.error_model if apply_error else "none"
    results["jitter_std"] = config.jitter_std if apply_error else 0.0
    results["jitter_iteration"] = config.jitter_iteration
    results["oracle_model"] = oracle_model
    results["objective"] = config.objective
    # Authoritative schema marker so downstream consumers do not have to
    # infer parameter columns from a hand-maintained reserved-column set.
    results["param_columns"] = ",".join(config.param_columns)

    # Regret columns
    results["y_opt"] = float(y_opt)
    results["best_true_so_far"] = best_true_list
    results["regret_inst_true"] = regret_inst_list
    results["regret_cum_true"] = regret_cum_list
    results["simple_regret_true"] = simple_regret_list
    results["regret_avg_true"] = regret_avg_list
    results["inference_value_true"] = inference_value_list
    results["inference_simple_regret_true"] = inference_regret_list

    return results


def summarize_adjustment(
    results: pd.DataFrame,
    jitter_iteration: int,
    param_columns: list[str],
) -> pd.Series:
    """Summarize a run relative to a jitter onset.

    The response delta uses the (t+1) -> (t+2) convention: noise first affects
    the observation at iteration jitter_iteration + 1, so the first candidate
    that can react to it is proposed at iteration jitter_iteration + 2. This
    matches evaluate_research_question.build_response_table and the README.
    """
    max_iter = int(results["iteration"].max())
    if jitter_iteration < 0 or jitter_iteration >= max_iter:
        raise ValueError("jitter_iteration must be within [0, max_iteration - 1].")

    start_iter = jitter_iteration + 1
    end_iter = jitter_iteration + 2
    if end_iter <= max_iter:
        current = results.loc[results["iteration"] == start_iter, param_columns].iloc[0]
        nxt = results.loc[results["iteration"] == end_iter, param_columns].iloc[0]
        delta = nxt - current
        l2_norm = float(np.linalg.norm(delta.to_numpy()))
        summary = {f"delta_{col}": float(delta[col]) for col in param_columns}
    else:
        # The response step falls outside the run; report NaN rather than a
        # silently wrong window.
        l2_norm = float("nan")
        summary = {f"delta_{col}": float("nan") for col in param_columns}
    summary["delta_l2_norm"] = l2_norm
    summary["iteration"] = jitter_iteration

    # Add run-level regret summaries (repeated per jitter_iteration row)
    summary["final_best_true"] = float(results["best_true_so_far"].iloc[-1])
    summary["final_simple_regret_true"] = float(results["simple_regret_true"].iloc[-1])
    summary["final_cum_regret_true"] = float(results["regret_cum_true"].iloc[-1])
    summary["final_avg_regret_true"] = float(results["regret_avg_true"].iloc[-1])
    summary["final_inference_simple_regret_true"] = float(
        results["inference_simple_regret_true"].iloc[-1]
    )

    sr = results["simple_regret_true"].to_numpy(dtype=float)
    summary["auc_simple_regret_true"] = float(np.trapezoid(sr, dx=1.0))
    ir = results["inference_simple_regret_true"].to_numpy(dtype=float)
    summary["auc_inference_simple_regret_true"] = float(np.trapezoid(ir, dx=1.0))
    summary["acq_opt_failures"] = int(results["acq_opt_failed"].sum())

    return pd.Series(summary)


def _load_resumable_run(path: Path, iterations: int) -> pd.DataFrame | None:
    """Load a previously written per-run CSV if it is complete and usable.

    Returns None (caller re-simulates) when the file is missing, truncated,
    or lacks the columns the summary step needs. Reusing files is safe
    because every run is fully determined by its seed and configuration.
    """
    if not path.is_file():
        return None
    required = {
        "iteration",
        "run_id",
        "jitter_std",
        "error_model",
        "simple_regret_true",
        "inference_simple_regret_true",
        "regret_cum_true",
        "regret_avg_true",
        "best_true_so_far",
        "acq_opt_failed",
        "dataset",
    }
    try:
        loaded = pd.read_csv(path)
    except Exception:
        return None
    if len(loaded) != iterations or not required.issubset(loaded.columns):
        return None
    return loaded


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

        X_known = df[dataset.param_columns].to_numpy(dtype=float)
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
                X_known=X_known,
            )
        else:
            y_opt = estimate_oracle_optimum(
                oracle=oracle,
                bounds=bounds,
                seed=args.oracle_opt_seed,
                n=args.oracle_opt_samples,
                batch_size=args.oracle_opt_batch_size,
                X_known=X_known,
            )

        clip_low, clip_high = parse_response_clip(
            args.response_clip,
            df,
            objective,
            objective_columns,
            args.normalize_objective,
            weights,
        )

        # Baseline runs carry neutral error metadata; the per-condition values
        # are substituted via dataclasses.replace for each jittered run.
        base_config = SimulationConfig(
            iterations=args.iterations,
            jitter_iteration=0,
            jitter_std=0.0,
            single_error=args.single_error,
            initial_samples=args.initial_samples,
            candidate_pool=args.candidate_pool,
            objective=objective,
            objective_columns=objective_columns,
            param_columns=dataset.param_columns,
            seed=seed,
            error_model="none",
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
            error_ar1_rho=args.error_ar1_rho,
            response_clip_low=clip_low,
            response_clip_high=clip_high,
            response_round=args.response_round,
            incumbent=args.incumbent,
        )

        for acq in acquisitions:
            # Baseline run
            if args.baseline_run:
                results_path = args.output_dir / (
                    f"bo_sensor_error_{dataset.name}_{objective}_{acq.name}_seed{seed}_baseline_{oracle_model}.csv"
                )
                baseline_results = (
                    _load_resumable_run(results_path, args.iterations) if args.resume else None
                )
                if baseline_results is not None:
                    baseline_run_id = str(baseline_results["run_id"].iloc[0])
                    baseline_runtime = 0.0
                else:
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
                        # Disambiguate variant parameters in the filename so
                        # non-default runs cannot overwrite each other.
                        variant_parts = []
                        if error_model == "bias" and args.error_bias != 0.2:
                            variant_parts.append(f"bias{args.error_bias}")
                        if error_model == "spike":
                            variant_parts.append(f"sp{args.error_spike_prob}-{args.error_spike_std}")
                        if error_model == "ar1" and args.error_ar1_rho != 0.8:
                            variant_parts.append(f"rho{args.error_ar1_rho}")
                        if args.single_error:
                            variant_parts.append("single")
                        variant_suffix = ("_" + "_".join(variant_parts)) if variant_parts else ""
                        results_path = args.output_dir / (
                            f"bo_sensor_error_{dataset.name}_{objective}_{acq.name}_seed{seed}_jittered_"
                            f"{oracle_model}_{error_model}_jit{jitter_iteration}_std{jitter_std}{variant_suffix}.csv"
                        )

                        results = (
                            _load_resumable_run(results_path, args.iterations)
                            if args.resume
                            else None
                        )
                        if results is not None:
                            run_id = str(results["run_id"].iloc[0])
                            run_runtime = 0.0
                        else:
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
    jitter_stds = resolve_sweep_values(
        args.jitter_stds,
        args.jitter_std,
        DEFAULT_JITTER_STDS,
        lambda raw: parse_float_list(raw, 0.2),
        ("--jitter-stds", "--jitter-std"),
    )
    jitter_iterations = resolve_sweep_values(
        args.jitter_iterations,
        args.jitter_iteration,
        DEFAULT_JITTER_ITERATIONS,
        lambda raw: parse_int_list(raw, 20),
        ("--jitter-iterations", "--jitter-iteration"),
    )
    validate_sweeps(jitter_iterations, jitter_stds, args.iterations)

    requested_oracle_models = parse_oracle_models(args.oracle_model, args.oracle_models)
    seeds = parse_seed_list(args.seeds, args.seed, args.num_seeds)

    acquisition_names = parse_acquisition_list(args.acq, args.acq_list)

    if args.combine_datasets and requested_oracle_models == [AUTO_ORACLE_MODEL]:
        raise ValueError(
            "--combine-datasets is not supported with --oracle-model auto: "
            "select_best_oracle_model.py produces no entry for the combined dataset. "
            "Pass an explicit oracle model instead."
        )

    dataset_configs = parse_dataset_configs(
        args.data_dir,
        args.dataset_config,
        args.dataset_cache_dir,
    )
    if args.combine_datasets:
        combined = combine_dataset_configs(dataset_configs)
        if combined is not None:
            dataset_configs.append(combined)

    oracle_selection = (
        load_oracle_selection(args.oracle_selection_path)
        if requested_oracle_models == [AUTO_ORACLE_MODEL]
        else None
    )

    output_dir = ensure_output_dir(args.output_dir)
    runtime_start = time.perf_counter()

    dataset_objectives: dict[str, list[str]] = {}
    objective_acquisition_names: dict[tuple[str, str], list[str]] = {}
    objective_oracle_models: dict[tuple[str, str], list[str]] = {}
    oracle_cv_scores: dict[str, float | None] = {}
    total_runs = 0
    for dataset in dataset_configs:
        objective_names = parse_objective_list(
            args.objective,
            args.objectives,
            dataset.objective_map,
        )
        dataset_objectives[dataset.name] = objective_names
        for objective_name in objective_names:
            key = (dataset.name, objective_name)
            filtered_acq_names = filter_acquisitions_for_objective(acquisition_names, objective_name)
            resolved_models = resolve_oracle_models_for_objective(
                requested_oracle_models,
                dataset.name,
                objective_name,
                oracle_selection,
            )
            objective_acquisition_names[key] = filtered_acq_names
            objective_oracle_models[key] = resolved_models

            # Oracle fidelity gate: the oracle is the ground-truth "human" for
            # all downstream claims, so a low cross-validated R^2 must never
            # pass silently.
            if oracle_selection is not None:
                entry = oracle_selection.get((dataset.name, objective_name), {})
                scores = entry.get("scores")
                best_model = entry.get("best_model")
                cv_score = None
                if isinstance(scores, dict) and isinstance(best_model, str):
                    raw_score = scores.get(best_model)
                    cv_score = float(raw_score) if raw_score is not None else None
                oracle_cv_scores[f"{dataset.name}:{objective_name}"] = cv_score
                if cv_score is not None:
                    if args.min_oracle_r2 is not None and cv_score < args.min_oracle_r2:
                        raise ValueError(
                            f"Auto-selected oracle '{best_model}' for "
                            f"{dataset.name}/{objective_name} has cross-validated "
                            f"R^2={cv_score:.3f} < --min-oracle-r2={args.min_oracle_r2}. "
                            "The simulated human has insufficient predictive validity."
                        )
                    if cv_score < 0.3:
                        print(
                            f"WARNING: oracle '{best_model}' for {dataset.name}/"
                            f"{objective_name} has low cross-validated R^2={cv_score:.3f}. "
                            "Results simulate a weakly human-grounded test function; "
                            "report this fidelity alongside any conclusions.",
                            file=sys.stderr,
                        )
            baseline_runs = (
                len(filtered_acq_names) * len(seeds) * len(resolved_models)
                if args.baseline_run
                else 0
            )
            jittered_runs = (
                len(filtered_acq_names)
                * len(seeds)
                * len(resolved_models)
                * len(error_models)
                * len(jitter_stds)
                * len(jitter_iterations)
            )
            total_runs += baseline_runs + jittered_runs

    use_parallel = args.parallel or len(seeds) > 1

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
    failed_seeds: list[dict[str, str]] = []

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
            ref_point = compute_reference_point(
                df, objective_name, objective_columns, args.normalize_objective
            )

            key = (dataset.name, objective_name)
            filtered_acq_names = objective_acquisition_names[key]
            acquisitions = [
                AcquisitionConfig(name=n, xi=args.xi, kappa=args.kappa) for n in filtered_acq_names
            ]
            resolved_oracle_models = objective_oracle_models[key]

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
                                resolved_oracle_models,
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
                                import traceback

                                print(
                                    f"\nError processing seed {seed} "
                                    f"({dataset.name}/{objective_name}): {e}",
                                    file=sys.stderr,
                                )
                                traceback.print_exc()
                                failed_seeds.append(
                                    {
                                        "seed": str(seed),
                                        "dataset": dataset.name,
                                        "objective": objective_name,
                                        "error": repr(e),
                                        "traceback": traceback.format_exc(),
                                    }
                                )
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
                for seed in seeds:
                    seed_summaries, _ = run_single_seed(
                        seed,
                        dataset,
                        objective_name,
                        resolved_oracle_models,
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

    progress.close()

    if not summaries:
        print("No simulation runs completed.")
        return

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "bo_sensor_error_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if args.baseline_run:
        jittered = summary_df[~summary_df["baseline"]]
        baseline = summary_df[summary_df["baseline"]]

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
        acquisition_names=objective_acquisition_names,
        error_models=error_models,
        requested_oracle_models=requested_oracle_models,
        resolved_oracle_models=objective_oracle_models,
        seeds=seeds,
        args=args,
        jitter_iterations=jitter_iterations,
        jitter_stds=jitter_stds,
    )

    def _git_commit(path: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    data_dir_commits = {}
    for dataset in dataset_configs:
        for data_dir in dataset.data_dirs:
            commit = _git_commit(data_dir)
            if commit is not None:
                data_dir_commits[str(data_dir)] = commit

    metadata_payload = {
        "args": vars(args),
        "runtime_sec": float(time.perf_counter() - runtime_start),
        "parallel_execution": bool(use_parallel and len(seeds) > 1),
        "n_workers": int(n_jobs) if (use_parallel and len(seeds) > 1) else 1,
        "git_commit": _git_commit(REPO_ROOT),
        "python_version": sys.version,
        "platform": sys.platform,
        "data_dir_commits": data_dir_commits,
        "resolved_oracle_cv_r2": oracle_cv_scores,
        "failed_seeds": failed_seeds,
        "effective_jitter_stds": jitter_stds,
        "effective_jitter_iterations": jitter_iterations,
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
        "requested_oracle_models": requested_oracle_models,
        "resolved_oracle_models": {
            f"{dataset_name}:{objective_name}": models
            for (dataset_name, objective_name), models in objective_oracle_models.items()
        },
        "oracle_selection_path": str(args.oracle_selection_path),
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
                "catboost",
                "statsmodels",
                "botorch",
                "torch",
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

    if failed_seeds:
        failed_ids = ", ".join(sorted({entry["seed"] for entry in failed_seeds}))
        print(
            f"ERROR: {len(failed_seeds)} seed run(s) failed (seeds: {failed_ids}). "
            "Summary outputs are incomplete; see run_metadata.json['failed_seeds'].",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    
    main()

