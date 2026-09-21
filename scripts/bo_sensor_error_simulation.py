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
import functools
import importlib.metadata
import json
import math
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
    AnalyticAcquisitionFunction,
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    _log_ei_helper,
    _scaled_improvement,
)
from botorch.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
try:
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
except ImportError:  # older BoTorch: build_thompson_sampling draws the Matheron path itself
    PathwiseThompsonSampling = None
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
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
from botorch.sampling.qmc import NormalQMCEngine
from botorch.utils.transforms import t_batch_mode_transform

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class _DefaultDtype:
    """Temporarily swap torch's process-wide default dtype and restore it."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self._previous: torch.dtype | None = None

    def __enter__(self) -> "_DefaultDtype":
        self._previous = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        return self

    def __exit__(self, *exc_info: object) -> bool:
        if self._previous is not None:
            torch.set_default_dtype(self._previous)
        return False


class TabPFNFloat32Regressor:
    """TabPFN under a float64 default dtype.

    This module sets torch's default dtype to float64 so BoTorch runs in double
    precision, but TabPFN's shipped checkpoints are float32 and it builds its
    internal tensors from the *default* dtype. Calling it in this process
    therefore raised ``RuntimeError: mat1 and mat2 must have the same dtype, but
    got Float and Double`` for every fit/predict, which is why the tabpfn oracle
    had never actually run. The adapter pins the default dtype to float32 for the
    duration of each TabPFN call and restores it afterwards, so BoTorch keeps its
    double precision.

    Kept as a module-level class (not a closure or a decorator) so instances
    survive the pickling that Windows' 'spawn' multiprocessing does.
    """

    def __init__(self, **kwargs: object) -> None:
        if TabPFNRegressor is None:
            raise ImportError(
                "tabpfn is required for oracle-model=tabpfn. Install it via requirements.txt."
            )
        self.kwargs = dict(kwargs)
        with _DefaultDtype(torch.float32):
            self.model = TabPFNRegressor(**self.kwargs)

    def fit(self, X: object, y: object) -> "TabPFNFloat32Regressor":
        with _DefaultDtype(torch.float32):
            self.model.fit(X, y)
        return self

    def predict(self, X: object) -> np.ndarray:
        with _DefaultDtype(torch.float32):
            preds = self.model.predict(X)
        return np.asarray(preds, dtype=float)

    def score(self, X: object, y: object) -> float:
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.kwargs)

    def __getattr__(self, name: str) -> object:
        # Only reached for attributes this wrapper does not define itself.
        # Guarding the private names keeps unpickling (which sets __dict__
        # before any attribute exists) from recursing.
        if name.startswith("_") or name in {"kwargs", "model"}:
            raise AttributeError(name)
        return getattr(self.__dict__["model"], name)

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

# Robust-BO baselines. Everything else in this list is a standard acquisition
# that happens to be run under noise; these two are the answers the literature
# actually gives to noisy observations, and without them the study can say which
# ordinary choice survives noise best but not how much a method built for the
# problem would recover.
#   qkg     the knowledge gradient, which values a candidate by the improvement
#           in the posterior MAXIMUM rather than in the observed one, and so does
#           not depend on any single noisy observation being right.
#   replei  the practical answer nobody writes papers about: spend half the
#           budget re-asking about the incumbent, averaging the rater's noise
#           down instead of modelling it. Compared at a fixed number of
#           EVALUATIONS, so it buys its replication with half the designs.
ROBUST_ACQUISITION_CHOICES = ["qkg", "replei"]

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
# Acquisition-side follow-ups, single-objective only. They are APPENDED to the
# end of ACQUISITION_CHOICES rather than added to SINGLE_ACQUISITION_CHOICES:
# that list precedes qkg/replei, the hypervolume family and the floors in the
# composed order, so growing it would move their indices and with them the
# noise every existing arm receives (see the note on ERROR_MODEL_CHOICES). Like
# qkg and replei they are opt-in and stay out of every "all" expansion.
#   ts   Thompson sampling: maximise one posterior sample path per iteration.
#   aei  augmented expected improvement (Huang, Allen, Notz & Zeng 2006): EI
#        against the posterior mean at the lower-confidence-bound incumbent,
#        discounted where the model is already about as sure as a rating is.
EXTENSION_ACQUISITION_CHOICES = ["ts", "aei", "shiplcb"]
# Acquisitions whose values are logs, so an average over them has to be taken
# in log space. replei optimises LogEI on its model-based trials.
LOG_VALUED_ACQUISITIONS = frozenset({"logei", "logpi", "replei", "aei"})
# Every name a run may USE. The robust baselines belong here: they are real
# acquisitions and --acq-list qkg,replei must validate.
ACQUISITION_CHOICES = (
    SINGLE_ACQUISITION_CHOICES
    + ROBUST_ACQUISITION_CHOICES
    + MULTI_ACQUISITION_CHOICES
    + BASELINE_ACQUISITION_CHOICES
    + EXTENSION_ACQUISITION_CHOICES
)
# What "all" EXPANDS to, which is NOT the same thing. qkg and replei were added
# after the main sweeps ran; folding them into "all" would silently redefine the
# design of every later arm that asks for it -- in particular the confirmatory
# arm, whose whole purpose is to replicate the exploratory design on fresh seeds.
# They are opt-in, and their own arm names them explicitly.
DEFAULT_ACQUISITION_CHOICES = (
    SINGLE_ACQUISITION_CHOICES + MULTI_ACQUISITION_CHOICES + BASELINE_ACQUISITION_CHOICES
)
# ORDER IS LOAD-BEARING. Every run's noise stream is seeded from
# ERROR_MODEL_CHOICES.index(error_model), so a name inserted anywhere but the
# end silently changes the noise every existing model receives: a rerun of a
# published sweep would stop reproducing it and a resumed arm would mix two
# streams. That happened once -- "none" was prepended -- and the full suite
# passed, because nothing pinned the order; tests/test_error_model_labels.py now
# does. Append new models; never insert.
#
# "none" (no response error, which is how the input-error arm runs) is kept OUT
# of this list for the same reason, and so that "--error-models all" keeps its
# meaning. It is handled explicitly where it is used, and run_error_label()
# keeps it reserved for the clean baseline.
ERROR_MODEL_CHOICES = ["gaussian", "bias", "dropout", "spike", "drift", "ar1"]

DEFAULT_ERROR_MODELS = "gaussian,bias"
DEFAULT_JITTER_ITERATIONS = "10,20,40"
DEFAULT_JITTER_STDS = "0.05,0.5,1,5"
# "lcb" (appended): the best lower confidence bound over the visited designs,
# posterior mean minus one latent SD. A design fitted to a single lucky rating
# still carries a wide band there, so it cannot raise the bar the way its
# posterior mean alone would.
INCUMBENT_CHOICES = ["posterior_mean", "observed_max", "lcb"]
OBSERVATION_NOISE_CHOICES = ["learned", "known", "self_report"]
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

    # --- INPUT error: the person acts on the wrong design ---------------------
    # Every error model above corrupts the reported VALUE: the optimizer is told
    # the wrong number about the design it proposed -- a wrong value at the right
    # location. This is the other corruption, and it is not a special case of
    # that one. The optimizer proposes x, the person actually experiences
    # x' != x (slider overshoot, wrong option pressed, wrong config applied) and
    # rates x' HONESTLY. A right value at the wrong location.
    #
    # It matters because the damage is set by the landscape's local geometry
    # rather than by a noise parameter: the same slip costs nothing on a plateau
    # and everything beside a narrow optimum. It is also the corruption a fitted
    # oracle cannot measure cleanly -- scoring it needs the objective AT THE
    # POINT ACTUALLY TOUCHED, so with a surrogate you get its error at a second
    # location on top of the slip's cost.
    #
    #   "none"      no input error.
    #   "slip"      x' = clip(x + delta), delta ~ N(0, s) per coordinate, s a
    #               FRACTION OF EACH COORDINATE'S RANGE. Pointer or slider
    #               imprecision: every trial, usually small.
    #   "misclick"  with probability p the trial lands on a uniformly random
    #               different design; otherwise exact. A discrete wrong press:
    #               rare, large.
    #
    # input_error_scale carries s for "slip" and p for "misclick"; both are
    # dimensionless in [0, 1], which is why one swept grid serves both. It is
    # NOT in landscape standard deviations, so it is not comparable with
    # jitter_std and results from the two must never be pooled.
    input_error_model: str = "none"
    input_error_scale: float = 0.0
    # Which design is written down. "proposed" is the real case -- nobody logs a
    # slip they did not notice, so the surrogate trains on (x, f(x')). "actual"
    # is the counterfactual where the slip is detected and logged, (x', f(x'));
    # the contrast separates the cost of going to the wrong place from the cost
    # of mislabelling it.
    input_error_recorded: str = "proposed"

    # Incumbent definition for improvement-based acquisitions.
    # "posterior_mean" avoids the noisy-max pitfall where a single positive
    # noise spike inflates best_f beyond any achievable value.
    incumbent: str = "posterior_mean"

    # How the surrogate learns about observation noise.
    #   "learned" (default, and what every result to date used) fits the noise as
    #     a free hyperparameter under BoTorch's default prior.
    #   "known" passes the TRUE injected variance as train_Yvar.
    # The contrast matters because "learned" confounds two things a study of
    # noisy feedback needs to keep apart: the information the error destroys, and
    # the surrogate's failure to realise the error is there. Only the first is a
    # property of the problem. Single-objective only.
    observation_noise: str = "learned"

    # --- process adaptations (the follow-up arms of docs/adaptations-proposal.md)
    # replicate_first: for the first N model-based proposals, rate each design
    #   twice -- the second rating is a fresh draw of the error process on the
    #   same recorded design, so two evaluations buy one design in that window
    #   and single ratings follow. Works with any acquisition; 0 disables it.
    #   The onset effect says early error is the expensive one, so this is
    #   replication spent where it should matter, against replei's uniform
    #   every-second-trial schedule.
    replicate_first: int = 0
    # final_rerate_top / final_rerate_reps: the last top x reps trials re-rate
    #   the top designs (by mean observation when the window opens), reps times
    #   each, and the deployed design is chosen by mean observation. This is
    #   "re-evaluate before you deploy" at a fixed evaluation budget.
    final_rerate_top: int = 0
    final_rerate_reps: int = 0
    # input_noise_model: "nigp" is a first-order noisy-input GP (McHutchon &
    #   Rasmussen, 2011): each training point's observation variance is inflated
    #   by the squared gradient of the posterior mean times the slip variance, so
    #   a recorded design that may have slipped is trusted less where the
    #   objective is steep. input_error_scale is the assumed slip SD, a fraction
    #   of each coordinate's range, as in the slip arm. Single-objective only.
    input_noise_model: str = "none"
    # inference_rule: how the deployed design is picked each iteration.
    #   "best_observed" (default, every result to date): the single highest
    #   observation. "best_mean": the design with the highest mean over its
    #   ratings, which is the point of replicating. The synthetic driver switches
    #   to best_mean whenever either replication option is on.
    inference_rule: str = "best_observed"
    # likelihood: "student_t" replaces the surrogate with an outlier-robust
    #   Student-t variational GP (scripts/robust_gp.py), for misclicks and gross
    #   sensor faults that a Gaussian likelihood bends through. Single-objective,
    #   learned noise only.
    likelihood: str = "gaussian"

    # --- acquisition-side follow-ups (equal-trial: none of them adds a rating)
    # input_uncertain_acq / input_uncertain_scale: average the acquisition over
    #   K fixed QMC normal perturbations of the design, SD scale x (high - low)
    #   per coordinate, so the optimiser prefers designs whose neighbourhood is
    #   good -- the acquisition-side answer to a slip. 0 disables it. A negative
    #   scale borrows the run's own input_error_scale, which is zero in the clean
    #   run; an explicit scale >= 0 applies to the clean run too, which is what
    #   prices the wrapper.
    input_uncertain_acq: int = 0
    input_uncertain_scale: float = -1.0
    # min_distance: a model-based proposal whose RMS distance (unit box) to a
    #   logged design is below this is treated as a near-repeat and redirected to
    #   the best screened pool point at least this far from every logged design.
    #   0 disables it.
    min_distance: float = 0.0

    # --- error-process extensions (equal-trial: none of them adds a rating) ---
    # noise_schedule: "none", a preset (front10, front20, U, back10) or a custom
    #   "1-10:2,11-:0.75". Effort e_t sets the gaussian error SD at trial t to
    #   jitter_std / sqrt(e_t), at a mean effort of 1 over the run.
    noise_schedule: str = "none"
    # missing_handling: what the surrogate gets for a rating lost to the
    #   missing_mcar / missing_low input processes, "drop" or "impute_low".
    missing_handling: str = "drop"
    # rater_assign / rater_offset_ratio: relay raters. "block:K" hands over every
    #   K trials, "roundrobin:R" cycles R raters, and rater r adds a fixed offset
    #   b_r ~ N(0, (ratio x jitter_std)^2) to noisy ratings after the onset.
    #   rater_model "backfit" estimates the offsets inside the scalar GP fit.
    rater_assign: str = "none"
    rater_offset_ratio: float = 0.0
    rater_model: str = "none"
    # response_ceiling: noisy ratings are capped at this quantile of the
    #   landscape (0 = off). ceiling_mode "anchored" raises the cap to 0.5 above
    #   the true value of the best-rated design so far.
    response_ceiling: float = 0.0
    ceiling_mode: str = "fixed"
    # anchor_rating: the proposal is judged beside the incumbent, so the error
    #   the rater shares between the pair cancels and the fresh part is
    #   differenced. The clean run is unchanged.
    anchor_rating: bool = False
    # confidence_noise: how coarse the rater's own precision report is, as the SD
    #   of a log-normal multiplier on the true squared error. 0 = a perfect report.
    confidence_noise: float = 0.5
    # anchor_every / anchor_set: every anchor_every trials the rater sees one of
    #   anchor_set fixed designs instead of the proposal. Their true value never
    #   moves, so any movement in their ratings is the rater drifting. 0 = off.
    anchor_every: int = 0
    anchor_set: int = 3
    # anchor_model: 'detrend' fits a line in the trial index to the anchors and
    #   subtracts it from every training rating before the surrogate is fitted.
    anchor_model: str = "none"
    # hold_early / hold_until: a design proposed in the first hold_early trials is
    #   not rated then; it is queued and rated from trial hold_until onward.
    hold_early: int = 0
    hold_until_frac: float = 0.6

    # --- multi-objective halo error (equal-trial: neither adds a rating) -----
    # error_cross_corr: rho in [0, 1]. A rater's overall impression of a design
    #   leaks into every objective's rating, so one trial's gaussian errors share
    #   a factor: e_j = sqrt(1 - rho) eps_j + sqrt(rho) z, eps drawn as in the
    #   standard run and z ONE extra N(0, jitter_std^2) draw after it. Each e_j
    #   keeps SD jitter_std; only the correlation between objectives is new.
    #   0 disables it and draws nothing.
    error_cross_corr: float = 0.0
    # mo_halo_model: "backfit" estimates the shared factor from the fitted
    #   ModelListGP's standardised residuals, refits on the corrected ratings and
    #   deploys the Pareto set of the corrected ratings (fit_mo_halo_backfit).
    mo_halo_model: str = "none"


INPUT_NOISE_MODEL_CHOICES = ["none", "nigp"]
INFERENCE_RULE_CHOICES = ["best_observed", "best_mean"]
# relevance_pursuit (appended): Ament et al.'s robust GP, which gives each
# training point its own outlier variance and selects how many are non-zero.
LIKELIHOOD_CHOICES = ["gaussian", "student_t", "relevance_pursuit"]
MISSING_HANDLING_CHOICES = ["drop", "impute_low"]
RATER_MODEL_CHOICES = ["none", "backfit"]
CEILING_MODE_CHOICES = ["fixed", "anchored"]
ANCHOR_MODEL_CHOICES = ["none", "detrend"]
MO_HALO_MODEL_CHOICES = ["none", "backfit"]


def _design_key(x: np.ndarray) -> tuple:
    return tuple(np.round(np.asarray(x, dtype=float), 10).tolist())


def _mean_by_design(X_list: list[np.ndarray], observed: list[float]) -> dict[tuple, tuple[float, int]]:
    """Mean observation per distinct recorded design, with the first index it appeared at."""
    sums: dict[tuple, list] = {}
    for idx, (x, y) in enumerate(zip(X_list, observed)):
        # A lost or imputed rating (NaN) is no rating of the design.
        if np.isnan(y):
            continue
        key = _design_key(x)
        if key not in sums:
            sums[key] = [0.0, 0, idx]
        sums[key][0] += float(y)
        sums[key][1] += 1
    return {key: (s / n, first) for key, (s, n, first) in sums.items()}


def _best_mean_index(X_list: list[np.ndarray], observed: list[float]) -> int:
    """Index of (the first rating of) the design with the highest mean observation."""
    means = _mean_by_design(X_list, observed)
    if not means:
        return 0  # nothing rated yet: the first design, as _nan_argmax does
    best = max(means.values(), key=lambda pair: pair[0])
    return int(best[1])


def _nan_argmax(values: list[float]) -> int:
    """argmax ignoring NaN (a lost or imputed rating); 0 when nothing is rated yet."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return 0
    return int(np.nanargmax(arr))


def _rerate_schedule(X_list: list[np.ndarray], observed: list[float], top: int, reps: int) -> list[np.ndarray]:
    """The top designs by mean observation, each repeated reps times, round-robin."""
    means = _mean_by_design(X_list, observed)
    ranked = sorted(means.values(), key=lambda pair: pair[0], reverse=True)[:top]
    designs = [np.array(X_list[first], dtype=float) for _, first in ranked]
    return [designs[i % len(designs)] for i in range(reps * len(designs))]


def _refit_with_input_noise(
    gp: "SingleTaskGP", train_X: torch.Tensor, train_Y: torch.Tensor, bounds: "Bounds",
    config: SimulationConfig,
) -> tuple["SingleTaskGP", ExactMarginalLogLikelihood]:
    """Refit with per-point observation variance inflated for input noise.

    First-order noisy-input GP (McHutchon & Rasmussen, 2011): if the recorded
    design is x and the evaluated one x + delta with delta ~ N(0, S), then to
    first order f(x + delta) ~ f(x) + grad f(x)^T delta, so the observation at x
    carries extra variance grad^T S grad. The gradient comes from the posterior
    mean of a first (ordinary) fit; the second fit takes the inflated variance
    as fixed per-point noise, the same path the known-noise arm uses.
    """
    gp.eval()
    slip_sd = config.input_error_scale * (np.asarray(bounds.high, dtype=float) - np.asarray(bounds.low, dtype=float))
    slip_var = torch.tensor(slip_sd ** 2, dtype=torch.double)
    X_req = train_X.clone().requires_grad_(True)
    mean_sum = gp.posterior(X_req).mean.sum()
    grad = torch.autograd.grad(mean_sum, X_req)[0]
    inflation = (grad.detach() ** 2 * slip_var).sum(dim=-1, keepdim=True)
    # The fitted noise lives in standardised units; bring it back to Y's.
    noise_var = gp.likelihood.noise.detach().reshape(-1)[0] * gp.outcome_transform.stdvs.detach().reshape(-1)[0] ** 2
    train_Yvar = noise_var + inflation + 1e-6
    refit = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds.tensor),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(refit.likelihood, refit)
    fit_gpytorch_mll(mll)
    return refit, mll


def adaptation_fields(args: "argparse.Namespace") -> dict:
    """SimulationConfig fields for the process adaptations, from the CLI.

    Shared by both drivers so the two cannot drift. The inference rule
    defaults to best_mean whenever any replication is on, which is what the
    replication is for; an explicit --inference-rule overrides that.
    """
    top, reps = (int(v) for v in str(getattr(args, "final_rerate", "0,0")).split(","))
    if (top > 0) != (reps > 0):
        raise ValueError("--final-rerate needs both TOP and REPS positive, or both zero.")
    replicate_first = int(getattr(args, "replicate_first", 0) or 0)
    rule = getattr(args, "inference_rule", None)
    if rule is None:
        rule = "best_mean" if (replicate_first or top) else "best_observed"
    # Any negative scale means "the run's own input_error_scale"; normalising it
    # to -1 keeps one filename per run.
    iu_scale = getattr(args, "input_uncertain_scale", -1.0)
    iu_scale = -1.0 if iu_scale is None or float(iu_scale) < 0 else float(iu_scale)
    return {
        "replicate_first": replicate_first,
        "final_rerate_top": top,
        "final_rerate_reps": reps,
        "input_noise_model": getattr(args, "input_noise_model", "none") or "none",
        "inference_rule": rule,
        "likelihood": getattr(args, "likelihood", "gaussian") or "gaussian",
        # The acquisition-side follow-ups. Read with getattr defaults because the
        # fitted-oracle driver has no such flags and must keep running with them off.
        "input_uncertain_acq": int(getattr(args, "input_uncertain_acq", 0) or 0),
        "input_uncertain_scale": iu_scale,
        "min_distance": float(getattr(args, "min_distance", 0.0) or 0.0),
        # The error-process extensions, likewise off for a driver without the flags.
        "noise_schedule": str(getattr(args, "noise_schedule", "none") or "none").strip(),
        "missing_handling": getattr(args, "missing_handling", "drop") or "drop",
        "rater_assign": str(getattr(args, "rater_assign", "none") or "none").strip(),
        "rater_offset_ratio": float(getattr(args, "rater_offset_ratio", 0.0) or 0.0),
        "rater_model": getattr(args, "rater_model", "none") or "none",
        "response_ceiling": float(getattr(args, "response_ceiling", None) or 0.0),
        "ceiling_mode": getattr(args, "ceiling_mode", "fixed") or "fixed",
        "anchor_rating": bool(getattr(args, "anchor_rating", False)),
        "confidence_noise": float(getattr(args, "confidence_noise", 0.5) or 0.0),
        "anchor_every": int(getattr(args, "anchor_every", 0) or 0),
        "anchor_set": int(getattr(args, "anchor_set", 3) or 3),
        "anchor_model": str(getattr(args, "anchor_model", "none") or "none"),
        "hold_early": int(getattr(args, "hold_early", 0) or 0),
        "hold_until_frac": float(getattr(args, "hold_until_frac", 0.6) or 0.6),
        # The multi-objective halo error and its remedy, likewise.
        "error_cross_corr": float(getattr(args, "error_cross_corr", 0.0) or 0.0),
        "mo_halo_model": getattr(args, "mo_halo_model", "none") or "none",
    }


@dataclasses.dataclass
class DatasetConfig:
    name: str
    data_dirs: list[Path]
    param_columns: list[str]
    objective_map: dict[str, list[str]]
    observation_glob: str = "ObservationsPerEvaluation.csv"
    # How the oracle target is built from repeated evaluations of a design:
    #   "individual" — one training row per (participant, design) rating (default).
    #   "mean"       — collapse to one row per design holding the mean rating, an
    #                  "average human" surface BO can actually learn. The
    #                  simulator's injected feedback noise then models the
    #                  individual deviation that averaging removes.
    oracle_target: str = "individual"


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
        "--input-error",
        type=str,
        default="none",
        choices=INPUT_ERROR_CHOICES,
        help="Corrupt the DESIGN rather than the rating: the optimizer proposes "
        "x, the person acts on x' != x and rates x' honestly. 'slip' adds "
        "gaussian positional error every trial (pointer/slider imprecision); "
        "'misclick' jumps to a uniformly random design with probability p (a "
        "wrong press). Magnitude comes from --input-error-scale, or from the "
        "swept --jitter-stds grid when --input-error-from-sweep is set. "
        "Single-objective only.",
    )
    parser.add_argument(
        "--input-error-scale",
        type=float,
        default=0.0,
        help="Slip SD as a FRACTION OF EACH COORDINATE'S RANGE, or the misclick "
        "probability. Dimensionless in [0,1] -- NOT landscape standard "
        "deviations, so it is not comparable with --jitter-stds and results "
        "from the two must not be pooled.",
    )
    parser.add_argument(
        "--input-error-from-sweep",
        action="store_true",
        default=False,
        help="Take the input-error magnitude from the swept --jitter-stds grid "
        "instead of --input-error-scale, so one sweep covers the grid. The "
        "recorded jitter_std column then means a box fraction (or a "
        "probability), not landscape SDs.",
    )
    parser.add_argument(
        "--input-error-recorded",
        type=str,
        default="proposed",
        choices=INPUT_ERROR_RECORDED_CHOICES,
        help="Which design is written down. 'proposed' (default) is the real "
        "case -- an unnoticed slip means the surrogate trains on (x, f(x')). "
        "'actual' is the counterfactual where the slip is detected and logged, "
        "separating the cost of going to the wrong place from the cost of "
        "mislabelling it.",
    )
    parser.add_argument(
        "--observation-noise",
        type=str,
        default="learned",
        choices=OBSERVATION_NOISE_CHOICES,
        help="'learned' (default) fits the GP's observation noise as a free "
        "hyperparameter, as every result to date does. 'known' passes the true "
        "injected variance as train_Yvar, which separates the information cost of "
        "the error from the surrogate's failure to model it. Single-objective only.",
    )
    parser.add_argument(
        "--replicate-first", type=int, default=0,
        help="Rate each of the first N model-based proposals twice (0 = off). Any acquisition.",
    )
    parser.add_argument(
        "--final-rerate", type=str, default="0,0",
        help="TOP,REPS: spend the last TOP x REPS trials re-rating the TOP best designs "
        "REPS times each, and deploy by mean observation. '0,0' = off.",
    )
    parser.add_argument(
        "--input-noise-model", type=str, default="none", choices=INPUT_NOISE_MODEL_CHOICES,
        help="'nigp': first-order noisy-input GP, inflating each point's observation "
        "variance by the squared posterior-mean gradient times the slip variance.",
    )
    parser.add_argument(
        "--likelihood", type=str, default="gaussian", choices=LIKELIHOOD_CHOICES,
        help="'student_t': outlier-robust Student-t variational GP surrogate (scripts/robust_gp.py).",
    )
    parser.add_argument(
        "--inference-rule", type=str, default=None, choices=INFERENCE_RULE_CHOICES,
        help="How the deployed design is picked: best single observation (default) or "
        "best mean over a design's ratings (the default once any replication is on).",
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


def aggregate_design_means(
    df: pd.DataFrame,
    param_columns: list[str],
    objective_columns: list[str],
) -> pd.DataFrame:
    """Collapse repeated evaluations of the same design to their mean response.

    Multiple participants rate the same design (identical ``param_columns``), so
    raw rows ask the oracle to predict between-participant variance from design
    parameters alone — an impossible regression that caps held-out R^2. Averaging
    per design yields an "average human" target whose surface BO can learn; the
    simulator's injected feedback noise models the individual deviation removed
    here. Returns one row per unique design with the mean of each required
    objective column.
    """
    required = list(dict.fromkeys(_objective_required_columns(objective_columns)))
    return df.groupby(param_columns, as_index=False)[required].mean()


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
    oracle_target: str = "individual",
) -> OracleModel:
    if oracle_target == "mean":
        df = aggregate_design_means(df, param_columns, objective_columns)
    elif oracle_target != "individual":
        raise ValueError(
            f"Unknown oracle_target '{oracle_target}' (expected 'individual' or 'mean')."
        )
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
            print(f"  train R^2 (in-sample, optimistic; CV fidelity is in best_oracle_models.json): {train_score:.4f}")
            print(f"  train RMSE: {train_rmse:.4f}")

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
    print(f"  train R^2 (in-sample, optimistic; CV fidelity is in best_oracle_models.json): {train_score:.4f}")
    print(f"  train RMSE: {train_rmse:.4f}")

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
        return TabPFNFloat32Regressor(
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
        return list(DEFAULT_ACQUISITION_CHOICES)
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
    # ts and aei are single-objective; the multi-objective branch above drops them.
    allowed = SINGLE_ACQUISITION_CHOICES + BASELINE_ACQUISITION_CHOICES + EXTENSION_ACQUISITION_CHOICES
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
        oracle_target = str(entry.get("oracle_target", "individual"))
        if oracle_target not in ("individual", "mean"):
            raise ValueError(
                f"Dataset '{name}' oracle_target must be 'individual' or 'mean', "
                f"got '{oracle_target}'."
            )

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
                oracle_target=oracle_target,
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

    if any(dataset.oracle_target != first.oracle_target for dataset in datasets[1:]):
        print("Cannot combine datasets with different oracle_target settings.", file=sys.stderr)
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
        oracle_target=first.oracle_target,
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


def self_reported_variance(error_magnitude, config: "SimulationConfig",
                           rng: np.random.Generator, apply_error: bool) -> float:
    """The variance a rater who reports their own confidence implies.

    The known-noise arm hands the GP the true but CONSTANT variance and
    recovers nothing, which is what a homoscedastic constant should do: the GP
    already learns one. A confidence report differs in kind because it is
    HETEROSCEDASTIC, saying which trials were hard. It is modelled as this
    trial's squared error seen through an honest but coarse rater, a log-normal
    multiplier of SD `confidence_noise`. At 0 the report is perfect, which
    prices the ceiling of the idea rather than a plausible rater.

    Drawn from a stream of its own, never the run rng, so a run with the arm on
    keeps common random numbers with the standard sweep.
    """
    if not apply_error:
        return 1e-6
    err = float(np.mean(np.atleast_1d(np.asarray(error_magnitude, dtype=float)) ** 2))
    scale = float(getattr(config, "confidence_noise", 0.5) or 0.0)
    if scale > 0:
        err *= float(np.exp(rng.normal(0.0, scale) - 0.5 * scale ** 2))
    return float(max(err, 1e-6))

def anchor_detrend(anchor_rows: list[int], anchor_ids: list[int],
                   observed: list[float], n_trials: int) -> np.ndarray:
    """A per-trial additive correction, read off the anchors alone.

    An anchor design has one true value, so the spread of its ratings across the
    session is pure rater movement: drift, a handover offset, a scale that
    compresses as the rater tires. Each anchor is centred on its own mean, which
    removes the design and leaves only the movement, and a least-squares line in
    the trial index is fitted to the pooled residuals. Two anchor ratings are
    needed before a line means anything; below that the correction is zero.

    A line, not a spline: with one anchor visit every few trials there are rarely
    more than a dozen points, and the failure this exists to fix -- a block
    handover confounded with the search trend -- is first order.
    """
    out = np.zeros(int(n_trials), dtype=float)
    if len(anchor_rows) < 3:
        return out
    rows = np.asarray(anchor_rows, dtype=int)
    ids = np.asarray(anchor_ids, dtype=int)
    vals = np.asarray([observed[r] for r in rows], dtype=float)
    resid = np.empty_like(vals)
    for a in np.unique(ids):
        sel = ids == a
        resid[sel] = vals[sel] - vals[sel].mean()
    if not np.all(np.isfinite(resid)) or np.ptp(rows) == 0:
        return out
    slope, intercept = np.polyfit(rows.astype(float), resid, 1)
    return intercept + slope * np.arange(n_trials, dtype=float)

def known_noise_variance(
    iteration: int, config: SimulationConfig, apply_error: bool
) -> float:
    """The variance of the error actually injected at this iteration.

    Mirrors ``apply_sensor_error`` exactly, so ``observation_noise="known"`` hands
    the GP the truth rather than an approximation of it:

      * before the onset (and on every iteration of a baseline run) the
        observation is exact;
      * ``gaussian`` / ``bias`` / ``drift`` all add N(0, jitter_std^2) -- the bias
        offset and the drift ramp move the MEAN, not the variance, and a GP's
        homoskedastic noise term models the variance;
      * ``ar1`` is stationary with SD ``jitter_std`` by construction;
      * ``spike`` adds an independent spike with probability p, so the marginal
        variance is ``std^2 + p * spike_std^2``;
      * ``dropout`` replaces the observation with the previous one, which is not
        additive noise at all and has no honest variance to declare;
      * a noise schedule divides the variance by the trial's effort e_t, and a
        rater offset moves the mean, like ``bias``, so it declares nothing;
      * a halo cross-correlation leaves each objective's variance at
        ``jitter_std^2`` -- it adds covariance between objectives, not variance.

    A small floor keeps the exact observations strictly positive, which BoTorch
    requires of ``train_Yvar``. 1e-6 rather than something smaller: on a
    standardised objective that is still numerically noise-free, but it leaves a
    nugget large enough that the Cholesky factorisation survives the
    near-duplicate designs BO piles up around an optimum.
    """
    if config.anchor_rating and apply_error and config.observation_noise == "known":
        # An anchored rating is the difference of two draws, so none of the
        # per-model variances below describes it. Rather than hand the GP a
        # number that is wrong by a factor of two, refuse the combination.
        raise ValueError(
            "--observation-noise known cannot be combined with --anchor-rating: "
            "the variance of a differenced rating is not the injected one.")
    floor = 1e-6
    if not apply_error or iteration <= config.jitter_iteration:
        return floor
    if config.single_error and iteration != config.jitter_iteration + 1:
        return floor
    if config.error_model == "none":
        # The input-error arm: the rating itself is exact, so there is no
        # observation noise to declare however large the positional slip was.
        return floor
    variance = float(config.jitter_std) ** 2
    if config.noise_schedule != "none":
        variance = variance / noise_effort(iteration, config)
    if config.error_model == "spike":
        variance += float(config.error_spike_prob) * float(config.error_spike_std) ** 2
    elif config.error_model == "dropout":
        raise ValueError(
            "observation_noise='known' is not defined for the dropout error model: "
            "holding the previous value is not additive noise."
        )
    return max(variance, floor)


# missing_mcar / missing_low (appended): the rating is LOST rather than the
# design moved -- the person skips the question, the sensor drops the sample.
# input_error_scale is the loss probability. See draw_missing_rating.
MISSING_INPUT_ERROR_CHOICES = ["missing_mcar", "missing_low"]
INPUT_ERROR_CHOICES = ["none", "slip", "misclick"] + MISSING_INPUT_ERROR_CHOICES
INPUT_ERROR_RECORDED_CHOICES = ["proposed", "actual"]


# ---------------------------------------------------------------------------
# Error-process extensions: effort schedules, missing ratings, relay raters and
# a rating-scale ceiling, with their remedies (relevance pursuit, imputation,
# offset backfitting). Each changes what a rating says; none adds a trial.
# ---------------------------------------------------------------------------

# Presets, written for the standard T = 50: only there do they average 1.
# back10 depends on T and is built in noise_schedule_efforts.
NOISE_SCHEDULE_PRESETS = {
    "front10": "1-10:2,11-:0.75",
    "front20": "1-20:1.5,21-:0.6667",
    "U": "1-20:1.6667,21-40:0.3333,41-:1",
}
_SCHEDULE_SEGMENT = re.compile(r"(?P<start>\d+)-(?P<end>\d*):(?P<effort>[^,:]+)")


@functools.lru_cache(maxsize=None)
def noise_schedule_efforts(spec: str, iterations: int) -> tuple[float, ...]:
    """Effort e_t for trials 1..T; the gaussian error SD at trial t is jitter_std / sqrt(e_t).

    A preset, or "a-b:effort" segments ("a-:effort" runs to T) covering every
    trial exactly once. A schedule only moves effort between trials, so it must
    average 1 over the run (within 1e-3); otherwise it would change how much
    error there is as well as when.
    """
    spec = str(spec).strip()
    T = int(iterations)
    if spec == "none":
        return (1.0,) * T
    hint = " The presets are written for T = 50." if spec in NOISE_SCHEDULE_PRESETS else ""
    if spec == "back10":
        if T <= 10:
            raise ValueError(f"--noise-schedule back10 needs more than 10 trials; this run has {T}.")
        custom = f"1-{T - 10}:0.75,{T - 9}-:2"
    else:
        custom = NOISE_SCHEDULE_PRESETS.get(spec, spec)
    efforts: list[float | None] = [None] * T
    for raw in custom.split(","):
        segment = raw.strip()
        match = _SCHEDULE_SEGMENT.fullmatch(segment)
        if match is None:
            raise ValueError(
                f"--noise-schedule segment {segment!r} is not 'a-b:effort' or 'a-:effort'. Use a "
                f"preset ({', '.join([*NOISE_SCHEDULE_PRESETS, 'back10'])}) or e.g. '1-10:2,11-:0.75'."
            )
        start = int(match["start"])
        end = int(match["end"]) if match["end"] else T
        try:
            effort = float(match["effort"])
        except ValueError:
            raise ValueError(f"--noise-schedule segment {segment!r}: the effort is not a number.") from None
        if not 1 <= start <= end <= T:
            raise ValueError(f"--noise-schedule segment {segment!r} covers trials outside 1..{T}.{hint}")
        if not (math.isfinite(effort) and effort > 0):
            raise ValueError(f"--noise-schedule segment {segment!r}: the effort must be positive.")
        for t in range(start, end + 1):
            if efforts[t - 1] is not None:
                raise ValueError(f"--noise-schedule {spec!r} gives trial {t} two efforts.")
            efforts[t - 1] = effort
    gaps = [t for t, e in enumerate(efforts, start=1) if e is None]
    if gaps:
        raise ValueError(f"--noise-schedule {spec!r} gives no effort to trial(s) {gaps[:5]} of 1..{T}.")
    mean = sum(efforts) / T
    if abs(mean - 1.0) > 1e-3:
        raise ValueError(
            f"--noise-schedule {spec!r} has mean effort {mean:.5f} over trials 1..{T}. A schedule "
            f"redistributes effort, so it must average 1 (within 1e-3).{hint}"
        )
    return tuple(float(e) for e in efforts)


def noise_effort(iteration: int, config: SimulationConfig) -> float:
    """The run's effort e_t at trial ``iteration`` (1-based)."""
    return noise_schedule_efforts(config.noise_schedule, config.iterations)[iteration - 1]


def noise_schedule_name(spec: str) -> str:
    """Filename-safe schedule name: the preset, or the custom spec with ':' -> '@' and ',' -> '+'.

    A colon is not a legal filename character on Windows.
    """
    spec = str(spec).strip()
    if spec in NOISE_SCHEDULE_PRESETS or spec == "back10":
        return spec
    return spec.replace(" ", "").replace(":", "@").replace(",", "+")


LANDSCAPE_SAMPLE_POINTS = 4096
LANDSCAPE_SAMPLE_SEED = 20_260_914
_LANDSCAPE_VALUES: dict[tuple, np.ndarray] = {}


def landscape_values(oracle: object, bounds: Bounds) -> np.ndarray:
    """The oracle at 4096 scrambled-Sobol points of the box, sorted, computed once per process.

    The Sobol seed is fixed, so a quantile is a property of the landscape alone
    and is shared by every seed, acquisition and condition. The engine has its
    own generator: neither the run rng nor torch's global one is touched.
    Analytic oracles are cached by name; any other oracle on the object itself.
    """
    low = np.asarray(bounds.low, dtype=float)
    high = np.asarray(bounds.high, dtype=float)
    name = getattr(oracle, "name", None)
    if isinstance(name, str):
        cache, key = _LANDSCAPE_VALUES, (type(oracle).__name__, name, tuple(low), tuple(high))
    else:
        cache, key = oracle.__dict__.setdefault("_landscape_values", {}), (tuple(low), tuple(high))
    if key not in cache:
        engine = torch.quasirandom.SobolEngine(dimension=len(low), scramble=True, seed=LANDSCAPE_SAMPLE_SEED)
        X = low + engine.draw(LANDSCAPE_SAMPLE_POINTS).to(torch.double).cpu().numpy() * (high - low)
        if hasattr(oracle, "predict_many"):
            values = np.asarray(oracle.predict_many(X), dtype=float).reshape(len(X), -1)[:, 0]
        else:
            values = np.array([float(np.asarray(oracle.predict(x), dtype=float).reshape(-1)[0]) for x in X])
        cache[key] = np.sort(values)
    return cache[key]


def landscape_quantile(oracle: object, bounds: Bounds, q: float) -> float:
    """The q-quantile of the oracle over its box (see landscape_values)."""
    return float(np.quantile(landscape_values(oracle, bounds), float(q)))


MISSING_LOW_QUANTILE = 0.4
IMPUTE_LOW_QUANTILE = 0.2
IMPUTE_LOW_MIN_RATINGS = 5


def draw_missing_rating(
    true_value: np.ndarray,
    iteration: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    low_threshold: float | None,
) -> bool:
    """Whether this trial's rating is lost.

    Onset as for the other channels: every rating up to t0 arrives.
    missing_mcar loses a rating with probability p = input_error_scale;
    missing_low loses it with the same p only when the design's true value is
    below the landscape's 40th percentile -- people skip what they dislike, so
    the loss is informative. One uniform is drawn per post-onset trial whatever
    the design, so a verdict never shifts the stream for the trials after it.
    """
    if config.input_error_model not in MISSING_INPUT_ERROR_CHOICES or iteration <= config.jitter_iteration:
        return False
    if config.single_error and iteration != config.jitter_iteration + 1:
        return False
    if rng.random() >= float(config.input_error_scale):
        return False
    if config.input_error_model == "missing_low":
        return float(true_value[0]) < float(low_threshold)
    return True


def impute_low_value(observed: list[float]) -> float | None:
    """The 20th percentile of the ratings received so far (their minimum below five); None if none."""
    values = np.asarray([v for v in observed if not np.isnan(v)], dtype=float)
    if values.size == 0:
        return None
    if values.size < IMPUTE_LOW_MIN_RATINGS:
        return float(values.min())
    return float(np.quantile(values, IMPUTE_LOW_QUANTILE))


def _missing_training_rows(observed: list[float], imputed: list[float]) -> tuple[list[int], list[float]]:
    """GP training rows and targets: each received rating, else its imputed value, else nothing."""
    rows: list[int] = []
    targets: list[float] = []
    for index, (rating, stand_in) in enumerate(zip(observed, imputed)):
        if not np.isnan(rating):
            rows.append(index)
            targets.append(float(rating))
        elif not np.isnan(stand_in):
            rows.append(index)
            targets.append(float(stand_in))
    return rows, targets


RATER_KINDS = ("none", "block", "roundrobin")
# Mixed into the rater-offset seed. The offsets must never come from jitter_rng:
# drawing them there would shift the rating noise away from the standard run's.
RATER_SEED_TAG = 52_617_465
BACKFIT_ROUNDS = 3


def parse_rater_assign(spec: str) -> tuple[str, int]:
    """("none", 0), ("block", K) or ("roundrobin", R) from "none", "block:K", "roundrobin:R"."""
    text = str(spec).strip()
    if text == "none":
        return "none", 0
    kind, sep, raw = text.partition(":")
    if kind not in RATER_KINDS[1:] or not sep:
        raise ValueError(f"--rater-assign must be 'none', 'block:K' or 'roundrobin:R'; got {text!r}.")
    try:
        count = int(raw)
    except ValueError:
        raise ValueError(f"--rater-assign {text!r}: {raw!r} is not a whole number.") from None
    if count < 1:
        raise ValueError(f"--rater-assign {text!r}: the count must be at least 1.")
    return kind, count


def rater_index(iteration: int, kind: str, count: int) -> int:
    """r(t) = (t - 1) // K for block hand-overs, (t - 1) % R for a round robin."""
    return (iteration - 1) // count if kind == "block" else (iteration - 1) % count


def rater_suffix(spec: str, ratio: float) -> str:
    """rater-<kind><count>-tau<ratio>; the colon of the flag is not legal in a Windows filename."""
    kind, count = parse_rater_assign(spec)
    return f"rater-{kind}{count}-tau{float(ratio):g}"


def draw_rater_offsets(
    seed: int, spec: str, ratio: float, jitter_std: float, iterations: int
) -> np.ndarray:
    """b_r ~ N(0, (ratio x jitter_std)^2) for every rater the run can meet.

    From a generator of its own, seeded by (seed, a fixed tag, the assignment,
    ratio), so the standard normals behind the offsets are shared across error
    magnitudes and acquisitions and the rating noise stays the standard run's.
    """
    kind, count = parse_rater_assign(spec)
    raters = math.ceil(int(iterations) / count) if kind == "block" else count
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), RATER_SEED_TAG, RATER_KINDS.index(kind), count, int(round(float(ratio) * 1_000_000))]
        )
    )
    return rng.standard_normal(raters) * (float(ratio) * float(jitter_std))


def fit_backfit_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    rater_ids: list[int],
    fit_gp: callable,
) -> tuple[object, object, dict[int, float]]:
    """Per-rater offsets estimated inside the GP fit, by backfitting.

    From b = 0, BACKFIT_ROUNDS rounds of: fit the GP on y - b[r]; set
    b_r = sum over rater r of (y - mu(x)) / (n_r + 1), the +1 shrinking a rater
    seen only a few times towards zero; centre b to an n-weighted mean of zero,
    because an offset every rater shares is the GP's constant mean. A final fit
    on y - b[r] is the model returned. b restarts at zero on every call, so the
    estimates depend on the current data alone.

    fit_gp(train_X, train_Y) -> (model, mll). Returns (model, mll, {rater: b_r}).
    """
    ids = np.asarray(rater_ids, dtype=int)
    raters, index = np.unique(ids, return_inverse=True)
    counts = np.bincount(index, minlength=len(raters)).astype(float)
    y = train_Y.detach().reshape(-1).cpu().numpy().astype(float)
    offsets = np.zeros(len(raters))
    for _ in range(BACKFIT_ROUNDS):
        model, _ = fit_gp(train_X, train_Y - torch.as_tensor(offsets[index], dtype=train_Y.dtype).reshape(-1, 1))
        with torch.no_grad():
            mu = model.posterior(train_X).mean.reshape(-1).cpu().numpy().astype(float)
        offsets = np.bincount(index, weights=y - mu, minlength=len(raters)) / (counts + 1.0)
        offsets = offsets - np.sum(counts * offsets) / np.sum(counts)
    model, mll = fit_gp(train_X, train_Y - torch.as_tensor(offsets[index], dtype=train_Y.dtype).reshape(-1, 1))
    return model, mll, {int(r): float(b) for r, b in zip(raters, offsets)}


def _fit_gaussian_gp(
    train_X: torch.Tensor, train_Y: torch.Tensor, train_Yvar: torch.Tensor | None, bounds_tensor: torch.Tensor
) -> tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
    """The sweep's standard surrogate, as run_simulation builds it inline."""
    gp = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp, mll


def fit_relevance_pursuit_gp(
    train_X: torch.Tensor, train_Y: torch.Tensor, bounds_tensor: torch.Tensor
) -> tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
    """Robust GP via Relevance Pursuit (Ament et al., 2024, arXiv:2410.24222).

    Every training point carries its own outlier variance; the fit decides how
    many are non-zero. Same input normalisation and outcome standardisation as
    the standard surrogate, fitted as BoTorch documents it: an
    ExactMarginalLogLikelihood through fit_gpytorch_mll, which dispatches to the
    model's custom_fit (backward relevance pursuit over the outlier support,
    then Bayesian model selection of its size).
    """
    from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP

    gp = RobustRelevancePursuitSingleTaskGP(
        train_X,
        train_Y,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp, mll


def _fit_model_list_gp(
    train_X: torch.Tensor, train_Y_list: list[torch.Tensor], bounds_tensor: torch.Tensor
) -> tuple[ModelListGP, SumMarginalLogLikelihood]:
    """The multi-objective surrogate: one standard GP per objective, fitted jointly."""
    gps = [
        SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
            outcome_transform=Standardize(m=1),
        )
        for train_Y in train_Y_list
    ]
    gp = ModelListGP(*gps)
    mll = SumMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp, mll


HALO_RHO_MAX = 0.95
# The correlation of two points is +-1 whatever the data, so it estimates nothing.
HALO_MIN_ROWS = 3


def halo_shared_factor(residuals: np.ndarray) -> tuple[float, np.ndarray]:
    """rho_hat and the estimated shared factor z_t, from standardised residuals (n trials x m objectives).

    If r_tj = sqrt(1 - rho) eps_tj + sqrt(rho) z_t with eps and z independent
    N(0, 1), r_t has covariance (1 - rho) I + rho 11^T, which maps 1 to
    (1 + (m - 1) rho) 1, so E[z_t | r_t] = sqrt(rho) m mean_j(r_tj) / (1 + (m - 1) rho).
    rho_hat is the mean off-diagonal correlation of the residual columns (pairs
    involving a constant column have none and are skipped), clipped to
    [0, 0.95]: a negative mean is no shared factor, and the cap keeps some of
    each residual for the objective's own error. Below three trials rho_hat is 0.
    """
    R = np.asarray(residuals, dtype=float)
    n, m = R.shape
    rho_hat = 0.0
    if n >= HALO_MIN_ROWS and m >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(R, rowvar=False)
        off_diagonal = corr[~np.eye(m, dtype=bool)]
        off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
        if off_diagonal.size:
            rho_hat = float(np.clip(off_diagonal.mean(), 0.0, HALO_RHO_MAX))
    shared = math.sqrt(rho_hat) * m * R.mean(axis=1) / (1.0 + (m - 1) * rho_hat)
    return rho_hat, shared


def fit_mo_halo_backfit(
    gp: ModelListGP,
    train_X: torch.Tensor,
    train_Y_list: list[torch.Tensor],
    fit_models: callable,
) -> tuple[ModelListGP, list[torch.Tensor], np.ndarray, float]:
    """Remove a per-trial factor shared by every objective's rating, then refit.

    From the fitted ModelListGP: r_tj = (y_tj - mu_j(x_t)) / sigma_nj, with
    mu_j the posterior mean at the trial's design and sigma_nj the learned
    noise SD of objective j in output units; rho_hat and z_t from
    halo_shared_factor; y_tj <- y_tj - sigma_nj sqrt(rho_hat) z_t; and
    fit_models(train_X, corrected) -> (model, mll) refits. One pass, from the
    current data alone, so earlier estimates never carry over.

    Returns (model, corrected targets, the (n, m) correction subtracted, rho_hat).
    """
    Y = torch.cat([y.detach() for y in train_Y_list], dim=1).cpu().numpy().astype(float)
    with torch.no_grad():
        mu = np.column_stack(
            [model.posterior(train_X).mean.reshape(-1).cpu().numpy().astype(float) for model in gp.models]
        )
    sigma = np.array([observation_noise_sd(model) for model in gp.models], dtype=float)
    rho_hat, shared = halo_shared_factor((Y - mu) / sigma[None, :])
    correction = sigma[None, :] * math.sqrt(rho_hat) * shared[:, None]
    if rho_hat == 0.0:
        # Nothing to remove: the corrected targets are the ratings themselves, and
        # a second fit on them could only move to another optimum of the same fit.
        # Exact zeros: sqrt(0) * z is -0.0 wherever z < 0, which reads oddly in a log.
        return gp, list(train_Y_list), np.zeros_like(correction), rho_hat
    corrected = [
        y - torch.as_tensor(correction[:, [j]], dtype=y.dtype) for j, y in enumerate(train_Y_list)
    ]
    model, _ = fit_models(train_X, corrected)
    return model, corrected, correction, rho_hat


CEILING_ANCHOR_MARGIN = 0.5


def response_ceiling_value(base: float, mode: str, observed: list[float], true: list[float]) -> float:
    """The cap on this trial's rating.

    fixed: the landscape quantile. anchored: at least 0.5 above the true value
    of the design with the highest rating among the EARLIER trials -- the person
    rates relative to the best design seen so far, which moves the top of the
    scale. The value is the one the person experienced (objective_true).
    """
    if mode == "anchored":
        ratings = np.asarray(observed, dtype=float)
        if ratings.size and not np.isnan(ratings).all():
            return max(float(base), float(true[int(np.nanargmax(ratings))]) + CEILING_ANCHOR_MARGIN)
    return float(base)


def validate_error_extensions(
    acquisition: str,
    settings: dict,
    *,
    is_multi: bool,
    noisy: bool = True,
) -> None:
    """Reject combinations the error-process extensions cannot run.

    ``settings`` holds SimulationConfig field names: run_simulation passes its
    config's fields, the synthetic driver the CLI's. Checks on the noisy
    process -- the response model a schedule or a rater needs, a missing rate
    being a probability -- apply only when ``noisy``, since a clean twin never
    draws them. Called per run and once before a sweep starts.
    """
    get = settings.get
    iterations = int(get("iterations"))
    error_model = str(get("error_model", "none"))
    input_error = str(get("input_error_model", "none"))
    likelihood = str(get("likelihood", "gaussian"))
    observation_noise = str(get("observation_noise", "learned"))
    input_noise_model = str(get("input_noise_model", "none"))
    missing_handling = str(get("missing_handling", "drop"))
    rater_model = str(get("rater_model", "none"))
    ceiling_mode = str(get("ceiling_mode", "fixed"))
    for value, choices, flag in (
        (likelihood, LIKELIHOOD_CHOICES, "--likelihood"),
        (input_error, INPUT_ERROR_CHOICES, "--input-error"),
        (missing_handling, MISSING_HANDLING_CHOICES, "--missing-handling"),
        (rater_model, RATER_MODEL_CHOICES, "--rater-model"),
        (ceiling_mode, CEILING_MODE_CHOICES, "--ceiling-mode"),
        (str(get("mo_halo_model", "none")), MO_HALO_MODEL_CHOICES, "--mo-halo-model"),
    ):
        if value not in choices:
            raise ValueError(f"{flag} {value!r} is not one of {choices}.")

    # (2) relevance pursuit: the restrictions of the Student-t surrogate.
    if likelihood == "relevance_pursuit" and (
        is_multi or observation_noise in ("known", "self_report") or input_noise_model != "none"
    ):
        raise ValueError(
            "likelihood='relevance_pursuit' is single-objective with learned noise and no "
            "input-noise model, like student_t: it fits its own per-point noise."
        )

    # (1) the effort schedule
    schedule = str(get("noise_schedule", "none"))
    if schedule != "none":
        noise_schedule_efforts(schedule, iterations)  # raises on a malformed or unbalanced schedule
        if noisy and error_model != "gaussian":
            raise ValueError(
                f"--noise-schedule rescales the gaussian rating error, and this run's response error "
                f"is {error_model!r}. Use --error-models gaussian, without --input-error-from-sweep."
            )

    # (3) missing ratings
    missing = input_error in MISSING_INPUT_ERROR_CHOICES
    if missing_handling != "drop" and not missing:
        raise ValueError(
            "--missing-handling acts only on a lost rating; pass --input-error missing_mcar or missing_low."
        )
    if missing:
        if is_multi:
            raise ValueError(f"{input_error} is single-objective only, like every input-error process.")
        if int(get("replicate_first", 0) or 0) or int(get("final_rerate_top", 0) or 0) or acquisition == "replei":
            raise ValueError(
                f"{input_error} cannot run with replication, re-rating or replei: each re-rates a design "
                "whose ratings may be lost, and none has a rule for a lost repeat."
            )
        if input_noise_model != "none":
            raise ValueError(f"{input_error} cannot run with the noisy-input GP: a lost rating is not a slip.")
        if str(get("input_error_recorded", "proposed")) != "proposed":
            raise ValueError(
                f"{input_error} never moves the design, so there is no 'actual' design to record; "
                "drop --input-error-recorded."
            )
        if missing_handling == "impute_low" and observation_noise in ("known", "self_report"):
            raise ValueError(
                "--missing-handling impute_low with --observation-noise known: an imputed value has no "
                "injected variance to declare."
            )
        rate = float(get("input_error_scale", 0.0))
        if noisy and not 0.0 <= rate <= 1.0:
            raise ValueError(f"The {input_error} rate is a probability and must lie in [0, 1]; got {rate:g}.")

    # (4) relay raters
    kind, _ = parse_rater_assign(str(get("rater_assign", "none")))
    ratio = float(get("rater_offset_ratio", 0.0))
    if not (math.isfinite(ratio) and ratio >= 0.0):
        raise ValueError(f"--rater-offset-ratio must be a non-negative number; got {ratio:g}.")
    if kind == "none":
        if ratio != 0.0:
            raise ValueError("--rater-offset-ratio needs --rater-assign block:K or roundrobin:R.")
        if rater_model != "none":
            raise ValueError("--rater-model backfit estimates per-rater offsets and needs --rater-assign.")
    else:
        if is_multi:
            raise ValueError("Relay raters are single-objective only; backfitting works on the scalar GP.")
        if ratio == 0.0 and rater_model == "none":
            raise ValueError(
                "--rater-assign with --rater-offset-ratio 0 and no --rater-model would do nothing."
            )
        if noisy and error_model != "gaussian":
            raise ValueError(
                f"Rater offsets are added to the gaussian rating error, and this run's response error is "
                f"{error_model!r}. Use --error-models gaussian, without --input-error-from-sweep."
            )
    if rater_model == "backfit" and (likelihood != "gaussian" or input_noise_model != "none"):
        raise ValueError(
            "--rater-model backfit refits the standard Gaussian GP; it cannot run with another "
            "--likelihood or with --input-noise-model."
        )

    # (5) the rating-scale ceiling
    ceiling = float(get("response_ceiling", 0.0))
    if not 0.0 <= ceiling < 1.0:
        raise ValueError(
            f"--response-ceiling is a quantile of the landscape and must lie in (0, 1); got {ceiling:g}."
        )
    if ceiling > 0.0 and is_multi:
        raise ValueError("--response-ceiling is single-objective only.")
    if ceiling == 0.0 and ceiling_mode != "fixed":
        raise ValueError("--ceiling-mode needs --response-ceiling.")

    # (6) the multi-objective halo error and its backfit
    cross_corr = float(get("error_cross_corr", 0.0))
    if not (math.isfinite(cross_corr) and 0.0 <= cross_corr <= 1.0):
        raise ValueError(
            f"--error-cross-corr is the correlation of one trial's rating errors across objectives and "
            f"must lie in [0, 1]; got {cross_corr:g}."
        )
    if cross_corr > 0.0:
        if not is_multi:
            raise ValueError(
                "--error-cross-corr correlates the rating errors of a design's objectives, so it is "
                "multi-objective only; pass --multi-objective."
            )
        if noisy and error_model != "gaussian":
            raise ValueError(
                f"--error-cross-corr shares a factor across the gaussian rating errors, and this run's "
                f"response error is {error_model!r}. Use --error-models gaussian."
            )
    if str(get("mo_halo_model", "none")) != "none":
        if not is_multi:
            raise ValueError(
                "--mo-halo-model backfit removes a factor shared across a ModelListGP's objectives, so it is "
                "multi-objective only; pass --multi-objective."
            )
        if acquisition in BASELINE_ACQUISITION_CHOICES:
            raise ValueError(
                f"--mo-halo-model backfit acts inside the surrogate fit, and '{acquisition}' fits no "
                "surrogate, so its runs would be the standard arm's. Name the hypervolume acquisitions "
                "with --acq-list."
            )


def run_error_label(error_model: str, input_error_model: str) -> str:
    """What corrupted a run, as it is written down.

    "none" is reserved for the clean baseline: the evaluator recognises a
    baseline by it, so a corrupted run labelled "none" is filed as a baseline
    and never paired. A run is therefore labelled by whatever was actually
    applied -- the response model, the input model, or both joined by "+" --
    and asking for a label when nothing was applied is an error, not a default.
    """
    response = str(error_model)
    inp = str(input_error_model)
    if inp == "none":
        if response == "none":
            raise ValueError(
                "A jittered run with no corruption: both the response error and the "
                "input error are 'none'. That is a baseline, and 'none' is its label."
            )
        return response
    if response == "none":
        return inp
    return f"{response}+{inp}"


def apply_input_error(
    candidate: np.ndarray,
    iteration: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    bounds: "Bounds",
) -> tuple[np.ndarray, bool]:
    """Where the person actually went, given where the optimizer sent them.

    Returns the design to EVALUATE and whether a slip occurred. Onset follows
    the same convention as the response models: exact for ``t <= t0``, so the
    first corrupted trial is ``t0 + 1``.

    The slip is clipped back into the box rather than resampled or reflected.
    A resample would make the error distribution depend on how close the
    proposal sits to a wall in a way that is hard to state; clipping is what a
    real bounded control does when you overshoot it -- the slider stops.
    """
    # A missing-rating process loses the rating, never the design; see draw_missing_rating.
    if (
        config.input_error_model == "none"
        or config.input_error_model in MISSING_INPUT_ERROR_CHOICES
        or iteration <= config.jitter_iteration
    ):
        return candidate, False
    if config.single_error and iteration != config.jitter_iteration + 1:
        return candidate, False

    scale = float(config.input_error_scale)
    if scale <= 0.0:
        return candidate, False

    low = np.asarray(bounds.low, dtype=float)
    high = np.asarray(bounds.high, dtype=float)

    if config.input_error_model == "slip":
        step = rng.normal(0.0, scale * (high - low), size=candidate.shape)
        return np.clip(candidate + step, low, high), True

    if config.input_error_model == "misclick":
        if rng.random() >= scale:
            return candidate, False
        return low + rng.random(candidate.shape) * (high - low), True

    raise ValueError(
        f"Unknown input error model: {config.input_error_model!r}. "
        f"Expected one of {INPUT_ERROR_CHOICES}."
    )


def apply_sensor_error(
    true_value: np.ndarray,
    iteration: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    previous_observed: np.ndarray,
    previous_error: np.ndarray | None = None,
    rater_offset: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # rater_offset: the relay rater's fixed offset, added after the error and
    # before the response instrument; None (the default) adds nothing.
    if iteration <= config.jitter_iteration:
        return true_value, np.zeros_like(true_value, dtype=float)
    if config.single_error and iteration != config.jitter_iteration + 1:
        return true_value, np.zeros_like(true_value, dtype=float)
    # "none" is not a no-op arm: it is how the INPUT-error arm runs, where the
    # design is corrupted and the rating of it is honest. Return before drawing
    # so the response channel consumes no randomness and the input channel gets
    # a clean stream from the shared jitter_rng.
    if config.error_model == "none":
        return true_value, np.zeros_like(true_value, dtype=float)

    jitter = rng.normal(0.0, config.jitter_std, size=true_value.shape)
    if config.error_cross_corr > 0 and config.error_model == "gaussian":
        # Halo error: the standard per-objective draw first, then ONE shared
        # draw, so rho = 0 consumes exactly the standard stream and rho > 0 keeps
        # every eps_j the standard run's.
        halo_rho = float(config.error_cross_corr)
        shared = rng.normal(0.0, config.jitter_std)
        jitter = math.sqrt(1.0 - halo_rho) * jitter + math.sqrt(halo_rho) * shared
    if config.noise_schedule != "none":
        # The standard run's draw, rescaled to SD jitter_std / sqrt(e_t): the
        # schedule moves error between trials without touching the stream.
        jitter = jitter / math.sqrt(noise_effort(iteration, config))

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
        # jitter_std at the final iteration, ON TOP of the ordinary gaussian
        # rating inconsistency (real fatigue adds to noise rather than
        # replacing it; a review found the earlier ramp-only variant made
        # "drift" the only deterministic, jitter-free condition). Peak ramp
        # = jitter_std keeps the sweep a single swept factor; note the
        # drift condition therefore carries more total error energy than
        # pure gaussian at the same nominal std.
        span = max(1, config.iterations - config.jitter_iteration)
        ramp = config.jitter_std * (iteration - config.jitter_iteration) / span
        observed = true_value + np.full_like(true_value, ramp, dtype=float) + jitter
    elif config.error_model == "ar1":
        # Serially correlated rating error: e_t = rho * e_{t-1} + innovation,
        # with stationary SD equal to jitter_std. The FIRST post-onset error
        # is a full stationary draw (reusing `jitter`, already N(0, std)),
        # so early errors are not systematically smaller than the stationary
        # SD (cold-start fix: e_0 = 0 gave first-step SD of only
        # std*sqrt(1-rho^2), ~0.6x std at rho=0.8, exactly inside the
        # response-step window the evaluation measures).
        rho = float(config.error_ar1_rho)
        if iteration == config.jitter_iteration + 1:
            observed = true_value + jitter
        else:
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

    if rater_offset is not None:
        observed = observed + rater_offset
    if config.anchor_rating:
        # Judged against the incumbent shown beside it. Everything the rater
        # gets wrong about BOTH designs at this moment -- a constant bias, the
        # drift ramp so far, the carried AR(1) state, a relay offset -- is
        # common to the pair and cancels. What is drawn fresh per judgement
        # does not, and the anchor contributes a draw of its own, so the
        # idiosyncratic part grows by sqrt(2). That trade is the arm: it should
        # pay under bias and drift and cost under pure gaussian noise.
        anchor_jitter = rng.normal(0.0, config.jitter_std, size=true_value.shape)
        observed = true_value + (jitter - anchor_jitter)
    observed = _postprocess_response(observed, config)
    return observed, observed - true_value


def screen_candidate_pool(
    acqf: object,
    bounds: Bounds,
    rng: np.random.Generator,
    candidate_pool: int,
    num_restarts: int,
    return_pool: bool = False,
) -> tuple[torch.Tensor, ...]:
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
    if return_pool:
        # The whole screened pool and its values, for the min-distance redirect.
        # The draw is the same either way, so asking for them changes nothing else.
        return best_candidate, initial_conditions, candidate_tensor, acq_values.detach()
    return best_candidate, initial_conditions


# ---------------------------------------------------------------------------
# Acquisition-side follow-ups: the lcb incumbent, Thompson sampling, augmented
# EI, the input-uncertain wrapper and the min-distance redirect. Each changes
# which design is proposed; none changes how many designs are rated.
# ---------------------------------------------------------------------------


def _posterior_mean_and_lcb(model: object, train_X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Latent posterior mean, and mean - 1 x latent SD, at the visited designs.

    Latent (no observation noise): the band is the model's uncertainty about f,
    which is what an incumbent should be discounted by, not the rater's noise.
    """
    with torch.no_grad():
        posterior = model.posterior(train_X)
        mean = posterior.mean.reshape(-1)
        sd = posterior.variance.clamp_min(0.0).sqrt().reshape(-1)
    return mean, mean - sd


def observation_noise_sd(model: object) -> float:
    """The surrogate's observation-noise SD in the model's OUTPUT units.

    The likelihood lives in standardised units (Standardize outcome transform),
    so its variance is scaled back by stdvs^2 -- the conversion
    _refit_with_input_noise uses. A fixed per-point noise (the nigp refit) is
    summarised by its mean variance, and a Student-t likelihood by its variance
    scale^2 * nu / (nu - 2), finite because robust_gp keeps nu > 2.
    """
    likelihood = model.likelihood
    noise = getattr(likelihood, "noise", None)
    if noise is None:
        # The relevance-pursuit likelihood has no single noise: it keeps the
        # inlier noise in a base module and adds a variance per outlier. A new
        # rating is an inlier's until shown otherwise, so the base noise applies.
        noise = likelihood.noise_covar.base_noise.noise
    variance = noise.detach().reshape(-1).to(torch.double).mean()
    deg_free = getattr(likelihood, "deg_free", None)
    if deg_free is not None:
        nu = float(deg_free.detach().reshape(-1)[0])
        variance = variance * nu / (nu - 2.0)
    stdvs = getattr(getattr(model, "outcome_transform", None), "stdvs", None)
    if stdvs is not None:
        variance = variance * stdvs.detach().reshape(-1)[0].to(torch.double) ** 2
    return float(variance.sqrt())


class ShipRuleExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Expected improvement in the value the CAUTIOUS SHIP RULE would deliver.

    Ordinary EI asks what a rating does to the best posterior mean. A study does
    not ship the best posterior mean; it ships the best lower confidence bound,
    and the decomposition of this paper says that choice, not the search, is
    where the cost of feedback error sits. So value a candidate by what one
    rating of it does to that bound:

        sigma_post^2 = sigma^2 sn^2 / (sigma^2 + sn^2)     the SD left after one rating
        v^2          = sigma^2 - sigma_post^2              how far the mean may move
        A(x)         = E[ max(0, mu + v Z - beta sigma_post - c) ]

    with c the bound the rule already achieves. The expectation is the usual
    analytic EI with threshold c + beta sigma_post and scale v, so this is no
    more expensive than EI. Where the model is already sure, v is small and the
    candidate cannot move the decision; where it is unsure, beta sigma_post
    penalises it for still being unsure AFTER the rating. Not log-valued: the
    threshold moves with x, so the log form has no stable best_f to subtract.
    """

    def __init__(
        self,
        model: object,
        best_lcb: float | torch.Tensor,
        noise_sd: float | torch.Tensor,
        beta: float = 1.0,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model)
        self.register_buffer("best_lcb", torch.as_tensor(best_lcb))
        self.register_buffer("noise_sd", torch.as_tensor(noise_sd))
        self.beta = float(beta)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        mu = posterior.mean.squeeze(-1).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
        if not self.maximize:
            mu = -mu
        sn = self.noise_sd.to(mu)
        sigma_post = (sigma * sn / torch.sqrt(sigma**2 + sn**2).clamp_min(1e-12))
        # How far one rating can still move the mean: the total variance minus
        # what is left. Clamped because both terms are estimates.
        v = (sigma**2 - sigma_post**2).clamp_min(1e-12).sqrt()
        threshold = self.best_lcb.to(mu) + self.beta * sigma_post
        z = (mu - threshold) / v
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return v * (z * normal.cdf(z) + torch.exp(normal.log_prob(z)))


def build_ship_rule_ei(model: object, train_X: torch.Tensor, noise_sd: float,
                       beta: float = 1.0) -> "ShipRuleExpectedImprovement":
    """The acquisition, with the bound the ship rule currently achieves."""
    # The helper is the loop's own lcb incumbent: mean minus ONE latent SD.
    _, lcb = _posterior_mean_and_lcb(model, train_X)
    return ShipRuleExpectedImprovement(
        model=model, best_lcb=float(lcb.max()), noise_sd=float(noise_sd), beta=beta)


class LogAugmentedExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Log augmented expected improvement (Huang, Allen, Notz & Zeng, 2006).

    ``log AEI(x) = LogEI(x; best_f) + log(1 - sigma_n / sqrt(sigma(x)^2 + sigma_n^2))``

    with sigma(x) the LATENT posterior SD and sigma_n the observation-noise SD,
    both in output units. The factor is near 1 where the model is unsure and
    falls to 0 as sigma(x) drops below sigma_n: another noisy rating of a design
    the model already pins down buys almost nothing. Log-valued like LogEI.
    """

    _log: bool = True

    def __init__(
        self,
        model: object,
        best_f: float | torch.Tensor,
        noise_sd: float | torch.Tensor,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("noise_sd", torch.as_tensor(noise_sd))
        self.maximize = maximize

    @staticmethod
    def log_penalty(sigma: torch.Tensor, noise_sd: torch.Tensor) -> torch.Tensor:
        """log(1 - s_n / t) with t = sqrt(s^2 + s_n^2), without the cancellation.

        1 - s_n / t = s^2 / (t (t + s_n)), which stays accurate for s << s_n --
        exactly where the discount does its work.
        """
        total = torch.sqrt(sigma**2 + noise_sd**2)
        return 2.0 * sigma.log() - total.log() - (total + noise_sd).log()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean, sigma = self._mean_and_sigma(X)  # latent posterior
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        log_ei = _log_ei_helper(u) + sigma.log()
        return (log_ei + self.log_penalty(sigma, self.noise_sd)).squeeze(-1)


def build_augmented_ei(
    model: object, train_X: torch.Tensor, noise_sd: float | None = None
) -> LogAugmentedExpectedImprovement:
    """AEI with its own threshold: the posterior mean at the visited design with
    the best mean - 1 latent SD.

    The method defines the threshold, so --incumbent and xi do not apply to it.
    noise_sd overrides the model's own estimate (the known-noise arm passes the
    variance of the rating about to be taken).
    """
    mean, lcb = _posterior_mean_and_lcb(model, train_X)
    threshold = float(mean[int(torch.argmax(lcb))])
    sd = observation_noise_sd(model) if noise_sd is None else float(noise_sd)
    return LogAugmentedExpectedImprovement(model=model, best_f=threshold, noise_sd=sd)


class _MatheronThompsonSampling(AcquisitionFunction):
    """One Matheron posterior path as an acquisition, for BoTorch releases that
    lack PathwiseThompsonSampling. Draws and evaluates the path the same way."""

    def __init__(self, model: object) -> None:
        super().__init__(model=model)
        self.batch_size: int | None = None
        self.path = None

    def redraw(self, batch_size: int) -> None:
        from botorch.sampling.pathwise import draw_matheron_paths

        self.path = draw_matheron_paths(self.model, sample_shape=torch.Size([batch_size]))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.path is None:
            self.batch_size = int(X.shape[-2])
            self.redraw(self.batch_size)
        # Each point is evaluated as its own n = 1 input, as the pathwise
        # acquisition does; a path is deterministic, so this is just its value.
        values = self.path(X.unsqueeze(-2))
        return values.reshape(*X.shape[:-1]).sum(-1)


def build_thompson_sampling(model: object, rng: np.random.Generator) -> AcquisitionFunction:
    """Thompson sampling: one posterior sample path, drawn now.

    The path's random features come from torch's global generator. Seeding it
    from the run rng makes the draw a function of the run seed alone, and
    fork_rng keeps the draw from shifting that generator for the rest of the run.
    """
    acqf = (
        PathwiseThompsonSampling(model=model)
        if PathwiseThompsonSampling is not None
        else _MatheronThompsonSampling(model)
    )
    path_seed = int(rng.integers(0, 2**31 - 1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(path_seed)
        # Draw eagerly: left alone, the path is drawn lazily on the first call,
        # outside the seeded block. The batch-size check needs batch_size set too.
        acqf.batch_size = 1
        acqf.redraw(batch_size=1)
    return acqf


class InputUncertainAcquisition(AcquisitionFunction):
    """An acquisition averaged over K fixed perturbations of the design.

    The value at x is the average of base(clip(x + delta_k)) over the K deltas:
    arithmetic for acquisitions on their natural scale, log-mean-exp for
    log-valued ones, so it is the acquisition that is averaged and not its log.
    The deltas are fixed for the object's lifetime, which keeps the averaged
    surface deterministic for the gradient optimiser.
    """

    def __init__(
        self,
        base: object,
        perturbations: torch.Tensor,
        bounds_tensor: torch.Tensor,
        log_space: bool,
    ) -> None:
        super().__init__(model=base.model)
        self.base = base
        self.register_buffer("perturbations", perturbations.to(torch.double))
        self.register_buffer("low", bounds_tensor[0].to(torch.double))
        self.register_buffer("high", bounds_tensor[1].to(torch.double))
        self.log_space = bool(log_space)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # One base call per perturbation rather than one call on a K-times larger
        # batch: peak memory stays that of the unwrapped acquisition, which
        # matters for qnei's joint posteriors over a 1000-point screening pool.
        values = torch.stack(
            [
                self.base(torch.minimum(torch.maximum(X + delta, self.low), self.high))
                for delta in self.perturbations
            ],
            dim=-1,
        )
        if self.log_space:
            return torch.logsumexp(values, dim=-1) - math.log(values.shape[-1])
        return values.mean(dim=-1)


def draw_input_perturbations(
    rng: np.random.Generator, bounds: Bounds, count: int, scale: float
) -> torch.Tensor:
    """K QMC normal perturbations in raw input units, SD scale x (high - low).

    The scrambled-Sobol seed comes from the run rng, so the deltas are fixed by
    the run seed; they are redrawn once per model-based iteration.
    """
    engine = NormalQMCEngine(d=len(bounds.low), seed=int(rng.integers(0, 2**31 - 1)))
    z = engine.draw(int(count)).to(torch.double)
    span = torch.as_tensor(
        np.asarray(bounds.high, dtype=float) - np.asarray(bounds.low, dtype=float),
        dtype=torch.double,
    )
    return z * (float(scale) * span)


def input_uncertain_effective_scale(config: SimulationConfig, apply_error: bool) -> float:
    """The perturbation SD (fraction of each range) the wrapper uses in a run; 0 = off.

    An explicit input_uncertain_scale >= 0 is used as given, in the clean run
    too. The default borrows the arm's slip SD, which exists only where a slip
    can happen -- a corrupted run of an input-error arm -- and is 0 elsewhere.
    It applies from the first trial even when slips have a later onset: the
    optimiser is not told when slips start.
    """
    if config.input_uncertain_acq <= 0:
        return 0.0
    if config.input_uncertain_scale >= 0:
        return float(config.input_uncertain_scale)
    if apply_error and config.input_error_model != "none":
        return float(config.input_error_scale)
    return 0.0


def validate_acquisition_extensions(
    acquisition: str,
    *,
    is_multi: bool,
    input_error_model: str = "none",
    input_uncertain_acq: int = 0,
    input_uncertain_scale: float = -1.0,
    min_distance: float = 0.0,
) -> None:
    """Reject combinations the acquisition-side follow-ups cannot run.

    Called per run by run_simulation and once up front by the synthetic driver,
    so a bad combination fails before a sweep starts instead of in every worker.
    """
    if acquisition in EXTENSION_ACQUISITION_CHOICES and is_multi:
        raise ValueError(
            f"Acquisition '{acquisition}' is single-objective only; the hypervolume "
            "family is the multi-objective counterpart."
        )
    if input_uncertain_acq < 0:
        raise ValueError("input_uncertain_acq must be >= 0 (0 = off).")
    if input_uncertain_acq > 0:
        if is_multi:
            raise ValueError(
                "The input-uncertain acquisition is single-objective only, like the "
                "input-error arm it answers."
            )
        if acquisition == "qkg":
            raise ValueError(
                "The input-uncertain acquisition cannot wrap qkg: a one-shot acquisition "
                "is optimised jointly with its fantasy points and has no screening pool."
            )
        if input_uncertain_scale < 0 and input_error_model in ("misclick", *MISSING_INPUT_ERROR_CHOICES):
            raise ValueError(
                f"The {input_error_model} arm's input_error_scale is a probability, not a slip SD, so "
                "the input-uncertain acquisition cannot borrow it; pass an explicit "
                "--input-uncertain-scale."
            )
        if input_uncertain_scale < 0 and input_error_model == "none":
            raise ValueError(
                "The input-uncertain acquisition borrows its scale from the input-error arm "
                "by default, and none is set, so it would do nothing; pass "
                "--input-uncertain-scale or run it with --input-error slip."
            )
    if not 0.0 <= min_distance < 1.0:
        raise ValueError(
            "min_distance is an RMS distance in the unit box, whose largest value is 1; "
            "it must lie in [0, 1)."
        )
    if min_distance > 0 and acquisition == "qkg":
        raise ValueError(
            "min_distance redirects to a screened pool point, and qkg (a one-shot "
            "acquisition) is optimised without a screening pool."
        )


def _rms_unit_distance(points: np.ndarray, logged: np.ndarray, bounds: Bounds) -> np.ndarray:
    """RMS distance from each point to its nearest logged design, in the unit box.

    RMS over coordinates rather than Euclidean, so one threshold means the same
    thing at every dimension (two random designs sit ~0.41 apart at any d).
    """
    low = np.asarray(bounds.low, dtype=float)
    span = np.asarray(bounds.high, dtype=float) - low
    p = (np.atleast_2d(np.asarray(points, dtype=float)) - low) / span
    q = (np.atleast_2d(np.asarray(logged, dtype=float)) - low) / span
    diff = p[:, None, :] - q[None, :, :]
    return np.sqrt(np.mean(diff**2, axis=-1)).min(axis=1)


def _min_distance_redirect(
    candidate: torch.Tensor,
    pool: torch.Tensor,
    pool_values: torch.Tensor,
    logged_X: np.ndarray,
    bounds: Bounds,
    min_distance: float,
) -> tuple[torch.Tensor, bool]:
    """Swap a near-repeat proposal for the best screened point far from the log.

    Returns (design, redirected). The candidate is kept when it is already at
    least min_distance from every logged design, or when no pool point is.
    """
    nearest = _rms_unit_distance(candidate.detach().cpu().numpy().reshape(1, -1), logged_X, bounds)
    if nearest[0] >= min_distance:
        return candidate, False
    far = _rms_unit_distance(pool.detach().cpu().numpy(), logged_X, bounds) >= min_distance
    if not far.any():
        return candidate, False
    # NaN would win np.argmax; an acquisition value that is NaN is no value.
    values = np.nan_to_num(pool_values.detach().cpu().numpy().astype(float), nan=-np.inf)
    eligible = np.flatnonzero(far)
    best = int(eligible[np.argmax(values[eligible])])
    return pool[best].reshape(candidate.shape).clone(), True


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
    input_uncertain_acq: int = 0,
    input_uncertain_scale: float = 0.0,
    min_distance: float = 0.0,
    logged_X: np.ndarray | None = None,
    noise_sd: float | None = None,
    diagnostics: dict | None = None,
) -> torch.Tensor:
    # The keyword extras are the acquisition-side follow-ups. At their defaults
    # this consumes the same randomness and returns the same candidate as before
    # they existed. input_uncertain_scale is the EFFECTIVE scale (see
    # input_uncertain_effective_scale); noise_sd overrides aei's model-derived
    # noise; diagnostics, when given, receives "min_distance_redirect".
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
    elif acq_config.name == "qkg":
        # num_fantasies is the cost knob: the acquisition optimises an inner
        # problem per fantasy, so 8 keeps a 50-iteration run affordable at the
        # price of a coarser estimate.
        acqf = qKnowledgeGradient(model=gp_model, num_fantasies=8)
    elif acq_config.name == "replei":
        if best_f is None:
            raise ValueError("best_f required for replei.")
        acqf = LogExpectedImprovement(model=gp_model, best_f=best_f + acq_config.xi)
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
        # qNEHVI builds its own partitioning from X_baseline; it does not
        # accept a precomputed `partitioning` argument. Passing one raised
        # TypeError on EVERY call, so the qnehvi arm silently ran as the
        # random fallback in all pre-fix sweeps (caught via the
        # acq_opt_failed audit). prune_baseline=True is the
        # botorch-recommended setting and bounds memory as X_baseline grows.
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qNoisyExpectedHypervolumeImprovement(
            model=gp_model,
            ref_point=ref_point.tolist(),
            X_baseline=train_X,
            sampler=sampler,
            prune_baseline=True,
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
        # accept a precomputed `partitioning` argument. prune_baseline=True
        # mirrors the qnehvi branch: it is the botorch-recommended setting, it
        # bounds memory as X_baseline grows, and — critically — it must match
        # qnehvi's setting or the two arms are not comparable (the acquisitions
        # would differ in baseline handling, not just in log-space smoothing).
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=gp_model,
            ref_point=ref_point.tolist(),
            X_baseline=train_X,
            sampler=sampler,
            prune_baseline=True,
        )
    elif acq_config.name == "ts":
        acqf = build_thompson_sampling(gp_model, rng)
    elif acq_config.name == "aei":
        acqf = build_augmented_ei(gp_model, train_X, noise_sd=noise_sd)
    elif acq_config.name == "shiplcb":
        acqf = build_ship_rule_ei(gp_model, train_X, noise_sd=noise_sd)
    else:
        raise ValueError(f"Unknown acquisition: {acq_config.name}")

    if isinstance(acqf, OneShotAcquisitionFunction):
        if input_uncertain_acq > 0 or min_distance > 0:
            raise ValueError(
                f"{acq_config.name} is a one-shot acquisition with no screening pool; it "
                "supports neither the input-uncertain wrapper nor the min-distance redirect."
            )
        # One-shot acquisitions (the knowledge gradient) are optimised jointly
        # over the candidate AND its fantasy points, so they expect a q-batch of
        # q + num_fantasies rows. The candidate screen evaluates single points
        # and would hand them a q-batch of 1, which they reject outright --
        # which, because the acquisition call sits inside the fallback handler,
        # would quietly degrade every iteration to random sampling. optimize_acqf
        # knows how to augment the batch, so let it generate its own restarts.
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds_tensor,
            q=1,
            num_restarts=int(num_restarts),
            raw_samples=int(raw_samples),
            options={"batch_limit": 5, "maxiter": int(maxiter)},
        )
        if candidate.numel() == 0:
            raise RuntimeError(f"{acq_config.name}: optimize_acqf returned no candidate.")
        return candidate.detach()

    if input_uncertain_acq > 0 and input_uncertain_scale > 0:
        # Wrapped before screening, so the pool, the restarts and the optimiser
        # all see the averaged surface.
        acqf = InputUncertainAcquisition(
            base=acqf,
            perturbations=draw_input_perturbations(rng, bounds, input_uncertain_acq, input_uncertain_scale),
            bounds_tensor=bounds_tensor,
            log_space=acq_config.name in LOG_VALUED_ACQUISITIONS,
        )

    screened = screen_candidate_pool(
        acqf=acqf,
        bounds=bounds,
        rng=rng,
        candidate_pool=candidate_pool,
        num_restarts=num_restarts,
        return_pool=min_distance > 0,
    )
    fallback_candidate, batch_initial_conditions = screened[0], screened[1]

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_tensor,
        q=1,
        num_restarts=int(batch_initial_conditions.shape[0]),
        raw_samples=int(raw_samples),
        options={"batch_limit": 5, "maxiter": int(maxiter)},
        batch_initial_conditions=batch_initial_conditions,
    )
    chosen = fallback_candidate if candidate.numel() == 0 else candidate.detach()
    if min_distance > 0:
        if logged_X is None:
            raise ValueError("min_distance needs logged_X, the designs logged so far.")
        chosen, redirected = _min_distance_redirect(
            chosen, screened[2], screened[3], logged_X, bounds, min_distance
        )
        if diagnostics is not None:
            diagnostics["min_distance_redirect"] = redirected
    return chosen


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
    noise_variances: list[float] = []
    # The confidence report gets a stream of its own. Drawing it from the run or
    # jitter rng would shift every later error and lose common random numbers
    # with the standard sweep, which is the whole basis of the pairing.
    confidence_rng = np.random.default_rng(
        np.random.SeedSequence([int(config.seed), 0xC0FFEE, len(str(config.error_model))]))
    fit_times: list[float] = []
    acq_failures: list[bool] = []
    input_error_l2s: list[float] = []
    input_error_flags: list[bool] = []
    # The objective at the design that was WRITTEN DOWN, which is the one that
    # would actually be deployed. Identical to y_true unless an unnoticed input
    # slip moved the evaluation somewhere else.
    y_deployed_list: list[np.ndarray] = []

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
    if acq.name == "replei" and is_multi:
        raise ValueError("replei ranks a scalar incumbent and has no multi-objective form.")
    if is_multi and (config.replicate_first or config.final_rerate_top or config.input_noise_model != "none"):
        raise NotImplementedError("The process adaptations are single-objective only.")
    if config.input_noise_model != "none" and config.observation_noise in ("known", "self_report"):
        raise ValueError("input_noise_model and observation_noise='known' both set train_Yvar; pick one.")
    if config.likelihood == "student_t" and (
        is_multi or config.observation_noise in ("known", "self_report") or config.input_noise_model != "none"
    ):
        raise ValueError("likelihood='student_t' is single-objective with learned noise and no input-noise model.")
    if config.final_rerate_top and config.final_rerate_top * config.final_rerate_reps >= config.iterations - config.initial_samples:
        raise ValueError("final_rerate_top x final_rerate_reps must leave at least one model-based trial.")
    validate_acquisition_extensions(
        acq.name,
        is_multi=is_multi,
        input_error_model=config.input_error_model,
        input_uncertain_acq=config.input_uncertain_acq,
        input_uncertain_scale=config.input_uncertain_scale,
        min_distance=config.min_distance,
    )
    validate_error_extensions(acq.name, vars(config), is_multi=is_multi, noisy=apply_error)
    input_uncertain_scale = input_uncertain_effective_scale(config, apply_error)
    # --- error-process extension state. All of it is inert at the defaults.
    missing_on = apply_error and config.input_error_model in MISSING_INPUT_ERROR_CHOICES
    low_threshold = (
        landscape_quantile(oracle, bounds, MISSING_LOW_QUANTILE)
        if missing_on and config.input_error_model == "missing_low"
        else None
    )
    missing_flags: list[bool] = []
    imputed_values: list[float] = []
    rater_kind, rater_count = parse_rater_assign(config.rater_assign)
    # Offsets exist only where they act: noisy runs. The clean twin still knows
    # who rated what, which is all a backfit needs.
    rater_offsets = (
        draw_rater_offsets(
            config.seed, config.rater_assign, config.rater_offset_ratio, config.jitter_std, config.iterations
        )
        if apply_error and rater_kind != "none"
        else None
    )
    rater_ids: list[int] = []
    rater_estimates: dict[int, float] = {}
    ceiling_on = apply_error and config.response_ceiling > 0
    ceiling_base = landscape_quantile(oracle, bounds, config.response_ceiling) if ceiling_on else None
    ceiling_values: list[float] = []
    min_distance_redirects: list[bool] = []
    # Halo backfit state: the latest fit's correction (trials it saw x objectives)
    # and the rho_hat each trial's fit estimated (NaN where none ran).
    halo_corrections: np.ndarray | None = None
    halo_rho_hats: list[float] = []
    # Replication state: the design due a second rating, how many have had one,
    # and the end-of-run re-rating schedule once its window opens.
    replication_pending: np.ndarray | None = None
    # Arm 5. The anchor designs are drawn from a stream of their own so the run
    # rng is untouched and the arm keeps common random numbers with the sweep.
    anchor_designs: np.ndarray | None = None
    anchor_rows: list[int] = []
    anchor_ids: list[int] = []
    if config.anchor_every > 0:
        anchor_rng = np.random.default_rng(
            np.random.SeedSequence([int(config.seed), 0xA9C40, int(config.anchor_set)]))
        anchor_designs = sample_uniform(bounds, anchor_rng, size=int(config.anchor_set))
    # Arm 10. Designs proposed early and rated late; the queue is FIFO, so the
    # order within the held set is preserved and only its position moves.
    held_designs: list[np.ndarray] = []
    held_count = 0
    hold_release = int(round(config.hold_until_frac * config.iterations))
    anchor_rng_hold = np.random.default_rng(
        np.random.SeedSequence([int(config.seed), 0x401D, int(config.hold_early)]))
    replicated = 0
    rerate_schedule: list[np.ndarray] = []
    rerate_window = config.final_rerate_top * config.final_rerate_reps
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
        redirected = False
        halo_rho = np.nan
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
        elif rerate_window and iteration > config.iterations - rerate_window:
            # Re-evaluate before deploying: the last trials go to the designs
            # that currently look best, so the final pick rests on more than
            # one rating each.
            if not rerate_schedule:
                rerate_schedule = _rerate_schedule(
                    X_list, objective_observed_scalar, config.final_rerate_top, config.final_rerate_reps
                )
            candidate_np = rerate_schedule.pop(0)
            fit_time = 0.0
        elif (config.anchor_every > 0 and anchor_designs is not None
              and iteration > config.initial_samples
              and (iteration - config.initial_samples) % config.anchor_every == 0):
            # An anchor trial. It buys no new design, which is the price; what it
            # buys is a reading of the rater that the search cannot confound.
            which = (len(anchor_rows)) % int(config.anchor_set)
            candidate_np = np.array(anchor_designs[which], dtype=float)
            anchor_rows.append(len(X_list))
            anchor_ids.append(which)
            fit_time = 0.0
        elif held_designs and iteration >= hold_release:
            # A design proposed early, judged now. Same design, later moment.
            candidate_np = held_designs.pop(0)
            fit_time = 0.0
        elif replication_pending is not None:
            # The second rating of a design proposed one trial ago.
            candidate_np = replication_pending
            replication_pending = None
            fit_time = 0.0
        elif acq.name == "replei" and (iteration - config.initial_samples) % 2 == 0:
            # Re-ask about the design that currently looks best. The oracle
            # returns the same true value and the error model draws fresh noise,
            # which is exactly what replicating a rating does: no new
            # information about the landscape, one more sample of the rater.
            best_index = int(np.argmax(np.asarray(objective_observed_scalar, dtype=float)))
            candidate_np = np.array(X_list[best_index], dtype=float)
            fit_time = 0.0
        elif missing_on and len(_missing_training_rows(objective_observed_scalar, imputed_values)[0]) < 2:
            # Lost ratings can leave too few to fit a surrogate on; the initial
            # design is extended the way it was drawn, which is not a failure.
            candidate_np = sample_uniform(bounds, rng, size=1)[0]
            fit_time = 0.0
        else:
            fit_start = time.perf_counter()

            # GP construction/fit lives INSIDE the fallback try: a fit failure
            # (degenerate data, Cholesky errors) must degrade to the recorded
            # random fallback like any acquisition failure, not crash a
            # multi-hour sweep (review finding).
            try:
                train_X = torch.tensor(np.vstack(X_list), dtype=torch.double)
                if config.anchor_model == "detrend" and config.anchor_every > 0:
                    # What the anchors say the rater has drifted by, per trial.
                    # Subtracted from every rating, the anchors' own included, so
                    # the surrogate sees one scale for the whole session.
                    anchor_offset = anchor_detrend(
                        anchor_rows, anchor_ids, objective_observed_scalar, len(X_list))
                else:
                    anchor_offset = None
                if is_multi:
                    train_Y_array = np.vstack(y_observed_list)
                    train_Y_list = [
                        torch.tensor(train_Y_array[:, idx].reshape(-1, 1), dtype=torch.double)
                        for idx in range(train_Y_array.shape[1])
                    ]
                    gp, mll = _fit_model_list_gp(train_X, train_Y_list, bounds_tensor)
                    if config.mo_halo_model == "backfit":
                        # The corrected ratings are the targets from here on, so the
                        # refit model and the hypervolume partitioning both read them.
                        gp, train_Y_list, halo_corrections, halo_rho = fit_mo_halo_backfit(
                            gp,
                            train_X,
                            train_Y_list,
                            lambda X, Ys: _fit_model_list_gp(X, Ys, bounds_tensor),
                        )
                    best_f = None
                    train_Y_for_acq: list[torch.Tensor] | torch.Tensor = train_Y_list
                else:
                    y_for_fit = np.array(y_observed_list, dtype=float).reshape(-1)
                    if anchor_offset is not None:
                        # The anchors' reading of the rater, removed. Applied to the
                        # ratings the surrogate trains on AND, through best_f and the
                        # deployment rule below, to what the run ships: correcting one
                        # and not the other would measure nothing.
                        y_for_fit = y_for_fit - anchor_offset[: len(y_for_fit)]
                    train_Y = torch.tensor(
                        y_for_fit.reshape(-1, 1), dtype=torch.double
                    )
                    train_Yvar = None
                    if config.observation_noise in ("known", "self_report"):
                        assert len(noise_variances) == train_Y.shape[0], (
                            "noise schedule out of step with the training targets"
                        )
                        train_Yvar = torch.tensor(
                            np.asarray(noise_variances, dtype=float).reshape(-1, 1),
                            dtype=torch.double,
                        )
                    train_rows = None
                    if missing_on:
                        # A lost rating trains nothing under "drop"; under
                        # "impute_low" its imputed value stands in for it.
                        train_rows, targets = _missing_training_rows(objective_observed_scalar, imputed_values)
                        row_index = torch.as_tensor(train_rows, dtype=torch.long)
                        train_X = train_X[row_index]
                        train_Y = torch.tensor(np.asarray(targets, dtype=float).reshape(-1, 1), dtype=torch.double)
                        if train_Yvar is not None:
                            train_Yvar = train_Yvar[row_index]
                    if config.likelihood == "student_t":
                        scripts_dir = str(Path(__file__).resolve().parent)
                        if scripts_dir not in sys.path:
                            sys.path.insert(0, scripts_dir)
                        from robust_gp import build_robust_gp

                        gp = build_robust_gp(train_X, train_Y, bounds=bounds_tensor)
                        mll = None
                    elif config.likelihood == "relevance_pursuit":
                        gp, mll = fit_relevance_pursuit_gp(train_X, train_Y, bounds_tensor)
                    elif config.rater_model == "backfit":
                        fit_ids = rater_ids if train_rows is None else [rater_ids[i] for i in train_rows]
                        gp, mll, rater_estimates = fit_backfit_gp(
                            train_X,
                            train_Y,
                            fit_ids,
                            lambda X, Y: _fit_gaussian_gp(X, Y, train_Yvar, bounds_tensor),
                        )
                        # The corrected ratings are the targets from here on, so
                        # observed_max and the acquisitions read them too.
                        train_Y = train_Y - torch.tensor(
                            [rater_estimates[r] for r in fit_ids], dtype=torch.double
                        ).reshape(-1, 1)
                    else:
                        gp = SingleTaskGP(
                            train_X,
                            train_Y,
                            train_Yvar=train_Yvar,
                            input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
                            outcome_transform=Standardize(m=1),
                        )
                        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                        fit_gpytorch_mll(mll)
                    if config.input_noise_model == "nigp":
                        gp, mll = _refit_with_input_noise(gp, train_X, train_Y, bounds, config)
                    if config.incumbent == "posterior_mean":
                        # Max posterior mean over visited points: robust to noisy
                        # observations (a single positive noise spike cannot
                        # inflate best_f beyond achievable values).
                        with torch.no_grad():
                            posterior_mean = gp.posterior(train_X).mean.reshape(-1)
                        best_f = posterior_mean.max().item()
                    elif config.incumbent == "lcb":
                        # Best mean - 1 latent SD over the visited designs.
                        best_f = _posterior_mean_and_lcb(gp, train_X)[1].max().item()
                    else:
                        best_f = train_Y.max().item()
                    train_Y_for_acq = train_Y

                aei_noise_sd = None
                if acq.name == "aei" and config.observation_noise in ("known", "self_report"):
                    # The model is told each rating's variance, so the discount
                    # uses the variance the rating about to be taken will carry.
                    aei_noise_sd = float(np.sqrt(known_noise_variance(iteration, config, apply_error)))
                candidate_info: dict = {}
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
                    input_uncertain_acq=config.input_uncertain_acq,
                    input_uncertain_scale=input_uncertain_scale,
                    min_distance=config.min_distance,
                    logged_X=np.vstack(X_list) if config.min_distance > 0 else None,
                    noise_sd=aei_noise_sd,
                    diagnostics=candidate_info,
                )
                candidate_np = candidate_tensor.cpu().numpy().flatten()
                redirected = bool(candidate_info.get("min_distance_redirect", False))
            except Exception as e:
                # Recorded per-iteration (acq_opt_failed column) so downstream
                # analysis can detect contaminated runs; never silently absorbed.
                acq_failed = True
                print(
                    f"[acq-fallback] run={run_id} acq={acq.name} iteration={iteration}: "
                    f"GP fit or acquisition optimization failed, falling back to random. Error: {e}",
                    file=sys.stderr,
                )
                candidate_np = sample_uniform(bounds, rng, size=1)[0]

            fit_time = time.perf_counter() - fit_start
            if config.replicate_first and replicated < config.replicate_first and not acq_failed:
                replication_pending = np.array(candidate_np, dtype=float)
                replicated += 1
            if (config.hold_early and held_count < config.hold_early
                    and iteration < hold_release and not acq_failed):
                # Hold it back rather than rate it now. The trial still has to go
                # somewhere, so the next proposal takes its place; the design is
                # not duplicated, it is moved.
                held_designs.append(np.array(candidate_np, dtype=float))
                held_count += 1
                candidate_np = sample_uniform(bounds, anchor_rng_hold, size=1)[0]

        # Where the person actually went. The objective is evaluated THERE --
        # that is what really happened and what the run really achieved -- while
        # what gets written down is governed by input_error_recorded. With the
        # default "proposed" the surrogate trains on (x, f(x')), which is the
        # honest model of an unnoticed slip.
        if apply_error and config.input_error_model != "none":
            if jitter_rng is None:
                raise ValueError("jitter_rng must be provided when apply_error is True.")
            if is_multi:
                raise NotImplementedError(
                    "Input error is single-objective only. The deployed-versus-"
                    "evaluated distinction is defined per point, and carrying it "
                    "through a Pareto set means deciding what the non-dominated "
                    "set of MISRECORDED points means -- a question this arm does "
                    "not answer."
                )
            actual_np, slipped = apply_input_error(
                candidate=candidate_np,
                iteration=iteration,
                config=config,
                rng=jitter_rng,
                bounds=bounds,
            )
        else:
            actual_np, slipped = candidate_np, False
        recorded_np = actual_np if config.input_error_recorded == "actual" else candidate_np
        input_error_l2 = float(np.linalg.norm(actual_np - candidate_np))

        true_value = oracle.predict(actual_np)
        # What deploying the recorded design would actually get you. Only a
        # second oracle call when a slip has separated the two.
        deployed_value = (
            true_value if input_error_l2 == 0.0 else oracle.predict(recorded_np)
        )
        y_deployed_list.append(deployed_value)

        if previous_observed is None:
            previous_observed = true_value

        rater = rater_index(iteration, rater_kind, rater_count) if rater_kind != "none" else 0
        # Decided before the rating is drawn, in the input channel's place in the stream.
        missing = (
            draw_missing_rating(true_value, iteration, config, jitter_rng, low_threshold)
            if missing_on
            else False
        )

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
                rater_offset=None if rater_offsets is None else float(rater_offsets[rater]),
            )
        else:
            observed_value, error_magnitude = true_value, np.zeros_like(true_value, dtype=float)

        if ceiling_on:
            # On every trial of a noisy run, onset or not: the top of the scale is
            # there from the first rating. It consumes no randomness.
            ceiling = response_ceiling_value(
                ceiling_base, config.ceiling_mode, objective_observed_scalar, objective_true_scalar
            )
            observed_value = np.minimum(observed_value, ceiling)
            error_magnitude = observed_value - true_value
            ceiling_values.append(ceiling)

        # The error process runs whether or not its rating reaches the log, so
        # the stateful models (ar1, dropout) carry the value that was drawn.
        previous_observed = observed_value
        previous_error = error_magnitude
        if missing_on:
            imputed = (
                impute_low_value(objective_observed_scalar)
                if missing and config.missing_handling == "impute_low"
                else None
            )
            missing_flags.append(bool(missing))
            imputed_values.append(np.nan if imputed is None else imputed)
            if missing:
                observed_value = np.full_like(true_value, np.nan, dtype=float)
                error_magnitude = np.full_like(true_value, np.nan, dtype=float)

        X_list.append(recorded_np)
        input_error_l2s.append(input_error_l2)
        input_error_flags.append(bool(slipped))
        if config.observation_noise == "self_report":
            # The rater says how sure they were; the GP believes them, coarsely.
            noise_variances.append(
                self_reported_variance(error_magnitude, config, confidence_rng, apply_error))
        else:
            noise_variances.append(known_noise_variance(iteration, config, apply_error))
        y_true_list.append(true_value)
        y_observed_list.append(observed_value)
        error_magnitudes.append(error_magnitude)
        fit_times.append(fit_time)
        acq_failures.append(acq_failed)
        min_distance_redirects.append(redirected)
        rater_ids.append(rater)
        halo_rho_hats.append(halo_rho)

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
            obs_matrix = np.vstack(y_observed_list)
            if halo_corrections is not None:
                # Deploy the Pareto set of the corrected ratings, under the latest
                # estimates (zero for a trial no fit has seen yet).
                obs_matrix[: len(halo_corrections)] -= halo_corrections
            obs_t = torch.tensor(obs_matrix, dtype=torch.double)
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
            # (highest OBSERVED value so far), scored by its TRUE value. With
            # replication on, the pick is by mean over a design's ratings.
            ratings = objective_observed_scalar
            if config.rater_model == "backfit":
                # Deploy on the corrected ratings, under the latest offset estimates
                # (zero for a rater no fit has seen yet).
                ratings = [y - rater_estimates.get(r, 0.0) for y, r in zip(ratings, rater_ids)]
            if config.inference_rule == "best_mean":
                best_obs_idx = _best_mean_index(X_list, ratings)
            elif missing_on:
                # A lost or imputed rating is never the one deployed.
                best_obs_idx = _nan_argmax(ratings)
            else:
                best_obs_idx = int(np.argmax(ratings))
            # Scored by the DEPLOYED design, not the evaluated one. They differ
            # only under an unnoticed input slip, and there the difference is
            # the whole point: the experimenter deploys what the log says, and
            # the log says x while the rating came from x'. Scoring f(x') here
            # would credit them with a design they never wrote down and would
            # hide most of what a misclick costs.
            inference_value = float(y_deployed_list[best_obs_idx][0])
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
    if config.input_error_model != "none":
        # objective_true is the objective WHERE THE PERSON WENT; this is the
        # objective at the design that was logged. Under the default
        # "proposed" recording they come apart exactly when a slip happened.
        results["objective_true_deployed"] = [float(v[0]) for v in y_deployed_list]
        results["input_error_l2"] = input_error_l2s
        results["input_error_applied"] = input_error_flags
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
    if config.min_distance > 0:
        # Only when the redirect is on, so default logs keep their schema.
        results["min_distance_redirect"] = min_distance_redirects
    # The error-process extensions log only where they act, so default logs --
    # and the clean twin of a run they leave alone -- keep their schema.
    if apply_error and config.noise_schedule != "none":
        results["noise_effort"] = [noise_effort(t, config) for t in range(1, config.iterations + 1)]
    if missing_on:
        results["missing"] = missing_flags
        if config.missing_handling == "impute_low":
            results["imputed_value"] = imputed_values
    if rater_kind != "none" and (apply_error or config.rater_model != "none"):
        results["rater_id"] = rater_ids
    if ceiling_on:
        results["ceiling_value"] = ceiling_values
    if config.mo_halo_model != "none":
        # Only where the remedy runs, so default logs keep their schema. The
        # corrections are the final fit's (0 for a trial it did not see), so
        # objective_observed_<obj> - halo_correction_<obj> replays the last
        # deployed Pareto set.
        results["halo_rho_hat"] = halo_rho_hats
        final_corrections = np.zeros((config.iterations, len(config.objective_columns)))
        if halo_corrections is not None:
            final_corrections[: len(halo_corrections)] = halo_corrections
        for idx, column in enumerate(config.objective_columns):
            results[f"halo_correction_{_objective_output_name(column)}"] = final_corrections[:, idx]
    results["seed"] = config.seed
    results["run_id"] = run_id
    # Label a corrupted run by what corrupted it -- never "none", the baseline
    # marker every downstream stage keys on. See run_error_label().
    results["error_model"] = (
        run_error_label(config.error_model, config.input_error_model)
        if apply_error
        else "none"
    )
    results["jitter_std"] = config.jitter_std if apply_error else 0.0
    results["jitter_iteration"] = config.jitter_iteration
    results["oracle_model"] = oracle_model
    results["objective"] = config.objective
    results["observation_noise"] = config.observation_noise
    results["known_noise_var"] = noise_variances
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
            oracle_target=dataset.oracle_target,
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
            observation_noise=args.observation_noise,
            **adaptation_fields(args),
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

