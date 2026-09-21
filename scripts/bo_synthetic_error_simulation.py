"""BO under noisy feedback on KNOWN functions -- the BOBA benchmark suite.

This is the sibling of ``bo_sensor_error_simulation.py``. It runs the identical
simulator core (same GP, same acquisition optimisation, same error models, same
per-iteration logs, same summary schema) with one substitution: the human is no
longer a regression oracle fitted to archival ratings, but an analytic benchmark
function with a published optimum.

Why that substitution matters
-----------------------------
Every claim the data-driven arm of this project makes is conditional on the
fitted oracle being a faithful stand-in for a person. It is not a very good one:
cross-validated R^2 tops out around 0.55 on the eHMI data and is near zero on
ProVoice. Any measured cost of feedback error is therefore entangled with the
oracle's own mis-specification, and ``y_opt`` -- the reference the regret is
measured against -- is itself a random-search estimate that the optimiser can
legitimately beat, which is why regret is deliberately not clamped at zero there.

On the BOBA suite both problems disappear:

  * the objective is exact, so nothing about the "human" is approximate;
  * ``y_opt`` is a verified supremum, so regret is a real regret;
  * the landscape geometry is measurable in advance
    (``scripts/boba_benchmarks.py``), so "which landscapes are fragile to
    feedback error" becomes an answerable question rather than a single
    dataset's anecdote.

What is deliberately identical to the data-driven arm
-----------------------------------------------------
The per-iteration CSV schema and filename pattern, so
``scripts/evaluate_research_question.py`` and
``scripts/confirmatory_followup.py`` consume this output unmodified. The
``dataset`` column carries the benchmark name and ``oracle_model`` is ``exact``.

Objective scaling
-----------------
Every benchmark is standardised to zero mean and unit variance over its own box
(constants in ``boba_landscape_stats.json``). The transform is affine and
increasing, so the optimisation problem is unchanged, but it makes every error
magnitude mean the same thing across a suite whose raw outputs span eight orders
of magnitude: ``--jitter-stds 0.5`` is half a landscape standard deviation
everywhere. ``opt_z`` in the stats file records how many standard deviations the
optimum sits above a random design, which is the second natural currency for
"how big is this error" and is left as a covariate for the analysis rather than
being normalised away.

Example
-------
  python scripts/bo_synthetic_error_simulation.py \\
    --functions ackley,shekel,branin --acq logei,ucb,random \\
    --iterations 50 --num-seeds 5 \\
    --error-models gaussian,bias --jitter-stds 0.05,0.5 --jitter-iterations 0,20 \\
    --output-dir output-boba
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Imported first: it sets the BLAS/OMP thread limits that every worker needs
# before numpy and torch are loaded, and it owns torch's default dtype.
import bo_sensor_error_simulation as sim  # noqa: E402

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import boba_benchmarks as bb  # noqa: E402
import boba_multiobjective as mob  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent

# Only the single-objective acquisitions apply: the BOBA suite is scalar, so the
# hypervolume family has nothing to optimise. random/sobol stay in as the
# model-free floors -- without them a "most robust" acquisition can simply be one
# that never learns anything in either condition.
# What "all" expands to -- the twelve the main sweep ran, and no more. See the
# note on DEFAULT_ACQUISITION_CHOICES in the simulation module.
DEFAULT_ACQUISITIONS = sim.SINGLE_ACQUISITION_CHOICES + sim.BASELINE_ACQUISITION_CHOICES
# What a run may NAME: the same twelve plus the opt-in robust baselines and the
# opt-in single-objective follow-ups (ts, aei).
SYNTHETIC_ACQUISITION_CHOICES = (
    sim.SINGLE_ACQUISITION_CHOICES
    + sim.ROBUST_ACQUISITION_CHOICES
    + sim.BASELINE_ACQUISITION_CHOICES
    + sim.EXTENSION_ACQUISITION_CHOICES
)
# The hypervolume family plus the same model-free floors. The log variants are
# the numerically stable ones and are the default here for that reason.
MO_ACQUISITION_CHOICES = sim.MULTI_ACQUISITION_CHOICES + sim.BASELINE_ACQUISITION_CHOICES

ORACLE_TAG = "exact"
OBJECTIVE_NAME = "value"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--functions",
        type=str,
        default="all",
        help="Comma-separated benchmark names; 'all' for the whole BOBA suite minus the "
        "stochastic 'typing' arm; 'extensions' for the levy dimension ladder and the "
        "confound-breaking bump family. Available: " + ", ".join(bb.ALL_ORDER),
    )
    parser.add_argument(
        "--multi-objective",
        action="store_true",
        default=False,
        help="Run the multi-objective suite instead: BoTorch problems with a published "
        "maximum hypervolume, standardised per objective so the same error magnitudes "
        "apply. --functions then names problems from boba_multiobjective, and the "
        "acquisitions are the hypervolume family plus the model-free floors.",
    )
    parser.add_argument("--mo-stats-path", type=Path, default=mob.DEFAULT_MO_STATS_PATH)
    parser.add_argument("--boba-root", type=str, default=None,
                        help="BOBA checkout; only needed for the 'typing' benchmark.")
    parser.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)

    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--initial-samples", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=1000)

    parser.add_argument("--acq", type=str, default="all",
                        help="Comma-separated acquisitions, or 'all' for the standard suite. "
                             "Scalar: " + ", ".join(SYNTHETIC_ACQUISITION_CHOICES) + ". "
                             "With --multi-objective: " + ", ".join(MO_ACQUISITION_CHOICES) + ". "
                             "'all' expands to the standard suite only; the robust baselines "
                             "(qkg, replei) and the extensions (ts, aei) have to be named.")
    parser.add_argument("--acq-list", type=str, default=None,
                        help="Same names as --acq, but taken verbatim (no 'all' expansion); "
                             "overrides --acq when given.")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num-seeds", type=int, default=5)

    parser.add_argument("--error-models", type=str, default="gaussian,bias,drift,ar1")
    parser.add_argument("--jitter-stds", type=str, default="0.05,0.25,1.0",
                        help="Error magnitudes in units of the landscape standard deviation.")
    parser.add_argument("--jitter-iterations", type=str, default="0,20",
                        help="Onset iterations. 0 = noisy from the first observation.")
    parser.add_argument("--single-error", action="store_true", default=False)

    parser.add_argument(
        "--error-bias-mode",
        type=str,
        default="scaled",
        choices=["scaled", "fixed"],
        help="'scaled' (default) sets the bias model's systematic offset equal to the "
        "swept error magnitude, so 'bias' and 'gaussian' contrast a systematic and a "
        "random error of the SAME size. 'fixed' reproduces the data-driven arm, where "
        "the offset stays at --error-bias while the gaussian component sweeps -- which "
        "makes the bias term negligible at the large levels.",
    )
    parser.add_argument("--error-bias", type=float, default=0.2)
    parser.add_argument("--error-spike-prob", type=float, default=0.1)
    parser.add_argument(
        "--error-spike-std-mode", type=str, default="scaled", choices=["scaled", "fixed"],
        help="As --error-bias-mode, for the spike model's spike magnitude.",
    )
    parser.add_argument("--error-spike-std", type=float, default=0.5)
    parser.add_argument("--error-ar1-rho", type=float, default=0.8)
    parser.add_argument("--dropout-strategy", type=str, default="hold_last", choices=["hold_last"])

    parser.add_argument(
        "--response-clip",
        type=str,
        default="none",
        help="'none', 'sample' (clip to the standardised sampled range of the landscape, "
        "the analogue of a bounded rating scale), or explicit 'low,high'.",
    )
    parser.add_argument(
        "--response-round", type=float, default=None,
        help="Round the observed response to this step, in landscape standard "
        "deviations -- the discreteness of a rating instrument. The three archival "
        "studies imply 0.55, 0.54 and 1.12 sigma respectively, so 0.55 is the realistic "
        "setting; pair it with --response-clip to bound the scale as well.",
    )

    parser.add_argument(
        "--input-error",
        type=str,
        default="none",
        choices=sim.INPUT_ERROR_CHOICES,
        help="Corrupt the DESIGN rather than the rating: the optimizer proposes "
        "x, the person acts on x' != x and rates x' honestly. 'slip' adds "
        "gaussian positional error every trial (pointer/slider imprecision); "
        "'misclick' jumps to a uniformly random design with probability p (a "
        "wrong press); 'missing_mcar' loses the rating with probability p and "
        "'missing_low' only for designs below the landscape's 40th percentile "
        "(see --missing-handling). Magnitude comes from --input-error-scale, or "
        "from the swept --jitter-stds grid when --input-error-from-sweep is set. "
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
        choices=sim.INPUT_ERROR_RECORDED_CHOICES,
        help="Which design is written down. 'proposed' (default) is the real "
        "case -- an unnoticed slip means the surrogate trains on (x, f(x')). "
        "'actual' is the counterfactual where the slip is detected and logged, "
        "separating the cost of going to the wrong place from the cost of "
        "mislabelling it.",
    )

    # The process adaptations of docs/adaptations-proposal.md. Parsed by
    # sim.adaptation_fields so both drivers read them the same way.
    parser.add_argument("--replicate-first", type=int, default=0,
                        help="Rate each of the first N model-based proposals twice (0 = off).")
    parser.add_argument("--final-rerate", type=str, default="0,0",
                        help="TOP,REPS: the last TOP x REPS trials re-rate the TOP best designs "
                             "REPS times each and the deployed design is picked by mean rating.")
    parser.add_argument("--input-noise-model", type=str, default="none",
                        choices=sim.INPUT_NOISE_MODEL_CHOICES,
                        help="'nigp': noisy-input GP, variance inflated by the squared posterior-mean "
                             "gradient times the slip variance (uses the swept slip SD).")
    parser.add_argument("--likelihood", type=str, default="gaussian", choices=sim.LIKELIHOOD_CHOICES,
                        help="'student_t': outlier-robust Student-t variational GP surrogate. "
                             "'relevance_pursuit': BoTorch's robust GP with per-point outlier variances "
                             "(changes the clean run, so it needs its own --output-dir).")
    parser.add_argument("--inference-rule", type=str, default=None, choices=sim.INFERENCE_RULE_CHOICES,
                        help="Deployed-design rule; defaults to best_mean when any replication is on.")
    parser.add_argument("--incumbent", type=str, default="posterior_mean",
                        choices=sim.INCUMBENT_CHOICES,
                        help="best_f of the improvement acquisitions: 'posterior_mean' (default), "
                             "'observed_max', or 'lcb' (best posterior mean - 1 latent SD over the "
                             "visited designs). Changes the clean run; the baseline file is named for it.")
    # Acquisition-side follow-ups, all equal-trial. Parsed by sim.adaptation_fields.
    parser.add_argument("--input-uncertain-acq", type=int, default=0,
                        help="Average the acquisition over K fixed QMC normal perturbations of the "
                             "design (log-mean-exp for log-valued acquisitions); 0 = off.")
    parser.add_argument("--input-uncertain-scale", type=float, default=-1.0,
                        help="Perturbation SD as a fraction of each coordinate's range. -1 (default) "
                             "borrows the run's slip SD, so it needs --input-error slip and is off in "
                             "the clean run. An explicit value >= 0 applies to the clean run too, so "
                             "such a run needs its own --output-dir.")
    parser.add_argument("--min-distance", type=float, default=0.0,
                        help="Redirect a model-based proposal whose RMS distance (unit box) to a logged "
                             "design is below this to the best screened pool point at least this far "
                             "from every logged design; 0 = off. Changes the clean run, so it needs "
                             "its own --output-dir.")
    # Error-process extensions, all equal-trial. Parsed by sim.adaptation_fields.
    parser.add_argument("--noise-schedule", type=str, default="none",
                        help="Move the gaussian rating error between trials: SD jitter_std / sqrt(e_t) at "
                             "trial t, from the standard run's own draws. Presets front10, front20, U, back10 "
                             "(mean effort 1 at T = 50) or custom '1-10:2,11-:0.75'. Gaussian only; the clean "
                             "run is unchanged.")
    parser.add_argument("--missing-handling", type=str, default="drop", choices=sim.MISSING_HANDLING_CHOICES,
                        help="With --input-error missing_mcar/missing_low: 'drop' trains and deploys without the "
                             "lost rating; 'impute_low' trains on the 20th percentile of the ratings so far "
                             "(their minimum below five) and never deploys the imputed design.")
    parser.add_argument("--rater-assign", type=str, default="none",
                        help="Relay raters: 'block:K' hands over every K trials, 'roundrobin:R' cycles R raters. "
                             "Each rater's fixed offset is added to noisy ratings after the onset. Gaussian only; "
                             "the clean run is unchanged unless --rater-model backfit.")
    parser.add_argument("--rater-offset-ratio", type=float, default=0.0,
                        help="Rater offsets ~ N(0, (ratio x jitter_std)^2), from a stream of their own.")
    parser.add_argument("--rater-model", type=str, default="none", choices=sim.RATER_MODEL_CHOICES,
                        help="'backfit': estimate per-rater offsets inside the GP fit and deploy on the corrected "
                             "ratings. Changes the clean run, so it needs its own --output-dir.")
    parser.add_argument("--anchor-every", type=int, default=0,
                        help="Every N trials, rate one of a small fixed anchor set instead of the "
                             "proposal. The anchors have a constant true value, so movement in their "
                             "ratings is the rater drifting, separated from the search trend. "
                             "Costs trials and changes the clean run: needs its own --output-dir.")
    parser.add_argument("--anchor-set", type=int, default=3,
                        help="How many fixed anchor designs to cycle through.")
    parser.add_argument("--anchor-model", type=str, default="none", choices=sim.ANCHOR_MODEL_CHOICES,
                        help="'detrend': fit a line in the trial index to the anchors and subtract "
                             "it from every rating before the surrogate is fitted.")
    parser.add_argument("--hold-early", type=int, default=0,
                        help="Hold back the first N model-based proposals and rate them late instead, "
                             "to decouple 'informative design' from 'early in the session'. The "
                             "design is moved, not duplicated. Changes the clean run.")
    parser.add_argument("--hold-until", type=float, default=0.6,
                        help="Fraction of the budget after which held designs are released.")
    parser.add_argument("--confidence-noise", type=float, default=0.5,
                        help="With --observation-noise self_report: how coarse the rater's own "
                             "precision report is, as the SD of a log-normal multiplier on the "
                             "trial's true squared error. 0 prices a perfect report.")
    parser.add_argument("--anchor-rating", action="store_true",
                        help="Judge each proposal against the incumbent shown beside it: the error "
                             "shared by the pair cancels and the fresh part is differenced, so the "
                             "idiosyncratic noise grows by sqrt(2). The clean run is unchanged.")
    parser.add_argument("--response-ceiling", type=float, default=None,
                        help="Cap every noisy rating at this quantile q of the landscape (0 < q < 1). Single-"
                             "objective; the clean run is unchanged.")
    parser.add_argument("--ceiling-mode", type=str, default="fixed", choices=sim.CEILING_MODE_CHOICES,
                        help="'anchored' raises the cap to 0.5 above the true value of the best-rated design so "
                             "far, when that is higher.")
    # Multi-objective halo error, equal-trial. Parsed by sim.adaptation_fields.
    parser.add_argument("--error-cross-corr", type=float, default=0.0,
                        help="Multi-objective, gaussian only: one trial's rating errors share a factor across "
                             "objectives, e_j = sqrt(1 - rho) eps_j + sqrt(rho) z, with eps the standard draws and z "
                             "one extra N(0, std^2) draw after them (a halo rater). 0 = off; the clean run is "
                             "unchanged.")
    parser.add_argument("--mo-halo-model", type=str, default="none", choices=sim.MO_HALO_MODEL_CHOICES,
                        help="'backfit': estimate the shared factor from the ModelListGP's standardised residuals, "
                             "refit on the corrected ratings and deploy their Pareto set. Multi-objective, "
                             "hypervolume acquisitions only. Changes the clean run, so it needs its own --output-dir.")
    parser.add_argument(
        "--observation-noise", type=str, default="learned",
        choices=sim.OBSERVATION_NOISE_CHOICES,
        help="'learned' (default) leaves the GP to fit its own noise level under "
        "BoTorch's prior. 'known' hands it the true injected variance, which is the "
        "contrast that separates the information the error destroys from the "
        "surrogate's failure to notice it is there.",
    )
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=2.0)

    parser.add_argument("--acq-num-restarts", type=int, default=10)
    parser.add_argument("--acq-raw-samples", type=int, default=512)
    parser.add_argument("--acq-maxiter", type=int, default=200)
    parser.add_argument("--acq-mc-samples", type=int, default=256)

    parser.add_argument("--baseline-run", action="store_true", default=True)
    parser.add_argument("--no-baseline-run", action="store_false", dest="baseline_run")

    parser.add_argument("--output-dir", type=Path, default=Path("output-boba"))
    parser.add_argument(
        "--per-function-dirs",
        action="store_true",
        default=True,
        help="Write each benchmark's per-iteration logs into its own subdirectory so the "
        "evaluation can be run per function without loading the whole sweep into memory.",
    )
    parser.add_argument("--flat-output", action="store_false", dest="per_function_dirs")

    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the run count and the resolved design, then exit without simulating.",
    )
    return parser.parse_args(argv)


def _parse_names(raw: str, available: list[str], what: str) -> list[str]:
    if raw.strip() == "all":
        return list(available)
    names = [v.strip() for v in raw.split(",") if v.strip()]
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"Unknown {what}: {unknown}. Available: {available}")
    if not names:
        raise ValueError(f"No {what} selected.")
    return names


def _parse_floats(raw: str) -> list[float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Empty float list.")
    return values


def _parse_ints(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Empty int list.")
    return values


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds requested.")
        return seeds
    return [args.seed + i for i in range(args.num_seeds)]


def resolve_clip(raw: str, stats_entry: dict[str, float]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Response clipping in STANDARDISED units."""
    if raw.strip() in {"", "none"}:
        return None, None
    if raw.strip() == "sample":
        mean, std = stats_entry["mean"], stats_entry["std"]
        low = (stats_entry["min"] - mean) / std
        high = (max(stats_entry["max"], stats_entry["y_opt"]) - mean) / std
        return np.array([low], dtype=float), np.array([high], dtype=float)
    parts = [float(v) for v in raw.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise ValueError(f"--response-clip must be 'none', 'sample' or 'low,high'; got {raw!r}")
    return np.array([parts[0]], dtype=float), np.array([parts[1]], dtype=float)


# ---------------------------------------------------------------------------
# One (function, seed) unit of work
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Task:
    function: str
    seed: int


def _variant_suffix(args: argparse.Namespace, error_model: str, error_bias: float,
                    spike_std: float) -> str:
    """Disambiguate condition parameters that are not already in the filename.

    Without this, two conditions that differ only in the bias magnitude would
    write to the same path and the second would silently overwrite the first.
    """
    parts: list[str] = []
    if error_model == "bias":
        parts.append(f"bias{error_bias:g}")
    if error_model == "spike":
        parts.append(f"sp{args.error_spike_prob:g}-{spike_std:g}")
    if error_model == "ar1" and args.error_ar1_rho != 0.8:
        parts.append(f"rho{args.error_ar1_rho:g}")
    if args.single_error:
        parts.append("single")
    # Resume matches on filename alone, so any setting that changes the run but
    # not the name would silently reuse the wrong CSVs.
    if args.observation_noise != "learned":
        parts.append(f"noise-{args.observation_noise}")
    if args.incumbent != "posterior_mean":
        parts.append(f"inc-{args.incumbent}")
    # The input-error arm. Which design gets written down changes the run, and
    # an input magnitude that is not the swept one is not the std in the name.
    if args.input_error != "none":
        if args.input_error_recorded != "proposed":
            parts.append(f"rec-{args.input_error_recorded}")
        if not args.input_error_from_sweep:
            parts.append(f"ie{args.input_error_scale:g}")
    # The process adaptations: each changes the run and none is in the name.
    adapt = sim.adaptation_fields(args)
    if adapt["replicate_first"]:
        parts.append(f"rep{adapt['replicate_first']}")
    if adapt["final_rerate_top"]:
        parts.append(f"rerate{adapt['final_rerate_top']}x{adapt['final_rerate_reps']}")
    if adapt["input_noise_model"] != "none":
        parts.append(adapt["input_noise_model"])
    if adapt["likelihood"] != "gaussian":
        parts.append(adapt["likelihood"].replace("_", ""))
    # The acquisition-side follow-ups. ts/aei are in the acquisition field of the
    # name and lcb in inc-<name>; these two are not named anywhere else.
    if adapt["input_uncertain_acq"]:
        parts.append(f"iu{adapt['input_uncertain_acq']}-{adapt['input_uncertain_scale']:g}")
    if adapt["min_distance"]:
        parts.append(f"mind{adapt['min_distance']:g}")
    # The error-process extensions. A missing process is in the error label, and
    # relevance_pursuit in the likelihood part above; the rest are named here only.
    if adapt["noise_schedule"] != "none":
        parts.append(f"sched-{sim.noise_schedule_name(adapt['noise_schedule'])}")
    if args.input_error in sim.MISSING_INPUT_ERROR_CHOICES:
        # Named even for the default "drop", so the handling is never implicit.
        parts.append(f"miss-{adapt['missing_handling']}")
    if adapt["rater_assign"] != "none":
        parts.append(sim.rater_suffix(adapt["rater_assign"], adapt["rater_offset_ratio"]))
    if adapt["rater_model"] != "none":
        parts.append(f"raterfit-{adapt['rater_model']}")
    if adapt["response_ceiling"] > 0:
        parts.append(f"ceil{adapt['response_ceiling']:g}-{adapt['ceiling_mode']}")
    if adapt["anchor_rating"]:
        parts.append("anchored")
    if adapt["anchor_every"] > 0:
        parts.append(f"anch{adapt['anchor_every']}x{adapt['anchor_set']}-{adapt['anchor_model']}")
    if adapt["hold_early"] > 0:
        parts.append(f"hold{adapt['hold_early']}@{adapt['hold_until_frac']:g}")
    # The multi-objective halo error and its remedy, named nowhere else.
    if adapt["error_cross_corr"] > 0:
        parts.append(f"xc{adapt['error_cross_corr']:g}")
    if adapt["mo_halo_model"] != "none":
        parts.append(f"halo-{adapt['mo_halo_model']}")
    return ("_" + "_".join(parts)) if parts else ""


# Settings that change the CLEAN run without appearing in its filename. The
# baseline is written as *_baseline_exact.csv with no variant suffix (only
# --observation-noise and --incumbent name it), so under --resume a baseline
# left by one such setting would silently be reused for another. The marker
# ties an output directory to one setting.
CLEAN_RUN_MARKER = "clean_run_settings.json"


def _clean_run_settings(args: argparse.Namespace) -> dict:
    adapt = sim.adaptation_fields(args)
    settings: dict[str, object] = {}
    if adapt["min_distance"] > 0:
        settings["min_distance"] = adapt["min_distance"]
    if adapt["input_uncertain_acq"] > 0 and adapt["input_uncertain_scale"] > 0:
        settings["input_uncertain"] = [adapt["input_uncertain_acq"], adapt["input_uncertain_scale"]]
    # The two error-process remedies that change the clean run. Both are new, so
    # no existing output directory lacks the marker they bring. student_t also
    # changes it but predates the marker, and tracking it now would refuse to
    # resume its arm.
    if adapt["likelihood"] == "relevance_pursuit":
        settings["likelihood"] = adapt["likelihood"]
    if adapt["rater_model"] != "none":
        # The assignment decides who rated what in the clean run too.
        settings["rater_model"] = [adapt["rater_model"], adapt["rater_assign"]]
    # Both of these replace proposals, so the clean run is a different search.
    if adapt["anchor_every"] > 0:
        settings["anchors"] = [adapt["anchor_every"], adapt["anchor_set"], adapt["anchor_model"]]
    if adapt["hold_early"] > 0:
        settings["hold_early"] = [adapt["hold_early"], adapt["hold_until_frac"]]
    # The halo backfit refits the clean run's surrogate too; new, so tracked.
    # --error-cross-corr acts on noisy ratings only and is not.
    if adapt["mo_halo_model"] != "none":
        settings["mo_halo_model"] = adapt["mo_halo_model"]
    return settings


def _guard_clean_run_settings(output_dir: Path, args: argparse.Namespace) -> None:
    """Refuse to mix clean runs of different clean-run-changing settings in one directory.

    Only the acquisition-side follow-ups and the clean-run-changing error-process
    remedies are tracked, and a default run into a directory without a marker
    touches nothing.
    """
    marker = output_dir / CLEAN_RUN_MARKER
    settings = _clean_run_settings(args)
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded != settings:
            raise ValueError(
                f"{output_dir} holds clean runs made with {recorded or 'the default settings'}, "
                f"and this run's clean-run settings are {settings or 'the defaults'}. The baseline "
                "filename does not carry them, so use a separate --output-dir."
            )
        return
    if not settings:
        return
    if next(output_dir.rglob("*_baseline_*.csv"), None) is not None:
        raise ValueError(
            f"{output_dir} already holds clean baselines made without {settings}. These settings "
            "change the clean run but not its filename, so use a fresh --output-dir."
        )
    marker.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def run_task(
    task: Task,
    args: argparse.Namespace,
    stats_entry: dict[str, float],
    acquisitions: list[str],
    error_models: list[str],
    jitter_stds: list[float],
    jitter_iterations: list[int],
    output_dir: Path,
    progress_q: object | None = None,
) -> tuple[list[dict], list[dict]]:
    """All acquisitions x conditions for one benchmark and one seed."""
    if args.boba_root:
        bb.set_boba_root(args.boba_root)

    function = task.function
    seed = task.seed
    multi = bool(args.multi_objective)
    if multi:
        oracle = mob.SyntheticMultiOracle.from_stats(function, {function: stats_entry})
        # y_opt is the published maximum hypervolume pushed through the same
        # per-objective affine map -- exact, not a sampling estimate.
        y_opt = float(stats_entry["max_hv"])
        ref_point = np.asarray(stats_entry["ref_point"], dtype=float)
        bounds = sim.Bounds(low=oracle.bounds_low, high=oracle.bounds_high)
        param_columns = oracle.param_columns
        objective_columns = oracle.objective_columns
        objective_name = "multi_objective"
        clip_low = clip_high = None
        dim = oracle.spec.dim
    else:
        spec = bb.BENCHMARKS[function]
        oracle = bb.SyntheticOracle.from_stats(function, {function: stats_entry},
                                               objective_name=OBJECTIVE_NAME)
        y_opt = (stats_entry["y_opt"] - stats_entry["mean"]) / stats_entry["std"]
        ref_point = None
        bounds = sim.Bounds(low=spec.bounds_low, high=spec.bounds_high)
        param_columns = spec.param_columns
        objective_columns = [OBJECTIVE_NAME]
        objective_name = OBJECTIVE_NAME
        clip_low, clip_high = resolve_clip(args.response_clip, stats_entry)
        dim = spec.dim

    base_config = sim.SimulationConfig(
        iterations=args.iterations,
        jitter_iteration=0,
        jitter_std=0.0,
        single_error=args.single_error,
        initial_samples=args.initial_samples,
        candidate_pool=args.candidate_pool,
        objective=objective_name,
        objective_columns=objective_columns,
        param_columns=param_columns,
        seed=seed,
        error_model="none",
        error_bias=args.error_bias,
        error_spike_prob=args.error_spike_prob,
        error_spike_std=args.error_spike_std,
        dropout_strategy=args.dropout_strategy,
        normalize_objective=False,
        objective_weights=None,
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
        input_error_model=args.input_error,
        input_error_scale=args.input_error_scale,
        input_error_recorded=args.input_error_recorded,
        **sim.adaptation_fields(args),
    )

    summaries: list[dict] = []
    diagnostics: list[dict] = []

    def _tick() -> None:
        if progress_q is not None:
            try:
                progress_q.put(1)
            except Exception:
                pass

    def _finish(results: pd.DataFrame, acq_name: str, error_model: str,
                jitter_std: float, jitter_iteration: int, run_id: str,
                runtime: float, baseline: bool, error_bias: float, spike_std: float) -> None:
        summary = sim.summarize_adjustment(results, int(jitter_iteration), param_columns)
        row = summary.to_dict()
        row.update(
            dataset=function,
            objective=objective_name,
            acquisition=acq_name,
            oracle_model=ORACLE_TAG,
            error_model=str(results["error_model"].iloc[0]),
            jitter_std=float(results["jitter_std"].iloc[0]),
            jitter_iteration=int(jitter_iteration),
            iterations=int(args.iterations),
            seed=int(seed),
            run_id=run_id,
            baseline=baseline,
            xi=float(args.xi),
            kappa=float(args.kappa),
            runtime_sec=float(runtime),
            y_opt=float(y_opt),
            param_columns=",".join(param_columns),
            dim=int(dim),
            error_bias_used=float(error_bias),
            error_spike_std_used=float(spike_std),
        )
        summaries.append(row)
        # Regret must be non-negative when y_opt is a true supremum. A negative
        # value means the verified optimum was wrong for this box, which would
        # invalidate every regret on this function -- so it is recorded rather
        # than left to be noticed by eye.
        diagnostics.append(
            {
                "dataset": function,
                "acquisition": acq_name,
                "seed": int(seed),
                "error_model": str(results["error_model"].iloc[0]),
                "jitter_std": float(results["jitter_std"].iloc[0]),
                "jitter_iteration": int(jitter_iteration),
                "min_simple_regret_true": float(results["simple_regret_true"].min()),
                "max_objective_true": float(results["objective_true"].max()),
                "acq_opt_failures": int(results["acq_opt_failed"].sum()),
            }
        )

    for acq_name in acquisitions:
        acq = sim.AcquisitionConfig(name=acq_name, xi=args.xi, kappa=args.kappa)

        if args.baseline_run:
            arm = ""
            if args.observation_noise != "learned":
                arm += f"_noise-{args.observation_noise}"
            if args.incumbent != "posterior_mean":
                arm += f"_inc-{args.incumbent}"
            path = output_dir / (
                f"bo_sensor_error_{function}_{objective_name}_{acq_name}_seed{seed}"
                f"_baseline_{ORACLE_TAG}{arm}.csv"
            )
            baseline_results = sim._load_resumable_run(path, args.iterations) if args.resume else None
            if baseline_results is not None:
                run_id = str(baseline_results["run_id"].iloc[0])
                runtime = 0.0
            else:
                run_id = str(uuid.uuid4())
                rng = np.random.default_rng(seed)
                torch.manual_seed(seed)
                start = time.perf_counter()
                baseline_results = sim.run_simulation(
                    oracle=oracle,
                    bounds=bounds,
                    config=dataclasses.replace(base_config, seed=seed),
                    acq=acq,
                    rng=rng,
                    jitter_rng=None,
                    run_id=run_id,
                    apply_error=False,
                    oracle_model=ORACLE_TAG,
                    y_opt=y_opt,
                )
                baseline_results["dataset"] = function
                runtime = time.perf_counter() - start
                baseline_results.to_csv(path, index=False)

            for jitter_iteration in jitter_iterations:
                _finish(baseline_results, acq_name, "none", 0.0, jitter_iteration,
                        run_id, runtime, True, args.error_bias, args.error_spike_std)
            _tick()

        for error_model in error_models:
            for jitter_std in jitter_stds:
                error_bias = (
                    float(jitter_std) if args.error_bias_mode == "scaled" else args.error_bias
                )
                spike_std = (
                    float(jitter_std) if args.error_spike_std_mode == "scaled"
                    else args.error_spike_std
                )
                for jitter_iteration in jitter_iterations:
                    suffix = _variant_suffix(args, error_model, error_bias, spike_std)
                    # Name the file after the corruption actually applied. With
                    # --input-error-from-sweep the loop's error_model only carries
                    # the swept grid -- the response channel is switched off -- so
                    # a "gaussian" filename would misreport the run, and resume
                    # would treat a slip run as a gaussian one. With no input error
                    # this is exactly error_model, so existing filenames (and
                    # therefore resume of existing arms) are unchanged.
                    channel = sim.run_error_label(
                        "none" if args.input_error_from_sweep else error_model,
                        args.input_error,
                    )
                    path = output_dir / (
                        f"bo_sensor_error_{function}_{objective_name}_{acq_name}_seed{seed}"
                        f"_jittered_{ORACLE_TAG}_{channel}_jit{jitter_iteration}"
                        f"_std{jitter_std}{suffix}.csv"
                    )
                    results = sim._load_resumable_run(path, args.iterations) if args.resume else None
                    if results is not None:
                        run_id = str(results["run_id"].iloc[0])
                        runtime = 0.0
                    else:
                        run_id = str(uuid.uuid4())
                        rng = np.random.default_rng(seed)
                        torch.manual_seed(seed)
                        # Common random numbers: the jitter stream deliberately
                        # does NOT depend on the benchmark, so the same seed and
                        # condition inject the identical standardised error
                        # sequence on every landscape. Cross-function contrasts
                        # then differ by geometry, not by noise realisation.
                        jitter_rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    seed,
                                    sim.ACQUISITION_CHOICES.index(acq_name),
                                    int(jitter_iteration),
                                    int(round(float(jitter_std) * 1_000_000)),
                                    sim.ERROR_MODEL_CHOICES.index(error_model),
                                ]
                            )
                        )
                        config = dataclasses.replace(
                            base_config,
                            seed=seed,
                            error_model=error_model,
                            jitter_std=float(jitter_std),
                            jitter_iteration=int(jitter_iteration),
                            error_bias=error_bias,
                            error_spike_std=spike_std,
                        )
                        if args.input_error_from_sweep:
                            # The swept grid drives the INPUT error instead. The
                            # response error is switched off so the arm measures
                            # one corruption at a time: leaving both on at the
                            # same swept value would confound them at every cell.
                            config = dataclasses.replace(
                                config,
                                input_error_scale=float(jitter_std),
                                error_model="none",
                            )
                        start = time.perf_counter()
                        results = sim.run_simulation(
                            oracle=oracle,
                            bounds=bounds,
                            config=config,
                            acq=acq,
                            rng=rng,
                            jitter_rng=jitter_rng,
                            run_id=run_id,
                            apply_error=True,
                            oracle_model=ORACLE_TAG,
                            y_opt=y_opt,
                        )
                        results["dataset"] = function
                        runtime = time.perf_counter() - start
                        results.to_csv(path, index=False)

                    # Record the arm under the corruption it actually applied.
                    # With --input-error-from-sweep the response channel is off
                    # and the swept grid drives the input channel, so labelling
                    # the row "gaussian" would be a straight misreport.
                    # The same label the filename carries, so the summary row, the
                    # per-run CSV and the path cannot disagree about what ran.
                    recorded_error_model = channel
                    _finish(results, acq_name, recorded_error_model, float(jitter_std),
                            int(jitter_iteration), run_id, runtime, False,
                            error_bias, spike_std)
                    _tick()

    return summaries, diagnostics


def _run_task_star(payload: tuple) -> tuple[list[dict], list[dict]]:
    return run_task(*payload)


# ---------------------------------------------------------------------------
# Aggregation. Mirrors bo_sensor_error_simulation.main()'s excess-summary block
# so the two arms produce comparable secondary outputs; the primary analysis is
# recomputed from the per-iteration logs by evaluate_research_question.py either
# way.
# ---------------------------------------------------------------------------


def build_excess_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    jittered = summary_df[~summary_df["baseline"]]
    baseline = summary_df[summary_df["baseline"]]
    merged = jittered.merge(
        baseline,
        on=["dataset", "acquisition", "objective", "iterations", "jitter_iteration",
            "seed", "oracle_model", "xi", "kappa", "param_columns"],
        suffixes=("_jitter", "_baseline"),
    )
    rows: list[dict] = []
    for _, row in merged.iterrows():
        param_columns = row["param_columns"].split(",")
        entry: dict[str, object] = {
            "dataset": row["dataset"],
            "objective": row["objective"],
            "acquisition": row["acquisition"],
            "oracle_model": row["oracle_model"],
            "error_model": row["error_model_jitter"],
            "jitter_std": row["jitter_std_jitter"],
            "jitter_iteration": row["jitter_iteration"],
            "seed": row["seed"],
            "param_columns": row["param_columns"],
            "dim": row["dim_jitter"],
        }
        deltas = []
        for col in param_columns:
            value = row[f"delta_{col}_jitter"] - row[f"delta_{col}_baseline"]
            entry[f"delta_excess_{col}"] = value
            deltas.append(value)
        entry["delta_excess_l2_norm"] = float(np.linalg.norm(np.asarray(deltas, dtype=float)))
        for metric in ("final_simple_regret_true", "final_cum_regret_true",
                       "final_avg_regret_true", "auc_simple_regret_true",
                       "final_inference_simple_regret_true",
                       "auc_inference_simple_regret_true"):
            key = metric.replace("final_", "final_").replace("_true", "_excess_true")
            entry[key] = row[f"{metric}_jitter"] - row[f"{metric}_baseline"]
        rows.append(entry)
    return pd.DataFrame(rows)


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.boba_root:
        bb.set_boba_root(args.boba_root)

    if args.multi_objective:
        functions = _parse_names(args.functions, mob.MO_ORDER, "multi-objective problem")
    elif args.functions == "extensions":
        functions = list(bb.EXTENSION_ORDER)
    else:
        functions = _parse_names(
            args.functions,
            bb.DEFAULT_SUITE if args.functions == "all" else bb.ALL_ORDER,
            "benchmark",
        )
    allowed = MO_ACQUISITION_CHOICES if args.multi_objective else SYNTHETIC_ACQUISITION_CHOICES
    # "all" expands to the standard suite; the robust baselines have to be asked
    # for by name. Naming one is still valid -- that is what `allowed` is for.
    default_all = MO_ACQUISITION_CHOICES if args.multi_objective else DEFAULT_ACQUISITIONS
    if args.acq_list:
        acquisitions = [a.strip() for a in args.acq_list.split(",") if a.strip()]
    elif args.acq.strip() == "all":
        acquisitions = list(default_all)
    else:
        acquisitions = _parse_names(args.acq, allowed, "acquisition")
    unknown_acq = [a for a in acquisitions if a not in allowed]
    if unknown_acq:
        raise ValueError(
            f"Acquisition(s) {unknown_acq} do not apply here. "
            + ("The multi-objective suite takes the hypervolume family plus the "
               "model-free floors: " if args.multi_objective
               else "The BOBA suite is single-objective, so the hypervolume family does "
                    "not apply: ")
            + ", ".join(allowed)
        )

    # The acquisition-side follow-ups, checked up front so a bad combination
    # fails here rather than inside every worker.
    adapt = sim.adaptation_fields(args)
    for acq_name in acquisitions:
        sim.validate_acquisition_extensions(
            acq_name,
            is_multi=bool(args.multi_objective),
            input_error_model=args.input_error,
            input_uncertain_acq=adapt["input_uncertain_acq"],
            input_uncertain_scale=adapt["input_uncertain_scale"],
            min_distance=adapt["min_distance"],
        )

    error_models = _parse_names(args.error_models, sim.ERROR_MODEL_CHOICES, "error model")
    jitter_stds = _parse_floats(args.jitter_stds)
    jitter_iterations = _parse_ints(args.jitter_iterations)
    seeds = resolve_seeds(args)

    # The error-process extensions, likewise up front: once per response model
    # the sweep runs, and per swept rate when the sweep drives the input channel.
    rates = jitter_stds if args.input_error_from_sweep else [args.input_error_scale]
    for acq_name in acquisitions:
        for error_model in error_models:
            for rate in rates:
                sim.validate_error_extensions(
                    acq_name,
                    {
                        **adapt,
                        "iterations": args.iterations,
                        "error_model": "none" if args.input_error_from_sweep else error_model,
                        "input_error_model": args.input_error,
                        "input_error_scale": float(rate),
                        "input_error_recorded": args.input_error_recorded,
                        "observation_noise": args.observation_noise,
                    },
                    is_multi=bool(args.multi_objective),
                )

    for onset in jitter_iterations:
        if onset < 0 or onset >= args.iterations:
            raise ValueError(f"--jitter-iterations entry {onset} outside [0, {args.iterations - 1}].")
    if any(value < 0 for value in jitter_stds):
        raise ValueError("--jitter-stds must be non-negative.")
    if args.initial_samples < 2:
        raise ValueError("--initial-samples must be >= 2 for a GP to be fit.")
    if args.initial_samples >= args.iterations:
        raise ValueError("--initial-samples must be < --iterations.")

    if args.multi_objective:
        stats = mob.load_mo_stats(args.mo_stats_path)
        regenerate = (f"  python scripts/boba_multiobjective.py --output {args.mo_stats_path}")
    else:
        stats = bb.load_stats(args.stats_path)
        regenerate = (f"  python scripts/boba_benchmarks.py --functions <names> "
                      f"--output {args.stats_path}")
    missing = [f for f in functions if f not in stats]
    if missing:
        raise ValueError(f"No landscape statistics for {missing}. Regenerate with:\n{regenerate}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_per_task = len(acquisitions) * (
        (1 if args.baseline_run else 0)
        + len(error_models) * len(jitter_stds) * len(jitter_iterations)
    )
    # Seed-major, and within a seed the highest-dimensional (slowest) benchmarks
    # first. Two reasons, both about a sweep that runs for a day:
    #   * seed-major means the FIRST seed completes across every benchmark before
    #     the second one starts, so a partial sweep is already a complete
    #     cross-benchmark dataset the analysis can run on. Function-major would
    #     give twenty seeds of one landscape and nothing else.
    #   * longest-first within a wave keeps the slow tasks from landing in the
    #     final wave, where they would idle most of the pool.
    def _dim(name: str) -> int:
        return (mob.MO_BENCHMARKS[name].dim if args.multi_objective
                else bb.BENCHMARKS[name].dim)

    ordered_functions = sorted(functions, key=lambda f: -_dim(f))
    tasks = [Task(function=f, seed=s) for s in seeds for f in ordered_functions]
    total_runs = runs_per_task * len(tasks)

    print(f"Benchmarks      : {len(functions)}  ({', '.join(functions)})")
    print(f"Acquisitions    : {len(acquisitions)} ({', '.join(acquisitions)})")
    print(f"Error models    : {error_models}")
    print(f"Error magnitudes: {jitter_stds}  (landscape SD units)")
    print(f"Onsets          : {jitter_iterations}")
    print(f"Seeds           : {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"Iterations/run  : {args.iterations}")
    print(f"Tasks           : {len(tasks)} (function x seed), {runs_per_task} runs each")
    print(f"TOTAL RUNS      : {total_runs:,}")

    if args.dry_run:
        return
    _guard_clean_run_settings(output_dir, args)

    if args.n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif args.n_jobs == -2:
        n_jobs = max(1, mp.cpu_count() - 2)
    elif args.n_jobs > 0:
        n_jobs = min(args.n_jobs, mp.cpu_count())
    else:
        n_jobs = 1
    n_jobs = min(n_jobs, len(tasks))
    print(f"Workers         : {n_jobs}\n")

    runtime_start = time.perf_counter()
    summaries: list[dict] = []
    diagnostics: list[dict] = []
    failures: list[dict] = []

    def _task_dir(function: str) -> Path:
        path = output_dir / function if args.per_function_dirs else output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    progress = tqdm(total=total_runs, desc="Simulation runs", unit="run", smoothing=0.02)

    if n_jobs > 1:
        manager = mp.Manager()
        progress_q = manager.Queue()

        import threading

        def _monitor(q, pbar):
            while True:
                message = q.get()
                if message is None:
                    break
                try:
                    pbar.update(int(message))
                except Exception:
                    pass

        monitor = threading.Thread(target=_monitor, args=(progress_q, progress), daemon=True)
        monitor.start()
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(
                        _run_task_star,
                        (task, args, stats[task.function], acquisitions, error_models,
                         jitter_stds, jitter_iterations, _task_dir(task.function), progress_q),
                    ): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        task_summaries, task_diagnostics = future.result()
                        summaries.extend(task_summaries)
                        diagnostics.extend(task_diagnostics)
                    except Exception as exc:  # pragma: no cover - reported, not raised
                        import traceback

                        print(f"\nTask failed: {task.function} seed {task.seed}: {exc}",
                              file=sys.stderr)
                        traceback.print_exc()
                        failures.append({
                            "function": task.function,
                            "seed": task.seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        })
        finally:
            try:
                progress_q.put(None)
                monitor.join(timeout=10)
                manager.shutdown()
            except Exception:
                pass
    else:
        for task in tasks:
            try:
                task_summaries, task_diagnostics = run_task(
                    task, args, stats[task.function], acquisitions, error_models,
                    jitter_stds, jitter_iterations, _task_dir(task.function), None,
                )
                summaries.extend(task_summaries)
                diagnostics.extend(task_diagnostics)
                progress.update(runs_per_task)
            except Exception as exc:  # pragma: no cover
                import traceback

                print(f"\nTask failed: {task.function} seed {task.seed}: {exc}", file=sys.stderr)
                traceback.print_exc()
                failures.append({
                    "function": task.function,
                    "seed": task.seed,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
    progress.close()

    if not summaries:
        print("No runs completed.", file=sys.stderr)
        sys.exit(1)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "bo_synthetic_error_summary.csv", index=False)

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(output_dir / "bo_synthetic_error_diagnostics.csv", index=False)

    # y_opt is a verified supremum, so a negative simple regret is a real defect
    # rather than the expected behaviour it is in the fitted-oracle arm.
    tolerance = 1e-6
    violations = diagnostics_df[diagnostics_df["min_simple_regret_true"] < -tolerance]
    if len(violations):
        worst = violations.groupby("dataset")["min_simple_regret_true"].min()
        print(
            f"\nWARNING: {len(violations)} run(s) beat the verified optimum "
            f"(negative simple regret). Worst per benchmark:\n{worst.to_string()}",
            file=sys.stderr,
        )

    failed_acq = int(diagnostics_df["acq_opt_failures"].sum())
    if failed_acq:
        by_dataset = (
            diagnostics_df.groupby(["dataset", "acquisition"])["acq_opt_failures"].sum()
        )
        by_dataset = by_dataset[by_dataset > 0]
        print(
            f"\nNOTE: {failed_acq} acquisition-optimisation failure(s) fell back to random "
            f"sampling. Affected cells:\n{by_dataset.to_string()}",
            file=sys.stderr,
        )

    if args.baseline_run:
        excess = build_excess_summary(summary_df)
        excess.to_csv(output_dir / "bo_synthetic_error_excess_summary.csv", index=False)

    metadata = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "functions": functions,
        "acquisitions": acquisitions,
        "error_models": error_models,
        "jitter_stds": jitter_stds,
        "jitter_iterations": jitter_iterations,
        "seeds": seeds,
        "total_runs": int(total_runs),
        "completed_summary_rows": int(len(summary_df)),
        "runtime_sec": float(time.perf_counter() - runtime_start),
        "n_workers": int(n_jobs),
        "failures": failures,
        "git_commit": _git_commit(REPO_ROOT),
        "boba_root": args.boba_root,
        "boba_commit": _git_commit(Path(args.boba_root)) if args.boba_root else None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "landscape_stats_path": str(args.stats_path),
        "landscape_stats": {f: stats[f] for f in functions},
        "multi_objective": bool(args.multi_objective),
        "package_versions": sim.collect_package_versions(
            ["numpy", "pandas", "scipy", "scikit-learn", "botorch", "torch", "gpytorch", "tqdm"]
        ),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str),
                                                  encoding="utf-8")

    elapsed = time.perf_counter() - runtime_start
    print(f"\nComplete in {elapsed / 3600:.2f} h ({elapsed / max(total_runs, 1):.2f} s/run).")
    print(f"Results in {output_dir}")

    if failures:
        print(f"ERROR: {len(failures)} task(s) failed; outputs are incomplete.", file=sys.stderr)
        sys.exit(1)
    (output_dir / "SWEEP_COMPLETE").write_text(
        json.dumps({"total_runs": total_runs, "runtime_sec": elapsed}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        mp.set_start_method("spawn", force=True)
    main()
