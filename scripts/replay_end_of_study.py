"""End-of-study procedures that change only the last trials, replayed from the logs.

A standard study spends all T = 50 trials searching and ships the design with the
single best rating. The two procedures here keep T fixed and spend the final
trials differently. Everything before those trials is unchanged, so each is
replayed EXACTLY from an existing run log: truncate the log where the procedure
starts, refit the loop's GP on that prefix, and simulate only the new trials.
No trial is added. The premise -- a changed ending leaves the prefix identical --
is checked against the re-rating arm, whose first T - 6 trials must equal the
standard run's (column prefix_identical).

(A) Tournament(k), k in {3, 5}
    Search trials 1..T-k. Candidates: the top k visited designs by posterior mean
    minus one latent SD (the ship rule) of the loop's GP refitted on that prefix
    (BoTorch default SingleTaskGP, Normalize with the benchmark box, Standardize),
    or, as a variant, the top k by observed rating. The last k trials are ONE
    comparative sitting, each candidate shown once:
        look_i = f(x_i) + e_i,    e_i ~ N(0, (rho * s)^2),    rho in {1, 0.5}
    The winner is the candidate with the highest posterior mean once the GP is
    conditioned on the looks as extra observations (hyperparameters held fixed),
    or, as a variant, the best look.
(B) Confirmation(k), k in {2, 4}
    Search trials 1..T-2k. D = the ship-rule choice at that point; C = the best of
    the initial designs by observed rating (what a study would compare against).
    The last 2k trials rate D and C k times each in counterbalanced order. Ship D
    if its mean rating is at least C's, else revert to C. The improvement claim is
    a one-sided paired t-test p < 0.05 that D beats C; it is false if f(D) <= f(C).
    The standard process claims an improvement when its best search rating exceeds
    C's rating, falsely when the design it ships is truly no better than C.

The rating noise s of the new trials
------------------------------------
Error shared by every trial of one sitting cancels when designs are compared, so
only the idiosyncratic part enters:
    gaussian, bias, drift   s = sigma                     (bias and the slow ramp are shared)
    ar1                     s = sigma * sqrt(1 - rho_ar^2) (the innovation; the state is shared)
    clean twin              s = 0
    slip, misclick          s = --slip-look-sd (0.25) in the noisy AND the clean twin:
                            the system renders the RECORDED design, so no slip
                            enters, and the rating noise belongs to the procedure,
                            not to the error under study.
Draws come from a generator seeded by the run file's name and the procedure, so a
replay is reproducible, independent of --workers, and the variants of one
procedure (candidate rule, rho, winner rule) share their draws.

Scores and summaries
--------------------
Per run: the deployed regret at T (inference_simple_regret_true) of each procedure,
of the standard process, and, where the arm exists, of the re-rating arm (top three
re-rated twice in the last six trials), each on the noisy run and its clean twin;
the claim and false-claim indicators. Recovery uses the estimand of
analyse_boba_adaptations.py -- cost = ref_noisy - ref_clean, gain = ref_noisy -
trt_noisy, price = trt_clean - ref_clean, in units of opt_z, landscape bootstrap by
its own summarise() -- by arm x procedure x error model x magnitude x onset.

    python scripts/replay_end_of_study.py --workers 5
    python scripts/replay_end_of_study.py --summary-only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import warnings
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import analyse_boba_adaptations as aba  # noqa: E402
import analyse_extra_runs as aer  # noqa: E402
import bo_sensor_error_simulation as sim  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.transforms.input import Normalize  # noqa: E402
from botorch.models.transforms.outcome import Standardize  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

RESPONSE_MODELS = ("gaussian", "bias", "drift", "ar1")
INPUT_MODELS = ("slip", "misclick")
DEFAULT_ARMS = "output-boba,output-boba-slip,output-boba-misclick"
# The re-rating arm that modifies each standard arm, for context at matched cells.
RERATE_DIRS = {"output-boba": "output-boba-adapt-rerate", "output-boba-slip": "output-boba-adapt-rerate-slip"}
PROC_TOURNAMENT, PROC_CONFIRM = 1, 2
TREATMENT_FAMILIES = ("tournament", "confirmation", "rerate")
META = ("family", "k", "candidates", "rho", "winner")
LOGGED_ATOL = 1e-9


class ReproductionError(RuntimeError):
    """The log does not reproduce its own recorded deployed regret."""


@dataclasses.dataclass(frozen=True)
class Settings:
    tournament_k: tuple[int, ...] = (3, 5)
    confirm_k: tuple[int, ...] = (2, 4)
    rhos: tuple[float, ...] = (1.0, 0.5)
    slip_look_sd: float = 0.25
    alpha: float = 0.05
    lcb_beta: float = 1.0


@dataclasses.dataclass(frozen=True)
class Filters:
    functions: frozenset[str] | None = None
    acquisitions: frozenset[str] | None = None
    seeds: frozenset[int] | None = None
    error_models: frozenset[str] | None = None
    stds: tuple[float, ...] | None = None
    onsets: frozenset[int] | None = None


@dataclasses.dataclass(frozen=True)
class ArmInfo:
    name: str
    root: Path
    iterations: int
    initial_samples: int
    ar1_rho: float
    input_arm: bool
    rerate_dir: Path | None = None
    rerate_suffix: str = ""
    rerate_window: int = 0


@dataclasses.dataclass
class RunLog:
    name: str
    X: np.ndarray            # recorded designs, (T, d)
    observed: np.ndarray     # the ratings the loop trained on
    deployed: np.ndarray     # f at the RECORDED design (differs from objective_true only after a slip)
    logged_inference: np.ndarray
    y_opt: float


@dataclasses.dataclass
class SearchState:
    """What the experimenter knows after the first n trials."""
    n: int
    first: np.ndarray        # row of each distinct design's first rating, in order of appearance
    obs_mean: np.ndarray     # mean rating per distinct design
    lcb: np.ndarray          # posterior mean - beta * latent SD per distinct design
    gp: object | None
    train_X: torch.Tensor
    train_Y: torch.Tensor
    bounds: torch.Tensor
    gp_failed: bool


# ---------------------------------------------------------------------------
# Settings, arms, logs
# ---------------------------------------------------------------------------


def validate_settings(settings: Settings) -> None:
    if not settings.tournament_k and not settings.confirm_k:
        raise ValueError("nothing to replay: both --tournament-k and --confirm-k are empty")
    if any(k < 2 for k in settings.tournament_k):
        raise ValueError("a tournament compares at least two candidates (--tournament-k >= 2)")
    if any(k < 2 for k in settings.confirm_k):
        raise ValueError("the paired t-test needs at least two ratings of each design (--confirm-k >= 2)")
    for label, values in (("--tournament-k", settings.tournament_k), ("--confirm-k", settings.confirm_k),
                          ("--rho", settings.rhos)):
        if len(set(values)) != len(values):
            raise ValueError(f"{label} repeats a value: {values}")
    if not settings.rhos or any(r < 0 for r in settings.rhos):
        raise ValueError("--rho needs at least one non-negative value")
    if settings.slip_look_sd < 0:
        raise ValueError("--slip-look-sd must be non-negative")
    if not 0.0 < settings.alpha < 1.0:
        raise ValueError("--alpha must lie in (0, 1)")
    if settings.lcb_beta < 0:
        raise ValueError("--lcb-beta must be non-negative")


def _rerate_window(args: dict) -> tuple[int, int]:
    raw = args.get("final_rerate")
    if raw in (None, "", "0,0"):
        return 0, 0
    top, reps = (int(v) for v in str(raw).split(","))
    return top, reps


def load_arm(root: Path, rerate: str = "auto") -> ArmInfo:
    """Budget and design of a STANDARD arm, from its run_metadata.json."""
    root = Path(root)
    meta_path = root / "run_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{root}: no run_metadata.json, so the budget and the initial design are unknown")
    args = json.loads(meta_path.read_text(encoding="utf-8")).get("args", {})
    # The replay refits the standard loop's GP on the logged prefix, so an arm that
    # changed the schedule or the surrogate is not a valid starting point.
    problems = []
    if _rerate_window(args)[0]:
        problems.append("it already re-rates at the end (final_rerate)")
    if int(args.get("replicate_first") or 0):
        problems.append("it replicates early ratings (replicate_first)")
    if (args.get("observation_noise") or "learned") != "learned":
        problems.append("its GP was given the noise (observation_noise)")
    if (args.get("likelihood") or "gaussian") != "gaussian":
        problems.append("its surrogate is not the Gaussian GP (likelihood)")
    if (args.get("input_noise_model") or "none") != "none":
        problems.append("its surrogate is a noisy-input GP (input_noise_model)")
    if args.get("single_error"):
        problems.append("its error hits one trial only (single_error)")
    if args.get("multi_objective"):
        problems.append("it is multi-objective")
    input_error = args.get("input_error") or "none"
    if input_error != "none" and not args.get("input_error_from_sweep"):
        problems.append("it combines a response error with a fixed input error")
    if problems:
        raise ValueError(f"{root} is not a standard single-objective arm: {'; '.join(problems)}")

    rerate_dir = None
    if rerate == "auto":
        candidate = root.parent / RERATE_DIRS.get(root.name, "")
        rerate_dir = candidate if root.name in RERATE_DIRS and candidate.is_dir() else None
    elif rerate not in ("", "none"):
        pairs = dict(part.split("=", 1) for part in rerate.split(",") if part.strip())
        if root.name in pairs:
            rerate_dir = Path(pairs[root.name])
    suffix, window = "", 0
    if rerate_dir is not None:
        rmeta = rerate_dir / "run_metadata.json"
        if not rmeta.is_file():
            raise FileNotFoundError(f"re-rating arm {rerate_dir} has no run_metadata.json")
        rargs = json.loads(rmeta.read_text(encoding="utf-8")).get("args", {})
        top, reps = _rerate_window(rargs)
        if not top:
            raise ValueError(f"{rerate_dir} was given as a re-rating arm but has no final_rerate")
        if int(rargs.get("iterations", -1)) != int(args["iterations"]):
            raise ValueError(f"{rerate_dir} has a different budget from {root}")
        suffix, window = f"_rerate{top}x{reps}", top * reps
    return ArmInfo(
        name=root.name, root=root, iterations=int(args["iterations"]),
        initial_samples=int(args["initial_samples"]), ar1_rho=float(args.get("error_ar1_rho", 0.8)),
        input_arm=input_error != "none", rerate_dir=rerate_dir, rerate_suffix=suffix, rerate_window=window,
    )


def validate_arm(arm: ArmInfo, settings: Settings) -> None:
    longest = max([*settings.tournament_k, *(2 * k for k in settings.confirm_k)])
    if arm.iterations - longest <= arm.initial_samples:
        raise ValueError(
            f"{arm.root}: T = {arm.iterations} with {arm.initial_samples} initial designs leaves no "
            f"model-based search before a {longest}-trial ending"
        )


def read_run(path: Path, iterations: int) -> RunLog:
    df = pd.read_csv(path)
    if len(df) != iterations or not np.array_equal(df["iteration"].to_numpy(), np.arange(1, iterations + 1)):
        raise ValueError(f"{Path(path).name}: expected iterations 1..{iterations}, found {len(df)} rows")
    params = str(df["param_columns"].iloc[0]).split(",")
    deployed = "objective_true_deployed" if "objective_true_deployed" in df.columns else "objective_true"
    return RunLog(
        name=Path(path).name,
        X=df[params].to_numpy(dtype=float),
        observed=df["objective_observed"].to_numpy(dtype=float),
        deployed=df[deployed].to_numpy(dtype=float),
        logged_inference=df["inference_simple_regret_true"].to_numpy(dtype=float),
        y_opt=float(df["y_opt"].iloc[0]),
    )


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def noise_seed(name: str, *parts: int) -> np.random.SeedSequence:
    # crc32 rather than hash(): Python salts str hashes per process, and a draw
    # must not depend on which worker replays the file.
    return np.random.SeedSequence([zlib.crc32(name.encode("utf-8")), *(int(p) for p in parts)])


def idiosyncratic_sd(error_model: str, jitter_std: float, ar1_rho: float = 0.8) -> float:
    """The part of a response error that does NOT cancel between trials of one sitting."""
    if error_model == "none":
        return 0.0
    if error_model in ("gaussian", "bias", "drift"):
        # bias is constant and the drift ramp moves by sigma / (T - onset) per trial,
        # so both are shared; the gaussian jitter on top of them is not.
        return float(jitter_std)
    if error_model == "ar1":
        # e_t = rho e_{t-1} + u_t: the state is shared, the innovation u_t is new.
        return float(jitter_std) * float(np.sqrt(max(0.0, 1.0 - ar1_rho ** 2)))
    raise ValueError(
        f"error model {error_model!r} has no idiosyncratic-noise model for the end-of-study sitting; "
        f"supported: {', '.join(RESPONSE_MODELS + INPUT_MODELS)}"
    )


def sitting_sd_fn(arm: ArmInfo, error_model: str, jitter_std: float, onset: int, settings: Settings):
    """Rating-noise SD of end-of-study trial t (1-based), before any rho scaling."""
    if arm.input_arm:
        base = float(settings.slip_look_sd)
        return lambda trial: base
    base = idiosyncratic_sd(error_model, jitter_std, arm.ar1_rho)
    # Same onset convention as the simulator: trial t is corrupted when t > onset.
    return lambda trial: base if trial > onset else 0.0


# ---------------------------------------------------------------------------
# The loop's GP on a prefix
# ---------------------------------------------------------------------------


def distinct_designs(X: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(first row, mean value) per distinct recorded design, keyed as the loop keys them."""
    means = sim._mean_by_design(list(X), [float(v) for v in values])
    first = np.array([f for _, f in means.values()], dtype=int)
    mean = np.array([m for m, _ in means.values()], dtype=float)
    return first, mean


def torch_seed(name: str, n: int) -> int:
    return int(noise_seed(name, 0, n).generate_state(1)[0])


def fit_loop_gp(train_X: torch.Tensor, train_Y: torch.Tensor, bounds: torch.Tensor, seed: int) -> SingleTaskGP:
    """The loop's surrogate exactly as run_simulation builds it for a learned-noise, Gaussian arm."""
    torch.manual_seed(seed)
    gp = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    return gp


def latent_mean_sd(gp: SingleTaskGP, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        post = gp.posterior(torch.tensor(X, dtype=torch.double))
        mean = post.mean.reshape(-1).numpy()
        sd = post.variance.clamp_min(0.0).sqrt().reshape(-1).numpy()
    return mean, sd


def conditioned_mean(gp: SingleTaskGP, train_X: torch.Tensor, train_Y: torch.Tensor, X_new: torch.Tensor,
                     y_new: torch.Tensor, X_query: torch.Tensor, bounds: torch.Tensor) -> np.ndarray:
    """Posterior mean at X_query once (X_new, y_new) are added as extra observations.

    Exact GP algebra with the prefix fit's hyperparameters held fixed: kernel,
    constant mean, learned homoscedastic noise and the Standardize statistics of
    the prefix. The looks are evidence about the same function, not a reason to
    re-learn its lengthscales, and the experimenter does not know rho, so they
    carry the learned noise. Written out rather than via condition_on_observations
    so every transform is explicit; the tests check it against BoTorch.
    """
    span = bounds[1] - bounds[0]
    A = (torch.cat([train_X, X_new]) - bounds[0]) / span
    Q = (X_query - bounds[0]) / span
    mu = gp.outcome_transform.means.reshape(())
    sd = gp.outcome_transform.stdvs.reshape(())
    y = (torch.cat([train_Y, y_new]).reshape(-1) - mu) / sd
    with torch.no_grad():
        noise = gp.likelihood.noise.reshape(-1)[0]
        eye = torch.eye(A.shape[0], dtype=A.dtype)
        K = gp.covar_module(A, A).to_dense() + noise * eye
        L, info = torch.linalg.cholesky_ex(K)
        jitter = 1e-8
        while int(info) != 0 and jitter <= 1e-3:
            L, info = torch.linalg.cholesky_ex(K + jitter * eye)
            jitter *= 10.0
        if int(info) != 0:
            raise torch.linalg.LinAlgError("conditioned covariance is not positive definite")
        resid = (y - gp.mean_module(A).reshape(-1)).unsqueeze(-1)
        weights = torch.cholesky_solve(resid, L).reshape(-1)
        out = gp.mean_module(Q).reshape(-1) + gp.covar_module(Q, A).to_dense() @ weights
    return (out * sd + mu).numpy()


def search_state(run: RunLog, n: int, bounds: torch.Tensor, beta: float) -> SearchState:
    X, y = run.X[:n], run.observed[:n]
    first, obs_mean = distinct_designs(X, y)
    train_X = torch.tensor(X, dtype=torch.double)
    train_Y = torch.tensor(y.reshape(-1, 1), dtype=torch.double)
    try:
        gp = fit_loop_gp(train_X, train_Y, bounds, torch_seed(run.name, n))
        mean, sd = latent_mean_sd(gp, X[first])
        lcb = mean - beta * sd
        if not np.all(np.isfinite(lcb)):
            raise FloatingPointError("non-finite posterior")
        failed = False
    except Exception as exc:  # noqa: BLE001 - recorded per run, as the loop records acq_opt_failed
        # A failed fit degrades visibly, never silently: the ship rule falls back
        # to the ratings and the row says so (gp_failed).
        print(f"[gp-fallback] {run.name} n={n}: {exc}", file=sys.stderr)
        gp, lcb, failed = None, obs_mean.copy(), True
    return SearchState(n=n, first=first, obs_mean=obs_mean, lcb=lcb, gp=gp, train_X=train_X,
                       train_Y=train_Y, bounds=bounds, gp_failed=failed)


# ---------------------------------------------------------------------------
# The procedures
# ---------------------------------------------------------------------------


def paired_one_sided_p(better: np.ndarray, worse: np.ndarray) -> float:
    """One-sided paired t-test p-value that ``better`` exceeds ``worse``."""
    diff = np.asarray(better, dtype=float) - np.asarray(worse, dtype=float)
    if diff.size < 2:
        raise ValueError("a paired t-test needs at least two pairs")
    if np.ptp(diff) == 0.0:
        # Exact ratings (a clean twin): the t statistic is undefined, its sign is not.
        return 0.0 if diff[0] > 0 else 1.0
    return float(sps.ttest_rel(better, worse, alternative="greater").pvalue)


def check_logged_inference(run: RunLog, ns: list[int], rule: str = "best_observed") -> None:
    """Recompute the loop's deployed design at each n and compare with the logged regret."""
    for n in ns:
        if rule == "best_observed":
            pick = int(np.argmax(run.observed[:n]))
        else:
            first, mean = distinct_designs(run.X[:n], run.observed[:n])
            pick = int(first[int(np.argmax(mean))])
        recomputed = run.y_opt - float(run.deployed[pick])
        if abs(recomputed - float(run.logged_inference[n - 1])) > LOGGED_ATOL:
            raise ReproductionError(
                f"{run.name}: deployed regret after {n} trials recomputes to {recomputed!r}, "
                f"the log says {run.logged_inference[n - 1]!r} ({rule})"
            )


def standard_process(run: RunLog, initial: int) -> dict:
    pick = int(np.argmax(run.observed))
    c_row = int(np.argmax(run.observed[:initial]))
    claim = bool(run.observed[pick] > run.observed[c_row])
    truly_better = bool(run.deployed[pick] > run.deployed[c_row])
    # The loop's own logged value (replay_run has just matched it to within 1e-9),
    # so the reference is bit-identical to final_inference_simple_regret_true.
    return {"family": "standard", "regret": float(run.logged_inference[-1]), "claim": claim,
            "truly_better": truly_better, "false_claim": claim and not truly_better}


def tournament(run: RunLog, state: SearchState, k: int, rhos: tuple[float, ...], sd_of_trial, T: int) -> dict:
    z = np.random.default_rng(noise_seed(run.name, PROC_TOURNAMENT, k)).standard_normal(k)
    sds = np.array([sd_of_trial(t) for t in range(T - k + 1, T + 1)], dtype=float)
    out: dict[str, dict] = {}
    for cand, scores in (("lcb", state.lcb), ("obs", state.obs_mean)):
        # Stable sort: ties go to the design visited first, as the loop's argmax does.
        rows = state.first[np.argsort(-scores, kind="stable")[:k]]
        m = len(rows)
        truth = run.deployed[rows]
        Xc = torch.tensor(run.X[rows], dtype=torch.double)
        for rho in rhos:
            looks = truth + rho * sds[:m] * z[:m]
            look_pick = int(np.argmax(looks))
            post_pick = look_pick
            if state.gp is not None:
                post = conditioned_mean(state.gp, state.train_X, state.train_Y, Xc,
                                        torch.tensor(looks, dtype=torch.double).reshape(-1, 1), Xc, state.bounds)
                post_pick = int(np.argmax(post))
            for winner, pick in (("post", post_pick), ("look", look_pick)):
                out[f"tournament_k{k}_{cand}_rho{rho:g}_{winner}"] = {
                    "family": "tournament", "k": k, "candidates": cand, "rho": float(rho), "winner": winner,
                    "regret": run.y_opt - float(truth[pick]),
                    "n_candidates": m,
                    "pick_changed": bool(pick != 0),
                    # the top-ranked candidate without a sitting, and a perfect sitting's ceiling
                    "top_candidate_regret": run.y_opt - float(truth[0]),
                    "best_candidate_regret": run.y_opt - float(truth.max()),
                    "gp_failed": state.gp_failed,
                }
    return out


def confirmation_decision(f_d: float, f_c: float, r_d: np.ndarray, r_c: np.ndarray, alpha: float) -> dict:
    shipped_d = bool(np.mean(r_d) >= np.mean(r_c))
    p = paired_one_sided_p(r_d, r_c)
    claim = bool(p < alpha)
    truly_better = bool(f_d > f_c)
    return {"shipped_value": f_d if shipped_d else f_c, "shipped_d": shipped_d, "p_value": p,
            "claim": claim, "truly_better": truly_better, "false_claim": claim and not truly_better}


def confirmation(run: RunLog, state: SearchState, k: int, initial: int, sd_of_trial, T: int, alpha: float) -> dict:
    d_row = int(state.first[int(np.argmax(state.lcb))])
    c_row = int(np.argmax(run.observed[:initial]))
    f_d, f_c = float(run.deployed[d_row]), float(run.deployed[c_row])
    rng = np.random.default_rng(noise_seed(run.name, PROC_CONFIRM, k))
    z_d, z_c = rng.standard_normal(k), rng.standard_normal(k)
    # Counterbalanced: pair j is (D, C) for even j and (C, D) for odd j. The shared
    # state cancels within a pair, so order only enters through the onset gate.
    start = T - 2 * k + 1
    d_trials = [start + 2 * j + (j % 2) for j in range(k)]
    c_trials = [start + 2 * j + 1 - (j % 2) for j in range(k)]
    r_d = f_d + np.array([sd_of_trial(t) for t in d_trials]) * z_d
    r_c = f_c + np.array([sd_of_trial(t) for t in c_trials]) * z_c
    decision = confirmation_decision(f_d, f_c, r_d, r_c, alpha)
    return {"family": "confirmation", "k": k, "regret": run.y_opt - decision.pop("shipped_value"), **decision,
            "d_is_c": sim._design_key(run.X[d_row]) == sim._design_key(run.X[c_row]),
            "gp_failed": state.gp_failed}


def replay_run(run: RunLog, arm: ArmInfo, settings: Settings, bounds: torch.Tensor, sd_of_trial) -> dict:
    T = arm.iterations
    ns = sorted({T - k for k in settings.tournament_k} | {T - 2 * k for k in settings.confirm_k})
    # Every prefix the replay starts from must reproduce the loop's own record.
    check_logged_inference(run, ns + [T])
    states = {n: search_state(run, n, bounds, settings.lcb_beta) for n in ns}
    out = {"standard": standard_process(run, arm.initial_samples)}
    for k in settings.tournament_k:
        out.update(tournament(run, states[T - k], k, settings.rhos, sd_of_trial, T))
    for k in settings.confirm_k:
        out[f"confirm_k{k}"] = confirmation(run, states[T - 2 * k], k, arm.initial_samples, sd_of_trial, T,
                                            settings.alpha)
    return out


def rerate_outcome(path: Path, standard_run: RunLog, arm: ArmInfo) -> dict | None:
    if not path.is_file():
        return None
    log = read_run(path, arm.iterations)
    check_logged_inference(log, [arm.iterations], rule="best_mean")
    prefix = arm.iterations - arm.rerate_window
    identical = bool(np.array_equal(log.X[:prefix], standard_run.X[:prefix])
                     and np.array_equal(log.observed[:prefix], standard_run.observed[:prefix]))
    return {"family": "rerate", "k": arm.rerate_window, "regret": float(log.logged_inference[-1]),
            "prefix_identical": identical}


# ---------------------------------------------------------------------------
# Tasks: one (landscape, acquisition, seed) stem per task, its clean twin replayed once
# ---------------------------------------------------------------------------


def _accepted_variant(error_model: str, jitter_std: float, variant: str) -> bool:
    # The standard process only: the bias arm's scaled bias is in the name, and
    # any other suffix (single error, known noise, ...) is a different arm.
    return variant == "" or (error_model == "bias" and variant == f"bias{jitter_std:g}")


def build_tasks(arm: ArmInfo, filters: Filters, settings: Settings, out_dir: Path,
                resume: bool = False) -> tuple[list[dict], Counter]:
    if filters.error_models is not None:
        unknown = sorted(set(filters.error_models) - set(RESPONSE_MODELS + INPUT_MODELS))
        if unknown:
            raise ValueError(f"unsupported error model(s) {unknown}; supported: {RESPONSE_MODELS + INPUT_MODELS}")
    allowed = set(INPUT_MODELS if arm.input_arm else RESPONSE_MODELS)
    if filters.error_models is not None:
        allowed &= set(filters.error_models)
    baselines, jittered = aer.index_runs(arm.root)
    counts: Counter = Counter()
    by_stem: dict[str, list[dict]] = {}
    for rec in jittered:
        if filters.functions is not None and rec["dataset"] not in filters.functions:
            continue
        if filters.acquisitions is not None and rec["acquisition"] not in filters.acquisitions:
            continue
        if filters.seeds is not None and rec["seed"] not in filters.seeds:
            continue
        if filters.onsets is not None and rec["jitter_iteration"] not in filters.onsets:
            continue
        if filters.stds is not None and not np.any(np.isclose(rec["jitter_std"], filters.stds)):
            continue
        if rec["error_model"] not in allowed:
            counts[f"skipped: error model {rec['error_model']}"] += 1
            continue
        if not _accepted_variant(rec["error_model"], rec["jitter_std"], rec["variant"]):
            counts[f"skipped: variant {rec['variant']}"] += 1
            continue
        by_stem.setdefault(rec["stem"], []).append(rec)
    tasks = []
    for stem, runs in sorted(by_stem.items()):
        clean = baselines.get(stem)
        if clean is None:
            counts["skipped: no clean twin"] += len(runs)
            continue
        dataset = runs[0]["dataset"]
        out_path = out_dir / "per_run" / arm.name / dataset / f"{stem}.csv"
        if resume and out_path.is_file():
            counts["resumed stems"] += 1
            continue
        counts["noisy runs"] += len(runs)
        tasks.append({
            "arm": arm, "settings": settings, "dataset": dataset, "stem": stem,
            "acquisition": runs[0]["acquisition"], "seed": runs[0]["seed"], "clean_path": str(clean),
            "runs": [{key: (str(v) if key == "path" else v) for key, v in r.items()} for r in runs],
            "out_path": str(out_path),
        })
    return tasks, counts


def _assemble_rows(base: dict, noisy_out: dict, clean_out: dict) -> list[dict]:
    ref_noisy, ref_clean = noisy_out["standard"]["regret"], clean_out["standard"]["regret"]
    rows = []
    for proc, n_res in noisy_out.items():
        c_res = clean_out.get(proc)
        if c_res is None:
            continue
        row = {**base, "procedure": proc, **{m: n_res.get(m) for m in META},
               "regret_noisy": n_res["regret"], "regret_clean": c_res["regret"],
               "ref_noisy": ref_noisy, "ref_clean": ref_clean}
        for suffix, res in (("noisy", n_res), ("clean", c_res)):
            for key, value in res.items():
                if key not in META and key != "regret":
                    row[f"{key}_{suffix}"] = value
        rows.append(row)
    return rows


def replay_stem_rows(task: dict) -> pd.DataFrame:
    arm, settings = task["arm"], task["settings"]
    spec = bb.BENCHMARKS[task["dataset"]]
    bounds = torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)
    clean_path = Path(task["clean_path"])
    clean = read_run(clean_path, arm.iterations)
    clean_out = replay_run(clean, arm, settings, bounds, sitting_sd_fn(arm, "none", 0.0, 0, settings))
    rerate_name = arm.rerate_suffix.lstrip("_")
    if arm.rerate_dir is not None:
        rr = rerate_outcome(arm.rerate_dir / task["dataset"] / clean_path.name, clean, arm)
        if rr is not None:
            clean_out[rerate_name] = rr
    rows: list[dict] = []
    for rec in task["runs"]:
        path = Path(rec["path"])
        noisy = read_run(path, arm.iterations)
        sd_fn = sitting_sd_fn(arm, rec["error_model"], rec["jitter_std"], rec["jitter_iteration"], settings)
        noisy_out = replay_run(noisy, arm, settings, bounds, sd_fn)
        if arm.rerate_dir is not None and rerate_name in clean_out:
            rr = rerate_outcome(arm.rerate_dir / task["dataset"] / f"{path.stem}{arm.rerate_suffix}.csv", noisy, arm)
            if rr is not None:
                noisy_out[rerate_name] = rr
        base = {"arm": arm.name, "dataset": task["dataset"], "acquisition": task["acquisition"],
                "seed": task["seed"], "error_model": rec["error_model"], "jitter_std": rec["jitter_std"],
                "jitter_iteration": rec["jitter_iteration"], "variant": rec["variant"], "file": path.name}
        rows.extend(_assemble_rows(base, noisy_out, clean_out))
    return pd.DataFrame(rows)


def _worker_init() -> None:
    # One thread per worker: small GPs are fastest single-threaded, and the fit
    # must not depend on how many workers share the machine.
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")


def replay_stem(task: dict) -> dict:
    start = time.perf_counter()
    frame = replay_stem_rows(task)
    out_path = Path(task["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out_path)  # a killed job never leaves a half-written stem for --resume to trust
    return {"stem": task["stem"], "rows": len(frame), "runs": len(task["runs"]),
            "seconds": time.perf_counter() - start}


def run_tasks(tasks: list[dict], workers: int, out_dir: Path) -> list[str]:
    failures: list[str] = []
    done_runs, total_runs = 0, sum(len(t["runs"]) for t in tasks)
    start = time.perf_counter()

    def report(info: dict) -> None:
        nonlocal done_runs
        done_runs += info["runs"]
        elapsed = time.perf_counter() - start
        eta = elapsed / max(done_runs, 1) * (total_runs - done_runs)
        print(f"  [{done_runs}/{total_runs} runs] {info['stem']}: {info['seconds']:.1f}s "
              f"(elapsed {elapsed / 60:.1f} min, eta {eta / 60:.1f} min)", flush=True)

    if workers <= 1:
        _worker_init()
        for task in tasks:
            try:
                report(replay_stem(task))
            except Exception as exc:  # noqa: BLE001 - listed and rerun with --resume
                failures.append(f"{task['stem']}: {exc!r}")
                print(f"  FAILED {task['stem']}: {exc!r}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = {pool.submit(replay_stem, task): task["stem"] for task in tasks}
            for future in as_completed(futures):
                try:
                    report(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{futures[future]}: {exc!r}")
                    print(f"  FAILED {futures[future]}: {exc!r}", file=sys.stderr)
    if failures:
        (out_dir / "failures.log").write_text("\n".join(failures) + "\n", encoding="utf-8")
    return failures


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else str(v).strip().lower() == "true")


def load_per_run(out_dir: Path, arm_names: list[str], filters: Filters) -> pd.DataFrame:
    files = [f for name in arm_names for f in sorted((out_dir / "per_run" / name).glob("*/*.csv"))]
    if not files:
        raise SystemExit(f"no replayed runs under {out_dir / 'per_run'} for arms {arm_names}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    keep = np.ones(len(frame), dtype=bool)
    if filters.functions is not None:
        keep &= frame["dataset"].isin(filters.functions)
    if filters.acquisitions is not None:
        keep &= frame["acquisition"].isin(filters.acquisitions)
    if filters.seeds is not None:
        keep &= frame["seed"].isin(filters.seeds)
    if filters.error_models is not None:
        keep &= frame["error_model"].isin(filters.error_models)
    if filters.onsets is not None:
        keep &= frame["jitter_iteration"].isin(filters.onsets)
    if filters.stds is not None:
        keep &= np.isclose(frame["jitter_std"].to_numpy()[:, None], np.asarray(filters.stds)[None, :]).any(axis=1)
    return frame[keep].reset_index(drop=True)


def recovery_table(frame: pd.DataFrame, opt_z: dict[str, float], split_acquisition: bool = False) -> pd.DataFrame:
    """analyse_boba_adaptations' estimand, per cell and pooled, for every treatment procedure."""
    df = frame[frame["family"].isin(TREATMENT_FAMILIES)]
    if df.empty:
        return pd.DataFrame()
    z = df["dataset"].map(lambda d: opt_z.get(d, 1.0)).astype(float)
    df = df.assign(ref_noisy=df["ref_noisy"] / z, ref_clean=df["ref_clean"] / z,
                   trt_noisy=df["regret_noisy"] / z, trt_clean=df["regret_clean"] / z)
    keys = ["arm", "procedure", "error_model"] + (["acquisition"] if split_acquisition else [])
    rows = []
    for key, group in df.groupby(keys, sort=True):
        base = {**dict(zip(keys, key)), **{m: group[m].iloc[0] for m in META}}
        rng = np.random.default_rng(aba.BOOTSTRAP_SEED)
        cells = [{**base, "scope": "cell", "jitter_std": float(std), "jitter_iteration": int(onset),
                  **aba.summarise(block, rng)}
                 for (std, onset), block in group.groupby(["jitter_std", "jitter_iteration"])]
        for cell, q in zip(cells, multipletests([c["wilcoxon_p"] for c in cells], method="fdr_bh")[1]):
            cell["wilcoxon_p_fdr"] = float(q)
        pooled = {**base, "scope": "pooled", "jitter_std": np.nan, "jitter_iteration": np.nan,
                  **aba.summarise(group, np.random.default_rng(aba.BOOTSTRAP_SEED)), "wilcoxon_p_fdr": np.nan}
        rows.extend(cells + [pooled])
    return pd.DataFrame(rows)


def claim_rates(block: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Shares of runs, with landscape-bootstrap intervals (landscapes resampled, counts pooled)."""
    claim = _as_bool(block["claim_noisy"])
    better = _as_bool(block["truly_better_noisy"])
    false = claim & ~better
    parts = pd.DataFrame({"dataset": block["dataset"].to_numpy(), "n": 1, "claim": claim.to_numpy(),
                          "false": false.to_numpy(), "better": better.to_numpy(),
                          "hit": (claim & better).to_numpy(), "not_better": (~better).to_numpy()})
    per = parts.groupby("dataset")[["n", "claim", "false", "better", "hit", "not_better"]].sum()
    cols = {c: i for i, c in enumerate(per.columns)}
    counts = per.to_numpy(dtype=float)
    boot = counts[rng.integers(0, len(per), size=(aba.BOOTSTRAP_REPS, len(per)))].sum(axis=1)
    total = counts.sum(axis=0)
    out = {"n_runs": int(total[cols["n"]]), "n_landscapes": int(len(per))}
    for name, (num, den) in {"claim_rate": ("claim", "n"), "false_claim_rate": ("false", "n"),
                             "false_discovery_share": ("false", "claim"), "power": ("hit", "better"),
                             "size": ("false", "not_better"), "truly_better_share": ("better", "n")}.items():
        out[name] = float(total[cols[num]] / total[cols[den]]) if total[cols[den]] > 0 else np.nan
        ok = boot[:, cols[den]] > 0
        ratio = boot[ok, cols[num]] / boot[ok, cols[den]]
        out[f"{name}_lo"] = float(np.percentile(ratio, 2.5)) if ok.sum() >= 100 else np.nan
        out[f"{name}_hi"] = float(np.percentile(ratio, 97.5)) if ok.sum() >= 100 else np.nan
    out["clean_claim_rate"] = float(_as_bool(block["claim_clean"]).mean())
    if "shipped_d_noisy" in block and block["shipped_d_noisy"].notna().all():
        out["shipped_d_share"] = float(_as_bool(block["shipped_d_noisy"]).mean())
        out["d_is_c_share"] = float(_as_bool(block["d_is_c_noisy"]).mean())
    return out


def claims_table(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame[frame["family"].isin(("confirmation", "standard"))]
    rows = []
    for (arm, proc, model), group in df.groupby(["arm", "procedure", "error_model"], sort=True):
        base = {"arm": arm, "procedure": proc, "family": group["family"].iloc[0], "k": group["k"].iloc[0],
                "error_model": model}
        rng = np.random.default_rng(aba.BOOTSTRAP_SEED)
        for (std, onset), block in group.groupby(["jitter_std", "jitter_iteration"]):
            rows.append({**base, "scope": "cell", "jitter_std": float(std), "jitter_iteration": int(onset),
                         **claim_rates(block, rng)})
        rows.append({**base, "scope": "pooled", "jitter_std": np.nan, "jitter_iteration": np.nan,
                     **claim_rates(group, rng)})
    return pd.DataFrame(rows)


def matched_to_rerate(frame: pd.DataFrame) -> pd.DataFrame:
    cell = ["arm", "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration"]
    keys = frame.loc[frame["family"] == "rerate", cell].drop_duplicates()
    return frame.merge(keys, on=cell, how="inner")


def print_summary(recovery: pd.DataFrame, claims: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    if not recovery.empty:
        cells = recovery[recovery["scope"] == "cell"].assign(
            cond=lambda d: d.apply(lambda r: f"{r.jitter_std:g}/it{int(r.jitter_iteration) + 1}", axis=1),
            text=lambda d: d.apply(lambda r: f"{r.recovered:+.0%} [{r.recovered_lo:+.0%},{r.recovered_hi:+.0%}] "
                                             f"price {r.price:+.3f}", axis=1))
        for (arm, model), block in cells.groupby(["arm", "error_model"]):
            print(f"\n--- recovered share of the deployed cost [95% landscape CI], price without error: "
                  f"{arm} / {model} ---")
            print(block.pivot_table(index="procedure", columns="cond", values="text", aggfunc="first").to_string())
    if not claims.empty:
        cells = claims[claims["scope"] == "cell"].assign(
            cond=lambda d: d.apply(lambda r: f"{r.jitter_std:g}/it{int(r.jitter_iteration) + 1}", axis=1),
            text=lambda d: d.apply(lambda r: f"claim {r.claim_rate:.0%} false {r.false_claim_rate:.0%} "
                                             f"power {r.power:.0%}", axis=1))
        for (arm, model), block in cells.groupby(["arm", "error_model"]):
            print(f"\n--- improvement claims against the best initial design: {arm} / {model} ---")
            print(block.pivot_table(index="procedure", columns="cond", values="text", aggfunc="first").to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _names(raw: str) -> frozenset[str] | None:
    return None if raw.strip().lower() == "all" else frozenset(v.strip() for v in raw.split(",") if v.strip())


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=DEFAULT_ARMS, help="comma-separated STANDARD arm directories")
    p.add_argument("--output-dir", type=Path, default=Path("output-boba/analysis/end_of_study"))
    p.add_argument("--functions", default="all")
    p.add_argument("--acquisitions", default="logei,qnei,ucb")
    p.add_argument("--seeds", default="7,8,9,10,11,12,13,14,15,16")
    p.add_argument("--error-models", default="all",
                   help=f"subset of {','.join(RESPONSE_MODELS + INPUT_MODELS)}; each arm keeps the ones it has")
    p.add_argument("--stds", default="all", help="magnitudes as in the file names (input-error arms: the scale)")
    p.add_argument("--onsets", default="all", help="jitter_iteration values (0-based, as in the file names)")
    p.add_argument("--tournament-k", default="3,5")
    p.add_argument("--confirm-k", default="2,4")
    p.add_argument("--rho", default="1,0.5", help="look noise as a multiple of the idiosyncratic SD")
    p.add_argument("--slip-look-sd", type=float, default=0.25,
                   help="rating-noise SD of a rendered design in the input-error arms (scaled by rho in a sitting)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lcb-beta", type=float, default=1.0, help="latent SDs subtracted by the ship rule")
    p.add_argument("--rerate-dirs", default="auto",
                   help="'auto' (the known re-rating arm beside each standard arm), 'none', or arm=dir,...")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="skip stems whose per-run file already exists")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--split-acquisition", action="store_true", help="recovery per acquisition as well")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    settings = Settings(
        tournament_k=tuple(int(v) for v in args.tournament_k.split(",") if v.strip()),
        confirm_k=tuple(int(v) for v in args.confirm_k.split(",") if v.strip()),
        rhos=tuple(float(v) for v in args.rho.split(",") if v.strip()),
        slip_look_sd=float(args.slip_look_sd), alpha=float(args.alpha), lcb_beta=float(args.lcb_beta),
    )
    validate_settings(settings)
    stds = None if args.stds.strip().lower() == "all" else tuple(float(v) for v in args.stds.split(",") if v.strip())
    onsets = _names(args.onsets)
    filters = Filters(
        functions=_names(args.functions), acquisitions=_names(args.acquisitions),
        seeds=None if _names(args.seeds) is None else frozenset(int(s) for s in _names(args.seeds)),
        error_models=_names(args.error_models), stds=stds,
        onsets=None if onsets is None else frozenset(int(o) for o in onsets),
    )
    arms = [load_arm(Path(a.strip()), args.rerate_dirs) for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        validate_arm(arm, settings)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    settings_path = out_dir / "settings.json"
    current = json.loads(json.dumps(dataclasses.asdict(settings)))
    if settings_path.is_file():
        saved = json.loads(settings_path.read_text(encoding="utf-8"))
        if saved != current:
            raise SystemExit(f"{out_dir} holds replays made with other settings {saved}; use another --output-dir")
    else:
        settings_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    if not args.summary_only:
        tasks = []
        for arm in arms:
            arm_tasks, counts = build_tasks(arm, filters, settings, out_dir, resume=args.resume)
            print(f"{arm.name}: {len(arm_tasks)} stems, {dict(counts)}; re-rating context: "
                  f"{arm.rerate_dir or 'none'}")
            tasks.extend(arm_tasks)
        failures = run_tasks(tasks, args.workers, out_dir)
        if failures:
            raise SystemExit(f"{len(failures)} stems failed (see {out_dir / 'failures.log'}); "
                             f"fix and rerun with --resume before summarising")

    frame = load_per_run(out_dir, [arm.name for arm in arms], filters)
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    frame.to_csv(out_dir / "end_of_study_per_run.csv.gz", index=False)
    recovery = recovery_table(frame, opt_z, args.split_acquisition)
    recovery.to_csv(out_dir / "end_of_study_recovery.csv", index=False)
    claims = claims_table(frame)
    claims.to_csv(out_dir / "end_of_study_claims.csv", index=False)
    written = ["end_of_study_per_run.csv.gz", "end_of_study_recovery.csv", "end_of_study_claims.csv"]
    if (frame["family"] == "rerate").any():
        matched = recovery_table(matched_to_rerate(frame), opt_z, args.split_acquisition)
        matched.to_csv(out_dir / "end_of_study_recovery_rerate_matched.csv", index=False)
        written.append("end_of_study_recovery_rerate_matched.csv")
        prefix_cols = [c for c in ("prefix_identical_noisy", "prefix_identical_clean") if c in frame]
        rr = frame[frame["family"] == "rerate"]
        print(f"re-rating context: {len(rr)} matched runs; prefix identical to the standard run in "
              + ", ".join(f"{c} {_as_bool(rr[c]).mean():.1%}" for c in prefix_cols))
    failed = frame.filter(like="gp_failed").apply(_as_bool).to_numpy().any(axis=1).mean() if len(frame) else 0.0
    print(f"{len(frame):,} procedure rows from {frame['file'].nunique():,} noisy runs on "
          f"{frame['dataset'].nunique()} landscapes; rows with a GP fallback: {failed:.2%}")
    print_summary(recovery, claims)
    print(f"\nWrote {', '.join(written)} under {out_dir}")


if __name__ == "__main__":
    main()
