"""Replay the multi-objective arm's deployment step with GP-refit fronts.

What the arm deploys
--------------------
``run_simulation``'s multi-objective branch reports the non-dominated set of the
OBSERVED objective vectors and scores it by the hypervolume of their TRUE values
(``inference_value_true``). Under rating error that set is partly the noise's
choice: a design with lucky ratings joins the front, a truly better one with
unlucky ratings leaves it. An independent recomputation found about half of the
reported front truly dominated at 1 SD.

What this replays
-----------------
Nothing is re-simulated and no trial is added. Every rule reads the same logged
ratings the standard process read, so the number of human trials is unchanged.
For each logged run one BoTorch default SingleTaskGP per objective (Normalize over
the design box, Standardize) is refit on all evaluated designs, and three
deployment rules each pick a set:

    observed        ND of the observed vectors -- the arm's own rule. It must
                    reproduce inference_value_true, and every run is checked.
    posterior_mean  ND of the posterior means at the evaluated designs.
    lower_bound     ND of posterior mean minus one latent SD per objective.

The refit rules are trimmed to the observed front's cardinality by greedy removal
of the smallest hypervolume contribution, computed in the rule's own estimate
space with the arm's reference point, so no rule can win by shipping more
designs. They are never padded: a refit front smaller than the observed one ships
fewer designs, which can only cost it hypervolume.

Scores
------
hv        (max_hv - true hypervolume of the deployed set) / floor gap, the gap
          being analyse_boba_mo.achievable_gain (max_hv minus the mean clean
          model-free final hypervolume).
dm        A hidden decision-maker. For 50 Dirichlet(1) weight vectors per seed it
          picks the deployed member with the best ESTIMATED weighted Chebyshev
          utility; the pick is scored by its TRUE utility regret against the best
          evaluated design. Selection only: zero in a clean run under the
          observed rule.
dm_level  The same pick's true Chebyshev shortfall from the utopia point. Unlike
          dm it also carries what the error did to the search.

Estimand
--------
As in analyse_boba_adaptations.py. A deployment rule changes the clean run's
deployment too, so each is scored against the STANDARD observed-ND rule:

    cost      = ref_noisy - ref_clean    what error costs the standard rule
    gain      = ref_noisy - trt_noisy    what the refit rule saves under error
    price     = trt_clean - ref_clean    what the refit rule costs without error
    recovered = gain / cost

per magnitude x onset, as ratios of problem means, with a cluster bootstrap over
problems, on the problems analyse_boba_mo.py's headroom screen admits.

    python scripts/replay_mo_front.py --n-jobs 6
    python scripts/replay_mo_front.py --input-dir output-boba-mo-halo --screen-dir output-boba-mo --error-cross-corr 0.85 --n-jobs 6
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import boba_multiobjective as mob  # noqa: E402

RULES = ("observed", "posterior_mean", "lower_bound")
REFERENCE_RULE = "observed"
# Response label -> per-run column. hv_loss is filled in once the floor gap is known.
RESPONSES = {"hv": "hv_loss", "dm": "dm_regret", "dm_level": "dm_shortfall"}
MODEL_FREE = ("random", "sobol")
HYPERVOLUME_ACQUISITIONS = ("qehvi", "qnehvi", "qlogehvi", "qlognehvi")
N_WEIGHTS = 50
WEIGHT_SEED = 20260914
UTOPIA_LOG2_SAMPLES = 13
LOWER_BOUND_SDS = 1.0
# The multi-objective arm is memory heavy and this machine is shared; see the
# worker-budget note in run_boba_gaps_mo.ps1.
MAX_JOBS = 6
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260914
CHECK_TOLERANCE = 1e-9
# Relative size below which a hypervolume difference is rounding, not volume.
_CONTRIBUTION_EPS = 1e-12

_BASELINE_RE = re.compile(
    r"^bo_sensor_error_(?P<dataset>[a-z0-9]+)_multi_objective_(?P<acquisition>[a-z0-9]+)"
    r"_seed(?P<seed>\d+)_baseline_exact(?P<suffix>.*)\.csv$"
)
# The std group cannot contain "_", so a variant suffix such as "_xcorr0.85" can
# never be read as part of the magnitude, dots and all.
_NOISY_RE = re.compile(
    r"^bo_sensor_error_(?P<dataset>[a-z0-9]+)_multi_objective_(?P<acquisition>[a-z0-9]+)"
    r"_seed(?P<seed>\d+)_jittered_exact_(?P<channel>[^_]+)_jit(?P<onset>\d+)"
    r"_std(?P<std>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?P<suffix>_.*)?\.csv$"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dir", type=Path, default=Path("output-boba-mo"))
    p.add_argument(
        "--screen-dir", type=Path, default=None,
        help="Where the floor gap and the headroom screen come from (its */evaluation "
        "outputs and run_metadata.json). Defaults to --input-dir. An arm that runs only "
        "one acquisition has no model-free floor of its own; point this at output-boba-mo.",
    )
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to <input-dir>/analysis/front_replay.")
    p.add_argument("--mo-stats-path", type=Path, default=mob.DEFAULT_MO_STATS_PATH,
                   help="Used only when --input-dir has no run_metadata.json.")
    p.add_argument("--problems", type=str, default=None,
                   help="Comma-separated; default every admissible problem found.")
    p.add_argument("--acquisitions", type=str, default=",".join(HYPERVOLUME_ACQUISITIONS))
    p.add_argument("--seeds", type=str, default=None, help="Default: every seed found.")
    p.add_argument("--jitter-stds", type=str, default=None, help="Default: every magnitude found.")
    p.add_argument("--jitter-iterations", type=str, default=None, help="Default: every onset found.")
    p.add_argument("--error-model", type=str, default="gaussian",
                   help="The error channel in the filename (run_error_label).")
    p.add_argument(
        "--error-cross-corr", type=float, default=None,
        help="Replay the arm run with the driver's --error-cross-corr RHO. The filename "
        "part is derived from the driver's own _variant_suffix; the clean runs stay the "
        "unsuffixed ones, since error correlation does not touch a clean run.",
    )
    p.add_argument("--variant-suffix", type=str, default="",
                   help="The noisy runs' filename variant part, spelled by hand (e.g. '_xcorr0.85').")
    p.add_argument("--baseline-suffix", type=str, default="",
                   help="The clean runs' filename arm part, spelled by hand (e.g. '_noise-known').")
    p.add_argument("--admissible", type=str, default=None,
                   help="Comma-separated problems to score, overriding the headroom screen.")
    p.add_argument("--include-inadmissible", action="store_true", default=False,
                   help="Score every problem with a floor gap, including those the screen drops.")
    p.add_argument("--lower-bound-sds", type=float, default=LOWER_BOUND_SDS)
    p.add_argument("--n-weights", type=int, default=N_WEIGHTS)
    p.add_argument("--check-tolerance", type=float, default=CHECK_TOLERANCE,
                   help="Relative tolerance for the observed-ND reproduction check.")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--resume", action="store_true", default=False,
                   help="Reuse per-group caches whose files and settings match.")
    p.add_argument("--dry-run", action="store_true", default=False)
    return p.parse_args(argv)


def _names(raw: str | None) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else []


def _floats(raw: str | None) -> list[float]:
    return [float(v) for v in _names(raw)]


def _ints(raw: str | None) -> list[int]:
    return [int(v) for v in _names(raw)]


def validate_args(args: argparse.Namespace) -> None:
    """Refuse combinations that would score the wrong runs or the wrong rule."""
    if not 1 <= args.n_jobs <= MAX_JOBS:
        raise ValueError(
            f"--n-jobs must be between 1 and {MAX_JOBS}: the hypervolume arm is memory "
            f"heavy and other jobs share the machine (got {args.n_jobs})."
        )
    if args.error_cross_corr is not None and args.variant_suffix:
        raise ValueError(
            "--error-cross-corr derives the filename variant from the driver and "
            "--variant-suffix spells it by hand; pass one, not both."
        )
    if args.error_cross_corr is not None and not -1.0 <= args.error_cross_corr <= 1.0:
        raise ValueError(f"--error-cross-corr is a correlation and must lie in [-1, 1]; got {args.error_cross_corr}.")
    for flag, value in (("--variant-suffix", args.variant_suffix), ("--baseline-suffix", args.baseline_suffix)):
        if value and not value.startswith("_"):
            raise ValueError(f"{flag} must start with '_', as the driver writes it; got {value!r}.")
    if args.error_model == "none":
        raise ValueError("'none' is the clean-run marker, not an error channel; name the channel the noisy runs used.")
    if not args.lower_bound_sds > 0:
        raise ValueError("--lower-bound-sds must be positive; at 0 the lower-bound rule is the posterior-mean rule.")
    if args.n_weights < 1:
        raise ValueError("--n-weights must be at least 1.")
    if args.check_tolerance < 0:
        raise ValueError("--check-tolerance must be non-negative.")
    acquisitions = _names(args.acquisitions)
    if not acquisitions:
        raise ValueError("--acquisitions is empty.")
    floors = [a for a in acquisitions if a in MODEL_FREE]
    if floors:
        raise ValueError(
            f"{floors} are the model-free floors that define the gap; they are not scored "
            "here, as in analyse_boba_mo.py."
        )
    if args.admissible and args.include_inadmissible:
        raise ValueError("--admissible names the problems itself; drop --include-inadmissible.")


# ---------------------------------------------------------------------------
# Finding the runs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunName:
    kind: str  # "clean" or "noisy"
    dataset: str
    acquisition: str
    seed: int
    suffix: str
    channel: str | None = None
    jitter_iteration: int | None = None
    jitter_std: float | None = None


@dataclasses.dataclass(frozen=True)
class RunGroup:
    """One clean run and the noisy runs that pair with it."""

    dataset: str
    acquisition: str
    seed: int
    baseline: Path
    noisy: tuple[tuple[float, int, Path], ...]  # (jitter_std, jitter_iteration, path)


def parse_run_name(name: str) -> RunName | None:
    """Parse a multi-objective per-run CSV name; None for anything else."""
    m = _NOISY_RE.match(name)
    if m:
        return RunName(
            kind="noisy", dataset=m["dataset"], acquisition=m["acquisition"], seed=int(m["seed"]),
            suffix=m["suffix"] or "", channel=m["channel"], jitter_iteration=int(m["onset"]),
            jitter_std=float(m["std"]),
        )
    m = _BASELINE_RE.match(name)
    if m:
        return RunName(kind="clean", dataset=m["dataset"], acquisition=m["acquisition"],
                       seed=int(m["seed"]), suffix=m["suffix"])
    return None


def _run_files(input_dir: Path) -> list[Path]:
    pattern = "bo_sensor_error_*_multi_objective_*.csv"
    return sorted(set(input_dir.glob(pattern)) | set(input_dir.glob(f"*/{pattern}")))


def available_suffixes(input_dir: Path) -> list[str]:
    found = {run.suffix for run in map(parse_run_name, (p.name for p in _run_files(input_dir)))
             if run is not None and run.kind == "noisy"}
    return sorted(found)


def discover_groups(
    input_dir: Path,
    *,
    acquisitions: set[str],
    error_model: str,
    suffix_for: Callable[[str, float], str],
    problems: set[str] | None = None,
    seeds: set[int] | None = None,
    jitter_stds: list[float] | None = None,
    jitter_iterations: set[int] | None = None,
    baseline_suffix: str = "",
) -> tuple[list[RunGroup], list[str]]:
    """Pair every selected noisy run with its identically seeded clean run.

    Returns the groups and the names of noisy runs that have no clean run.
    """
    baselines: dict[tuple[str, str, int], Path] = {}
    noisy: dict[tuple[str, str, int], list[tuple[float, int, Path]]] = {}
    for path in _run_files(input_dir):
        run = parse_run_name(path.name)
        if run is None or run.acquisition not in acquisitions:
            continue
        if problems is not None and run.dataset not in problems:
            continue
        if seeds and run.seed not in seeds:
            continue
        key = (run.dataset, run.acquisition, run.seed)
        if run.kind == "clean":
            if run.suffix == baseline_suffix:
                baselines[key] = path
            continue
        if run.channel != error_model:
            continue
        if jitter_stds and not any(math.isclose(run.jitter_std, v) for v in jitter_stds):
            continue
        if jitter_iterations and run.jitter_iteration not in jitter_iterations:
            continue
        # Exact match: the standard files carry no suffix, so asking for "" must
        # not also sweep up every variant arm stored next to them.
        if run.suffix != suffix_for(run.channel, run.jitter_std):
            continue
        noisy.setdefault(key, []).append((run.jitter_std, run.jitter_iteration, path))

    groups: list[RunGroup] = []
    orphans: list[str] = []
    for key in sorted(noisy):
        if key not in baselines:
            orphans.extend(p.name for *_, p in noisy[key])
            continue
        groups.append(RunGroup(*key, baseline=baselines[key], noisy=tuple(sorted(noisy[key]))))
    return groups, orphans


def cross_corr_suffix(rho: float, error_model: str, jitter_std: float) -> str:
    """The filename part the synthetic driver writes for --error-cross-corr RHO.

    Asked of the driver itself -- its parser and _variant_suffix -- rather than
    spelled again here, so the replay cannot drift from the names the runs were
    written under.
    """
    if float(rho) == 0.0:
        # A setting at its default adds no part to the name: rho = 0 is the standard files.
        return ""
    import contextlib
    import io

    import bo_synthetic_error_simulation as drv

    try:
        # The driver's parser reports an unknown flag on stderr under THIS script's
        # name, which reads as if the replay had rejected its own option; the
        # ValueError below says what actually happened.
        with contextlib.redirect_stderr(io.StringIO()):
            dargs = drv.parse_args(["--multi-objective", f"--error-cross-corr={float(rho)!r}"])
    except SystemExit as exc:
        raise ValueError(
            "The synthetic driver does not accept --error-cross-corr, so the variant's "
            "filename part cannot be derived from it. Spell that part by hand with "
            "--variant-suffix."
        ) from exc
    error_bias = float(jitter_std) if dargs.error_bias_mode == "scaled" else float(dargs.error_bias)
    spike_std = float(jitter_std) if dargs.error_spike_std_mode == "scaled" else float(dargs.error_spike_std)
    suffix = drv._variant_suffix(dargs, error_model, error_bias, spike_std)
    if not suffix:
        raise ValueError(
            f"The driver accepts --error-cross-corr but its _variant_suffix does not name "
            f"it (rho = {rho:g}), so its runs would overwrite the standard ones. Fix the "
            "driver before replaying this arm."
        )
    return suffix


def make_suffix_resolver(args: argparse.Namespace) -> Callable[[str, float], str]:
    if args.error_cross_corr is None:
        fixed = args.variant_suffix
        return lambda error_model, jitter_std: fixed
    cache: dict[tuple[str, float], str] = {}

    def resolve(error_model: str, jitter_std: float) -> str:
        key = (error_model, float(jitter_std))
        if key not in cache:
            cache[key] = cross_corr_suffix(args.error_cross_corr, error_model, jitter_std)
        return cache[key]

    return resolve


def load_run_stats(input_dir: Path, mo_stats_path: Path | None = None) -> dict:
    """The per-problem constants the runs were made with."""
    meta_path = input_dir / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stats = meta.get("landscape_stats") or {}
        if stats and all("ref_point" in v for v in stats.values()):
            return stats
    return mob.load_mo_stats(mo_stats_path)


# ---------------------------------------------------------------------------
# Fronts, hypervolume, trimming
# ---------------------------------------------------------------------------


def non_dominated(Y: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows under BoTorch's defaults.

    maximize=True and deduplicate=True are what run_simulation calls, and the
    observed rule reproduces the logged hypervolume only if this does the same.
    """
    import torch
    from botorch.utils.multi_objective import is_non_dominated

    Y = np.asarray(Y, dtype=float)
    if Y.shape[0] == 0:
        return np.zeros(0, dtype=int)
    mask = is_non_dominated(torch.as_tensor(Y, dtype=torch.double))
    return np.flatnonzero(mask.cpu().numpy())


def hypervolume(Y: np.ndarray, ref_point: np.ndarray) -> float:
    """Mirror of bo_sensor_error_simulation._compute_hypervolume."""
    import torch
    from botorch.utils.multi_objective import is_non_dominated
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    Y = np.asarray(Y, dtype=float)
    if Y.shape[0] == 0:
        return 0.0
    Y_t = torch.as_tensor(Y, dtype=torch.double)
    pareto = Y_t[is_non_dominated(Y_t)]
    hv = Hypervolume(ref_point=torch.as_tensor(np.asarray(ref_point, dtype=float), dtype=torch.double))
    return float(hv.compute(pareto))


def strictly_dominated(Y: np.ndarray) -> np.ndarray:
    """Row i is dominated if some row is at least as good everywhere and better somewhere."""
    Y = np.asarray(Y, dtype=float)
    geq = (Y[:, None, :] >= Y[None, :, :]).all(axis=-1)  # geq[j, i]: j >= i everywhere
    gt = (Y[:, None, :] > Y[None, :, :]).any(axis=-1)
    return (geq & gt).any(axis=0)


def trim_to_cardinality(Y_est: np.ndarray, members: np.ndarray, k: int,
                        ref_point: np.ndarray) -> np.ndarray:
    """Greedily drop the member with the smallest hypervolume contribution until k remain.

    Contributions are exclusive (HV of the set minus HV without the member) and
    live in the estimate space the rule chose the set in -- the true values are
    what is being predicted, so they must not steer the trimming.
    """
    if k < 1:
        raise ValueError("cannot trim a front to fewer than one design")
    ref_point = np.asarray(ref_point, dtype=float)
    kept = [int(i) for i in members]
    while len(kept) > k:
        points = np.asarray(Y_est, dtype=float)[kept]
        total = hypervolume(points, ref_point)
        contribution = np.array(
            [total - hypervolume(np.delete(points, j, axis=0), ref_point) for j in range(len(kept))]
        )
        # A member that adds nothing can come out at +-1e-13 from rounding, and
        # that rounding must not decide which of several zero members goes.
        contribution[np.abs(contribution) <= _CONTRIBUTION_EPS * max(abs(total), 1.0)] = 0.0
        # Members outside the reference box all contribute zero. Among those, drop
        # the one farthest from entering the box (its worst margin over the
        # reference point), then the lowest index, so the result is deterministic.
        margin = (points - ref_point).min(axis=1)
        order = np.lexsort((np.asarray(kept), margin, contribution))
        kept.pop(int(order[0]))
    return np.asarray(sorted(kept), dtype=int)


# ---------------------------------------------------------------------------
# The hidden decision-maker
# ---------------------------------------------------------------------------


def dirichlet_weights(seed: int, num_objectives: int, n_weights: int = N_WEIGHTS) -> np.ndarray:
    """Weight vectors, identical for every rule, run and problem with this seed and M.

    Common random numbers: a difference between two rules, or between a noisy run
    and its clean twin, is then never a difference between the people asked.
    """
    rng = np.random.default_rng(np.random.SeedSequence([WEIGHT_SEED, int(seed), int(num_objectives)]))
    return rng.dirichlet(np.ones(int(num_objectives)), size=int(n_weights))


def utopia_point(name: str, stats_entry: dict, log2_n: int = UTOPIA_LOG2_SAMPLES) -> np.ndarray:
    """Per-objective maximum of the standardised objectives over a Sobol sample of the box.

    The decision-maker's anchor must be a property of the problem, like the
    reference point: the true values are hidden from it and the observed ones are
    what is on trial. The reference point itself is a poor anchor -- it sits 36 SD
    below the front on zdt1's f0 and 6 SD on f1, so the weights would mean little.
    Objectives are standardised, so a unit of shortfall is one landscape SD on
    every objective and the weights need no rescaling.
    """
    X = mob.sobol_sample(name, log2_n=log2_n, seed=mob.STATS_SEED)
    mean = np.asarray(stats_entry["mean"], dtype=float)
    std = np.asarray(stats_entry["std"], dtype=float)
    return ((mob.evaluate(name, X) - mean) / std).max(axis=0)


def chebyshev_utility(Y: np.ndarray, weights: np.ndarray, utopia: np.ndarray) -> np.ndarray:
    """Weighted Chebyshev utility, (n_weights, n_designs); higher is better."""
    shortfall = np.asarray(utopia, dtype=float)[None, None, :] - np.asarray(Y, dtype=float)[None, :, :]
    return -(np.asarray(weights, dtype=float)[:, None, :] * shortfall).max(axis=-1)


def decision_maker(Y_est: np.ndarray, Y_true: np.ndarray, members: np.ndarray,
                   weights: np.ndarray, utopia: np.ndarray) -> tuple[float, float]:
    """Mean true regret and mean true shortfall of the member each weight vector picks."""
    members = np.asarray(members, dtype=int)
    if members.size == 0:
        return float("nan"), float("nan")
    estimated = chebyshev_utility(np.asarray(Y_est)[members], weights, utopia)
    pick = members[np.argmax(estimated, axis=1)]
    true_all = chebyshev_utility(Y_true, weights, utopia)
    chosen = true_all[np.arange(len(weights)), pick]
    # Against the best EVALUATED design: the rule can only ship what was tried.
    regret = true_all.max(axis=1) - chosen
    return float(regret.mean()), float((-chosen).mean())


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def load_run(path: Path, num_objectives: int) -> dict:
    frame = pd.read_csv(path)
    param_columns = str(frame["param_columns"].iloc[0]).split(",")
    objectives = [f"f{m}" for m in range(num_objectives)]
    needed = (param_columns + [f"objective_observed_{c}" for c in objectives]
              + [f"objective_true_{c}" for c in objectives] + ["inference_value_true", "y_opt"])
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    failed = frame["acq_opt_failed"].astype(str).str.lower().eq("true") if "acq_opt_failed" in frame else None
    return {
        "X": frame[param_columns].to_numpy(dtype=float),
        "Y_observed": frame[[f"objective_observed_{c}" for c in objectives]].to_numpy(dtype=float),
        "Y_true": frame[[f"objective_true_{c}" for c in objectives]].to_numpy(dtype=float),
        "logged_inference": float(frame["inference_value_true"].iloc[-1]),
        "y_opt": float(frame["y_opt"].iloc[0]),
        "n_iterations": int(len(frame)),
        "error_model": str(frame["error_model"].iloc[0]),
        "jitter_std": float(frame["jitter_std"].iloc[0]),
        "jitter_iteration": int(frame["jitter_iteration"].iloc[0]),
        "acq_opt_failures": int(failed.sum()) if failed is not None else 0,
    }


def refit_posteriors(X: np.ndarray, Y: np.ndarray, low: np.ndarray, high: np.ndarray,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and latent SD at the evaluated designs, one default GP per objective."""
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_X = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.double)
    # The loop's own bounds tensor (sim.Bounds.tensor): Normalize maps the design
    # box, not the range the data happen to cover.
    box = torch.stack([torch.from_numpy(np.asarray(low, dtype=float)),
                       torch.from_numpy(np.asarray(high, dtype=float))])
    # Seeded because fit_gpytorch_mll resamples hyperparameters when a fit fails.
    torch.manual_seed(int(seed))
    means, sds = [], []
    for m in range(Y.shape[1]):
        # One model and one marginal likelihood per objective. The loop fits a
        # ModelListGP under SumMarginalLogLikelihood, which BoTorch fits model by
        # model, so this is the same fit.
        gp = SingleTaskGP(
            train_X,
            torch.as_tensor(np.asarray(Y, dtype=float)[:, [m]], dtype=torch.double),
            input_transform=Normalize(d=train_X.shape[-1], bounds=box),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(train_X)
        means.append(posterior.mean.reshape(-1).cpu().numpy())
        # Latent SD, without observation noise: the uncertainty about a design's
        # true value, which is what shipping it risks.
        sds.append(posterior.variance.clamp_min(0.0).sqrt().reshape(-1).cpu().numpy())
    return np.column_stack(means), np.column_stack(sds)


def replay_run(run: dict, *, ref_point: np.ndarray, utopia: np.ndarray, weights: np.ndarray,
               low: np.ndarray, high: np.ndarray, seed: int,
               lower_bound_sds: float = LOWER_BOUND_SDS) -> list[dict]:
    """Every rule's deployed set and scores for one logged run."""
    Y_obs, Y_true = run["Y_observed"], run["Y_true"]
    k = int(len(non_dominated(Y_obs)))
    started = time.perf_counter()
    mean, sd = refit_posteriors(run["X"], Y_obs, low, high, seed)
    refit_seconds = time.perf_counter() - started
    estimates = {
        "observed": Y_obs,
        "posterior_mean": mean,
        "lower_bound": mean - float(lower_bound_sds) * sd,
    }
    dominated = strictly_dominated(Y_true)
    rows = []
    for rule in RULES:
        Y_est = estimates[rule]
        front = non_dominated(Y_est)
        deployed = trim_to_cardinality(Y_est, front, k, ref_point)
        hv_true = hypervolume(Y_true[deployed], ref_point)
        regret, shortfall = decision_maker(Y_est, Y_true, deployed, weights, utopia)
        rows.append({
            "rule": rule,
            "k_observed": k,
            "n_front": int(len(front)),
            "n_deployed": int(len(deployed)),
            "hv_true": hv_true,
            "dm_regret": regret,
            "dm_shortfall": shortfall,
            "truly_dominated_share": float(dominated[deployed].mean()),
            "deployed_iterations": ";".join(str(int(i) + 1) for i in deployed),
            "refit_seconds": refit_seconds,
        })
    return rows


def build_context(stats: dict, problems: list[str], *, n_weights: int = N_WEIGHTS,
                  lower_bound_sds: float = LOWER_BOUND_SDS,
                  check_tolerance: float = CHECK_TOLERANCE,
                  utopia_log2: int = UTOPIA_LOG2_SAMPLES, cache_dir: Path | None = None,
                  resume: bool = False, tag: str = "") -> dict:
    """Everything a worker needs, picklable for Windows' spawn."""
    unknown = [p for p in problems if p not in stats]
    if unknown:
        raise ValueError(f"no statistics for {unknown}")
    return {
        "stats": {p: stats[p] for p in problems},
        "utopia": {p: utopia_point(p, stats[p], utopia_log2).tolist() for p in problems},
        "utopia_log2": int(utopia_log2),
        "n_weights": int(n_weights),
        "lower_bound_sds": float(lower_bound_sds),
        "check_tolerance": float(check_tolerance),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "resume": bool(resume),
        "tag": tag,
    }


def replay_group(group: RunGroup, context: dict) -> pd.DataFrame:
    """Replay one clean run and its noisy twins; refuses anything it cannot reproduce."""
    spec = mob.MO_BENCHMARKS[group.dataset]
    entry = context["stats"][group.dataset]
    ref_point = np.asarray(entry["ref_point"], dtype=float)
    max_hv = float(entry["max_hv"])
    low, high = mob.bounds(group.dataset)
    weights = dirichlet_weights(group.seed, spec.num_objectives, context["n_weights"])
    utopia = np.asarray(context["utopia"][group.dataset], dtype=float)

    rows: list[dict] = []
    clean_rows = None
    for condition, expected, path in [("clean", None, group.baseline)] + [
        ("noisy", (std, onset), p) for std, onset, p in group.noisy
    ]:
        run = load_run(path, spec.num_objectives)
        if condition == "clean":
            clean_rows = run["n_iterations"]
            if run["error_model"] != "none":
                raise ValueError(f"{path.name}: a clean run must carry error_model 'none', not {run['error_model']!r}")
        else:
            if run["n_iterations"] != clean_rows:
                raise ValueError(
                    f"{path.name}: {run['n_iterations']} iterations against {clean_rows} in its "
                    "clean run; a truncated run cannot be paired (finish it with the driver's --resume)."
                )
            if not (math.isclose(run["jitter_std"], expected[0]) and run["jitter_iteration"] == expected[1]):
                raise ValueError(f"{path.name}: the file's condition columns disagree with its name")
        if not math.isclose(run["y_opt"], max_hv, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{path.name}: y_opt {run['y_opt']!r} is not the statistics' max_hv {max_hv!r}; "
                "these are not the constants the run was made with."
            )
        try:
            per_rule = replay_run(run, ref_point=ref_point, utopia=utopia, weights=weights, low=low,
                                  high=high, seed=group.seed, lower_bound_sds=context["lower_bound_sds"])
        except Exception as exc:
            raise RuntimeError(f"{path.name}: replay failed: {exc}") from exc
        observed = next(r for r in per_rule if r["rule"] == REFERENCE_RULE)
        error = abs(observed["hv_true"] - run["logged_inference"])
        if error > context["check_tolerance"] * max(1.0, abs(run["logged_inference"])):
            raise ValueError(
                f"{path.name}: the observed non-dominated set has true hypervolume "
                f"{observed['hv_true']!r}, but the run logged inference_value_true = "
                f"{run['logged_inference']!r}. The replay is not reading what the simulator "
                "deployed (reference point, column order or front convention); refusing to "
                "score rules against it."
            )
        for r in per_rule:
            r.update({
                "dataset": group.dataset,
                "acquisition": group.acquisition,
                "seed": group.seed,
                "condition": condition,
                "error_model": run["error_model"],
                "jitter_std": run["jitter_std"] if condition == "noisy" else 0.0,
                # A clean run has no onset; it pairs with every onset downstream.
                "jitter_iteration": run["jitter_iteration"] if condition == "noisy" else -1,
                "n_iterations": run["n_iterations"],
                "max_hv": max_hv,
                "observed_check_abs_error": error,
                "acq_opt_failures": run["acq_opt_failures"],
                "source_file": path.name,
                "lower_bound_sds": context["lower_bound_sds"],
                "n_weights": context["n_weights"],
                "utopia_log2": context["utopia_log2"],
            })
            rows.append(r)
    return pd.DataFrame(rows)


def replay_group_cached(group: RunGroup, context: dict) -> pd.DataFrame:
    cache = None
    if context.get("cache_dir"):
        cache = Path(context["cache_dir"]) / f"{group.dataset}_{group.acquisition}_seed{group.seed}{context['tag']}.csv"
        if context.get("resume") and cache.exists():
            cached = pd.read_csv(cache)
            expected = {group.baseline.name} | {p.name for *_, p in group.noisy}
            same_files = set(cached["source_file"]) == expected
            same_settings = all(
                key in cached and np.allclose(cached[key].astype(float), float(context[key]))
                for key in ("lower_bound_sds", "n_weights", "utopia_log2")
            )
            if same_files and same_settings:
                return cached
    frame = replay_group(group, context)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(cache)
    return frame


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def run_groups(groups: list[RunGroup], context: dict, n_jobs: int) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    step = max(1, len(groups) // 20)
    started = time.perf_counter()

    def progress(done: int) -> None:
        if done % step == 0 or done == len(groups):
            print(f"  {done}/{len(groups)} groups, {time.perf_counter() - started:,.0f} s", flush=True)

    if n_jobs == 1:
        for i, group in enumerate(groups, 1):
            frames.append(replay_group_cached(group, context))
            progress(i)
        return frames
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as pool:
        futures = [pool.submit(replay_group_cached, g, context) for g in groups]
        for i, future in enumerate(as_completed(futures), 1):
            frames.append(future.result())
            progress(i)
    return frames


# ---------------------------------------------------------------------------
# Scoring against the standard rule
# ---------------------------------------------------------------------------


def load_screen(screen_dir: Path) -> tuple[dict[str, float], pd.DataFrame]:
    """Floor gap and headroom, computed by analyse_boba_mo.py's own functions."""
    import analyse_boba_mo as amo

    paired = amo.ab.load_paired(screen_dir)
    gain = amo.achievable_gain(screen_dir, paired)
    return {str(d): float(g) for d, g in zip(gain["dataset"], gain["gain"])}, amo.headroom(paired)


def paired_cells(runs: pd.DataFrame, gap: dict[str, float]) -> pd.DataFrame:
    """Per (problem, acquisition, seed, magnitude, onset, rule, response): the four arms."""
    runs = runs.copy()
    runs["hv_loss"] = (runs["max_hv"] - runs["hv_true"]) / runs["dataset"].map(gap)
    if runs["hv_loss"].isna().any():
        raise ValueError(f"no floor gap for {sorted(runs.loc[runs.hv_loss.isna(), 'dataset'].unique())}")
    unit = ["dataset", "acquisition", "seed"]
    cell = unit + ["error_model", "jitter_std", "jitter_iteration"]
    long = runs.melt(id_vars=cell + ["condition", "rule"], value_vars=list(RESPONSES.values()),
                     var_name="metric", value_name="value")
    noisy = long[long["condition"] == "noisy"].drop(columns="condition")
    clean = long[long["condition"] == "clean"][unit + ["rule", "metric", "value"]]

    def arm(frame: pd.DataFrame, rule: str, name: str) -> pd.DataFrame:
        return frame[frame["rule"] == rule].drop(columns="rule").rename(columns={"value": name})

    ref_noisy = arm(noisy, REFERENCE_RULE, "ref_noisy")
    # The standard rule's clean run is the yardstick for every rule and onset.
    ref_clean = arm(clean, REFERENCE_RULE, "ref_clean")
    frames = []
    for rule in RULES:
        if rule == REFERENCE_RULE:
            continue
        m = (
            ref_noisy.merge(arm(noisy, rule, "trt_noisy"), on=cell + ["metric"], validate="one_to_one")
            .merge(ref_clean, on=unit + ["metric"], validate="many_to_one")
            .merge(arm(clean, rule, "trt_clean"), on=unit + ["metric"], validate="many_to_one")
        )
        if len(m) != len(ref_noisy):
            raise ValueError(f"{rule}: {len(ref_noisy) - len(m)} noisy cells lost their pairing")
        frames.append(m.assign(rule=rule))
    out = pd.concat(frames, ignore_index=True)
    label = {column: name for name, column in RESPONSES.items()}
    return out.assign(response=out["metric"].map(label)).drop(columns="metric")


def summarise(block: pd.DataFrame, rng: np.random.Generator) -> dict:
    """analyse_boba_adaptations.summarise, on problems: ratios of problem means, problem bootstrap.

    Kept as a copy rather than an import so a change there cannot silently move
    these numbers; tests/test_mo_front.py checks the two agree.
    """
    from scipy.stats import wilcoxon

    per = block.groupby("dataset")[["ref_noisy", "ref_clean", "trt_noisy", "trt_clean"]].mean()
    cost_l = (per.ref_noisy - per.ref_clean).to_numpy()
    gain_l = (per.ref_noisy - per.trt_noisy).to_numpy()
    price_l = (per.trt_clean - per.ref_clean).to_numpy()
    cost, gain, price = cost_l.mean(), gain_l.mean(), price_l.mean()
    n = len(per)
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, n)
        c = cost_l[idx].mean()
        draws.append(gain_l[idx].mean() / c if c > 0 else np.nan)
    draws = np.array(draws)
    ok = np.isfinite(draws)
    if np.allclose(gain_l, 0.0):
        p = 1.0
    else:
        try:
            p = float(wilcoxon(gain_l).pvalue)
        except ValueError:
            p = 1.0
    return {
        "n_problems": int(n), "n_cells": int(len(block)),
        "cost": float(cost), "gain": float(gain), "price": float(price),
        "recovered": float(gain / cost) if cost > 0 else np.nan,
        "recovered_lo": float(np.percentile(draws[ok], 2.5)) if ok.sum() >= 100 else np.nan,
        "recovered_hi": float(np.percentile(draws[ok], 97.5)) if ok.sum() >= 100 else np.nan,
        "wilcoxon_p": p,
    }


CELL = ["error_model", "jitter_std", "jitter_iteration", "response", "rule"]


def recovery_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, block in cells.groupby(CELL, sort=True):
        # A fresh generator per cell, so a cell's interval does not depend on
        # which other cells happen to be in the run.
        rows.append({**dict(zip(CELL, keys)), **summarise(block, np.random.default_rng(BOOTSTRAP_SEED))})
    return pd.DataFrame(rows)


def per_problem_table(cells: pd.DataFrame) -> pd.DataFrame:
    per = cells.groupby(["dataset"] + CELL)[["ref_noisy", "ref_clean", "trt_noisy", "trt_clean"]].mean().reset_index()
    per["cost"] = per.ref_noisy - per.ref_clean
    per["gain"] = per.ref_noisy - per.trt_noisy
    per["price"] = per.trt_clean - per.ref_clean
    per["recovered"] = np.where(per.cost > 0, per.gain / per.cost.where(per.cost > 0), np.nan)
    return per


def front_table(runs: pd.DataFrame) -> pd.DataFrame:
    """What each rule ships: sizes and the share that is truly dominated, problem means."""
    runs = runs.assign(front_smaller=(runs["n_front"] < runs["k_observed"]).astype(float))
    cols = ["k_observed", "n_front", "n_deployed", "front_smaller", "truly_dominated_share"]
    keys = ["condition", "error_model", "jitter_std", "jitter_iteration", "rule"]
    per = runs.groupby(keys + ["dataset"])[cols].mean()
    return per.groupby(level=list(range(len(keys)))).mean().reset_index()


def _onset(it: int) -> str:
    return "clean" if int(it) < 0 else f"trial {int(it) + 1}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"replay_mo_front: {exc}") from None
    started = time.perf_counter()
    input_dir = args.input_dir
    screen_dir = args.screen_dir or input_dir
    output_dir = args.output_dir or (input_dir / "analysis" / "front_replay")

    import analyse_boba_mo as amo

    stats = load_run_stats(input_dir, args.mo_stats_path)
    gap, hr = load_screen(screen_dir)
    if args.admissible:
        admissible = _names(args.admissible)
        unknown = [p for p in admissible if p not in gap]
        if unknown:
            raise SystemExit(f"replay_mo_front: no floor gap in {screen_dir} for {unknown}")
    elif args.include_inadmissible:
        admissible = sorted(gap)
    else:
        admissible = sorted(hr.loc[hr["headroom"] >= amo.HEADROOM_MIN, "dataset"])
    requested = _names(args.problems)
    if requested:
        dropped = sorted(set(requested) - set(admissible))
        if dropped:
            print(f"Not admissible, skipped: {dropped} (headroom < {amo.HEADROOM_MIN})")
        problems = [p for p in requested if p in admissible]
    else:
        problems = list(admissible)
    for p in problems:
        if not gap.get(p, 0.0) > 0:
            raise SystemExit(f"replay_mo_front: {p} has no positive floor gap ({gap.get(p)})")
        if not math.isclose(float(stats[p]["max_hv"]), float(json.loads(
                (screen_dir / "run_metadata.json").read_text(encoding="utf-8"))["landscape_stats"][p]["max_hv"]),
                rel_tol=1e-12):
            raise SystemExit(f"replay_mo_front: {p}: the input and screen directories disagree on max_hv")

    try:
        suffix_for = make_suffix_resolver(args)
        groups, orphans = discover_groups(
            input_dir,
            acquisitions=set(_names(args.acquisitions)),
            error_model=args.error_model,
            suffix_for=suffix_for,
            problems=set(problems),
            seeds=set(_ints(args.seeds)) or None,
            jitter_stds=_floats(args.jitter_stds) or None,
            jitter_iterations=set(_ints(args.jitter_iterations)) or None,
            baseline_suffix=args.baseline_suffix,
        )
    except ValueError as exc:
        raise SystemExit(f"replay_mo_front: {exc}") from None
    if orphans:
        raise SystemExit(
            f"replay_mo_front: {len(orphans)} noisy runs have no clean run to pair with, e.g. "
            f"{orphans[:3]}. Finish the arm or restrict --seeds."
        )
    if not groups:
        raise SystemExit(
            f"replay_mo_front: no paired runs under {input_dir} for this selection. "
            f"Variant suffixes present: {available_suffixes(input_dir) or ['(none)']}"
        )
    suffixes = sorted({parse_run_name(p.name).suffix for g in groups for *_, p in g.noisy})
    variant = suffixes[0] if len(suffixes) == 1 else "_mixed"
    tag = "" if (args.error_model == "gaussian" and not variant) else f"_{args.error_model}{variant}"

    n_noisy = sum(len(g.noisy) for g in groups)
    print(f"\nReplaying {len(groups)} clean runs and {n_noisy} noisy runs "
          f"({', '.join(problems)}; variant {variant or '(standard)'}) with {args.n_jobs} worker(s).")
    if args.dry_run:
        return

    context = build_context(
        stats, sorted({g.dataset for g in groups}), n_weights=args.n_weights,
        lower_bound_sds=args.lower_bound_sds, check_tolerance=args.check_tolerance,
        cache_dir=output_dir / "runs", resume=args.resume, tag=tag,
    )
    runs = pd.concat(run_groups(groups, context, args.n_jobs), ignore_index=True)
    runs = runs.sort_values(["dataset", "acquisition", "seed", "condition", "jitter_std",
                             "jitter_iteration", "rule"]).reset_index(drop=True)
    runs["hv_loss"] = (runs["max_hv"] - runs["hv_true"]) / runs["dataset"].map(gap)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / f"front_replay_runs{tag}.csv", index=False)

    n_runs = int((runs["rule"] == REFERENCE_RULE).sum())
    worst = float(runs["observed_check_abs_error"].max())
    print("\n=== CHECK: observed-ND replay vs logged inference_value_true ===")
    print(f"  runs = {n_runs}   max |difference| = {worst:.3e}   PASS")

    fronts = front_table(runs)
    fronts.to_csv(output_dir / f"front_replay_fronts{tag}.csv", index=False)
    print("\n=== WHAT EACH RULE SHIPS (problem means) ===")
    show = fronts.assign(onset=fronts["jitter_iteration"].map(_onset))
    print(show[["condition", "jitter_std", "onset", "rule", "k_observed", "n_front", "n_deployed",
                "front_smaller", "truly_dominated_share"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    cells = paired_cells(runs, gap)
    recovery = recovery_table(cells)
    per_problem = per_problem_table(cells)
    recovery.to_csv(output_dir / f"front_replay_recovery{tag}.csv", index=False)
    per_problem.to_csv(output_dir / f"front_replay_per_problem{tag}.csv", index=False)

    print("\n=== RECOVERY against the standard observed-ND rule "
          "(ratios of problem means; 95% problem bootstrap) ===")
    for _, r in recovery.iterrows():
        ci = (f"[{r.recovered_lo:+.0%}, {r.recovered_hi:+.0%}]"
              if np.isfinite(r.recovered_lo) else "[n/a]")
        rec = f"{r.recovered:+.0%}" if np.isfinite(r.recovered) else "n/a"
        print(f"  sigma {r.jitter_std:>4g}  {_onset(r.jitter_iteration):<8s}  {r.response:<8s}  "
              f"{r.rule:<14s}  cost {r.cost:7.4f}  gain {r.gain:+8.4f}  price {r.price:+8.4f}  "
              f"recovered {rec:>6s} {ci}  ({r.n_problems} problems, {r.n_cells} cells)")

    import botorch

    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "screen_dir": str(screen_dir),
        "problems": problems,
        "admissible": admissible,
        "headroom_min": amo.HEADROOM_MIN,
        "floor_gap": {p: gap[p] for p in problems},
        "utopia": context["utopia"],
        "variant_suffix": variant,
        "weight_seed": WEIGHT_SEED,
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "n_groups": len(groups),
        "n_runs": n_runs,
        "max_observed_check_abs_error": worst,
        "botorch_version": botorch.__version__,
        "runtime_sec": time.perf_counter() - started,
    }
    (output_dir / f"front_replay_metadata{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {output_dir} in {meta['runtime_sec']:,.0f} s")


if __name__ == "__main__":
    main()
