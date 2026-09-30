"""How many of T human trials should be spent identifying rather than searching?

A study with a fixed budget of T rated trials currently spends all T searching
and ships whichever design was rated highest. scripts/decompose_regret.py shows
that under a noisy rater most of the error-caused regret of the shipped design is
SELECTION loss -- the run visited something better than what it ships -- and no
acquisition function can reach that term. Spending the last k trials on a
comparative sitting among the top k visited designs reaches it, at the price of k
fewer search trials. This script derives k from quantities the experimenter has
at trial T - k, and then scores the derived k against what actually happened.

The rule
--------
Write mu_i, sigma_i for the loop GP's posterior latent mean and SD at the
distinct designs visited in the first T - k trials, and order the designs by the
ship rule lcb_i = mu_i - beta sigma_i. A k-tournament shows the top k candidates
once each in one sitting and ships the best-rated of them, so with s the
idiosyncratic rating SD of that sitting,

    f_i  ~  N(mu_i, sigma_i^2)          the latent value, as the GP sees it
    y_i  =  f_i + N(0, s^2)             the one look the sitting buys
    U(k) =  E[ f_{argmax_i y_i} ]       the value of what gets shipped

U(k) is the identification term. It is NOT monotone in k: a larger k is more
likely to contain the best visited design (coverage) and more likely to let a
worse one win the sitting on a lucky look (discrimination). U(k) is evaluated by
Monte Carlo over the posterior, which costs nothing next to the GP fit.

The search term is what the k forgone trials would have found. At trial T - k the
experimenter does not know, so the rule uses the run's OWN recent rate of
improvement: the mean gain in the best observed rating per trial over the last
--window trials, floored at zero, extrapolated over k. Call it Shat(k).

    k_hat = argmax_k  U(k) - Shat(k),   k in {0} u --k-grid

k = 0 is "no tournament": ship the lcb design at T. Every quantity above is
available to a real experimenter. The true objective is used ONLY to score the
result afterwards, never to choose k.

When the rule decides (--decision)
----------------------------------
'fixed' (the default, the decide-once rule) scores every option under the one
posterior at n0 = T - max(--k-grid), the last trial at which all of them are
still open. With the published grid (max k = 30, T = 50) that is trial 20, so
at the late onset (error from trial 21, the simulator corrupting trial t only
when t > onset) the rule decides on twenty exact ratings, before any error
exists, and makes the same choice at every magnitude. Capping the grid
(--k-grid ...,25) moves n0 past the onset but still commits once, early.

'sequential' decides at each option's own last moment. At n_j = T - k_j, for
k_j from the largest k down, it scores the options still open (ship by the
posterior mean or lcb at T, or a sitting of any k <= k_j) under the one
posterior at n_j, exactly as 'fixed' does at n0, with the sitting SD read from
that posterior. If the sitting that expires now (k_j) scores best it commits to
it; otherwise it keeps searching to the next decision point. At the smallest k
the choice is final. Every comparison is still within one posterior, which is
what the single-posterior design was for, and every commitment uses every
rating observed by then, which the replay (whose sitting of k starts at T - k)
also uses. By-onset scores of either rule, on all seeds and on each seed half
of heldout_remedies.py (7-11, 12-16), are written to <name>_by_onset<suffix>.csv.

Scoring
-------
The realised regret of every (run, k) comes from the k-sweep of
scripts/replay_end_of_study.py, joined on the run's file and k, so the rule is
never scored on its own simulation. Five policies are compared, in units of the
landscape's achievable improvement:

    standard        all T trials searching, ship the best rating (ref_noisy)
    k = 0           all T trials searching, ship the lcb design
    fixed k         the best single k for the whole suite -- what a practitioner
                    would do after reading a paper's table
    derived k_hat   this rule, per run
    oracle k        the best k for this run, which nothing can beat

The published outputs read both replays and the whole grid (paper/COMMANDS.md
has the replays): with DIRS = output-boba/analysis/end_of_study_ksweep,
output-boba/analysis/end_of_study_kwide (comma-joined, no space) and
GRID = 2,3,5,8,12,16,20,25,30,

    python scripts/budget_split.py --workers 5 --ksweep-dir DIRS --k-grid GRID
    python scripts/budget_split.py --workers 5 --ksweep-dir DIRS --k-grid GRID --rho 0.5 --output-suffix _rho0.5

and the two recomputations with a decision point after the late onset:

    python scripts/budget_split.py --workers 5 --ksweep-dir DIRS --k-grid GRID --decision sequential --output-suffix _decision_sequential
    python scripts/budget_split.py --workers 5 --ksweep-dir DIRS --k-grid 2,3,5,8,12,16,20,25 --output-suffix _kmax25

(each also with --rho 0.5 and the suffix + _rho0.5). --rate-source truth is a
clairvoyant diagnostic, never the rule: the search term reads the gain in the
best true value instead of the best rating, which prices what rating error in
the search term costs (suffixes _kmax25_truerate, _decision_sequential_truerate).
--twin clean prices a rule: it derives the rule on each run's clean twin (exact
ratings) and charges it the clean twin's replayed regret of what it chose, less
the clean standard process's (price = trt_clean - ref_clean, the estimand of
analyse_boba_adaptations.py), beside the same price for every fixed k and the
no-trial rules (budget_split_price<suffix>.csv; suffixes _clean,
_decision_sequential_clean):

    python scripts/budget_split.py --workers 5 --ksweep-dir DIRS --k-grid GRID --twin clean --output-suffix _clean

The worker count does not change any output: every run's Monte Carlo stream and
GP fit are seeded by the run. --summary-only rescores an existing _derived file
without refitting. The defaults (one directory, grid 2..12) are a smaller study,
not the paper's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402
# The replay owns the GP fit, the candidate ordering and the sitting's noise
# model; importing them keeps the rule and the thing it is scored against from
# drifting apart.
import replay_end_of_study as eos  # noqa: E402

MC_DRAWS = 4000
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260916
OUTPUT_NAME = "budget_split"
DECISIONS = ("fixed", "sequential")
# What the search term's rate of improvement is read from: the ratings (the rule)
# or the true values of the same visited designs (a clairvoyant diagnostic).
RATE_SOURCES = ("rating", "truth")
# The per-run column that holds the onset (0-based, error on trials t > onset).
ONSET_COLUMN = "jitter_iteration"
# The paired comparisons the by-onset table reports beside each policy's gain:
# k = 8 and k = 12 (the fixed k chosen on seeds 12-16 and on seeds 7-11), k = 16
# (the best fixed k at rho = 0.5) and the two no-trial ship rules.
COMPARATORS = ("fixed_k8", "fixed_k12", "fixed_k16", "always_pm", "always_lcb")
# The seed halves of heldout_remedies.py (TRAIN_SEEDS, TEST_SEEDS), so the
# by-onset table carries the held-out numbers the paper quotes.
SEED_SETS = (("all", None), ("7-11", (7, 8, 9, 10, 11)), ("12-16", (12, 13, 14, 15, 16)))
# Which run the rule is derived on: the noisy run (the rule, scored) or its clean
# twin (the rule's price, trt_clean - ref_clean, the estimand of
# analyse_boba_adaptations.py).
TWINS = ("noisy", "clean")
# The comparisons the price table pairs the derived rule's price against.
PRICE_COMPARATORS = ("fixed_k8", "fixed_k12", "fixed_k16", "always_pm", "always_lcb")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("output-boba"))
    p.add_argument("--ksweep-dir", type=str, default="output-boba/analysis/end_of_study_ksweep",
                   help="the k-sweep(s) whose per-run regrets the derived k is scored against; "
                        "comma-separated to pool grids run in separate directories")
    p.add_argument("--output-dir", type=Path, default=None, help="default: <input-dir>/analysis")
    p.add_argument("--k-grid", type=str, default="2,3,5,8,12")
    p.add_argument("--rho", type=float, default=1.0,
                   help="the sitting's noise as a multiple of the idiosyncratic SD. 1.0 is the "
                        "assumption-free setting: a side-by-side comparison cancels the error "
                        "shared across the designs of one sitting and leaves the idiosyncratic "
                        "part intact. 0.5 additionally assumes the comparison halves what is "
                        "left, which doubles every gain and moves the peak from k=8 to k=16, so "
                        "it is a sensitivity and not the headline.")
    p.add_argument("--lcb-beta", type=float, default=1.0)
    p.add_argument("--window", type=int, default=10,
                   help="trials of recent improvement the search term is extrapolated from")
    p.add_argument("--noise-source", choices=("gp", "truth"), default="gp",
                   help="'gp': the rule estimates s from the surrogate, as an experimenter must. "
                        "'truth': it is given the true idiosyncratic SD, to price that knowledge.")
    p.add_argument("--rate-source", choices=RATE_SOURCES, default="rating",
                   help="'rating': the search term extrapolates the run's recent gain in its best "
                        "rating, as an experimenter must. 'truth': the same gain in the best TRUE "
                        "value of the visited designs, a clairvoyant diagnostic that prices how much "
                        "rating error in the search term costs; it needs --output-suffix.")
    p.add_argument("--functions", type=str, default="all")
    p.add_argument("--acquisitions", type=str, default="logei,qnei")
    p.add_argument("--error-models", type=str, default="gaussian,bias,drift,ar1")
    p.add_argument("--seeds", type=str, default="7,8,9,10,11,12,13,14,15,16")
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    p.add_argument("--decision", choices=DECISIONS, default="fixed",
                   help="'fixed': one decision at n0 = T - max(--k-grid) (the decide-once rule). "
                        "'sequential': a decision at each T - k, committing to the sitting that "
                        "expires there only if it scores best of the options still open.")
    p.add_argument("--output-suffix", type=str, default="",
                   help="appended to every output name (budget_split_derived<suffix>.csv, "
                        "budget_split_policies<suffix>.csv, budget_split_by_onset<suffix>.csv), "
                        "so a variant never overwrites the published files; e.g. _rho0.5")
    p.add_argument("--twin", choices=TWINS, default="noisy",
                   help="'noisy': derive the rule on every noisy run and score it (the default). "
                        "'clean': derive it on each run's clean twin instead and write its price, "
                        "trt_clean - ref_clean, beside every fixed k's and the no-trial rules' "
                        "(budget_split_derived<suffix>.csv, budget_split_price<suffix>.csv); it "
                        "needs --output-suffix.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    if any(ch in args.output_suffix for ch in "/\\:"):
        p.error("--output-suffix must be a plain file-name suffix")
    if args.rate_source == "truth" and not args.output_suffix:
        # A rule that reads the true objective must never land in the published file names.
        p.error("--rate-source truth reads the true objective: name its outputs with --output-suffix")
    if args.twin == "clean" and not args.output_suffix:
        # The clean twins' choices must never overwrite the noisy runs' published file.
        p.error("--twin clean writes the rule's choices on the clean twins: name them with --output-suffix")
    return args


def output_paths(out_dir: Path, suffix: str = "") -> dict[str, Path]:
    """Every file one invocation writes; the suffix keeps variants apart."""
    return {part: out_dir / f"{OUTPUT_NAME}_{part}{suffix}.csv"
            for part in ("derived", "policies", "by_onset")}


def price_path(out_dir: Path, suffix: str) -> Path:
    """The price table a --twin clean invocation writes beside its derived file."""
    return out_dir / f"{OUTPUT_NAME}_price{suffix}.csv"


def twin_stem(file: str) -> str:
    """The (landscape, objective, acquisition, seed) stem a noisy run shares with its clean twin.

    analyse_extra_runs.index_runs keys the clean twins by the part of the name before
    "_baseline_"; a noisy run's name carries the same stem before "_jittered_".
    """
    name = Path(str(file)).stem
    for marker in ("_jittered_", "_baseline_"):
        if marker in name:
            return name.split(marker, 1)[0]
    raise ValueError(f"{file} names neither a noisy run nor a clean twin")


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def expected_shipped_value(mu: np.ndarray, sd: np.ndarray, s: float, rng: np.random.Generator,
                           draws: int = MC_DRAWS) -> float:
    """E[f(winner)] when each of the candidates is shown once and the best look wins.

    Coverage and discrimination are both in here: adding a candidate can only
    raise the best latent value on offer, and can only raise the chance that a
    worse one wins the sitting.
    """
    k = len(mu)
    if k == 0:
        return float("nan")
    if k == 1:
        return float(mu[0])
    f = rng.normal(mu, sd, size=(draws, k))
    y = f + rng.normal(0.0, max(s, 1e-12), size=(draws, k))
    return float(f[np.arange(draws), y.argmax(axis=1)].mean())


def recent_improvement_rate(observed: np.ndarray, n: int, window: int) -> float:
    """Mean gain per trial in the best rating so far, over the last `window` trials.

    This is what an experimenter at trial n can see of how fast the search is
    still improving. Floored at zero: a run that has stopped improving forgoes
    nothing by stopping, and a noisy rating can make the best-so-far look like it
    went backwards.
    """
    if n < 2:
        return 0.0
    best = np.maximum.accumulate(observed[:n])
    start = max(0, n - window - 1)
    span = (n - 1) - start
    if span <= 0:
        return 0.0
    return max(0.0, float((best[n - 1] - best[start]) / span))


# The two policies that buy no trials at all. A tournament has to beat the better
# of them, not just one of them: comparing only against the lcb pick let a
# useless sitting look attractive whenever lcb happened to rank below the
# posterior mean, which is a reason to change the ship rule, not to buy a sitting.
NO_TRIAL_POLICIES = ("pm", "lcb")


def derive_k(run, bounds: torch.Tensor, k_grid: tuple[int, ...], beta: float, window: int,
             sitting_sd: float, rng: np.random.Generator, T: int, state=None,
             rate_source: str = "rating") -> dict:
    """The chosen policy, and the rule's score for every policy it considered.

    The choice is over {ship by posterior mean, ship by lcb, tournament of k}.

    Every policy is scored under ONE posterior, the one at the decision point
    n0 = T - max(k), which is the last trial at which every option is still open.
    Scoring each k under its own T - k posterior instead looks more faithful and
    is not: two GPs fitted on different numbers of points put their maxima on
    different scales, so the comparison picked up that difference rather than the
    difference between the policies, and a sitting too noisy to tell anything
    apart still came out ahead. A policy that searches on past n0 is credited
    with the run's own recent rate of improvement for the trials it keeps, which
    is what an experimenter at n0 can forecast.

    The candidate set is therefore the top k by lcb as seen at n0, where the
    replay picks it at T - k. The rule is a decision rule, not a simulator of the
    procedure it is choosing.

    `state` is the search state at n0 when the caller has already fitted it (the
    fit is seeded by the run and n0, so refitting gives the same GP).

    `rate_source` 'truth' reads the rate from the true values of the visited
    designs (run.deployed) instead of the ratings: a clairvoyant diagnostic,
    never the rule. Nothing else changes, the fit and the sitting included.
    """
    if rate_source not in RATE_SOURCES:
        raise ValueError(f"rate_source must be one of {RATE_SOURCES}, not {rate_source!r}")
    k_max = max(k_grid)
    n0 = T - k_max
    if n0 < 2:
        return {"k_hat": None, "gp_failed": True, "scores": {}, "detail": {}}
    if state is None:
        state = eos.search_state(run, n0, bounds, beta)
    elif state.n != n0:
        raise ValueError(f"state is at trial {state.n}, the decision point is {n0}")
    if state.gp is None:
        return {"k_hat": None, "gp_failed": True, "scores": {}, "detail": {}}
    mu, sd = eos.latent_mean_sd(state.gp, run.X[:n0][state.first])
    order = np.argsort(-state.lcb)
    rate = recent_improvement_rate(run.deployed if rate_source == "truth" else run.observed, n0, window)

    scores: dict[object, float] = {}
    detail: dict[object, dict] = {}
    # Search all the way to T, then ship with no sitting.
    for key, idx in (("pm", int(np.argmax(mu))), ("lcb", int(order[0]))):
        gained = k_max * rate
        scores[key] = float(mu[idx]) + gained
        detail[key] = {"utility": float(mu[idx]), "extra_search": gained}
    # Search to T - k, then spend k on the sitting.
    for k in k_grid:
        pick = order[:k]
        utility = expected_shipped_value(mu[pick], sd[pick], sitting_sd, rng)
        gained = (k_max - k) * rate
        scores[k] = utility + gained
        detail[k] = {"utility": utility, "extra_search": gained}

    choice = max(scores, key=lambda key: scores[key])
    return {"k_hat": choice, "gp_failed": False, "scores": scores, "detail": detail}


def gp_sitting_sd(state, rho: float) -> float:
    """The sitting's look SD as the surrogate at this decision point estimates it.

    gp.likelihood.noise is the noise variance of the STANDARDISED targets,
    while gp.posterior (and so the mu, sigma the rule scores candidates with) is
    untransformed back into objective units. Multiplying by the outcome
    transform's scale puts the two on one axis. Without it the rule read a
    sitting as 1.0 to 5.6 times more precise than it is, and that is the whole
    input to the coverage/discrimination trade-off.
    """
    noise_sd_standardised = float(np.sqrt(state.gp.likelihood.noise.mean().item()))
    scale = float(state.gp.outcome_transform.stdvs.reshape(-1)[0])
    return rho * noise_sd_standardised * scale


def decision_points(k_grid: tuple[int, ...], T: int) -> list[tuple[int, int]]:
    """(k_j, n_j = T - k_j) for every sitting that can still start, largest k first."""
    return [(k, T - k) for k in sorted(set(k_grid), reverse=True) if T - k >= 2]


def derive_k_sequential(run, bounds: torch.Tensor, k_grid: tuple[int, ...], beta: float, window: int,
                        rng: np.random.Generator, T: int, sitting_sd_of, rate_source: str = "rating") -> dict:
    """The rule with a decision at each option's own last moment.

    At n_j = T - k_j (largest k first) the options still open are the no-trial
    policies and every sitting of k <= k_j. derive_k scores them under the one
    posterior at n_j; if the sitting that expires now scores best the rule
    commits to it, and otherwise it searches on to the next decision point. At
    the smallest k the choice is final. `sitting_sd_of(state)` gives the look SD
    the rule assumes at that point (the surrogate's, or the truth's).

    A decision point whose fit fails is skipped, as an experimenter without a
    surrogate there would search on; the run fails only if every fit fails.
    """
    points = decision_points(k_grid, T)
    if not points:
        return {"k_hat": None, "gp_failed": True, "scores": {}, "detail": {}, "decided_at": None,
                "sitting_sd_used": float("nan"), "path": "", "n_fit_failures": 0}
    path, failures, last = [], 0, None
    for i, (k_j, n_j) in enumerate(points):
        state = eos.search_state(run, n_j, bounds, beta)
        if state.gp is None:
            failures += 1
            path.append(f"{k_j}:fail")
            continue
        sitting_sd = float(sitting_sd_of(state))
        open_grid = tuple(k for k in k_grid if k <= k_j)
        out = derive_k(run, bounds, open_grid, beta, window, sitting_sd, rng, T, state=state,
                       rate_source=rate_source)
        path.append(f"{k_j}:{out['k_hat']}")
        last = (out, n_j, sitting_sd)
        final = i == len(points) - 1
        if out["k_hat"] == k_j or final:
            return {**out, "decided_at": n_j, "sitting_sd_used": sitting_sd,
                    "path": ";".join(path), "n_fit_failures": failures}
    if last is None:
        return {"k_hat": None, "gp_failed": True, "scores": {}, "detail": {}, "decided_at": None,
                "sitting_sd_used": float("nan"), "path": ";".join(path), "n_fit_failures": failures}
    # Every fit after the last good one failed: the last good point's choice
    # stands only if it is still open, which a no-trial policy always is.
    out, n_j, sitting_sd = last
    k_hat = out["k_hat"] if out["k_hat"] in NO_TRIAL_POLICIES else max(
        NO_TRIAL_POLICIES, key=lambda key: out["scores"][key])
    return {**out, "k_hat": k_hat, "decided_at": n_j, "sitting_sd_used": sitting_sd,
            "path": ";".join(path), "n_fit_failures": failures}


# ---------------------------------------------------------------------------
# Per-run work
# ---------------------------------------------------------------------------


def _init_worker() -> None:
    torch.set_num_threads(1)


def derive_for_run(task: dict) -> dict:
    run = eos.read_run(Path(task["path"]), task["iterations"])
    bounds = torch.tensor(np.stack([task["bounds_low"], task["bounds_high"]]), dtype=torch.double)
    # crc32, not hash(): Python salts string hashing per process, so hash() gave
    # this run a different Monte Carlo stream on every invocation and the chosen
    # k was not reproducible.
    rng = np.random.default_rng(zlib.crc32(task["file"].encode("utf-8")))

    def sitting_sd_of(state) -> float:
        # What an experimenter has at the decision point: the surrogate's own
        # noise estimate there, scaled by how much of it a comparative sitting is
        # assumed to remove. Read at the decision point, not at T, so the rule
        # never uses a trial it is still deciding whether to spend.
        if task["noise_source"] == "truth":
            return task["true_sitting_sd"]
        return gp_sitting_sd(state, task["rho"])

    rate_source = task.get("rate_source", "rating")
    if task.get("decision", "fixed") == "sequential":
        out = derive_k_sequential(run, bounds, task["k_grid"], task["beta"], task["window"], rng,
                                  task["iterations"], sitting_sd_of, rate_source=rate_source)
        return {"file": task["file"], "k_hat": out["k_hat"], "gp_failed": out["gp_failed"],
                "sitting_sd_used": out["sitting_sd_used"],
                "scores": json.dumps({str(k): v for k, v in out["scores"].items()}),
                "decided_at": out["decided_at"], "path": out["path"],
                "n_fit_failures": out["n_fit_failures"]}

    n0 = task["iterations"] - max(task["k_grid"])
    state = eos.search_state(run, n0, bounds, task["beta"]) if n0 >= 2 else None
    if task["noise_source"] == "truth":
        sitting_sd = task["true_sitting_sd"]
    else:
        if state is None or state.gp is None:
            return {"file": task["file"], "k_hat": None, "gp_failed": True}
        sitting_sd = sitting_sd_of(state)
    out = derive_k(run, bounds, task["k_grid"], task["beta"], task["window"], sitting_sd, rng,
                   task["iterations"], state=state, rate_source=rate_source)
    return {"file": task["file"], "k_hat": out["k_hat"], "gp_failed": out["gp_failed"],
            "sitting_sd_used": sitting_sd,
            "scores": json.dumps({str(k): v for k, v in out["scores"].items()})}


def _task(file: str, path: Path, dataset: str, bench, arm, settings: dict, true_sd: float) -> dict:
    return {
        "file": file,
        "path": str(path),
        "dataset": dataset,
        "iterations": arm.iterations,
        "bounds_low": np.asarray(bench.bounds_low, dtype=float),
        "bounds_high": np.asarray(bench.bounds_high, dtype=float),
        "k_grid": settings["k_grid"],
        "beta": settings["beta"],
        "window": settings["window"],
        "rho": settings["rho"],
        "noise_source": settings["noise_source"],
        "true_sitting_sd": settings["rho"] * true_sd,
        "decision": settings.get("decision", "fixed"),
        "rate_source": settings.get("rate_source", "rating"),
    }


def build_tasks(ksweep: pd.DataFrame, arm_root: Path, arm, settings: dict) -> list[dict]:
    stats = bb.load_stats(settings["stats_path"])
    tasks = []
    for file, group in ksweep.groupby("file", sort=True):
        row = group.iloc[0]
        # The box lives on the benchmark, not in the stats file, which holds the
        # standardisation constants and the descriptors.
        bench = bb.BENCHMARKS.get(row["dataset"])
        if bench is None or row["dataset"] not in stats:
            continue
        true_sd = eos.idiosyncratic_sd(row["error_model"], float(row["jitter_std"]))
        # The k-sweep names a run by its basename, the ship-rule rescoring by
        # "<landscape>/<basename>"; the log lives under the landscape either way.
        tasks.append(_task(file, arm_root / row["dataset"] / Path(file).name, row["dataset"], bench, arm,
                           settings, true_sd))
    return tasks


def build_clean_tasks(ksweep: pd.DataFrame, baselines: dict[str, Path], arm, settings: dict) -> list[dict]:
    """One task per clean twin of the swept runs: the rule derived on exact ratings.

    `baselines` maps a stem to its clean log (analyse_extra_runs.index_runs). The
    twin's ratings are exact, so a sitting SD given as the truth is zero, which is
    also what the replay's clean twin assumes for its sitting.
    """
    stats = bb.load_stats(settings["stats_path"])
    tasks, seen = [], set()
    for file, group in ksweep.groupby("file", sort=True):
        stem = twin_stem(file)
        if stem in seen:
            continue
        seen.add(stem)
        dataset = group["dataset"].iloc[0]
        bench = bb.BENCHMARKS.get(dataset)
        if bench is None or dataset not in stats:
            continue
        clean = baselines.get(stem)
        if clean is None:
            raise SystemExit(f"no clean twin for {stem}")
        tasks.append(_task(Path(clean).name, Path(clean), dataset, bench, arm, settings, 0.0))
    return tasks


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _boot_mean_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean of a - b over landscapes, with a landscape bootstrap."""
    d = a - b
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    draws = [d[rng.integers(0, n, n)].mean() for _ in range(BOOTSTRAP_REPS)]
    return float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def score_policies(joined: pd.DataFrame, no_trial: pd.DataFrame, opt_z: dict[str, float],
                   rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Per policy: the deployed regret it would have produced, per landscape.

    `joined` carries the realised regret of every (run, k) from the k-sweep;
    `no_trial` the realised regret of the two policies that buy no trials, from
    the ship-rule rescoring. The derived policy is scored on whichever of the
    three it actually chose, so it is never given a choice it could not make.
    """
    z = joined["dataset"].map(lambda d: opt_z.get(d, 1.0))
    joined = joined.assign(regret_n=joined["regret_noisy"] / z, ref_n=joined["ref_noisy"] / z)

    per_run = joined.pivot_table(index=["dataset", "file"], columns="k", values="regret_n")
    ref = joined.groupby(["dataset", "file"])["ref_n"].first()
    k_hat = joined.groupby(["dataset", "file"])["k_hat"].first()

    # The no-trial policies, on the same index and the same scale.
    nt = no_trial.set_index("file")
    for name, column in (("pm", "regret_pm"), ("lcb", "regret_lcb1")):
        vals = [nt[column].get(f, np.nan) / opt_z.get(d, 1.0) for d, f in per_run.index]
        per_run[name] = vals

    def _look_up(idx, key):
        if pd.isna(key):
            return np.nan
        key = key if key in per_run.columns else _coerce(key, per_run.columns)
        return per_run.loc[idx, key] if key is not None else np.nan

    chosen = pd.Series([_look_up(i, k) for i, k in k_hat.items()], index=per_run.index)
    ks = [c for c in per_run.columns if c not in NO_TRIAL_POLICIES]
    oracle = per_run[ks + list(NO_TRIAL_POLICIES)].min(axis=1)

    frame = pd.DataFrame({"standard": ref, "derived": chosen, "oracle": oracle})
    for name in NO_TRIAL_POLICIES:
        frame[f"always_{name}"] = per_run[name]
    for k in ks:
        frame[f"fixed_k{k}"] = per_run[k]

    by_landscape = frame.groupby(level="dataset").mean()
    rows = []
    base = by_landscape["standard"].to_numpy()
    for policy in frame.columns:
        mean, lo, hi = _boot_mean_diff(base, by_landscape[policy].to_numpy(), rng)
        rows.append({"policy": policy,
                     "mean_regret": float(by_landscape[policy].mean()),
                     "gain_vs_standard": mean, "gain_lo": lo, "gain_hi": hi,
                     "n_landscapes": len(by_landscape)})
    return pd.DataFrame(rows), frame, k_hat


def _coerce(key, columns):
    """A k read back from CSV is a string; the pivot's columns are integers."""
    try:
        as_int = int(float(key))
    except (TypeError, ValueError):
        return None
    return as_int if as_int in columns else None


def policy_name(column: str) -> str:
    """'fixed_k8.0' (a k read as a float) and 'fixed_k8' name one policy."""
    if str(column).startswith("fixed_k"):
        return f"fixed_k{int(float(str(column)[len('fixed_k'):]))}"
    return str(column)


def choice_name(key) -> str:
    """A chosen policy as one string, whether it came from memory (8) or a CSV ('8', '8.0')."""
    try:
        return str(int(float(key)))
    except (TypeError, ValueError):
        return str(key)


def score_by_onset(joined: pd.DataFrame, no_trial: pd.DataFrame, opt_z: dict[str, float],
                   comparators: tuple[str, ...] = COMPARATORS, seeds=None,
                   seeds_label: str = "all") -> pd.DataFrame:
    """Every policy's deployed-design gain within each onset, and the derived rule paired against fixed rules.

    Pooled over onsets the derived rule mixes cells in which it decides after
    the error has begun with cells in which, at the published grid, it cannot
    have seen any, so the pooled gain says little about either. For all onsets
    and for each onset separately: every policy's gain over the standard process
    (landscape means of regret / opt_z, landscape bootstrap), the derived rule's
    gain minus each comparator's (paired: the difference is taken within each
    landscape before resampling), and how often the rule chose each policy.

    Every interval draws from its own generator seeded with BOOTSTRAP_SEED, the
    convention of heldout_remedies.gain_summary, so a number does not depend on
    what else is in the table. Point estimates for all onsets equal the policies
    table's; its intervals share one generator across policies and so differ in
    the third decimal. `seeds` restricts the runs (e.g. the held-out 12-16).
    """
    if seeds is not None:
        joined = joined[joined["seed"].isin(list(seeds))]
    rows = []
    onsets = sorted(joined[ONSET_COLUMN].dropna().unique()) if ONSET_COLUMN in joined else []
    blocks = [("all", joined)] + [(str(int(o)), joined[joined[ONSET_COLUMN] == o]) for o in onsets]
    for label, sub in blocks:
        _, frame, k_hat = score_policies(sub, no_trial, opt_z, np.random.default_rng(BOOTSTRAP_SEED))
        n_runs = int(len(frame))
        by_landscape = frame.rename(columns=policy_name).groupby(level="dataset").mean()
        base = by_landscape["standard"].to_numpy()
        common = {"seeds": seeds_label, "onset": label, "n_landscapes": len(by_landscape), "n_runs": n_runs}
        for policy in by_landscape.columns:
            est, lo, hi = _boot_mean_diff(base, by_landscape[policy].to_numpy(),
                                          np.random.default_rng(BOOTSTRAP_SEED))
            rows.append({**common, "kind": "gain", "policy": policy,
                         "mean_regret": float(by_landscape[policy].mean()), "estimate": est, "lo": lo, "hi": hi})
        for comp in comparators:
            if comp not in by_landscape:
                continue
            # derived gain minus comparator gain = comparator regret minus derived regret.
            est, lo, hi = _boot_mean_diff(by_landscape[comp].to_numpy(), by_landscape["derived"].to_numpy(),
                                          np.random.default_rng(BOOTSTRAP_SEED))
            rows.append({**common, "kind": "derived_minus", "policy": comp, "mean_regret": np.nan,
                         "estimate": est, "lo": lo, "hi": hi})
        for key, share in k_hat.map(choice_name).value_counts(normalize=True).items():
            rows.append({**common, "kind": "choice_share", "policy": key, "mean_regret": np.nan,
                         "estimate": float(share), "lo": np.nan, "hi": np.nan})
    columns = ["seeds", "onset", "kind", "policy", "mean_regret", "estimate", "lo", "hi", "n_landscapes", "n_runs"]
    return pd.DataFrame(rows, columns=columns)


def by_onset_tables(joined: pd.DataFrame, no_trial: pd.DataFrame, opt_z: dict[str, float],
                    comparators: tuple[str, ...] = COMPARATORS, seed_sets=SEED_SETS) -> pd.DataFrame:
    """score_by_onset on all seeds and on each seed half; a half with no runs is left out."""
    parts = []
    for label, seeds in seed_sets:
        if seeds is not None and ("seed" not in joined or not joined["seed"].isin(list(seeds)).any()):
            continue
        parts.append(score_by_onset(joined, no_trial, opt_z, comparators=comparators, seeds=seeds,
                                    seeds_label=label))
    return pd.concat(parts, ignore_index=True)


def price_frame(sweep: pd.DataFrame, derived_clean: pd.DataFrame, no_trial: pd.DataFrame,
                opt_z: dict[str, float]) -> tuple[pd.DataFrame, pd.Series, int]:
    """Per clean twin: what each policy adds to the regret of a run with no error, / opt_z.

    The replay scores every sitting on each run's clean twin too (regret_clean,
    against the clean standard process ref_clean), and those numbers repeat on
    every noisy run of the stem; the no-trial rules' clean regrets are the clean
    twin's rows of the ship-rule rescoring. The derived rule is charged the clean
    regret of the policy it chose on the clean twin. Twins whose fit failed are
    dropped and counted.
    """
    sweep = sweep.assign(stem=sweep["file"].map(twin_stem))
    for column in ("regret_clean", "ref_clean"):
        spread = sweep.groupby(["stem", "k"])[column].agg(lambda s: float(np.ptp(s.to_numpy())))
        if len(spread) and spread.max() > 1e-9:
            raise ValueError(f"{column} differs between the noisy runs of one clean twin")
    keys = ["dataset", "stem", "seed"]
    per_stem = sweep.pivot_table(index=keys, columns="k", values="regret_clean", aggfunc="first")
    per_stem.columns = [int(float(c)) for c in per_stem.columns]
    ref = sweep.groupby(keys)["ref_clean"].first()

    clean = derived_clean.assign(stem=derived_clean["file"].map(twin_stem))
    failed = eos._as_bool(clean["gp_failed"]) if "gp_failed" in clean else pd.Series(False, index=clean.index)
    n_failed = int(failed.sum())
    clean = clean[~failed].drop_duplicates("stem").set_index("stem")
    nt = no_trial.drop_duplicates("file").set_index("file")

    rows, choices = {}, {}
    for (dataset, stem, seed) in per_stem.index:
        if stem not in clean.index:
            continue
        z = opt_z.get(dataset, 1.0)
        base = float(ref.loc[(dataset, stem, seed)])
        clean_file = clean.loc[stem, "file"]
        record = {f"fixed_k{k}": (float(per_stem.loc[(dataset, stem, seed), k]) - base) / z
                  for k in per_stem.columns}
        for name, column in (("pm", "regret_pm"), ("lcb", "regret_lcb1")):
            value = nt[column].get(clean_file, np.nan)
            record[f"always_{name}"] = (float(value) - base) / z
        choice = choice_name(clean.loc[stem, "k_hat"])
        key = f"always_{choice}" if choice in NO_TRIAL_POLICIES else f"fixed_k{choice}"
        record["derived"] = record.get(key, np.nan)
        rows[(dataset, stem, seed)] = record
        choices[(dataset, stem, seed)] = choice
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=keys)
    return frame, pd.Series(choices, name="k_hat"), n_failed


def score_price(sweep: pd.DataFrame, derived_clean: pd.DataFrame, no_trial: pd.DataFrame,
                opt_z: dict[str, float], comparators: tuple[str, ...] = PRICE_COMPARATORS,
                seed_sets=SEED_SETS) -> pd.DataFrame:
    """The price of every policy, trt_clean - ref_clean, and the derived rule's price less each comparator's.

    Landscape means of the per-twin price / opt_z, then the mean over landscapes
    with a landscape bootstrap, every interval from its own generator seeded with
    BOOTSTRAP_SEED. A positive price is regret the policy adds when there is no
    error at all; it is the same at every magnitude and onset, since the clean
    twin carries neither.
    """
    frame, choices, n_failed = price_frame(sweep, derived_clean, no_trial, opt_z)
    rows = []
    for label, seeds in seed_sets:
        sub = frame if seeds is None else frame[frame.index.get_level_values("seed").isin(list(seeds))]
        if sub.empty:
            continue
        by_landscape = sub.groupby(level="dataset").mean()
        common = {"seeds": label, "n_landscapes": len(by_landscape), "n_twins": len(sub),
                  "n_failed": n_failed}
        zero = np.zeros(len(by_landscape))
        for policy in by_landscape.columns:
            est, lo, hi = _boot_mean_diff(by_landscape[policy].to_numpy(), zero,
                                          np.random.default_rng(BOOTSTRAP_SEED))
            rows.append({**common, "kind": "price", "policy": policy, "estimate": est, "lo": lo, "hi": hi})
        for comp in comparators:
            if comp not in by_landscape:
                continue
            est, lo, hi = _boot_mean_diff(by_landscape["derived"].to_numpy(), by_landscape[comp].to_numpy(),
                                          np.random.default_rng(BOOTSTRAP_SEED))
            rows.append({**common, "kind": "derived_minus", "policy": comp, "estimate": est, "lo": lo, "hi": hi})
        picked = choices.reindex(sub.index)
        for key, share in picked.value_counts(normalize=True).items():
            rows.append({**common, "kind": "choice_share", "policy": key, "estimate": float(share),
                         "lo": np.nan, "hi": np.nan})
    columns = ["seeds", "kind", "policy", "estimate", "lo", "hi", "n_landscapes", "n_twins", "n_failed"]
    return pd.DataFrame(rows, columns=columns)


def load_sweep(ksweep_dir: str, rho: float, acquisitions: str | None = "logei,qnei",
               error_models: str | None = "gaussian,bias,drift,ar1") -> pd.DataFrame:
    """The replayed tournaments the rule is scored against, one row per (run, k)."""
    frames = []
    for raw in str(ksweep_dir).split(","):
        per_run_path = Path(raw.strip()) / "end_of_study_per_run.csv.gz"
        if not per_run_path.is_file():
            raise SystemExit(f"{per_run_path} not found: run the k-sweep of replay_end_of_study.py first")
        frames.append(pd.read_csv(per_run_path, low_memory=False))
    # Grids run in separate directories share every column; a k appearing in two
    # of them would be scored twice, so the first copy wins.
    sweep = pd.concat(frames, ignore_index=True)
    sweep = sweep.drop_duplicates(subset=["file", "procedure"], keep="first")
    # The tournament family the decomposition points at: lcb candidates, ship the
    # fresh look. rho is the sitting's assumed precision.
    sweep = sweep[(sweep["family"] == "tournament") & (sweep["candidates"] == "lcb")
                  & (sweep["winner"] == "look") & (sweep["rho"] == rho)]
    if acquisitions:
        sweep = sweep[sweep["acquisition"].isin({a.strip() for a in acquisitions.split(",")})]
    if error_models:
        sweep = sweep[sweep["error_model"].isin({e.strip() for e in error_models.split(",")})]
    if sweep.empty:
        raise SystemExit("no k-sweep rows match the filters")
    return sweep


def join_derived(sweep: pd.DataFrame, derived: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The sweep with each run's chosen policy; runs whose fit failed are dropped and counted."""
    joined = sweep.merge(derived[["file", "k_hat", "gp_failed"]], on="file", how="inner")
    failed = eos._as_bool(joined["gp_failed"])
    n_failed = int(failed.groupby(joined["file"]).first().sum())
    joined = joined[~failed]
    if joined.empty:
        raise SystemExit("every run's GP fit failed")
    return joined, n_failed


def load_no_trial(input_dir: Path) -> pd.DataFrame:
    """The two no-trial policies' realised regret, from the ship-rule rescoring."""
    ship_path = Path(input_dir) / "analysis" / "ship_rules_per_run.csv"
    if not ship_path.is_file():
        raise SystemExit(f"{ship_path} not found: the no-trial policies come from rescore_ship_rules.py")
    no_trial = pd.read_csv(ship_path, usecols=["file", "regret_pm", "regret_lcb1"])
    # Both tables key a run by its log; only the ship-rule one carries the
    # landscape directory, so they are matched on the basename.
    no_trial["file"] = no_trial["file"].map(lambda f: Path(f).name)
    return no_trial


def load_opt_z(stats_path: Path = bb.DEFAULT_STATS_PATH) -> dict[str, float]:
    stats = bb.load_stats(stats_path)
    return {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}


def _derive_all(tasks: list[dict], workers: int) -> pd.DataFrame:
    print(f"deriving k for {len(tasks):,} runs with {workers} worker(s)", flush=True)
    records = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, os.cpu_count() or 1),
                                 initializer=_init_worker) as pool:
            for i, rec in enumerate(pool.map(derive_for_run, tasks, chunksize=4), 1):
                records.append(rec)
                if i % 200 == 0:
                    print(f"  {i:,}/{len(tasks):,}", flush=True)
    else:
        _init_worker()
        records = [derive_for_run(t) for t in tasks]
    return pd.DataFrame(records)


def _read_derived(derived_path: Path, decision: str) -> pd.DataFrame:
    derived = pd.read_csv(derived_path)
    # Only the sequential rule records where it decided; rescoring one rule's
    # choices under the other's name would mislabel every output.
    if ("decided_at" in derived.columns) != (decision == "sequential"):
        raise SystemExit(f"{derived_path} was not written with --decision {decision}")
    return derived


def main_clean(args: argparse.Namespace, sweep: pd.DataFrame, settings: dict, out_dir: Path) -> pd.DataFrame:
    """--twin clean: the rule derived on each clean twin, and the price table."""
    derived_path = output_paths(out_dir, args.output_suffix)["derived"]
    if args.summary_only and derived_path.is_file():
        derived = _read_derived(derived_path, args.decision)
    else:
        import analyse_extra_runs as aer
        arm = eos.load_arm(args.input_dir)
        baselines, _ = aer.index_runs(args.input_dir)
        derived = _derive_all(build_clean_tasks(sweep, baselines, arm, settings), args.workers)
        derived.to_csv(derived_path, index=False)
    table = score_price(sweep, derived, load_no_trial(args.input_dir), load_opt_z(args.stats_path))
    out = price_path(out_dir, args.output_suffix)
    table.to_csv(out, index=False)
    print(f"\nprice of each policy on the clean twins (regret it adds without error, / opt_z; "
          f"decision {args.decision}, grid {','.join(map(str, settings['k_grid']))}):")
    for seeds, block in table.groupby("seeds", sort=False):
        price = block[block["kind"] == "price"]
        text = "; ".join(f"{r.policy} {r.estimate:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]"
                         for r in price.itertuples(index=False)
                         if r.policy in ("derived", "fixed_k8", "fixed_k12", "fixed_k16", "always_pm", "always_lcb"))
        print(f"  seeds {seeds:>5} ({int(block['n_twins'].iloc[0])} twins, {int(block['n_failed'].iloc[0])} "
              f"failed fits): {text}")
    print(f"\nWrote {derived_path} and {out}")
    return table


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    out_dir = args.output_dir or (args.input_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir, args.output_suffix)
    derived_path = paths["derived"]

    sweep = load_sweep(args.ksweep_dir, args.rho, args.acquisitions, args.error_models)

    k_grid = tuple(int(k) for k in args.k_grid.split(","))
    settings = {"k_grid": k_grid, "beta": args.lcb_beta, "window": args.window, "rho": args.rho,
                "noise_source": args.noise_source, "stats_path": args.stats_path,
                "decision": args.decision, "rate_source": args.rate_source}

    if args.twin == "clean":
        return main_clean(args, sweep, settings, out_dir)

    if args.summary_only and derived_path.is_file():
        derived = _read_derived(derived_path, args.decision)
    else:
        arm = eos.load_arm(args.input_dir)
        tasks = build_tasks(sweep, args.input_dir, arm, settings)
        derived = _derive_all(tasks, args.workers)
        derived.to_csv(derived_path, index=False)

    joined, n_failed = join_derived(sweep, derived)
    no_trial = load_no_trial(args.input_dir)
    opt_z = load_opt_z(args.stats_path)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary, frame, k_hat = score_policies(joined, no_trial, opt_z, rng)
    summary.to_csv(paths["policies"], index=False)
    by_onset = by_onset_tables(joined, no_trial, opt_z)
    by_onset.to_csv(paths["by_onset"], index=False)

    counts = k_hat.astype(str).value_counts()
    print(f"\n{len(frame):,} runs, {summary['n_landscapes'].iloc[0]} landscapes, "
          f"{n_failed} dropped for a failed fit; noise for the rule from {args.noise_source}; "
          f"search rate from the {args.rate_source}; decision {args.decision}, grid {','.join(map(str, k_grid))}")
    print("chosen policy: " + ", ".join(f"{k}: {n}" for k, n in counts.items()))
    print("\nDeployed regret (fraction of the achievable improvement), and the gain over the standard process:")
    for _, r in summary.sort_values("mean_regret").iterrows():
        print(f"  {r['policy']:<12} regret {r['mean_regret']:.4f}   "
              f"gain {r['gain_vs_standard']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}]")
    print("\nBy onset: the derived rule's deployed-design gain, and its gain minus each comparator's:")
    for (seeds, label), block in by_onset.groupby(["seeds", "onset"], sort=False):
        gain = block[(block["kind"] == "gain") & (block["policy"] == "derived")].iloc[0]
        diffs = block[block["kind"] == "derived_minus"]
        text = "; ".join(f"minus {r.policy} {r.estimate:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]"
                         for r in diffs.itertuples(index=False))
        print(f"  seeds {seeds:>5}, onset {label:>3}: derived {gain['estimate']:+.4f} "
              f"[{gain['lo']:+.4f}, {gain['hi']:+.4f}]; {text}")
    print(f"\nWrote {paths['derived']}, {paths['policies']} and {paths['by_onset']}")
    return summary


if __name__ == "__main__":
    main()
