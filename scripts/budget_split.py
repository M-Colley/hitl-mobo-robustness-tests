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

    python scripts/budget_split.py --workers 8
    python scripts/budget_split.py --summary-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("output-boba"))
    p.add_argument("--ksweep-dir", type=str, default="output-boba/analysis/end_of_study_ksweep",
                   help="the k-sweep(s) whose per-run regrets the derived k is scored against; "
                        "comma-separated to pool grids run in separate directories")
    p.add_argument("--output-dir", type=Path, default=None, help="default: <input-dir>/analysis")
    p.add_argument("--k-grid", type=str, default="2,3,5,8,12")
    p.add_argument("--rho", type=float, default=0.5,
                   help="the sitting's noise as a multiple of the idiosyncratic SD")
    p.add_argument("--lcb-beta", type=float, default=1.0)
    p.add_argument("--window", type=int, default=10,
                   help="trials of recent improvement the search term is extrapolated from")
    p.add_argument("--noise-source", choices=("gp", "truth"), default="gp",
                   help="'gp': the rule estimates s from the surrogate, as an experimenter must. "
                        "'truth': it is given the true idiosyncratic SD, to price that knowledge.")
    p.add_argument("--functions", type=str, default="all")
    p.add_argument("--acquisitions", type=str, default="logei,qnei")
    p.add_argument("--error-models", type=str, default="gaussian,bias,drift,ar1")
    p.add_argument("--seeds", type=str, default="7,8,9,10,11,12,13,14,15,16")
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args(argv)


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
             sitting_sd: float, rng: np.random.Generator, T: int) -> dict:
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
    """
    k_max = max(k_grid)
    n0 = T - k_max
    if n0 < 2:
        return {"k_hat": None, "gp_failed": True, "scores": {}, "detail": {}}
    state = eos.search_state(run, n0, bounds, beta)
    if state.gp is None:
        return {"k_hat": None, "gp_failed": True, "scores": {}, "detail": {}}
    mu, sd = eos.latent_mean_sd(state.gp, run.X[:n0][state.first])
    order = np.argsort(-state.lcb)
    rate = recent_improvement_rate(run.observed, n0, window)

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


# ---------------------------------------------------------------------------
# Per-run work
# ---------------------------------------------------------------------------


def _init_worker() -> None:
    torch.set_num_threads(1)


def derive_for_run(task: dict) -> dict:
    run = eos.read_run(Path(task["path"]), task["iterations"])
    bounds = torch.tensor(np.stack([task["bounds_low"], task["bounds_high"]]), dtype=torch.double)
    rng = np.random.default_rng(abs(hash(task["file"])) % (2**32))
    if task["noise_source"] == "truth":
        sitting_sd = task["true_sitting_sd"]
    else:
        # What an experimenter has at the decision point: the surrogate's own
        # noise estimate there, scaled by how much of it a comparative sitting is
        # assumed to remove. Read at n0, not at T, so the rule never uses a trial
        # it is still deciding whether to spend.
        n0 = task["iterations"] - max(task["k_grid"])
        state = eos.search_state(run, n0, bounds, task["beta"])
        if state.gp is None:
            return {"file": task["file"], "k_hat": None, "gp_failed": True}
        sitting_sd = task["rho"] * float(np.sqrt(state.gp.likelihood.noise.mean().item()))
    out = derive_k(run, bounds, task["k_grid"], task["beta"], task["window"], sitting_sd, rng,
                   task["iterations"])
    return {"file": task["file"], "k_hat": out["k_hat"], "gp_failed": out["gp_failed"],
            "sitting_sd_used": sitting_sd,
            "scores": json.dumps({str(k): v for k, v in out["scores"].items()})}


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
        tasks.append({
            "file": file,
            # The k-sweep names a run by its basename, the ship-rule rescoring by
            # "<landscape>/<basename>"; the log lives under the landscape either way.
            "path": str(arm_root / row["dataset"] / Path(file).name),
            "dataset": row["dataset"],
            "iterations": arm.iterations,
            "bounds_low": np.asarray(bench.bounds_low, dtype=float),
            "bounds_high": np.asarray(bench.bounds_high, dtype=float),
            "k_grid": settings["k_grid"],
            "beta": settings["beta"],
            "window": settings["window"],
            "rho": settings["rho"],
            "noise_source": settings["noise_source"],
            "true_sitting_sd": settings["rho"] * true_sd,
        })
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


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    out_dir = args.output_dir or (args.input_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_path = out_dir / f"{OUTPUT_NAME}_derived.csv"

    frames = []
    for raw in str(args.ksweep_dir).split(","):
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
                  & (sweep["winner"] == "look") & (sweep["rho"] == args.rho)]
    if args.acquisitions:
        sweep = sweep[sweep["acquisition"].isin({a.strip() for a in args.acquisitions.split(",")})]
    if args.error_models:
        sweep = sweep[sweep["error_model"].isin({e.strip() for e in args.error_models.split(",")})]
    if sweep.empty:
        raise SystemExit("no k-sweep rows match the filters")

    k_grid = tuple(int(k) for k in args.k_grid.split(","))
    settings = {"k_grid": k_grid, "beta": args.lcb_beta, "window": args.window, "rho": args.rho,
                "noise_source": args.noise_source, "stats_path": args.stats_path}

    if args.summary_only and derived_path.is_file():
        derived = pd.read_csv(derived_path)
    else:
        arm = eos.load_arm(args.input_dir)
        tasks = build_tasks(sweep, args.input_dir, arm, settings)
        print(f"deriving k for {len(tasks):,} runs with {args.workers} worker(s)", flush=True)
        records = []
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=min(args.workers, os.cpu_count() or 1),
                                     initializer=_init_worker) as pool:
                for i, rec in enumerate(pool.map(derive_for_run, tasks, chunksize=4), 1):
                    records.append(rec)
                    if i % 200 == 0:
                        print(f"  {i:,}/{len(tasks):,}", flush=True)
        else:
            _init_worker()
            records = [derive_for_run(t) for t in tasks]
        derived = pd.DataFrame(records)
        derived.to_csv(derived_path, index=False)

    joined = sweep.merge(derived[["file", "k_hat", "gp_failed"]], on="file", how="inner")
    n_failed = int(joined.groupby("file")["gp_failed"].first().sum())
    joined = joined[~joined["gp_failed"]]
    if joined.empty:
        raise SystemExit("every run's GP fit failed")

    ship_path = args.input_dir / "analysis" / "ship_rules_per_run.csv"
    if not ship_path.is_file():
        raise SystemExit(f"{ship_path} not found: the no-trial policies come from rescore_ship_rules.py")
    no_trial = pd.read_csv(ship_path, usecols=["file", "regret_pm", "regret_lcb1"])
    # Both tables key a run by its log; only the ship-rule one carries the
    # landscape directory, so they are matched on the basename.
    no_trial["file"] = no_trial["file"].map(lambda f: Path(f).name)

    stats = bb.load_stats(args.stats_path)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary, frame, k_hat = score_policies(joined, no_trial, opt_z, rng)
    summary.to_csv(out_dir / f"{OUTPUT_NAME}_policies.csv", index=False)

    counts = k_hat.astype(str).value_counts()
    print(f"\n{len(frame):,} runs, {summary['n_landscapes'].iloc[0]} landscapes, "
          f"{n_failed} dropped for a failed fit; noise for the rule from {args.noise_source}")
    print("chosen policy: " + ", ".join(f"{k}: {n}" for k, n in counts.items()))
    print("\nDeployed regret (fraction of the achievable improvement), and the gain over the standard process:")
    for _, r in summary.sort_values("mean_regret").iterrows():
        print(f"  {r['policy']:<12} regret {r['mean_regret']:.4f}   "
              f"gain {r['gain_vs_standard']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}]")
    print(f"\nWrote {out_dir / (OUTPUT_NAME + '_policies.csv')}")
    return summary


if __name__ == "__main__":
    main()
