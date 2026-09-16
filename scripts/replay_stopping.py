"""Stopping rules that spend FEWER human trials, replayed exactly from the logs.

Every other arm keeps the budget at T trials and changes what happens inside it.
These two rules END the study early, so they can only ever use fewer trials
than the standard process -- never more.

Why a replay is exact
---------------------
The loop's proposal at trial t depends only on trials 1..t-1 and on the random
streams consumed up to then, so a study that stops at trial t IS the logged run
truncated at t. Nothing is re-simulated: the stopping decision is a function of
the log prefix, and the design shipped at the stop is a refit on that prefix.
The drift ramp is set by the PLANNED budget, which a stopped study shares, so
that holds for drift as well. Every worker re-derives the standard process's
deployed design (the best rating so far) at every prefix and checks it against
the logged ``inference_simple_regret_true``; a mismatch aborts the analysis.

The ship rule
-------------
Refit the loop's own surrogate (BoTorch default SingleTaskGP: RBF-ARD kernel,
Normalize with the benchmark box, Standardize) on the trials in hand and ship
the visited design with the highest posterior mean minus one latent posterior
SD. It is scored at the stop and, for reference, at T.

Rule A -- onset detection and freeze
------------------------------------
Fit the GP on the first n0 = 15 trials and hold its hyperparameters and its
standardisation fixed. For t = n0+1..T the one-step-ahead residual

    z_t = (y_t - mu_{t-1}(x_t)) / sqrt(var_{t-1}(x_t) + noise)

comes from exact conditioning on trials 1..t-1 (linear algebra, no refit), and
a one-sided CUSUM on its square, S_t = max(0, S_{t-1} + z_t^2 - 1 - kappa),
alarms at the first t with S_t > h. On alarm the study stops at t and ships the
ship-rule design refitted on trials 1..t-w: the last w trials are the ones most
likely to be corrupted already. (h, kappa, w) are tuned on the tuning seeds to
minimise deployed regret subject to a false-alarm rate of at most 10% on clean
runs and on runs with error from trial 1 -- in both, nothing changes after n0,
so any alarm is false -- and then scored on the held-out seeds. A late-onset
alarm at t <= onset is counted as a false alarm too.

Rule B -- decision-stable stopping
----------------------------------
At checkpoints 20, 30, 40 refit the GP. Stop at a checkpoint when the ship-rule
design is the same one as at the previous checkpoint AND

    max over visited designs of (mean + SD)  -  (mean - SD) of the ship design  <  delta

i.e. nothing already visited could plausibly beat what would be shipped. delta
is 0.25 or 0.5 in "model" units (the SD of the ratings the GP standardised by,
which a practitioner can compute) or in "landscape" units (the objective's own
standardisation, which only a simulation knows); both are reported. Without a
stop the study runs to T and ships the ship-rule design there.

The floored variant guards against a surrogate that has explained the error
away as signal: the GP noise SD is floored at the sample SD of the first
design's ratings -- its logged rating plus three simulated repeats drawn from
the run's own error process as it stands at trials 2..4 (so a late onset leaves
the floor at zero, honestly). The repeats are real trials: they are counted as
trials used (t + 3 at a stop) and, without a stop, the study ships at T - 3 so
the total never exceeds T.

Scores and estimand
-------------------
Per run: trials used, deployed regret at the stop under the ship rule, deployed
regret of the standard process at T (best rating, the logged column), and the
ship rule at T. Summaries use the estimand of analyse_boba_adaptations.py
(its summarise(), imported): per paired cell cost = standard noisy - standard
clean, gain = standard noisy - rule noisy, price = rule clean - standard clean,
recovered = gain / cost, regret divided by opt_z, intervals resampling
landscapes. The rule is applied to the clean twin as well, so false alarms and
early stops without error show up as price. ship_T is scored the same way, so
"A minus ship_T" is what the stopping itself adds. Plus trials saved (mean and
median), stop and false-alarm rates, by error model x magnitude x onset.

Cost
----
A GP fit on <= 50 points is 0.05-0.35 s. Each run needs the n0 fit, the
checkpoint fits and a handful of prefix refits at the alarm times the rule-A
grid produces; refits are cached per run in <output-dir>/cache_*/<landscape>.jsonl
so an interrupted analysis resumes (the machine sleeps). Score seeds only get
refits for the tuned (h, kappa, w).

    python scripts/replay_stopping.py --input-dir output-boba --workers 5
    python scripts/replay_stopping.py --input-dir output-boba-budget100 --rules A \\
        --error-models gaussian --tune-seeds "" --score-seeds 7,8,9,10,11 \\
        --a-params-from output-boba/analysis/stopping/stopping_tuned_A.json --workers 5
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Imported first, as in the drivers: it sets the BLAS/OMP thread limits every
# worker needs before numpy and torch load, and owns torch's float64 default.
import bo_sensor_error_simulation as sim  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.transforms import Normalize, Standardize  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import analyse_boba_adaptations as adapt  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
from analyse_extra_runs import JITTERED, STEM  # noqa: E402

ORACLE_TAG = "exact"
MODEL_FREE = ("random", "sobol")
# Appended to the run's jitter SeedSequence so the simulated repeat ratings get a
# stream of their own: they must not replay draws the logged trials already used.
REPEAT_STREAM_TAG = 20260914
NEVER = float("inf")
H_GRID = "2,5,10,20,50,100,200,500,1000,2000,5000,10000"
KAPPA_GRID = "0,1,4,16"
W_GRID = "0,1,2,3"
PAIR = ["dataset", "acquisition", "seed"]
CELL = ["error_model", "jitter_std", "jitter_iteration"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("output-boba"))
    p.add_argument("--output-dir", type=Path, default=None, help="default: <input-dir>/analysis/stopping")
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    p.add_argument("--functions", type=str, default="all")
    p.add_argument("--acquisitions", type=str, default="logei,qnei")
    p.add_argument("--error-models", type=str, default="gaussian,bias,drift,ar1")
    p.add_argument("--jitter-stds", type=str, default="all")
    p.add_argument("--onsets", type=str, default="all",
                   help="jitter_iteration values; 0 = error from trial 1, 20 = from trial 21")
    p.add_argument("--tune-seeds", type=str, default="7,8,9,10,11",
                   help="seeds rule A is tuned on; empty with --a-params-from")
    p.add_argument("--score-seeds", type=str, default="12,13,14,15,16")
    p.add_argument("--rules", type=str, default="A,B")
    p.add_argument("--n0", type=int, default=15, help="rule A: trials the frozen GP is fitted on")
    p.add_argument("--h-grid", type=str, default=H_GRID)
    p.add_argument("--kappa-grid", type=str, default=KAPPA_GRID)
    p.add_argument("--w-grid", type=str, default=W_GRID)
    p.add_argument("--max-false-alarm", type=float, default=0.10)
    p.add_argument("--false-alarm-scope", choices=["pooled", "cell"], default="pooled",
                   help="'pooled': the onset-0 false-alarm rate over all onset-0 runs; 'cell': the worst "
                        "error model x magnitude cell must meet the cap")
    p.add_argument("--tune-objective", choices=["regret", "trials"], default="regret",
                   help="'regret': lowest mean deployed regret (opt_z units, landscape-weighted) over the "
                        "tuning runs; 'trials': fewest trials among settings no worse than never stopping")
    p.add_argument("--a-params-from", type=Path, default=None,
                   help="stopping_tuned_A.json of an earlier analysis: skip tuning and apply its (h, kappa, w)")
    p.add_argument("--grid-on-score-seeds", action="store_true",
                   help="also refit every grid setting on the score seeds (sensitivity; costs more fits)")
    p.add_argument("--checkpoints", type=str, default="20,30,40")
    p.add_argument("--deltas", type=str, default="0.25,0.5")
    p.add_argument("--delta-units", type=str, default="model,landscape")
    p.add_argument("--floor-repeats", type=int, default=3)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--no-evaluation-check", action="store_true",
                   help="skip the cross-check against <landscape>/evaluation/paired_excess_metrics.csv")
    return p.parse_args(argv)


def _floats(raw: str) -> list[float]:
    return [float(v) for v in raw.split(",") if v.strip()]


def _ints(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def resolve_config(args: argparse.Namespace) -> dict:
    """Parse and validate every option; incompatible combinations fail here, loudly."""
    rules = [r.strip().upper() for r in args.rules.split(",") if r.strip()]
    if not rules or set(rules) - {"A", "B"}:
        raise ValueError(f"--rules must name A and/or B, got {args.rules!r}")
    tune, score = _ints(args.tune_seeds), _ints(args.score_seeds)
    if set(tune) & set(score):
        raise ValueError(f"seeds {sorted(set(tune) & set(score))} are in both --tune-seeds and --score-seeds")
    if not score and not tune:
        raise ValueError("no seeds selected")
    models = [m.strip() for m in args.error_models.split(",") if m.strip()]
    bad = [m for m in models if m not in sim.ERROR_MODEL_CHOICES]
    if bad:
        # Input-error runs (slip, misclick) rate honestly; the floor and the
        # residual monitor are defined for response error only.
        raise ValueError(f"--error-models {bad} are not response-error models {sim.ERROR_MODEL_CHOICES}")
    acqs = [a.strip() for a in args.acquisitions.split(",") if a.strip()]
    if any(a in MODEL_FREE for a in acqs):
        raise ValueError(f"model-free floors {MODEL_FREE} have no surrogate to stop on; drop them from --acquisitions")
    cfg = dict(rules=rules, tune_seeds=tune, score_seeds=score, error_models=models, acquisitions=acqs,
               n0=int(args.n0), h_grid=_floats(args.h_grid), kappa_grid=_floats(args.kappa_grid),
               w_grid=_ints(args.w_grid), checkpoints=_ints(args.checkpoints), deltas=_floats(args.deltas),
               delta_units=[u.strip() for u in args.delta_units.split(",") if u.strip()],
               floor_repeats=int(args.floor_repeats), max_false_alarm=float(args.max_false_alarm))
    if "A" in rules:
        if cfg["n0"] < 3:
            raise ValueError("--n0 must be >= 3 for a GP fit")
        if not cfg["h_grid"] or min(cfg["h_grid"]) <= 0:
            raise ValueError("--h-grid must be positive")
        if not cfg["kappa_grid"] or min(cfg["kappa_grid"]) < 0:
            raise ValueError("--kappa-grid must be non-negative")
        if not cfg["w_grid"] or min(cfg["w_grid"]) < 0 or max(cfg["w_grid"]) > cfg["n0"] - 2:
            raise ValueError(f"--w-grid must lie in [0, n0 - 2] = [0, {cfg['n0'] - 2}] so the refit keeps >= 3 trials")
        if not 0.0 <= cfg["max_false_alarm"] < 1.0:
            raise ValueError("--max-false-alarm must lie in [0, 1)")
        if args.a_params_from is None and not tune:
            raise ValueError("rule A needs --tune-seeds, or tuned parameters via --a-params-from")
        if args.a_params_from is not None and tune:
            raise ValueError("--a-params-from skips tuning; pass --tune-seeds \"\" so no seed is held back unused")
    if "B" in rules:
        cps = cfg["checkpoints"]
        if len(cps) < 2 or any(b <= a for a, b in zip(cps, cps[1:])) or cps[0] < 3:
            raise ValueError("--checkpoints needs at least two strictly increasing trial counts >= 3")
        if not cfg["deltas"] or min(cfg["deltas"]) <= 0:
            raise ValueError("--deltas must be positive")
        if not cfg["delta_units"] or set(cfg["delta_units"]) - {"model", "landscape"}:
            raise ValueError("--delta-units takes 'model' and/or 'landscape'")
        if cfg["floor_repeats"] < 1:
            raise ValueError("--floor-repeats must be >= 1 (the floored variant needs repeat ratings)")
    return cfg


def check_budget(cfg: dict, T: int) -> None:
    if "A" in cfg["rules"] and cfg["n0"] >= T - 1:
        raise ValueError(f"--n0 {cfg['n0']} leaves no trials to monitor in a {T}-trial run")
    if "B" in cfg["rules"]:
        # A stop at the last checkpoint plus the repeats must still fit the budget.
        if cfg["checkpoints"][-1] + cfg["floor_repeats"] > T:
            raise ValueError(f"last checkpoint {cfg['checkpoints'][-1]} + {cfg['floor_repeats']} repeats "
                             f"exceeds the {T}-trial budget")


# ---------------------------------------------------------------------------
# Indexing the logs
# ---------------------------------------------------------------------------


def index_runs(input_dir: Path, functions: list[str], cfg: dict, stds: list[float] | None,
               onsets: list[int] | None, opt_z: dict[str, float]) -> pd.DataFrame:
    """Standard-process runs (no variant suffix) and their clean twins, one row per log."""
    split_of = {**{s: "tune" for s in cfg["tune_seeds"]}, **{s: "score" for s in cfg["score_seeds"]}}
    rows = []
    for dataset in functions:
        for path in sorted((input_dir / dataset).glob(f"bo_sensor_error_{dataset}_*.csv")):
            name = path.stem
            if "_baseline_" in name:
                stem, rest = name.split("_baseline_", 1)
                if rest != ORACLE_TAG:
                    continue  # a variant arm's clean run (inc-, noise-) is another process
                model, std, onset, bias = "none", 0.0, np.nan, np.nan
            elif "_jittered_" in name:
                stem, rest = name.split("_jittered_", 1)
                m = JITTERED.match(rest)
                if not m or m["oracle"] != ORACLE_TAG:
                    continue
                model, onset, std = m["model"], float(m["onset"]), float(m["std"])
                suffix, bias = m["suffix"].lstrip("_"), np.nan
                # The bias model always carries its offset in the name; any other
                # suffix is a process variant, not the standard process.
                if model == "bias" and re.fullmatch(r"bias[0-9.eE+-]+", suffix):
                    bias, suffix = float(suffix[4:]), ""
                if suffix or model not in cfg["error_models"]:
                    continue
                if stds is not None and not np.isclose(std, stds).any():
                    continue
                if onsets is not None and int(onset) not in onsets:
                    continue
            else:
                continue
            s = STEM.match(stem)
            if not s or s["dataset"] != dataset or s["acq"] not in cfg["acquisitions"]:
                continue
            seed = int(s["seed"])
            if seed not in split_of:
                continue
            st = path.stat()
            rows.append(dict(file=path.name, path=str(path), stem=stem, dataset=dataset, acquisition=s["acq"],
                             seed=seed, split=split_of[seed], error_model=model, jitter_std=std,
                             jitter_iteration=onset, bias=bias, clean=model == "none",
                             opt_z=float(opt_z[dataset]), stamp=f"{st.st_size}-{st.st_mtime_ns}"))
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit(f"no matching runs under {input_dir}")
    # Every noisy run is scored against its identically seeded clean twin.
    clean_keys = set(map(tuple, runs.loc[runs.clean, PAIR].to_numpy().tolist()))
    has_twin = runs.apply(lambda r: r.clean or (r.dataset, r.acquisition, r.seed) in clean_keys, axis=1)
    if (~has_twin).any():
        print(f"WARNING: {int((~has_twin).sum())} noisy runs have no clean twin and are dropped", file=sys.stderr)
    return runs[has_twin].reset_index(drop=True)


def read_run_args(input_dir: Path) -> dict:
    meta = input_dir / "run_metadata.json"
    if not meta.is_file():
        return {}
    return json.loads(meta.read_text(encoding="utf-8")).get("args", {})


# ---------------------------------------------------------------------------
# The surrogate: fit, one-step residuals, ship rule
# ---------------------------------------------------------------------------


def fit_loop_gp(X: torch.Tensor, Y: torch.Tensor, bounds: torch.Tensor, key: str,
                train_Yvar: torch.Tensor | None = None) -> tuple[SingleTaskGP, bool, int]:
    """The loop's surrogate, built exactly as run_simulation builds it.

    Seeded from the run and prefix so a fit that falls back to fit_gpytorch_mll's
    random restarts is still reproducible, whatever order the passes run in.
    A failed fit keeps the initial hyperparameters and is flagged.
    """
    torch.manual_seed(zlib.crc32(key.encode()) & 0x7FFFFFFF)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp = SingleTaskGP(X, Y, train_Yvar=train_Yvar,
                          input_transform=Normalize(d=X.shape[-1], bounds=bounds),
                          outcome_transform=Standardize(m=1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            ok = True
        except Exception:
            ok = False
    gp.eval()
    return gp, ok, len(caught)


def one_step_residuals(gp: SingleTaskGP, X: torch.Tensor, Y: torch.Tensor, n0: int) -> np.ndarray:
    """z_t for t = n0+1..T, conditioning exactly on trials 1..t-1 with gp's hyperparameters fixed.

    Worked in the model's standardised units with ITS standardisation (fitted
    on the first n0 ratings), so nothing about the model moves after n0; z is
    scale-free anyway. The kernel, mean and noise are read off the fitted
    modules rather than re-implemented, so a change of BoTorch default kernel
    cannot silently desynchronise this from the loop.
    """
    T = X.shape[0]
    with torch.no_grad():
        K = gp.covar_module(gp.input_transform(X)).to_dense()
        c = gp.mean_module.constant.reshape(-1)[0]
        noise = gp.likelihood.noise.reshape(-1)[0]
        ys = (Y.reshape(-1) - gp.outcome_transform.means.reshape(-1)[0]) / gp.outcome_transform.stdvs.reshape(-1)[0]
        eye = torch.eye(T, dtype=K.dtype)
        z = np.empty(T - n0)
        for t in range(n0, T):  # 0-based t is trial t + 1
            # The noise constraint (>= 1e-4) keeps this factorisation well posed
            # even when BO has piled designs on top of each other.
            L = torch.linalg.cholesky(K[:t, :t] + noise * eye[:t, :t])
            k = K[:t, t]
            alpha = torch.cholesky_solve((ys[:t] - c).unsqueeze(-1), L).squeeze(-1)
            v = torch.linalg.solve_triangular(L, k.unsqueeze(-1), upper=False).squeeze(-1)
            var = (K[t, t] - v @ v).clamp_min(0.0)
            z[t - n0] = float((ys[t] - (c + k @ alpha)) / torch.sqrt(var + noise))
    return z


def ship_record(gp: SingleTaskGP, X: torch.Tensor, m: int, y_deployed: np.ndarray, y_opt: float,
                ok: bool, n_warn: int) -> dict:
    """The ship-rule decision on trials 1..m: argmax over visited designs of mean - 1 latent SD."""
    with torch.no_grad():
        post = gp.posterior(X[:m])
        mu = post.mean.reshape(-1).numpy()
        sd = post.variance.clamp_min(0.0).sqrt().reshape(-1).numpy()
    lcb = mu - sd
    ship, pm = int(np.argmax(lcb)), int(np.argmax(mu))
    stdv = float(gp.outcome_transform.stdvs.reshape(-1)[0])
    return dict(ok=bool(ok), n_warn=int(n_warn), ship_idx=ship, pm_idx=pm,
                regret=float(y_opt - y_deployed[ship]), pm_regret=float(y_opt - y_deployed[pm]),
                max_ucb=float((mu + sd).max()), lcb_ship=float(lcb[ship]), y_std=stdv,
                noise_var=float(gp.likelihood.noise.detach().reshape(-1)[0]) * stdv ** 2,
                ship_design=[round(float(v), 10) for v in X[ship].tolist()])


# ---------------------------------------------------------------------------
# The floored variant's repeat ratings
# ---------------------------------------------------------------------------


def error_config(task: dict, T: int) -> sim.SimulationConfig:
    """Enough of a SimulationConfig for apply_sensor_error to reproduce the run's error process."""
    a = task["run_args"]
    if str(a.get("response_clip", "none")).strip() not in ("", "none"):
        raise NotImplementedError("the floored variant does not model a clipped response scale; "
                                  "drop the floor (--rules A) or extend error_config")
    std = float(task["jitter_std"])
    bias = task["bias"] if np.isfinite(task["bias"]) else (
        std if a.get("error_bias_mode", "scaled") == "scaled" else float(a.get("error_bias", 0.2)))
    spike = std if a.get("error_spike_std_mode", "scaled") == "scaled" else float(a.get("error_spike_std", 0.5))
    return sim.SimulationConfig(
        iterations=T, jitter_iteration=int(task["jitter_iteration"]), jitter_std=std,
        single_error=bool(a.get("single_error", False)), initial_samples=1, candidate_pool=1,
        objective="value", objective_columns=["value"], param_columns=["x0"], seed=int(task["seed"]),
        error_model=task["error_model"], error_bias=float(bias),
        error_spike_prob=float(a.get("error_spike_prob", 0.1)), error_spike_std=float(spike),
        dropout_strategy="hold_last", normalize_objective=False, objective_weights=None,
        acq_num_restarts=1, acq_raw_samples=1, acq_maxiter=1, acq_mc_samples=1, ref_point=None,
        error_ar1_rho=float(a.get("error_ar1_rho", 0.8)), response_round=a.get("response_round"),
    )


def repeat_ratings(frame: pd.DataFrame, task: dict) -> list[float]:
    """The first design's logged rating plus the simulated repeats, drawn at trials 2, 3, ...

    Drawn with the run's own error model in sequence after trial 1, so an AR(1)
    rater's repeats are as correlated as consecutive real ratings would be and
    a late onset leaves them exact. The stream is seeded like the run's jitter
    stream plus a tag -- common random numbers across landscapes, as in the driver.
    """
    first_true = float(frame["objective_true"].iloc[0])
    first_obs = float(frame["objective_observed"].iloc[0])
    ratings = [first_obs]
    if task["clean"]:
        return ratings + [first_true] * task["floor_repeats"]
    T = len(frame)
    config = error_config(task, T)
    rng = np.random.default_rng(np.random.SeedSequence([
        int(task["seed"]), sim.ACQUISITION_CHOICES.index(task["acquisition"]), int(task["jitter_iteration"]),
        int(round(float(task["jitter_std"]) * 1_000_000)), sim.ERROR_MODEL_CHOICES.index(task["error_model"]),
        REPEAT_STREAM_TAG]))
    prev_obs = np.array([first_obs])
    prev_err = np.array([float(frame["error_magnitude"].iloc[0]) if "error_magnitude" in frame else first_obs - first_true])
    for k in range(task["floor_repeats"]):
        obs, err = sim.apply_sensor_error(true_value=np.array([first_true]), iteration=2 + k, config=config,
                                          rng=rng, previous_observed=prev_obs, previous_error=prev_err)
        ratings.append(float(obs[0]))
        prev_obs, prev_err = obs, err
    return ratings


# ---------------------------------------------------------------------------
# One run (worker)
# ---------------------------------------------------------------------------


def replay_run(task: dict) -> dict:
    """Everything the rules need from one log, for the parts task asks for."""
    path = Path(task["path"])
    frame = pd.read_csv(path).sort_values("iteration").reset_index(drop=True)
    T = len(frame)
    if T != task["T"] or not np.array_equal(frame["iteration"].to_numpy(), np.arange(1, T + 1)):
        raise ValueError(f"{path.name}: {T} rows, expected iterations 1..{task['T']}")
    cols = str(frame["param_columns"].iloc[0]).split(",")
    X = torch.tensor(frame[cols].to_numpy(dtype=float), dtype=torch.double)
    y_obs = frame["objective_observed"].to_numpy(dtype=float)
    Y = torch.tensor(y_obs.reshape(-1, 1), dtype=torch.double)
    # What shipping a logged design really gets you (differs from objective_true
    # only under an unnoticed input slip).
    y_dep = frame["objective_true_deployed" if "objective_true_deployed" in frame else "objective_true"].to_numpy(dtype=float)
    y_opt = float(frame["y_opt"].iloc[0])
    spec = bb.BENCHMARKS[task["dataset"]]
    bounds = sim.Bounds(low=spec.bounds_low, high=spec.bounds_high).tensor
    rec: dict = {"file": path.name, "stamp": task["stamp"], "T": T, "fits": {}}

    if task["need_log"]:
        # The standard process replayed at every prefix: deploy the best rating so far.
        replay = np.array([y_opt - y_dep[int(np.argmax(y_obs[:t]))] for t in range(1, T + 1)])
        logged = frame["inference_simple_regret_true"].to_numpy(dtype=float)
        rec.update(y_opt=y_opt, regret_standard=logged.tolist(),
                   log_check_max_abs=float(np.max(np.abs(replay - logged))))

    if task["need_z"]:
        n0 = task["n0"]
        gp, ok, n_warn = fit_loop_gp(X[:n0], Y[:n0], bounds, f"{path.name}|n0={n0}")
        try:
            z = one_step_residuals(gp, X, Y, n0).tolist()
        except Exception:  # a factorisation failure leaves the monitor silent, flagged
            z, ok = [float("nan")] * (T - n0), False
        rec.update(z=z, z_fit_ok=bool(ok), z_n_warn=int(n_warn),
                   z_noise_sd=float(gp.likelihood.noise.detach().reshape(-1)[0].sqrt() * gp.outcome_transform.stdvs.reshape(-1)[0]))

    base: dict[int, dict] = {}

    def fit_prefix(m: int) -> dict:
        if m not in base:
            gp, ok, n_warn = fit_loop_gp(X[:m], Y[:m], bounds, f"{path.name}|m={m}")
            base[m] = {**ship_record(gp, X, m, y_dep, y_opt, ok, n_warn), "floored": False}
        return base[m]

    floor_sd = None
    for key in task["fits"]:
        if not key.endswith("f"):
            rec["fits"][key] = fit_prefix(int(key))
            continue
        m = int(key[:-1])
        if floor_sd is None:
            ratings = repeat_ratings(frame, task)
            floor_sd = float(np.std(ratings, ddof=1))
            rec["floor_sd"], rec["floor_ratings"] = floor_sd, ratings
        learned = fit_prefix(m)
        if floor_sd > 0 and learned["noise_var"] < floor_sd ** 2:
            # With the floor binding, the constrained noise optimum sits on the
            # floor, so fixing the noise there and refitting the kernel is the
            # floored fit.
            gp, ok, n_warn = fit_loop_gp(X[:m], Y[:m], bounds, f"{path.name}|m={m}|floor",
                                         train_Yvar=torch.full((m, 1), floor_sd ** 2, dtype=torch.double))
            rec["fits"][key] = {**ship_record(gp, X, m, y_dep, y_opt, ok, n_warn), "floored": True}
        else:
            rec["fits"][key] = {**learned, "floored": False}
    for m, fit in base.items():
        rec["fits"].setdefault(str(m), fit)
    return rec


# ---------------------------------------------------------------------------
# Cache and pool
# ---------------------------------------------------------------------------


class ReplayCache:
    """Per-run replay records, one JSONL per landscape, merged on load.

    Written only by the main process as workers finish, so an interrupted
    analysis resumes where it stopped. A record whose log changed since (size or
    mtime) is ignored.
    """

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        for path in sorted(root.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._merge(json.loads(line))

    def _merge(self, rec: dict) -> None:
        cur = self.records.get(rec["file"])
        if cur is None or cur["stamp"] != rec["stamp"]:
            self.records[rec["file"]] = {**rec, "fits": dict(rec.get("fits", {}))}
            return
        fits = {**cur["fits"], **rec.get("fits", {})}
        cur.update({k: v for k, v in rec.items() if k != "fits"})
        cur["fits"] = fits

    def add(self, rec: dict, dataset: str) -> None:
        self._merge(rec)
        with open(self.root / f"{dataset}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def missing(self, run, need_log: bool, need_z: bool, keys: list[str]) -> tuple[bool, bool, list[str]]:
        rec = self.records.get(run.file)
        if rec is None or rec["stamp"] != run.stamp:
            return need_log, need_z, list(keys)
        return (need_log and "regret_standard" not in rec, need_z and "z" not in rec,
                [k for k in keys if k not in rec["fits"]])

    def fit(self, file: str, key) -> dict:
        return self.records[file]["fits"][str(key)]


def run_pool(tasks: list[dict], workers: int, cache: ReplayCache, label: str) -> None:
    if not tasks:
        print(f"{label}: nothing to compute (cached)")
        return
    n_fits = sum(len(t["fits"]) + int(t["need_z"]) for t in tasks)
    print(f"{label}: {len(tasks)} runs, <= {n_fits} GP fits, {workers} worker(s)", flush=True)
    start, every = time.perf_counter(), max(1, len(tasks) // 20)

    def done(i: int, rec: dict, dataset: str) -> None:
        cache.add(rec, dataset)
        if (i + 1) % every == 0 or i + 1 == len(tasks):
            el = time.perf_counter() - start
            print(f"  {i + 1}/{len(tasks)} runs, {el:.0f}s elapsed, ~{el / (i + 1) * (len(tasks) - i - 1):.0f}s left",
                  flush=True)

    if workers <= 1:
        for i, task in enumerate(tasks):
            done(i, replay_run(task), task["dataset"])
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(replay_run, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            done(i, future.result(), futures[future]["dataset"])


def make_task(run, T: int, cfg: dict, run_args: dict, need_log: bool, need_z: bool, keys: list[str]) -> dict:
    return dict(path=run.path, file=run.file, dataset=run.dataset, stamp=run.stamp, T=T, n0=cfg["n0"],
                need_log=need_log, need_z=need_z, fits=keys, clean=bool(run.clean), error_model=run.error_model,
                jitter_std=float(run.jitter_std),
                jitter_iteration=-1 if run.clean else int(run.jitter_iteration),
                seed=int(run.seed), acquisition=run.acquisition, bias=float(run.bias),
                floor_repeats=cfg["floor_repeats"], run_args=run_args)


def ensure(runs: pd.DataFrame, cache: ReplayCache, cfg: dict, run_args: dict, T: int, workers: int, label: str,
           keys_of, need_log: bool = False, need_z: bool = False) -> None:
    tasks = []
    for run in runs.itertuples():
        nl, nz, keys = cache.missing(run, need_log, need_z, sorted(set(keys_of(run)), key=lambda k: (int(k.rstrip("f")), k)))
        if nl or nz or keys:
            tasks.append(make_task(run, T, cfg, run_args, nl, nz, keys))
    run_pool(tasks, workers, cache, label)


# ---------------------------------------------------------------------------
# Rule A
# ---------------------------------------------------------------------------


def cusum_paths(z2: np.ndarray, kappa: float) -> np.ndarray:
    """S_t for every run (rows) and monitored trial (columns); S before the first is 0."""
    S = np.zeros_like(z2)
    s = np.zeros(z2.shape[0])
    for j in range(z2.shape[1]):
        s = np.maximum(0.0, s + z2[:, j] - 1.0 - kappa)
        S[:, j] = s
    return S


def first_alarm(S: np.ndarray, h: float, n0: int) -> np.ndarray:
    """Trial number of the first S_t > h per run, 0 for none. Column j is trial n0 + 1 + j."""
    if not np.isfinite(h):
        return np.zeros(S.shape[0], dtype=int)
    hit = S > h
    return np.where(hit.any(axis=1), n0 + 1 + hit.argmax(axis=1), 0).astype(int)


def z_matrix(runs: pd.DataFrame, cache: ReplayCache) -> np.ndarray:
    # A monitor whose fit failed stays silent rather than alarming on garbage.
    z = np.array([cache.records[f]["z"] for f in runs["file"]], dtype=float)
    return np.nan_to_num(z, nan=0.0) ** 2


def _lmean(values: np.ndarray, codes: np.ndarray) -> float:
    """Mean of per-landscape means: every landscape weighs the same."""
    return float((np.bincount(codes, values) / np.bincount(codes)).mean())


def grid_table(runs: pd.DataFrame, mask: np.ndarray, alarms: dict, combos: list[tuple], cache: ReplayCache,
               T: int, max_fa: float, scope: str) -> pd.DataFrame:
    """Rule A's false alarms, trials and deployed regret for each (h, kappa, w) on the masked runs."""
    sub = runs[mask]
    pos = np.flatnonzero(mask)
    codes = pd.factorize(sub["dataset"])[0]
    opt_z = sub["opt_z"].to_numpy()
    clean = sub["clean"].to_numpy(bool)
    onset = sub["jitter_iteration"].to_numpy(float)
    onset0 = ~clean & (onset == 0)
    late = ~clean & (onset > 0)
    files = sub["file"].tolist()
    ship_T = np.array([cache.fit(f, T)["regret"] for f in files])
    cells = (sub["error_model"] + "|" + sub["jitter_std"].astype(str)).to_numpy()[onset0]
    rows = []
    for h, kappa, w in [(NEVER, np.nan, 0)] + list(combos):
        t = np.zeros(len(sub), dtype=int) if not np.isfinite(h) else alarms[(h, kappa)][pos]
        regret = ship_T.copy()
        for i in np.flatnonzero(t > 0):
            regret[i] = cache.fit(files[i], int(t[i]) - w)["regret"]
        fired = t > 0
        fa_clean = float(fired[clean].mean()) if clean.any() else np.nan
        if scope == "cell" and onset0.any():
            fa0 = float(pd.Series(fired[onset0]).groupby(cells).mean().max())
        else:
            fa0 = float(fired[onset0].mean()) if onset0.any() else np.nan
        rows.append(dict(
            h=h, kappa=kappa, w=w, n_runs=int(len(sub)),
            false_alarm_clean=fa_clean, false_alarm_onset0=fa0,
            pre_onset_alarm_late=float((fired & (t <= onset))[late].mean()) if late.any() else np.nan,
            detection_rate_late=float((t > onset)[late].mean()) if late.any() else np.nan,
            mean_trials_used=_lmean(np.where(fired, t, T).astype(float), codes),
            mean_regret_norm=_lmean(regret / opt_z, codes),
            mean_regret_norm_ship_T=_lmean(ship_T / opt_z, codes),
            feasible=bool((np.isnan(fa_clean) or fa_clean <= max_fa) and (np.isnan(fa0) or fa0 <= max_fa)),
        ))
    return pd.DataFrame(rows)


def choose_params(grid: pd.DataFrame, objective: str) -> dict:
    """The tuned (h, kappa, w). Never stopping is always a feasible candidate, so this cannot fail.

    Ties go to the more conservative setting (larger h, larger kappa, smaller w):
    two settings that behave identically on the tuning seeds are not identical
    on new ones, and the larger threshold is the one less likely to fire falsely.
    """
    feas = grid[grid["feasible"]].copy()
    never = float(grid.loc[~np.isfinite(grid["h"]), "mean_regret_norm"].iloc[0])
    feas["neg_h"], feas["neg_kappa"] = -feas["h"], -feas["kappa"].fillna(np.inf)
    if objective == "trials":
        feas = feas[feas["mean_regret_norm"] <= never + 1e-12]
        order = ["mean_trials_used", "mean_regret_norm", "neg_h", "neg_kappa", "w"]
    else:
        order = ["mean_regret_norm", "mean_trials_used", "neg_h", "neg_kappa", "w"]
    best = feas.sort_values(order, kind="mergesort").iloc[0]
    return dict(h=float(best["h"]), kappa=float(best["kappa"]) if np.isfinite(best["h"]) else 0.0, w=int(best["w"]),
                objective=objective, tune_row=best.drop(["neg_h", "neg_kappa"]).to_dict())


# ---------------------------------------------------------------------------
# Rule B
# ---------------------------------------------------------------------------


def rule_b_stop(fits: list[dict], checkpoints: list[int], delta: float, units: str) -> int | None:
    """The checkpoint rule B stops at, or None. fits[i] is the refit at checkpoints[i]."""
    for prev, cur, t in zip(fits, fits[1:], checkpoints[1:]):
        if cur["ship_design"] != prev["ship_design"]:
            continue
        scale = cur["y_std"] if units == "model" else 1.0
        if (cur["max_ucb"] - cur["lcb_ship"]) / scale < delta:
            return t
    return None


def b_variants(cfg: dict) -> list[tuple[str, float, str, bool]]:
    return [(f"B_d{d:g}_{u}" + ("_floor" if floor else ""), d, u, floor)
            for d in cfg["deltas"] for u in cfg["delta_units"] for floor in (False, True)]


def b_keys(cfg: dict, T: int) -> list[str]:
    cps = cfg["checkpoints"]
    return [str(c) for c in cps] + [f"{c}f" for c in cps] + [f"{T - cfg['floor_repeats']}f"]


# ---------------------------------------------------------------------------
# Per-run scores and summaries
# ---------------------------------------------------------------------------


def per_run_table(runs: pd.DataFrame, cache: ReplayCache, cfg: dict, T: int, a_params: dict | None,
                  a_alarm: np.ndarray | None) -> pd.DataFrame:
    rows = []
    for i, run in enumerate(runs.itertuples()):
        rec = cache.records[run.file]
        std_regret = rec["regret_standard"]
        ship_T = cache.fit(run.file, T)
        onset = np.nan if run.clean else float(run.jitter_iteration)
        base = dict(split=run.split, dataset=run.dataset, acquisition=run.acquisition, seed=run.seed,
                    error_model=run.error_model, jitter_std=run.jitter_std, jitter_iteration=onset,
                    opt_z=run.opt_z, T=T, regret_standard_T=std_regret[T - 1], regret_ship_T=ship_T["regret"],
                    file=run.file)

        def add(rule, stop_t, trials, fit, **extra):
            stopped = stop_t is not None
            rows.append({**base, "rule": rule, "stopped": stopped, "stop_t": stop_t if stopped else np.nan,
                         "trials_used": trials, "trials_saved": T - trials, "regret": fit["regret"],
                         "regret_best_observed_at_stop": std_regret[(stop_t if stopped else T) - 1],
                         "ship_idx": fit["ship_idx"], "fit_ok": fit["ok"], **extra})

        add("ship_T", None, T, ship_T)
        if a_params is not None:
            t = int(a_alarm[i])
            fit = cache.fit(run.file, t - a_params["w"]) if t else ship_T
            false_alarm = bool(t) and (run.clean or onset == 0 or t <= onset)
            delay = t - onset if (t and not run.clean and onset > 0 and t > onset) else np.nan
            add("A", t or None, t or T, fit, h=a_params["h"], kappa=a_params["kappa"], w=a_params["w"],
                false_alarm=false_alarm, detection_delay=delay, z_fit_ok=rec.get("z_fit_ok"))
        if "B" in cfg["rules"]:
            cps, rep = cfg["checkpoints"], cfg["floor_repeats"]
            for name, delta, units, floor in b_variants(cfg):
                fits = [cache.fit(run.file, f"{c}f" if floor else c) for c in cps]
                stop = rule_b_stop(fits, cps, delta, units)
                if floor:
                    # The repeats are trials: t + rep at a stop, and without one the
                    # study ships at T - rep so the total is still T.
                    fit = fits[cps.index(stop)] if stop else cache.fit(run.file, f"{T - rep}f")
                    trials = stop + rep if stop else T
                else:
                    fit = fits[cps.index(stop)] if stop else ship_T
                    trials = stop if stop else T
                add(name, stop, trials, fit, delta=delta, delta_units=units, floor=floor,
                    floor_sd=rec.get("floor_sd") if floor else np.nan, ship_floored=fit.get("floored", False))
    out = pd.DataFrame(rows)
    for col in ("regret", "regret_standard_T", "regret_ship_T"):
        out[f"{col}_norm"] = out[col] / out["opt_z"]
    return out


def _landscape_mean(block: pd.DataFrame, col: str) -> float:
    return float(block.groupby("dataset")[col].mean().mean())


def trial_stats(block: pd.DataFrame) -> dict:
    stats = dict(
        n_runs=int(len(block)), n_landscapes=int(block["dataset"].nunique()),
        mean_trials_used=float(block["trials_used"].mean()), mean_trials_saved=float(block["trials_saved"].mean()),
        median_trials_saved=float(block["trials_saved"].median()), stop_rate=float(block["stopped"].mean()),
        mean_regret_norm=_landscape_mean(block, "regret_norm"),
        mean_regret_standard_T_norm=_landscape_mean(block, "regret_standard_T_norm"),
        mean_regret_ship_T_norm=_landscape_mean(block, "regret_ship_T_norm"),
    )
    if "false_alarm" in block and block["false_alarm"].notna().any():
        onset = block["jitter_iteration"]
        late = onset.notna() & (onset > 0)
        stats.update(
            false_alarm_rate=float(block["false_alarm"].astype(float).mean()),
            detection_rate_late=float((block["stopped"] & (block["stop_t"] > onset))[late].mean()) if late.any() else np.nan,
            median_detection_delay=float(block["detection_delay"].median()) if block["detection_delay"].notna().any() else np.nan,
        )
    return stats


def summarise_rules(per_run: pd.DataFrame) -> pd.DataFrame:
    """Recovery against the standard process, trials saved and stop/false-alarm rates per cell."""
    out = []
    for (split, rule), rows in per_run.groupby(["split", "rule"], sort=False):
        clean =rows[rows["error_model"] == "none"]
        noisy = rows[rows["error_model"] != "none"]
        twin = clean[PAIR + ["regret", "regret_standard_T", "regret_ship_T"]].rename(
            columns={"regret": "regret_clean", "regret_standard_T": "regret_standard_T_clean",
                     "regret_ship_T": "regret_ship_T_clean"})
        paired = noisy.merge(twin, on=PAIR, how="inner", validate="many_to_one")
        z = paired["opt_z"]
        paired = paired.assign(ref_noisy=paired["regret_standard_T"] / z, ref_clean=paired["regret_standard_T_clean"] / z,
                               trt_noisy=paired["regret"] / z, trt_clean=paired["regret_clean"] / z,
                               gain_ship=(paired["regret_ship_T"] - paired["regret"]) / z,
                               price_ship=(paired["regret_clean"] - paired["regret_ship_T_clean"]) / z)
        head = dict(split=split, rule=rule)
        out.append({**head, "error_model": "none", "jitter_std": 0.0, "jitter_iteration": np.nan,
                    "level": "clean", **trial_stats(clean)})

        def cell(block: pd.DataFrame, labels: dict, level: str) -> dict:
            return {**head, **labels, "level": level, **trial_stats(block),
                    **adapt.summarise(block, np.random.default_rng(adapt.BOOTSTRAP_SEED)),
                    # What stopping adds beyond shipping by the same rule at T.
                    "gain_vs_ship_T": _landscape_mean(block, "gain_ship"),
                    "price_vs_ship_T": _landscape_mean(block, "price_ship")}

        cells = [cell(block, dict(zip(CELL, key)), "cell") for key, block in paired.groupby(CELL)]
        for r, q in zip(cells, multipletests([r["wilcoxon_p"] for r in cells], method="fdr_bh")[1]):
            r["wilcoxon_p_fdr"] = float(q)
        out.extend(cells)
        for model, block in paired.groupby("error_model"):
            out.append(cell(block, dict(error_model=model, jitter_std=np.nan, jitter_iteration=np.nan), "model"))
        out.append(cell(paired, dict(error_model="pooled", jitter_std=np.nan, jitter_iteration=np.nan), "pooled"))
    return pd.DataFrame(out)


def print_summary(summary: pd.DataFrame, split: str) -> None:
    block = summary[summary["split"] == split]
    if block.empty:
        return
    print(f"\n=== {split} seeds: recovery of the standard process's deployed cost, trials saved ===")
    print("(recovered = gain/cost [landscape bootstrap 95%]; saved = mean/median trials; "
          "stop% ; FA% = false alarms (rule A); shipT = gain beyond the ship rule at T, opt_z)")
    for rule, rows in block.groupby("rule", sort=False):
        print(f"\n--- {rule} ---")
        for r in rows.itertuples():
            if r.level == "clean":
                fa = f"  FA {r.false_alarm_rate:4.0%}" if hasattr(r, "false_alarm_rate") and pd.notna(getattr(r, "false_alarm_rate", np.nan)) else ""
                print(f"  clean runs         saved {r.mean_trials_saved:5.1f}/{r.median_trials_saved:4.0f}  "
                      f"stop {r.stop_rate:4.0%}{fa}  regret {r.mean_regret_norm:.3f} vs standard "
                      f"{r.mean_regret_standard_T_norm:.3f}")
                continue
            if r.level == "cell":
                label = f"{r.error_model:>8s} {r.jitter_std:>4g} it.{int(r.jitter_iteration) + 1:<3d}"
            else:
                label = f"{r.error_model:>8s} all"
            fa = getattr(r, "false_alarm_rate", np.nan)
            fa_txt = f"  FA {fa:4.0%}" if pd.notna(fa) else ""
            print(f"  {label:19s} recovered {r.recovered:+6.0%} [{r.recovered_lo:+.0%}, {r.recovered_hi:+.0%}]  "
                  f"saved {r.mean_trials_saved:5.1f}/{r.median_trials_saved:4.0f}  stop {r.stop_rate:4.0%}{fa_txt}  "
                  f"cost {r.cost:.3f} price {r.price:+.3f} shipT {r.gain_vs_ship_T:+.3f}")


# ---------------------------------------------------------------------------
# Cross-check against the evaluation outputs
# ---------------------------------------------------------------------------


def evaluation_check(input_dir: Path, runs: pd.DataFrame, cache: ReplayCache, T: int) -> dict:
    """The standard process's deployed regret at T against evaluate_research_question's table."""
    worst, matched = 0.0, 0
    for dataset, block in runs.groupby("dataset"):
        path = input_dir / dataset / "evaluation" / "paired_excess_metrics.csv"
        if not path.is_file():
            continue
        ev = pd.read_csv(path, usecols=["acquisition", "seed", "error_model", "jitter_std", "jitter_iteration",
                                        "final_inference_simple_regret_true_jitter",
                                        "final_inference_simple_regret_true_baseline"])
        noisy = block[~block["clean"]].copy()
        noisy["mine"] = [cache.records[f]["regret_standard"][T - 1] for f in noisy["file"]]
        clean = block[block["clean"]].copy()
        clean["mine_clean"] = [cache.records[f]["regret_standard"][T - 1] for f in clean["file"]]
        noisy = noisy.merge(clean[["acquisition", "seed", "mine_clean"]], on=["acquisition", "seed"])
        for frame in (noisy, ev):
            frame["std_key"] = frame["jitter_std"].round(6)
            frame["jitter_iteration"] = frame["jitter_iteration"].astype(int)
        m = noisy.merge(ev, on=["acquisition", "seed", "error_model", "std_key", "jitter_iteration"])
        if len(m):
            matched += len(m)
            worst = max(worst, float((m["mine"] - m["final_inference_simple_regret_true_jitter"]).abs().max()),
                        float((m["mine_clean"] - m["final_inference_simple_regret_true_baseline"]).abs().max()))
    return dict(evaluation_rows_matched=matched, evaluation_max_abs_diff=worst)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = resolve_config(args)
    t_start = time.perf_counter()
    out = args.output_dir or (args.input_dir / "analysis" / "stopping")
    out.mkdir(parents=True, exist_ok=True)
    stats = bb.load_stats(args.stats_path)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    run_args = read_run_args(args.input_dir)
    if args.functions.strip() == "all":
        functions = sorted(p.name for p in args.input_dir.iterdir() if p.is_dir() and p.name in bb.BENCHMARKS)
    else:
        functions = [f.strip() for f in args.functions.split(",") if f.strip()]
        unknown = [f for f in functions if f not in bb.BENCHMARKS or f not in opt_z]
        if unknown:
            raise ValueError(f"unknown landscapes or no opt_z: {unknown}")
    stds = None if args.jitter_stds.strip() == "all" else _floats(args.jitter_stds)
    onsets = None if args.onsets.strip() == "all" else _ints(args.onsets)

    runs = index_runs(args.input_dir, functions, cfg, stds, onsets, opt_z)
    T = int(run_args.get("iterations") or len(pd.read_csv(runs["path"].iloc[0], usecols=["iteration"])))
    check_budget(cfg, T)
    print(f"{len(runs)} logs ({int(runs['clean'].sum())} clean) from {runs['dataset'].nunique()} landscapes, "
          f"T = {T}; splits {runs['split'].value_counts().to_dict()}")
    if "A" in cfg["rules"] and args.a_params_from is None:
        tune = runs[runs["split"] == "tune"]
        if not tune["clean"].any() or not (tune["jitter_iteration"] == 0).any():
            raise ValueError("tuning rule A needs clean runs and onset-0 runs on the tuning seeds "
                             "(the false-alarm constraint is defined on both)")

    cache = ReplayCache(out / f"cache_n0{cfg['n0']}_rep{cfg['floor_repeats']}")
    keys1 = [str(T)] + (b_keys(cfg, T) if "B" in cfg["rules"] else [])
    ensure(runs, cache, cfg, run_args, T, args.workers, "pass 1 (logs, residuals, checkpoint fits)",
           lambda run: keys1, need_log=True, need_z="A" in cfg["rules"])
    bad = [f for f in runs["file"] if cache.records[f]["log_check_max_abs"] > 1e-9]
    if bad:
        raise SystemExit(f"the replayed standard process disagrees with the logged inference regret in "
                         f"{len(bad)} logs, e.g. {bad[:3]}; these are not standard-process runs")
    metadata: dict = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                          config=cfg, T=T, n_logs=int(len(runs)),
                          log_check_max_abs=float(max(cache.records[f]["log_check_max_abs"] for f in runs["file"])))

    a_params, a_alarm, grids = None, None, []
    if "A" in cfg["rules"]:
        z2 = z_matrix(runs, cache)
        paths = {k: cusum_paths(z2, k) for k in cfg["kappa_grid"]}
        alarms = {(h, k): first_alarm(paths[k], h, cfg["n0"]) for k in cfg["kappa_grid"] for h in cfg["h_grid"]}
        combos = [(h, k, w) for (h, k) in alarms for w in cfg["w_grid"]]

        def prefix_keys(mask: np.ndarray, settings: list[tuple]):
            need: dict[str, set] = {f: set() for f in runs.loc[mask, "file"]}
            for h, k, w in settings:
                t = first_alarm(cusum_paths(z2, k), h, cfg["n0"]) if (h, k) not in alarms else alarms[(h, k)]
                for i in np.flatnonzero(mask & (t > 0)):
                    need[runs.at[i, "file"]].add(str(int(t[i]) - w))
            return lambda run: need.get(run.file, set())

        tune_mask = (runs["split"] == "tune").to_numpy()
        score_mask = (runs["split"] == "score").to_numpy()
        if args.a_params_from is not None:
            loaded = json.loads(Path(args.a_params_from).read_text(encoding="utf-8"))
            a_params = dict(h=float(loaded["h"]), kappa=float(loaded["kappa"]), w=int(loaded["w"]),
                            objective=loaded.get("objective"), source=str(args.a_params_from))
            if a_params["w"] > cfg["n0"] - 2:
                raise ValueError(f"loaded w = {a_params['w']} exceeds n0 - 2")
        else:
            ensure(runs[tune_mask], cache, cfg, run_args, T, args.workers, "pass 2 (rule A grid refits, tuning seeds)",
                   prefix_keys(tune_mask, combos))
            grid = grid_table(runs, tune_mask, alarms, combos, cache, T, cfg["max_false_alarm"], args.false_alarm_scope)
            grids.append(grid.assign(split="tune"))
            a_params = choose_params(grid, args.tune_objective)
        chosen = [(a_params["h"], a_params["kappa"], a_params["w"])] if np.isfinite(a_params["h"]) else []
        a_alarm = (first_alarm(cusum_paths(z2, a_params["kappa"]), a_params["h"], cfg["n0"])
                   if chosen else np.zeros(len(runs), dtype=int))
        score_settings = combos if args.grid_on_score_seeds else chosen
        # Tuning seeds need the chosen setting too when parameters were loaded.
        ensure(runs, cache, cfg, run_args, T, args.workers, "pass 3 (rule A refits, tuned setting)",
               prefix_keys(np.ones(len(runs), bool), chosen))
        if score_mask.any():
            ensure(runs[score_mask], cache, cfg, run_args, T, args.workers, "pass 3b (rule A grid, score seeds)",
                   prefix_keys(score_mask, score_settings))
            local = dict(alarms)
            if chosen and chosen[0][:2] not in local:
                local[chosen[0][:2]] = a_alarm
            grids.append(grid_table(runs, score_mask, local, score_settings, cache, T, cfg["max_false_alarm"],
                                    args.false_alarm_scope).assign(split="score"))
        (out / "stopping_tuned_A.json").write_text(json.dumps(a_params, indent=2, default=float), encoding="utf-8")
        metadata["rule_A"] = a_params
        metadata["rule_A_monitor_fit_failures"] = int(sum(not cache.records[f].get("z_fit_ok", True) for f in runs["file"]))
        print(f"\nrule A tuned ({a_params.get('objective') or 'loaded'}): h = {a_params['h']:g}, "
              f"kappa = {a_params['kappa']:g}, w = {a_params['w']}")
        if grids and "tune_row" in a_params:
            g = grids[0]
            print("  best feasible settings on the tuning seeds:")
            show = g[g["feasible"]].sort_values(["mean_regret_norm", "mean_trials_used"]).head(6)
            print(show[["h", "kappa", "w", "false_alarm_clean", "false_alarm_onset0", "pre_onset_alarm_late",
                        "detection_rate_late", "mean_trials_used", "mean_regret_norm",
                        "mean_regret_norm_ship_T"]].to_string(index=False))

    per_run = per_run_table(runs, cache, cfg, T, a_params, a_alarm)
    summary = summarise_rules(per_run)
    per_run.to_csv(out / "stopping_per_run.csv", index=False)
    summary.to_csv(out / "stopping_summary.csv", index=False)
    if grids:
        pd.concat(grids, ignore_index=True).to_csv(out / "stopping_grid_A.csv", index=False)
    metadata["ship_fit_failures"] = int((~per_run["fit_ok"].astype(bool)).sum())
    if not args.no_evaluation_check:
        metadata.update(evaluation_check(args.input_dir, runs, cache, T))
        print(f"\nstandard process at T vs evaluation/paired_excess_metrics.csv: "
              f"{metadata['evaluation_rows_matched']} rows, max |diff| {metadata['evaluation_max_abs_diff']:.2e}")
    print(f"replayed best-rating deployment vs logged inference regret, every prefix: max |diff| "
          f"{metadata['log_check_max_abs']:.2e}; ship-rule fit failures {metadata['ship_fit_failures']}")
    for split in ("tune", "score"):
        print_summary(summary, split)
    metadata["runtime_sec"] = float(time.perf_counter() - t_start)
    (out / "stopping_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out / 'stopping_summary.csv'}, stopping_per_run.csv"
          + (", stopping_grid_A.csv, stopping_tuned_A.json" if grids or a_params else "")
          + f" ({metadata['runtime_sec']:.0f}s)")


if __name__ == "__main__":
    main()
