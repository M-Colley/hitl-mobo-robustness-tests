"""The noise-aware ship rules, with the rater's noise variance given to the final refit.

Question
--------
Shipping the visited design with the highest posterior mean less one or two latent SDs (LCB1, LCB2; PM
subtracts none) after all T = 50 trials costs no trial, but the rules pay a price without error: 0.021-0.023
of the achievable improvement for LogEI and qNEI, 0.015 over the ten acquisitions. The price comes from the
final refit's learned noise term, which reads a rugged landscape's roughness as rating noise. This check
refits the final GP of every logged run with a FIXED observation variance (BoTorch's train_Yvar) that a study
could know or measure, ships PM, LCB1 and LCB2 from that refit, and scores them with the process-change
estimand of analyse_boba_adaptations.py against the standard process (the single best rating). Nothing is
re-simulated except, in the two inside-the-budget levels, the three repeat ratings.

The refit is rescore_ship_rules.fit_loop_gp, the loop's own surrogate (BoTorch's default SingleTaskGP,
Normalize over the landscape's box, Standardize(m=1)), on every logged trial, torch seeded per file
(rescore_ship_rules.file_seed), and every rule picks among the visited designs (ties to the earliest trial).
train_Yvar is given in rating units; Standardize divides it by the ratings' variance itself, so a study needs
the noise variance of its ratings and no estimate of the objective's spread. On these landscapes rating units
are landscape SDs, because each objective is standardised over its box.

Information levels, and exactly what train_Yvar holds at trial t = 1..T (t0 = the onset, sigma_e = the swept
magnitude, FLOOR = 1e-6, the simulator's floor for an exact rating). The clean twin of every level except
'learned' gets FLOOR on every trial: a rater known, or measured, to be exact has variance zero. The two
inside-the-budget levels apply their whole procedure to the clean twin (three exact repeats, estimate zero).

  learned        None: the loop's learned noise term. This is rescore_ship_rules.py's own refit, and it must
                 reproduce that script's ship_rules_per_run.csv (regret of best_observed, PM, LCB1, LCB2) to
                 1e-9 in every run or the check stops: the control.
  mag            sigma_e^2 on every trial: the magnitude known, the onset not. For gaussian, bias and drift
                 sigma_e^2 is the variance of the fresh draw eps_t (the bias offset and the drift ramp are
                 mean shifts that no variance describes); for AR(1) it is the stationary marginal variance of
                 e_t (the innovation's is sigma_e^2 (1 - rho^2) = 0.36 sigma_e^2). Exactly the post-onset value
                 of bo_sensor_error_simulation.known_noise_variance, which the known-noise arm handed its GP.
  mag_onset      FLOOR for t <= t0, sigma_e^2 after: the magnitude and the onset known, the known-noise arm's
                 information (known_noise_variance trial by trial; at t0 = 0 the same vector as mag).
  mag_half       (sigma_e / 2)^2 on every trial: the SD understated twofold (a sensitivity).
  mag_double     (2 sigma_e)^2 on every trial: the SD overstated twofold (a sensitivity).
  retest3        sigma_hat^2 on every trial, from r = 3 test-retest pairs rated in one calibration sitting
                 after the study (six ratings outside the 50-trial budget). Under the paper's shared-sitting
                 model the shared error cancels within a sitting, so a pair's difference carries twice the
                 idiosyncratic variance s^2 (replay_end_of_study.idiosyncratic_sd: sigma_e for gaussian, bias
                 and drift, sigma_e sqrt(1 - rho^2) for AR(1), whose persistent part a quick retest cannot
                 see). The pairs are simulated, d_i = e_i1 - e_i2 with e ~ N(0, s^2), and sigma_hat^2 =
                 sum_i d_i^2 / (2 r), distributed as s^2 chi2_3 / 3; drawn from a generator seeded by the run's
                 file name and RETEST_TAG.
  cusum          FLOOR for t <= a - w and sigma_e^2 after, where a is the alarm trial of the freeze rule's
                 detector (replay_stopping.py rule A at its tuned point, stopping_tuned_A.json: kappa 16, h 2000,
                 w 3; stopping_per_run.csv) and a - w the last trial the freeze rule itself treats as clean;
                 sigma_e^2 on every trial when the detector did not alarm: the magnitude known, the onset
                 estimated. LogEI and qNEI on the main sweep only, the runs the detector was replayed on.
  retest3_cusum  as cusum with sigma_hat^2 of retest3 in place of sigma_e^2: the deployable pipeline.
  inside_random  inside the budget: the study searches T - 3 = 47 trials (the logged prefix, exact, since a
                 proposal depends only on the trials before it) and spends trials 48-50 re-rating three of the
                 47 visited designs drawn at random (seeded by the file name and INSIDE_CHOICE_TAG). The
                 repeats are drawn from the run's own error process in sequence, as the simulator defines it
                 (bo_sensor_error_simulation.apply_sensor_error at iterations 48, 49, 50, the AR(1) state
                 carried on from trial 47's logged error, the drift ramp at those trials, the bias offset), from
                 a stream seeded like replay_stopping.repeat_ratings with INSIDE_NOISE_TAG. sigma_hat^2 =
                 sum_i (repeat_i - original_i)^2 / 6. The final refit uses all 50 ratings with sigma_hat^2 on
                 every one and ships among the 47 visited designs. The standard process it is scored against
                 keeps all 50 search trials, so its price includes the three lost search trials. A pair spans
                 time, so under bias from trial 21, drift and AR(1) the estimate also carries the change in the
                 shared error between the two ratings.
  inside_top     as inside_random, re-rating the top three distinct designs by LCB1 of the learned-noise
                 refit on the 47-trial prefix (the final sitting's candidate rule), best first.

Estimand (analyse_boba_adaptations.py, through analyse_ship_rules.twin_frame and aba.paired_frame /
aba.summarise / aba.share_flags, imported): per paired run, regret / opt_z,
    cost  = std_noisy - std_clean,  gain = std_noisy - rule_noisy,  price = rule_clean - std_clean,
landscape means, mean over landscapes; recovered = gain / cost as a ratio of landscape means with a
landscape-bootstrap interval (a fresh generator seeded aba.BOOTSTRAP_SEED per summary, as
analyse_ship_rules.py, so the learned rows reproduce its published rows); a share is withheld where the
reference cost is below 0.01 of opt_z or negative on a landscape (share_flags). Gain and price intervals and
the paired increments come from generators of their own (EXTRA_BOOTSTRAP_SEED), so no summary value depends
on them. The landscapes share their random numbers by design; the bootstrap treats them as independent.

Scopes: LogEI and qNEI under the four processes (gaussian, bias, drift, AR(1)) on the main sweep, seeds 7-11,
12-16 and 7-16, cell by cell and pooled, and each process alone; gaussian error, LogEI and qNEI, on seeds
7-16 and on the untouched seeds 27-36 (output-boba-confirmatory); the cautious rule's scope, ten acquisitions
under gaussian error, seeds 7-16, 12-16 and 27-36, pooled over 0.25 sigma and above. Every choice is made on
seeds 7-11 and scored on seeds 12-16: per cell, the rule (PM, LCB1, LCB2) with the largest gain on seeds 7-11
for each level, and one rule for all cells by the gain pooled over the eight cells, in the four-process scope
and, scored on seeds 12-16 and on the untouched seeds 27-36, in the gaussian scope; between the two
inside-the-budget protocols; and between the two outside-the-budget pipelines (the retest alone or with the
detector's onset). This analysis was built after the paper's other end-of-study results: it is post hoc, and
it is outside the paper's Holm families (heldout_remedies.py's 27 and 41 procedures, the nine k of
sitting_by_magnitude.py). The Holm values printed here are over the eight cells of one level, a family of this
check only.

Two comparisons on the same runs: the final sitting at the k seeds 7-11 chose in the cell and scope
(sitting_by_magnitude_selection.json and sitting_by_magnitude_gaussian_selection.json; the sitting's per-run
regrets from end_of_study_ksweep and _kwide, joined by file name, the standard process checked to 1e-9),
against each level's rule chosen on seeds 7-11 in the same cell and scope; and the known-noise arm (output-boba-knownnoise,
whose GP was told the true variance during the whole run, LogEI and qNEI, gaussian, seeds 7-11), refitted by
rescore_ship_rules.rescore_frame with its logged variance and scored against the MAIN sweep's standard process,
beside mag_onset (the same information given to the final refit alone). For that arm the check also confirms
that its logged known_noise_var equals the mag_onset vector trial by trial.

Outputs. Refits are cached per arm in output-boba/analysis/review/known_variance_ship/per_run/<arm>.jsonl
(git-ignored), so a rerun refits only what is missing; --overwrite discards the cache. Then:
    output-boba/analysis/review/known_variance_ship_per_run.csv.gz   one row per run, every level and rule
    output-boba/analysis/review/known_variance_ship_cells.csv         every scope x seeds x level x rule x cell
    output-boba/analysis/review/known_variance_ship_selection.json    the choices on 7-11, scored on 12-16
    output-boba/analysis/review/register_checks/known_variance_ship.txt   what the score stage prints
    paper/tables/known_variance.tex   the paper's tab:knownvar (this script is its producer)
A fresh checkout holds the per-run table but not the run logs, so --stage score runs there; the refit stage
needs the logs of output-boba, output-boba-confirmatory and output-boba-knownnoise, the git-ignored
stopping_per_run.csv, and, for the control, the git-ignored ship_rules_fresh table of the confirmatory arm.
Cost: 381,193 GP fits for the 47,400 runs, 54 minutes on eight workers; the score stage takes three minutes.

    python scripts/checks/known_variance_ship.py --workers 8
    python scripts/checks/known_variance_ship.py --stage score
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported first, as in the drivers and replay_stopping.py: it sets the BLAS/OMP thread limits before numpy
# and torch load, and torch's float64 default.
import bo_sensor_error_simulation as sim  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import analyse_boba_adaptations as aba  # noqa: E402
import analyse_ship_rules as asr  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
import replay_end_of_study as ree  # noqa: E402
import replay_stopping as rstop  # noqa: E402
import rescore_ship_rules as rs  # noqa: E402

REVIEW = REPO / "output-boba" / "analysis" / "review"
CACHE_DIR = REVIEW / "known_variance_ship" / "per_run"        # git-ignored: output-boba/analysis/**/per_run/
PER_RUN = REVIEW / "known_variance_ship_per_run.csv.gz"
CELLS = REVIEW / "known_variance_ship_cells.csv"
SELECTION = REVIEW / "known_variance_ship_selection.json"
REGISTER = REVIEW / "register_checks" / "known_variance_ship.txt"
TABLE = REPO / "paper" / "tables" / "known_variance.tex"
STOPPING = REPO / "output-boba" / "analysis" / "stopping" / "stopping_per_run.csv"
TUNED_A = REPO / "output-boba" / "analysis" / "stopping" / "stopping_tuned_A.json"
SITTING_SELECTION = REVIEW / "sitting_by_magnitude_selection.json"
STORED = {
    "output-boba": REPO / "output-boba" / "analysis" / "ship_rules_per_run.csv",
    "output-boba-confirmatory": REPO / "output-boba-confirmatory" / "analysis" / "ship_rules_fresh" / "ship_rules_per_run.csv",
}
# Bumped whenever a change moves a cached refit; a cache written under another version is refused.
CACHE_VERSION = 1

TRAIN_SEEDS = tuple(range(7, 12))
TEST_SEEDS = tuple(range(12, 17))
ALL_SEEDS = TRAIN_SEEDS + TEST_SEEDS
UNTOUCHED_SEEDS = tuple(range(27, 37))
TEN = tuple(aba.TEN.split(","))
PAIR_ACQS = ("logei", "qnei")
PROCESSES = ("gaussian", "bias", "drift", "ar1")
RULES = tuple(rs.POSTERIOR_RULES)                  # pm, lcb1, lcb2: 0, 1 and 2 latent SDs below the mean
GRID = aba.GRID
FLOOR = float(sim.known_noise_variance(1, types.SimpleNamespace(anchor_rating=False, jitter_iteration=0),
                                       apply_error=False))
R_PAIRS = 3
RETEST_TAG, INSIDE_CHOICE_TAG, INSIDE_NOISE_TAG = 20261009, 20261010, 20261011
EXTRA_BOOTSTRAP_SEED = aba.BOOTSTRAP_SEED + 101
REPS = aba.BOOTSTRAP_REPS
MAX_WORKERS = 8
POOL_MIN = 0.25

LEVELS = {
    "learned": "learned noise (the loop's refit; the control)",
    "mag": "magnitude known: sigma_e^2 on every trial",
    "mag_onset": "magnitude and onset known: 1e-6 up to the onset, sigma_e^2 after",
    "mag_half": "SD understated twofold: (sigma_e/2)^2 on every trial",
    "mag_double": "SD overstated twofold: (2 sigma_e)^2 on every trial",
    "retest3": "three test-retest pairs after the study (six ratings outside the budget)",
    "cusum": "magnitude known, onset from the freeze rule's CUSUM alarm",
    "retest3_cusum": "three test-retest pairs, onset from the CUSUM alarm",
    "inside_random": "inside the budget: trials 48-50 re-rate three random visited designs",
    "inside_top": "inside the budget: trials 48-50 re-rate the top three by LCB1 at trial 47",
}
VECTOR_LEVELS = ("mag", "mag_onset", "mag_half", "mag_double", "retest3", "cusum", "retest3_cusum")
CUSUM_LEVELS = ("cusum", "retest3_cusum")
INSIDE_LEVELS = ("inside_random", "inside_top")
ALL_LEVELS = tuple(LEVELS)
NO_CUSUM_LEVELS = tuple(lv for lv in ALL_LEVELS if lv not in CUSUM_LEVELS)

# Which runs each arm refits, and which levels. The known-noise arm is refitted only with its own GP (its
# logged variance, as rescore_ship_rules does), for the comparison with the final refit alone.
ARM_SPECS = {
    "output-boba": (
        dict(acqs=PAIR_ACQS, models=PROCESSES, seeds=ALL_SEEDS, levels=ALL_LEVELS),
        dict(acqs=tuple(a for a in TEN if a not in PAIR_ACQS), models=("gaussian",), seeds=ALL_SEEDS,
             levels=NO_CUSUM_LEVELS),
    ),
    "output-boba-confirmatory": (dict(acqs=TEN, models=("gaussian",), seeds=UNTOUCHED_SEEDS, levels=NO_CUSUM_LEVELS),),
    "output-boba-knownnoise": (dict(acqs=PAIR_ACQS, models=("gaussian",), seeds=TRAIN_SEEDS, levels=("own_gp",)),),
}


# ---------------------------------------------------------------------------
# The variance each level passes
# ---------------------------------------------------------------------------


def parse_bias(variant: str) -> float:
    """The bias offset a main-sweep bias run names in its variant ('bias1' -> 1.0), NaN without one."""
    v = str(variant or "")
    if v.startswith("bias"):
        try:
            return float(v[4:])
        except ValueError:
            pass
    return float("nan")


def error_config(run_args: dict, info: dict, T: int):
    """The run's error process as a SimulationConfig (replay_stopping.error_config, imported)."""
    return rstop.error_config({"run_args": run_args, "jitter_std": float(info["jitter_std"]),
                               "bias": parse_bias(info.get("variant", "")),
                               "jitter_iteration": int(info["jitter_iteration"]), "seed": int(info["seed"]),
                               "error_model": info["error_model"]}, T)


def retest_estimate(error_model: str, sigma: float, rho: float, name: str, r: int = R_PAIRS) -> float:
    """sigma_hat^2 from r test-retest pairs rated in one sitting: sum d_i^2 / (2 r), d_i = e_i1 - e_i2."""
    s = ree.idiosyncratic_sd(error_model, sigma, rho)
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(name.encode("utf-8")), RETEST_TAG]))
    e = rng.normal(0.0, s, size=(r, 2))
    d = e[:, 0] - e[:, 1]
    return float(np.sum(d ** 2) / (2 * r))


def onset_vector(v: float, alarm: float | None, w: int, T: int) -> np.ndarray:
    """FLOOR for t <= alarm - w, v after; v on every trial without an alarm."""
    if alarm is None or not np.isfinite(alarm):
        return np.full(T, float(v))
    t = np.arange(1, T + 1)
    return np.where(t > float(alarm) - w, float(v), FLOOR)


def variance_vectors(cfg, T: int, info: dict, name: str, levels, alarm: float | None = None,
                     w: int = 3) -> tuple[dict[str, np.ndarray], float]:
    """train_Yvar of every vector level for one noisy run, and the retest estimate."""
    per_trial = np.array([sim.known_noise_variance(int(t), cfg, True) for t in range(1, T + 1)], float)
    v = float(sim.known_noise_variance(T, cfg, True))
    sigma = float(info["jitter_std"])
    if info["error_model"] in PROCESSES and not np.isclose(v, sigma ** 2, rtol=1e-12, atol=0.0):
        raise ValueError(f"{name}: known_noise_variance gives {v!r} after the onset, not sigma_e^2 = {sigma ** 2!r}")
    v_hat = retest_estimate(info["error_model"], sigma, float(cfg.error_ar1_rho), name)
    vh = max(v_hat, FLOOR)
    full = {
        "mag": np.full(T, v), "mag_onset": per_trial, "mag_half": np.full(T, max(v / 4.0, FLOOR)),
        "mag_double": np.full(T, 4.0 * v), "retest3": np.full(T, vh),
        "cusum": onset_vector(v, alarm, w, T), "retest3_cusum": onset_vector(vh, alarm, w, T),
    }
    return {k: full[k] for k in VECTOR_LEVELS if k in levels}, v_hat


# ---------------------------------------------------------------------------
# The refit and the rules
# ---------------------------------------------------------------------------


def posterior_picks(X: np.ndarray, y: np.ndarray, lo, hi, yvar: np.ndarray | None, seed: int,
                    n_cand: int | None = None) -> tuple[dict[str, int], dict]:
    """Refit the loop's GP (rescore_ship_rules.fit_loop_gp) and pick among the first n_cand rows by each rule."""
    gp, diag = rs.fit_loop_gp(X, y, lo, hi, yvar=yvar, seed=seed)
    n = len(X) if n_cand is None else int(n_cand)
    mu, sd = rs.latent_posterior(gp, X[:n])
    picks = {}
    for rule, k in rs.POSTERIOR_RULES.items():
        crit = mu - k * sd
        picks[rule] = int(np.argmax(crit)) if np.all(np.isfinite(crit)) else -1
    return picks, diag


def repeat_ratings(cfg, info: dict, true_values: np.ndarray, m: int, prev_obs: float, prev_err: float) -> np.ndarray:
    """Ratings of the given designs at trials m+1, m+2, ... from the run's own error process, in sequence.

    The stream is seeded like replay_stopping.repeat_ratings (common random numbers across landscapes, as in
    the driver) with a tag of its own, so it replays no draw of the logged trials.
    """
    rng = np.random.default_rng(np.random.SeedSequence([
        int(info["seed"]), sim.ACQUISITION_CHOICES.index(info["acquisition"]), int(info["jitter_iteration"]),
        int(round(float(info["jitter_std"]) * 1_000_000)), sim.ERROR_MODEL_CHOICES.index(info["error_model"]),
        INSIDE_NOISE_TAG]))
    prev_o, prev_e = np.array([float(prev_obs)]), np.array([float(prev_err)])
    out = []
    for j, g in enumerate(np.asarray(true_values, float)):
        obs, err = sim.apply_sensor_error(true_value=np.array([g]), iteration=m + 1 + j, config=cfg, rng=rng,
                                          previous_observed=prev_o, previous_error=prev_e)
        out.append(float(obs[0]))
        prev_o, prev_e = obs, err
    return np.array(out)


def top_distinct(order: np.ndarray, X: np.ndarray, r: int) -> np.ndarray:
    """The first r rows of ``order`` whose designs are distinct."""
    picked, seen = [], set()
    for i in order:
        key = tuple(np.round(X[int(i)], 12))
        if key in seen:
            continue
        seen.add(key)
        picked.append(int(i))
        if len(picked) == r:
            break
    if len(picked) < r:
        raise ValueError(f"fewer than {r} distinct visited designs")
    return np.array(picked)


def inside_budget(X, y, g, score, err, y_opt, lo, hi, seed: int, name: str, choose: str, cfg, info: dict,
                  r: int = R_PAIRS) -> dict:
    """Search T - r trials, re-rate r visited designs, refit all T ratings with their estimate, ship by each rule."""
    T = len(y)
    m = T - r
    if choose == "random":
        rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(name.encode("utf-8")), INSIDE_CHOICE_TAG]))
        idx = rng.choice(m, size=r, replace=False)
    elif choose == "top":
        gp, _ = rs.fit_loop_gp(X[:m], y[:m], lo, hi, yvar=None, seed=seed)
        mu, sd = rs.latent_posterior(gp, X[:m])
        idx = top_distinct(np.argsort(-(mu - sd), kind="stable"), X[:m], r)
    else:
        raise ValueError(f"unknown choice {choose!r}")
    idx = np.asarray(idx, int)
    y_orig = y[idx]
    # A clean twin's rater is exact: the repeats equal the true values and the estimate is zero.
    y_rep = g[idx].astype(float) if cfg is None else repeat_ratings(cfg, info, g[idx], m, y[m - 1], err[m - 1])
    s2 = float(np.sum((y_rep - y_orig) ** 2) / (2 * r))
    Xf = np.vstack([X[:m], X[idx]])
    yf = np.concatenate([y[:m], y_rep])
    picks, diag = posterior_picks(Xf, yf, lo, hi, np.full(m + r, max(s2, FLOOR)), seed, n_cand=m)
    return {"picks": picks, "regret": {k: (float(y_opt - score[i]) if i >= 0 else np.nan) for k, i in picks.items()},
            "sigma2": s2, "idx": [int(i) for i in idx], "diag": diag}


def run_arrays(df: pd.DataFrame, info: dict) -> dict:
    """The arrays the levels refit, from the same rows rescore_ship_rules.rescore_frame uses."""
    T = len(df)
    if not np.array_equal(df["iteration"].to_numpy(), np.arange(1, T + 1)):
        raise ValueError(f"{info['file']}: iterations are not 1..{T}")
    spec = bb.BENCHMARKS[info["dataset"]]
    cols = str(df["param_columns"].iloc[0]).split(",")
    observed = df["objective_observed"].to_numpy(float)
    if not np.all(np.isfinite(observed)):
        raise ValueError(f"{info['file']}: a missing rating; the levels assume every trial was rated")
    score_column = "objective_true_deployed" if "objective_true_deployed" in df.columns else "objective_true"
    g = df["objective_true"].to_numpy(float)
    err = (df["error_magnitude"].to_numpy(float) if "error_magnitude" in df.columns else observed - g)
    return {"X": df[cols].to_numpy(float), "y": observed, "g": g, "score": df[score_column].to_numpy(float),
            "err": err, "y_opt": float(df["y_opt"].iloc[0]), "lo": np.asarray(spec.bounds_low, float),
            "hi": np.asarray(spec.bounds_high, float), "T": T}


ID_KEYS = ("arm", "dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed", "baseline",
           "variant", "file")


def refit_run(task: dict) -> dict:
    """Worker: one log, every level the task asks for."""
    df = pd.read_csv(task["path"], float_precision="round_trip")
    name = Path(task["file"]).name
    seed = rs.file_seed(task["file"])
    std = rs.rescore_frame(df, task, task["expected_rule"], seed)
    rec = {"cache_version": CACHE_VERSION, **{k: task[k] for k in ID_KEYS}, "y_opt": std["y_opt"],
           "n_train": std["n_train"], "regret_best_observed": std["regret_best_observed"],
           **{f"regret_learned_{r}": std[f"regret_{r}"] for r in RULES},
           **{f"idx_learned_{r}": std[f"idx_{r}"] for r in RULES},
           "learned_gp_noise_sd": std["gp_noise_sd"], "gp_known_noise": bool(std["gp_known_noise"]),
           "n_fits": 1, "n_fit_failures": int(not std["gp_fit_ok"]), "n_fit_warnings": int(std["gp_n_warnings"] > 0)}
    levels = task["levels"]
    a = run_arrays(df, task)
    T = a["T"]
    if levels == ("own_gp",):
        # The known-noise arm: its GP was told known_noise_variance trial by trial, which is mag_onset.
        want = (np.full(T, FLOOR) if task["baseline"] else
                np.array([sim.known_noise_variance(t, error_config(task["run_args"], task, T), True)
                          for t in range(1, T + 1)]))
        rec["known_var_max_abs_diff"] = float(np.max(np.abs(df["known_noise_var"].to_numpy(float) - want)))
        return rec
    if task["baseline"]:
        cfg = None
        vectors = {lv: np.full(T, FLOOR) for lv in VECTOR_LEVELS if lv in levels}
        rec["sigma2_retest3"] = 0.0
    else:
        cfg = error_config(task["run_args"], task, T)
        vectors, v_hat = variance_vectors(cfg, T, task, name, levels, task.get("alarm"), task.get("w", 3))
        rec["sigma2_retest3"] = v_hat
    rec["alarm"] = task.get("alarm", np.nan)
    fitted: dict[bytes, tuple[dict, dict]] = {}
    for lv, vec in vectors.items():
        key = np.asarray(vec, float).tobytes()
        if key not in fitted:
            fitted[key] = posterior_picks(a["X"], a["y"], a["lo"], a["hi"], vec, seed)
            rec["n_fits"] += 1
            rec["n_fit_failures"] += int(not fitted[key][1]["gp_fit_ok"])
            rec["n_fit_warnings"] += int(fitted[key][1]["gp_n_warnings"] > 0)
        picks, _ = fitted[key]
        for r, i in picks.items():
            rec[f"regret_{lv}_{r}"] = float(a["y_opt"] - a["score"][i]) if i >= 0 else np.nan
            rec[f"idx_{lv}_{r}"] = int(i)
    for lv in INSIDE_LEVELS:
        if lv not in levels:
            continue
        out = inside_budget(a["X"], a["y"], a["g"], a["score"], a["err"], a["y_opt"], a["lo"], a["hi"], seed, name,
                            lv.split("_", 1)[1], cfg, task)
        rec["n_fits"] += 2 if lv == "inside_top" else 1
        rec["n_fit_failures"] += int(not out["diag"]["gp_fit_ok"])
        rec["n_fit_warnings"] += int(out["diag"]["gp_n_warnings"] > 0)
        rec[f"sigma2_{lv}"] = out["sigma2"]
        rec[f"rerated_{lv}"] = ";".join(str(i) for i in out["idx"])
        for r in RULES:
            rec[f"regret_{lv}_{r}"] = out["regret"][r]
            rec[f"idx_{lv}_{r}"] = int(out["picks"][r])
    return rec


def _init_worker() -> None:
    import torch

    # One thread per worker, as rescore_ship_rules.py: the job is many small fits.
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Tasks, cache, refit stage
# ---------------------------------------------------------------------------


def detector_alarms() -> tuple[dict[str, float], int]:
    """Rule A's alarm trial per noisy run (NaN without an alarm) and its w, from replay_stopping.py's outputs."""
    tuned = json.loads(TUNED_A.read_text(encoding="utf-8"))
    st = pd.read_csv(STOPPING, usecols=["file", "rule", "error_model", "stopped", "stop_t", "h", "kappa", "w"])
    st = st[(st["rule"] == "A") & (st["error_model"] != "none")]
    for col, key in (("h", "h"), ("kappa", "kappa"), ("w", "w")):
        if not np.allclose(st[col].astype(float), float(tuned[key])):
            raise ValueError(f"stopping_per_run.csv rule A rows do not all use the tuned {key} = {tuned[key]}")
    stopped = st["stopped"].astype(str).str.lower().isin(("true", "1"))
    alarms = dict(zip(st["file"].map(lambda f: Path(str(f)).name),
                      np.where(stopped, st["stop_t"].astype(float), np.nan)))
    return alarms, int(tuned["w"])


def build_tasks(arms: list[str], datasets: set[str] | None) -> list[dict]:
    tasks: list[dict] = []
    alarms, w = None, 3
    for arm in arms:
        arm_dir = REPO / arm
        settings = rs.arm_settings(arm_dir)
        if settings["inference_rule"] != "best_observed":
            raise ValueError(f"{arm}: ships by {settings['inference_rule']}, not the standard best_observed")
        run_args = rstop.read_run_args(arm_dir)
        seen: set[str] = set()
        for spec in ARM_SPECS[arm]:
            runs = rs.index_runs(arm_dir, datasets, set(spec["acqs"]), set(spec["seeds"]), set(spec["models"]))
            for t in runs:
                if t["file"] in seen:      # a clean twin is listed by every spec of its acquisition
                    continue
                seen.add(t["file"])
                levels = spec["levels"]
                task = {**t, "arm": arm, "run_args": run_args, "expected_rule": settings["inference_rule"],
                        "levels": tuple(levels)}
                if any(lv in CUSUM_LEVELS for lv in levels) and not t["baseline"]:
                    if alarms is None:
                        alarms, w = detector_alarms()
                    name = Path(t["file"]).name
                    if name not in alarms:
                        raise KeyError(f"{name}: no rule-A row in {STOPPING}")
                    task["alarm"], task["w"] = float(alarms[name]), w
                tasks.append(task)
    return tasks


def cache_path(arm: str) -> Path:
    return CACHE_DIR / f"{arm}.jsonl"


def read_cache(arm: str) -> list[dict]:
    path = cache_path(arm)
    records = rs.read_journal(path)
    stale = {r.get("cache_version") for r in records} - {CACHE_VERSION}
    if stale:
        raise SystemExit(f"{path} holds refits of cache version(s) {sorted(map(str, stale))}, not {CACHE_VERSION}: "
                         "pass --overwrite to refit")
    return records


def refit_stage(arms: list[str], workers: int, datasets: set[str] | None, max_files: int | None,
                overwrite: bool) -> None:
    """Refit every run the cache lacks. One worker pool serves every arm, so the workers import this module
    once, at the start; outputs are seeded per run and do not depend on the worker count or order."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(arms, datasets)
    by_arm: dict[str, list[dict]] = {}
    for t in tasks:
        by_arm.setdefault(t["arm"], []).append(t)
    todo: list[dict] = []
    for arm, arm_tasks in by_arm.items():
        path = cache_path(arm)
        if overwrite and path.exists():
            path.unlink()
        done = {r["file"] for r in read_cache(arm)}
        arm_todo = [t for t in arm_tasks if t["file"] not in done]
        if max_files is not None:
            arm_todo = arm_todo[:max_files]
        print(f"{arm}: {len(arm_tasks):,} runs, {len(done & {t['file'] for t in arm_tasks}):,} cached, "
              f"{len(arm_todo):,} to refit", flush=True)
        todo += arm_todo
    if not todo:
        return
    n_workers = max(1, min(workers, MAX_WORKERS, os.cpu_count() or 1, len(todo)))
    print(f"refitting {len(todo):,} runs with {n_workers} worker(s)", flush=True)
    t0 = time.time()
    handles = {arm: open(cache_path(arm), "a", encoding="utf-8") for arm in by_arm}
    try:
        def emit(i: int, rec: dict) -> None:
            fh = handles[rec["arm"]]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 500 == 0 or i + 1 == len(todo):
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  {i + 1:,}/{len(todo):,}  {rate:.1f} runs/s  eta {(len(todo) - i - 1) / rate / 60:.1f} min",
                      flush=True)

        if n_workers == 1:
            _init_worker()
            for i, task in enumerate(todo):
                emit(i, refit_run(task))
        else:
            from multiprocessing import get_context

            with get_context("spawn").Pool(n_workers, initializer=_init_worker) as pool:
                for i, rec in enumerate(pool.imap_unordered(refit_run, todo, chunksize=4)):
                    emit(i, rec)
    finally:
        for fh in handles.values():
            fh.close()
    print(f"refit stage: {time.time() - t0:.0f}s", flush=True)


def per_run_columns() -> list[str]:
    cols = list(ID_KEYS) + ["y_opt", "n_train", "regret_best_observed"]
    cols += [f"regret_{lv}_{r}" for lv in ALL_LEVELS for r in RULES]
    cols += ["learned_gp_noise_sd", "gp_known_noise", "sigma2_retest3", "sigma2_inside_random", "sigma2_inside_top",
             "rerated_inside_random", "rerated_inside_top", "alarm", "known_var_max_abs_diff", "n_fits",
             "n_fit_failures", "n_fit_warnings"]
    return cols


def assemble(arms: list[str]) -> pd.DataFrame:
    frames = []
    for arm in arms:
        recs = read_cache(arm)
        if recs:
            frames.append(pd.DataFrame(recs).drop_duplicates("file", keep="last"))
    if not frames:
        raise SystemExit(f"no cached refits under {CACHE_DIR}; run the refit stage")
    table = pd.concat(frames, ignore_index=True)
    for c in per_run_columns():
        if c not in table.columns:
            table[c] = np.nan
    table = table[per_run_columns()]
    return table.sort_values(["arm", "dataset", "acquisition", "seed", "baseline", "error_model", "variant",
                              "jitter_std", "jitter_iteration"], na_position="first").reset_index(drop=True)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def control_check(per_run: pd.DataFrame) -> list[str]:
    """The learned refit against rescore_ship_rules.py's stored table: equal to 1e-9 in every run, or stop."""
    lines = []
    cols = ["regret_best_observed"] + [f"regret_{r}" for r in RULES]
    for arm, path in STORED.items():
        mine = per_run[per_run["arm"] == arm]
        if mine.empty:
            continue
        if not path.is_file():
            lines.append(f"{arm}: {_rel(path)} is not here; the control was not checked in this run")
            continue
        stored = pd.read_csv(path, usecols=["file"] + cols, float_precision="round_trip")
        stored["name"] = stored["file"].map(lambda f: Path(str(f)).name)
        if stored["name"].duplicated().any():
            raise ValueError(f"{path}: duplicated run names")
        stored = stored.drop(columns="file").rename(columns={c: f"{c}_stored" for c in cols})
        m = mine.assign(name=mine["file"].map(lambda f: Path(str(f)).name)).merge(
            stored, on="name", how="left", validate="one_to_one")
        lost = m["regret_best_observed_stored"].isna()
        if lost.any():
            raise KeyError(f"{int(lost.sum())} refitted runs of {arm} are not in {path}, "
                           f"e.g. {m.loc[lost, 'name'].iloc[0]}")
        mine_cols = {"regret_best_observed": "regret_best_observed",
                     **{f"regret_{r}": f"regret_learned_{r}" for r in RULES}}
        worst, exact = 0.0, 0
        same = np.ones(len(m), bool)
        for stored_col, my_col in mine_cols.items():
            d = (m[my_col].astype(float) - m[f"{stored_col}_stored"].astype(float)).abs().to_numpy()
            worst = max(worst, float(np.nanmax(d)))
            same &= d == 0.0
        exact = int(same.sum())
        if worst > rs.REPRODUCTION_TOL:
            raise rs.ReproductionError(f"{arm}: the learned refit differs from {path} by up to {worst:g}")
        lines.append(f"{arm}: the learned refit reproduces {_rel(path)} in {len(m):,} of {len(m):,} runs "
                     f"(best_observed, PM, LCB1, LCB2; max |difference| {worst:g}, bit-identical in {exact:,})")
    return lines


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def wilcoxon_p(v) -> float:
    v = np.asarray(v, float)
    return float(wilcoxon(v).pvalue) if np.any(v != 0) else float("nan")


def boot_mean_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    v = np.asarray(v, float)
    draws = v[rng.integers(0, len(v), (REPS, len(v)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summary(block: pd.DataFrame) -> dict:
    """aba.summarise and aba.share_flags of one block, plus gain and price intervals from their own generator."""
    s = asr._flagged_summary(block)
    per = block.groupby("dataset")[["ref_noisy", "ref_clean", "trt_noisy", "trt_clean"]].mean()
    g = (per["ref_noisy"] - per["trt_noisy"]).to_numpy()
    p = (per["trt_clean"] - per["ref_clean"]).to_numpy()
    rng = np.random.default_rng(EXTRA_BOOTSTRAP_SEED)
    s["gain_lo"], s["gain_hi"] = boot_mean_ci(g, rng)
    s["price_lo"], s["price_hi"] = boot_mean_ci(p, rng)
    s["landscapes_gaining"], s["landscapes_losing"] = int((g > 0).sum()), int((g < 0).sum())
    return s


def increment(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """b's paired advantage over a on the noisy runs: per run a.trt_noisy - b.trt_noisy, landscape means."""
    keys = aba.PAIR_KEYS
    m = a[keys + ["trt_noisy", "trt_clean"]].merge(b[keys + ["trt_noisy", "trt_clean"]], on=keys,
                                                   suffixes=("_a", "_b"), validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("the two rules do not cover the same runs")
    per = m.assign(inc=m["trt_noisy_a"] - m["trt_noisy_b"], dprice=m["trt_clean_b"] - m["trt_clean_a"]).groupby(
        "dataset")[["inc", "dprice"]].mean()
    v = per["inc"].to_numpy()
    lo, hi = boot_mean_ci(v, np.random.default_rng(EXTRA_BOOTSTRAP_SEED + 1))
    return {"inc": float(v.mean()), "inc_lo": lo, "inc_hi": hi, "inc_p": wilcoxon_p(v),
            "inc_landscapes_ahead": int((v > 0).sum()), "inc_landscapes_behind": int((v < 0).sum()),
            "inc_n_landscapes": int(len(v)), "inc_price": float(per["dprice"].mean())}


def blocks(p: pd.DataFrame):
    """(cell label, sigma, onset, block): the eight cells, then the pools the paper uses."""
    p = p.assign(jitter_std=p["jitter_std"].astype(float), jitter_iteration=p["jitter_iteration"].astype(int))
    for (sig, ons), sub in sorted(p.groupby(["jitter_std", "jitter_iteration"])):
        yield f"{sig:g}sigma_from_trial_{ons + 1}", sig, ons, sub
    big = p[p["jitter_std"] >= POOL_MIN - 1e-12]
    for ons, sub in sorted(big.groupby("jitter_iteration")):
        yield f"pooled_ge_{POOL_MIN:g}sigma_from_trial_{ons + 1}", np.nan, ons, sub
    yield f"pooled_ge_{POOL_MIN:g}sigma", np.nan, np.nan, big
    for sig, sub in sorted(p.groupby("jitter_std")):
        yield f"{sig:g}sigma_both_onsets", sig, np.nan, sub
    yield "pooled_all_cells", np.nan, np.nan, p


SCOPES = (
    # name, arm, acquisitions, error processes, seed sets
    ("four processes", "output-boba", PAIR_ACQS, PROCESSES,
     {"7-11": TRAIN_SEEDS, "12-16": TEST_SEEDS, "7-16": ALL_SEEDS}),
    ("gaussian", "output-boba", PAIR_ACQS, ("gaussian",), {"7-11": TRAIN_SEEDS, "12-16": TEST_SEEDS, "7-16": ALL_SEEDS}),
    ("bias", "output-boba", PAIR_ACQS, ("bias",), {"12-16": TEST_SEEDS, "7-16": ALL_SEEDS}),
    ("drift", "output-boba", PAIR_ACQS, ("drift",), {"12-16": TEST_SEEDS, "7-16": ALL_SEEDS}),
    ("ar1", "output-boba", PAIR_ACQS, ("ar1",), {"12-16": TEST_SEEDS, "7-16": ALL_SEEDS}),
    ("gaussian untouched", "output-boba-confirmatory", PAIR_ACQS, ("gaussian",), {"27-36": UNTOUCHED_SEEDS}),
    ("ten acquisitions, gaussian", "output-boba", TEN, ("gaussian",),
     {"7-11": TRAIN_SEEDS, "12-16": TEST_SEEDS, "7-16": ALL_SEEDS}),
    ("ten acquisitions, gaussian untouched", "output-boba-confirmatory", TEN, ("gaussian",), {"27-36": UNTOUCHED_SEEDS}),
)


def scope_runs(per_run: pd.DataFrame, arm: str, acqs, models) -> pd.DataFrame:
    x = per_run[(per_run["arm"] == arm) & per_run["acquisition"].isin(acqs)]
    return x[x["baseline"] | x["error_model"].isin(models)].reset_index(drop=True)


def scope_levels(arm: str, acqs) -> tuple[str, ...]:
    return ALL_LEVELS if arm == "output-boba" and set(acqs) <= set(PAIR_ACQS) else NO_CUSUM_LEVELS


def paired_frames(runs: pd.DataFrame, levels, opt_z: dict[str, float]) -> dict[tuple[str, str], pd.DataFrame]:
    """(level, rule) -> the paired frame against the standard process (analyse_ship_rules.twin_frame)."""
    ref = asr.twin_frame(runs, "best_observed")
    out = {}
    for lv in levels:
        for r in RULES:
            trt = asr.twin_frame(runs, f"{lv}_{r}")
            out[(lv, r)] = aba.paired_frame(ref, trt, asr.RESPONSE, opt_z, pool=False)
    return out


def score_scopes(per_run: pd.DataFrame, opt_z: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    rows, frames = [], {}
    for name, arm, acqs, models, seed_sets in SCOPES:
        runs = scope_runs(per_run, arm, acqs, models)
        if runs.empty or not (~runs["baseline"]).any():
            print(f"  scope {name}: no runs, skipped")
            continue
        levels = scope_levels(arm, acqs)
        pf = paired_frames(runs, levels, opt_z)
        frames[name] = pf
        for seeds_label, seeds in seed_sets.items():
            sub = {k: v[v["seed"].isin(seeds)] for k, v in pf.items()}
            if sub[("learned", "lcb1")].empty:
                continue
            learned_blocks = {r: {c: b for c, _, _, b in blocks(sub[("learned", r)])} for r in RULES}
            for lv in levels:
                for r in RULES:
                    for cell, sig, ons, block in blocks(sub[(lv, r)]):
                        row = {"scope": name, "seeds": seeds_label, "level": lv, "rule": r, "cell": cell,
                               "jitter_std": sig, "jitter_iteration": ons, "n_runs": int(len(block)),
                               **summary(block)}
                        if lv != "learned":
                            row.update({f"over_learned_{k}": v
                                        for k, v in increment(learned_blocks[r][cell], block).items()})
                        rows.append(row)
        print(f"  scope {name}: {len(runs):,} runs scored", flush=True)
    return pd.DataFrame(rows), frames


CELL_ORDER = [f"{s:g}sigma_from_trial_{o + 1}" for s in GRID for o in (0, 20)]


def cell_of(frame: pd.DataFrame, cell: str) -> pd.DataFrame:
    return dict((c, b) for c, _, _, b in blocks(frame))[cell]


TEST_KEYS = ("gain", "gain_lo", "gain_hi", "cost", "recovered", "recovered_lo", "recovered_hi", "recovered_suppressed",
             "wilcoxon_p", "price", "price_lo", "price_hi", "landscapes_gaining", "landscapes_losing", "n_landscapes")


def _record(row, keys=TEST_KEYS) -> dict:
    return {k: (None if pd.isna(row[k]) else (row[k] if isinstance(row[k], str) else float(row[k]))) for k in keys}


def select_on_train(cells: pd.DataFrame, frames: dict, scope: str = "four processes",
                    untouched: str | None = None) -> dict:
    """Choices on seeds 7-11, scored on seeds 12-16 (and, with ``untouched``, on seeds 27-36 of that scope):
    per cell, and one rule per level for all cells."""
    c = cells[cells["scope"] == scope]
    u = cells[(cells["scope"] == untouched) & (cells["seeds"] == "27-36")] if untouched else None
    pf = frames[scope]
    pu = frames.get(untouched) if untouched else None
    levels = sorted(set(c["level"]), key=list(ALL_LEVELS).index)
    out: dict = {"scope": f"{scope}, LogEI and qNEI, seeds 7-11 choose, seeds 12-16 score"
                          + (", and seeds 27-36 of the replication arm" if untouched else ""),
                 "by_cell": {}, "one_rule": {}, "inside_protocol": {}}

    def gain(seeds, lv, r, cell):
        x = c[(c["seeds"] == seeds) & (c["level"] == lv) & (c["rule"] == r) & (c["cell"] == cell)]
        return float(x["gain"].iloc[0])

    def test_row(lv, r, cell):
        return c[(c["seeds"] == "12-16") & (c["level"] == lv) & (c["rule"] == r) & (c["cell"] == cell)].iloc[0]

    def untouched_row(lv, r, cell):
        x = u[(u["level"] == lv) & (u["rule"] == r) & (u["cell"] == cell)]
        return None if x.empty else x.iloc[0]

    learned_choice = {cell: max(RULES, key=lambda r: gain("7-11", "learned", r, cell)) for cell in CELL_ORDER}
    for lv in levels:
        per_cell, p_gain, p_inc = {}, [], []
        for cell in CELL_ORDER:
            r = max(RULES, key=lambda rr: gain("7-11", lv, rr, cell))
            t = test_row(lv, r, cell)
            lr = learned_choice[cell]
            base = cell_of(pf[("learned", lr)][pf[("learned", lr)]["seed"].isin(TEST_SEEDS)], cell)
            mine = cell_of(pf[(lv, r)][pf[(lv, r)]["seed"].isin(TEST_SEEDS)], cell)
            inc = increment(base, mine) if lv != "learned" else {}
            per_cell[cell] = {
                "rule": r, "train_gains": {rr: gain("7-11", lv, rr, cell) for rr in RULES},
                "test": _record(t),
                "learned_rule_chosen_on_7_11": lr,
                "over_learned_chosen": inc,
            }
            if u is not None and pu is not None and (lv, r) in pu:
                ur = untouched_row(lv, r, cell)
                if ur is not None:
                    per_cell[cell]["untouched"] = _record(ur)
                    if lv != "learned":
                        per_cell[cell]["untouched_over_learned_chosen"] = increment(
                            cell_of(pu[("learned", lr)], cell), cell_of(pu[(lv, r)], cell))
            p_gain.append(t["wilcoxon_p"])
            p_inc.append(inc.get("inc_p", np.nan))
        for key, ps in (("gain_p_holm_over_cells", p_gain), ("over_learned_chosen_p_holm_over_cells", p_inc)):
            ps = np.asarray(ps, float)
            adj = np.full(len(ps), np.nan)
            ok = np.isfinite(ps)
            if ok.any():
                adj[ok] = multipletests(ps[ok], method="holm")[1]
            for cell, a in zip(CELL_ORDER, adj):
                per_cell[cell][key] = None if not np.isfinite(a) else float(a)
        out["by_cell"][lv] = per_cell
        r1 = max(RULES, key=lambda rr: gain("7-11", lv, rr, "pooled_all_cells"))
        pooled_cells = CELL_ORDER + [f"pooled_ge_{POOL_MIN:g}sigma", "pooled_all_cells"]
        out["one_rule"][lv] = {"rule": r1, "train_pooled_gain": {rr: gain("7-11", lv, rr, "pooled_all_cells")
                                                                  for rr in RULES},
                               "test": {cell: _record(test_row(lv, r1, cell)) for cell in pooled_cells}}
        if u is not None:
            rows_u = {cell: untouched_row(lv, r1, cell) for cell in pooled_cells}
            out["one_rule"][lv]["untouched"] = {cell: _record(x) for cell, x in rows_u.items() if x is not None}
    # The deployable pipelines, each at its one rule: inside the budget (which designs to re-rate) and outside
    # it (the test-retest alone, or with the detector's onset).
    for key, options in (("inside_protocol", INSIDE_LEVELS), ("outside_protocol", ("retest3", "retest3_cusum"))):
        if all(lv in out["one_rule"] for lv in options):
            train = {lv: max(out["one_rule"][lv]["train_pooled_gain"].values()) for lv in options}
            best = max(options, key=lambda lv: train[lv])
            out[key] = {"chosen": best, "train_pooled_gain_at_its_rule": train, "rule": out["one_rule"][best]["rule"]}
    return out


def sitting_comparison(per_run: pd.DataFrame, opt_z: dict[str, float], selections: dict[str, dict]) -> list[dict]:
    """Each level's rule (chosen on seeds 7-11 in the cell and scope) against the final sitting at the k
    sitting_by_magnitude.py chose on seeds 7-11 for that cell and scope, on the same runs, seeds 12-16."""
    import sitting_by_magnitude as sbm

    analysis = REPO / "output-boba" / "analysis"
    files = {"four processes": SITTING_SELECTION, "gaussian": REVIEW / "sitting_by_magnitude_gaussian_selection.json"}
    if not all((analysis / d / "end_of_study_per_run.csv.gz").is_file() for d in sbm.REPLAY_DIRS) \
            or not all(f.is_file() for f in files.values()):
        return []
    sit = sbm.load(analysis, bb.DEFAULT_STATS_PATH)
    sit = sit[["file", "k", "regret_noisy", "regret_clean", "ref_noisy", "ref_clean"]]
    rows = []
    for scope in ("four processes", "gaussian"):
        selection = selections.get(scope)
        if selection is None:
            continue
        chosen_k = {s["cell"]: s["k_chosen_on_7_11"] for s in json.loads(files[scope].read_text())
                    if s.get("k_chosen_on_7_11") is not None}
        runs = scope_runs(per_run, "output-boba", PAIR_ACQS, PROCESSES if scope == "four processes" else ("gaussian",))
        noisy = runs[~runs["baseline"]].assign(name=lambda d: d["file"].map(lambda f: Path(str(f)).name))
        clean = runs[runs["baseline"]].set_index(["dataset", "acquisition", "seed"])
        for cell in ("1sigma_from_trial_1", "5sigma_from_trial_21", "1sigma_from_trial_21", "5sigma_from_trial_1"):
            k = chosen_k.get(cell)
            if k is None:
                continue
            sig, ons = float(cell.split("sigma")[0]), int(cell.rsplit("_", 1)[1]) - 1
            nb = noisy[(noisy["jitter_std"] == sig) & (noisy["jitter_iteration"] == ons) & noisy["seed"].isin(TEST_SEEDS)]
            m = nb.merge(sit[sit["k"] == k], left_on="name", right_on="file", how="left", suffixes=("", "_sit"),
                         validate="one_to_one")
            if m["regret_noisy"].isna().any():
                raise KeyError(f"{cell}: {int(m['regret_noisy'].isna().sum())} runs have no sitting replay at k = {k}")
            gap = float((m["ref_noisy"] - m["regret_best_observed"]).abs().max())
            if gap > 1e-9:
                raise ValueError(f"{cell}: the sitting replay's standard process differs from the refit's by {gap:g}")
            z = m["dataset"].map(opt_z)
            m["price_sitting"] = (m["regret_clean"] - m["ref_clean"]) / z
            for lv in ("learned", "mag", "mag_onset", "retest3", "cusum", "retest3_cusum", "inside_random", "inside_top"):
                r = selection["by_cell"].get(lv, {}).get(cell, {}).get("rule")
                if r is None:
                    continue
                twin = clean[f"regret_{lv}_{r}"]
                rule_clean = m.set_index(["dataset", "acquisition", "seed"]).index.map(twin)
                inc = (m["regret_noisy"] - m[f"regret_{lv}_{r}"]) / z
                per = pd.DataFrame({"dataset": m["dataset"], "inc": inc,
                                    "price_rule": (np.asarray(rule_clean, float) - m["ref_clean"]) / z,
                                    "price_sitting": m["price_sitting"]}).groupby("dataset").mean()
                v = per["inc"].to_numpy()
                lo, hi = boot_mean_ci(v, np.random.default_rng(EXTRA_BOOTSTRAP_SEED + 2))
                rows.append({"scope": scope, "cell": cell, "sitting_k": int(k), "level": lv, "rule": r,
                             "rule_over_sitting": float(v.mean()), "lo": lo, "hi": hi, "p": wilcoxon_p(v),
                             "landscapes_ahead": int((v > 0).sum()), "n_landscapes": int(len(v)),
                             "price_rule": float(per["price_rule"].mean()),
                             "price_sitting": float(per["price_sitting"].mean()), "n_runs": int(len(m))})
    return rows


def knownnoise_comparison(per_run: pd.DataFrame, opt_z: dict[str, float]) -> list[dict]:
    """The known-noise arm's rules (GP told the variance during the run) against the final refit alone."""
    kn = per_run[per_run["arm"] == "output-boba-knownnoise"]
    if kn.empty:
        return []
    main = scope_runs(per_run, "output-boba", PAIR_ACQS, ("gaussian",))
    # The arm ran seeds 7-11; every (landscape, acquisition, seed) it holds is paired with the main sweep's.
    stems = kn[["dataset", "acquisition", "seed"]].drop_duplicates()
    main = main.merge(stems, on=["dataset", "acquisition", "seed"], how="inner")
    if main.empty:
        return []
    ref = asr.twin_frame(main, "best_observed")
    rows = []
    for r in RULES:
        arm_p = aba.paired_frame(ref, asr.twin_frame(kn, f"learned_{r}"), asr.RESPONSE, opt_z, pool=False)
        fin_p = aba.paired_frame(ref, asr.twin_frame(main, f"mag_onset_{r}"), asr.RESPONSE, opt_z, pool=False)
        for (cell, sig, ons, a_blk), (_, _, _, f_blk) in zip(blocks(arm_p), blocks(fin_p)):
            sa, sf = summary(a_blk), summary(f_blk)
            rows.append({"rule": r, "cell": cell, "arm_gain": sa["gain"], "arm_gain_lo": sa["gain_lo"],
                         "arm_gain_hi": sa["gain_hi"], "arm_recovered": sa["recovered"],
                         "arm_recovered_suppressed": sa["recovered_suppressed"], "arm_price": sa["price"],
                         "final_refit_gain": sf["gain"], "final_refit_price": sf["price"],
                         **{f"arm_over_final_refit_{k}": v for k, v in increment(f_blk, a_blk).items()}})
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _f(v, fmt="+.4f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


def _share(r) -> str:
    if isinstance(r.get("recovered_suppressed"), str) and r["recovered_suppressed"]:
        return "no share (" + r["recovered_suppressed"] + ")"
    if r.get("recovered") is None or not np.isfinite(r["recovered"]):
        return "share n/a"
    return f"{100 * r['recovered']:+.1f}% [{100 * r['recovered_lo']:+.0f}, {100 * r['recovered_hi']:+.0f}]"


def _row(cells: pd.DataFrame, scope, seeds, level, rule, cell) -> dict | None:
    x = cells[(cells["scope"] == scope) & (cells["seeds"] == seeds) & (cells["level"] == level)
              & (cells["rule"] == rule) & (cells["cell"] == cell)]
    if x.empty:
        return None
    d = x.iloc[0].to_dict()
    if not isinstance(d.get("recovered_suppressed"), str):
        d["recovered_suppressed"] = ""
    return d


def _line(r: dict, with_inc: bool = True) -> str:
    s = (f"gain {_f(r['gain'])} [{_f(r['gain_lo'])}, {_f(r['gain_hi'])}]  {_share(r)}  p {_f(r['wilcoxon_p'], '.3g')}  "
         f"gaining {int(r['landscapes_gaining'])}/{int(r['n_landscapes'])}  price {_f(r['price'], '.4f')}")
    if with_inc and r.get("over_learned_inc") is not None and np.isfinite(r.get("over_learned_inc", np.nan)):
        s += (f"  vs learned {_f(r['over_learned_inc'])} [{_f(r['over_learned_inc_lo'])}, "
              f"{_f(r['over_learned_inc_hi'])}] p {_f(r['over_learned_inc_p'], '.3g')}")
    return s


def _chosen_line(cell: str, c: dict, key: str, inc_key: str, holm: bool = True) -> str:
    t = c[key]
    n = int(t["n_landscapes"]) if t.get("n_landscapes") is not None else 0
    line = (f"   {cell:<22s} {c['rule'].upper():4s} gain {_f(t['gain'])} [{_f(t['gain_lo'])}, {_f(t['gain_hi'])}]  "
            f"{_share({**t, 'recovered_suppressed': t['recovered_suppressed'] or ''})}  p {_f(t['wilcoxon_p'], '.3g')}"
            + (f" (Holm over cells {_f(c['gain_p_holm_over_cells'], '.3g')})" if holm else "")
            + f"  gaining {int(t['landscapes_gaining'])}/{n}  price {_f(t['price'], '.4f')}")
    o = c.get(inc_key)
    if o:
        line += (f"  vs learned {c['learned_rule_chosen_on_7_11'].upper()} {_f(o['inc'])} [{_f(o['inc_lo'])}, "
                 f"{_f(o['inc_hi'])}] p {_f(o['inc_p'], '.3g')} ahead {o['inc_landscapes_ahead']}/{o['inc_n_landscapes']}")
    return line


def report(per_run: pd.DataFrame, cells: pd.DataFrame, selection: dict, sitting: list[dict],
           known: list[dict], control: list[str]) -> list[str]:
    L = []
    say = L.append
    say("The noise-aware ship rules with the rater's noise variance given to the final refit")
    say("Gains over the standard process (the best single rating) on the deployed design at T = 50, in units of opt_z;")
    say("price = regret added to the clean twin; recovered = gain / cost, a ratio of landscape means with a landscape")
    say("bootstrap; 'no share' where the cost is below 0.01 of opt_z or negative on a landscape (the gain is then the")
    say("result). 'vs learned' is the paired advantage over the same rule after the learned-noise refit. p is a")
    say("two-sided Wilcoxon over the per-landscape values. Post hoc, outside the paper's Holm families; Holm values")
    say("below are over the eight cells of one level only. LogEI and qNEI unless the scope says ten acquisitions.")
    say("")
    say("== 0. The control, the fits and what each level passes")
    for line in control:
        say("   " + line)
    kn = per_run[per_run["arm"] == "output-boba-knownnoise"]
    if len(kn):
        ok = int((kn["known_var_max_abs_diff"].astype(float) == 0).sum())
        say(f"   output-boba-knownnoise: its logged known_noise_var equals the mag_onset vector exactly in {ok:,} of "
            f"{len(kn):,} runs (max |difference| {kn['known_var_max_abs_diff'].astype(float).max():g})")
    say(f"   GP fits {int(per_run['n_fits'].sum()):,}; failed fits {int(per_run['n_fit_failures'].sum()):,}; runs with a fit "
        f"warning {int((per_run['n_fit_warnings'] > 0).sum()):,} of {len(per_run):,}")
    for lv, text in LEVELS.items():
        say(f"   {lv:<14s} {text}")
    noisy = per_run[(per_run["arm"] == "output-boba") & ~per_run["baseline"] & per_run["acquisition"].isin(PAIR_ACQS)]
    if len(noisy):
        say("   the variance estimates, median (10th, 90th percentile) of sigma_hat^2 / sigma_e^2, LogEI and qNEI, seeds 7-16:")
        for model in PROCESSES:
            x = noisy[noisy["error_model"] == model]
            parts = []
            for lv in ("retest3", "inside_random", "inside_top"):
                q = (x[f"sigma2_{lv}"].astype(float) / x["jitter_std"].astype(float) ** 2).quantile([0.5, 0.1, 0.9])
                parts.append(f"{lv} {q.iloc[0]:.2f} ({q.iloc[1]:.2f}, {q.iloc[2]:.2f})")
            say(f"      {model:<8s} " + "   ".join(parts))
        a = noisy["alarm"].astype(float)
        say(f"   the CUSUM alarmed in {int(np.isfinite(a).sum()):,} of {len(noisy):,} noisy runs (all seeds, four processes)")

    say("")
    say("== 1. Price without error (clean twin), by level and rule")
    oz = aba.dataset_scale("opt_z")
    for label, arm, acqs, seeds in (("LogEI and qNEI, seeds 7-16", "output-boba", PAIR_ACQS, ALL_SEEDS),
                                    ("ten acquisitions, seeds 7-16", "output-boba", TEN, ALL_SEEDS),
                                    ("ten acquisitions, seeds 27-36", "output-boba-confirmatory", TEN, UNTOUCHED_SEEDS)):
        cl = per_run[(per_run["arm"] == arm) & per_run["baseline"] & per_run["acquisition"].isin(acqs)
                     & per_run["seed"].isin(seeds)]
        if cl.empty:
            continue
        z = cl["dataset"].map(oz)
        rough = cl["learned_gp_noise_sd"].astype(float) > 0.2
        parts = []
        for r in ("lcb1", "lcb2"):
            pr = (cl[f"regret_learned_{r}"] - cl["regret_best_observed"]) / z
            parts.append(f"{r.upper()} {pr[rough].sum() / pr.sum():.0%}" if pr.sum() > 0 else f"{r.upper()} n/a")
        same = [int((cl[f"regret_mag_{r}"] == cl["regret_best_observed"]).sum()) for r in RULES]
        say(f"   {label}: {rough.mean():.0%} of {len(cl):,} clean twins have a learned noise SD above 0.2 landscape SDs "
            f"and carry this share of the learned rule's summed price: " + ", ".join(parts)
            + f"; the floor refit's choice has exactly the standard choice's regret in {same[0]:,}/{same[1]:,}/{same[2]:,} "
            f"(PM/LCB1/LCB2) of them")
    for scope, seeds in (("four processes", "7-16"), ("four processes", "12-16"), ("ten acquisitions, gaussian", "7-16"),
                         ("ten acquisitions, gaussian untouched", "27-36"), ("gaussian untouched", "27-36")):
        say(f"   {scope}, seeds {seeds}:")
        for lv in ALL_LEVELS:
            parts = []
            for r in RULES:
                x = _row(cells, scope, seeds, lv, r, "pooled_all_cells")
                if x is not None:
                    parts.append(f"{r.upper()} {x['price']:.4f} [{x['price_lo']:.4f}, {x['price_hi']:.4f}]")
            if parts:
                say(f"      {lv:<14s} " + "   ".join(parts))

    say("")
    four = selection["four processes"]
    say("== 2. Four processes, seeds 12-16 (held out), the rule chosen per cell on seeds 7-11")
    for lv, per_cell in four["by_cell"].items():
        say(f"-- {lv}: {LEVELS[lv]}")
        for cell in CELL_ORDER:
            say(_chosen_line(cell, per_cell[cell], "test", "over_learned_chosen"))

    gauss = selection.get("gaussian")
    if gauss:
        say("")
        say("== 2b. Gaussian error, the rule chosen per cell on seeds 7-11 of the main sweep; scored on seeds 12-16 and")
        say("       on the untouched seeds 27-36 of the replication arm (second line of each cell)")
        for lv, per_cell in gauss["by_cell"].items():
            if lv in CUSUM_LEVELS or lv in ("mag_half", "mag_double"):
                continue
            say(f"-- {lv}: {LEVELS[lv]}")
            for cell in CELL_ORDER:
                say(_chosen_line(cell, per_cell[cell], "test", "over_learned_chosen"))
                if "untouched" in per_cell[cell]:
                    say(_chosen_line("  seeds 27-36", per_cell[cell], "untouched", "untouched_over_learned_chosen",
                                     holm=False))

    say("")
    say("== 3. Four processes, one rule per level for all cells, chosen on seeds 7-11 by the gain over all eight cells")
    for lv, d in four["one_rule"].items():
        say(f"-- {lv}: {d['rule'].upper()} (seeds 7-11 pooled gains: "
            + ", ".join(f"{r.upper()} {v:+.4f}" for r, v in d["train_pooled_gain"].items()) + ")")
        for cell in CELL_ORDER + [f"pooled_ge_{POOL_MIN:g}sigma", "pooled_all_cells"]:
            t = d["test"][cell]
            say(f"   {cell:<22s} gain {_f(t['gain'])} [{_f(t['gain_lo'])}, {_f(t['gain_hi'])}]  "
                f"{_share({**t, 'recovered_suppressed': t['recovered_suppressed'] or ''})}  p {_f(t['wilcoxon_p'], '.3g')}  "
                f"price {_f(t['price'], '.4f')}")
    for key, label in (("inside_protocol", "inside the budget"), ("outside_protocol", "outside the budget")):
        if four.get(key):
            ip = four[key]
            say(f"   {label}, seeds 7-11 choose {ip['chosen']} with {ip['rule'].upper()} (pooled gains over the eight "
                "cells at each one's rule: " + ", ".join(f"{k} {v:+.4f}" for k, v in ip["train_pooled_gain_at_its_rule"].items())
                + ")")

    say("")
    say("== 4. Small error: the 0.05 and 0.25 sigma cells, four processes, every rule (seeds 12-16 and 7-16)")
    for seeds in ("12-16", "7-16"):
        for cell in ("0.05sigma_from_trial_1", "0.05sigma_from_trial_21", "0.25sigma_from_trial_1", "0.25sigma_from_trial_21"):
            parts = []
            for lv in ("learned", "mag", "mag_onset", "retest3", "retest3_cusum", "inside_random", "inside_top"):
                g = [_row(cells, "four processes", seeds, lv, r, cell) for r in RULES]
                if all(x is not None for x in g):
                    parts.append(f"{lv} " + "/".join(f"{x['gain']:+.4f}" for x in g))
            say(f"   seeds {seeds} {cell:<24s} (PM/LCB1/LCB2) " + "  ".join(parts))

    say("")
    say("== 5. Gaussian error, LogEI and qNEI: held out and untouched")
    for cell in ("1sigma_from_trial_1", "1sigma_from_trial_21", "5sigma_from_trial_1", "5sigma_from_trial_21",
                 "0.25sigma_from_trial_1"):
        say(f"-- {cell}")
        for scope, seeds in (("gaussian", "7-11"), ("gaussian", "12-16"), ("gaussian untouched", "27-36")):
            for lv in ("learned", "mag", "mag_onset", "retest3", "inside_random", "inside_top"):
                for r in ("lcb1", "lcb2"):
                    x = _row(cells, scope, seeds, lv, r, cell)
                    if x is not None:
                        say(f"   seeds {seeds:<5s} {lv:<13s} {r.upper():4s} {_line(x)}")

    say("")
    say("== 6. The cautious rule's scope: ten acquisitions, gaussian error")
    for cell in (f"pooled_ge_{POOL_MIN:g}sigma", "1sigma_both_onsets", "0.25sigma_both_onsets", "5sigma_both_onsets",
                 "0.05sigma_both_onsets"):
        say(f"-- {cell}")
        for scope, seeds in (("ten acquisitions, gaussian", "7-16"), ("ten acquisitions, gaussian", "7-11"),
                             ("ten acquisitions, gaussian", "12-16"), ("ten acquisitions, gaussian untouched", "27-36")):
            for lv in ("learned", "mag", "mag_onset", "mag_half", "mag_double", "retest3", "inside_random", "inside_top"):
                for r in ("lcb1", "lcb2"):
                    x = _row(cells, scope, seeds, lv, r, cell)
                    if x is not None:
                        say(f"   seeds {seeds:<5s} {lv:<13s} {r.upper():4s} {_line(x)}")

    if sitting:
        say("")
        say("== 7. Against the final sitting at the k seeds 7-11 chose, on the same runs (seeds 12-16)")
        say("   rule_over_sitting = (regret of the sitting - regret of the rule) / opt_z, landscape means; the rule is the")
        say("   level's choice on seeds 7-11 in the cell (section 2).")
        for r in sitting:
            say(f"   {r['scope']:<15s} {r['cell']:<22s} k = {r['sitting_k']:2d}  {r['level']:<13s} {r['rule'].upper():4s} "
                f"{r['rule_over_sitting']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] p {_f(r['p'], '.3g')} ahead "
                f"{r['landscapes_ahead']}/{r['n_landscapes']}  price rule {r['price_rule']:.4f}, "
                f"sitting {r['price_sitting']:.4f}")

    if known:
        say("")
        say("== 8. The known-noise arm (GP told the variance during the whole run) against the final refit alone")
        say("   gaussian, seeds 7-11; both against the MAIN sweep's standard process; 'arm - final' is the paired advantage")
        say("   of the arm's rule over the same rule after a final refit told the same variance (mag_onset).")
        for r in known:
            if r["cell"] not in CELL_ORDER + [f"pooled_ge_{POOL_MIN:g}sigma"]:
                continue
            say(f"   {r['rule'].upper():4s} {r['cell']:<22s} arm gain {r['arm_gain']:+.4f} [{r['arm_gain_lo']:+.4f}, "
                f"{r['arm_gain_hi']:+.4f}] price {r['arm_price']:+.4f}  final refit gain {r['final_refit_gain']:+.4f} "
                f"price {r['final_refit_price']:+.4f}  arm - final {r['arm_over_final_refit_inc']:+.4f} "
                f"[{r['arm_over_final_refit_inc_lo']:+.4f}, {r['arm_over_final_refit_inc_hi']:+.4f}] "
                f"p {_f(r['arm_over_final_refit_inc_p'], '.3g')}")

    say("")
    say("== 9. By error process, seeds 12-16, the cells where the levels differ")
    for cell in ("1sigma_from_trial_1", "1sigma_from_trial_21", "5sigma_from_trial_1", "5sigma_from_trial_21"):
        say(f"-- {cell}")
        for proc in PROCESSES:
            parts = []
            for lv in ("learned", "mag", "mag_onset", "retest3", "cusum", "retest3_cusum", "inside_random", "inside_top"):
                g = [_row(cells, proc, "12-16", lv, r, cell) for r in ("lcb1", "lcb2")]
                if all(x is not None for x in g):
                    parts.append(f"{lv} {g[0]['gain']:+.3f}/{g[1]['gain']:+.3f}")
            say(f"   {proc:<8s} (LCB1/LCB2) " + "  ".join(parts))
    return L


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


TABLE_LABELS = {
    "learned": "nothing (learned noise)", "mag": "the magnitude", "mag_onset": "the magnitude and the onset",
    "mag_half": "the SD, understated twofold", "mag_double": "the SD, overstated twofold",
    "retest3": r"three test--retest pairs$^\dagger$", "cusum": "the magnitude, CUSUM onset",
    "retest3_cusum": r"three pairs$^\dagger$, CUSUM onset", "inside_random": r"three random re-ratings$^\ddagger$",
    "inside_top": r"the top three re-rated$^\ddagger$",
}
TABLE_CELLS = ("1sigma_from_trial_1", "1sigma_from_trial_21", "5sigma_from_trial_1", "5sigma_from_trial_21")
SMALL_CELLS = ("0.05sigma_from_trial_1", "0.05sigma_from_trial_21", "0.25sigma_from_trial_1",
               "0.25sigma_from_trial_21")


def _tex_num(v: float, digits: int = 3) -> str:
    """A signed decimal for the table; -0.000 prints as 0.000."""
    s = f"{v:+.{digits}f}"
    return "$" + ("0." + "0" * digits if float(s) == 0 else s) + "$"


def latex_table(selection: dict) -> str:
    """paper/tables/known_variance.tex: each level at the one rule seeds 7-11 choose for it, scored on seeds 12-16
    (four processes): the small-error range, the four larger cells, the pooled share at 0.25 sigma and above, the
    price. The paper's tab:knownvar inputs it; this script is its producer."""
    one = selection["four processes"]["one_rule"]
    pooled = f"pooled_ge_{POOL_MIN:g}sigma"
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"the final refit is given & $0.05$--$0.25\sigma$ & $1\sigma$, trial 1 & $1\sigma$, trial 21 & "
             r"$5\sigma$, trial 1 & $5\sigma$, trial 21 & pooled & price \\", r"\midrule"]
    for lv in ALL_LEVELS:
        if lv not in one:
            continue
        t = one[lv]["test"]
        small = [t[c]["gain"] for c in SMALL_CELLS]
        p = t[pooled]
        share = (r"\multicolumn{1}{c}{--}" if p.get("recovered_suppressed") else
                 f"${100 * p['recovered']:.1f}\\%$ $[{100 * p['recovered_lo']:.0f}, {100 * p['recovered_hi']:.0f}]$")
        price = float(t[TABLE_CELLS[0]]["price"])
        cells = " & ".join(_tex_num(t[c]["gain"]) for c in TABLE_CELLS)
        lines.append(f"{TABLE_LABELS[lv]} & {_tex_num(min(small))} to {_tex_num(max(small))} & {cells} & {share} & "
                     f"${abs(price):.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def score_stage(per_run: pd.DataFrame, control: list[str]) -> None:
    opt_z = aba.dataset_scale("opt_z")
    t0 = time.time()
    cells, frames = score_scopes(per_run, opt_z)
    cells.to_csv(CELLS, index=False)
    selection = {"four processes": select_on_train(cells, frames, "four processes")}
    if "gaussian" in frames:
        selection["gaussian"] = select_on_train(cells, frames, "gaussian", untouched="gaussian untouched")
    sitting = sitting_comparison(per_run, opt_z, selection)
    known = knownnoise_comparison(per_run, opt_z)
    record = {"producer": "scripts/checks/known_variance_ship.py", "post_hoc": True,
              "holm_note": "outside the paper's Holm families; Holm here is over the eight cells of one level",
              "levels": LEVELS, "floor": FLOOR, "retest_pairs": R_PAIRS, "selection": selection,
              "versus_sitting": sitting, "known_noise_arm": known, "control": control}
    SELECTION.write_text(json.dumps(_jsonable(record), indent=1), encoding="utf-8")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(latex_table(selection), encoding="utf-8")
    lines = report(per_run, cells, selection, sitting, known, control)
    REGISTER.parent.mkdir(parents=True, exist_ok=True)
    REGISTER.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {_rel(CELLS)}, {_rel(SELECTION)}, {_rel(TABLE)} and {_rel(REGISTER)} "
          f"({time.time() - t0:.0f}s)")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "refit", "score"), default="all",
                   help="all: refit what the cache lacks (where the run logs are), then score; score: from the per-run table")
    p.add_argument("--arms", default=",".join(ARM_SPECS), help="arms to refit (comma-separated)")
    p.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"worker processes (at most {MAX_WORKERS})")
    p.add_argument("--datasets", default=None, help="restrict the refit to these landscapes (a smoke test)")
    p.add_argument("--max-files", type=int, default=None, help="refit at most this many new runs per arm (timing)")
    p.add_argument("--overwrite", action="store_true", help="discard the refit cache and start again")
    args = p.parse_args(argv)
    unknown = [a for a in args.arms.split(",") if a and a not in ARM_SPECS]
    if unknown:
        p.error(f"unknown arms {unknown}; choose from {list(ARM_SPECS)}")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    arms = [a for a in args.arms.split(",") if a]
    datasets = {d for d in args.datasets.split(",") if d} if args.datasets else None
    if args.stage in ("all", "refit"):
        present = [a for a in arms if (REPO / a).is_dir() and any((REPO / a).glob("*/bo_sensor_error_*.csv"))]
        if present:
            refit_stage(present, args.workers, datasets, args.max_files, args.overwrite)
        missing = sorted(set(arms) - set(present))
        if missing:
            print(f"no run logs for {missing} here; their refits come from the cache or the per-run table")
        if any(cache_path(a).is_file() for a in arms):
            per_run = assemble([a for a in arms if cache_path(a).is_file()])
            control = control_check(per_run)
            # mtime 0: the same refits write the same bytes, so a rerun leaves the tracked file unchanged.
            per_run.to_csv(PER_RUN, index=False, compression={"method": "gzip", "mtime": 0})
            print(f"Wrote {_rel(PER_RUN)} ({len(per_run):,} runs)")
        if args.stage == "refit":
            return
    if not PER_RUN.is_file():
        raise SystemExit(f"no per-run table {PER_RUN}; run the refit stage where the run logs are")
    per_run = pd.read_csv(PER_RUN, float_precision="round_trip", low_memory=False)
    per_run["baseline"] = per_run["baseline"].astype(str).str.lower().isin(("true", "1"))
    per_run["variant"] = per_run["variant"].fillna("").astype(str)
    if args.datasets:
        print("note: --datasets restricts only the refit; the score stage reads the whole per-run table")
    control = control_check(per_run)
    score_stage(per_run, control)


if __name__ == "__main__":
    main()
