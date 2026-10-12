"""Selection loss as a winner's curse on the visited set.

Section 5 splits the deployed design's regret at the final trial into search
loss (the run never visited anything better) and selection loss (it visited
something better and shipped the wrong one; eq. decomp). This check asks how
much of the selection term the visited designs and the error process explain
on their own. For every noisy run it keeps the designs the run visited, in the
order it visited them, redraws the rating error of the run's own process on
them, applies the standard ship rule and averages the selection loss over
DRAWS fresh error paths:

    pred(V) = E_e [ max_t g_t  -  g_d ],    d = first index of max_t (g_t + e_t)

with g_t the exact objective of trial t in landscape SDs. The error law is the
simulator's (bo_sensor_error_simulation.apply_sensor_error) to the bit: ratings
are exact for t <= t0; after the onset GAUSSIAN adds N(0, s^2), BIAS adds the
offset of the run's file-name variant (bias<b>, b = s in every arm read here)
plus N(0, s^2), DRIFT adds the ramp s (t - t0)/(T - t0) plus N(0, s^2), and
AR(1) starts stationary at t0 + 1 and carries its state with rho = 0.8 (rho<r>
in the variant otherwise). ``rate`` computes the ratings with the simulator's
own floating-point operations, so fed a run's own noise seed it must reproduce
the logged ratings exactly; ``scan`` checks that for every run it reads. Ties go
to the first index, as np.argmax does in the simulator; under continuous error
only exact (pre-onset) ratings can tie, and those are of designs with the same
true value. pred(V) is the paper's one-shot selection loss (eq:frag,
app:mediation) taken over the run's own visited designs and error process
instead of a random design and independent noise. It uses the visited designs'
TRUE values, so it explains the selection loss; it is not a rule a study could run.

What the comparison means. Were the visited designs fixed in advance, a run's
realised selection loss would be one draw from the distribution pred(V)
averages, so realised / predicted would be 1 in expectation. The designs were
chosen by a search that saw the same errors the ship rule then reads, so
realised / predicted - 1 is what that adaptivity adds. The same fresh error
paths are applied, as common random numbers, to the CLEAN twin's visited
designs: pred(noisy set) / pred(clean set) says how much more selection-prone
error has made the set itself (the clean run's own selection loss is exactly
zero; this is what it would be if only the final choice saw the error).

Scope. The main sweep (output-boba: ten model-based acquisitions, seeds 7-16,
four processes, four magnitudes, onsets 0 and 20, T = 50), the replication
planned in advance (output-boba-confirmatory: seeds 27-36, GAUSSIAN only, the
untouched seeds), and the budget arms (output-boba-budget25/-budget100: six
acquisitions, seeds 7-11, GAUSSIAN, T = 25 and 100 with onsets 0 and 0.4 T).
The budget table compares them with the main sweep restricted to the same six
acquisitions, seeds and process.

Estimands. Every aggregate is a ratio of landscape means: each run's value is
divided by its landscape's opt_z (the regret unit of the paper; landscape SDs
where a row says so), averaged within a landscape over acquisitions and seeds,
then over the twenty landscapes, and the interval is a 95% percentile interval
over 2000 landscape resamples (one resample matrix shared by every estimand).
The landscapes share their random numbers by design, so the bootstrap treats
correlated clusters as independent. A selection SHARE is selection / deployed
excess over the clean twin, screened like every share of a cost
(analyse_boba_adaptations.share_of_reference: suppressed when the pooled
deployed excess is below 0.01 of opt_z or negative on any landscape); the
predicted share divides the predicted selection loss by the same realised
deployed excess, so predicted share / realised share = predicted / realised
selection loss. An onset ratio is a ratio of landscape means of the onset-0
and the late-onset value at the final trial, suppressed by the same screen on
its late-onset denominator; in the main sweep the final-trial search excess from
trial 21 is negative on one to four landscapes at every magnitude, so its onset
ratio is never reported, and the onset table gives the differences instead (deployed =
search + selection, with the search part's share of the difference). The
deployed onset ratio is exactly the search and selection onset ratios averaged
with the late-onset shares as weights.

Checks. ``scan`` stops unless every run's realised selection loss reproduces
its log (deployed minus search regret at the final trial, to 1e-12) and every
clean twin's ratings are exact, and it records how exactly the error stream
rebuilt from the run's noise seed reproduces each logged rating and error.
``report`` stops unless every one of those streams is reproduced exactly, the
main sweep's per-run deployed and search regret equal ship_rules_per_run.csv,
and the cell means and shares of regret_decomposition.csv and the per-landscape
deployed excess and selection share of per_landscape_headline.csv are
reproduced to 1e-12.

Subcommands

    scan     read every per-run log of the four arms, replay its error stream,
             and run the Monte Carlo (about 5 minutes on 4 workers); writes the
             per-run table winners_curse_runs.parquet (git-ignored)
    report   the tables, the checks against the decomposition, and the register
             file; prints exactly what it writes to
             register_checks/winners_curse.txt

    python scripts/checks/winners_curse.py scan --workers 4
    python scripts/checks/winners_curse.py report

Outputs (output-boba/analysis/review/): winners_curse_runs.parquet (per run),
winners_curse_cells.csv (per arm, process, magnitude and onset),
winners_curse_landscapes.csv (per landscape; make_paper_figures.py draws the
main sweep's 1 sigma, first-rating rows), winners_curse_onset.csv (the onset
decomposition), winners_curse_budget.csv (T = 25, 50, 100) and
winners_curse_budget_contrast.csv (T = 100 against T = 25),
winners_curse_checks.csv (the reproduction checks) and register_checks/winners_curse.txt;
and, in paper/tables/, winners_curse.tex and winners_curse_landscapes.tex (the main
sweep's cells, and its headline cell per landscape, as LaTeX tabulars that the paper
inputs; this script is their producer, not make_boba_paper_tables.py).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OUT_DIR = REPO / "output-boba" / "analysis" / "review"
TABLES_DIR = REPO / "paper" / "tables"
RUNS_PATH = OUT_DIR / "winners_curse_runs.parquet"
REGISTER_TXT = OUT_DIR / "register_checks" / "winners_curse.txt"

# arm key -> (directory, label). T is read from each log, never assumed.
ARMS = {
    "main": ("output-boba", "main sweep, ten acquisitions, seeds 7-16"),
    "untouched": ("output-boba-confirmatory", "untouched seeds 27-36, ten acquisitions, GAUSSIAN"),
    "T25": ("output-boba-budget25", "T = 25, six acquisitions, seeds 7-11, GAUSSIAN"),
    "T100": ("output-boba-budget100", "T = 100, six acquisitions, seeds 7-11, GAUSSIAN"),
}
PROCESSES = ("gaussian", "bias", "drift", "ar1")
AR1_RHO = 0.8                 # the simulator's default; a run that changed it says so in its variant
DRAWS = 4000                  # fresh error paths per run
MC_SEED = 20261009
BOOT = 2000
BOOT_SEED = 20261010
NEAR = 0.1                    # "near the best": within 0.1 landscape SD of the best visited design
LOG_TOL = 1e-12               # realised selection must reproduce the log's deployed - search regret
DECOMP_TOL = 1e-12            # and the cell means of regret_decomposition.csv
BUDGET_ACQS = ("logei", "ei", "pi", "ucb", "qucb", "qnei")
BUDGET_SEEDS = (7, 8, 9, 10, 11)
HEADLINE = (1.0, 0)           # 1 sigma from the first rating
PAIR_KEYS = ["dataset", "acquisition", "seed"]
USECOLS = ("iteration", "objective_true", "objective_observed", "error_magnitude", "y_opt",
           "simple_regret_true", "inference_simple_regret_true", "dataset", "acquisition", "seed",
           "error_model", "jitter_std", "jitter_iteration")


# ---------------------------------------------------------------------------
# The error law and the ship rule
# ---------------------------------------------------------------------------


def process_parameters(error_model: str, variant: str) -> tuple[float, float]:
    """(bias offset, AR(1) rho) of a run, from its file-name variant.

    A run's variant lives only in its file name (AGENTS.md). The synthetic
    driver writes ``bias<b>`` for every BIAS run and ``rho<r>`` only when rho is
    not 0.8. Any other part names a setting this law does not model, so it raises.
    """
    if error_model not in PROCESSES:
        raise ValueError(f"error model {error_model!r} is not one of {PROCESSES}")
    bias, rho = 0.0, AR1_RHO
    parts = [p for p in str(variant or "").split("_") if p]
    for part in parts:
        if error_model == "bias" and part.startswith("bias"):
            bias = float(part[len("bias"):])
        elif error_model == "ar1" and part.startswith("rho"):
            rho = float(part[len("rho"):])
        else:
            raise ValueError(f"variant part {part!r} of a {error_model} run is not modelled here")
    if error_model == "bias" and not any(p.startswith("bias") for p in parts):
        raise ValueError(f"a bias run without a bias<b> variant ({variant!r}): its offset is unknown")
    return bias, rho


def stream_normals(process: str, n: int, rng: np.random.Generator, m: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Standard normals for n post-onset trials, in the order the simulator draws them.

    Each post-onset trial draws its jitter, rng.normal(0, s); AR(1) then draws its
    innovation on every trial after the first noisy one (the jitter it draws there
    is discarded). Returns (z_jitter, z_innovation), each (m, n); column 0 of the
    innovations is never used. With m = 1 and the run's jitter generator this is
    the run's own stream; with m > 1 each row is an independent path.
    """
    if n < 1:
        return np.zeros((m, 0)), np.zeros((m, 0))
    if process == "ar1":
        z = rng.standard_normal((m, 2 * n - 1))
        z_jit = np.empty((m, n))
        z_inn = np.zeros((m, n))
        z_jit[:, 0] = z[:, 0]
        z_jit[:, 1:] = z[:, 1::2]
        z_inn[:, 1:] = z[:, 2::2]
        return z_jit, z_inn
    return rng.standard_normal((m, n)), np.zeros((m, n))


def rate(true_post: np.ndarray, z_jit: np.ndarray, z_inn: np.ndarray, process: str, sigma: float,
         onset: int, T: int, bias: float = 0.0, rho: float = AR1_RHO) -> np.ndarray:
    """Ratings of the post-onset trials t = onset+1 .. T, shape (m, n).

    The operations are apply_sensor_error's, in its order, so that the run's own
    stream reproduces its logged ratings bit for bit: jitter = 0.0 + s z (what
    rng.normal(0, s) returns); BIAS (g + b) + jitter; DRIFT (g + s (t - t0)/span)
    + jitter with span = max(1, T - t0); AR(1) g + jitter on the first noisy
    trial, then (g + rho e_prev) + s sqrt(1 - rho^2) z with e_prev the previous
    rating minus its true value.
    """
    g = np.asarray(true_post, dtype=float)
    n = g.shape[-1]
    jitter = sigma * np.asarray(z_jit, dtype=float)
    if process == "gaussian":
        return g + jitter
    if process == "bias":
        return (g + bias) + jitter
    if process == "drift":
        span = max(1, T - onset)
        ramp = sigma * (np.arange(onset + 1, T + 1) - onset) / span
        return (g + ramp) + jitter
    if process == "ar1":
        scale = sigma * np.sqrt(max(0.0, 1.0 - rho ** 2))
        innovation = scale * np.asarray(z_inn, dtype=float)
        obs = np.empty(jitter.shape)
        obs[:, 0] = g[0] + jitter[:, 0]
        prev = obs[:, 0] - g[0]
        for j in range(1, n):
            obs[:, j] = (g[j] + rho * prev) + innovation[:, j]
            prev = obs[:, j] - g[j]
        return obs
    raise ValueError(f"unknown error process {process!r}")


def ship_indices(ratings: np.ndarray) -> np.ndarray:
    """The standard ship rule on each row: the FIRST index of the best rating (np.argmax)."""
    return np.asarray(ratings).argmax(axis=-1)


def realised_selection(true: np.ndarray, observed: np.ndarray) -> float:
    """max g - g[first argmax of the ratings]: the run's own selection loss."""
    g = np.asarray(true, dtype=float)
    return float(g.max() - g[int(np.argmax(np.asarray(observed, dtype=float)))])


def selection_draws(g: np.ndarray, onset: int, process: str, sigma: float, z_jit: np.ndarray,
                    z_inn: np.ndarray, bias: float = 0.0, rho: float = AR1_RHO):
    """Selection loss, ship index and the shipped design's rating error on each path.

    Rows of (z_jit, z_inn) are error paths; g is one visited set in trial order.
    """
    g = np.asarray(g, dtype=float)
    T = len(g)
    obs = rate(g[onset:], z_jit, z_inn, process, sigma, onset, T, bias, rho)
    m = obs.shape[0]
    if onset:
        ratings = np.empty((m, T))
        ratings[:, :onset] = g[:onset]
        ratings[:, onset:] = obs
    else:
        ratings = obs
    idx = ship_indices(ratings)
    loss = g.max() - g[idx]
    error = ratings[np.arange(m), idx] - g[idx]
    return loss, idx, error


def expected_selection(g: np.ndarray, onset: int, process: str, sigma: float, draws: int,
                       rng: np.random.Generator, bias: float = 0.0, rho: float = AR1_RHO) -> dict:
    """pred(V) by Monte Carlo: mean selection loss over ``draws`` fresh error paths."""
    z_jit, z_inn = stream_normals(process, len(g) - onset, rng, draws)
    loss, idx, error = selection_draws(g, onset, process, sigma, z_jit, z_inn, bias, rho)
    return {"mean": float(loss.mean()), "var": float(loss.var(ddof=1)), "p_wrong": float((loss > 0).mean()),
            "p_post": float((idx >= onset).mean()), "error": float(error.mean())}


def path_seed(arm_dir: str, file: str) -> np.random.SeedSequence:
    """The seed of one run's fresh error paths: MC_SEED and 128 bits of the run's path.

    A 32-bit hash would give about one colliding pair among the ~90,000 runs read
    here (it did, under CRC32); 128 bits make a collision negligible.
    """
    digest = hashlib.sha256(f"{arm_dir}/{file}".encode("utf-8")).digest()
    return np.random.SeedSequence([MC_SEED, int.from_bytes(digest[:16], "little")])


def jitter_seed(seed: int, acq_index: int, onset: int, sigma: float, em_index: int) -> np.random.SeedSequence:
    """The synthetic driver's noise seed (bo_synthetic_error_simulation.run_task)."""
    return np.random.SeedSequence([int(seed), int(acq_index), int(onset), int(round(float(sigma) * 1_000_000)),
                                   int(em_index)])


def replay_ratings(g: np.ndarray, onset: int, process: str, sigma: float, seed_seq: np.random.SeedSequence,
                   bias: float = 0.0, rho: float = AR1_RHO) -> np.ndarray:
    """The run's own ratings rebuilt from its noise seed (exact before the onset)."""
    g = np.asarray(g, dtype=float)
    z_jit, z_inn = stream_normals(process, len(g) - onset, np.random.default_rng(seed_seq), 1)
    out = g.copy()
    out[onset:] = rate(g[onset:], z_jit, z_inn, process, sigma, onset, len(g), bias, rho)[0]
    return out


def near_best(g: np.ndarray, within: float) -> tuple[int, int]:
    """Trials, and distinct designs (distinct true values), within ``within`` of the best visited."""
    g = np.asarray(g, dtype=float)
    close = g[g >= g.max() - within]
    return int(close.size), int(np.unique(close).size)


# ---------------------------------------------------------------------------
# scan: one task per (arm, landscape, acquisition, seed)
# ---------------------------------------------------------------------------


def read_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in USECOLS, float_precision="round_trip")
    df = df.sort_values("iteration", kind="stable").reset_index(drop=True)
    if not np.array_equal(df["iteration"].to_numpy(), np.arange(1, len(df) + 1)):
        raise ValueError(f"{path}: iterations are not 1..T")
    return df


def _final_regrets(df: pd.DataFrame, path: str) -> dict:
    """Deployed and search regret at the final trial, rebuilt as rescore_ship_rules does and checked."""
    g = df["objective_true"].to_numpy(dtype=float)
    y = df["objective_observed"].to_numpy(dtype=float)
    y_opt = float(df["y_opt"].iloc[0])
    d, b = int(np.argmax(y)), int(np.argmax(g))
    deployed, search = y_opt - float(g[d]), y_opt - float(g[b])
    logged_dep = float(df["inference_simple_regret_true"].iloc[-1])
    logged_search = float(df["simple_regret_true"].iloc[-1])
    if abs(deployed - logged_dep) > LOG_TOL or abs(search - logged_search) > LOG_TOL:
        raise ValueError(f"{path}: rebuilt deployed/search regret {deployed!r}/{search!r} disagree with the log "
                         f"{logged_dep!r}/{logged_search!r}")
    return {"g": g, "y": y, "y_opt": y_opt, "ship_idx": d, "best_idx": b, "deployed": deployed,
            "search": search, "T": int(len(df))}


def _check_identity(df: pd.DataFrame, info: dict, path: str) -> None:
    """The log must be the run its file name says it is."""
    want = {"dataset": info["dataset"], "acquisition": info["acquisition"], "seed": info["seed"]}
    if not info["baseline"]:
        want.update(error_model=info["error_model"], jitter_iteration=info["jitter_iteration"])
    for col, value in want.items():
        got = df[col].iloc[0]
        if str(got) != str(value):
            raise ValueError(f"{path}: column {col}={got!r} but the file name says {value!r}")
    if not info["baseline"] and abs(float(df["jitter_std"].iloc[0]) - float(info["jitter_std"])) > 1e-12:
        raise ValueError(f"{path}: jitter_std {df['jitter_std'].iloc[0]!r} but the file name says {info['jitter_std']!r}")


def scan_group(task: dict) -> list[dict]:
    """The clean twin and every noisy run of one (arm, landscape, acquisition, seed)."""
    draws = int(task["draws"])
    clean_df = read_log(task["clean"]["path"])
    _check_identity(clean_df, task["clean"], task["clean"]["path"])
    c = _final_regrets(clean_df, task["clean"]["path"])
    if not np.array_equal(c["y"], c["g"]) or c["deployed"] != c["search"]:
        raise ValueError(f"{task['clean']['path']}: a clean twin with a rating error or a selection loss")
    c_near, c_near_unique = near_best(c["g"], NEAR)
    key = {k: task[k] for k in ("arm", "dataset", "acquisition", "seed")}
    rows = [{**key, "baseline": True, "error_model": "none", "jitter_std": 0.0, "onset": -1, "variant": "",
             "file": task["clean"]["file"], "T": c["T"], "y_opt": c["y_opt"], "deployed": c["deployed"],
             "search": c["search"], "n_near": c_near, "n_near_unique": c_near_unique}]
    for info in task["noisy"]:
        path = info["path"]
        df = read_log(path)
        _check_identity(df, info, path)
        r = _final_regrets(df, path)
        if r["T"] != c["T"]:
            raise ValueError(f"{path}: {r['T']} trials against the clean twin's {c['T']}")
        g, y, T = r["g"], r["y"], r["T"]
        onset, sigma, process = int(info["jitter_iteration"]), float(info["jitter_std"]), info["error_model"]
        bias, rho = process_parameters(process, info["variant"])
        # The run's own error stream, replayed from its noise seed.
        seed_seq = jitter_seed(info["seed"], task["acq_index"], onset, sigma, task["em_index"][process])
        replayed = replay_ratings(g, onset, process, sigma, seed_seq, bias, rho)
        logged_error = df["error_magnitude"].to_numpy(dtype=float)
        # Fresh error paths, seeded by the run's identity alone (worker count and order do not matter);
        # the same paths are applied to the clean twin's visited set (common random numbers).
        rng = np.random.default_rng(path_seed(task["arm_dir"], info["file"]))
        z_jit, z_inn = stream_normals(process, T - onset, rng, draws)
        loss, idx, err = selection_draws(g, onset, process, sigma, z_jit, z_inn, bias, rho)
        loss_c, _, err_c = selection_draws(c["g"], onset, process, sigma, z_jit, z_inn, bias, rho)
        d = r["ship_idx"]
        n_near, n_near_unique = near_best(g, NEAR)
        rows.append({
            **key, "baseline": False, "error_model": process, "jitter_std": sigma, "onset": onset,
            "variant": info["variant"], "file": info["file"], "T": T, "bias": bias, "rho": rho, "draws": draws,
            "y_opt": r["y_opt"], "deployed": r["deployed"], "search": r["search"],
            "selection_direct": realised_selection(g, y),
            "deployed_clean": c["deployed"], "search_clean": c["search"],
            "ship_idx": d, "best_idx": r["best_idx"],
            "wrong_real": float(g[d] < g.max()), "post_real": float(d >= onset),
            "over_real": float(y[d] - g[d]),
            "sel_pred": float(loss.mean()), "sel_pred_var": float(loss.var(ddof=1)),
            "p_wrong_pred": float((loss > 0).mean()), "p_post_pred": float((idx >= onset).mean()),
            "over_pred": float(err.mean()),
            "sel_pred_clean": float(loss_c.mean()), "sel_pred_clean_var": float(loss_c.var(ddof=1)),
            "over_pred_clean": float(err_c.mean()),
            "n_near": n_near, "n_near_unique": n_near_unique,
            "n_near_clean": c_near, "n_near_unique_clean": c_near_unique,
            "n_within_sigma": int((g >= g.max() - sigma).sum()),
            "n_within_sigma_clean": int((c["g"] >= c["g"].max() - sigma).sum()),
            "replay_max_abs_rating": float(np.max(np.abs(replayed - y))),
            "replay_max_abs_error": float(np.max(np.abs((replayed - g) - logged_error))),
            "logged_pre_onset_exact": bool(np.array_equal(y[:onset], g[:onset])
                                           and not np.any(logged_error[:onset])),
        })
    return rows


def build_tasks(draws: int) -> list[dict]:
    """One task per (arm, landscape, acquisition, seed), with its clean twin and noisy runs."""
    import bo_sensor_error_simulation as sim  # the seed recipe's name lists; main process only
    import rescore_ship_rules as rs

    em_index = {p: sim.ERROR_MODEL_CHOICES.index(p) for p in PROCESSES}
    tasks = []
    for arm, (arm_dir, _label) in ARMS.items():
        groups: dict[tuple, dict] = {}
        for info in rs.index_runs(REPO / arm_dir):    # model-free floors are left out, as everywhere
            k = (info["dataset"], info["acquisition"], info["seed"])
            grp = groups.setdefault(k, {"clean": None, "noisy": []})
            if info["baseline"]:
                if info["variant"]:
                    raise ValueError(f"{arm_dir}/{info['file']}: a clean twin with a variant")
                if grp["clean"] is not None:
                    raise ValueError(f"{arm_dir}: two clean twins for {k}")
                grp["clean"] = info
            else:
                grp["noisy"].append(info)
        for (dataset, acq, seed), grp in sorted(groups.items()):
            if grp["clean"] is None:
                raise ValueError(f"{arm_dir}: no clean twin for {(dataset, acq, seed)}")
            # The noise seed uses the acquisition's position in ACQUISITION_CHOICES. The
            # model-based acquisitions hold the same positions as before qkg and replei
            # were inserted (check_provenance.ACQUISITION_ORDER_BEFORE_ROBUST); only the
            # floors moved, and they are not read here. The replay confirms every seed.
            tasks.append({"arm": arm, "arm_dir": arm_dir, "dataset": dataset, "acquisition": acq, "seed": seed,
                          "clean": grp["clean"], "noisy": grp["noisy"], "draws": draws,
                          "acq_index": sim.ACQUISITION_CHOICES.index(acq), "em_index": em_index})
    return tasks


def cmd_scan(args: argparse.Namespace) -> None:
    tasks = build_tasks(args.draws)
    n_noisy = sum(len(t["noisy"]) for t in tasks)
    print(f"{len(tasks)} clean twins and {n_noisy} noisy runs over {len(ARMS)} arms; {args.draws} error paths each",
          flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, part in enumerate(pool.map(scan_group, tasks, chunksize=4)):
            rows.extend(part)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} groups", flush=True)
    runs = pd.DataFrame(rows)
    # NaN on the clean rows, which have no onset; a nullable boolean keeps parquet happy.
    runs["logged_pre_onset_exact"] = runs["logged_pre_onset_exact"].astype("boolean")
    noisy = runs[~runs["baseline"]]
    exact = (noisy["replay_max_abs_rating"] == 0) & (noisy["replay_max_abs_error"] == 0)
    print(f"error stream replayed exactly in {int(exact.sum())} of {len(noisy)} noisy runs "
          f"(largest deviation {noisy[['replay_max_abs_rating', 'replay_max_abs_error']].max().max():.3e}); "
          f"pre-onset ratings exact in {int(noisy['logged_pre_onset_exact'].sum())}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_parquet(RUNS_PATH, index=False)
    print(f"wrote {RUNS_PATH.relative_to(REPO)} ({len(runs)} rows)")


# ---------------------------------------------------------------------------
# report: estimands
# ---------------------------------------------------------------------------


def opt_z_map() -> dict[str, float]:
    import boba_benchmarks as bb

    return {k: float(v["opt_z"]) for k, v in bb.load_stats().items() if "opt_z" in v}


def boot_matrix(n: int) -> np.ndarray:
    """The landscape resamples every interval uses (2000 x n, fixed seed)."""
    return np.random.default_rng(BOOT_SEED).integers(0, n, (BOOT, n))


def boot_ratio(num: np.ndarray, den: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    """Ratio of landscape means, with a percentile landscape-bootstrap interval."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    point = float(num.mean() / den.mean()) if den.mean() != 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = num[B].mean(axis=1) / den[B].mean(axis=1)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_mean(values: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    draws = values[B].mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def share_guard(ref: np.ndarray, part: np.ndarray) -> str:
    """Why a share of ``ref`` is not reported ('' when it is): the paper's share screen."""
    from analyse_boba_adaptations import share_of_reference

    return share_of_reference(pd.DataFrame({"ref": ref, "trt": np.asarray(ref) - np.asarray(part)}))[
        "share_suppressed"]


def with_units(runs: pd.DataFrame, opt_z: dict[str, float]) -> pd.DataFrame:
    """Per-run quantities in units of opt_z (``*_z``), the paper's regret unit."""
    lacking = sorted(set(runs["dataset"]) - set(opt_z))
    if lacking:
        raise KeyError(f"no opt_z for {lacking}")
    out = runs.copy()
    z = out["dataset"].map(opt_z).astype(float)
    out["opt_z"] = z
    # As decompose_regret.decompose: divide each regret by opt_z, then difference.
    out["deployed_z"] = out["deployed"] / z
    out["search_z"] = out["search"] / z
    out["selection_z"] = (out["deployed_z"] - out["search_z"]).clip(lower=0.0)
    out["selection_sd"] = (out["deployed"] - out["search"]).clip(lower=0.0)
    noisy = ~out["baseline"]
    out.loc[noisy, "dep_excess_z"] = out.loc[noisy, "deployed_z"] - out.loc[noisy, "deployed_clean"] / z[noisy]
    out.loc[noisy, "search_excess_z"] = out.loc[noisy, "search_z"] - out.loc[noisy, "search_clean"] / z[noisy]
    for col in ("sel_pred", "sel_pred_clean"):
        out.loc[noisy, f"{col}_z"] = out.loc[noisy, col] / z[noisy]
    # Monte Carlo variance of each run's predicted mean, in opt_z units.
    out.loc[noisy, "sel_pred_mcvar_z"] = out.loc[noisy, "sel_pred_var"] / out.loc[noisy, "draws"] / z[noisy] ** 2
    return out


PER_RUN_MEANS = ["selection_z", "sel_pred_z", "sel_pred_clean_z", "dep_excess_z", "search_excess_z",
                 "selection_sd", "sel_pred", "sel_pred_clean", "wrong_real", "p_wrong_pred", "post_real",
                 "p_post_pred", "over_real", "over_pred", "over_pred_clean", "n_near", "n_near_clean",
                 "n_near_unique", "n_near_unique_clean", "n_within_sigma", "n_within_sigma_clean"]


def landscape_means(block: pd.DataFrame, landscapes: list[str]) -> pd.DataFrame:
    per = block.groupby("dataset")[PER_RUN_MEANS].mean()
    missing = sorted(set(landscapes) - set(per.index))
    if missing:
        raise ValueError(f"cell lacks landscapes {missing}")
    per = per.loc[landscapes]
    n = block.groupby("dataset").size().loc[landscapes]
    # Monte Carlo variance of each landscape mean of the prediction.
    per["mc_var"] = block.groupby("dataset")["sel_pred_mcvar_z"].sum().loc[landscapes] / n ** 2
    per["n_runs"] = n
    return per


def cell_row(block: pd.DataFrame, landscapes: list[str], B: np.ndarray) -> dict:
    """One cell: realised against predicted selection loss, and the visited-set comparison."""
    from scipy.stats import spearmanr

    per = landscape_means(block, landscapes)
    sigma = float(block["jitter_std"].iloc[0]) if block["jitter_std"].nunique() == 1 else float("nan")
    real, pred, dep = per["selection_z"].to_numpy(), per["sel_pred_z"].to_numpy(), per["dep_excess_z"].to_numpy()
    row = {"n_landscapes": len(per), "n_runs": int(per["n_runs"].sum()),
           "dep_excess": float(dep.mean()), "search_excess": float(per["search_excess_z"].mean()),
           "sel_real": float(real.mean()), "sel_pred": float(pred.mean()),
           "sel_pred_mc_se": float(np.sqrt(per["mc_var"].sum()) / len(per))}
    row.update(zip(("ratio", "ratio_lo", "ratio_hi"), boot_ratio(real, pred, B)))
    guard = share_guard(dep, real)
    row["share_suppressed"] = guard
    for name, part in (("share_real", real), ("share_pred", pred)):
        point, lo, hi = boot_ratio(part, dep, B)
        row.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
    clean = per["sel_pred_clean_z"].to_numpy()
    row["sel_pred_cleanset"] = float(clean.mean())
    row.update(zip(("set_effect", "set_effect_lo", "set_effect_hi"), boot_ratio(pred, clean, B)))
    for col in ("n_near", "n_near_clean", "n_near_unique", "n_near_unique_clean", "n_within_sigma",
                "n_within_sigma_clean", "wrong_real", "p_wrong_pred", "post_real", "p_post_pred"):
        row[col] = float(per[col].mean())
    # The rule of thumb, in units of the rater's noise: how far below the best
    # visited design the shipped one lies, and how far its rating overstates it.
    if sigma == sigma and sigma > 0:
        for name, col in (("sel_real_per_sigma", "selection_sd"), ("sel_pred_per_sigma", "sel_pred"),
                          ("over_real_per_sigma", "over_real"), ("over_pred_per_sigma", "over_pred"),
                          ("over_pred_cleanset_per_sigma", "over_pred_clean")):
            point, lo, hi = boot_mean(per[col].to_numpy() / sigma, B)
            row.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
    # Across landscapes: is the landscape ranking of the selection loss predicted?
    # In landscape SDs the two sides share no normaliser; in units of opt_z and as
    # shares they share opt_z or the deployed excess, which inflates a rank correlation.
    row["spearman_landscape_sd"] = float(spearmanr(per["selection_sd"], per["sel_pred"]).statistic)
    row["spearman_landscape_optz"] = float(spearmanr(real, pred).statistic)
    row["spearman_landscape_share"] = (float(spearmanr(real / dep, pred / dep).statistic)
                                       if dep.min() >= 0.01 else float("nan"))
    ratio_l = real / pred
    row["ratio_landscape_min"], row["ratio_landscape_max"] = float(ratio_l.min()), float(ratio_l.max())
    row["landscape_ratio_min_at"] = landscapes[int(np.argmin(ratio_l))]
    row["landscape_ratio_max_at"] = landscapes[int(np.argmax(ratio_l))]
    row["spearman_run"] = float(spearmanr(block["selection_sd"], block["sel_pred"]).statistic)
    return row


def cells_table(runs: pd.DataFrame, landscapes: list[str], B: np.ndarray) -> pd.DataFrame:
    rows = []
    noisy = runs[~runs["baseline"]]
    for arm in ARMS:
        a = noisy[noisy["arm"] == arm]
        processes = sorted(a["error_model"].unique())
        scopes = ([("pooled", a)] if len(processes) > 1 else []) + [(p, a[a["error_model"] == p]) for p in processes]
        for process, block in scopes:
            for (sigma, onset), cell in block.groupby(["jitter_std", "onset"]):
                rows.append({"arm": arm, "error_model": process, "jitter_std": sigma, "onset": int(onset),
                             "T": int(cell["T"].iloc[0]), **cell_row(cell, landscapes, B)})
    return pd.DataFrame(rows)


def landscape_table(runs: pd.DataFrame, landscapes: list[str], opt_z: dict[str, float]) -> pd.DataFrame:
    """Per landscape and cell: realised and predicted selection loss and shares."""
    rows = []
    noisy = runs[~runs["baseline"]]
    for arm in ARMS:
        a = noisy[noisy["arm"] == arm]
        processes = sorted(a["error_model"].unique())
        scopes = ([("pooled", a)] if len(processes) > 1 else []) + [(p, a[a["error_model"] == p]) for p in processes]
        for process, block in scopes:
            for (sigma, onset), cell in block.groupby(["jitter_std", "onset"]):
                per = landscape_means(cell, landscapes)
                for name, r in per.iterrows():
                    dep = float(r["dep_excess_z"])
                    rows.append({
                        "arm": arm, "error_model": process, "jitter_std": sigma, "onset": int(onset),
                        "landscape": name, "opt_z": opt_z[name], "n_runs": int(r["n_runs"]),
                        "dep_excess": dep, "search_excess": float(r["search_excess_z"]),
                        "sel_real": float(r["selection_z"]), "sel_pred": float(r["sel_pred_z"]),
                        "sel_real_sd": float(r["selection_sd"]), "sel_pred_sd": float(r["sel_pred"]),
                        "ratio": float(r["selection_z"] / r["sel_pred_z"]),
                        # A landscape's share only where its own deployed excess is a cost.
                        "share_real": float(r["selection_z"] / dep) if dep >= 0.01 else float("nan"),
                        "share_pred": float(r["sel_pred_z"] / dep) if dep >= 0.01 else float("nan"),
                        "n_near": float(r["n_near"]), "n_near_clean": float(r["n_near_clean"]),
                    })
    return pd.DataFrame(rows)


def onset_table(runs: pd.DataFrame, landscapes: list[str], B: np.ndarray) -> pd.DataFrame:
    """The deployed excess at the final trial by onset, split into its search and selection parts.

    As ratios: dep_0 / dep_late = w (search_0 / search_late) + (1 - w) (sel_0 / sel_late),
    with w = search_late / dep_late, all ratios of landscape means (opt_z units); a
    ratio is suppressed when its late-onset denominator fails the share screen (the
    late-onset search excess is negative on some landscapes at every magnitude). As
    differences, which need no screen: dep_0 - dep_late = (search_0 - search_late) +
    (sel_0 - sel_late), with the search part's share of the difference screened on
    the deployed difference.
    """
    rows = []
    noisy = runs[~runs["baseline"]]
    for arm in ARMS:
        a = noisy[noisy["arm"] == arm]
        processes = sorted(a["error_model"].unique())
        scopes = ([("pooled", a)] if len(processes) > 1 else []) + [(p, a[a["error_model"] == p]) for p in processes]
        for process, block in scopes:
            for sigma, cell in block.groupby("jitter_std"):
                onsets = sorted(cell["onset"].unique())
                if len(onsets) != 2 or onsets[0] != 0:
                    raise ValueError(f"{arm} {process} {sigma}: onsets {onsets}")
                early = landscape_means(cell[cell["onset"] == 0], landscapes)
                late = landscape_means(cell[cell["onset"] == onsets[1]], landscapes)
                row = {"arm": arm, "error_model": process, "jitter_std": sigma, "late_onset": int(onsets[1])}
                for name, col in (("deployed", "dep_excess_z"), ("search", "search_excess_z"),
                                  ("selection", "selection_z"), ("selection_pred", "sel_pred_z")):
                    num, den = early[col].to_numpy(), late[col].to_numpy()
                    row[f"{name}_early"], row[f"{name}_late"] = float(num.mean()), float(den.mean())
                    point, lo, hi = boot_ratio(num, den, B)
                    guard = share_guard(den, den)     # the late-onset value is the denominator
                    row.update({f"{name}_ratio": point, f"{name}_ratio_lo": lo, f"{name}_ratio_hi": hi,
                                f"{name}_ratio_suppressed": guard})
                    point, lo, hi = boot_mean(num - den, B)
                    row.update({f"{name}_diff": point, f"{name}_diff_lo": lo, f"{name}_diff_hi": hi})
                d_dep = early["dep_excess_z"].to_numpy() - late["dep_excess_z"].to_numpy()
                d_search = early["search_excess_z"].to_numpy() - late["search_excess_z"].to_numpy()
                point, lo, hi = boot_ratio(d_search, d_dep, B)
                row.update({"search_part_of_diff": point, "search_part_of_diff_lo": lo,
                            "search_part_of_diff_hi": hi, "search_part_of_diff_suppressed": share_guard(d_dep, d_search)})
                w = row["search_late"] / row["deployed_late"]
                row["search_weight_late"] = w
                row["identity_gap"] = row["deployed_ratio"] - (w * row["search_ratio"]
                                                               + (1 - w) * row["selection_ratio"])
                rows.append(row)
    return pd.DataFrame(rows)


def budget_table(runs: pd.DataFrame, landscapes: list[str], B: np.ndarray) -> pd.DataFrame:
    """T = 25, 50, 100 under GAUSSIAN error, six acquisitions, seeds 7-11, onsets 0 and 0.4 T."""
    noisy = runs[~runs["baseline"]]
    main = noisy[(noisy["arm"] == "main") & noisy["acquisition"].isin(BUDGET_ACQS)
                 & noisy["seed"].isin(BUDGET_SEEDS) & (noisy["error_model"] == "gaussian")]
    rows = []
    for label, block in (("T25", noisy[noisy["arm"] == "T25"]), ("T50", main), ("T100", noisy[noisy["arm"] == "T100"])):
        if set(block["acquisition"]) != set(BUDGET_ACQS) or set(block["seed"]) != set(BUDGET_SEEDS):
            raise ValueError(f"{label}: acquisitions {sorted(set(block['acquisition']))}, seeds {sorted(set(block['seed']))}")
        for (sigma, onset), cell in block.groupby(["jitter_std", "onset"]):
            T = int(cell["T"].iloc[0])
            r = cell_row(cell, landscapes, B)
            rows.append({"T": T, "jitter_std": sigma, "onset": int(onset),
                         "onset_class": "first rating" if onset == 0 else "0.4 T",
                         **{k: r[k] for k in ("n_runs", "dep_excess", "search_excess", "sel_real", "sel_pred",
                                              "ratio", "ratio_lo", "ratio_hi", "share_real", "share_real_lo",
                                              "share_real_hi", "share_pred", "share_suppressed", "set_effect",
                                              "n_near", "n_near_clean", "sel_real_per_sigma")}})
    out = pd.DataFrame(rows)
    out["_late"] = out["onset"] > 0
    return out.sort_values(["_late", "jitter_std", "T"]).drop(columns="_late").reset_index(drop=True)


def budget_contrast(runs: pd.DataFrame, landscapes: list[str], B: np.ndarray) -> pd.DataFrame:
    """How each term moves from T = 25 to T = 100: ratios of landscape means, landscapes paired.

    The two budget arms share landscapes, acquisitions and seeds, so the
    landscape bootstrap resamples the pair. A ratio is screened on its T = 25
    denominator like every share of a cost.
    """
    noisy = runs[~runs["baseline"]]
    main = noisy[(noisy["arm"] == "main") & noisy["acquisition"].isin(BUDGET_ACQS)
                 & noisy["seed"].isin(BUDGET_SEEDS) & (noisy["error_model"] == "gaussian")]
    arms = {25: noisy[noisy["arm"] == "T25"], 50: main, 100: noisy[noisy["arm"] == "T100"]}
    rows = []
    for late in (False, True):
        for sigma in sorted(main["jitter_std"].unique()):
            per = {T: landscape_means(b[(b["jitter_std"] == sigma) & ((b["onset"] > 0) == late)], landscapes)
                   for T, b in arms.items()}
            row = {"jitter_std": sigma, "onset_class": "0.4 T" if late else "first rating"}
            for name, col in (("deployed", "dep_excess_z"), ("search", "search_excess_z"),
                              ("selection", "selection_z"), ("selection_pred", "sel_pred_z")):
                a, b = per[25][col].to_numpy(), per[100][col].to_numpy()
                point, lo, hi = boot_ratio(b, a, B)
                row.update({f"{name}_T25": float(a.mean()), f"{name}_T50": float(per[50][col].mean()),
                            f"{name}_T100": float(b.mean()), f"{name}_ratio_100_25": point,
                            f"{name}_ratio_100_25_lo": lo, f"{name}_ratio_100_25_hi": hi,
                            f"{name}_ratio_100_25_suppressed": share_guard(a, a)})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# report: checks against the existing decomposition
# ---------------------------------------------------------------------------


def check_against_ship_rules(runs: pd.DataFrame) -> dict:
    """Per run, the main sweep's deployed and search regret against ship_rules_per_run.csv."""
    path = REPO / "output-boba" / "analysis" / "ship_rules_per_run.csv"
    s = pd.read_csv(path, usecols=["file", "regret_best_observed", "regret_best_visited"],
                    float_precision="round_trip")
    main = runs[runs["arm"] == "main"]
    m = main.merge(s, on="file", how="outer", indicator=True, validate="one_to_one")
    unmatched = int((m["_merge"] != "both").sum())
    both = m[m["_merge"] == "both"]
    return {"n_runs": int(len(both)), "n_unmatched": unmatched,
            "max_abs_deployed": float((both["deployed"] - both["regret_best_observed"]).abs().max()),
            "max_abs_search": float((both["search"] - both["regret_best_visited"]).abs().max())}


def decomposition_rows(runs: pd.DataFrame) -> pd.DataFrame:
    """regret_decomposition.csv's cell means and point shares, recomputed from the logs."""
    main = runs[runs["arm"] == "main"]
    noisy, clean = main[~main["baseline"]], main[main["baseline"]]
    base = clean.groupby(PAIR_KEYS)[["deployed_z", "search_z"]].mean().rename(
        columns={"deployed_z": "deployed_clean_z", "search_z": "search_clean_z"})

    def summary(group: pd.DataFrame) -> dict:
        per = group.groupby("dataset")[["deployed_z", "search_z", "selection_z"]].mean()
        m = group.merge(base, on=PAIR_KEYS, how="inner")
        m = m.assign(excess_deployed=m["deployed_z"] - m["deployed_clean_z"],
                     excess_search=m["search_z"] - m["search_clean_z"], excess_selection=m["selection_z"])
        perx = m.groupby("dataset")[["excess_deployed", "excess_search", "excess_selection"]].mean()
        return {"n_runs": len(group), "mean_deployed": per["deployed_z"].mean(), "mean_search": per["search_z"].mean(),
                "mean_selection": per["selection_z"].mean(),
                "selection_share": per["selection_z"].mean() / per["deployed_z"].mean(),
                "mean_excess_deployed": perx["excess_deployed"].mean(),
                "mean_excess_search": perx["excess_search"].mean(),
                "mean_excess_selection": perx["excess_selection"].mean(),
                "excess_selection_share": perx["excess_selection"].mean() / perx["excess_deployed"].mean()}

    rows = []
    for (model, std, onset), g in noisy.groupby(["error_model", "jitter_std", "onset"]):
        rows.append({"error_model": model, "jitter_std": str(float(std)), "jitter_iteration": str(float(onset)),
                     **summary(g)})
    for model, g in noisy.groupby("error_model"):
        rows.append({"error_model": model, "jitter_std": "pooled", "jitter_iteration": "pooled", **summary(g)})
    for (std, onset), g in noisy.groupby(["jitter_std", "onset"]):
        rows.append({"error_model": "pooled", "jitter_std": str(float(std)), "jitter_iteration": str(float(onset)),
                     **summary(g)})
    rows.append({"error_model": "pooled", "jitter_std": "pooled", "jitter_iteration": "pooled", **summary(noisy)})
    return pd.DataFrame(rows)


DECOMP_COLUMNS = ["n_runs", "mean_deployed", "mean_search", "mean_selection", "selection_share",
                  "mean_excess_deployed", "mean_excess_search", "mean_excess_selection", "excess_selection_share"]


def check_against_headline(lands: pd.DataFrame) -> dict:
    """The per-landscape deployed excess and selection share of the headline figure's table."""
    path = OUT_DIR / "per_landscape_headline.csv"
    if not path.is_file():
        return {"n_rows": 0, "max_abs_deployed": float("nan"), "max_abs_share": float("nan")}
    ref = pd.read_csv(path)
    mine = lands[(lands["arm"] == "main")].rename(columns={"onset": "jitter_iteration"})
    m = ref.merge(mine, on=["landscape", "error_model", "jitter_std", "jitter_iteration"], how="inner",
                  suffixes=("_ref", ""))
    share = m["selection_share"].where(m["deployed"] >= 0.01)
    return {"n_rows": int(len(m)), "max_abs_deployed": float((m["deployed"] - m["dep_excess"]).abs().max()),
            "max_abs_share": float((share - m["share_real"]).abs().max())}


def check_against_decomposition(runs: pd.DataFrame) -> pd.DataFrame:
    """Every row of regret_decomposition.csv: largest absolute deviation per column."""
    path = REPO / "output-boba" / "analysis" / "regret_decomposition.csv"
    ref = pd.read_csv(path, dtype={"jitter_std": str, "jitter_iteration": str})
    ref["jitter_std"] = ref["jitter_std"].map(lambda v: v if v == "pooled" else str(float(v)))
    ref["jitter_iteration"] = ref["jitter_iteration"].map(lambda v: v if v == "pooled" else str(float(v)))
    mine = decomposition_rows(runs)
    keys = ["error_model", "jitter_std", "jitter_iteration"]
    m = ref.merge(mine, on=keys, how="outer", suffixes=("_ref", "_logs"), indicator=True)
    if (m["_merge"] != "both").any():
        raise SystemExit(f"regret_decomposition.csv and the logs disagree on the cells: "
                         f"{m.loc[m['_merge'] != 'both', keys].to_dict('records')}")
    out = []
    for col in DECOMP_COLUMNS:
        dev = (m[f"{col}_ref"].astype(float) - m[f"{col}_logs"].astype(float)).abs()
        out.append({"column": col, "n_rows": len(m), "max_abs_deviation": float(dev.max())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# report: text
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(row, name: str, pct: bool = False, digits: int = 2) -> str:
    v, lo, hi = row[name], row[f"{name}_lo"], row[f"{name}_hi"]
    if pct:
        return f"{100 * v:.1f}% [{100 * lo:.0f}, {100 * hi:.0f}]"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _cell_label(sigma: float, onset: int) -> str:
    return f"{sigma:g} sigma from {'the first rating' if onset == 0 else f'trial {onset + 1}'}"


def latex_table(cells: pd.DataFrame) -> str:
    """The main sweep's cells (four processes pooled) as a tabular body for the appendix."""
    c = cells[(cells["arm"] == "main") & (cells["error_model"] == "pooled")].sort_values(["onset", "jitter_std"])
    proc = cells[(cells["arm"] == "main") & (cells["error_model"] != "pooled")]
    lines = [r"\begin{tabular}{rr rrr rr r rrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{selection loss, fraction of $\optz$} & \multicolumn{2}{c}{share of deployed excess}"
             r" & set & \multicolumn{4}{c}{realised / predicted by process} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){9-12}",
             r"$\sigma_e$ & from trial & realised & predicted & ratio & realised & predicted & effect"
             r" & \textsc{gauss.} & \textsc{bias} & \textsc{drift} & \textsc{ar}(1) \\", r"\midrule"]
    for _, r in c.iterrows():
        per_proc = []
        for p in PROCESSES:
            q = proc[(proc["error_model"] == p) & (proc["jitter_std"] == r["jitter_std"]) & (proc["onset"] == r["onset"])]
            per_proc.append(f"${float(q['ratio'].iloc[0]):.2f}$")
        share = (f"${100 * r['share_real']:.0f}\\%$ & ${100 * r['share_pred']:.0f}\\%$" if not r["share_suppressed"]
                 else r"-- & --")
        lines.append(
            f"${r['jitter_std']:g}$ & {int(r['onset']) + 1 if r['onset'] else 1} & ${r['sel_real']:.3f}$ & "
            f"${r['sel_pred']:.3f}$ & ${r['ratio']:.2f}$ $[{r['ratio_lo']:.2f}, {r['ratio_hi']:.2f}]$ & {share} & "
            f"${r['set_effect']:.2f}$ & " + " & ".join(per_proc) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_landscape_table(lands: pd.DataFrame, sigma: float = HEADLINE[0], onset: int = HEADLINE[1]) -> str:
    """The headline cell per landscape (what make_paper_figures.fig_winners_curse draws), sorted by share."""
    from make_paper_figures import PRETTY

    h = lands[(lands["arm"] == "main") & (lands["error_model"] == "pooled") & (lands["jitter_std"] == sigma)
              & (lands["onset"] == onset)].sort_values("share_real")
    lines = [r"\begin{tabular}{lr rr rr r rr}", r"\toprule",
             r" & & \multicolumn{2}{c}{selection share} & \multicolumn{2}{c}{selection loss, landscape SDs} & "
             r"& \multicolumn{2}{c}{trials within $0.1$ SD of best} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}",
             r"landscape & $\optz$ & realised & predicted & realised & predicted & ratio & noisy & clean twin \\",
             r"\midrule"]
    for _, r in h.iterrows():
        lines.append(f"{PRETTY.get(r['landscape'], r['landscape'])} & ${r['opt_z']:.2f}$ & "
                     f"${100 * r['share_real']:.1f}\\%$ & ${100 * r['share_pred']:.1f}\\%$ & ${r['sel_real_sd']:.3f}$ & "
                     f"${r['sel_pred_sd']:.3f}$ & ${r['ratio']:.2f}$ & ${r['n_near']:.1f}$ & ${r['n_near_clean']:.1f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_report(runs: pd.DataFrame, cells: pd.DataFrame, lands: pd.DataFrame, onset: pd.DataFrame,
                 budget: pd.DataFrame, contrast: pd.DataFrame, checks: dict) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        noisy = runs[~runs["baseline"]]
        draws = sorted(noisy["draws"].astype(int).unique())
        print("Selection loss as a winner's curse on the visited set (scripts/checks/winners_curse.py).")
        print("Selection loss = deployed minus best-visited regret at the final trial; predicted = its mean over "
              f"{'/'.join(map(str, draws))} fresh error paths of the run's own process on the run's own visited designs, standard ship "
              "rule (first index of the best rating). Ratios of landscape means over twenty landscapes, units of "
              "opt_z unless a line says landscape SDs; 95% percentile intervals from 2000 landscape resamples, "
              "which treat the landscapes (which share seeds) as independent clusters.\n")
        print("Inputs and reproduction checks")
        for arm, (arm_dir, label) in ARMS.items():
            a = runs[runs["arm"] == arm]
            print(f"  {arm:<9} {arm_dir:<26} {int((~a['baseline']).sum()):>6} noisy runs, "
                  f"{int(a['baseline'].sum()):>5} clean twins ({label})")
        exact = (noisy["replay_max_abs_rating"] == 0) & (noisy["replay_max_abs_error"] == 0)
        print(f"  the error stream rebuilt from each run's noise seed reproduces its logged ratings and errors exactly "
              f"in {int(exact.sum())} of {len(noisy)} noisy runs (largest deviation "
              f"{noisy[['replay_max_abs_rating', 'replay_max_abs_error']].max().max():.1e}); pre-onset ratings are "
              f"exact in {int(noisy['logged_pre_onset_exact'].sum())}")
        print(f"  realised selection loss reproduces each log's deployed minus search regret to {LOG_TOL:g} "
              f"(checked in scan); largest |max g - g[d] - (deployed - search)| "
              f"{float((noisy['selection_direct'] - (noisy['deployed'] - noisy['search'])).abs().max()):.1e}")
        sr = checks["ship_rules"]
        print(f"  main sweep against ship_rules_per_run.csv: {sr['n_runs']} runs matched, {sr['n_unmatched']} "
              f"unmatched, largest deviation {max(sr['max_abs_deployed'], sr['max_abs_search']):.1e} "
              "(deployed and search regret)")
        dc = checks["decomposition"]
        print(f"  main sweep against regret_decomposition.csv ({int(dc['n_rows'].iloc[0])} rows): largest deviation "
              f"{dc['max_abs_deviation'].max():.1e} over {', '.join(dc['column'])}")
        hl = checks["headline"]
        if hl["n_rows"]:
            print(f"  main sweep against per_landscape_headline.csv ({hl['n_rows']} landscape-cell rows): largest "
                  f"deviation {max(hl['max_abs_deployed'], hl['max_abs_share']):.1e} (deployed excess, selection share)")
        else:
            print("  per_landscape_headline.csv not found; that check was not run")
        rel = cells["sel_pred_mc_se"] / cells["sel_pred"]
        print("  Monte Carlo standard error of a cell's predicted selection loss, at most: "
              + ", ".join(f"{arm} {100 * float(rel[cells['arm'] == arm].max()):.2f}%" for arm in ARMS)
              + " of the prediction\n")

        print("Main sweep, four processes pooled, ten acquisitions: selection loss at the final trial")
        main = cells[(cells["arm"] == "main") & (cells["error_model"] == "pooled")].sort_values(["onset", "jitter_std"])
        print(f"  {'cell':<30} {'realised':>8} {'predicted':>9}  {'realised/predicted':<20} "
              f"{'share realised':<17} {'share predicted':<17} {'set effect':<18}")
        for _, r in main.iterrows():
            share_r = _ci(r, "share_real", pct=True) if not r["share_suppressed"] else "-- (" + r["share_suppressed"] + ")"
            share_p = _ci(r, "share_pred", pct=True) if not r["share_suppressed"] else "--"
            print(f"  {_cell_label(r['jitter_std'], r['onset']):<30} {r['sel_real']:8.4f} {r['sel_pred']:9.4f}  "
                  f"{_ci(r, 'ratio'):<20} {share_r:<17} {share_p:<17} {_ci(r, 'set_effect'):<18}")
        print("  (set effect = predicted selection loss on the run's visited designs over that on its clean twin's, "
              "same error paths)")
        print("\nRealised / predicted by process (main sweep, ten acquisitions)")
        proc = cells[(cells["arm"] == "main") & (cells["error_model"] != "pooled")]
        for p in PROCESSES:
            q = proc[proc["error_model"] == p].sort_values(["onset", "jitter_std"])
            print(f"  {p:<9} " + "  ".join(f"{r['jitter_std']:g}/{r['onset']}: {r['ratio']:.2f} "
                                           f"[{r['ratio_lo']:.2f}, {r['ratio_hi']:.2f}]" for _, r in q.iterrows()))
        print(f"  range over the 8 pooled cells {main['ratio'].min():.2f} to {main['ratio'].max():.2f}; over the 32 "
              f"process cells {proc['ratio'].min():.2f} to {proc['ratio'].max():.2f}")

        print("\nThe visited set: designs near the best (main sweep, pooled; trials within 0.1 landscape SD of the best "
              "visited, the best included; distinct designs in brackets) and within sigma_e of it")
        for _, r in main.iterrows():
            print(f"  {_cell_label(r['jitter_std'], r['onset']):<30} noisy {r['n_near']:5.1f} ({r['n_near_unique']:4.1f})"
                  f"  clean twin {r['n_near_clean']:5.1f} ({r['n_near_unique_clean']:4.1f});  within sigma_e: noisy "
                  f"{r['n_within_sigma']:5.1f}, clean {r['n_within_sigma_clean']:5.1f} of {int(r['T'])} trials")

        print("\nRule of thumb, in units of the rater's noise sigma_e (landscape SDs, mean over landscapes)")
        for scope in ("pooled", "gaussian"):
            q = cells[(cells["arm"] == "main") & (cells["error_model"] == scope)].sort_values(["onset", "jitter_std"])
            print(f"  {'four processes pooled' if scope == 'pooled' else 'GAUSSIAN'}:")
            for _, r in q.iterrows():
                print(f"    {_cell_label(r['jitter_std'], r['onset']):<30} shipped design below the best visited by "
                      f"{_ci(r, 'sel_real_per_sigma')} sigma_e (predicted {r['sel_pred_per_sigma']:.2f}); its rating "
                      f"overstates its true value by {_ci(r, 'over_real_per_sigma')} sigma_e (predicted "
                      f"{r['over_pred_per_sigma']:.2f}); wrong design shipped in {_pct(r['wrong_real'])} "
                      f"(predicted {_pct(r['p_wrong_pred'])})")

        print("\nPer landscape (main sweep, four processes pooled): realised against predicted")
        for _, r in main.iterrows():
            print(f"  {_cell_label(r['jitter_std'], r['onset']):<30} Spearman over landscapes: landscape SDs "
                  f"{r['spearman_landscape_sd']:.3f}, opt_z units {r['spearman_landscape_optz']:.3f}, shares "
                  f"{r['spearman_landscape_share']:.3f}; realised/predicted per landscape {r['ratio_landscape_min']:.2f} "
                  f"({r['landscape_ratio_min_at']}) to {r['ratio_landscape_max']:.2f} ({r['landscape_ratio_max_at']}); "
                  f"per run {r['spearman_run']:.2f}")
        s, o = HEADLINE
        h = lands[(lands["arm"] == "main") & (lands["error_model"] == "pooled") & (lands["jitter_std"] == s)
                  & (lands["onset"] == o)].sort_values("share_real")
        print(f"  {_cell_label(s, o)}: selection share realised {_pct(h['share_real'].min())} "
              f"({h['landscape'].iloc[0]}) to {_pct(h['share_real'].max())} ({h['landscape'].iloc[-1]}), predicted "
              f"{_pct(h['share_pred'].min())} to {_pct(h['share_pred'].max())}; selection loss in landscape SDs "
              f"realised {h['sel_real_sd'].min():.3f} to {h['sel_real_sd'].max():.3f}, predicted "
              f"{h['sel_pred_sd'].min():.3f} to {h['sel_pred_sd'].max():.3f}")

        print("\nOnset decomposition of the deployed excess at the final trial (main sweep, four processes pooled): "
              "from the first rating against from trial 21, landscape means")
        q = onset[(onset["arm"] == "main") & (onset["error_model"] == "pooled")].sort_values("jitter_std")
        for _, r in q.iterrows():
            parts = [f"{label} {r[f'{name}_early']:.3f} vs {r[f'{name}_late']:.3f}, difference "
                     f"{_ci(r, f'{name}_diff', digits=3)}"
                     for name, label in (("deployed", "deployed"), ("search", "search"), ("selection", "selection"),
                                         ("selection_pred", "selection predicted"))]
            split = (f"search part of the difference {_ci(r, 'search_part_of_diff', pct=True)}"
                     if not r["search_part_of_diff_suppressed"] else
                     f"search part of the difference -- ({r['search_part_of_diff_suppressed']})")
            print(f"  {r['jitter_std']:g} sigma: " + "; ".join(parts) + f"; {split}")
            ratios = []
            for name, label in (("deployed", "deployed"), ("search", "search"), ("selection", "selection"),
                                ("selection_pred", "selection predicted")):
                if r[f"{name}_ratio_suppressed"]:
                    ratios.append(f"{label} -- ({r[f'{name}_ratio_suppressed']})")
                else:
                    ratios.append(f"{label} {_ci(r, f'{name}_ratio')}")
            print("      onset ratios: " + "; ".join(ratios)
                  + f"; late-onset search share {r['search_weight_late']:.3f} (identity gap {r['identity_gap']:.1e})")

        print("\nReplication: untouched seeds 27-36 and the budget arms (GAUSSIAN)")
        for arm in ("untouched", "T25", "T100"):
            q = cells[cells["arm"] == arm].sort_values(["onset", "jitter_std"])
            print(f"  {arm:<9} realised/predicted " + "  ".join(
                f"{r['jitter_std']:g}/{r['onset']}: {r['ratio']:.2f} [{r['ratio_lo']:.2f}, {r['ratio_hi']:.2f}]"
                for _, r in q.iterrows()))
        g = cells[(cells["arm"] == "main") & (cells["error_model"] == "gaussian")].sort_values(["onset", "jitter_std"])
        print("  main GAUSSIAN realised/predicted " + "  ".join(
            f"{r['jitter_std']:g}/{r['onset']}: {r['ratio']:.2f}" for _, r in g.iterrows()))
        for arm in ("untouched",):
            q = cells[cells["arm"] == arm].sort_values(["onset", "jitter_std"])
            for _, r in q.iterrows():
                share = _ci(r, "share_real", pct=True) if not r["share_suppressed"] else "--"
                print(f"  untouched {_cell_label(r['jitter_std'], r['onset']):<30} realised {r['sel_real']:.4f}, "
                      f"predicted {r['sel_pred']:.4f}, share realised {share}, predicted "
                      f"{'--' if r['share_suppressed'] else _pct(r['share_pred'])}")

        print("\nBudget (GAUSSIAN, six acquisitions, seeds 7-11; T = 50 is the main sweep on the same scope): final trial")
        print(f"  {'cell':<26} {'T':>4} {'deployed':>9} {'search':>8} {'selection':>9} {'predicted':>9} "
              f"{'share realised':<17} {'share pred.':>11}  realised/predicted")
        for _, r in budget.iterrows():
            share = _ci(r, "share_real", pct=True) if not r["share_suppressed"] else "--"
            pred = _pct(r["share_pred"]) if not r["share_suppressed"] else "--"
            print(f"  {r['jitter_std']:g} sigma, {r['onset_class']:<13} {int(r['T']):4d} {r['dep_excess']:9.4f} "
                  f"{r['search_excess']:8.4f} {r['sel_real']:9.4f} {r['sel_pred']:9.4f} {share:<17} {pred:>11}  "
                  f"{_ci(r, 'ratio')}")
        print("\nFrom T = 25 to T = 100 (same scope; ratio of landscape means T = 100 / T = 25, landscapes paired)")
        for _, r in contrast.iterrows():
            parts = []
            for name in ("deployed", "search", "selection", "selection_pred"):
                if r[f"{name}_ratio_100_25_suppressed"]:
                    parts.append(f"{name} -- ({r[f'{name}_ratio_100_25_suppressed']})")
                else:
                    parts.append(f"{name} {_ci(r, f'{name}_ratio_100_25')}")
            print(f"  {r['jitter_std']:g} sigma, {r['onset_class']:<13} " + "; ".join(parts))
    return buf.getvalue()


def cmd_report(args: argparse.Namespace) -> None:
    if not RUNS_PATH.is_file():
        raise SystemExit(f"{RUNS_PATH.relative_to(REPO)} not found: run `winners_curse.py scan` first")
    opt_z = opt_z_map()
    runs = with_units(pd.read_parquet(RUNS_PATH), opt_z)
    landscapes = sorted(runs["dataset"].unique())
    if len(landscapes) != 20:
        raise SystemExit(f"expected twenty landscapes, found {len(landscapes)}")
    noisy = runs[~runs["baseline"]]
    if not ((noisy["replay_max_abs_rating"] == 0) & (noisy["replay_max_abs_error"] == 0)).all():
        raise SystemExit("some run's error stream is not reproduced exactly from its seed; see the per-run table")
    B = boot_matrix(len(landscapes))
    checks = {"ship_rules": check_against_ship_rules(runs), "decomposition": check_against_decomposition(runs)}
    sr, dc = checks["ship_rules"], checks["decomposition"]
    if sr["n_unmatched"] or max(sr["max_abs_deployed"], sr["max_abs_search"]) > 0:
        raise SystemExit(f"the logs and ship_rules_per_run.csv disagree: {sr}")
    if dc["max_abs_deviation"].max() > DECOMP_TOL:
        raise SystemExit(f"the logs do not reproduce regret_decomposition.csv:\n{dc}")
    cells = cells_table(runs, landscapes, B)
    lands = landscape_table(runs, landscapes, opt_z)
    checks["headline"] = hl = check_against_headline(lands)
    if hl["n_rows"] and max(hl["max_abs_deployed"], hl["max_abs_share"]) > DECOMP_TOL:
        raise SystemExit(f"the logs do not reproduce per_landscape_headline.csv: {hl}")
    onset = onset_table(runs, landscapes, B)
    budget = budget_table(runs, landscapes, B)
    contrast = budget_contrast(runs, landscapes, B)
    text = write_report(runs, cells, lands, onset, budget, contrast, checks)
    cells.to_csv(OUT_DIR / "winners_curse_cells.csv", index=False)
    lands.to_csv(OUT_DIR / "winners_curse_landscapes.csv", index=False)
    onset.to_csv(OUT_DIR / "winners_curse_onset.csv", index=False)
    budget.to_csv(OUT_DIR / "winners_curse_budget.csv", index=False)
    contrast.to_csv(OUT_DIR / "winners_curse_budget_contrast.csv", index=False)
    check_rows = [{"check": "ship_rules_per_run", "statistic": k, "value": v} for k, v in sr.items()]
    check_rows += [{"check": "regret_decomposition", "statistic": f"max_abs_{r['column']}",
                    "value": r["max_abs_deviation"]} for _, r in dc.iterrows()]
    check_rows += [{"check": "per_landscape_headline", "statistic": k, "value": v} for k, v in hl.items()]
    exact = (noisy["replay_max_abs_rating"] == 0) & (noisy["replay_max_abs_error"] == 0)
    check_rows += [{"check": "error_replay", "statistic": "runs_exact", "value": int(exact.sum())},
                   {"check": "error_replay", "statistic": "runs", "value": int(len(noisy))}]
    pd.DataFrame(check_rows).to_csv(OUT_DIR / "winners_curse_checks.csv", index=False)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "winners_curse.tex").write_text(latex_table(cells), encoding="utf-8")
    (TABLES_DIR / "winners_curse_landscapes.tex").write_text(latex_landscape_table(lands), encoding="utf-8")
    REGISTER_TXT.parent.mkdir(parents=True, exist_ok=True)
    REGISTER_TXT.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="read the logs, replay the error streams, run the Monte Carlo")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--draws", type=int, default=DRAWS)
    sub.add_parser("report", help="tables, checks and the register file")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "scan":
        cmd_scan(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
