"""Re-rating with memory: the final sitting and the confirmation trials when the rater remembers (post hoc).

The paper's end-of-study procedures re-rate designs the rater has already rated, and every replay
(replay_end_of_study.py, --sitting-process shared) draws each new look's error independently of the
earlier rating of the same design. The designs re-rated were chosen partly for a positive original error,
so a rater who remembers the earlier rating carries that error into the re-rating. retest_memory.py
tested memory for the three test-retest pairs only; this check tests (i) the final sitting and (ii) the
four confirmation trials.

POST HOC, and outside every Holm family of the paper; the memory strengths are assumed, not measured.

Model
-----
A look at a design whose original rating had error e_orig gets the error

    e = rho_m * e_orig + sqrt(1 - rho_m^2) * e_fresh,        rho_m in {0, 0.25, 0.5, 0.75},

where e_fresh is exactly the look error the published replay draws for that look (the same draws, so
rho_m = 0 is the published procedure; the control checks it bit for bit and stops on any difference) and
e_orig is the logged error of the design's original rating, objective_observed - objective_true at its
trial in the run log (checked against the loop's logged error_magnitude to 1e-9 in every run). A design
rated more than once before the procedure starts (trials 1..T-k for a sitting of k, 1..T-2k for
confirmation) takes the rating the run's ship rule reads: its highest rating among those trials, ties to
the earliest (the standard process ships the design with the best single rating). e_orig is the whole
logged error at that trial: under bias, drift and AR(1) it carries the bias, the ramp or the AR state of
the original trial, which the shared sitting model otherwise cancels between looks, so a design first
rated before the onset (exact, e_orig = 0) and one rated after it no longer differ only by their fresh
draws. Under Gaussian error the look keeps its SD and correlates with the original error at rho_m. Memory
acts on the noisy runs only: a clean twin's ratings are exact (e_orig = 0, checked in every clean twin's
log by the sitting replay) and its looks carry no fresh error, so its looks, its shipped design and every
price are unchanged by construction (checked run by run in the sitting; the confirmation's clean-twin
claims are the published ones). As a sensitivity, every rate is also scored with the error of the
design's first rating in place of the ship rule's rating.

(i) The final sitting. Cells: the sitting of two at 1 sigma from the first rating (jitter_std 1.0,
jitter_iteration 0) and the sitting of sixteen at 5 sigma from trial 21 (jitter_std 5.0,
jitter_iteration 20); main sweep (output-boba), LogEI and qNEI, gaussian, bias, drift and ar1, seeds 7-16.
Candidates and look errors come from sitting_order_effects.py (its cached replay inputs, which reproduce
the published sittings bit for bit); the replay stage refits each noisy run's prefix GP with
replay_end_of_study's own functions to recover each candidate's row, checks the candidates, true values
and look errors against those inputs and against replay_end_of_study.tournament in process, and reads
e_orig from the log. The best look ships (ties to the earliest candidate, as published). Estimand
(analyse_boba_adaptations): per run in units of opt_z, against the standard process, cost = ref_noisy -
ref_clean, gain = ref_noisy - trt_noisy, price = trt_clean - ref_clean; increments over the memoryless
(published) sitting and over LCB1 and LCB2 after all 50 trials with no extra trial (regret_lcb1 and
regret_lcb2 of ship_rules_per_run.csv, joined by sitting_by_magnitude.attach_ship_rules, which checks the
standard process to 1e-9). Landscape means, then the mean over the 20 landscapes; landscape bootstrap
(2,000 resamples; the landscapes share their random numbers, so it treats correlated clusters as
independent); two-sided Wilcoxon over the 20 landscape means; Holm over the six (cell, rho_m > 0) rows
per seed set and estimand (rho_m = 0 is the published sitting, tested in the paper). Seeds 12-16 (held
out) and 7-16, four processes and both acquisitions pooled.

(ii) The confirmation trials (confirm_k2 of replay_end_of_study.py): search trials 1..46; D = the
LCB1 choice of the prefix GP, C = the best initial design by its rating; trials 47-50 rate D and C twice
each in counterbalanced order; an improvement is claimed when a one-sided paired t-test at alpha 0.05
prefers D. Under memory both designs' fresh ratings carry rho_m of their original errors. Scope as
published: gaussian error, LogEI, qNEI and UCB, seeds 7-16, all four magnitudes and both onsets (4,800
noisy runs). Each D comes from the prefix GP refitted with replay_end_of_study's own functions; every
run is checked against replay_end_of_study.confirmation in process and against the published per-run
table. Claim definitions are the published ones (replay_end_of_study.claim_rates, landscape bootstrap):
false-claim rate = runs that claim while f(D) <= f(C), over all runs, the 24 runs whose D is C allowed to
claim (the published 0.5%; *_excl_dc counts no claim there); detection = claims among the runs with
f(D) > f(C) (the published 51%); 'no test' = claiming whenever D's mean fresh rating exceeds C's (the
published 6.6%, 6.3% without the D = C runs; detection 88%). The standard process re-rates nothing, so
its claims (15.7% false) are unchanged.

    python scripts/checks/rerating_memory.py replay --workers 2      # GP refits from the run logs (git-ignored)
    python scripts/checks/rerating_memory.py score                   # from the cached inputs; writes the register

replay writes review/rerating_memory_sitting_inputs.csv.gz and review/rerating_memory_confirm_inputs.csv.gz
(gzip without a timestamp; --resume keeps the runs already there; the worker count changes no number;
about 14 minutes for the sitting's 3,200 refits and 30 for the confirmation's 4,800 on 2 workers, plus
indexing the arm).
score reads them, sitting_order_effects_inputs.csv.gz, the published per-run tables, end_of_study_claims.csv
and ship_rules_per_run.csv, and writes review/rerating_memory_sitting{,_by_process}.csv,
review/rerating_memory_sitting_per_run.csv.gz, review/rerating_memory_confirm_claims.csv,
review/rerating_memory_confirm_per_run.csv.gz and register_checks/rerating_memory.txt (exactly what it
prints).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import analyse_boba_adaptations as aba  # noqa: E402
import sitting_by_magnitude as sbm  # noqa: E402
import sitting_order_effects as soe  # noqa: E402

ARM_DIR = soe.ARM_DIR
ANALYSIS = soe.ANALYSIS
REVIEW = soe.REVIEW
SITTING_INPUTS = REVIEW / "rerating_memory_sitting_inputs.csv.gz"
CONFIRM_INPUTS = REVIEW / "rerating_memory_confirm_inputs.csv.gz"
SITTING_PER_RUN = REVIEW / "rerating_memory_sitting_per_run.csv.gz"
SITTING_SUMMARY = REVIEW / "rerating_memory_sitting.csv"
SITTING_BY_PROCESS = REVIEW / "rerating_memory_sitting_by_process.csv"
CONFIRM_PER_RUN = REVIEW / "rerating_memory_confirm_per_run.csv.gz"
CONFIRM_CLAIMS = REVIEW / "rerating_memory_confirm_claims.csv"
REGISTER = REVIEW / "register_checks" / "rerating_memory.txt"
# The published confirmation replay (replay_end_of_study.py with its defaults).
CONFIRM_PUBLISHED = ANALYSIS / "end_of_study"

MEMORY = (0.0, 0.25, 0.5, 0.75)
CELLS = soe.CELLS
SEED_SETS = soe.SEED_SETS
CONFIRM_K = 2
CONFIRM_PROCEDURE = f"confirm_k{CONFIRM_K}"
CONFIRM_ACQS = ("logei", "qnei", "ucb")
CONFIRM_PROCESSES = ("gaussian",)
CONFIRM_SEEDS = tuple(range(7, 17))
ALPHA = 0.05
LCB_BETA = 1.0
BOOTSTRAP_SEED = 20261107
HOLM_FAMILY = "the six (cell, rho_m > 0) rows of (i), per seed set and estimand"
ESTIMANDS = ("gain", "inc_published", "inc_lcb1", "inc_lcb2")


class ControlError(RuntimeError):
    """rho_m = 0 does not reproduce the published procedure."""


def mem_label(rho_m: float) -> str:
    return f"mem{float(rho_m):g}"


# ---------------------------------------------------------------------------
# The model (pure numpy; the tests pin each formula)
# ---------------------------------------------------------------------------


def memory_errors(e_orig, e_fresh, rho_m: float) -> np.ndarray:
    """rho_m * e_orig + sqrt(1 - rho_m^2) * e_fresh, elementwise.

    At rho_m = 0 this is e_fresh exactly (0 * e_orig adds a signed zero), so a
    look built from it is the published look bit for bit.
    """
    rho_m = float(rho_m)
    if not 0.0 <= rho_m <= 1.0:
        raise ValueError(f"rho_m must lie in [0, 1], not {rho_m}")
    e_orig = np.asarray(e_orig, dtype=float)
    e_fresh = np.asarray(e_fresh, dtype=float)
    if e_orig.shape != e_fresh.shape:
        raise ValueError(f"e_orig {e_orig.shape} and e_fresh {e_fresh.shape} differ in shape")
    if not np.all(np.isfinite(e_orig)):
        raise ValueError("an original error is not finite")
    return rho_m * e_orig + np.sqrt(1.0 - rho_m ** 2) * e_fresh


def sitting_pick(truth, e_fresh, e_orig, rho_m: float) -> int:
    """The candidate (rank index) with the best look under memory; ties to the earliest, as published."""
    looks = np.asarray(truth, dtype=float) + memory_errors(e_orig, e_fresh, rho_m)
    return int(np.argmax(looks))


def confirmation_ratings(f: float, e_fresh, e_orig: float, rho_m: float) -> np.ndarray:
    """Fresh ratings f + e of one design under memory; e_fresh holds the published draws sd(t) * z."""
    e_fresh = np.asarray(e_fresh, dtype=float)
    return float(f) + memory_errors(np.full(e_fresh.shape, float(e_orig)), e_fresh, rho_m)


def rating_row(keys: list, observed, row: int, n: int) -> tuple[int, int, int]:
    """(rating row, first row, number of ratings) of the design recorded at ``row``, among rows 0..n-1.

    The rating row is the design's highest rating there, ties to the earliest:
    the rating the standard ship rule (best single rating) reads for it.
    """
    if not 0 <= int(row) < int(n):
        raise ValueError(f"row {row} is not among the first {n} trials")
    key = keys[int(row)]
    rows = np.array([r for r in range(int(n)) if keys[r] == key], dtype=int)
    best = rows[int(np.argmax(np.asarray(observed, dtype=float)[rows]))]
    return int(best), int(rows[0]), int(len(rows))


def original_errors(run, rows, n: int) -> dict:
    """e_orig (at the rating row) and the error of the first rating, per design given by one of its rows."""
    import bo_sensor_error_simulation as sim

    keys = [sim._design_key(x) for x in run.X[:n]]
    info = [rating_row(keys, run.observed, int(r), n) for r in rows]
    rating = np.array([i[0] for i in info], dtype=int)
    first = np.array([i[1] for i in info], dtype=int)
    count = np.array([i[2] for i in info], dtype=int)
    # objective_true: deployed is the true value at the recorded design, which is
    # objective_true in a response-error arm (no slip); check_logged_error ties it to error_magnitude.
    return {"rows_rating": rating, "rows_first": first, "n_ratings": count,
            "e_orig": run.observed[rating] - run.deployed[rating],
            "e_first": run.observed[first] - run.deployed[first]}


# ---------------------------------------------------------------------------
# replay: the candidates' rows and original errors, from the run logs
# ---------------------------------------------------------------------------


def _vec(values) -> str:
    return soe._vec(values)


def _ivec(values) -> str:
    return soe._ivec(values)


def parse_vec(text) -> np.ndarray:
    return np.array([], dtype=float) if pd.isna(text) or str(text) == "" else soe.parse_vec(text)


def parse_ivec(text) -> np.ndarray:
    return np.array([], dtype=int) if pd.isna(text) or str(text) == "" else soe.parse_ivec(text)


def sitting_replay_stem(task: dict) -> list[dict]:
    """One stem's sitting runs: refit, check against the cached inputs and tournament(), read e_orig."""
    import torch

    import boba_benchmarks as bb
    import replay_end_of_study as eos

    arm, settings = task["arm"], task["settings"]
    T = arm.iterations
    spec = bb.BENCHMARKS[task["dataset"]]
    bounds = torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)
    cached = task["cached"]
    out = []
    clean = eos.read_run(Path(task["clean_path"]), T)
    eos.check_logged_error(clean, T)
    for k in task["clean_ks"]:
        n = T - int(k)
        gap = float(np.max(np.abs(clean.observed[:n] - clean.deployed[:n])))
        if gap != 0.0:
            raise eos.ReproductionError(f"{clean.name}: the clean twin's rating error is {gap!r}, not zero")
        m = int(cached[(clean.name, int(k))]["n_candidates"])
        out.append({"file": clean.name, "k": int(k), "baseline": True, "rows_first": "", "rows_rating": "",
                    "n_ratings": "", "e_orig": _vec(np.zeros(m)), "e_first": _vec(np.zeros(m)),
                    "max_abs_prefix_error": gap})
    for r in task["runs"]:
        run = eos.read_run(Path(r["path"]), T)
        k = int(CELLS[soe.cell_of(r["jitter_std"], r["jitter_iteration"])][2])
        n = T - k
        eos.check_logged_inference(run, [n, T])
        eos.check_logged_error(run, T)
        sd_fn = eos.sitting_sd_fn(arm, r["error_model"], float(r["jitter_std"]), int(r["jitter_iteration"]),
                                  settings, variant=str(r.get("variant") or "") or None)
        state = eos.search_state(run, n, bounds, settings.lcb_beta)
        rec = soe.sitting_inputs(run, state, k, sd_fn, T)  # raises unless tournament() agrees in process
        c = cached[(run.name, k)]
        for col in ("truth", "error"):
            if rec[col] != c[col]:
                raise eos.ReproductionError(f"{run.name}, k = {k}: the refitted sitting's {col} differs from "
                                            f"{soe.INPUTS.name}")
        if int(rec["n_candidates"]) != int(c["n_candidates"]):
            raise eos.ReproductionError(f"{run.name}, k = {k}: {rec['n_candidates']} candidates, cached "
                                        f"{c['n_candidates']}")
        rows = state.first[np.argsort(-state.lcb, kind="stable")[:k]]
        if not np.array_equal(run.deployed[rows], soe.parse_vec(c["truth"])):
            raise eos.ReproductionError(f"{run.name}, k = {k}: the candidates' rows do not give the cached truth")
        orig = original_errors(run, rows, n)
        if not np.array_equal(orig["rows_first"], rows):
            raise eos.ReproductionError(f"{run.name}, k = {k}: a candidate row is not its design's first rating")
        out.append({"file": run.name, "k": k, "baseline": False, "rows_first": _ivec(orig["rows_first"]),
                    "rows_rating": _ivec(orig["rows_rating"]), "n_ratings": _ivec(orig["n_ratings"]),
                    "e_orig": _vec(orig["e_orig"]), "e_first": _vec(orig["e_first"]),
                    "max_abs_prefix_error": float(np.max(np.abs(run.observed[:n] - run.deployed[:n])))})
    return out


def confirmation_inputs(run, state, k: int, initial: int, sd_fn, T: int, alpha: float) -> dict:
    """D, C, their true values, original errors and the published fresh errors of one run.

    The expressions are replay_end_of_study.confirmation's (shared model); the
    decision recomputed at rho_m = 0 is checked against confirmation() itself.
    """
    import bo_sensor_error_simulation as sim
    import replay_end_of_study as eos

    if eos._uses_process(sd_fn):
        raise ValueError(f"{run.name}: only the shared sitting model is reproduced here")
    d_row = int(state.first[int(np.argmax(state.lcb))])
    c_row = int(np.argmax(run.observed[:initial]))
    f_d, f_c = float(run.deployed[d_row]), float(run.deployed[c_row])
    rng = np.random.default_rng(eos.noise_seed(run.name, eos.PROC_CONFIRM, k))
    z_d, z_c = rng.standard_normal(k), rng.standard_normal(k)
    start = T - 2 * k + 1
    d_trials = [start + 2 * j + (j % 2) for j in range(k)]
    c_trials = [start + 2 * j + 1 - (j % 2) for j in range(k)]
    err_d = np.array([sd_fn(t) for t in d_trials]) * z_d
    err_c = np.array([sd_fn(t) for t in c_trials]) * z_c
    n = T - 2 * k
    orig = original_errors(run, [d_row, c_row], n)
    dec = eos.confirmation_decision(f_d, f_c, confirmation_ratings(f_d, err_d, orig["e_orig"][0], 0.0),
                                    confirmation_ratings(f_c, err_c, orig["e_orig"][1], 0.0), alpha)
    pub = eos.confirmation(run, state, k, initial, sd_fn, T, alpha)
    regret = run.y_opt - dec["shipped_value"]
    same = (regret == pub["regret"] and dec["p_value"] == pub["p_value"] and dec["claim"] == pub["claim"]
            and dec["shipped_d"] == pub["shipped_d"] and dec["truly_better"] == pub["truly_better"]
            and dec["fresh_mean_claim"] == pub["fresh_mean_claim"])
    d_is_c = sim._design_key(run.X[d_row]) == sim._design_key(run.X[c_row])
    if not same or d_is_c != pub["d_is_c"]:
        raise eos.ReproductionError(f"{run.name}: the recomputed confirmation differs from confirmation() "
                                    f"(regret {regret!r} against {pub['regret']!r}, p {dec['p_value']!r} against "
                                    f"{pub['p_value']!r})")
    return {"y_opt": float(run.y_opt), "d_row": d_row, "c_row": c_row,
            "d_rating_row": int(orig["rows_rating"][0]), "c_rating_row": int(orig["rows_rating"][1]),
            "n_ratings_d": int(orig["n_ratings"][0]), "n_ratings_c": int(orig["n_ratings"][1]),
            "f_d": f_d, "f_c": f_c, "e_d": float(orig["e_orig"][0]), "e_c": float(orig["e_orig"][1]),
            "e_d_first": float(orig["e_first"][0]), "e_c_first": float(orig["e_first"][1]),
            "err_d": _vec(err_d), "err_c": _vec(err_c), "d_is_c": bool(d_is_c), "gp_failed": bool(state.gp_failed),
            "regret": float(pub["regret"]), "p_value": float(pub["p_value"]), "claim": bool(pub["claim"]),
            "shipped_d": bool(pub["shipped_d"]), "truly_better": bool(pub["truly_better"]),
            "fresh_mean_claim": bool(pub["fresh_mean_claim"])}


def confirm_replay_stem(task: dict) -> list[dict]:
    """One stem's gaussian runs: refit the prefix GP at T - 2k and record the confirmation's inputs."""
    import torch

    import boba_benchmarks as bb
    import replay_end_of_study as eos

    arm, settings = task["arm"], task["settings"]
    T = arm.iterations
    spec = bb.BENCHMARKS[task["dataset"]]
    bounds = torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)
    out = []
    for r in task["runs"]:
        run = eos.read_run(Path(r["path"]), T)
        eos.check_logged_inference(run, [T - 2 * CONFIRM_K, T])
        eos.check_logged_error(run, T)
        sd_fn = eos.sitting_sd_fn(arm, r["error_model"], float(r["jitter_std"]), int(r["jitter_iteration"]),
                                  settings, variant=str(r.get("variant") or "") or None)
        state = eos.search_state(run, T - 2 * CONFIRM_K, bounds, settings.lcb_beta)
        rec = confirmation_inputs(run, state, CONFIRM_K, arm.initial_samples, sd_fn, T, settings.alpha)
        out.append({"file": run.name, "dataset": task["dataset"], "acquisition": task["acquisition"],
                    "seed": int(task["seed"]), "error_model": r["error_model"], "jitter_std": float(r["jitter_std"]),
                    "jitter_iteration": int(r["jitter_iteration"]), "variant": str(r.get("variant") or ""), **rec})
    return out


def sitting_tasks() -> list[dict]:
    """sitting_order_effects' stems (LogEI and qNEI, seeds 7-16, both cells), with the cached inputs attached."""
    inputs = soe.load_inputs(soe.INPUTS)
    cached = {(r["file"], int(r["k"])): r for r in inputs.to_dict("records")}
    tasks, _ = soe.replay_tasks()
    for t in tasks:
        ks = sorted({int(CELLS[soe.cell_of(r["jitter_std"], r["jitter_iteration"])][2]) for r in t["runs"]})
        clean = Path(t["clean_path"]).name
        names = [(clean, k) for k in ks] + [
            (Path(r["path"]).name, int(CELLS[soe.cell_of(r["jitter_std"], r["jitter_iteration"])][2]))
            for r in t["runs"]]
        missing = [nm for nm in names if nm not in cached]
        if missing:
            raise KeyError(f"{len(missing)} runs of {t['stem']} are not in {soe.INPUTS.name}, e.g. {missing[0]}")
        t["cached"] = {nm: {c: cached[nm][c] for c in ("truth", "error", "n_candidates")} for nm in names}
        t["clean_ks"] = ks
        t["names"] = names
    return tasks


def confirm_tasks() -> list[dict]:
    """replay_end_of_study's stems for the published confirmation scope (gaussian, LogEI, qNEI, UCB, 7-16)."""
    import replay_end_of_study as eos

    arm = eos.load_arm(ARM_DIR, "none")
    settings = eos.Settings(tournament_k=(), confirm_k=(CONFIRM_K,), rhos=(1.0,), alpha=ALPHA, lcb_beta=LCB_BETA)
    eos.validate_settings(settings)
    eos.validate_arm(arm, settings)
    filters = eos.Filters(acquisitions=frozenset(CONFIRM_ACQS), seeds=frozenset(CONFIRM_SEEDS),
                          error_models=frozenset(CONFIRM_PROCESSES))
    tasks, _ = eos.build_tasks(arm, filters, settings, Path("unused"), resume=False)
    for t in tasks:
        t["names"] = [Path(r["path"]).name for r in t["runs"]]
    return tasks


def _write_rows(rows: list[dict], path: Path, keys: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    soe._write_gz(frame, path)
    return frame


def _run_pool(tasks: list[dict], worker, workers: int, rows: list[dict], out: Path, keys: list[str],
              label: str) -> list[dict]:
    import replay_end_of_study as eos

    start = time.perf_counter()
    if workers <= 1:
        eos._worker_init()
        for i, t in enumerate(tasks, 1):
            rows += worker(t)
            if i % 25 == 0 or i == len(tasks):
                _write_rows(rows, out, keys)
                print(f"  {label}: {i}/{len(tasks)} stems, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=eos._worker_init) as pool:
            futures = {pool.submit(worker, t): t["stem"] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                rows += fut.result()
                if i % 25 == 0 or i == len(tasks):
                    _write_rows(rows, out, keys)
                    rate = i / (time.perf_counter() - start)
                    print(f"  {label}: {i}/{len(tasks)} stems, {(time.perf_counter() - start) / 60:.1f} min, "
                          f"eta {(len(tasks) - i) / rate / 60:.1f} min", flush=True)
    return rows


def run_replay(args) -> None:
    start = time.perf_counter()
    for proc in args.procedures:
        if proc == "sitting":
            tasks, out, keys, worker = sitting_tasks(), args.sitting_out, ["file", "k"], sitting_replay_stem
            done_key = lambda r: (r["file"], int(r["k"]))  # noqa: E731
        else:
            tasks, out, keys, worker = confirm_tasks(), args.confirm_out, ["file"], confirm_replay_stem
            done_key = lambda r: r["file"]  # noqa: E731
        rows: list[dict] = []
        if args.resume and out.is_file():
            old = pd.read_csv(out, float_precision="round_trip", low_memory=False, keep_default_na=False,
                              dtype={c: str for c in ("rows_first", "rows_rating", "n_ratings", "e_orig", "e_first",
                                                      "err_d", "err_c", "variant")})
            rows = old.to_dict("records")
            done = {done_key(r) for r in rows}
            tasks = [t for t in tasks if not set(t["names"]) <= done]
        if args.limit:
            tasks = tasks[:args.limit]  # a smoke test; never the published inputs
        print(f"{proc}: {len(tasks)} stems to replay ({sum(len(t['runs']) for t in tasks)} noisy runs); "
              f"{len(rows)} rows kept from an earlier pass", flush=True)
        if tasks:
            rows = _run_pool(tasks, worker, args.workers, rows, out, keys, proc)
        frame = _write_rows(rows, out, keys)
        print(f"wrote {soe.rel(out)}: {len(frame)} rows; every run matched replay_end_of_study in process; "
              f"{(time.perf_counter() - start) / 60:.1f} min on {args.workers} worker(s)", flush=True)


# ---------------------------------------------------------------------------
# score (i): the sitting
# ---------------------------------------------------------------------------


def load_sitting_memory(path: Path = SITTING_INPUTS) -> pd.DataFrame:
    d = pd.read_csv(path, float_precision="round_trip", low_memory=False, keep_default_na=False,
                    dtype={c: str for c in ("rows_first", "rows_rating", "n_ratings", "e_orig", "e_first")})
    d["baseline"] = d["baseline"].astype(str).str.lower().isin(("true", "1"))
    d["k"] = d["k"].astype(int)
    return d


def sitting_regrets(inputs: pd.DataFrame, mem: pd.DataFrame) -> pd.DataFrame:
    """Per input row: the shipped candidate and its regret at every rho_m, from e_orig and from the first rating."""
    d = inputs.merge(mem[["file", "k", "e_orig", "e_first", "n_ratings"]], on=["file", "k"], how="left",
                     validate="one_to_one", indicator=True)
    if (d["_merge"] != "both").any():
        raise KeyError(f"{int((d['_merge'] != 'both').sum())} sitting rows have no original errors; run replay")
    out = []
    for r in d.itertuples(index=False):
        truth, error = soe.parse_vec(r.truth), soe.parse_vec(r.error)
        e_orig, e_first = parse_vec(r.e_orig), parse_vec(r.e_first)
        if not (len(truth) == len(error) == len(e_orig) == len(e_first) == int(r.n_candidates)):
            raise ValueError(f"{r.file}, k = {r.k}: candidate vectors of unequal length")
        if r.baseline and (np.any(e_orig != 0.0) or np.any(error != 0.0)):
            raise ValueError(f"{r.file}: a clean twin with a rating error")
        row = {"file": r.file, "k": int(r.k)}
        pub = soe.published_pick(truth, error)
        for rho in MEMORY:
            lab = mem_label(rho)
            c = sitting_pick(truth, error, e_orig, rho)
            row[f"cand_{lab}"], row[f"reg_{lab}"] = c, r.y_opt - float(truth[c])
            c1 = sitting_pick(truth, error, e_first, rho)
            row[f"cand_{lab}first"], row[f"reg_{lab}first"] = c1, r.y_opt - float(truth[c1])
        if row[f"cand_{mem_label(0.0)}"] != pub or row[f"cand_{mem_label(0.0)}first"] != pub:
            raise ControlError(f"{r.file}, k = {r.k}: rho_m = 0 ships candidate {row['cand_mem0']}, the published "
                               f"sitting {pub}")
        # soe.per_run_frame checks 'none' against the published tables: here it is rho_m = 0 itself.
        row["cand_none"], row["reg_none"] = row["cand_mem0"], row["reg_mem0"]
        row["mean_e_orig_over_sigma"] = (float(np.mean(e_orig)) / float(r.jitter_std)) if not r.baseline else 0.0
        row["e_orig_top_over_sigma"] = (float(e_orig[0]) / float(r.jitter_std)) if not r.baseline else 0.0
        row["e_orig_published_pick_over_sigma"] = (float(e_orig[pub]) / float(r.jitter_std)) if not r.baseline else 0.0
        row["share_exact_original"] = float(np.mean(e_orig == 0.0))
        row["n_multi_rated"] = int(np.sum(parse_ivec(r.n_ratings) > 1)) if not r.baseline else 0
        out.append(row)
    return pd.DataFrame(out)


def attach_lcb2(w: pd.DataFrame, ship: pd.DataFrame) -> pd.DataFrame:
    """LCB2 after all 50 trials on the same runs and their clean twins, by sitting_by_magnitude's own join."""
    attach = pd.DataFrame({"file": w["file"], "dataset": w["dataset"], "acquisition": w["acquisition"],
                           "seed": w["seed"], "opt_z": w["opt_z"], "regret_noisy": w["reg_none"],
                           "regret_clean": w["reg_none_clean"], "ref_noisy": w["ref_regret"],
                           "ref_clean": w["ref_regret_clean"], "top_candidate_regret_noisy": w["top_candidate_regret"]})
    m = sbm.attach_ship_rules(attach, ship)[["file", "regret_lcb1", "regret_lcb1_clean", "regret_lcb2",
                                             "regret_lcb2_clean"]]
    chk = w[["file", "regret_lcb1", "regret_lcb1_clean"]].merge(m, on="file", suffixes=("", "_2"), validate="one_to_one")
    if not (np.array_equal(chk["regret_lcb1"], chk["regret_lcb1_2"])
            and np.array_equal(chk["regret_lcb1_clean"], chk["regret_lcb1_clean_2"])):
        raise ValueError("the two joins of ship_rules_per_run.csv disagree on LCB1")
    out = w.merge(m[["file", "regret_lcb2", "regret_lcb2_clean"]], on="file", how="left", validate="one_to_one")
    if out[["regret_lcb2", "regret_lcb2_clean"]].isna().any().any():
        raise KeyError("a noisy run or its clean twin has no LCB2 rescoring")
    return out


def scaled_block(w: pd.DataFrame, name: str, opt_z: dict[str, float]) -> pd.DataFrame:
    """sitting_order_effects.scaled_block (aba.paired_frame of the standard process and the sitting at one rho_m,
    the increment over the memoryless sitting and over LCB1), plus the increment over LCB2."""
    pf = soe.scaled_block(w, name, opt_z)
    lcb2 = w[["file", "regret_lcb2", "regret_lcb2_clean"]]
    pf = pf.merge(lcb2, on="file", validate="one_to_one")
    z = pf["dataset"].map(opt_z).astype(float)
    return pf.assign(inc_lcb2=pf["regret_lcb2"] / z - pf["trt_noisy"],
                     gain_lcb2=pf["ref_noisy"] - pf["regret_lcb2"] / z,
                     price_lcb2=pf["regret_lcb2_clean"] / z - pf["ref_clean"])


def summarise_block(pf: pd.DataFrame, rng: np.random.Generator) -> dict:
    """analyse_boba_adaptations' estimand on one block, the increments and the share guard."""
    s = aba.summarise(pf, rng)
    flags = aba.share_flags(pf)
    cols = ["cost", "gain", "price", "inc_published", "inc_lcb1", "inc_lcb2", "gain_lcb1", "price_lcb1",
            "gain_lcb2", "price_lcb2", "price_over_published"]
    per = pf.groupby("dataset")[cols].mean()
    out = {"n_runs": int(len(pf)), "n_landscapes": int(len(per)), "cost": s["cost"]}
    for col in ("gain", "price", "inc_published", "inc_lcb1", "inc_lcb2"):
        ci = sbm.mean_ci(per[col], rng)
        out.update({col: ci["mean"], f"{col}_lo": ci["lo"], f"{col}_hi": ci["hi"]})
        if col != "price":
            out[f"{col}_landscapes_positive"] = int((per[col] > 0).sum())
            out[f"{col}_p"] = sbm.wilcoxon_p(per[col].to_numpy())
    if abs(out["gain"] - s["gain"]) > 1e-12 or abs(out["price"] - s["price"]) > 1e-12:
        raise AssertionError("the gain or price differs from analyse_boba_adaptations.summarise")
    suppressed = flags["recovered_suppressed"]
    out.update({"share": np.nan if suppressed else s["recovered"],
                "share_lo": np.nan if suppressed else s["recovered_lo"],
                "share_hi": np.nan if suppressed else s["recovered_hi"], "share_suppressed": suppressed,
                "lcb1_gain": float(per["gain_lcb1"].mean()), "lcb1_price": float(per["price_lcb1"].mean()),
                "lcb2_gain": float(per["gain_lcb2"].mean()), "lcb2_price": float(per["price_lcb2"].mean()),
                "price_over_published": float(per["price_over_published"].mean()),
                "changed_share_noisy": float(pf["changed"].mean()),
                "changed_share_clean": float(pf.drop_duplicates(["dataset", "acquisition", "seed"])["changed_clean"]
                                             .mean())})
    return out


def score_sitting(w: pd.DataFrame, opt_z: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, by_proc = [], []
    first_gap = 0.0
    for seeds_name, seeds in SEED_SETS.items():
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for cell, (sigma, onset, k) in CELLS.items():
            sub = w[(w["cell"] == cell) & w["seed"].isin(seeds)]
            for rho in MEMORY:
                name = mem_label(rho)
                pf = scaled_block(sub, name, opt_z)
                base = {"seeds": seeds_name, "cell": cell, "k": k, "rho_m": rho}
                rows.append({**base, **summarise_block(pf, rng)})
                # The same sitting with the first rating's error in place of the ship rule's rating.
                pf1 = scaled_block(sub, f"{name}first", opt_z)
                g1 = pf1.groupby("dataset")["gain"].mean().mean()
                first_gap = max(first_gap, abs(float(g1) - rows[-1]["gain"]))
                for proc, g in pf.groupby("error_model"):
                    per = g.groupby("dataset")[["cost", "gain", "price", "inc_published", "inc_lcb1",
                                                "inc_lcb2"]].mean()
                    by_proc.append({**base, "error_model": proc, "n_runs": int(len(g)),
                                    **{c: float(per[c].mean()) for c in per.columns},
                                    "changed_share_noisy": float(g["changed"].mean())})
    table = pd.DataFrame(rows)
    for col in ESTIMANDS:
        table[f"{col}_p_holm6"] = np.nan
        for seeds_name in SEED_SETS:
            fam = (table["seeds"] == seeds_name) & (table["rho_m"] > 0)
            table.loc[fam, f"{col}_p_holm6"] = sbm.holm(table.loc[fam, f"{col}_p"].to_numpy())
    return table, pd.DataFrame(by_proc), {"first_rating_max_gain_shift": first_gap}


# ---------------------------------------------------------------------------
# score (ii): the confirmation trials
# ---------------------------------------------------------------------------


CONFIRM_COLS = ["regret", "p_value", "claim", "shipped_d", "truly_better", "d_is_c"]


def load_confirm_memory(path: Path = CONFIRM_INPUTS) -> pd.DataFrame:
    d = pd.read_csv(path, float_precision="round_trip", low_memory=False, keep_default_na=False,
                    dtype={"err_d": str, "err_c": str, "variant": str})
    for col in ("d_is_c", "gp_failed", "claim", "shipped_d", "truly_better", "fresh_mean_claim"):
        d[col] = d[col].astype(str).str.lower().isin(("true", "1"))
    return d


def read_confirm_published(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The published confirm_k2 rows: from the replay's per-stem files (exact) and from the assembled table."""
    cols = ["arm", "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "variant",
            "file", "procedure", "family", "k", "regret_noisy", "claim_noisy", "truly_better_noisy",
            "false_claim_noisy", "claim_clean", "shipped_d_noisy", "p_value_noisy", "d_is_c_noisy"]
    frames = []
    stems = files.assign(stem=files["file"].str.split("_jittered_", n=1).str[0])
    for (dataset, stem), _ in stems.groupby(["dataset", "stem"]):
        p = pd.read_csv(CONFIRM_PUBLISHED / "per_run" / ARM_DIR.name / dataset / f"{stem}.csv", usecols=cols,
                        float_precision="round_trip", low_memory=False)
        frames.append(p[p["procedure"] == CONFIRM_PROCEDURE])
    per_stem = pd.concat(frames, ignore_index=True)
    table = pd.read_csv(CONFIRM_PUBLISHED / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False,
                        float_precision="round_trip")
    table = table[(table["arm"] == ARM_DIR.name) & (table["procedure"] == CONFIRM_PROCEDURE)]

    def scope(d: pd.DataFrame) -> pd.DataFrame:
        """The published confirmation scope: gaussian error, LogEI, qNEI and UCB, seeds 7-16."""
        keep = (d["error_model"].isin(CONFIRM_PROCESSES) & d["acquisition"].isin(CONFIRM_ACQS)
                & d["seed"].isin(CONFIRM_SEEDS))
        return d[keep].reset_index(drop=True)

    return scope(per_stem), scope(table)


def check_confirm_control(cache: pd.DataFrame, published: pd.DataFrame) -> dict:
    """The cached in-process confirmation must equal the published per-run rows bit for bit."""
    pub = published.rename(columns={f"{c}_noisy": c for c in CONFIRM_COLS + ["false_claim"]})
    m = cache[["file"] + CONFIRM_COLS].merge(pub[["file"] + CONFIRM_COLS + ["false_claim"]], on="file", how="left",
                                            suffixes=("", "_pub"), validate="one_to_one", indicator=True)
    missing = int((m["_merge"] != "both").sum())
    if missing:
        raise ControlError(f"{missing} confirmation runs have no published row, e.g. "
                           f"{m.loc[m['_merge'] != 'both', 'file'].iloc[0]}")
    worst = {}
    for c in CONFIRM_COLS:
        a, b = m[c].to_numpy(), m[f"{c}_pub"].to_numpy()
        if c in ("regret", "p_value"):
            a, b = a.astype(float), b.astype(float)
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            worst[c] = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
        else:
            a = np.array([str(v).lower() in ("true", "1") for v in a])
            b = np.array([str(v).lower() in ("true", "1") for v in b])
            bad = a != b
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ControlError(f"{int(bad.sum())} confirmation runs differ from the published replay in {c}, e.g. "
                               f"{m['file'].iloc[i]}: {a[i]!r} against {b[i]!r}")
    fc = m["claim"].astype(bool) & ~m["truly_better"].astype(bool)
    if not np.array_equal(fc.to_numpy(), m["false_claim"].astype(str).str.lower().isin(("true", "1")).to_numpy()):
        raise ControlError("the published false_claim is not claim and not truly_better")
    return {"n_runs": int(len(m)), "max_abs_difference": worst}


def confirm_decisions(cache: pd.DataFrame, rho_m: float, first: bool = False) -> pd.DataFrame:
    """Every run's confirmation decision when both designs' fresh ratings remember rho_m of their original errors."""
    import replay_end_of_study as eos

    out = []
    for r in cache.itertuples(index=False):
        err_d, err_c = parse_vec(r.err_d), parse_vec(r.err_c)
        e_d, e_c = (r.e_d_first, r.e_c_first) if first else (r.e_d, r.e_c)
        dec = eos.confirmation_decision(r.f_d, r.f_c, confirmation_ratings(r.f_d, err_d, e_d, rho_m),
                                        confirmation_ratings(r.f_c, err_c, e_c, rho_m), ALPHA)
        out.append({"file": r.file, "regret": r.y_opt - dec["shipped_value"], "p_value": dec["p_value"],
                    "claim": dec["claim"], "shipped_d": dec["shipped_d"], "truly_better": dec["truly_better"],
                    "false_claim": dec["false_claim"], "fresh_mean_claim": dec["fresh_mean_claim"],
                    "d_is_c": bool(r.d_is_c)})
    return pd.DataFrame(out)


def claims_frame(dec: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """A per-run frame shaped as replay_end_of_study.claims_table reads it (the clean twin's claim is unchanged)."""
    base = published[["arm", "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration",
                      "variant", "file", "procedure", "family", "k", "claim_clean"]]
    d = dec.rename(columns={c: f"{c}_noisy" for c in ("regret", "p_value", "claim", "shipped_d", "truly_better",
                                                       "false_claim", "fresh_mean_claim", "d_is_c")})
    return base.merge(d, on="file", how="inner", validate="one_to_one")


def compare_claims(mine: pd.DataFrame, published: pd.DataFrame) -> int:
    """Every value of the published claims rows must be reproduced exactly; returns the number compared."""
    keys = ["arm", "procedure", "error_model", "scope", "jitter_std", "jitter_iteration"]
    m = published.merge(mine, on=keys, how="left", suffixes=("_pub", ""), validate="one_to_one", indicator=True)
    if (m["_merge"] != "both").any():
        raise ControlError("a published claims row is not reproduced at rho_m = 0")
    n = 0
    for c in published.columns:
        if c in keys:
            continue
        a, b = m[c], m[f"{c}_pub"]
        if pd.api.types.is_numeric_dtype(b):
            a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        else:
            bad = (a.astype(str) != b.astype(str)).to_numpy()
        if bad.any():
            raise ControlError(f"claims column {c} differs from end_of_study_claims.csv at rho_m = 0")
        n += len(a)
    return n


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------


def _ci(v: float, lo: float, hi: float) -> str:
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _p(p: float) -> str:
    return "  nan" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}")


def _share(r) -> str:
    if r["share_suppressed"]:
        return "withheld"
    return f"{r['share']:+.0%} [{r['share_lo']:+.0%}, {r['share_hi']:+.0%}]"


def _pct(v: float, lo: float | None = None, hi: float | None = None) -> str:
    if lo is None:
        return f"{100 * v:.1f}%"
    return f"{100 * v:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def sitting_lines(table: pd.DataFrame, by_proc: pd.DataFrame, diag: pd.DataFrame) -> list[str]:
    L = []
    for seeds_name in SEED_SETS:
        scope = "held-out seeds 12-16" if seeds_name == "12-16" else "seeds 7-16"
        for cell, (sigma, onset, k) in CELLS.items():
            t = table[(table["seeds"] == seeds_name) & (table["cell"] == cell)]
            r0 = t.iloc[0]
            L.append(f"\n=== (i) {cell.replace('_', ' ')}, sitting of k = {k}; {scope}; LogEI and qNEI, four processes "
                     f"pooled; {int(r0['n_runs'])} noisy runs on {int(r0['n_landscapes'])} landscapes ===")
            L.append(f"cost of error of the standard process {r0['cost']:.3f} of opt_z; after 50 trials LCB1 gains "
                     f"{r0['lcb1_gain']:+.3f} (price {r0['lcb1_price']:.3f}), LCB2 {r0['lcb2_gain']:+.3f} "
                     f"(price {r0['lcb2_price']:.3f})")
            L.append(f"{'rho_m':<6s}{'gain over standard':>26s}{'ahd':>4s}{'p':>6s}{'Holm':>6s}{'share of cost':>20s}"
                     f"{'price':>8s}{'over memoryless':>26s}{'ahd':>4s}{'p':>6s}{'Holm':>6s}"
                     f"{'over LCB1':>26s}{'ahd':>4s}{'p':>6s}{'Holm':>6s}{'over LCB2':>26s}{'ahd':>4s}{'p':>6s}"
                     f"{'Holm':>6s}{'moved':>7s}")
            for _, r in t.iterrows():
                cells = []
                for col in ESTIMANDS:
                    cells.append(f"{_ci(r[col], r[f'{col}_lo'], r[f'{col}_hi']):>26s}"
                                 f"{int(r[f'{col}_landscapes_positive']):>4d}{_p(r[f'{col}_p']):>6s}"
                                 f"{_p(r[f'{col}_p_holm6']):>6s}")
                L.append(f"{r['rho_m']:<6g}{cells[0]}{_share(r):>20s}{r['price']:>8.3f}{cells[1]}{cells[2]}{cells[3]}"
                         f"{r['changed_share_noisy']:>7.1%}")
    L.append("\n(i) by error process, held-out seeds 12-16 (landscape means, then the mean over landscapes; no intervals):")
    bp = by_proc[by_proc["seeds"] == "12-16"]
    for cell in CELLS:
        b = bp[bp["cell"] == cell]
        for col, what in (("gain", "gain over the standard process"), ("inc_published", "change from the memoryless "
                                                                                       "sitting"),
                          ("inc_lcb1", "increment over LCB1")):
            piv = b.pivot_table(index="rho_m", columns="error_model", values=col, aggfunc="first")
            piv = piv[[p for p in soe.PROCESSES if p in piv.columns]]
            L.append(f"{cell}, {what}:")
            L.append(piv.round(3).to_string())
    L.append("\n(i) selection on the original error (noisy runs, seeds 7-16; mean e_orig in units of the cell's sigma, "
             "which includes the bias and the drift ramp of the original trial):")
    for cell, (sigma, onset, k) in CELLS.items():
        g = diag[diag["cell"] == cell]
        parts = []
        for col, what in (("mean_e_orig_over_sigma", "the k candidates"), ("e_orig_top_over_sigma", "the top-ranked"),
                          ("e_orig_published_pick_over_sigma", "the published look winner")):
            m = g.groupby("error_model")[col].mean()
            parts.append(f"{what} " + ", ".join(f"{p} {m[p]:+.2f}" for p in soe.PROCESSES if p in m.index))
        L.append(f"  {cell}: " + "; ".join(parts) + f"; candidates first rated before the onset (exact, e_orig = 0) "
                 f"{g['share_exact_original'].mean():.1%}")
    return L


def confirm_lines(claims: dict[float, pd.DataFrame], standard: pd.Series, diag: dict) -> list[str]:
    L = [f"\n=== (ii) confirmation trials ({CONFIRM_PROCEDURE}: D and C rated twice each in trials 47-50); gaussian "
         f"error, LogEI, qNEI and UCB, seeds 7-16, four magnitudes and both onsets pooled; "
         f"{diag['n_runs']} noisy runs on 20 landscapes ==="]
    L.append(f"standard process (re-rates nothing, so unchanged by memory): false claims "
             f"{_pct(standard['false_claim_rate'], standard['false_claim_rate_lo'], standard['false_claim_rate_hi'])}, "
             f"detection {_pct(standard['power'], standard['power_lo'], standard['power_hi'])}")
    L.append(f"{'rho_m':<6s}{'test: claims':>13s}{'false claims':>23s}{'excl D=C':>10s}{'size':>23s}{'detection':>23s}"
             f"{'no test: false claims':>25s}{'excl D=C':>10s}{'detection':>23s}{'ships D':>9s}")
    for rho, c in claims.items():
        p = c[c["scope"] == "pooled"].iloc[0]
        L.append(f"{rho:<6g}{_pct(p['claim_rate']):>13s}"
                 f"{_pct(p['false_claim_rate'], p['false_claim_rate_lo'], p['false_claim_rate_hi']):>23s}"
                 f"{_pct(p['false_claim_rate_excl_dc']):>10s}"
                 f"{_pct(p['size'], p['size_lo'], p['size_hi']):>23s}"
                 f"{_pct(p['power'], p['power_lo'], p['power_hi']):>23s}"
                 f"{_pct(p['fresh_mean_false_claim_rate'], p['fresh_mean_false_claim_rate_lo'], p['fresh_mean_false_claim_rate_hi']):>25s}"
                 f"{_pct(p['fresh_mean_excl_dc_false_claim_rate']):>10s}"
                 f"{_pct(p['fresh_mean_power'], p['fresh_mean_power_lo'], p['fresh_mean_power_hi']):>23s}"
                 f"{_pct(p['shipped_d_share']):>9s}")
    L.append("\n(ii) by cell (the test's false-claim rate / size / detection, %; each cell 600 runs):")
    cells = claims[MEMORY[0]][claims[MEMORY[0]]["scope"] == "cell"][["jitter_std", "jitter_iteration"]]
    head = f"{'rho_m':<6s}" + "".join(f"{f'{s:g}s/t{int(o) + 1}':>16s}" for s, o in cells.itertuples(index=False))
    L.append(head)
    for rho, c in claims.items():
        cc = c[c["scope"] == "cell"]
        L.append(f"{rho:<6g}" + "".join(
            f"{f'{100 * r.false_claim_rate:.1f}/{100 * r.size:.1f}/{100 * r.power:.0f}':>16s}"
            for r in cc.itertuples(index=False)))
    L.append(f"\n(ii) selection on the original error: mean (e_D - e_C) / sigma {diag['mean_dc_gap']:+.3f} over the "
             f"runs with D not C ({diag['mean_dc_gap_false']:+.3f} where f(D) <= f(C)); D was rated more than once "
             f"before trial 47 in {diag['multi_d']} runs, C in {diag['multi_c']}.")
    return L


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def score(args) -> None:
    opt_z = aba.dataset_scale("opt_z")
    lines = [__doc__.split("\n\n")[0], "",
             "POST HOC, outside every Holm family of the paper; the memory strengths rho_m are assumed. Model: a look "
             "at a design whose original rating had error e_orig gets rho_m * e_orig + sqrt(1 - rho_m^2) * e_fresh, "
             "e_fresh the published replay's own draw for that look, e_orig = objective_observed - objective_true of "
             "the design's original rating in the run log (a design rated more than once before the procedure: its "
             "highest rating, ties to the earliest, the rating the standard ship rule reads). Noisy runs only: a "
             "clean twin's original errors are zero and its looks exact, so its shipped design and every price are "
             "unchanged by construction (checked run by run in the sitting; the confirmation's clean-twin claims are "
             "the published ones)."]

    # (i) the sitting
    inputs = soe.load_inputs(args.sitting_order_inputs)
    mem = load_sitting_memory(args.sitting_inputs)
    reg = sitting_regrets(inputs, mem)
    ship = soe.read_ship_rules()
    w, control = soe.per_run_frame(inputs, reg, ship, opt_z)
    w = attach_lcb2(w, ship)
    expected = len(soe.ACQS) * len(soe.PROCESSES) * len(SEED_SETS["7-16"])
    counts = w.groupby(["cell", "dataset"]).size()
    if (counts != expected).any() or w["dataset"].nunique() != 20 or set(w["cell"]) != set(CELLS):
        raise SystemExit(f"incomplete sitting inputs: expected {expected} noisy runs per cell and landscape on 20 "
                         f"landscapes, found {counts.describe().to_dict()}")
    clean_moved = {rho: int((w[f"reg_{mem_label(rho)}_clean"] != w["reg_none_clean"]).sum()) for rho in MEMORY}
    if any(clean_moved.values()):
        raise ControlError(f"memory moved a clean twin's sitting: {clean_moved}")
    table, by_proc, extra = score_sitting(w, opt_z)
    noisy_mem = mem[~mem["baseline"]]
    n_cand = int(noisy_mem["n_ratings"].map(lambda s: len(parse_ivec(s))).sum())
    n_multi = int(noisy_mem["n_ratings"].map(lambda s: int(np.sum(parse_ivec(s) > 1))).sum())
    raw = control["raw_gap_to_table"]
    lines.append(f"\nControl (i): at rho_m = 0 every one of {control['per_stem']['n_runs']} noisy runs (and its clean twin) "
                 f"ships the published sitting ({soe.procedure_name(2)} of {soe.PUBLISHED[2].name}, "
                 f"{soe.procedure_name(16)} of {soe.PUBLISHED[16].name}) bit for bit: largest difference "
                 f"{max(control['per_stem']['max_abs_difference'].values()):.1e} against the replay's per-stem outputs "
                 f"and {max(control['table']['max_abs_difference'].values()):.1e} against end_of_study_per_run.csv.gz "
                 f"after the replay's own CSV round trip (without it {raw['runs_differing']} runs differ by at most "
                 f"{raw['max_abs']:.1e}, the published table's float parsing). The replay stage refitted every noisy "
                 f"run's prefix GP and matched replay_end_of_study.tournament and {soe.INPUTS.name} (candidates, true "
                 f"values, look errors) bit for bit; every logged error_magnitude equals observed - true to 1e-9; "
                 f"every clean twin's prefix ratings are exact, and no clean twin's shipped design moves at any "
                 f"rho_m (so every price below is the published price).")
    lines.append(f"Original ratings: {n_multi} of {n_cand} candidates ({n_multi / n_cand:.2%}) were rated more than "
                 f"once before the sitting; with the first rating's error in their place no gain over the standard "
                 f"process moves by more than {extra['first_rating_max_gain_shift']:.4f} of opt_z.")
    lines.append("\nUnits: shares of the achievable improvement opt_z, deployed design at T = 50. gain = standard process "
                 "less the sitting with memory (noisy run); price = the sitting less the standard process on the clean "
                 "twin (unchanged by memory); 'over memoryless' = the published sitting's regret less the regret with "
                 "memory, per run (negative: memory costs the sitting); 'over LCB1'/'over LCB2' = the regret of LCB1/"
                 "LCB2 after all 50 trials (no extra trial) less the regret with memory. [95% landscape bootstrap, "
                 "2,000 resamples; the landscapes share their random numbers, so it treats correlated clusters as "
                 "independent]; ahd = landscapes of 20 with a positive value; p = two-sided Wilcoxon over the 20 "
                 "landscape means; Holm = " + HOLM_FAMILY + " (rho_m = 0, the published sitting, stands outside); "
                 "share of cost = gain over the standard process's cost of error, a ratio of landscape means, withheld "
                 "where that cost is below 0.01 or negative on a landscape; moved = share of noisy runs whose shipped "
                 "design differs from the published sitting's. The intervals of the rho_m = 0 rows come from this "
                 "script's bootstrap and differ in the third decimal from the paper's generator.")
    lines += sitting_lines(table, by_proc, reg.merge(w[["file", "k", "cell", "error_model"]], on=["file", "k"]))

    # (ii) the confirmation trials
    import replay_end_of_study as eos

    cache = load_confirm_memory(args.confirm_inputs)
    per_stem, gz = read_confirm_published(cache[["file", "dataset"]])
    if len(cache) != 4800 or cache["file"].nunique() != 4800 or len(gz) != 4800:
        raise SystemExit(f"incomplete confirmation inputs: {len(cache)} cached runs, {len(gz)} published, "
                         f"expected 4,800")
    ctrl_stem = check_confirm_control(cache, per_stem)
    ctrl_table = check_confirm_control(soe.csv_round_trip(cache[["file"] + CONFIRM_COLS]), gz)
    published_claims = pd.read_csv(CONFIRM_PUBLISHED / "end_of_study_claims.csv", float_precision="round_trip")
    pub_rows = published_claims[(published_claims["arm"] == ARM_DIR.name)
                                & (published_claims["procedure"] == CONFIRM_PROCEDURE)
                                & published_claims["error_model"].isin(CONFIRM_PROCESSES)]
    standard = published_claims[(published_claims["arm"] == ARM_DIR.name)
                                & (published_claims["procedure"] == "standard")
                                & (published_claims["error_model"] == "gaussian")
                                & (published_claims["scope"] == "pooled")].iloc[0]
    claims, decisions, first_rates = {}, {}, {}
    for rho in MEMORY:
        dec = confirm_decisions(cache, rho)
        if rho == 0.0:
            m = dec.merge(cache[["file"] + CONFIRM_COLS + ["fresh_mean_claim"]], on="file", suffixes=("", "_c"))
            for c in CONFIRM_COLS + ["fresh_mean_claim"]:
                if not np.array_equal(m[c].to_numpy(), m[f"{c}_c"].to_numpy()):
                    raise ControlError(f"rho_m = 0 does not reproduce the published confirmation in {c}")
            if not np.array_equal(dec["fresh_mean_claim"].to_numpy(), dec["shipped_d"].to_numpy()):
                raise ControlError("a fresh-mean claim differs from shipped_d in a noisy run (a tie)")
        frame = claims_frame(dec, gz)
        c = eos.claims_table(frame)
        if rho == 0.0:
            n_compared = compare_claims(c, pub_rows)
        claims[rho] = c.assign(rho_m=rho)
        decisions[rho] = dec
        dec1 = confirm_decisions(cache, rho, first=True)
        better = dec1["truly_better"].to_numpy(dtype=bool)
        first_rates[rho] = (float(dec1["false_claim"].mean()),
                            float((dec1["claim"].to_numpy(dtype=bool) & better).sum() / better.sum()))
    dc = cache["d_is_c"].to_numpy()
    dc_moved = sum(int((decisions[rho]["claim"].to_numpy()[dc] != decisions[0.0]["claim"].to_numpy()[dc]).sum())
                   for rho in MEMORY)
    sig = cache["jitter_std"].astype(float)
    gap = (cache["e_d"] - cache["e_c"]) / sig
    notc = ~cache["d_is_c"]
    diag = {"n_runs": int(len(cache)), "mean_dc_gap": float(gap[notc].mean()),
            "mean_dc_gap_false": float(gap[notc & ~cache["truly_better"]].mean()),
            "multi_d": int((cache["n_ratings_d"] > 1).sum()), "multi_c": int((cache["n_ratings_c"] > 1).sum())}
    lines.append(f"\nControl (ii): the replay stage refitted every run's prefix GP at trial {50 - 2 * CONFIRM_K} and matched "
                 f"replay_end_of_study.confirmation in process; the {ctrl_stem['n_runs']} cached runs equal the "
                 f"published {CONFIRM_PROCEDURE} rows of {CONFIRM_PUBLISHED.name} bit for bit (regret, p value, claim, "
                 f"shipped design, truly better, D = C) against the per-stem outputs (largest difference "
                 f"{max(ctrl_stem['max_abs_difference'].values()):.1e}) and against end_of_study_per_run.csv.gz after "
                 f"the CSV round trip ({max(ctrl_table['max_abs_difference'].values()):.1e}); at rho_m = 0 every run's "
                 f"decision is reproduced, and replay_end_of_study.claims_table reproduces all {n_compared} values of "
                 f"the published end_of_study_claims.csv rows ({CONFIRM_PROCEDURE}, gaussian, 8 cells and pooled) "
                 f"exactly. GP fallbacks {int(cache['gp_failed'].sum())}.")
    lines.append("Claim definition (published, replay_end_of_study.claim_rates): the test claims an improvement when a "
                 "one-sided paired t-test of D's two fresh ratings against C's (paired in counterbalanced order) gives "
                 "p < 0.05; a claim is false when f(D) <= f(C); the false-claim rate is over all 4,800 runs, the runs "
                 "whose D is C allowed to claim ('excl D=C' counts no claim there, as the standard process makes "
                 "none); size = claims among the runs with f(D) <= f(C) (the test's one-sided level is 0.05); "
                 "detection = claims among the runs with f(D) > f(C). 'No test' claims whenever D's mean fresh "
                 f"rating exceeds C's. [95% landscape bootstrap, the published draws.] Of the {int(dc.sum())} runs whose "
                 f"D is C, {dc_moved} change their claim at any rho_m (both designs remember the same rating, and the "
                 f"t statistic does not change with a common scale).")
    lines += confirm_lines(claims, standard, diag)
    lines.append("With the first rating's error in place of the ship rule's rating (it differs only where a design was "
                 "rated more than once): false claims " + " / ".join(f"{100 * v[0]:.1f}%" for v in first_rates.values())
                 + ", detection " + " / ".join(f"{100 * v[1]:.1f}%" for v in first_rates.values())
                 + " at rho_m = " + " / ".join(f"{r:g}" for r in first_rates) + " (pooled, no intervals).")
    lines.append(f"\nWrote {', '.join(soe.rel(p) for p in (args.per_run, args.summary, args.by_process, args.confirm_per_run, args.confirm_claims, args.register))}")

    # outputs
    REVIEW.mkdir(parents=True, exist_ok=True)
    cols = ["file", "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "variant",
            "cell", "k", "n_candidates", "opt_z", "ref_regret", "ref_regret_clean", "regret_lcb1", "regret_lcb1_clean",
            "regret_lcb2", "regret_lcb2_clean", "file_clean"]
    regs = [c for c in w.columns if c.startswith(("reg_mem", "cand_mem"))]
    soe._write_gz(w[cols + regs].sort_values(["cell", "file"]), args.per_run)
    table.to_csv(args.summary, index=False)
    by_proc.to_csv(args.by_process, index=False)
    conf = cache[["file", "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "y_opt",
                  "d_row", "c_row", "d_rating_row", "c_rating_row", "n_ratings_d", "n_ratings_c", "f_d", "f_c", "e_d",
                  "e_c", "d_is_c"]].copy()
    for rho, dec in decisions.items():
        lab = mem_label(rho)
        d = dec.set_index("file")
        for c in ("regret", "p_value", "claim", "shipped_d", "fresh_mean_claim"):
            conf[f"{c}_{lab}"] = conf["file"].map(d[c])
    soe._write_gz(conf.sort_values("file"), args.confirm_per_run)
    pd.concat(claims.values(), ignore_index=True).to_csv(args.confirm_claims, index=False)
    text = "\n".join(lines) + "\n"
    args.register.parent.mkdir(parents=True, exist_ok=True)
    args.register.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")
    r = sub.add_parser("replay", help="refit the prefix GPs and write the candidates' original errors")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--resume", action="store_true", help="keep the runs already in the inputs files")
    r.add_argument("--procedures", default="sitting,confirm", help="comma-separated subset of sitting,confirm")
    r.add_argument("--sitting-out", type=Path, default=SITTING_INPUTS)
    r.add_argument("--confirm-out", type=Path, default=CONFIRM_INPUTS)
    r.add_argument("--limit", type=int, default=0, help="replay only the first N stems (a smoke test)")
    s = sub.add_parser("score", help="apply memory and score it (the default)")
    s.add_argument("--sitting-order-inputs", type=Path, default=soe.INPUTS)
    s.add_argument("--sitting-inputs", type=Path, default=SITTING_INPUTS)
    s.add_argument("--confirm-inputs", type=Path, default=CONFIRM_INPUTS)
    s.add_argument("--per-run", type=Path, default=SITTING_PER_RUN)
    s.add_argument("--summary", type=Path, default=SITTING_SUMMARY)
    s.add_argument("--by-process", type=Path, default=SITTING_BY_PROCESS)
    s.add_argument("--confirm-per-run", type=Path, default=CONFIRM_PER_RUN)
    s.add_argument("--confirm-claims", type=Path, default=CONFIRM_CLAIMS)
    s.add_argument("--register", type=Path, default=REGISTER)
    args = ap.parse_args(argv)
    if args.command == "replay":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        procs = [p.strip() for p in args.procedures.split(",") if p.strip()]
        if not procs or set(procs) - {"sitting", "confirm"}:
            raise SystemExit("--procedures takes a comma-separated subset of sitting,confirm")
        args.procedures = procs
        run_replay(args)
    else:
        if args.command is None:
            args = ap.parse_args(["score"] + list(argv or []))
        score(args)


if __name__ == "__main__":
    main()
