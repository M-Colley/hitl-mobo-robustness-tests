"""Order effects between the designs of a final comparative sitting (a post-hoc sensitivity analysis).

Every sitting number in the paper treats the looks of one sitting as independent
given the designs: look_j = f(x_j) + e_j, where e_j is the idiosyncratic part of
the run's rating error (replay_end_of_study.py, --sitting-process shared). The
Limitations section and app:endofstudy name order effects beyond the simulated
processes (contrast, anchoring between consecutive designs, fatigue beyond the
drift ramp) as untested. This check tests them on the two sittings the paper
reports in the cells where the sitting matters.

POST HOC, and outside every Holm family of the paper. The design below was fixed
before any result of this check was seen, but after the sitting's own results
were known. Each adjusted p printed here names its own family: the 20
(cell, effect, order) rows of this analysis, per seed set and estimand.

Design
------
Cells: the sitting of k = 2 at 1 sigma from the first rating (jitter_std 1.0,
jitter_iteration 0) and the sitting of k = 16 at 5 sigma from trial 21
(jitter_std 5.0, jitter_iteration 20). Main sweep (output-boba), LogEI and qNEI,
the error processes gaussian, bias, drift and ar1, seeds 7-16, every noisy run
and its clean twin (one clean twin per landscape, acquisition and seed, shared
by every cell and process).

Each run's published sitting is reproduced exactly with replay_end_of_study's
own functions: the prefix state search_state(run, T - k, bounds, beta = 1)
(the loop's GP refitted on trials 1..T-k with its torch seed), the candidates
(the top k distinct visited designs by posterior mean less one latent SD, in
rank order; m <= k of them when the prefix holds fewer distinct designs), their
true values truth_j, and the look errors e_j = rho * sd_j * z_j with rho = 1,
sd_j the shared model's SD of the run's process at trial T - k + 1 + j
(sitting_sd_fn) and z the tournament's own draw
default_rng(noise_seed(run, PROC_TOURNAMENT, k)).standard_normal(k). The
published look winner is argmax(truth + e). The replay stage checks every run
against replay_end_of_study.tournament in process; the score stage checks every
run against the published per-run tables (procedure tournament_k{k}_lcb_rho1_look
of end_of_study_ksweep for k = 2 and end_of_study_kwide for k = 16; regret of the
noisy run and of its clean twin, and the standard process of both), bit for bit,
and stops on any difference.

The m candidates are then presented in an order, and the look at presentation
position p = 0..m-1 (t_p, e_p: the true value and the look error of the design
shown there) is modified by one effect at a time:

    none          y_p = t_p + e_p                                         the published sitting
    anchoring a   y_0 = t_0 + e_0,  y_p = (1 - a)(t_p + e_p) + a y_{p-1}   a in {0.25, 0.5}
    contrast c    y_0 = t_0 + e_0,  y_p = t_p + e_p + c (t_p - t_{p-1})    c in {0.25, 0.5}
    fatigue f     y_p = t_p + e_p (1 + f p / (m - 1))                      f = 1 (m = 1: unchanged)

Orders: random, a permutation per run from
default_rng(SeedSequence([crc32(run name), k, 24601])) (noise_seed(run name, k,
ORDER_TAG); no existing draw uses that stream, so every published draw is
untouched); rank, the best-ranked candidate first. The design with the best
modified look ships (ties to the earliest position); 'none' does not depend on
the order, which the score stage checks run by run. The clean twin's rater is
exact (e = 0), but anchoring and contrast still act on the true values, so a
noiseless rater can still anchor; fatigue does nothing there. The clean twin's
random order is drawn from its own file name, so it is shared by the four
processes as the twin is.

Estimand (analyse_boba_adaptations.py)
--------------------------------------
Per run, in units of the landscape's achievable improvement opt_z
(analyse_boba_adaptations.dataset_scale('opt_z')), against the STANDARD process
(the run's own logged deployed regret at T, replay_end_of_study.standard_process):
cost = ref_noisy - ref_clean, gain = ref_noisy - trt_noisy, price = trt_clean -
ref_clean (aba.paired_frame, aba.summarise). Landscape means, then the mean over
the twenty landscapes; landscape bootstrap (2,000 resamples, seed
BOOTSTRAP_SEED; the landscapes share their random numbers, so the bootstrap
treats correlated clusters as independent); two-sided Wilcoxon over the twenty
per-landscape values. A share of the cost is withheld where the reference cost
is below 0.01 of opt_z or negative on any landscape (aba.share_flags). Paired
increments, per run: over the published sitting, (regret of 'none' - regret
with the effect) / opt_z, the change the effect makes; over LCB1 after all 50
trials with no extra trial (regret_lcb1 of ship_rules_per_run.csv, joined by
sitting_by_magnitude.attach_ship_rules, which checks that the rescoring
reproduces the standard process to 1e-9), (regret_lcb1 - regret with the
effect) / opt_z. Primary scope: seeds 12-16 (the paper's held-out seeds); also
seeds 7-16. Pooled over the four processes and both acquisitions; per-process
means are in sitting_order_effects_by_process.csv.

Two subcommands:

    python scripts/checks/sitting_order_effects.py replay --workers 2
    python scripts/checks/sitting_order_effects.py score > output-boba/analysis/review/register_checks/sitting_order_effects.txt

replay refits 4,000 prefix GPs (3,200 noisy runs, 400 clean twins at two
prefixes) from the git-ignored run logs and writes
review/sitting_order_effects_inputs.csv.gz (per run and clean twin: the
candidates' true values, look errors and random order, in rank order; gzip
without a timestamp, so a rerun is byte-identical; --resume keeps the stems
already there). The worker count changes no number. score reads that file, the
published per-run tables and ship_rules_per_run.csv, and writes
review/sitting_order_effects_per_run.csv.gz, review/sitting_order_effects.csv
and review/sitting_order_effects_by_process.csv.
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

ARM_DIR = REPO / "output-boba"
ANALYSIS = ARM_DIR / "analysis"
REVIEW = ANALYSIS / "review"
INPUTS = REVIEW / "sitting_order_effects_inputs.csv.gz"
PER_RUN = REVIEW / "sitting_order_effects_per_run.csv.gz"
SUMMARY = REVIEW / "sitting_order_effects.csv"
BY_PROCESS = REVIEW / "sitting_order_effects_by_process.csv"
SHIP_RULES = ANALYSIS / "ship_rules_per_run.csv"
# The published replays that hold each sitting size (shared sitting model).
PUBLISHED = {2: ANALYSIS / "end_of_study_ksweep", 16: ANALYSIS / "end_of_study_kwide"}

# cell label -> (jitter_std, jitter_iteration, sitting size k)
CELLS = {"1sigma_from_trial_1": (1.0, 0, 2), "5sigma_from_trial_21": (5.0, 20, 16)}
ACQS = sbm.ACQS
PROCESSES = sbm.PROCESSES
SEED_SETS = {"12-16": sbm.TEST_SEEDS, "7-16": sbm.TRAIN_SEEDS + sbm.TEST_SEEDS}
RHO = 1.0
LCB_BETA = 1.0
# The presentation order's own stream: SeedSequence([crc32(run name), k, ORDER_TAG]).
ORDER_TAG = 24601
EFFECTS = (("none", 0.0), ("anchoring", 0.25), ("anchoring", 0.5), ("contrast", 0.25), ("contrast", 0.5),
           ("fatigue", 1.0))
ORDERS = ("random", "rank")
BOOTSTRAP_SEED = 20261105
HOLM_FAMILY = "the 20 (cell, effect, order) rows of this analysis, per seed set and estimand"


class ControlError(RuntimeError):
    """The effect-free sitting does not reproduce the published sitting."""


def label(effect: str, level: float, order: str) -> str:
    return "none" if effect == "none" else f"{effect}{level:g}_{order}"


# Every (effect, level, order) scored; 'none' once, since the order cannot move it.
COMBOS = [("none", 0.0, "rank")] + [(e, lv, o) for e, lv in EFFECTS if e != "none" for o in ORDERS]


def procedure_name(k: int) -> str:
    """The published procedure: candidates by LCB1, rho = 1, the best look ships."""
    return f"tournament_k{int(k)}_lcb_rho{RHO:g}_look"


def rel(path: Path) -> str:
    """A path as printed into a register file: relative to the repository, else its name alone."""
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).name


def cell_of(jitter_std: float, jitter_iteration: int) -> str | None:
    for name, (sigma, onset, _) in CELLS.items():
        if np.isclose(float(jitter_std), sigma) and int(jitter_iteration) == onset:
            return name
    return None


# ---------------------------------------------------------------------------
# The effects (pure numpy; the tests pin each formula)
# ---------------------------------------------------------------------------


def presentation(order: str, perm_random: np.ndarray) -> np.ndarray:
    """Candidate (rank index) shown at each position: the run's random permutation, or rank order."""
    perm_random = np.asarray(perm_random, dtype=int)
    if order == "random":
        return perm_random
    if order == "rank":
        return np.arange(len(perm_random))
    raise ValueError(f"unknown order {order!r}; choose from {ORDERS}")


def presented_looks(truth: np.ndarray, error: np.ndarray, shown: np.ndarray, effect: str, level: float) -> np.ndarray:
    """The modified looks in presentation order; ``shown[p]`` is the candidate shown at position p."""
    shown = np.asarray(shown, dtype=int)
    t = np.asarray(truth, dtype=float)[shown]
    e = np.asarray(error, dtype=float)[shown]
    m = len(t)
    if effect == "none":
        return t + e
    if effect == "anchoring":
        a = float(level)
        y = np.empty(m)
        for p in range(m):
            y[p] = t[p] + e[p] if p == 0 else (1.0 - a) * (t[p] + e[p]) + a * y[p - 1]
        return y
    if effect == "contrast":
        y = t + e
        y[1:] = y[1:] + float(level) * (t[1:] - t[:-1])
        return y
    if effect == "fatigue":
        scale = 1.0 + float(level) * np.arange(m) / (m - 1) if m > 1 else np.ones(m)
        return t + e * scale
    raise ValueError(f"unknown effect {effect!r}")


def shipped(truth: np.ndarray, error: np.ndarray, shown: np.ndarray, effect: str, level: float) -> int:
    """The candidate (rank index) with the best modified look; ties to the earliest position."""
    y = presented_looks(truth, error, shown, effect, level)
    return int(np.asarray(shown, dtype=int)[int(np.argmax(y))])


def published_pick(truth: np.ndarray, error: np.ndarray) -> int:
    """The published sitting's look winner, as replay_end_of_study.tournament computes it (shared model)."""
    return int(np.argmax(np.asarray(truth, dtype=float) + np.asarray(error, dtype=float)))


def check_control(reproduced: pd.DataFrame, published: pd.DataFrame) -> dict:
    """Every run's effect-free sitting must reproduce the published one bit for bit.

    ``reproduced`` and ``published`` both hold file, k, regret_noisy, regret_clean,
    ref_noisy and ref_clean (raw regrets). Raises ControlError on a missing run or
    any difference; returns the counts compared.
    """
    cols = ["regret_noisy", "regret_clean", "ref_noisy", "ref_clean"]
    m = reproduced[["file", "k"] + cols].merge(published[["file", "k"] + cols], on=["file", "k"], how="left",
                                               suffixes=("", "_pub"), validate="one_to_one", indicator=True)
    missing = int((m["_merge"] != "both").sum())
    if missing:
        raise ControlError(f"{missing} reproduced runs have no published row, e.g. "
                           f"{m.loc[m['_merge'] != 'both', 'file'].iloc[0]}")
    worst = {}
    for c in cols:
        a, b = m[c].to_numpy(dtype=float), m[f"{c}_pub"].to_numpy(dtype=float)
        bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ControlError(f"{int(bad.sum())} runs differ from the published sitting in {c}, e.g. "
                               f"{m['file'].iloc[i]} (k = {m['k'].iloc[i]}): {a[i]!r} against {b[i]!r}")
        worst[c] = float(np.max(np.abs(a - b))) if len(a) else 0.0
    return {"n_runs": int(len(m)), "max_abs_difference": worst}


# ---------------------------------------------------------------------------
# replay: the published sitting's inputs, from the run logs
# ---------------------------------------------------------------------------


def _vec(values) -> str:
    """Space-separated shortest round-trip reprs, so a float survives the CSV bit for bit."""
    return " ".join(repr(float(v)) for v in values)


def _ivec(values) -> str:
    return " ".join(str(int(v)) for v in values)


def parse_vec(text) -> np.ndarray:
    return np.array([float(v) for v in str(text).split()], dtype=float)


def parse_ivec(text) -> np.ndarray:
    return np.array([int(v) for v in str(text).split()], dtype=int)


def sitting_inputs(run, state, k: int, sd_fn, T: int) -> dict:
    """Candidates, true values, look errors and random order of the published sitting of one run.

    The candidates and the looks are tournament()'s own expressions (shared model,
    rho = 1); the result is checked against tournament() itself.
    """
    import replay_end_of_study as eos

    if eos._uses_process(sd_fn):
        raise ValueError(f"{run.name}: only the shared sitting model is reproduced here")
    z = np.random.default_rng(eos.noise_seed(run.name, eos.PROC_TOURNAMENT, k)).standard_normal(k)
    sds = np.array([sd_fn(t) for t in range(T - k + 1, T + 1)], dtype=float)
    rows = state.first[np.argsort(-state.lcb, kind="stable")[:k]]
    m = len(rows)
    truth = run.deployed[rows]
    error = RHO * sds[:m] * z[:m]
    regret = run.y_opt - float(truth[published_pick(truth, error)])
    pub = eos.tournament(run, state, k, (RHO,), sd_fn, T)[procedure_name(k)]
    top = run.y_opt - float(truth[0])
    if regret != pub["regret"] or int(pub["n_candidates"]) != m or top != pub["top_candidate_regret"]:
        raise eos.ReproductionError(f"{run.name}, k = {k}: the sitting's inputs give regret {regret!r} on {m} "
                                    f"candidates, tournament() {pub['regret']!r} on {pub['n_candidates']}")
    perm = np.random.default_rng(eos.noise_seed(run.name, k, ORDER_TAG)).permutation(m)
    return {"n_candidates": m, "y_opt": float(run.y_opt), "published_regret": float(pub["regret"]),
            "top_candidate_regret": top, "truth": _vec(truth), "error": _vec(error), "perm_random": _ivec(perm),
            "gp_failed": bool(state.gp_failed)}


def replay_stem(task: dict) -> list[dict]:
    """The sitting inputs of one stem: its clean twin at each k the stem needs, and each noisy run of a cell."""
    import torch

    import boba_benchmarks as bb
    import replay_end_of_study as eos

    arm, settings = task["arm"], task["settings"]
    T = arm.iterations
    spec = bb.BENCHMARKS[task["dataset"]]
    bounds = torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)
    runs = [r for r in task["runs"] if cell_of(r["jitter_std"], r["jitter_iteration"])]
    ks = sorted({CELLS[cell_of(r["jitter_std"], r["jitter_iteration"])][2] for r in runs})
    entries = [{"path": task["clean_path"], "baseline": True, "error_model": "none", "jitter_std": 0.0,
                "jitter_iteration": 0, "variant": "", "cell": "", "k": k} for k in ks]
    entries += [{"path": r["path"], "baseline": False, "error_model": r["error_model"],
                 "jitter_std": float(r["jitter_std"]), "jitter_iteration": int(r["jitter_iteration"]),
                 "variant": str(r.get("variant") or ""),
                 "cell": cell_of(r["jitter_std"], r["jitter_iteration"]),
                 "k": CELLS[cell_of(r["jitter_std"], r["jitter_iteration"])][2]} for r in runs]
    out, logs = [], {}
    for e in entries:
        path = Path(e["path"])
        if path not in logs:
            logs[path] = eos.read_run(path, T)
        run = logs[path]
        k = int(e["k"])
        # The prefix the sitting starts from and the final trial must reproduce the loop's own record.
        eos.check_logged_inference(run, [T - k, T])
        sd_fn = eos.sitting_sd_fn(arm, e["error_model"], e["jitter_std"], e["jitter_iteration"], settings,
                                  variant=e["variant"] or None)
        state = eos.search_state(run, T - k, bounds, settings.lcb_beta)
        ref = eos.standard_process(run, arm.initial_samples)["regret"]
        out.append({"file": run.name, "dataset": task["dataset"], "acquisition": task["acquisition"],
                    "seed": int(task["seed"]), "baseline": e["baseline"], "error_model": e["error_model"],
                    "jitter_std": e["jitter_std"], "jitter_iteration": e["jitter_iteration"],
                    "variant": e["variant"], "cell": e["cell"], "k": k, "ref_regret": float(ref),
                    **sitting_inputs(run, state, k, sd_fn, T)})
    return out


def replay_tasks() -> tuple[list[dict], dict]:
    """replay_end_of_study's own stems: standard variants, LogEI and qNEI, seeds 7-16, both cells' runs."""
    import replay_end_of_study as eos

    arm = eos.load_arm(ARM_DIR, "none")
    settings = eos.Settings(tournament_k=tuple(sorted({k for _, _, k in CELLS.values()})), confirm_k=(),
                            rhos=(RHO,), lcb_beta=LCB_BETA)
    eos.validate_settings(settings)
    eos.validate_arm(arm, settings)
    filters = eos.Filters(acquisitions=frozenset(ACQS),
                          seeds=frozenset(int(s) for s in SEED_SETS["7-16"]),
                          error_models=frozenset(PROCESSES),
                          stds=tuple(sorted({s for s, _, _ in CELLS.values()})),
                          onsets=frozenset(o for _, o, _ in CELLS.values()))
    tasks, counts = eos.build_tasks(arm, filters, settings, Path("unused"), resume=False)
    for t in tasks:
        t["runs"] = [r for r in t["runs"] if cell_of(r["jitter_std"], r["jitter_iteration"])]
    tasks = [t for t in tasks if t["runs"]]
    return tasks, dict(counts)


def _write_gz(frame: pd.DataFrame, path: Path) -> None:
    """Atomically, gzip without a timestamp, so a rerun is byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
    tmp.replace(path)


def _write_inputs(rows: list[dict], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(["file", "k"]).reset_index(drop=True)
    _write_gz(frame, path)
    return frame


def run_replay(workers: int, out: Path, resume: bool) -> pd.DataFrame:
    import replay_end_of_study as eos

    start = time.perf_counter()
    tasks, counts = replay_tasks()
    rows: list[dict] = []
    if resume and out.is_file():
        old = load_inputs(out)
        rows = old.to_dict("records")
        done = set(zip(old["file"], old["k"]))

        def finished(t: dict) -> bool:
            names = {(Path(t["clean_path"]).name, CELLS[cell_of(r["jitter_std"], r["jitter_iteration"])][2])
                     for r in t["runs"]}
            names |= {(Path(r["path"]).name, CELLS[cell_of(r["jitter_std"], r["jitter_iteration"])][2])
                      for r in t["runs"]}
            return names <= done
        tasks = [t for t in tasks if not finished(t)]
    n_runs = sum(len(t["runs"]) for t in tasks)
    print(f"{ARM_DIR.name}: {len(tasks)} stems to replay, {n_runs} noisy runs in the two cells; "
          f"{len(rows)} rows kept from an earlier pass (build_tasks counts before the cell filter: {counts})",
          flush=True)
    if tasks:
        if workers <= 1:
            eos._worker_init()
            results = (replay_stem(t) for t in tasks)
            for i, res in enumerate(results, 1):
                rows += res
                if i % 25 == 0 or i == len(tasks):
                    _write_inputs(rows, out)
                    print(f"  {i}/{len(tasks)} stems, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=eos._worker_init) as pool:
                futures = {pool.submit(replay_stem, t): t["stem"] for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    rows += fut.result()
                    if i % 25 == 0 or i == len(tasks):
                        _write_inputs(rows, out)
                        rate = i / (time.perf_counter() - start)
                        print(f"  {i}/{len(tasks)} stems, {(time.perf_counter() - start) / 60:.1f} min, "
                              f"eta {(len(tasks) - i) / rate / 60:.1f} min", flush=True)
    frame = _write_inputs(rows, out)
    print(f"wrote {rel(out)}: {len(frame)} rows "
          f"({int((~frame['baseline']).sum())} noisy runs, {int(frame['baseline'].sum())} clean-twin prefixes); "
          f"GP fallbacks {int(frame['gp_failed'].sum())}; every run matched tournament() in process; "
          f"{(time.perf_counter() - start) / 60:.1f} min on {workers} worker(s)")
    return frame


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def load_inputs(path: Path = INPUTS) -> pd.DataFrame:
    d = pd.read_csv(path, float_precision="round_trip", low_memory=False,
                    dtype={"truth": str, "error": str, "perm_random": str, "variant": str, "cell": str})
    for col in ("baseline", "gp_failed"):
        d[col] = d[col].astype(str).str.lower().isin(("true", "1"))
    d["variant"] = d["variant"].fillna("").astype(str)
    d["cell"] = d["cell"].fillna("").astype(str)
    d["k"] = d["k"].astype(int)
    return d


def regrets(inputs: pd.DataFrame) -> pd.DataFrame:
    """Per input row: the shipped candidate and its regret under every (effect, order), and 'none' both ways."""
    out = []
    for r in inputs.itertuples(index=False):
        truth, error, perm = parse_vec(r.truth), parse_vec(r.error), parse_ivec(r.perm_random)
        if not (len(truth) == len(error) == len(perm) == int(r.n_candidates)):
            raise ValueError(f"{r.file}, k = {r.k}: candidate vectors of unequal length")
        row = {"file": r.file, "k": int(r.k)}
        pick = published_pick(truth, error)
        row["cand_published"], row["reg_published"] = pick, r.y_opt - float(truth[pick])
        for effect, level, order in COMBOS + [("none", 0.0, "random")]:
            c = shipped(truth, error, presentation(order, perm), effect, level)
            name = label(effect, level, order) + ("_random" if (effect, order) == ("none", "random") else "")
            row[f"cand_{name}"], row[f"reg_{name}"] = c, r.y_opt - float(truth[c])
        out.append(row)
    reg = pd.DataFrame(out)
    # 'none' is the published sitting under either order (a tie can only move the pick between equal values).
    for a, b in (("reg_none", "reg_published"), ("reg_none_random", "reg_published")):
        if not np.array_equal(reg[a].to_numpy(), reg[b].to_numpy()):
            raise ControlError(f"{a} differs from {b} on {int((reg[a] != reg[b]).sum())} rows")
    return reg


PUBLISHED_COLS = ["arm", "file", "procedure", "k", "regret_noisy", "regret_clean", "ref_noisy", "ref_clean"]


def read_published() -> pd.DataFrame:
    """The published sitting rows of end_of_study_per_run.csv.gz (k = 2 and k = 16), parsed exactly."""
    frames = []
    for k, d in PUBLISHED.items():
        p = pd.read_csv(d / "end_of_study_per_run.csv.gz", usecols=PUBLISHED_COLS, float_precision="round_trip",
                        low_memory=False)
        p = p[(p["arm"] == ARM_DIR.name) & (p["procedure"] == procedure_name(k))]
        frames.append(p.assign(k=int(k)))
    return pd.concat(frames, ignore_index=True)


def read_published_stems(files: pd.DataFrame) -> pd.DataFrame:
    """The same rows from the replay's own per-stem outputs (per_run/<arm>/<landscape>/<stem>.csv), parsed exactly.

    ``files`` holds file, dataset and k of the noisy runs. These files are what
    replay_stem wrote from the in-process floats; end_of_study_per_run.csv.gz was
    assembled from them by load_per_run, which re-reads them with pandas' default
    float parser (see csv_round_trip).
    """
    frames = []
    stems = files.assign(stem=files["file"].str.split("_jittered_", n=1).str[0])
    for (k, dataset, stem), _ in stems.groupby(["k", "dataset", "stem"]):
        path = PUBLISHED[int(k)] / "per_run" / ARM_DIR.name / dataset / f"{stem}.csv"
        p = pd.read_csv(path, usecols=PUBLISHED_COLS, float_precision="round_trip", low_memory=False)
        frames.append(p[p["procedure"] == procedure_name(k)].assign(k=int(k)))
    return pd.concat(frames, ignore_index=True)


def csv_round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` written as replay_stem writes a per-stem file and read back as load_per_run reads it.

    pandas' default ('high') float parser is not round-trip exact: it can move a
    value by an ulp (it reads '20.473721531015673' as 20.473721531015677), so the
    published end_of_study_per_run.csv.gz carries that rounding. Passing the
    reproduced values through the same write and read reproduces it exactly.
    """
    import io

    return pd.read_csv(io.StringIO(frame.to_csv(index=False)), low_memory=False)


def read_ship_rules() -> pd.DataFrame:
    cols = ["dataset", "acquisition", "seed", "baseline", "file", "regret_best_observed", "regret_pm",
            "regret_lcb1", "regret_lcb2"]
    s = pd.read_csv(SHIP_RULES, usecols=cols, float_precision="round_trip", low_memory=False)
    return s[s["acquisition"].isin(ACQS)].reset_index(drop=True)


def per_run_frame(inputs: pd.DataFrame, reg: pd.DataFrame, ship: pd.DataFrame,
                  opt_z: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """One row per noisy run: every regret of the run and of its clean twin, LCB1 of both, and the control."""
    d = inputs.drop(columns=["truth", "error", "perm_random"]).merge(reg, on=["file", "k"], validate="one_to_one")
    noisy, clean = d[~d["baseline"]], d[d["baseline"]]
    keys = ["dataset", "acquisition", "seed", "k"]
    keep = [c for c in clean.columns if c.startswith(("reg_", "cand_"))] + ["ref_regret", "file", "gp_failed"]
    w = noisy.merge(clean[keys + keep], on=keys, suffixes=("", "_clean"), how="left", validate="many_to_one")
    if w["file_clean"].isna().any():
        raise KeyError(f"{int(w['file_clean'].isna().sum())} noisy runs have no clean-twin sitting")
    rep = pd.DataFrame({"file": w["file"], "k": w["k"], "regret_noisy": w["reg_none"],
                        "regret_clean": w["reg_none_clean"], "ref_noisy": w["ref_regret"],
                        "ref_clean": w["ref_regret_clean"]})
    # Exact against the replay's own per-stem outputs, and exact against the published table once the
    # reproduced values take the same CSV round trip; the raw gap to that table is only parser rounding.
    gz = read_published()
    control = {"per_stem": check_control(rep, read_published_stems(w[["file", "dataset", "k"]])),
               "table": check_control(csv_round_trip(rep), gz)}
    raw = rep.merge(gz, on=["file", "k"], suffixes=("", "_pub"), validate="one_to_one")
    cols = ["regret_noisy", "regret_clean", "ref_noisy", "ref_clean"]
    gaps = np.column_stack([np.abs(raw[c].to_numpy() - raw[f"{c}_pub"].to_numpy()) for c in cols])
    control["raw_gap_to_table"] = {"max_abs": float(gaps.max()),
                                   "runs_differing": int((gaps > 0).any(axis=1).sum()),
                                   "max_rel": float(np.max(gaps / np.maximum(np.abs(
                                       np.column_stack([raw[c].to_numpy() for c in cols])), 1e-300)))}
    # LCB1 after all 50 trials on the same runs, joined and checked by sitting_by_magnitude's own join.
    z = w["dataset"].map(opt_z)
    if z.isna().any():
        raise KeyError(f"no opt_z for {sorted(w.loc[z.isna(), 'dataset'].unique())}")
    attach = pd.DataFrame({"file": w["file"], "dataset": w["dataset"], "acquisition": w["acquisition"],
                           "seed": w["seed"], "opt_z": z, "regret_noisy": w["reg_none"],
                           "regret_clean": w["reg_none_clean"], "ref_noisy": w["ref_regret"],
                           "ref_clean": w["ref_regret_clean"], "top_candidate_regret_noisy": w["top_candidate_regret"]})
    m = sbm.attach_ship_rules(attach, ship)[["file", "regret_lcb1", "regret_lcb1_clean"]]
    w = w.assign(opt_z=z.to_numpy()).merge(m, on="file", how="left", validate="one_to_one")
    if w[["regret_lcb1", "regret_lcb1_clean"]].isna().any().any():
        raise KeyError("a noisy run or its clean twin has no LCB1 rescoring")
    return w.reset_index(drop=True), control


def scaled_block(w: pd.DataFrame, name: str, opt_z: dict[str, float]) -> pd.DataFrame:
    """aba.paired_frame of the standard process (ref) and the sitting with one effect (trt), plus increments."""
    resp = aba.RESPONSES["deployed"]
    keys = aba.PAIR_KEYS
    ref = w[keys].assign(**{f"{resp}_jitter": w["ref_regret"].to_numpy(),
                            f"{resp}_baseline": w["ref_regret_clean"].to_numpy()})
    trt = w[keys].assign(**{f"{resp}_jitter": w[f"reg_{name}"].to_numpy(),
                            f"{resp}_baseline": w[f"reg_{name}_clean"].to_numpy()})
    pf = aba.paired_frame(ref, trt, resp, opt_z, pool=False)
    # dict.fromkeys: 'none' names its own candidate columns twice
    extra = w[keys + list(dict.fromkeys(["file", "cell", "k", "reg_none", "reg_none_clean", "regret_lcb1",
                                         "regret_lcb1_clean", "cand_none", f"cand_{name}", "cand_none_clean",
                                         f"cand_{name}_clean"]))]
    pf = pf.merge(extra, on=keys, validate="one_to_one")
    z = pf["dataset"].map(opt_z).astype(float)
    return pf.assign(
        cost=pf["ref_noisy"] - pf["ref_clean"], gain=pf["ref_noisy"] - pf["trt_noisy"],
        price=pf["trt_clean"] - pf["ref_clean"],
        inc_published=pf["reg_none"] / z - pf["trt_noisy"],
        inc_lcb1=pf["regret_lcb1"] / z - pf["trt_noisy"],
        price_over_published=pf["trt_clean"] - pf["reg_none_clean"] / z,
        gain_lcb1=pf["ref_noisy"] - pf["regret_lcb1"] / z,
        price_lcb1=pf["regret_lcb1_clean"] / z - pf["ref_clean"],
        changed=(pf[f"cand_{name}"] != pf["cand_none"]).astype(float),
        changed_clean=(pf[f"cand_{name}_clean"] != pf["cand_none_clean"]).astype(float),
    )


def summarise_block(pf: pd.DataFrame, rng: np.random.Generator) -> dict:
    """The estimand of analyse_boba_adaptations on one block, the increments, and the share guard."""
    s = aba.summarise(pf, rng)
    flags = aba.share_flags(pf)
    per = pf.groupby("dataset")[["cost", "gain", "price", "inc_published", "inc_lcb1", "gain_lcb1", "price_lcb1",
                                 "price_over_published", "changed"]].mean()
    out = {"n_runs": int(len(pf)), "n_landscapes": int(len(per)), "cost": s["cost"]}
    for col, name in (("gain", "gain"), ("price", "price"), ("inc_published", "inc_published"),
                      ("inc_lcb1", "inc_lcb1")):
        ci = sbm.mean_ci(per[col], rng)
        out.update({name: ci["mean"], f"{name}_lo": ci["lo"], f"{name}_hi": ci["hi"]})
        if col != "price":
            out[f"{name}_landscapes_positive"] = int((per[col] > 0).sum())
            out[f"{name}_p"] = sbm.wilcoxon_p(per[col].to_numpy())
    if abs(out["gain"] - s["gain"]) > 1e-12 or abs(out["price"] - s["price"]) > 1e-12:
        raise AssertionError("the gain or price differs from analyse_boba_adaptations.summarise")
    suppressed = flags["recovered_suppressed"]
    out.update({"share": np.nan if suppressed else s["recovered"],
                "share_lo": np.nan if suppressed else s["recovered_lo"],
                "share_hi": np.nan if suppressed else s["recovered_hi"], "share_suppressed": suppressed,
                "lcb1_gain": float(per["gain_lcb1"].mean()), "lcb1_price": float(per["price_lcb1"].mean()),
                "price_over_published": float(per["price_over_published"].mean()),
                "changed_share_noisy": float(pf["changed"].mean()),
                "changed_share_clean": float(pf.drop_duplicates(["dataset", "acquisition", "seed"])["changed_clean"]
                                             .mean())})
    return out


def score_all(w: pd.DataFrame, opt_z: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, by_proc = [], []
    for seeds_name, seeds in SEED_SETS.items():
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for cell, (sigma, onset, k) in CELLS.items():
            sub = w[(w["cell"] == cell) & w["seed"].isin(seeds)]
            for effect, level, order in COMBOS:
                name = label(effect, level, order)
                pf = scaled_block(sub, name, opt_z)
                base = {"seeds": seeds_name, "cell": cell, "k": k, "effect": effect, "level": level,
                        "order": "-" if effect == "none" else order}
                rows.append({**base, **summarise_block(pf, rng)})
                for proc, g in pf.groupby("error_model"):
                    per = g.groupby("dataset")[["cost", "gain", "price", "inc_published", "inc_lcb1",
                                                "gain_lcb1"]].mean()
                    by_proc.append({**base, "error_model": proc, "n_runs": int(len(g)),
                                    **{c: float(per[c].mean()) for c in per.columns},
                                    "changed_share_noisy": float(g["changed"].mean())})
    table = pd.DataFrame(rows)
    for col in ("gain", "inc_published", "inc_lcb1"):
        table[f"{col}_p_holm20"] = np.nan
        for seeds_name in SEED_SETS:
            fam = (table["seeds"] == seeds_name) & (table["effect"] != "none")
            table.loc[fam, f"{col}_p_holm20"] = sbm.holm(table.loc[fam, f"{col}_p"].to_numpy())
    return table, pd.DataFrame(by_proc)


def _ci(v: float, lo: float, hi: float) -> str:
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _p(p: float) -> str:
    return "  nan" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}")


def _share(r) -> str:
    if r["share_suppressed"]:
        return "withheld"
    return f"{r['share']:+.0%} [{r['share_lo']:+.0%}, {r['share_hi']:+.0%}]"


def print_tables(table: pd.DataFrame, by_proc: pd.DataFrame) -> None:
    for seeds_name in SEED_SETS:
        scope = "held-out seeds 12-16" if seeds_name == "12-16" else "seeds 7-16"
        for cell, (sigma, onset, k) in CELLS.items():
            t = table[(table["seeds"] == seeds_name) & (table["cell"] == cell)]
            r0 = t.iloc[0]
            print(f"\n=== {cell.replace('_', ' ')}, sitting of k = {k}; {scope}; LogEI and qNEI, four processes pooled; "
                  f"{int(r0['n_runs'])} noisy runs on {int(r0['n_landscapes'])} landscapes ===")
            print(f"cost of error of the standard process {r0['cost']:.3f} of opt_z; LCB1 after 50 trials gains "
                  f"{r0['lcb1_gain']:+.3f} at a price of {r0['lcb1_price']:.3f}")
            print(f"{'effect':<15s}{'order':<7s}{'gain over standard':>27s}{'gain':>5s}{'share of cost':>21s}"
                  f"{'price':>25s}{'over published sitting':>27s}{'ahd':>4s}{'p':>6s}{'Holm':>6s}"
                  f"{'over LCB1':>27s}{'ahd':>4s}{'p':>6s}{'Holm':>6s}{'moved':>7s}")
            for _, r in t.iterrows():
                eff = "none" if r["effect"] == "none" else f"{r['effect']} {r['level']:g}"
                print(f"{eff:<15s}{r['order']:<7s}{_ci(r['gain'], r['gain_lo'], r['gain_hi']):>27s}"
                      f"{int(r['gain_landscapes_positive']):>5d}{_share(r):>21s}"
                      f"{_ci(r['price'], r['price_lo'], r['price_hi']):>25s}"
                      f"{_ci(r['inc_published'], r['inc_published_lo'], r['inc_published_hi']):>27s}"
                      f"{int(r['inc_published_landscapes_positive']):>4d}{_p(r['inc_published_p']):>6s}"
                      f"{_p(r['inc_published_p_holm20']):>6s}"
                      f"{_ci(r['inc_lcb1'], r['inc_lcb1_lo'], r['inc_lcb1_hi']):>27s}"
                      f"{int(r['inc_lcb1_landscapes_positive']):>4d}{_p(r['inc_lcb1_p']):>6s}"
                      f"{_p(r['inc_lcb1_p_holm20']):>6s}{r['changed_share_noisy']:>7.1%}")
    print("\nBy error process, held-out seeds 12-16 (landscape means, then the mean over landscapes; no intervals):")
    bp = by_proc[by_proc["seeds"] == "12-16"].copy()
    bp["row"] = bp.apply(lambda r: "none" if r["effect"] == "none"
                         else f"{r['effect']} {r['level']:g} {r['order']}", axis=1)
    for cell in CELLS:
        b = bp[bp["cell"] == cell]
        for col, what in (("gain", "gain over the standard process"),
                          ("inc_published", "increment over the published sitting"),
                          ("inc_lcb1", "increment over LCB1")):
            piv = b.pivot_table(index="row", columns="error_model", values=col, aggfunc="first", sort=False)
            print(f"\n{cell}, {what}:")
            print(piv[[p for p in PROCESSES if p in piv.columns]].round(3).to_string())


def score(args) -> None:
    opt_z = aba.dataset_scale("opt_z")
    inputs = load_inputs(args.inputs)
    reg = regrets(inputs)
    w, control = per_run_frame(inputs, reg, read_ship_rules(), opt_z)
    expected = len(ACQS) * len(PROCESSES) * len(SEED_SETS["7-16"])
    counts = w.groupby(["cell", "dataset"]).size()
    if (counts != expected).any() or w["dataset"].nunique() != 20 or set(w["cell"]) != set(CELLS):
        raise SystemExit(f"incomplete inputs: expected {expected} noisy runs per cell and landscape on 20 "
                         f"landscapes, found {counts.describe().to_dict()}")
    table, by_proc = score_all(w, opt_z)
    REVIEW.mkdir(parents=True, exist_ok=True)
    cols = ["file", "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "variant",
            "cell", "k", "n_candidates", "opt_z", "ref_regret", "ref_regret_clean", "regret_lcb1",
            "regret_lcb1_clean", "file_clean"]
    regs = [c for c in w.columns if c.startswith(("reg_", "cand_"))]
    _write_gz(w[cols + regs].sort_values(["cell", "file"]), args.per_run)
    table.to_csv(args.summary, index=False)
    by_proc.to_csv(args.by_process, index=False)

    print(__doc__.split("\n\n")[0])
    print("\nPOST HOC: the design was fixed before any result of this check, after the sitting's own results were")
    print("known. Outside every Holm family of the paper; each adjusted p below is Holm over " + HOLM_FAMILY + ".")
    raw = control["raw_gap_to_table"]
    print(f"\nControl: every one of {control['per_stem']['n_runs']} noisy runs (and its clean twin) reproduces the "
          f"published sitting ({procedure_name(2)} of {PUBLISHED[2].name}, {procedure_name(16)} of "
          f"{PUBLISHED[16].name}) bit for bit, in the regret of the noisy run and of its clean twin and the standard "
          f"process of both: (1) against the replay's own per-stem outputs (per_run/{ARM_DIR.name}/<landscape>/"
          f"<stem>.csv), largest difference {max(control['per_stem']['max_abs_difference'].values()):.1e}; "
          f"(2) against end_of_study_per_run.csv.gz once the reproduced values take the replay's own CSV round "
          f"trip (load_per_run re-reads the per-stem files with pandas' default float parser, which is not "
          f"round-trip exact), largest difference {max(control['table']['max_abs_difference'].values()):.1e}. "
          f"Without that round trip {raw['runs_differing']} runs differ from the table by at most "
          f"{raw['max_abs']:.1e} ({raw['max_rel']:.1e} relative), the parser's rounding of the published table, "
          f"not a different sitting. 'none' under a random order ships the same regret as under rank order in "
          f"every run and clean twin.")
    print(f"Inputs: {len(inputs)} rows ({int((~inputs['baseline']).sum())} noisy runs, "
          f"{int(inputs['baseline'].sum())} clean-twin prefixes); GP fallbacks {int(inputs['gp_failed'].sum())}; "
          f"sittings short of k looks: " + ", ".join(
              f"k = {k}: {float((inputs[(inputs['k'] == k) & ~inputs['baseline']]['n_candidates'] < k).mean()):.1%}"
              for k in sorted(inputs["k"].unique())))
    print("\nUnits: shares of the achievable improvement opt_z, deployed design at T = 50. gain = standard process less "
          "the sitting with the effect (noisy run); price = the sitting with the effect less the standard process on "
          "the clean twin; 'over published sitting' = regret of the published (effect-free) sitting less the regret "
          "with the effect, per run (negative: the effect costs the sitting); 'over LCB1' = regret of LCB1 after all "
          "50 trials less the regret with the effect. [95% landscape bootstrap, 2,000 resamples; the landscapes "
          "share their random numbers, so it treats correlated clusters as independent]; gain/ahd = landscapes of "
          "20 with a positive value; p = two-sided Wilcoxon over the 20 landscape means; Holm = " + HOLM_FAMILY +
          "; share of cost = gain over the standard process's cost of error, a ratio of landscape means, withheld "
          "where that cost is below 0.01 or negative on a landscape; moved = share of noisy runs whose shipped "
          "design differs from the published sitting's.")
    print_tables(table, by_proc)
    print("\nPrice of each effect over the published sitting's price (clean twins, opt_z units; the clean rater is "
          "exact, so only anchoring and contrast can move it), and the share of clean twins whose shipped design moves:")
    t = table[(table["effect"] != "none")]
    for _, r in t.iterrows():
        print(f"  seeds {r['seeds']:<6s} {r['cell']:<22s} {r['effect']:>9s} {r['level']:<5g} {r['order']:<7s} "
              f"{r['price_over_published']:+.4f}   moved {r['changed_share_clean']:.1%}")
    print("\nStructural notes (algebra, not findings). With two designs, anchoring cannot reverse the comparison: "
          "y_1 = (1 - a)(t_1 + e_1) + a y_0 exceeds y_0 exactly when t_1 + e_1 does, so the sitting of two is "
          "unchanged by it in every run. With two designs, contrast ships the second design when "
          "(1 + c)(t_1 - t_0) > e_0 - e_1 in either order: it scales the true difference by 1 + c, so as specified "
          "(on the true values, not on the previous rating) it can only sharpen a comparison of two. With more "
          "designs it also favours a design shown just after a poor one, which is what its price on the clean "
          "twins measures. Fatigue scales only the look error, so it is inert on a clean twin and its price is "
          "the published sitting's.")
    print(f"\nWrote {', '.join(rel(p) for p in (args.per_run, args.summary, args.by_process))}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")
    r = sub.add_parser("replay", help="refit the prefix GPs and write the sitting inputs")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--resume", action="store_true", help="keep the stems already in the inputs file")
    r.add_argument("--out", type=Path, default=INPUTS)
    s = sub.add_parser("score", help="apply the order effects and score them (the default)")
    s.add_argument("--inputs", type=Path, default=INPUTS)
    s.add_argument("--per-run", type=Path, default=PER_RUN)
    s.add_argument("--summary", type=Path, default=SUMMARY)
    s.add_argument("--by-process", type=Path, default=BY_PROCESS)
    args = ap.parse_args(argv)
    if args.command == "replay":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        run_replay(args.workers, args.out, args.resume)
    else:
        if args.command is None:
            args = ap.parse_args(["score"] + list(argv or []))
        score(args)


if __name__ == "__main__":
    main()
