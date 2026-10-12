"""Final sittings gated on the run's own noise estimate, and the sitting of sixteen on the untouched seeds.

Both analyses are post hoc. They were built after the per-cell results of
Section 7 were known (sitting_by_magnitude.py on seeds 7-16) and after a
prototype had looked at every seed set, including the untouched seeds 27-36.
None of their tests belongs to a Holm family of the paper; every block names the
family its adjusted p is taken over.

Three subcommands.

refit
    For every LogEI and qNEI run of the main sweep (output-boba, seeds 7-16, the
    four response processes, and the clean twins) and of the replication arm
    (output-boba-confirmatory, seeds 27-36, gaussian, and the clean twins), refit
    the loop's GP on trials 1..T-k for the two gated sizes k = 16 and k = 2, with
    replay_end_of_study.search_state itself (the same data, transforms and torch
    seed, torch_seed(run name, T - k)), so the gate reads the very GP that ranks
    the sitting's candidates. Per run and prefix it records the fitted noise SD in
    the objective's units, sqrt(gp.likelihood.noise * outcome_transform.stdvs ** 2)
    (the likelihood noise is the variance of the standardised targets), its
    scale-free version sqrt(gp.likelihood.noise) (the noise SD over the SD of the
    ratings the GP standardised by), and the regret of the top LCB1 candidate,
    which must reproduce the replay's top_candidate_regret to 1e-9 (checked by
    'score'). Every prefix also re-derives the logged deployed regret
    (replay_end_of_study.check_logged_inference). Writes
    output-boba/analysis/review/gated_sitting_noise.csv.gz (gzip without a
    timestamp, so a rerun is byte-identical; --resume keeps the runs already
    there); the run logs it reads are git-ignored and live on the simulation
    machine. About 12 minutes on 8 workers; the worker count changes no number.

score (the default)
    The gated sittings, scored with the estimand of analyse_boba_adaptations.py
    (per run, in units of the landscape's achievable improvement opt_z:
    cost = ref_noisy - ref_clean, gain = ref_noisy - trt_noisy, price =
    trt_clean - ref_clean; landscape means; landscape bootstrap; two-sided
    Wilcoxon over the twenty per-landscape values):

    G16     at trial 34 = T - 16, if the GP's noise SD on trials 1..34 exceeds
            tau16, spend trials 35..50 on a sitting of sixteen (candidates by the
            posterior mean less one latent SD, each shown once, the best look
            ships, looks at the full idiosyncratic SD, the shared sitting:
            end_of_study_kwide, k = 16, rho 1); otherwise search on and ship the
            best rating.
    G2      the same at trial 48 with a sitting of two (end_of_study_ksweep, k = 2).
    G16+G2  both gates in sequence with their own thresholds: G16 at trial 34;
            if it does not fire, G2 at trial 48. No new choice is made for it.

    k is not chosen here: 16 and 2 are the sizes sitting_by_magnitude.py chooses
    on seeds 7-11 in the two cells where the sitting gains most (5 sigma from
    trial 21 and 1 sigma from trial 1). A run whose gate does not fire is the
    logged standard process exactly; a run whose gate fires is the replayed
    sitting, whose prefix is the logged run. The clean twin is gated by its own
    prefix fit, so a gate that fires without error is the price.

    The threshold rule, fixed before seeds 12-16 and 27-36 were scored by this
    script: tau is the value of TAU_GRID (0 = always sit, inf = never) that
    maximises, on seeds 7-11 (LogEI and qNEI, the four processes pooled), the
    mean of the gated sitting's gain over the standard process across the eight
    (magnitude, onset) cells and the error-free case, where the gain is minus
    the price: V(tau) = (sum over cells of the landscape-mean gain - price) / 9.
    Ties go to the larger tau (fewer sittings). The thresholds are then scored
    on seeds 12-16 (held out, four processes pooled) and 27-36 (untouched,
    gaussian error only; G16 reads the k = 16 replay of 'fresh-k16'). Per cell:
    the gain, its share of the cost of error (withheld where the cost is below
    0.01 or negative on a landscape, sitting_by_magnitude.share_suppressed), the
    increments over LCB1 and LCB2 after all 50 trials on the same runs
    (sitting_by_magnitude.attach_ship_rules; their prices are 0.020-0.023), the
    gated sitting's increment over the ungated one, the gate's firing rate, and
    the price. Holm families: the eight cells of one procedure ('Holm over the
    cells') and the 24 (procedure, cell) tests of the three gated procedures in
    one scope ('Holm over the 24'), separately for each estimand.

    The threshold is in landscape SDs: the objective of every landscape is
    standardised (z units), so a study needs the spread of its objective over
    the design space to set tau on its own rating scale. The scale-free gate
    (the noise SD over the SD of the ratings in hand, FRAC_GRID) is chosen by the
    same rule and scored as a comparison. The sensitivity over every tau of the
    grid is printed for all three seed sets.

fresh-k16
    The sitting of sixteen on the untouched seeds 27-36, which the paper could
    not test (sizes above twelve were never replayed there). The replay:

        python scripts/replay_end_of_study.py --arms output-boba-confirmatory
            --output-dir output-boba-confirmatory/analysis/end_of_study_fresh_k16
            --acquisitions logei,qnei,ucb --seeds 27,28,29,30,31,32,33,34,35,36
            --error-models gaussian --stds 0.05,0.25,1.0,5.0 --onsets 0,20
            --tournament-k 16 --confirm-k "" --rho 1 --rerate-dirs none --workers 8

    scored with fresh_seed_replication's own functions (reference_table,
    rule_increments, summarise) exactly as it scores k = 2, 5 and 12: the gain
    over the standard process, the increments over LCB1, LCB2 and PM after all 50
    trials, the clean-twin price, Wilcoxon p, with Holm over the eight cells at
    k = 16, over the four sizes 2, 5, 12, 16 within a cell, and over the 32
    (cell, k). Also the sitting of sixteen over the sitting of twelve on the same
    runs (the size the paper's untouched test used in its place), and the main
    sweep's gaussian-only k = 16 gain on seeds 12-16 beside it. Writes
    review/gated_sitting_fresh_k16.csv.

    python scripts/checks/gated_sitting.py refit --workers 8
    python scripts/checks/gated_sitting.py fresh-k16 > output-boba/analysis/review/register_checks/fresh_k16.txt
    python scripts/checks/gated_sitting.py score > output-boba/analysis/review/register_checks/gated_sitting.txt
"""
from __future__ import annotations

import argparse
import json
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
import sitting_by_magnitude as sbm  # noqa: E402

T = 50
# gate name -> sitting size k; the gate is evaluated at trial T - k on the GP fitted to trials 1..T-k
GATES = {"G16": 16, "G2": 2}
TWO_STAGE = "G16+G2"
PROCEDURES = ("G16", "G2", TWO_STAGE)
# The threshold grids, in landscape SDs (noise_sd) and as a share of the ratings' SD (noise_frac).
TAU_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, np.inf)
FRAC_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf)
STATS = {"noise_sd": TAU_GRID, "noise_frac": FRAC_GRID}
TRAIN_SEEDS, TEST_SEEDS = sbm.TRAIN_SEEDS, sbm.TEST_SEEDS
UNTOUCHED_SEEDS = tuple(range(27, 37))
ACQS = sbm.ACQS
PROCESSES = sbm.PROCESSES
MAIN_ARM, FRESH_ARM = "output-boba", "output-boba-confirmatory"
MAIN_ANALYSIS = REPO / MAIN_ARM / "analysis"
FRESH_ANALYSIS = REPO / FRESH_ARM / "analysis"
REVIEW = MAIN_ANALYSIS / "review"
NOISE_CACHE = REVIEW / "gated_sitting_noise.csv.gz"
STATS_PATH = REPO / "boba_landscape_stats.json"
# Which replay holds which sitting size, per arm (the main sweep's k-sweep and k-wide; the untouched
# seeds' fresh replay and its k = 16 companion).
REPLAYS = {MAIN_ARM: ("end_of_study_ksweep", "end_of_study_kwide"),
           FRESH_ARM: ("end_of_study_fresh", "end_of_study_fresh_k16")}
SHIP_RULES_CSV = {MAIN_ARM: MAIN_ANALYSIS / "ship_rules_per_run.csv",
                  FRESH_ARM: FRESH_ANALYSIS / "ship_rules_fresh" / "ship_rules_per_run.csv"}
REFIT_SCOPE = {MAIN_ARM: (TRAIN_SEEDS + TEST_SEEDS, PROCESSES), FRESH_ARM: (UNTOUCHED_SEEDS, ("gaussian",))}
SCORE_SEED = 20261009
TOP_TOL = 1e-9
CELLS = [(s, o) for s in (0.05, 0.25, 1.0, 5.0) for o in (0, 20)]


def cell_label(sigma: float, onset: int) -> str:
    return f"{sigma:g}sigma_from_trial_{int(onset) + 1}"


def rel(path: Path) -> str:
    """A path as printed into a register file: relative to the repository, never the local home directory."""
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).name


def json_safe(obj):
    """Infinite thresholds as the string 'inf', so the JSON is standard."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return "inf" if obj > 0 else ("-inf" if obj < 0 else "nan")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# refit: the GP behind each gate, from the run logs
# ---------------------------------------------------------------------------


def gp_noise(gp) -> tuple[float, float, float]:
    """(noise SD in objective units, scale-free noise SD, SD of the ratings) of a fitted SingleTaskGP.

    gp.likelihood.noise is the noise VARIANCE of the standardised targets
    (AGENTS.md), so the objective-unit SD is sqrt(noise * stdvs ** 2).
    """
    noise_var = float(gp.likelihood.noise.detach().reshape(-1)[0])
    stdvs = float(gp.outcome_transform.stdvs.detach().reshape(-1)[0])
    frac = float(np.sqrt(noise_var))
    return frac * stdvs, frac, stdvs


def refit_stem(task: dict, ks: tuple[int, ...] = tuple(GATES.values())) -> list[dict]:
    """The gate statistics of one stem: its clean twin and every noisy run, at n = T - k for each k."""
    import torch

    import boba_benchmarks as bb
    import replay_end_of_study as eos

    arm = task["arm"]
    spec = bb.BENCHMARKS[task["dataset"]]
    bounds = torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)
    ns = sorted({arm.iterations - int(k) for k in ks})
    entries = [{"path": task["clean_path"], "baseline": True, "error_model": "none", "jitter_std": 0.0,
                "jitter_iteration": np.nan, "variant": ""}]
    entries += [{"path": r["path"], "baseline": False, "error_model": r["error_model"],
                 "jitter_std": float(r["jitter_std"]), "jitter_iteration": int(r["jitter_iteration"]),
                 "variant": str(r.get("variant") or "")} for r in task["runs"]]
    rows = []
    for e in entries:
        run = eos.read_run(Path(e["path"]), arm.iterations)
        # Every prefix the gate reads must reproduce the loop's own record.
        eos.check_logged_inference(run, ns + [arm.iterations])
        for n in ns:
            state = eos.search_state(run, n, bounds, 1.0)
            if state.gp is None:
                noise_sd = noise_frac = stdvs = np.nan
            else:
                noise_sd, noise_frac, stdvs = gp_noise(state.gp)
            top = int(state.first[int(np.argmax(state.lcb))])
            rows.append({"arm": arm.name, "file": run.name, "dataset": task["dataset"],
                         "acquisition": task["acquisition"], "seed": int(task["seed"]),
                         "baseline": e["baseline"], "error_model": e["error_model"], "jitter_std": e["jitter_std"],
                         "jitter_iteration": e["jitter_iteration"], "variant": e["variant"], "n": int(n),
                         "k": int(arm.iterations - n), "noise_sd": noise_sd, "noise_frac": noise_frac,
                         "stdvs": stdvs, "top_regret": run.y_opt - float(run.deployed[top]),
                         "gp_failed": bool(state.gp_failed)})
    return rows


def refit_tasks(arm_dir: Path, seeds, processes, acqs=ACQS) -> tuple[list[dict], dict]:
    """replay_end_of_study's own stems (standard variants, clean twin found), so the runs are the replay's."""
    import replay_end_of_study as eos

    arm = eos.load_arm(arm_dir, "none")
    filters = eos.Filters(acquisitions=frozenset(acqs), seeds=frozenset(int(s) for s in seeds),
                          error_models=frozenset(processes))
    settings = eos.Settings(tournament_k=tuple(GATES.values()), confirm_k=())
    tasks, counts = eos.build_tasks(arm, filters, settings, Path("unused"), resume=False)
    return tasks, dict(counts)


def _write_cache(rows: list[dict], path: Path) -> pd.DataFrame:
    """The refit cache, sorted, written atomically; gzip without a timestamp, so a rerun is byte-identical."""
    frame = pd.DataFrame(rows).sort_values(["arm", "file", "n"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
    tmp.replace(path)
    return frame


def run_refit(workers: int, out: Path, resume: bool, arms=(MAIN_ARM, FRESH_ARM)) -> pd.DataFrame:
    import replay_end_of_study as eos

    done_rows: list[dict] = []
    done = set()
    if resume and out.is_file():
        old = read_noise(out)
        old["variant"] = old["variant"].fillna("").astype(str)
        done_rows = old.to_dict("records")
        done = set(zip(old["arm"], old["file"]))
    tasks = []
    for name in arms:
        seeds, processes = REFIT_SCOPE[name]
        arm_tasks, counts = refit_tasks(REPO / name, seeds, processes)
        todo = [t for t in arm_tasks
                if not ({(name, Path(t["clean_path"]).name)} | {(name, Path(r["path"]).name) for r in t["runs"]})
                <= done]
        print(f"{name}: {len(arm_tasks)} stems ({counts}); {len(arm_tasks) - len(todo)} already refitted", flush=True)
        tasks += todo
    rows = list(done_rows)
    start = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, initializer=eos._worker_init) as pool:
            futures = {pool.submit(refit_stem, t): t["stem"] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                rows += fut.result()
                if i % 50 == 0 or i == len(tasks):
                    _write_cache(rows, out)
                    rate = i / (time.perf_counter() - start)
                    print(f"  {i}/{len(tasks)} stems, {rate:.2f}/s, eta {(len(tasks) - i) / rate / 60:.1f} min",
                          flush=True)
    frame = _write_cache(rows, out)
    print(f"wrote {rel(out)} ({len(frame)} rows; GP fallbacks {int(frame['gp_failed'].astype(bool).sum())})")
    return frame


# ---------------------------------------------------------------------------
# score: the per-run frame of one arm
# ---------------------------------------------------------------------------


def read_noise(path: Path = NOISE_CACHE) -> pd.DataFrame:
    noise = pd.read_csv(path, float_precision="round_trip", low_memory=False)
    noise["baseline"] = noise["baseline"].astype(str).str.lower().isin(("true", "1"))
    return noise


def attach_noise(frame: pd.DataFrame, noise: pd.DataFrame, arm: str, ks=tuple(GATES.values())) -> pd.DataFrame:
    """Add each noisy run's and its clean twin's gate statistics at n = T - k (noise_sd_<n>, ..._clean).

    Every run and clean twin of ``frame`` must have its refit, and the refit's top LCB1 candidate must
    reproduce the replay's top_candidate_regret (top<k>, top<k>_clean) to TOP_TOL: the gate reads the
    GP that ranks the sitting's candidates, or the analysis stops.
    """
    nz = noise[noise["arm"] == arm]
    out = frame.copy()
    for k in ks:
        n = T - int(k)
        at = nz[nz["n"] == n]
        noisy = at[~at["baseline"]].set_index("file")
        clean = at[at["baseline"]].set_index(["dataset", "acquisition", "seed"])
        key = pd.MultiIndex.from_frame(out[["dataset", "acquisition", "seed"]])
        for stat in ("noise_sd", "noise_frac", "top_regret", "gp_failed"):
            out[f"{stat}_{n}"] = out["file"].map(noisy[stat])
            out[f"{stat}_{n}_clean"] = clean[stat].reindex(key).to_numpy()
        missing = out[f"top_regret_{n}"].isna() | out[f"top_regret_{n}_clean"].isna()
        if missing.any():
            raise KeyError(f"{int(missing.sum())} runs of {arm} have no refit at n = {n} "
                           f"(e.g. {out.loc[missing, 'file'].iloc[0]}); run 'refit' first")
        for sfx in ("", "_clean"):
            gap = (out[f"top_regret_{n}{sfx}"] - out[f"top{k}{sfx}"]).abs()
            if float(gap.max()) > TOP_TOL:
                bad = out.loc[gap.idxmax(), "file"]
                raise ValueError(f"{arm}, k = {k}{sfx or ' (noisy)'}: the refit's top LCB1 candidate differs from "
                                 f"the replay's by {float(gap.max()):g} ({bad}); the gate would not read the GP "
                                 f"that ranks the sitting's candidates")
    return out


def load_replays(analysis: Path, replay_dirs, processes, ks) -> pd.DataFrame:
    """The sitting rows of the replays (sitting_by_magnitude.load's filters) with the clean top candidate."""
    d = sbm.load(analysis, STATS_PATH, processes, tuple(replay_dirs))
    d = d[d["k"].isin([int(k) for k in ks])]
    tops = []
    for name in replay_dirs:
        f = pd.read_csv(analysis / name / "end_of_study_per_run.csv.gz", low_memory=False,
                        usecols=["file", "family", "k", "candidates", "rho", "winner", "top_candidate_regret_clean"])
        f = f[(f["family"] == "tournament") & (f["candidates"] == "lcb") & (f["winner"] == "look")
              & (f["rho"] == 1.0)]
        tops.append(f[["file", "k", "top_candidate_regret_clean"]].assign(k=lambda x: x["k"].astype(int)))
    top = pd.concat(tops, ignore_index=True).drop_duplicates(["file", "k"], keep="first")
    return d.merge(top, on=["file", "k"], how="left", validate="one_to_one")


def run_frame(arm: str, noise: pd.DataFrame, ks=tuple(GATES.values()), processes=None,
              gate_ks=tuple(GATES.values())) -> pd.DataFrame:
    """One row per noisy run of ``arm`` (LogEI and qNEI): the standard process, the zero-trial rules, the
    sittings of each k in ``ks`` and the gate statistics before the sittings of ``gate_ks``, every regret
    in units of opt_z."""
    analysis = MAIN_ANALYSIS if arm == MAIN_ARM else FRESH_ANALYSIS
    processes = processes or REFIT_SCOPE[arm][1]
    d = load_replays(analysis, REPLAYS[arm], processes, ks)
    ship = pd.read_csv(SHIP_RULES_CSV[arm], low_memory=False)
    return build_frame(d, ship, noise, arm, ks, gate_ks)


def build_frame(d: pd.DataFrame, ship: pd.DataFrame, noise: pd.DataFrame, arm: str,
                ks=tuple(GATES.values()), gate_ks=tuple(GATES.values())) -> pd.DataFrame:
    """The per-run frame from replay rows ``d`` (sitting_by_magnitude.load's columns plus
    top_candidate_regret_clean), the ship-rule rescoring and the refits (for the sizes in gate_ks,
    which must be among ks)."""
    if not set(gate_ks) <= set(ks):
        raise ValueError(f"the gated sizes {gate_ks} must be among the sittings loaded {ks}")
    m = sbm.attach_ship_rules(d, ship)
    keep = ["file", "dataset", "acquisition", "seed", "error_model", "sigma", "onset", "opt_z", "ref_noisy",
            "ref_clean"] + [f"regret_{r}" for r in sbm.SHIP_RULES] + [f"regret_{r}_clean" for r in sbm.SHIP_RULES]
    frame = m.drop_duplicates("file")[keep].reset_index(drop=True)
    for k in ks:
        x = m[m["k"] == int(k)].set_index("file")
        frame[f"sit{k}"] = frame["file"].map(x["regret_noisy"])
        frame[f"sit{k}_clean"] = frame["file"].map(x["regret_clean"])
        frame[f"top{k}"] = frame["file"].map(x["top_candidate_regret_noisy"])
        frame[f"top{k}_clean"] = frame["file"].map(x["top_candidate_regret_clean"])
        if frame[[f"sit{k}", f"sit{k}_clean"]].isna().any().any():
            raise KeyError(f"{arm}: {int(frame[f'sit{k}'].isna().sum())} runs have no sitting of {k}")
    frame = attach_noise(frame, noise, arm, gate_ks)
    # Every regret in units of the achievable improvement (the gate statistics stay in landscape SDs).
    z = frame["opt_z"].to_numpy()
    for c in ["ref_noisy", "ref_clean"] + [c for c in frame.columns if c.startswith(("regret_", "sit"))]:
        frame[c] = frame[c].to_numpy() / z
    return frame


# ---------------------------------------------------------------------------
# The gated procedures and their scores
# ---------------------------------------------------------------------------


def gate_fires(frame: pd.DataFrame, k: int, tau: float, stat: str = "noise_sd") -> tuple[np.ndarray, np.ndarray]:
    """Does the gate before a sitting of k fire, on each noisy run and on its clean twin?

    A run without a noise estimate (a failed GP fit) does not sit: NaN > tau is False.
    """
    n = T - int(k)
    return (frame[f"{stat}_{n}"].to_numpy(dtype=float) > tau,
            frame[f"{stat}_{n}_clean"].to_numpy(dtype=float) > tau)


def gated(frame: pd.DataFrame, k: int, tau: float, stat: str = "noise_sd") -> tuple[np.ndarray, np.ndarray]:
    """Deployed regret (opt_z units) of the gated sitting of k, noisy run and clean twin."""
    fn, fc = gate_fires(frame, k, tau, stat)
    return (np.where(fn, frame[f"sit{k}"], frame["ref_noisy"]),
            np.where(fc, frame[f"sit{k}_clean"], frame["ref_clean"]))


def two_stage(frame: pd.DataFrame, tau16: float, tau2: float, stat: str = "noise_sd") -> tuple[np.ndarray, np.ndarray]:
    """G16, and if it does not fire at trial 34, G2 at trial 48 (the run is the logged one until then)."""
    f16n, f16c = gate_fires(frame, 16, tau16, stat)
    f2n, f2c = gate_fires(frame, 2, tau2, stat)
    return (np.where(f16n, frame["sit16"], np.where(f2n, frame["sit2"], frame["ref_noisy"])),
            np.where(f16c, frame["sit16_clean"], np.where(f2c, frame["sit2_clean"], frame["ref_clean"])))


def procedure_regret(frame: pd.DataFrame, name: str, taus: dict, stat: str = "noise_sd"):
    if name == TWO_STAGE:
        return two_stage(frame, taus["G16"], taus["G2"], stat)
    return gated(frame, GATES[name], taus[name], stat)


def per_run_scores(frame: pd.DataFrame, trt_n, trt_c) -> pd.DataFrame:
    """The estimand per noisy run (opt_z units): gain, cost, price of its clean twin, increments over LCB1/LCB2."""
    return pd.DataFrame({
        "dataset": frame["dataset"].to_numpy(), "acquisition": frame["acquisition"].to_numpy(),
        "seed": frame["seed"].to_numpy(), "error_model": frame["error_model"].to_numpy(),
        "sigma": frame["sigma"].to_numpy(), "onset": frame["onset"].to_numpy(),
        "gain": frame["ref_noisy"].to_numpy() - trt_n, "cost": (frame["ref_noisy"] - frame["ref_clean"]).to_numpy(),
        "price": np.asarray(trt_c) - frame["ref_clean"].to_numpy(),
        "inc_lcb1": frame["regret_lcb1"].to_numpy() - trt_n, "inc_lcb2": frame["regret_lcb2"].to_numpy() - trt_n,
    })


def stem_prices(scores: pd.DataFrame) -> pd.Series:
    """Per-landscape mean price over the clean twins (one per landscape, acquisition and seed)."""
    stems = scores.drop_duplicates(["dataset", "acquisition", "seed"])
    return stems.groupby("dataset")["price"].mean()


def uniform_value(frame: pd.DataFrame, trt_n, trt_c) -> float:
    """V = (sum over the cells of the landscape-mean gain - the landscape-mean price) / (cells + 1)."""
    s = per_run_scores(frame, trt_n, trt_c)
    cells = s.groupby(["sigma", "onset", "dataset"])["gain"].mean().groupby(["sigma", "onset"]).mean()
    return float((cells.sum() - stem_prices(s).mean()) / (len(cells) + 1))


def choose_threshold(train: pd.DataFrame, k: int, grid=TAU_GRID, stat: str = "noise_sd") -> tuple[float, pd.DataFrame]:
    """The rule fixed in advance: the tau of ``grid`` with the largest V on the training runs; ties to the larger."""
    rows = [{"tau": float(tau), "value": uniform_value(train, *gated(train, k, tau, stat))} for tau in grid]
    table = pd.DataFrame(rows)
    best = table["value"].max()
    tau = float(table.loc[table["value"] >= best - 1e-12, "tau"].max())
    return tau, table


def summarise_cell(scores: pd.DataFrame, rng: np.random.Generator, extra: tuple[str, ...] = ()) -> dict:
    """Gain, share of the cost, increments and their tests over the twenty landscapes of one cell."""
    cols = ["gain", "cost", "inc_lcb1", "inc_lcb2", *extra]
    per = scores.groupby("dataset")[cols].mean()
    g, c = per["gain"].to_numpy(), per["cost"].to_numpy()
    lo, hi = sbm.boot_ci(g, rng)
    slo, shi = sbm.ratio_ci(g, c, rng)            # drawn either way, so a withheld share moves no later draw
    why = sbm.share_suppressed(c)
    out = {"n_runs": int(len(scores)), "n_landscapes": int(len(per)), "cost": float(c.mean()),
           "gain": float(g.mean()), "gain_lo": lo, "gain_hi": hi, "gain_p": sbm.wilcoxon_p(g),
           "landscapes_gaining": int((g > 0).sum()),
           "share": np.nan if why else float(g.mean() / c.mean()),
           "share_lo": np.nan if why else slo, "share_hi": np.nan if why else shi, "share_suppressed": why}
    for col in ("inc_lcb1", "inc_lcb2", *extra):
        v = per[col].to_numpy()
        vlo, vhi = sbm.boot_ci(v, rng)
        out.update({col: float(v.mean()), f"{col}_lo": vlo, f"{col}_hi": vhi, f"{col}_p": sbm.wilcoxon_p(v),
                    f"{col}_ahead": int((v > 0).sum())})
    return out


def score_procedure(frame: pd.DataFrame, name: str, trt_n, trt_c, rng: np.random.Generator, scope: str,
                    fires: tuple[np.ndarray, np.ndarray] | None = None,
                    ungated: tuple[np.ndarray, np.ndarray] | None = None) -> list[dict]:
    """One row per cell, plus a pooled row, for one procedure in one scope; the price is scope-wide."""
    s = per_run_scores(frame, trt_n, trt_c)
    extra: tuple[str, ...] = ()
    if ungated is not None:
        # positive when gating the sitting does better than always sitting on the same run
        s["inc_ungated"] = np.asarray(ungated[0]) - np.asarray(trt_n)
        extra = ("inc_ungated",)
    if fires is not None:
        s["fires"], s["fires_clean"] = fires[0].astype(float), fires[1].astype(float)
    price_l = stem_prices(s)
    plo, phi = sbm.boot_ci(price_l.to_numpy(), rng)
    rows = []
    for (sig, ons), sub in s.groupby(["sigma", "onset"], sort=True):
        row = {"procedure": name, "scope": scope, "cell": cell_label(sig, ons), "sigma": float(sig),
               "onset": int(ons), **summarise_cell(sub, rng, extra)}
        if fires is not None:
            row["fire_rate"] = float(sub["fires"].mean())
        rows.append(row)
    pooled = {"procedure": name, "scope": scope, "cell": "pooled", "sigma": np.nan, "onset": np.nan,
              **summarise_cell(s, rng, extra)}
    if fires is not None:
        pooled["fire_rate"] = float(s["fires"].mean())
    rows.append(pooled)
    clean_rate = (float(s.drop_duplicates(["dataset", "acquisition", "seed"])["fires_clean"].mean())
                  if fires is not None else np.nan)
    for r in rows:
        r.update({"price": float(price_l.mean()), "price_lo": plo, "price_hi": phi, "fire_rate_clean": clean_rate,
                  "value_uniform": np.nan})
    cells = [r for r in rows if r["cell"] != "pooled"]
    v = (sum(r["gain"] for r in cells) - float(price_l.mean())) / (len(cells) + 1)
    for r in rows:
        r["value_uniform"] = v
    return rows


def add_holm(table: pd.DataFrame) -> pd.DataFrame:
    """Holm over the eight cells of each (scope, procedure) and over the 24 (procedure, cell) of a scope."""
    table = table.copy()
    for col in ("gain", "inc_lcb1", "inc_lcb2"):
        table[f"{col}_p_holm_cells"] = np.nan
        table[f"{col}_p_holm_24"] = np.nan
        for scope, block in table.groupby("scope"):
            in_cells = block[(block["cell"] != "pooled") & block["procedure"].isin(PROCEDURES)]
            table.loc[in_cells.index, f"{col}_p_holm_24"] = sbm.holm(in_cells[f"{col}_p"].to_numpy())
            for _, sub in in_cells.groupby("procedure"):
                table.loc[sub.index, f"{col}_p_holm_cells"] = sbm.holm(sub[f"{col}_p"].to_numpy())
    return table


def _fmt_p(p: float) -> str:
    return "n/a" if not np.isfinite(p) else f"{p:.2g}"


def _share(r) -> str:
    if not np.isfinite(r["share"]):
        return "share n/a"
    return f"{100 * r['share']:.0f}% [{100 * r['share_lo']:.0f}, {100 * r['share_hi']:.0f}]"


def print_rows(table: pd.DataFrame, procedures, scope: str) -> None:
    for name in procedures:
        block = table[(table["procedure"] == name) & (table["scope"] == scope)]
        if block.empty:
            continue
        r0 = block.iloc[0]
        print(f"  {name}: price {r0['price']:.4f} [{r0['price_lo']:.4f}, {r0['price_hi']:.4f}]"
              + (f"; gate fires on {100 * r0['fire_rate_clean']:.1f}% of clean twins"
                 if np.isfinite(r0["fire_rate_clean"]) else "")
              + f"; V (8 cells and no error) {r0['value_uniform']:+.4f}")
        for _, r in block.iterrows():
            line = (f"    {r['cell']:24s} gain {r['gain']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}] "
                    f"{_share(r):18s} gaining {int(r['landscapes_gaining']):2d}/20 p {_fmt_p(r['gain_p'])}")
            if "gain_p_holm_cells" in r and np.isfinite(r.get("gain_p_holm_cells", np.nan)):
                line += f" (Holm cells {_fmt_p(r['gain_p_holm_cells'])}, 24 {_fmt_p(r['gain_p_holm_24'])})"
            line += (f" | over LCB1 {r['inc_lcb1']:+.4f} [{r['inc_lcb1_lo']:+.4f}, {r['inc_lcb1_hi']:+.4f}] "
                     f"p {_fmt_p(r['inc_lcb1_p'])} | over LCB2 {r['inc_lcb2']:+.4f} [{r['inc_lcb2_lo']:+.4f}, "
                     f"{r['inc_lcb2_hi']:+.4f}] p {_fmt_p(r['inc_lcb2_p'])}")
            if "inc_ungated" in r and np.isfinite(r.get("inc_ungated", np.nan)):
                line += f" | over always sitting {r['inc_ungated']:+.4f} [{r['inc_ungated_lo']:+.4f}, " \
                        f"{r['inc_ungated_hi']:+.4f}]"
            if "fire_rate" in r and np.isfinite(r.get("fire_rate", np.nan)):
                line += f" | fires {100 * r['fire_rate']:.0f}%"
            print(line)


HEADLINES = (("G16", "5sigma_from_trial_21"), ("G16", "5sigma_from_trial_1"), ("G2", "1sigma_from_trial_1"),
             ("G16+G2", "5sigma_from_trial_21"), ("G16+G2", "1sigma_from_trial_1"))


def print_headlines(table: pd.DataFrame, scopes) -> None:
    """The cells the text quotes, with every interval, p and Holm value, and the gate's footprint elsewhere."""
    print("\n== headline cells (gain over the standard process; increments over LCB1, LCB2 and always sitting;")
    print("   p two-sided Wilcoxon over the twenty landscapes, Holm over the cells of the procedure and over the 24)")
    for scope in scopes:
        print(f"  {scope}")
        for name, cell in HEADLINES:
            r = table[(table["scope"] == scope) & (table["procedure"] == name) & (table["cell"] == cell)].iloc[0]
            line = (f"    {name:6s} {cell:22s} gain {r['gain']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}] "
                    f"{_share(r)} p {_fmt_p(r['gain_p'])} (Holm {_fmt_p(r['gain_p_holm_cells'])}, "
                    f"{_fmt_p(r['gain_p_holm_24'])})")
            for col, lab in (("inc_lcb1", "LCB1"), ("inc_lcb2", "LCB2")):
                line += (f"; over {lab} {r[col]:+.4f} [{r[f'{col}_lo']:+.4f}, {r[f'{col}_hi']:+.4f}] "
                         f"ahead {int(r[f'{col}_ahead'])}/20 p {_fmt_p(r[f'{col}_p'])} "
                         f"(Holm {_fmt_p(r[f'{col}_p_holm_cells'])}, {_fmt_p(r[f'{col}_p_holm_24'])})")
            if np.isfinite(r.get("inc_ungated", np.nan)):
                line += (f"; over always sitting {r['inc_ungated']:+.4f} [{r['inc_ungated_lo']:+.4f}, "
                         f"{r['inc_ungated_hi']:+.4f}] p {_fmt_p(r['inc_ungated_p'])}")
            line += f"; price {r['price']:.4f} [{r['price_lo']:.4f}, {r['price_hi']:.4f}]"
            if np.isfinite(r.get("fire_rate", np.nan)):
                line += f"; fires on {100 * r['fire_rate']:.0f}% of the cell's runs, {100 * r['fire_rate_clean']:.1f}% " \
                        f"of clean twins"
            print(line)
        for name in PROCEDURES:
            b = table[(table["scope"] == scope) & (table["procedure"] == name) & (table["sigma"] <= 1.0)]
            worst = b.loc[b["gain"].abs().idxmax()]
            print(f"    {name}: largest |gain| at 0.05-1 sigma {worst['gain']:+.4f} ({worst['cell']}); V "
                  f"{b['value_uniform'].iloc[0]:+.4f}")
        for k in GATES.values():
            a = table[(table["scope"] == scope) & (table["procedure"] == f"always sit {k}")].iloc[0]
            print(f"    always sitting {k}: price {a['price']:.4f} [{a['price_lo']:.4f}, {a['price_hi']:.4f}], "
                  f"V {a['value_uniform']:+.4f}")


def score(args) -> None:
    noise = read_noise(Path(args.noise))
    main = run_frame(MAIN_ARM, noise)
    fresh = run_frame(FRESH_ARM, noise)
    train = main[main["seed"].isin(TRAIN_SEEDS)].reset_index(drop=True)
    scopes = {"seeds 7-11 (selection)": train,
              "seeds 12-16 (held out)": main[main["seed"].isin(TEST_SEEDS)].reset_index(drop=True),
              "seeds 27-36 (untouched, gaussian)": fresh}
    print("Gated final sittings, LogEI and qNEI, deployed regret in units of the achievable improvement opt_z.")
    print("Post hoc: built after the per-cell results of Section 7 and a prototype on every seed set; outside")
    print("every Holm family of the paper. Gate statistic: the noise SD of the GP that ranks the sitting's")
    print("candidates (fitted on trials 1..T-k), in landscape SDs; a study needs its objective's spread to")
    print("set the threshold on its own scale.")
    print(f"runs: main {len(main)} noisy (seeds 7-16, {', '.join(PROCESSES)}), untouched {len(fresh)} noisy "
          f"(seeds 27-36, gaussian); every refit reproduced the replay's top LCB1 candidate to {TOP_TOL:g}")
    fails = {a: int(noise.loc[noise["arm"] == a, "gp_failed"].astype(bool).sum()) for a in (MAIN_ARM, FRESH_ARM)}
    print(f"GP fallbacks in the refits (no noise estimate, the gate does not fire): {fails}")
    clean = noise[noise["baseline"]]
    for n in sorted(clean["n"].unique()):
        q = clean.loc[(clean["n"] == n) & (clean["arm"] == MAIN_ARM), "noise_sd"].quantile([0.5, 0.9, 0.99])
        print(f"   clean twins (seeds 7-16), noise SD on trials 1..{n}: median {q.iloc[0]:.3f}, 90th percentile "
              f"{q.iloc[1]:.3f}, 99th {q.iloc[2]:.3f} landscape SDs")

    selection: dict = {"rule": "argmax over the grid of V = (sum of the eight cells' landscape-mean gains - price) / 9 "
                               "on seeds 7-11, LogEI and qNEI, four processes pooled; ties to the larger tau",
                       "grid": {s: [float(x) for x in g] for s, g in STATS.items()}, "chosen": {}, "curves": {}}
    taus: dict[str, dict] = {}
    for stat, grid in STATS.items():
        taus[stat] = {}
        for gate, k in GATES.items():
            tau, curve = choose_threshold(train, k, grid, stat)
            taus[stat][gate] = tau
            selection["chosen"].setdefault(stat, {})[gate] = tau
            selection["curves"].setdefault(stat, {})[gate] = curve.to_dict("records")
    print("\n== thresholds chosen on seeds 7-11 by the rule fixed in advance (V over the eight cells and no error)")
    for stat in STATS:
        unit = "landscape SDs" if stat == "noise_sd" else "of the ratings' SD (scale-free)"
        for gate in GATES:
            curve = pd.DataFrame(selection["curves"][stat][gate])
            print(f"   {gate} on {stat}: tau = {taus[stat][gate]:g} {unit}; V by tau: "
                  + ", ".join(f"{t:g}: {v:+.4f}" for t, v in zip(curve["tau"], curve["value"])))

    rng = np.random.default_rng(SCORE_SEED)
    rows = []
    for scope, frame in scopes.items():
        for stat in STATS:
            tag = "" if stat == "noise_sd" else " (scale-free)"
            for name in PROCEDURES:
                trt_n, trt_c = procedure_regret(frame, name, taus[stat], stat)
                fires = gate_fires(frame, GATES[name], taus[stat][name], stat) if name in GATES else None
                ungated = (frame[f"sit{GATES[name]}"].to_numpy(), frame[f"sit{GATES[name]}_clean"].to_numpy()) \
                    if name in GATES else None
                for r in score_procedure(frame, name + tag, trt_n, trt_c, rng, scope, fires, ungated):
                    r["stat"], r["taus"] = stat, json.dumps(json_safe({g: taus[stat][g] for g in GATES}))
                    rows.append(r)
        # the ungated sittings and the zero-trial rules on the same runs, for reference
        for k in GATES.values():
            for r in score_procedure(frame, f"always sit {k}", frame[f"sit{k}"].to_numpy(),
                                     frame[f"sit{k}_clean"].to_numpy(), rng, scope):
                r["stat"], r["taus"] = "", ""
                rows.append(r)
        for rule in ("lcb1", "lcb2"):
            for r in score_procedure(frame, rule.upper(), frame[f"regret_{rule}"].to_numpy(),
                                     frame[f"regret_{rule}_clean"].to_numpy(), rng, scope):
                r["stat"], r["taus"] = "", ""
                rows.append(r)
    table = add_holm(pd.DataFrame(rows))
    REVIEW.mkdir(parents=True, exist_ok=True)
    table.to_csv(REVIEW / "gated_sitting.csv", index=False)

    for scope in scopes:
        print(f"\n== {scope}: gain over the standard process, share of its cost of error, increments over the")
        print("   zero-trial rules after all 50 trials (same runs), price without error; Holm over the eight cells")
        print("   of the procedure and over the 24 (procedure, cell) of the three gated procedures in this scope")
        print_rows(table, PROCEDURES, scope)
        print("  -- for reference, on the same runs:")
        print_rows(table, [f"always sit {k}" for k in GATES.values()] + ["LCB1", "LCB2"], scope)
    print("\n== the scale-free gate (noise SD over the SD of the ratings in hand), chosen by the same rule")
    for scope in list(scopes)[1:]:
        print(f"  {scope}")
        print_rows(table, [p + " (scale-free)" for p in PROCEDURES], scope)
    print_headlines(table, list(scopes)[1:])

    # Sensitivity over every threshold of the grid, on the three seed sets.
    sens = []
    for scope, frame in scopes.items():
        for stat, grid in STATS.items():
            for gate, k in GATES.items():
                for tau in grid:
                    s = per_run_scores(frame, *gated(frame, k, tau, stat))
                    cells = s.groupby(["sigma", "onset", "dataset"])["gain"].mean().groupby(["sigma", "onset"]).mean()
                    fn, fc = gate_fires(frame, k, tau, stat)
                    row = {"scope": scope, "stat": stat, "procedure": gate, "tau": float(tau),
                           "price": float(stem_prices(s).mean()),
                           "fire_rate_clean": float(pd.Series(fc).groupby(
                               [frame["dataset"], frame["acquisition"], frame["seed"]]).first().mean())}
                    row.update({cell_label(sg, o): float(v) for (sg, o), v in cells.items()})
                    row["value_uniform"] = (float(cells.sum()) - row["price"]) / (len(cells) + 1)
                    sens.append(row)
    sens = pd.DataFrame(sens)
    sens.to_csv(REVIEW / "gated_sitting_sensitivity.csv", index=False)
    print("\n== sensitivity over the threshold (gain over the standard process per cell, price, V)")
    show = [cell_label(s, o) for s, o in CELLS]
    short = {c: c.replace("sigma_from_trial_", "s/t") for c in show}
    for scope in scopes:
        for gate in GATES:
            block = sens[(sens["scope"] == scope) & (sens["stat"] == "noise_sd") & (sens["procedure"] == gate)]
            print(f"  {scope}, {gate} (noise SD in landscape SDs)")
            print("    " + block[["tau", "price", "fire_rate_clean"] + show + ["value_uniform"]].rename(
                columns=short).round(4).to_string(index=False).replace("\n", "\n    "))
    (REVIEW / "gated_sitting_selection.json").write_text(json.dumps(json_safe(selection), indent=2),
                                                         encoding="utf-8")
    print(f"\nwrote {rel(REVIEW / 'gated_sitting.csv')}, gated_sitting_sensitivity.csv, gated_sitting_selection.json")


# ---------------------------------------------------------------------------
# fresh-k16
# ---------------------------------------------------------------------------


FRESH_COLS = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file", "family", "k",
              "candidates", "rho", "winner", "regret_noisy", "ref_noisy", "ref_clean", "regret_clean",
              "top_candidate_regret_noisy"]


def paired_over_k(frame: pd.DataFrame, oz: dict[str, float], k_hi: int, k_lo: int, rng) -> pd.DataFrame:
    """The sitting of k_hi over the sitting of k_lo on the same runs, per cell: (regret_lo - regret_hi) / opt_z."""
    import fresh_seed_replication as fsr

    d = fsr.sitting(frame, oz)
    hi = d[d["k"] == k_hi].set_index("file")
    lo = d[d["k"] == k_lo].set_index("file")
    common = hi.index.intersection(lo.index)
    if len(common) != len(hi) or len(common) != len(lo):
        raise ValueError(f"the sittings of {k_hi} and {k_lo} do not cover the same runs")
    x = hi.loc[common].assign(inc=(lo.loc[common, "regret_noisy"] - hi.loc[common, "regret_noisy"])
                              / hi.loc[common, "dataset"].map(oz)).reset_index()
    rows = []
    for label, sub in fsr._cells(x):
        v = sub.groupby("dataset")["inc"].mean()
        mean, vlo, vhi = fsr._mean_ci(v, rng)
        rows.append({"cell": label, f"inc_k{k_lo}": mean, f"inc_k{k_lo}_lo": vlo, f"inc_k{k_lo}_hi": vhi,
                     f"inc_k{k_lo}_ahead": int((v > 0).sum()), f"inc_k{k_lo}_p": sbm.wilcoxon_p(v.to_numpy())})
    return pd.DataFrame(rows)


def fresh_k16(args) -> None:
    import fresh_seed_replication as fsr

    oz = fsr.opt_z()
    k16_dir = FRESH_ANALYSIS / "end_of_study_fresh_k16"
    settings = json.loads((k16_dir / "settings.json").read_text(encoding="utf-8"))
    k16 = pd.read_csv(k16_dir / "end_of_study_per_run.csv.gz", usecols=FRESH_COLS, low_memory=False)
    fresh = pd.read_csv(FRESH_ANALYSIS / "end_of_study_fresh" / "end_of_study_per_run.csv.gz", usecols=FRESH_COLS,
                        low_memory=False)
    ship = pd.read_csv(SHIP_RULES_CSV[FRESH_ARM], low_memory=False)
    looks = pd.read_csv(k16_dir / "end_of_study_looks.csv") if (k16_dir / "end_of_study_looks.csv").is_file() else None
    both = pd.concat([fresh, k16], ignore_index=True)
    if set(fresh["k"].dropna().astype(int)) & {16}:
        raise ValueError("end_of_study_fresh already holds k = 16")

    print("== The sitting of sixteen on the untouched seeds 27-36 (gaussian error, LogEI and qNEI)")
    print(f"   replay {k16_dir.relative_to(REPO).as_posix()}: settings {settings}")
    if looks is not None:
        for r in looks.itertuples():
            print(f"   {r.arm}, k = {r.k}: {100 * r.share_short_noisy:.1f}% of the noisy runs (all acquisitions) are "
                  f"short of 16 distinct prefix designs, mean {r.mean_looks_noisy:.2f} looks")
    rules, inc, prices = fsr.reference_table(k16, ship, oz)
    other = fsr.rule_increments(k16, ship, oz)
    # The Holm families over the four sizes: Wilcoxon p does not draw, so these are the p of the
    # published k = 2, 5, 12 rows and of k = 16 above.
    _, inc_all, _ = fsr.reference_table(both, ship, oz)
    other_all = fsr.rule_increments(both, ship, oz)
    rng = np.random.default_rng(SCORE_SEED + 16)
    over12 = paired_over_k(both, oz, 16, 12, rng)
    d16 = fsr.sitting(k16, oz)
    gain_rows = []
    for label, sub in fsr._cells(d16):
        per = sub.groupby("dataset")[["gain", "cost"]].mean()
        gain_rows.append({"cell": label, "gain_p": sbm.wilcoxon_p(per["gain"].to_numpy()),
                          "landscapes_gaining": int((per["gain"] > 0).sum())})
    gp = pd.DataFrame(gain_rows)
    gp["gain_p_holm_8"] = np.nan
    in_cells = gp["cell"] != "pooled"
    gp.loc[in_cells, "gain_p_holm_8"] = sbm.holm(gp.loc[in_cells, "gain_p"].to_numpy())

    out = inc.merge(other, on=["cell", "k"], validate="one_to_one").merge(gp, on="cell", validate="one_to_one")
    out = out.merge(over12, on="cell", validate="one_to_one")
    out["cost"] = out["cell"].map(rules.drop_duplicates("cell").set_index("cell")["cost"])
    a16 = inc_all[inc_all["k"] == 16].set_index("cell")
    o16 = other_all[other_all["k"] == 16].set_index("cell")
    out["inc_lcb1_p_holm_4k"] = out["cell"].map(a16["inc_lcb1_p_holm_over_k"])
    out["inc_lcb1_p_holm_32"] = out["cell"].map(a16["inc_lcb1_p_holm_all_cells"])
    for rule in ("lcb2", "pm"):
        out[f"inc_{rule}_p_holm_4k"] = out["cell"].map(o16[f"inc_{rule}_p_holm_over_k"])
        out[f"inc_{rule}_p_holm_32"] = out["cell"].map(o16[f"inc_{rule}_p_holm_all_cells"])
    out["inc_k12_p_holm_8"] = np.nan
    out.loc[out["cell"] != "pooled", "inc_k12_p_holm_8"] = sbm.holm(
        out.loc[out["cell"] != "pooled", "inc_k12_p"].to_numpy())
    # fresh_seed_replication names Holm over every in-cell row "over the 24"; with k = 16 alone it is the 8 cells.
    out = out.rename(columns={"inc_lcb1_p_holm_all_cells": "inc_lcb1_p_holm_8", "inc_lcb2_p_holm_all_cells":
                              "inc_lcb2_p_holm_8", "inc_pm_p_holm_all_cells": "inc_pm_p_holm_8"})
    out = out.drop(columns=[c for c in out.columns if c.endswith("_p_holm_over_k")])
    price16 = prices[prices["procedure"] == "sitting k = 16"].iloc[0]
    out["price_lo"], out["price_hi"] = float(price16["lo"]), float(price16["hi"])
    REVIEW.mkdir(parents=True, exist_ok=True)
    out.to_csv(REVIEW / "gated_sitting_fresh_k16.csv", index=False)

    print("   per cell (gain over the standard process; the paired increments over LCB1, LCB2 and PM after all 50")
    print("   trials and over the sitting of twelve, per run / opt_z, landscape means, landscape bootstrap; two-sided")
    print("   Wilcoxon over the twenty landscapes; Holm over the 8 cells at k = 16, over k in {2, 5, 12, 16} within")
    print("   the cell (4k) and over the 32 (cell, k)):")
    for _, r in out.iterrows():
        share = ("share n/a (cost not positive on every landscape)" if not np.isfinite(r["share"])
                 else f"share {r['share']:+.3f} [{r['share_lo']:+.3f}, {r['share_hi']:+.3f}]")
        print(f"   {r['cell']}  (cost of error {r['cost']:.4f})")
        print(f"      gain {r['gain']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}] {share}, gaining "
              f"{int(r['landscapes_gaining'])}/20, p {_fmt_p(r['gain_p'])} (Holm 8 {_fmt_p(r['gain_p_holm_8'])})")
        for rule in ("lcb1", "lcb2", "pm"):
            print(f"      over {rule.upper():4s} ({rule.upper()} gain {r[f'{rule}_gain']:+.4f}) {r[f'inc_{rule}']:+.4f} "
                  f"[{r[f'inc_{rule}_lo']:+.4f}, {r[f'inc_{rule}_hi']:+.4f}] ahead "
                  f"{int(r[f'inc_{rule}_landscapes_ahead'])}/20 p {_fmt_p(r[f'inc_{rule}_p'])} (Holm 8 "
                  f"{_fmt_p(r[f'inc_{rule}_p_holm_8'])}, 4k {_fmt_p(r[f'inc_{rule}_p_holm_4k'])}, 32 "
                  f"{_fmt_p(r[f'inc_{rule}_p_holm_32'])})")
        print(f"      over the sitting of 12 {r['inc_k12']:+.4f} [{r['inc_k12_lo']:+.4f}, {r['inc_k12_hi']:+.4f}] "
              f"ahead {int(r['inc_k12_ahead'])}/20 p {_fmt_p(r['inc_k12_p'])} (Holm 8 {_fmt_p(r['inc_k12_p_holm_8'])}); "
              f"over its own top candidate {r['inc_top']:+.4f}; {int(r['n_runs'])} runs")
    print(f"   clean-twin price of the sitting of 16: {price16['price']:+.4f} [{price16['lo']:+.4f}, "
          f"{price16['hi']:+.4f}] of opt_z (the zero-trial rules: "
          + ", ".join(f"{p.procedure} {p.price:+.4f}" for p in prices.itertuples() if not p.procedure.startswith("sit"))
          + ")")
    # The main sweep's held-out seeds, gaussian error alone, for comparison.
    kw = pd.read_csv(MAIN_ANALYSIS / "end_of_study_kwide" / "end_of_study_per_run.csv.gz", usecols=FRESH_COLS,
                     low_memory=False)
    ms = fsr.sitting(kw, oz)
    ms = ms[(ms["k"] == 16) & ms["seed"].between(12, 16)]
    late = ms[(ms["jitter_std"].astype(float) == 5.0) & (ms["jitter_iteration"].astype(int) == 20)]
    early = ms[(ms["jitter_std"].astype(float) == 1.0) & (ms["jitter_iteration"].astype(int) == 0)]
    print("   for comparison, the main sweep's seeds 12-16, gaussian error alone, k = 16:")
    print(f"      5 sigma from trial 21: {fsr.summarise(late, rng)}")
    print(f"      1 sigma from trial 1:  {fsr.summarise(early, rng)}")
    print(f"\nwrote {rel(REVIEW / 'gated_sitting_fresh_k16.csv')}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="score", choices=("refit", "score", "fresh-k16"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--noise", default=str(NOISE_CACHE), help="the refit cache 'refit' writes and 'score' reads")
    ap.add_argument("--resume", action="store_true", help="refit: keep the runs already in the cache")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    if args.command == "refit":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        run_refit(args.workers, Path(args.noise), args.resume)
    elif args.command == "fresh-k16":
        fresh_k16(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
