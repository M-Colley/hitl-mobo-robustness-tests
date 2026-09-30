"""How much does the standard ship rule's first-index tie-break matter?

The standard ship rule deploys the visited design with the best observed rating,
``int(np.argmax(ratings))`` in the simulator (bo_sensor_error_simulation.py, the
inference incumbent), so a tie at the maximum goes to the design visited FIRST.
replay_end_of_study.py keeps the same convention with a stable sort. With
continuous rating error a tie between designs of different true value has
probability zero: the main sweep's ties are all between exact ratings (in a
clean twin, or before an onset at trial 21) of designs with the same true
value, which ``rescore`` checks from the logs. A response cap, clipping,
rounding or a discrete scale makes the maximum tied between different designs
in many runs, and the first-index rule then deploys the earliest design that
reached the top of the scale.

This check re-scores the logs; it runs no simulation, since the ship rule never
feeds back into the search. For every per-run log it computes

    n_tied           designs whose rating is within TIE_TOL of the maximum
    regret_first     y_opt - f(x_first tied)       the logged rule
    regret_uniform   y_opt - mean_{i tied} f(x_i)  the EXACT expectation under a
                                                   uniformly random tie-break
                                                   (no seed is involved)
    regret_best_tie / regret_worst_tie             the two extreme tie-breaks

and checks that regret_first reproduces the logged final deployed regret. f is
the objective at the DEPLOYED design (objective_true_deployed where the log
has it, which differs from objective_true only under an unnoticed slip), and
lost or imputed ratings are never deployed, as in the simulator.

Subcommands

    scan      every arm directory (output-boba*, output-fitted*, output-oracle-iso*):
              per-run table and the share of runs with a tied maximum, per arm,
              variant and condition, with the share of consequential ties in
              which the first tied design is worse than the mean tied design
    rescore   every deployed-metric number the paper quotes from an arm with
              ties, under both conventions, with the estimand unchanged
              (analyse_boba_adaptations.paired_frame / summarise for arm
              contrasts, replay_hitl_remedies.recovery_table for the replayed
              remedies, whose reference is the standard rule), and where
              the main sweep's tied maxima come from (main_sweep_tie_origin.csv:
              re-read from the logs of the tied runs, are the tied ratings
              exact and before the onset?)

    python scripts/review_checks/tie_break.py scan --workers 6
    python scripts/review_checks/tie_break.py rescore

rescore prints exactly what it writes to register_checks/tie_break.txt, so it
can also run under run_review_checks.py with the argument ``rescore``; scan must
have run first (it reads the ~240,000 per-run logs, rescore only its table).
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

TIE_TOL = 1e-12          # two ratings within this are a tie
MATCH_TOL = 1e-9         # the logged regret must be reproduced to this
SPREAD_TOL = 1e-9        # tied designs whose true values differ by more make the tie matter
OUT_DIR = REPO / "output-boba" / "analysis" / "review" / "tie_break"
REGISTER_TXT = REPO / "output-boba" / "analysis" / "review" / "register_checks" / "tie_break.txt"
ARM_GLOBS = ("output-boba*", "output-fitted*", "output-oracle-iso*")
SKIP_PARTS = {"analysis", "evaluation", "analysis-sensitivity", "contrast", "configs", "data"}


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------


def tied_indices(ratings: np.ndarray, tol: float = TIE_TOL) -> np.ndarray:
    """Indices whose rating is within tol of the maximum, NaN ignored.

    Empty when nothing is rated. The first element is what np.nanargmax returns,
    i.e. the logged first-index rule.
    """
    r = np.asarray(ratings, dtype=float)
    if r.size == 0 or np.isnan(r).all():
        return np.array([], dtype=int)
    top = np.nanmax(r)
    with np.errstate(invalid="ignore"):
        return np.flatnonzero(np.abs(r - top) <= tol)


def tie_regrets(ratings: np.ndarray, deployed_true: np.ndarray, y_opt: float,
                tol: float = TIE_TOL) -> dict:
    """Deployed regret of the best-observed-rating rule under four tie-breaks.

    ``regret_uniform`` is the exact expectation over a uniformly random choice
    among the tied designs, y_opt minus the MEAN true value over the tied set;
    it equals ``regret_first`` whenever the maximum is unique or every tied
    design has the same true value.
    """
    idx = tied_indices(ratings, tol)
    f = np.asarray(deployed_true, dtype=float)
    if idx.size == 0:
        # The simulator deploys trial 1 when nothing is rated (_nan_argmax).
        idx = np.array([0])
    vals = f[idx]
    # Tied designs whose true values agree to SPREAD_TOL are one design for this
    # purpose (a revisit, a piecewise-constant oracle, float noise of 1e-13).
    distinct = int(np.unique(np.round(vals / SPREAD_TOL)).size)
    return {
        "n_tied": int(idx.size),
        "n_tied_distinct_true": distinct,
        "first_idx": int(idx[0]),
        "regret_first": float(y_opt - vals[0]),
        "regret_uniform": float(y_opt - vals.mean()),
        "regret_best_tie": float(y_opt - vals.max()),
        "regret_worst_tie": float(y_opt - vals.min()),
        "regret_last_tie": float(y_opt - vals[-1]),
    }


def _wanted(col: str) -> bool:
    return col in {
        "iteration", "objective_true", "objective_observed", "objective_true_deployed", "missing",
        "y_opt", "inference_simple_regret_true", "dataset", "acquisition", "seed", "error_model",
        "jitter_std", "jitter_iteration", "oracle_model", "objective",
    }


def run_record(path: Path) -> dict:
    """One per-run log reduced to its tie statistics."""
    from evaluate_research_question import parse_variant

    df = pd.read_csv(path, usecols=_wanted)
    df = df.sort_values("iteration", kind="stable")
    last = df.iloc[-1]
    oracle = str(last["oracle_model"])
    rec = {
        "file": path.name,
        "subdir": path.parent.name,
        "variant": parse_variant(path.name, oracle),
        "dataset": str(last["dataset"]),
        "acquisition": str(last["acquisition"]),
        "seed": int(last["seed"]),
        "error_model": str(last["error_model"]),
        "jitter_std": float(last["jitter_std"]),
        "jitter_iteration": int(last["jitter_iteration"]),
        "oracle_model": oracle,
        "objective": str(last["objective"]),
        "baseline": str(last["error_model"]) == "none",
        "n_iterations": int(len(df)),
        "y_opt": float(last["y_opt"]),
        "logged_final": float(last["inference_simple_regret_true"]),
    }
    if rec["objective"] == "multi_objective":
        # The multi-objective ship rule deploys a Pareto set, not an argmax, so
        # there is no tie to break.
        rec.update(is_multi=True, match_first=np.nan)
        return rec
    ratings = df["objective_observed"].to_numpy(dtype=float).copy()
    if "missing" in df.columns:
        lost = df["missing"].astype(str).str.lower().eq("true").to_numpy()
        ratings[lost] = np.nan
    deployed = (df["objective_true_deployed"] if "objective_true_deployed" in df.columns
                else df["objective_true"]).to_numpy(dtype=float)
    rec.update(tie_regrets(ratings, deployed, rec["y_opt"]))
    rec["is_multi"] = False
    rec["match_first"] = bool(abs(rec["regret_first"] - rec["logged_final"])
                              <= MATCH_TOL * max(1.0, abs(rec["logged_final"])))
    return rec


REQUIRED = ("iteration", "objective_true", "objective_observed", "y_opt", "inference_simple_regret_true",
            "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "oracle_model",
            "objective")


def _scan_dir(task: tuple[str, str]) -> list[dict]:
    """Every per-run log of one directory; a file that is not one is recorded, not fatal."""
    arm, directory = task
    out = []
    for path in sorted(Path(directory).glob("bo_sensor_error_*.csv")):
        header = pd.read_csv(path, nrows=0).columns
        lacking = [c for c in REQUIRED if c not in header]
        if lacking:
            out.append({"arm": arm, "file": path.name, "subdir": path.parent.name,
                        "not_a_run_log": ",".join(lacking), "is_multi": False})
            continue
        rec = run_record(path)
        rec["arm"] = arm
        out.append(rec)
    return out


def _rel(path: Path) -> Path:
    """A path relative to the repository where it lies inside it, so no local path is printed."""
    try:
        return Path(path).resolve().relative_to(REPO)
    except ValueError:
        return Path(path).name


def arm_directories(repo: Path = REPO) -> list[Path]:
    dirs = []
    for pattern in ARM_GLOBS:
        dirs.extend(p for p in sorted(repo.glob(pattern)) if p.is_dir())
    return dirs


def run_directories(arm_dir: Path) -> list[Path]:
    """Every directory under an arm that holds per-run logs, analysis dirs excluded."""
    found = set()
    for path in arm_dir.rglob("bo_sensor_error_*.csv"):
        rel = path.relative_to(arm_dir).parts[:-1]
        if any(part in SKIP_PARTS for part in rel):
            continue
        found.add(path.parent)
    return sorted(found)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for arm_dir in arm_directories():
        if args.arms and arm_dir.name not in args.arms:
            continue
        for d in run_directories(arm_dir):
            tasks.append((arm_dir.name, str(d)))
    print(f"{len(tasks)} run directories under {len({t[0] for t in tasks})} arms")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, part in enumerate(pool.map(_scan_dir, tasks, chunksize=1)):
            rows.extend(part)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)} directories, {len(rows)} runs", flush=True)
    runs = pd.DataFrame(rows)
    if "not_a_run_log" not in runs.columns:
        runs["not_a_run_log"] = np.nan
    odd = runs[runs["not_a_run_log"].notna()]
    if len(odd):
        print(f"{len(odd)} files named like run logs lack run-log columns and are left out, e.g. "
              + "; ".join(f"{r.arm}/{r.subdir}/{r.file} ({r.not_a_run_log})" for r in odd.head(3).itertuples()))
    runs_path = OUT_DIR / "tie_runs.parquet"
    if args.arms and runs_path.is_file():
        old = pd.read_parquet(runs_path)
        runs = pd.concat([old[~old["arm"].isin(set(runs["arm"]))], runs], ignore_index=True)
    runs.to_parquet(runs_path, index=False)
    share = write_shares(runs)
    print(f"wrote {_rel(runs_path)} ({len(runs)} runs), tie_share_by_arm.csv, tie_share_by_condition.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(share.to_string(index=False))


def _log_path(arm_dir: Path, subdir: str, name: str) -> Path:
    """The per-run log a row of the run table came from (analysis copies excluded)."""
    direct = arm_dir / subdir / name
    if direct.is_file():
        return direct
    for hit in sorted(arm_dir.rglob(name)):
        rel = hit.relative_to(arm_dir).parts[:-1]
        if hit.parent.name == subdir and not any(part in SKIP_PARTS for part in rel):
            return hit
    raise FileNotFoundError(f"{arm_dir.name}/{subdir}/{name}")


def tie_origin(runs: pd.DataFrame, arm: str = "output-boba", repo: Path = REPO) -> pd.DataFrame:
    """Where an arm's tied maxima come from, re-read from the logs of its tied runs.

    One row per run of ``arm`` with a tied maximum: the tied trials (0-based),
    whether every tied rating is exact (equal to the design's true value within
    TIE_TOL), and, for a noisy run, whether every tied trial comes before the
    onset (0-based index below jitter_iteration, since the error starts at trial
    jitter_iteration + 1). A tie between exact ratings is a tie between designs
    of the same true value, so it never changes the deployed design.
    """
    ok = runs[runs["not_a_run_log"].isna() & ~runs["is_multi"].astype(bool) & (runs["arm"] == arm)]
    tied = ok[ok["n_tied"] >= 2]
    rows = []
    for r in tied.itertuples():
        df = pd.read_csv(_log_path(repo / arm, r.subdir, r.file), usecols=_wanted)
        df = df.sort_values("iteration", kind="stable")
        ratings = df["objective_observed"].to_numpy(dtype=float).copy()
        if "missing" in df.columns:
            ratings[df["missing"].astype(str).str.lower().eq("true").to_numpy()] = np.nan
        truth = df["objective_true"].to_numpy(dtype=float)
        idx = tied_indices(ratings)
        baseline = str(r.baseline) == "True"
        rows.append({
            "arm": arm, "subdir": r.subdir, "file": r.file, "baseline": baseline,
            "error_model": r.error_model, "jitter_std": r.jitter_std, "jitter_iteration": int(r.jitter_iteration),
            "n_tied": int(idx.size), "first_tied_idx": int(idx[0]), "last_tied_idx": int(idx[-1]),
            "all_tied_exact": bool(np.all(np.abs(ratings[idx] - truth[idx]) <= TIE_TOL)),
            "all_tied_before_onset": np.nan if baseline else bool(idx[-1] < int(r.jitter_iteration)),
            "tie_matters": bool((r.regret_worst_tie - r.regret_best_tie) > SPREAD_TOL),
        })
    cols = ["arm", "subdir", "file", "baseline", "error_model", "jitter_std", "jitter_iteration", "n_tied",
            "first_tied_idx", "last_tied_idx", "all_tied_exact", "all_tied_before_onset", "tie_matters"]
    return pd.DataFrame(rows, columns=cols)


def tie_matters(runs: pd.DataFrame) -> pd.Series:
    """A tie matters when the tied designs' true values span more than SPREAD_TOL."""
    return (runs["regret_worst_tie"] - runs["regret_best_tie"]) > SPREAD_TOL


def write_shares(runs: pd.DataFrame) -> pd.DataFrame:
    share = tie_share(runs)
    share.to_csv(OUT_DIR / "tie_share_by_arm.csv", index=False)
    tie_share(runs, by_condition=True).to_csv(OUT_DIR / "tie_share_by_condition.csv", index=False)
    return share


def tie_share(runs: pd.DataFrame, by_condition: bool = False) -> pd.DataFrame:
    """Share of runs whose final maximum rating is tied, per arm and variant."""
    runs = runs[runs["not_a_run_log"].isna()] if "not_a_run_log" in runs.columns else runs
    r = runs[~runs["is_multi"].astype(bool)].copy()
    r["tied"] = r["n_tied"] >= 2
    r["tie_matters"] = tie_matters(r)
    # Among the runs whose tie matters: is the earliest tied design worse than
    # the average tied design (so the logged rule adds to the measured cost)?
    r["first_worse"] = r["tie_matters"] & (r["regret_first"] > r["regret_uniform"])
    keys = ["arm", "variant", "baseline"]
    if by_condition:
        keys += ["error_model", "jitter_std", "jitter_iteration"]
    g = r.groupby(keys, dropna=False)
    out = g.agg(n_runs=("file", "size"), share_tied=("tied", "mean"),
                share_tie_matters=("tie_matters", "mean"), median_n_tied=("n_tied", "median"),
                max_n_tied=("n_tied", "max"), share_logged_is_first=("match_first", "mean"),
                mean_regret_first=("regret_first", "mean"),
                mean_regret_uniform=("regret_uniform", "mean"),
                _n_matters=("tie_matters", "sum"), _n_first_worse=("first_worse", "sum")).reset_index()
    out["share_first_worse_than_tied_mean"] = (out.pop("_n_first_worse")
                                               / out.pop("_n_matters").where(lambda s: s > 0))
    multi = runs[runs["is_multi"].astype(bool)]
    if len(multi):
        m = multi.groupby(["arm", "variant", "baseline"], dropna=False).size().reset_index(name="n_runs")
        m["note"] = "multi-objective: Pareto-set ship rule, no argmax tie"
        out = pd.concat([out, m], ignore_index=True)
    return out.sort_values(keys, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# rescore
# ---------------------------------------------------------------------------


def load_runs() -> pd.DataFrame:
    return pd.read_parquet(OUT_DIR / "tie_runs.parquet")


def attach_uniform(evaluation: pd.DataFrame, runs: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Add the uniform tie-break's final deployed regret to an evaluation frame.

    The evaluation rows (paired_excess_metrics.csv, one per noisy run with its
    clean twin) gain ``final_inference_uniform_tie_jitter`` and ``_baseline``,
    so paired_frame can be called with response ``final_inference_uniform_tie``
    and the estimand is otherwise unchanged. Every row must find its run, and
    the logged first-index value must match the evaluation's own value.
    """
    r = runs[(runs["arm"] == arm) & runs["not_a_run_log"].isna()] if "not_a_run_log" in runs.columns \
        else runs[runs["arm"] == arm]
    r = r.assign(baseline=r["baseline"].astype(bool))
    noisy = r[~r["baseline"]][["dataset", "acquisition", "seed", "error_model", "jitter_std",
                              "jitter_iteration", "variant", "oracle_model", "regret_first",
                              "regret_uniform"]]
    clean = r[r["baseline"]][["dataset", "acquisition", "seed", "oracle_model", "regret_first",
                             "regret_uniform"]]
    if clean.duplicated(["dataset", "acquisition", "seed", "oracle_model"]).any():
        raise ValueError(f"{arm}: duplicate clean runs")
    ev = evaluation.copy()
    ev["variant"] = ev["variant"].fillna("") if "variant" in ev.columns else ""
    # jitter_std is a float read from two different files; match on a rounded key.
    ev["_std"] = ev["jitter_std"].round(9)
    noisy = noisy.assign(_std=noisy["jitter_std"].round(9)).drop(columns="jitter_std")
    keys = ["dataset", "acquisition", "seed", "error_model", "_std", "jitter_iteration", "variant",
            "oracle_model"]
    m = ev.merge(noisy, on=keys, how="left", validate="one_to_one")
    if m["regret_uniform"].isna().any() and (ev["variant"] == "").all():
        # An evaluation that predates the variant column (analyse_boba_adaptations.load
        # then writes ""), in a directory whose runs carry one suffix: pair without
        # it, provided the remaining keys still identify one run.
        keys = [k for k in keys if k != "variant"]
        bare = noisy.drop(columns="variant")
        if bare.duplicated(keys).any():
            raise ValueError(f"{arm}: the evaluation has no variant and the runs need one to pair")
        m = ev.merge(bare, on=keys, how="left", validate="one_to_one")
    m = m.merge(clean, on=["dataset", "acquisition", "seed", "oracle_model"], how="left",
                suffixes=("", "_clean"), validate="many_to_one")
    if m["regret_uniform"].isna().any() or m["regret_uniform_clean"].isna().any():
        raise ValueError(f"{arm}: {int(m['regret_uniform'].isna().sum())} evaluation rows without a run")
    for col, logged in (("regret_first", "final_inference_simple_regret_true_jitter"),
                        ("regret_first_clean", "final_inference_simple_regret_true_baseline")):
        bad = ~np.isclose(m[col], m[logged], rtol=0, atol=1e-9)
        if bad.any():
            raise ValueError(f"{arm}: {int(bad.sum())} rows where the first-index regret "
                             f"does not reproduce {logged}")
    return m.assign(final_inference_uniform_tie_jitter=m["regret_uniform"],
                    final_inference_uniform_tie_baseline=m["regret_uniform_clean"]).drop(
        columns=["_std", "regret_first", "regret_uniform", "regret_first_clean", "regret_uniform_clean"])


FIRST = "final_inference_simple_regret_true"      # the logged rule, as the evaluation stores it
UNIFORM = "final_inference_uniform_tie"           # the same column under a uniform tie-break
CONVENTIONS = (("first", FIRST), ("uniform", UNIFORM))
SUMMARY_COLS = ("n_landscapes", "n_cells", "cost", "gain", "price", "recovered", "recovered_lo",
                "recovered_hi", "wilcoxon_p")


def _opt_z() -> dict[str, float]:
    import boba_benchmarks as bb
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    return {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}


_SCALES: dict[str, dict[str, float]] = {}


def _dataset_scale(kind: str) -> dict[str, float]:
    """analyse_boba_adaptations.dataset_scale, computed once per kind."""
    if kind not in _SCALES:
        import analyse_boba_adaptations as aba
        _SCALES[kind] = aba.dataset_scale(kind)
    return _SCALES[kind]


def _moved(frame: pd.DataFrame) -> int:
    """Rows whose deployed regret differs between the two conventions, noisy or clean."""
    d = (frame[f"{UNIFORM}_jitter"] - frame[f"{FIRST}_jitter"]).abs() \
        + (frame[f"{UNIFORM}_baseline"] - frame[f"{FIRST}_baseline"]).abs()
    return int((d > SPREAD_TOL).sum())


def arm_contrast(name: str, runs: pd.DataFrame, opt_z: dict[str, float]) -> tuple[list[dict], str]:
    """One analyse_boba_adaptations arm on the deployed design, under both conventions.

    The loading, pairing and summaries are the script's own (load, paired_frame,
    summarise, and main's order of generator use), so the first-index rows
    reproduce adaptations_recovery.csv; only the response column differs.
    """
    import analyse_boba_adaptations as aba

    spec = aba.ARMS[name]
    arm_dir, ref_dir = REPO / spec["dir"], REPO / spec["ref"]
    seeds = {int(s) for s in spec["seeds"].split(",")}
    ref_df = aba.load(ref_dir, set(spec["ref_acqs"].split(",")), seeds, spec["error_model"], spec.get("ref_variant"))
    arm_df = aba.load(arm_dir, set(spec["acqs"].split(",")), seeds, spec["error_model"], spec.get("variant"))
    own_rule = ""
    try:
        ref_df = attach_uniform(ref_df, runs, ref_dir.name)
    except ValueError as exc:
        return [], f"not re-scored: {exc}"
    try:
        arm_df = attach_uniform(arm_df, runs, arm_dir.name)
    except ValueError as exc:
        if "does not reproduce" not in str(exc):
            return [], f"not re-scored: {exc}"
        # The treatment ships by its own rule (by mean over re-ratings, on
        # corrected ratings): it has no argmax tie to break, so it keeps its
        # logged value and only the reference is re-scored.
        arm_df = arm_df.assign(**{f"{UNIFORM}_jitter": arm_df[f"{FIRST}_jitter"],
                                  f"{UNIFORM}_baseline": arm_df[f"{FIRST}_baseline"]})
        own_rule = "; the treatment keeps its own (non-argmax) ship rule, only the reference is re-scored"
    moved = {"ref": _moved(ref_df), "trt": _moved(arm_df)}
    if spec["relative"]:
        ref_df, arm_df = aba.rank_magnitudes(ref_df), aba.rank_magnitudes(arm_df)
    # Each arm names its dataset scale (opt_z, the multi-objective floor gap or the
    # fitted achievable improvement); paired_frame raises for a dataset without
    # one, so the scale comes from the arm spec, exactly as in
    # analyse_boba_adaptations.main.
    scale = opt_z if spec["scale"] == "opt_z" else _dataset_scale(spec["scale"])
    rows = []
    for convention, response in CONVENTIONS:
        paired = aba.paired_frame(ref_df, arm_df, response, scale, spec["pool"])
        rng = np.random.default_rng(aba.BOOTSTRAP_SEED)
        for (std, onset), block in paired.groupby(["jitter_std", "jitter_iteration"]):
            rows.append({"arm": name, "convention": convention, "scope": "condition", "jitter_std": float(std),
                         "jitter_iteration": int(onset), **aba.summarise(block, rng)})
        pooled = aba.summarise(paired, np.random.default_rng(aba.BOOTSTRAP_SEED))
        rows.append({"arm": name, "convention": convention, "scope": "pooled", "jitter_std": np.nan,
                     "jitter_iteration": np.nan, **pooled})
    for r in rows:
        r.update(ref_runs_moved=moved["ref"], trt_runs_moved=moved["trt"],
                 n_ref_runs=len(ref_df), n_trt_runs=len(arm_df))
    return rows, f"re-scored ({moved['ref']} reference and {moved['trt']} treatment runs move){own_rule}"


def arm_excess(runs: pd.DataFrame, opt_z: dict[str, float], arms: list[str]) -> pd.DataFrame:
    """Deployed excess (noisy minus its clean twin, / opt_z) under five tie-breaks, per condition.

    A ratio of landscape means over the model-based acquisitions of each arm
    directory, variant and condition, straight from the scan: first index (the
    logged rule), uniform (the sensitivity), last index, and the best and worst
    tied design, which bound what any tie-break could do.
    """
    rows = []
    cols = ["regret_first", "regret_uniform", "regret_last_tie", "regret_best_tie", "regret_worst_tie"]
    for arm in arms:
        r = runs[(runs["arm"] == arm) & runs["not_a_run_log"].isna() & ~runs["is_multi"].astype(bool)]
        r = r[~r["acquisition"].isin(("random", "sobol"))].assign(baseline=lambda d: d["baseline"].astype(bool))
        clean = r[r["baseline"]].set_index(["dataset", "acquisition", "seed", "oracle_model"])
        noisy = r[~r["baseline"]]
        key = pd.MultiIndex.from_frame(noisy[["dataset", "acquisition", "seed", "oracle_model"]])
        z = noisy["dataset"].map(opt_z)
        if z.isna().any():
            continue
        ex = pd.DataFrame({c: (noisy[c].to_numpy() - clean[c].reindex(key).to_numpy()) / z.to_numpy()
                           for c in cols}, index=noisy.index)
        ex = pd.concat([noisy[["dataset", "variant", "error_model", "jitter_std", "jitter_iteration"]], ex], axis=1)
        ex["moved"] = tie_matters(noisy)
        for keys, g in ex.groupby(["variant", "error_model", "jitter_std", "jitter_iteration"], dropna=False):
            land = g.groupby("dataset")[cols].mean()
            rows.append({"arm": arm, "variant": keys[0], "error_model": keys[1], "jitter_std": keys[2],
                         "jitter_iteration": keys[3], "n_runs": len(g), "n_landscapes": len(land),
                         "share_runs_moved": float(g["moved"].mean()),
                         **{c.replace("regret_", "excess_"): float(land[c].mean()) for c in cols}})
    return pd.DataFrame(rows)


def check_against_published(rows: pd.DataFrame, published: pd.DataFrame) -> float:
    """Max abs difference of the first-index rows from adaptations_recovery.csv (deployed)."""
    pub = published[published["response"] == "deployed"]
    worst = 0.0
    for r in rows[rows["convention"] == "first"].itertuples():
        p = pub[pub["arm"] == r.arm]
        if r.scope == "pooled":
            ref = p.iloc[0]
            pairs = [(r.recovered, ref["pooled_recovered"]), (r.recovered_lo, ref["pooled_recovered_lo"]),
                     (r.recovered_hi, ref["pooled_recovered_hi"]), (r.cost, ref["pooled_cost"]),
                     (r.gain, ref["pooled_gain"]), (r.price, ref["pooled_price"])]
        else:
            ref = p[np.isclose(p["jitter_std"], r.jitter_std) & (p["jitter_iteration"] == r.jitter_iteration)].iloc[0]
            pairs = [(r.recovered, ref["recovered"]), (r.recovered_lo, ref["recovered_lo"]),
                     (r.recovered_hi, ref["recovered_hi"]), (r.cost, ref["cost"]), (r.gain, ref["gain"]),
                     (r.price, ref["price"])]
        for a, b in pairs:
            if np.isfinite(a) or np.isfinite(b):
                worst = max(worst, abs(a - b))
    return worst


# The replayed remedies whose reference is the standard rule: (arm directory,
# output subdirectory, seeds). The rank refit and the shortlist are quoted from
# the main sweep, the gross-fault arm and the capped-scale arm.
REMEDY_RUNS = (("output-boba", "hitl_remedies", "7-16"),
               ("output-boba-spike", "hitl_remedies", "7-11"),
               ("output-boba-spike", "hitl_remedies_heldout", "12-16"),
               ("output-boba-ceiling", "hitl_remedies", "7-11"),
               ("output-boba-ceiling", "hitl_remedies_heldout", "12-16"))


def remedies_with_uniform_reference(per_run: pd.DataFrame, runs: pd.DataFrame, arm: str) -> pd.DataFrame:
    """replay_hitl_remedies' per-run table with the standard rule re-scored.

    ref_noisy and ref_clean (and the ``standard`` procedure's own regret) become
    the uniform tie-break's expected regret of the noisy run and its clean twin.
    The treatment's regret is the replayed rule's and does not change: the rank
    refit and the shortlist choose by a GP criterion, which has no exact ties.
    """
    r = runs[(runs["arm"] == arm) & runs["not_a_run_log"].isna()].assign(
        baseline=lambda d: d["baseline"].astype(bool))
    noisy = r[~r["baseline"]].set_index("file")
    clean = r[r["baseline"]].set_index(["dataset", "acquisition", "seed"])
    f = per_run.copy()
    first_n = f["file"].map(noisy["regret_first"])
    key = pd.MultiIndex.from_frame(f[["dataset", "acquisition", "seed"]])
    first_c = pd.Series(clean["regret_first"].reindex(key).to_numpy(), index=f.index)
    uni_n = f["file"].map(noisy["regret_uniform"])
    uni_c = pd.Series(clean["regret_uniform"].reindex(key).to_numpy(), index=f.index)
    if first_n.isna().any() or first_c.isna().any():
        raise ValueError(f"{arm}: {int(first_n.isna().sum())} replayed runs not found in the scan")
    for got, want, label in ((first_n, f["ref_noisy"], "ref_noisy"), (first_c, f["ref_clean"], "ref_clean")):
        bad = ~np.isclose(got, want, rtol=0, atol=1e-9)
        if bad.any():
            raise ValueError(f"{arm}: the replay's {label} is not the first-index rule in {int(bad.sum())} rows")
    f["ref_noisy"], f["ref_clean"] = uni_n.to_numpy(), uni_c.to_numpy()
    std = f["procedure"] == "standard"
    f.loc[std, "regret_noisy"] = f.loc[std, "ref_noisy"]
    f.loc[std, "regret_clean"] = f.loc[std, "ref_clean"]
    return f


def replayed_remedies(runs: pd.DataFrame, opt_z: dict[str, float]) -> tuple[pd.DataFrame, list[str], dict]:
    import replay_hitl_remedies as rhr

    out, notes, frames = [], [], {}
    for arm, sub, seeds in REMEDY_RUNS:
        base = REPO / arm / "analysis" / sub
        per = pd.read_csv(base / "hitl_remedies_per_run.csv.gz", low_memory=False)
        published = pd.read_csv(base / "hitl_remedies_recovery.csv")
        uni = remedies_with_uniform_reference(per, runs, arm)
        frames[(arm, sub)] = {"first": per, "uniform": uni}
        moved = int(((uni["ref_noisy"] - per["ref_noisy"]).abs() > SPREAD_TOL).groupby(per["file"]).any().sum())
        for convention, frame in (("first", per), ("uniform", uni)):
            table = rhr.recovery_table(frame, opt_z)
            if convention == "first":
                cols = ["cost", "gain", "price", "recovered", "recovered_lo", "recovered_hi"]
                keys = ["procedure", "error_model", "jitter_std", "jitter_iteration"]
                m = table.merge(published, on=keys, suffixes=("", "_pub"))
                diff = max(float((m[c] - m[f"{c}_pub"]).abs().max()) for c in cols)
                notes.append(f"{arm}/{sub}: {len(m)} rows of hitl_remedies_recovery.csv reproduced, "
                             f"max abs difference {diff:.1e}; {moved} noisy runs move under the uniform rule")
            out.append(table.assign(arm=arm, subdir=sub, seeds=seeds, convention=convention, runs_moved=moved))
    return pd.concat(out, ignore_index=True), notes, frames


def rank_rule_landscape_splits(frames: dict, opt_z: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    """tab:heldout's landscape-split column for the rank rule, both conventions.

    heldout_remedies.py's own Cube, Problem, landscape_splits and
    summarise_splits, on the capped and gross-fault arms. The partitions are
    drawn per partition before any problem is visited, so one problem alone
    sees the same 2000 folds as the published run.
    """
    import heldout_remedies as hr

    rows, notes = [], []
    published = pd.read_csv(REPO / "output-boba" / "analysis" / "review" / "heldout_remedies.csv", low_memory=False)
    for arm in ("output-boba-spike", "output-boba-ceiling"):
        name = "rank_" + arm.replace("output-boba-", "")
        for convention, per in frames[(arm, "hitl_remedies")].items():
            long = hr._normalise(per[per["procedure"] != "standard"], opt_z)
            cube = hr.Cube(long)
            prob = hr.Problem(name, f"rank-rule variant ({convention} ties)", cube, ["ordinal_lcb1", "ordinal_pm"],
                              "recovery", hr.PAPER_RANK_RULE, schemes=("landscape",))
            full = {c: hr.recovery_summary(long, c, hr.ALL_SEEDS) for c in prob.candidates}
            full["_best"] = prob.candidates[prob.pick(np.array([full[c]["value"] for c in prob.candidates]))]
            splits = hr.landscape_splits([prob], {name: full["_best"]}, hr.N_PARTITIONS, hr.SPLIT_SEED)
            _, heads = hr.summarise_splits(splits, {name: full["_best"]}, {name: full}, {name: hr.PAPER_RANK_RULE})
            h = heads[(name, "landscape")]
            rows.append({"problem": name, "convention": convention, "full_best": full["_best"],
                         "full_value": full[full["_best"]]["value"], "full_lo": full[full["_best"]]["lo"],
                         "full_hi": full[full["_best"]]["hi"], "test_median": h["test_median"],
                         "test_p2_5": h["test_p2_5"], "test_p97_5": h["test_p97_5"],
                         "share_selected_full_best": h["share_selected_full_best"], "n_folds": h["n_folds"]})
            if convention == "first":
                pub = published[(published["section"] == "2_landscape_split") & (published["problem"] == name)].iloc[0]
                diff = max(abs(h[c] - pub[c]) for c in ("test_median", "test_p2_5", "test_p97_5"))
                notes.append(f"{name}: landscape-split median and range of heldout_remedies.csv reproduced, "
                             f"max abs difference {diff:.1e}")
    return pd.DataFrame(rows), notes


SEEDS_7_11 = frozenset(range(7, 12))
SEEDS_7_16 = frozenset(range(7, 17))


def instrument_screen(runs: pd.DataFrame, opt_z: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    """design_rules_from_pilot's instrument screen with the capped arm re-scored.

    Two estimands, each under both conventions.

    ``script``: its own instrument() and screen_fit(), called on a temporary copy
    of the capped arm's evaluation tree restricted to a seed set, with the
    deployed excess replaced by the uniform tie-break's under ``uniform``. The
    published file (design_rule_instrument_screen.csv) was produced when the arm
    had seeds 7-11 only; the arm now also holds seeds 12-16 for the fixed cap,
    so the script re-run as it stands gives other numbers, and both seed sets
    are reported. Its uncapped side is the main sweep's cell means over all ten
    acquisitions and seeds, not the capped arm's two acquisitions and seeds.

    ``paired``: the like-for-like version, per landscape and magnitude, from the
    ceiling-cost pairing of analyse_boba_adaptations (LogEI and qNEI, seeds
    7-11, each capped run against the main-sweep run with the same seed): extra
    cost = mean capped noisy - mean uncapped noisy (the clean twins coincide).
    screen_fit is applied to it unchanged.
    """
    import tempfile

    from scipy.stats import norm

    import analyse_boba_adaptations as aba
    import boba_benchmarks as bb
    import design_rules_from_pilot as dr

    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    ceiling = REPO / "output-boba-ceiling"
    out, notes = [], []
    for seed_label, seeds in (("7-11", SEEDS_7_11), ("7-16", SEEDS_7_16)):
        with tempfile.TemporaryDirectory() as tmp:
            roots = {"first": Path(tmp) / "first", "uniform": Path(tmp) / "uniform"}
            for path in sorted(ceiling.glob("*/evaluation/paired_excess_metrics.csv")):
                ev = pd.read_csv(path)
                ev = ev[ev["seed"].isin(seeds)]
                att = attach_uniform(ev, runs, ceiling.name)
                uni = att.assign(final_inference_simple_regret_excess_true=att[f"{UNIFORM}_jitter"]
                                 - att[f"{UNIFORM}_baseline"])
                for convention, frame in (("first", ev), ("uniform", uni)):
                    dst = roots[convention] / path.parent.parent.name / "evaluation"
                    dst.mkdir(parents=True)
                    frame.to_csv(dst / "paired_excess_metrics.csv", index=False)
            for convention, root in roots.items():
                inst = dr.instrument(root, REPO / "output-boba", stats, opt_z, 0.9,
                                     np.random.default_rng(dr.SEED))
                out.append(dr.screen_fit(inst).assign(convention=convention, estimand="script",
                                                      capped_seeds=seed_label))
    published = pd.read_csv(REPO / "output-boba" / "analysis" / "design_rule_instrument_screen.csv")
    cols = ["median_extra_cost", "rho_headroom", "extra_cost_low_headroom", "extra_cost_high_headroom"]
    for seed_label in ("7-11", "7-16"):
        mine = out[0 if seed_label == "7-11" else 2].merge(published, on="variant", suffixes=("", "_pub"))
        diff = max(float((mine[c] - mine[f"{c}_pub"]).abs().max()) for c in cols)
        notes.append(f"instrument screen, capped seeds {seed_label}: design_rule_instrument_screen.csv "
                     f"{'reproduced' if diff < 1e-12 else 'NOT reproduced'}, max abs difference {diff:.1e}")

    # The like-for-like pairing.
    cap_z = float(norm.ppf(0.9))
    ref = aba.load(REPO / "output-boba", {"logei", "qnei"}, set(SEEDS_7_11), "gaussian")
    ref = attach_uniform(ref, runs, "output-boba")
    for variant in ("ceil0.9-fixed", "ceil0.9-anchored"):
        trt = aba.load(ceiling, {"logei", "qnei"}, set(SEEDS_7_11), "gaussian", variant)
        trt = attach_uniform(trt, runs, ceiling.name)
        for convention, response in CONVENTIONS:
            p = aba.paired_frame(ref, trt, response, opt_z, pool=False)
            cells = p.groupby(["dataset", "jitter_std"]).agg(
                cost_capped=("trt_noisy", "mean"), cost_uncapped=("ref_noisy", "mean"),
                clean_capped=("trt_clean", "mean"), clean_uncapped=("ref_clean", "mean")).reset_index()
            cells["cost_capped"] -= cells.pop("clean_capped")
            cells["cost_uncapped"] -= cells.pop("clean_uncapped")
            cells["extra_cost"] = cells["cost_capped"] - cells["cost_uncapped"]
            cells["variant"] = variant
            cells["opt_z"] = cells["dataset"].map(opt_z)
            cells["cap_z"] = cap_z
            cells["headroom_above_cap"] = cells["opt_z"] - cap_z
            cells["frag"] = [float(stats[d].get(f"frag_{s:g}", np.nan))
                             for d, s in zip(cells["dataset"], cells["jitter_std"])]
            out.append(dr.screen_fit(cells).assign(convention=convention, estimand="paired",
                                                   capped_seeds="7-11"))
    return pd.concat(out, ignore_index=True), notes


def cautious_rule() -> tuple[pd.DataFrame, list[str]]:
    """The cautious (LCB2) and LCB1 ship rules on the main sweep, by cell and by pool.

    analyse_ship_rules' own loader, twin_frame, paired_frame and a fresh
    generator per summary, so the cell rows reproduce ship_rules_recovery.csv.
    The main sweep has no consequential ties, so there is one convention here.
    """
    import analyse_boba_adaptations as aba
    import analyse_ship_rules as asr

    runs = asr.load_per_run(REPO / "output-boba" / "analysis" / "ship_rules_per_run.csv")
    opt_z = asr.landscape_opt_z(runs["dataset"].unique())
    ref = asr.twin_frame(runs, asr.STANDARD)
    ref = ref[(ref["error_model"] == "gaussian") & (ref["variant"] == "")]
    published = pd.read_csv(REPO / "output-boba" / "analysis" / "ship_rules_recovery.csv")
    rows, notes = [], []
    for rule in ("lcb2", "lcb1"):
        trt = asr.twin_frame(runs, rule)
        trt = trt[(trt["error_model"] == "gaussian") & (trt["variant"] == "")]
        paired = aba.paired_frame(ref, trt, asr.RESPONSE, opt_z, pool=False)
        pools = {"0.25 sigma, both onsets": paired["jitter_std"].eq(0.25),
                 "1 sigma, both onsets": paired["jitter_std"].eq(1.0),
                 "5 sigma, both onsets": paired["jitter_std"].eq(5.0),
                 ">= 0.25 sigma, both onsets": paired["jitter_std"].ge(0.25),
                 ">= 0.25 sigma without 5 sigma (0.25 and 1 sigma)": paired["jitter_std"].isin([0.25, 1.0])}
        for (std, onset), block in paired.groupby(["jitter_std", "jitter_iteration"]):
            s = asr._summary(block)
            rows.append({"rule": rule, "pool": f"{std:g} sigma, onset {int(onset)}", "jitter_std": std,
                         "jitter_iteration": onset, **s, "n_rows": len(block)})
        for label, mask in pools.items():
            s = asr._summary(paired[mask])
            rows.append({"rule": rule, "pool": label, "jitter_std": np.nan, "jitter_iteration": np.nan, **s,
                         "n_rows": int(mask.sum())})
        # The rank refit is replayed on LogEI and qNEI only; its price is compared
        # with this rule's on the same two acquisitions (a price reads only the
        # clean twins, so the error process does not enter).
        two = paired["acquisition"].isin(["logei", "qnei"]) & paired["jitter_std"].ge(0.25)
        s = asr._summary(paired[two])
        rows.append({"rule": rule, "pool": ">= 0.25 sigma, both onsets, LogEI and qNEI only", "jitter_std": np.nan,
                     "jitter_iteration": np.nan, **s, "n_rows": int(two.sum())})
        pub = published[(published["arm"] == "output-boba") & (published["rule"] == rule)
                        & (published["error_model"] == "gaussian") & (published["cell"] == "condition")]
        mine = pd.DataFrame([r for r in rows if r["rule"] == rule and np.isfinite(r["jitter_std"])])
        m = mine.merge(pub, on=["jitter_std", "jitter_iteration"], suffixes=("", "_pub"))
        diff = max(float((m[c] - m[f"{c}_pub"]).abs().max()) for c in ("recovered", "recovered_lo", "recovered_hi"))
        notes.append(f"{rule}: {len(m)} gaussian condition rows of ship_rules_recovery.csv reproduced, "
                     f"max abs difference {diff:.1e}")
    out = pd.DataFrame(rows)
    # Each cell has the same number of runs per landscape, so a cell's share of
    # the pooled denominator is its cost over the sum of the cell costs.
    for rule, block in out.groupby("rule"):
        cells = block[block["jitter_std"].ge(0.25)]
        share5 = cells.loc[cells["jitter_std"].eq(5.0), "cost"].sum() / cells["cost"].sum()
        out.loc[out["rule"] == rule, "share_of_pooled_cost_at_5sigma"] = share5
    return out, notes


def _pct(v, lo=None, hi=None, sign=1.0) -> str:
    v = sign * v
    if lo is None or not np.isfinite(lo):
        return f"{100 * v:.1f}%"
    lo, hi = sorted((sign * lo, sign * hi))
    return f"{100 * v:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def quoted_numbers(arms: pd.DataFrame, rem: pd.DataFrame, splits: pd.DataFrame, screen: pd.DataFrame,
                   caut: pd.DataFrame) -> pd.DataFrame:
    """The paper's deployed-design numbers from arms with ties, under both conventions."""
    rows = []

    def add(where, quantity, source, first, uniform):
        rows.append({"where": where, "quantity": quantity, "source": source, "first": first, "uniform": uniform})

    def arm_row(name, conv, scope="pooled", std=None):
        g = arms[(arms["arm"] == name) & (arms["convention"] == conv) & (arms["scope"] == scope)]
        if std is not None:
            g = g[np.isclose(g["jitter_std"], std)]
        return g.iloc[0]

    src = "tie_break/arm_contrasts_deployed.csv"
    for scope, std, label in (("pooled", None, "pooled 0.25 and 1 sigma"), ("condition", 0.25, "0.25 sigma"),
                              ("condition", 1.0, "1 sigma")):
        f, u = arm_row("ceiling-cost", "first", scope, std), arm_row("ceiling-cost", "uniform", scope, std)
        add("Limitations, app:shiprules, app:budgetneutral, app:remedies (screen)",
            f"capped scale: extra cost over an uncapped scale, share of its cost ({label})",
            src, _pct(f.recovered, f.recovered_lo, f.recovered_hi, -1), _pct(u.recovered, u.recovered_lo,
                                                                              u.recovered_hi, -1))
    for scope, std, label in (("pooled", None, "pooled"), ("condition", 0.25, "0.25 sigma"),
                              ("condition", 1.0, "1 sigma")):
        f, u = arm_row("ceiling-anchored", "first", scope, std), arm_row("ceiling-anchored", "uniform", scope, std)
        add("tab:remedies, tab:budgetneutral, app:shiprules", f"re-anchored cap, recovered ({label}); price 0", src,
            _pct(f.recovered, f.recovered_lo, f.recovered_hi), _pct(u.recovered, u.recovered_lo, u.recovered_hi))
    for sp, label in (("sp0.15-20", "15% of trials at 20 SD"), ("sp0.05-20", "5% at 20 SD"),
                      ("sp0.05-5", "5% at 5 SD"), ("sp0.15-5", "15% at 5 SD")):
        f, u = arm_row(f"spike-clip-{sp}", "first"), arm_row(f"spike-clip-{sp}", "uniform")
        add("tab:remedies, tab:budgetneutral, app:shiprules",
            f"clip to the landscape's known range, recovered ({label}); price 0",
            src, _pct(f.recovered, f.recovered_lo, f.recovered_hi), _pct(u.recovered, u.recovered_lo,
                                                                          u.recovered_hi))
    for name, label in (("q-iu-0.05", "slip-aware acquisition, 0.05 slip"),
                        ("q-iu-0.15", "slip-aware acquisition, 0.15 slip"),
                        ("q-iu-0.4", "slip-aware acquisition, 0.4 slip"),
                        ("nigp", "noisy-input GP under a slip"), ("rerate-slip", "re-rating under a slip")):
        if not (arms["arm"] == name).any():
            continue
        f, u = arm_row(name, "first"), arm_row(name, "uniform")
        add("tab:budgetneutral / tab:adapt", f"{label}, recovered", src,
            _pct(f.recovered, f.recovered_lo, f.recovered_hi), _pct(u.recovered, u.recovered_lo, u.recovered_hi))

    src = "tie_break/replayed_remedies_deployed.csv"
    pooled = rem[rem["error_model"] == "pooled"]

    def rem_row(arm, sub, proc, conv):
        return pooled[(pooled["arm"] == arm) & (pooled["subdir"] == sub) & (pooled["procedure"] == proc)
                      & (pooled["convention"] == conv)].iloc[0]

    for arm, sub, label in (("output-boba-ceiling", "hitl_remedies", "capped scale, seeds 7-11"),
                            ("output-boba-ceiling", "hitl_remedies_heldout", "capped scale, seeds 12-16"),
                            ("output-boba-spike", "hitl_remedies", "gross faults, seeds 7-11"),
                            ("output-boba-spike", "hitl_remedies_heldout", "gross faults, seeds 12-16"),
                            ("output-boba", "hitl_remedies", "main sweep, seeds 7-16")):
        f, u = rem_row(arm, sub, "ordinal_lcb1", "first"), rem_row(arm, sub, "ordinal_lcb1", "uniform")
        add("tab:remedies, Sec. 7, app:remedies, tab:shortlist, tab:heldout",
            f"rank refit (ordinal_lcb1), recovered, {label}; price {f.price:.4f}", src,
            _pct(f.recovered, f.recovered_lo, f.recovered_hi), _pct(u.recovered, u.recovered_lo, u.recovered_hi))
    for m in (1, 2, 3, 5):
        f = rem_row("output-boba-ceiling", "hitl_remedies", f"shortlist_m{m}", "first")
        u = rem_row("output-boba-ceiling", "hitl_remedies", f"shortlist_m{m}", "uniform")
        add("tab:shortlist (capped scale)", f"ship {m} by LCB1, recovered; price {f.price:.4f}", src,
            _pct(f.recovered, f.recovered_lo, f.recovered_hi), _pct(u.recovered, u.recovered_lo, u.recovered_hi))
    for problem in ("rank_ceiling", "rank_spike"):
        f = splits[(splits["problem"] == problem) & (splits["convention"] == "first")].iloc[0]
        u = splits[(splits["problem"] == problem) & (splits["convention"] == "uniform")].iloc[0]
        add("tab:heldout", f"{problem}: landscape-split median [2.5, 97.5 percentiles] over 2000 folds",
            "tie_break/rank_rule_landscape_splits.csv", _pct(f.test_median, f.test_p2_5, f.test_p97_5),
            _pct(u.test_median, u.test_p2_5, u.test_p97_5))
    src = "tie_break/instrument_screen.csv"
    for est, seeds in (("script", "7-11"), ("paired", "7-11"), ("script", "7-16")):
        for variant in ("ceil0.9-fixed", "ceil0.9-anchored"):
            g = screen[(screen["estimand"] == est) & (screen["capped_seeds"] == seeds) & (screen["variant"] == variant)]
            f, u = g[g["convention"] == "first"].iloc[0], g[g["convention"] == "uniform"].iloc[0]
            fmt = (lambda r: f"median {r.median_extra_cost:.3f}, rho {r.rho_headroom:.2f}, "
                             f"halves {r.extra_cost_low_headroom:.3f}/{r.extra_cost_high_headroom:.3f}")
            add("app:remedies (screen)", f"instrument screen ({est}, capped seeds {seeds}), {variant}", src, fmt(f), fmt(u))
    src = "tie_break/cautious_rule_cells.csv (main sweep: no ties, one convention)"
    for r in caut[caut["rule"] == "lcb2"].itertuples():
        v = _pct(r.recovered, r.recovered_lo, r.recovered_hi) + f"; price {r.price:.4f}"
        add("Sec. 7, app:shiprules, tab:remedies", f"cautious rule (LCB2), {r.pool}", src, v, v)
    return pd.DataFrame(rows)


def cmd_rescore(args: argparse.Namespace) -> None:
    import analyse_boba_adaptations as aba

    runs = load_runs()
    if "not_a_run_log" not in runs.columns:
        runs["not_a_run_log"] = np.nan
    opt_z = _opt_z()
    lines: list[str] = []
    say = lines.append

    # 1. where the maximum is tied
    # Rebuilt from the per-run table on every call, so the shares always follow
    # the current definition of a tie that matters.
    share = write_shares(runs)
    share["variant"] = share["variant"].fillna("")
    ok = runs[runs["not_a_run_log"].isna() & ~runs["is_multi"].astype(bool)]
    say("Tie-break of the standard ship rule (best observed rating): first index, as logged, "
        "against a uniformly random choice among tied maxima (exact expectation, no seed).")
    say(f"runs scanned: {len(runs):,} in {runs['arm'].nunique()} arm directories; logged final deployed "
        f"regret reproduced by the first-index rule in {int(ok['match_first'].sum()):,} of {len(ok):,} "
        f"single-objective runs.")
    main = ok[ok["arm"] == "output-boba"]
    for b, label in ((False, "noisy"), (True, "clean")):
        g = main[main["baseline"].astype(bool) == b]
        say(f"main sweep {label} runs: {len(g):,}; tied maximum in {int((g['n_tied'] >= 2).sum())}, "
            f"a tie between designs of different true value in {int(tie_matters(g).sum())}.")
    origin = tie_origin(runs)
    origin.to_csv(OUT_DIR / "main_sweep_tie_origin.csv", index=False)
    for b, label in ((False, "noisy"), (True, "clean")):
        g = origin[origin["baseline"] == b]
        onsets = ", ".join(f"trial {int(k) + 1}: {int(v)}"
                           for k, v in g.groupby("jitter_iteration").size().items())
        text = (f"main sweep {label} runs with a tied maximum, re-read from their logs: {len(g)}; every tied "
                f"rating exact (equal to the true value within {TIE_TOL:g}) in {int(g['all_tied_exact'].sum())}")
        if not b:
            text += (f"; error onset {onsets}; every tied trial before the onset in "
                     f"{int(g['all_tied_before_onset'].astype(bool).sum())}")
        say(text + f"; tied designs of different true value in {int(g['tie_matters'].sum())}.")
    tied = share[(share["baseline"].astype(str) == "False") & (share["share_tie_matters"] > 0)]
    say("\narm directories with any noisy run whose tied maxima differ in true value "
        "(share of runs; median designs tied; among the consequential ties, the share in which the "
        "first tied design is worse than the mean tied design):")
    for r in tied.itertuples():
        say(f"  {r.arm:34s} {str(r.variant):28s} n={int(r.n_runs):5d}  tied {r.share_tied:6.1%}  "
            f"consequential {r.share_tie_matters:6.1%}  median tied {r.median_n_tied:g}  "
            f"first worse {r.share_first_worse_than_tied_mean:6.1%}")
    clean_tied = share[(share["baseline"].astype(str) == "True") & (share["share_tie_matters"] > 0)]
    if len(clean_tied):
        say("clean-twin (error_model none) directories with a consequential tie:")
        for r in clean_tied.itertuples():
            say(f"  {r.arm:34s} {str(r.variant):28s} n={int(r.n_runs):5d}  consequential "
                f"{r.share_tie_matters:6.1%}")
    excess = arm_excess(runs, opt_z, sorted(set(tied["arm"])))
    excess.to_csv(OUT_DIR / "arm_excess_deployed.csv", index=False)
    say("\ndeployed excess of each such arm (ratio of landscape means over its model-based acquisitions, "
        "share of the achievable improvement), per condition: first-index | uniform | last-index | "
        "best tied | worst tied")
    for r in excess.itertuples():
        say(f"  {r.arm:30s} {str(r.variant):20s} {r.error_model:8s} {r.jitter_std:g}/{int(r.jitter_iteration):<2d} "
            f"n={r.n_runs:4d} moved {r.share_runs_moved:5.1%}  {r.excess_first:.4f} | {r.excess_uniform:.4f} | "
            f"{r.excess_last_tie:.4f} | {r.excess_best_tie:.4f} | {r.excess_worst_tie:.4f}")
    not_first = ok[~ok["match_first"].astype(bool)].groupby("arm").size()
    if len(not_first):
        say("\nruns whose logged deployed regret is NOT the first-index argmax of the logged ratings "
            "(a different ship rule: re-rating by mean, per-rater backfit, and so on); not re-scored:")
        for arm, n in not_first.items():
            say(f"  {arm}: {n}")

    # 2. the arm contrasts of analyse_boba_adaptations
    published = pd.read_csv(REPO / "output-boba" / "analysis" / "adaptations_recovery.csv")
    rows, status = [], {}
    for name in aba.ARMS:
        spec = aba.ARMS[name]
        dirs = {Path(spec["dir"]).name, Path(spec["ref"]).name}
        if not (tied["arm"].isin(dirs)).any():
            status[name] = "no consequential ties in either directory; unchanged"
            continue
        r, st = arm_contrast(name, runs, opt_z)
        rows.extend(r)
        status[name] = st
    arms = pd.DataFrame(rows)
    arms.to_csv(OUT_DIR / "arm_contrasts_deployed.csv", index=False)
    say(f"\narm contrasts (analyse_boba_adaptations.ARMS, deployed design): first-index rows reproduce "
        f"adaptations_recovery.csv, max abs difference {check_against_published(arms, published):.1e}")
    for name, st in status.items():
        if not st.endswith("unchanged"):
            say(f"  {name}: {st}")
    say(f"  {sum(s.endswith('unchanged') for s in status.values())} other arms touch no directory with "
        "consequential ties and are unchanged")
    for name in [n for n in status if not status[n].endswith("unchanged") and not status[n].startswith("not")]:
        say(f"  {name}:")
        for r in arms[arms["arm"] == name].itertuples():
            cell = "pooled" if r.scope == "pooled" else f"{r.jitter_std:g} sigma, onset {int(r.jitter_iteration)}"
            say(f"    {r.convention:7s} {cell:22s} cost {r.cost:.4f} gain {r.gain:+.4f} price {r.price:+.4f} "
                f"recovered {r.recovered:+.1%} [{r.recovered_lo:+.1%}, {r.recovered_hi:+.1%}]")

    # 3. the replayed remedies whose reference is the standard rule
    rem, notes, frames = replayed_remedies(runs, opt_z)
    rem.to_csv(OUT_DIR / "replayed_remedies_deployed.csv", index=False)
    say("\nreplayed remedies (replay_hitl_remedies.recovery_table), reference = the standard rule:")
    for n in notes:
        say("  " + n)
    for (arm, sub, seeds), g in rem[rem["error_model"] == "pooled"].groupby(["arm", "subdir", "seeds"], sort=False):
        say(f"  {arm}/{sub} (seeds {seeds}):")
        for proc in ("ordinal_lcb1", "ordinal_pm", "shortlist_m1", "shortlist_m2", "shortlist_m3", "shortlist_m5"):
            parts = []
            for conv in ("first", "uniform"):
                x = g[(g["procedure"] == proc) & (g["convention"] == conv)].iloc[0]
                parts.append(f"{conv} {x.recovered:+.1%} [{x.recovered_lo:+.1%}, {x.recovered_hi:+.1%}] "
                             f"(cost {x.cost:.4f}, gain {x.gain:+.4f}, price {x.price:+.4f})")
            say(f"    {proc:13s} " + "; ".join(parts))

    # 4. tab:heldout's landscape splits for the rank rule
    splits, notes = rank_rule_landscape_splits(frames, opt_z)
    splits.to_csv(OUT_DIR / "rank_rule_landscape_splits.csv", index=False)
    say("\nrank rule, 2000 landscape-split folds (heldout_remedies.py's own functions):")
    for n in notes:
        say("  " + n)
    for r in splits.itertuples():
        say(f"  {r.problem:13s} {r.convention:7s} full {r.full_value:+.1%} [{r.full_lo:+.1%}, {r.full_hi:+.1%}]  "
            f"held-out median {r.test_median:+.1%} [{r.test_p2_5:+.1%}, {r.test_p97_5:+.1%}]  "
            f"folds choosing {r.full_best} {r.share_selected_full_best:.1%}")

    # 5. the instrument screen
    screen, notes = instrument_screen(runs, opt_z)
    screen.to_csv(OUT_DIR / "instrument_screen.csv", index=False)
    say("\ninstrument screen (design_rules_from_pilot.instrument / screen_fit), capped arm re-scored:")
    for n in notes:
        say("  " + n)
    say("  (script = design_rules_from_pilot as written, capped LogEI/qNEI against the main sweep's cell means "
        "over ten acquisitions; paired = LogEI/qNEI seeds 7-11 against the same runs of the main sweep)")
    for r in screen.itertuples():
        say(f"  {r.estimand:6s} capped seeds {r.capped_seeds:5s} {r.convention:7s} {r.variant:18s} "
            f"median extra cost {r.median_extra_cost:+.3f}  rho(headroom) {r.rho_headroom:+.2f}  "
            f"low-headroom half {r.extra_cost_low_headroom:+.3f}  high {r.extra_cost_high_headroom:+.3f}")

    # 6. the cautious rule (main sweep, no ties) and the rank refit's price
    caut, notes = cautious_rule()
    caut.to_csv(OUT_DIR / "cautious_rule_cells.csv", index=False)
    say("\ncautious ship rule (LCB2) and LCB1, gaussian error, main sweep, ten acquisitions, seeds 7-16:")
    for n in notes:
        say("  " + n)
    for r in caut.itertuples():
        say(f"  {r.rule} {r.pool:50s} cost {r.cost:.4f} gain {r.gain:+.4f} price {r.price:+.4f} "
            f"recovered {r.recovered:+.1%} [{r.recovered_lo:+.1%}, {r.recovered_hi:+.1%}]")
    for rule, g in caut.groupby("rule"):
        say(f"  {rule}: the 5 sigma cells carry {g['share_of_pooled_cost_at_5sigma'].iloc[0]:.1%} of the pooled "
            ">= 0.25 sigma cost")

    # 7. every quoted deployed-design number from an arm with ties, side by side
    quoted = quoted_numbers(arms, rem, splits, screen, caut)
    quoted.to_csv(OUT_DIR / "quoted_numbers.csv", index=False)
    say("\nquoted numbers, first-index (as logged) | uniform tie-break (deployed design throughout):")
    for r in quoted.itertuples():
        say(f"  {r.where:34s} {r.quantity:78s} {r.first:>22s} | {r.uniform}")

    REGISTER_TXT.parent.mkdir(parents=True, exist_ok=True)
    REGISTER_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # stdout is exactly the register text, so run_review_checks.py (which saves a
    # check's stdout as its register file) writes the same file; the paths go to
    # stderr and never into a tracked output.
    print("\n".join(lines))
    print(f"wrote {_rel(REGISTER_TXT)} and the CSVs in {_rel(OUT_DIR)}", file=sys.stderr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    p.add_argument("--arms", nargs="*", default=None)
    p.set_defaults(func=cmd_scan)
    p = sub.add_parser("rescore")
    p.set_defaults(func=cmd_rescore)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
