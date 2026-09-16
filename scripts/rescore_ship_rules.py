"""Re-score the design a study ships, under other ship rules, from the logs alone.

Every regret this project reports for the deployed design uses one ship rule:
the design with the single best observed rating (inference_simple_regret_true).
Under response error that rule chases lucky ratings. This script asks what the
same trials would have shipped under other rules, with no new trials: it refits
the loop's own surrogate on everything the loop saw and picks a visited design.

The surrogate is the one the loop fits at every step (bo_sensor_error_simulation
.run_simulation): BoTorch's default SingleTaskGP -- whatever kernel and priors the
installed version defaults to, no substitute -- with Normalize over the
benchmark's box and Standardize(m=1). It is fitted on every logged row with a
finite objective_observed (X = the logged param columns, i.e. the RECORDED
design), with the logged known_noise_var as train_Yvar where the run used
observation_noise="known", and with a torch seed fixed per file so a rerun
reproduces it.

Rules, each choosing among the visited designs (ties go to the earliest trial,
as np.argmax and the loop do):

    best_observed  the highest single rating -- the standard rule; it must
                   reproduce the log's final inference_simple_regret_true
    best_mean      the design with the highest mean over its ratings (the rule
                   the replication arms log)
    pm             argmax of the posterior mean
    lcb1, lcb2     argmax of posterior mean minus 1 or 2 latent (noise-free) SDs
    best_visited   the oracle: the visited design with the best true value, the
                   ceiling any rule that ships a visited design can reach

A chosen design is scored by objective_true_deployed where the log has it (an
unnoticed slip: the study deploys the design it wrote down, not the one the
rating came from) and by objective_true otherwise; regret = y_opt - value.

The reproduction check. The arm's own inference rule is read from
run_metadata.json through the simulator's adaptation_fields (replication arms
default to best_mean) and must reproduce the logged final regret to 1e-9, or the
script stops: a mismatch means the logs and this re-scoring disagree about what
was shipped, and every number downstream would be wrong. Where the arm's rule is
not best_observed, the per-run column best_observed_reproduces records whether
best_observed happens to coincide, and the summary reports how often it does not.

One row per run goes to <output-dir>/ship_rules_per_run.csv (default
<input-dir>/analysis). Progress is journalled to ship_rules_per_run.jsonl, so
--resume continues an interrupted job without refitting what is done.

    python scripts/rescore_ship_rules.py --input-dir output-boba --acquisitions logei,qnei --seeds 7-16 --workers 5 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import boba_benchmarks as bb  # noqa: E402

RULES = ("best_observed", "best_mean", "pm", "lcb1", "lcb2", "best_visited")
# Posterior rules: how many latent SDs are subtracted from the posterior mean.
POSTERIOR_RULES = {"pm": 0.0, "lcb1": 1.0, "lcb2": 2.0}
MODEL_FREE = ("random", "sobol")
REPRODUCTION_TOL = 1e-9
OUTPUT_NAME = "ship_rules_per_run"
PREFIX = "bo_sensor_error_"
# The surrogate variants this script cannot refit: their loop GP is not the
# default SingleTaskGP (noisy-input refit, Student-t variational GP).
UNSUPPORTED_VARIANT_PARTS = ("nigp", "studentt")

STEM = re.compile(r"^bo_sensor_error_(?P<dataset>.+)_(?P<objective>[a-z]+)_(?P<acq>[a-z]+)_seed(?P<seed>\d+)$")
JITTERED = re.compile(r"^exact_(?P<model>[a-z0-9]+)_jit(?P<onset>\d+)_std(?P<std>[0-9.]+)(?P<suffix>.*)$")

ID_COLUMNS = ["dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed",
              "baseline", "variant"]


class ReproductionError(RuntimeError):
    """The arm's own ship rule does not reproduce the logged deployed regret."""


# ---------------------------------------------------------------------------
# File index
# ---------------------------------------------------------------------------


def parse_run_name(name: str) -> dict | None:
    """Identify a per-run log from its file name, or None if it is not one.

    Only the synthetic ("exact") oracle is recognised: the refit needs the
    benchmark's box, which a fitted-oracle arm does not have. Baselines keep any
    suffix after "exact" as their variant ("inc-observed_max", "noise-known").
    """
    stem = name[:-4] if name.endswith(".csv") else name
    if "_baseline_" in stem:
        head, rest = stem.split("_baseline_", 1)
        s = STEM.match(head)
        if not s or not (rest == "exact" or rest.startswith("exact_")):
            return None
        return {"dataset": s["dataset"], "acquisition": s["acq"], "seed": int(s["seed"]),
                "objective": s["objective"], "baseline": True, "error_model": "none",
                "jitter_std": 0.0, "jitter_iteration": np.nan, "variant": rest[len("exact"):].lstrip("_")}
    if "_jittered_" in stem:
        head, rest = stem.split("_jittered_", 1)
        s, m = STEM.match(head), JITTERED.match(rest)
        if not s or not m:
            return None
        return {"dataset": s["dataset"], "acquisition": s["acq"], "seed": int(s["seed"]),
                "objective": s["objective"], "baseline": False, "error_model": m["model"],
                "jitter_std": float(m["std"]), "jitter_iteration": int(m["onset"]),
                "variant": m["suffix"].lstrip("_")}
    return None


def parse_seeds(value: str | None) -> set[int] | None:
    """'7-16' or '7,8,9' (or a mix) to a set of seeds."""
    if not value:
        return None
    seeds: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"empty seed range {part!r}")
            seeds.update(range(lo, hi + 1))
        else:
            seeds.add(int(part))
    return seeds


def _csv_set(value: str | None) -> set[str] | None:
    return {v.strip() for v in value.split(",") if v.strip()} if value else None


def index_runs(input_dir: Path, datasets: set[str] | None = None, acquisitions: set[str] | None = None,
               seeds: set[int] | None = None, error_models: set[str] | None = None) -> list[dict]:
    """Every per-run log under <input_dir>/<landscape>/ that passes the filters.

    Clean baselines are kept whenever their landscape, acquisition and seed pass:
    they are the twin of every error model. Without an acquisition list the
    model-free floors are left out, as in every recovery analysis.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"no such arm directory: {input_dir}")
    tasks = []
    for entry in sorted(os.scandir(input_dir), key=lambda e: e.name):
        if not entry.is_dir() or entry.name == "analysis":
            continue
        if datasets and entry.name not in datasets:
            continue
        for f in sorted(os.scandir(entry.path), key=lambda e: e.name):
            if not (f.name.startswith(PREFIX) and f.name.endswith(".csv")):
                continue
            info = parse_run_name(f.name)
            if info is None or info["dataset"] != entry.name:
                continue
            if acquisitions is not None:
                if info["acquisition"] not in acquisitions:
                    continue
            elif info["acquisition"] in MODEL_FREE:
                continue
            if seeds is not None and info["seed"] not in seeds:
                continue
            if not info["baseline"] and error_models is not None and info["error_model"] not in error_models:
                continue
            tasks.append({**info, "path": f.path, "file": f"{entry.name}/{f.name}"})
    return tasks


def arm_settings(input_dir: Path, rule_override: str = "auto") -> dict:
    """The arm's logged ship rule and surrogate, from run_metadata.json.

    Read through the simulator's own adaptation_fields so the rule is decided
    exactly as the driver decided it (replication defaults to best_mean).
    """
    meta_path = input_dir / "run_metadata.json"
    args: dict = {}
    if meta_path.is_file():
        args = json.loads(meta_path.read_text(encoding="utf-8")).get("args", {}) or {}
    import bo_sensor_error_simulation as sim

    # None in the metadata means "the default", which is what getattr gives.
    fields = sim.adaptation_fields(argparse.Namespace(**{k: v for k, v in args.items() if v is not None}))
    if args.get("multi_objective"):
        raise NotImplementedError(f"{input_dir} is a multi-objective arm; ship rules here are scalar.")
    if fields["input_noise_model"] != "none" or fields["likelihood"] != "gaussian":
        raise NotImplementedError(
            f"{input_dir}: the loop's surrogate is not the default SingleTaskGP "
            f"(input_noise_model={fields['input_noise_model']}, likelihood={fields['likelihood']}); "
            "refitting a different surrogate would re-score a different process.")
    rule = fields["inference_rule"] if rule_override == "auto" else rule_override
    if rule not in ("best_observed", "best_mean"):
        raise ValueError(f"unknown inference rule {rule!r}")
    return {"inference_rule": rule, "source": "run_metadata.json" if meta_path.is_file() else
            "default (no run_metadata.json)" if rule_override == "auto" else "--inference-rule",
            "observation_noise": args.get("observation_noise", "learned") or "learned"}


# ---------------------------------------------------------------------------
# The refit and the rules
# ---------------------------------------------------------------------------


def file_seed(file: str) -> int:
    """The torch seed of one log, fixed by its name (fit retries draw from it)."""
    return zlib.crc32(Path(file).name.encode("utf-8")) & 0x7FFFFFFF


def fit_loop_gp(X: np.ndarray, y: np.ndarray, bounds_low: np.ndarray, bounds_high: np.ndarray,
                yvar: np.ndarray | None = None, seed: int = 0):
    """Refit the loop's surrogate; returns (model, fit diagnostics).

    Built exactly as run_simulation builds it for a single objective, so no
    kernel, prior or transform is chosen here.
    """
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    import bo_sensor_error_simulation as sim

    torch.manual_seed(seed)
    bounds_tensor = sim.Bounds(low=np.asarray(bounds_low, float), high=np.asarray(bounds_high, float)).tensor
    train_X = torch.tensor(np.asarray(X, float), dtype=torch.double)
    train_Y = torch.tensor(np.asarray(y, float).reshape(-1, 1), dtype=torch.double)
    train_Yvar = None if yvar is None else torch.tensor(np.asarray(yvar, float).reshape(-1, 1), dtype=torch.double)
    gp = SingleTaskGP(
        train_X, train_Y, train_Yvar=train_Yvar,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    error = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit_gpytorch_mll(mll)
        except Exception as exc:  # noqa: BLE001 -- recorded per run, never silently absorbed
            error = f"{type(exc).__name__}: {exc}"[:300]
    noise_sd = np.nan
    if yvar is None:
        noise_sd = float((gp.likelihood.noise.detach().reshape(-1)[0]
                          * gp.outcome_transform.stdvs.detach().reshape(-1)[0] ** 2).sqrt())
    return gp, {"gp_fit_ok": not error, "gp_fit_error": error, "gp_n_warnings": len(caught),
                "gp_noise_sd": noise_sd, "gp_kernel": type(gp.covar_module).__name__}


def latent_posterior(gp, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and latent (noise-free) SD at X, in the objective's units."""
    import torch

    gp.eval()
    with torch.no_grad():
        post = gp.posterior(torch.tensor(np.asarray(X, float), dtype=torch.double))
        mu = post.mean.reshape(-1).cpu().numpy()
        sd = post.variance.clamp_min(0).sqrt().reshape(-1).cpu().numpy()
    return mu, sd


def select_indices(X: np.ndarray, observed: np.ndarray, score: np.ndarray,
                   mu: np.ndarray | None, sd: np.ndarray | None) -> dict[str, int]:
    """Row index (into the given arrays) each rule ships; -1 where it cannot choose.

    best_mean uses the simulator's own grouping of repeated designs, so a
    re-scored replication arm ties and rounds exactly as its log did.
    """
    import bo_sensor_error_simulation as sim

    X = np.asarray(X, float)
    idx = {
        "best_observed": int(np.argmax(observed)),
        "best_mean": int(sim._best_mean_index(list(X), [float(v) for v in observed])),
        "best_visited": int(np.argmax(score)),
    }
    for rule, k in POSTERIOR_RULES.items():
        crit = None if mu is None or sd is None else np.asarray(mu, float) - k * np.asarray(sd, float)
        # A non-finite posterior cannot rank designs; the rule is left undecided
        # (NaN regret) and the analysis refuses to average over it.
        idx[rule] = int(np.argmax(crit)) if crit is not None and np.all(np.isfinite(crit)) else -1
    return idx


def rescore_frame(df: pd.DataFrame, info: dict, expected_rule: str, seed: int) -> dict:
    """One per-run row: the regret of every rule, with the reproduction check."""
    if "objective" in df.columns and (df["objective"].astype(str) == "multi_objective").any():
        raise NotImplementedError(f"{info['file']}: multi-objective log; ship rules here are scalar.")
    if any(part in str(info.get("variant", "")).split("_") for part in UNSUPPORTED_VARIANT_PARTS):
        raise NotImplementedError(f"{info['file']}: variant {info['variant']!r} does not use the default GP.")
    dataset = info["dataset"]
    if dataset not in bb.BENCHMARKS:
        raise KeyError(f"{info['file']}: {dataset!r} is not a BOBA benchmark; its box is unknown.")
    spec = bb.BENCHMARKS[dataset]
    cols = str(df["param_columns"].iloc[0]).split(",")
    if cols != spec.param_columns:
        raise ValueError(f"{info['file']}: param columns {cols} are not {dataset}'s {spec.param_columns}")
    # The log must be the run its name says it is.
    checks = {"acquisition": info["acquisition"], "seed": info["seed"], "dataset": dataset}
    if not info["baseline"]:
        checks.update(jitter_iteration=info["jitter_iteration"], jitter_std=info["jitter_std"])
    for col, want in checks.items():
        if col in df.columns:
            got = df[col].iloc[0]
            same = abs(float(got) - float(want)) < 1e-12 if col == "jitter_std" else str(got) == str(want)
            if not same:
                raise ValueError(f"{info['file']}: column {col}={got!r} but the file name says {want!r}")

    X_all = df[cols].to_numpy(dtype=float)
    observed_all = df["objective_observed"].to_numpy(dtype=float)
    score_column = "objective_true_deployed" if "objective_true_deployed" in df.columns else "objective_true"
    score_all = df[score_column].to_numpy(dtype=float)
    y_opt = float(df["y_opt"].iloc[0])
    rows = np.flatnonzero(np.isfinite(observed_all))
    if len(rows) < 2:
        raise ValueError(f"{info['file']}: fewer than two finite observations")
    X, observed, score = X_all[rows], observed_all[rows], score_all[rows]
    known = "observation_noise" in df.columns and str(df["observation_noise"].iloc[0]) == "known"
    yvar = df["known_noise_var"].to_numpy(dtype=float)[rows] if known else None

    t0 = time.perf_counter()
    gp, diag = fit_loop_gp(X, observed, spec.bounds_low, spec.bounds_high, yvar=yvar, seed=seed)
    mu, sd = latent_posterior(gp, X)
    fit_sec = time.perf_counter() - t0
    idx = select_indices(X, observed, score, mu, sd)

    regret = {rule: (y_opt - float(score[i])) if i >= 0 else np.nan for rule, i in idx.items()}
    logged = float(df["inference_simple_regret_true"].iloc[-1])
    expected = regret[expected_rule]
    if not abs(expected - logged) <= REPRODUCTION_TOL:
        raise ReproductionError(
            f"{info['file']}: the arm's ship rule {expected_rule} gives regret {expected!r}, the log says "
            f"{logged!r} (best_observed {regret['best_observed']!r}, best_mean {regret['best_mean']!r}). "
            "The log and the re-scoring disagree about what was shipped.")
    return {
        **{k: info[k] for k in ID_COLUMNS}, "file": info["file"], "y_opt": y_opt,
        "n_trials": int(len(df)), "n_train": int(len(rows)), "score_column": score_column,
        "logged_rule": expected_rule, "logged_regret": logged,
        "best_observed_reproduces": bool(abs(regret["best_observed"] - logged) <= REPRODUCTION_TOL),
        **{f"regret_{rule}": regret[rule] for rule in RULES},
        # Indices into the log's rows, so any pick can be audited against the CSV.
        **{f"idx_{rule}": int(rows[i]) if i >= 0 else -1 for rule, i in idx.items()},
        "gp_known_noise": bool(known), "gp_seed": int(seed), "fit_sec": float(fit_sec), **diag,
    }


def rescore_file(task: dict) -> dict:
    """Worker entry point: read one log and re-score it."""
    df = pd.read_csv(task["path"], float_precision="round_trip")
    return rescore_frame(df, task, task["expected_rule"], file_seed(task["file"]))


def _init_worker() -> None:
    import torch

    # One thread per worker: the job is many small fits, and --workers is the
    # only knob that should decide how much CPU it takes.
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Journal (resume)
# ---------------------------------------------------------------------------


def read_journal(path: Path) -> list[dict]:
    """Complete records of an earlier run; a line cut off by a crash is dropped."""
    if not path.is_file():
        return []
    records, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        # Rewrite without the broken line, or the next append would glue a
        # record onto it and lose both.
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        print(f"  dropped {bad} incomplete journal line(s) from {path.name}")
    return records


def write_table(records: list[dict], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(records)
    if table.empty:
        return table
    table = table.drop_duplicates("file", keep="last")
    table = table.sort_values(["dataset", "acquisition", "seed", "baseline", "error_model", "variant",
                               "jitter_std", "jitter_iteration"], na_position="first").reset_index(drop=True)
    table.to_csv(path, index=False)
    return table


def report(table: pd.DataFrame, settings: dict) -> None:
    if table.empty:
        print("no runs re-scored")
        return
    n = len(table)
    print(f"\n{n:,} runs ({int(table['baseline'].sum()):,} clean) from {table['dataset'].nunique()} landscapes; "
          f"arm ship rule {settings['inference_rule']} ({settings['source']}), reproduced to "
          f"{REPRODUCTION_TOL:g} in every run")
    if settings["inference_rule"] != "best_observed":
        differs = int((~table["best_observed_reproduces"].astype(bool)).sum())
        print(f"  the arm does not ship by best_observed: best_observed differs from the logged pick in "
              f"{differs:,} of {n:,} runs")
    fails = int((~table["gp_fit_ok"].astype(bool)).sum())
    undecided = int(table[[f"regret_{r}" for r in POSTERIOR_RULES]].isna().any(axis=1).sum())
    print(f"  GP refit failures {fails:,}; runs with fit warnings {int((table['gp_n_warnings'] > 0).sum()):,}; "
          f"runs a posterior rule could not decide {undecided:,}; median fit {table['fit_sec'].median():.2f}s")
    for rule in RULES[1:]:
        same = float((table[f"idx_{rule}"] == table["idx_best_observed"]).mean())
        print(f"  {rule:<13} ships the best_observed design in {same:6.1%} of runs; "
              f"mean regret {table[f'regret_{rule}'].mean():+.4f} vs {table['regret_best_observed'].mean():+.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="an arm directory, e.g. output-boba")
    p.add_argument("--output-dir", type=Path, default=None, help="default: <input-dir>/analysis")
    p.add_argument("--acquisitions", type=str, default=None,
                   help="comma-separated; default every model-based acquisition found")
    p.add_argument("--seeds", type=str, default=None, help="e.g. 7-16 or 7,8,9")
    p.add_argument("--datasets", type=str, default=None, help="comma-separated landscapes (default all)")
    p.add_argument("--error-models", type=str, default=None,
                   help="comma-separated error models as in the file names (gaussian, bias, slip, ...); "
                        "clean baselines are always kept")
    p.add_argument("--workers", type=int, default=4, help="worker processes (capped at the CPU count)")
    p.add_argument("--chunksize", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="skip runs already in the journal")
    p.add_argument("--overwrite", action="store_true", help="discard an existing journal and start again")
    p.add_argument("--inference-rule", choices=("auto", "best_observed", "best_mean"), default="auto",
                   help="the arm's logged ship rule; auto reads run_metadata.json")
    p.add_argument("--max-files", type=int, default=None, help="stop after this many new runs (timing)")
    args = p.parse_args(argv)
    if args.resume and args.overwrite:
        p.error("--resume and --overwrite contradict each other")
    if args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    input_dir = args.input_dir
    out_dir = args.output_dir or (input_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / f"{OUTPUT_NAME}.jsonl"
    table_path = out_dir / f"{OUTPUT_NAME}.csv"

    settings = arm_settings(input_dir, args.inference_rule)
    tasks = index_runs(input_dir, _csv_set(args.datasets), _csv_set(args.acquisitions),
                       parse_seeds(args.seeds), _csv_set(args.error_models))
    if not tasks:
        raise SystemExit(f"no per-run logs under {input_dir} match the filters")

    if journal.exists() and not (args.resume or args.overwrite):
        raise SystemExit(f"{journal} exists: pass --resume to continue it or --overwrite to start again")
    if args.overwrite and journal.exists():
        journal.unlink()
    records = read_journal(journal) if args.resume else []
    done = {r["file"] for r in records}
    todo = [t for t in tasks if t["file"] not in done]
    if args.max_files is not None:
        todo = todo[: args.max_files]
    for t in todo:
        t["expected_rule"] = settings["inference_rule"]
    workers = max(1, min(args.workers, os.cpu_count() or 1, len(todo) or 1))
    print(f"{input_dir}: {len(tasks):,} runs match, {len(done & {t['file'] for t in tasks}):,} already done, "
          f"{len(todo):,} to re-score with {workers} worker(s); ship rule to reproduce: "
          f"{settings['inference_rule']} ({settings['source']})", flush=True)

    t0 = time.time()
    with open(journal, "a", encoding="utf-8") as fh:
        def _emit(i: int, rec: dict) -> None:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            records.append(rec)
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  {i + 1:,}/{len(todo):,}  {rate:.2f} runs/s  eta {(len(todo) - i - 1) / rate / 60:.1f} min",
                      flush=True)

        if workers == 1:
            _init_worker()
            for i, task in enumerate(todo):
                _emit(i, rescore_file(task))
        elif todo:
            from multiprocessing import get_context

            with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
                for i, rec in enumerate(pool.imap_unordered(rescore_file, todo, chunksize=args.chunksize)):
                    _emit(i, rec)

    table = write_table(records, table_path)
    report(table, settings)
    print(f"Wrote {table_path} ({time.time() - t0:.0f}s)")
    return table


if __name__ == "__main__":
    main()
