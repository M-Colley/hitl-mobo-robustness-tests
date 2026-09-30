"""Rerun one logged run per arm with the current code and compare it with the log.

The recorded code provenance of most arms is unreliable: the synthetic driver
stamped only ``git rev-parse HEAD``, and for weeks HEAD did not contain the
simulator that was running (it was uncommitted), so ``git_commit`` in an arm's
run_metadata.json does not say which code produced it. This script is the
evidence that replaces that field. For each arm it

  1. reads the arm's run_metadata.json (the settings of its LAST invocation;
     a run's variant lives only in its file name, see AGENTS.md),
  2. picks one logged noisy run whose settings are recoverable from that
     metadata and its file name: the file name the current code gives those
     settings must exist in the arm's directory (for the instrument arms, whose
     logs predate the clip/round parts of the name, the legacy name is used and
     reported),
  3. reruns exactly that run with the current code into --out-dir, and
  4. compares the new per-trial CSV with the logged one, column by column,
     first exactly and then with np.allclose(rtol, atol).

run_id (a fresh uuid per run) and fit_time_sec (wall clock) are excluded; a
column present in only one of the two files is reported, not compared. The
status per arm is ``exact``, ``within tolerance`` or ``differs`` (with the
first differing column in file order and the first differing trial), or
``unrecoverable`` when no logged run matches the recorded settings.

A run that differs is retried once with the one change that is known to have
moved since some arms ran, and the verdict says whether that explains it:

  * synthetic arms: the acquisition order before qkg and replei were inserted
    (it seeds the noise of the hypervolume family and the model-free floors);
  * fitted-oracle arms: a dataset config given with --dataset-config-override
    (the -prefix arms read the configs from before the 2026-09-23 data fixes,
    which run_metadata.json records only in part); ``git:REV`` takes the arm's
    entry of datasets.json at commit REV.

--prefer-acq random,sobol picks a model-free floor first, which checks the
acquisitions whose noise seed moved with that insertion.

Arms: every output-boba*/ directory with a run_metadata.json (synthetic
driver), every output-fitted*/<dataset>/ directory (fitted-oracle driver),
for each output-oracle-iso*/ arm its runs/iso_branin/ directory (the
fitted-oracle driver on a synthetic archival dataset), and the comparison loop
(scripts/elicitation_compare.py): output-elicitation-rerun/, the output of the
command recorded for it in paper/COMMANDS.md, and the older log in
output-elicitation/, which came from a code state the repository does not hold.
The comparison loop has no run_metadata.json and logs one row per run, not per
trial: its settings are the recorded command (``--functions`` resolved as
elicitation_compare.landscape_names resolves it, so ``suite`` is the twenty
landscapes of boba_benchmarks.DEFAULT_SUITE), and the check reruns the four
runs of its first cell (first landscape, seed, fault and magnitude; rating and
comparison loops, noisy and clean) and compares those rows, wall-clock seconds
excluded.

The code the reruns used is recorded with the report: code_stamp is taken when
the check starts and again when it ends (commit, sha256 of the uncommitted diff
of scripts/, the dataset configs and the landscape statistics, dirty files),
both stamps go into <report>.stamp.json, the start stamp into columns checker_*
of the CSV and a header of the .md. The files are also hashed one by one at
both ends: checker_code_unchanged says that no file a rerun imported (and no
dataset config or statistics file) changed during the check, and
checker_tree_unchanged that the whole diff did not, which other agents editing
analysis scripts in the same checkout can break without touching the reruns.
The patch itself goes to the git-ignored store of code_diff_store().

  python scripts/check_provenance.py --out-dir <scratch> --workers 6 \\
      --dataset-config-override "output-fitted*-prefix/*=git:3d2323aed~1" \\
      --report output-boba/analysis/review/provenance_check.csv
  python scripts/check_provenance.py --out-dir <scratch2> --workers 6 \\
      --prefer-acq random,sobol --arms "output-boba,output-boba-mo,..." \\
      --report output-boba/analysis/review/provenance_check_random.csv
  python scripts/check_provenance.py --out-dir <scratch3> --workers 1 \\
      --arms output-elicitation-rerun --elicitation-scope clean \\
      --report output-boba/analysis/review/provenance_check_elicitation_rerun.csv
  (--arms output-elicitation with --report .../provenance_check_elicitation.csv
  checks the older log instead.)

Nothing under the arms' directories is written; every rerun goes to --out-dir.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fnmatch
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

EXCLUDED_COLUMNS = ("run_id", "fit_time_sec")
RTOL = 1e-6
ATOL = 1e-9

# Cheapest informative choices first: a model-based acquisition that reads
# best_f, then the others, then the model-free floors.
ACQ_PREFERENCE = ["logei", "ei", "ucb", "qnei", "qei", "logpi", "pi", "qucb", "qpi", "greedy",
                  "aei", "ts", "qkg", "replei", "qlognehvi", "qlogehvi", "qnehvi", "qehvi",
                  "random", "sobol"]
ISO_LANDSCAPE = "iso_branin"
FITTED_DATASETS = ("ehmi", "opticarvis", "provoice")

# The comparison loop (scripts/elicitation_compare.py): one row per run, keyed by
# these columns; its wall-clock time is not compared.
ELICITATION_ARM = "output-elicitation-rerun"   # the recorded command's --output-dir
ELICITATION_LEGACY_ARM = "output-elicitation"  # older log; its code state is not in the repository
ELICITATION_RUNS = "elicitation_runs.csv"
ELICITATION_SCRIPT = "scripts/elicitation_compare.py"
ELICITATION_KEYS = ["dataset", "elicitation", "error_model", "magnitude", "seed", "apply_error"]
ELICITATION_EXCLUDED = ("seconds",)
COMMANDS_MD = REPO_ROOT / "paper" / "COMMANDS.md"

# ACQUISITION_CHOICES before qkg and replei were inserted at positions 10-11
# (2026-09-10, tests/test_acquisition_seed_order.py). The position seeds a
# jittered run's noise, so a run made before the insertion with a hypervolume
# acquisition or a model-free floor draws different noise under the current
# order. A synthetic run that differs is retried under this order.
ACQUISITION_ORDER_BEFORE_ROBUST = [
    "logei", "logpi", "ei", "pi", "ucb", "qucb", "qei", "qpi", "qnei", "greedy",
    "qehvi", "qnehvi", "qlogehvi", "qlognehvi",
    "random", "sobol",
]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_runs(logged_path: Path, new_path: Path, rtol: float = RTOL, atol: float = ATOL,
                 excluded: tuple[str, ...] = EXCLUDED_COLUMNS) -> dict:
    """Compare two per-trial CSVs. Returns status, first differing column and trial."""
    import numpy as np
    import pandas as pd

    logged = pd.read_csv(logged_path, float_precision="round_trip")
    new = pd.read_csv(new_path, float_precision="round_trip")
    out: dict[str, object] = {
        "n_rows_logged": int(len(logged)),
        "n_rows_new": int(len(new)),
        "columns_only_logged": ";".join(c for c in logged.columns if c not in new.columns),
        "columns_only_new": ";".join(c for c in new.columns if c not in logged.columns),
        "first_diff_column": "",
        "first_diff_trial": "",
        "max_abs_diff": 0.0,
        "n_diff_columns": 0,
        "n_diff_rows": 0,
    }
    columns = [c for c in logged.columns if c in new.columns and c not in excluded]
    out["n_columns_compared"] = len(columns)
    if len(logged) != len(new):
        out["status"] = "differs"
        out["first_diff_column"] = "<row count>"
        return out
    exact = True
    close = True
    first_col, first_row = None, None
    max_abs = 0.0
    n_diff = 0
    row_differs = np.zeros(len(logged), dtype=bool)
    for col in columns:
        a, b = logged[col], new[col]
        numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        if numeric:
            x = a.to_numpy(dtype=float)
            y = b.to_numpy(dtype=float)
            same = (x == y) | (np.isnan(x) & np.isnan(y))
            near = np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                max_abs = max(max_abs, float(np.max(np.abs(x[finite] - y[finite]))))
        else:
            same = (a.astype(str).to_numpy() == b.astype(str).to_numpy())
            near = same
        row_differs |= ~same
        if not same.all():
            exact = False
            n_diff += 1
            row = int(np.argmin(same))
            if first_col is None:
                first_col, first_row = col, row
        if not near.all():
            close = False
    out["max_abs_diff"] = max_abs
    out["n_diff_columns"] = n_diff
    out["n_diff_rows"] = int(row_differs.sum())
    if first_col is not None:
        trial = logged["iteration"].iloc[first_row] if "iteration" in logged.columns else first_row
        out["first_diff_column"] = first_col
        out["first_diff_trial"] = int(trial)
        out["first_diff_logged"] = str(logged[first_col].iloc[first_row])
        out["first_diff_new"] = str(new[first_col].iloc[first_row])
    out["status"] = "exact" if exact else ("within tolerance" if close else "differs")
    return out


# ---------------------------------------------------------------------------
# Namespaces from recorded metadata
# ---------------------------------------------------------------------------


def _capture_parser(parse):
    """Call a driver's parse function and return (parser, defaults namespace)."""
    import argparse as _argparse

    captured = {}
    original = _argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        captured["parser"] = self
        return original(self, args, namespace)

    _argparse.ArgumentParser.parse_args = spy
    try:
        defaults = parse()
    finally:
        _argparse.ArgumentParser.parse_args = original
    return captured["parser"], defaults


def namespace_from_metadata(parser, defaults: argparse.Namespace, recorded: dict) -> argparse.Namespace:
    """The driver's defaults overlaid with the recorded args, path options as Path.

    A flag added after the arm ran is absent from its record and keeps its
    default, which is off for every flag that changes a run.
    """
    path_dests = {a.dest for a in parser._actions if a.type is Path or isinstance(a.default, Path)}
    ns = argparse.Namespace(**vars(defaults))
    for key, value in (recorded or {}).items():
        if key in path_dests and isinstance(value, str):
            value = Path(value)
        setattr(ns, key, value)
    return ns


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Check:
    arm: str
    driver: str                  # "synthetic" or "sensor"
    arm_dir: str                 # directory holding run_metadata.json
    logged_path: str = ""
    name_note: str = ""
    params: dict = dataclasses.field(default_factory=dict)
    recorded_commit: str = ""
    recorded_dirty: object = None
    n_invocations: int = 0
    note: str = ""


def _ordered(values, preference):
    values = list(values)
    ranked = [v for v in preference if v in values]
    return ranked + [v for v in values if v not in ranked]


def _std_order(stds):
    return sorted(stds, key=lambda s: (abs(math.log(float(s))) if float(s) > 0 else 99.0, float(s)))


def _onset_order(onsets):
    return sorted(onsets, key=lambda t: (0 if int(t) == 20 else 1, int(t)))


def _metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stamp_fields(meta: dict) -> dict:
    invocations = meta.get("invocations") or []
    return {
        "recorded_commit": str(meta.get("git_commit") or "")[:12],
        "recorded_dirty": meta.get("code_dirty"),
        "n_invocations": len(invocations),
    }


def synthetic_candidates(arm_dir: Path, meta: dict, prefer: list[str] = ACQ_PREFERENCE):
    """Yield (params, logged path, name note) for the recorded settings, best first."""
    import bo_synthetic_error_simulation as synth

    sim = synth.sim
    parser, defaults = _capture_parser(lambda: synth.parse_args([]))
    ns = namespace_from_metadata(parser, defaults, meta.get("args") or {})
    multi = bool(getattr(ns, "multi_objective", False))
    dims = (lambda f: synth.mob.MO_BENCHMARKS[f].dim) if multi else (lambda f: synth.bb.BENCHMARKS[f].dim)
    functions = sorted(meta.get("functions") or [], key=lambda f: (dims(f), f != "branin", f))
    objective = "multi_objective" if multi else synth.OBJECTIVE_NAME
    legacy_ns = argparse.Namespace(**{**vars(ns), "response_clip": "none", "response_round": None})
    for seed in sorted(meta.get("seeds") or []):
        for function in functions:
            for acq in _ordered(meta.get("acquisitions") or [], prefer):
                for error_model in meta.get("error_models") or []:
                    for std in _std_order(meta.get("jitter_stds") or []):
                        for onset in _onset_order(meta.get("jitter_iterations") or []):
                            bias = float(std) if ns.error_bias_mode == "scaled" else ns.error_bias
                            spike = float(std) if ns.error_spike_std_mode == "scaled" else ns.error_spike_std
                            channel = sim.run_error_label(
                                "none" if ns.input_error_from_sweep else error_model, ns.input_error)
                            folder = arm_dir / function if ns.per_function_dirs else arm_dir
                            stem = (f"bo_sensor_error_{function}_{objective}_{acq}_seed{seed}"
                                    f"_jittered_{synth.ORACLE_TAG}_{channel}_jit{onset}_std{float(std)}")
                            params = {"function": function, "seed": int(seed), "acq": acq,
                                      "error_model": error_model, "jitter_std": float(std),
                                      "jitter_iteration": int(onset)}
                            name = stem + synth._variant_suffix(ns, error_model, bias, spike) + ".csv"
                            if (folder / name).is_file():
                                yield params, folder / name, ""
                                continue
                            legacy = stem + synth._variant_suffix(legacy_ns, error_model, bias, spike) + ".csv"
                            if legacy != name and (folder / legacy).is_file():
                                yield params, folder / legacy, (
                                    "logged under the legacy name (clip/round not in the name); "
                                    "the rerun is named " + name)


def sensor_candidates(run_dir: Path, meta: dict, prefer: list[str] = ACQ_PREFERENCE):
    import bo_sensor_error_simulation as sim  # noqa: F401  (thread limits first)

    args = meta.get("args") or {}
    datasets = [d["name"] for d in meta.get("datasets") or []]
    for name in datasets:
        for objective in (meta.get("objectives") or {}).get(name, []):
            oracles = (meta.get("resolved_oracle_models") or {}).get(f"{name}:{objective}", [])
            for oracle in oracles:
                for seed in sorted(meta.get("seeds") or []):
                    for acq in _ordered(meta.get("acquisitions") or [], prefer):
                        for error_model in meta.get("error_models") or []:
                            for std in _std_order(meta.get("effective_jitter_stds") or []):
                                for onset in _onset_order(meta.get("effective_jitter_iterations") or []):
                                    parts = []
                                    if error_model == "bias" and args.get("error_bias", 0.2) != 0.2:
                                        parts.append(f"bias{args.get('error_bias')}")
                                    if error_model == "spike":
                                        parts.append(f"sp{args.get('error_spike_prob')}-{args.get('error_spike_std')}")
                                    if error_model == "ar1" and args.get("error_ar1_rho", 0.8) != 0.8:
                                        parts.append(f"rho{args.get('error_ar1_rho')}")
                                    if args.get("single_error"):
                                        parts.append("single")
                                    suffix = ("_" + "_".join(parts)) if parts else ""
                                    fname = (f"bo_sensor_error_{name}_{objective}_{acq}_seed{seed}_jittered_"
                                             f"{oracle}_{error_model}_jit{onset}_std{float(std)}{suffix}.csv")
                                    if (run_dir / fname).is_file():
                                        yield {"dataset": name, "objective": objective, "oracle": oracle,
                                               "seed": int(seed), "acq": acq, "error_model": error_model,
                                               "jitter_std": float(std), "jitter_iteration": int(onset)}, \
                                            run_dir / fname, ""


def recorded_elicitation_argv(commands_md: Path = COMMANDS_MD) -> list[str] | None:
    """The arguments of the one command paper/COMMANDS.md records as the comparison
    loop's producer (not its --summary-only reread), or None if there is not exactly one."""
    import shlex

    if not commands_md.is_file():
        return None
    lines = [line.strip() for line in commands_md.read_text(encoding="utf-8").splitlines()
             if line.strip().startswith(f"python {ELICITATION_SCRIPT}") and "--summary-only" not in line]
    if len(set(lines)) != 1:
        return None
    return shlex.split(lines[0])[2:]


ELICITATION_SCOPES = ("cell", "clean")


def elicitation_tasks(argv: list[str], scope: str = "cell",
                      logged_names: set[str] | None = None) -> tuple[dict, list[dict], list[str]]:
    """The runs of the recorded command to rerun, with a description and the
    landscapes the command itself covers.

    cell   the first cell (first landscape, seed, fault and magnitude): four runs,
           rating and comparison loops, noisy and clean;
    clean  the clean runs of the first seed of both loops on every landscape the
           command and the log share (a clean run is the same for every fault, so
           those of the first fault stand for all).
    """
    import elicitation_compare as ec

    if scope not in ELICITATION_SCOPES:
        raise ValueError(f"unknown elicitation scope {scope!r}")
    args = ec.parse_args(argv)
    stats = ec.bb.load_stats(args.stats_path)
    # 'all', 'suite' or a comma-separated list, resolved exactly as the script resolves it
    names = ec.landscape_names(args.functions, stats)
    tasks = ec.build_tasks(args, names)
    first = tasks[0]
    keys = ("dataset", "seed", "error_model", "magnitude")
    if scope == "cell":
        chosen = [t for t in tasks if all(t[k] == first[k] for k in keys)]
        desc = {k: first[k] for k in keys}
    else:
        shared = [n for n in names if logged_names is None or n in logged_names]
        chosen = [t for t in tasks if t["dataset"] in shared and not t["apply_error"]
                  and all(t[k] == first[k] for k in ("seed", "error_model", "magnitude"))]
        desc = {"dataset": f"{len(shared)} landscapes", "seed": first["seed"],
                "error_model": f"{first['error_model']} (clean runs)", "magnitude": first["magnitude"]}
    desc["iterations"] = args.iterations
    return desc, chosen, names


def elicitation_cell(argv: list[str]) -> tuple[dict, list[dict]]:
    """The first cell of the recorded command and its four tasks (scope 'cell')."""
    desc, tasks, _ = elicitation_tasks(argv, "cell")
    return desc, tasks


def _elicitation_frame(frame):
    return frame.sort_values(ELICITATION_KEYS, kind="mergesort").reset_index(drop=True)


def discover(arms_filter: list[str] | None = None) -> list[Check]:
    checks: list[Check] = []
    for meta_path in sorted(REPO_ROOT.glob("output-boba*/run_metadata.json")):
        arm_dir = meta_path.parent
        checks.append(Check(arm=arm_dir.name, driver="synthetic", arm_dir=str(arm_dir)))
    for arm_dir in sorted(p for p in REPO_ROOT.glob("output-fitted*") if p.is_dir()):
        for dataset in FITTED_DATASETS:
            if (arm_dir / dataset / "run_metadata.json").is_file():
                checks.append(Check(arm=f"{arm_dir.name}/{dataset}", driver="sensor",
                                    arm_dir=str(arm_dir / dataset)))
    for arm_dir in sorted(p for p in REPO_ROOT.glob("output-oracle-iso*") if p.is_dir()):
        run_dir = arm_dir / "runs" / ISO_LANDSCAPE
        if (run_dir / "run_metadata.json").is_file():
            checks.append(Check(arm=f"{arm_dir.name}/{ISO_LANDSCAPE}", driver="sensor", arm_dir=str(run_dir)))
    for arm in (ELICITATION_ARM, ELICITATION_LEGACY_ARM):
        if (REPO_ROOT / arm / ELICITATION_RUNS).is_file():
            checks.append(Check(arm=arm, driver="elicitation", arm_dir=str(REPO_ROOT / arm)))
    if arms_filter:
        checks = [c for c in checks if any(fnmatch.fnmatch(c.arm, pat) for pat in arms_filter)]
    return checks


def select_elicitation(check: Check, scope: str = "cell") -> Check:
    import pandas as pd

    argv = recorded_elicitation_argv()
    if argv is None:
        check.note = f"paper/COMMANDS.md does not record exactly one producing command of {ELICITATION_SCRIPT}"
        return check
    logged = pd.read_csv(Path(check.arm_dir) / ELICITATION_RUNS, float_precision="round_trip")
    logged_names = set(logged["dataset"])
    desc, tasks, names = elicitation_tasks(argv, scope, logged_names)
    keys = {tuple(t[k] for k in ELICITATION_KEYS) for t in tasks}
    hits = logged[[tuple(r) in keys for r in logged[ELICITATION_KEYS].itertuples(index=False, name=None)]]
    if not tasks or len(hits) != len(tasks):
        check.note = (f"the log holds {len(hits)} of the {len(tasks)} chosen runs of the "
                      "command recorded in paper/COMMANDS.md")
        return check
    check.params = {**desc, "scope": scope, "argv": " ".join(argv)}
    check.logged_path = str(Path(check.arm_dir) / ELICITATION_RUNS)
    what = ("rating and comparison loops, noisy and clean" if scope == "cell"
            else "rating and comparison loops, clean")
    check.name_note = (f"{len(tasks)} runs ({desc['dataset']}, seed {desc['seed']}, {desc['error_model']} at "
                       f"{desc['magnitude']}; {what}), settings from the command recorded in "
                       "paper/COMMANDS.md; wall-clock seconds not compared.")
    if set(names) != logged_names:
        check.note = (f"the recorded command covers {len(names)} landscapes, the log {len(logged_names)}; "
                      f"{len(set(names) - logged_names)} are in the command only, "
                      f"{len(logged_names - set(names))} in the log only.")
    return check


def select(check: Check, prefer: list[str] | None = None, elicitation_scope: str = "cell") -> Check:
    if check.driver == "elicitation":
        return select_elicitation(check, elicitation_scope)
    meta = _metadata(Path(check.arm_dir) / "run_metadata.json")
    for key, value in _stamp_fields(meta).items():
        setattr(check, key, value)
    prefer = list(prefer or []) + [a for a in ACQ_PREFERENCE if a not in (prefer or [])]
    generator = (synthetic_candidates if check.driver == "synthetic" else sensor_candidates)(
        Path(check.arm_dir), meta, prefer)
    for params, path, note in generator:
        check.params, check.logged_path, check.name_note = params, str(path), note
        return check
    check.note = "no logged run matches the settings recorded in run_metadata.json"
    return check


# ---------------------------------------------------------------------------
# Reruns (run in worker processes)
# ---------------------------------------------------------------------------


def _rerun_synthetic(check: Check, out_dir: Path, acquisition_order: list[str] | None = None) -> Path:
    import bo_synthetic_error_simulation as synth

    if acquisition_order is not None:
        # run_task seeds the noise from sim.ACQUISITION_CHOICES.index(acq)
        saved = synth.sim.ACQUISITION_CHOICES
        synth.sim.ACQUISITION_CHOICES = list(acquisition_order)
        try:
            return _rerun_synthetic(check, out_dir, None)
        finally:
            synth.sim.ACQUISITION_CHOICES = saved

    meta = _metadata(Path(check.arm_dir) / "run_metadata.json")
    parser, defaults = _capture_parser(lambda: synth.parse_args([]))
    ns = namespace_from_metadata(parser, defaults, meta.get("args") or {})
    ns.baseline_run = False
    ns.resume = False
    if ns.boba_root:
        synth.bb.set_boba_root(ns.boba_root)
    function = check.params["function"]
    if ns.multi_objective:
        stats = synth.mob.load_mo_stats(Path(ns.mo_stats_path))
    else:
        stats = synth.bb.load_stats(Path(ns.stats_path))
    entry = stats.get(function)
    recorded = (meta.get("landscape_stats") or {}).get(function)
    if entry is None:
        entry = recorded
        check.note += " landscape stats missing from the current file; the recorded ones were used."
    elif recorded is not None and json.dumps(recorded, sort_keys=True, default=str) != json.dumps(
            entry, sort_keys=True, default=str):
        check.note += " the current landscape stats differ from the recorded ones."
    target = out_dir / function
    target.mkdir(parents=True, exist_ok=True)
    synth.run_task(synth.Task(function=function, seed=check.params["seed"]), ns, entry,
                   [check.params["acq"]], [check.params["error_model"]], [check.params["jitter_std"]],
                   [check.params["jitter_iteration"]], target, None)
    produced = sorted(target.glob("*_jittered_*.csv"))
    if len(produced) != 1:
        raise RuntimeError(f"expected one rerun file in {target}, found {len(produced)}")
    return produced[0]


def recorded_dataset_changes(meta: dict, ns: argparse.Namespace) -> list[str]:
    """The recorded dataset fields (run_metadata.json 'datasets') that the dataset
    config the run names no longer has. column_ranges and oracle_target are not
    recorded, so a change there cannot be seen."""
    import bo_sensor_error_simulation as sim

    try:
        current = {d.name: d for d in sim.parse_dataset_configs(None, Path(ns.dataset_config),
                                                                Path(ns.dataset_cache_dir))}
    except Exception as exc:
        return [f"config unreadable ({type(exc).__name__})"]
    changed = []
    for entry in meta.get("datasets") or []:
        now = current.get(entry.get("name"))
        if now is None:
            changed.append(f"{entry.get('name')}: missing")
            continue
        for field in ("param_columns", "objective_map", "observation_glob"):
            if field in entry and entry[field] != getattr(now, field):
                changed.append(f"{entry['name']}.{field}")
    return changed


def materialise_dataset_config(spec: str, dataset: str, out_dir: Path) -> Path:
    """A dataset config path from an override: a file as given, or, for
    ``git:REV``, the entry of ``git show REV:datasets.json`` named ``dataset``
    written on its own (as make_per_dataset_configs.py splits the file, entry
    whole) into out_dir."""
    if not spec.startswith("git:"):
        return Path(spec)
    import subprocess

    rev = spec[len("git:"):]
    text = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{rev}:datasets.json"],
                          capture_output=True, text=True, check=True).stdout
    entries = [e for e in json.loads(text) if e.get("name") == dataset]
    if len(entries) != 1:
        raise ValueError(f"datasets.json at {rev} has {len(entries)} entries named {dataset!r}")
    safe = "".join(ch if ch.isalnum() else "-" for ch in rev)
    path = out_dir / f"datasets-{dataset}-at-{safe}.json"
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path


def _rerun_sensor(check: Check, out_dir: Path, dataset_config: str | None = None) -> Path:
    import bo_sensor_error_simulation as sim

    meta = _metadata(Path(check.arm_dir) / "run_metadata.json")
    if dataset_config is not None:
        dataset_config = str(materialise_dataset_config(dataset_config, check.params["dataset"], out_dir))
    saved_argv = sys.argv
    sys.argv = ["bo_sensor_error_simulation.py"]
    try:
        parser, defaults = _capture_parser(sim.parse_args)
    finally:
        sys.argv = saved_argv
    ns = namespace_from_metadata(parser, defaults, meta.get("args") or {})
    if dataset_config is not None:
        ns.dataset_config = Path(dataset_config)
    changed = recorded_dataset_changes(meta, ns)
    if changed:
        check.note = (check.note + " the dataset config now differs from the recorded dataset in: "
                      + ", ".join(changed) + ".").strip()
    p = check.params
    ns.acq, ns.acq_list = p["acq"], p["acq"]
    ns.seeds, ns.seed, ns.num_seeds = str(p["seed"]), p["seed"], 1
    ns.error_model, ns.error_models = None, p["error_model"]
    ns.jitter_std, ns.jitter_stds = None, repr(p["jitter_std"])
    ns.jitter_iteration, ns.jitter_iterations = None, str(p["jitter_iteration"])
    # the oracle the run recorded, named explicitly: an 'auto' selection file
    # may have been refreshed since (it names a model and nothing else)
    ns.oracle_model, ns.oracle_models = p["oracle"], None
    ns.objective, ns.objectives = p["objective"], None
    ns.output_dir = out_dir
    ns.resume, ns.parallel, ns.n_jobs, ns.baseline_run = False, False, 1, False
    original = sim.parse_args
    sim.parse_args = lambda: ns
    try:
        sim.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError(f"the simulator exited with {exc.code}") from exc
    finally:
        sim.parse_args = original
    produced = sorted(out_dir.glob("*_jittered_*.csv"))
    if len(produced) != 1:
        raise RuntimeError(f"expected one rerun file in {out_dir}, found {len(produced)}")
    return produced[0]


def _rerun_elicitation(check: Check, out_dir: Path) -> Path:
    """Rerun the chosen cell of the comparison loop as its driver runs a task (one
    torch thread, warnings off) and write the logged rows of that cell beside it."""
    import elicitation_compare as ec
    import pandas as pd

    logged = pd.read_csv(check.logged_path, float_precision="round_trip")
    _, tasks, _ = elicitation_tasks(recorded_elicitation_argv() or [], check.params.get("scope", "cell"),
                                    set(logged["dataset"]))
    ec._init_worker()
    new = _elicitation_frame(pd.DataFrame([ec.run_cell(task) for task in tasks]))
    keys = {tuple(t[k] for k in ELICITATION_KEYS) for t in tasks}
    logged = _elicitation_frame(
        logged[[tuple(r) in keys for r in logged[ELICITATION_KEYS].itertuples(index=False, name=None)]])
    logged.to_csv(out_dir / "logged_cell.csv", index=False)
    path = out_dir / "rerun_cell.csv"
    new[[c for c in logged.columns if c in new.columns] + [c for c in new.columns if c not in logged.columns]] \
        .to_csv(path, index=False)
    return path


def _retry(check: Check, overrides: dict[str, str]) -> tuple[str, dict] | None:
    """What to change for a second attempt at a run that differs, or None.

    A synthetic run whose acquisition sits at a different position in the order
    used before qkg and replei were inserted is retried under that order. A
    fitted-oracle run is retried with a dataset config given for its arm by
    --dataset-config-override (for instance the config of an older commit).
    """
    if check.driver == "elicitation":
        return None
    if check.driver == "synthetic":
        import bo_sensor_error_simulation as sim

        acq = check.params.get("acq")
        old = ACQUISITION_ORDER_BEFORE_ROBUST
        if acq in old and old.index(acq) != sim.ACQUISITION_CHOICES.index(acq):
            return ("acquisition order before qkg/replei were inserted",
                    {"acquisition_order": old})
        return None
    for pattern, spec in overrides.items():
        if fnmatch.fnmatch(check.arm, pattern):
            path, _, label = spec.partition("=")
            if not label:
                label = (f"datasets.json at {path[len('git:'):]}" if path.startswith("git:")
                         else Path(path).name)
            return (f"dataset config {label}", {"dataset_config": path})
    return None


def _attempt(check: Check, out_dir: Path, **kwargs) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rerun.log", "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), \
            contextlib.redirect_stderr(log):
        rerun = {"synthetic": _rerun_synthetic, "sensor": _rerun_sensor,
                 "elicitation": _rerun_elicitation}[check.driver]
        return rerun(check, out_dir, **kwargs)


def run_check(check: Check, out_root: str, overrides: dict[str, str] | None = None) -> dict:
    started = time.perf_counter()
    row = dataclasses.asdict(check)
    row.pop("params")
    row.update({f"param_{k}": v for k, v in check.params.items()})
    if not check.logged_path:
        row.update(status="unrecoverable", verdict="unrecoverable", runtime_sec=0.0)
        return row
    out_dir = Path(out_root) / check.arm.replace("/", "__")
    try:
        new_path = _attempt(check, out_dir)
        row["rerun_path"] = str(new_path)
        if check.driver == "elicitation":
            row.update(compare_runs(out_dir / "logged_cell.csv", new_path,
                                    excluded=EXCLUDED_COLUMNS + ELICITATION_EXCLUDED))
        else:
            if Path(check.logged_path).name != new_path.name:
                row["name_note"] = (row.get("name_note") or "") + " rerun file name differs from the logged one."
            row.update(compare_runs(Path(check.logged_path), new_path))
    except Exception as exc:  # reported, not raised: one broken arm must not hide the rest
        row.update(status="rerun failed", note=f"{check.note} {type(exc).__name__}: {exc}".strip(),
                   traceback=traceback.format_exc()[-2000:])
    row["verdict"] = row["status"]
    retry = _retry(check, overrides or {}) if row["status"] in ("differs", "rerun failed") else None
    if retry is not None:
        label, kwargs = retry
        row["retry"] = label
        try:
            second = _attempt(check, out_dir / "retry", **kwargs)
            result = compare_runs(Path(check.logged_path), second)
            row["retry_status"] = result["status"]
            row["retry_first_diff_column"] = result["first_diff_column"]
            row["retry_first_diff_trial"] = result["first_diff_trial"]
            row["retry_max_abs_diff"] = result["max_abs_diff"]
            if result["status"] in ("exact", "within tolerance"):
                row["verdict"] = f"{result['status']} with {label}"
        except Exception as exc:
            row["retry_status"] = f"rerun failed: {type(exc).__name__}: {exc}"
    row["note"] = (check.note or row.get("note") or "").strip()
    row["runtime_sec"] = round(time.perf_counter() - started, 1)
    row["_imported"] = imported_repo_files()
    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _relative(path: str) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _public_arg(arg: str) -> str:
    """A command-line argument fit for a tracked report: a path inside the repository
    relative to it, a path outside it reduced to <outside>/<name>."""
    if arg.startswith("~"):
        arg = str(Path.home()) + arg[1:]
    path = Path(arg)
    if not path.is_absolute():
        return arg
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return "<outside>/" + path.name


def checker_stamp() -> dict:
    """code_stamp of the code the reruns use, its patch written to the git-ignored store;
    argv holds no path outside the repository."""
    import bo_sensor_error_simulation as sim

    stamp = sim.code_stamp(REPO_ROOT, patch_dir=sim.code_diff_store(REPO_ROOT))
    stamp["argv"] = [_public_arg(str(a)) for a in sys.argv]
    return stamp


def file_hashes() -> dict[str, str]:
    """sha256 of every file code_stamp covers (the .py files under scripts/, the
    dataset configs, the landscape statistics), keyed by repository-relative path."""
    import hashlib

    import bo_sensor_error_simulation as sim

    out: dict[str, str] = {}
    for entry in sim.PROVENANCE_PATHS:
        path = REPO_ROOT / entry
        files = sorted(path.rglob("*.py")) if path.is_dir() else ([path] if path.is_file() else [])
        for f in files:
            if "__pycache__" in f.parts:
                continue
            try:
                out[f.relative_to(REPO_ROOT).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
            except OSError:
                out[f.relative_to(REPO_ROOT).as_posix()] = "unreadable"
    return out


def imported_repo_files() -> list[str]:
    """The repository files of the modules this process has imported."""
    found = set()
    for module in list(sys.modules.values()):
        name = (getattr(module, "__dict__", None) or {}).get("__file__")
        if not isinstance(name, str) or not Path(name).is_absolute():
            continue  # a relative __file__ (some extension modules) is not a repository file
        try:
            found.add(Path(name).resolve().relative_to(REPO_ROOT).as_posix())
        except (ValueError, OSError):
            continue
    return sorted(found)


def stamp_columns(start: dict, end: dict, changed_imported: list[str] | None = None) -> dict:
    """checker_code_unchanged: same commit, and no file a rerun imported changed
    during the check. checker_tree_unchanged: the whole uncommitted diff is the
    same at the end (other files may be edited concurrently)."""
    same_commit = start.get("git_commit") == end.get("git_commit")
    return {
        "checker_git_commit": start.get("git_commit") or "",
        "checker_code_diff_sha256": start.get("code_diff_sha256") or "",
        "checker_n_dirty_files": len(start.get("dirty_files") or []),
        "checker_code_unchanged": bool(same_commit and not (changed_imported or [])),
        "checker_tree_unchanged": bool(same_commit
                                       and start.get("code_diff_sha256") == end.get("code_diff_sha256")),
    }


def stamp_header(start: dict, end: dict, changed_imported: list[str] | None = None,
                 changed_other: list[str] | None = None) -> list[str]:
    cols = stamp_columns(start, end, changed_imported)
    dirty = start.get("dirty_files") or []
    lines = [f"Code of the reruns: commit {cols['checker_git_commit'] or 'unknown'} plus the uncommitted "
             f"diff of scripts/, the dataset configs and the landscape statistics, sha256 "
             f"{cols['checker_code_diff_sha256'] or 'unknown'} over {len(dirty)} dirty file(s)"
             + (f" (patch {start['code_diff_patch']}, git-ignored)" if start.get("code_diff_patch") else "")
             + f", stamped at {start.get('stamped_at')}."]
    if cols["checker_code_unchanged"]:
        lines.append("No file that a rerun imported changed while the check ran"
                     + ("." if cols["checker_tree_unchanged"] else
                        f"; {len(changed_other or [])} other file(s) under scripts/ did (listed in the "
                        f".stamp.json), so the end-of-check diff hash is {end.get('code_diff_sha256')}."))
    else:
        lines.append("CHANGED while the check ran: " + ", ".join(f"`{f}`" for f in changed_imported or [])
                     + (f"; commit at the end {end.get('git_commit')}" if end.get("git_commit")
                        != start.get("git_commit") else "") + ".")
    if dirty:
        lines += ["", "Dirty files at the start:"] + [f"- `{line}`" for line in dirty]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="where the reruns are written")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", type=str, default="", help="comma-separated fnmatch patterns over arm names")
    ap.add_argument("--report", type=Path, default=None,
                    help="CSV report (default <out-dir>/provenance_check.csv); a .md table is written beside it")
    ap.add_argument("--list", action="store_true", help="print the selected runs and exit")
    ap.add_argument("--prefer-acq", type=str, default="",
                    help="comma-separated acquisitions to pick first (default: LogEI first, floors last)")
    ap.add_argument("--dataset-config-override", action="append", default=[], metavar="PATTERN=PATH[=LABEL]",
                    help="retry a fitted-oracle arm that differs with this dataset config; PATH may be "
                         "git:REV for the arm's entry of datasets.json at commit REV (LABEL names it in "
                         "the report)")
    ap.add_argument("--elicitation-scope", choices=ELICITATION_SCOPES, default="cell",
                    help="which runs of the comparison loop to rerun: its first cell (4 runs, default) or "
                         "the clean runs of its first seed on every landscape it shares with the log")
    args = ap.parse_args(argv)

    import pandas as pd

    patterns = [p.strip() for p in args.arms.split(",") if p.strip()] or None
    prefer = [a.strip() for a in args.prefer_acq.split(",") if a.strip()]
    overrides = dict(item.split("=", 1) for item in args.dataset_config_override)
    checks = [select(c, prefer, args.elicitation_scope) for c in discover(patterns)]
    if args.list:
        for c in checks:
            print(f"{c.arm:<44s} {_relative(c.logged_path) if c.logged_path else c.note}")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_stamp = checker_stamp()
    start_hashes = file_hashes()
    rows: list[dict] = []
    # the slow ones (fitted oracles, multi-objective) first, so they do not
    # land in the last wave
    order = sorted(checks, key=lambda c: (c.driver != "sensor", "-mo" not in c.arm, c.arm))
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_check, c, str(args.out_dir), overrides): c for c in order}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['arm']:<44s} {row['verdict']:<17s} {row.get('first_diff_column', '')} "
                  f"({row.get('runtime_sec', 0)} s)", flush=True)
    end_stamp = checker_stamp()
    end_hashes = file_hashes()
    imported = sorted(set(imported_repo_files()).union(*(set(r.pop("_imported", []) or []) for r in rows)))
    changed = sorted(k for k in set(start_hashes) | set(end_hashes) if start_hashes.get(k) != end_hashes.get(k))
    # the dataset configs and landscape statistics are read, not imported: always relevant
    relevant = set(imported) | {k for k in set(start_hashes) | set(end_hashes) if not k.endswith(".py")}
    changed_imported = [f for f in changed if f in relevant]
    changed_other = [f for f in changed if f not in relevant]
    columns = stamp_columns(start_stamp, end_stamp, changed_imported)
    table = pd.DataFrame(rows).sort_values("arm")
    for col in ("arm_dir", "logged_path", "rerun_path"):
        if col in table:
            table[col] = table[col].map(lambda v: _relative(v) if isinstance(v, str) and v else v)
    for key, value in columns.items():
        table[key] = value
    report = args.report or (args.out_dir / "provenance_check.csv")
    report.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(report, index=False)
    report.with_suffix(".stamp.json").write_text(json.dumps(
        {**columns, "start": start_stamp, "end": end_stamp,
         "files_imported_by_the_reruns": imported,
         "imported_file_sha256_at_start": {f: start_hashes.get(f) for f in imported if f in start_hashes},
         "changed_during_check_imported": changed_imported,
         "changed_during_check_other": changed_other},
        indent=2) + "\n", encoding="utf-8")
    counts = table["verdict"].value_counts().to_dict()
    lines = stamp_header(start_stamp, end_stamp, changed_imported, changed_other) + [""]
    lines += ["| arm | driver | run | status | first differing column (trial or row) | rows differing "
              "| max abs diff | verdict |",
              "|---|---|---|---|---|---|---|---|"]

    def _cell(value) -> str:
        return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)

    for _, r in table.iterrows():
        trial = _cell(r.get("first_diff_trial"))
        diff = _cell(r.get("first_diff_column")) + (f" ({int(float(trial))})" if trial else "")
        mad = r.get("max_abs_diff")
        n_rows, n_diff_rows = _cell(r.get("n_rows_logged")), _cell(r.get("n_diff_rows"))
        rows_differing = (f"{int(float(n_diff_rows))} of {int(float(n_rows))}" if n_rows and n_diff_rows else "")
        lines.append(f"| {r['arm']} | {r['driver']} | {Path(str(r.get('logged_path') or '')).name} | "
                     f"{r['status']} | {diff} | {rows_differing} | {'' if pd.isna(mad) else f'{float(mad):.3g}'} | "
                     f"{r['verdict']} |")
    lines.append("")
    lines.append("Verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    notes = [(r["arm"], r["note"]) for _, r in table.iterrows() if _cell(r.get("note"))]
    notes += [(r["arm"], r["name_note"]) for _, r in table.iterrows() if _cell(r.get("name_note"))]
    if notes:
        lines += ["", "Notes:"] + [f"- {arm}: {note}" for arm, note in notes]
    report.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {report}")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    if sys.platform == "win32":
        mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
