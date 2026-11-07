"""The oracle-isolation experiment with the error grid calibrated per run seed.

The published arms (output-oracle-iso, -gaussian_process, -mlp; run by
run_oracle_isolation.ps1) scale every seed's error grid to the SEED-7 oracle's
sigma_f (oracle_isolation.calibrate), but the simulator refits the oracle for
every run seed (build_oracle(seed=seed)), and the GP and MLP oracles of seeds
8-11 have a sigma_f of 0.57-1.98 times seed 7's (scripts/checks/
oracle_sigmaf_seeds.py), so their nominal 0.25/1/5 sigma were not those
magnitudes. This driver reruns the three arms with each seed's grid scaled to
that seed's own oracle, into output-oracle-iso{,-gaussian_process,-mlp}-perseed.
Everything else is run_oracle_isolation.ps1's: the same dataset configs
(output-oracle-iso/configs), the same selection files (the published arm's
selection/), the same augmentation per family (jitter for the selected tree
ensembles, none for the forced GP and MLP), acquisitions, budget, onset and
seeds. Steps, in order:

  manifest     manifest_seeds.csv per family: one row per (landscape, seed) with
               that seed's oracle sigma_f and its jitter_stds, the grid
               oracle_isolation.SIGMA_GRID times sigma_f formatted as calibrate()
               formats it ("%.6f"). sigma_f is read from
               output-oracle-iso/oracle_sigmaf_seeds.csv, which
               scripts/checks/oracle_sigmaf_seeds.py writes with
               oracle_isolation.oracle_stats, the function calibrate() uses. Seed 7
               must reproduce the published manifest: its sigma_f to 1e-12 and its
               jitter_stds string exactly, so seed 7's runs are the published
               configuration. Also writes the arm's manifest.csv (the published
               oracle description, without jitter_stds, which no longer has one
               value per landscape; a reader that needs it fails loudly).
  check-clean  reruns one clean twin per check (CLEAN_CHECKS: one per family, and
               for the tree arm one per oracle model) with the current code into a
               temporary directory outside the repository and compares it with the
               logged one (check_provenance.compare_runs: every column but run_id
               and fit_time_sec, exactly). Writes <arm>/clean_identity.json.
  copy-clean   a clean twin does not depend on the error grid, so the published
               arm's *_baseline_<oracle>.csv of seeds 7-11 are copied into the new
               run directories, for a family only if every check of check-clean
               was exact under the simulator file that is there now. Each copy is
               recorded with its sha256 in <arm>/clean_copies.csv. The copies carry
               no run_config_fingerprint (they predate it), so a run directory with
               copies is run with --allow-legacy-resume, and only after checking
               that every run file there without a fingerprint is a recorded,
               unchanged copy.
  run          one simulator invocation per (family, landscape, seed), with
               --seeds <s> and that seed's --jitter-stds, into
               <arm>/runs/<dataset> with --resume. The simulator parallelises over
               seeds only, so each invocation runs sequentially (--n-jobs 1 is
               passed to it) and --n-jobs here is the number of invocations run at
               once. Two invocations never share a run directory: each rewrites the
               directory's run_metadata.json (appending its own invocation record)
               and its summary CSVs, which then describe only the last seed; the
               per-run files carry every setting in their names. A finished
               invocation writes .done_seed<s> (JSON: the command and its stds); when
               all five seeds of a dataset are done, the run files are checked
               against manifest_seeds.csv (one clean twin and the three noisy runs
               of that seed's grid per acquisition, nothing else) and
               evaluate_research_question.py writes <dataset>/evaluation, then .done.
               Simulator output goes to <arm>/logs/<dataset>_seed<s>.log, progress
               to run_oracle_isolation_perseed.log.

    python scripts/run_oracle_isolation_perseed.py manifest
    python scripts/run_oracle_isolation_perseed.py check-clean
    python scripts/run_oracle_isolation_perseed.py copy-clean
    python scripts/run_oracle_isolation_perseed.py run --n-jobs 20
    python scripts/checks/oracle_perseed.py

The noisy-run file names carry the std (..._std<float>.csv), so a per-seed run
never takes a published run's name except on seed 7, whose grid is the published
one; no noisy run is copied. Never edit bo_sensor_error_simulation.py while a run
is in progress: the workers re-import it, and every run's fingerprint hashes it,
so --resume refuses runs made under another version of the file. `run` itself
refuses to add runs to an arm whose finished invocations (.done_seed<s> records
the file's sha256) or whose clean identity check used another version of it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import oracle_isolation as iso  # noqa: E402  (imports no simulator)

# family -> (published arm, per-seed arm, augmentation, label in oracle_sigmaf_seeds.csv)
FAMILIES = {
    "tree": ("output-oracle-iso", "output-oracle-iso-perseed", "jitter", "tree"),
    "gaussian_process": ("output-oracle-iso-gaussian_process", "output-oracle-iso-gaussian_process-perseed",
                         "none", "gaussian_process"),
    "mlp": ("output-oracle-iso-mlp", "output-oracle-iso-mlp-perseed", "none", "mlp"),
}
DATA_ROOT = "output-oracle-iso"  # the synthetic datasets and their configs, shared by every family
SIGMAF_TABLE = "output-oracle-iso/oracle_sigmaf_seeds.csv"
SIM = "scripts/bo_sensor_error_simulation.py"
EVALUATE = "scripts/evaluate_research_question.py"
ACQ_LIST = ("logei", "qnei", "ucb", "ei", "random", "sobol")  # run_oracle_isolation.ps1's order
OBJECTIVE = "composite"
ERROR_MODEL = "gaussian"
ITERATIONS, INITIAL_SAMPLES, ONSET = 50, 5, 0
SEED7 = 7
LOG = "run_oracle_isolation_perseed.log"
# One clean twin per check: (landscape, seed, acquisition). The tree arm's oracle
# model differs by landscape (gradient boosting, extra trees, CatBoost), so it has
# one check per model; seeds other than 7, since the copies serve seeds 7-11.
CLEAN_CHECKS = {
    "tree": (("branin", 8, "logei"), ("eggholder", 9, "logei"), ("moving_peaks", 10, "logei")),
    "gaussian_process": (("branin", 9, "logei"),),
    "mlp": (("branin", 10, "logei"),),
}
EXCLUDED_COLUMNS = ("run_id", "fit_time_sec")  # a fresh uuid and the wall clock
_LOG_LOCK = threading.Lock()


class PerSeedError(RuntimeError):
    """A run directory or manifest that the driver must not run on (raised in worker threads too)."""


# ---------------------------------------------------------------------------
# paths and names
# ---------------------------------------------------------------------------


def arm_root(family: str) -> Path:
    return REPO / FAMILIES[family][1]


def published_root(family: str) -> Path:
    return REPO / FAMILIES[family][0]


def dataset_name(landscape: str) -> str:
    return f"iso_{landscape}"


def run_dir(family: str, landscape: str) -> Path:
    return arm_root(family) / "runs" / dataset_name(landscape)


def seed_grid(sigma_f: float) -> str:
    """The error grid of one oracle, formatted exactly as oracle_isolation.calibrate formats it."""
    return ",".join(f"{s * sigma_f:.6f}" for s in iso.SIGMA_GRID)


def parse_stds(stds: str) -> list[float]:
    """As the simulator parses --jitter-stds (bo_sensor_error_simulation.parse_float_list)."""
    return [float(v.strip()) for v in stds.split(",") if v.strip()]


def baseline_name(landscape: str, acq: str, seed: int, model: str) -> str:
    return f"bo_sensor_error_{dataset_name(landscape)}_{OBJECTIVE}_{acq}_seed{seed}_baseline_{model}.csv"


def noisy_name(landscape: str, acq: str, seed: int, model: str, std: float) -> str:
    """The simulator's name for a noisy run (run_single_seed): the std is the parsed float's repr."""
    return (f"bo_sensor_error_{dataset_name(landscape)}_{OBJECTIVE}_{acq}_seed{seed}_jittered_"
            f"{model}_{ERROR_MODEL}_jit{ONSET}_std{float(std)}.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulator_sha256() -> str:
    """The file every run's fingerprint hashes (bo_sensor_error_simulation._resume_environment)."""
    return sha256(REPO / SIM)


def default_python() -> str:
    # $env:PYTHON as every other driver, $env:HITL_PYTHON as run_oracle_isolation.ps1's alias, else this interpreter
    return os.environ.get("PYTHON") or os.environ.get("HITL_PYTHON") or sys.executable


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def mask_home_text(text: str) -> str:
    """text with the home directory replaced by '~' in each spelling a Windows checkout writes it, so a record
    of the interpreter or a scratch directory names no local user; the same rule as the simulator's mask_home,
    restated here because this driver never imports the simulator."""
    home = str(Path.home())
    spellings = {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}
    if len(home) > 2 and home[1] == ":":
        tail = home[2:].replace("\\", "/")
        spellings |= {"/" + home[0].lower() + tail, "/" + home[0].upper() + tail}
        spellings |= {s[0].swapcase() + s[1:] for s in list(spellings) if s[:1].isalpha()}
    for spelling in sorted(spellings, key=len, reverse=True):
        if len(spelling.strip("/\\")) > 1:
            text = text.replace(spelling, "~")
    return text


def log(message: str) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with open(REPO / LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# ---------------------------------------------------------------------------
# 1. the per-seed manifest
# ---------------------------------------------------------------------------


def build_manifest(family: str, sigmaf: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """One row per (landscape, seed): that seed's oracle sigma_f and its error grid.

    ``sigmaf`` is oracle_sigmaf_seeds.csv, ``published`` the published arm's
    manifest.csv. Raises unless every landscape has all five seeds, the oracle
    models agree, and seed 7 reproduces the published sigma_f (to 1e-12) and
    jitter_stds (exactly)."""
    label = FAMILIES[family][3]
    table = sigmaf[sigmaf.family == label]
    pub = published.set_index("landscape")
    rows = []
    for landscape in pub.index:
        block = table[table.landscape == landscape].set_index("seed")
        missing = sorted(set(iso.SEEDS) - set(block.index))
        if missing:
            raise ValueError(f"{family} {landscape}: no sigma_f for seeds {missing} in {SIGMAF_TABLE}")
        model = str(pub.loc[landscape, "oracle_model"])
        if set(block.oracle_model) != {model}:
            raise ValueError(f"{family} {landscape}: {SIGMAF_TABLE} fitted {set(block.oracle_model)}, "
                             f"the published arm {model}")
        seven = float(block.loc[SEED7, "sigma_f"])
        pub_sigma = float(pub.loc[landscape, "sigma_f_fitted"])
        if abs(seven - pub_sigma) > 1e-12:
            raise ValueError(f"{family} {landscape}: seed 7 sigma_f {seven!r} is not the manifest's {pub_sigma!r}")
        if seed_grid(seven) != str(pub.loc[landscape, "jitter_stds"]):
            raise ValueError(f"{family} {landscape}: seed 7's grid {seed_grid(seven)} is not the published "
                             f"{pub.loc[landscape, 'jitter_stds']}")
        for seed in iso.SEEDS:
            sigma_f = float(block.loc[seed, "sigma_f"])
            rows.append({
                "family": family, "landscape": landscape, "dataset": dataset_name(landscape), "seed": seed,
                "oracle_model": model, "augmentation": FAMILIES[family][2],
                "sigma_f": sigma_f, "sigma_f_seed7": seven, "ratio_to_seed7": sigma_f / seven,
                "sigma_f_minus_published_seed7": seven - pub_sigma,
                "jitter_stds": seed_grid(sigma_f),
                "published_jitter_stds": str(pub.loc[landscape, "jitter_stds"]),
                "sigma_multiples": ",".join(f"{s:g}" for s in iso.SIGMA_GRID),
                "sigma_f_source": f"{SIGMAF_TABLE} (oracle_isolation.oracle_stats)",
                "dataset_config": f"{DATA_ROOT}/configs/datasets-{dataset_name(landscape)}.json",
                "selection_path": f"{FAMILIES[family][0]}/selection/{dataset_name(landscape)}.json",
            })
    return pd.DataFrame(rows)


def oracle_manifest(published: pd.DataFrame) -> pd.DataFrame:
    """The arm's manifest.csv: the published oracle description (the same oracles,
    selection and seed-7 fidelity) without jitter_stds, which oracle_isolation
    and oracle_achievable read for the landscape list and the oracle columns."""
    out = published.drop(columns=["jitter_stds"]).copy()
    out["calibration"] = "per_seed: " + iso.SEEDS_MANIFEST
    return out


def _write_if_new(frame: pd.DataFrame, path: Path, force: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    if path.is_file():
        if tmp.read_bytes() == path.read_bytes():
            tmp.unlink()
            return "unchanged"
        if not force:
            tmp.unlink()
            raise SystemExit(f"{_rel(path)} exists with other content; runs may depend on it (--force to replace)")
    os.replace(tmp, path)
    return "written"


def cmd_manifest(args) -> None:
    sigmaf = pd.read_csv(REPO / SIGMAF_TABLE)
    for family in args.families:
        published = pd.read_csv(published_root(family) / "manifest.csv")
        seeds = build_manifest(family, sigmaf, published)
        root = arm_root(family)
        if args.force and any(root.glob("runs/*/.done_seed*")):
            raise SystemExit(f"{_rel(root)} has finished invocations; a new grid needs a fresh directory")
        a = _write_if_new(seeds, root / iso.SEEDS_MANIFEST, args.force)
        b = _write_if_new(oracle_manifest(published), root / "manifest.csv", args.force)
        off = seeds[seeds.seed != SEED7].ratio_to_seed7
        log(f"[manifest] {family}: {_rel(root / iso.SEEDS_MANIFEST)} {a}, manifest.csv {b}; "
            f"{len(seeds)} rows; seed 7 = published grid on all {seeds.landscape.nunique()} landscapes "
            f"(max |sigma_f - published| {seeds.sigma_f_minus_published_seed7.abs().max():.1e}); "
            f"sigma_f(seed)/sigma_f(7) over seeds 8-11 in [{off.min():.3f}, {off.max():.3f}]")


def load_manifest(family: str) -> pd.DataFrame:
    path = arm_root(family) / iso.SEEDS_MANIFEST
    if not path.is_file():
        raise PerSeedError(f"{_rel(path)} is missing: run `run_oracle_isolation_perseed.py manifest` first")
    return pd.read_csv(path, dtype={"jitter_stds": str})


# ---------------------------------------------------------------------------
# the simulator's command line, as run_oracle_isolation.ps1 builds it
# ---------------------------------------------------------------------------


def sim_command(family: str, landscape: str, seed: int, stds: str, out_dir: Path, python: str,
                acq_list: tuple[str, ...] = ACQ_LIST, resume: bool = True,
                allow_legacy_resume: bool = False) -> list[str]:
    """run_oracle_isolation.ps1's simulator call for one seed: --seeds <seed> and
    that seed's --jitter-stds, and --n-jobs 1 (the simulator parallelises over
    seeds only, so one seed runs sequentially whatever it is given)."""
    data = dataset_name(landscape)
    cmd = [python, SIM,
           "--dataset-config", f"{DATA_ROOT}/configs/datasets-{data}.json", "--objective", OBJECTIVE,
           "--acq-list", ",".join(acq_list),
           "--oracle-model", "auto",
           "--oracle-selection-path", f"{FAMILIES[family][0]}/selection/{data}.json",
           "--oracle-augmentation", FAMILIES[family][2],
           "--iterations", str(ITERATIONS), "--initial-samples", str(INITIAL_SAMPLES),
           "--error-models", ERROR_MODEL, "--jitter-stds", stds, "--jitter-iterations", str(ONSET),
           "--seeds", str(seed), "--output-dir", str(out_dir), "--parallel", "--n-jobs", "1"]
    if resume:
        cmd.append("--resume")
    if allow_legacy_resume:
        cmd.append("--allow-legacy-resume")
    return cmd


def evaluate_command(out_dir: Path, python: str) -> list[str]:
    return [python, EVALUATE, "--input-dir", str(out_dir), "--output-dir", str(out_dir / "evaluation")]


def _env() -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


# ---------------------------------------------------------------------------
# 3. clean twins: the identity check and the copies
# ---------------------------------------------------------------------------


def cmd_check_clean(args) -> None:
    from check_provenance import compare_runs
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="oracle_perseed_identity_"))
    work = work.resolve()
    if work == REPO or REPO in work.parents:
        raise SystemExit(f"the identity check must run outside the repository, not in {work}")
    catboost_info = REPO / "catboost_info"
    before = sorted((p.relative_to(REPO).as_posix(), p.stat().st_mtime_ns) for p in catboost_info.rglob("*")) \
        if catboost_info.is_dir() else []
    for family in args.families:
        manifest = load_manifest(family).set_index(["landscape", "seed"])
        sim_hash = simulator_sha256()
        results = []
        for landscape, seed, acq in CLEAN_CHECKS[family]:
            row = manifest.loc[(landscape, seed)]
            model = str(row.oracle_model)
            out = work / family / f"{dataset_name(landscape)}_seed{seed}_{acq}"
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            one_sigma = f"{float(row.sigma_f):.6f}"  # one noisy run comes along; only the clean twin is compared
            cmd = sim_command(family, landscape, seed, one_sigma, out, args.python, acq_list=(acq,), resume=False)
            started = time.perf_counter()
            with open(out / "simulator.log", "w", encoding="utf-8") as handle:
                code = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, env=_env()).returncode
            seconds = time.perf_counter() - started
            logged = published_root(family) / "runs" / dataset_name(landscape) / baseline_name(landscape, acq, seed, model)
            new = out / baseline_name(landscape, acq, seed, model)
            if code != 0 or not new.is_file():
                comparison = {"status": "failed", "note": f"simulator exit {code}; see {out / 'simulator.log'}"}
            else:
                comparison = compare_runs(logged, new, excluded=EXCLUDED_COLUMNS)
            result = {"landscape": landscape, "seed": seed, "acquisition": acq, "oracle_model": model,
                      "logged": _rel(logged), "rerun": str(new), "seconds": round(seconds, 1),
                      "excluded_columns": list(EXCLUDED_COLUMNS), **comparison}
            results.append(result)
            log(f"[check-clean] {family} {landscape} seed {seed} {acq} ({model}): {comparison.get('status')} "
                f"({comparison.get('n_columns_compared', 0)} columns compared; only in the rerun: "
                f"{comparison.get('columns_only_new') or '-'}; first difference "
                f"{comparison.get('first_diff_column') or '-'}) in {seconds:.0f} s")
        record = {"family": family, "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
                  "simulator_sha256": sim_hash, "simulator_sha256_after": simulator_sha256(),
                  "python": args.python, "work_dir": str(work),
                  "identical": all(r.get("status") == "exact" for r in results), "checks": results}
        (arm_root(family) / "clean_identity.json").write_text(
            mask_home_text(json.dumps(record, indent=2, default=str)), encoding="utf-8")
        log(f"[check-clean] {family}: {'IDENTICAL' if record['identical'] else 'NOT IDENTICAL'} "
            f"-> {_rel(arm_root(family) / 'clean_identity.json')}")
    after = sorted((p.relative_to(REPO).as_posix(), p.stat().st_mtime_ns) for p in catboost_info.rglob("*")) \
        if catboost_info.is_dir() else []
    if after != before:
        log("[check-clean] WARNING: catboost_info/ at the repository root changed during the reruns")


def identity_record(family: str) -> dict | None:
    path = arm_root(family) / "clean_identity.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def cmd_copy_clean(args) -> None:
    for family in args.families:
        record = identity_record(family)
        if record is None or not record.get("identical"):
            log(f"[copy-clean] {family}: SKIPPED, the clean identity check has not passed "
                f"({'no record' if record is None else 'not identical'}); its clean twins will be rerun")
            continue
        if record.get("simulator_sha256") != simulator_sha256() or \
                record.get("simulator_sha256_after") != record.get("simulator_sha256"):
            raise SystemExit(f"{family}: {SIM} changed since the identity check; rerun check-clean")
        manifest = load_manifest(family)
        pub = published_root(family)
        rows, copied, present = [], 0, 0
        for landscape, model in manifest.groupby("landscape").oracle_model.first().items():
            dst_dir = run_dir(family, landscape)
            dst_dir.mkdir(parents=True, exist_ok=True)
            for seed in iso.SEEDS:
                for acq in ACQ_LIST:
                    name = baseline_name(landscape, acq, seed, model)
                    src = pub / "runs" / dataset_name(landscape) / name
                    dst = dst_dir / name
                    if not src.is_file():
                        raise SystemExit(f"no published clean twin {_rel(src)}")
                    digest = sha256(src)
                    if dst.is_file():
                        if sha256(dst) != digest:
                            raise SystemExit(f"{_rel(dst)} exists and differs from {_rel(src)}")
                        present += 1
                    else:
                        tmp = dst.with_name(dst.name + ".tmp")
                        shutil.copy2(src, tmp)
                        os.replace(tmp, dst)
                        copied += 1
                    rows.append({"dataset": dataset_name(landscape), "seed": seed, "acquisition": acq,
                                 "file": name, "sha256": digest, "source": _rel(src)})
        pd.DataFrame(rows).to_csv(arm_root(family) / "clean_copies.csv", index=False)
        log(f"[copy-clean] {family}: {copied} copied, {present} already present, {len(rows)} recorded in "
            f"{_rel(arm_root(family) / 'clean_copies.csv')}")


def recorded_copies(family: str) -> dict[str, str]:
    """{run file path relative to the repository: sha256} of the copied clean twins."""
    path = arm_root(family) / "clean_copies.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path)
    return {f"{FAMILIES[family][1]}/runs/{d}/{f}": s for d, f, s in zip(table.dataset, table.file, table.sha256)}


def _has_fingerprint(path: Path) -> bool:
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    return "run_config_fingerprint" in header


def legacy_guard(family: str, landscape: str) -> bool:
    """Whether the run directory needs --allow-legacy-resume: True when it holds
    run files without a fingerprint, and then every one of them must be a
    recorded copy with its recorded sha256 (else PerSeedError)."""
    copies = recorded_copies(family)
    legacy = []
    for path in sorted(run_dir(family, landscape).glob("bo_sensor_error_*_seed*_*.csv")):
        if not _has_fingerprint(path):
            legacy.append(path)
    for path in legacy:
        key = path.relative_to(REPO).as_posix()
        if key not in copies:
            raise PerSeedError(f"{key} has no configuration fingerprint and is not a recorded clean copy")
        if sha256(path) != copies[key]:
            raise PerSeedError(f"{key} differs from the clean twin copied there")
    return bool(legacy)


# ---------------------------------------------------------------------------
# 2. the runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    family: str
    landscape: str
    seed: int
    stds: str
    oracle_model: str

    @property
    def key(self) -> tuple[str, str]:  # one run directory
        return (self.family, self.landscape)

    @property
    def label(self) -> str:
        return f"{self.family}/{dataset_name(self.landscape)}/seed{self.seed}"


def seed_marker(task: Task) -> Path:
    return run_dir(task.family, task.landscape) / f".done_seed{task.seed}"


def done_marker(family: str, landscape: str) -> Path:
    return run_dir(family, landscape) / ".done"


def tasks_for(families, landscapes=None, seeds=None) -> list[Task]:
    """Seed-major, so that consecutive tasks fall in different run directories."""
    out = []
    for family in families:
        unknown = set(landscapes or ()) - set(load_manifest(family).landscape)
        if unknown:
            raise PerSeedError(f"{family}: no landscapes {sorted(unknown)} in {iso.SEEDS_MANIFEST}")
    for seed in (seeds or iso.SEEDS):
        for family in families:
            manifest = load_manifest(family)
            for row in manifest[manifest.seed == seed].itertuples():
                if landscapes and row.landscape not in landscapes:
                    continue
                out.append(Task(family, row.landscape, int(seed), str(row.jitter_stds), str(row.oracle_model)))
    return out


def marker_state(task: Task) -> str:
    """'todo', 'done', or raises when a finished invocation used another grid."""
    marker = seed_marker(task)
    if not marker.is_file():
        return "todo"
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("jitter_stds") != task.stds:
        raise PerSeedError(f"{_rel(marker)} ran with stds {record.get('jitter_stds')}, the manifest now says "
                         f"{task.stds}: the manifest changed after runs were made")
    return "done"


def expected_files(family: str, landscape: str, seed: int, stds: str, model: str) -> list[str]:
    names = []
    for acq in ACQ_LIST:
        names.append(baseline_name(landscape, acq, seed, model))
        names.extend(noisy_name(landscape, acq, seed, model, s) for s in parse_stds(stds))
    return names


def check_run_files(family: str, landscape: str) -> list[str]:
    """Problems with a run directory against manifest_seeds.csv: a missing run, or
    a run file no seed's grid explains (another std, seed or acquisition)."""
    manifest = load_manifest(family)
    block = manifest[manifest.landscape == landscape]
    expected = set()
    for row in block.itertuples():
        expected.update(expected_files(family, landscape, int(row.seed), str(row.jitter_stds), str(row.oracle_model)))
    present = {p.name for p in run_dir(family, landscape).glob("bo_sensor_error_*_seed*_*.csv")}
    problems = [f"missing {n}" for n in sorted(expected - present)]
    problems += [f"unexpected {n}" for n in sorted(present - expected)]
    return problems


def run_task(task: Task, python: str) -> tuple[bool, float, str]:
    out = run_dir(task.family, task.landscape)
    out.mkdir(parents=True, exist_ok=True)
    allow_legacy = legacy_guard(task.family, task.landscape)
    cmd = sim_command(task.family, task.landscape, task.seed, task.stds, out.relative_to(REPO), python,
                      allow_legacy_resume=allow_legacy)
    logs = arm_root(task.family) / "logs"
    logs.mkdir(exist_ok=True)
    sim_log = logs / f"{dataset_name(task.landscape)}_seed{task.seed}.log"
    before = {p.name for p in out.glob("bo_sensor_error_*_seed*_*.csv")}
    log(f"[run] {task.label} stds={task.stds} oracle={task.oracle_model}"
        f"{' (legacy clean copies present)' if allow_legacy else ''}")
    started = time.perf_counter()
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    with open(sim_log, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== {started_at} {subprocess.list2cmdline(cmd)}\n")
        handle.flush()
        code = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, env=_env()).returncode
    seconds = time.perf_counter() - started
    if code != 0:
        return False, seconds, f"simulator exited {code}; see {_rel(sim_log)}"
    names = set(expected_files(task.family, task.landscape, task.seed, task.stds, task.oracle_model))
    present = {p.name for p in out.glob("bo_sensor_error_*_seed*_*.csv")}
    if names - present:
        return False, seconds, f"{len(names - present)} expected run files missing after the run"
    new = sorted(names & (present - before))
    record = {"family": task.family, "dataset": dataset_name(task.landscape), "seed": task.seed,
              "oracle_model": task.oracle_model, "jitter_stds": task.stds, "command": cmd,
              "started_at": started_at, "seconds": round(seconds, 1), "exit_code": code,
              "runs_written": len(new), "runs_reused": len(names) - len(new),
              "simulator_sha256": simulator_sha256()}
    seed_marker(task).write_text(mask_home_text(json.dumps(record, indent=2)), encoding="utf-8")
    return True, seconds, f"{len(new)} runs written, {len(names) - len(new)} reused, {seconds:.0f} s"


def evaluate_dataset(family: str, landscape: str, python: str) -> tuple[bool, str]:
    problems = check_run_files(family, landscape)
    if problems:
        return False, f"run files do not match manifest_seeds.csv: {problems[:3]} ({len(problems)} problems)"
    out = run_dir(family, landscape)
    sim_log = arm_root(family) / "logs" / f"{dataset_name(landscape)}_evaluate.log"
    with open(sim_log, "a", encoding="utf-8") as handle:
        code = subprocess.run(evaluate_command(out.relative_to(REPO), python), cwd=REPO, stdout=handle,
                              stderr=subprocess.STDOUT, env=_env()).returncode
    if code != 0:
        return False, f"evaluation exited {code}; see {_rel(sim_log)}"
    done_marker(family, landscape).write_text(dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    return True, "evaluated"


def all_seeds_done(family: str, landscape: str) -> bool:
    return all((run_dir(family, landscape) / f".done_seed{s}").is_file() for s in iso.SEEDS)


class Scheduler:
    """At most n_jobs invocations at once, never two in one run directory."""

    def __init__(self, tasks: list[Task], n_jobs: int, python: str, evaluate: bool):
        self.pending = list(tasks)
        self.busy: set[tuple[str, str]] = set()
        self.cond = threading.Condition()
        self.n_jobs, self.python, self.evaluate = n_jobs, python, evaluate
        self.failed: list[str] = []
        self.seconds: list[float] = []

    def _next(self) -> Task | None:
        with self.cond:
            while True:
                if not self.pending:
                    return None
                for i, task in enumerate(self.pending):
                    if task.key not in self.busy:
                        self.busy.add(task.key)
                        return self.pending.pop(i)
                self.cond.wait()

    def _release(self, key) -> None:
        with self.cond:
            self.busy.discard(key)
            self.cond.notify_all()

    def worker(self) -> None:
        while True:
            task = self._next()
            if task is None:
                return
            try:
                ok, seconds, note = run_task(task, self.python)
                log(f"[{'done' if ok else 'FAIL'}] {task.label}: {note}")
                if ok:
                    self.seconds.append(seconds)
                else:
                    self.failed.append(task.label)
                if ok and self.evaluate and all_seeds_done(*task.key) and not done_marker(*task.key).is_file():
                    ok, note = evaluate_dataset(task.family, task.landscape, self.python)
                    log(f"[{'eval' if ok else 'FAIL'}] {task.family}/{dataset_name(task.landscape)}: {note}")
                    if not ok:
                        self.failed.append(f"evaluate {task.family}/{dataset_name(task.landscape)}")
            except Exception as exc:  # one failed task must not stop the others
                log(f"[FAIL] {task.label}: {exc!r}")
                self.failed.append(task.label)
            finally:
                self._release(task.key)

    def run(self) -> None:
        threads = [threading.Thread(target=self.worker, daemon=True) for _ in range(self.n_jobs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def simulator_guard(families) -> None:
    """Refuse to add runs under another simulator file than the arm's earlier runs and
    the clean identity check used: a code change needs a fresh directory (AGENTS.md),
    and the copied clean twins were validated against one version of the file."""
    current = simulator_sha256()
    for family in families:
        record = identity_record(family)
        if (arm_root(family) / "clean_copies.csv").is_file() and (record or {}).get("simulator_sha256") != current:
            raise PerSeedError(f"{family}: {SIM} is not the file the clean identity check validated "
                               f"(sha256 {(record or {}).get('simulator_sha256', '?')[:12]}, now {current[:12]}); "
                               "rerun `check-clean`, or use fresh directories")
        for marker in sorted(arm_root(family).glob("runs/*/.done_seed*")):
            used = json.loads(marker.read_text(encoding="utf-8")).get("simulator_sha256")
            if used != current:
                raise PerSeedError(f"{_rel(marker)} ran under another {SIM} (sha256 {str(used)[:12]}, now "
                                   f"{current[:12]}); runs of two code versions must not share an arm")


def cmd_run(args) -> None:
    tasks = tasks_for(args.families, args.landscapes, args.seeds)
    todo = [t for t in tasks if marker_state(t) == "todo"]
    if todo and not args.dry_run:
        simulator_guard(args.families)
    log(f"=== run: {len(tasks)} (family, dataset, seed) tasks selected, {len(tasks) - len(todo)} already done, "
        f"{len(todo)} to run with {args.n_jobs} at once; python {args.python}")
    for family in args.families:
        record = identity_record(family)
        if not (arm_root(family) / "clean_copies.csv").is_file():
            log(f"[note] {family}: no copied clean twins (identity check "
                f"{'not run' if record is None else ('passed, copy-clean not run' if record.get('identical') else 'failed')});"
                " the clean twins will be simulated")
    if args.dry_run:
        for t in todo:
            allow = bool(recorded_copies(t.family))
            print(subprocess.list2cmdline(sim_command(t.family, t.landscape, t.seed, t.stds,
                                                      run_dir(t.family, t.landscape).relative_to(REPO), args.python,
                                                      allow_legacy_resume=allow)))
        return
    # datasets left finished but unevaluated by an earlier invocation
    if args.evaluate:
        for family in args.families:
            for landscape in sorted({t.landscape for t in tasks if t.family == family}):
                if all_seeds_done(family, landscape) and not done_marker(family, landscape).is_file():
                    ok, note = evaluate_dataset(family, landscape, args.python)
                    log(f"[{'eval' if ok else 'FAIL'}] {family}/{dataset_name(landscape)}: {note}")
    scheduler = Scheduler(todo, max(1, min(args.n_jobs, os.cpu_count() or 1)), args.python, args.evaluate)
    scheduler.run()
    if scheduler.seconds:
        s = pd.Series(scheduler.seconds)
        log(f"=== invocations finished: {len(s)}, median {s.median():.0f} s, total {s.sum() / 3600:.2f} worker-hours")
    if scheduler.failed:
        log(f"=== {len(scheduler.failed)} FAILED: {', '.join(scheduler.failed[:10])}")
        raise SystemExit(1)
    log("=== run complete")


def cmd_status(args) -> None:
    for family in args.families:
        tasks = tasks_for([family], args.landscapes, args.seeds)
        done = sum(marker_state(t) == "done" for t in tasks)
        landscapes = sorted({t.landscape for t in tasks})
        evaluated = sum(done_marker(family, l).is_file() for l in landscapes)
        print(f"{family:17s} invocations {done}/{len(tasks)} done; datasets evaluated {evaluated}/{len(landscapes)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["manifest", "check-clean", "copy-clean", "run", "status"])
    p.add_argument("--families", default=",".join(FAMILIES), help="comma-separated: tree,gaussian_process,mlp")
    p.add_argument("--landscapes", default=None, help="comma-separated landscape names (default: all twenty)")
    p.add_argument("--seeds", default=None, help="comma-separated run seeds (default: 7-11)")
    p.add_argument("--n-jobs", type=int, default=1, help="simulator invocations at once (each runs one seed)")
    p.add_argument("--python", default=default_python(), help="interpreter for the simulator ($PYTHON, $HITL_PYTHON)")
    p.add_argument("--no-evaluate", dest="evaluate", action="store_false",
                   help="do not run evaluate_research_question.py when a dataset's five seeds are done")
    p.add_argument("--dry-run", action="store_true", help="print the simulator commands and stop")
    p.add_argument("--force", action="store_true", help="manifest: replace a manifest with other content")
    p.add_argument("--work-dir", default=None, help="check-clean: where the reruns go (outside the repository)")
    args = p.parse_args(argv)
    args.families = [f for f in args.families.split(",") if f]
    unknown = set(args.families) - set(FAMILIES)
    if unknown:
        p.error(f"unknown families {sorted(unknown)}; choose from {list(FAMILIES)}")
    args.landscapes = [x for x in args.landscapes.split(",") if x] if args.landscapes else None
    args.seeds = [int(x) for x in args.seeds.split(",") if x] if args.seeds else None
    if args.seeds and set(args.seeds) - set(iso.SEEDS):
        p.error(f"seeds must be among {iso.SEEDS}")
    return args


def main(argv=None) -> None:
    os.chdir(REPO)
    args = parse_args(argv)
    try:
        {"manifest": cmd_manifest, "check-clean": cmd_check_clean, "copy-clean": cmd_copy_clean,
         "run": cmd_run, "status": cmd_status}[args.step](args)
    except PerSeedError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
