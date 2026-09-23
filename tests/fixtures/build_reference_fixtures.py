"""Regenerate the vendored reference outputs in tests/fixtures (register item D1).

    python tests/fixtures/build_reference_fixtures.py boba
    python tests/fixtures/build_reference_fixtures.py backup [--source PATH]

``boba`` needs a BOBA checkout (``BOBA_ROOT``, default a ``BOBA`` directory next to this repository)
and writes boba_reference.json: BOBA's torch implementation of every analytic
benchmark at the parity points of test_boba_benchmarks.py, and the four registry
tables of BOBA/parallel_main.py.

``backup`` needs the simulator as it stood on 2026-09-14 and writes
bo_sim_backup_2026-09-14.json: its name lists, parse_error_models(None, "all"),
and the default multi-objective run of test_mo_halo.py. ``--source`` may be the
.py or, as on 2026-09-23 when only it survived, its CPython bytecode.

Both are evaluated through the same helpers the tests call, so a fixture holds
exactly what the live comparison would have produced. Not collected by pytest.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.machinery
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent
TESTS = FIXTURES.parent
REPO = TESTS.parent
for path in (REPO / "scripts", TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _reference_fixtures as ref  # noqa: E402

BACKUP_PYC = ref.BACKUP_SIM.parent / "__pycache__" / "bo_sensor_error_simulation.cpython-312.pyc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _dumps(obj, level: int = 0) -> str:
    """Indented JSON with every list of scalars on one line, so a fixture stays readable."""
    pad = " " * (level + 1)
    if isinstance(obj, dict):
        body = ",\n".join(f"{pad}{json.dumps(k)}: {_dumps(v, level + 1)}" for k, v in obj.items())
        return "{\n" + body + "\n" + " " * level + "}"
    if isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        return "[\n" + ",\n".join(pad + _dumps(v, level + 1) for v in obj) + "\n" + " " * level + "]"
    return json.dumps(obj, allow_nan=False)


def _repo_state() -> dict:
    dirty = _git(REPO, "status", "--porcelain", "--untracked-files=no", "--", "scripts")
    return {"git_commit": _git(REPO, "rev-parse", "HEAD"),
            "scripts_dirty": bool(dirty) if dirty is not None else None}


def _write(path: Path, payload: dict) -> None:
    text = _dumps(payload) + "\n"
    assert json.loads(text) == payload
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(text.encode()) / 1024:.1f} KiB)")


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def build_boba() -> None:
    import boba_benchmarks as bb

    if not ref.HAS_BOBA:
        raise SystemExit(f"no BOBA checkout at {ref.BOBA_ROOT} (set BOBA_ROOT)")
    files = ["bayes_opt/simulation.py", "parallel_main.py"]
    dirty = _git(ref.BOBA_ROOT, "status", "--porcelain", "--", *files)
    functions = {}
    for name in bb.BOBA_ORDER:
        spec = bb.BENCHMARKS[name]
        if spec.kind == "stochastic":
            continue
        X = ref.parity_points(spec)
        functions[name] = {"dim": spec.dim, "lo": spec.lo, "hi": spec.hi,
                           "points_sha256": ref.points_digest(X),
                           "values": [float(v) for v in ref.live_boba_values(name, X)]}
    payload = {
        "_header": {
            "what": "BOBA's own implementations of the analytic benchmarks at the parity points, and "
                    "BOBA's registry tables, for test_boba_benchmarks.py where no BOBA checkout exists",
            "produced_at": _now(),
            "produced_by": "python tests/fixtures/build_reference_fixtures.py boba",
            "how": (f"per function: X = lo + np.random.default_rng({ref.PARITY_SEED}).random(({ref.PARITY_POINTS}, dim))"
                    " * (hi - lo) with lo/hi/dim from the vendored registry; values[i] = "
                    "float(bayes_opt.simulation.<name>(torch.tensor(X[i], dtype=torch.double))); "
                    "points_sha256 = sha256 of X as little-endian float64. tables: the four list "
                    "literals of parallel_main.py, exec'd in an empty namespace."),
            "source": {
                "repository": "BOBA (upstream URL withheld for anonymous review)",
                "checkout": "a local checkout (path withheld)",
                "git_commit": _git(ref.BOBA_ROOT, "rev-parse", "HEAD"),
                "git_commit_date": _git(ref.BOBA_ROOT, "log", "-1", "--format=%ci"),
                "files_dirty": bool(dirty) if dirty is not None else None,
                "files_sha256": {f: _sha256(ref.BOBA_ROOT / f) for f in files},
            },
            "repository": _repo_state(),
            "environment": ref.environment(),
        },
        "tables": ref.live_boba_tables(),
        "functions": functions,
    }
    _write(ref.BOBA_FIXTURE, payload)


def _load_backup(source: Path):
    name = "bo_sim_backup_2026_09_14_fixture"
    if source.suffix == ".pyc":
        loader = importlib.machinery.SourcelessFileLoader(name, str(source))
        spec = importlib.util.spec_from_loader(name, loader)
    else:
        spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pyc_header(path: Path) -> dict:
    head = path.read_bytes()[:16]
    flags, mtime, size = struct.unpack("<III", head[4:16])
    return {"magic_hex": head[:4].hex(), "matches_running_interpreter": head[:4] == importlib.util.MAGIC_NUMBER,
            "flags": flags, "source_mtime": datetime.datetime.fromtimestamp(mtime).astimezone().isoformat(),
            "source_size_bytes": size}


def build_backup(source: Path) -> None:
    import test_mo_halo  # the test's own _run, so the fixture is exactly its call

    backup = _load_backup(source)
    constants = {name: list(value) for name, value in sorted(vars(backup).items())
                 if name.endswith("_CHOICES") and isinstance(value, (list, tuple))}
    run = test_mo_halo._without_timing(test_mo_halo._run(backup, iterations=6))
    source_info = {"path": str(source), "sha256": _sha256(source), "bytes": source.stat().st_size}
    if source.suffix == ".pyc":
        source_info["pyc_header"] = _pyc_header(source)
        source_info["note"] = ("the .py copy had been deleted; this is the CPython 3.12 bytecode "
                               "compiled from it on 2026-09-14, loaded with SourcelessFileLoader")
    payload = {
        "_header": {
            "what": "bo_sensor_error_simulation.py as it stood on 2026-09-14, before the acquisition-"
                    "extension, error-extension and multi-objective halo builds: its name lists and "
                    "one default multi-objective run, for the provenance tests of those builds",
            "produced_at": _now(),
            "produced_by": "python tests/fixtures/build_reference_fixtures.py backup --source " + source.name,
            "how": ("constants: every module-level *_CHOICES list of the loaded module. calls: "
                    "module.parse_error_models(None, 'all'). runs.mo_default_run: "
                    "tests/test_mo_halo.py::_run(module, iterations=6) -- branincurrin, qlognehvi, "
                    "seed 7, gaussian noise 0.5 SD from trial 1, jitter_rng seed 99 -- with "
                    "fit_time_sec dropped; floats stored by repr, NaN as null. The current repository "
                    "code is on sys.path for the module's own imports, as in the original tests."),
            "source": source_info,
            "repository": _repo_state(),
            "environment": ref.environment(),
        },
        "constants": constants,
        "calls": {"parse_error_models(None, 'all')": list(backup.parse_error_models(None, "all"))},
        "runs": {"mo_default_run": ref.frame_to_record(run)},
    }
    _write(ref.BACKUP_FIXTURE, payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=["boba", "backup"])
    parser.add_argument("--source", type=Path, default=None,
                        help="the 2026-09-14 simulator, .py or .pyc (backup only)")
    args = parser.parse_args(argv)
    if args.which == "boba":
        build_boba()
    else:
        source = args.source or (ref.BACKUP_SIM if ref.BACKUP_SIM.is_file() else BACKUP_PYC)
        if not source.is_file():
            raise SystemExit(f"no 2026-09-14 simulator at {source}")
        build_backup(source)


if __name__ == "__main__":
    main()
