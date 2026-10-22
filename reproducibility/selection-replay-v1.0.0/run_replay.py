"""Verify and reproduce the bundled synthetic, fixed-archive decision replay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_payload() -> dict:
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            failures.append(item["path"])
        elif digest(path) != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise RuntimeError(f"Artifact checksum verification failed: {failures}")
    return manifest


def install_no_evaluation_guards() -> dict[str, int]:
    """Reject the helper entry points that would do new evaluations or fits."""
    import boba_benchmarks
    import boba_multiobjective
    import botorch.fit
    import replay_mo_front

    attempted = {"objective_evaluations": 0, "gp_fits": 0}

    def evaluation_forbidden(*args, **kwargs):
        attempted["objective_evaluations"] += 1
        raise RuntimeError("This artifact only replays logs; new objective evaluation is forbidden.")

    def fitting_forbidden(*args, **kwargs):
        attempted["gp_fits"] += 1
        raise RuntimeError("This artifact only replays logs; GP fitting is forbidden.")

    boba_benchmarks.evaluate = evaluation_forbidden
    boba_multiobjective.evaluate = evaluation_forbidden
    replay_mo_front.refit_posteriors = fitting_forbidden
    botorch.fit.fit_gpytorch_mll = fitting_forbidden
    return attempted


def compare_outputs(output: Path) -> dict:
    reference = json.loads((ROOT / "reference/fingerprints.json").read_text())
    checked = {}
    for name, expected in reference["files_sha256"].items():
        checked[name] = digest(output / name) == expected
    with gzip.open(output / "per_preference_decisions.csv.gz", "rb") as stream:
        raw = stream.read()
    checked["per_preference_decisions_uncompressed"] = (
        hashlib.sha256(raw).hexdigest() == reference["raw_decisions_uncompressed_sha256"]
    )
    rows = len(raw.splitlines()) - 1
    if not all(checked.values()) or rows != 98100:
        raise RuntimeError(f"Replay differs from the published reference: {checked}; rows={rows}")
    return {"exact_reference_matches": checked, "scored_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("replay-output"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = verify_payload()
    if args.verify_only:
        print(json.dumps({"verified_files": len(manifest["files"]), "archives": 100}, indent=2))
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty output directory.")
    helper_root = ROOT / "source/hitl/scripts"
    sys.path.insert(0, str(helper_root))
    guards = install_no_evaluation_guards()
    sys.argv = [
        str(helper_root / "replay_schmitt_decisions.py"),
        "--input-dir", str(ROOT / "data/output-boba-mo-halo"),
        "--openbo-src", str(ROOT / "source/openbo/src"),
        "--output-dir", str(output), "--run",
    ]
    runpy.run_path(str(helper_root / "replay_schmitt_decisions.py"), run_name="__main__")
    result = {"artifact_version": "1.0.0", "input_archives": 100,
              "archived_trials": 5000, "guarded_new_operations_attempted": guards,
              **compare_outputs(output)}
    (output / "artifact_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
