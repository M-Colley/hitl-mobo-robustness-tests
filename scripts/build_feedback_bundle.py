"""Build a source-only transfer ZIP. Does not import or run experiment code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path) -> tuple[int, str]:
    files = set()
    for directory, suffixes in (("scripts", {".py"}), ("tests", {".py", ".json"}),
                                ("configs", {".json"}), ("docs", {".md"}),
                                ("paper/tables", {".tex"})):
        files.update(p for p in (ROOT/directory).rglob("*") if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    for pattern in ("requirements*.txt", "*.ps1", "*.bat"):
        files.update(ROOT.glob(pattern))
    for name in ("README.md", "AGENTS.md", "LICENSE", "pyproject.toml", ".gitignore",
                 "boba_landscape_stats.json", "boba_mo_stats.json", "datasets.json",
                 "datasets-ehmi.json", "datasets-extended.json", "datasets-provoice.json",
                 "paper/COMMANDS.md", "paper/main.tex", "paper/build_overleaf_bundle.py",
                 ".github/workflows/tests.yml"):
        path = ROOT/name
        if path.is_file():
            files.add(path)
    required = ("scripts/human_feedback_benchmark.py", "scripts/feedback_policies.py",
                "configs/feedback_smoke.json", "docs/HUMAN_FEEDBACK.md", "requirements-feedback.txt")
    for name in required:
        if ROOT/name not in files:
            raise FileNotFoundError(ROOT/name)
    payloads = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(files)}
    manifest = {"format": 1, "contents": "source, configuration, tests and public paper text; no raw run logs or caches",
                "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing transfer artifacts.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("TRANSFER_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
    return len(payloads), hashlib.sha256(destination.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"dist"/"human-feedback-portable.zip")
    args = parser.parse_args()
    count, sha = build(args.output)
    print(f"Packed {count} files into {args.output.resolve()}\nSHA256 {sha}")


if __name__ == "__main__":
    main()
