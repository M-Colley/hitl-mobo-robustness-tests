"""Run the register-triage checks and keep what each prints.

The paper quotes numbers these checks compute (the currency-test curvature and
spline checks, the manipulation intervals, the structural extra-trial zeros,
the like-for-like Kendall's W, the multi-objective front, the nugget-radius
sensitivity, the fitted-oracle companion ratio and the augmentation contrast).
Each check is a script under scripts/review_checks/; this runs them in turn and
writes each one's output to output-boba/analysis/review/register_checks/<name>.txt.

    python scripts/run_review_checks.py
    python scripts/run_review_checks.py --only c6_ratio,aug_contrast_per_dataset
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKS = ['currency_curvature', 'manipulation_intervals', 'structural_zeros', 'scalar_w', 'mo_front_check', 'nugget_threshold', 'ehmi_sigmaf', 'c6_ratio', 'aug_contrast_per_dataset']
ARGS = {"ehmi_sigmaf": ["datasets.json"], "nugget_threshold": ["datasets.json"]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="comma-separated check names")
    args = p.parse_args(argv)
    out = REPO / "output-boba" / "analysis" / "review" / "register_checks"
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in CHECKS if not args.only or n in args.only.split(",")]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for name in names:
        cmd = [sys.executable, str(REPO / "scripts" / "review_checks" / f"{name}.py")] + ARGS.get(name, [])
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
        (out / f"{name}.txt").write_text(r.stdout, encoding="utf-8")
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        print(f"{name:<26s} {status}")
        if r.returncode != 0:
            failed.append(name)
            print(r.stderr[-2000:])
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
