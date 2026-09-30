"""Run the register-triage checks and keep what each prints.

The paper quotes numbers these checks compute: the currency-test curvature and
spline checks, the manipulation intervals, the structural extra-trial zeros,
the like-for-like Kendall's W, the multi-objective front, the nugget-radius
sensitivity, the fitted-oracle companion ratio and its estimators, the
augmentation contrast, the headline on landscapes matched to the human studies'
opt_z, the oracle-isolation result by oracle family (with each seed's refitted
oracle and the achievable-improvement unit), the untouched-seed test of the
final sitting, the sitting against the zero-trial ship rules and under the
sequential rating process, the budget rule by onset, the model-free floors'
deployed regret, the multiplicity of the Friedman tests, the archival
instruments' scales, the multi-objective onset bound, the reach trial of the
extra-trial appendix, the one-shot predictor's residuals and out-of-sample fit,
the comparison loop's determinism in another worker schedule, and the
tie-break sensitivity of every deployed-design number from an arm with ties.

Each check is a script under scripts/review_checks/; this runs them in turn, in
the order below (a check that reads another's output comes after it), and writes
each one's stdout to output-boba/analysis/review/register_checks/<name>.txt.
Some checks also write CSV files of their own beside that directory; a few write
their register file themselves as well, with exactly what they print.

    python scripts/run_review_checks.py
    python scripts/run_review_checks.py --only c6_ratio,aug_contrast_per_dataset

A fresh checkout holds every table and figure input but not the run logs, so
some checks run only on the machine that holds the arms. structural_zeros,
mo_front_check, budget_split_by_onset, extra_reach, mo_onset_bound and
oracle_companion_estimators read git-ignored run logs. sitting_sequential reads
the per-run tables of end_of_study_ksweep_seq, end_of_study_kwide_seq and
end_of_study_ksweep_seqrank (over 5 MB each, untracked). tie_break reads the
per-run table of ``python scripts/review_checks/tie_break.py scan --workers 6``
(about 15 minutes, git-ignored). ehmi_sigmaf, nugget_threshold and
instrument_scale read the archival clones in .dataset_cache/, and
instrument_scale also output-boba-instrument/run_metadata.json. The other
checks run from a fresh checkout; sitting_vs_shiprule collates the tracked
outputs of scripts/sitting_by_magnitude.py (paper/COMMANDS.md).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKS = [
    "currency_curvature", "manipulation_intervals", "structural_zeros", "scalar_w", "mo_front_check",
    "nugget_threshold", "ehmi_sigmaf",
    # The fitted-oracle companion (App. B.5) and the oracle-isolation arms: the
    # achievable-improvement files that oracle_families reads are written by
    # oracle_achievable, which reads oracle_sigmaf_seeds' per-seed oracles.
    "c6_ratio", "aug_contrast_per_dataset", "oracle_companion_estimators", "matched_optz",
    "oracle_sigmaf_seeds", "oracle_achievable", "oracle_families",
    "anchor_propagation",
    # The final sitting and the budget rule (Sec. 7, App. E): sitting_vs_shiprule
    # collates the sitting outputs and the untouched-seed test.
    "fresh_seed_replication", "sitting_sequential", "sitting_vs_shiprule", "budget_split_by_onset",
    # Sec. 6 and the appendices.
    "floor_deployed", "friedman_multiplicity", "instrument_scale", "mo_onset_bound", "extra_reach",
    # App. C.2's residuals and out-of-sample fit; the comparison loop's rerun in
    # another worker schedule (reproducibility statement, about two minutes).
    "frag_residuals", "elicitation_determinism",
    # Last: it re-scores deployed numbers from every arm with ties.
    "tie_break",
]
ARGS = {"ehmi_sigmaf": ["datasets.json"], "nugget_threshold": ["datasets.json"], "tie_break": ["rescore"]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="comma-separated check names")
    args = p.parse_args(argv)
    out = REPO / "output-boba" / "analysis" / "review" / "register_checks"
    out.mkdir(parents=True, exist_ok=True)
    wanted = [n for n in args.only.split(",") if n]
    unknown = sorted(set(wanted) - set(CHECKS))
    if unknown:
        raise SystemExit(f"unknown check(s) {unknown}; choose from {CHECKS}")
    names = [n for n in CHECKS if not wanted or n in wanted]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for name in names:
        cmd = [sys.executable, str(REPO / "scripts" / "review_checks" / f"{name}.py")] + ARGS.get(name, [])
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", env=env)
        if r.returncode == 0:
            (out / f"{name}.txt").write_text(r.stdout, encoding="utf-8")
            status = "ok"
        else:
            # A failed check leaves its last good register file in place.
            status = f"FAILED ({r.returncode}); {name}.txt not rewritten"
            failed.append(name)
        print(f"{name:<28s} {status}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
