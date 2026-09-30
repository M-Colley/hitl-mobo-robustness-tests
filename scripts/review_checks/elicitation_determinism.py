"""Does the comparison-loop rerun reproduce in another worker schedule?

output-elicitation-rerun/ is the log the comparison appendix reads (the command
in paper/COMMANDS.md, Section 8, twelve workers). This reruns a subset of it,
three landscapes and seeds 8 and 9 (96 of its 960 runs: two loops, noisy and
clean, four faults), with six workers into a temporary directory, merges the new
rows with the logged ones on dataset, elicitation, error_model, magnitude, seed
and apply_error, and compares every other column except the wall-clock
``seconds``, exactly and then with np.isclose(rtol=1e-9, atol=1e-12).

    python scripts/review_checks/elicitation_determinism.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import elicitation_compare as ec  # noqa: E402

LOG = REPO / "output-elicitation-rerun" / "elicitation_runs.csv"
KEYS = ["dataset", "elicitation", "error_model", "magnitude", "seed", "apply_error"]
EXCLUDED = {"seconds"}
SUBSET = ["--functions", "branin,stevens,weber_fechner", "--error-models", "bias,drift,ceiling,gaussian",
          "--magnitudes", "1", "--seeds", "8,9", "--iterations", "20", "--workers", "6"]


def main() -> int:
    if not LOG.is_file():
        print(f"{LOG.relative_to(REPO)} is missing; run the command of paper/COMMANDS.md, Section 8")
        return 1
    logged = pd.read_csv(LOG, float_precision="round_trip")
    with tempfile.TemporaryDirectory(prefix="elic-determinism-") as tmp:
        # The script's own report names the temporary directory (a local path);
        # keep it out of the register file.
        with contextlib.redirect_stdout(io.StringIO()):
            ec.main(SUBSET + ["--output-dir", tmp])
        new = pd.read_csv(Path(tmp) / "elicitation_runs.csv", float_precision="round_trip")
    merged = new.merge(logged, on=KEYS, how="left", suffixes=("_new", "_log"), indicator=True)
    missing = int((merged["_merge"] != "both").sum())
    columns = [c for c in new.columns if c not in KEYS and c not in EXCLUDED]
    print(f"rerun: {len(new)} runs ({' '.join(SUBSET)}); matched in the log: {len(new) - missing}")
    exact_all, close_all = True, True
    for c in columns:
        a, b = merged[f"{c}_new"], merged[f"{c}_log"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            fa, fb = a.to_numpy(float), b.to_numpy(float)
            both_nan = np.isnan(fa) & np.isnan(fb)
            exact = int(np.sum(~((fa == fb) | both_nan)))
            close = int(np.sum(~(np.isclose(fa, fb, rtol=1e-9, atol=1e-12) | both_nan)))
        else:
            exact = close = int(np.sum(a.astype(str).to_numpy() != b.astype(str).to_numpy()))
        exact_all &= exact == 0
        close_all &= close == 0
        if exact:
            print(f"  {c}: {exact} rows differ exactly, {close} beyond tolerance")
    verdict = "exact" if exact_all and not missing else ("within tolerance" if close_all and not missing
                                                         else "differs")
    print(f"columns compared: {len(columns)} (every column but the keys and {sorted(EXCLUDED)})")
    print(f"verdict: {verdict}")
    return 0 if verdict != "differs" else 2


if __name__ == "__main__":
    sys.exit(main())
