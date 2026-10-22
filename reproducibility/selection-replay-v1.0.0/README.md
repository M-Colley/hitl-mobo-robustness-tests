# Synthetic MOBO decision replay, v1.0.0

This self-contained artifact supplies the 100 raw synthetic CSV archives,
the exact replay/helper source closure, the historical OpenBO selector,
benchmark statistics, reference results and hashes for the Schmitt decision
comparison. No private checkout or participant data is needed.

## Reproduce

Use Python 3.15 and the pinned CPU environment. The validated host was
Windows x86-64; path handling is portable, but other platforms are unverified.

```text
python -m venv .venv
# Activate the virtual environment using the command for your shell.
python -m pip install -r requirements.txt
python run_replay.py --verify-only
python run_replay.py --output-dir ../selection-replay-output
```

The runner verifies every bundled payload hash, performs only the fixed-log
replay, and checks five reference files byte-for-byte plus the uncompressed
98,100-row choice table. It rejects a nonempty output directory. Entry-point
guards reject analytic objective evaluations or GP fitting if attempted.
The run does not regenerate the optimization trajectories.

The pinned installation can be large (Torch dependencies); the data/source
archive itself is small. Reuse an existing environment matching environment.json
to avoid installing another Torch copy. No package installation was needed for
the release's extraction validation.

## Data and provenance

All 100 archives are exact analytic BraninCurrin, ZDT1, DTLZ2 or VehicleSafety
benchmark logs: qLogNEHVI, seeds 12--16 and 50 trials each. There are 20 clean
and 80 Gaussian-error archives (1 landscape SD, onset 0/20, cross-correlation
0/.85). They contain numerical design/outcome/error fields, simulator run
UUIDs and allowlisted categorical metadata. There are no participant names,
ratings from real people, contacts, clinical records or free-text responses.
synthetic-data-review.json records the allowlist and all original hash checks.

HITL tracked helper content is pinned to commit
e16f588b87f74f8c33ac4676e1df9b35488b82cd. OpenBO's base import dependencies
come from commit 2057d1a9cb4299b2bc8393b163a2081f91abc1c9. The replay and
historical decision package were uncommitted measurement-time additions;
their exact bytes and hashes are included explicitly. See source-provenance.json
and artifact_manifest.json. This is a dependency subset, not either full repo.
The bundled tracked helpers preserve the measured Windows CRLF bytes; provenance
also records canonical Git blob hashes and verifies that only line endings differ.
The bundled scalar and MO statistics are context for exact helper imports;
the replay does not calculate or refit them.

## Result and scope

The replay compares weighted sum, weighted nearest ideal and knee-weighted
selection with alpha=.5 on each fixed candidate archive. Declared linear and
quadratic-distance truth utilities have separate matching references.
Preference-weight noise (log SD 0/.25/1) is separate from recorded objective
error. Clean price uses exact outcomes and weights on the same archive.

Under recorded objective errors and exact preferences, pooled final selection
regret was .237706 for the linear weighted reference versus .332108 for the
knee rule (clean cost +.072301). Under quadratic-distance truth, the matching
nearest-ideal reference was .452719 versus .547495 for knee (clean cost
+.032571). These are within-archive true-front-normalized utility units.
The data support descriptive four-problem comparisons, not general robustness,
global optimization, real human benefit or MPC stability claims.

## Integrity and licensing

artifact_manifest.json hashes the payload; the GitHub release supplies the ZIP
SHA256 and its pinned source commit. Public validation records that all 98,100
rows and reference metrics reproduced from a fresh extraction. Original
measurement records remain unchanged. HITL/OpenBO source license texts are
included as LICENSE-HITL.txt and LICENSE-OpenBO.txt; the synthetic log data and
new artifact wrapper are distributed under the HITL MIT terms.
