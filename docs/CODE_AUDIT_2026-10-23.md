# Code audit, 2026-10-23

Reviewed the available codebase and the current worktree, including the new
Schmitt decision replay. The inventory includes 90 tracked production Python
modules, 58 tracked test modules, 26 PowerShell launchers, the paper builder,
and the additional replay and test files. Review combined static checks,
parallel inspection of the major subsystems, the full regression suite, and
focused reproductions. This is not a proof of every mathematical implementation
or a rerun of the historical experiment sweeps.

The existing changes to `paper/COMMANDS.md`,
`scripts/replay_schmitt_decisions.py`, and its test file were preserved.
No implementation or generated research output was changed during the initial
audit. The requested follow-up corrections are recorded below; the finding
descriptions and line numbers retain the pre-fix state for reference.

## Follow-up corrections

All eleven findings have been addressed in the working tree:

| Finding | Correction | Regression coverage |
|---|---|---|
| 1 | Apply landscape/seed filters before derivation, cached scoring and clean pricing; exclude unrelated cached failures. | All four CLI branches, complete held-out single-landscape runs, numerical comparator checks with oversized caches. |
| 2 | Validate actual logged source-screen seeds before copying, simulation or evaluation. | Reject overlap in both execution modes, honor logged rather than filename identities, allow valid cached fresh-seed analyses. |
| 3 | Reject multi-objective known/self-reported noise explicitly, consistent with the documented single-objective scope. | Direct clean/noisy simulation and CLI dry-run validation before loading objectives. |
| 4 | Accept underscore error labels and full scientific-notation magnitudes with explicit suffix boundaries. | Missing-response indexing and exponent/variant parsing. |
| 5 | Preserve registered treatment/reference variants, including the explicit empty variant. | Registered cross-arm selection on multi-variant inputs. |
| 6 | Include the external robust surrogate digest in both Student-t fingerprints. | Changed surrogate rejects stale resume; unrelated likelihood inputs and cached environment remain unaffected. |
| 7 | Record evaluator failures, remove stale completion markers and write arm/global success only after successful work. | Actual PowerShell function success/failure/stale-marker cases and complete-driver failure exit. |
| 8 | Dispatch final rerating before random/Sobol proposals. | Rerating schedules for LogEI/random/Sobol and exact ordinary model-free draw order. |
| 9 | Use continuous empirical-Bayes shrinkage formulas valid at zero between-user variance. | Actual zero variance estimate, near-zero limit and positive variance. |
| 10 | Suppress unstable per-problem ratios and export a suppression flag while retaining gain/price. | Negative, zero, tiny, boundary and safe costs, including mixed problem costs. |
| 11 | Derive plot descriptions and units from the selected metric and normalization. | Trajectory/final metrics and normalized/integrated units. |

Focused checks passed: 127 core/adaptation tests; 34 selected analysis regressions;
26 hierarchical/replay tests; and 15 confirmatory/PowerShell tests. The replay
and hierarchical checks also ran under Python 3.15 (9 passed, 3 optional skips).
The complete suite subsequently passed **1,334 tests, with 11 skipped and zero
failures**, on Python 3.13.12. All 70 added regression cases passed.

The paper table producer regenerated 29 tables into
`validation-local/code-fixes-20261023/tables`; all were byte-identical to the
published files. The figure producer regenerated six figures in the adjacent
`figures` directory, and their PNGs were visually inspected. Historical study
logs, tables and figures were preserved. Corrected runs need fresh output
directories: strict resume identities intentionally reject changed simulator
code, and this repair does not relabel previous experimental outputs.

The scoped provenance replay on `output-boba-missing-drop` differs from its
archive starting at trial 6. A control loaded the pre-fix simulation core from
commit `e16f588b87f74f8c33ac4676e1df9b35488b82cd` and replayed the same case
under the same installed environment: it matched the corrected core exactly
on all 50 rows and all 32 compared fields, excluding runtime/run identity and
the intentionally changed code fingerprint. Thus this archived-run difference
predates the repairs. The evidence is retained under
`validation-local/code-fixes-20261023/provenance` and in
`validation-local/code-fixes-20261023/prefixed_core_comparison.json`.

## Confirmed findings

### 1. [P1] Requested budget-analysis seed and landscape filters are ignored

Location: `scripts/budget_split.py:892`.

The CLI accepts `--functions` and `--seeds`, but neither argument is used after
parsing. `main()` loads and scores the sweep without those filters. A request
for held-out seeds can therefore include the selection seeds, and a request
for one landscape can include the entire suite.

Reproduction used the actual saved sweep and a temporary copy of the existing
derived-choice CSV. Calling `main()` with `--functions branin --seeds
12,13,14,15,16 --summary-only` passed 64,000 rows covering all 20 landscapes and
seeds 7–16 into `score_policies`. Scoring was intercepted before calculation
or output writes. Apply both filters before deriving choices and scoring,
including the clean-twin and summary-only paths.

### 2. [P1] Screening seeds can be relabeled as confirmatory seeds

Location: `scripts/confirmatory_followup.py:764–773`.

The overlap check defines screening seeds by first subtracting the requested
confirmatory seeds from the pooled data. Any screening seed inside that block
is consequently invisible to the check and is later included in confirmatory
tests. It checks for unexpected seeds above the boundary, rather than
disjointness from the actual broad screen.

A fixture with four broad-screen logs for acquisitions pi/logpi and seeds
12/13, `--seed-start 12`, and two requested seeds copied all four logs and
exported seeds 12/13 as confirmatory without running a simulation. Evaluation
and plotting were stubbed; the actual copy, seed-selection and statistical
reporting code ran. Record the source screen's seed identities before pooling
and reject an overlapping requested block.

### 3. [P2] Multi-objective known-noise modes silently fit learned-noise models

Locations: `scripts/bo_sensor_error_simulation.py:4296` and `:2992`.

The multi-objective branch calls `_fit_model_list_gp` without observation
variances. Its component GPs therefore learn noise even when the configuration
and log say `observation_noise="known"` or `"self_report"`. The fixed-variance
path is implemented only in the scalar branch.

The synthetic CLI accepts both combinations. A three-trial, two-objective toy
run, with fitting and candidate optimization mocked, logged known variances
of one while constructing ordinary `GaussianLikelihood` models for both
objectives, with no acquisition fallbacks. Supply the per-objective variances
or reject unsupported combinations before execution.

### 4. [P2] Ship-rule filename parsing drops valid missing-response runs

Location: `scripts/rescore_ship_rules.py:80`.

The error-model regex accepts only `[a-z0-9]+`, excluding supported
`missing_low` and `missing_mcar` labels. All 800 actual noisy missing-response
logs under `output-boba-missing-drop` return `None` from `parse_run_name`.
Indexing Ackley/LogEI/`missing_low` returns only five clean baselines, silently
discarding the noisy evidence.

Example rejected log:
`bo_sensor_error_ackley_value_logei_seed10_jittered_exact_missing_low_jit0_std0.15_miss-drop.csv`.

The same regex's numeric grammar, and the variant parser at
`scripts/evaluate_research_question.py:99`, mishandle scientific notation.
The simulators directly interpolate a float such as `1e-05`; rescoring reads
that as magnitude 1 and variant `e-05`, while evaluation invents the variant.
Parse the complete supported label and float grammars, with suffix boundaries
made explicit.

### 5. [P2] Built-in ship-rule cross comparisons discard their variant filters

Location: `scripts/analyse_ship_rules.py:270`.

`cross_specs()` replaces registered `ARMS` treatment/reference variants with
`"auto"`. Built-in ceiling, spike and relay comparisons then fail to select
their intended variants from a multi-variant per-run table.

`cross_specs(["ceiling-anchored"], [])` followed by `pick_variant()` on a fixture
containing both fixed and anchored ceiling variants raises instead of choosing
the registered one. The corresponding rescoring tables are absent locally,
so this reproduction used an in-memory fixture. Preserve the variants from
the registered arm specification.

### 6. [P2] Student-t resume identity omits the surrogate implementation

Location: `scripts/bo_sensor_error_simulation.py:4767–4777`.

The strict resume fingerprint hashes the simulation core and timing helper,
but not `robust_gp.py`, which implements the Student-t surrogate. Changing
that module can therefore leave the fingerprint unchanged and reuse stale
runs during a partial resume.

Instrumenting the source hashing for a Student-t configuration showed only
the core and timing files being hashed. Changing the simulated digest for
`robust_gp.py` and clearing the environment cache left the fingerprint
identical. Include the surrogate implementation in the run identity.

### 7. [P2] Budget-neutral driver marks failed evaluations complete

Location: `run_boba_budget_neutral.ps1:180–182`.

Evaluation failure only writes a log message. It does not append to the
failure list or prevent `ARM_COMPLETE` from being written, so the driver can
report successful completion with stale or absent analysis files.

Executing the actual `Run-Arm` function in a temporary directory with an
evaluator that exits 23 produced `FailureCount=0` and `ArmComplete=true`.
Propagate the evaluator's failure and write the completion marker only after
every required evaluation succeeds.

### 8. [P2] Random and Sobol silently skip requested final rerating

Location: `scripts/bo_sensor_error_simulation.py:4224–4231`.

The random/Sobol candidate branches precede the final-rerate branch, so the
accepted `--final-rerate` option has no effect on their proposals. `--acq all`
includes both methods, allowing a rerating arm to mix different procedures.

A deterministic random-search toy run with eight trials, two initial samples,
and final rerating of one design twice still visited eight unique designs;
neither final trial repeated an earlier design. Honor final rerating before
dispatching new proposals, or reject the unsupported combinations explicitly.

### 9. [P2] Zero user-effect variance disables shrinkage instead of fixing it

Location: `scripts/hierarchical_oracle_test.py:515–518`.

`variance_components()` legitimately floors the estimated between-user
variance `tau2` at zero. `warm_evaluation()` then sets prior precision to zero,
which treats that boundary as an unregularized user offset. A zero variance
actually fixes the offset at zero. This can distort warm prediction error,
likelihood and interval coverage.

An AST-extracted fixture executed the real functions without NumPyro.
Residuals `[-1,1,-1,1]` across two groups yield `tau2=0`, `sigma2=2`.
With a pooled prediction of zero and a context residual of ten, the warm query
mean becomes ten and its SD becomes two; they should remain zero and square
root of two. At `tau2=1e-12` the implementation returns approximately the
correct limiting values, confirming the discontinuity. The equivalent formulas
`alpha=tau2*sum(residuals)/(sigma2+n*tau2)` and
`alpha_var=tau2*sigma2/(sigma2+n*tau2)` handle the zero boundary.

### 10. [P2] Schmitt per-problem recovery exports bypass denominator safeguards

Location: `scripts/replay_schmitt_decisions.py:278–280`.

The new replay suppresses pooled recovery ratios when reference cost is below
0.01 or negative on a problem, but exports per-problem recovery ratios without
the corresponding safeguard. This violates the repository's rule against
reporting recovery shares with near-zero reference cost.

A four-problem fixture with reference noisy regret 0.005, clean regret zero,
and treatment noisy regret zero yields a suppressed pooled recovery, but
`recovered=1.0` in every per-problem row. Apply a per-problem threshold and
suppression flag, retaining absolute gain and price.

### 11. [P2] Arm-comparison plot labels the wrong units and sometimes time base

Location: `scripts/compare_boba_arms.py:258`.

The x-axis always says post-onset per-iteration excess regret in landscape SDs.
The default analytic comparison has already divided values by `opt_z`, so its
unit is the fraction of achievable improvement. Selecting a final deployed
metric also leaves the trajectory label unchanged. Derive the label from the
selected metric and scale so the plot states the measured quantity correctly.

## Initial audit validation

`python -m pytest tests/ -q -ra` completed successfully on Python 3.13.12:
**1,264 passed, 11 skipped, zero failures**. The complete log is retained at
`C:/Users/Mark/AppData/Local/Temp/hitl-code-review-20261023/pytest-py313.log`.
Nine skips require historical run logs or arm metadata absent from this
checkout; two require OpenBO selector integration. In particular, the main
sweep's exact logged-run regression tests did not execute. The existing
warnings include numerical and deprecation warnings from the scientific
libraries and deliberate boundary fixtures.

Checks already completed:

- `python scripts/check_paper.py --paper paper`: passed input, reference,
  citation and table-shape checks; no todo markers. No LaTeX toolchain is
  installed, so compilation was not verified.
- `ruff check scripts tests paper/build_overleaf_bundle.py --select
  E9,F63,F7,F82 --output-format concise`: passed.
- All tracked PowerShell scripts parsed successfully using PowerShell's AST
  parser.
- `git diff --check`: passed.
- Focused reproductions above used temporary files or in-memory fixtures;
  historical run logs and generated analyses were preserved.

The Schmitt selector integration dependency is unavailable at the documented
local OpenBO source path. Its integration coverage is therefore limited to
the existing optional-test behavior, with the pure replay code reviewed and
tested independently. No empirical research claims were re-estimated by this
audit; the downstream numerical effects of the defects remain unmeasured.

## Final repair validation

- `python -m pytest tests/ -o addopts= -q -ra`: **1,334 passed, 11 skipped**,
  141 warnings, 21m45s. Complete log:
  `validation-local/code-fixes-20261023/pytest-py313.log`.
- The same nine historical-data/metadata and two OpenBO integration skips
  remain. Exact replay of the absent main-sweep logs is still unavailable.
- Fatal Python lint, PowerShell parsing, static paper checks and
  `git diff --check` passed after the implementation changes.
- All 29 regenerated paper tables matched the published files byte for byte;
  six figures were regenerated and visually inspected in the validation
  directory. No published study outputs were overwritten.
- The same-environment, pre-fix/fixed-core control matched exactly across
  50 trials and 32 compared fields. The available historical archive mismatch
  is documented above and predates the fixes.

All eleven software findings are resolved in this repair. The pre-existing
replay work and command history were preserved.

Before the local checkout is removed, its ignored validation evidence and smoke
results are retained separately in `hitl-mobo-local-results-2026-10-24.zip`.
The archive excludes virtual environments, caches and distribution bundles;
its manifest records the committed source revision and file hashes. The saved
pre-fix control helper is pinned to commit
`e16f588b87f74f8c33ac4676e1df9b35488b82cd`, so a subsequent commit does not change
which simulation core it compares.
