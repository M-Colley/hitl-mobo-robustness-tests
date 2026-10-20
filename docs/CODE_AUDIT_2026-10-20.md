# Code review and extension, 2026-10-20

This records the initial review. The [2026-10-21 follow-up](VALIDATION_2026-10-21.md)
adds actual smoke benchmarks and Python 3.15 tests, and fixes the resume and
reaction-timing issues identified below. Keep the two stages distinct when
interpreting their validation scope.

Scope: static review of the simulation core, elicitation comparison,
evaluation timing, resume logic and launch/dependency workflow, plus focused
tests. No benchmark, provenance replay, model-fitting simulation sweep or paper
regeneration was executed. This is not an exhaustive review of every analysis
or a new empirical robustness result.

The supplied working folder lacked `scripts/`, `tests/`, requirements and
launchers, although Git HEAD `b285b907a` contained them. Recovered 174 missing
source/test/dependency/launcher files using create-only writes. Existing files
and the Git index were preserved. Recovered `paper/COMMANDS.md` separately to
record this extension's producers. Old result directories were not modified.

## Fixed defects

| Priority | Defect | Correction and verification |
|---|---|---|
| High | Learned-noise dropout simulation called `known_noise_variance` for its log and raised because dropout has no additive known variance. | `logged_noise_variance` records NaN for affected learned-noise metadata. Explicit known-noise dropout still fails. Deterministic unit tests cover active/onset/single-error/clean branches and other models. |
| High | Anchor detrending corrected GP inputs, but final deployment selection used the raw maximum rating. | `anchor_corrected_ratings` is used before deployment selection when the anchor model is enabled. An exact fixture shows the raw maximum chooses a worse late design and corrected ratings recover the better visited design. |
| Medium | Anchor regression centered ratings within anchor identity but left time centered globally, attenuating exact linear drift. | Both time and ratings are centered within each identity. Tests cover interleaved anchors, no repeat anchors and constant values. |
| High | `elicitation_compare.py --summary-only` started an experiment if the expected input CSV was absent. | The command fails before loading landscape stats, creating output directories, building tasks or starting workers. A regression test replaces those operations with forbidden calls. |
| Medium | Invalid/unknown elicitation faults could silently become clean feedback; duplicate task dimensions and nonfinite settings were accepted. | Validate fault names, magnitudes, probabilities, seeds, budgets, candidate counts, worker counts and landscape lists before dispatch. Valid historical settings keep their numerical behavior. |

The default no-anchor/no-dropout simulation branch keeps the original draw
order. The corrected anchor arms need fresh output directories: old logged
values describe the former implementation. No bit-for-bit historical replay
has been performed in this session.

## Remaining legacy issues

1. **Resume matches too little.** `_load_resumable_run` checks log shape,
   iteration sequence and columns, not all effective settings. Existing
   filenames and `_clean_run_settings` omit some combinations of initial
   samples, candidate pools, xi/kappa, replication, likelihood and inference
   choices. Use a fresh directory after changing code/settings. The new runner
   uses an immutable content/version manifest; the old pipeline has not been
   retrofitted because old directories lack sufficient identity metadata.
2. **Onset-zero reaction timing is mislabeled.** The evaluation response
   diagnostic uses onset+2, which compares initial random proposals at onset
   zero when five initial samples are used. The first adaptive proposal is
   trial six in that configuration. Final regret and its search/selection
   decomposition are separate from this diagnostic. Historical response
   tables were not silently regenerated; correcting them requires explicit
   initial-sample provenance and fresh outputs.
3. **Elicitation interruption loses work.** The legacy comparison holds
   results in memory and writes once after the pool finishes. It also repeats
   the same clean computation for every error/magnitude and recalculates some
   environment setup per task. The new runner checkpoints each completed task,
   bounds queued futures, and computes each clean run once.
4. **Legacy proposal/error RNG coupling.** The old elicitation loop uses one
   RNG for response error and proposal-pool draws, so error can change future
   candidate pools independently of feedback learning. The new arm isolates
   catalog, policy, error, presentation and rater streams. Historical streams
   were preserved rather than relabeled as equivalent.
5. **Legacy pairwise ceiling ties favor the first option.** Its new proposal
   occupies that position. The new arm randomizes pair presentation and
   represents ties as soft preference labels of .5. Old capped/rounded numeric
   results still require earliest and uniform tie values side by side.

## New implementation and performance choices

`scripts/feedback_policies.py` provides random/Sobol controls, finite-catalog
CEM-inspired search, RBF ridge UCB, a leader/challenger dueling bandit,
Bradley–Terry learned-reward policy optimization, categorical DPO and GP LogEI.
The reward-policy and DPO implementations are explicitly one-step finite-action
analogues. They do not implement sequential RLHF or LLM training.

`scripts/human_feedback_benchmark.py` provides the environment, human-error
processes, judgment/exposure accounting, source/version-protected resume and
standard-process analysis. It reuses `paired_frame` and `summarise`, reports
clean price with gain, preserves first-index numeric ties and exact uniform
tie expectations, and emits the landscape count and per-landscape values.

The new implementation uses a shared Sobol catalog, cached policy fits, an
incremental ridge inverse, sparse dueling statistics, bounded process futures,
atomic per-run checkpoints and one BLAS thread per worker. Fit failures stop
the task instead of substituting random proposals. Source and settings hashes
protect resumability. Timing is logged per task; no speedup or runtime claim
has been measured. The finite catalog improves compute predictability but
introduces a discretization gap that is logged and must be reported.

During independent review, the first dueling implementation was found to query
disjoint unseen pairs until catalog exhaustion. At the intended large catalog
and small budget that would never use its adaptive branch. It now revisits the
feedback-selected leader after the first query, with a test proving the next
proposal changes when the first preference changes.

## Validation and running elsewhere

Final focused result: **109 tests passed** on Python 3.13.12. Two warnings
reported the installed torch JIT deprecation. The smoke, screening and held-out
plans validated without experiment execution; `compileall` passed. Dependency
installation in a fresh environment has not been exercised here.

Reproduction command for the focused checks (no real objective evaluations):

```
python -m pytest -o addopts= -q tests/test_feedback_policies.py tests/test_human_feedback_benchmark.py tests/test_core_audit.py tests/test_elicitation.py::test_summary_only_never_starts_experiments_when_the_run_log_is_missing tests/test_elicitation.py::test_invalid_experiment_arguments_fail_before_dispatch tests/test_elicitation.py::test_invalid_landscape_lists_fail_before_dispatch tests/test_boba_benchmarks.py::test_known_noise_schedule_matches_the_injected_error tests/test_simulation_utils.py::test_apply_sensor_error_uses_jitter_for_dropout
```

The focused suite covers helper fixes, read-only modes, policy numerical
contracts, standard-process estimands, budget accounting with a fake
environment, unsafe recovery suppression and resume identities. Synthetic
feedback fixtures do not evaluate any analytic benchmark. Static compilation
and all supplied configuration plans are checked separately. The full existing
test suite includes optimizer/simulation loops and was deliberately not run
under the request to avoid benchmark execution.

See [HUMAN_FEEDBACK.md](HUMAN_FEEDBACK.md) for setup, exact execution commands,
method limitations, error definitions, held-out design and primary references.
Build a transfer archive with `python scripts/build_feedback_bundle.py`.
The ZIP includes source/configuration/tests only; no prior result logs,
dataset caches, Git internals or code-diff store are bundled.
