Code of the reruns: commit 736d91bd3be3d4ed89daec327e6518282a685953 plus the uncommitted diff of scripts/, the dataset configs and the landscape statistics, sha256 d869500c7b7eab68e22a6e7605bfd5accd3db81a9b86ffe7dcb1c90c6ae82b87 over 47 dirty file(s) (patch output/code_diffs/code_diff_d869500c7b7e.patch, git-ignored), stamped at 2026-09-30T19:56:57+0100.
No file that a rerun imported changed while the check ran.

Dirty files at the start:
- ` M datasets-ehmi.json`
- ` M datasets-extended.json`
- ` M datasets-provoice.json`
- ` M datasets.json`
- ` M scripts/analyse_boba_adaptations.py`
- ` M scripts/analyse_boba_mo.py`
- ` M scripts/analyse_boba_robustness.py`
- ` M scripts/analyse_extra_runs.py`
- ` M scripts/analyse_fitted_companion.py`
- ` M scripts/anchor_noise_scale.py`
- ` M scripts/bo_sensor_error_simulation.py`
- ` M scripts/bo_synthetic_error_simulation.py`
- ` M scripts/budget_split.py`
- ` M scripts/calibrate_noise_from_data.py`
- ` M scripts/changepoint_compare.py`
- ` M scripts/compare_boba_arms.py`
- ` M scripts/design_rules_from_pilot.py`
- ` M scripts/elicitation_compare.py`
- ` M scripts/estimate_archival_error_processes.py`
- ` M scripts/evaluate_research_question.py`
- ` M scripts/factor_variance_shares.py`
- ` M scripts/heldout_remedies.py`
- ` M scripts/make_boba_paper_tables.py`
- ` M scripts/make_paper_figures.py`
- ` M scripts/onset_bound.py`
- ` M scripts/oracle_isolation.py`
- ` M scripts/oracle_isolation_acq_ranking.py`
- ` M scripts/replay_end_of_study.py`
- ` M scripts/replay_hitl_remedies.py`
- ` M scripts/replay_stopping.py`
- ` M scripts/review_checks/fresh_seed_replication.py`
- ` M scripts/review_checks/oracle_families.py`
- ` M scripts/run_review_checks.py`
- ` M scripts/sitting_by_magnitude.py`
- `?? scripts/check_provenance.py`
- `?? scripts/review_checks/budget_split_by_onset.py`
- `?? scripts/review_checks/extra_reach.py`
- `?? scripts/review_checks/floor_deployed.py`
- `?? scripts/review_checks/friedman_multiplicity.py`
- `?? scripts/review_checks/instrument_scale.py`
- `?? scripts/review_checks/mo_onset_bound.py`
- `?? scripts/review_checks/oracle_achievable.py`
- `?? scripts/review_checks/oracle_companion_estimators.py`
- `?? scripts/review_checks/oracle_sigmaf_seeds.py`
- `?? scripts/review_checks/sitting_sequential.py`
- `?? scripts/review_checks/sitting_vs_shiprule.py`
- `?? scripts/review_checks/tie_break.py`

| arm | driver | run | status | first differing column (trial or row) | rows differing | max abs diff | verdict |
|---|---|---|---|---|---|---|---|
| output-elicitation-rerun | elicitation | elicitation_runs.csv | exact |  | 0 of 40 | 0 | exact |

Verdicts: exact 1

Notes:
- output-elicitation-rerun: 40 runs (20 landscapes, seed 7, bias (clean runs) at 1.0; rating and comparison loops, clean), settings from the command recorded in paper/COMMANDS.md; wall-clock seconds not compared.
