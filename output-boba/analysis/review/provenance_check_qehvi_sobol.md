Code of the reruns: commit 736d91bd3be3d4ed89daec327e6518282a685953 plus the uncommitted diff of scripts/, the dataset configs and the landscape statistics, sha256 1a1600eed4eb435fb11d9208b88fef38fed600b31921ed8ba49fe09652003560 over 37 dirty file(s) (patch output/code_diffs/code_diff_1a1600eed4eb.patch, git-ignored), stamped at 2026-09-30T11:21:56+0100.
No file that a rerun imported changed while the check ran.

Dirty files at the start:
- ` M datasets-ehmi.json`
- ` M datasets-extended.json`
- ` M datasets-provoice.json`
- ` M datasets.json`
- ` M scripts/analyse_boba_adaptations.py`
- ` M scripts/analyse_boba_mo.py`
- ` M scripts/analyse_boba_robustness.py`
- ` M scripts/analyse_fitted_companion.py`
- ` M scripts/anchor_noise_scale.py`
- ` M scripts/bo_sensor_error_simulation.py`
- ` M scripts/bo_synthetic_error_simulation.py`
- ` M scripts/budget_split.py`
- ` M scripts/calibrate_noise_from_data.py`
- ` M scripts/changepoint_compare.py`
- ` M scripts/compare_boba_arms.py`
- ` M scripts/estimate_archival_error_processes.py`
- ` M scripts/evaluate_research_question.py`
- ` M scripts/factor_variance_shares.py`
- ` M scripts/onset_bound.py`
- ` M scripts/oracle_isolation.py`
- ` M scripts/oracle_isolation_acq_ranking.py`
- ` M scripts/replay_end_of_study.py`
- ` M scripts/review_checks/fresh_seed_replication.py`
- ` M scripts/review_checks/oracle_families.py`
- ` M scripts/sitting_by_magnitude.py`
- `?? scripts/check_provenance.py`
- `?? scripts/review_checks/budget_split_by_onset.py`
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
| output-boba | synthetic | bo_sensor_error_branin_value_sobol_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | objective_observed (21) | 30 of 50 | 5.32 | exact with acquisition order before qkg/replei were inserted |
| output-boba-extensions | synthetic | bo_sensor_error_bump_a16_w0.05_value_sobol_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | objective_observed (21) | 30 of 50 | 5.32 | exact with acquisition order before qkg/replei were inserted |
| output-boba-ladder | synthetic | bo_sensor_error_bump_d4_value_sobol_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | objective_observed (21) | 30 of 50 | 5.32 | exact with acquisition order before qkg/replei were inserted |
| output-boba-mo | synthetic | bo_sensor_error_branincurrin_multi_objective_qehvi_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | x0 (22) | 30 of 50 | 5.19 | exact with acquisition order before qkg/replei were inserted |

Verdicts: exact with acquisition order before qkg/replei were inserted 4
