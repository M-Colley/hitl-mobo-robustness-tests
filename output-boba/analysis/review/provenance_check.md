Code of the reruns: commit 736d91bd3be3d4ed89daec327e6518282a685953 plus the uncommitted diff of scripts/, the dataset configs and the landscape statistics, sha256 270839b3b7269964ba1a52b883fc2aefca9583f9ec52cc14dfebe2bcfd6438b6 over 50 dirty file(s) (patch output/code_diffs/code_diff_270839b3b726.patch, git-ignored), stamped at 2026-09-30T21:35:17+0100.
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
- ` M scripts/review_checks/c6_ratio.py`
- ` M scripts/review_checks/fresh_seed_replication.py`
- ` M scripts/review_checks/oracle_families.py`
- ` M scripts/run_review_checks.py`
- ` M scripts/sitting_by_magnitude.py`
- `?? scripts/check_provenance.py`
- `?? scripts/review_checks/budget_split_by_onset.py`
- `?? scripts/review_checks/elicitation_determinism.py`
- `?? scripts/review_checks/extra_reach.py`
- `?? scripts/review_checks/floor_deployed.py`
- `?? scripts/review_checks/frag_residuals.py`
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
| output-boba | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-adapt-nigp | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_slip_jit20_std0.4_nigp.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-adapt-rep10 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_rep10.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-adapt-rep10-obs | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_inc-observed_max_rep10.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-adapt-rerate | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_rerate3x2.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-adapt-rerate-slip | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_slip_jit20_std0.4_rerate3x2.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-adapt-studentt | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_misclick_jit20_std0.4_studentt.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-adapt-studentt-rbf | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_misclick_jit20_std0.4_studenttrbf.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-budget100 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0.csv | exact |  | 0 of 100 | 0 | exact |
| output-boba-budget25 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0.csv | exact |  | 0 of 25 | 0 | exact |
| output-boba-ceiling | synthetic | bo_sensor_error_branin_value_logei_seed12_jittered_exact_gaussian_jit0_std1.0_ceil0.9-fixed.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-confirmatory | synthetic | bo_sensor_error_branin_value_logei_seed27_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-extensions | synthetic | bo_sensor_error_bump_a16_w0.05_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-idea-anchor | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_anchored.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-idea-anchors | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_anch5x3-detrend.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-idea-hold | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_hold5@0.6.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-idea-selfreport | synthetic | bo_sensor_error_branin_value_logei_seed12_jittered_exact_gaussian_jit20_std1.0_noise-self_report.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-idea-selfreport-r0 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_noise-self_report_conf-r0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-idea-selfreport-r0.3 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_noise-self_report_conf-r0.3.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-idea-selfreport-r0.6 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_noise-self_report_conf-r0.6.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-idea-shiplcb | synthetic | bo_sensor_error_branin_value_shiplcb_seed7_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-incumbent | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_inc-observed_max.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-instrument | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-knownnoise | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_noise-known.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-ladder | synthetic | bo_sensor_error_bump_d4_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-misclick | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_misclick_jit20_std0.4.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-missing-drop | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_missing_low_jit0_std0.3_miss-drop.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-missing-impute | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_missing_low_jit0_std0.3_miss-impute_low.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-mo | synthetic | bo_sensor_error_branincurrin_multi_objective_qlognehvi_seed7_jittered_exact_gaussian_jit20_std1.0.csv | differs | x0 (22) | 30 of 50 | 4.75 | exact with acquisition order before qkg/replei were inserted |
| output-boba-mo-halo | synthetic | bo_sensor_error_branincurrin_multi_objective_qlognehvi_seed7_jittered_exact_gaussian_jit20_std1.0_xc0.85.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-mo-halo-backfit | synthetic | bo_sensor_error_branincurrin_multi_objective_qlognehvi_seed7_jittered_exact_gaussian_jit20_std1.0_xc0.85_halo-backfit.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-q-aei | synthetic | bo_sensor_error_branin_value_aei_seed7_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-q-inclcb | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_inc-lcb.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-q-iu-0.05 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_slip_jit20_std0.05_iu16-0.05.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-q-iu-0.15 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_slip_jit20_std0.15_iu16-0.15.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-q-iu-0.4 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_slip_jit20_std0.4_iu16-0.4.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-q-mind | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_mind0.05.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-q-ts | synthetic | bo_sensor_error_branin_value_ts_seed7_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-relay | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_rater-roundrobin5-tau2.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-relay-backfit-block | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_rater-block10-tau2_raterfit-backfit.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-relay-backfit-rr | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_rater-roundrobin5-tau2_raterfit-backfit.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-robust | synthetic | bo_sensor_error_branin_value_qkg_seed7_jittered_exact_gaussian_jit20_std1.0.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-sched-U | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_sched-U.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-sched-back10 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_sched-back10.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-sched-front10 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_sched-front10.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-sched-front20 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0_sched-front20.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-session100 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.4142.csv | exact |  | 0 of 100 | 0 | exact |
| output-boba-session25 | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std0.7071.csv | exact |  | 0 of 25 | 0 | exact |
| output-boba-slip | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_slip_jit20_std0.4.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-slip-actual | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_slip_jit20_std0.4_rec-actual.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-spike | synthetic | bo_sensor_error_branin_value_logei_seed12_jittered_exact_spike_jit0_std0.25_sp0.15-20.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-spike-clip | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_spike_jit0_std0.25_sp0.15-20.csv | exact |  | 0 of 50 | 0 | exact |
| output-boba-spike-rrp | synthetic | bo_sensor_error_branin_value_logei_seed7_jittered_exact_spike_jit0_std0.25_sp0.15-20_relevancepursuit.csv | exact |  | 0 of 50 | 0 | exact |
| output-elicitation | elicitation | elicitation_runs.csv | differs | shipped_true (0) | 4 of 4 | 4.84 | differs |
| output-elicitation-rerun | elicitation | elicitation_runs.csv | exact |  | 0 of 4 | 0 | exact |
| output-fitted-adapt-rep10-prefix/opticarvis | sensor | bo_sensor_error_opticarvis_composite_logei_seed7_jittered_gradient_boosting_gaussian_jit20_std1.868543.csv | differs | Trajectory (6) | 50 of 50 | 189 | exact with dataset config datasets.json at 3d2323aed~1 |
| output-fitted-adapt-rep10-prefix/provoice | sensor | bo_sensor_error_provoice_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std1.49182.csv | differs | InterventionLightingTransparency (6) | 50 of 50 | 26.4 | exact with dataset config datasets.json at 3d2323aed~1 |
| output-fitted-adapt-rep10/ehmi | sensor | bo_sensor_error_ehmi_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std1.823566.csv | exact |  | 0 of 50 | 0 | exact |
| output-fitted-adapt-rep10/opticarvis | sensor | bo_sensor_error_opticarvis_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std0.781517.csv | exact |  | 0 of 50 | 0 | exact |
| output-fitted-adapt-rep10/provoice | sensor | bo_sensor_error_provoice_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std0.448589.csv | exact |  | 0 of 50 | 0 | exact |
| output-fitted-noaug-prefix/opticarvis | sensor | bo_sensor_error_opticarvis_composite_logei_seed7_jittered_gradient_boosting_gaussian_jit20_std1.868543.csv | differs | Trajectory (6) | 50 of 50 | 181 | exact with dataset config datasets.json at 3d2323aed~1 |
| output-fitted-noaug-prefix/provoice | sensor | bo_sensor_error_provoice_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std1.49182.csv | differs | InterventionLightingTransparency (6) | 50 of 50 | 5.49 | exact with dataset config datasets.json at 3d2323aed~1 |
| output-fitted-noaug/ehmi | sensor | bo_sensor_error_ehmi_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std1.823566.csv | exact |  | 0 of 50 | 0 | exact |
| output-fitted-noaug/opticarvis | sensor | bo_sensor_error_opticarvis_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std0.781517.csv | exact |  | 0 of 50 | 0 | exact |
| output-fitted-noaug/provoice | sensor | bo_sensor_error_provoice_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std0.448589.csv | exact |  | 0 of 50 | 0 | exact |
| output-fitted-prefix/opticarvis | sensor | bo_sensor_error_opticarvis_composite_logei_seed7_jittered_gradient_boosting_gaussian_jit20_std1.868543.csv | differs | Trajectory (6) | 50 of 50 | 188 | exact with dataset config datasets.json at 3d2323aed~1 |
| output-fitted-prefix/provoice | sensor | bo_sensor_error_provoice_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std1.49182.csv | differs | InterventionLightingTransparency (6) | 50 of 50 | 29.7 | exact with dataset config datasets.json at 3d2323aed~1 |
| output-fitted/ehmi | sensor | bo_sensor_error_ehmi_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std1.823566.csv | exact |  | 0 of 50 | 0 | exact |
| output-fitted/opticarvis | sensor | bo_sensor_error_opticarvis_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std0.781517.csv | exact |  | 0 of 50 | 0 | exact |
| output-fitted/provoice | sensor | bo_sensor_error_provoice_composite_logei_seed7_jittered_extra_trees_gaussian_jit20_std0.448589.csv | exact |  | 0 of 50 | 0 | exact |
| output-oracle-iso-gaussian_process/iso_branin | sensor | bo_sensor_error_iso_branin_composite_logei_seed7_jittered_gaussian_process_gaussian_jit0_std0.911017.csv | exact |  | 0 of 50 | 0 | exact |
| output-oracle-iso-mlp/iso_branin | sensor | bo_sensor_error_iso_branin_composite_logei_seed7_jittered_mlp_gaussian_jit0_std0.836126.csv | exact |  | 0 of 50 | 0 | exact |
| output-oracle-iso/iso_branin | sensor | bo_sensor_error_iso_branin_composite_logei_seed7_jittered_gradient_boosting_gaussian_jit0_std1.033559.csv | exact |  | 0 of 50 | 0 | exact |

Verdicts: differs 1, exact 65, exact with acquisition order before qkg/replei were inserted 1, exact with dataset config datasets.json at 3d2323aed~1 6

Notes:
- output-fitted-adapt-rep10-prefix/provoice: the dataset config now differs from the recorded dataset in: provoice.objective_map.
- output-fitted-noaug-prefix/provoice: the dataset config now differs from the recorded dataset in: provoice.objective_map.
- output-fitted-prefix/provoice: the dataset config now differs from the recorded dataset in: provoice.objective_map.
- output-boba-instrument: logged under the legacy name (clip/round not in the name); the rerun is named bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0_clip-8,8_round0.55.csv rerun file name differs from the logged one.
- output-boba-spike-clip: logged under the legacy name (clip/round not in the name); the rerun is named bo_sensor_error_branin_value_logei_seed7_jittered_exact_spike_jit0_std0.25_sp0.15-20_clip-sample.csv rerun file name differs from the logged one.
- output-elicitation: 4 runs (ackley, seed 7, bias at 1.0; rating and comparison loops, noisy and clean), settings from the command recorded in paper/COMMANDS.md; wall-clock seconds not compared.
- output-elicitation-rerun: 4 runs (ackley, seed 7, bias at 1.0; rating and comparison loops, noisy and clean), settings from the command recorded in paper/COMMANDS.md; wall-clock seconds not compared.
