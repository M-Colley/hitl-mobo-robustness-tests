# Commands behind the paper's numbers

Every table under `paper/tables/` is written by
`python scripts/make_boba_paper_tables.py --analysis output-boba/analysis --out paper/tables`,
and every figure under `paper/figures/` by `python scripts/make_paper_figures.py`.
This file records the commands that produce the analysis outputs those two read,
in the order they must run, including the ones that were once run by hand.
Seeds are fixed inside each script or given here.

## Main sweep and its syntheses

```
powershell -File run_boba_pipeline.ps1                 # output-boba: sweep, evaluation, GP noise diagnostic, synthesis
python scripts/diagnose_gp_noise.py --input-dir output-boba --per-cell 3 --at-iterations 8,15,25,50
python scripts/analyse_boba_robustness.py --input-dir output-boba --output-dir output-boba/analysis
python scripts/rescore_ship_rules.py                   # ship_rules_per_run.csv
python scripts/decompose_regret.py --input-dir output-boba
python scripts/replay_end_of_study.py --workers 5      # end_of_study*/ (k-sweep and k-wide runs set --tournament-k)
python scripts/budget_split.py                         # rho = 1, the paper's sitting
python scripts/replay_hitl_remedies.py                 # hitl_remedies/
python scripts/analyse_boba_adaptations.py             # adaptations_recovery.csv
```

## The 2026-09-22 review round (outputs in `output-boba/analysis/review/`)

```
python scripts/onset_bound.py
python scripts/heldout_remedies.py
python scripts/remedy_decomposition.py
python scripts/smooth_subset_and_gp_diagnostics.py
python scripts/bootstrap_coverage.py
python scripts/estimate_archival_error_processes.py
python scripts/oracle_isolation.py make-data
python scripts/oracle_isolation.py select
python scripts/oracle_isolation.py calibrate
powershell -File run_oracle_isolation.ps1
python scripts/oracle_isolation.py analyse
powershell -File run_boba_review_heldout.ps1           # spike, capped-scale, self-report seeds 12-16
powershell -File run_boba_review_calibration.ps1       # self-report at rank correlation 0.6, 0.3, 0
python scripts/analyse_boba_adaptations.py --arms idea-selfreport-heldout,idea-selfreport-r0.6,idea-selfreport-r0.3,idea-selfreport-r0 --no-extra-trials
python scripts/replay_hitl_remedies.py --input-dir output-boba-spike --output-dir output-boba-spike/analysis/hitl_remedies_heldout --seeds 12,13,14,15,16 --variants sp0.15-20 --error-models spike --stds 0.25 --onsets 0
python scripts/replay_hitl_remedies.py --input-dir output-boba-ceiling --output-dir output-boba-ceiling/analysis/hitl_remedies_heldout --seeds 12,13,14,15,16 --variants ceil0.9-fixed --error-models gaussian --stds 0.25,1 --onsets 0
```

## The archival datasets: noise anchor and fitted-oracle companion

The three datasets are read through `datasets.json`. Since 2026-09-23 it drops
the 40 `opticarvis` rows logged on the raw instrument scales (`column_ranges`)
and enters `provoice`'s Predictability, a lower-is-better construct, with a
minus sign. The pre-fix outputs are kept in `output/pre_fix_2026-09-23/` and
`output-fitted*-prefix/`.

```
python scripts/select_best_oracle_model.py --dataset-config datasets.json --objective composite --output-path output/best_oracle_models_postfix.json
    # merged into output/best_oracle_models.json for opticarvis and provoice; ehmi and the
    # multi-objective entries are unchanged
python scripts/calibrate_noise_from_data.py --dataset-config datasets.json   # output/noise_calibration.csv
python scripts/anchor_noise_scale.py                                          # output/noise_anchor.csv
python scripts/make_per_dataset_configs.py                                    # output/per_dataset/
powershell -File run_fitted_postfix.ps1                                       # output-fitted, -noaug, -adapt-rep10
python scripts/analyse_fitted_companion.py                                    # output-fitted/analysis
```

## Arms run by hand once

```
python scripts/elicitation_compare.py --error-models bias,drift,ceiling,gaussian --magnitudes 1 --iterations 20
```
