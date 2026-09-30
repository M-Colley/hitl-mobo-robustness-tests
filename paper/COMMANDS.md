# Commands behind the paper's numbers

Every table under `paper/tables/` is written by
`python scripts/make_boba_paper_tables.py --analysis output-boba/analysis --out paper/tables`,
except `paper/tables/sitting_by_magnitude.tex`, which `scripts/sitting_by_magnitude.py`
writes (Section 5 below), and the hand-written caption wrapper
`paper/tables/benchmarks_wrapper.tex`; every figure under `paper/figures/` is written by
`python scripts/make_paper_figures.py`.
This file records the commands that produce the analysis outputs those two read,
in the order they must run, including the ones that were once run by hand.
The sections are in run order, and so are the commands within each section.
A section number followed by 'above' or 'below' points into this file; any other
section or appendix number is the paper's.
Seeds are fixed inside each script or given here.

Run everything from the repository root with the Python 3.12 interpreter that
has torch and BoTorch 0.18.2.dev23 (`requirements-eval.txt` lists the
environment); the PowerShell drivers read `$env:PYTHON`. `<scratch>` stands for
any directory outside the repository. A command written as `... > file.txt`
writes a check's standard output, which is its whole report; run it with
`PYTHONIOENCODING=utf-8`, as `scripts/run_review_checks.py` does for the checks
it lists. The run logs of most arms stay on the simulation machine (they are
git-ignored); the analysis outputs the tables and figures read are tracked,
the per-landscape `evaluation/paired_excess_metrics.csv` files of the git-ignored
arms by force-adding them, apart from those arms' recorded run settings
(`run_metadata.json`, which hold local paths). No table reads a `run_metadata.json`.

## 1. Sweeps

Each driver is resumable and writes its own directories. Never rerun an arm into
its old directory after a code change (see `AGENTS.md`), and never resume
`output-boba-instrument` or `output-boba-spike-clip` in place: their clipped or
rounded runs carry the standard file names of the time, and the simulator now
refuses such a directory.

```
python scripts/boba_benchmarks.py --output boba_landscape_stats.json   # the twenty landscapes, their optima and opt_z
powershell -File run_boba_pipeline.ps1                 # output-boba: correctness gate, sweep, evaluation, GP noise diagnostic, synthesis
powershell -File run_boba_followups.ps1                # output-boba-knownnoise, -incumbent, -extensions, their evaluation and synthesis, and the two arm contrasts of Section 6 below
powershell -File run_boba_gaps.ps1                     # output-boba-ladder, -mo, -budget25, -budget100, -confirmatory (seeds 27-36), output-fitted
powershell -File run_boba_gaps_rest.ps1                # the same arms without -mo at 10 workers; resumes what run_boba_gaps.ps1 left
powershell -File run_boba_gaps_mo.ps1                  # output-boba-mo, resumed on its own
powershell -File run_boba_gaps2.ps1                    # output-boba-instrument, -robust, output-fitted-noaug
powershell -File run_boba_inputerror.ps1               # output-boba-slip, -slip-actual, -misclick, then their evaluation
python scripts/analyse_boba_inputerror.py              # output-boba-slip/analysis: the input-error arms
powershell -File run_boba_budget_neutral.ps1 -Arm all -Jobs 20   # q-*, sched-*, session*, spike*, relay*, missing*, ceiling, mo-halo*
powershell -File run_boba_adapt_queue.ps1              # output-boba-adapt-*: replication, re-rating, noisy-input GP
powershell -File run_boba_adapt_studentt.ps1 -Jobs 20  # output-boba-adapt-studentt
powershell -File run_boba_adapt_studentt_rbf.ps1       # output-boba-adapt-studentt-rbf (--likelihood student_t_rbf)
powershell -File run_boba_hitl_ideas.ps1               # output-boba-idea-*
powershell -File run_boba_review_heldout.ps1           # spike, capped-scale, self-report seeds 12-16
powershell -File run_boba_review_calibration.ps1       # self-report at rank correlation 0.6, 0.3, 0
```

The fitted-oracle and oracle-isolation sweeps need the archival inputs first;
they are in Sections 3 and 4 below.

## 2. Main sweep and its syntheses

```
python scripts/diagnose_gp_noise.py --input-dir output-boba --per-cell 3 --at-iterations 8,15,25,50
python scripts/analyse_boba_robustness.py --input-dir output-boba --output-dir output-boba/analysis
    # among its outputs: descriptor_vif.csv (the VIFs of the primary descriptor model's own design,
    # indicators included: the tab:descriptors column), descriptor_vif_all.csv (the seven-descriptor
    # design: log opt_z 9.2, sparsity 5.8, skew 6.3), bias_onset0_control.csv (with landscape-,
    # stream- and two-way clustered SEs) and mediator_model_normalised.csv (the opt_z-free rows on
    # both units; report block 3b prints the frag coefficient beside the magnitude)
python scripts/factor_variance_shares.py               # factor_variance_shares.csv: main-effect shares on both responses (fragility as a fraction of opt_z; excess_sd in landscape SDs), the factor sentence of Section 4
python scripts/rescore_ship_rules.py                   # ship_rules_per_run.csv
python scripts/analyse_ship_rules.py                   # ship_rules_recovery.csv: the cautious rule's 10.1% [4.6, 15.9] at price 0.015 (gaussian, 0.25 sigma and above, ten acquisitions, seeds 7-16)
python scripts/decompose_regret.py --input-dir output-boba
python scripts/onset_bound.py                          # review/onset_bound.{csv,md} and onset_bound_runs.parquet (tracked: the input of review_checks/mo_onset_bound.py)
python scripts/remedy_decomposition.py
python scripts/smooth_subset_and_gp_diagnostics.py
python scripts/bootstrap_coverage.py                   # review/bootstrap_coverage.csv: the selection-share t interval [32.7, 51.0] and the 20,000-resample percentile interval [33.6, 50.2]
python scripts/analyse_pilot_frag.py                   # pilot_frag_summary.csv (tab:pilot: pooled columns over n = 80 landscape-by-magnitude cells)
python scripts/review_checks/frag_residuals.py > output-boba/analysis/review/register_checks/frag_residuals.txt
    # Appendix C.2: the residuals of frag with the indicators (tab:mediation's mediator_only model) against log
    # opt_z and log tail weight over the 640 cells and per landscape, and the leave-one-landscape-out R^2 of that
    # model (0.44) and of the descriptors with the magnitude (0.41), reusing analyse_boba_robustness's cells
python scripts/review_checks/floor_deployed.py > output-boba/analysis/review/register_checks/floor_deployed.txt
    # the floors' absolute deployed regret against each acquisition at onsets 0 and 20: paired landscape-bootstrap
    # differences (2000 resamples), Holm over the ten comparisons, a signed-rank check, and the split of the
    # deployed excess into search and selection (Section 6, 'On the deployed design'); writes review/floor_deployed.csv
python scripts/review_checks/friedman_multiplicity.py > output-boba/analysis/review/register_checks/friedman_multiplicity.txt
    # BH, Holm and Bonferroni over the 32 Friedman p-values of the acquisition ranking (28 raw, 28 BH, 25 Holm; median W 0.247)
python scripts/test_confirmatory_hypotheses.py --screening output-boba --confirmatory output-boba-confirmatory
    # output-boba-confirmatory/analysis/confirmatory_{report.txt,results.json}: the replication planned in advance (void by its own rule)
python scripts/test_confirmatory_hypotheses.py --screening output-boba --confirmatory output-boba-confirmatory --exclude-benchmarks rosenbrock --output-dir output-boba-confirmatory/analysis-sensitivity
    # the post-hoc sensitivity analysis tab:confirmatory prints; both confirmatory commands reproduce their outputs byte for byte
```

## 3. The archival datasets: noise anchor and fitted-oracle companion

The three datasets are read through `datasets.json`. Since 2026-09-23 it drops
the 40 `opticarvis` rows logged on the raw instrument scales (`column_ranges`)
and enters `provoice`'s Predictability, a lower-is-better construct, with a
minus sign. Each remote dataset is pinned by its `data_commit`. The pre-fix
outputs are kept in `output/pre_fix_2026-09-23/` and `output-fitted*-prefix/`.

```
python scripts/select_best_oracle_model.py --dataset-config datasets.json --objective composite --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting --output-path output/best_oracle_models_postfix.json
    # (the 2026-09-23 run also offered tabpfn; extra trees won on all three datasets, which is
    # also the best of the seven families above, so the selected oracles are the same)
python scripts/select_best_oracle_model.py --dataset-config datasets.json --objective composite --oracle-augmentation none --output-path output/best_oracle_models_postfix_noaug.json
    # the held-out R2 without augmentation, compared within the seven families
    # merged into output/best_oracle_models.json for opticarvis and provoice; ehmi is unchanged
python scripts/select_best_oracle_model.py --dataset-config <datasets.json restricted to opticarvis and provoice> --objective multi_objective --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting --output-path output/best_oracle_models_postfix_mo.json
    # the multi-objective entries for the two fixed datasets (extra trees on both, CV R2 -0.16
    # and -0.12), merged the same way; no result in the paper reads them
python scripts/calibrate_noise_from_data.py --dataset-config datasets.json   # output/noise_calibration.csv (ends in a data_commit column)
python scripts/anchor_noise_scale.py                                          # output/noise_anchor.csv (ends in a data_commit column)
python scripts/make_per_dataset_configs.py                                    # output/per_dataset/
powershell -File run_fitted_postfix.ps1                                       # output-fitted, -noaug, -adapt-rep10
python scripts/analyse_fitted_companion.py
    # output-fitted/analysis: fitted_vs_synthetic.csv (the floor gap formed within each run seed's oracle;
    # tab:fitted), fitted_vs_synthetic_best_clean.csv, and fitted_vs_synthetic_achievable{,_best_clean}.csv
    # (the share of the achievable improvement, Appendix B.5)
git show 3d2323aed~1:datasets.json > <scratch>/datasets-prefix.json
python scripts/estimate_archival_error_processes.py --dataset-config <scratch>/datasets-prefix.json --oracle-selection-path output/pre_fix_2026-09-23/best_oracle_models.json --manifest output/pre_fix_2026-09-23/per_dataset/manifest.json --noise-calibration output/pre_fix_2026-09-23/noise_calibration.csv --noise-anchor output/pre_fix_2026-09-23/noise_anchor.csv
    # review/archival_error_processes.{csv,md,json}: the tracked outputs come from the pre-fix inputs;
    # run with its defaults on the current datasets.json, the opticarvis and provoice rows change
```

## 4. Oracle isolation

Every run seed of a fitted-oracle arm refits its oracle, so every normaliser is
formed within each seed and aggregated as a ratio of landscape means. The three
checks at the end of this section must run in the order given:
`oracle_achievable.py` reads `oracle_sigmaf_seeds.csv`, and `oracle_families.py`
reads the `*_achievable*` files.

```
python scripts/oracle_isolation.py make-data
python scripts/oracle_isolation.py select
python scripts/oracle_isolation.py calibrate
powershell -File run_oracle_isolation.ps1                                # output-oracle-iso, tree ensemble
python scripts/oracle_isolation.py select --family gaussian_process      # the isolation datasets against forced smooth
python scripts/oracle_isolation.py calibrate --family gaussian_process   # oracles, no jitter augmentation
powershell -File run_oracle_isolation.ps1 -Family gaussian_process       # output-oracle-iso-gaussian_process
python scripts/oracle_isolation.py select --family mlp
python scripts/oracle_isolation.py calibrate --family mlp
powershell -File run_oracle_isolation.ps1 -Family mlp                    # output-oracle-iso-mlp
for d in output-oracle-iso output-oracle-iso-gaussian_process output-oracle-iso-mlp; do ls $d/runs/iso_*/bo_sensor_error_iso_*_seed*.csv | wc -l; done
    # run counts for tab:arms: 2,400 run files per family (600 clean, 1,800 noisy); the other CSVs there are summaries
python scripts/oracle_isolation.py analyse
    # tree ensemble: the floor gap within each run seed's oracle (primary optimum: the seed's logged y_opt),
    # with the *_best_clean and *_visited summaries: the floor-gap blocks of tab:oracleiso, Appendix B.5
python scripts/oracle_isolation.py analyse --family gaussian_process
python scripts/oracle_isolation.py analyse --family mlp
python scripts/review_checks/oracle_sigmaf_seeds.py --workers 6 > output-boba/analysis/review/register_checks/oracle_sigmaf_seeds.txt
    # sigma_f, fidelity and mean of every run seed's refitted oracle for the three families
    # (output-oracle-iso/oracle_sigmaf_seeds.csv): the seed-7 calibration caveat of Appendix B.5
python scripts/review_checks/oracle_achievable.py > output-boba/analysis/review/register_checks/oracle_achievable.txt
    # the share of the achievable improvement for the three families (tab:oracleiso achievable blocks,
    # Section 4, Appendices B.1 and B.5): *_achievable*.csv under seven estimators, oracle (primary:
    # the seed's y_opt less that oracle's mean over the search box), best_clean, empirical, visited,
    # sigma_f, sigma_f_seed7 and both_best_clean; prints the mean_f check, A over sigma_f*opt_z, the
    # exact best clean over opt_z and the floor-share decomposition
python scripts/run_review_checks.py --only oracle_families
    # register_checks/oracle_families.txt: the three families in both units under the primary, best_clean,
    # sigma_f and both_best_clean normalisers; ratios, medians and the ratio without Branin, Powell and
    # Rosenbrock; counts lower, Wilcoxon p, Spearman(cost) and Spearman(fidelity, ratio); the paired family
    # contrasts, the fidelity counts, the six laws and the fidelity-matched subset; each family's own
    # achievable improvement; and the closing table with every estimator side by side
python scripts/review_checks/oracle_companion_estimators.py > output-boba/analysis/review/register_checks/oracle_companion_estimators.txt
    # the companion arm under the pooled, per-seed y_opt and per-seed best-clean optima, in both units, for
    # the corrected and the as-logged data, with the landscape-and-dataset ratio bootstrap and the
    # known-function per-landscape ranges (Appendix B.5)
python scripts/run_review_checks.py --only c6_ratio,aug_contrast_per_dataset
    # the companion ratio interval and the per-dataset augmentation contrast, which import
    # analyse_fitted_companion's functions (Appendix B.5)
python scripts/oracle_isolation_acq_ranking.py
    # tree ensemble: acquisition ranking on both metrics (oracle_isolation_acq_ranking{,_deployed}.csv) and the
    # exact arm's split-half reliability ceiling (oracle_isolation_acq_reliability.csv): Section 4, Appendix B.5
python scripts/oracle_isolation_acq_ranking.py --iso output-oracle-iso-gaussian_process --out output-oracle-iso-gaussian_process/oracle_isolation_acq_ranking.csv
python scripts/oracle_isolation_acq_ranking.py --iso output-oracle-iso-mlp --out output-oracle-iso-mlp/oracle_isolation_acq_ranking.csv
```

The in-process oracle fits of this pipeline go through
`oracle_isolation.fit_outside_repo`, and CatBoost oracles are built with
`allow_writing_files=False`, so none of these commands writes `catboost_info/`.

## 5. End of study: replays, the final sitting and the budget rule

The shared sitting model (`--sitting-process shared`, the default) is the model
of every sitting number in the paper; the sequential replays are its sensitivity.

```
python scripts/replay_end_of_study.py --workers 5
    # end_of_study/: confirmation-trial false-claim rates and power on seeds 7-16, the standard process's
    # 98% claim rate and 42.7% false claims at 5 sigma (end_of_study_claims.csv, gaussian, 4,800 runs pooled)
python scripts/replay_end_of_study.py --workers 5 --output-dir output-boba/analysis/end_of_study_ksweep --acquisitions logei,qnei --tournament-k 2,3,5,8,12 --confirm-k 2,4,6
python scripts/replay_end_of_study.py --workers 5 --output-dir output-boba/analysis/end_of_study_kwide --acquisitions logei,qnei --tournament-k 16,20,25,30 --confirm-k 2
    # the k-sweep and k-wide replays over the three standard arms (the default --arms); checked by
    # byte-identical spot reruns of published stems
python scripts/replay_end_of_study.py --arms output-boba --output-dir output-boba/analysis/end_of_study_ksweep_seq --tournament-k 2,3,5,8,12 --confirm-k 2,4,6 --rho 1,0.5 --sitting-process sequential --workers 12 --resume
python scripts/replay_end_of_study.py --arms output-boba --output-dir output-boba/analysis/end_of_study_kwide_seq --tournament-k 16,20,25,30 --confirm-k 2 --rho 1,0.5 --sitting-process sequential --workers 12 --resume
    # the sitting as a sequence of trials: the drift ramp and the AR(1) state continue through it, candidates in random order
python scripts/replay_end_of_study.py --arms output-boba --output-dir output-boba/analysis/end_of_study_ksweep_seqrank --error-models drift,ar1 --tournament-k 2,3,5,8,12 --confirm-k 2,4,6 --rho 1,0.5 --sitting-process sequential --sitting-order rank --workers 12 --resume
python scripts/replay_end_of_study.py --arms output-boba --output-dir output-boba/analysis/end_of_study_kwide_seqrank --error-models drift,ar1 --tournament-k 16,20,25,30 --confirm-k 2 --rho 1,0.5 --sitting-process sequential --sitting-order rank --workers 12 --resume
    # the same with the candidates shown best-ranked first (drift and ar1 only; gaussian and bias are unchanged by construction)
for sd in 0 0.1 0.25 0.5; do python scripts/replay_end_of_study.py --arms output-boba-slip,output-boba-misclick --tournament-k 5 --confirm-k "" --rho 1 --slip-look-sd $sd --rerate-dirs none --output-dir output-boba/analysis/end_of_study_slipsd$sd --workers 12 --resume; done
    # the input arms' sitting over the assumed look SD (the 0.25 run reproduces end_of_study)
python scripts/replay_end_of_study.py --arms output-boba-confirmatory --output-dir output-boba-confirmatory/analysis/end_of_study_fresh --acquisitions logei,qnei,ucb --seeds 27,28,29,30,31,32,33,34,35,36 --error-models gaussian --stds 0.05,0.25,1.0,5.0 --onsets 0,20 --tournament-k 2,5,12 --confirm-k 2 --rho 1 --rerate-dirs none
python scripts/rescore_ship_rules.py --input-dir output-boba-confirmatory --output-dir output-boba-confirmatory/analysis/ship_rules_fresh --acquisitions logei,ei,qei,pi,logpi,qpi,ucb,qucb,qnei,greedy --seeds 27,28,29,30,31,32,33,34,35,36 --error-models gaussian
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25,30
    # the derived budget rule as published, rho = 1 (the paper's sitting): budget_split_derived.csv and
    # budget_split_policies.csv (reproduced byte for byte), budget_split_by_onset.csv (all seeds and the
    # halves 7-11 and 12-16); the worker count changes no output. The defaults (one replay directory, grid
    # 2..12, so n0 = 38) are not the paper's study.
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25,30 --rho 0.5 --output-suffix _rho0.5
    # the rho = 0.5 sensitivity: budget_split_derived_rho0.5.csv and budget_split_policies_rho0.5.csv (byte
    # for byte; heldout_remedies.py reads them) and budget_split_by_onset_rho0.5.csv
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25,30 --decision sequential --output-suffix _decision_sequential
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25,30 --decision sequential --rho 0.5 --output-suffix _decision_sequential_rho0.5
    # the sequential rule (a decision at each T - k), built after the by-onset result: outside the Holm families
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25 --output-suffix _kmax25
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25 --rho 0.5 --output-suffix _kmax25_rho0.5
    # the capped grid, one decision at n0 = 25 (a diagnostic: it needs the onset to be known)
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25 --rate-source truth --output-suffix _kmax25_truerate
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25,30 --decision sequential --rate-source truth --output-suffix _decision_sequential_truerate
    # the clairvoyant search-credit diagnostics (they read the true objective; never a rule)
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25,30 --twin clean --output-suffix _clean
python scripts/budget_split.py --workers 5 --ksweep-dir output-boba/analysis/end_of_study_ksweep,output-boba/analysis/end_of_study_kwide --k-grid 2,3,5,8,12,16,20,25,30 --decision sequential --twin clean --output-suffix _decision_sequential_clean
    # the prices without error (trt_clean - ref_clean): each rule derived on the clean twins, beside every fixed k
    # and the no-trial rules: budget_split_{derived,price}_clean.csv and budget_split_{derived,price}_decision_sequential_clean.csv
python scripts/review_checks/budget_split_by_onset.py
    # register_checks/budget_split_by_onset.txt: the derived rule by onset and magnitude, without the 5 sigma
    # cell from trial 21, where it decides and whether it is blind, landscape groups, fit correlations, the
    # search-term inflation from the logs, the posterior-mean-to-LCB and clairvoyant counterfactuals, the
    # clean-twin prices, and a reproduction of heldout_remedies. Its 'gains at 0.05 sigma' block is not a price.
python scripts/replay_hitl_remedies.py --input-dir output-boba --workers 8
    # hitl_remedies/: the best-arm (LUCB) sitting against one look each to the top eight at half the
    # idiosyncratic SD, seeds 7-16: 16.1% [10.8, 20.6] at price 0.017 against 18.7% [13.1, 23.6] at 0.015
    # (hitl_remedies_recovery.csv, error_model pooled, procedures lucb_k8_rho0.5 and tournament_k8_lcb_rho0.5_look)
python scripts/heldout_remedies.py
    # review/heldout_remedies.{csv,md}: held-out scopes, landscapes gaining and Holm p (section 4_holm: columns
    # scope, landscapes_gaining_test, p_holm_core_test, p_holm_all_test), for Section 7 and the Multiplicity
    # paragraph of Appendix E.3: the cautious rule's held-out 10.7% [5, 16] on seeds 12-16, the five rho-0.5
    # sitting survivors, fixed k = 5 (0.25), k = 8 (13 of 20, 0.27), k = 30 (0.029), and the best-arm sitting of
    # eight looks at half the idiosyncratic SD (17 of 20, adjusted p = 0.008). Name the Holm family with every p.
    # Section 4b_holm_rank_scope: the rank rules tested on the gross-fault and capped arms' seeds 12-16, the scope
    # of their recoveries (ordinal_lcb1 48.8% [35, 63], 19 of 20, p 5.7e-6; 37.9% [31, 45], 20 of 20, p 1.9e-6;
    # Holm 1.4e-4 and 5.1e-5 over the 27 with the two main-sweep rank tests replaced, column
    # p_holm_core_lcb1_replaced_test). Section 4c_holm_lucb_comparators: the four one-look sittings the best-arm
    # sittings compete with, added to the all family (45 tests; the top-eight one at half the SD gains on 18 of
    # 20, adjusted p 0.0086; lucb_k8_rho0.5 0.0086; all nine all-family survivors survive). Appendix E.3.
python scripts/sitting_by_magnitude.py
    # the sitting cell by cell against the standard process and the zero-trial ship rules on the same runs,
    # four processes pooled: PM, LCB1 and LCB2 after 50 trials; the sitting's paired increments over LCB1, LCB2
    # and PM, overall and by process, with Holm over the nine k, over the eight chosen cells and over all 72; the
    # zero-trial rule chosen on seeds 7-11 and the increment over it; the looks-alone increment; the clean-twin
    # prices. Writes review/sitting_by_magnitude{.csv,_ship_rules.csv,_selection.json} and
    # paper/tables/sitting_by_magnitude.tex (Section 7, Appendix E.3, tab:sittingcell, tab:remedies)
python scripts/sitting_by_magnitude.py --processes gaussian --tag gaussian
    # the same under gaussian error alone: review/sitting_by_magnitude_gaussian{.csv,_ship_rules.csv,_selection.json}, no table
python scripts/sitting_by_magnitude.py --replay-dirs end_of_study_ksweep_seq,end_of_study_kwide_seq --tag seq
    # the sequential replays, random order: review/sitting_by_magnitude_seq{.csv,_ship_rules.csv,_selection.json}
python scripts/sitting_by_magnitude.py --tag seqrank --replay-dirs end_of_study_ksweep_seqrank=drift+ar1,end_of_study_kwide_seqrank=drift+ar1,end_of_study_ksweep=gaussian+bias,end_of_study_kwide=gaussian+bias
    # rank order: drift and AR(1) from the rank-order replays, gaussian and bias from the shared ones
python scripts/run_review_checks.py --only fresh_seed_replication
    # the untouched-seed test (seeds 27-36, gaussian): the zero-trial ship rules; the sitting's increment over
    # LCB1 per cell and k, with Holm over the 3 k and the 24 (cell, k); the looks-alone increment; the clean-twin
    # prices; and, in a last block, the increments over LCB2 and PM with the same Holm families
python scripts/review_checks/sitting_vs_shiprule.py
    # register_checks/sitting_vs_shiprule.txt: the sitting against the zero-trial ship rules side by side (four
    # processes and gaussian; seeds 7-11, 12-16, 7-16 and 27-36; the quoted cells; the pooled k-curve; the Holm
    # families; the prices; the look model; the increments over LCB2 and PM with the rule chosen on seeds 7-11).
    # It collates the outputs above, so it runs after them.
python scripts/review_checks/sitting_sequential.py > output-boba/analysis/review/register_checks/sitting_sequential.txt
    # the shared against the sequential sitting: every pooled gain of the fixed-allocation sitting the paper
    # prints, re-scored with sitting_by_magnitude.run and heldout_remedies' k-curve scoring (the chosen k per
    # cell, the derived rule, the tab:heldout k row with its landscape-split column, the optimism of the choice
    # of k, the 41-procedure Holm family), plus the slip-SD and spike summaries. Its shared side must reproduce
    # the published outputs first, and it exits non-zero otherwise.
```

## 6. Process changes, faulty raters and ties

`analyse_boba_adaptations.py` scores every arm on an explicit scale: `opt_z` for
the landscapes, the hypervolume floor gap for the halo problems, and the fitted
achievable improvement for the archival datasets. It needs the `output-boba-mo`
evaluation and the `output-fitted` run logs.

```
python scripts/analyse_boba_mo.py
    # output-boba-mo/analysis: the headroom screen at 0.15 (mo_headroom.csv: DH2 0.0, Penicillin 0.197), the
    # dose-response, the acquisition agreement and the cross-arm floor-gap cells (mo_vs_scalar.csv); all six
    # outputs reproduce byte for byte
python scripts/analyse_boba_adaptations.py
    # adaptations_recovery.csv, adaptations_per_dataset.csv (per-dataset cost, gain, price and recovery) and
    # adaptations_extra_runs.csv: tab:adapt, tab:budgetneutral, tab:remedies, Appendices E.4 and E.5. The
    # response-clipping rows (spike-clip-*) have price zero by construction.
python scripts/analyse_boba_adaptations.py --arms idea-selfreport-heldout,idea-selfreport-r0.6,idea-selfreport-r0.3,idea-selfreport-r0 --no-extra-trials
python scripts/analyse_boba_adaptations.py --arms studentt-rbf
python scripts/compare_boba_arms.py --reference output-boba --treatment output-boba-knownnoise --label-reference "learned noise" --label-treatment "known noise" --output-dir output-boba-knownnoise/analysis
python scripts/compare_boba_arms.py --reference output-boba --treatment output-boba-incumbent --label-reference "posterior-mean incumbent" --label-treatment "observed-max incumbent" --output-dir output-boba-incumbent/analysis
    # each arm over its own clean twin, on the post-onset trajectory loss; shares are suppressed where the
    # pooled reference is below 0.01 of the achievable improvement or negative on any landscape
    # (arm_contrast_{pairs,summary,by_benchmark,by_acquisition}.csv, arm_contrast_report.txt,
    # arm_contrast_summary.json: tab:knownnoise, tab:incumbent, tab:incumbentacq, Appendices B.1, D.1 and D.2).
    # The same two commands are in run_boba_followups.ps1.
python scripts/replay_hitl_remedies.py --input-dir output-boba-spike --output-dir output-boba-spike/analysis/hitl_remedies --seeds 7,8,9,10,11 --variants sp0.15-20 --error-models spike --stds 0.25 --onsets 0 --sitting-k 8 --workers 12
python scripts/replay_hitl_remedies.py --input-dir output-boba-spike --output-dir output-boba-spike/analysis/hitl_remedies_heldout --seeds 12,13,14,15,16 --variants sp0.15-20 --error-models spike --stds 0.25 --onsets 0
    # the gross-fault arm's remedies (rank refit 51% [36, 63] on seeds 7-11, 49% [34, 63] held out). Both
    # directories were rewritten in place by these two commands after the fix of the spike look SD (0.268
    # against 7.75): the look-buying rows (tournament_*, lucb_*) now equal those of *_spikesd below, and the
    # rank-rule and shortlist rows did not change. No paper number uses the look-buying rows. The seeds 7-11
    # command is the corrected twin of the held-out one; the original seeds 7-11 run was not logged.
python scripts/replay_hitl_remedies.py --input-dir output-boba-spike --output-dir output-boba-spike/analysis/hitl_remedies_spikesd --seeds 7,8,9,10,11 --variants sp0.15-20 --error-models spike --stds 0.25 --onsets 0 --sitting-k 8 --workers 12
python scripts/replay_hitl_remedies.py --input-dir output-boba-spike --output-dir output-boba-spike/analysis/hitl_remedies_spikesd_heldout --seeds 12,13,14,15,16 --variants sp0.15-20 --error-models spike --stds 0.25 --onsets 0 --workers 12
    # the same replays at the run's own spike SD, reproduced exactly: the corrected look-buying rows
python scripts/replay_end_of_study.py --arms output-boba-spike --variants sp0.15-20 --error-models spike --acquisitions logei,qnei --seeds 7,8,9,10,11 --tournament-k 8 --confirm-k "" --rho 1,0.5 --rerate-dirs none --output-dir output-boba-spike/analysis/end_of_study_spikesd --workers 12 --resume
python scripts/replay_end_of_study.py --arms output-boba-spike --variants sp0.15-20 --error-models spike --acquisitions logei,qnei --seeds 12,13,14,15,16 --tournament-k 4,8 --confirm-k "" --rho 1,0.5 --rerate-dirs none --output-dir output-boba-spike/analysis/end_of_study_spikesd_heldout --workers 12 --resume
    # the gross-fault arm's fixed-allocation sitting with a spike drawn per look (not quoted in the paper)
python scripts/replay_hitl_remedies.py --input-dir output-boba-ceiling --output-dir output-boba-ceiling/analysis/hitl_remedies --seeds 7,8,9,10,11 --variants ceil0.9-fixed --error-models gaussian --stds 0.25,1 --onsets 0 --sitting-k 8 --workers 6
    # the capped scale's remedies on seeds 7-11 (rank refit 36% [29, 43] under the first-index rule); this
    # command reproduces hitl_remedies_recovery.csv byte for byte
python scripts/replay_hitl_remedies.py --input-dir output-boba-ceiling --output-dir output-boba-ceiling/analysis/hitl_remedies_heldout --seeds 12,13,14,15,16 --variants ceil0.9-fixed --error-models gaussian --stds 0.25,1 --onsets 0
    # the capped scale's held-out remedies; its looks are replayed without the cap, so only the rows that buy no look are quoted
python scripts/replay_stopping.py --input-dir output-boba --acquisitions logei,qnei --error-models gaussian,bias,drift,ar1 --tune-seeds 7,8,9,10,11 --score-seeds 12,13,14,15,16 --rules A,B --n0 15 --h-grid 2,5,10,20,50,100,200,500,1000,2000,5000,10000 --kappa-grid 0,1,4,16 --w-grid 0,1,2,3 --max-false-alarm 0.1 --false-alarm-scope pooled --tune-objective regret --checkpoints 20,30,40 --deltas 0.25,0.5 --delta-units model,landscape --floor-repeats 3 --workers 5
    # output-boba/analysis/stopping/: detect-and-freeze (rule A, tuned on seeds 7-11: h 2000, kappa 16, w 3 in
    # stopping_tuned_A.json) and decision-stable stopping (rule B), scored on 12-16; the options are those of
    # stopping_metadata.json. About 8,000 s from scratch on 5 workers; with the cache_n015_rep3 fits about 25 s,
    # and it then reproduces stopping_per_run.csv, stopping_grid_A.csv and stopping_tuned_A.json byte for byte
python scripts/changepoint_compare.py
    # output-boba/analysis/changepoint/: CUSUM, GLR and BOCPD on the freeze rule's residual cache at matched
    # false-alarm budgets (changepoint_detectors.csv), and the freeze rule's own operating point with every
    # detector re-tuned to its rate and a check against rule A's logged alarms (changepoint_freeze_point.csv,
    # changepoint_metadata.json). Runs after replay_stopping.py; about 6 s
python scripts/design_rules_from_pilot.py --ceiling-seeds 7,8,9,10,11
    # design_rule_{instrument,instrument_screen,sizing}.csv, reproduced byte for byte: the instrument screen
    # (Spearman 0.65 over the 40 landscape-by-magnitude cells at 0.25 and 1 sigma; half split 0.153 / 0.316;
    # medians 0.219 / 0.104). Without --ceiling-seeds it reads the capped arm's seeds 12-16 as well and gives
    # other numbers. Its uncapped side is the main sweep's cell means over ten acquisitions and all seeds.
python scripts/review_checks/tie_break.py scan --workers 6
    # every per-run log of output-boba*, output-fitted* and output-oracle-iso*: per run, the first-index and the
    # uniform-tie (exact expectation) deployed regret of the noisy run and its clean twin, and the share of runs
    # with a tied maximum rating per arm, variant and condition. About 15 minutes on 6 workers. Writes
    # review/tie_break/tie_runs.parquet (git-ignored, 13 MB) and tie_share_by_{arm,condition}.csv. Rerun it
    # after any new run logs: rescore reads only its parquet.
python scripts/review_checks/tie_break.py rescore
    # every quoted deployed-design number of an arm with ties under both conventions, with the estimand
    # unchanged (arm contrasts through analyse_boba_adaptations' load, paired_frame and summarise; replayed
    # remedies through replay_hitl_remedies.recovery_table; the rank rule's landscape splits through
    # heldout_remedies; the instrument screen through design_rules_from_pilot on capped seeds 7-11 and 7-16, and
    # a paired version; the cautious rule's LCB2 and LCB1 cells through analyse_ship_rules), plus the main
    # sweep's tie origin and the rank refit's price. Writes review/tie_break/{arm_contrasts_deployed,
    # arm_excess_deployed,replayed_remedies_deployed,rank_rule_landscape_splits,instrument_screen,
    # cautious_rule_cells,quoted_numbers,main_sweep_tie_origin,tie_share_by_arm,tie_share_by_condition}.csv and
    # register_checks/tie_break.txt, which equals its standard output. About 1 minute. Among its numbers: the
    # capped scale's extra cost 185% [132, 258] (first index) against 135% [92, 196] (uniform), and the cautious
    # rule's 7.7% [2.7, 12.8] at 1 sigma over both onsets.
```

## 7. Checks behind individual sentences

Each check behind an individual sentence is a script under
`scripts/review_checks/`: the currency-test curvature and spline checks, the
manipulated-landscape intervals, the structural extra-trial zeros, the
like-for-like Kendall's W, the multi-objective front, the nugget-radius
sensitivity, the ehmi sigma_f on two surfaces, the companion ratio interval and
its estimators, the augmentation contrast per dataset, the matched opt_z, the
oracle-isolation checks of Section 4 above, the anchor interval propagated, the
untouched-seed test, the sitting against the zero-trial ship rules and under the
sequential rating process, the budget rule by onset, the floors' deployed
regret, the Friedman multiplicity, the archival instruments' scales, the
multi-objective onset bound, the extra-trial reach, the one-shot predictor's
residuals and out-of-sample fit, the comparison loop's determinism and the
tie-break sensitivity. `python scripts/run_review_checks.py` runs them all in dependency
order and writes each one's standard output to
`output-boba/analysis/review/register_checks/<check>.txt`; `--only a,b` runs a
subset, and the direct invocations given elsewhere in this file are equivalent.
Its inputs must exist first: the analyses of Sections 2 to 6 above, the
tie-break scan, `onset_bound_runs.parquet` and, for the checks the docstring of
`run_review_checks.py` lists, the git-ignored run logs and archival clones.

```
python scripts/run_review_checks.py        # every check in its CHECKS list, in that order
python scripts/run_review_checks.py --only oracle_families,anchor_propagation,fresh_seed_replication,matched_optz
    # the three-family isolation table, the anchor interval propagated (review_checks/anchor_propagation.py), the
    # untouched-seed test (review_checks/fresh_seed_replication.py) and the headline on matched opt_z
python scripts/run_review_checks.py --only ehmi_sigmaf,nugget_threshold
    # the ehmi sigma_f on two surfaces (review_checks/ehmi_sigmaf.py) and the nugget-radius sensitivity; both take
    # datasets.json and read the archival data clones in .dataset_cache/
python scripts/run_review_checks.py --only mo_front_check
    # the reported multi-objective front: about half truly dominated at 1 sigma, and a shortfall above the evaluated set's in all eight cells
python scripts/run_review_checks.py --only manipulation_intervals
    # wide- and narrow-spike manipulation contrasts per magnitude, trajectory metric, five seeds (Appendices B.2 and D.3)
python scripts/run_review_checks.py --only matched_optz
    # the clean twin's deployed regret, 0.1284 of the achievable improvement at 1 sigma from the first rating (Discussion)
python scripts/review_checks/instrument_scale.py > output-boba/analysis/review/register_checks/instrument_scale.txt
    # each archival instrument's composite step, coarsest-item step, span and ceiling in its study's sigma_f
    # (review/instrument_scale.csv), the ratings at each item's logged ends, and where the instrument arm's
    # +-8 sigma clip meets the twenty landscapes (read from output-boba-instrument/run_metadata.json)
python scripts/review_checks/mo_onset_bound.py --workers 6 > output-boba/analysis/review/register_checks/mo_onset_bound.txt
    # the running-maximum onset bound for the multi-objective arm, with the scalar arm on the same floor-gap
    # basis (review/mo_onset_bound.csv), and the split of the cross-arm log gap in raw onset ratio into bound and
    # normalised parts; needs onset_bound_runs.parquet from onset_bound.py and mo_vs_scalar.csv from analyse_boba_mo.py
python scripts/analyse_extra_runs.py --input-dir output-boba-budget100 --k 25,50 --tolerance 0,0.01
    # output-boba-budget100/analysis/extra_runs.csv and extra_runs_per_run.csv (tab:extra); the extra trials are
    # counted from clean_origin, the trial at which the clean twin first reached its own k-trial regret. The per-run
    # table also carries clean_origin, reach_trial and reach_over_k, and extra_runs.csv median_clean_origin,
    # median_reach_trial and median_reach_over_k: Appendix B.7's reach trial 56 and 2.2 (2.24) times k at k = 25,
    # gaussian 1 sigma from the first rating, tolerance 0.01. 'multiplier' is (k + extra) / k, not the reach over k
python scripts/review_checks/extra_reach.py > output-boba/analysis/review/register_checks/extra_reach.txt
    # Appendix B.7: the noisy run's reach trial (origin + extra; median 56 at k = 25, 1 sigma from the first
    # rating, tolerance 0.01) and its ratio to k (2.2), recomputed with analyse_extra_runs' own functions
```

## 8. The comparison loop (Appendix app:elicitation)

The log in `output-elicitation/` came from a code state the repository does not
hold: the committed script does not regenerate it (39 of 40 clean seed-7 runs
ship another design), and the command once recorded here ran every landscape of
the stats file (29), where that log has 20. The appendix reads a rerun of the
current script on the paper's twenty landscapes (960 runs, seeds 7-9), into a
fresh directory:

```
python scripts/elicitation_compare.py --functions ackley,branin,eggholder,griewank,hartmann_3,hartmann_6,hicks_law,levy_10,michalewicz,moving_peaks,powell,power_law_practice,rastrigin,rosenbrock,schwefel,shekel,steering_law,stevens,weber_fechner,yerkes_dodson --error-models bias,drift,ceiling,gaussian --magnitudes 1 --iterations 20 --workers 12 --output-dir output-elicitation-rerun
python scripts/elicitation_compare.py --summary-only --output-dir output-elicitation-rerun
    # elicitation_runs.csv and elicitation_summary.csv: the own-twin share, the standard-process estimand
    # (cost, gain and price against the rating loop), and both tie conventions for the rating loop under the cap
    # (Appendix app:elicitation, tab:arms)
python scripts/review_checks/elicitation_determinism.py > output-boba/analysis/review/register_checks/elicitation_determinism.txt
    # the determinism check of the reproducibility statement: the same command on branin, stevens and weber_fechner,
    # seeds 8 and 9 (96 of the 960 runs) with six workers, into a temporary directory; every column but seconds
    # equals output-elicitation-rerun's rows (merged on dataset, elicitation, error_model, magnitude, seed,
    # apply_error): verdict exact. About two minutes
```

The twenty names are `sorted(boba_benchmarks.DEFAULT_SUITE)`, written out so that the
record names its landscapes itself; `check_provenance.py` reads the command from this
line and resolves `--functions` (a list, `suite` or `all`) as the script does. Each run seeds numpy's and torch's
global generators from its own seed: BoTorch's `PairwiseGP` perturbs every Laplace start
with numpy's global generator, so without that seeding no two executions of the
comparison loop agree. The provenance check of Section 9 below reads this record:
`check_provenance.ELICITATION_ARM` is `output-elicitation-rerun`, whose 40 clean
seed-7 runs it reruns exactly, and the old log in `output-elicitation/` is checked
separately against the same command.

## 9. Provenance and environment

The recorded `git_commit` of an arm run before 2026-09-30 does not identify its
code. The evidence is a rerun of one logged run per arm, compared with the log.
The reports record the code of the reruns (commit plus the sha256 of the
uncommitted diff, in the `.stamp.json`); rerun them once the tree is committed.

```
python scripts/check_provenance.py --out-dir <scratch>/prov --workers 6 --dataset-config-override "output-fitted*-prefix/*=git:3d2323aed~1" --report output-boba/analysis/review/provenance_check.csv
    # one logged run of each arm (output-boba*, output-fitted*/<dataset>, output-oracle-iso*/runs/iso_branin) and the
    # first cell of each comparison-loop log (output-elicitation-rerun exact, the old output-elicitation log differs),
    # rerun and compared, 73 checks; the as-logged fitted arms are retried with datasets.json at 3d2323aed~1
python scripts/check_provenance.py --out-dir <scratch>/prov_random --workers 6 --prefer-acq random,sobol --arms "output-boba,output-boba-confirmatory,output-boba-extensions,output-boba-ladder,output-boba-misclick,output-boba-mo,output-boba-mo-halo,output-boba-mo-halo-backfit,output-boba-slip,output-boba-slip-actual" --report output-boba/analysis/review/provenance_check_random.csv
    # the model-free floors (random first) of the ten arms listed
python scripts/check_provenance.py --out-dir <scratch>/prov_qehvi_sobol --workers 6 --prefer-acq qehvi,sobol --arms "output-boba,output-boba-extensions,output-boba-ladder,output-boba-mo" --report output-boba/analysis/review/provenance_check_qehvi_sobol.csv
    # the sobol floors and the qehvi run of the multi-objective arm
python scripts/check_provenance.py --out-dir <scratch>/prov_elicitation_rerun --workers 1 --arms output-elicitation-rerun --elicitation-scope clean --report output-boba/analysis/review/provenance_check_elicitation_rerun.csv
    # the comparison loop's clean seed-7 runs on its 20 landscapes (40 runs) against the rerun of Section 8 above,
    # output-elicitation-rerun/elicitation_runs.csv: exact, 40 of 40, with one worker where the log used twelve
python scripts/check_provenance.py --out-dir <scratch>/prov_elicitation --workers 1 --arms output-elicitation --elicitation-scope clean --report output-boba/analysis/review/provenance_check_elicitation.csv
    # the same 40 runs against the old log, output-elicitation/elicitation_runs.csv (39 of 40 differ: the log of
    # Section 8 above, from a code state the repository does not hold)
python -m pytest tests/test_regression_logged_run.py -q -p no:cacheprovider
    # the current simulator reproduces a logged main-sweep run (branin, LogEI, seed 7, gaussian 1 sigma from
    # trial 21) and its clean twin exactly; skips where output-boba is absent
python -m pytest tests/test_tie_break.py tests/test_sitting_by_magnitude.py tests/test_replay_sitting_process.py -q -p no:cacheprovider
```

Of the 71 BO runs checked (one per output directory and archival dataset, and
`iso_branin` for each oracle-isolation arm), 64 reproduce exactly from their
record; 7 need a setting the record does not hold: the acquisition order used
before `qkg` and `replei` were inserted (one multi-objective run; the floors of
`output-boba`, `-extensions`, `-ladder` and all of `-mo` ran under it, and
`check_provenance.ACQUISITION_ORDER_BEFORE_ROBUST` reconstructs it), or the
dataset configuration of the time for the as-logged fitted arms, whose record
holds the objective map but not the column ranges. The comparison loop's old log
does not reproduce; its rerun does (Section 8 above). The smooth-family isolation arms recorded
`git_commit` 915d5e069 with the then uncommitted `_SmoothOracle` (committed in
bce12bdaf); their `iso_branin` reruns with the current code are exact.

The acquisition settings of Section 3.3 (xi, kappa, candidate pool, restarts,
L-BFGS-B iterations, QMC samples, the knowledge gradient's fantasies and raw
samples) are the `args` recorded in each arm's `run_metadata.json`, as the
drivers of Section 1 above set them. The BoTorch build (0.18.2.dev23, commit
eea35f51a) and the other package versions are recorded there as well, under
`package_versions`.

## 10. Tables, figures and the paper

```
python scripts/make_boba_paper_tables.py --analysis output-boba/analysis --out paper/tables
    # stops with exit 1, naming the table and the file, when any input of a table main.tex \inputs is missing;
    # --allow-missing keeps the old table instead, for a checkout without the git-ignored arms
python scripts/make_paper_figures.py                   # kcurve.pdf draws the pooled curve and the 1 sigma, first-rating curve from review/sitting_by_magnitude*
python scripts/check_paper.py --paper paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && pdflatex main
python paper/build_overleaf_bundle.py                  # compiles in a clean directory, repeating pdflatex until main.log asks for no rerun, and prints where the main text ends
```
