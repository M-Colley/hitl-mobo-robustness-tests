# Commands behind the paper's numbers

Every table under `paper/tables/` is written by
`python scripts/make_boba_paper_tables.py --analysis output-boba/analysis --out paper/tables`,
except `paper/tables/sitting_by_magnitude.tex`, which `scripts/sitting_by_magnitude.py`
writes (Section 5 below), `known_variance.tex` (`scripts/checks/known_variance_ship.py`,
Section 5), `decision_map.tex` (`scripts/checks/decision_map.py`, Section 6),
`winners_curse.tex` and `winners_curse_landscapes.tex` (`scripts/checks/winners_curse.py
report`, Section 2), and the hand-written caption wrapper
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
`PYTHONIOENCODING=utf-8`, as `scripts/run_checks.py` does for the checks
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
    # also ceiling_screen.csv (clean runs at the optimum at trial 21 and at T, and the share with no post-onset
    # gain) and headroom_screen.csv's low_headroom column (formerly the inverted ceiling_free; not a ceiling test).
    # Among its other outputs: descriptor_vif.csv (the VIFs of the primary descriptor model's own design,
    # indicators included: the tab:descriptors column), descriptor_vif_all.csv (the seven-descriptor
    # design: log opt_z 9.2, sparsity 5.8, skew 6.3), bias_onset0_control.csv (with landscape-,
    # stream- and two-way clustered SEs) and mediator_model_normalised.csv (the opt_z-free rows on
    # both units; report block 3b prints the frag coefficient beside the magnitude)
python scripts/factor_variance_shares.py               # factor_variance_shares.csv: main-effect shares on both responses (fragility as a fraction of opt_z; excess_sd in landscape SDs), the factor sentence of Section 4
python scripts/rescore_ship_rules.py                   # ship_rules_per_run.csv
python scripts/analyse_ship_rules.py                   # ship_rules_recovery.csv: the cautious rule's 10.1% [4.6, 15.9] at price 0.015 (gaussian, 0.25 sigma and above, ten acquisitions, seeds 7-16)
    # recovered_suppressed and n_cost_negative flag shares whose cost is near zero or negative on a landscape; the
    # BH family is arm x rule x error process across that process's variants (fdr_family_size; until 2026-10-09
    # bias was corrected over two cells only)
python scripts/decompose_regret.py --input-dir output-boba
    # excess_selection_share_suppressed: no share at 0.05 sigma or at 0.25 sigma from trial 21 (tab:dose prints --);
    # a missing opt_z raises
python scripts/onset_bound.py                          # review/onset_bound.{csv,md} and onset_bound_runs.parquet (tracked: the input of checks/mo_onset_bound.py)
python scripts/remedy_decomposition.py
python scripts/smooth_subset_and_gp_diagnostics.py --reuse-gp
    # re-aggregates review/gp_diagnostics_per_run.csv; the three convex or unimodal landscapes get a range, not an
    # interval (tab:smooth); without --reuse-gp it refits the 800 GPs first
python scripts/bootstrap_coverage.py                   # review/bootstrap_coverage.csv: the selection-share t interval [32.7, 51.0] and the 20,000-resample percentile interval [33.6, 50.2]
python scripts/analyse_pilot_frag.py                   # pilot_frag_summary.csv (tab:pilot: pooled columns over n = 80 landscape-by-magnitude cells)
python scripts/checks/frag_residuals.py > output-boba/analysis/review/register_checks/frag_residuals.txt
    # Appendix C.2: the residuals of frag with the indicators (tab:mediation's mediator_only model) against log
    # opt_z and log tail weight over the 640 cells and per landscape, and the leave-one-landscape-out R^2 of that
    # model (0.44) and of the descriptors with the magnitude (0.41), reusing analyse_boba_robustness's cells
python scripts/checks/floor_deployed.py > output-boba/analysis/review/register_checks/floor_deployed.txt
    # the floors' absolute deployed regret against each acquisition at onsets 0 and 20: paired landscape-bootstrap
    # differences (100,000 resamples), a t test and a signed-rank test, each with Holm over the ten comparisons
    # (at 5 sigma only LogPI, by the bootstrap alone, adjusted p 0.03; at 1 sigma six acquisitions under all three),
    # and the split of the deployed excess into search and selection (Section 6); writes review/floor_deployed.csv
python scripts/checks/ceiling_clean_runs.py > output-boba/analysis/review/register_checks/ceiling_clean_runs.txt
    # after onset_bound.py and analyse_boba_robustness.py: the clean runs at the optimum per landscape (four laws
    # ceiling-limited from trial 21) and the late-onset numbers with and without them (review/ceiling_clean_runs.csv,
    # ceiling_sensitivity.csv): the controls of Section 3.5 and Appendix A.1
python scripts/checks/winners_curse.py scan --workers 4
    # reads the git-ignored run logs of output-boba, -confirmatory, -budget25 and -budget100 (about 5 minutes) and
    # writes review/winners_curse_runs.parquet (git-ignored); rerun it after any change to apply_sensor_error
python scripts/checks/winners_curse.py report
    # selection loss as a winner's curse on the visited designs (Section 5, app:winners): stops unless the error
    # streams replay exactly and regret_decomposition.csv and per_landscape_headline.csv reproduce; writes
    # review/winners_curse_{cells,landscapes,onset,budget,budget_contrast,checks}.csv,
    # register_checks/winners_curse.txt and paper/tables/winners_curse{,_landscapes}.tex (tab:winners,
    # tab:winnerslandscapes); make_paper_figures.py draws figures/winners_curse.pdf from winners_curse_landscapes.csv
python scripts/checks/friedman_multiplicity.py > output-boba/analysis/review/register_checks/friedman_multiplicity.txt
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
python scripts/checks/oracle_sigmaf_seeds.py --workers 6 > output-boba/analysis/review/register_checks/oracle_sigmaf_seeds.txt
    # sigma_f, fidelity and mean of every run seed's refitted oracle for the three families
    # (output-oracle-iso/oracle_sigmaf_seeds.csv): the seed-7 calibration caveat of Appendix B.5
python scripts/checks/oracle_achievable.py > output-boba/analysis/review/register_checks/oracle_achievable.txt
    # the share of the achievable improvement for the three families (tab:oracleiso achievable blocks,
    # Section 4, Appendices B.1 and B.5): *_achievable*.csv under seven estimators, oracle (primary:
    # the seed's y_opt less that oracle's mean over the search box), best_clean, empirical, visited,
    # sigma_f, sigma_f_seed7 and both_best_clean; prints the mean_f check, A over sigma_f*opt_z, the
    # exact best clean over opt_z and the floor-share decomposition
python scripts/run_checks.py --only oracle_families
    # register_checks/oracle_families.txt: the three families in both units under the primary, best_clean,
    # sigma_f and both_best_clean normalisers; ratios, medians and the ratio without Branin, Powell and
    # Rosenbrock; counts lower, Wilcoxon p, Spearman(cost) and Spearman(fidelity, ratio); the paired family
    # contrasts, the fidelity counts, the six laws and the fidelity-matched subset; each family's own
    # achievable improvement; and the closing table with every estimator side by side
python scripts/checks/oracle_companion_estimators.py > output-boba/analysis/review/register_checks/oracle_companion_estimators.txt
    # the companion arm under the pooled, per-seed y_opt and per-seed best-clean optima, in both units, for
    # the corrected and the as-logged data, with the landscape-and-dataset ratio bootstrap and the
    # known-function per-landscape ranges (Appendix B.5)
python scripts/run_checks.py --only c6_ratio,aug_contrast_per_dataset
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
python scripts/replay_end_of_study.py --workers 5 --summary-only
python scripts/replay_end_of_study.py --workers 5 --output-dir output-boba/analysis/end_of_study_ksweep --acquisitions logei,qnei --tournament-k 2,3,5,8,12 --confirm-k 2,4,6 --summary-only
python scripts/replay_end_of_study.py --workers 5 --output-dir output-boba/analysis/end_of_study_kwide --acquisitions logei,qnei --tournament-k 16,20,25,30 --confirm-k 2 --summary-only
    # rerun 2026-10-09 from the per-run files: n_looks and unused_trials per tournament row and
    # end_of_study_looks.csv (k = 30 shows every distinct prefix design once, 19.6 on average; k = 25, 20, 16 and 12
    # fall short in 8.4%, 4.0%, 1.8% and 0.3% of runs), and end_of_study_claims.csv with both claim definitions
    # (the fresh-mean claim 6.6% when runs whose chosen design is the initial design may claim, 6.3% [4.8, 8.2]
    # when they make none). The recovery tables were restored, not rewritten: they predate summarise's interval
    # refusal, and a rewrite would blank intervals in low-cost cells no published number uses
python scripts/replay_end_of_study.py --arms output-boba --output-dir output-boba/analysis/end_of_study_ksweep_seq --tournament-k 2,3,5,8,12 --confirm-k 2,4,6 --rho 1,0.5 --sitting-process sequential --workers 12 --resume
python scripts/replay_end_of_study.py --arms output-boba --output-dir output-boba/analysis/end_of_study_kwide_seq --tournament-k 16,20,25,30 --confirm-k 2 --rho 1,0.5 --sitting-process sequential --workers 12 --resume
    # the sitting as a sequence of trials: the drift ramp and the AR(1) state continue through it, candidates in random order
python scripts/replay_end_of_study.py --arms output-boba --output-dir output-boba/analysis/end_of_study_ksweep_seqrank --error-models drift,ar1 --tournament-k 2,3,5,8,12 --confirm-k 2,4,6 --rho 1,0.5 --sitting-process sequential --sitting-order rank --workers 12 --resume
python scripts/replay_end_of_study.py --arms output-boba --output-dir output-boba/analysis/end_of_study_kwide_seqrank --error-models drift,ar1 --tournament-k 16,20,25,30 --confirm-k 2 --rho 1,0.5 --sitting-process sequential --sitting-order rank --workers 12 --resume
    # the same with the candidates shown best-ranked first (drift and ar1 only; gaussian and bias are unchanged by construction)
for sd in 0 0.1 0.25 0.5; do python scripts/replay_end_of_study.py --arms output-boba-slip,output-boba-misclick --tournament-k 5 --confirm-k "" --rho 1 --slip-look-sd $sd --rerate-dirs none --output-dir output-boba/analysis/end_of_study_slipsd$sd --workers 12 --resume; done
    # the input arms' sitting over the assumed look SD (the 0.25 run reproduces end_of_study)
python scripts/replay_end_of_study.py --arms output-boba-confirmatory --output-dir output-boba-confirmatory/analysis/end_of_study_fresh --acquisitions logei,qnei,ucb --seeds 27,28,29,30,31,32,33,34,35,36 --error-models gaussian --stds 0.05,0.25,1.0,5.0 --onsets 0,20 --tournament-k 2,5,12 --confirm-k 2 --rho 1 --rerate-dirs none
python scripts/replay_end_of_study.py --arms output-boba-confirmatory --output-dir output-boba-confirmatory/analysis/end_of_study_fresh --acquisitions logei,qnei,ucb --seeds 27,28,29,30,31,32,33,34,35,36 --error-models gaussian --stds 0.05,0.25,1.0,5.0 --onsets 0,20 --tournament-k 2,5,12 --confirm-k 2 --rho 1 --rerate-dirs none --summary-only
    # the same summaries for the untouched seeds (both claim definitions)
python scripts/rescore_ship_rules.py --input-dir output-boba-confirmatory --output-dir output-boba-confirmatory/analysis/ship_rules_fresh --acquisitions logei,ei,qei,pi,logpi,qpi,ucb,qucb,qnei,greedy --seeds 27,28,29,30,31,32,33,34,35,36 --error-models gaussian
python scripts/replay_end_of_study.py --arms output-boba-confirmatory --output-dir output-boba-confirmatory/analysis/end_of_study_fresh_k16 --acquisitions logei,qnei,ucb --seeds 27,28,29,30,31,32,33,34,35,36 --error-models gaussian --stds 0.05,0.25,1.0,5.0 --onsets 0,20 --tournament-k 16 --confirm-k "" --rho 1 --rerate-dirs none --workers 8
python scripts/checks/gated_sitting.py fresh-k16 > output-boba/analysis/review/register_checks/fresh_k16.txt
    # post hoc: the sitting of sixteen on the untouched seeds 27-36 (LogEI and qNEI, gaussian): +0.146 [0.094, 0.200]
    # at 5 sigma from trial 21, +0.121 over LCB1 and +0.064 over LCB2 (Holm over the eight cells 0.007 and 0.11),
    # +0.009 [-0.020, +0.040] over the sitting of twelve; review/gated_sitting_fresh_k16.csv (Section 7, app:heldout).
    # output-boba-confirmatory is git-ignored: force-add the replay's named files only
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
python scripts/checks/budget_split_by_onset.py
    # register_checks/budget_split_by_onset.txt: the derived rule by onset and magnitude, without the 5 sigma
    # cell from trial 21, where it decides and whether it is blind, landscape groups, fit correlations, the
    # search-term inflation from the logs, the posterior-mean-to-LCB and clairvoyant counterfactuals, the
    # clean-twin prices, and a reproduction of heldout_remedies. Its 'gains at 0.05 sigma' block is not a price.
python scripts/replay_hitl_remedies.py --input-dir output-boba --workers 8
    # hitl_remedies/ (rerun 2026-10-09 with the corrected best-arm rule: kriging-anchored bounds of one SD, the
    # first look per candidate shared with the tournament; every LUCB number before that date is void): at eight
    # looks and the full idiosyncratic SD the best-arm sitting recovers 3.4% [-2.5, 8.8] at price 0.024 against
    # 9.7% [4.8, 14.0] at 0.015 for one look each, and at half the SD 9.6% [3.0, 15.4] against 18.7% [13.1, 23.6]
    # (hitl_remedies_recovery.csv, error_model pooled, procedures lucb_k8_rho* and tournament_k8_lcb_rho*_look).
    # The seven remedy replays of Section 6 below were rerun with it; their non-LUCB rows are unchanged
python scripts/heldout_remedies.py
    # review/heldout_remedies.{csv,md}: held-out scopes, landscapes gaining and Holm p (section 4_holm: columns
    # scope, landscapes_gaining_test, p_holm_core_test, p_holm_all_test), for Section 7 and the Multiplicity
    # paragraph of Appendix E.3: the cautious rule's held-out 10.7% [5, 16] on seeds 12-16, the five rho-0.5
    # sitting survivors, fixed k = 5 (0.27), k = 8 (13 of 20, 0.29), k = 30 (0.03); no best-arm sitting survives
    # (the strongest gains on 14 of 20, adjusted p 0.20), and section 4d tests each against one look each at the same
    # k and noise (ahead on at most 5 of 20, Holm p <= 0.011 over the four pairs). Name the Holm family with every p.
    # Section 4b_holm_rank_scope: the rank rules tested on the gross-fault and capped arms' seeds 12-16, the scope
    # of their recoveries (ordinal_lcb1 48.8% [35, 63], 19 of 20, p 5.7e-6; 37.9% [31, 45], 20 of 20, p 1.9e-6;
    # Holm 1.4e-4 and 5.1e-5 over the 27 with the two main-sweep rank tests replaced, column
    # p_holm_core_lcb1_replaced_test). Section 4c_holm_lucb_comparators: the four one-look sittings the best-arm
    # sittings compete with, added to the all family (45 tests; the top-eight one at half the SD gains on 18 of
    # 20, adjusted p 0.0086; all eight all-family survivors survive). Appendix E.3.
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
python scripts/run_checks.py --only fresh_seed_replication
    # the untouched-seed test (seeds 27-36, gaussian): the zero-trial ship rules; the sitting's increment over
    # LCB1 per cell and k, with Holm over the 3 k and the 24 (cell, k); the looks-alone increment; the clean-twin
    # prices; and, in a last block, the increments over LCB2 and PM with the same Holm families
python scripts/checks/sitting_vs_shiprule.py
    # register_checks/sitting_vs_shiprule.txt: the sitting against the zero-trial ship rules side by side (four
    # processes and gaussian; seeds 7-11, 12-16, 7-16 and 27-36; the quoted cells; the pooled k-curve; the Holm
    # families; the prices; the look model; the increments over LCB2 and PM with the rule chosen on seeds 7-11).
    # It collates the outputs above, so it runs after them.
python scripts/checks/sitting_sequential.py > output-boba/analysis/review/register_checks/sitting_sequential.txt
    # the shared against the sequential sitting: every pooled gain of the fixed-allocation sitting the paper
    # prints, re-scored with sitting_by_magnitude.run and heldout_remedies' k-curve scoring (the chosen k per
    # cell, the derived rule, the tab:heldout k row with its landscape-split column, the optimism of the choice
    # of k, the 41-procedure Holm family), plus the slip-SD and spike summaries. Its shared side must reproduce
    # the published outputs first, and it exits non-zero otherwise.
python scripts/checks/gated_sitting.py refit --workers 8
    # post hoc: the noise SD of the GP that ranks each sitting's candidates, refitted on trials 1..T-k (T-k = 34
    # and 48) for every noisy run and clean twin of LogEI and qNEI on seeds 7-16 and 27-36; needs the run logs,
    # about 12 minutes; writes review/gated_sitting_noise.csv.gz (tracked) and stops unless every refit
    # reproduces the replay's top candidate to 1e-9
python scripts/checks/gated_sitting.py score > output-boba/analysis/review/register_checks/gated_sitting.txt
    # the gated sittings G16 and G2: thresholds chosen on seeds 7-11 by the rule fixed in advance (1 and 0.6
    # landscape SDs), scored on 12-16 and 27-36, the threshold sensitivity and the scale-free gate;
    # review/gated_sitting{.csv,_sensitivity.csv,_selection.json} (Section 7, tab:remedies, app:gated)
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
python scripts/bo_synthetic_error_simulation.py --functions all --acq-list shiplcb --seeds 7,8,9,10,11 --jitter-stds 0.05,0.25,1.0,5.0 --jitter-iterations 0,20 --error-models gaussian --n-jobs 16 --output-dir output-boba-idea-shiplcb2 --resume
for d in output-boba-idea-shiplcb2/*/; do python scripts/evaluate_research_question.py --input-dir $d --output-dir ${d}evaluation; done
    # the ship-rule acquisition rerun after a fault in the first arm (output-boba-idea-shiplcb) made every trial
    # fall back to a random proposal; 0 fallbacks here. analyse_boba_adaptations' idea-shiplcb rows read this arm
    # (+7% [-7, 21] of the deployed cost against LogEI, price +0.2 points). The simulator now refuses to mark a
    # sweep complete when an acquisition fell back (--allow-acq-fallback overrides), and load() raises on such runs
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
python scripts/checks/known_variance_ship.py --workers 8
    # post hoc: the final refit given the rater's noise variance (train_Yvar) at ten levels of knowledge, for the
    # posterior-mean, LCB1 and LCB2 rules. Needs the run logs of output-boba, -confirmatory and -knownnoise, the
    # git-ignored stopping_per_run.csv of replay_stopping.py above (the CUSUM levels) and ship_rules_fresh of
    # Section 5 (the control); caches the refits in review/known_variance_ship/per_run/ (git-ignored, 109 MB) and
    # writes review/known_variance_ship_per_run.csv.gz. 381,193 GP fits, 54 minutes on eight workers
python scripts/checks/known_variance_ship.py --stage score
    # from the tracked per-run table only, about four minutes: review/known_variance_ship_{cells.csv,selection.json},
    # register_checks/known_variance_ship.txt and paper/tables/known_variance.tex (tab:knownvar, app:knownvar,
    # the Section 7 paragraph and tab:remedies rows). The learned level must reproduce ship_rules_per_run.csv bit
    # for bit; the price of every level given the variance is zero by construction
python scripts/checks/decision_map.py > output-boba/analysis/review/register_checks/decision_map.txt
    # post hoc: F1 (freeze only on an alarm) against the published freeze rule, and the per-cell choice of
    # procedure on seeds 7-11 scored on 12-16, in four candidate sets (published; with F1 and the gated sittings;
    # with the self-gating combinations; with the known-variance rules at the knowledge each policy has). Reads
    # the git-ignored stopping_per_run.csv, gated_sitting_noise.csv.gz and known_variance_ship_per_run.csv.gz, so
    # it runs after both; writes review/decision_map_*.csv and paper/tables/decision_map.tex (tab:decisionmap, app:decisionmap)
python scripts/design_rules_from_pilot.py --ceiling-seeds 7,8,9,10,11
    # design_rule_{instrument,instrument_screen,sizing}.csv, reproduced byte for byte: the instrument screen
    # (Spearman 0.65 over the 40 landscape-by-magnitude cells at 0.25 and 1 sigma; half split 0.153 / 0.316;
    # medians 0.219 / 0.104). Without --ceiling-seeds it reads the capped arm's seeds 12-16 as well and gives
    # other numbers. Its uncapped side is the main sweep's cell means over ten acquisitions and all seeds.
python scripts/checks/tie_break.py scan --workers 6
    # every per-run log of output-boba*, output-fitted* and output-oracle-iso*: per run, the first-index and the
    # uniform-tie (exact expectation) deployed regret of the noisy run and its clean twin, and the share of runs
    # with a tied maximum rating per arm, variant and condition. About 15 minutes on 6 workers. Writes
    # review/tie_break/tie_runs.parquet (git-ignored, 13 MB) and tie_share_by_{arm,condition}.csv. Rerun it
    # after any new run logs: rescore reads only its parquet.
python scripts/checks/tie_break.py rescore
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
`scripts/checks/`: the currency-test curvature and spline checks, the
manipulated-landscape intervals, the structural extra-trial zeros, the
like-for-like Kendall's W, the multi-objective front, the nugget-radius
sensitivity, the ehmi sigma_f on two surfaces, the companion ratio interval and
its estimators, the augmentation contrast per dataset, the matched opt_z, the
oracle-isolation checks of Section 4 above, the anchor interval propagated, the
untouched-seed test, the sitting against the zero-trial ship rules and under the
sequential rating process, the budget rule by onset, the ship rules given the
rater's noise variance, the gated sittings, the decision map, the floors' deployed
regret, the ceiling-limited clean runs, the Friedman multiplicity, the archival
instruments' scales, the multi-objective onset bound, the extra-trial reach, the
winner's curse, the one-shot predictor's residuals and out-of-sample fit, the
comparison loop's determinism, the library-version rerun and the tie-break
sensitivity (Sections 2, 5, 6 and 9 above give the direct invocations of the
new ones). `python scripts/run_checks.py` runs them all in dependency
order and writes each one's standard output to
`output-boba/analysis/review/register_checks/<check>.txt`; `--only a,b` runs a
subset, and the direct invocations given elsewhere in this file are equivalent.
Its inputs must exist first: the analyses of Sections 2 to 6 above, the
tie-break scan, `onset_bound_runs.parquet` and, for the checks the docstring of
`run_checks.py` lists, the git-ignored run logs and archival clones.

```
python scripts/run_checks.py        # every check in its CHECKS list, in that order
python scripts/run_checks.py --only oracle_families,anchor_propagation,fresh_seed_replication,matched_optz
    # the three-family isolation table, the anchor interval propagated (checks/anchor_propagation.py), the
    # untouched-seed test (checks/fresh_seed_replication.py) and the headline on matched opt_z
python scripts/run_checks.py --only ehmi_sigmaf,nugget_threshold
    # the ehmi sigma_f on two surfaces (checks/ehmi_sigmaf.py) and the nugget-radius sensitivity; both take
    # datasets.json and read the archival data clones in .dataset_cache/
python scripts/run_checks.py --only mo_front_check
    # the reported multi-objective front: about half truly dominated at 1 sigma, and a shortfall above the evaluated set's in all eight cells
python scripts/run_checks.py --only manipulation_intervals
    # wide- and narrow-spike manipulation contrasts per magnitude, trajectory metric, five seeds (Appendices B.2 and D.3)
python scripts/run_checks.py --only matched_optz
    # the clean twin's deployed regret, 0.1284 of the achievable improvement at 1 sigma from the first rating (Discussion)
python scripts/checks/instrument_scale.py > output-boba/analysis/review/register_checks/instrument_scale.txt
    # each archival instrument's composite step, coarsest-item step, span and ceiling in its study's sigma_f
    # (review/instrument_scale.csv), the ratings at each item's logged ends, and where the instrument arm's
    # +-8 sigma clip meets the twenty landscapes (read from output-boba-instrument/run_metadata.json)
python scripts/checks/mo_onset_bound.py --workers 6 > output-boba/analysis/review/register_checks/mo_onset_bound.txt
    # the running-maximum onset bound for the multi-objective arm, with the scalar arm on the same floor-gap
    # basis (review/mo_onset_bound.csv), and the split of the cross-arm log gap in raw onset ratio into bound and
    # normalised parts; needs onset_bound_runs.parquet from onset_bound.py and mo_vs_scalar.csv from analyse_boba_mo.py
python scripts/analyse_extra_runs.py --input-dir output-boba-budget100 --k 25,50 --tolerance 0,0.01
    # output-boba-budget100/analysis/extra_runs.csv and extra_runs_per_run.csv (tab:extra); the extra trials are
    # counted from clean_origin, the trial at which the clean twin first reached its own k-trial regret. The per-run
    # table also carries clean_origin, reach_trial and reach_over_k, and extra_runs.csv median_clean_origin,
    # median_reach_trial and median_reach_over_k: Appendix B.7's reach trial 56 and 2.2 (2.24) times k at k = 25,
    # gaussian 1 sigma from the first rating, tolerance 0.01. 'multiplier' is (k + extra) / k, not the reach over k
python scripts/checks/extra_reach.py > output-boba/analysis/review/register_checks/extra_reach.txt
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
python scripts/checks/elicitation_determinism.py > output-boba/analysis/review/register_checks/elicitation_determinism.txt
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
uncommitted diff, in the `.stamp.json`). The reports in the repository were
rerun on 2026-10-02 on commit 6268a4360 with a clean tree (the empty diff's
sha256, e3b0c442…); rerun them after any change to the simulator.

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
<venv>/Scripts/python scripts/checks/version_sensitivity.py run --out-dir output-version-check --workers 12
<venv>/Scripts/python scripts/checks/version_sensitivity.py run --out-dir output-version-check --workers 16 --seeds 12,13,14,15,16
<venv>/Scripts/python scripts/checks/version_sensitivity.py evaluate --out-dir output-version-check
python scripts/checks/version_sensitivity.py compare --out-dir output-version-check
    # the reproducibility statement's library check: <venv> is a separate, git-ignored environment with torch
    # 2.14.1+cpu, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, pyarrow 26.0.0 and tqdm 4.70.1 over the
    # pinned BoTorch commit, NumPy and SciPy. It reruns the main sweep's LogEI and qNEI runs in two gaussian cells
    # (1 sigma from the first rating, 5 sigma from trial 21; twenty landscapes, seeds 7-16) with the settings of
    # output-boba/run_metadata.json into output-version-check/ (git-ignored), and compare writes
    # review/version_sensitivity.csv and register_checks/version_sensitivity.txt. In that environment
    # tests/test_regression_logged_run.py checks the first designs and skips the exact comparison
```

`requirements.txt` and `pyproject.toml` pin BoTorch to commit eea35f51a and SciPy
below 1.19: the 0.18.1 release on PyPI and SciPy 1.19 or later take another path
through L-BFGS-B (`tests/test_repro_hygiene.py` checks the pins). The simulator
switches off BoTorch's fused qLogEHVI C++ kernel, so multi-objective runs take
the pure-Python path every logged run took.

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

## 11. Finite-candidate feedback alternatives (implementation, 2026-10-20)

These are a NEW scalar finite-candidate arm, not inputs to existing paper
tables. No experiment command below was executed during the code review.
The comparison producers reuse analyse_boba_adaptations.paired_frame and
summarise, with bo_logei:observed_max as the fixed standard reference.

Read-only planning commands (no objective evaluation or model fitting):

```
python scripts/human_feedback_benchmark.py --config configs/feedback_smoke.json --dry-run
python scripts/human_feedback_benchmark.py --config configs/feedback_screen.json --dry-run
python scripts/human_feedback_benchmark.py --config configs/feedback_heldout.json --dry-run
```

Commands prepared for execution on the OTHER PC, in this order:

```
python scripts/human_feedback_benchmark.py --config configs/feedback_smoke.json --output-dir output-feedback-smoke --workers 2 --run
python scripts/human_feedback_benchmark.py --config configs/feedback_screen.json --output-dir output-feedback-screen --workers 4 --run
# freeze all choices from seeds 7-11 before evaluating held-out seeds:
python scripts/human_feedback_benchmark.py --config configs/feedback_heldout.json --output-dir output-feedback-heldout --workers 4 --run
python scripts/human_feedback_benchmark.py --output-dir output-feedback-heldout --summarize
```

Each output directory holds manifest.json, atomic runs/<task-hash>.json,
per_run_metrics.csv, comparison_summary.csv and comparison_per_landscape.csv.
The CSV comparisons report FINAL-budget search/deployed/selection endpoints,
gain and clean price in fractions of opt_z, with landscape bootstrap intervals
and unsafe recovery-ratio suppression. They do not produce a post-onset AUC or
confirmatory p-values. Use the same command plus --resume after interruption;
changed source/settings/environment requires a fresh directory.

Source-only distribution producer (no experiment execution):

```
python scripts/build_feedback_bundle.py --output dist/human-feedback-portable.zip
```

## 12. Initial execution and Python 3.15 validation (2026-10-21)

Section 11 records the original no-execution implementation. The user then
authorized initial real benchmark runs. These are software smoke tests, not
scientific robustness rankings. The smoke configuration now uses Hartmann 3D,
Hartmann 6D, Branin and Ackley; eight methods; eight judgments; one seed;
Gaussian/drift magnitude 1 at onset 0/4; 160 tasks including clean twins.

```
python scripts/human_feedback_benchmark.py --config configs/feedback_smoke.json --output-dir output-feedback-smoke-py313-20261021 --workers 2 --run
python scripts/human_feedback_benchmark.py --config configs/feedback_smoke.json --output-dir output-feedback-smoke-py313-20261021 --workers 1 --run --resume
python scripts/bo_synthetic_error_simulation.py --functions hartmann_3,hartmann_6,branin --acq logei,random,sobol --iterations 8 --initial-samples 3 --candidate-pool 32 --acq-num-restarts 2 --acq-maxiter 20 --seeds 7 --error-models gaussian,dropout --jitter-stds 1 --jitter-iterations 0,4 --n-jobs 2 --output-dir output-feedback-legacy-smoke-py313-20261021
```

Native Windows Python 3.15.0 was tested in `.venv-feedback315`, using official
PyTorch CPU wheels. `requirements-feedback-py315.txt` records the portable
installation route and was checked with a binary-only clean resolver.

```
py -3.15 -m venv .venv-feedback315
.venv-feedback315/Scripts/python.exe -m pip install --only-binary=:all: -r requirements-feedback-py315.txt
.venv-feedback315/Scripts/python.exe scripts/human_feedback_benchmark.py --config configs/feedback_smoke.json --output-dir output-feedback-smoke-py315-20261021 --workers 2 --run
.venv-feedback315/Scripts/python.exe scripts/human_feedback_benchmark.py --config configs/feedback_smoke.json --output-dir output-feedback-smoke-py315-20261021 --workers 1 --run --resume
.venv-feedback315/Scripts/python.exe scripts/human_feedback_benchmark.py --config configs/feedback_edge_smoke.json --output-dir output-feedback-edge-smoke-py315-20261021 --workers 2 --run
```

The edge configuration spends six budget units per run and uses a clipped,
rounded instrument. Comparisons cost two units and receive three judgments;
ratings receive six. All eleven fault definitions are exercised. The outputs
retain final search/deployed/selection metrics and both numeric tie rules.
Full test and static-check logs are local under `validation-local/`; the
review's results and limitations are recorded in `docs/VALIDATION_2026-10-21.md`.

Final legacy smoke and resume with native Python 3.15, after fingerprint and
completion-marker fixes:

```
.venv-feedback315/Scripts/python.exe scripts/bo_synthetic_error_simulation.py --functions hartmann_3,hartmann_6,branin --acq logei,random,sobol --iterations 8 --initial-samples 3 --candidate-pool 32 --acq-num-restarts 2 --acq-maxiter 20 --seeds 7 --error-models gaussian,dropout --jitter-stds 1 --jitter-iterations 0,4 --n-jobs 2 --output-dir output-feedback-legacy-smoke-py315-final-20261021
# repeat the exact command with --resume: all 45 per-run CSV byte hashes remain unchanged
# --resume with candidate-pool 33 is rejected and removes SWEEP_COMPLETE;
# restoring candidate-pool 32 and resuming restores completion with unchanged CSVs
python -m pytest tests/ -o addopts= -q
.venv-feedback315/Scripts/python.exe -m pytest tests/ -o addopts= -q
python scripts/check_paper.py --paper paper
python scripts/build_feedback_bundle.py --output dist/human-feedback-portable-2026-10-21.zip
```

## 2026-10-21 exploratory Schmitt decision replay (outside the paper)

This new log-only replay does not regenerate or contribute numbers to the
existing paper tables/figures. It fixes 100 available MOBO archives before
scoring: four problems, qLogNEHVI, seeds 12–16, clean plus Gaussian 1 SD at
onsets 0/20 and objective-error correlation 0/.85. All selectors and exact
feedback counterfactuals share the same evaluated designs within each archive.
It tests declared linear and weighted-quadratic utilities separately, with
matching reference rules, and independent illustrative log-weight errors.

Run from the repository root. The output directory must be empty for a run;
use a new directory to reproduce instead of overwriting the first result.

```powershell
.\.venv-feedback315\Scripts\python.exe -m pytest tests/test_replay_schmitt_decisions.py -q
.\.venv-feedback315\Scripts\python.exe scripts/replay_schmitt_decisions.py --input-dir output-boba-mo-halo --openbo-src C:\Users\Mark\Desktop\openbo\src --output-dir C:\Users\Mark\Documents\Codex\2026-10-21\wo\outputs\selection-replay
.\.venv-feedback315\Scripts\python.exe scripts/replay_schmitt_decisions.py --input-dir output-boba-mo-halo --openbo-src C:\Users\Mark\Desktop\openbo\src --output-dir C:\Users\Mark\Documents\Codex\2026-10-21\wo\outputs\selection-replay --run
.\.venv-feedback315\Scripts\python.exe -m pytest tests/ -q --durations=10 *> C:\Users\Mark\Documents\Codex\2026-10-21\wo\outputs\selection-replay\validation_full_tests.txt
.\.venv-feedback315\Scripts\python.exe scripts/check_paper.py --paper paper *> C:\Users\Mark\Documents\Codex\2026-10-21\wo\outputs\selection-replay\validation_check_paper.txt
```

Outputs are `plan.json`, `manifest.json` (input and source SHA256 hashes and
package versions), `per_preference_decisions.csv.gz`, `per_archive_metrics.csv`,
`comparison_summary.csv`, `comparison_per_problem.csv`, `preference_groups.csv`
and `report.txt` in that external output directory. Units are final selection
regret in fixed within-archive true-front normalized utility, not global
optimization regret or hypervolume. Bootstrap intervals are descriptive with
only four correlated problem clusters; no p-values or significance stars are
reported. The clean comparator is the same-archive exact-feedback/true-weight
counterfactual, not a different search trajectory. No new objective trials,
acquisition runs or GP refits are performed.

### 2026-10-22 replay-test portability check

The two OpenBO integration tests now accept an installed package or an
`OPENBO_SRC` source root; the three pure replay tests always run. An explicit
invalid source root raises a setup error. For a sibling OpenBO checkout,
the source case can be repeated without changing the parent environment:

```bash
python -c "import os, subprocess, sys; child_env=os.environ.copy(); child_env['OPENBO_SRC']='../openbo/src'; raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', 'tests/test_replay_schmitt_decisions.py', '-o', 'addopts=', '-q', '-rs'], env=child_env))"
```

Replace the relative source root with the checkout being tested. With an
installed OpenBO package, no environment variable is needed. Without either,
the same pytest command runs three pure tests and skips only the two
integration tests. Four isolated child-environment checks gave: source root
5 passed; importable package without `OPENBO_SRC` 5 passed; unavailable
OpenBO 3 passed/2 skipped; invalid explicit source 3 passed/2 expected setup
errors. Logs are `portability_*.txt` with `portability_summary.json` in the
external selection-replay output directory. The fixture-only correction did
not rerun or change the experiment; the earlier complete-suite result of
1,250 passed remains a separate historical validation.

## 2026-10-23 codebase audit (software validation only)

This review did not regenerate study tables, figures, or experiment results.
Confirmed defects, reproductions, and validation limits are recorded in
`docs/CODE_AUDIT_2026-10-23.md`. Existing uncommitted replay work was preserved.
The validation commands ran in this order from the repository root; the test
suite ran concurrently with the subsequent static checks.

```powershell
$reviewDir = Join-Path $env:TEMP 'hitl-code-review-20261023'
New-Item -ItemType Directory -Force -Path $reviewDir | Out-Null
python -m pytest tests/ -q -ra *> (Join-Path $reviewDir 'pytest-py313.log')
python scripts/check_paper.py --paper paper
ruff check scripts tests paper/build_overleaf_bundle.py --select E9,F63,F7,F82 --output-format concise

$repoPath = (Get-Location).Path
git ls-files '*.ps1' | ForEach-Object {
    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $repoPath $_), [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors) { $parseErrors }
}
git diff --check
```

### 2026-10-23 audit corrections and regression validation

The follow-up fixes all eleven recorded software findings. These commands
validate software behavior; they do not rerun or revise the paper's study.
The temporary producer outputs below preserve the published files; all 29
regenerated tables were byte-identical. The complete implementation test log
is `validation-local/code-fixes-20261023/pytest-py313.log` (git-ignored).

Focused checks, launched before the full suite (some finished concurrently):

```powershell
python -m pytest tests/test_confirmatory_followup.py tests/test_budget_neutral_driver.py -o addopts= -q
python -m pytest tests/test_core_audit.py tests/test_adaptations.py tests/test_error_extensions.py -o addopts= -q
python -m pytest tests/test_hierarchical_oracle.py tests/test_replay_schmitt_decisions.py -o addopts= -q
.\.venv-feedback315\Scripts\python.exe -m pytest tests/test_hierarchical_oracle.py tests/test_replay_schmitt_decisions.py -o addopts= -q
python -m pytest tests/test_budget_split.py tests/test_ship_rules.py tests/test_variant_evaluation.py tests/test_compare_boba_arms.py -o addopts= -q -k "cli_filters_apply or default_run_filters or selected_noisy_scores or selected_clean_prices or cached_cli_finishes or index_keeps_missing or registered_cross_specs or float_magnitudes_never or plot_labels"
```

Producer and final checks, in launch order (the full suite ran concurrently
with the subsequent read-only checks):

```powershell
python scripts/make_boba_paper_tables.py --analysis output-boba/analysis --out validation-local/code-fixes-20261023/tables
python scripts/make_paper_figures.py --analysis output-boba/analysis --out validation-local/code-fixes-20261023/figures
python scripts/check_provenance.py --out-dir validation-local/code-fixes-20261023/provenance --arms 'output-boba-missing-drop' --list
$fixDir = Join-Path (Get-Location) 'validation-local/code-fixes-20261023'
New-Item -ItemType Directory -Force -Path $fixDir | Out-Null
python -m pytest tests/ -o addopts= -q -ra *> (Join-Path $fixDir 'pytest-py313.log')
python scripts/check_paper.py --paper paper
ruff check scripts tests paper/build_overleaf_bundle.py --select E9,F63,F7,F82 --output-format concise
git diff --check
python scripts/check_provenance.py --out-dir validation-local/code-fixes-20261023/provenance --arms 'output-boba-missing-drop' --workers 1
```

All tracked PowerShell files were also reparsed with the AST check from the
initial audit. The provenance scope names one available arm explicitly;
the main sweep's absent run logs still prevent its exact regression replay.

The available missing-drop replay differed from its historical archive. A
same-environment control then loaded the pre-fix core from Git HEAD
(`e16f588b87f74f8c33ac4676e1df9b35488b82cd`) into an isolated module and
replayed the same selected case through `check_provenance._attempt`. It matched
the corrected run exactly on all 50 rows and 32 compared fields, excluding
runtime/run identity and the code fingerprint. This verifies that the
historical difference predates these repairs. The local control command and
its JSON output were:

```powershell
python validation-local/code-fixes-20261023/compare_prefixed_core.py
# writes validation-local/code-fixes-20261023/provenance_before_fix/ and
# validation-local/code-fixes-20261023/prefixed_core_comparison.json
```

Final implementation-suite result: **1,334 passed, 11 skipped, zero failures**
(Python 3.13.12, 141 warnings, 21m45s). All 70 added regression cases passed.
The skipped tests require absent historical logs/metadata or OpenBO selector
integration. Fatal Python lint, PowerShell parsing and static paper checks
passed. Published study outputs were not overwritten.

## 13. The 2026-11-05 round: open items of the Limitations

Every analysis below was built after the procedures it tests and lies outside
the paper's Holm families; each adjusted p it prints names its own family. In
run order.

The rank rule's family-wise test with tied ratings at the cap broken at random
(app:heldout, Limitations). Needs `tie_break.py scan` (Section 6):

```
python scripts/checks/rank_rule_random_ties.py > output-boba/analysis/review/register_checks/rank_rule_random_ties.txt
    # control: the first-index tests reproduce heldout_remedies.py section 4b exactly; then the gross-fault and
    # capped-scale arms' ordinal_lcb1 tests on seeds 12-16 with the standard process's regret replaced by the
    # exact expectation over a uniformly random tied design, Holm over the core family with the two main-sweep
    # rank tests replaced. run_checks.py runs it last.
```

Test-retest pairs whose second rating remembers the first (app:knownvar, Section 7, Limitations):

```
python scripts/checks/retest_memory.py --workers 8
    # refits the three-pair level of known_variance_ship.py with the estimate scaled by (1 - rho_m), rho_m in
    # 0, 0.5, 0.75, 0.9; LogEI and qNEI, four processes, seeds 7-16 (12,800 noisy runs; the control rho_m = 0
    # reproduces known_variance_ship_per_run.csv's retest3 regrets in every run). Needs the run logs; caches the
    # refits in review/retest_memory/per_run/ (git-ignored). Writes review/retest_memory_per_run.csv.gz,
    # review/retest_memory_cells.csv and register_checks/retest_memory.txt
python scripts/checks/retest_memory.py --stage score
    # the same outputs from the tracked per-run table only
```

The best-arm sitting's shortfall per cell (app:remedies):

```
python scripts/checks/lucb_shortfall_by_cell.py > output-boba/analysis/review/register_checks/lucb_shortfall_by_cell.txt
    # from hitl_remedies_per_run.csv.gz: the LUCB sitting of eight less one look each, per magnitude and onset
```

Order effects between the designs of the final sitting (app:endofstudy, Limitations):

```
python scripts/checks/sitting_order_effects.py replay --workers 2
    # anchoring, contrast and fatigue at assumed sizes, the candidates in rank and in random order, for the
    # sitting of two at 1 sigma from the first rating and of sixteen at 5 sigma from trial 21 (main sweep,
    # LogEI and qNEI, four processes, seeds 7-16). Needs the run logs; checks every run's published sitting
    # against replay_end_of_study.tournament. Writes review/sitting_order_effects_inputs.csv.gz
python scripts/checks/sitting_order_effects.py score > output-boba/analysis/review/register_checks/sitting_order_effects.txt
    # checks every run against the published per-run tables bit for bit, then writes
    # review/sitting_order_effects{,_by_process}.csv and review/sitting_order_effects_per_run.csv.gz
```

The held-out stage of the finite-candidate feedback benchmark, with the
selection frozen before any held-out run (`docs/FEEDBACK_HELDOUT_2026-11-05.md`;
not in the paper):

```
python scripts/human_feedback_benchmark.py --config configs/feedback_screen.json --output-dir output-feedback-screen --workers 16 --run
python scripts/feedback_select.py freeze --screen output-feedback-screen --out configs/feedback_frozen_selection.json
    # 2026-11-05T19:31Z; records the sha256 of every screen output it read and of itself. single = the one
    # (method, ship rule) with the largest V over the 48 cells; map = per cell the largest screen gain, or the
    # reference. The file's own sha256 was recorded before the held-out run and checked before scoring
python scripts/human_feedback_benchmark.py --config configs/feedback_heldout.json --output-dir output-feedback-heldout --workers 14 --run
python scripts/feedback_select.py score --heldout output-feedback-heldout --selection configs/feedback_frozen_selection.json
    # output-feedback-heldout/analysis/frozen_selection_scores.{csv,txt}: landscape bootstrap (2,000), Wilcoxon
    # over the 20 landscapes, Holm over the cells where each policy acts
```

The clip a study can run: each rating after the initial design clipped to two
robust SDs (1.4826 MAD) beyond the range of the ratings recorded so far, the
margin fixed before any run (`--response-clip running2`). It acts on the clean
twin as well, so every setting has its own directory and is scored against the
standard process; a clean twin's resume fingerprint holds the error settings,
so two spike sizes cannot share one directory:

```
bash run_boba_cliprun.sh 8
    # output-boba-cliprun2-sp{0.05-5,0.15-5,0.05-20} (seeds 7-11), output-boba-cliprun2-sp0.15-20 (seeds 7-16)
    # and output-boba-cliprun2-gaussian (gaussian 1 sigma from the first rating, seeds 7-16); LogEI and qNEI,
    # 0.25 sigma from the first rating for the spikes; evaluate_research_question.py per landscape
python scripts/analyse_boba_adaptations.py --arms cliprun-sp0.05-5,cliprun-sp0.15-5,cliprun-sp0.05-20,cliprun-sp0.15-20,cliprun-sp0.15-20-heldout,cliprun-gaussian --no-extra-trials
    # adds the cliprun-* rows to adaptations_recovery.csv and adaptations_per_dataset.csv (against
    # output-boba-spike at the same spike size, or the main sweep for cliprun-gaussian): tab:budgetneutral,
    # tab:remedies, app:budgetneutral. Every earlier row is unchanged bit for bit (the merge re-reads with
    # float_precision="round_trip") except pooled_wilcoxon_p_fdr, whose BH family gains the new arms
python scripts/checks/tie_break.py scan --workers 4 --arms output-boba-cliprun2-sp0.05-5 output-boba-cliprun2-sp0.15-5 output-boba-cliprun2-sp0.05-20 output-boba-cliprun2-sp0.15-20 output-boba-cliprun2-gaussian
    # merges the five directories into review/tie_break/tie_runs.parquet: no noisy run of the running clip has a
    # tied top rating, and no clean twin's tie matters, so its numbers need no second tie rule
```

The oracle-isolation arms rerun with each run seed's error grid scaled to that
seed's own oracle (Section 4, app:fitted, Limitations). Never edit
`bo_sensor_error_simulation.py` while `run` is in progress:

```
python scripts/run_oracle_isolation_perseed.py manifest
    # manifest_seeds.csv per family from output-oracle-iso/oracle_sigmaf_seeds.csv; seed 7 must reproduce the
    # published grid exactly
python scripts/run_oracle_isolation_perseed.py check-clean
    # one clean twin per family (and per tree oracle model) rerun and compared exactly: <arm>/clean_identity.json
python scripts/run_oracle_isolation_perseed.py copy-clean
    # the published clean twins of seeds 7-11 copied into the new arms, with their sha256 in <arm>/clean_copies.csv
python scripts/run_oracle_isolation_perseed.py run --n-jobs 6
    # 300 (family, landscape, seed) invocations into output-oracle-iso{,-gaussian_process,-mlp}-perseed/runs
    # (5,400 noisy runs; seed 7's noisy runs equal the published configuration), then evaluation per dataset
python scripts/checks/oracle_perseed.py
    # control: the published pipeline reproduces its own files byte for byte; then the per-seed arms' floor-gap
    # and achievable-improvement files in their own directories, register_checks/oracle_perseed.txt and
    # review/oracle_perseed_comparison.csv (both calibrations side by side under the four normalisers)
```

## 14. The 2026-11-07 review round: three mock reviews and three experiments

Three mock TMLR reviews (methodology, Bayesian optimization, human-in-the-loop) read
a frozen copy of the paper. Their requests that need data are E1-E3 below; the rest
is writing and analysis of existing outputs. Everything here is post hoc with
respect to the paper's procedures. The analysis of E1-E3 was fixed in
`configs/review_round_plan.json` before any of their runs; its sha256
(75ccf817d949cc702ff2716049354511c126b7e348e590d0b86e043efbbe7cd2) is written into
`scripts/checks/review_experiments.py`, which refuses to score if the file changed.

```
python scripts/checks/headline_intervals.py
    # 24 headline quantities (deployed loss, selection shares, LCB1, LCB2, sittings of 2 and 16 and their
    # increments, cautious rule, rank refit, prices) under the paper's percentile landscape bootstrap, a t
    # interval (19 df, delta method) and a two-way landscape x seed bootstrap, plus a seed-effect permutation
    # test; every estimate reproduces its producer. Writes review/headline_intervals.csv and
    # register_checks/headline_intervals.txt (app:estimands, Section 3.4)
python scripts/checks/rerating_memory.py replay --workers 2
    # needs the git-ignored output-boba run logs (about 45 minutes): the sitting of 2 (1 sigma, trial 1) and of 16
    # (5 sigma, trial 21) and the four confirmation trials, with a look's error carrying rho_m of the design's
    # earlier rating error; writes review/rerating_memory_{sitting,confirm}_inputs.csv.gz
python scripts/checks/rerating_memory.py score
    # rho_m = 0 reproduces every published run bit for bit; review/rerating_memory_*.csv and
    # register_checks/rerating_memory.txt (Section 7, app:endofstudy, Limitations, tab:evidence)
python scripts/checks/acq_ranking_shiprule.py > output-boba/analysis/review/register_checks/acq_ranking_shiprule.txt
    # the ten acquisitions ranked on the deployed design's cost of error under the best observed rating (reproduces
    # tab:acq), the posterior-mean maximiser and LCB2: Spearman with the trajectory ranking 0.59, 0.83 and 0.79;
    # Greedy falls to seventh (Section 6)
python scripts/checks/selection_share_optz.py > output-boba/analysis/review/register_checks/selection_share_optz.txt
    # Spearman over the twenty landscapes, 1 sigma from the first rating: opt_z against the selection share
    # -0.82 (p 8e-6), against the deployed loss -0.01 (Section 5, Contributions)
python scripts/make_boba_paper_tables.py --analysis output-boba/analysis
    # tab:acq gains the noisy run's absolute deployed regret (the floors 52.6% against 32-36%); the excess
    # column is reproduced from the per-landscape evaluations or the table is refused; no other table changes
bash run_review_experiments.sh 16
    # E1 output-boba-confirm4p-{1s-t1,5s-t21}: seeds 27-36, bias, drift, ar1 (gaussian is output-boba-confirmatory's)
    # E2 output-boba-driftdown-{1s-t1,5s-t21}: seeds 7-16, drift_down, paired with the main sweep's drift
    # E3 output-boba-gplegacy-{1s-t1,5s-t21}: seeds 7-16, four processes, --gp-prior legacy
    # LogEI and qNEI, T = 50, the two headline cells, one directory per cell; evaluate_research_question.py per
    # landscape and ARM_COMPLETE per directory
python scripts/checks/review_experiments.py verify-plan
python scripts/checks/review_experiments.py replay --workers 4
    # rescore_ship_rules.py on all six directories (E3 refits with the legacy prior); replay_end_of_study.py
    # (sitting of 2 in the 1-sigma cell, 16 in the 5-sigma trial-21 cell, rho 1, shared sitting) on E1 and E2
    # into <dir>/analysis/end_of_study_review
python scripts/checks/review_experiments.py score > output-boba/analysis/review/register_checks/review_experiments.txt
    # review/review_experiments_{e1,e2,e3}.csv: E1's family H1-H4 with Holm, every quantity per process,
    # three interval methods; E2 and E3 as paired contrasts with the main sweep. E1: all four tests positive
    # (LCB2 7.4% at 1 sigma, Holm 0.019; sitting of 16 41.5% at 5 sigma trial 21). E2: reversed drift costs
    # 0.304 against 0.267 at 1 sigma (selection) and 0.493 against 0.637 at 5 sigma trial 21; the sitting's lead
    # over LCB2 falls to +0.020. E3 (legacy GP): cost +0.036 at 1 sigma (LogEI's search loss), -0.134 at 5 sigma
    # trial 21 (selection loss); LCB2 2.7% [-9, 12] against 8.1% [1, 15]. E3's per-acquisition and per-process
    # rows are beyond the plan (point estimates). Abstract, Claims, Contributions, tab:evidence, Section 7,
    # Section 5, Limitations, Discussion, sec:robustness (B.1), sec:noisediag, app:endofstudy
    # (run_checks.py runs this score stage as the check review_experiments)
```
