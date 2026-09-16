# Third pass, 2026-09-16

This pass differs from the first two in three ways. Code was executed where the sandbox allowed it. The claims were checked against the real side-arm outputs on the Mac. The earlier audit documents were themselves red-teamed and corrected. PyPI is blocked by organisation policy in both sandboxes, so torch, botorch and statsmodels could not be installed. pytest 9.0.3 ran from a local cache. A short numpy shim stood in for the two statsmodels functions the repository uses (Benjamini–Hochberg and OLS with R²). Every result below that relies on the shim says so. The main sweep (`output-boba/`), the confirmatory runs, the budget-100 arm and the fitted-oracle outputs are not on the Mac, so numbers that need them remain unverified.

Status labels. EXECUTED means code ran here and produced the stated result. DATA means the result was computed from the real output files on the Mac. VERIFIED means the cited lines were read and confirmed. RELAYED means a reviewer checked it and it is consistent with the code read here.

## H. What was executed, and what it showed

### H1. EXECUTED. 98 tests pass offline and none fails on an assertion

With pytest from the cache and the statsmodels shim, 98 tests ran and passed. They are in `test_changepoint.py` (18), `test_mo_front.py` (19), `test_variant_evaluation.py` (17), `test_paper_tables.py` (15), `test_decompose_regret.py` (9), `test_inputerror_analysis.py` (7), `test_currency.py` (6), `test_extra_runs.py` (5) and `test_cli_smoke.py` (2). The three files marked with the shim are `test_currency.py`, `test_inputerror_analysis.py` and `test_variant_evaluation.py`. Every other test failed at import on torch, jax or tqdm. No test failed on an assertion. Roughly 345 of the 443 test functions remain unexecuted. As shipped, `python -m pytest tests/` aborts with "Interrupted: 1 error during collection" whenever jax and numpyro are absent, so it runs no test at all (confirmed here with `test_changepoint.py` plus `test_hierarchical_oracle.py`).

### H2. EXECUTED. The landscape statistics reproduce across platforms

`boba_benchmarks.landscape_stats` was rerun on Linux with numpy 2.4.4 for all 29 entries of `boba_landscape_stats.json` (Windows, Python 3.12.9). For the 22 landscapes that need no torch, every field the paper uses (mean, SD, y_opt, opt_z, sparsity, ruggedness, skew, tail ratio, frag at 0.05, 0.25, 1 and 5) agrees to at most 1.6e-13 relative error. Every opt_z and frag(1) value printed in Tables 12 and 19 for those landscapes matches. The other seven (branin, rosenbrock, rastrigin, michalewicz and the three Levy entries) call BoTorch test functions and were not checked. The polish-derived field `opt_best_found` differs at the 1e-16 absolute level, which does not enter opt_z.

### H3. EXECUTED and DATA. The input-error tables are reproducible end to end

`make_boba_paper_tables.py` regenerated `inputerror_dose.tex`, `inputerror_mechanism.tex`, `inputerror_deployed.tex` and `inputerror_main.tex` from the analysis CSVs on the Mac. All four are byte-identical to `paper/tables/`. An independent implementation of the primary metric (post-onset trapezoid AUC of `simple_regret_true` per interval, noisy minus clean, divided by opt_z, averaged per landscape) recomputed two Table 5 cells from the raw per-run logs. A 15% slip from trial 1 gives 11.49% (CSV 11.49%). A 40% slip from trial 21 gives 6.43% (CSV 6.43%). The three input-error rows of Table 24 also rebuild exactly from `extra_runs.csv`.

### H4. DATA. The price column of Table 26 reproduces from raw summaries

The clean runs of `output-boba-relay` are standard-process clean runs for LogEI and qNEI, seeds 7 to 11. Using them as the reference, the deployed-design price is +1.63 points of opt_z for early replication (paper +1.6), +0.84 for final re-rating (+0.8), +1.55 for the Student-t surrogate (+1.5) and −1.48 for the noisy-input GP (−1.5). This confirms that those arms share their clean runs with the standard process.

### H5. DATA. Integrity of the 33 side arms on the Mac

Across all 33 side arms with summaries, the acquisition-optimisation fallback count is zero, no final regret is NaN and no final regret is negative. Clean-run selection loss is zero to 1e-15 in every arm except the three backfit arms (`relay-backfit-block`, `relay-backfit-rr`, `mo-halo-backfit`), where the remedy changes the clean run's ship rule. The per-arm `bo_synthetic_error_summary.csv` holds only the last invocation's runs in directories with several variants. No analysis script reads those summaries, so no paper number is affected.

### H6. DATA. The slip draws are shared between the unnoticed and the logged slip arms

Model-free floors propose independently of the data. Their true-value sequences are bit-identical between `output-boba-slip` and `output-boba-slip-actual` in 80 of 80 checked runs (five landscapes, two floors, two seeds, two magnitudes, two onsets). Learners are identical until their records diverge. Table 22's mislabelling column is therefore a genuine paired contrast.

## I. New findings

### I1. BLOCKER. DATA. The deployed design loses two to three times what the headline metric reports

A1 established that Table 1, Sections 4, 6 and 7, the Table 4 "absolute loss" column and Appendix C use the post-onset time average of search loss. The real data now size the gap. At the final trial, the deployed-design excess divided by the search excess is 2.45 in `session25` (LogEI and qNEI, T = 25, 0.71σ and 3.5σ from trial 1) and 2.64 in `session100` (T = 100, 1.41σ and 7.1σ). Across all response-error side arms the ratio ranges from 1.6 to 5.9. The paper's own pooled decomposition gives the same answer, since a deployed excess of 0.218 against a final search excess of 0.083 is a ratio of 2.6. The abstract's "costs 14.1% of the available improvement" therefore understates what a study ships by a factor of about two to three. Table 4's caption calls the trajectory metric "the deployment number", which is false. Fix. Report a deployed-design column beside every search-loss number, pair it with `final_simple_regret_excess_true / opt_z` so both use the same time base, and rename the current metric "post-onset time-averaged search loss".

### I2. MAJOR. DATA. Selection is about half of the deployed cost at realistic magnitudes

For the standard process (LogEI and qNEI, error from trial 1), selection is 52% of the final deployed excess at 0.71σ (T = 25) and 56% at 1.41σ (T = 100). In the augmented-EI, Thompson-sampling and minimum-distance arms it reaches 77 to 91% at the late onset from 0.25σ upward, where search loss is small. The pooled 61.8% is an excess-weighted average pulled up by those cells. A title or abstract that says "most" of the cost is selection needs the 1σ, onset-0 cell of `regret_decomposition.csv` (main sweep) with its interval. If that interval does not lie above 0.5, say "about half".

### I3. MAJOR. DATA. The "−979%" row of Table 25 divides by a cost near zero

Computed from the raw logs, dropping a lost rating costs +0.60 and −0.67 points of opt_z at missingness 0.15 and 0.30 when low ratings go missing, and −0.25 and +2.17 points when ratings go missing at random. The reference cost changes sign across cells, so any recovery ratio is unstable. Imputing a lost rating low costs +6.2 and +14.5 points under random missingness and +0.26 and −0.34 points when low ratings go missing. The substantive finding is that low imputation is harmful when ratings go missing at random. Report the absolute difference and drop the ratio.

### I4. MAJOR. VERIFIED. Appendix L describes the confirmation rule differently from the code

The paper says the chosen design "ships only if a one-sided paired t-test at α = 0.05 prefers it" (main.tex:1427). `replay_end_of_study.confirmation_decision` ships the chosen design whenever its mean rating is at least the comparator's and uses the t-test only for the improvement claim (lines 22–28 and 485–491). The "3.2 to 3.4% of the cost" and the false-claim rates follow the code. Fix the sentence.

### I5. MAJOR. DATA. The late-onset "no extra trials" is guaranteed by construction

In `session100` (T = 100, LogEI and qNEI), 171 of 200 clean runs (86%) had already reached their trial-50 regret, within 1% of opt_z, by trial 41. A noisy twin is identical to its clean run up to trial 41, so each of these runs scores exactly zero extra trials whatever the error does. The reported median of 0 in Table 2's right column carries no information. In Table 24 (T = 50, error from trial 1, target the clean trial-10 regret), 142 of 600 clean runs (24%) reached the target within the five shared initial designs, so a quarter of that table's zeros are structural. Report the share of structural zeros, or restrict each cell to runs whose clean twin improved after the onset.

### I6. MAJOR. VERIFIED. Augmented EI and Thompson sampling are scored against a ten-acquisition average

`analyse_boba_adaptations.py:129-134` scores each of these single acquisitions against the mean of `logei, ei, pi, ucb, qucb, qnei, logpi, qei, qpi, greedy` (line 70). That pool contains the four weakest acquisitions of Table 4. Table 4 already places qUCB 36% below the ten-acquisition mean excess (4.14% against 6.44%). Augmented EI's 41% recovery of the trajectory cost therefore mixes its advantage as an acquisition with any effect on the error. It is not a clean test of the decomposition. Score it against EI and LogEI and report its price.

### I7. MAJOR. VERIFIED and DATA. Table 25 is typed by hand and has no price column

Table 25 is a literal tabular at main.tex:1488-1516 with no generator, although the paper says every table is scripted. Prices computed here from raw data, in points of opt_z on the deployed design, are 0.00 for response clipping, +1.12 for relevance pursuit, +0.75 for per-rater backfit with round-robin handover, −0.41 with block handover, +1.38 for the LCB incumbent (LogEI), +4.69 for minimum-distance proposals (LogEI), +2.82 for the slip-aware acquisition at 0.05 and +14.55 at 0.4. The abstract's "near-zero price" holds for clipping and relevance pursuit relative to their costs. Several acquisition-side arms carry large prices. Generate Table 25 from `analyse_boba_adaptations.py` output and add the price column.

### I8. MAJOR. VERIFIED and RELAYED. The fitted-oracle arm rests on one usable oracle for its fidelity claim

The deployed provoice composite oracle has held-out R² −0.03, so it is worse than a constant. It still feeds Table 7, Table 20 and the fitted row of Table 26. The quality gate `--min-oracle-r2` exists (bo_sensor_error_simulation.py:931, 4473) and no driver sets it. Turning augmentation off changes oracle fidelity by +0.01 to +0.03 on ehmi and barely at all on provoice. Only opticarvis moves (+0.29). `analyse_fitted_companion.py:185-197` pools the augmentation contrast over all three datasets with no per-dataset breakdown. The abstract's "improving its oracles by up to 0.29 held-out R² does not close the gap" therefore rests on opticarvis alone, which is never reported separately. Report the contrast per dataset, and either drop provoice or state its fidelity in the text.

### I9. MAJOR. VERIFIED. Appendix D.3's null holds at one magnitude only

Raising the narrow spike from amplitude 4 to 16 changes the cost at 1σ from 0.246 to 0.245. At 0.25σ the same change lowers it from 0.152 to 0.026, a sixfold drop. At 5σ it falls from 0.324 to 0.262, and at 0.05σ it rises from 0.007 to 0.014 (Table 12). "Raising a narrow optimum changes nothing" (main.tex:442, 596 and 980) is true only at 1σ. State the full pattern.

### I10. MINOR. VERIFIED. BIAS is always positive

`apply_sensor_error` adds `+error_bias` (bo_sensor_error_simulation.py:2819-2821), and `--error-bias-mode` offers only `scaled` and `fixed`. The paper describes "a systematically generous or harsh rater" (main.tex:228). Every BIAS result describes a generous rater, and an upward step is not symmetric with a downward one for a maximiser. Say so, or add a harsh-rater arm.

### I11. MINOR. VERIFIED. The landscapes share their random numbers

The noise seed has no landscape key (bo_synthetic_error_simulation.py:754-771), and the initial designs come from the same seeded stream on every landscape. This is deliberate and tested. The landscape bootstrap and the landscape-level Wilcoxon tests treat landscapes as independent clusters, so their intervals may be too narrow. Disclose the design and add a sensitivity check that resamples seeds within landscapes.

### I12. MINOR. VERIFIED. One cell has two intervals

The known-function arm at 1σ from trial 1 has interval [55, 155] at main.tex:840 and [54, 154] at main.tex:1093 and in Table 13. Use one bootstrap for both.

### I13. MINOR. VERIFIED. The fitted optimum mixes seed-specific functions

`fitted_optimum` takes the maximum clean-run best over all seeds (analyse_fitted_companion.py:79-88), and each seed trains its own oracle. The maximum is taken over ten different functions, so the gap can exceed any single seed's gap. The paper's "the bias works against the result below" (main.tex:833-836) is therefore not guaranteed. The search box is the min/max box of a prior campaign (`bounds_from_data`), so σf and opt_z include regions no rating supports.

### I14. MINOR. VERIFIED. All three archival studies are automated-vehicle interface studies

The parameter lists in `datasets.json` describe an external HMI, a trajectory visualisation and an in-vehicle intervention interface. Scope the 12–20% anchor and Appendix C's "property of human objectives" to that domain.

### I15. MINOR. VERIFIED. The fitted-arm inputs have no recorded producer

No driver runs `calibrate_noise_from_data.py` or `anchor_noise_scale.py`, which produce `output/noise_calibration.csv` and `output/noise_anchor.csv`. Neither script records data provenance. The simulator records `data_dir_commits`, but README tells authors to quote those SHAs in the paper, which would de-anonymise the submission. The Reproducibility statement's "scripted end to end ... to every table" is false for Table 20, Table 25 and the fitted arm.

### I16. MINOR. VERIFIED. The environment manifests contradict each other

`requirements.txt` sets floors above the pins in `requirements-eval.txt` for numpy (2.5.3 against 2.5.2), scipy (1.18.1 against 1.18.0), tqdm (4.70.0 against 4.68.3) and tabpfn (8.5.0 against 8.2.0). `pyproject.toml` requires Python 3.13 or newer and claims its floors match `requirements.txt`. The results were produced on Python 3.12.9 with a CUDA torch build. CI tests Python 3.13 and 3.14 and installs no jax or numpyro, so its test step should abort at collection. No manifest describes the environment behind the numbers.

### I17. MINOR. VERIFIED. Identifiers and third-party data outside the paper

`LICENSE` names the author. Four test files (`test_boba_benchmarks.py:36`, `test_mo_halo.py:37`, `test_error_extensions.py:39`, `test_acquisition_extensions.py:39`) and `docs/known-function-arm-2026-09-06.md` carry the Windows username. `docs/code-review-2026-06-12.md`, `datasets*.json` and `tests/test_cli_smoke.py` carry GitHub account names. Every `run_metadata.json` carries the username in its paths. `sandbox/` and `external_datasets/` hold third-party ASHRAE and NISQA data that the paper does not use, including third-party contact addresses. The `.ps1` drivers themselves carry no username (they read `$env:PYTHON` or `$env:LOCALAPPDATA`).

### I18. MINOR. RELAYED. Resumed runs carry no code stamp

`--resume` reuses per-run CSVs that record no code version or environment, and `run_metadata.json` reflects only the last invocation. The recorded settings may not match most runs of a resumed arm.

### I19. MINOR. VERIFIED. The preregistration has no external timestamp

`output-boba-confirmatory/HYPOTHESIS.md` is a local file. A reviewer cannot verify that it predates the fresh seeds. A dated commit hash or an anonymised registry entry would settle this.

### I20. MINOR. VERIFIED. Section 3.3 omits the optimiser settings

Each iteration screens 1000 uniform points on the shared stream, starts L-BFGS-B from the best 10, and runs up to 200 iterations. Monte-Carlo acquisitions use 256 samples (64 in the multi-objective arm). Improvement acquisitions add ξ = 0.01, and UCB uses β = 4. None of this appears in the paper.

### I21. MINOR. VERIFIED. Appendix D.1 miscounts observations

Iterations include the five initial designs (bo_sensor_error_simulation.py:3577-3582). "The first ten of fifty iterations (fifteen observations)" (main.tex:872-873) should read "the first fifteen iterations".

### I22. MINOR. VERIFIED. Three-cluster stars

The fitted row of Table 26 (−161%, starred, [−3504, −49]) takes its star from a bootstrap over three datasets. A Wilcoxon test over three per-dataset gains cannot fall below p = 0.25. Remove the star or state the cluster count.

### I23. MINOR. VERIFIED. The instrument's span has two values

The paper says the three instruments "span sixteen to twenty σ" (main.tex:1190-1191). The driver comment says "13, 17 and 22 sigma" (run_boba_gaps2.ps1:7). No script computes either.

## J. Corrections applied to the earlier documents

The red-team read found errors in the handover documents. Each correction below was verified here and has been applied to the document named.

| document | earlier statement | corrected statement | evidence |
|---|---|---|---|
| readiness_review §5 | "Wednesday 17 ... Thursday 25" | 17 Sep is a Thursday and 25 Sep a Friday. Every weekday label moved by one day. | Python calendar |
| readiness_review, title | "Most of what noisy feedback costs ... is selection" | Conditional on the 1σ onset-0 share. Neutral title proposed. | I2 |
| readiness_review §2.1 | augmented EI result called a clean mechanism result | It uses a ten-acquisition reference. | I6 |
| readiness_review §2.2 | the pilot has "no optimizer in the loop" | The pilot is a GP fitted to a clean LogEI run. The 0.52 against 0.78 comparison uses exact frag. | main.tex:1701 |
| readiness_review §2.3 | the shekel sensitivity weakens Section 6 | Dropping shekel moves βz from 0.82–0.86 to 1.04–1.10 and the gap from 0.36–0.51 to 0.57–0.73, which strengthens both rejections. | main.tex:439-441 |
| readiness_review §2.3 | "no acquisition-side change is distinguishable from zero" | No interval lies above zero, and seven of twelve rows are significantly negative. | Table 25 |
| readiness_review §2.3 | rerun the confirmatory arm "with the headroom threshold set at 0.05" | Keep 0.10, exclude Rosenbrock by name, and date a new preregistration before the run. | main.tex:1255 |
| readiness_review §2.3 | "Picheny et al. 2013 report regret at the best observed point" | Removed as unverified. | none |
| readiness_review §2.2 | Wang et al. 2024 on robust DPO | Replaced by Chowdhury, Kini and Natarajan (ICML 2024) and Lee et al. (B-Pref, NeurIPS Datasets and Benchmarks 2021). The noisy-input GP is attributed to McHutchon and Rasmussen (2011). | reference check |
| readiness_review §3 | topic "applications to human-facing systems" | Not a 2027 topic. Probabilistic methods, optimization, and datasets and benchmarks fit. | ICLR 2027 CfP |
| readiness_review §3 | abbreviated reviewer-venue list | Full list added, plus the rule that authors on three or more papers review six. | Author Guidelines |
| readiness_review §3 | AI statement "must follow the template" and should cite "the 443 tests" | The template may be adapted. The statement may list only checks that were performed. | AI Policy for Authors |
| readiness_review §4 | abstract numbers 0.051 and 0.85 | 0.051 is an absolute gain in achievable-improvement units. The pilot's Spearman is 0.32 at 1σ and 0.85 pooled. | Table 27 |
| readiness_review §6 | split into a separate HCI paper | A parallel, substantially similar submission breaches ICLR's dual-submission policy. The option is restated without a venue. | Author Guidelines |
| code_audit A1 | "The deployed design's cost is larger, by the selection term." | The two metrics have different time bases. Compare deployed excess with final search excess. Scope extended to Table 4's caption, Appendix C and Table 7. | I1 |
| code_audit B3 | the paper describes the re-presentation estimator | The paper names the nearest-neighbour estimator. Only "repeated ratings" is loose. | main.tex:320 |
| code_audit B4 | Table 4's floor row is a literal | Only the excess cell is literal. The script with the `== 0.0` test sits outside pytest's testpaths. | make_boba_paper_tables.py:178, 186 |
| code_audit B5 | the sitting model is wrong | It is a modelling assumption that should be stated with its range of validity. | replay_end_of_study.py:290-314 |
| code_audit B7 | Appendix K gives +5.8 to +18.5% on Gaussian ≥ 0.25σ | Appendix K reports the 1σ and 5σ cells. Section 5's shares are of selection loss, Appendix K's of deployed cost. | main.tex:1397-1400 |
| code_audit B9, second_pass F4 | σ² is not the injected variance for AR(1) and drift | σ² is the marginal variance of both noise components. The known-noise GP misses the AR(1) correlation and the drift ramp. | bo_sensor_error_simulation.py:2831-2862 |
| code_audit C3 | 19.7% attributed to a third of the budget | 19.7% is the best of 19 variants, a five-candidate sitting. A third of the budget (k = 16) gains +0.051 absolute. The twelve rows above the rule are seven kinds of change. | main.tex:528-541 |
| code_audit C4 | VIFs in Table 19 | Table 21. | descriptors.tex |
| code_audit C13 | README still stale | README is correct. The stale values sit in `docs/known-function-arm-2026-09-06.md`, `scripts/decompose_regret.py:41` and `tests/test_currency.py:18`. | grep |
| code_audit D | the guard tracks everything except student_t | It tracks min_distance, input_uncertain, relevance pursuit, rater_model and mo_halo_model only. | bo_synthetic_error_simulation.py:494-514 |
| code_audit header | "Nothing was executed" | Reviewers ran numpy and scipy checks for the BCa and quasi-Poisson items. | first-pass reports |
| second_pass D2 | 14 tests error at collection | The module errors once and pytest aborts the whole run. | H1 |
| second_pass D3 | no SHA recorded | The simulator records `data_dir_commits`. README's advice to quote them would de-anonymise. The venue appears in the data repository's name. | bo_sensor_error_simulation.py:4828-4843 |
| second_pass D4 | rerun provoice "via run_provoice_normalized.ps1" | That legacy driver uses 20 seeds, other error models, onsets and grid. A new driver and a change to `anchor_noise_scale.py:80` are needed. | run_provoice_normalized.ps1:32 |
| second_pass D5 | opticarvis uses ExtraTrees with leaf size 2 | opticarvis uses gradient boosting (depth 3). The memorisation argument applies to provoice. The box extrapolation (I13) is an alternative explanation for opticarvis's opt_z. | known-function-arm doc |
| second_pass D10 | 256 distinct resamples | 35 distinct multisets for four clusters. | C(7,4) |
| second_pass E3 | augmented EI's numbers imply selection loss rose | Invalid. The trajectory metric is a time average of search loss and the deployed metric is a final value. The point that the acquisition shapes the visited set stands. | reasoning |
| second_pass E7 | shares 0.24 onset, 0.17 landscape | Magnitude 0.24, onset 0.17, landscape 0.10, acquisition 0.04, process 0.002. | main.tex:311 |
| second_pass E11 | Table 23's "evaluated" column is an oracle | Ratings are exact in the input-error arms, so the best-rated trial's actual design is the best evaluated one and is identifiable. | H6, bo_synthetic_error_simulation.py:781-790 |
| second_pass E12 | the section names none of the required categories | It describes design, implementation and analysis. It lacks the negative list and the verification method. | main.tex:679-686 |
| second_pass G | "worse under iid noise" survived | Its interval is [−79, +15]. Removed from the list. | main.tex:1551 |
| AGENTS.md | 19.7% for a third of the budget, "cannot reach", "recover nothing", "−41% for LogPI", "harmful below 0.25σ" for every remedy, username, hard-coded interpreter, defect table | Rewritten with verified, scoped statements. The defect table moved to `handover/README.md`. | above |
| handover README | one core-second per run, two weeks on four workers | 19.7 h × 24 workers / 79,200 runs is about 21.5 worker-seconds per run, about five days on four workers of equal speed. | arithmetic |
| handover README | back up `output-boba/analysis` | Back up the analysis and evaluation directories, `run_metadata.json`, and every directory `make_boba_paper_tables.main` reads. | make_boba_paper_tables.py:1088-1167 |
| handover README | rerun Student-t, budget_split, provoice | Student-t needs a fresh output directory. budget_split must run without `--summary-only` and with the recorded k-grid and ρ. provoice needs a new driver. | run_boba_adapt.ps1:65, 75; budget_split.py:94-95, 378 |

## K. Still unverifiable here

The main-sweep CSVs (`output-boba/`, Tables 1–4, 6, 9–11, 13–21, 25, 27), the confirmatory runs, the budget-100 arm, the fitted-arm outputs, the elicitation and change-point outputs, and every test that needs torch. The installed BoTorch default kernel can be read from the `gp_kernel` column of `ship_rules_per_run.csv` on the simulation machine. The 1σ, onset-0 selection share can be read from `regret_decomposition.csv` there.
