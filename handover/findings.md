# Findings register

This register is the single list of known problems as of 2026-09-16. It merges three audit passes. The evidence for each item is in `history/`, where the IDs come from (A to C in `code_audit.md`, D to F in `second_pass_findings.md`, H to J in `third_pass_findings.md`). Update the status column when an item is fixed.

Basis codes. X means executed in an audit sandbox. D means computed from the real output files. V means confirmed by reading the cited lines. R means reported by one reviewer and consistent with the code read, without independent re-derivation.

Severity. BLOCKER changes a headline number, invalidates a claim, or triggers a desk rejection. MAJOR is likely to cause a rejection. MINOR is worth fixing before submission.

## 1. Claims and numbers in the paper

| ID | severity | problem | where | fix | basis | status |
|---|---|---|---|---|---|---|
| A1, I1 | BLOCKER | The headline metric is the post-onset time average of search loss. The text, the abstract and Table 4's caption present it as the cost of the shipped design. The shipped design loses 2.45 to 2.64 times as much at the final trial for the standard process, and the paper's pooled numbers give 0.218 against 0.083. | main.tex:356, abstract, Table 4 caption, Appendix C, Table 7 | Add a deployed column beside every search-loss number, compare final with final, rename the metric | V, D | fixed 2026-09-17 |
| A2 | BLOCKER for Section 9 | The budget rule takes the sitting SD from GP noise in standardised units and combines it with a posterior in objective units. | budget_split.py:235 | Multiply by `outcome_transform.stdvs**2`, rerun without `--summary-only` and with the recorded k-grid and ρ | V | fixed 2026-09-16 |
| A3 | MAJOR | The interaction and quadratic terms are formed from uncentred logs, so "the magnitude is significant again" is a slope at opt_z = 1 and σ = 1. | analyse_boba_robustness.py:848-861, main.tex:797-799, Appendix G | Centre, refit, rewrite both passages | V | open |
| I2 | MAJOR | Selection is about half of the deployed cost near 1σ from trial 1 (52% at 0.71σ, 56% at 1.41σ). The 61.8% holds only for the excess-weighted pool. | abstract, title, decompose_regret output | Report the 1σ onset-0 share with its interval before choosing the title | D | fixed 2026-09-17 |
| E4 | MAJOR | The pooled shares 61.8%, 5.0%, 15.6% and the multi-objective "at most half" are excess-weighted and dominated by 5σ cells. | main.tex:387-390, 585, 590, 601 | Report the 1σ onset-0 value beside each pool | R | fixed 2026-09-17 |
| E3 | MAJOR | "The term an acquisition function cannot reach" and "the reason is structural" are stated as facts. The acquisition shapes the visited set. | abstract, main.tex:133-136, 369-371, 395-396 | State that no acquisition-side change tested reduced it | V | fixed 2026-09-17 |
| I6 | MAJOR | Augmented EI and Thompson sampling are scored against the mean of ten acquisitions, including the four weakest. | analyse_boba_adaptations.py:129-134 | Score against EI or LogEI and report the price | V | open |
| E2, I3, I7 | MAJOR | Table 25 is typed by hand, has no price column, defines none of its fault models or remedies, and its "−979%" row divides by a reference cost near zero that changes sign. | main.tex:1462-1516 | Generate the table, add prices and definitions, report absolute differences for missing ratings | V, D | open |
| I4 | MAJOR | Appendix L says the chosen design ships only if a t-test prefers it. The code ships on the higher mean rating and uses the t-test for the claim only. | main.tex:1427, replay_end_of_study.py:22-28, 485-491 | Rewrite the sentence | V | fixed 2026-09-17 |
| I5, D8 | MAJOR | "Zero extra trials" for late error is structural in 86% of runs, and 24% of Table 24's runs are structural zeros. | analyse_extra_runs.py:139-144, main.tex:341, Tables 2 and 24 | Report the structural share or restrict each cell | D | open |
| A4 | MINOR | "A budget of 2.6×" is computed as (k + extra)/k with extra measured from an earlier origin. | analyse_extra_runs.py:183, main.tex:339 | Drop it or compute trial/k | V | open |
| E9 | MAJOR | frag is compared with an additive model the paper calls misspecified, using in-sample R² only, with no σe-only baseline, and the pilot uses a clean run. | main.tex:462-468, 789-811, Appendix R | Add leave-one-landscape-out R², a σe-only row and a noisy-pilot row | R | open |
| E5 | MAJOR | The power law is fitted on four magnitudes while Appendix B finds curvature in log σe. Poisson pseudo-likelihood is uncited. | main.tex:406-441 | Keep "SPREAD fails", drop the GAIN verdict or refit per magnitude pair, cite PPML | R | open |
| I9, E6 | MAJOR | Appendix D.3's "no effect" holds at 1σ only (0.152 to 0.026 at 0.25σ). The dimension claim is a two-point comparison without intervals. | main.tex:442, 596, 980, 1002-1005 | Add intervals to Table 12 and state the full pattern | V | open |
| E8 | MAJOR | The best of 19 replayed variants is reported without selection correction. The k-curve peak is unresolved, the replay design is unstated, and the false-claim rates lack a denominator. | main.tex:527-546, 1427-1432 | Intervals for every k, a table of variants, the runs and denominators | R | open |
| I8, D4 | MAJOR | The provoice composite is 81% mental demand and its oracle has held-out R² −0.03. The augmentation contrast is pooled over three datasets, so "better oracles do not close the gap" rests on opticarvis alone. | analyse_fitted_companion.py:185-197, run_boba_gaps_rest.ps1:151-157, main.tex:85-86, 846-852 | Per-dataset contrast, and a normalised provoice rerun with a new driver or removal of provoice | V, R | open |
| B3, D5 | MAJOR | The rater-noise anchor uses a nearest-neighbour estimator that is overestimated for opticarvis, and its three denominators are SDs of different kinds of surface. | anchor_noise_scale.py:80, 103-108, Table 20 | One estimator, one kind of surface, both bias directions | V, R | open |
| E1 | MAJOR | The "standard construction" of fitted regression oracles is uncited. | main.tex:45-46, 103, 106 | Cite third-party instances | V | open |
| E10 | MAJOR | Methods the paper runs are uncited (relevance pursuit, Student-t GP, noisy-input GP, continuous knowledge gradient, Thompson sampling, CUSUM, corruption-tolerant BO), the HITL-BO literature is missing, and `audibert2010best` is never cited. | references.bib, Section 2 | Add about twelve references (list in `readiness_review.md` Section 2.2) | R | open |
| C1, D7 | MAJOR | Appendix P and three multi-objective sentences cite numbers no script produces (the freeze ceiling of 82% and 54%, "5.2 of 10.9", recall and shortfall). | main.tex:1106-1110, 1563-1596 | Cut them or add the scripts | V | open |
| C5, C6, C7, D9, I15 | MAJOR | Table 8's row set, the Appendix C ratio interval, the elicitation T = 20, the scalar Kendall's W of 0.52 and the noise-anchor inputs have no recorded producer, so "scripted end to end" is false. | main.tex:658-661 and the cited passages | Record commands in `paper/COMMANDS.md` and add missing scripts | V, R | open |
| B1 | MAJOR | The instrument arm discretises only noisy post-onset ratings, so its onset-21 row measures a scale that switches on mid-run. | bo_sensor_error_simulation.py:2793-2869, Table 16 | State it and drop the onset-21 row | V | open |
| B2 | MAJOR | The Student-t arm also switches the kernel to Matérn-5/2. | robust_gp.py:245 | Rerun with RBF in a fresh directory or state the confound | V | open |
| B4 | MAJOR | The floor check prints and continues, Table 4's floor excess cell is a literal, and the paper says "exactly zero". | analyse_boba_robustness.py:266-277, make_boba_paper_tables.py:186 | Raise on failure, read the CSV, write "zero to 1e-9, observed 0.000e+00" | V | fixed 2026-09-17 |
| E14 | MINOR | Several intervals that include zero are worded as effects. | main.tex:388-389, 503-504, 600-607, Table 3 | Reword | R | fixed 2026-09-17 |
| E16 | MINOR | Invariances that hold by construction are presented as findings. | main.tex:172-175, 1543-1553 | Say "by construction" | V | fixed 2026-09-17 |
| E7 | MINOR | Factors are ranked by range ratios. The variance shares rank onset (0.17) above landscape (0.10). | abstract, main.tex:306-311 | Lead with the variance shares | V | open |
| E11 | MINOR | Three abstract numbers appear only in appendices, and "near-zero price" has no table. | abstract | Move them or drop them | V | open |
| E13 | MINOR | There is no data or ethics statement for the three archival human-rating datasets. | paper | Add one paragraph | V | partly 2026-09-17: an Ethics statement in the 2027 template's slot states what the paper does and does not collect; approval and consent terms are a todo for the authors |
| I14 | MINOR | All three archival studies are automated-vehicle interface studies. | Section 4, Appendix C | Scope the claims | V | open |
| I10 | MINOR | BIAS always models a generous rater. | bo_sensor_error_simulation.py:2819-2821, main.tex:228 | State it or add a harsh-rater arm | V | open |
| I20, C8 | MINOR | Section 3.3 omits the optimiser settings, ξ = 0.01 and β = 4. BIAS matches GAUSSIAN at onset 0 in distribution only. | main.tex:241-255, 313 | Add the settings and reword | V | open |
| B5 | MINOR | The within-sitting noise model is an assumption that is optimistic for AR(1) and drift at large k. | replay_end_of_study.py:290-314 | State it and split the k-curve by process | V | open |
| B6 | MINOR | "Cautious rule" means posterior mean minus two SDs in Section 5 and minus one SD in the replays. | replay_end_of_study.py:113, budget_split.py:97 | Rename one of them | V | open |
| B7 | MINOR | Section 5's ship-rule shares pool processes and magnitudes and divide by selection loss. Appendix K reports 1σ and 5σ cells and divides by deployed cost. | decompose_regret.py:246-248, main.tex:393-396, 1397-1400 | State both scopes | V | open |
| B8 | MINOR | Appendix E's 4.7e-16 checks the scaling identity on a sampled front. "No run beats the published maximum" is asserted nowhere, and the MO headroom screen is 0.15. | boba_multiobjective.py:196-228, analyse_boba_mo.py:60, main.tex:1045-1060 | Reword | V | open |
| B9 | MINOR | The known-noise clean twin uses `train_Yvar` 1e-6, and "3% or less at 1σ and above" holds only for the pool over onsets. | bo_sensor_error_simulation.py:2094-2132, tables/known_noise.tex | One sentence each | R | open |
| B10 | MINOR | The confirmatory script tests H3a over conditions, applies a BH correction that is a no-op, fits H1 without onset indicators, and the docs give 7.3e-5 where the paper gives 7.2e-5. | scripts/test_confirmatory_hypotheses.py | Disclose in Appendix G | V | open |
| C9, F8 | MINOR | The post-onset window includes intervals that are zero by construction, which deflates onset ratios unequally, and the MO onset ratio weights problems by 1/gap. | evaluate_research_question.py:257-259, analyse_boba_mo.py:298-300 | Disclose or trim the window | V, R | open |
| D6 | MINOR | The MO "absolute loss" column is an excess. | analyse_boba_mo.py:140, make_boba_paper_tables.py:439-446 | Relabel | V | open |
| D10 | MINOR | The halo row of Table 25 comes from four MO problems, one acquisition, one magnitude and seeds 7 to 16, with a four-cluster bootstrap. | run_boba_budget_neutral.ps1:111-114, 173 | State the design | V, D | open |
| C4 | MINOR | Table 21 prints VIFs from a seven-column design beside coefficients from a five-column fit. | analyse_boba_robustness.py:743-744 | Recompute on the primary design | V | open |
| E15, I11, I22 | MINOR | Twenty landscapes share their random numbers yet are bootstrapped as independent clusters, BH runs within small families, and one star rests on three clusters. | analyses, Table 26 fitted row | State cluster counts and the shared stream, drop the star | V, R | open |
| F1, F2, F3, I13 | MINOR | The fitted arm's σf and optimum come from seed-specific oracles over an extrapolated box, its bootstrap has three clusters, and its R² figures mix protocols. | analyse_fitted_companion.py:79-88, anchor_noise_scale.py:80, main.tex:107, 833-836 | Disclose | V, R | open |
| C2, C3, E17, I12, I21, I23 | MINOR | Counting errors and internal contradictions (8, 6 or 9 extra trials, arm counts, "adopted unchanged", two intervals for one cell, observation count, instrument spans). | see `history/` | Prose edits | V | open |
| C15 | BLOCKER | The draft uses the 2026 style and header, spills onto page 10, and carries four `\todo` markers. | main.tex:1-15, 33, 319, 675, 686 | Switch style, cut, resolve | V | partly 2026-09-17: style, header and check_paper.py switched to the official 2027 kit (only the running header differs, .bst identical); page overflow now ~0.5 page (main text ends 47% down page 10) after cutting Setup 116->~80 lines, moving Controls to the appendix and trimming Discussion; a probe shows moving the error equations to the appendix would buy only 12% of a page, so the rest must come from other sections; five author todos open |
| review | MAJOR | The abstract has 524 words and the paper has no figure. | main.tex | See `readiness_review.md` Section 4 | V | open |
| E12 | MAJOR | The AI use statement lacks a negative list and a verification method and cites the 2026 policy. | main.tex:679-686 | The authors write it and list only checks that were performed | V | partly 2026-09-17: heading, required/not-used/recommended structure, verification sentence and 2027 policy reference now follow the template; the negative list is a todo only the authors can fill |

### Note on A1/I1, I2 and E4, 2026-09-17

A1/I1. `tables/dose_response.tex` is regenerated with three blocks: the post-onset
per-iteration average of search loss, the deployed design's excess at the final
trial, and the selection share of that excess. Section 3.4 now defines both
responses and says which is primary, and the abstract and Section 4 quote both.
The standalone decomposition table was folded into the same float, so the paper
gained no table. The generator raises if `cell_means.csv` and
`regret_decomposition.csv` disagree on the deployed excess by more than 5e-4;
they currently agree to four decimals in all eight cells, which is an independent
check of the two pipelines.

I2. The decomposition now emits rows pooled over the error processes per
(magnitude, onset), with landscape-bootstrap intervals, and the paper leads with
1 sigma from the first rating: **41.8% [34, 50]**, rising to 83.1% [75, 90] at 5
sigma from trial 21. The 61.8% pool is still reported, with its scope stated (the
5 sigma cells carry 66% of the total excess it divides by, the 1 sigma cells 25%).

Discrepancy to resolve. I2 cites 52% at 0.71 sigma and 56% at 1.41 sigma. Those
magnitudes are not in the main grid; they are the `session25` and `session100`
arms, which run T = 25 and T = 100 and so split the cost differently. On the main
sweep at exactly 1 sigma the share is 41.8%. The paper now quotes the main-sweep
value. If the intended claim was about the session arms, that needs its own
sentence and its own scope.

E4, the other three pools, 2026-09-17. Each now carries its 1 sigma onset-0 value
beside the pool, in the abstract and in Section 8. Known noise removes 5.0% of the
excess-weighted cost and **costs** 7% at 1 sigma, which strengthens the paper's
own conclusion. The incumbent removes 15.6% pooled but 12% at 1 sigma and 65-67%
below 0.25 sigma, so that pool understates the effect at small error and
overstates it at realistic error. The multi-objective claim was a bound ("at most
half") that held at every magnitude and hid the fact that the gap is narrowest
where it matters: a seventh at 0.05 sigma, a half at 1 sigma and 5 sigma; the
main text now gives the pattern instead of the bound. The two ablation appendices
already carried per-magnitude qualifiers and were left as they were, except that
B9's "3% or less at 1 sigma and above" is still the pool over onsets and remains
open.

E3, I4, B4, E14 and E16, 2026-09-17.

E3. The decomposition no longer says an acquisition function cannot reach the
selection term. It says the acquisition addresses search directly and can reach
selection only indirectly, by changing which designs the ratings must rank, and
that none of the seven acquisition-side changes we ran reduced the deployed loss.
"The reason is structural" is now "a likely reason, which we do not test
directly", with the defensible claim stated separately: four rules that only
re-read data in hand recover nothing, and the procedures that buy new ratings do.

I4. Verified against `confirmation_decision` in replay_end_of_study.py:486. The
code ships on the higher MEAN rating; the t-test governs only the claim. Appendix
L now says both, and says they are separate decisions, which is why the procedure
changes little about what ships and much about what may be claimed.

B4. `floor_check` now raises instead of printing to stderr and continuing, and
the acquisitions table reads the floor's excess from `floor_check.csv` instead of
printing a hard-coded 0.00%, refusing to print a zero over a failed control. The
observed maximum over all 12,800 comparisons is 0.000e+00. Section 3.5 states
the mechanism rather than only the value.
`tests/test_error_model_labels.py` pinned the old print-and-continue behaviour
and now pins the raise.

E14. "Gains +0.001 [-0.018, +0.018]" and "+0.004 [-0.013, +0.021]" are now
"indistinguishable from" their references. The intervals in Table 25 are left for
E2/I3/I7, which rebuilds that table.

E16. The comparison loop's immunity to a shared strictly monotone fault is
labelled a consequence of the construction, with the run described as confirming
the implementation rather than the claim; the bias arm is labelled the same way.
The related-work sentence no longer calls the invariance a measurement.

C15, E12, E13 and a BibTeX error, 2026-09-17.

The official ICLR 2027 kit was fetched from ICLR/Master-Template and diffed
against 2026: the .sty differs only in the two running-header strings, the .bst
is byte-identical, and the paper's natbib.sty, fancyhdr.sty and math_commands.tex
were the unmodified 2026 copies (line endings only). main.tex, paper/README.md
and check_paper.py (now year-agnostic) point at 2027. The 2027 instructions
confirm a STRICT 9 pages of main text at submission, 10 at rebuttal; the main
text ends about a tenth of the way into page 11.

The AI use statement was retitled and restructured to the 2027 template: tasks
used, tasks not used (a todo -- only the authors know), recommended-disclosure
tasks, and a verification sentence that names what was actually done (697 tests,
cross-pipeline checks, three audit passes). An Ethics statement was added in the
template's slot. Both sit outside the page count.

BibTeX exited 2 on every build: the provenance comment at the top of
references.bib contained the literal text "@article", and BibTeX has no comment
syntax between entries, so it tried to parse the comment as an entry. No entry
was lost (24 cited, 24 in the .bbl) and no citation was undefined, but Overleaf
would have shown a red error on every compile. Reworded.

A ready-to-upload zip is built by paper/build_overleaf_bundle.py; it contains
only what a submission needs and a README_OVERLEAF.md listing the six open todos
and the page overflow. The bundle is test-compiled from a clean directory before
it is written.

## 2. Code, tests and reproducibility

| ID | severity | problem | where | fix | basis | status |
|---|---|---|---|---|---|---|
| D2 | MAJOR | pytest aborts at collection when jax and numpyro are absent, so the suite runs nothing and CI should fail. | tests/test_hierarchical_oracle.py:38 | Module-level `pytest.importorskip("numpyro")` | X | fixed 2026-09-17 |
| D1 | MAJOR | The BOBA parity and provenance tests skip on every machine but the author's. | tests/test_boba_benchmarks.py:36 | Vendor the reference outputs into `tests/fixtures` | V | open |
| C10 | MINOR | The budget rule seeds its Monte Carlo from Python's salted `hash()`. | budget_split.py:223 | Use `zlib.crc32` | V | fixed 2026-09-17 |
| C11 | MINOR | `decompose_regret` does not filter variants and drops unpaired runs silently, pooled adaptation references lack a completeness check, and a missing opt_z falls back to 1.0. | decompose_regret.py:225-226, analyse_boba_adaptations.py:250-256 | Add the checks | R | open |
| C12 | MINOR | The pilot frag is correlated with the opt_z-normalised cost, while Section 7 uses raw excess. | analyse_pilot_frag.py:240-249 | Say so in Appendix R | R | open |
| J12 | MINOR | The paired table carries only the noisy run's fallback count, and global warning filters hide optimiser warnings. | evaluate_research_question.py:237-239, 316, bo_sensor_error_simulation.py:202-203 | Carry both counts | R | open |
| F4 | MINOR | Some tests pin approximations as definitions, and some are vacuous. | test_end_of_study.py:84-90, test_boba_benchmarks.py:700-704, test_ship_rules.py:218-221, test_elicitation.py:126-128, test_scientific_correctness.py:343-353, test_variant_evaluation.py:74-82 | Rewrite them | V | open |
| F5 | MINOR | The code behind Tables 1, 6, 19, 20, 21, 25 and 26 and the fitted arm has no tests. | see `history/second_pass_findings.md` F5 | Add tests | R | open |
| F6 | MINOR | The robustness synthesis also runs on the input-error directories and writes meaningless mediator files there. | run_boba_inputerror_finish.ps1:121 | Skip those directories | R | open |
| F7 | MINOR | `boba_benchmarks.main` overwrites the stats file with the requested subset, two comments are stale, and `steering_law` is labelled interior. | boba_benchmarks.py:519-523, 605, 1038-1056 | Merge instead of overwrite, fix comments | R | open |
| F9 | MINOR | Bootstrap generators are seeded differently across scripts. | replay_mo_front.py:860, analyse_boba_mo.py:146 | Align | R | open |
| I16 | MINOR | The requirement files contradict each other, `pyproject.toml` requires Python 3.13 while the results used 3.12.9, and CI tests other versions without jax or numpyro. | requirements.txt, requirements-eval.txt, pyproject.toml:5-6, .github/workflows/tests.yml | Freeze the environment of the simulation machine | V | open |
| I18, I19 | MINOR | Resumed runs carry no code stamp, run metadata reflects the last invocation, and the preregistration has no external timestamp. | bo_sensor_error_simulation.py:4072-4100, output-boba-confirmatory/HYPOTHESIS.md | Stamp runs, cite a dated commit | V, R | open |
| C13 | MINOR | Stale values remain in `docs/known-function-arm-2026-09-06.md`, `scripts/decompose_regret.py:41`, `tests/test_currency.py:18` and `scripts/analyse_fitted_companion.py:23`. | those files | Update | V | partly 2026-09-17: decompose_regret.py and test_currency.py corrected to the real 74x span; docs/known-function-arm-2026-09-06.md is a dated record and analyse_fitted_companion.py's 0.37 vs the paper's 0.29 held-out R2 needs the authors to say which is right |
| check | MINOR | `check_paper.py` hard-codes the 2026 style name. | check_paper.py:38 | Update with the style switch | V | open |

## 3. Anonymity and submission hygiene

| ID | severity | problem | where | fix | basis | status |
|---|---|---|---|---|---|---|
| C14, D3, I17 | BLOCKER for the supplementary | Identifiers appear in `LICENSE`, four test files, two docs files, `datasets*.json`, two more test files with account URLs, every `run_metadata.json`, and `.git`. Third-party ASHRAE and NISQA data sit in `sandbox/` and `external_datasets/`. | see the checklist in `README.md` | Build the supplementary from a scrubbed export | V | open |
| readme | MINOR | README told readers to quote dataset commit hashes in the paper. | README.md | Limited to the camera-ready version | V | fixed 2026-09-16 |
| readme | MINOR | README recommended `run_full_workflow.bat` in one section and warned against it in another. | README.md | Section rewritten to point at the paper's pipeline | V | fixed 2026-09-16 |
| agents | MINOR | AGENTS.md held stale values, overclaims and the Windows username. | AGENTS.md | Rewritten | V | fixed 2026-09-16 |

## 4. Retracted or corrected during the audit

These statements from earlier passes were wrong and have been corrected in the history files. They are listed so that nobody re-raises them. E3's arithmetic linking augmented EI's trajectory and deployed recoveries was invalid, because the two metrics have different time bases. E11's claim that Table 23's "evaluated" column is an oracle was wrong, because ratings are exact in the input-error arms. E7 mislabelled the variance shares. B5 called the sitting model wrong, and it is an assumption. D5 attributed memorisation to the opticarvis oracle, which is gradient boosting. D2 counted 14 collection errors, and pytest in fact aborts the whole run. D10 counted 256 distinct resamples, and there are 35. B9 and F4 said σ² is not the injected variance, and σ² is the marginal variance of both processes. C13 said README was stale, and it was already correct. D3 said no dataset commit is recorded, and the simulator records it. Section G listed the elicitation loop's disadvantage under independent noise as surviving, and its interval includes zero. The first readiness review had every weekday of the day plan shifted by one, overstated the compute cost by a factor of about three, proposed lowering the preregistered headroom screen, and cited an unverified reference. The full table is Section J of `history/third_pass_findings.md`.
