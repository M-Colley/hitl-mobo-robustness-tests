# ICLR 2027 readiness review: "What Noisy Feedback Costs Bayesian Optimization, and the One Number That Predicts It"

Reviewed 2026-09-16 against the repository `hitl-mobo-robustness-tests` (paper/main.pdf, 28 pages, compiled 2026-09-16), README, AGENTS.md, docs/known-function-arm-2026-09-06.md, the preregistration in output-boba-confirmatory/HYPOTHESIS.md, the generated tables, and the ICLR 2027 Call for Papers and Author Guidelines.

## Verdict

Recommendation as an ICLR reviewer today. Reject. The measurement is careful and the controls are unusually good. The paper as written would not survive the ICLR reviewer pool for three reasons that have nothing to do with the data. It has no figures. The title and abstract oversell one predictor that the paper's own analysis demotes. The paper reads as a report of thirty experiments rather than as an argument with one claim. All three are fixable before 25 September. The decisive risk is not the science. It is presentation and formal compliance.

Hard deadlines. Abstract with the final author list is due Friday 18 September 2026, 23:59 AoE. No author can be added or removed after that. Full paper and supplementary material are due 25 September 2026, 23:59 AoE. Reviews come 5 November. Author discussion runs 5 to 18 November. Decisions on 16 December.

## 1. What the project is

The repository simulates Bayesian optimization driven by a noisy human rater. The human is replaced by an exact analytic objective. Twenty benchmark landscapes with verified optima are affinely standardised to zero mean and unit variance over their box. Four error processes (Gaussian, bias, drift, AR(1)) are injected at four magnitudes (0.05 to 5 landscape SD) and two onsets (trial 1 and trial 21). Twelve acquisition functions run for 50 trials at ten seeds. Every noisy run is paired with an identically seeded clean twin. The primary response is post-onset excess simple regret, divided by opt_z (the standardised height of the optimum over a random design) for reporting. The main sweep is 79,200 runs. Around thirty side arms with five seeds and two acquisitions test remedies, input errors, budget, rating-scale discretisation, multi-objective transfer, and a fitted-oracle companion on three archival HCI datasets.

The story the paper tells is in five steps. Error costs 14% of the achievable improvement at 1 SD from the first rating and 2.8% when it arrives at trial 21. Onset and landscape move the cost far more than the kind of error does. Deployed regret splits into search loss and selection loss, and selection loss is 61.8% of what the error costs. Because selection loss is zero under exact observation and is untouched by the acquisition function, acquisition-side remedies recover nothing on the deployed design. Spending a third of the budget on a comparative sitting, or modelling the fault explicitly, recovers 22 to 38%.

## 2. Contribution assessment

### 2.1 What is strong

The search versus selection decomposition (Section 5, Equation 5) is the real contribution. It is simple, exact, and it explains a pattern that the field has treated as puzzling (noise-aware acquisitions do not help much under human noise). It gives a testable prediction, and the paper tests it with seven acquisition-side changes that all fail on the deployed design while augmented EI recovers 41% of the trajectory cost. That is a clean mechanism result. It should be the title and the first sentence of the abstract.

The estimand discipline is better than most of the BO literature. Scoring process changes against the standard process rather than against their own clean twin (Appendix Q) is correct and the paper shows that the wrong estimand flips the sign of the replication result. The model-free floor as a machine-precision negative control, the uniform-offset control, the headroom screen, the seed-balance check, the UCB and qNEI zero-movement control on the incumbent switch, and the landscape bootstrap all belong in a methods paper. The exact replay of end-of-study procedures from logged prefixes is elegant and cheap.

The negative results are valuable and honest. Known noise variance removes 5% of the cost. Early replication makes things worse. Thompson sampling loses 58%. Preference elicitation is exactly immune to drift and useless against saturation. The CUSUM already matches the GLR optimum. A void preregistered replication is reported as void. This is the open science posture the CfP asks for. Reviewers at ICLR reward this less than HCI reviewers do, so the paper must lead with the positive mechanism and keep the negatives as support.

### 2.2 What is weak, in order of damage

The title claim is not supported at the strength the title states. frag(σe) explains R² = 0.52 with onset and process indicators, 0.48 alone. The descriptor model with a magnitude by opt_z interaction and a quadratic in log σe reaches 0.78 without frag. Adding frag gives 0.79. Preregistered H1b (magnitude adds nothing given frag) holds under the additive specification only and the paper says so in Appendix G. "The one number that predicts it" is therefore a claim the paper's own Table 6 undercuts. An ICLR reviewer will open Table 6, see 0.52 against 0.78, and write "overclaim" in the summary. The honest framing is that a ten-trial pilot computation with no optimizer in the loop predicts two thirds of what the best post-hoc descriptor model explains, and the pilot version works as well as the exact one (Spearman 0.94 with exact frag, 0.85 with cost). That is a good result. It is not a one-number law.

There is no figure in 28 pages. Not one. The dose-response by onset, the search versus selection split by magnitude, the inverted U in identification budget k, the frag against measured cost scatter, and the recovery forest plot of Table 25 all beg to be drawn. The analysis pipeline already emits onset_contrast.png, fragility_by_benchmark.png and fragility_currency_collapse.png in every arm's analysis folder, so the raw material exists. A results paper with zero figures at ICLR signals to reviewers that the authors did not look at their data. It also makes the paper far harder to read than it needs to be.

The abstract is 524 words and lists roughly twenty numbers. It reads as a table of contents. An ICLR abstract is 150 to 250 words with one claim and three supporting numbers. This abstract also needs to be submitted by Friday, and the final abstract must "stay close" to it, so the rewrite must happen before the abstract deadline, not after.

The paper has too many results and no hierarchy. Sections 4 to 11 are eight results sections in nine pages, each with its own bolded sub-claims, plus nineteen appendices (A to S) with 27 tables. Every ablation is treated as equally important. Reviewers will not find the mechanism claim because it sits at Section 5 behind a dose-response table, a power-law scale test, and a descriptor regression. The main text should carry one argument. Cost (Section 4, short). Decomposition (Section 5, the centre). Why acquisitions cannot help and what does (Sections 8 and 9 merged). Robustness checks in one compact paragraph pointing to the appendix. Sections 6 and 7 (SPREAD versus GAIN, frag predictor) are supporting evidence and can be compressed to half a page each or moved.

The novelty question will be asked and the paper does not preempt it. ICLR reviewers will say "this is a benchmark study with standard BoTorch components and no new method." The defence is that the decomposition is new as a reporting practice, that the field's default metric (regret at the best observed point) provably hides the dominant loss term under noise, and that the recommended fix (a comparative sitting, confirmation trials) is a protocol contribution with measured effect sizes. The paper needs a paragraph in the introduction that says exactly this, and a paragraph in related work that names what is missing from prior noisy-BO benchmarks (Picheny et al. 2013 report regret at the best observed point and never separate the two terms).

The related work is thin for ICLR. Twenty-five references. Missing and likely to be raised by reviewers. Bogunovic et al. 2020, corruption-tolerant GP bandit optimization (adversarially corrupted observations, directly relevant to slips and spikes). Martinez-Cantin et al. 2018, practical BO in the presence of outliers (Student-t GP, the paper runs this but does not cite the origin). Fröhlich et al. 2020, noisy-input entropy search (the NIGP arm). Kirschner and Krause 2018 on heteroscedastic bandits. Oliveira et al. 2019, BO under uncertain inputs. Letham and Bakshy 2019 on Bayesian optimization for policy search via online-offline experimentation. The HITL BO literature the paper claims to speak to. Khajah et al. 2016 (CHI), Koyama et al. 2017 and 2020 (sequential line search, sequential gallery), Kadner et al. 2021 (AdaptiFont), Chan et al. 2022 (positive and negative qualities of HITL optimization), Liao et al. 2023 (multi-objective HITL BO). The RLHF preference-noise literature is the bridge to the ICLR audience and it is absent. Label noise in preference data (Wang et al. 2024 on robust DPO, Chowdhury et al. 2024 on noisy preferences, the Christiano et al. 2017 base). Two sentences connecting "noisy human rater in BO" to "noisy human labeller in preference learning" would tell the ICLR reader why this paper is at ICLR.

### 2.3 Empirical risks a careful reviewer will probe

Posterior-mean shipping does worse than best-observed rating by 8.7% (Section 5). Under Gaussian noise with a GP whose fitted noise is 0.83 to 0.89 of the injected value at n = 50, shipping the posterior-mean maximiser over visited designs should beat the noisy argmax. That it loses is surprising. Either the refitted rescoring GP differs from the loop GP in a way that matters, or the posterior-mean rule is choosing points in flat regions where the mean is inflated. Check this before submission. Run the rescoring at σe = 0.05 and confirm the posterior-mean rule converges to the best-observed rule. Run it at 5σ and confirm the direction. If posterior mean still loses at 5σ, explain why in the text, because a reviewer will assume a bug.

The remedies rest on five seeds and two acquisitions (LogEI, qNEI). The headline "acquisition changes recover nothing on the deployed design" is generalised from LogEI and qNEI to all acquisitions. State the scope in the sentence, not only in the appendix. The intervals in Table 25 are wide (augmented EI +2% [−6, +11]) and the "recover nothing" wording should be "no acquisition-side change is distinguishable from zero on the deployed design."

The scale test leans on one landscape. Without shekel (opt_z 56.8) βz moves to 1.04 to 1.10 and the GAIN gap to 0.57 to 0.73. Appendix D.3 then shows opt_z is not causal. Section 6 is therefore a section whose conclusion ("neither scale fits") is weakened by its own sensitivity analysis and then undercut by the next appendix. Compress Section 6 to three sentences and a pointer, and let D.3 carry the mechanism.

The preregistered replication is void. This is reported with exemplary honesty. It still means the paper has zero confirmed preregistered claims. Compute permitting, rerun the confirmatory arm with the headroom threshold set at 0.05 and Rosenbrock excluded a priori, on seeds 37 to 46. The design note says 19.7 hours on 24 workers for the full sweep. The Gaussian-only confirmatory arm is a quarter of that. It fits before 25 September if the Windows machine is free this weekend. A confirmed replication converts the weakest paragraph of the paper into one of its strongest.

The rater-noise anchor (0.74 to 3.35 σf) passes through fitted oracles the paper otherwise rejects. The paper says so. Reviewers will still ask whether 1σ is realistic. One extra sentence with within-rater test-retest reliabilities from published rating-scale psychometrics (ICC around 0.7 implies noise near 0.65 σ of the trait) would anchor the claim without the fitted oracle.

The suite "was assembled for a separate study." If BOBA is public, cite it in the third person. If it is the authors' own unpublished project, say "a benchmark suite under separate submission" and nothing more. An unexplained suite of twenty functions with six human-performance laws will draw a question about provenance and about anonymity.

### 2.4 Things I checked and found consistent

Every number I traced from the main text to the generated tables in paper/tables and to docs/known-function-arm-2026-09-06.md matched (dose response 14.1/2.8, onset ratios 2.3/5.1/11.7, incumbent 15.6% and the UCB and qNEI zeros, known noise 5.0%, adaptations table, input error 5.4/11.5/23.2, confirmatory 0.65/0.69, pilot frag 0.94/0.85). The 76,800 pairs and 64,000 model-based noisy runs are arithmetically consistent with 20 × 12 × 10 × 32 and 20 × 10 × 10 × 32. The 0.000e+00 floor control and the 10⁻⁹ replay assertions are described as tested in the code and the test suite has 443 test functions across 29 files. The README's opt_z range (1.25 to 56.8, 45×) is stale against the paper (0.76 to 56.8, 74×). Fix the README before it ships as supplementary.

## 3. Formal compliance with ICLR 2027

Style file. The draft uses iclr2026_conference.sty and the running header reads "Under review as a conference paper at ICLR 2026." Download https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip and switch. A 2026 header on a 2027 submission looks like a recycled rejection and invites a format desk-reject.

Page limit. ICLR 2027 allows 9 pages of main text at submission. Ethics, reproducibility, AI use statement, acknowledgements and references do not count. The Limitations paragraph currently spills two lines onto page 10. Papers over the limit are desk-rejected. The compression in Section 4 below brings the main text to about 8 pages, which also makes room for two figures.

AI use statement. ICLR 2027 makes an "AI use statement" mandatory in the paper and in the submission form. The current section is titled "LLM usage" and says an LLM "proposed and implemented parts of the experimental design and analysis, wrote and tested most of the simulation and analysis code, ran and monitored the experiments, drafted and edited the manuscript." The policy requires disclosure of exactly these uses and adds a template ending with "We have reviewed all AI-assisted work" plus how verification was done. Rename the section, follow the template, and state the verification concretely (the 443 tests, the parity check against source implementations, the machine-precision controls, human review of every number against the analysis CSVs). Given how extensive the AI role was, an honest and specific statement here is the safer path, and the policy explicitly permits this scope of use. The TODO on this section must be resolved by the authors, not by the model.

Reciprocal reviewing. Each submission needs at least one author registered to review three papers, and a qualified reviewer needs one accepted paper at ICLR, NeurIPS, ICML, UAI, AISTATS, JMLR, TMLR, ACL, EMNLP, EACL, CVPR, ICCV, ECCV, AAAI, IJCAI, JAIR, ICRA, IROS, RSS, CoRL, KDD or COLT. CHI is not on the list. If no author qualifies the paper is exempt, but each author may then be on at most one such paper. Check the author list against this before Friday.

Anonymity. The paper text is clean. The supplementary is not. run_metadata.json in every output arm contains "C:\Users\markc\Desktop\hitl-mobo-robustness-tests". AGENTS.md names the interpreter path with the same username. The .git history contains a merge from "M-Colley/claude/...". Any of these in the supplementary zip is a desk reject. Build the supplementary from a fresh export, strip .git, scrub the username from every json and md, and check with grep before zipping. The three archival datasets (ehmi, opticarvis, provoice) must be cited in the third person with the TODO in Section 4 resolved.

Reproducibility. The main sweep directory output-boba is gitignored and is not in the folder. The paper's headline tables come from output-boba/analysis. The supplementary should include at least the analysis CSVs that make_boba_paper_tables.py consumes, the landscape stats JSON, and the run_metadata for the main sweep, so a reviewer can regenerate every table without 20 hours of compute. The Reproducibility statement should say this.

OpenReview. All authors need profiles before 25 September. Non-institutional emails take about two weeks to moderate, so anyone without a profile must create one now with an institutional address.

Topics. The CfP lists optimization, probabilistic methods, and applications to human-facing systems. Fit is acceptable if the introduction frames the paper as a study of learning from corrupted human feedback rather than as an HCI simulation.

## 4. Recommended restructuring of the main text

Title. "Most of what noisy feedback costs Bayesian optimization is selection, not search." The current title should go.

Abstract, 150 words. Sentence one states the decomposition. Sentence two gives the 61.8% share and the exact-observation zero. Sentence three gives the cost at 1σ and the onset ratio. Sentence four states that seven acquisition-side changes recover nothing on the deployed design while a comparative sitting over a third of the budget recovers 0.051 and confirmation trials cut false claims from 15.7% to 0.5%. Sentence five names the twenty exact landscapes and the zero-excess control. Sentence six states that a ten-trial pilot quantity predicts the cost at Spearman 0.85.

Introduction, 1 page. Problem, the confound in the fitted-oracle design, the decomposition as the lens, four contributions. Contribution list ordered by importance. Decomposition first, remedies second, measurement protocol third, predictor fourth. Drop the input-error arm and the fitted-oracle companion from the contribution list and cite them as appendices.

Related work, 0.75 page. Add the missing noisy and corrupted BO work, the HITL BO work, and two sentences on preference-noise in RLHF.

Setup, 1.25 pages. Unchanged in content. Figure 1 here. A single panel showing one landscape's clean and noisy trajectories with the visited-best and shipped design marked, which makes selection loss visible before it is defined.

Cost, 0.75 page. Table 1 and Figure 2 (dose response by onset, with landscape spread as a band). Drop Table 2 to the appendix and keep one sentence on extra trials.

Decomposition, 1.5 pages. Equation 5, the 61.8% result, the exact-observation zero, the failure of re-reading rules. Figure 3 shows the split by magnitude and onset as stacked bars. This is the centre of the paper and should get the space.

What helps and what does not, 1.75 pages. Merge Sections 8 and 9. Table 4 stays (acquisition ranking) with one sentence on the incumbent confound. Figure 4 is a forest plot of Table 25 with the acquisition-side arms above a rule and the fault-model arms below it. The inverted U in k gets one line in the text. Ship-rule and budget-rule findings get one paragraph each.

Predicting the cost, 0.5 page. frag definition, the R² numbers stated at their true strength, the pilot result.

Checks, 0.5 page. Six one-sentence robustness statements pointing to appendices, as Section 11 does now but at half the length.

Discussion and Limitations, 0.75 page. Keep the two unexplained results. Add the scope statement on remedies (five seeds, two acquisitions). Add one sentence on what a study with real raters would need to measure to test the decomposition.

That is roughly 8.75 pages with four figures. Everything cut moves to the appendix, which has no limit.

## 5. Action plan to 25 September

Wednesday 17 to Thursday 18. Fix the author list and check reciprocal-reviewing eligibility. Rewrite the abstract and title as above. Register the abstract on OpenReview by Friday 23:59 AoE with the final author order. Create any missing OpenReview profiles with institutional emails.

Thursday 18 to Saturday 20. Switch to the 2027 style files. Restructure the main text per Section 4. Produce four figures from the existing analysis CSVs using the pipeline's plotting code. Resolve the three TODOs (third-person citations for the three datasets, the repository URL as an anonymised placeholder, the AI use statement). Extend related work by twelve to fifteen references.

Saturday 20 to Monday 22, compute permitting. Launch the confirmatory rerun on seeds 37 to 46 with the corrected headroom rule. Run the posterior-mean ship-rule sanity check at 0.05σ and 5σ.

Monday 22 to Wednesday 24. Fold in the confirmatory result if it finishes. Recompile and verify main text ends on or before page 9. Build the supplementary from a clean export, scrub usernames and .git, include the analysis CSVs for the main sweep, run scripts/check_paper.py and the test suite. Have a co-author read the paper cold and mark every place where the argument is not followable without the appendix.

Thursday 25. Submit by 23:59 AoE with a margin of hours, not minutes.

## 6. One unconventional option

Submit the decomposition paper to ICLR and split the remedies into a second paper. Sections 9, 10 and Appendices K to Q are a self-contained protocol paper (comparative sitting, confirmation trials, change-point freeze, budget rule) with its own estimand and its own negative results. Packed into one ICLR submission they dilute the mechanism claim and they cannot be given the space their intervals need. A CHI 2027 or TOCHI paper on "how to end a human-in-the-loop optimization study" would reach the practitioners who ship designs, and it could carry the fitted-oracle companion and the rater-noise anchor, which are HCI content. The ICLR paper would then be one claim, one decomposition, one predictor, and twenty landscapes. That is a paper reviewers can hold in their heads.

## Sources

ICLR 2027 Call for Papers, https://iclr.cc/Conferences/2027/CallForPapers. ICLR 2027 Author Guidelines, https://iclr.cc/Conferences/2027/AuthorGuidelines. ICLR 2027 AI Policy for Authors, https://iclr.cc/Conferences/2027/AIPolicyForAuthors.
