# ICLR 2027 readiness review

Paper "What Noisy Feedback Costs Bayesian Optimization, and the One Number That Predicts It", repository `hitl-mobo-robustness-tests`. First written 2026-09-16 and revised the same day after three audit passes. For anything about what the code computes, `findings.md` in this folder takes precedence, and the three pass reports in `history/` hold the evidence. `README.md` in this folder is the entry point.

## Verdict

Recommendation as an ICLR reviewer today is reject. The decisive flaw is that the headline number measures the search trajectory while the text presents it as the cost of the design a study ships. On the real outputs the shipped design loses two to three times as much (third pass I1). The paper also has no figures. Its title promises a single predictor that its own Table 6 ranks below a descriptor model. Its thirty experiments carry no hierarchy. Several remedy results rest on estimands that cannot support the stated conclusion (third pass I3, I6, I7). The measurement itself is careful, and the controls hold on the data that could be checked. Everything except the optional reruns can be fixed before 25 September.

Deadlines. The abstract and the final author list are due Friday 18 September 2026, 23:59 AoE. No author can be added or removed afterwards. The full paper and supplementary material are due Friday 25 September 2026, 23:59 AoE. Reviews arrive 5 November, the discussion runs 5 to 18 November, and decisions come on 16 December.

## 1. What the project is

The repository simulates Bayesian optimization driven by a noisy human rater. The rater is replaced by an exact analytic objective. Twenty benchmark landscapes with verified optima are standardised to zero mean and unit variance over their box. Four error processes (Gaussian, bias, drift, AR(1)) are injected at four magnitudes (0.05 to 5 landscape SDs) and two onsets (trial 1 and trial 21). Twelve acquisition functions run for 50 trials at ten seeds, and every noisy run is paired with an identically seeded clean twin. The main sweep is 79,200 runs. About thirty side arms with five seeds and mostly two acquisitions test remedies, input errors, budget, rating-scale discretisation, multi-objective transfer, and a fitted-oracle companion on three archival automated-vehicle interface studies.

The paper's argument runs in five steps. Error costs 14.1% of the achievable improvement at 1σ from the first rating and 2.8% when it arrives at trial 21. Onset and landscape move the cost more than the kind of error does. Deployed regret splits into search loss and selection loss, and selection is 61.8% of the pooled cost of error. Selection loss is zero under exact observation, and the acquisition function does not target it, so the paper expects acquisition-side remedies to fail on the deployed design. Spending trials on identification or modelling the fault recovers part of the cost.

## 2. Contribution assessment

### 2.1 What is strong

The search versus selection decomposition (Section 5, Equation 5) is the real contribution. It is exact and simple. It explains why noise-aware acquisitions help little under human noise. The field's default report, regret at the best visited design, contains no selection loss at all, and the paper can show that this default misses a large share of what a study ships.

The estimand discipline is better than most of the BO literature. Appendix Q scores process changes against the standard process, and the paper shows that the naive estimand flips the sign of the replication result. The model-free floor, the uniform-offset control, the headroom screen, the seed-balance check and the zero movement of UCB and qNEI under the incumbent switch all belong in a methods paper. On the data available for checking, the floors are bit-identical across paired arms, the landscape statistics reproduce across operating systems to 1.6e-13, and four appendix tables regenerate exactly (third pass H2, H3, H6).

The negative results are useful. Known noise variance removes 5% of the pooled cost. Early replication makes things worse. Thompson sampling loses 58% on the deployed design. Preference elicitation cannot help against saturation. The void preregistered replication is reported as void. ICLR reviewers reward negative results less than HCI reviewers do, so the paper should lead with the decomposition and use the negatives as support.

The acquisition-side evidence needs one repair before it can illustrate the mechanism. The seven kinds of acquisition-side change tested (twelve rows in Table 25) do not improve the deployed design, since no interval lies above zero and seven rows are significantly negative. Augmented EI's 41% recovery of the trajectory cost is measured against the mean of ten standard acquisitions, including the four weakest in Table 4 (third pass I6). Scoring it against EI or LogEI would show whether the contrast between its trajectory and deployed recovery survives.

### 2.2 What is weak, in order of damage

The headline metric is mislabelled (code audit A1, third pass I1). Table 1, Sections 4, 6 and 7, Table 4's "absolute loss" column and Appendix C use the post-onset time average of the best-visited design's regret. The text calls this "the regret of the design a study would deploy", and Table 4's caption calls it "the deployment number". At the final trial the deployed design's excess is 2.45 to 2.64 times the search excess for the standard process, and the paper's own pooled numbers give 0.218 against 0.083. The abstract's 14.1% therefore understates what a study ships by a factor of about two to three. This has to be fixed before any restructuring, because the restructured paper leads with the two metrics.

The title overclaims. frag(σe) explains R² 0.52 of the cell-mean variation with onset and process indicators, and 0.48 alone. The descriptor model with a magnitude by opt_z interaction reaches 0.78 without frag, and adding frag gives 0.79. The comparison uses exact frag. The practitioner version is a GP fitted to the first ten trials of a clean LogEI run, which tracks exact frag at Spearman 0.94 and correlates 0.85 with cost when pooled over magnitudes but only 0.32 within 1σ. Preregistered H1b held only under the additive specification, and the interaction model behind that statement formed its product term uncentred (code audit A3), so the refit must come first. A fair summary is that a cheap pilot quantity ranks the pooled cost well, and it is weak within one magnitude.

There is no figure in 28 pages. The dose-response by onset, the search and selection split by magnitude, the identification-budget curve, frag against measured cost, and the remedy intervals all call for plots. The analysis pipeline already writes `onset_contrast.png`, `fragility_by_benchmark.png` and `fragility_currency_collapse.png` for every arm. A results paper without figures tells ICLR reviewers the authors did not look at their data.

The abstract has 524 words and about twenty numbers. An ICLR abstract carries one claim and three supporting numbers in 150 to 250 words. The final abstract must stay close to the one registered on 18 September, so the rewrite has to happen before that date.

The paper has no hierarchy. Sections 4 to 11 are eight results sections in nine pages, backed by nineteen appendices and 27 tables. The mechanism sits in Section 5, after the dose-response section. Sections 6 and 7 (the power-law scale test and the frag regression) follow it and read as competing headlines. The main text should carry one argument.

The novelty question is not pre-empted. Reviewers will say the paper uses standard BoTorch components and proposes no method. The defence is that the default reporting metric provably omits the selection term, that the omission is large, and that the paper supplies a protocol and measured effect sizes for end-of-study procedures. The introduction needs a paragraph that says this, and related work needs a sentence on which noisy-BO benchmarks report only best-visited regret, checked against the cited papers.

Related work is thin for ICLR, with 24 references in the compiled paper. The arms the paper runs need their sources cited. These are relevance pursuit (Ament et al., NeurIPS 2024), the Student-t GP for outliers (Martinez-Cantin et al., AISTATS 2018, and Jylänki et al., JMLR 2011), the noisy-input GP (McHutchon and Rasmussen, NeurIPS 2011), continuous knowledge gradient (Wu and Frazier, NeurIPS 2016), Thompson sampling, and CUSUM (Page, Biometrika 1954). The corruption setting needs Bogunovic et al. (AISTATS 2020). Related robust-BO work includes Oliveira et al. (AISTATS 2019), Fröhlich et al. (AISTATS 2020) and Kirschner and Krause (COLT 2018). The human-in-the-loop BO literature in HCI includes Khajah et al. (CHI 2016), Koyama et al. (SIGGRAPH 2017 and 2020), Dudley et al. (CHI 2019), Kadner et al. (CHI 2021), Chan et al. (CHI 2022) and Liao et al. (IEEE Pervasive Computing 2023). The bridge to the ICLR audience is noisy preference data, with Christiano et al. (NeurIPS 2017), Lee et al. (B-Pref, NeurIPS Datasets and Benchmarks 2021, which simulates noisy and irrational teachers) and Chowdhury, Kini and Natarajan (Provably Robust DPO, ICML 2024). The "standard construction" of fitted regression oracles that the introduction criticises also needs third-party citations (second pass E1).

### 2.3 Empirical risks a careful reviewer will probe

Selection is about half of the deployed cost at realistic magnitudes (third pass I2). For LogEI and qNEI with error from trial 1, the share is 52% at 0.71σ and 56% at 1.41σ. The pooled 61.8% is excess-weighted and pulled up by late onsets and 5σ cells. Read the 1σ, onset-0 cell from `regret_decomposition.csv` before choosing a title.

Posterior-mean shipping recovers −8.7% of the selection loss in Section 5. The rescoring chain was traced and contains no bug. The figure pools all four error processes and all magnitudes, including 0.05σ, where a rule's fixed price outweighs a tiny selection loss. Appendix K reports the cautious rule at 1σ and 5σ as shares of the deployed cost, which is a different denominator. State both scopes in the text.

The remedy conclusions need their designs in the sentences that state them. The rating-error rows of Table 26 use LogEI and qNEI at five seeds. Table 25 mixes LogEI with LogPI, LogEI with PI, LogEI with UCB, a pooled ten-acquisition reference, and a four-problem multi-objective arm at seeds 7 to 16. The accurate summary is that no acquisition-side change improved the deployed design, with no interval above zero and seven of twelve rows significantly negative. Table 25 is typed by hand, has no price column, and contains a "−979%" row whose reference cost is near zero and changes sign (third pass I3, I7).

Section 6 should be shortened. Dropping shekel moves βz from the range 0.82 to 0.86 up to 1.04 to 1.10, and it moves the GAIN gap from 0.36 to 0.51 up to 0.57 to 0.73. Both moves strengthen the rejections. The power law is fitted on four magnitude levels while Appendix B finds the response curved in log σe, so the GAIN test examines a functional form the paper later rejects (second pass E5). Appendix D.3's statement that raising a narrow optimum changes nothing holds at 1σ only, since the same change lowers the cost sixfold at 0.25σ (third pass I9).

The preregistered replication is void on its own control rule. If compute allows, rerun the Gaussian-only confirmatory arm on seeds 37 to 46 with the 0.10 headroom screen kept, Rosenbrock excluded by name, and a new preregistration dated before the run. That is about 21,600 runs, roughly 130 worker-hours or five to six hours on 24 workers. A confirmed replication would turn the weakest paragraph into a strong one.

The rater-noise anchor (0.74 to 3.35 σf) is fragile. The opticarvis numerator uses a nearest-neighbour estimator that the project's June notes call overestimated (code audit B3). The three denominators are SDs of different kinds of surface (second pass D5, corrected). The provoice composite is dominated by mental demand, and its oracle has held-out R² −0.03 (second pass D4, third pass I8). All three studies are automated-vehicle interfaces (third pass I14). One sentence on test-retest reliability from rating-scale psychometrics would anchor "1σ is realistic" without a fitted oracle.

The benchmark suite "was assembled for a separate study". If that suite is public, cite it in the third person. If it is the authors' own unpublished work, describe it without identifying details. The paper also says the suite was adopted unchanged while it corrects one published optimum by 64%.

### 2.4 Checks performed on the paper

Every number traced from the main text to the generated tables matched, with the exceptions listed in the three audit documents (for example the two intervals for one cell, third pass I12). Run counts are consistent with the design. Four input-error tables regenerate byte-identically from the analysis CSVs, two Table 5 cells reproduce from raw logs, Table 24's input-error rows rebuild exactly, and Table 26's price column reproduces for four arms (third pass H3, H4).

## 3. Formal compliance with ICLR 2027

Style file. The draft uses `iclr2026_conference.sty` and prints "Under review as a conference paper at ICLR 2026". Switch to https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip. `scripts/check_paper.py:38` hard-codes the 2026 file name and must be updated with it.

Page limit. The main text may have 9 pages at submission. References, the AI use statement, the optional ethics and reproducibility statements and acknowledgements do not count. The Limitations paragraph currently spills two lines onto page 10, and papers over the limit are desk-rejected.

AI use statement. The section is mandatory, and the submission form asks for the same information. The policy gives a template ("In this work, we used generative AI tools for ... We have not used generative AI tools for ...") and says authors need not follow it exactly. Disclosure is required for, among others, proposing or refining hypotheses, designing or giving feedback on methodology or experiments, implementing methods, generating synthetic data and interpreting results. The current "LLM usage" section describes several such uses but lacks the negative list and the verification method. The verification sentence may list only checks that were actually performed, with dates. As of 16 September these are a static audit of the code, offline execution of 98 tests, reproduction of the landscape statistics for 22 landscapes, regeneration of four tables from analysis files, and data checks on the side arms. The 443-test suite has not been run since the audit, and the parity tests against the source implementations only run on the author's machine. The authors must write and confirm this section. The policy treats an LLM-produced substantial falsehood as a Code of Ethics violation, which raises the stakes of the untraceable numbers listed in `README.md` in this folder.

Reciprocal reviewing. Each submission needs at least one author registered to review at least three papers. A qualified reviewer has an accepted paper at ICLR, NeurIPS, ICML, UAI, AISTATS, JMLR, TMLR, ACL, EMNLP, EACL, NAACL, IJCNLP-AACL, CL, TACL, COLM, CVPR, ICCV, ECCV, PAMI, 3DV, AAAI, IJCAI, JAIR, ICRA, IROS, RSS, CoRL, KDD or COLT. Authors on three or more submissions must review at least six papers. If no author qualifies, the paper is exempt, and each author may then appear on at most one such paper. Registration opens after the abstract deadline, and a submission without a registered reviewer is desk-rejected.

Dual submission. ICLR does not allow submissions that are identical or substantially similar to work submitted in parallel to other conferences or journals.

Anonymity. The paper text is clean apart from the dataset names in Table 20. The supplementary is not. The identifiers to remove are listed in `README.md` in this folder (third pass I17, second pass D3).

Reproducibility. The main-sweep outputs exist only on the simulation machine. The supplementary should contain the analysis and evaluation CSVs that `make_boba_paper_tables.py` reads, the landscape statistics, and anonymised run metadata, so a reviewer can regenerate every table without 470 worker-hours of compute. The Reproducibility statement's "scripted end to end ... to every table" is currently false (third pass I15).

OpenReview. All authors need profiles before 25 September. Profiles without an institutional email can take up to two weeks to be moderated.

Topics. The 2027 list includes probabilistic methods, optimization, and datasets and benchmarks. The introduction should frame the paper as a study of learning from corrupted human feedback.

## 4. Recommended restructuring of the main text

Title. "Search Loss and Selection Loss in Bayesian Optimization with Noisy Human Feedback". A title with "most of the cost is selection" is defensible only if the 1σ, onset-0 share and its interval lie above one half.

Abstract, 150 to 250 words. The first sentence states the decomposition. The second states that the default metric omits the selection term and gives the deployed-design cost beside the search cost at 1σ from the first rating. The third gives the selection share at that cell and states that it is zero under exact observation. The fourth states that no acquisition-side change improved the deployed design, that a final comparative sitting gains an absolute 0.05 of the achievable improvement (with the sitting noise stated), and that four confirmation trials cut false improvement claims from 15.7% to 0.5%. The fifth names the twenty exact landscapes and the zero-excess control. A frag sentence is optional and should report the within-magnitude Spearman beside the pooled one.

Introduction, 1 page. Problem, the fitted-oracle confound with third-party citations, the decomposition, and three or four contributions ordered by importance (decomposition, measurement protocol with both metrics, end-of-study procedures, predictor). The input-error arm and the fitted-oracle companion move to the appendix.

Related work, 0.75 page, extended as in Section 2.2.

Setup, 1.25 pages, with Figure 1 showing one landscape's clean and noisy trajectories with the best visited and the shipped design marked. Section 3.3 adds the optimiser settings (third pass I20).

Cost, 0.75 page, with Table 1 carrying both metrics and Figure 2 showing the dose-response by onset.

Decomposition, 1.5 pages, with Equation 5, the per-cell selection share, the exact-observation zero, the ship-rule results with their scopes, and Figure 3 as stacked bars by magnitude and onset.

What helps and what does not, 1.75 pages. Sections 8 and 9 merge. Table 4 keeps one sentence on the incumbent. Figure 4 plots Table 25 once it is generated by script with a price column and each row's design is stated.

Predicting the cost, 0.5 page.

Checks, 0.5 page, as one-sentence pointers to the appendix.

Discussion and Limitations, 0.75 page, including the remedy scope, the automated-vehicle scope of the rater-noise anchor, and what a study with real raters would need to measure.

That totals about 8.75 pages with four figures, which leaves little margin under 9 pages. Cut Section 7 to a paragraph if the compile runs over.

## 5. Action plan to 25 September

Wednesday 16 and Thursday 17. On the simulation machine, add `pytest.importorskip("numpyro")` to `tests/test_hierarchical_oracle.py`, run the test suite and keep the log. Fix code audit A2, A3, A4 and B4 and regenerate the affected numbers (see `README.md` in this folder for the exact reruns). Back up the output tree. Read the 1σ, onset-0 selection share. Decide the metric framing (A1) and the title.

Friday 18 by 23:59 AoE. Fix the author list, check reciprocal-reviewer eligibility, create missing OpenReview profiles with institutional emails, and register the rewritten abstract.

Friday 18 to Sunday 20. Switch to the 2027 style files. Restructure the main text as in Section 4. Draw four figures from the existing analysis files. Resolve the four `\todo` markers. Extend related work. Correct the confirmation-rule sentence (third pass I4). Generate Table 25 by script with a price column and drop its ratio row for missing ratings.

Sunday 20 to Tuesday 22, if compute allows. Rerun the Student-t arm with an RBF kernel in a fresh output directory, write and date a new preregistration and launch the confirmatory rerun, and decide on provoice (normalised rerun with a new driver, or removal).

Tuesday 22 to Thursday 24. Fold in rerun results. Recompile and confirm the main text ends on page 9. Build the supplementary from a clean export and scrub the identifiers. Run `scripts/check_paper.py` and the test suite. Ask a co-author to read the paper cold and mark every place where the argument depends on the appendix.

Friday 25 by 23:59 AoE. Submit with hours of margin.

## 6. Unconventional options

Reposition the paper as a benchmark and reporting-standard contribution under the "datasets and benchmarks" topic. The contribution becomes the twenty-landscape paired-twin protocol with two metrics and a small library that reports search and selection loss for any BO run. The remedies then serve as a demonstration of what the protocol can distinguish. This framing answers the "no new method" objection directly.

Alternatively, remove the end-of-study procedures from this submission and publish them separately once the ICLR decision is known. The ICLR paper would then carry one decomposition, one predictor and twenty landscapes. A parallel submission of substantially similar material elsewhere would breach ICLR's dual-submission policy, so the split has to happen before 25 September and the second paper has to wait.

## Sources

ICLR 2027 Call for Papers, https://iclr.cc/Conferences/2027/CallForPapers. ICLR 2027 Author Guidelines, https://iclr.cc/Conferences/2027/AuthorGuidelines. ICLR 2027 AI Policy for Authors, https://iclr.cc/Conferences/2027/AIPolicyForAuthors.
