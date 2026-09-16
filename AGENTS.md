# Working in this repository

<!-- HANDOVER-START -->
> Handover in progress. Start with `handover/README.md`. The `handover/` folder and this block are deleted before submission with `python handover/strip_handover_blocks.py`.
<!-- HANDOVER-END -->

This file is for whoever works on the repository next, human or agent. `README.md` explains what the project is. This file explains how not to break it and where the traps are. Everything here was learned by hitting it.

## What the project claims

Bayesian optimization driven by human ratings is measured on twenty analytic landscapes with published optima, with four error processes injected at four magnitudes and two onsets. The deployed design's regret splits into search loss (the run never visited anything better) and selection loss (it visited something better and shipped the wrong one). Selection loss is zero under exact observation, so a noiseless benchmark cannot show it, and no acquisition function targets it directly. Pooled over all cells, selection is 61.8% of the deployed cost of error. Near 1σ from the first rating it is about half. No acquisition-side change tested improved the deployed design. A final comparative sitting of about a third of the budget gains an absolute 0.05 of the achievable improvement, and four confirmation trials cut false improvement claims from 15.7% to 0.5%. Explicit models of the fault, such as response clipping, relevance pursuit and per-rater offsets, recover a substantial share of the cost of gross faults.

## Two metrics, never interchangeable

`simple_regret_true` is `y_opt − best_true_so_far`, the regret of the best design visited so far. It is search loss. The post-onset per-iteration AUC of its excess (`auc_simple_regret_excess_true_postonset_per_iter`) is the time-averaged search loss behind the dose-response, power-law, descriptor, frag and acquisition-ranking results. `inference_simple_regret_true` is the regret of the design with the best observed rating, which is what a study ships. At the final trial the shipped design's excess is typically two to three times the search excess. Name the metric and its time base in every sentence that quotes a number, and compare a final deployed value only with a final search value.

## Non-negotiables

Score a process change against the standard process, never against its own clean twin. Replicating ratings, re-rating before deploying, or changing the acquisition also changes the clean run, so excess over the arm's own twin measures it against a handicapped reference. This once made replication look as if it recovered a third of the cost when it recovers nothing. `scripts/analyse_boba_adaptations.py` implements the right estimand (`cost = ref_noisy − ref_clean`, `gain = ref_noisy − trt_noisy`, `price = trt_clean − ref_clean`). Import `paired_frame` and `summarise` from it instead of writing a second copy. When the reference is a pooled mean over several acquisitions, say so, because a strong single acquisition then looks like a remedy.

Never report a recovery ratio whose reference cost is near zero or changes sign across cells. Report the absolute difference instead.

Report the price of every process change beside its recovery.

Never report an acquisition robustness ranking without naming the incumbent. The observed-max incumbent removes 15.6% of the measured cost on its own and 41% for LogPI, and moves UCB and qNEI by exactly 0.000 because they never read `best_f`. That zero is the control that validates the pairing. Every ranking here ranks acquisition-and-incumbent pairs.

Do not re-raise the GP-noise confound. Supplying the true injected variance as `train_Yvar` removes 5.0% of the pooled cost, nothing survives FDR correction, and the onset ratio does not shrink. The onset effect remains unexplained, and the text should say so.

`random` and `sobol` must show 0.000e+00 excess search regret under response error. They ignore observations, so any other number means the pairing is broken. Their deployed design is still chosen by noisy ratings, so they are no zero control on the deployed metric, and under input error they evaluate displaced points and are no zero control at all.

Aggregate as ratios of landscape means with a landscape bootstrap, never as means of per-run ratios. The suite's achievable improvement (`opt_z`) spans 74×, from 0.76 on `powell` to 56.8 on `shekel`, so per-run ratios let a landscape with tiny regret outvote one with large regret. The landscapes share their random numbers by design, so state that the bootstrap treats correlated clusters as independent. With three or four clusters, report the cluster count and do not star results.

`gp.likelihood.noise` is the noise variance of the standardised targets. Multiply it by `gp.outcome_transform.stdvs ** 2` before combining it with anything in objective units.

Centre variables before forming interaction or quadratic terms, otherwise the linear coefficient is a conditional slope at zero of the raw variable.

Record the command behind every table in `paper/COMMANDS.md`. A number without a recorded producer does not go into the paper.

## Traps that have cost real time

A run's variant lives only in its file name. One output directory can hold several spike sizes or two ceiling modes. `run_metadata.json` and the per-arm summary CSVs are overwritten by the last invocation, so they describe only the last variant. `evaluate_research_question.py` parses the suffix into a `variant` column and puts it in `CONDITION_COLS`.

A variant that changes the clean run needs its own `--output-dir`. The simulator's guard (`_clean_run_settings` in `bo_synthetic_error_simulation.py`) tracks only minimum distance, the input-uncertain acquisition, relevance pursuit, the rater model and the halo model. It does not track the Student-t likelihood, the noisy-input GP, `--replicate-first`, `--final-rerate` or `--inference-rule`, so check those by hand.

Rerunning an arm into its old directory reuses its old runs. The drivers skip a directory that contains `SWEEP_COMPLETE`, and every run uses `--resume`, which matches on file name only. Code changes therefore need a fresh directory.

A baseline's oracle name can contain an underscore (`extra_trees`). Parse it from the logged `oracle_model`, never by regex over the file name.

The noise seed depends on the position of the acquisition in `ACQUISITION_CHOICES` and of the error model in `ERROR_MODEL_CHOICES`. Append new names at the end and never insert. `tests/test_acquisition_seed_order.py` and `tests/test_error_model_labels.py` pin the order.

Bash heredocs with an unquoted delimiter eat backslashes. Use a quoted delimiter or a file editor for LaTeX, regexes and Windows paths.

`nohup bash -c '...' &` does not survive the tool call that started it, while `nohup python ... &` does. Background jobs also die when the machine sleeps. Always use `--resume`, into the right directory.

The simulation machine is a Windows box with 24 usable workers. The drivers read `$env:PYTHON` and otherwise use `$env:LOCALAPPDATA\Programs\Python\Python312`. torch there is a CUDA build, so do not install `requirements-eval.txt` over it. The simulator never uses the GPU. A run costs about 21.5 worker-seconds, and the main sweep costs about 470 worker-hours. Move analysis files between machines, not simulations.

The main sweep (`output-boba/`) and several other arms are gitignored and exist only on that machine. The paper tables read their analysis and evaluation directories. Back them up separately.

`tests/test_hierarchical_oracle.py` imports jax and numpyro at module level. Without them pytest aborts the whole run at collection.

Smoke-test a driver before an overnight run. `run_boba_budget_neutral.ps1 -Smoke` runs every variant of every arm at toy size. It has caught an arm silently skipped by a `--help` guard and an arm whose two variants collided in one directory.

## The analyses, and what each is for

| script | question |
|---|---|
| `bo_synthetic_error_simulation.py` | the known-function arm's simulator (20 analytic landscapes) |
| `bo_sensor_error_simulation.py` | the fitted-oracle arm's simulator, sharing the core |
| `evaluate_research_question.py` | pairs noisy runs with clean twins, per landscape |
| `analyse_boba_robustness.py` | cross-benchmark synthesis, rankings, descriptors |
| `analyse_boba_adaptations.py` | a process change against the standard process |
| `rescore_ship_rules.py`, `analyse_ship_rules.py` | what the same trials would have shipped under another rule |
| `decompose_regret.py` | deployed regret split into search loss and selection loss |
| `replay_end_of_study.py` | tournaments and confirmation trials, replayed from logged prefixes |
| `replay_stopping.py` | onset detection and freezing |
| `budget_split.py` | how many of T trials to spend on identification |
| `changepoint_compare.py` | CUSUM, GLR and BOCPD on the same residual streams |
| `elicitation_compare.py` | a comparison loop against a rating loop at equal human cost |
| `make_boba_paper_tables.py` | every generated file in `paper/tables/` |
| `check_paper.py` | static checks on `\input`, `\ref`, `\cite` and table shapes |

A replay reproduces the logged prefix exactly, because the proposal at trial t depends only on trials 1 to t − 1. The new trials of a replay use a modelled sitting noise and hyperparameters frozen at the prefix fit. Every replay re-derives the standard ship rule from the prefix and aborts if it disagrees with the log. The GP refit and the acquisition are not checked against the log, since hyperparameters are not logged.

## What is settled, and what is not

Settled, with the measurement. Selection is a large share of the deployed cost of error, about half near 1σ and more at the late onset and at 5σ. No acquisition-side change tested improved the deployed design. Early replication and Thompson sampling make the deployed design worse. Four confirmation trials cut false improvement claims from 15.7% to 0.5%. A comparison loop is immune to shared monotone faults such as drift by construction and does not help against a saturating scale. The CUSUM already matches the classical optimal detector.

Not settled. The human is simulated. The best fitted oracle has held-out R² of about 0.55, and the provoice oracle is worse than a constant. The end-of-study procedures and the cautious ship rule pay a fixed price whether or not error is present, so they cost more than they recover below 0.25σ. Response clipping has no price. The onset effect is unexplained. The model-based budget rule must be recomputed before it can be quoted. The preregistered replication is void on its own control rule, and the post-hoc sensitivity analysis clears all four thresholds.

## Before you claim a result

1. Run `python -m pytest tests/ -q` and keep the log.
2. Run `python scripts/check_paper.py --paper paper`. It checks inputs, references and table shapes, and it does not compile the document.
3. Compile the paper (`pdflatex`, `bibtex`, `pdflatex` twice) and check that the main text through Limitations ends on page 9.
4. Regenerate tables with `python scripts/make_boba_paper_tables.py --analysis output-boba/analysis` after any analysis change. Never edit generated files in `paper/tables/`.
