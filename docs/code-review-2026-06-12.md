# Code & Methodology Review — 2026-06-12

Multi-agent review (6 dimensions, 31 agents). 24 findings adversarially verified (0 refuted), 66 additional suggestions, 7 critic findings from auditing the committed `output/` corpus. The two critical data findings were independently re-verified by hand.

## Dimension summaries

### sim-correctness
Core BO loop, error-injection timing, and pairing logic in scripts/bo_sensor_error_simulation.py are sound: noise first affects the observation at jitter_iteration+1 (apply_sensor_error returns clean values for iteration <= jitter_iteration, and the jitter RNG is a separate, untouched generator before that point), the first reactive candidate is jitter_iteration+2 exactly as the README states, baseline and jittered runs are properly paired via identical np/torch seeding, regret is computed on true oracle values, and evaluate_research_question.py reads metadata from CSV columns rather than parsing filenames, so underscore-containing dataset/objective names round-trip safely. The most consequential defects are in the plotting/aggregation layer: plot_combined_aspects.plot_regret_trajectories structurally can never draw the promised baseline curve, and plot_sensor_error_results pools incomparable datasets by omitting the dataset key. In the simulator itself, the singular --jitter-std/--jitter-iteration flags are silently dead, the multi-objective reference point ignores --normalize-objective, and the bias/dropout error models do not quite match their labels (fixed 0.2 bias with a noise sweep; permanent random-walk hold). Reproducibility is the other weak spot: evaluation globs every per-run CSV in the (committed, 3000-file) output directory without config keys or baseline dedup, filenames omit several distinguishing parameters, and parallel mode silently drops failed seeds. select_best_oracle_model.py is clean on leakage (train-fold-only normalization, GroupKFold), but its selection protocol (no augmentation) differs from how the chosen oracle is trained in the simulation, and the simulation reports inflated in-sample oracle R².

### stats-methodology
The pairing core is mostly sound: both the simulation summary merge and evaluate_research_question.build_paired_table match jittered runs to baselines on dataset/objective/acquisition/seed/oracle (plus jitter_iteration), use inner joins with correct many-to-one structure, and CIs/effect sizes that exist are computed over seeds (the correct unit), not iterations. However, the analysis has one critical methodological flaw — the 'confirmatory' follow-up pools the broad-screen seeds that selected the pi-vs-logpi hypothesis into the confirmatory tests, invalidating its inferential claims — and several high-severity issues: the evaluation script never checks that baseline and jittered traces share the same support (run length is not a merge key and the README/batch/confirmatory defaults disagree on --iterations, with stale committed CSVs globbed in), unnormalized excess AUC and ranks are averaged across incommensurate error conditions in the overall rankings that drive the headline answer, and the default n=5 seeds makes the 2-acquisition Wilcoxon cells mathematically incapable of significance while Friedman's chi-square approximation is unreliable. Secondary issues include silent pooling across oracle models in the ranking pivots, an uncorrected dual-p-value confirmatory table that declares winners by sign alone, legacy plots of a metric the README itself documents as identically zero, and inferential tests on the noise-contaminated observed objective. FDR (BH) correction and Cohen's dz/Kendall's W are present in parts of the pipeline but are applied inconsistently across the paper-facing outputs.

### code-quality
The codebase is a competent but monolithic research pipeline: seven standalone scripts with no package structure, importing each other via sys.path accidents and duplicating loaders, paired statistics, FDR correction, and four plotting implementations. The most correctness-threatening issues are (1) the simulation's silent fallback to random sampling when acquisition optimization fails — the warning is suppressed by a module-level UserWarning filter and nothing is recorded, so acquisition-function comparisons could be contaminated invisibly; (2) singular CLI flags (--error-model, --jitter-std) that are silently ignored in favor of plural-flag defaults, so users can run the wrong experiment without error; and (3) evaluation that globs an output directory containing 118 MB of committed results, silently mixing stale runs, while three mutually inconsistent dependency manifests and a committed run generated under a fourth environment undermine reproducibility claims. Cross-script file contracts are broken (effect-size CSVs land where neither plot_combined_aspects nor the dashboard looks), the streamlit dashboard is undeclared and partially dead-on-arrival, and the README documents only four of seven scripts while run_full_workflow.bat runs a different experiment (50 iterations, cv-folds 5) than the README recommends (100, 3). For CHI artifact evaluation, the priority order is: instrument and surface the acquisition fallback, fix the silently-ignored flags, unify dependencies and regenerate/caveat committed results, then extract a shared package to collapse the duplication.

### reproducibility
Internal experimental determinism is the repo's strength: every stochastic component is explicitly seeded (per-run np.random.default_rng(seed), torch.manual_seed, SeedSequence-derived jitter RNGs, random_state on every sklearn/xgboost/lightgbm/catboost/tabpfn oracle), thread counts are pinned (OMP/MKL=1, torch.set_num_threads(1), n_jobs=1), everything runs on CPU, and there are zero uses of unseeded np.random. However, environment and data reproducibility are in poor shape for a CHI artifact: the three dependency files mutually conflict and have drifted through four recent 'update requirements' commits; the committed 108 MB of results was generated with an environment (tabpfn 6.4.1, torch 2.10.0+cu128, numpy 2.3.5) that no current requirements file can recreate, and with code one month older than the current scripts. The experiment data is not in the repo at all — it is git-cloned at unpinned HEAD from three external GitHub repos (one third-party) with silent cache reuse, no recorded commit SHAs, and no licensing/provenance documentation. run_metadata.json records CLI args and 14 package versions but neither the code commit, the data snapshot, the Python version, nor the platform. Finally, the repo's three run recipes (README, run_full_workflow.bat, and the recorded args of the committed run) disagree on iterations and CV folds, and CI exercises only Ubuntu while the paper results were demonstrably produced on Windows.

### tests
Test quality, where tests exist, is genuinely good: regret math is verified against hand-computed values through run_simulation, apply_sensor_error boundary semantics and per-model error formulas are asserted exactly, _paired_test_from_diff is checked against scipy, and everything is seeded and deterministic (no flaky or slow tests gating CI). However, coverage is concentrated on unit-level helpers while the scientifically load-bearing integration paths are untested: no script main() ever runs (the 'cli_smoke' file is a misnomer), the README's t+1/t+2 jitter-timing claim is only tested against apply_sensor_error itself rather than through the run_simulation loop, the trapezoid AUC underlying the primary metric is never asserted numerically, and the headline ranking machinery (compute_condition_rankings, Friedman/Kendall's W, overall rankings) plus all of confirmatory_followup.py have zero tests. There is also an unpinned methodological inconsistency: summarize_adjustment still computes the t->t+1 response delta that the README explicitly disavows, feeding an untested, main()-embedded excess-summary merge that duplicates evaluate_research_question's pairing logic with different keys. Cross-platform discovery works (absolute pathlib + testpaths), but the 2144-line torch-importing module is re-executed four times at collection with no conftest, and CI installs the full torch stack uncached on ubuntu-only while development is on Windows. The highest-value additions are a tiny end-to-end pipeline smoke test, an integration test of error onset timing and the error_applied column, and numeric tests for AUC and the ranking/confirmatory statistics.

### chi-readiness
This is a competently engineered simulation pipeline (proper baseline/jittered pairing, seeded reproducibility, exploratory-then-confirmatory analysis with FDR correction, multi-oracle infrastructure), but as a CHI submission it currently overclaims on its central premise: there are no humans, the error models are sensor models (the code says so literally), noise levels (std 0.05-5) are uncalibrated against the ~1.6-point achievable range of the 7-point-scale objectives, observations are never clipped or discretized to the instrument, and every run gets 10-40 noise-free warm-up iterations — the opposite of real human feedback. The evidence base is thinner than the framing implies: 5 seeds for the main grid (the pipeline's own report admits low power), only 2 of 3 configured datasets actually run, all datasets from a single automotive-UX domain, no random-search floor, and no robust-BO competitors. The headline metric (excess AUC regret) can crown uniformly bad acquisitions as 'robust' and lacks any deployment-relevant translation for an HCI audience. The most impactful fixes, roughly in order: empirically calibrate noise from the within-participant variance in the source datasets (turning 'simulation' into 'empirically grounded simulation'), add noise-from-start and human-plausible error models (drift, heteroscedasticity, ordinal clipping), add random + robust baselines, scale to >=20 seeds with preregistered confirmatory hypotheses, and report absolute noisy-condition performance with CIs, effect sizes, and scale-point interpretations alongside excess regret.

## Verified findings (adversarially confirmed)

### [high/bug] Regret-trajectory plots can never show the baseline curve
**Where:** `scripts/plot_combined_aspects.py:240-277` (dimension: sim-correctness)

plot_regret_trajectories promises 'baseline vs jittered, mean ± 95 % CI' (docstring line 218) and iterates `for is_baseline, label, color in [(True, "Baseline (no noise)", ...), (False, "Jittered", ...)]`. But the groupby keys are `group_cols = ["objective", "error_model", "jitter_std", "jitter_iteration"]` (plus dataset). Baseline runs are written with `error_model = "none"` and `jitter_std = 0.0` (bo_sensor_error_simulation.py line 1724-1725), so baseline rows can never fall into a jittered group (e.g. error_model='gaussian', jitter_std=0.5), and the only group containing baseline rows is skipped by `if meta["error_model"] == "none": continue` (line 249). Therefore `sub = acq_df[acq_df["baseline"] == is_baseline]` is always empty for is_baseline=True and every figure silently shows only the jittered curve, despite the legend/label implying a comparison.

**Fix:** Select baseline rows separately: for each jittered group, fetch the matching baseline runs by (dataset, objective, acquisition, seed) ignoring error_model/jitter_std/jitter_iteration, e.g. pre-split df into baseline_df and jittered_df and join baseline_df on the non-error keys inside the loop.

### [medium/bug] --jitter-std and --jitter-iteration CLI flags are silently dead
**Where:** `scripts/bo_sensor_error_simulation.py:286-296, 1205-1216, 2006-2007` (dimension: sim-correctness)

parse_args defines `--jitter-iteration` (default 20) and `--jitter-std` (default 0.2), but main() builds the sweep with `jitter_stds = parse_float_list(args.jitter_stds, args.jitter_std)` and `jitter_iterations = parse_int_list(args.jitter_iterations, args.jitter_iteration)`. parse_float_list/parse_int_list only fall back to the singular value `if value is None`, and `--jitter-stds`/`--jitter-iterations` have non-None string defaults ("0.05,0.5,1,5" / "10,20,40"). So a user running `--jitter-std 2.0` or `--jitter-iteration 30` gets the full default sweep with no warning; the singular flags can never take effect. The module docstring example (`--jitter-iterations 20 --jitter-stds 0.1`) only works because it uses the plural form.

**Fix:** Either remove the singular flags, or set the plural defaults to None and fall back to the singular values, or raise an error when a singular flag is passed together with the (defaulted) plural flag.

### [medium/bug] Reference point ignores --normalize-objective, corrupting multi-objective hypervolume/regret when normalization is enabled
**Where:** `scripts/bo_sensor_error_simulation.py:1249-1260, 666-677, 1826-1833, 2106` (dimension: sim-correctness)

compute_reference_point computes `min_vals - 0.1 * ranges` from RAW objective values (`_extract_objective_values(df, objective_columns)` with no normalization). But with `--normalize-objective` and objective=multi_objective, build_oracle trains on normalized targets (`Y = compute_objective_matrix(df, objective_columns, normalize)`), so the oracle predicts values in [0,1] while the ref point stays on the raw scale (e.g. ~0.4 for a 1-7 Likert objective, or large values for wider scales). estimate_oracle_hypervolume (y_opt) and _compute_hypervolume in run_simulation both use this mismatched ref point, so hypervolume and all hypervolume-based regret metrics are computed against the wrong dominated region. Only latent because --normalize-objective defaults to False.

**Fix:** Compute the reference point in the same space as the oracle outputs: pass the normalization through compute_reference_point (use compute_objective_matrix(df, cols, normalize) before taking min/range), or assert that normalize_objective is False for multi_objective.

### [medium/bug] Legacy plots pool incomparable datasets: plot_objectives and plot_excess_adjustments group without 'dataset'
**Where:** `scripts/plot_sensor_error_results.py:394-425, 456-490` (dimension: sim-correctness)

plot_objectives groups by `["objective", "acquisition", "error_model", "jitter_std", "jitter_iteration", "oracle_model", "iteration"]` — no "dataset" — and the output filename also omits dataset. With multiple datasets (ehmi, opticarvis, provoice) all defining objective="composite" on different value scales, the plotted trajectory is a meaningless cross-dataset average. plot_excess_adjustments similarly aggregates bo_sensor_error_excess_summary.csv with `excess.groupby(["objective", "acquisition", "error_model", "jitter_iteration", "jitter_std"])`, pooling delta_excess_l2_norm across datasets whose parameter spaces have completely different dimensionality and units (9 vs 16 vs 4 parameters). plot_adjustments additionally facets by error_model with hue="baseline", but baseline rows only exist under error_model="none", so the baseline/jitter contrast never appears in any panel.

**Fix:** Add "dataset" to all groupby keys and filenames in this script (as evaluate_research_question.py already does), and pair baseline rows explicitly instead of relying on hue within error_model facets.

### [low/bug] --combine-datasets is incompatible with --oracle-model auto (guaranteed KeyError)
**Where:** `scripts/bo_sensor_error_simulation.py:998-1002, 2020-2023, 2048-2053` (dimension: sim-correctness)

main() appends a synthetic dataset named 'combined' when --combine-datasets is set, then calls resolve_oracle_models_for_objective for every dataset. In auto mode this does `oracle_selection.get((dataset_name, objective_name))` and raises `KeyError(f"No auto-selected oracle found for dataset='combined' ...")` because select_best_oracle_model.py has no --combine-datasets option and never writes a 'combined' entry. So the README-recommended `--oracle-model auto` plus --combine-datasets always crashes before any simulation runs.

**Fix:** Add a --combine-datasets option to select_best_oracle_model.py, or fall back to a default/explicit oracle for the combined dataset, or reject the flag combination with a clear error at argument-validation time.

### [critical/methodology] Confirmatory analysis double-dips on the screening seeds that generated the hypothesis
**Where:** `scripts/confirmatory_followup.py:63-72, 538-543` (dimension: stats-methodology)

copy_existing_logs() copies the broad-screen per-iteration logs (seeds 7-11 per run_full_workflow.bat) into the confirmatory raw dir: 'for source in sorted(base_input_dir.glob(pattern)): ... shutil.copy2(source, destination)'. run_evaluation() then evaluates the pooled raw dir, and all downstream 'confirmatory' tests (summarize_condition_tests, summarize_dataset_tests, main_confirmatory_table.tex) include those screening seeds alongside the 20 new seeds (12-31). The pi-vs-logpi comparison itself was selected FROM the broad screen (base_overall rankings), so the same data that picked the winner is reused in the confirmatory test. This is classic double dipping/selection bias and invalidates the 'confirmatory' label and the nominal p-values in the LaTeX table intended for the paper.

**Fix:** Filter the confirmatory tests to seed >= args.seed_start (new seeds only), or at minimum report screening-seed and new-seed results separately and label the pooled analysis as exploratory. State in the paper that the comparison was selected on independent seeds.

### [high/bug] No same-support guard for AUC pairing; runs with different --iterations are silently paired and pooled
**Where:** `scripts/evaluate_research_question.py:43-47, 137-139, 150-167` (dimension: stats-methodology)

load_iteration_logs() globs every 'bo_sensor_error_*_seed*_*.csv' in the input dir, and build_paired_table() merges on ['dataset','objective','acquisition','seed','oracle_model','jitter_iteration'] only — run length is not a key and there is no assertion that baseline and jittered traces have equal max_iter. AUC is 'float(np.trapezoid(run_df["simple_regret_true"]..., dx=1.0))' over each run's full trace, so a 100-iteration baseline paired with a 50-iteration jittered run (or vice versa) produces a meaningless excess AUC. This is a live hazard: README's quickstart uses --iterations 100 while run_full_workflow.bat and confirmatory_followup.run_simulation() use the default 50 and write into the same output tree; output/ also contains committed historical CSVs. The simulation-side merge (bo_sensor_error_simulation.py line 2214) does include 'iterations' as a key, so the evaluation script is strictly weaker than the in-memory summary.

**Fix:** Carry n_iterations (max_iter) into the response table, add it to the merge keys, and assert max_iter_jitter == max_iter_baseline per pair; additionally normalize AUC by the number of iterations and/or refuse to run when the input dir mixes run lengths. Pass merge(validate='many_to_one') to catch duplicate baselines.

### [high/methodology] Overall rankings average ranks and raw excess AUC across incommensurate conditions
**Where:** `scripts/evaluate_research_question.py:309-324` (dimension: stats-methodology)

compute_overall_rankings() does rankings.groupby(['dataset','objective','acquisition']).agg(mean_rank=('mean_rank','mean'), mean_auc_simple_regret_excess_true=('mean_auc_simple_regret_excess_true','mean'), ...), pooling over error_model, jitter_iteration (10/20/40) and jitter_std (0.05-5). Excess AUC is unnormalized, so its magnitude mechanically scales with the post-onset window (iterations - jitter_iteration) and with jitter_std; the pooled mean AUC is dominated by the std=5, jitter=10 cells. mean_rank is averaged across conditions whose acquisition sets (and hence rank scales, k) can differ, since compute_condition_rankings only requires pivot.shape[1] >= 2. The README research question is explicitly per 'dataset and error level', yet this pooled table is what feeds the confirmatory pair selection (base_overall in confirmatory_followup.py line 545).

**Fix:** Normalize: divide AUC by the affected window length, or rank-transform within condition before pooling (mean_rank is fine only if k is constant — enforce a complete acquisition set per condition or normalize ranks to [0,1] by (rank-1)/(k-1)). Report pooled results only as a secondary, clearly-labeled aggregate.

### [high/methodology] n=5 seeds: Wilcoxon cells can never reach p<0.05 and Friedman chi-square approximation is invalid; mixed test families share one BH correction
**Where:** `scripts/evaluate_research_question.py:263-303` (dimension: stats-methodology)

The default design is 5 seeds (bo_sensor_error_simulation.py --num-seeds default 5; run_full_workflow.bat BROAD_SEEDS=7,8,9,10,11). For 2-acquisition conditions the code runs wilcoxon(diff, zero_method='wilcox', alternative='two-sided'); with n=5 the minimum attainable two-sided exact p is 2/2^5 = 0.0625, so these tests can never be significant even before FDR — the pipeline silently guarantees null results for those cells. For >2 acquisitions, friedmanchisquare's chi-square approximation is unreliable at n=5 blocks (scipy itself recommends n>10). Finally, multipletests(..., method='fdr_bh') at line 300 pools Friedman omnibus p-values and Wilcoxon pairwise p-values into a single family, mixing hypotheses of different types and granularity.

**Fix:** Increase seeds for the broad screen (>=20) or switch to exact/permutation tests and say explicitly that 2-acquisition cells are descriptive only at n=5; run BH separately per test family (or per dataset/objective), and report the attainable minimum p in the report text.

### [medium/bug] oracle_model is a pairing key but is dropped from condition grouping; pivot aggfunc='mean' silently pools across oracles
**Where:** `scripts/evaluate_research_question.py:190, 212, 219-224` (dimension: stats-methodology)

build_paired_table pairs on oracle_model, but summarize_conditions uses condition_cols = ['dataset','objective','error_model','jitter_iteration','jitter_std','acquisition'] and compute_condition_rankings uses ['dataset','objective','error_model','jitter_iteration','jitter_std'] — neither includes oracle_model. The per-condition pivot 'group.pivot_table(index="seed", columns="acquisition", values=PRIMARY_METRIC, aggfunc="mean")' will silently average over multiple oracle models (or any duplicated logs) within a seed cell, hiding pseudo-replication; n_seeds=('seed','nunique') then under-reports the number of rows behind each mean/std. The same pattern exists in confirmatory_followup.build_head_to_head_table (lines 162-164: index_cols lack oracle_model, aggfunc='mean').

**Fix:** Add oracle_model to all condition grouping keys (or assert a single oracle per dataset/objective before analysis), and replace aggfunc='mean' with a duplicate check (e.g., validate via groupby.size()==1) so unexpected pooling fails loudly.

### [low/bug] Winner map assigns ties and NaN cells to the reference acquisition
**Where:** `scripts/confirmatory_followup.py:403-421` (dimension: stats-methodology)

plot_winner_map uses 'np.where(mean_data["mean_diff"] < 0, challenger, reference)': a cell with mean_diff exactly 0 (possible, e.g., jitter_std=0 sweeps or all-zero diffs) and a cell whose mean is NaN are both colored as a reference win rather than being masked — np.where(NaN < 0) is False. The heatmap mask 'mask=pivot.isna()' operates on winner_code, which is never NaN because the winner string is always mapped to 0/1, so missing conditions render as reference wins instead of blanks.

**Fix:** Mask cells where mean_diff is NaN or where |mean_diff| is below a tolerance (render as 'tie'), e.g., compute winner_code as np.select([diff<0, diff>0], [1,0], default=np.nan) and let the existing mask hide NaNs.

### [high/bug] Failed acquisition optimization silently falls back to random sampling, and the warning is globally suppressed
**Where:** `scripts/bo_sensor_error_simulation.py:113-114, 1638-1640` (dimension: code-quality)

In run_simulation, any exception during BoTorch acquisition optimization is caught with `except Exception as e: warnings.warn(f"BoTorch optimization failed, falling back to random. Error: {e}"); candidate_np = sample_uniform(...)`. The module sets `warnings.filterwarnings("ignore", category=UserWarning)` at import time (lines 113-114), and warnings.warn defaults to UserWarning, so this fallback message is never shown — in sequential mode or in spawned workers (which re-import the module and re-install the filter). Nothing is recorded in the per-run CSV either: a run labeled acquisition='qei' may contain an unknown number of purely random iterations and the paper's acquisition-function comparison cannot detect it. The same global filter also silences the `warnings.warn("Cannot combine datasets...")` messages in combine_dataset_configs (1148-1165).

**Fix:** Record fallback events in the results DataFrame (e.g., an `acq_opt_failed` boolean column per iteration) and in run_metadata.json; print to stderr or use logging instead of warnings.warn; replace the blanket UserWarning/RuntimeWarning filters with narrowly scoped `warnings.catch_warnings()` around the specific noisy library calls.

### [high/bug] Singular CLI flags --error-model, --jitter-std, --jitter-iteration are silently ignored
**Where:** `scripts/bo_sensor_error_simulation.py:286-296, 345-346, 925-935, 1205-1216, 2005-2007` (dimension: code-quality)

The plural flags have non-empty string defaults (--error-models='gaussian,bias', --jitter-stds='0.05,0.5,1,5', --jitter-iterations='10,20,40'), and the parsers use `raw = error_models or error_model` / `parse_float_list(args.jitter_stds, args.jitter_std)` where the fallback branch is unreachable from the CLI. Concretely, `python scripts/bo_sensor_error_simulation.py --error-model spike` silently runs gaussian+bias, and `--jitter-std 0.3` silently runs the default sweep. The singular values still leak into base_config, so baseline per-run CSVs carry a misleading `jitter_iteration` column equal to the dead --jitter-iteration default (20) at line 1726. For an experiment-configuration interface this can silently run the wrong experiment.

**Fix:** Remove the singular flags or make plural flags default to None and fall back explicitly; emit an error if a singular flag is passed alongside a plural one; set baseline rows' jitter_iteration to NaN or omit it.

### [medium/bug] infer_param_columns reconstructs the run-CSV schema from a hand-maintained reserved-column set — fragile and silently corrupting
**Where:** `scripts/evaluate_research_question.py:53-88` (dimension: code-quality)

evaluate_research_question infers parameter columns by listing every column NOT in a hardcoded `reserved` set mirroring run_simulation's output schema. If any new metadata or metric column is added to run_simulation's results (or appears via the loader's column normalization), it is silently treated as a design parameter and folded into `response_l2` (line 117), distorting the REACTION_METRIC without any error. The set is already one schema drift away from wrong (e.g. `error_applied` is included only because someone remembered to add it). Note the simulator already writes `param_columns` into the summary CSV (bo_sensor_error_simulation.py:1918) but not into the per-iteration logs that this script consumes.

**Fix:** Have run_simulation write the parameter column list into each per-iteration CSV (a `param_columns` column, or a sidecar JSON), and make infer_param_columns read it; fall back to an explicit allowlist with a loud error on unknown columns.

### [medium/bug] Broken cross-script file contract: effect_sizes_cohens_dz.csv is expected in the evaluation dir but written by the legacy script to output/plots — forest plots and two dashboard pages silently never work in the documented workflow
**Where:** `scripts/plot_combined_aspects.py:320-337` (dimension: code-quality)

plot_combined_aspects.plot_effect_size_forest reads `evaluation_dir / 'effect_sizes_cohens_dz.csv'` and silently returns [] if missing. That file is produced only by plot_sensor_error_results.py (line 228) into its own --output-dir (default output/plots), and plot_sensor_error_results.py is not part of run_full_workflow.bat nor the README's recommended workflow. So the workflow never produces forest plots and gives no indication why. dashboard.py (lines 61-66) likewise looks for effect_sizes_cohens_dz.csv and final_outcome_paired_tests.csv in output/evaluation, where the recommended pipeline never writes them — its 'Effect Sizes' and 'Paired Statistics' pages are dead-on-arrival with the documented workflow.

**Fix:** Move the effect-size computation into evaluate_research_question.py (the canonical evaluation step) so all consumers find it in the evaluation dir, or at minimum print a warning naming the missing file and the script that produces it.

### [medium/bug] Parallel seed failures are printed and swallowed; final summaries are silently incomplete
**Where:** `scripts/bo_sensor_error_simulation.py:2156-2164` (dimension: code-quality)

In parallel mode, a worker exception is handled with `print(f"\nError processing seed {seed}: {e}"); traceback.print_exc()` and the loop continues. The run then completes 'successfully' (exit code 0, run_full_workflow.bat proceeds), but bo_sensor_error_summary.csv and the excess summaries simply lack that seed's rows, and run_metadata.json still lists the seed as requested with no failure record. Downstream evaluation just sees fewer seeds per condition (the report prints seed counts, but nothing flags the discrepancy). There are also six broad `except Exception: pass` blocks around progress-queue/manager plumbing (lines 1798, 1803, 2126, 2168, 2172, 2176) — acceptable for progress reporting, but the seed-level swallow is not.

**Fix:** Record failed seeds (with tracebacks) in run_metadata.json, exit non-zero if any seed failed, or add a --keep-going flag that defaults off; narrow the bare except blocks to the specific queue/BrokenPipe exceptions.

### [high/methodology] Two divergent 'response step' conventions (t->t+1 vs t+1->t+2) and duplicate excess-merge logic are not pinned by any test
**Where:** `scripts/bo_sensor_error_simulation.py:1741-1768, 2210-2271` (dimension: tests)

README (lines 119-121) states the meaningful response step is t+1 -> t+2. evaluate_research_question.build_response_table implements that (start=jit+1, end=jit+2, tested). But summarize_adjustment (line 1750-1752) still computes delta from jitter_iteration -> jitter_iteration+1 (the convention the README explicitly calls wrong), and its outputs feed bo_sensor_error_excess_summary.csv via the main()-embedded merge at lines 2210-2271 (delta_excess_*, delta_excess_l2_norm). The only test of summarize_adjustment (test_summarize_adjustment_includes_avg_regret, tests/test_simulation_utils.py:443) checks one regret field and never asserts which iterations the delta uses. The excess merge itself (merge keys at lines 2216-2227 including xi/kappa/param_columns) is inline in main() and untestable/untested, and duplicates eval_mod.build_paired_table with different keys.

**Fix:** Add a test pinning summarize_adjustment's delta to specific iterations (linearly increasing params, assert delta == params[jit+1]-params[jit]) and document in the test docstring that this is the legacy t->t+1 convention vs the eval pipeline's t+1->t+2 — or change it to match the README and test that. Extract the lines 2210-2271 merge into a function build_excess_summary(summary_df) and add a test asserting its auc_simple_regret_excess_true equals eval_mod.build_paired_table's value for the same runs.

### [high/methodology] Error models omit the best-documented human response behaviors
**Where:** `scripts/bo_sensor_error_simulation.py:1350-1382` (dimension: chi-readiness)

apply_sensor_error implements i.i.d. gaussian noise, constant bias (+jitter), hold-last dropout, and spikes — and only gaussian and bias are run by default (line 346, confirmed by the 3000 committed output CSVs which contain only gaussian/bias). As models of HUMAN rating error this misses every behavior the survey-methodology and psychophysics literatures consider first-order: temporal drift and fatigue (error variance growing with trial count), anchoring/learning effects early in a session, heteroscedasticity (more inconsistency for mid-scale/indifferent stimuli, less at the extremes), scale compression and central-tendency/extreme-response styles, and serial correlation between consecutive ratings. The i.i.d.-after-onset structure is a sensor model, not a human model.

**Fix:** Add at least: (1) a drift/fatigue model where bias or std increases linearly or sigmoidally with iteration; (2) a heteroscedastic model where std depends on distance from scale midpoint or from the participant's running mean; (3) an autocorrelated (AR(1)) error model. Cite human-rating-noise literature (e.g., test-retest reliability of Likert ratings, response styles in survey methodology, fatigue effects in repeated perceptual judgments) to justify the chosen forms. Also actually run the already-implemented dropout and spike models or remove them from the paper's claimed scope.

### [high/methodology] Noisy observations are neither clipped nor discretized to the response scale — impossible human ratings
**Where:** `scripts/bo_sensor_error_simulation.py:1362-1380` (dimension: chi-readiness)

Objectives are means of Likert-type constructs (Trust, Understanding, etc.); a sample committed run shows objective_true in [3.77, 5.40] with y_opt=5.37, i.e. roughly a 7-point scale. apply_sensor_error returns true_value + N(0, jitter_std) with no clipping and no quantization, so at jitter_std=5 the simulated 'human' routinely reports ratings like -4.6 or 13.2 — values no bounded ordinal instrument can produce. Real humans produce censored, discrete responses; censoring at scale endpoints changes the noise distribution qualitatively (it becomes biased toward the interior near the optimum), which can change which acquisition functions are robust.

**Fix:** Clip observed values to the instrument range and, ideally, round to the scale's granularity (the source data's resolution), then re-run at least the high-noise conditions to check whether conclusions change. Report this as an explicit design decision either way; reviewers with quantitative-methods background will check for it.

### [high/methodology] Noise magnitudes (std 0.05–5) are unjustified relative to the ~1.6-point achievable objective range
**Where:** `scripts/bo_sensor_error_simulation.py:296` (dimension: chi-readiness)

Default sweep is --jitter-stds 0.05,0.5,1,5 (line 296; run_metadata.json confirms these were used). On the ehmi composite, the entire range of objective values observed in a run is ~1.6 points (3.77–5.40) on a ~1–7 scale. So std=0.05 is negligible (3% of signal range), std=5 is ~3x the ENTIRE signal range (noise that obliterates all information — no human is that inconsistent), and the sweep jumps 0.05→0.5 and 1→5 with nothing in the plausible-human middle. None of the four levels is tied to any empirical estimate of human rating variability, and normalize_objective defaults to False so the same absolute stds mean different things across datasets with different scales.

**Fix:** Calibrate the noise grid empirically: compute within-participant / within-condition rating SDs from the three source datasets and define noise levels as multiples of that empirical SD (e.g., 0.5x, 1x, 2x, 4x), or as fractions of the instrument range. Report noise as a signal-to-noise ratio so results transfer across datasets. Drop or clearly label std=5 as a stress test, not a human-plausible condition.

### [high/methodology] Noise-free warm-up (jitter onset at iteration 10/20/40) contradicts the human-feedback framing
**Where:** `scripts/bo_sensor_error_simulation.py:1357-1358` (dimension: chi-readiness)

apply_sensor_error returns the true value unchanged for all iterations <= jitter_iteration, so every run enjoys 10–40 iterations of perfectly clean feedback before errors begin. That is a sensor-degradation scenario (sensor works, then breaks), not a human one: human raters are noisy from trial 1, and if anything become noisier (fatigue) or more consistent (calibration) over time. The README's own semantics (lines 112–123) are built around this onset design. As a result the headline conditions never test the practically dominant HITL case — noise present throughout the optimization, including during the initial-sample phase where the GP prior is formed.

**Fix:** Add jitter_iteration=0 (noise from the first observation) as the primary condition and reposition the delayed-onset conditions as a secondary 'when does noise hurt most' analysis. This single addition aligns the experiment with the paper's framing and is cheap to run with the existing sweep machinery.

### [high/methodology] Five seeds per condition for the main grid — the pipeline's own report admits the analysis is underpowered
**Where:** `scripts/evaluate_research_question.py:460` (dimension: chi-readiness)

The committed results use seeds 7–11 (run_metadata.json), i.e., n=5 replicates per cell, and evaluate_research_question.py literally writes '- With only 5 seeds per condition, p-values are low power.' into its own report (line 460). Friedman tests over 10 acquisition functions with 5 blocks, and 2-method Wilcoxon tests where the minimum attainable two-sided p with n=5 is 0.0625 (can never reject at alpha=.05 even before FDR correction), make the per-condition inferential layer essentially decorative. BO benchmarking papers typically use 20–50 replicates; the confirmatory follow-up does use 20 seeds (confirmatory_followup.py defaults, lines 32–33) but only for the single pi-vs-logpi comparison on the composite objective.

**Fix:** Increase the main grid to >=20 seeds (the simulation is already parallelized; runtime_sec=47466 for 5 seeds suggests ~2.5 days at 20 seeds, or prune the acquisition set first). Alternatively, present the 5-seed grid as explicitly exploratory/descriptive (mean ranks only, no p-values) and expand the confirmatory phase to the top-3 vs bottom-3 acquisitions across all datasets. Justify the final n with a power analysis based on the observed seed-level variance in the committed output.

### [high/methodology] No random-search floor and no purpose-built noise-robust BO competitors
**Where:** `scripts/bo_sensor_error_simulation.py:151-164` (dimension: chi-readiness)

ACQUISITION_CHOICES contains 10 single-objective and 2 multi-objective BoTorch acquisitions but no random/Sobol baseline (greedy is UCB with beta=0, which is exploitation, not a floor). Without random search, a reviewer cannot tell whether at std=5 the 'best' acquisition is meaningfully better than not modeling at all — quite plausibly it is not, given noise 3x the signal range. The noise-aware acquisitions qNEI/qNEHVI are included (good), but there are no explicitly robust competitors: GP with Student-t likelihood for outliers, re-querying/replication of suspicious observations, median-of-replicates strategies, or input warping. The paper's implied takeaway ('which acquisition should HITL practitioners use under noise') is weak without these comparison points.

**Fix:** Add (1) uniform-random and Sobol candidate selection as floors, (2) at least one robust-likelihood GP variant (Student-t likelihood is a few lines in GPyTorch), and (3) a simple practitioner heuristic such as querying each design twice and averaging (same budget accounting). These define the floor and ceiling that make the acquisition rankings interpretable.

### [high/methodology] Ranking purely on excess regret can crown acquisitions that are uniformly bad
**Where:** `scripts/evaluate_research_question.py:30, 219-230` (dimension: chi-readiness)

The primary metric is auc_simple_regret_excess_true = AUC(jittered) - AUC(baseline), and condition rankings (compute_condition_rankings, lines 219–230) rank acquisitions by this difference alone. An acquisition that performs near-randomly in BOTH baseline and noisy conditions has excess ~0 and ranks as maximally 'robust', while a method that is excellent clean and merely good under noise ranks worse. For an HCI audience the deployment question is 'which method gives the best design under realistic noise', i.e., absolute noisy-condition performance, with robustness as a secondary property. The current headline metric can systematically recommend the wrong method.

**Fix:** Report a two-dimensional result: absolute jittered regret (primary, answers 'what should I deploy') and excess regret (secondary, answers 'how much does noise cost'). A scatter of baseline AUC vs excess AUC per acquisition, or a rank aggregation over both, would expose any 'robust because flat' artifacts. At minimum verify and state in the paper that no top-ranked acquisition is dominated on absolute performance.

## Critic findings (from auditing committed output/ and the BoTorch layer)

### [critical/reproducibility] Committed run corpus contradicts committed run_metadata.json about which ground-truth oracle generated half the data
**Where:** `output/run_metadata.json:resolved_oracle_models block; writer at scripts/bo_sensor_error_simulation.py:2380-2387`

run_metadata.json records resolved_oracle_models = {'opticarvis:composite': ['gradient_boosting'], 'opticarvis:multi_objective': ['gradient_boosting']}, but every one of the ~1,500 committed opticarvis run CSVs is named '..._extra_trees...' and carries oracle_model='extra_trees' in its rows (e.g. bo_sensor_error_opticarvis_composite_ei_seed10_baseline_extra_trees.csv). The metadata writer derives resolved_oracle_models from the same objective_oracle_models dict that drives the runs, so this metadata cannot describe these files. Git history confirms: the run CSVs were committed 2026-01-30 (79497b6 'Added Test Results'), while run_metadata.json (589c9a6) and best_oracle_models.json (48a5e8d) were replaced on 2026-03-10. The committed corpus is therefore a January data set described by March metadata and a March oracle-selection file under which opticarvis's actual oracle (extra_trees, CV R^2 0.57) ranks 4th behind gradient_boosting (R^2 ~1.0). Additionally, the metadata claims provoice was part of the invocation (datasets and objectives include provoice, resolved provoice:multi_objective='tabpfn'), yet zero provoice output files exist and no failure is recorded — a concrete committed instance of the swallowed-parallel-failure problem. Any paper statement sourced from run_metadata.json about the experiments misstates the ground-truth function for half of all results.

**Fix:** Regenerate the entire output corpus in one invocation and commit data + metadata + oracle-selection atomically; add the resolved oracle model, git SHA, and a per-file manifest (filename -> config hash) to run_metadata.json; make evaluate_research_question.py cross-check each run CSV's oracle_model/jitter parameters against the manifest and fail loudly on mismatch.

### [critical/methodology] Committed oracle-selection results show the 'ground-truth human' has near-zero or negative cross-validated R^2, and auto-selection has no quality gate
**Where:** `output/best_oracle_models.json:ehmi/provoice entries; argmax at scripts/select_best_oracle_model.py:300-303 and scripts/bo_sensor_error_simulation.py:990-1008`

Nobody read the committed selection file. It reports, under user-grouped CV: ehmi composite best R^2 = 0.097 (extra_trees); ehmi multi_objective best R^2 = -0.172 with ALL eight models negative; provoice composite best R^2 = -0.008 (all negative); provoice multi_objective best R^2 = -0.030. Negative grouped R^2 means the chosen oracle predicts held-out users WORSE than a constant. select_best_oracle_model.py picks max(scores) with an RMSE tie-break and resolve_oracle_models_for_objective consumes 'best_model' with no minimum-quality threshold, so the simulation proceeds to treat a function with essentially zero predictive validity for real humans as the deterministic ground truth for the ehmi half of all committed experiments (and would do so for provoice). This is stronger than the existing 'unreported fidelity' finding: fidelity IS reported in the repo and it is damning, and any CHI reviewer can read it. Robustness conclusions about 'noisy human feedback' are drawn on a simulated human that does not predict humans.

**Fix:** Gate auto-selection on a minimum grouped-CV R^2 (refuse to run or warn prominently below it); report oracle CV fidelity alongside every result table; reframe paper claims as robustness on 'data-derived synthetic test functions' rather than human ground truth, or demonstrate conclusions are insensitive to oracle misspecification.

### [high/methodology] opticarvis oracle selection exhibits a perfect-fit leakage signature (R^2 = 0.9999999999, RMSE ~1e-8 across held-out source files)
**Where:** `output/best_oracle_models.json:opticarvis entries; CV construction at scripts/select_best_oracle_model.py:69-95`

For opticarvis, gradient_boosting/xgboost/catboost achieve R^2 = 0.99999+ with RMSE down to 1.5e-8 under GroupKFold with group_source='__source_file' and effective_cv_folds=2 (only two observation files, no User_ID). Machine-precision generalization across held-out files means the two ObservationsPerEvaluation.csv files contain (near-)duplicated X->y mappings or y is deterministically computed from X — i.e., grouping by source file does not prevent identical rows appearing in both folds. Meanwhile lightgbm and hist_gradient_boosting score exactly 0.0 (constant predictions), a degenerate-data signature. The two datasets that constitute ALL committed results therefore live in incomparable ground-truth regimes — opticarvis: a deterministic interpolator of apparently duplicated data; ehmi: an R^2~0.1 noise-fitter — yet downstream rankings treat them as two comparable replications of 'human feedback'. The silent fold-count downgrade (5 requested -> 2 effective) is recorded in JSON but surfaced nowhere.

**Fix:** Audit the opticarvis raw files for duplicated rows / derived objective columns before any rerun; deduplicate (X,y) pairs before CV; warn loudly when effective_cv_folds < requested or when CV R^2 is implausibly close to 1; discuss the qualitative ground-truth difference between datasets in the paper or drop the degenerate dataset.

### [high/methodology] Improvement-based acquisitions use the max of NOISY observations as the incumbent (best_f), confounding the robustness ranking with a known-bad implementation choice
**Where:** `scripts/bo_sensor_error_simulation.py:1617, 1426-1455, 1475-1485`

best_f = train_Y.max().item() (line 1617) where train_Y are the error-corrupted observations, then EI/logEI/PI/logPI/qEI/qPI receive best_f + xi (lines 1426-1455). With the committed gaussian sweep up to std=5 on objectives whose true range is ~0.7-2.8, a single positive noise spike inflates the incumbent far above any achievable value, driving improvement-based acquisitions to ~0 everywhere so they effectively stall — a textbook noisy-BO pitfall (the standard remedies are a posterior-mean incumbent or the noisy variants). qNEI by contrast uses X_baseline (lines 1459-1465) and qEHVI/qNEHVI partition on the noisy observed front (1475-1505), so the paper's headline comparison ('which acquisition is robust to feedback noise') partially measures incumbent-definition artifacts rather than intrinsic acquisition robustness: EI/PI-family fragility is exaggerated by construction and qNEI is advantaged by construction. No reviewer examined the BoTorch modeling layer at all.

**Fix:** Use the maximum posterior mean over visited points (or over the design space) as best_f for EI/PI-family under noise, or explicitly frame the experiment as comparing 'off-the-shelf noise-naive vs noise-aware implementations' and add the posterior-mean-incumbent variants as conditions; add a unit test that best_f under injected noise does not exceed the posterior-mean incumbent.

### [high/methodology] Simple regret is computed with an omniscient incumbent: the run's best TRUE value, which no noisy-feedback experimenter could identify
**Where:** `scripts/bo_sensor_error_simulation.py:1676-1692`

best_true_so_far = max(best_true_so_far, scalar_true) (line 1683) and s_t = max(0, y_opt - best_true_so_far) (1686): the 'recommendation' credited to the optimizer at every iteration is the best point by its TRUE objective value, even in jittered runs where the experimenter only sees corrupted ratings. In a real HITL deployment, noise harms you twice — it misdirects search AND prevents you from recognizing which evaluated design was actually best. The metric removes the second mechanism entirely (identification is free and perfect), so excess regret systematically understates the cost of feedback noise, biasing the paper's central robustness conclusions optimistically. Combined with the noisy-max best_f issue this means noise corrupts the acquisition's incumbent but never the reported recommendation — the inverse of deployment reality. No reviewer questioned the incumbent/recommendation rule.

**Fix:** Report 'inference regret': select the incumbent by observed data (e.g., max observed value or max GP posterior mean), then score its true value; keep the omniscient version as a secondary diagnostic. This is a one-line change in the regret bookkeeping plus a paired metric column.

### [medium/methodology] The search space is the per-column min/max box of a prior optimization campaign's trace; committed runs show some dimensions confined to a sliver while the oracle extrapolates over the rest
**Where:** `scripts/bo_sensor_error_simulation.py:822-831 (bounds_from_data, sample_uniform), 2105`

bounds_from_data takes df[param].min()/max() over the observation logs, which for these datasets are themselves the trajectories of earlier human-in-the-loop BO studies — so the simulated design space is the box hull of wherever a previous optimizer happened to sample, not the actual UI design space. Committed evidence (30 opticarvis baseline files, including 150 uniform initial samples): EgoTrajectory never leaves [0.7449, 0.8456] and CarStatus never leaves [0.7443, 0.8536] (width ~0.1), while sibling parameters span ~[0.16, 0.79]. Data coverage inside this box is concentrated along the prior optimizer's path, yet (a) the tree-based oracle is queried uniformly across the box where it extrapolates as piecewise constants, and (b) y_opt is estimated by 200k uniform random draws over the same mostly-unsupported box (estimate_oracle_optimum, 833-852). Neither the box construction nor its coverage was examined by any reviewer, and no test covers bounds_from_data.

**Fix:** Document and justify the design-space definition per dataset (ideally from the original studies' declared parameter ranges, not data hulls); report training-data coverage of the search box; restrict the oracle/optimum estimation to a supported region or use the original studies' bounds; add a test that bounds match the source studies' documented ranges.

### [medium/methodology] The zero-regret clamp is not hypothetical: 11 of 50 committed opticarvis composite baselines end at exactly 0 simple regret because BO beats the random-search y_opt
**Where:** `scripts/bo_sensor_error_simulation.py:1677, 1684, 1686; evidence in output/bo_sensor_error_opticarvis_composite_*_baseline_extra_trees.csv`

New committed-data evidence escalating what reviewers filed as a low-severity edge case: in bo_sensor_error_opticarvis_composite_ei_seed10_baseline_extra_trees.csv, best_true_so_far reaches 2.7520 while y_opt = 2.5681 — the optimizer exceeds the 'optimum' by 0.18 — and across the 50 committed opticarvis composite baselines, 11 (22%) finish with simple_regret_true exactly 0.0. Once a baseline saturates at the floor, final_simple_regret_excess_true = jitter - baseline is mechanically >= 0 and the AUC tail difference is censored at an arbitrary point set by the y_opt underestimate, so the primary metric's distribution is a censored mixture in exactly the dataset that supplies half the rankings. Instantaneous and cumulative regret (r_t = max(0, y_opt - true)) are clipped the same way (line 1684).

**Fix:** Drop the max(0, .) clamps and allow negative regret relative to the y_opt estimate (or estimate y_opt by optimizing the oracle directly, e.g., multi-start gradient-free search seeded with training optima plus all BO-visited points across runs); at minimum report the fraction of floor-censored runs per condition.

## Additional suggestions (not adversarially verified)

### [medium/reproducibility] Evaluation globs all per-iteration CSVs in the output dir; stale runs silently corrupt pairing (no iterations key, no dedup)
**Where:** `scripts/evaluate_research_question.py:42-50, 149-167` (dimension: sim-correctness)

load_iteration_logs concatenates every `bo_sensor_error_*_seed*_*.csv` in --input-dir. The README and run_full_workflow.bat write into the committed `output/` directory (currently ~3000 CSVs). Per-run CSV filenames do not encode iterations, error_bias, spike params, --single-error, or augmentation settings, so re-running with changed settings leaves stale files that are silently mixed into the analysis. build_paired_table merges on `["dataset", "objective", "acquisition", "seed", "oracle_model", "jitter_iteration"]` without an `iterations` (run-length) key and without deduplicating multiple baseline runs per key: a leftover 50-iteration baseline plus a new 100-iteration baseline produces a many-to-many merge (duplicate pairs) and pairs AUC values integrated over different horizons. The same glob is used in plot_sensor_error_results.py (line 44), plot_combined_aspects.py (line 225), and dashboard.py (line 46).

**Fix:** Add max-iteration / config columns to the pairing keys, deduplicate baselines (assert exactly one baseline per key), and either write each simulation invocation to a fresh directory or filter loaded files against run_metadata.json (e.g. by run_id set).

### [medium/methodology] Composite objective averages raw objective columns on different scales (no normalization by default)
**Where:** `scripts/bo_sensor_error_simulation.py:536-554` (dimension: sim-correctness)

compute_objective returns `values.mean(axis=1)` over raw (sign-adjusted) objective columns when weights is None, and --normalize-objective defaults to False everywhere (README workflow never sets it). For the provoice dataset the composite is mean(Predictability, 'Percieved Usefulness', -'Mental Demand') on raw scales; if Mental Demand uses a wider scale (e.g. NASA-TLX style) than the other items, the negated column dominates the composite and effectively becomes the objective. The same fixed `--error-bias 0.2` and jitter_std sweep (0.05..5) is applied to all datasets/objectives regardless of their objective scale, so 'std=5' is a vastly different relative perturbation for an unnormalized composite on one dataset vs another, confounding cross-dataset robustness comparisons.

**Fix:** Enable per-objective min-max normalization by default for composite (and document it), or express jitter_std and error_bias relative to each objective's observed range so error severity is comparable across datasets.

### [low/methodology] Bias error model sweeps noise, not bias: jitter_std sweep adds Gaussian noise on top of a fixed 0.2 bias
**Where:** `scripts/bo_sensor_error_simulation.py:1362-1369` (dimension: sim-correctness)

apply_sensor_error always draws `jitter = rng.normal(0.0, config.jitter_std, ...)` and for error_model='bias' returns `true_value + bias + jitter` with bias fixed at --error-bias (default 0.2, never swept). So the conditions labeled 'bias, std=0.05 .. std=5' actually vary the Gaussian noise around a constant small bias; at std=5 the condition is dominated by noise and nearly indistinguishable from the gaussian model (the offset 0.2 is negligible). If the paper claims to compare 'bias' severity levels, the manipulation does not match the label.

**Fix:** Sweep error_bias for the bias model (and keep jitter_std at 0 or a small fixed value), or rename the conditions to 'bias+noise' and report both parameters explicitly.

### [low/methodology] Dropout error model implements a permanent random walk, not transient dropout
**Where:** `scripts/bo_sensor_error_simulation.py:1370-1374, 1646-1667` (dimension: sim-correctness)

For error_model='dropout' (strategy 'hold_last'), every post-jitter observation is `observed = previous_observed + jitter` where previous_observed is the previous OBSERVED value (run_simulation line 1667 sets `previous_observed = observed_value`). Once jitter starts, the signal never returns to the true value: observations become a random walk anchored at the last clean observation, with variance growing linearly in iterations (each step adds N(0, jitter_std^2)). 'hold_last' suggests a sensor repeating its last reading during a dropout episode, but this is a permanent, drifting dropout with no recovery. Not in the default error_models ('gaussian,bias'), but it would be reported under a misleading name.

**Fix:** Either document the permanent-random-walk semantics explicitly, or add a dropout probability/duration so the signal intermittently recovers to the true value.

### [low/methodology] Oracle selection protocol differs from how the chosen oracle is actually trained in the simulation
**Where:** `scripts/select_best_oracle_model.py:133-139, 258` (dimension: sim-correctness)

select_best_oracle_model.py benchmarks models via CV on the raw data with no augmentation (`model.fit(X_train, y_train)` line 136) and uses tree_scale 0.7 in --oracle-fast mode. The simulation then trains the auto-selected oracle WITH jitter augmentation by default (--oracle-augmentation default 'jitter', repeats=2, bo_sensor_error_simulation.py lines 376-383, 666-724) and uses tree_scale 0.35 in fast mode (line 662). The model ranked best under one training protocol is deployed under a different one, so the 'best_model' choice may not be best for the actual ground-truth function used in the experiments. The CV itself is clean (normalization fit on train folds only, GroupKFold by User_ID — no leakage found).

**Fix:** Apply the same augmentation (fit-time only, on the training fold) and tree_scale settings during model selection as used by build_oracle in the simulation.

### [low/quality] Oracle quality reported as training-set R²/RMSE on augmented data
**Where:** `scripts/bo_sensor_error_simulation.py:689-695, 726-732` (dimension: sim-correctness)

build_oracle prints `train_score = model.score(X_aug_df, y_aug)` and RMSE computed on the same (augmented) data the model was just fit on: 'Oracle (extra_trees) ... R^2 score: ...'. For ExtraTrees/RandomForest with min_samples_leaf=2 this is near-1.0 by construction and says nothing about generalization; if these printed numbers are quoted in the paper as oracle fidelity they overstate it (the honest numbers are the CV scores in best_oracle_models.json).

**Fix:** Label the printed metrics explicitly as in-sample/training metrics, or report the CV metrics from the selection JSON instead.

### [low/methodology] Simple regret clamped at zero against a random-search estimate of the optimum
**Where:** `scripts/bo_sensor_error_simulation.py:833-852, 1677-1692` (dimension: sim-correctness)

y_opt is the max of oracle predictions over 200k uniform random points (estimate_oracle_optimum), and regret is `r_t = max(0.0, y_opt - scalar_true)` / `s_t = max(0.0, y_opt - best_true_so_far)`. Because y_opt is a lower bound on the true supremum, BO can exceed it; the max(0, ...) clamp then floors regret at exactly 0. In excess-regret differences (jittered - baseline) this clamp is asymmetric: once both runs hit the floor the difference is forced to 0, shrinking measured robustness differences specifically in the easy/late-iteration regime. For piecewise-constant tree oracles 200k samples is usually adequate, so the practical impact is likely small, but it biases auc_simple_regret_excess_true toward 0.

**Fix:** Estimate y_opt with a margin (e.g. also evaluate all BO-visited points post hoc and take the global max across runs per oracle), or keep the unclamped signed regret in a separate column so the floor effect can be quantified.

### [low/reproducibility] Jittered-run filenames omit error_bias/spike parameters and --single-error, so variant runs overwrite each other
**Where:** `scripts/bo_sensor_error_simulation.py:1966-1970` (dimension: sim-correctness)

The per-run CSV name is `bo_sensor_error_{dataset}_{objective}_{acq.name}_seed{seed}_jittered_{oracle_model}_{error_model}_jit{jitter_iteration}_std{jitter_std}.csv`. Runs differing only in --error-bias, --error-spike-prob/std, --single-error, --iterations, or oracle augmentation settings map to the identical filename and silently overwrite previous results (or, worse, coexist as stale files if the parameter is changed together with another one). Combined with the glob-everything loaders this makes mixed-config analyses easy to produce by accident. (Float formatting itself round-trips fine: parse_float_list yields 1.0 -> 'std1.0', and no reader parses values back out of filenames — all readers use CSV columns.)

**Fix:** Encode the remaining distinguishing parameters (bias, spike prob/std, single_error, iterations) in the filename or in a per-invocation subdirectory keyed by a config hash.

### [low/reproducibility] Parallel mode swallows per-seed failures; summary outputs silently miss seeds
**Where:** `scripts/bo_sensor_error_simulation.py:2156-2164, 1601-1617` (dimension: sim-correctness)

In the parallel branch, `except Exception as e: print(f"Error processing seed {seed}: {e}"); traceback.print_exc()` continues after a seed-level failure, so bo_sensor_error_summary.csv and all downstream rankings are computed from fewer seeds with no machine-readable record of the loss (only a console print, easily missed under tqdm output). A failure is plausible because `fit_gpytorch_mll(mll)` (lines 1602/1616) is outside the per-iteration try/except that guards only get_botorch_candidate, so a GP fitting error aborts the entire seed including all its remaining acquisition/error conditions. Note also that --parallel is auto-enabled whenever len(seeds) > 1 (line 2071), which is the default (--num-seeds 5).

**Fix:** Record failed seeds in run_metadata.json and exit nonzero (or re-raise) when any seed fails; move fit_gpytorch_mll inside the existing fallback try/except.

### [low/quality] Wrong return type annotation on _evaluate_model_multi_objective
**Where:** `scripts/select_best_oracle_model.py:146-193` (dimension: sim-correctness)

`def _evaluate_model_multi_objective(...) -> float:` actually returns `{"r2": ..., "rmse": ...}` (dict), matching the single-objective sibling. Purely cosmetic but misleading for reviewers/static analysis.

**Fix:** Change the annotation to `-> dict[str, float]`.

### [low/methodology] Per-seed oracle retraining makes the ground-truth function seed-dependent
**Where:** `scripts/bo_sensor_error_simulation.py:1806-1841` (dimension: sim-correctness)

run_single_seed calls build_oracle(..., seed=seed) so the tree ensemble defining the synthetic ground truth (and hence y_opt, re-estimated per seed at lines 1826-1841) differs across seeds. Baseline/jittered pairing is still valid (both share the same per-seed oracle and y_opt), but across-seed variance now mixes BO stochasticity with ground-truth-function variability, which changes the interpretation of seed-level statistics (Friedman blocks in evaluate_research_question.py treat seeds as replicates of the same problem). This may be an intentional design (robustness across plausible ground truths) but should be stated in the paper; it also multiplies runtime since y_opt is re-estimated with 200k oracle evaluations per seed per objective.

**Fix:** Either fix the oracle seed (train once per dataset/objective/oracle_model and share across seeds) or explicitly document that each seed is a different sampled ground-truth function and that conclusions are marginal over oracles.

### [medium/methodology] Main confirmatory table reports raw, uncorrected p-values from two tests and declares a winner by sign regardless of significance
**Where:** `scripts/confirmatory_followup.py:256-280, 311-373` (dimension: stats-methodology)

build_main_confirmatory_table takes dataset_tests (one t-test and one Wilcoxon per dataset x objective x metric) and writes p_value_t and p_value_wilcoxon to the paper-facing main_confirmatory_table.tex with no multiplicity correction (FDR is computed only for the condition-level table in summarize_condition_tests) and no declared primary test — reporting both p_t and p_W invites picking the smaller. 'table["winner"] = np.where(table["mean_diff"] < 0, challenger, reference)' labels a winner purely by the sign of the mean difference, even when the CI spans zero, and a mean_diff of exactly 0 or NaN is credited to the challenger/reference asymmetrically.

**Fix:** Pre-specify one primary test (paired t on seed-averaged diffs or Wilcoxon), apply Holm or BH across the dataset x objective cells feeding the table, and only declare a winner when the corrected test rejects (otherwise print 'n.s.').

### [medium/methodology] Dataset-level confirmatory test averages excess AUC across incommensurate error conditions before testing
**Where:** `scripts/confirmatory_followup.py:221-229, 232-248` (dimension: stats-methodology)

summarize_dataset_tests collapses each seed to 'float(group[diff_col].mean())' over all error_model x jitter_iteration x jitter_std cells (gaussian and bias, std 0.05 to 5, jitter 10-40), then runs ttest_1samp across seeds. Because unnormalized excess AUC scales with jitter_std and with the post-onset window, this unweighted seed average is dominated by the std=5/jitter=10 cells; the headline confirmatory conclusion therefore answers 'who wins under an arbitrary mixture dominated by extreme noise', not the stated research question of which acquisition works at which error level. The unweighted mean is also unbalanced if any seed is missing cells (group sizes are not checked).

**Fix:** Either stratify the confirmatory claim by error level (test per jitter_std, or per error_model, with correction), or standardize the per-condition diffs (e.g., divide by per-condition pooled SD or use within-condition ranks) before averaging across conditions; check that every seed contributes the same set of conditions.

### [medium/quality] Legacy plots visualize delta_excess_l2_norm and baseline-vs-jitter delta_l2 that are identically zero/identical by construction
**Where:** `scripts/plot_sensor_error_results.py:428-490` (dimension: stats-methodology)

summarize_adjustment (bo_sensor_error_simulation.py lines 1750-1756) measures the step x(t+1)-x(t), but both candidates are chosen before any noisy observation can influence the model (noise first affects the observation at t+1, first reactive candidate at t+2 — as the README and evaluate_research_question's own report text state: 'In the current outputs, delta_excess_l2_norm is identically zero'). plot_adjustments() nevertheless bar-plots delta_l2_mean with hue='baseline' (the two hues are identical by construction) and plot_excess_adjustments() bar-plots delta_excess_l2_norm means (identically zero). These figures are pure noise-free artifacts that could mislead readers if they leak into the paper.

**Fix:** Delete plot_adjustments/plot_excess_adjustments or recompute the response step as x(t+2)-x(t+1) (matching evaluate_research_question.build_response_table), and remove delta_l2/delta_excess_l2 from the simulation summary CSVs or rename them with an explicit 'pre-reaction' warning.

### [medium/methodology] Paired tests on objective_observed trivially detect the injected error, not robustness, and dominate the reported 'largest effects'
**Where:** `scripts/plot_sensor_error_results.py:26-33, 198-211, 223, 279-303` (dimension: stats-methodology)

ANALYSIS_METRICS includes 'objective_observed'; evaluate_final_outcomes_improved runs paired t/Wilcoxon on final objective_observed_jitter - objective_observed_baseline. Under the bias error model this difference contains the injected offset by construction, so significance is guaranteed and uninformative about acquisition behavior. effect_sizes (line 223) is restricted to OBJECTIVE_METRICS = ['objective_true','objective_observed'], so the effect-size heatmaps and the 'Largest paired effects' section of statistical_report.txt (sorted by |cohens_dz| over all metrics) are likely dominated by this artifact. Additionally, final-iteration objective_true is the last evaluated candidate, not best-so-far, making it a high-variance outcome compared to simple_regret_true.

**Fix:** Drop objective_observed from inferential testing (keep it descriptive), restrict effect-size reporting to true-value metrics (simple_regret_true / best_true_so_far), and exclude observed-metric rows from the 'largest effects' ranking.

### [medium/quality] Silent run/row drops: jitter_iteration+2 > max_iter discards the whole run's regret metrics; inner joins drop unmatched rows without reporting
**Where:** `scripts/evaluate_research_question.py:109-115, 162-170` (dimension: stats-methodology)

In build_response_table, 'if end_iter > max_iter: continue' skips the entire row — including final_simple_regret_true and auc_simple_regret_true, which do not depend on the response step — so any condition with jitter_iteration >= iterations-1 silently vanishes from ALL metrics (the simulation itself allows jitter_iteration = iterations-1 and keeps it in bo_sensor_error_summary.csv, so the two outputs can disagree on coverage). build_paired_table's how='inner' merge then drops jittered rows lacking a baseline with no count of dropped rows; the only failure surfaced is the fully-empty case. '.iloc[0]' on lines 114-115 also raises an opaque IndexError if an iteration is missing from a truncated CSV.

**Fix:** Decouple the response-step computation from the regret/AUC metrics (emit regret metrics even when t+2 is out of range, with response_l2=NaN); log the number of jittered rows that failed to pair and the number of skipped runs; use merge(indicator=True) to report drop counts, and validate iteration completeness per run.

### [low/methodology] Best acquisition declared from omnibus tests without post-hoc pairwise inference or effect sizes for 2-method cells
**Where:** `scripts/evaluate_research_question.py:255-292` (dimension: stats-methodology)

Each condition row reports 'best_acquisition': str(mean_ranks.index[0]) alongside a Friedman omnibus p-value; a significant Friedman test only says some acquisitions differ, not that the top-ranked one beats the runner-up — no post-hoc (e.g., Nemenyi or paired Wilcoxon with Holm) is run. For 2-acquisition conditions, kendalls_w is set to NaN and no effect size (matched-pairs rank-biserial or Cohen's dz) is provided, so the Wilcoxon cells carry a p-value but no magnitude. Ties in mean rank are broken arbitrarily by sort order in 'mean_ranks.index[0]'.

**Fix:** Add post-hoc pairwise comparisons (top-1 vs top-2 at minimum) with Holm correction when the omnibus test rejects; report rank-biserial r or dz for 2-method cells; mark best_acquisition as tied when isclose to the runner-up's mean rank.

### [low/quality] Condition summary and heatmaps show point estimates with no uncertainty at n=5
**Where:** `scripts/evaluate_research_question.py:189-205, 327-419` (dimension: stats-methodology)

summarize_conditions reports mean/median/std of the excess metrics but no SEM or CI, and the mean-rank / mean-excess-AUC heatmaps (plot_condition_heatmaps) annotate raw means with no indication of seed-level variability. With n_seeds=5 the standard error of a mean rank is large (for k=6 acquisitions, SD of a single rank is ~1.7, so SEM ~0.76); heatmap differences of <1 rank are visually salient but statistically indistinguishable. CIs that are computed elsewhere (confirmatory paired_stats_from_diff) correctly use seed-level differences with t critical values, so the unit of analysis is right where CIs exist — they are simply absent here.

**Fix:** Add bootstrap or t-based 95% CIs over seeds to condition_summary and overlay significance/CI half-width (or grey out cells where the CI covers 0) in the heatmaps.

### [low/quality] Degenerate-difference edge cases produce NaN p-values and inconsistent n in _paired_test_from_diff
**Where:** `scripts/plot_sensor_error_results.py:110-144` (dimension: stats-methodology)

If all paired differences are equal and non-zero (std_diff ~ 0 but not all-zero), the t-branch 'if n_pairs >= 2 and not np.isclose(std_diff, 0.0)' is skipped and the elif requires np.allclose(diffs, 0.0), so p_value_t stays NaN even though the evidence of a systematic shift is maximal (a constant offset should be reported as significant or at least flagged, not NaN — NaN rows are then silently excluded from the BH family in _apply_fdr, changing m). Also wilcoxon(zero_method='wilcox') discards zero differences while the reported 'n_pairs' counts all finite diffs, so the effective n of the test is overstated in the output table. The same wilcox zero-handling exists in evaluate_research_question.py line 278 and confirmatory_followup.py line 152.

**Fix:** Handle the constant-nonzero case explicitly (report p ~ 0 with a degenerate-variance flag), use zero_method='pratt' or report n_nonzero alongside n_pairs, and document that NaN-p rows are excluded from the FDR family.

### [medium/reproducibility] README, batch workflow, and confirmatory defaults disagree on iterations and write into one shared output tree with committed stale CSVs
**Where:** `run_full_workflow.bat:33-41, 56-66` (dimension: stats-methodology)

README quickstart runs the simulation with '--iterations 100'; run_full_workflow.bat omits --iterations (default 50, bo_sensor_error_simulation.py line 284) and confirmatory_followup.run_simulation (lines 75-100) also never passes --iterations, so a user following the README and then the batch/confirmatory flow produces 100- and 50-iteration logs in the same directories. Baseline filenames ('..._seed{seed}_baseline_{oracle}.csv') do not encode iterations or sweep settings, so reruns partially overwrite old files while leaving orphaned jittered logs behind; load_iteration_logs in both analysis scripts globs everything present, and the large committed output/ directory means a fresh clone already contains result CSVs that any default invocation will silently mix with. Combined with the missing same-support check (separate finding), this makes published numbers dependent on directory history rather than on the declared configuration.

**Fix:** Pin --iterations in run_full_workflow.bat and confirmatory_followup to the README value; write each invocation into a timestamped/config-hashed subdirectory (run_metadata.json already exists — also check it at load time and refuse to mix configs); remove or relocate committed result CSVs out of the default --input-dir path.

### [high/quality] No installable package; scripts import each other via sys.path accidents and module-level side effects
**Where:** `pyproject.toml:1-31` (dimension: code-quality)

pyproject.toml declares [project] metadata and dependencies but no [build-system], no package discovery, and no entry points — `pip install .` would install nothing usable. scripts/ has no __init__.py; select_best_oracle_model.py does `from bo_sensor_error_simulation import (...)` (lines 14-26), which only works because Python prepends the script's directory to sys.path when invoked as `python scripts/select_best_oracle_model.py`. All three test files use sys.path/importlib hacks (e.g. tests/test_cli_smoke.py:11-23 and test_scientific_correctness.py:23-36 exec the 2,436-line simulation module repeatedly under different names). Worse, importing bo_sensor_error_simulation as a 'library' mutates global state: it sets OMP/MKL/OPENBLAS env vars (lines 21-28), calls torch.set_default_dtype/torch.set_num_threads (104-105), installs blanket warning filters (113-114), and contains an inexplicable `if __name__ not in sys.modules:` sys.modules patch (124-127) that is effectively dead cargo-cult code for multiprocessing.

**Fix:** Restructure into an installable src/ package (e.g. hitl_mobo/{data.py, oracle.py, simulation.py, metrics.py, stats.py, plotting.py}) with console-script entry points; keep scripts/ as thin CLIs; move env-var/torch/warnings setup into main() guarded code, not import side effects; delete the sys.modules hack.

### [high/reproducibility] Three conflicting dependency manifests, and committed results were generated under a fourth, unpinned environment
**Where:** `requirements-eval.txt:1-14` (dimension: code-quality)

pyproject.toml, requirements.txt, and requirements-eval.txt disagree: scikit-learn ==1.8.0 (eval) vs >=1.9.0 (requirements) vs >=1.8.0 (pyproject); tabpfn ==8.0.3 vs >=8.0.8 vs >=6.4.1; botorch ==0.17.2 vs >=0.18.1; tqdm ==4.67.3 vs >=4.68.2; torch >=2.9.1 vs ==2.12.0. requirements.txt floors exceed requirements-eval.txt pins, so the two files are mutually unsatisfiable. ImportError messages in bo_sensor_error_simulation.py (e.g. line 773: "Install it via requirements.txt") point at a different file than the README's recommended `pip install -r requirements-eval.txt`. Meanwhile the committed output/run_metadata.json records numpy 2.3.5, tabpfn 6.4.1, torch 2.10.0+cu128, botorch 0.17.2 — an environment matching none of the manifests — so the shipped results are not reproducible from any documented install.

**Fix:** Make pyproject.toml the single source of truth; generate one fully pinned lock file (pip-compile/uv lock) used by CI and the README; delete or clearly repurpose the redundant requirements file; regenerate (or clearly caveat) the committed results under the pinned environment.

### [high/reproducibility] 118 MB / 3,371 result files committed in output/, and evaluation globs the whole directory so stale runs silently mix into the analysis
**Where:** `scripts/evaluate_research_question.py:42-50` (dimension: code-quality)

output/ contains 3,371 committed files (118 MB) of per-run CSVs plus legacy output/plots figures, but NOT the recommended output/evaluation results. load_iteration_logs concatenates every file matching `bo_sensor_error_*_seed*_*.csv` in --input-dir with no manifest check: re-running the simulation with different settings (iterations, sweeps, oracle) into the same default `output/` mixes old and new runs without any warning, directly corrupting rankings and statistics. The committed run_metadata.json also shows the shipped data used --iterations 50 while the README's recommended command uses --iterations 100, so reproducing the README does not reproduce the committed results.

**Fix:** Move result data out of git (release asset/Zenodo/git-lfs) for artifact evaluation; have the simulation write a manifest of run_ids/files and make evaluate_research_question load only manifest-listed files (or at least cross-check iterations/config in run_metadata.json and fail loudly on mixed configurations).

### [medium/quality] Heavy code duplication across scripts: log loading, paired statistics, FDR correction, and four near-identical plotting implementations
**Where:** `scripts/plot_sensor_error_results.py:43-48, 80-181` (dimension: code-quality)

The iteration-log loader and its filename-glob contract `bo_sensor_error_*_seed*_*.csv` are reimplemented four times (evaluate_research_question.py:43, plot_sensor_error_results.py:44, plot_combined_aspects.py:225, dashboard.py:46) — plot_sensor_error_results uses unsorted `list(...)` while the others sort. Paired-test statistics are duplicated as `_paired_test_from_diff` (plot_sensor_error_results.py:80-159) and `paired_stats_from_diff` (confirmatory_followup.py:115-157) with subtly different NaN/zero-variance handling. BH-FDR application is implemented three times (plot_sensor_error_results._apply_fdr:162-181, confirmatory_followup:199-212, evaluate_research_question:296-304). The rank heatmap exists in evaluate_research_question.plot_condition_heatmaps and dashboard.page_rank_heatmap; the regret-trajectory plot in plot_combined_aspects.plot_regret_trajectories and dashboard.page_trajectories; the Cohen's-dz forest plot in plot_combined_aspects.plot_effect_size_forest and dashboard.page_forest_plot; the excess-AUC heatmap in plot_combined_aspects and confirmatory_followup.plot_confirmatory_heatmaps. Divergence between copies is a real risk for paper figures/statistics.

**Fix:** Extract shared modules (io.py with the loader + filename schema constants; stats.py with paired tests + FDR; plotting.py with heatmap/trajectory/forest builders) in a package, and have all five consumer scripts plus the dashboard import them.

### [medium/quality] God functions: main() ~430 lines, run_single_seed ~227 lines, run_simulation ~197 lines, get_botorch_candidate ~120-line if/elif chain
**Where:** `scripts/bo_sensor_error_simulation.py:1542-1738, 1771-1997, 2001-2427` (dimension: code-quality)

main() (2001-2427) mixes run planning, parallel orchestration with an inline progress-monitor thread, pandas excess-summary computation via iterrows (2231-2266), groupby aggregation, and metadata/config file writing. run_single_seed (1771-1997) mixes oracle training, optimum estimation, simulation, per-run CSV writing, summary-row construction (the ~20-field summary dict is duplicated verbatim for baseline at 1903-1919 and jittered at 1977-1992), and progress callbacks. run_simulation (1542-1738) interleaves GP fitting, acquisition optimization, error injection, regret bookkeeping, and DataFrame assembly, so the simulation core cannot be reused without producing its exact output format. get_botorch_candidate (1410-1529) is a 13-branch if/elif chain that would be a dict-based registry. None are individually buggy, but they make the core experiment nearly untestable in isolation — visible in the tests, which mostly exercise small parsers instead of the loop.

**Fix:** Split into: a pure simulate() returning arrays/dataclasses, a writer that serializes results, a summary builder shared by baseline/jittered branches, an acquisition-factory dict, and a thin orchestration layer; this also removes the duplicated 20-field summary block.

### [medium/quality] Deprecated delta_l2_norm metric still computed, written to all summary CSVs, and plotted, despite README/report declaring it wrong
**Where:** `scripts/bo_sensor_error_simulation.py:1741-1768, 2231-2351` (dimension: code-quality)

summarize_adjustment computes the t→t+1 parameter delta (delta_l2_norm), which the README ('Important Note About the Immediate Adjustment Metric') and evaluate_research_question's own report text (lines 440-444: 'delta_excess_l2_norm is identically zero') document as the wrong response step. Yet the simulator still computes it per run, derives delta_excess_l2_norm in main(), aggregates it into bo_sensor_error_summary.csv, bo_sensor_error_excess_summary.csv and bo_sensor_error_dataset_effects.csv, and plot_sensor_error_results.plot_adjustments/plot_excess_adjustments (428-490) still produce bar charts of it. Related legacy debris: DATA_DIR points to a nonexistent repo dir 'eHMI-bo-participantdata' (line 109); PARAM_COLUMNS/OBJECTIVE_MAP (129-149) duplicate the ehmi entry of datasets.json in code; _normalize_qehvi_columns (439-450) silently strips ' QEHVI' suffixes from dataset columns with no documentation.

**Fix:** Either delete the metric and its plots, or rename it (e.g. delta_l2_norm_legacy) with a comment pointing at the corrected response_l2 in evaluate_research_question; remove the dead DATA_DIR fallback and derive the in-code defaults from datasets.json; document the QEHVI column munging.

### [medium/quality] dashboard.py is unmaintained and cannot run from any documented install: streamlit is declared nowhere
**Where:** `scripts/dashboard.py:20, 14-16` (dimension: code-quality)

dashboard.py imports streamlit, which appears in no requirements file, not in pyproject.toml, and not in CI — `streamlit run scripts/dashboard.py` fails with ImportError on a fresh documented install. The README never mentions the dashboard. Two of its four pages depend on legacy outputs that the recommended workflow never produces (see the effect_sizes finding). Minor issues: unused `import sys` (line 16); module-level argument parsing into a global OUTPUT_DIR (lines 29-38); no test coverage. The code itself is reasonable Streamlit, but as shipped it is an undocumented, undeclared, partially broken extra.

**Fix:** Either commit to it — add an optional-dependency group (`dashboard = ["streamlit"]`), document it in the README, and fix its input paths to the evaluation dir contract — or remove it before artifact submission.

### [medium/methodology] --oracle-fast uses tree_scale 0.35 in the simulation but 0.7 in oracle selection — selection results do not transfer
**Where:** `scripts/bo_sensor_error_simulation.py:661-664` (dimension: code-quality)

build_oracle sets `tree_scale = 0.35` when oracle_fast is on (bo_sensor_error_simulation.py:661-664), while select_best_oracle_model.py:258 sets `tree_scale = 0.7 if args.oracle_fast else 1.0`. With --oracle-fast, the model-selection step benchmarks oracles at twice the capacity the simulation actually trains, so the 'best_model' written to best_oracle_models.json and consumed via --oracle-model auto may not be the best model for the simulation's oracles. The magic scale factors (and per-model n_estimators 600/500/400/800 in _build_oracle_model:742-819) are undocumented.

**Fix:** Define one shared ORACLE_FAST_TREE_SCALE constant in a common module used by both scripts; document the fast-mode capacity reduction in the README and in best_oracle_models.json metadata.

### [medium/paper] README and run_full_workflow.bat describe different experiments; 3 of 7 scripts are undocumented; orchestration is Windows-only
**Where:** `run_full_workflow.bat:1-77` (dimension: code-quality)

The bat file runs select_best_oracle_model with --cv-folds 5 (README recommends 3), omits --iterations (so the simulation runs the 50-iteration default while the README's 'main experiment' uses 100), and adds two whole stages (plot_combined_aspects.py and confirmatory_followup.py with hardcoded pi/logpi acquisitions and seeds 12-31) that the README never mentions. The README's 'What Each Script Does' covers only 4 of 7 scripts — plot_combined_aspects.py, confirmatory_followup.py, and dashboard.py are undocumented — and never mentions run_full_workflow.bat itself. The orchestration is Windows-only batch with no shell counterpart, while CI runs on ubuntu (tests only, never the workflow). The README's output list also omits bo_sensor_error_summary_stats.csv and the evaluation CSVs response_metrics.csv / paired_excess_metrics.csv. For CHI artifact evaluation, reviewers following the README will not reproduce the committed pipeline.

**Fix:** Document every script and the full workflow (including the confirmatory follow-up's role and its hardcoded shortlist) in the README; align cv-folds/iterations between README and bat; provide a cross-platform runner (Makefile, Python orchestrator, or run_full_workflow.sh) and exercise it in CI on a tiny configuration.

### [low/quality] Inconsistent CLI conventions and precedence semantics across the seven scripts
**Where:** `scripts/bo_sensor_error_simulation.py:306-407, 925-952` (dimension: code-quality)

Overlapping flag pairs use different precedence idioms: parse_oracle_models uses `oracle_models if oracle_models is not None else oracle_model` while parse_error_models uses `error_models or error_model` (empty string behaves differently). Multi-value flags have three names across scripts: --acq/--acq-list (simulation), --acquisitions (confirmatory_followup), --shortlist (plot_combined_aspects). Input-dir naming varies: --input-dir (evaluate, plot_sensor_error_results), --evaluation-dir (plot_combined_aspects), --base-input-dir/--base-evaluation-dir (confirmatory). Seed selection is --seeds OR --seed+--num-seeds with the surprising default that a bare run executes 5 seeds (range(seed, seed+5)) and auto-enables multiprocessing (line 2071). select_best_oracle_model duplicates the simulation's argparse block for --data-dir/--dataset-config/--objectives/--normalize-objective/--objective-weights and reimplements parse_oracle_models (lines 52-62) minus auto.

**Fix:** Centralize shared argparse groups (dataset selection, objective selection, oracle selection) in a common module; standardize on one list-flag convention and one precedence rule; document the multi-seed default prominently or default to a single seed.

### [low/quality] Type-hint and docstring defects in load-bearing functions
**Where:** `scripts/select_best_oracle_model.py:146-193` (dimension: code-quality)

_evaluate_model_multi_objective is annotated `-> float` but returns `{"r2": ..., "rmse": ...}` (dict). bo_sensor_error_simulation.run_single_seed annotates `progress_update: callable | None` using the builtin `callable` instead of typing.Callable (line 1786) and `progress_q: object | None` instead of a Queue protocol. OracleModel.model is typed `object` though it is a regressor or list of regressors. Core scientific functions — run_simulation, summarize_adjustment, apply_sensor_error, build_response_table — have no docstrings explaining units, sign conventions (maximization), or the jitter timing contract that the README spells out, so the documented semantics live only in the README and a code comment in evaluate_research_question.py (107-108).

**Fix:** Fix the wrong return annotation, use Callable[[int], None] / queue protocols, type the oracle as a Protocol with fit/predict, and add docstrings stating the jitter_iteration+1 / +2 timing contract where it is implemented (apply_sensor_error, summarize_adjustment, build_response_table).

### [low/quality] Undocumented magic numbers in scientifically meaningful defaults
**Where:** `scripts/bo_sensor_error_simulation.py:1249-1260, 1931-1939` (dimension: code-quality)

compute_reference_point uses `min_vals - 0.1 * ranges` (hypervolume reference) with no justification or citation; --oracle-opt-seed defaults to 10_007; optimize_acqf hardcodes `{"batch_limit": 5}` (line 1524); the jitter RNG seed quantizes jitter_std via `int(round(float(jitter_std) * 1_000_000))` (line 1936); oracle data augmentation defaults to on (--oracle-augmentation jitter, repeats=2, std=0.02, lines 376-383) — a methodology-relevant choice an artifact reviewer would not discover without reading the argparse block; estimate_oracle_optimum approximates y_opt by 200k random samples, which determines all regret values. These all directly shape the reported metrics.

**Fix:** Name these as module-level constants with comments explaining the choice (and cite the BoTorch convention for the 0.1-range reference-point margin); state the y_opt approximation method and the default oracle augmentation in the README's methods notes.

### [low/quality] Minor robustness/performance smells: unsorted glob, iterrows aggregation, import inside plot loop
**Where:** `scripts/plot_sensor_error_results.py:44` (dimension: code-quality)

plot_sensor_error_results.load_iteration_logs uses `list(input_dir.glob(...))` (unsorted, OS-dependent order) while the other three loaders sort — harmless for grouped stats but makes 'identical' runs byte-diff in intermediate CSVs. bo_sensor_error_simulation.main builds the excess summary by iterating merged rows with iterrows (lines 2231-2266) instead of vectorized column arithmetic — slow at the committed scale of ~3,000 runs. plot_combined_aspects.plot_regret_trajectories does `from scipy.stats import t as student_t_dist` inside the per-group loop body (line 285).

**Fix:** Sort the glob, vectorize the excess computation (the per-column subtraction is a one-liner on the merged frame), and hoist the scipy import to module level.

### [critical/reproducibility] Committed results were produced with an environment no requirements file can reproduce
**Where:** `output/run_metadata.json:package_versions block` (dimension: reproducibility)

The committed output/run_metadata.json (last committed 2026-03-10, git 589c9a6) records the environment that generated the 3371 committed result CSVs: numpy 2.3.5, matplotlib 3.10.8, torch 2.10.0+cu128, tabpfn 6.4.1, scikit-learn 1.8.0, botorch 0.17.2. The supposedly-pinned requirements-eval.txt now specifies numpy==2.4.6, matplotlib==3.10.9, torch==2.12.0, tabpfn==8.0.3 — and requirements.txt demands tabpfn>=8.0.8. tabpfn jumped two major versions (6.4.1 -> 8.x) and tabpfn is the resolved oracle for the provoice multi_objective condition per output/run_config.txt ('multi_objective: oracles=[tabpfn]'), so re-running with any current file will not reproduce the committed results. Additionally, the simulation script was last modified 2026-04-12 (commit 0a3c904), a month AFTER the committed results, so the committed CSVs were generated by an older version of the code than what is in the repo.

**Fix:** Regenerate all results with the final code and the exact pinned environment, commit the refreshed run_metadata.json alongside, and freeze requirements-eval.txt at exactly those versions (ideally a full `pip freeze` lockfile, not just 14 top-level packages). Treat any future code or dependency change as invalidating the committed results.

### [high/reproducibility] requirements.txt, requirements-eval.txt and pyproject.toml mutually conflict
**Where:** `requirements.txt:3, 7, 11, 13` (dimension: reproducibility)

The three dependency specifications are incompatible with each other: requirements.txt requires scikit-learn>=1.9.0, tqdm>=4.68.2, tabpfn>=8.0.8, botorch>=0.18.1, but requirements-eval.txt pins scikit-learn==1.8.0, tqdm==4.67.3, tabpfn==8.0.3, botorch==0.17.2 (lines 3, 7, 11, 13 of each file). Installing both files in one resolver run fails. pyproject.toml declares a third, different set of floors (scikit-learn>=1.8.0, tabpfn>=6.4.1, torch>=2.9.1, numpy>=2.4.2) and pytest>=8.3 in [project.optional-dependencies].dev while requirements-dev.txt says pytest>=9.0.3. The recent commit history (2a33056, 02d4509, 2f8652a, c40c14a — four 'update requirements' commits in May 2026) shows the files are being edited independently and drifting apart. It is undefined which of the three specifications is authoritative for the paper.

**Fix:** Designate one source of truth (e.g. pyproject.toml for loose bounds, plus a single generated lockfile such as requirements-eval.txt produced by `pip-compile` or `uv pip compile` from pyproject). Delete or auto-generate the others, and add a CI check that the lockfile is consistent with pyproject.

### [high/reproducibility] Experiment data lives in three external GitHub repos cloned at unpinned HEAD
**Where:** `datasets.json:4, 28, 54` (dimension: reproducibility)

datasets.json points data_dir at https://github.com/M-Colley/ehmi-optimization-chi25-data, https://github.com/M-Colley/opticarvis-data, and https://github.com/JSusak/ProVoiceData (a third-party account). No data is committed to this repo. fetch_remote_dataset (scripts/bo_sensor_error_simulation.py:1038-1048) runs `git clone --depth 1 <url>` — no tag, branch, or commit SHA — and if the cache directory already exists it silently reuses whatever snapshot is there ('if target_dir.exists(): return target_dir'). Two users running the pipeline at different times can therefore silently get different data, and the data repos can be force-pushed, renamed, or deleted at any time. run_metadata.json does not record which commit of the data repos was used, so the data snapshot behind the committed results is unrecoverable.

**Fix:** Pin each dataset to an immutable reference: add a 'ref'/'commit' field to datasets.json and clone with `git fetch <url> <sha> && git checkout <sha>` (or use release tarballs / Zenodo DOIs). Record the resolved data-repo commit SHAs in run_metadata.json, and verify the cached clone matches the pinned SHA instead of blindly reusing it.

### [high/quality] 108 MB of result CSVs are git-tracked while output/ is simultaneously in .gitignore
**Where:** `.gitignore:221` (dimension: reproducibility)

3371 files (108.4 MB) under output/ are tracked in git, yet .gitignore line 221 ignores 'output/'. The tracked files predate the ignore rule, producing a contradictory state: modifications to already-tracked CSVs show up in git status, but any NEW result file (new conditions, new seeds) and all derived evaluation artifacts are silently invisible to git — which is exactly why output/evaluation/ and output/confirmatory/ (the rankings and statistics the paper's claims rest on, per README and run_full_workflow.bat) are absent from the repo while raw per-run CSVs are committed. The committed results are also stale relative to the code (see the run_metadata finding): results committed 2026-03-10, simulation code modified 2026-04-12, dependencies updated through 2026-05-30.

**Fix:** Pick one policy. Either (a) move results out of the code repo into a versioned artifact (GitHub release asset, Zenodo, Git LFS) and let .gitignore stand, or (b) commit the complete, regenerated artifact set — including output/evaluation/ — and remove 'output/' from .gitignore. The current half-tracked state guarantees code/results desynchronization.

### [medium/reproducibility] run_metadata.json omits git commit, Python version, platform, and data-repo SHAs
**Where:** `scripts/bo_sensor_error_simulation.py:2365-2420` (dimension: reproducibility)

The metadata payload records args (vars(args), line 2366), runtime, datasets (resolved local cache paths only), seeds, and importlib.metadata versions of 14 packages (collect_package_versions, lines 1340-1347, called at 2390). It does NOT record: the git commit of this repo's code, the commit SHAs of the cloned dataset repos, sys.version / Python version, OS/platform, or CPU/BLAS info. The committed metadata proves the gap matters: the environment shows 'torch': '2.10.0+cu128' but nothing identifies which code revision or data snapshot produced the run, and the script subsequently changed (commit 0a3c904, 2026-04-12).

**Fix:** Extend metadata_payload with: `subprocess.run(['git','rev-parse','HEAD'])` for the code repo (plus a dirty-tree flag), `git rev-parse HEAD` inside each cached dataset clone, `sys.version`, `platform.platform()`, and ideally the full `importlib.metadata.distributions()` list instead of a hand-picked 14-package subset.

### [medium/reproducibility] README's documented commands cannot reproduce the committed results
**Where:** `README.md:39-56, 199-208` (dimension: reproducibility)

README's 'Recommended Workflow' and 'If You Only Remember One Thing' sections instruct `--iterations 100` and `--cv-folds 3`, but the committed output/run_metadata.json records iterations=50, and run_full_workflow.bat (lines 34-41) passes no --iterations flag (default 50) and uses ORACLE_CV_FOLDS=5 (line 15). So there are three divergent recipes (README, bat file, actual recorded run), and a reviewer following the README will run a different experiment (twice the iterations, different CV) than the one whose outputs are committed. The bat file also runs a confirmatory step (seeds 12-31, shortlist pi,logpi,qucb) that the README never mentions.

**Fix:** Make run_full_workflow.bat (or a cross-platform equivalent script) the single canonical recipe, have the README defer to it, and ensure its defaults exactly match the args recorded in the committed run_metadata.json (iterations, cv-folds, seeds, jitter sweeps).

### [medium/reproducibility] CI tests only ubuntu-latest while the canonical workflow and committed results are Windows
**Where:** `.github/workflows/tests.yml:9-13, 23-30` (dimension: reproducibility)

CI runs on `runs-on: ubuntu-latest` only, Python 3.13/3.14, installing `-r requirements-eval.txt -r requirements-dev.txt` and running `python -m pytest -q`. But the orchestration entry point is a Windows batch file (run_full_workflow.bat), the simulation has a Windows-specific multiprocessing spawn path (bo_sensor_error_simulation.py:2430-2434, `if os.name == 'nt': mp.set_start_method('spawn', force=True)`), and the committed run_config.txt shows Windows backslash paths ('.dataset_cache\ehmi-optimization-chi25-data'), proving the paper results were produced on Windows. The OS actually used for the experiments is never exercised in CI, and no end-to-end tiny-budget pipeline smoke run exists in CI (unit tests only). Note also commit b93313b ('remove catboost (missing Python 3.14 availability)') while catboost==1.2.10 is back in requirements-eval.txt and CI still targets 3.14 — fragile.

**Fix:** Add windows-latest to the matrix (at least for the test job), and add a fast end-to-end smoke job that runs bo_sensor_error_simulation.py with ~2 seeds / few iterations on a tiny local fixture dataset to catch pipeline-level breakage.

### [medium/paper] No licensing or provenance documentation for the three human-subject datasets
**Where:** `README.md:186-197` (dimension: reproducibility)

LICENSE is MIT and covers only 'the Software'. The README 'Datasets' section says only that datasets.json defines 'data location' — it never states what the ehmi, opticarvis, and provoice datasets are, who collected them, under what license/ethics approval they are distributed, or how to cite them. One repo (JSusak/ProVoiceData) belongs to a different GitHub account entirely. There is no CITATION.cff and no data-license statement anywhere in the repo. For a CHI artifact built on human-participant data, undocumented data provenance and licensing is a review-blocking gap.

**Fix:** Add a 'Data provenance' README section per dataset: original paper citation, collection context, license of the data repo, and ethics/consent statement reference. Add CITATION.cff for the artifact itself and verify the third-party ProVoiceData repo's license permits redistribution/reuse.

### [low/reproducibility] Python version only loosely bounded; version used for results not recorded
**Where:** `pyproject.toml:5` (dimension: reproducibility)

`requires-python = ">=3.13"` has no upper bound, CI tests both 3.13 and 3.14, the README never states a Python version, and run_metadata.json does not record sys.version. The exact interpreter used to produce the committed results is therefore unknown, and future Python releases are implicitly claimed to be supported.

**Fix:** Declare `requires-python = ">=3.13,<3.15"` (or similar), state the exact version used for the paper runs in the README, and record sys.version in run_metadata.json.

### [low/methodology] Jitter noise stream is shared across datasets, objectives, and oracle models
**Where:** `scripts/bo_sensor_error_simulation.py:1931-1940` (dimension: reproducibility)

The jitter RNG is seeded via np.random.SeedSequence([seed, ACQUISITION_CHOICES.index(acq.name), jitter_iteration, round(jitter_std*1e6), ERROR_MODEL_CHOICES.index(error_model)]) — the dataset name, objective, and oracle model are not part of the entropy. Consequently the exact same error-noise draw sequence is injected for, e.g., ehmi/composite and provoice/composite at the same (seed, acq, jit, std, error_model), correlating the 'independent' dataset conditions. This is fully deterministic (good) but the cross-dataset error realizations are not independent samples, which slightly weakens any claim that robustness rankings replicate across datasets.

**Fix:** Include stable integer identifiers for dataset, objective, and oracle model in the SeedSequence entropy list (e.g. indices into sorted name lists recorded in metadata), so each condition gets an independent noise stream while remaining reproducible.

### [low/quality] Default cache and oracle-selection paths are CWD-relative
**Where:** `scripts/bo_sensor_error_simulation.py:111, 336-340` (dimension: reproducibility)

DEFAULT_ORACLE_SELECTION_PATH = Path("output") / "best_oracle_models.json" (line 111) and --dataset-cache-dir default Path(".dataset_cache") (lines 336-340) resolve relative to the current working directory, unlike DEFAULT_DATASET_CONFIG_PATH which is anchored to REPO_ROOT (line 110). Running the scripts from any directory other than the repo root silently creates a second dataset cache (triggering a fresh unpinned clone, possibly of different data) and fails to find the oracle selection file. The committed run_config.txt's relative '.dataset_cache\...' paths also mean the recorded data location is ambiguous.

**Fix:** Anchor both defaults to REPO_ROOT like DEFAULT_DATASET_CONFIG_PATH, and write absolute (or repo-root-relative, explicitly labeled) paths into run_config.txt/run_metadata.json.

### [low/paper] Docstring claims '95% bootstrap CI' but code computes a t-distribution CI
**Where:** `scripts/plot_combined_aspects.py:218-287` (dimension: reproducibility)

plot_regret_trajectories' docstring says 'A shaded band represents the 95 % bootstrap CI across seeds' (line 222), but the implementation computes mean ± t_crit * SE using scipy.stats.t.ppf(0.975, df=count-1) (lines 283-287) — a parametric t-interval, not a bootstrap. If the figure captions or methods section of the CHI paper are written from this docstring, the statistical method will be misreported.

**Fix:** Fix the docstring to 'parametric 95% t-distribution CI across seeds' (or actually implement a seeded bootstrap if that is what the paper claims), and audit figure captions against the code before submission.

### [high/quality] No end-to-end smoke test: no script main() is ever executed, despite the file being named test_cli_smoke.py
**Where:** `tests/test_cli_smoke.py:1-172` (dimension: tests)

test_cli_smoke.py only tests parse_args round-trips and three pure helper functions; it never runs bo_sensor_error_simulation.py, evaluate_research_question.py, or any plot script end-to-end, not even with a tiny config. evaluate_research_question.main() (load_iteration_logs -> build_response_table -> build_paired_table -> compute_condition_rankings -> write_report) and the simulation's main() (including the --baseline-run excess-summary branch at scripts/bo_sensor_error_simulation.py:2210-2320) have zero integration coverage. Column-schema drift between writer and readers, CSV separator issues, or a broken --baseline-run branch would pass CI silently. The 'smoke' name gives false confidence.

**Fix:** Add a true smoke test: run bo_sensor_error_simulation.py via subprocess (or by calling main() with monkeypatched sys.argv) on a synthetic 2-parameter dataset with iterations=6, initial_samples=3, 2 seeds, acquisitions 'greedy,ei', error-model gaussian, --baseline-run, writing to tmp_path; assert the per-run CSVs, bo_sensor_error_summary.csv and bo_sensor_error_excess_summary.csv exist with expected row counts; then run evaluate_research_question.py on that output dir and assert condition_rankings.csv/overall_rankings.csv are non-empty and contain both acquisitions.

### [high/quality] Jitter-timing (README t+1/t+2 claim) only unit-tested on apply_sensor_error, not through run_simulation
**Where:** `tests/test_scientific_correctness.py:79-122` (dimension: tests)

TestErrorModelBoundary calls bo_sim.apply_sensor_error directly with hand-chosen iteration numbers, so it tests the function's own convention against itself. An off-by-one in the caller (scripts/bo_sensor_error_simulation.py:1577 'for iteration in range(1, ...)' passing 'iteration' at line 1652-1658) would not be caught — e.g. switching to enumerate(0-based) or passing len(X_list) would shift error onset by one iteration and silently invalidate the paper's central timing claim (README lines 119-121). The error_applied column is also re-derived independently at lines 1705-1711 rather than from the actual injected error, and no test checks that the column agrees with where observations actually diverge.

**Fix:** Add an integration test: run run_simulation with apply_error=True, a deterministic FixedOracle, jitter_std large (e.g. 10.0), jitter_iteration=k, initial_samples=iterations (greedy, no GP). Assert objective_observed == objective_true exactly for iteration <= k, objective_observed != objective_true for iteration k+1..n, and that results['error_applied'] is True exactly where error_magnitude != 0 (and matches iteration > k). Repeat with single_error=True asserting only row k+1 diverges.

### [high/quality] AUC of simple regret (the paper's PRIMARY_METRIC building block) has no numeric test
**Where:** `scripts/evaluate_research_question.py:137-139` (dimension: tests)

auc_simple_regret_true is computed with np.trapezoid(sr, dx=1.0) in two independent places: evaluate_research_question.build_response_table (line 137-139) and bo_sensor_error_simulation.summarize_adjustment (line 1766). No test asserts the value: TestBuildPairedTable injects auc_regret values by hand, and test_summarize_adjustment_includes_avg_regret only checks final_avg_regret_true. Trapezoid half-weights the endpoints (sum vs trapezoid differ), so a silent change to np.sum or a different dx would alter every excess-regret ranking with no test failure. PRIMARY_METRIC = 'auc_simple_regret_excess_true' (line 30) is built directly on this.

**Fix:** Add a test feeding a known simple_regret series, e.g. [3,2,1] -> expected trapezoid AUC 4.0 (= 3/2 + 2 + 1/2), through both build_response_table and summarize_adjustment, asserting both return the same value. Also assert the two implementations agree on a randomized series to guard against divergence.

### [high/quality] compute_condition_rankings / compute_overall_rankings (the paper's headline acquisition ranking) are completely untested
**Where:** `scripts/evaluate_research_question.py:208-325` (dimension: tests)

The functions that produce condition_rankings.csv and overall_rankings.csv — seed-wise rank pivot (line 219-230), Friedman test for k>2 acquisitions (line 263), Kendall's W = stat/(n*(k-1)) (line 264), Wilcoxon fallback for k=2 (line 274-291), FDR over condition tests (line 295-304), and 'condition_win' tie handling via math.isclose (line 251) — have zero tests. These outputs are the research question's answer. A sign flip in 'ascending=True' ranking (lower excess regret should rank better), the dropna(axis=0, how='any') seed-filtering at line 225, or a wrong Kendall's W denominator would be invisible.

**Fix:** Add tests with a hand-built paired DataFrame: (1) 3 acquisitions x 4 seeds with a strict dominance ordering -> assert mean_rank ordering, best_acquisition, condition_win flags, and Friedman p < 0.05 with Kendall's W close to 1; (2) 2 acquisitions -> assert test_name == 'wilcoxon'; (3) a seed missing one acquisition -> assert that seed is dropped from the pivot (n_seeds reflects it); (4) assert compute_overall_rankings averages mean_rank correctly across conditions.

### [high/quality] confirmatory_followup.py statistical functions have zero test coverage
**Where:** `scripts/confirmatory_followup.py:115-249` (dimension: tests)

No test file references confirmatory_followup (grep for 'confirmatory' across tests/ returns nothing). paired_stats_from_diff (line 115), build_head_to_head_table (line 160), summarize_condition_tests with its own FDR application (line 176-213), summarize_dataset_tests (line 216), and the LaTeX table writer write_main_confirmatory_table (line 311) — i.e. the confirmatory statistics that will appear verbatim in the paper — are untested. This is a parallel, independent implementation of t-test/Wilcoxon/FDR from the one in plot_sensor_error_results (which IS well tested), so existing tests provide no protection here.

**Fix:** Add tests/test_confirmatory_followup.py mirroring TestPairedTestFromDiff: assert paired_stats_from_diff matches scipy ttest_1samp/wilcoxon on known diffs and handles all-zero/single-element inputs; assert build_head_to_head_table subtracts reference-minus-challenger with correct seed alignment on a small fabricated paired table; assert summarize_condition_tests' FDR-adjusted p-values >= raw p-values and that _format_p_value/_latex_escape produce expected strings.

### [medium/quality] test_fdr_correction_applied_per_metric is effectively tautological
**Where:** `tests/test_scientific_correctness.py:343-353` (dimension: tests)

The test's name and docstring claim 'FDR is applied within each metric group independently', but the only assertion is result['p_value_fdr_bh'].notna().all() — it would pass whether FDR is applied per-metric, globally, or not at all (e.g. if the function copied raw p-values into the new column). The comment in the test even acknowledges the chosen p-values cannot distinguish the behaviors.

**Fix:** Use p-values where per-metric and global BH differ numerically, e.g. metric 'a': [0.01, 0.02, 0.03], metric 'b': [0.04]; per-metric BH gives a-group adjusted [0.03, 0.03, 0.03] and b-group [0.04], while a single global BH over 4 values gives [0.04, 0.04, 0.04, 0.04]. Assert the exact per-metric values.

### [medium/quality] Filename round-trip between simulation writer and downstream loaders is untested
**Where:** `tests/test_cli_smoke.py:109-128` (dimension: tests)

The simulation writes 'bo_sensor_error_{dataset}_{objective}_{acq}_seed{seed}_baseline_{oracle_model}.csv' / '..._jittered_...' (scripts/bo_sensor_error_simulation.py:1893, 1967); both plot_sensor_error_results.load_iteration_logs (line 44) and evaluate_research_question.load_iteration_logs (line 43) glob 'bo_sensor_error_*_seed*_*.csv'. The only test of the loader (test_plot_loader_and_final_row_summary) fabricates its own filename 'bo_sensor_error_demo_seed7_baseline.csv', so a change to the writer's naming (e.g. dropping the prefix or the 'seed' token) would break the pipeline while all tests pass. evaluate_research_question raises FileNotFoundError on mismatch, but only at runtime on the user's machine.

**Fix:** Add a round-trip test that constructs the filename with the same f-string logic the writer uses (ideally extract a build_run_filename(dataset, objective, acq, seed, baseline, oracle_model) helper used by main()) for both baseline and jittered variants, writes minimal CSVs under tmp_path with those names, and asserts both loaders' globs discover all of them and none extra.

### [medium/quality] Oracle-selection JSON schema round-trip between producer and consumer is untested
**Where:** `tests/test_simulation_utils.py:262-292` (dimension: tests)

test_load_and_resolve_oracle_selection fabricates best_oracle_models.json by hand and tests only the consumer (bo_sim.load_oracle_selection / resolve_oracle_models_for_objective). The producer side in select_best_oracle_model.py — that it writes the same {'datasets': [{'name', 'objectives': {obj: {'best_model', 'scores'}}}]} structure, and that best_model is actually the argmax of the scores dict — is never tested. If the writer's schema or the argmax logic drifted, '--oracle-model auto' would silently pick wrong or fail at runtime, and no test would notice.

**Fix:** Add a test that calls select_best_oracle_model's selection/serialization function (or runs its main on the tiny grouped DataFrame already used in test_select_best_oracle_model_uses_grouped_validation_for_user_ids with two models) to produce the JSON file, then feeds that exact file to bo_sim.load_oracle_selection and asserts resolve_oracle_models_for_objective(['auto'], ...) returns the model with the higher score.

### [medium/quality] infer_param_columns has no schema-sync test against real simulation output
**Where:** `scripts/evaluate_research_question.py:53-88` (dimension: tests)

infer_param_columns treats any column not in a hard-coded reserved set (lines 54-77) as a parameter column. If run_simulation ever adds a metadata column (it already appends ~15 such columns at scripts/bo_sensor_error_simulation.py:1694-1731) without updating the reserved set, the new column is silently included in response_l2 = ||params_end - params_start||, corrupting the reaction metric for every run. Current tests only exercise infer_param_columns indirectly via _make_log, which by construction contains exactly the reserved columns plus p1/p2 — so the test and the reserved set can never disagree.

**Fix:** Add a contract test: run run_simulation once (tiny, greedy, initial_samples=iterations) for both 'composite' and 'multi_objective', then assert infer_param_columns(results) == config.param_columns exactly. This fails the moment a new output column is added without updating the reserved set.

### [medium/quality] Multi-objective BO path (ModelListGP + qEHVI/qNEHVI) is never executed by tests
**Where:** `tests/test_simulation_utils.py:498-569` (dimension: tests)

All run_simulation tests set initial_samples == iterations, so the GP-fitting branch (scripts/bo_sensor_error_simulation.py:1581-1642) never runs; multi-objective coverage is limited to hypervolume regret bookkeeping with a SequenceOracle and 'greedy'. The ModelListGP construction (lines 1585-1604), SumMarginalLogLikelihood fitting, and the qehvi/qnehvi branches of get_botorch_candidate are untested — yet 'multi_objective' is a default objective (test_parse_objective_list_defaults_to_composite_and_multi asserts it). A botorch API change or a ref_point shape bug there would only surface in a full research run. Note the broad except at line 1638-1640 silently falls back to random sampling on any acquisition failure, so even a fully broken acquisition would produce plausible-looking CSVs.

**Fix:** Add one slow-marked but small test: run_simulation with iterations=5, initial_samples=3, objective='multi_objective', acq='qehvi' (and one single-objective acq like 'ei'), tiny acq settings, asserting it completes with no warnings.warn fallback (use pytest.warns/recwarn to assert the 'BoTorch optimization failed' warning is NOT emitted) and produces finite hypervolume columns.

### [low/quality] Heavy module loaded 4+ times via importlib at collection time; no conftest.py
**Where:** `tests/test_scientific_correctness.py:23-36` (dimension: tests)

Each test file re-executes the 2144-line bo_sensor_error_simulation.py (which imports torch/botorch at module top) under a different name: 'bo_sim' (test_simulation_utils.py:16-20), 'bo_sim_sci' (test_scientific_correctness.py:34), and twice more in test_cli_smoke.py (lines 40, 66), plus plot/eval modules twice. This slows collection, creates duplicate class identities (bo_sim.SimulationConfig is not bo_sim_sci.SimulationConfig — a latent isinstance hazard), and any import failure aborts collection of an entire file rather than failing one test. test_cli_smoke.py additionally inserts SCRIPTS_DIR into sys.path (lines 11-14) even though it then bypasses it with spec_from_file_location. The mechanism does work cross-platform (absolute pathlib paths, testpaths set in pyproject), so this is robustness/speed debt rather than breakage.

**Fix:** Add tests/conftest.py with session-scoped fixtures (or a single module-level loader) that load each script exactly once under one canonical name and share it across test files; remove the redundant sys.path manipulation. Longer term, move reusable logic into an installable package (pyproject already exists) and import normally.

### [low/reproducibility] CI installs the full GPU-capable torch stack on every run with no cache, no timeout, and never runs evaluate/confirmatory entry points
**Where:** `.github/workflows/tests.yml:8-31` (dimension: tests)

The workflow installs requirements-eval.txt (torch==2.12.0, botorch, catboost, tabpfn — multi-GB) on every push/PR for two Python versions with no pip cache (setup-python 'cache: pip' is absent) and no timeout-minutes, so most CI wall time is dependency download. It runs only 'python -m pytest -q', which (per the gaps above) executes no script end-to-end. Tests run only on ubuntu-latest while development happens on Windows (run_full_workflow.bat), so Windows-specific path/encoding regressions in the scripts are never exercised in CI. The tests themselves are seeded and deterministic (fixed default_rng seeds throughout), which is good — flakiness is not the issue; coverage and cost are.

**Fix:** Add 'cache: pip' to actions/setup-python (keyed on requirements-eval.txt + requirements-dev.txt), set timeout-minutes on the job, install CPU-only torch via the cpu index to cut download size, and add a windows-latest entry (single Python version) to the matrix. Once the end-to-end smoke test exists, it runs automatically under this job.

### [low/quality] plot_combined_aspects.py and dashboard.py have zero coverage, including their data-aggregation logic
**Where:** `scripts/plot_combined_aspects.py:` (dimension: tests)

No test imports plot_combined_aspects.py or dashboard.py (grep across tests/ returns nothing). While pure plotting is low priority, these scripts contain their own loading/aggregation of the result CSVs; if their groupby/merge logic disagrees with evaluate_research_question.py, figures in the paper could contradict the reported statistics. plot_sensor_error_results.py is the only plotting module with (good) tests of its statistical helpers.

**Fix:** Add at least an import-and-run test for each: load the module via the existing _load helper, feed it the tiny end-to-end output directory from the new smoke test (or fabricated frames matching the writer schema), call its top-level aggregation/plot functions with a tmp_path output dir, and assert the expected PNG/CSV files exist and aggregated values match a hand-computed mean for one cell.

### [critical/paper] "HITL" framing overclaims: no humans anywhere in the loop, and the code itself says "sensor error"
**Where:** `README.md:1-13` (dimension: chi-readiness)

The repo title and README frame this as Human-in-the-Loop MOBO robustness, but the "human" is a tree-ensemble oracle (default extra_trees, scripts/bo_sensor_error_simulation.py line 365) fit on archival study data, and the perturbation is literally named 'sensor error' throughout (script name bo_sensor_error_simulation.py, apply_sensor_error(), output files bo_sensor_error_*.csv). Reviewer 2 will write: 'This is a simulation study of Bayesian optimization under observation noise on surrogate benchmarks derived from HCI datasets. No human provided feedback; no human behavior was modeled or validated. Calling it HITL is misleading.' The sensor-error vocabulary in the artifacts will be quoted back at the authors as evidence the human framing was retrofitted.

**Fix:** Reframe the contribution honestly as 'simulation-based robustness analysis of preference-driven BO with surrogate users derived from three HCI studies', and explicitly position the noise models as approximations of human rating error with citations to psychometrics (test-retest reliability, intra-rater consistency, response styles). Strongest fix: add a human-grounded validation — the source datasets (ObservationsPerEvaluation.csv per participant) contain repeated human ratings of similar designs; estimate the empirical within-participant rating SD and show the simulated noise levels bracket it. A small confirmatory human study (even N=12 re-rating designs twice) would convert this from 'simulation only' to 'empirically calibrated simulation' and defuse the main rejection risk.

### [medium/methodology] The 'bias' error model is confounded with gaussian noise, and bias magnitude/sign are never varied
**Where:** `scripts/bo_sensor_error_simulation.py:1366-1369` (dimension: chi-readiness)

The bias model returns true_value + error_bias + N(0, jitter_std): a fixed +0.2 offset (default, line 348, never swept) plus the SAME gaussian jitter as the gaussian model, with jitter_std as the swept factor. Consequently at std=0.5/1/5 the 'bias' condition is statistically almost identical to the 'gaussian' condition (a 0.2 constant buried under noise of 2.5–25x its size), so the two 'error models' in the headline comparison are not actually distinct treatments except at std=0.05. Additionally only a positive (inflating) bias is tested, and a constant shift applied to all post-onset observations is the mildest possible systematic error for a GP with standardized outcomes.

**Fix:** Make bias a pure, swept factor: bias in {-1, -0.5, +0.5, +1} (scale-relative) with jitter_std=0, crossed separately with noise. Better still, model human-plausible systematic errors: drift (bias growing over iterations) and scale compression (responses pulled toward the midpoint, i.e., multiplicative not additive), which interact with BO very differently from a constant offset.

### [medium/methodology] Whole-run AUC confounds noise-onset timing with noise-exposure duration
**Where:** `scripts/evaluate_research_question.py:137-139` (dimension: chi-readiness)

auc_simple_regret_true is np.trapezoid over ALL 50 iterations (line 137–139). For jitter_iteration=40 only the last ~10 iterations can differ from baseline, versus ~40 for jitter_iteration=10, so excess AUC is mechanically smaller for late onsets regardless of any robustness property. Within-condition rankings are unaffected (onset is part of the condition key), but any cross-onset statement in the paper — e.g., 'late-onset noise is less harmful' or heatmaps aggregating over jitter_iteration — is an artifact of integration-window length, not of BO dynamics.

**Fix:** Additionally compute excess AUC restricted to the post-onset window (iterations jitter_iteration+1..50) and/or normalized per noisy iteration, and use that for any across-onset comparison or aggregation. Keep the full-run AUC for within-condition rankings if desired, but state the confound explicitly.

### [high/paper] Single application domain (automotive UX), and only 2 of the 3 configured datasets were actually run
**Where:** `datasets.json:1-69` (dimension: chi-readiness)

All three datasets are automated-vehicle UX (ehmi: external HMI design; opticarvis: AR visualization in AVs; provoice: voice/intervention in autonomous driving), two from the same research group, and ehmi/opticarvis share the identical construct set (Trust, Understanding, PerceivedSafety, Aesthetics, Acceptance). Worse, the committed output/ contains runs only for ehmi and opticarvis (composite: 50 file-combos each; multi_objective: 10 each) — provoice, the only dataset with different constructs and a different group, has zero runs. Claims about 'HITL MOBO robustness' generalize from N=2 highly similar automotive surrogate benchmarks with 9–16 design parameters.

**Fix:** Run provoice (it is already configured), and add at least one non-automotive HITL optimization dataset — candidates with public data include haptics/vibrotactile preference tuning, font/typography legibility, web-UI or visualization parameter studies, or game-difficulty tuning. If broadening is infeasible, scope every claim to 'BO-driven design optimization in automated-vehicle UX' and state the single-domain limitation prominently; do not title the paper generically.

### [medium/methodology] Oracle as ground-truth human is a pooled average user with unreported fidelity
**Where:** `scripts/bo_sensor_error_simulation.py:642-820` (dimension: chi-readiness)

The 'true' objective is the prediction of a tree ensemble (default extra_trees) fit on data pooled across all participants of the source studies, lightly augmented with jitter (oracle_augment_std=0.02). This bakes in three issues: (1) the simulated human is an average user, so the inter-individual variability that motivates HITL personalization is absent — only intra-rater noise is simulated; (2) tree ensembles give piecewise-constant response surfaces with plateaus, which interact pathologically with GP-based acquisitions (zero gradients, ties in EI) in ways a real human utility surface would not; (3) the paper's regret is regret against this oracle, so oracle misfit silently redefines the optimum, yet no oracle accuracy (CV R^2/RMSE from select_best_oracle_model.py) appears in the planned outputs.

**Fix:** Report oracle cross-validated accuracy per dataset/objective in the paper. Run a sensitivity check with a smooth oracle (the GP-free alternative: gradient boosting vs a smoothed model) to show rankings are stable — the multi-oracle infrastructure already exists (--oracle-models). For the HITL story, fit per-participant or per-cluster oracles (user_id/group_id hooks already exist in the CLI, lines 353–354) and show robustness conclusions hold across simulated individual users, not just the average.

### [medium/paper] Reporting gaps for a quantitative CHI paper: no CIs/effect sizes on the main grid, no preregistered hypotheses, no power justification
**Where:** `scripts/evaluate_research_question.py:189-306` (dimension: chi-readiness)

The main evaluation outputs mean/median/std of excess regret and FDR-corrected Friedman/Wilcoxon p-values, but no confidence intervals or effect sizes at the condition level (Kendall's W is computed but buried in a CSV). Effect sizes (Cohen's dz) and 95% CIs exist only in the legacy plot script and in confirmatory_followup.py, and there only for the single pi-vs-logpi head-to-head. There is no written hypothesis statement anywhere in the repo — the 'research question' is 'which parts work how well', which a CHI AC will read as a fishing expedition. The exploratory/confirmatory split is actually a strength of this pipeline, but nothing documents which hypotheses the confirmatory phase was meant to test or why 20 seeds suffices.

**Fix:** Write explicit, falsifiable hypotheses (e.g., 'H1: log-variant acquisitions show lower excess AUC regret than their plain counterparts under gaussian noise >= 1 SD') and preregister the confirmatory phase (OSF or AsPredicted), including the seed range, metric, and test. Add bootstrap 95% CIs on mean excess regret per condition and report Kendall's W alongside every Friedman test in the paper. Justify replicate count with a power analysis using the seed-level variance already in output/.

### [medium/paper] No deployment-relevant or practically interpretable outcome measures for an HCI audience
**Where:** `scripts/evaluate_research_question.py:30-33` (dimension: chi-readiness)

The metric stack (excess AUC simple regret, excess final simple regret, response-step L2) is pure optimization-theory vocabulary in oracle units. CHI reviewers will ask: what does an excess AUC regret of 2.3 mean for a designer? The quantities practitioners care about are absent: quality of the design actually recommended at budget exhaustion in instrument units ('0.4 Likert points lower Trust'), extra human queries needed to recover baseline quality (sample-efficiency cost of noise, directly translatable to participant time and money), and probability of recommending a near-optimal design (P(simple regret < epsilon)). Without these, the practical-significance section of the paper cannot be written from the current outputs.

**Fix:** Add three derived metrics to the evaluation: (1) final recommended-design quality expressed in raw scale points and as % of achievable range; (2) 'iterations-to-match': post-onset iterations needed for the noisy run to reach the baseline's final best (capped at budget) — this converts robustness into human effort; (3) success rate at epsilon = 0.1 and 0.25 scale points. Lead the paper's results with these, keep AUC excess regret as the technical ranking metric.

### [medium/paper] MOBO claim rests on a two-method comparison evaluated by a scalarized proxy
**Where:** `scripts/bo_sensor_error_simulation.py:163, 1669-1677` (dimension: chi-readiness)

The title promises Multi-Objective BO robustness, but the multi_objective condition compares only qEHVI vs qNEHVI (MULTI_ACQUISITION_CHOICES, line 163; 10 run-combos per dataset in output/), evaluated via hypervolume regret stored in the same simple_regret_true column. Most of the experimental mass (50 of 60 combos per dataset) is the 'composite' condition — an unweighted mean scalarization of five constructs (compute_objective, line 550), which is single-objective BO, not MOBO. A reviewer will note that (a) an equal-weight mean of Trust/Aesthetics/etc. is itself a strong unvalidated assumption about user utility, and (b) two methods cannot support general MOBO robustness claims.

**Fix:** Either expand the MO arm (add qNParEGO, random scalarization, and a random baseline with hypervolume tracking) and report Pareto-front quality metrics beyond hypervolume (e.g., IGD+), or retitle/reframe around scalarized preference optimization with an MO supplementary analysis. Also justify or vary the equal weighting of the five constructs (a weight-perturbation ablation would be cheap and convincing).

## Critic summary
The six reviewers never opened the committed output corpus or the BoTorch modeling layer, and both hide paper-killing problems. The committed artifacts are internally contradictory: run_metadata.json (March 2026) claims opticarvis ran under gradient_boosting while all ~1,500 committed opticarvis CSVs (January 2026) used extra_trees, and the committed best_oracle_models.json shows the 'ground-truth human' for ehmi/provoice has near-zero or negative grouped-CV R^2 while opticarvis shows a machine-precision (R^2=0.9999999, RMSE~1e-8) leakage/duplication signature — none of which any reviewer read despite the file being 200 lines of JSON. On the modeling side, nobody audited the two incumbent definitions: improvement acquisitions use the max of noise-corrupted observations as best_f (confounding the robustness ranking, and structurally favoring qNEI which uses X_baseline), while regret uses an omniscient true-value incumbent that makes the final recommendation immune to noise — together these bracket the paper's central quantity from both sides. Spot-checks of committed CSVs also confirmed that the search space is the box hull of a prior BO campaign's trace (some dimensions ~0.1 wide) and that the zero-regret clamp actually binds in 22% of opticarvis baselines, censoring the primary metric. Classes of bugs left unhunted include data/metadata consistency checks on the committed corpus, GP/acquisition implementation review against noisy-BO best practice, and design-space/coverage validity.
