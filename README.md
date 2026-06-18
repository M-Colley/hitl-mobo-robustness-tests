# HITL MOBO Robustness Tests

This repository simulates Bayesian optimization under noisy human feedback.

It is meant to answer a simple question:

- Which optimization choices work best for which dataset and error level?

The "human" in the loop is simulated: a regression oracle fit on archival
study data stands in for participants, and feedback errors are injected on
top of its predictions. Conclusions therefore describe robustness on
data-derived synthetic test functions; report the oracle's cross-validated
fidelity (written to `output/best_oracle_models.json` and echoed into
`run_metadata.json`) alongside any claims.

In practice, the workflow is:

1. Choose or benchmark the oracle model.
2. Run the BO robustness simulation.
3. Evaluate the results with simple rankings, excess-regret summaries, and plots.

## What Each Script Does

- `scripts/select_best_oracle_model.py`
  - Benchmarks candidate oracle models on the dataset and writes the best choice to `output/best_oracle_models.json`.
  - Audits duplicated rows (leakage), warns on silently downgraded CV folds, and flags implausibly perfect or near-zero scores.
- `scripts/bo_sensor_error_simulation.py`
  - Runs baseline and noisy BO simulations and writes per-run CSV logs plus summary files.
- `scripts/evaluate_research_question.py`
  - The simplest recommended evaluation step. It ranks acquisitions by robustness (excess regret) AND absolute noisy performance for each dataset and error condition, with per-family FDR correction and paired effect sizes.
- `scripts/confirmatory_followup.py`
  - Runs a preregistration-style confirmatory comparison on fresh seeds (screening seeds are excluded from confirmatory tests).
- `scripts/plot_combined_aspects.py`
  - Paper-ready combined figures from the evaluation outputs.
- `scripts/plot_sensor_error_results.py`
  - Extra legacy plots and paired final-outcome tests. Useful, but not the main workflow anymore.
- `scripts/dashboard.py`
  - Optional Streamlit dashboard over the evaluation outputs (`pip install streamlit`).

## Recommended Workflow

### 1. Install

```bash
python -m pip install --upgrade -r requirements-eval.txt
```

### 2. Pick the Oracle Model

If you already know which oracle model you want, skip this step.

```bash
python scripts/select_best_oracle_model.py \
  --oracle-models all \
  --cv-folds 3 \
  --output-path output/best_oracle_models.json
```

### 3. Run the Simulation

This is the main experiment.

```bash
python scripts/bo_sensor_error_simulation.py \
  --iterations 50 \
  --acq all \
  --oracle-model auto \
  --oracle-selection-path output/best_oracle_models.json \
  --output-dir output
```

(Use the same `--iterations` everywhere; the evaluation step refuses to mix
run lengths. `run_full_workflow.bat` uses the default of 50.)

What this does:

- loads the datasets from `datasets.json`
- trains the oracle model for each dataset/objective
- runs BO with the requested acquisition functions, including the model-free
  `random` and `sobol` floors
- runs both a clean baseline and noisy versions
- writes per-iteration logs and summary CSVs to `output/`

Important methodological defaults:

- Improvement-based acquisitions (EI/PI families) use a posterior-mean
  incumbent (`--incumbent posterior_mean`), so a single positive noise spike
  cannot stall them; pass `--incumbent observed_max` for the classic
  noise-naive variant.
- Per-iteration logs include `inference_simple_regret_true`: the true value of
  the design the experimenter would pick from the noisy observations. This is
  the deployment-relevant recommendation quality; plain `simple_regret_true`
  assumes the best design is recognized for free.
- Regret is NOT clamped at zero: `y_opt` is a sampling-based estimate
  (anchored on the training data) that BO can legitimately exceed.
- Use `--jitter-iterations 0,...` to include the human-plausible
  "noisy from the first observation" condition, `--response-clip auto
  --response-round <scale step>` to keep noisy ratings on the instrument
  scale, and the `drift` / `ar1` error models for systematic and serially
  correlated human error.
- Any acquisition-optimization failure falls back to random sampling and is
  recorded per iteration in the `acq_opt_failed` column (and summed in the
  summaries) — check it before comparing acquisitions.

### 4. Evaluate the Results

This is the easiest way to answer the research question.

```bash
python scripts/evaluate_research_question.py \
  --input-dir output \
  --output-dir output/evaluation
```

Start by opening these files:

- `output/evaluation/evaluation_report.txt`
- `output/evaluation/overall_rankings.csv`
- `output/evaluation/condition_rankings.csv`
- `output/evaluation/mean_rank_*.png`
- `output/evaluation/mean_excess_auc_*.png`

## What the Evaluation Means

The recommended evaluation is based on **excess regret**:

- baseline run = no sensor error
- jittered run = same condition, but with noisy feedback
- excess regret = `jittered - baseline`

Interpretation:

- lower excess regret is better
- negative excess regret means the noisy run happened to do better than baseline
- acquisitions are ranked separately for each dataset, error model, error size, and jitter start

### Recommended Metrics

Report BOTH of these — excess regret alone can crown an acquisition that is
uniformly bad in baseline and noisy conditions alike (the `random`/`sobol`
floors make this visible):

1. `auc_simple_regret_excess_true` — robustness (lower is better, comparable
   only within one condition; for across-onset comparisons use
   `auc_simple_regret_excess_true_postonset_per_iter`).
2. `mean_auc_simple_regret_true_jitter` — absolute performance under noise
   (the deployment question: which method actually finds good designs).

Then:

3. `final_inference_simple_regret_excess_true` — extra cost of noise on the
   design the experimenter would actually recommend.
4. `final_simple_regret_excess_true` — endpoint summary.
5. `response_l2_excess` — diagnostic only; how much the first error-affected
   recommendation changes.

Statistical caveats baked into the outputs: tests are FDR-corrected per test
family, Wilcoxon cells report their minimum attainable p (n=5 seeds cannot
reach p<0.05 — use >=20 seeds for confirmatory claims), Kendall's W
accompanies Friedman tests, and `effect_sizes_cohens_dz.csv` carries paired
effect sizes with 95% CIs.

## Important Note About the "Immediate Adjustment" Metric

The summary files in `output/` include `delta_l2_norm` and `delta_excess_l2_norm`.
Those are diagnostic, not primary, metrics.

Timing convention (used consistently by the simulator's summaries and by
`scripts/evaluate_research_question.py`):

- noise starts affecting the observed signal at iteration `jitter_iteration + 1`
- the first candidate that can react to that noisy observation is at iteration `jitter_iteration + 2`
- so the "response step" is from `t+1` to `t+2`

## Most Important Output Files

### Raw simulation outputs

Written by `scripts/bo_sensor_error_simulation.py`:

- `output/bo_sensor_error_<dataset>_<objective>_<acq>_seed<seed>_baseline_<oracle>.csv`
- `output/bo_sensor_error_<dataset>_<objective>_<acq>_seed<seed>_jittered_<oracle>_<error>_jit<iter>_std<std>.csv`
- `output/bo_sensor_error_summary.csv`
- `output/bo_sensor_error_excess_summary.csv`
- `output/bo_sensor_error_dataset_effects.csv`
- `output/run_metadata.json`
- `output/run_config.txt`

### Simplest evaluation outputs

Written by `scripts/evaluate_research_question.py`:

- `output/evaluation/evaluation_report.txt`
  - Human-readable summary.
- `output/evaluation/overall_rankings.csv`
  - Best acquisitions overall for each dataset/objective.
- `output/evaluation/condition_rankings.csv`
  - Rankings for each dataset/error condition.
- `output/evaluation/condition_summary.csv`
  - Mean and median excess metrics per condition.
- `output/evaluation/mean_rank_*.png`
  - Heatmaps of acquisition ranks.
- `output/evaluation/mean_excess_auc_*.png`
  - Heatmaps of excess AUC regret.

## Common Variants

### Run a single objective

```bash
python scripts/bo_sensor_error_simulation.py --objective composite
python scripts/bo_sensor_error_simulation.py --objective multi_objective
```

### Run specific seeds

```bash
python scripts/bo_sensor_error_simulation.py --seeds 7,8,9,10,11
```

### Use a fixed oracle instead of auto selection

```bash
python scripts/bo_sensor_error_simulation.py --oracle-model extra_trees
```

### Change the error sweep

```bash
python scripts/bo_sensor_error_simulation.py \
  --jitter-iterations 0,10,20,40 \
  --jitter-stds 0.05,0.5,1,5 \
  --error-models gaussian,bias,drift,ar1 \
  --response-clip auto
```

Notes:

- `--jitter-iterations 0` applies noise from the first observation onward —
  the realistic human-feedback condition; the larger onsets isolate *when*
  noise hurts most.
- Calibrate `--jitter-stds` against the response scale: estimate the
  within-participant rating SD from the source data and express noise levels
  as multiples of it. The legacy default of `5` is roughly 3x the entire
  observable objective range on these datasets — a stress test, not a
  human-plausible condition.
- Singular flags (`--jitter-std`, `--jitter-iteration`, `--error-model`) now
  work, and passing both a singular and its plural variant is an error
  instead of being silently ignored.

## Datasets

By default, the scripts read `datasets.json` from the repo root.

That file defines:

- dataset name
- data location (a Git URL cloned into `.dataset_cache/`, or a local path)
- parameter columns
- objective columns
- `oracle_target` (optional, default `"individual"`) — set to `"mean"` to fit the
  oracle on per-design *mean* ratings (an "average human" surface validated on
  held-out designs) instead of individual participant rows. Use this when raw
  ratings carry large between-participant variance the design parameters can't
  explain; on `ehmi` it raises held-out R² from ~0.14 to ~0.55. The simulator's
  injected feedback noise then represents the individual deviation that
  averaging removes.
- `observation_glob` — the filename pattern of the observation CSVs.
  This must match the per-participant files (for opticarvis:
  `data/S_*ObservationsPerEvaluation.csv`); matching only an aggregate export
  silently trains the oracle on a handful of rows.

If you want to use different data, update `datasets.json` or pass `--dataset-config`.

Reproducibility notes:

- Remote datasets are cloned at HEAD; `run_metadata.json` records the
  commit SHA of each cloned data directory (`data_dir_commits`) — quote those
  SHAs in the paper.
- The simulation exits non-zero and records `failed_seeds` in
  `run_metadata.json` if any parallel seed fails, instead of silently writing
  incomplete summaries.
- Document the licenses/provenance of the three study datasets before
  publishing an artifact; they live in external repositories.

## If You Only Remember One Thing

Run `run_full_workflow.bat` (Windows), or the equivalent steps:

```bash
python -m pip install --upgrade -r requirements-eval.txt
python scripts/select_best_oracle_model.py --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting --cv-folds 5 --output-path output/best_oracle_models.json
python scripts/bo_sensor_error_simulation.py --acq all --oracle-model auto --oracle-selection-path output/best_oracle_models.json --seeds 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 --output-dir output --parallel
python scripts/evaluate_research_question.py --input-dir output --output-dir output/evaluation
```

(tabpfn is excluded as an oracle candidate: the oracle is queried hundreds of
thousands of times per run, which is intractable for TabPFN on CPU.)

Then open `output/evaluation/evaluation_report.txt`.
