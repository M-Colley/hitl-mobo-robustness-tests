# HITL MOBO Robustness Tests

This repository simulates Bayesian optimization under noisy human feedback.

It is meant to answer a simple question:

- Which optimization choices work best for which dataset and error level?

In practice, the workflow is:

1. Choose or benchmark the oracle model.
2. Run the BO robustness simulation.
3. Evaluate the results with simple rankings, excess-regret summaries, and plots.

## What Each Script Does

- `scripts/select_best_oracle_model.py`
  - Benchmarks candidate oracle models on the dataset and writes the best choice to `output/best_oracle_models.json`.
- `scripts/bo_sensor_error_simulation.py`
  - Runs baseline and noisy BO simulations and writes per-run CSV logs plus summary files.
- `scripts/evaluate_research_question.py`
  - The simplest recommended evaluation step. It ranks acquisitions by robustness for each dataset and error condition.
- `scripts/plot_sensor_error_results.py`
  - Extra legacy plots and paired final-outcome tests. Useful, but not the main workflow anymore.

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
  --iterations 100 \
  --acq all \
  --oracle-model auto \
  --oracle-selection-path output/best_oracle_models.json \
  --output-dir output
```

What this does:

- loads the datasets from `datasets.json`
- trains the oracle model for each dataset/objective
- runs BO with the requested acquisition functions
- runs both a clean baseline and noisy versions
- writes per-iteration logs and summary CSVs to `output/`

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

Use these in this order:

1. `auc_simple_regret_excess_true`
   - Best overall robustness metric.
   - Lower is better.
2. `final_simple_regret_excess_true`
   - Useful endpoint summary.
   - Lower is better.
3. `response_l2_excess`
   - Diagnostic only.
   - Shows how much the first error-affected recommendation changes.

## Important Note About the "Immediate Adjustment" Metric

The old summary files in `output/` include `delta_l2_norm` and `delta_excess_l2_norm`.
Those are **not** the best metrics to answer the research question.

Why:

- noise starts affecting the observed signal at iteration `jitter_iteration + 1`
- the first candidate that can react to that noisy observation is at iteration `jitter_iteration + 2`
- so the useful "response step" is from `t+1` to `t+2`, not from `t` to `t+1`

`scripts/evaluate_research_question.py` already handles this correctly.

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
  --jitter-iterations 10,20,40 \
  --jitter-stds 0.05,0.5,1,5 \
  --error-models gaussian,bias
```

## Datasets

By default, the scripts read `datasets.json` from the repo root.

That file defines:

- dataset name
- data location
- parameter columns
- objective columns

If you want to use different data, update `datasets.json` or pass `--dataset-config`.

## If You Only Remember One Thing

Use this order:

```bash
python -m pip install --upgrade -r requirements-eval.txt
python scripts/select_best_oracle_model.py --oracle-models all --cv-folds 3 --output-path output/best_oracle_models.json
python scripts/bo_sensor_error_simulation.py --iterations 100 --acq all --oracle-model auto --oracle-selection-path output/best_oracle_models.json --output-dir output
python scripts/evaluate_research_question.py --input-dir output --output-dir output/evaluation
```

Then open `output/evaluation/evaluation_report.txt`.
