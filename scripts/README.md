# Sensor-error simulation for HITL BO

This folder provides a reproducible, data-driven simulation that answers the following
questions using the eHMI study data:

- **Effects of sensor errors in implicit HITL optimization**: inject Gaussian jitter
  into the feedback signal after a specified iteration and observe how the next
  parameter suggestion changes.
- **Testing 100 iterations (or more/less)**: run iterative Bayesian Optimization
  for a configurable number of iterations.
- **After iteration 20, add artificial jitter to feedback values**: the simulation
  supports `--jitter-iteration 20` and `--jitter-std` for magnitude. Use
  `--single-error` to inject only one error after the jitter iteration. For
  sweeps, use `--jitter-iterations` and `--jitter-stds` to evaluate multiple
  start points and magnitudes.
- **Testing different acquisition functions**: run `ei`, `pi`, `ucb`, or all.

## What this simulation does

1. Loads all `ObservationsPerEvaluation.csv` files from one or more datasets.
2. Trains **oracle models** to map the 9 eHMI parameters to a target objective
   (composite, single-objective, or multi-objective).
3. Runs iterative BO with a **Gaussian Process** surrogate.
4. Trains one or more **oracle models** (Random Forest, Extra Trees, Gradient Boosting,
   HistGradientBoosting, XGBoost, LightGBM) to map eHMI parameters to the target objective.
5. Injects **sensor error** (Gaussian jitter) into the observed feedback after a chosen
   iteration.
6. Writes a per-iteration CSV and a summary of the **parameter adjustment** from
   iteration *N* to *N+1* (e.g., 20 → 21).
7. Displays a progress bar for each simulation run.

## Install (latest compatible versions)

```bash
python -m pip install --upgrade -r scripts/requirements.txt
```

## Run

```bash
python scripts/bo_sensor_error_simulation.py \
  --iterations 100 \
  --jitter-iterations 20,25,30 \
  --jitter-stds 0.05,0.1,0.2,0.4 \
  --initial-samples 5 \
  --candidate-pool 1000 \
  --objective composite \
  --oracle-models random_forest,lightgbm,xgboost \
  --acq all \
  --seed 7 \
  --output-dir output/bo_sensor_error
```

By default, the script now sweeps jitter start points (`20,25,30`) and jitter
magnitudes (`0.05,0.1,0.2,0.4`). Override these with
`--jitter-iterations` and `--jitter-stds` or pass a single
`--jitter-iteration`/`--jitter-std` pair for a focused run.

Oracle models can be swept with `--oracle-models`, or set to a single model with
`--oracle-model`. The default sweep now covers **Random Forest**, **LightGBM**, and
**XGBoost**. Use `--oracle-model all` to run all available options:

```bash
python scripts/bo_sensor_error_simulation.py --oracle-model all
```

### Baseline (no-jitter) comparison

By default the simulation runs **both** a baseline (no jitter) and a jittered run
for each acquisition and seed. This enables an **excess adjustment** calculation
that isolates the impact of sensor error:

- `bo_sensor_error_summary.csv` includes `baseline=true/false`.
- `bo_sensor_error_excess_summary.csv` reports `delta_excess_<param>` and
  `delta_excess_l2_norm` (jittered − baseline).

Disable the baseline with:

```bash
python scripts/bo_sensor_error_simulation.py --no-baseline-run
```

## Outputs

- `bo_sensor_error_<dataset>_<objective>_<acq>_seed<seed>_baseline_<oracle_model>.csv`
  - Full iteration log: parameter values, true objective, observed objective,
    `error_applied`, and `error_magnitude`.
- `bo_sensor_error_<dataset>_<objective>_<acq>_seed<seed>_jittered_<oracle_model>_<error_model>_jit<iter>_std<std>.csv`
- `bo_sensor_error_summary.csv`
  - One row per acquisition function.
  - `delta_<param>`: change in each parameter from iteration *N* to *N+1*.
  - `delta_l2_norm`: L2 norm of the parameter change (overall adjustment magnitude).
- `bo_sensor_error_excess_summary.csv`
  - Per-acquisition/seed **excess change** (jittered − baseline).
- `bo_sensor_error_dataset_effects.csv`
  - Per-dataset and cross-dataset mean/std for jitter excess metrics when baselines are enabled.
- `bo_sensor_error_summary_stats.csv`
  - Mean/std summary by acquisition and baseline flag.
- `final_outcome_significance.csv`
  - Paired t-test results comparing baseline vs jittered **final outcomes**
    (`objective_true` and `objective_observed`) for each sweep configuration.
- `run_metadata.json`
  - CLI args, dataset path, package versions, and total runtime.
- `run_config.txt`
  - Human-readable configuration summary.

### How these outputs answer your questions

**1) “Effects of sensor errors in implicit HITL optimization”**  
Sensor error is simulated by adding Gaussian jitter to the feedback after
`--jitter-iteration`. The per-iteration CSVs (`bo_sensor_error_<dataset>_<objective>_<acq>_seed<seed>_*.csv`)
contain both `objective_true` (oracle signal) and `objective_observed`
jittered values. Comparing these columns after the jitter point shows how
the optimization is driven by noisy feedback rather than the underlying
oracle signal.

**2) “Testing 100 iterations (more or less) and integrating one simulated error as feedback … what is the parameter value adjustment in the following iteration?”**  
Set `--iterations 100` and `--jitter-iteration 20` to match the scenario.
The summary CSV reports the *parameter adjustment* immediately after the
first noisy feedback is injected:

- `delta_<param>` = (parameter at iteration 21) − (parameter at iteration 20)  
- `delta_l2_norm` = overall magnitude of the change

These deltas quantify how the algorithm changes its suggested parameters
in response to the first noisy observation.

**3) “After iteration 20, add artificial jitter to the feedback values”**  
This is the default behavior when `--jitter-iteration 20` is set. All
iterations > 20 use `objective_observed = objective_true + jitter`.
For a single injected error (only iteration 21), add `--single-error`.

**4) “Testing different acquisition functions”**  
Run `--acq all` to generate per-seed logs such as
`bo_sensor_error_composite_logei_seed7_baseline_xgboost.csv` and
`bo_sensor_error_composite_qei_seed7_jittered_xgboost_gaussian_jit20_std0.1.csv`,
plus a summary row for each acquisition. Compare `delta_l2_norm` and
per-parameter deltas across acquisitions to see which method reacts most
strongly to sensor error.

## Objective options

Use `--objective`/`--objectives` to control which dataset column(s) are optimized:

- `composite`: mean of `Trust`, `Understanding`, `PerceivedSafety`,
  `Aesthetics`, `Acceptance`
- `multi_objective`: runs multi-objective BO using the five outcomes directly
- `trust`, `understanding`, `perceived_safety`, `aesthetics`, `acceptance`

By default, the script runs **both** `composite` and `multi_objective`. Override
this with either:

```bash
python scripts/bo_sensor_error_simulation.py --objective trust
python scripts/bo_sensor_error_simulation.py --objectives composite,trust
```

### Objective normalization and weighting

Use `--normalize-objective` to scale each objective column to [0, 1] before
aggregation. Use `--objective-weights` to apply weights matching the objective
columns, for example:

```bash
python scripts/bo_sensor_error_simulation.py \
  --objective composite \
  --normalize-objective \
  --objective-weights 0.3,0.2,0.2,0.2,0.1
```

## Data location

By default the script looks for `eHMI-bo-participantdata` in the repository root
relative to the script location. If your data lives elsewhere, pass it explicitly:

```bash
python scripts/bo_sensor_error_simulation.py \
  --data-dir /path/to/eHMI-bo-participantdata
```

You can also point `data_dir` (or any entry in `data_dirs`) to a remote Git repo
URL. The script will clone it into `--dataset-cache-dir` before loading observations.

### Multiple datasets and custom objectives

To evaluate **multiple datasets** (including datasets with different objective
columns), provide a JSON dataset config and optionally enable a combined dataset.

```json
[
  {
    "name": "ehmi",
    "data_dir": "../eHMI-bo-participantdata",
    "param_columns": ["verticalPosition", "verticalWidth", "horizontalWidth", "r", "g", "b", "a", "blinkFrequency", "volume"],
    "objective_map": {
      "composite": ["Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance"],
      "multi_objective": ["Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance"],
      "trust": ["Trust"],
      "understanding": ["Understanding"],
      "perceived_safety": ["PerceivedSafety"],
      "aesthetics": ["Aesthetics"],
      "acceptance": ["Acceptance"]
    }
  },
  {
    "name": "opticarvis",
    "data_dir": "https://github.com/M-Colley/opticarvis-data",
    "param_columns": ["verticalPosition", "verticalWidth", "horizontalWidth", "r", "g", "b", "a", "blinkFrequency", "volume"],
    "objective_map": {
      "composite": ["Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance"],
      "multi_objective": ["Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance"]
    }
  }
]
```

Run with:

```bash
python scripts/bo_sensor_error_simulation.py \
  --dataset-config datasets.json \
  --combine-datasets \
  --objective composite,multi_objective
```

Notes:
* The dataset name becomes part of the per-iteration CSV filename and is also stored
  in the `dataset` column of summary outputs.
* `--combine-datasets` adds an extra dataset that concatenates the raw observations
  only when **parameter columns** and **objective columns** match across datasets.

### Participant or group filtering

Limit the dataset to a single participant or condition group:

```bash
python scripts/bo_sensor_error_simulation.py --user-id 10
python scripts/bo_sensor_error_simulation.py --group-id 1
```

## Interpreting the parameter adjustment

The summary row answers: **“How much did the suggested parameter vector change right
after the sensor error starts?”**

- The `delta_<param>` values show per-parameter adjustments.
- `delta_l2_norm` provides a single scalar magnitude for the adjustment.

To study stability under sensor error, compare `delta_l2_norm` across acquisition
functions or across multiple seeds.

## Multiple seeds

Use `--seeds` to provide an explicit list, or `--num-seeds` to run sequential seeds:

```bash
python scripts/bo_sensor_error_simulation.py --seeds 7,8,9
python scripts/bo_sensor_error_simulation.py --seed 7 --num-seeds 10
```

The summary stats file reports mean and standard deviation across runs.

## Sensor error models

Use `--error-model` to control the error type after `--jitter-iteration`. The
default sweep now uses **gaussian** and **drift**.

- `gaussian` (default): `objective_observed = true + N(0, jitter_std)`
- `bias`: constant offset `+ error_bias`
- `drift`: linearly increasing offset `+ error_drift * (iteration - jitter_iteration)`
- `dropout`: hold the last observed value (`--dropout-strategy hold_last`)
- `spike`: occasional spikes with probability `error_spike_prob`

Example:

```bash
python scripts/bo_sensor_error_simulation.py \
  --error-model spike \
  --error-spike-prob 0.2 \
  --error-spike-std 0.6
```

## Deterministic jitter

Sensor-error draws now use a dedicated random generator that is seeded from the
base seed plus the jitter iteration, jitter magnitude, and error model. This
makes jitter deterministic across sweeps and prevents the jitter random draws
from perturbing the BO candidate sampling randomness.

## Acquisition hyperparameters

Use `--xi` (EI/PI) and `--kappa` (UCB) to control exploration:

```bash
python scripts/bo_sensor_error_simulation.py --xi 0.02 --kappa 2.5
```

### Modern acquisition functions

The script supports BoTorch's modern analytic and Monte Carlo acquisitions:

- `logei`, `logpi`, `ei`, `pi`, `ucb`, `greedy`
- `qei`, `qpi`, `qnei`, `qucb`
- Multi-objective: `qehvi`, `qnehvi`

Use `--acq-mc-samples` to control Monte Carlo sampling for `q*` acquisitions.

## Oracle improvements & speed knobs

- **Data augmentation:** enable `--oracle-augmentation jitter` (default) to
  add Gaussian-noise copies of each training sample. Configure with
  `--oracle-augment-repeats` and `--oracle-augment-std`.
- **Fast oracles:** use `--oracle-fast` to reduce estimator counts for quicker
  experimentation.

## Plotting

Generate plots from the simulation outputs:

```bash
python scripts/plot_sensor_error_results.py \
  --input-dir output/bo_sensor_error \
  --output-dir output/bo_sensor_error/plots
```

This produces objective trajectory plots per acquisition and a bar chart of
mean `delta_l2_norm`, plus p-value scatter plots summarizing whether the final
outcomes differ significantly between baseline and jittered runs.
