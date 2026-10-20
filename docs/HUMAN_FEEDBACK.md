# Human feedback experiments on another PC

This extension compares optimizers on the same finite Sobol catalog of exact,
scalar analytic landscapes. It is a new experiment arm. The original project
contains both scalar BO and multi-objective BO; the main known-function study
is scalar. A new method's results should be compared with the reference inside
this arm, rather than merged with the continuous BO/MOBO paper tables.

The initial implementation was checked without benchmark execution. The
2026-10-21 follow-up adds real Hartmann/Branin/Ackley smoke runs and Python 3.15
validation. See [the validation record](VALIDATION_2026-10-21.md).

## Install and plan

Copy the latest `dist/human-feedback-portable*.zip` to the other PC and extract it. Open a
terminal in the extracted folder. Use a fresh virtual environment; the new
dependency file is smaller than the archival-data environment.

Windows PowerShell:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-feedback-py315.txt
.\.venv\Scripts\python.exe -m pip freeze > feedback-environment.txt
.\.venv\Scripts\python.exe scripts/human_feedback_benchmark.py --config configs/feedback_smoke.json --dry-run
```

Python 3.13 is also checked; use `requirements-feedback.txt` for that version.
Python 3.15 uses the official PyTorch CPU index because its Torch wheels are
not distributed through PyPI. The tested Python 3.15 stack uses Torch
2.14.1+cpu, BoTorch 0.18.1, NumPy 2.5.4 and SciPy 1.18.1. It runs on the CPU
and does not require a CUDA installation. On Linux/macOS, use
`python3 -m venv .venv` and `.venv/bin/python` in place of the Windows paths.
The Python 3.15 CPU requirements are checked locally on Windows x86-64; CI is
configured to exercise Ubuntu too. Other OS/architecture combinations may need the
corresponding wheel from the official PyTorch index. The new dependency ranges
support fresh experiments; they do not reproduce the
paper's exact library environment. Keep the frozen environment for screening
and held-out experiments. The archive contains a SHA256 transfer manifest.

To run the entire repository test suite on Python 3.15, install
`requirements-tests-py315.txt` and run `python -m pytest tests/ -q`. That adds
the XGBoost/LightGBM oracle tests and the PyArrow parquet engine used by the
tie-break scanner. These are not needed by the new finite-candidate runner.

`--dry-run`, and running without an action flag, only validate and print the
plan. They do not construct a policy, evaluate an objective, fit a model,
create an output directory, or launch workers. `--summarize` reads completed
checkpoints only and fails if any are missing. Starting an experiment requires
`--run` explicitly.

## Run on the other PC

Use the smoke configuration before a long job:

```powershell
.\.venv\Scripts\python.exe scripts/human_feedback_benchmark.py --config configs/feedback_smoke.json --output-dir output-feedback-smoke --workers 2 --run
.\.venv\Scripts\python.exe scripts/human_feedback_benchmark.py --config configs/feedback_screen.json --output-dir output-feedback-screen --workers 4 --run
```

The smoke plan contains 160 tasks: Hartmann 3D, Hartmann 6D, Branin and Ackley,
eight methods, eight judgments, one seed, Gaussian/drift error at magnitude 1
and onset 0/4, plus clean twins. It exercises Windows process spawning with two
workers. The screening plan contains 39,200 tasks:
20 landscapes, eight methods, five seeds, 48 noisy cells and one clean run
per landscape/method/seed. This is a substantial study; no runtime estimate has
been measured. GP fits will usually dominate compute. Start with a few workers
and inspect memory and saved per-run seconds before increasing parallelism.

To resume an interrupted job, use its identical command plus `--resume`:

```powershell
.\.venv\Scripts\python.exe scripts/human_feedback_benchmark.py --config configs/feedback_screen.json --output-dir output-feedback-screen --workers 4 --run --resume
.\.venv\Scripts\python.exe scripts/human_feedback_benchmark.py --output-dir output-feedback-screen --summarize
```

Each completed task is written atomically to `runs/<task hash>.json`; incomplete
tasks are rerun. A manifest binds the configuration, source hashes, landscape
statistics, Python, platform and numerical package versions. Incompatible
resume attempts fail. Use a fresh directory after changing code or settings.
The worker count can change without changing random streams. Do not edit a
module while Windows workers are running.

Choose methods, hyperparameters, budgets and ship rules using seeds 7–11 only.
Then freeze those choices and use `configs/feedback_heldout.json` (seeds 12–16)
in a fresh output directory. This supplied configuration matches the full
screen; if you choose a smaller or tuned study, copy the choices into both
configurations before evaluating the held-out set. A further untouched set can
use seeds 27–36. This implementation does not select a policy using test seeds
or claim a confirmatory multiple-testing family.

## What the methods test

| Method | Feedback | Mechanism and scope |
|---|---|---|
| `bo_logei` | Rating | GP LogEI scored on the common catalog. The acquisition incumbent is the highest latent posterior mean at a visited design. The reference ships the earliest best observed rating. |
| `random` | Rating | Uniform catalog sampling without replacement until exhaustion. Response-independent search control. |
| `sobol` | Rating | Catalog order, then cycling after exhaustion. Response-independent search control. |
| `cem` | Rating | Cross-entropy-inspired search: fit a diagonal Gaussian to high-rated visited designs and choose an available catalog design of high density, with explicit random exploration. It is a finite-catalog adaptation, not a canonical population CEM implementation. |
| `ridge_ucb` | Rating | RBF-feature ridge reward regression and an upper confidence exploration proxy. This is a bandit baseline, without a GP acquisition. |
| `dueling_ucb` | Comparison | Revisit the empirical Borda leader and choose an opponent by pairwise upper confidence. This is a documented leader/challenger variant, not a claim to reproduce canonical RUCB. |
| `reward_policy` | Comparison | Regularized Bradley–Terry RBF reward model, then an exact KL-regularized categorical policy update against a uniform reference; sample from the policy with uniform exploration. A one-step RLHF analogue. |
| `dpo` | Comparison | Fit categorical policy logits directly with the DPO preference loss against a uniform reference, with explicit L2 regularization. A finite-action DPO analogue; it does not generalize across nearby candidates as the RBF reward model does. |

The policies never receive objective values, the optimum, or ratings of unshown
designs. The environment precomputes the catalog's exact values for scoring.
Only shown designs enter search regret, and recommendations must be shown
designs. The continuous published optimum remains the regret reference; the
catalog's best value is logged to expose the discretization gap. Increasing
catalog size changes the problem and needs a new output directory.

RLHF is appropriate when the goal includes learning a reusable policy from
human preferences. Full sequential RLHF needs an environment with states,
actions, transitions and trajectory preferences. Here an experiment selects a
static design, so finite-action bandits and preference optimization provide a
more direct first comparison. This code does not implement PPO, trajectory
learning, or language-model fine-tuning.

For multi-objective extension, keep the existing `boba_multiobjective.py` arm
and define what the person prefers over outcome vectors before training a
policy. Pareto-front recovery and scalar utility recovery are different
endpoints. Preference BO/BOPE, already related to the repository's PairwiseGP
loop, is also worth studying but remains a BO approach. Other useful future
comparators include CMA-ES/differential evolution, robust best-arm
identification, active ranking, contextual bandits across participants, and
inverse reinforcement learning when demonstrations rather than ratings exist.

## Error models and study design

The screen includes Gaussian error, constant bias, drift, stationary AR(1),
gross spikes and presentation-position bias. Additional models are available
by editing JSON: `lapse`, `contrast`, `fatigue`, `heteroscedastic`, and `rater`.
These extension definitions intentionally have their own provenance and are
not interchangeable with the old sweep's similarly named error processes.

| Error | Meaning of magnitude |
|---|---|
| Gaussian | Independent rating noise SD, in landscape SDs. |
| Bias | Constant additive offset, in landscape SDs; no extra Gaussian component. |
| Drift | Final ramp offset, in landscape SDs. |
| AR(1) | Stationary offset SD, in landscape SDs, with `ar1_rho` controlling serial correlation. Simultaneous pairs share the current state. |
| Spike | Ordinary Gaussian SD plus a spike of `spike_multiplier` times that SD with `spike_probability`. Default spike settings are 15% and 20. |
| Position | First-versus-second preference offset in landscape SDs. Pair presentation order is independently randomized. Scalar ratings have no left/right effect. |
| Lapse | Probability, capped at one, of replacing that judgment's perceived values with standard normal draws. Use a dedicated probability grid such as .05/.15/.30, not the screen's SD grid. |
| Contrast | Dimensionless gain on current value minus the previous sitting's mean. In an unbounded instrument it preserves pair ordering. |
| Fatigue | Gaussian SD starts at magnitude and grows to three times magnitude over the affected budget interval. |
| Heteroscedastic | Rating SD is magnitude times `.25 + sigmoid(true value)`; the environment uses truth to simulate difficulty, but policies do not receive it. |
| Rater | SD of fixed per-rater offsets, in landscape SDs. Rotate `n_raters` in blocks of `rater_block` budget units. |

`onsets` count spent budget units before a judgment starts, so onset 0 affects
the first judgment. Catalog, policy, presentation, rater-offset and error RNG
streams are independent. Error streams are shared across landscapes by design.
Under pure response error, random/Sobol propose identical clean/noisy sequences
and must have exactly zero excess final search regret. Their deployed regret
can still worsen through wrong selection.

`rating_clip: [lo, hi]` and `rating_step` define an instrument in standardized
landscape units and apply to BOTH clean and noisy runs. They do not use the
known optimum to set deployable bounds. Round first, then clip. Shared additive
faults cancel in simultaneous comparisons before this instrumentation; clipping
and rounding can change ties. Pairwise ties are soft labels of .5, not silently
forced wins. Numeric-rating outputs include both earliest-max and the exact
expected true value under uniformly random ties among top-rated designs.

The default `comparison_cost: 1` equates one comparison judgment with one rating
judgment and logs two design exposures per comparison. Repeat the study with
`comparison_cost: 2` to charge for exposures instead. These are different
cost assumptions, and neither measures real participant time. A cost that does
not divide the budget leaves a remainder unused; `budget_spent` records it.
Compare final endpoints at the permitted budget, and report actual spend.

`configs/feedback_edge_smoke.json` combines a clipped/rounded instrument with
comparison cost 2 on Hartmann 3D. Its 184 short tasks exercise all eleven error
models. The common magnitude .25 means a probability for lapse and a
dimensionless gain for contrast; each output row names that unit explicitly.

## Read the outputs

`per_run_metrics.csv` contains final search regret, final deployed regret, and
their difference (final selection loss), in landscape SDs, with clean twins,
judgment counts, design exposures, actual spend and wall time. Checkpoints also
contain the per-judgment search and recommendation trace. No post-onset AUC is
computed by this extension; do not call a final endpoint a trajectory average.

`comparison_summary.csv` compares every method/ship rule with the fixed
`bo_logei:observed_max` reference, within each error/magnitude/onset cell. It
uses the existing `analyse_boba_adaptations.paired_frame` and `summarise`:

```
cost  = reference noisy regret - reference clean regret
gain  = reference noisy regret - treatment noisy regret
price = treatment clean regret - reference clean regret
```

All three are final-budget values in fractions of each landscape's achievable
improvement (`opt_z`). Recovery is a ratio of landscape-mean gain to
landscape-mean reference cost. It is suppressed for a near-zero pooled cost or
a negative per-landscape cost. Absolute noisy and clean reference/treatment
means and gain/price bootstrap intervals accompany it. Selection-component
recovery ratios are omitted; report the absolute selection-loss difference.
`comparison_per_landscape.csv` exposes cluster values and suppresses tiny-cost
ratios there too.

The landscape bootstrap treats correlated landscapes as independent despite
their shared random streams. With three or four landscapes, report that count
and every landscape value; do not interpret stars or the interval as strong
confirmatory evidence. No p-values or stars are emitted. The smoke run is a
software check, not evidence for a scientific ranking.

Every numeric-rating method also reports `policy`, `observed_max`,
`uniform_ties`, and `rank_mean` ship rules. `rank_mean` is a zero-query RBF ridge
refit on ranks, not the old paper's GP rank refit. It is deployable from the
shown ratings and features only. Compare its gain AND clean price with the
same acquisition and standard process. Avoid calling a method robust just
because its own clean run is poor; inspect absolute noisy regret too.

## Sources

- [Christiano et al., human preference RL](https://arxiv.org/abs/1706.03741): learning rewards from trajectory comparisons, then optimizing a policy.
- [Rafailov et al., DPO](https://arxiv.org/abs/2305.18290): direct preference policy optimization. The categorical implementation here is an analogue of that objective.
- [Bengs et al., dueling bandits](https://www.jmlr.org/papers/v22/18-546.html): preference-based online learning and assumptions.
- [Li et al., contextual bandits](https://arxiv.org/abs/1003.0146): linear confidence-based bandit methods; the RBF ridge baseline is an adaptation.
- [de Boer et al., cross-entropy tutorial](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf): population/distribution search underlying the CEM-inspired comparator.
- [BoTorch preference BO tutorial](https://botorch.org/docs/tutorials/preference_bo): a comparison-feedback method that remains BO.
- [PyTorch 2.14 release notes](https://pytorch.org/blog/pytorch-2-14-release-blog/): Python 3.15 wheels use the official PyTorch package index.
