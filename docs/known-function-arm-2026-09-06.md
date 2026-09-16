# Known-function arm, and the TabPFN 8.5 oracle re-benchmark — 2026-09-06

Two things were asked for: re-run the oracle benchmark now that TabPFN 8.5 is
installed, and extend the project with *known* test functions — all of the ones
in the BOBA repository — to see what feedback error does when the objective is
not a fitted approximation of a person.

This document records what was built, what was verified, what was found, and
what is still running.

---

## 1. Why a known-function arm

Every claim the data-driven arm makes is conditional on the fitted oracle being
a faithful stand-in for a person, and it is not a good one: cross-validated R²
is 0.55 at best on eHMI (after the per-design "mean" target fix) and near zero
on ProVoice. Two consequences run through everything downstream:

- a measured cost of feedback error is entangled with the surrogate human's own
  mis-specification, and
- `y_opt`, the reference regret is measured against, is a random-search estimate
  that the optimiser can legitimately beat — which is why the data-driven arm
  deliberately does not clamp regret at zero.

On an analytic benchmark both problems disappear. The objective is exact, the
optimum is a verified supremum, and — the part that makes this more than a
robustness check — the landscape's geometry can be measured *before* the sweep
and used as a predictor of which landscapes feedback error actually damages.

## 2. What was built

| file | what it is |
|---|---|
| `scripts/boba_benchmarks.py` | The 20 non-stochastic BOBA benchmarks, vectorised over numpy, plus the standardisation constants, landscape descriptors, a one-shot fragility mediator, and two purpose-built extension families. |
| `scripts/bo_synthetic_error_simulation.py` | The simulator. Reuses `bo_sensor_error_simulation`'s core unchanged — same GP, same acquisition optimisation, same error models, same log schema — and swaps the fitted oracle for an exact analytic one. |
| `scripts/analyse_boba_robustness.py` | Cross-benchmark synthesis: floor check, seed-balance check, beneficial-regime stratum, headroom screen, the "what makes an error big" contrast, the landscape-descriptor regression, the one-shot mediator, and acquisition rankings blocked on benchmark. |
| `scripts/diagnose_gp_noise.py` | Refits the surrogate from the sweep's own logs and compares the noise it *learns* with the noise that was *injected*. |
| `boba_landscape_stats.json` | The standardisation constants and descriptor table, regenerable from `boba_benchmarks.py`. |
| `run_boba_pipeline.ps1` | The driver: correctness gate → sweep → per-benchmark evaluation → GP-noise diagnostic → synthesis. |
| `tests/test_boba_benchmarks.py` | 148 tests. Parity against BOBA's own implementations, attainability of every optimum, the standardisation algebra, determinism, pickling under Windows spawn, the log schema, and an end-to-end check that `evaluate_research_question.py` consumes the output unmodified. |

`tests/conftest.py` was added because `test_scientific_correctness.py` failed at
collection whenever it ran before a module that happened to put `scripts/` on
`sys.path` — the suite's result depended on collection order. Whole suite:
**271 passed**.

## 3. Verification of the benchmark suite

All 20 non-stochastic BOBA functions agree with BOBA's own torch
implementations to **≤1.3e-15 relative** on random points in their boxes. Two
discrepancies in BOBA's recorded optima were found by searching each box
(2^15 Sobol + 60 L-BFGS-B polishes) and comparing against `Y_BEST`:

**`power_law_practice` — BOBA's optimum is stale, by 64%.** It records
−0.216422. That is exactly `-4·201^-0.55`, the supremum under the *original*
`n_max = 200`. The function now ships `n_max = 30`, whose supremum is
`-4·31^-0.55 = -0.6050775778`, attained at the all-ones corner (log f is concave
in mean(x), so the extremum is an endpoint, and t always wants to be 1). The
vendored registry uses the corrected value; `tests/test_boba_benchmarks.py`
asserts both the correction and the provenance of BOBA's constant, so the test
fails loudly if BOBA ever fixes it. **This affects BOBA itself** — every regret
it has computed on that function is offset by 0.389.

**`schwefel` — the attainable maximum is −5.09e-05, not 0.** BOBA uses the
truncated constant 418.9829 instead of 418.98288727243369. Immaterial here
(1.3e-7 landscape SDs) but worth knowing, since this arm claims `y_opt` is
exact. The registry keeps the conventional 0, which only ever over-states regret
by that constant, and the constant cancels in every excess-regret contrast.

Every other recorded optimum is attainable to ≤6e-07 relative.

**`typing` (BOBA's 21st function) is excluded from the default suite.** It is a
Monte-Carlo simulation whose value depends on call history rather than only on
x (σ ≈ 0.075 across repeats at fixed x), it discards its time coordinate, and
its `Y_BEST = 1.0` is a clamp ceiling that is never attained (sampled max
0.9498). Including it would reintroduce precisely the sampled-`y_opt` defect
this arm exists to remove. It remains runnable as a labelled side arm
(`--functions typing --boba-root <path>`).

## 4. Design of the sweep

20 benchmarks × 12 acquisitions × 10 seeds × [1 noise-free baseline + 4 error
models × 4 magnitudes × 2 onsets] = **79,200 runs** of 50 BO iterations.

**Objectives are standardised** to zero mean and unit variance over their own
box before any error is injected. Raw outputs span eight orders of magnitude
across the suite (`hartmann_6` tops out at 3.3, `powell` reaches −6e4), so
without this a single grid of error magnitudes would mean "imperceptible" on one
function and "signal destroyed" on the next. The transform is affine and
increasing, so the optimisation problem itself is untouched.

**Magnitudes {0.05, 0.25, 1.0, 5.0} landscape SDs.** The endpoints match the
data-driven sweep's grid once its 1–7 rating scale is expressed in SDs.

**`opt_z` is deliberately left in.** It records how many landscape SDs the
optimum stands above an average random design and ranges from 1.25 (`branin`) to
56.8 (`shekel`) — so the *same* nominal error is trivial on one benchmark and
catastrophic on another. That 45× spread is the lever arm that identifies which
of the two candidate scales fragility actually follows, and normalising it away
would destroy the only variation that can answer the question.

**`bias` scales with the swept magnitude.** With the data-driven arm's fixed 0.2
offset, the bias condition becomes indistinguishable from plain gaussian at the
top of the sweep, so the systematic-vs-random contrast is untestable there.

**Seed-major task ordering, slowest benchmark first**, so a partial sweep is
already a complete cross-benchmark dataset rather than twenty seeds of one
landscape.

**Ten seeds, not twenty.** Unlike the data-driven arm — where seeds are the only
replication and five cannot reach p<0.05 on a Wilcoxon — the primary tests here
block on the twenty *benchmarks*, so seeds only sharpen each cell mean. Widening
`$SEEDS` and re-running reuses every completed run.

### Analysis, and one thing it must not do

The regression response is the raw post-onset per-iteration excess simple regret
in landscape SDs. It is emphatically **not** that quantity divided by `opt_z`:
dividing by `opt_z` and then regressing on `log opt_z` manufactures the very
coefficient the currency test is supposed to estimate — if the raw excess were
perfectly landscape-invariant, the normalised response alone would still return
"achievable gain is the currency", by arithmetic. The normalised version is kept
for reporting magnitudes and for the acquisition rankings, where a
benchmark-constant divisor cannot change a within-benchmark rank.

Inference is a 2000-replicate bootstrap that resamples **benchmarks**, not rows
or seeds: the descriptors vary only between benchmarks, so that is the unit of
replication, and with ~20 clusters the asymptotic cluster-robust sandwich is
both anti-conservative and prone to coming out rank-deficient.

### Built-in controls

- **The model-free floor is an exact control.** `random` and `sobol` choose
  candidates without looking at any observation, so their excess regret must be
  *identically* zero. Measured on a pilot: 0.000e+00. They are excluded from the
  robustness ranking — a method that ignores its data wins any such ranking
  while learning nothing.
- **A constant bias from iteration 1 must cost nothing.** The GP standardises
  its targets and the incumbent is a posterior mean, so an offset present in
  every observation is algebraically invisible. The `bias` arm at onset 0 should
  therefore match the `gaussian` arm up to its noise draw; the analysis reports
  the difference.
- **Headroom screen.** On an easy landscape BO reaches the optimum in the
  noise-free baseline and stays there, so a null is guaranteed by construction.
  The analysis reports each benchmark's noise-free advantage over the floor and
  flags those under 10%.
- **Seed-balance check.** `--resume` matches on filename, and the filename does
  not encode the seed *list*, so a relaunch with a narrower `--seeds` leaves the
  old runs on disk to be silently folded into some cells and not others. (This
  happened: 183 stray seed-17..26 runs from an aborted launch were deleted, and
  the check now fails loudly rather than biasing the cell means.)
- **Beneficial-regime stratum.** Excess regret can be reliably *negative* —
  post-convergence noise on a multimodal landscape is an escape mechanism. Every
  test is two-sided and cells that move the other way are named rather than
  averaged into a smaller positive mean.

### Two extension families (built, not yet run)

Neither is part of BOBA; both exist to break a confound the real suite cannot.

- **`levy_4d`, `levy_7d`** — the same Levy function at d=4 and d=7 which,
  together with `levy_10` (d=11), turns dimension from a single-point covariate
  into a three-point manipulation inside one landscape family.
- **`bump_a{4,16}_w{0.05,0.15}`** — a cos-field background plus one narrow
  spike. Across the real suite, `opt_z`, sparsity and skew are correlated at
  0.87–0.92 and are close to a structural identity for a bounded function with
  an isolated optimum, so no amount of adding real benchmarks separates them.
  Here amplitude moves `opt_z` (5.0 → 21.7 at fixed width) while width moves
  sparsity at roughly fixed `opt_z` — a real, if partial, decoupling. A narrow
  spike is invisible to any finite Sobol screen, so `FunctionSpec` gained an
  `argmax_hint` that `verify_optimum` polishes from; without it the "verified
  optimum" of a needle landscape is just the background maximum.

## 5. TabPFN 8.5

**The tabpfn oracle path had never executed.** `bo_sensor_error_simulation`
sets `torch.set_default_dtype(float64)` for BoTorch; TabPFN's checkpoints are
float32 and it builds its internal tensors from the *default* dtype, so every
fit and predict raised `mat1 and mat2 must have the same dtype, but got Float
and Double`. Any earlier statement that TabPFN "was benchmarked and lost" inside
`select_best_oracle_model.py` is false — it crashed before producing a number.
Fixed by `TabPFNFloat32Regressor`, a module-level (therefore picklable, which
Windows' spawn multiprocessing requires) adapter that pins the default dtype to
float32 around each call and restores it afterwards, leaving BoTorch in double
precision.

**It is not viable as a deployed oracle, and the README's stated reason was
wrong.** The README blamed `--oracle-opt-samples` (200,000 predictions for the
optimum estimate). That is real but tunable, and worth about 14 CPU-days. The
binding cost is the *single-row* `oracle.predict` that `run_simulation` issues
once per BO iteration: an in-context model pays the full forward pass over its
training context every call, measured at **28.3 s per single-row predict** at
the production one-thread setting against **~43,400 rows/s for `extra_trees`**.
That is a ~750× slowdown of a sweep that already takes days. The README now
records the measured numbers. (The machine has an RTX 4090 and
`_build_oracle_model` hardcodes `device="cpu"`; a GPU oracle would be far
faster but contends with 24+ CPU workers.)

**Fidelity: under the deployed protocol, TabPFN 8.5 loses on every slice** — but
that protocol turns out to be the reason (see below). Full re-benchmark,
`--oracle-models all --cv-folds 5`, same protocol as the committed selection
(the tree scores reproduce it to the last digit — `extra_trees` on
ehmi/composite came back 0.5474910874610203 again — so the comparison is
clean). R², higher is better; TabPFN's rank out of 8 in brackets:

| dataset / objective | best model | best R² | tabpfn R² | rank |
|---|---|---|---|---|
| ehmi / composite | extra_trees | 0.5475 | 0.4494 | 8/8 |
| ehmi / multi_objective | extra_trees | 0.4454 | 0.3737 | 6/8 |
| opticarvis / composite | gradient_boosting | 0.5181 | 0.3899 | 8/8 |
| opticarvis / multi_objective | random_forest | 0.3893 | 0.2388 | 8/8 |
| provoice / composite | extra_trees | −0.0324 | −0.0757 | 5/8 |
| provoice / multi_objective | extra_trees | −0.1186 | −0.2027 | 6/8 |

Written to `output/tabpfn85/best_oracle_models_tabpfn85.json`; the committed
selection in `output/best_oracle_models.json` was not touched.

**Consequences.** Under the deployed protocol the *selected* oracle is unchanged
on every slice, so nothing downstream needs re-running on account of TabPFN — a
changed selection would have meant a different test function and therefore a
different sweep, not a re-analysis. The narrow claim is that under this
repository's current selection protocol TabPFN 8.5 does not reach the tree
ensembles on any of the three datasets. It
does *not* support "TabPFN 8.5 is worse than TabPFN 8.2" — the only 8.2 numbers
on record come from a different protocol (cold GroupKFold, no augmentation,
`n_estimators=2`), so that comparison would need `sandbox/oracle_experiments/`
re-run as well.

### The augmentation reverses it — and is the bigger finding

The default `--oracle-augmentation jitter` triples the training set with copies
of every row displaced by 2% of each feature's range, carrying the same target.
That is a questionable input for an in-context learner specifically, so the
benchmark was repeated with `--oracle-augmentation none`. The two runs are
identical in every recorded setting — folds, grouping, seed, model list, CV
strategy — except that flag.

Every model improves, most of them by a lot, and TabPFN improves most:

| dataset / objective | best (jitter) | R² | best (none) | R² | tabpfn Δ | every-model Δ |
|---|---|---|---|---|---|---|
| ehmi / composite | extra_trees | 0.5475 | **tabpfn** | **0.5988** | +0.149 | +0.010 … +0.149 (2 down) |
| ehmi / multi_objective | extra_trees | 0.4454 | **tabpfn** | **0.4966** | +0.123 | +0.008 … +0.123 (1 down) |
| opticarvis / composite | gradient_boosting | 0.5181 | xgboost | **0.8229** | +0.374 | **+0.192 … +0.374** |
| opticarvis / multi_objective | random_forest | 0.3893 | xgboost | **0.6545** | +0.373 | **+0.170 … +0.373** |
| provoice / composite | extra_trees | −0.0324 | **tabpfn** | **+0.0204** | +0.096 | +0.024 … +0.096 (1 down) |
| provoice / multi_objective | extra_trees | −0.1186 | **tabpfn** | −0.0554 | +0.147 | +0.024 … +0.147 (1 down) |

Two conclusions, and they point in opposite directions from each other:

**On TabPFN.** Under the protocol the simulator actually deploys — `build_oracle`
also defaults to `jitter` — TabPFN 8.5 loses on all six slices. Remove the
augmentation and it wins four of six, including the headline ehmi/composite
number (0.5988 against the best tree's 0.5669). It is not that TabPFN is a poor
oracle; it is that near-duplicate rows are unusually bad for a model that
conditions on its training set in context, and the pipeline feeds it two thirds
of them. Any statement about TabPFN's fidelity has to name the augmentation
setting.

**On the augmentation itself, which matters more.** It is costing this project a
great deal of oracle fidelity on datasets that are not eHMI. On opticarvis every
one of the eight models gains between **+0.17 and +0.37 R²** when it is turned
off — the deployed oracle scores 0.52 where an identical model without
augmentation scores 0.82. On provoice, composite fidelity crosses from negative
to (barely) positive. The eHMI gains are small for the tree ensembles
(+0.01 … +0.03) and only large for TabPFN, which is consistent with the
`oracle_target="mean"` aggregation there already doing the smoothing that the
jitter was presumably meant to provide.

The mechanism is plausible on inspection: `augment_oracle_data` displaces every
coordinate by 2% of its range, which in opticarvis's 16-dimensional design space
is 8% of the box diagonal, and asserts the rating is unchanged there. That is a
strong and evidently false smoothness assumption, and it is imposed three times
over.

**What this does and does not license.** It does not retrofit onto existing
results: the oracle *is* the objective, so training it differently makes a
different test function, and `output-chi/` would have to be regenerated rather
than re-analysed. It does say that the next sweep should not use jitter
augmentation, and that "the human oracle is very difficult to model" is partly
self-inflicted. Caveat on the evidence: these are 5-fold CV means and per-fold
spreads were not recorded, so the small eHMI tree deltas should not be leaned
on; the opticarvis effect is an order of magnitude larger than any plausible
fold noise.


## 6. Two confounds the applied literature does not control

Both were named up front as threats, both were measured rather than argued
about, and they came out opposite ways. Results are in 7.6 and 7.8; this section
records why they were run.

**The surrogate fits its own noise.** `run_simulation` builds
`SingleTaskGP(train_X, train_Y, ...)` with no `train_Yvar`, so the observation
noise is a free hyperparameter under BoTorch's default prior — a prior whose
mass sits well below the levels this study injects. If the fitted noise were far
below the injected noise, the measured degradation would be partly the cost of a
mis-fitted hyperparameter rather than the cost of the information the error
destroys, and a null for `qnei` (the acquisition built for noisy observations)
could not distinguish "noise-awareness does not help" from "noise-awareness was
never switched on". `scripts/diagnose_gp_noise.py` measures the gap and
`--observation-noise known` closes it. **Verdict: real but inconsequential** —
the estimate is 8× low for twenty iterations, and fixing it changes almost
nothing (7.8).

**The incumbent shrinks under noise.** `best_f` is the maximum posterior mean at
visited points, which contracts toward the standardised mean as noise rises,
making improvement-based acquisitions automatically more exploratory. The
acquisitions differ in exposure: logei/logpi/ei/pi/qei consume `best_f`, ucb and
qucb do not, qnei ignores it entirely — so a robustness ranking over
acquisitions is partly a ranking over incumbent sensitivity.
`--incumbent observed_max` separates them, with ucb and qnei as controls that
must not move. **Verdict: substantial** — it accounts for 35–41% of the
probability-of-improvement family's measured fragility (7.8).

## 7. Results

**79,200 runs, 76,800 paired jittered-vs-baseline comparisons, 20 landscapes,
12 acquisitions, 4 error models x 4 magnitudes x 2 onsets, 10 seeds. 19.7 h on
24 workers, 0.90 s/run.** Every control passes: model-free excess regret exactly
`0.000e+00` across all 2,560 `random`/`sobol` comparisons, balanced seed panel,
no benchmark ceiling-limited (lowest headroom `rosenbrock` 0.176), and the
uniform-offset control lands within 0.011 of zero at every magnitude.

### 7.1 How much it costs

Fraction of the achievable improvement destroyed, averaged over landscapes and
model-based acquisitions:

| error present | 0.05σ | 0.25σ | 1σ | 5σ |
|---|---|---|---|---|
| from iteration 1 | 0.95% | 5.33% | **14.09%** | **21.97%** |
| from iteration 21 | 0.29% | 1.19% | 2.81% | 4.90% |

Two things stand out. Bayesian optimisation is considerably more robust than the
framing of the applied literature implies — error with the same standard
deviation as the landscape itself, present from the very first rating, costs 14%
of what was available, and even a 5σ error costs 22%. And **when the error
starts matters about five times more than how large it is**.

The *type* of error barely matters at matched magnitude and onset: gaussian
0.109, drift 0.113, bias 0.110, ar1 0.092. Serially correlated error is if
anything gentler than white noise, and a systematic drift is indistinguishable
from a random one. The four-way error taxonomy the data-driven arm sweeps over
is close to a single factor.

The **landscape**, by contrast, dominates everything. At an identical 1σ error
from iteration 1: `powell` 0.038, `rosenbrock` 0.038, `eggholder` 0.039 at one
end; `power_law_practice` 0.219, `hicks_law` 0.241, `hartmann_6` 0.303 at the
other. An 8x spread across landscapes against a 4x spread across the entire
error grid.

### 7.2 The mechanism: it reduces to one noisy pick

`frag(σ_e)` is the expected loss from a *single* greedy choice among 256 Sobol
candidates when each is observed with N(0, σ_e²) error — no GP, no sequence, no
acquisition function, about 0.2 s per landscape to estimate. Fitted on the
benchmark x condition cell means, predictors z-scored, 95% intervals from a
2000-replicate bootstrap over benchmarks:

| model | R² | β(frag) | β(log σ_e) | β(log opt_z) | β(log tail) |
|---|---|---|---|---|---|
| mediator only | 0.520 | **+0.75** | — | — | — |
| descriptors only | 0.526 | — | +0.37*** | +0.64*** | +0.20** |
| both | **0.701** | **+0.66 [0.28, 0.74]*** | **−0.07 [−0.12, 0.08], p=0.44** | +0.42** | +0.20** |

**Conditional on the one-shot selection loss, the error magnitude explains
nothing.** A dose only matters through what that dose costs a single choice on
that particular landscape. Dimension and ruggedness are null throughout.

The practical consequence is the reason to care: `frag(σ_e)` is computable from
a design space and a rating-noise estimate before any optimisation is run.


> **Correction (2026-09-12): the null on magnitude is specification-dependent.**
> The additive descriptor model is misspecified for the raw response: section 7.3
> finds the cost is a power law in σ_e and opt_z whose exponents do not sum to one,
> i.e. the two interact. `mediator_model.csv` now carries two more models with the
> product log σ_e × log opt_z and a quadratic in log σ_e. Descriptors + interaction:
> R² = 0.779 with no frag term (additive: 0.526).
> Adding frag: R² = 0.792, β(frag) = +0.31 [+0.18, +0.45],
> β(log σ_e) = -0.38 [-0.60, -0.08] (significant again),
> β(interaction) = +0.67 [+0.12, +0.95]. So frag predicts on
> its own (0.52 with the indicators, 0.48 alone) about two thirds of what the best
> descriptor model explains and stays informative beside it, but it does not make
> the magnitude redundant. The paper's Section 6 now says "predicts", not
> "mediates", and the preregistered H1b is noted as holding under the additive
> specification only.

### 7.3 What makes an error "big"

Power law in the mean, `E[excess] = A · σ_e^β_c · opt_z^β_z`, fitted to
landscape × magnitude cell means by quasi-Poisson estimating equations (which
keep the cells where noise happened to help), with intervals from resampling
landscapes. SPREAD predicts β_z = 0; GAIN (excess = opt_z · h(σ_e / opt_z) with
a power-law h) predicts β_c + β_z = 1.

| error model | onset | β_c | β_z | 95% BCa CI | β_c + β_z − 1 | 95% BCa CI | verdict |
|---|---|---|---|---|---|---|---|
| gaussian | 0 | 0.51 | 0.86 | [0.47, 1.13] | +0.37 | [+0.15, +0.70] | neither |
| gaussian | 20 | 0.62 | 1.15 | [0.64, 1.48] | +0.76 | [+0.03, +1.03] | neither |
| bias | 0 | 0.51 | 0.85 | [0.49, 1.08] | +0.36 | [+0.17, +0.65] | neither |
| bias | 20 | 0.70 | 1.14 | [0.61, 1.58] | +0.85 | [+0.04, +1.08] | neither |
| drift | 0 | 0.55 | 0.82 | [0.55, 1.11] | +0.37 | [-0.01, +0.63] | gain |
| drift | 20 | 0.65 | 1.15 | [0.61, 1.55] | +0.80 | [-0.00, +1.14] | gain |
| ar1 | 0 | 0.66 | 0.85 | [0.60, 1.15] | +0.51 | [+0.14, +0.77] | neither |
| ar1 | 20 | 0.71 | 1.14 | [0.64, 1.49] | +0.85 | [+0.13, +1.11] | neither |

Intervals are BCa over resampled landscapes (percentile intervals stay in the
`*_pct_*` columns of `noise_currency.csv`); the bootstrap is right-skewed and leans
on `shekel`, the one landscape with opt_z above 20: without it β_z is
1.04–1.10 from the first observation and the GAIN gap 0.57–0.73.
SPREAD fails in every condition. GAIN fails in six of eight — drift's gap touches
zero at both onsets — and OLS on log(excess) over the positive cells rejects it in
all eight. The cost grows with opt_z faster than the achievable-gain scaling allows.
> **Correction (2026-09-10).** An earlier version of this section regressed the
> *raw* excess on log10 σ_e and log10 opt_z and read GAIN as β(opt_z) = 1 − β(noise).
> That restriction holds for the exponents of a power law, not for slopes on a
> semi-log scale, so its "gain fits at the mid-run onset" verdict was not a test
> of GAIN. The table above replaces it.

### 7.4 Which acquisition

| acquisition | mean rank | excess regret | absolute loss |
|---|---|---|---|
| **qUCB** | **3.95** | 0.041 | 0.279 |
| qNEI | 4.34 | 0.049 | **0.252** |
| UCB | 4.41 | 0.047 | 0.274 |
| LogEI | 4.62 | 0.052 | 0.260 |
| qEI | 4.73 | 0.054 | 0.256 |
| EI | 4.98 | 0.054 | 0.262 |
| qPI | 6.29 | 0.079 | 0.310 |
| Greedy | 6.57 | 0.081 | 0.325 |
| LogPI | 7.48 | 0.092 | 0.313 |
| PI | 7.66 | 0.094 | 0.317 |
| *model-free floor* | *—* | *0.000* | *0.410* |

Confidence-bound and noisy-EI acquisitions are the robust ones and the
probability-of-improvement family is uniformly worst, on both robustness and
absolute performance. Friedman over the 20 landscapes is significant in 28 of 32
conditions and 115 of 288 pairwise comparisons survive FDR — but **Kendall's W is
0.247**, so which acquisition wins is mostly a property of the landscape (qUCB
takes 10 of 32 conditions, UCB and qNEI 6 each, LogEI 5). Any "use X under noise"
recommendation drawn from one design space is over-generalising.

### 7.5 Noise almost never helps

At one seed, 25 cells showed reliably negative excess regret and it looked like a
regime. At ten seeds exactly **one** cell survives (`michalewicz`, ar1, 0.05σ,
onset 20, dz −0.21). The escape-from-a-local-optimum story is not supported;
the earlier signal was sampling noise, and is corrected here.

### 7.6 The surrogate is slow to notice — and it barely matters

`scripts/diagnose_gp_noise.py` refits each GP from the sweep's own logs using
only the first *n* observations. Ratios of fitted to injected noise SD, gaussian
error from iteration 1:

| observations | 0.05σ | 0.25σ | 1σ | 5σ |
|---|---|---|---|---|
| 8 | 1.43 | 0.30 | **0.12** | **0.11** |
| 15 | 1.50 | 0.41 | **0.14** | **0.10** |
| 25 | 1.37 | 0.72 | 0.66 | 0.56 |
| 50 | 1.24 | 0.88 | 0.83 | 0.89 |

For the first twenty-odd of its fifty iterations the surrogate believes its
observations are about **eight times cleaner than they are**, and by the end of
the run the estimate has converged — so a diagnostic evaluated only at n=50
looks reassuring and is misleading. A wide noise prior recovers the injected
level far sooner (0.55 rather than 0.14 at n=15, 1σ), so the shrinkage is the
prior's doing rather than a limit of the data.

The obvious inference — that this explains the onset effect, since at onset 0
the error arrives exactly when the noise cannot yet be estimated — **was tested
and is wrong.** See 7.8.

### 7.7 Is 1σ a lot? Anchoring against real raters

A landscape standard deviation is only a useful unit if real feedback noise can
be expressed in it. `scripts/calibrate_noise_from_data.py` estimates within-rater
noise from the three archival studies with a nearest-neighbour (semivariogram
nugget) estimator over repeated ratings by the same participant;
`scripts/anchor_noise_scale.py` divides that by the standard deviation of each
study's own fitted objective over its design box — the same quantity `σ_f`
denotes for a benchmark — and reports each study's `opt_z` on the same scale as
the twenty landscapes.

| study | σ_f | opt_z | suite percentile | rating noise | **noise / σ_f** |
|---|---|---|---|---|---|
| ehmi | 0.365 | 2.37 | 30th | 0.269 | **0.74** |
| provoice | 0.298 | 4.00 | 70th | 0.304 | **1.02** |
| opticarvis | 0.374 | 13.36 | 90th | 1.254 | **3.35** |

Real rating noise lands between **0.74 and 3.35 landscape standard deviations**,
and participants are noisy from their first rating — the expensive onset. Read
off the onset-0 row of the dose–response table, those studies lose roughly
**12–20% of the achievable improvement** to rating noise alone. The magnitudes at
which this study measures its largest costs are the magnitudes human raters
actually produce.

The caveat is structural and has to be stated: σ_f for a real study can only be
computed *through a fitted oracle*, which is precisely the object this arm exists
to avoid. It anchors the interesting range; it is not itself a measurement.

### 7.8 Ablations

Three follow-up arms, each scoped to the smallest design that answers its own
question, paired within (landscape, acquisition, magnitude, onset, seed).

**Known noise — the confound is ruled out.** Supplying the true injected
variance as `train_Yvar` across 4,800 paired cells removes **5.0%** of the
measured cost, **no condition survives FDR correction** (all p ≥ 0.888), and the
onset-0 to onset-20 ratio at 1σ does not shrink — it goes from 4.0 to 5.2. Per
landscape the largest reduction is ackley at 47%; per acquisition the sign is
inconsistent, helping pi (22%) and qnei (15%) but *hurting* qucb (−29%).

Two things follow. BO is remarkably insensitive to its own noise hyperparameter
in this regime, despite getting it badly wrong for the first twenty iterations.
And the cost this study measures is the **information** cost of the error, not
an artefact of a mis-specified surrogate — which is what licenses reading the
dose–response table as a property of the problem. It also means the onset effect
is currently **unexplained**, and my earlier framing of 7.6 as its mechanism was
wrong.

**Incumbent — the confound is ruled in.** Switching `best_f` from the posterior
mean to the observed maximum removes **15.6%** of the measured cost, with four
of eight conditions FDR-significant (down to p = 0.0005). The per-acquisition
breakdown is the finding:

| acquisition | reference | treatment | Δ | share removed |
|---|---|---|---|---|
| LogPI | 0.555 | 0.328 | −0.227 | **+41%** |
| PI | 0.572 | 0.373 | −0.199 | **+35%** |
| EI | 0.459 | 0.382 | −0.077 | +17% |
| LogEI | 0.493 | 0.481 | −0.011 | +2% |
| qNEI | 0.378 | 0.378 | **+0.000** | 0% |
| UCB | 0.346 | 0.346 | **+0.000** | 0% |
| qEI | 0.445 | 0.454 | +0.009 | −2% |

qNEI and UCB never read `best_f`, so they move by *exactly* zero — a control
that validates the pairing. A substantial part of "PI is fragile under noise" is
really "the posterior-mean incumbent is fragile under noise, and PI is the
acquisition most exposed to it". Every acquisition-robustness ranking here,
including 7.4, is a ranking of acquisition-and-incumbent pairs.

**Manipulations — the descriptor regression is wrong in two places.** Raw
post-onset excess regret (landscape SDs), gaussian error from iteration 1:

| landscape | d | opt_z | sparsity | 0.05σ | 0.25σ | 1σ | 5σ |
|---|---|---|---|---|---|---|---|
| bump_a4_w0.05 | 4 | 5.0 | <2e-5 | 0.007 | 0.152 | 0.246 | 0.324 |
| bump_a16_w0.05 | 4 | 21.7 | <2e-5 | 0.014 | 0.026 | **0.245** | 0.262 |
| bump_a4_w0.15 | 4 | 5.3 | 4e-4 | 0.133 | 0.289 | 0.495 | 0.704 |
| bump_a16_w0.15 | 4 | 12.3 | 2e-4 | 0.298 | 0.730 | **1.888** | 3.173 |
| levy_4d | 4 | 1.5 | 3e-1 | 0.006 | 0.048 | **0.114** | 0.200 |
| levy_7d | 7 | 2.1 | 7e-2 | 0.051 | 0.115 | **0.222** | 0.305 |
| levy_10 | 11 | 2.6 | 1e-2 | 0.001 | 0.147 | **0.264** | 0.481 |

*`opt_z` on its own is not causal.* Holding the spike narrow and raising its
amplitude 4× moves opt_z from 5.0 to 21.7 and changes excess at 1σ from 0.246 to
0.245 — nothing. Widen the spike and the same amplitude change takes excess from
0.495 to 1.888. So the observational β(log opt_z) = +0.42 is an interaction, not
a main effect: a taller optimum costs more to miss only when it is broad enough
that BO would otherwise have found it.

*Dimension does matter, and the observational analysis missed it.* Within the
Levy family, excess at 1σ rises monotonically 0.114 → 0.222 → 0.264 from d=4 to
d=11, while the cross-suite regression puts β(dim) at −0.03, n.s. Across twenty
benchmarks dimension is confounded with everything else; within one family it is
not. (Caveat: opt_z also rises along the ladder, 1.53 → 2.07 → 2.62, so this is
a within-family comparison rather than a pure dimension manipulation.)

### 7.9 Input error: the person applies the wrong design

Three arms (`output-boba-slip`, `output-boba-misclick`, `output-boba-slip-actual`),
each 20 landscapes x 6 model-based acquisitions (logei, ei, pi, ucb, qucb, qnei) +
2 floors x 4 magnitudes x 2 onsets x 5 seeds = 6,400 paired runs. A **slip**
displaces every trial by N(0, s^2) per coordinate (s a fraction of the range,
clipped to the box); a **misclick** replaces a trial by a uniform random design
with probability p. The rating is exact. Under `recorded=proposed` the log holds
the proposal x and the surrogate trains on (x, f(x')); under `recorded=actual`
it holds x'. Analysis: `scripts/analyse_boba_inputerror.py` -> 
`output-boba-slip/analysis/inputerror_{dose,mechanism,mislabel,deployed}.csv`.
All numbers are the fraction of the achievable improvement destroyed, in percent,
with 95% cluster-bootstrap intervals over landscapes.

**Dose-response (model-based acquisitions).**

| magnitude | slip, it. 1 | slip, it. 21 | misclick, it. 1 | misclick, it. 21 |
|---|---|---|---|---|
| 1% | 1.5 [0, 3] | 1.1 [0, 2] | 0.4 [0, 1] | 0.1 [0, 0] |
| 5% | 5.4 [3, 8] | 3.0 [2, 5] | 2.8 [2, 4] | 0.5 [0, 1] |
| 15% | 11.5 [8, 15] | 5.0 [3, 8] | 6.8 [5, 9] | 2.0 [1, 3] |
| 40% | 23.2 [16, 31] | 6.4 [4, 10] | 13.5 [10, 18] | 3.5 [2, 6] |

A 5% slip on every trial costs about what a 0.25 sigma rating error costs (5.4% vs
5.3%), a 15% slip about a 1 sigma rating error (11.5% vs 14.1%), a 40% slip more
than a 5 sigma one (23.2% vs 22.0%). A misclick is cheaper per unit: p = 0.4 costs
13.5%, less than a 15% slip. The early/late ratio for a slip is 1.4-3.6x against
5.0x for a 1 sigma rating error.

**Mechanism (slip, error from the first trial).** `floor`: model-free arms, so the
geometric cost of evaluating displaced points with nothing to corrupt; `logged`:
the slip-actual arm, a learner whose record is correct; `unnoticed`: the slip arm;
`mislabelling`: unnoticed - logged, paired on shared seeds and slip draws (the
arms share their slip stream). Computed on the 100 complete (landscape, seed)
tasks common to both arms; the column gap equals the paired cost to 1e-16.

| slip | floor | logged | unnoticed | mislabelling |
|---|---|---|---|---|
| 1% | 0.3 [-0, 1] | 0.7 [-0, 2] | 1.5 [0, 3] | 0.8 [-0, 2] |
| 5% | 0.2 [-1, 1] | 2.8 [2, 4] | 5.4 [3, 8] | 2.7 [1, 4] |
| 15% | 1.2 [-1, 3] | 7.1 [5, 10] | 11.5 [8, 15] | 4.4 [2, 6] |
| 40% | 6.4 [0, 13] | 19.6 [12, 28] | 23.2 [16, 31] | 3.6 [2, 6] |

The floor barely moves (a random design displaced is still random) except at 40%,
where clipping piles evaluations at the box edges. A learner with a correct record
loses far more than the floor. Mislabelling is about half the total at 1-5%,
two-fifths at 15% and a sixth at 40%. At the late onset the mislabelling cost is
0.1-1.0 points.

**What ships (slip, unnoticed).** Final-iteration excess simple regret of the best
design actually experienced (`evaluated`) against the design the log recommends,
scored at the recorded location of the best-rated trial (`deployed`).

| slip | evaluated, it. 1 | deployed, it. 1 | evaluated, it. 21 | deployed, it. 21 |
|---|---|---|---|---|
| 1% | 1.7 [0, 3] | 2.7 [1, 5] | 1.5 [1, 3] | 2.4 [1, 4] |
| 5% | 6.5 [4, 10] | 16.4 [9, 24] | 3.9 [2, 6] | 9.6 [5, 15] |
| 15% | 13.4 [8, 20] | 34.3 [24, 45] | 6.9 [3, 10] | 13.4 [8, 19] |
| 40% | 24.9 [16, 34] | 65.8 [53, 78] | 9.0 [5, 14] | 13.3 [8, 19] |

The deployed design does 1.6-2.6x worse than the design that earned its rating:
the optimizer's progress survives an input error far better than the record of
it does.

Two notes for anyone re-running this. The floor check in
`analyse_boba_robustness.py` is reported, not asserted, for these arms: a
model-free floor evaluates the wrong points too, so its excess is not zero. And
the first slip arm labelled its corrupted runs `none`, which the evaluator reads
as the baseline marker; it was regenerated with `run_error_label` and the old
directory is `output-boba-slip-mislabelled` (safe to delete).
### 7.10 frag from a pilot (experiment E7)

`scripts/analyse_pilot_frag.py` (tests: `tests/test_pilot_frag.py`). For each landscape and each
of the ten seeds, the study's own GP is fitted to the first k trials of the clean LogEI run and
frag(σ_e) is evaluated on its posterior mean, on the same 256 Sobol designs and with the same
4,000 error draws as the exact value (common random numbers; the replica reproduces
`selection_fragility` and the stored JSON bit for bit). Spearman correlations across landscapes,
the measured cost being the early-onset fraction destroyed averaged over the ten model-based
acquisitions and four error processes:

| frag computed from | with exact frag (pooled) | with measured cost (pooled) | cost at 0.25σ | cost at 1σ | cost at 5σ |
|---|---|---|---|---|---|
| first 10 clean trials | 0.94 | 0.85 | -0.20 | 0.32 | 0.61 |
| first 20 clean trials | 0.96 | 0.85 | 0.01 | 0.20 | 0.66 |
| first 50 clean trials | 0.96 | 0.85 | 0.17 | 0.15 | 0.63 |
| exact objective | 1.00 | 0.84 | 0.42 | 0.42 | 0.45 |

Ten clean trials are enough: the pilot frag tracks the exact one and predicts the pooled cost as
well as the exact frag does. Within one magnitude neither ranks landscapes well, so the predictive
power is mostly across magnitudes. At 5σ a ten-point pilot underestimates frag by about a third on average
(mean bias −0.78 against a mean exact value of 2.27; the median landscape by a fifth). Outputs: `output-boba/analysis/pilot_frag*.csv`.
### 7.11 Extra trials: what the error costs in budget

`scripts/analyse_extra_runs.py`. For each noisy run and its identically seeded clean
twin, the number of extra trials the noisy run needs to reach the true best-so-far
regret the clean run had after k trials, within a tolerance of 1% of opt_z (the
achievable improvement), measured from the trial at which the clean run itself first
got there. Runs that never get there within the budget T are censored at T − origin,
so the mean is a lower bound; the median is exact while fewer than half are censored.
Model-free floors excluded. Medians and shares are rounded half up. Output:
`<arm>/analysis/extra_runs.csv` (+ `_per_run.csv`).

**Budget-100 arm (gaussian, six acquisitions, seeds 7–11), tolerance 1%.**

| σ_e | error from it. 1, match a clean 25-trial study | error from it. 41, match a clean 50-trial study |
|---|---|---|
| 0.05σ | 1 (never 4%; mean 7.8 [5.6, 10.1]) | 0 (never 1%; mean 1.6 [0.7, 2.8]) |
| 0.25σ | 7 (never 14%; mean 20.5 [15.8, 25.6]) | 0 (never 3%; mean 2.5 [1.1, 4.2]) |
| 1σ | 41 (never 39%; mean 43.3 [36.9, 49.4]) | 0 (never 8%; mean 4.7 [2.1, 7.8]) |
| 5σ | >78 (never 70%; mean 64.0 [59.6, 68.3]) | 0 (never 12%; mean 7.0 [3.3, 11.4]) |

At 1σ from the first rating the median run needs 41 extra trials to match a clean
25-trial study (×2.6 budget) and 39% of runs never do within 100; at 5σ, 70% never
do. At 0.05–0.25σ the shortfall is 1–7 trials. Late error (from trial 41) costs a
50-trial study a median of no extra trials, though 8–12% of runs at ≥1σ never recover.
Extra runs buy back small error, not the magnitudes real raters produce — which is
the case for changing the method rather than the budget (see
`docs/adaptations-proposal.md`).

**Every arm, early onset, match a clean 10-trial study within T = 50, tolerance 1%
(median extra trials; never-reached share).**

| arm | magnitudes | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|---|
| gaussian | 0.05 / 0.25 / 1 / 5 | 0 (1%) | 1 (8%) | 9 (26%) | 26 (44%) |
| bias | 0.05 / 0.25 / 1 / 5 | 0 (1%) | 2 (9%) | 8 (27%) | 28 (45%) |
| drift | 0.05 / 0.25 / 1 / 5 | 0 (1%) | 2 (9%) | 8 (28%) | 36 (49%) |
| ar1 | 0.05 / 0.25 / 1 / 5 | 0 (1%) | 1 (4%) | 3 (22%) | 23 (42%) |
| slip (unnoticed) | 0.01 / 0.05 / 0.15 / 0.4 | 0 (3%) | 3 (7%) | 8 (15%) | 24 (40%) |
| slip (logged) | 0.01 / 0.05 / 0.15 / 0.4 | 0 (2%) | 2 (5%) | 4 (10%) | 14 (25%) |
| misclick | 0.01 / 0.05 / 0.15 / 0.4 | 0 (1%) | 0 (3%) | 0 (9%) | 6 (23%) |
### 7.12 Corrections from the paper audit (2026-09-12)

Found by adversarial review of the draft against the outputs; each is now in the
paper and, where it touches a table, in `make_boba_paper_tables.py`.

- **Reference rows matched to their arms.** The budget, instrument and robust
  arms ran gaussian error on seeds 7–11 with six acquisitions (or qKG + replEI),
  but their reference rows were the full sweep (four processes, ten acquisitions,
  ten seeds). `_matched_dose_grid` now restricts the main sweep to the arm's own
  settings. Matched T=50 row (gaussian × six × seeds 7–11): early 0.6/3.8/11.7/21.3,
  late 0.1/1.0/2.3/4.7; onset ratios 2.3×, 5.1×, 11.7×. The instrument arm's
  apparent 17% benefit at 1σ (11.7 vs 14.1) was this mismatch — matched, 11.7 vs
  11.7. The robust reference (gaussian × ten × seeds 7–11) is 13.9% and 21.5%, so
  qKG/replication recover a third at 1σ and a quarter at 5σ. *(Superseded by
  §7.13: that is excess over their own handicapped clean runs; against the
  standard process neither recovers the cost.)*
- **Fixed-opt_z ladder.** The 0.031/0.028/0.048 quoted for bump_d4/d7/d11 were
  fractions of opt_z; in the table's raw units they are 0.28/0.25/0.43. bump_d4 has
  headroom 0.098, below the 0.10 screen, so the admissible comparison is d = 7 → 11,
  where the fixed-opt_z ladder rises 1.7× against Levy's 1.2×: dimension does raise
  the cost when opt_z is held fixed. "Most of the dimension effect was opt_z" is
  withdrawn. The ladder rows are now in the manipulation table.
- **Noise-estimate window.** The fitted noise is an eighth of the injected value
  through 15 observations (iteration 10), and 0.56–0.66 at 25; "first twenty
  iterations" overstated it.
- **Pooled shares are excess-weighted.** Known-noise removes 5.0% of the
  excess-weighted cost (40% at 0.25σ, n.s.; ≤3% at 1σ and above); the incumbent
  switch removes 15.6% (14% at 1σ, 67–86% at 0.25σ and below).
- **Multi-objective scalar column** is now gaussian-only (the MO arm ran gaussian
  only), which makes it the known-function row of the fitted-oracle comparison
  (10.2/44.3/99.0/148.4 early on the floor gap).
- **H3a** was tested over the eight conditions rather than the twenty benchmarks as
  preregistered; run as preregistered, p = 3.6e-5 (7.3e-5 without Rosenbrock).
- **Fitted-oracle gap.** Ratio 3.0 at 1σ with cluster-bootstrap interval [1.1, 17];
  the optimum estimate cannot be the mechanism (it cancels in the excess), and the
  three fitted design spaces sit at the low end of the twenty-landscape spread.
- **Oracle-fidelity manipulation** is +0.29 held-out R² for the deployed oracle
  (opticarvis gradient_boosting 0.52 → 0.81), not the +0.37 of TabPFN.
- **opt_z range** is 0.76 (powell) to 56.8 (shekel), 74×, not 1.25 to 56.8.

### 7.13 Process adaptations (experiments E1–E6, 2026-09-13)

The question after §7.11: if extra trials cannot absorb the error, can the process
be changed so that it costs less? The proposal (`docs/adaptations-proposal.md`)
listed candidate changes; these are the ones that ran. Paper: Appendix H
(`app:adapt`, `tables/adaptations.tex`).

**The estimand had to change.** Rating a design twice, re-rating before deploying
or re-asking about the incumbent also changes the *clean* run, and a repeated
rating of an exact objective carries no information. Excess over a method's own
clean run therefore flatters any change that handicaps its clean twin. Every change
is scored against the **standard** process instead. Per paired cell (landscape,
acquisition, magnitude, onset, seed), regret as a fraction of opt_z:

    recovered = (R_std,noisy − R_adapt,noisy) / (R_std,noisy − R_std,clean)
    price     =  R_adapt,clean − R_std,clean

Aggregated as ratios of landscape means with landscape-bootstrap intervals and
Wilcoxon tests (BH within arm × response), on two responses: the post-onset
trajectory (`auc_simple_regret_true_postonset_per_iter`) and the deployed design
(`final_inference_simple_regret_true`). Where the change is a different
acquisition (qKG, replEI) the reference is the mean over the ten standard
acquisitions (pooled mode). Extra trials are counted against the standard
process's clean run (`analyse_extra_runs.py --baseline-dir`). Script:
`scripts/analyse_boba_adaptations.py` → `output-boba/analysis/adaptations_recovery.csv`,
`adaptations_extra_runs.csv`.

**What was built.** Driver flags `--replicate-first N` (rate each of the first N
proposals twice), `--final-rerate TOP,REPS` (re-rate the TOP designs by mean
rating REPS times in the last TOP×REPS trials), `--input-noise-model nigp`
(first-order noisy-input GP: each point's noise variance += ‖∇μ‖² · slip
variance, refit), `--likelihood student_t` (variational Student-t GP,
`scripts/robust_gp.py`) and `--inference-rule best_mean` (default when
replicating). In adapted arms the clean-run files keep the plain
`_baseline_exact.csv` name but are adapted runs. Runners: `run_boba_adapt.ps1`,
`run_boba_adapt_queue.ps1`, `run_boba_adapt_studentt.ps1`, `run_fitted_adapt.ps1`.

**Results** (20 landscapes, seeds 7–11, onsets 1 and 21, four magnitudes, T = 50,
LogEI + qNEI unless stated; recovered in % of the standard process's cost, 95%
landscape-bootstrap interval; price in points of opt_z without error):

| change | error | recovered, trajectory | recovered, deployed | price traj / deployed |
|---|---|---|---|---|
| observed-max incumbent (re-score of §7.8) | gaussian | +19 [10, 27] | −10 [−15, −5] | 0.1 / 0.2 |
| GP given the true noise variance (re-score of §7.6) | gaussian | −3 [−16, 8] | −3 [−11, 5] | 0.4 / 0.0 |
| first ten proposals rated twice (E4) | gaussian | −88 [−113, −67] | −14 [−21, −8] | 5.1 / 1.6 |
| … plus observed-max incumbent, LogEI (E1) | gaussian | −93 [−127, −58] | −44 [−57, −32] | 4.7 / 1.1 |
| … with observed-max incumbent, vs the incumbent arm | gaussian | −88 [−107, −63] | −20 [−31, −9] | 3.9 / −0.5 |
| last six trials re-rate the top three (E5) | gaussian | −1 [−2, −1] | +4 [0.04, 9] | 0.1 / 0.8 |
| qKG (pooled reference) | gaussian | −4 [−23, 14] | −32 [−42, −24] | 2.7 / 3.6 |
| re-rate the incumbent every second trial (pooled) | gaussian | −61 [−88, −33] | −5 [−12, 3] | 5.4 / 4.3 |
| noisy-input GP (E3) | unnoticed slip | −28 [−43, −13] | −19 [−31, −9] | −0.1 / −1.5 |
| last six trials re-rate the top three (E5) | unnoticed slip | 0 [−0.3, 0.2] | −6 [−15, 1] | 0.1 / 0.8 |
| first ten rated twice, three fitted oracles (E6) | gaussian | −161 [−3504, −49] | −8 [−17, 6] | — |
| Student-t likelihood (E2) | misclick | −2 [−38, 31] | −13 [−50, 16] | 0.2 / 1.5 |

Per-condition structure worth keeping:

- **Incumbent.** Trajectory recovery with error from trial 1 is +67 / +47 / +24 / +10%
  at 0.05 / 0.25 / 1 / 5σ; the deployed-design loss is concentrated at the late
  onset (−27% at 1σ, −36% at 5σ). Reason not isolated.
- **qKG** helps with early error at ≥0.25σ (+43 / +12 / +23% of the trajectory
  cost at 0.25 / 1 / 5σ) and hurts with late error (−146 / −114 / −48%); this early benefit is part of what the old "recovers a third"
  was seeing; the rest was the handicapped clean runs.
- **Re-rate before deploy** reaches +18% [5, 29] of the deployed cost at 5σ late.
  Under slips it hurts with large late slips (−36% at 15%, −74% at 40%).
- **Noisy-input GP** recovers +76% [27, 245] of a 1% slip and +17% of a 5% slip
  from trial 1, and −129% / −205% of 15% / 40% slips from trial 21. Likely but untested reason:
  late in the run the search sits near a steep peak whose flanks have the largest
  gradients, so the observations that locate the peak are discounted most.
- **Student-t likelihood** (E2, misclick) recovers −1% [−10, 6] of the trajectory
  cost at 40% misclicks from trial 1, where misclicks cost most, and no cell or pooled
  value is distinguishable from zero. It fits a variational Student-t GP
  (`scripts/robust_gp.py`); the arm had no acquisition-optimisation failures.

**Extra trials to match a clean standard 10-trial study, error from trial 1,
within T = 50, tolerance 1%** (median, never-reached share; the reference differs
by arm because it is restricted to the arm's own acquisitions):

| arm | magnitudes | arm | standard process |
|---|---|---|---|
| incumbent | 0.05 / 0.25 / 1 / 5σ | 0 (0%) / 1 (2%) / 6 (13%) / 21 (39%) | 0 (1%) / 1 (6%) / 8 (21%) / 36 (48%) |
| known noise | 0.05 / 0.25 / 1 / 5σ | 0 (1%) / 1 (6%) / 5 (19%) / 22 (43%) | 0 (0%) / 0 (3%) / 5 (16%) / 25 (43%) |
| first ten rated twice | 0.05 / 0.25 / 1 / 5σ | 2 (1%) / 4 (5%) / 8 (17%) / 31 (46%) | 0 (1%) / 1 (2%) / 6 (18%) / 25 (42%) |
| … plus incumbent (LogEI) | 0.05 / 0.25 / 1 / 5σ | 2 (1%) / 5 (4%) / 12 (15%) / 34 (38%) | 0 (0%) / 1 (2%) / 5 (21%) / 26 (46%) |
| re-rate top three | 0.05 / 0.25 / 1 / 5σ | 0 (2%) / 1 (2%) / 6 (22%) / 25 (44%) | 0 (1%) / 1 (2%) / 6 (18%) / 25 (42%) |
| noisy-input GP | 0.01 / 0.05 / 0.15 / 0.4 slip | 1 (5%) / 3 (8%) / 8 (24%) / 30 (43%) | 1 (4%) / 3 (7%) / 8 (18%) / 35 (46%) |
| re-rate top three | 0.01 / 0.05 / 0.15 / 0.4 slip | 1 (2%) / 3 (5%) / 8 (18%) / 35 (46%) | 1 (4%) / 3 (7%) / 8 (18%) / 35 (46%) |
| Student-t likelihood | 0.01 / 0.05 / 0.15 / 0.4 misclick | 0 (0%) / 0 (1%) / 1 (9%) / 8 (24%) | 0 (0%) / 0 (3%) / 1 (11%) / 8 (28%) |

**E7** (frag from a pilot) is §7.10: a ten-trial pilot works as well as the exact frag.

**Corrections this supersedes.**

- §7.12's "qKG/replication recover a third at 1σ and a quarter at 5σ" was excess
  over each method's own clean run, and both clean runs are handicapped (replEI
  spends every second trial on a zero-information repeat; qKG's clean run is 2.7
  points worse than the standard one). Withdrawn; the paper's robust-baselines
  paragraph now says neither recovers the cost.
- The incumbent's 15.6% and known noise's 5.0% (§7.8, §7.12) are also excess over
  their own clean runs. Against the standard process the incumbent recovers +19% of
  the trajectory cost but −10% of the deployed design's, and known noise −3% on
  both. The earlier shares stay in the paper as descriptions of those ablations.
- The proposal's recommendations A2 (noise-aware acquisition) and A3 (early
  replication) are refuted; A4 (re-rate before deploying) is small; S3 (noisy-input
  GP) is mixed and net harmful; S4 (Student-t likelihood) has no effect.
## 8. The paper

`paper/` holds an ICLR 2026 draft against the **official** style files
(`iclr2026_conference.sty`/`.bst`, `fancyhdr.sty`, `natbib.sty`, downloaded from
ICLR/Master-Template). Every table under `paper/tables/` is generated by
`scripts/make_boba_paper_tables.py` from the analysis CSVs, so the paper cannot
drift from the sweep; rebuild rather than edit. `scripts/check_paper.py` runs
static checks (missing `\input` targets, dangling `\ref`s, unknown citation
keys, malformed generated tables); the document also compiles under TeX Live
2026 with no errors, undefined references or overfull boxes, and the main text
through the Limitations paragraph ends on page 9 (the Reproducibility and
LLM-usage statements do not count toward ICLR's limit). `\iclrfinalcopy` is
commented out and the author block is a TODO: a non-anonymous submission is
rejected without review.

## 9. How to reproduce

```powershell
$env:PYTHON = "C:\Users\markc\AppData\Local\Programs\Python\Python312\python.exe"
powershell -ExecutionPolicy Bypass -File run_boba_pipeline.ps1
```

Resumable: re-run after any interruption and completed runs are reused.
