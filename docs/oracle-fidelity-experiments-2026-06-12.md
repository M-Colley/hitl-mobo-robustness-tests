# Oracle ("simulated human") fidelity: experiments and verdict — 2026-06-12

Six parallel modeling experiments against a shared harness ([sandbox/oracle_experiments/](../sandbox/oracle_experiments/)), every result adversarially verified for protocol violations and re-run (all six confirmed). Two evaluation protocols:

- **cold** = pooled model, 5-fold GroupKFold over users — "predict an unseen user" (this is what `select_best_oracle_model.py` measures).
- **warm** = within-user split, first 70% of a user's session → train, rest → test — "predict the *same* user later in their session", which is the oracle's actual job in the simulation. Warm R² is structurally pessimistic (late-session BO evaluations cluster near the optimum, so test variance collapses); judge warm by **RMSE vs the intra-rater noise floor** (`output/noise_calibration.csv`: ehmi 0.27, opticarvis ~0.66 (see below), provoice ≤0.65).

## Verdict per dataset

### ehmi — the oracle is near its ceiling; the low cold R² is an identity, not a deficiency
- Variance decomposition: **63.4% of composite variance is between-user intercepts**, which no pooled model can predict for unseen users. Cold ceiling ≈ **0.226**; deployed extra_trees achieves 0.152 = **67% of ceiling**. TabPFN reference reaches only 0.162 — the model class is maxed out.
- Warm: ceiling ≈ 0.847; baseline R² 0.635 / RMSE 0.416 = **75% of ceiling**. Best deployable warm variant: **kNN (k=10, distance weights, scaled): RMSE 0.388**.
- Hyperparameter tuning (nested), feature engineering (HSV/luminance/poly2), and the model zoo all failed to beat the deployed config cold. Remaining error is user heterogeneity, not capacity or representation.

### opticarvis — honest and at/above measurable ceiling
- Cold 0.651 / RMSE 0.739 (after the observation_glob fix; previously a fake 0.9999999).
- The calibrated noise floor 1.25 is **overestimated** (in 16-dim parameter space "close" NN pairs aren't close, so the estimate absorbs real design variance); the user-train-mean oracle implies true intra-rater SD ≲ 0.66. Even so, cold RMSE is already at rater repeatability.
- Warm: per-user trees overfit badly (RMSE 0.922, worse than predicting the user's mean, 0.660). Regularized models fix this: **SVR RBF C=1: RMSE 0.210**, ridge 0.664-0.674. Lesson: for per-user simulation, use regularized/linear models, not trees.

### provoice — two real problems found, one fixed, one needs a decision
1. **Composite scale bug (paper-relevant, affects the running sweep):** Mental Demand spans 1–17 (NASA-TLX-style 0–20) vs 1–5 for Predictability/Usefulness. Signed variances 8.24/1.08/0.89 → **−Mental Demand is 80.7% of composite variance (r = 0.965)**. The provoice "composite" is effectively negative mental demand. Fix: rerun provoice with `--normalize-objective` (rescales constructs to [0,1] before averaging) — note this changes the composite's scale, so jitter-std levels need re-interpretation/re-calibration for that run.
2. **Warm fidelity was an evaluation artifact, now fixed in the harness protocol:** `infer_oracle_groups` merges each participant's two condition sessions under one User_ID, and `rglob` orders "Condition 2" before "Condition 1" while timestamps show Condition 1 mostly ran first — the "temporal" split was anti-chronological. Grouping by session file (the only reliable session key; P17's Condition_ID is mislabeled): RMSE 0.851 → **0.631**, below the ≤0.65 noise floor; adding the iteration index as a context feature: **0.594** (R² +0.138). Within-session simulation fidelity for provoice is fine.
3. Cold remains ≈0 R² for structural reasons: 57% of rows are a shared initial design rated by all sessions (64.8% of composite variance lies within identical-X rows), and no construct carries cross-user signal. Tuning crosses zero (+0.005) at best. Treat provoice cold as at-its-data-ceiling.

## What this means for the paper
- Report oracle fidelity as **% of the heterogeneity/noise ceiling**, not raw R²: ehmi 67% (cold) / 75% (warm), opticarvis at ceiling, provoice within-session at noise floor.
- The pooled extra_trees/gradient_boosting oracle selection is confirmed near-optimal — no change to the deployed oracle.
- For a per-user (personalized) oracle mode, use regularized models: kNN-k10 (ehmi), ridge/SVR (opticarvis), small ExtraTrees msl=5 + iteration feature (provoice).
- Flag the provoice composite scale bug and either rerun provoice normalized or scope provoice composite results out.

Experiment scripts: `sandbox/oracle_experiments/exp_{zoo,tuned_trees,features,warm,ceiling,provoice}.py`; shared protocol in `harness.py`. Baseline reproduction is asserted inside each script.
