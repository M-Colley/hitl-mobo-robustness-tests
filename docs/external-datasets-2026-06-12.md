# External datasets: feasibility, conversion, and verdicts — 2026-06-12

Four candidates from the data-sourcing research were downloaded, converted where possible, and fidelity-tested through the existing pipeline (verification agents independently reproduced every claim). Two are includable as benchmark domains and are ready to run via [datasets-extended.json](../datasets-extended.json); `datasets.json` is untouched so the main sweep's scope is stable.

| dataset | verdict | role | oracle signal (cold R²) | noise |
|---|---|---|---|---|
| **NISQA-sim** (speech quality) | **benchmark** ✓ | tune transmission params → MOS | **0.589** (strongest in suite) | inter-rater SD 0.74–0.99/5-pt; panel-mean SEM ~0.30 |
| **ASHRAE DB II** (thermal comfort) | **benchmark** ✓ | tune climate params → comfort | 0.02 user-grouped, ~0 building-grouped | intra-rater SD ~0.98 on 1–6 scale |
| KonIQ-10k (image quality) | calibration only | rating-noise anchor | n/a (no design space, no rater IDs) | inter-rater SD 0.575/5-pt, heteroscedastic (peaks mid-scale) |
| AdaptiFont / Koyama SLS | not usable | — | only 11 cluster centroids / code-only | — |

## NISQA-sim (`external_datasets/nisqa_sim/`, 12,500 rows × 28 params)
The simulated subsets publish the full distortion parameters per clip (bandpass cutoffs, frame-erasure rate, SNRs, codec chain with bitrate mode and packet loss...). Framing: "tune transmission parameters for perceived quality" — a real parameterized design space with 5 rating dimensions (MOS, noisiness, coloration, discontinuity, loudness; all 1–5). Metadata extracted via HTTP range requests into the 16 GB Zenodo zip — audio never downloaded (~20 MB on disk).
Caveats for the paper: ratings are ~5-vote **crowd-panel means** (the oracle simulates a small panel, not one human — use single-vote SD 0.74 to simulate one rater, SEM 0.30 for the panel label); grouping is by source **speaker** (content), since no rater IDs exist; codec identities are ordinal-encoded; license is non-commercial research.

## ASHRAE DB II (`external_datasets/ashrae/`, 6,684 rows, 4,036 subjects, 112 buildings, CC0)
Six classical PMV inputs (air/radiant temperature, humidity, air velocity, met, clo) → ThermalComfort (1–6) and |ThermalSensation| (precomputed so `-ThermalSensationAbs` expresses "maximize neutrality"). Honest but **noise-dominated**: cold R² ≈ 0.02 (subject-grouped) and ≈ 0 (building-grouped) against an intra-rater noise SD of ~0.98 on a 1–6 scale. Include it as the *hard, high-noise* regime — it stress-tests exactly what the paper is about — or use it purely as an external noise-calibration reference. Note met/clo are occupant covariates, not controllable setpoints; parameter ranges are narrow (climate-controlled offices). A wider-range variant (~27k rows, building-grouped) is prepared in `sandbox/external_datasets_eval/variant_b_building.py`.

## KonIQ-10k — calibration numbers for the paper
"Crowd image-quality ratings on a 5-point ACR scale show a per-stimulus inter-rater SD of 0.57 on average (IQR 0.53–0.61), ~14% of the scale range, with noise variance ≈ between-stimulus signal variance (single-rating reliability 0.48)." Noise is heteroscedastic — largest mid-scale (0.60 at MOS 2.5–3.0), smaller at the extremes — which empirically motivates the heteroscedastic error model direction.

## AdaptiFont / SLS / near-misses
AdaptiFont publishes only 11 per-subject font centroids (no trials); SLS is code-only. Best author-contact target found: the Tactons vibrotactile dataset (arXiv 2502.00268 — 154 parameterized stimuli × 36 participants, roughness/valence/arousal on 0–100 scales) — ideal shape, no public deposit.

## Do they actually contribute to the error-robustness experiment?

The "human simulation" has two halves: the **oracle** = the preference surface f(x), and the **error model** = the injected rating noise. `apply_sensor_error` reads only the CLI `jitter_std` — never the dataset — so a dataset contributes **only through the shape of f(x)**; the empirical noise is used to *calibrate* the noise grid, not injected. So external datasets enrich the preference half, not the noise half.

As landscapes they are genuinely distinct regimes ([sandbox/landscape_snr.py](../sandbox/landscape_snr.py), extra_trees oracle, uniform design-space sample vs calibrated noise SD):

| dataset | dims | SNR_std (f_std / noiseSD) | SNR_top ((y_opt−p95)/noiseSD) | oracle cold R² |
|---|---|---|---|---|
| ehmi | 9 | 1.14 | 0.92 | 0.14 |
| opticarvis | 16 | 0.09 | 0.32 | 0.52 |
| provoice | 4 | 0.33 | 0.44 | ~0 |
| **ashrae** | 6 | 0.50 | 0.49 | 0.02 |
| **nisqa_sim** | 28 | 0.32 | **0.89** | **0.59** |

Two takeaways: (1) at empirically-realistic noise, signal ≈ noise on **every** dataset (SNR_std 0.09–1.14) — the paper's premise, quantified. (2) **NISQA is the standout new landscape**: the only oracle that genuinely predicts held-out humans (R² 0.59) *and* a sharp, identifiable optimum (SNR_top 0.89) in a 28-D space — the first non-degenerate, trustworthy test bed. **ASHRAE's oracle is near-flat** (R² 0.02, SNR ~0.5) — it adds domain breadth and a noise-dominated stress regime, but the error test on it risks a null result (little signal for noise to corrupt). Whether they change the *conclusions* (do robustness rankings differ across SNR regimes?) is answered only by the full extended sweep.

## How to run the extended suite
```bash
python scripts/select_best_oracle_model.py --dataset-config datasets-extended.json --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting --cv-folds 5 --output-path output/extended/best_oracle_models.json
python scripts/bo_sensor_error_simulation.py --dataset-config datasets-extended.json ...
```
Calibrated noise grids per dataset: ehmi 0.27·{½,1,2,4}, nisqa_sim 0.74·{...} (single-rater) on 1–5, ashrae 0.98·{...} on 1–6 — i.e. set `--jitter-stds` per dataset run, or report noise as multiples of each dataset's empirical SD.

Conversion/verification scripts: `sandbox/external_datasets_eval/`. Verification verdicts: ashrae **confirmed**, nisqa_sim **confirmed** (numbers reproduce within 0.001; pipeline-load checks pass; provenance URLs resolve).
