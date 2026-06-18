# Results — BO Robustness Under Noisy Human Feedback

This folder is a curated, browsable snapshot of one full evaluation run. The
raw per-run logs (`output/`) are intentionally git-ignored; everything needed to
read the result is reproduced here. Regenerate end-to-end with
`run_full_pipeline.ps1` (resumable).

## What was run

- **Datasets:** `ehmi`, `opticarvis`, `provoice` (three archival HCI studies).
- **Objective:** `composite` (scalarized single objective) — see the
  multi-objective note below for why the Pareto/hypervolume variant is excluded.
- **Acquisitions (12):** EI, logEI, PI, logPI, UCB, qUCB, qEI, qPI, qNEI, Greedy,
  plus the model-free floors Random and Sobol.
- **Noise grid:** error models `{gaussian, bias}` × std `{0.05, 0.5, 1.0, 5.0}`
  (× the response scale) × jitter onset `{10, 20, 40}` = 24 noisy conditions,
  each paired against a clean baseline.
- **Seeds:** 20 per condition (seeds 7–26). 24,000 BO runs of 50 iterations.
- **Coverage:** 17,280 paired baseline/jitter rows, **20 seeds in every
  condition**, **0 acquisition-optimization fallbacks**.

## Oracle fidelity — read this first

The "human" is a regression oracle fit on archival study data, so every result
below describes robustness on **data-derived synthetic test functions, not human
ground truth.** Cross-validated (group-k-fold, held-out participant) R² of the
selected oracles:

| dataset / objective | oracle | CV R² |
|---|---|---|
| opticarvis / composite | gradient_boosting | **0.52** (decent) |
| ehmi / composite | extra_trees | 0.14 (weak) |
| provoice / composite | extra_trees | −0.03 (≈ chance) |

Only `opticarvis` has a meaningfully predictive oracle. Treat `ehmi` and
especially `provoice` as synthetic surfaces with little human grounding. Full
record in [`best_oracle_models.json`](best_oracle_models.json).

## Headline findings

**1. Report absolute performance, not excess regret alone.** Ranked purely by
*excess* AUC regret (jittered − baseline), the model-free floors `Random`/`Sobol`
"win" almost everywhere (see [`figures/winner_map_composite.png`](figures/winner_map_composite.png))
— but only because a method that never learns has ~0 excess by construction. On
the deployment metric (absolute noisy-run AUC, lower = better) they are the
*worst*:

| dataset | best model-based (abs. AUC) | Random (abs. AUC) |
|---|---|---|
| ehmi | UCB ≈ 4.5 | 15.7 |
| opticarvis | qNEI ≈ 187 | 200 |
| provoice | qNEI ≈ 28.7 | 35.6 |

This is the exact trap the project README warns about, reproduced cleanly.

**2. Model-based acquisitions degrade gracefully under noise.** The Cohen's-dz
forest plots (`figures/forest_effect_sizes_*`) show the jittered−baseline effect
is small-to-moderate and positive (noise hurts a little); UCB/EI families are the
most affected, qNEI/Greedy/logEI the least. The per-iteration trajectories
(`figures/regret_trajectory_*`) show the jittered and baseline curves staying
close until large noise (std ≥ 1).

**3. Statistical power is real at 20 seeds.** Per-condition Friedman omnibus
tests are significant (FDR-corrected) across most of the noise grid, with
Kendall's W up to ~0.5 — none flagged underpowered. Larger noise and earlier
onset produce the strongest acquisition differences.

See [`evaluation_report.txt`](evaluation_report.txt),
[`overall_rankings.csv`](overall_rankings.csv), and
[`condition_rankings.csv`](condition_rankings.csv) for the full numbers.

## Multi-objective (`multi_objective`) — excluded, and why

The 5-objective Pareto variant (`ehmi`, `opticarvis`) is **computationally
infeasible on commodity hardware** and is therefore not part of this snapshot:

- Exact hypervolume box-decomposition blows up at 5 objectives: a single
  `qEHVI`/`qNEHVI` run peaks at **~22 GB**, and 20 parallel workers OOM-crash the
  process pool (`BrokenProcessPool`).
- The numerically-stable log variants (`qLogEHVI`/`qLogNEHVI`, now implemented in
  the simulator) do not reduce that cost — the box-decomposition dominates and is
  independent of the MC-sample count. On this Windows box they were worse still
  (~42–56 GB and pathologically slow, because the smooth-max path falls back to a
  pure-Python implementation when the MSVC `cl` compiler is absent).

`provoice` multi-objective has only 3 objectives and is tractable; it can be added
on a high-RAM machine, or by switching to a scalarization-based MO method
(e.g. ParEGO) that avoids hypervolume entirely. The single-objective `composite`
study above answers the core research question on its own.

## Reproducing

```powershell
powershell -ExecutionPolicy Bypass -File run_full_pipeline.ps1
```

Resumable (uses `--resume`), suppresses system sleep while running, and writes
`output/PIPELINE_COMPLETE` on success. Figures land in `output/evaluation/figures/`.
