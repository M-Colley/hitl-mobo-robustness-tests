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
ground truth.** Cross-validated R² of the selected oracles:

| dataset / objective | oracle | CV R² (individual) | CV R² (per-design mean) |
|---|---|---|---|
| opticarvis / composite | gradient_boosting | **0.52** (held-out participant) | — |
| ehmi / composite | extra_trees | 0.14 (held-out participant) | **0.55** (held-out design) |
| provoice / composite | extra_trees | −0.03 | −0.19 (no signal either way) |

**Oracle target.** Individual ratings ask the oracle to predict between-participant
variance from design parameters alone, capping R². For **ehmi** the project now
fits a **per-design *mean* ("average human") oracle** (`oracle_target: "mean"` in
`datasets.json`), validated on held-out *designs* — the regime BO actually faces.
That raises ehmi's R² from 0.14 → **0.55**. `opticarvis` already has a strong
design→rating signal and keeps the individual-row oracle; `provoice` has ≈ 0 / negative
R² for every model and target, so it must be read as a **pure synthetic function**.
The simulator's injected feedback noise models the individual deviation that
averaging removes. Full record in [`best_oracle_models.json`](best_oracle_models.json).

> Note: the figures/tables in *this* snapshot were produced with the earlier
> individual-row ehmi oracle and the composite objective only. Re-running
> `run_full_pipeline.ps1` refreshes them with the mean ehmi oracle **and** the
> multi-objective results (now feasible — see below).

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

## Multi-objective (`multi_objective`) — feasible, run by the driver

Not in *this* snapshot (which is composite-only), but the 5-objective Pareto
variant **is feasible** and `run_full_pipeline.ps1` runs it. What we measured
per run at 5 objectives:

| acquisition | sampling | peak RAM | per run |
|---|---|---|---|
| qEHVI | full (raw 512 / mc 256) | 22 GB | 414 s |
| **qEHVI** | **reduced (raw 128 / mc 64)** | **6.1 GB** | **141 s** |
| qLogEHVI | reduced, no MSVC `cl` on PATH | 56 GB | hours (broken) |
| qLogEHVI | reduced, `cl` exposed via `vcvars64` | 7.5 GB | 559 s |

The 22 GB at full sampling — not the box-decomposition — was the wall; **reduced
sampling drops plain qEHVI/qNEHVI to ~6 GB**, so ~8 workers fit in 64 GB without
the `BrokenProcessPool` OOM. The driver therefore runs multi-objective with plain
`qEHVI/qNEHVI` at reduced sampling. (The numerically-stable `qLogEHVI/qLogNEHVI`
are also implemented and become tractable if `cl` — already installed with VS
2022 — is exposed via `vcvars64`; they are ~2–4× slower.)

## Reproducing

```powershell
powershell -ExecutionPolicy Bypass -File run_full_pipeline.ps1
```

Resumable (uses `--resume`), suppresses system sleep while running, and writes
`output/PIPELINE_COMPLETE` on success. Figures land in `output/evaluation/figures/`.
