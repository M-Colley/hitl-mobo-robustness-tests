# Human-feedback benchmark: frozen selection, held-out result (2026-11-05)

This is the first held-out result of the finite-candidate feedback benchmark (`docs/HUMAN_FEEDBACK.md`).
It is not in the paper; whether it belongs there is the authors' decision.

## Design, fixed before any held-out run

- **Screen:** `configs/feedback_screen.json`, seeds 7–11, 39,200 tasks. The output is in `output-feedback-screen/` (git-ignored).
  - 20 landscapes, eight methods (`bo_logei`, `random`, `sobol`, `cem`, `ridge_ucb`, `dueling_ucb`, `reward_policy`, `dpo`).
  - Their ship rules: `observed_max`, `policy`, `rank_mean`, `uniform_ties`.
  - Six error processes (gaussian, bias, drift, ar1, spike, position), magnitudes 0.05, 0.25, 1 and 5, onsets 0 and 20, budget 50.
- **Choice:** `python scripts/feedback_select.py freeze` wrote `configs/feedback_frozen_selection.json` on 2026-11-05T19:31Z.
  - The file records the sha256 of every screen output it read and of the script itself.
  - Its own sha256 was recorded before the held-out run started. It was unchanged when scored.
  - The estimand is the repository's: the deployed design's regret at the final judgment, gain against the standard process (`bo_logei`, best observed rating), price on the clean twins, landscape means.
  - **single:** the one (method, ship rule) with the largest V = (sum over the 48 cells of the gain − the price) / 49. The choice is `cem:observed_max`, with V +0.019 on the screen.
  - **map:** per cell, the procedure with the largest screen gain, or the reference where none gains. It departs from the reference in 22 of 48 cells.
- **Held out:** `configs/feedback_heldout.json`, seeds 12–16, the same 39,200-task plan, in `output-feedback-heldout/` (git-ignored).
- **Scoring:** `python scripts/feedback_select.py score`.
  - Landscape bootstrap, 2,000 resamples.
  - Two-sided Wilcoxon over the 20 landscapes.
  - Holm over the cells where the policy acts, separately for single and map.
  - These families belong to this analysis only.

## Result (seeds 12–16, fractions of opt_z)

| Policy | V [95% CI] | Landscapes ahead | Price without error |
|---|---|---|---|
| single (`cem:observed_max`) | +0.017 [−0.012, +0.046] | 12/20 | 0.025 |
| map | +0.073 [+0.054, +0.092] | 19/20 | none: it uses the reference when there is no error |
| map − single | +0.056 [+0.029, +0.081] | 17/20, Wilcoxon p 0.0006 | |

**The single procedure** loses a little in most cells up to 1σ, where it mostly pays its price of about 0.025 (−0.061 [−0.109, −0.013] under gaussian error at 1σ from the first rating). It gains a lot where large error arrives late:
- spike 5σ from trial 21: +0.356 [0.238, 0.489], Holm p 2.7e-4
- spike 1σ from trial 21: +0.242 [0.161, 0.332], Holm p 9.2e-5
- gaussian 5σ from trial 21: +0.141 [0.053, 0.237], Holm p 0.28
- drift 5σ from trial 21: +0.131 [0.068, 0.191], Holm p 0.056
- ar1 5σ from trial 21: +0.138 [0.048, 0.228], Holm p 0.40

**Map cells that survive Holm** (with the chosen procedure's price, which the map pays only in a misjudged cell):

| Cell | Procedure | Held-out gain [95% CI] | Landscapes | Holm p | Price |
|---|---|---|---|---|---|
| ar1, 5σ, trial 1 | dpo | +0.433 [0.351, 0.515] | 20/20 | 4.2e-5 | 0.122 |
| ar1, 5σ, trial 21 | dpo | +0.399 [0.314, 0.490] | 19/20 | 7.6e-5 | 0.122 |
| drift, 5σ, trial 1 | dpo | +0.155 [0.090, 0.222] | 16/20 | 0.003 | 0.122 |
| drift, 5σ, trial 21 | dpo | +0.272 [0.179, 0.366] | 17/20 | 0.001 | 0.122 |
| spike, 0.25σ, trial 1 | bo_logei:rank_mean | +0.262 [0.167, 0.361] | 18/20 | 9.1e-4 | 0.087 |
| spike, 0.25σ, trial 21 | bo_logei:rank_mean | +0.248 [0.158, 0.337] | 19/20 | 0.002 | 0.087 |
| spike, 1σ, trial 1 | dpo | +0.316 [0.208, 0.427] | 18/20 | 0.001 | 0.122 |
| spike, 1σ, trial 21 | bo_logei:rank_mean | +0.410 [0.313, 0.511] | 19/20 | 0.002 | 0.087 |
| spike, 5σ, trial 21 | reward_policy | +0.440 [0.281, 0.618] | 20/20 | 4.2e-5 | 0.264 |

No acting cell under gaussian or bias error survives Holm. The map's choice at gaussian 1σ from the first rating is `bo_logei:policy`: +0.016 [−0.009, +0.041], price 0.036.

## Reading

- **Consistent with the paper:** at ordinary noise nothing beats the standard process measurably. Under large or structured error, procedures that read the ratings robustly (pairwise or ranks, `dpo`, `rank_mean`) recover a lot. Their prices without error are large (0.09–0.26), so a study should choose them only knowing the fault.
- **The map's value is conditional:** it assumes the cell, meaning the process, the magnitude and the onset, is known. Any single procedure must pay its price on error-free runs.
- **Scope of the benchmark:**
  - finite catalogs of 512 candidate designs;
  - one comparison costs as much as one rating (`comparison_cost: 1`);
  - simulated raters.
  It does not reproduce the paper's continuous BO loop.
- **Position error costs the rating-based reference nothing, by construction:** it is a preference for the first design of a pair (`docs/HUMAN_FEEDBACK.md`), and scalar ratings have no pair order. Its cells therefore show only each method's price and say nothing about robustness.

## Commands

```
python scripts/human_feedback_benchmark.py --config configs/feedback_screen.json --output-dir output-feedback-screen --workers 16 --run
python scripts/feedback_select.py freeze --screen output-feedback-screen --out configs/feedback_frozen_selection.json
python scripts/human_feedback_benchmark.py --config configs/feedback_heldout.json --output-dir output-feedback-heldout --workers 14 --run
python scripts/feedback_select.py score --heldout output-feedback-heldout --selection configs/feedback_frozen_selection.json
```

The last command writes `output-feedback-heldout/analysis/frozen_selection_scores.{csv,txt}`.

The runs used Python 3.12.9, torch 2.13.0+cu126 (CPU only), BoTorch 0.18.2.dev23 (eea35f51a), NumPy 2.5.3 and SciPy 1.18.1. Each run directory's `manifest.json` records them.
