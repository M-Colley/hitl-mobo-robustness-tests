"""The fitted-oracle companion arm under each optimum estimator.

Every run seed refits the oracle, so analyse_fitted_companion.py forms the floor
gap within each seed's oracle and takes each dataset's fraction as the ratio of
seed means, mean_s(excess_s) / mean_s(gap_s). This prints, for the corrected
data (output-fitted, output-fitted-noaug) and the as-logged data (the
output-fitted*-prefix arms for opticarvis and provoice, whose logging defects
the fix removed; ehmi was unaffected and is read from output-fitted), each
dataset's post-onset trajectory fraction under

  pooled      the estimator used up to 2026-09-28 (maximum clean best over all
              seeds' oracles less the floor pooled over all seeds), for reference
  oracle      the primary: each seed's logged y_opt
  best_clean  the sensitivity: the best value any clean run of the seed reached

with the arm mean, the known-function arm's mean on the same denominator, their
ratio with the two-sample bootstrap of scripts/review_checks/c6_ratio.py
(landscapes and datasets resampled independently, seed 20260910), and how many
of the twenty landscapes' known-function fractions each dataset's 1 sigma,
first-rating fraction exceeds. It also counts the seeds on which any run beat
the logged y_opt. Then the same on the achievable improvement, the paper's
headline unit (analyse_fitted_companion.share_fraction): the known-function
arm's excess / opt_z against each dataset's mean_s(excess_s) / mean_s(optimum_s
- box mean_s), the box mean estimated from the designs of the seed's clean
random and Sobol runs (analyse_fitted_companion.box_means), for the oracle and
best_clean optima. Reads the evaluated logs and the floor runs' logs only.

    python scripts/review_checks/oracle_companion_estimators.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))
import analyse_boba_robustness as ab  # noqa: E402
import analyse_fitted_companion as fc  # noqa: E402
import boba_benchmarks as bb  # noqa: E402


def pooled_optimum(paired: pd.DataFrame) -> pd.Series:
    """The committed estimator up to 2026-09-28, indexed by dataset only."""
    return paired.groupby("dataset")[fc.BASELINE_BEST].max()


def fractions(paired: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"pooled": fc.floor_fraction(paired, pooled_optimum(paired))}
    for how in fc.OPTIMA:
        out[how] = fc.floor_fraction(paired, fc.fitted_optimum(paired, how))
    return out


def beaten(paired: pd.DataFrame) -> str:
    y_opt = fc.fitted_optimum(paired, "oracle")
    best = paired.groupby(["dataset", "seed"])[["final_best_true_baseline", "final_best_true_jitter"]].max()
    over = best.sub(y_opt, axis=0)
    parts = []
    for ds, block in over.groupby(level="dataset"):
        parts.append(f"{ds}: clean {int((block.final_best_true_baseline > 1e-12).sum())}/{len(block)}, "
                     f"noisy {int((block.final_best_true_jitter > 1e-12).sum())}/{len(block)} seeds")
    return "; ".join(parts)


def report(frames: dict[str, pd.DataFrame], synth_cells: pd.Series) -> None:
    """Per estimator, magnitude and onset: each dataset, the arm mean, the known-function
    mean, their ratio with the two-sample bootstrap, and the landscapes each dataset exceeds."""
    for how, frame in frames.items():
        cells = frame.groupby(["sigma_multiple", "jitter_iteration", "dataset"])["frac"].mean()
        print(f"   -- optimum {how}")
        for (mag, onset), block in cells.groupby(level=[0, 1]):
            per = block.droplevel([0, 1])
            s = synth_cells.loc[(mag, onset)]
            line = "  ".join(f"{d} {v:6.3f}" for d, v in per.items())
            ratio = s.mean() / per.mean()
            rng = np.random.default_rng(20260910)
            fv, sv = per.to_numpy(), s.to_numpy()
            draws = np.array([sv[rng.integers(0, len(sv), len(sv))].mean()
                              / fv[rng.integers(0, len(fv), len(fv))].mean() for _ in range(2000)])
            lo, hi = np.percentile(draws, [2.5, 97.5])
            above = "  ".join(f"{d}>{int((s < v).sum())}" for d, v in per.items())
            print(f"      {mag:5g} sigma onset {onset:2d}: {line}  mean {per.mean():6.3f}  "
                  f"known-function {s.mean():6.3f}  ratio {ratio:5.2f} [{lo:.2f}, {hi:.2f}]  "
                  f"fitted/known {per.mean() / s.mean():5.2f}  landscapes below: {above}")


def main() -> None:
    synth_raw = ab.load_paired(REPO / "output-boba")
    synth_raw = synth_raw[synth_raw.error_model == "gaussian"]
    # opt_z from the tracked statistics file, not the git-ignored run_metadata.json (which
    # records local paths); the two agree exactly on the main sweep's twenty landscapes.
    optz = pd.Series({k: float(v["opt_z"]) for k, v in bb.load_stats(bb.DEFAULT_STATS_PATH).items()
                      if isinstance(v, dict) and "opt_z" in v})
    synth = fc.floor_fraction(synth_raw, optz)
    synth["sigma_multiple"] = synth["jitter_std"].round(2)
    synth_cells = synth.groupby(["sigma_multiple", "jitter_iteration", "dataset"])["frac"].mean()
    synth_ach = fc.share_fraction(synth_raw, optz)
    synth_ach["sigma_multiple"] = synth_ach["jitter_std"].round(2)
    synth_ach_cells = synth_ach.groupby(["sigma_multiple", "jitter_iteration", "dataset"])["frac"].mean()

    corrected = fc.relative_magnitude(ab.load_paired(REPO / "output-fitted"))
    noaug = fc.relative_magnitude(ab.load_paired(REPO / "output-fitted-noaug"))
    arms = {"corrected, augmentation on": corrected, "corrected, augmentation off": noaug}
    means = {"corrected, augmentation on": fc.box_means(REPO / "output-fitted", corrected),
             "corrected, augmentation off": fc.box_means(REPO / "output-fitted-noaug", noaug)}
    prefix = REPO / "output-fitted-prefix"
    if prefix.is_dir():
        as_logged = fc.relative_magnitude(ab.load_paired(prefix))
        as_logged = as_logged[as_logged.dataset != "ehmi"]
        ehmi = corrected[corrected.dataset == "ehmi"]
        arms["as-logged, augmentation on"] = pd.concat([ehmi, as_logged], ignore_index=True)
        means["as-logged, augmentation on"] = pd.concat(
            [fc.box_means(REPO / "output-fitted", ehmi), fc.box_means(prefix, as_logged)])

    # The spread the three datasets' ranks are read against, in both units.
    for unit, cells in (("floor gap", synth_cells), ("achievable improvement", synth_ach_cells)):
        s = cells.loc[(1.0, 0)]
        print(f"known-function per-landscape fraction at 1 sigma from the first rating, {unit}: "
              f"range [{s.min():.3f}, {s.max():.3f}] ({s.idxmin()} to {s.idxmax()}), median {s.median():.3f}")

    for label, paired in arms.items():
        print(f"\n== {label}: {len(paired):,} paired runs, seeds {sorted(paired.seed.unique())}")
        print(f"   runs beating the logged y_opt: {beaten(paired)}")
        print("   floor gap (tab:fitted's unit)")
        report(fractions(paired), synth_cells)
        print("   achievable improvement (the headline unit: known-function excess / opt_z)")
        report({how: fc.share_fraction(paired, fc.fitted_achievable(paired, means[label], how)) for how in fc.OPTIMA},
               synth_ach_cells)


if __name__ == "__main__":
    main()
