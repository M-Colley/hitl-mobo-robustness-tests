"""The fitted-oracle companion arm, and what it does and does not settle.

The synthetic arm's premise is that a fitted human oracle confounds the
measurement. That is an argument until the two designs are run on a matched
grid, which is what this compares: the data-driven simulator on the three
archival datasets, at the same error magnitudes expressed in each dataset's own
sigma_f, against the known-function arm.

TWO THINGS MAKE THE COMPARISON AWKWARD, and pretending otherwise would be the
whole error the paper is about.

1. The fitted arm has no verified optimum. Its ``y_opt`` is a random-search
   estimate the optimizer can legitimately beat, so ``opt_z`` -- the synthetic
   arm's normaliser -- does not exist here. Both arms are therefore put on the
   gap between the best value reached and what a same-budget model-free design
   reaches, which is computable in both and means the same thing in both.

2. The fitted arm's excess mixes the injected error with the oracle's own
   mis-specification. Nothing in this script removes that; the comparison shows
   whether the two arms AGREE, and disagreement is exactly as consistent with
   "the oracle is wrong" as with "analytic landscapes are unrepresentative".
   The augmentation contrast is the closest thing to a handle on it: the
   deployed jitter augmentation costs up to 0.37 held-out R^2, so running the
   companion with it off changes the oracle's fidelity while holding the design
   fixed.

    python scripts/analyse_fitted_companion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyse_boba_robustness as ab  # noqa: E402

RESPONSE = "auc_simple_regret_excess_true_postonset_per_iter"
BASELINE_BEST = "final_best_true_baseline"
MODEL_FREE = ("random", "sobol")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260910


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fitted", type=Path, default=Path("output-fitted"))
    p.add_argument("--fitted-noaug", type=Path, default=Path("output-fitted-noaug"))
    p.add_argument("--synthetic", type=Path, default=Path("output-boba"))
    p.add_argument("--manifest", type=Path, default=Path("output/per_dataset/manifest.json"))
    p.add_argument("--output-dir", type=Path, default=Path("output-fitted/analysis"))
    return p.parse_args(argv)


def floor_fraction(paired: pd.DataFrame, optimum: pd.Series) -> pd.DataFrame:
    """Excess as a fraction of the gap from the model-free floor to the optimum."""
    floor = (
        paired[paired.acquisition.isin(MODEL_FREE)]
        .groupby("dataset")[BASELINE_BEST]
        .mean()
    )
    gain = (optimum - floor).rename("gain")
    learners = paired[~paired.acquisition.isin(MODEL_FREE)].copy()
    learners = learners.merge(gain, left_on="dataset", right_index=True, how="left")
    learners["frac"] = learners[RESPONSE] / learners["gain"]
    return learners


def fitted_optimum(paired: pd.DataFrame) -> pd.Series:
    """Best value any arm reached, which is all a fitted oracle offers.

    This is NOT a verified supremum and the paper says so: it is the same
    random-search-style estimate the applied literature uses, and the optimizer
    can reach it by construction, so the resulting fraction is optimistic in a
    way the synthetic arm's is not.
    """
    return paired.groupby("dataset")[BASELINE_BEST].max()


def relative_magnitude(paired: pd.DataFrame) -> pd.DataFrame:
    """Recover the sweep multiple from per-dataset sigma_f-scaled magnitudes.

    Each dataset was swept at 0.05, 0.25, 1 and 5 times ITS OWN sigma_f, so the
    raw jitter_std differs per dataset and pooling on it would compare 0.05
    sigma_f on one dataset with 0.25 sigma_f on another.
    """
    out = []
    for name, block in paired.groupby("dataset"):
        levels = sorted(block["jitter_std"].unique())
        nonzero = [v for v in levels if v > 0]
        if not nonzero:
            continue
        unit = min(nonzero) / 0.05
        block = block.copy()
        block["sigma_multiple"] = (block["jitter_std"] / unit).round(2)
        out.append(block)
    return pd.concat(out, ignore_index=True)


def dose(frame: pd.DataFrame, magnitude: str) -> pd.DataFrame:
    return frame.pivot_table(
        index=magnitude, columns="jitter_iteration", values="frac", aggfunc="mean"
    )


def bootstrap(frame: pd.DataFrame, magnitude: str, arm: str) -> pd.DataFrame:
    """Cluster bootstrap over datasets/landscapes -- the cluster is the design space."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (mag, onset), cell in frame.groupby([magnitude, "jitter_iteration"]):
        per = cell.groupby("dataset")["frac"].mean().to_numpy()
        if len(per) == 0:
            continue
        draws = np.array(
            [per[rng.integers(0, len(per), len(per))].mean() for _ in range(BOOTSTRAP_REPS)]
        )
        rows.append(
            {
                "arm": arm,
                "sigma_multiple": float(mag),
                "jitter_iteration": int(onset),
                "n_design_spaces": len(per),
                "mean": float(per.mean()),
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fitted = relative_magnitude(ab.load_paired(args.fitted))
    fitted = floor_fraction(fitted, fitted_optimum(fitted))
    print("\n=== FITTED companion: fraction of the achievable gain destroyed ===")
    print(dose(fitted, "sigma_multiple").to_string(float_format=lambda v: f"{v:+.4f}"))

    synth = ab.load_paired(args.synthetic)
    synth = synth[synth.error_model == "gaussian"]
    meta = json.loads((args.synthetic / "run_metadata.json").read_text(encoding="utf-8"))
    optz = pd.Series({k: float(v["opt_z"]) for k, v in meta["landscape_stats"].items()})
    synth = floor_fraction(synth, optz)
    synth["sigma_multiple"] = synth["jitter_std"].round(2)
    print("\n=== SYNTHETIC arm, same denominator, gaussian only ===")
    print(dose(synth, "sigma_multiple").to_string(float_format=lambda v: f"{v:+.4f}"))

    noaug = None
    if args.fitted_noaug.is_dir():
        noaug = relative_magnitude(ab.load_paired(args.fitted_noaug))
        noaug = floor_fraction(noaug, fitted_optimum(noaug))
        print("\n=== FITTED, augmentation OFF ===")
        print(dose(noaug, "sigma_multiple").to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\n=== onset ratio (early / late) -- free of the denominator ===")
    arms = [("fitted", fitted), ("synthetic", synth)]
    if noaug is not None:
        arms.append(("fitted no-aug", noaug))
    for label, frame in arms:
        grid = dose(frame, "sigma_multiple")
        early, late = grid.columns[0], grid.columns[-1]
        ratio = grid[early] / grid[late]
        print(f"  {label:<14s} " + "  ".join(f"{i:g}x: {v:,.1f}" for i, v in ratio.items()))

    parts = [bootstrap(fitted, "sigma_multiple", "fitted"),
             bootstrap(synth, "sigma_multiple", "synthetic")]
    if noaug is not None:
        parts.append(bootstrap(noaug, "sigma_multiple", "fitted no-aug"))
    cross = pd.concat(parts, ignore_index=True)
    cross.to_csv(args.output_dir / "fitted_vs_synthetic.csv", index=False)
    print("\n=== cell means with 95% cluster-bootstrap intervals ===")
    print(cross.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # The augmentation contrast, paired at the finest grain the two arms share.
    if noaug is not None:
        keys = ["dataset", "acquisition", "error_model", "jitter_std",
                "jitter_iteration", "seed"]
        merged = fitted.merge(noaug, on=keys, suffixes=("_aug", "_noaug"))
        if len(merged):
            delta = merged["frac_noaug"] - merged["frac_aug"]
            print(f"\n=== augmentation contrast, paired on {len(merged):,} cells ===")
            print(f"  mean change with augmentation OFF: {delta.mean():+.4f} "
                  f"(median {delta.median():+.4f})")
            print(f"  augmentation-off is worse in {100 * (delta > 0).mean():.1f}% of cells")
        else:
            print("\n  augmentation contrast: the two arms share no cells.")

    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
