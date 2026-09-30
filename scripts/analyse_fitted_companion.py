"""The fitted-oracle companion arm, and what it does and does not settle.

The synthetic arm's premise is that a fitted human oracle confounds the
measurement. That is an argument until the two designs are run on a matched
grid, which is what this compares: the data-driven simulator on the three
archival datasets, at the same error magnitudes expressed in each dataset's own
sigma_f, against the known-function arm.

TWO THINGS MAKE THE COMPARISON AWKWARD, and pretending otherwise would be the
whole error the paper is about.

1. The fitted arm has no verified optimum, and every run seed refits the
   oracle (bo_sensor_error_simulation.py builds it with seed=seed), so each
   dataset has ten oracles, one per seed 7-16. Both arms are put on the floor
   gap, the optimum less what a same-budget model-free design (random, Sobol)
   reaches. The synthetic arm has one function per landscape, so its gap is
   opt_z less the floor pooled over seeds. The fitted arm's gap is formed
   within each seed's oracle and averaged over seeds, and a dataset's fraction
   is the ratio of seed means, mean_s(excess_s) / mean_s(gap_s). Two optima:

     oracle      (primary, the analogue of opt_z) the logged y_opt of that
                 seed's oracle, the simulator's random-search estimate of its
                 maximum (200,000 points plus the training designs), fixed before
                 any run. A run can beat it: on the corrected data with
                 augmentation no clean run does (a noisy run does on 1 of the 10
                 provoice seeds); without augmentation clean runs do on 2 of the
                 10 ehmi seeds (scripts/review_checks/oracle_companion_estimators.py).
     best_clean  (sensitivity) the best value any clean run of that seed
                 reached. It is attainable by construction, so it shrinks the gap
                 and inflates the fitted fraction.

   Up to 2026-09-28 the gap was the maximum clean best over all ten oracles less
   the floor pooled over all ten, which mixes ten functions.

   The floor gap is not the only normaliser both arms have. The paper's
   headline unit, the achievable improvement (opt_z: the optimum less the
   landscape mean), has a fitted-arm analogue formed within each seed, A_s =
   optimum_s less the oracle's mean over the search box, estimated by the mean
   over the designs the seed's clean random and Sobol runs visited (100 per
   seed, read from the run logs). The floor gap can approach zero where random
   search nearly reaches the optimum (Branin, Powell and Rosenbrock in the
   synthetic arm); the achievable improvement cannot. Each dataset's share is
   mean_s(excess_s) / mean_s(A_s), and fitted_vs_synthetic_achievable{,_best_clean}.csv
   hold it beside the floor-gap files.

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
BASELINE_REGRET = "final_simple_regret_true_baseline"  # y_opt - BASELINE_BEST
MODEL_FREE = ("random", "sobol")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260910
# The fitted arm's optimum estimators, primary first, and the suffix of the
# output file each writes.
OPTIMA = {"oracle": "", "best_clean": "_best_clean"}


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


def floor_gap(paired: pd.DataFrame, optimum: pd.Series) -> pd.Series:
    """The floor gap per dataset.

    ``optimum`` indexed by dataset is one function per dataset (the synthetic
    arm): the gap is the optimum less the floor mean pooled over seeds. Indexed
    by (dataset, seed) it is one oracle per seed (the fitted arm): the gap is
    formed within each seed and averaged over the seeds, mean_s(opt_s - floor_s).
    """
    floors = paired[paired.acquisition.isin(MODEL_FREE)]
    if optimum.index.nlevels == 1:
        return (optimum - floors.groupby("dataset")[BASELINE_BEST].mean()).rename("gain")
    floor = floors.groupby(["dataset", "seed"])[BASELINE_BEST].mean()
    learners = paired[~paired.acquisition.isin(MODEL_FREE)]
    missing = set(map(tuple, learners[["dataset", "seed"]].drop_duplicates().to_numpy())) - set(floor.index)
    if missing:
        raise ValueError(f"(dataset, seed) cells with learners but no floor runs: {sorted(missing)[:5]}")
    opt = optimum.reindex(floor.index)
    if opt.isna().any():
        raise ValueError(f"no optimum for {list(opt.index[opt.isna()])[:5]}")
    return (opt - floor).groupby(level="dataset").mean().rename("gain")


def share_fraction(paired: pd.DataFrame, gain: pd.Series) -> pd.DataFrame:
    """Each learner's response over its dataset's normaliser ``gain`` (indexed by dataset)."""
    gain = gain.rename("gain")
    learners = paired[~paired.acquisition.isin(MODEL_FREE)].copy()
    learners = learners.merge(gain, left_on="dataset", right_index=True, how="left")
    if learners["gain"].isna().any():
        raise ValueError(f"no normaliser for {sorted(learners.loc[learners.gain.isna(), 'dataset'].unique())}")
    learners["frac"] = learners[RESPONSE] / learners["gain"]
    return learners


def floor_fraction(paired: pd.DataFrame, optimum: pd.Series) -> pd.DataFrame:
    """Excess as a fraction of the gap from the model-free floor to the optimum.

    With a per-seed optimum the gap is mean_s(gap_s), so a cell's mean ``frac``
    over a dataset balanced across seeds is mean_s(excess_s) / mean_s(gap_s).
    """
    return share_fraction(paired, floor_gap(paired, optimum))


def box_means(root: Path, paired: pd.DataFrame) -> pd.Series:
    """Each seed's oracle mean over its search box, indexed by (dataset, seed).

    Estimated by the mean true value over the designs the seed's clean random and
    Sobol runs visited (uniform and quasi-uniform in the box the runs search),
    read from the run logs, so no oracle is refitted.
    """
    out = {}
    for dataset, seed in paired[["dataset", "seed"]].drop_duplicates().itertuples(index=False):
        values = []
        for acq in MODEL_FREE:
            files = sorted((root / dataset).glob(f"bo_sensor_error_{dataset}_*_{acq}_seed{seed}_baseline_*.csv"))
            if len(files) != 1:
                raise ValueError(f"{root / dataset}: expected one clean {acq} run of seed {seed}, found {len(files)}")
            values.append(pd.read_csv(files[0], usecols=["objective_true"]).objective_true.to_numpy())
        out[(dataset, int(seed))] = float(np.mean(np.concatenate(values)))
    return pd.Series(out).rename_axis(["dataset", "seed"])


def fitted_achievable(paired: pd.DataFrame, means: pd.Series, how: str = "oracle") -> pd.Series:
    """mean_s(optimum_s - mean_s) per dataset: the fitted arm's achievable improvement."""
    optimum = fitted_optimum(paired, how)
    gap = optimum - means.reindex(optimum.index)
    if gap.isna().any() or (gap <= 0).any():
        raise ValueError(f"no positive achievable improvement for {list(gap.index[gap.isna() | (gap <= 0)])[:5]}")
    return gap.groupby(level="dataset").mean()


def fitted_optimum(paired: pd.DataFrame, how: str = "oracle") -> pd.Series:
    """The optimum of each seed's fitted oracle, indexed by (dataset, seed).

    ``oracle`` (primary) is the logged y_opt, BASELINE_BEST + BASELINE_REGRET of
    any run of the seed; it is constant within a seed, which is checked. It is a
    random-search estimate, not a verified supremum. ``best_clean`` is the best
    value any clean run of the seed reached, which the optimizer reaches by
    construction, so the fraction it gives is optimistic in a way the synthetic
    arm's is not.
    """
    if how == "oracle":
        y_opt = (paired[BASELINE_BEST] + paired[BASELINE_REGRET]).groupby([paired["dataset"], paired["seed"]])
        spread = float((y_opt.max() - y_opt.min()).max())
        if spread > 1e-9 * max(1.0, float(y_opt.max().abs().max())):
            raise ValueError(f"y_opt differs between runs of one seed by {spread:g}")
        return y_opt.mean()
    if how == "best_clean":
        return paired.groupby(["dataset", "seed"])[BASELINE_BEST].max()
    raise ValueError(f"unknown optimum estimator {how!r}; choose from {list(OPTIMA)}")


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

    synth_raw = ab.load_paired(args.synthetic)
    synth_raw = synth_raw[synth_raw.error_model == "gaussian"]
    meta = json.loads((args.synthetic / "run_metadata.json").read_text(encoding="utf-8"))
    optz = pd.Series({k: float(v["opt_z"]) for k, v in meta["landscape_stats"].items()})
    synth = floor_fraction(synth_raw, optz)
    synth["sigma_multiple"] = synth["jitter_std"].round(2)
    print("\n=== SYNTHETIC arm, same denominator, gaussian only ===")
    print(dose(synth, "sigma_multiple").to_string(float_format=lambda v: f"{v:+.4f}"))

    raw_fitted = relative_magnitude(ab.load_paired(args.fitted))
    raw_noaug = relative_magnitude(ab.load_paired(args.fitted_noaug)) if args.fitted_noaug.is_dir() else None
    for how, suffix in OPTIMA.items():
        print(f"\n##### fitted-arm optimum: {how} (per seed) #####")
        fitted = floor_fraction(raw_fitted, fitted_optimum(raw_fitted, how))
        print("\n=== FITTED companion: fraction of the achievable gain destroyed ===")
        print(dose(fitted, "sigma_multiple").to_string(float_format=lambda v: f"{v:+.4f}"))

        noaug = None
        if raw_noaug is not None:
            noaug = floor_fraction(raw_noaug, fitted_optimum(raw_noaug, how))
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
        cross.to_csv(args.output_dir / f"fitted_vs_synthetic{suffix}.csv", index=False)
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

    # The same comparison on the achievable improvement, the paper's headline unit:
    # excess / opt_z in the synthetic arm, mean_s(excess_s) / mean_s(optimum_s -
    # box mean_s) in the fitted arms.
    synth_ach = share_fraction(synth_raw, optz)
    synth_ach["sigma_multiple"] = synth_ach["jitter_std"].round(2)
    means = {"fitted": box_means(args.fitted, raw_fitted)}
    if raw_noaug is not None:
        means["fitted no-aug"] = box_means(args.fitted_noaug, raw_noaug)
    for how, suffix in OPTIMA.items():
        print(f"\n##### achievable improvement, fitted-arm optimum: {how} (per seed) #####")
        parts = []
        for label, raw in (("fitted", raw_fitted), ("fitted no-aug", raw_noaug)):
            if raw is None:
                continue
            frame = share_fraction(raw, fitted_achievable(raw, means[label], how))
            print(f"\n=== {label.upper()}: share of the achievable improvement destroyed ===")
            print(dose(frame, "sigma_multiple").to_string(float_format=lambda v: f"{v:+.4f}"))
            parts.append(bootstrap(frame, "sigma_multiple", label))
        print("\n=== SYNTHETIC arm, share of the achievable improvement (opt_z) ===")
        print(dose(synth_ach, "sigma_multiple").to_string(float_format=lambda v: f"{v:+.4f}"))
        parts.insert(1, bootstrap(synth_ach, "sigma_multiple", "synthetic"))
        cross = pd.concat(parts, ignore_index=True)
        cross.to_csv(args.output_dir / f"fitted_vs_synthetic_achievable{suffix}.csv", index=False)
        print("\n=== cell means with 95% cluster-bootstrap intervals ===")
        print(cross.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
