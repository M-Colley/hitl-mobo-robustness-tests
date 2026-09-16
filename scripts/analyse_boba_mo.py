"""Analysis for the multi-objective arm.

The scalar analysis (analyse_boba_robustness.py) cannot be reused wholesale: it
is built around ``opt_z`` and the landscape descriptors, and a Pareto problem has
no scalar optimum to stand above a random design. What DOES transfer is the
paired design, the post-onset per-iteration excess metric, the model-free
control and the headroom screen, so this mirrors those and nothing else.

The response is the same column the scalar arm calls ``excess_sd``:
``auc_simple_regret_excess_true_postonset_per_iter``. Here its units are
standardised hypervolume rather than landscape standard deviations, because the
objectives are standardised per objective and hypervolume is their product --
which is also why an M-objective problem's numbers are not comparable with an
M'-objective problem's until they are divided by something in the same units.

That something is the ACHIEVABLE HYPERVOLUME GAIN: the published maximum
hypervolume minus what a model-free design of the same budget reaches. It is the
analogue of ``opt_z`` -- how much there is to win over not learning -- and the
quotient is the same "fraction of the achievable improvement destroyed" the
scalar arm reports.

Cross-arm comparison is the delicate part, and the obvious way to do it is
wrong. The scalar arm divides by ``opt_z``, a z-score against a random design's
MEAN; this arm divides by the gap to a same-budget model-free FLOOR. Those
denominators differ by a lot, so quoting one arm's published fraction against
the other's manufactures a difference. So it is done two ways instead: (1) both
arms recomputed against the floor denominator, identically; and (2) the onset
ratio, which is a ratio of two costs on the same landscape and therefore free of
the denominator altogether.

    python scripts/analyse_boba_mo.py
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
BASELINE_AUC = "auc_simple_regret_true_postonset_per_iter_baseline"
MODEL_FREE = ("random", "sobol")
# Matches the scalar arm: resample PROBLEMS, not runs. With six problems the
# asymptotic standard error is meaningless, and the cluster is the problem.
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260909
# Below this the clean run barely beats a design that reads nothing, so error has
# nothing to destroy and a null is guaranteed rather than earned. The scalar
# suite's weakest admissible landscape (Rosenbrock) sits at 0.176.
HEADROOM_MIN = 0.15


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dir", type=Path, default=Path("output-boba-mo"))
    p.add_argument("--scalar-dir", type=Path, default=Path("output-boba"))
    p.add_argument("--output-dir", type=Path, default=Path("output-boba-mo/analysis"))
    return p.parse_args(argv)


def achievable_gain(input_dir: Path, paired: pd.DataFrame) -> pd.DataFrame:
    """max_hv minus what a model-free design of the same budget reaches."""
    meta = json.loads((input_dir / "run_metadata.json").read_text(encoding="utf-8"))
    stats = pd.DataFrame(meta["landscape_stats"]).T.rename(columns={"name": "dataset"})
    for column in ("dim", "num_objectives", "max_hv", "sampled_hv"):
        stats[column] = pd.to_numeric(stats[column])

    floor = (
        paired[paired.acquisition.isin(MODEL_FREE)]
        .groupby("dataset")["final_best_true_baseline"]
        .mean()
        .rename("floor_hv")
    )
    out = stats[["dataset", "dim", "num_objectives", "max_hv", "sampled_hv"]].merge(
        floor, left_on="dataset", right_index=True, how="left"
    )
    out["gain"] = out["max_hv"] - out["floor_hv"]
    return out


def headroom(paired: pd.DataFrame) -> pd.DataFrame:
    """The scalar arm's screen, unchanged: 1 - best clean learner AUC / the floor's."""
    base = paired.groupby(["dataset", "acquisition"])[BASELINE_AUC].mean().reset_index()
    rows = []
    for name, block in base.groupby("dataset"):
        floors = block[block.acquisition.isin(MODEL_FREE)]
        learners = block[~block.acquisition.isin(MODEL_FREE)]
        if floors.empty or learners.empty:
            continue
        floor_auc = float(floors[BASELINE_AUC].mean())
        best = learners.loc[learners[BASELINE_AUC].idxmin()]
        rows.append(
            {
                "dataset": name,
                "floor_auc": floor_auc,
                "best_learner": best.acquisition,
                "best_auc": float(best[BASELINE_AUC]),
                "headroom": 1.0 - float(best[BASELINE_AUC]) / floor_auc
                if floor_auc
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("headroom")


def floor_fraction(
    paired: pd.DataFrame,
    gain: pd.DataFrame | None = None,
    y_opt: pd.Series | None = None,
) -> pd.DataFrame:
    """Excess as a fraction of the gap between the optimum and the model-free floor.

    Computed identically for both arms, which is the only basis on which their
    absolute levels may be quoted against each other.
    """
    learners = paired[~paired.acquisition.isin(MODEL_FREE)].copy()
    if gain is not None:
        learners = learners.merge(gain[["dataset", "gain"]], on="dataset", how="left")
    else:
        floor = (
            paired[paired.acquisition.isin(MODEL_FREE)]
            .groupby("dataset")["final_best_true_baseline"]
            .mean()
        )
        learners = learners.merge(
            (y_opt - floor).rename("gain"), left_on="dataset", right_index=True, how="left"
        )
    learners["frac"] = learners[RESPONSE] / learners["gain"]
    return learners


def bootstrap_cells(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Cluster-bootstrap each (magnitude, onset) cell over problems."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (std, onset), cell in df.groupby(["jitter_std", "jitter_iteration"]):
        per = cell.groupby("dataset")["frac"].mean().to_numpy()
        draws = np.array(
            [per[rng.integers(0, len(per), len(per))].mean() for _ in range(BOOTSTRAP_REPS)]
        )
        rows.append(
            {
                "arm": arm,
                "jitter_std": std,
                "jitter_iteration": onset,
                "n_problems": len(per),
                "mean": float(per.mean()),
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def acquisition_agreement(frac: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Do the problems agree on which hypervolume acquisition is most robust?"""
    from scipy.stats import friedmanchisquare

    cond = ["error_model", "jitter_std", "jitter_iteration"]
    cell = frac.groupby(cond + ["dataset", "acquisition"])["frac"].mean().reset_index()
    cell["rank"] = cell.groupby(cond + ["dataset"])["frac"].rank()

    significant = total = 0
    for _, block in cell.groupby(cond):
        wide = block.pivot(index="dataset", columns="acquisition", values="frac").dropna()
        if wide.shape[0] < 3 or wide.shape[1] < 3:
            continue
        total += 1
        stat = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
        significant += int(stat.pvalue < 0.05)

    wide = cell.pivot_table(
        index="dataset", columns="acquisition", values="frac", aggfunc="mean"
    )
    ranks = wide.rank(axis=1)
    n, k = ranks.shape
    spread = ((ranks.sum(axis=0) - ranks.sum(axis=0).mean()) ** 2).sum()
    kendall_w = 12 * spread / (n**2 * (k**3 - k))

    table = (
        cell.groupby("acquisition")
        .agg(mean_rank=("rank", "mean"), mean_frac=("frac", "mean"))
        .sort_values("mean_rank")
    )
    summary = {
        "n_problems": int(n),
        "n_acquisitions": int(k),
        "conditions_total": int(total),
        "conditions_significant": int(significant),
        "kendall_w": float(kendall_w),
        "winners": {str(d): str(a) for d, a in wide.idxmin(axis=1).items()},
    }
    return table, summary


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired = ab.load_paired(args.input_dir)
    gain = achievable_gain(args.input_dir, paired)

    print("\n=== CONTROL: model-free excess must be identically zero ===")
    mf = paired[paired.acquisition.isin(MODEL_FREE)]
    worst = float(mf[RESPONSE].abs().max())
    print(
        f"  n = {len(mf):,}   max |excess| = {worst:.3e}   "
        f"{'PASS' if worst == 0.0 else 'FAIL'}"
    )

    print("\n=== CONTROL: acquisition-optimisation fallbacks ===")
    print(f"  total = {int(paired.acq_opt_failures.sum())}")

    print("\n=== SUITE ===")
    print(gain.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    hr = headroom(paired)
    print("\n=== HEADROOM SCREEN ===")
    print(hr.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    admissible = sorted(hr.loc[hr.headroom >= HEADROOM_MIN, "dataset"])
    dropped = sorted(hr.loc[hr.headroom < HEADROOM_MIN, "dataset"])
    print(f"  admissible: {admissible}")
    print(f"  DROPPED   : {dropped}  (headroom < {HEADROOM_MIN})")
    hr.to_csv(args.output_dir / "mo_headroom.csv", index=False)

    frac = floor_fraction(paired, gain=gain)
    frac = frac[frac.dataset.isin(admissible)]

    print("\n=== DOSE-RESPONSE: fraction of achievable hypervolume gain destroyed ===")
    dose = frac.pivot_table(
        index="jitter_std", columns="jitter_iteration", values="frac", aggfunc="mean"
    )
    print(dose.to_string(float_format=lambda v: f"{v:+.4f}"))
    dose.to_csv(args.output_dir / "mo_dose_response.csv")

    print("\n=== per problem, at sigma = 1 ===")
    per = frac[frac.jitter_std == 1.0].pivot_table(
        index="dataset", columns="jitter_iteration", values="frac", aggfunc="mean"
    )
    print(per.to_string(float_format=lambda v: f"{v:+.4f}"))
    per.to_csv(args.output_dir / "mo_per_problem.csv")

    print("\n=== acquisition ranking (mean rank over problems within condition) ===")
    acq, agreement = acquisition_agreement(frac)
    print(acq.to_string(float_format=lambda v: f"{v:,.4f}"))
    print(
        f"  Friedman p < 0.05 in {agreement['conditions_significant']} of "
        f"{agreement['conditions_total']} conditions; "
        f"Kendall's W = {agreement['kendall_w']:.3f}"
    )
    print(f"  per-problem winner: {agreement['winners']}")
    acq.to_csv(args.output_dir / "mo_acquisitions.csv")
    (args.output_dir / "mo_acquisition_agreement.json").write_text(
        json.dumps(agreement, indent=2), encoding="utf-8"
    )

    # --- cross-arm: one common basis, plus one denominator-free ratio ---------
    print("\n=== CROSS-ARM ===")
    sc_paired = ab.load_paired(args.scalar_dir)
    # The multi-objective arm ran gaussian error only, so its scalar comparator
    # must too; pooling the scalar arm's four processes here made the "common
    # footing" column differ from the known-function row of the fitted-oracle
    # comparison for no reason a reader could see.
    sc_paired = sc_paired[sc_paired["error_model"] == "gaussian"]
    sc_meta = json.loads(
        (args.scalar_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    # opt_z, NOT y_opt. The runs report standardised objective values, and
    # y_opt in landscape_stats is the RAW optimum (0.0 for Ackley, whose runs
    # sit around +11.7 on the standardised scale). opt_z = (y_best - mean)/std
    # IS the optimum on the scale the runs are measured on; using y_opt here
    # silently produced negative "fractions destroyed".
    y_opt = pd.Series(
        {k: float(v["opt_z"]) for k, v in sc_meta["landscape_stats"].items()}
    )
    sc = floor_fraction(sc_paired, y_opt=y_opt)
    sc_dose = sc.pivot_table(
        index="jitter_std", columns="jitter_iteration", values="frac", aggfunc="mean"
    )
    print("\n  scalar arm, floor denominator:")
    print("  " + sc_dose.to_string(float_format=lambda v: f"{v:+.4f}").replace("\n", "\n  "))
    print("\n  multi-objective arm, floor denominator:")
    print("  " + dose.to_string(float_format=lambda v: f"{v:+.4f}").replace("\n", "\n  "))

    print("\n  onset ratio (early onset / late onset) -- denominator-free:")
    for label, table in (("scalar", sc_dose), ("multi-objective", dose)):
        early, late = table.columns[0], table.columns[-1]
        ratio = table[early] / table[late]
        print(
            f"    {label:<16s} "
            + "  ".join(f"sigma={i:g}: {v:,.1f}x" for i, v in ratio.items())
        )

    # Cell means with cluster-bootstrap intervals, both arms, one file. The
    # intervals are the point of it: with six problems against twenty, the two
    # arms can differ by a factor of two in the point estimate and still overlap.
    cross = pd.concat(
        [bootstrap_cells(sc, "scalar"), bootstrap_cells(frac, "multi-objective")],
        ignore_index=True,
    )
    print("\n  cell means with 95% cluster-bootstrap intervals over problems:")
    print("  " + cross.to_string(index=False,
                                 float_format=lambda v: f"{v:+.4f}").replace("\n", "\n  "))
    cross.to_csv(args.output_dir / "mo_vs_scalar.csv", index=False)
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
