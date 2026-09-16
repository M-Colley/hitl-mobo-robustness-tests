"""Analysis for the input-error arm: the person acts on the wrong design.

Every number is in the paper's own currency -- excess regret divided by opt_z,
the fraction of the achievable improvement destroyed -- so a slip and a response
error can be set side by side on the DAMAGE axis, even though their MAGNITUDE
axes are not comparable (a fraction of the box against landscape standard
deviations).

Three questions, one per contrast:

1. HOW MUCH. Dose-response by magnitude and onset for slip and misclick, over
   the model-based acquisitions, cluster-bootstrapped over landscapes.

2. WHERE FROM. Three quantities separate the mechanisms:
     floor     model-free acquisitions under slip. They never read an
               observation, so their excess is the pure geometric cost of
               evaluating the wrong points -- nothing learned, nothing corrupted.
     actual    model-based, slip detected and logged as (x', f(x')). The
               learner's data are true; it just cannot place its evaluations.
     proposed  model-based, slip unnoticed, (x, f(x')). The same, plus a
               surrogate trained on mislabelled points.
   proposed - actual is the cost of mislabelling, paired on shared cells: the
   two arms share seeds and therefore slip draws. actual - floor is what a
   learner loses beyond what displacement costs a non-learner. All of these are
   computed on ONE common basis -- see mechanism_tables().

3. WHAT SHIPS. Under slip/proposed the design a practitioner deploys is the
   logged x, never the x' that earned its rating. The simulator scores the
   inference metric at the recorded point, so deployed-design regret against
   evaluated regret is the price of shipping what the log says. The evaluated
   metric can even improve under a misclick -- a random jump can land somewhere
   good -- which is exactly why the deployed metric is the one that counts.

    python scripts/analyse_boba_inputerror.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyse_boba_robustness as ab  # noqa: E402
import boba_benchmarks as bb  # noqa: E402

MODEL_FREE = ("random", "sobol")
AUC = "auc_simple_regret_excess_true_postonset_per_iter"
EVALUATED = "final_simple_regret_excess_true"
DEPLOYED = "final_inference_simple_regret_excess_true"
PAIR_KEYS = ["dataset", "acquisition", "jitter_std", "jitter_iteration", "seed"]
# One task of the driver: every acquisition and condition for one landscape at
# one seed. The mechanism table is only ever computed on whole tasks.
BASIS = ["dataset", "seed"]
TASK_CELLS = ["acquisition", "jitter_std", "jitter_iteration"]
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260910


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--slip", type=Path, default=Path("output-boba-slip"))
    p.add_argument("--misclick", type=Path, default=Path("output-boba-misclick"))
    p.add_argument("--slip-actual", type=Path, default=Path("output-boba-slip-actual"))
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    p.add_argument("--output-dir", type=Path, default=Path("output-boba-slip/analysis"))
    return p.parse_args(argv)


def load_arm(path: Path, opt_z: pd.Series, label: str) -> pd.DataFrame | None:
    """Paired metrics for one arm, normalised by opt_z, or None if not evaluated."""
    if not any(path.glob("*/evaluation/paired_excess_metrics.csv")):
        print(f"{label}: no evaluation outputs under {path} -- skipped.")
        return None
    df = ab.load_paired(path)
    missing = sorted(set(df["dataset"]) - set(opt_z.index))
    if missing:
        raise ValueError(f"{label}: no opt_z for {missing}")
    z = df["dataset"].map(opt_z)
    return df.assign(
        frac=df[AUC] / z,
        frac_evaluated=df[EVALUATED] / z,
        frac_deployed=df[DEPLOYED] / z,
        arm=label,
    )


def bootstrap(frame: pd.DataFrame, value: str, by: list[str], arm: str) -> pd.DataFrame:
    """Per-landscape means, resampled over landscapes -- the landscape is the cluster."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for key, cell in frame.groupby(by):
        per = cell.groupby("dataset")[value].mean().to_numpy()
        per = per[np.isfinite(per)]
        if len(per) == 0:
            continue
        draws = np.array(
            [per[rng.integers(0, len(per), len(per))].mean() for _ in range(BOOTSTRAP_REPS)]
        )
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            "arm": arm,
            **dict(zip(by, key)),
            "n_landscapes": len(per),
            "mean": float(per.mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
        })
    return pd.DataFrame(rows)


def _complete_tasks(frame: pd.DataFrame) -> pd.DataFrame:
    """(landscape, seed) tasks carrying every (acquisition, magnitude, onset) cell the arm has.

    A task is the unit the driver runs, so a partly-finished arm has tasks with
    only some of their cells. Averaging one of those against a complete task in
    another arm changes the COMPOSITION of a column -- which acquisitions it
    averages over -- not just its landscapes.
    """
    expected = len(frame[TASK_CELLS].drop_duplicates())
    per_task = frame.drop_duplicates(BASIS + TASK_CELLS).groupby(BASIS).size()
    return per_task[per_task == expected].reset_index()[BASIS]


def _on_basis(frame: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(common, on=BASIS, how="inner")


def mechanism_tables(
    slip: pd.DataFrame | None,
    actual: pd.DataFrame | None,
    cond: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Floor, logged and unnoticed costs on ONE basis, plus the paired mislabelling cost.

    The rows are read side by side, so they must describe the same work. Two
    earlier versions did not, silently, on partial data. The first averaged
    each column over whatever its arm had finished, so at a 40% slip
    "unnoticed" sat below "logged" while the paired mislabelling cost came out
    positive. The second matched on (landscape, seed), but a half-finished task
    still changed which acquisitions a column averaged over, leaving the
    columns and the paired cost up to 0.06 apart. The basis is now the whole
    task, counted only once it is complete in every contributing arm -- with
    complete data, that is everything -- and the invariant the table is read
    by is checked at the end rather than assumed.
    """
    if slip is None:
        return pd.DataFrame(), pd.DataFrame()
    sources = {
        "floor": slip[slip.acquisition.isin(MODEL_FREE)],
        "proposed": slip[~slip.acquisition.isin(MODEL_FREE)],
    }
    if actual is not None:
        sources["actual"] = actual[~actual.acquisition.isin(MODEL_FREE)]
    sources = {name: frame for name, frame in sources.items() if not frame.empty}
    if not sources:
        return pd.DataFrame(), pd.DataFrame()

    common = None
    for frame in sources.values():
        tasks = _complete_tasks(frame)
        common = tasks if common is None else common.merge(tasks, on=BASIS, how="inner")
    if common.empty:
        print("\n  no (landscape, seed) task is complete in every arm yet -- "
              "mechanism table skipped.")
        return pd.DataFrame(), pd.DataFrame()
    print(f"\n  common basis for the mechanism rows: {len(common):,} complete tasks, "
          f"{common['dataset'].nunique()} landscapes")

    mechanism = pd.concat(
        [bootstrap(_on_basis(frame, common), "frac", cond, name)
         for name, frame in sources.items()],
        ignore_index=True,
    )

    mislabel = pd.DataFrame()
    if "proposed" in sources and "actual" in sources:
        paired = _on_basis(sources["proposed"], common)[PAIR_KEYS + ["frac"]].merge(
            _on_basis(sources["actual"], common)[PAIR_KEYS + ["frac"]],
            on=PAIR_KEYS, suffixes=("_proposed", "_actual"),
        )
        paired["mislabel_cost"] = paired["frac_proposed"] - paired["frac_actual"]
        print(f"  proposed/actual pairs on the common basis: {len(paired):,}")
        mislabel = bootstrap(paired, "mislabel_cost", cond, "proposed - actual")
        # The SHARE of the unnoticed slip's cost that is mislabelling, with its
        # own interval: a ratio of two landscape means, so it is bootstrapped as
        # one quantity per resample rather than assembled from two intervals.
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        shares = []
        for key, cell in paired.groupby(cond):
            per = cell.groupby("dataset")[["mislabel_cost", "frac_proposed"]].mean().to_numpy()
            draws = []
            for _ in range(BOOTSTRAP_REPS):
                pick = per[rng.integers(0, len(per), len(per))]
                total = pick[:, 1].mean()
                draws.append(pick[:, 0].mean() / total if total > 0 else np.nan)
            draws = np.array(draws)
            key = key if isinstance(key, tuple) else (key,)
            shares.append({**dict(zip(cond, key)),
                           "share": float(per[:, 0].mean() / per[:, 1].mean()),
                           "share_lo": float(np.nanpercentile(draws, 2.5)),
                           "share_hi": float(np.nanpercentile(draws, 97.5))})
        mislabel = mislabel.merge(pd.DataFrame(shares), on=cond, how="left")

        # On one basis the paired cost IS the gap between the two columns. Check
        # it: this table has had two composition bugs, and both would have
        # produced a plausible-looking table with no error at all.
        columns = mechanism.pivot_table(index=cond, columns="arm", values="mean")
        gap = columns["proposed"] - columns["actual"]
        worst = float((gap - mislabel.set_index(cond)["mean"]).abs().max())
        if not np.isfinite(worst) or worst > 1e-9:
            raise ValueError(
                "mechanism table is internally inconsistent: the gap between the "
                "'proposed' and 'actual' columns differs from the paired mislabelling "
                f"cost by up to {worst:.3g}, so the columns are not on one basis."
            )
    return mechanism, mislabel


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if table.empty:
        print("  (nothing to show)")
    else:
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = bb.load_stats(args.stats_path)
    opt_z = pd.Series({name: float(entry["opt_z"]) for name, entry in stats.items()})

    arms = {
        "slip": load_arm(args.slip, opt_z, "slip"),
        "misclick": load_arm(args.misclick, opt_z, "misclick"),
        "slip-actual": load_arm(args.slip_actual, opt_z, "slip-actual"),
    }
    cond = ["jitter_std", "jitter_iteration"]

    # 1. How much -------------------------------------------------------------
    dose = pd.concat(
        [bootstrap(df[~df.acquisition.isin(MODEL_FREE)], "frac", cond, name)
         for name, df in arms.items() if df is not None],
        ignore_index=True,
    )
    show("1. fraction of the achievable improvement destroyed (model-based)", dose)
    dose.to_csv(args.output_dir / "inputerror_dose.csv", index=False)

    # 2. Where from -----------------------------------------------------------
    mechanism, mislabel = mechanism_tables(arms["slip"], arms["slip-actual"], cond)
    show("2. geometric cost (floor) against learning under slip (actual, proposed)", mechanism)
    mechanism.to_csv(args.output_dir / "inputerror_mechanism.csv", index=False)
    show("2b. the cost of mislabelling: proposed - actual, paired", mislabel)
    mislabel.to_csv(args.output_dir / "inputerror_mislabel.csv", index=False)

    # 3. What ships -----------------------------------------------------------
    ships = pd.DataFrame()
    if arms["slip"] is not None:
        s = arms["slip"][~arms["slip"].acquisition.isin(MODEL_FREE)]
        ships = pd.concat([
            bootstrap(s, "frac_evaluated", cond, "evaluated"),
            bootstrap(s, "frac_deployed", cond, "deployed"),
        ], ignore_index=True)
    show("3. slip/proposed: best point reached against the design deployed (final)", ships)
    ships.to_csv(args.output_dir / "inputerror_deployed.csv", index=False)

    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
