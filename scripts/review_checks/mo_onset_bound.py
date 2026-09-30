"""The multi-objective arm's onset ratio, against its running-maximum bound.

The multi-objective appendix (sec:multiobjective) compares the arm's raw onset ratio (search
loss with error from the first rating over search loss with error from trial
21) with the scalar arm's and reads the flatter ratio behaviourally. For the
scalar arm, scripts/onset_bound.py shows the raw ratio is almost entirely the
clean run's post-onset improvement B(t) = r_c(s) - r_c(t), the most a noisy run
that shares designs 1..s with its clean twin can lose. The same bound holds
here: ``simple_regret_true`` is the published maximum hypervolume minus the
true hypervolume of the designs evaluated so far, and the hypervolume of a set
never falls when a design is added, so the noisy run's regret after s is at most
r_c(s). The per-run window means come from onset_bound.summarise_pair, which
also checks the shared prefix and the bound itself in every pair.

Both arms are put on the denominator of the published cross-arm comparison
(analyse_boba_mo.py, tables/multiobjective.tex): the gap between the optimum and
a same-budget model-free floor, per problem, gaussian error only, learners only,
the multi-objective problems that pass analyse_boba_mo.HEADROOM_MIN. With E the
problem mean of the post-onset window excess and B that of the bound (both over
the floor gap), per magnitude

    raw ratio        E_early / E_late          (the published onset ratio)
    bound ratio      B_early / B_late          (mechanical: improvement left to lose)
    normalised ratio (E/B)_early / (E/B)_late  (behavioural: share of it lost)

with 95% percentile bootstraps resampling problems (landscapes), paired across
onsets (onset_bound._boot). The six problems and twenty landscapes share their
random numbers by design; the bootstrap treats them as independent clusters.
Since raw = bound x normalised in each arm, the log gap between the two arms'
raw ratios splits exactly into a bound part and a normalised part; the printout
gives that split and the bound's share of it (cross_arm_log_share, point
estimates only).

The scalar arm's per-run summaries are onset_bound.py's cache
(output-boba/analysis/review/onset_bound_runs.parquet); the multi-objective
arm's are computed here from the logs in output-boba-mo. Both are checked
against the pipeline's paired tables, and E reproduces mo_vs_scalar.csv.

    python scripts/review_checks/mo_onset_bound.py [--workers 6]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import onset_bound as ob  # noqa: E402

MO_BASE_RE = re.compile(
    r"^bo_sensor_error_(?P<ds>.+)_multi_objective_(?P<acq>[^_]+)_seed(?P<seed>\d+)_baseline_exact\.csv$")
MO_NOISY_RE = re.compile(
    r"^bo_sensor_error_(?P<ds>.+)_multi_objective_(?P<acq>[^_]+)_seed(?P<seed>\d+)_jittered_exact_"
    r"(?P<em>.+?)_jit(?P<onset>\d+)_std(?P<std>\d+(?:\.\d+)?)(?P<suffix>_.*)?\.csv$")
ERROR_MODEL = "gaussian"
QUANTITIES = ("raw_ratio", "bound_ratio", "normalised_ratio")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo-dir", type=Path, default=REPO / "output-boba-mo")
    p.add_argument("--scalar-dir", type=Path, default=REPO / "output-boba")
    p.add_argument("--scalar-runs", type=Path,
                   default=REPO / "output-boba" / "analysis" / "review" / "onset_bound_runs.parquet",
                   help="onset_bound.py's per-run cache for the scalar arm")
    p.add_argument("--published", type=Path, default=REPO / "output-boba-mo" / "analysis" / "mo_vs_scalar.csv",
                   help="analyse_boba_mo.py's cross-arm cell means, which E must reproduce")
    p.add_argument("--out", type=Path, default=REPO / "output-boba" / "analysis" / "review" / "mo_onset_bound.csv")
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args(argv)


def mo_runs(root: Path, workers: int) -> pd.DataFrame:
    """Per-pair window summaries of the multi-objective arm, by onset_bound's own code."""
    tasks, _ = ob.discover("multi-objective", root, None, None, base_re=MO_BASE_RE, noisy_re=MO_NOISY_RE)
    if not tasks:
        raise SystemExit(f"no multi-objective run logs found under {root}")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(ob.process_group, *task) for task in tasks]):
            rows += fut.result()
    df = pd.DataFrame(rows)
    bad = df[df["error"] != ""]
    if len(bad):
        raise SystemExit(f"{len(bad)} multi-objective pairs failed, e.g. {bad['error'].iloc[0]}")
    print(f"multi-objective arm: {len(df):,} pairs from {df['dataset'].nunique()} problems "
          f"({time.time() - t0:.0f}s)")
    # Workers finish in any order; a fixed row order keeps the float sums, and so the file, reproducible.
    order = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "onset", "variant"]
    return df.drop(columns="error").sort_values(order, kind="mergesort").reset_index(drop=True)


def floor_gaps(scalar_dir: Path, mo_dir: Path) -> tuple[dict, dict, list[str]]:
    """Each arm's floor-gap divisor, and the multi-objective problems that pass the headroom screen."""
    import analyse_boba_mo as mo
    import analyse_boba_robustness as ab

    mo_paired = ab.load_paired(mo_dir)
    gain = mo.achievable_gain(mo_dir, mo_paired).set_index("dataset")["gain"].to_dict()
    hr = mo.headroom(mo_paired)
    admissible = sorted(hr.loc[hr["headroom"] >= mo.HEADROOM_MIN, "dataset"])

    import boba_benchmarks as bb

    sc_paired = ab.load_paired(scalar_dir)
    # opt_z from the tracked statistics file, not the git-ignored run_metadata.json (which records
    # local paths); the two agree exactly on the main sweep's twenty landscapes.
    opt_z = pd.Series({k: float(v["opt_z"]) for k, v in bb.load_stats(bb.DEFAULT_STATS_PATH).items()
                       if isinstance(v, dict) and "opt_z" in v})
    floor = (sc_paired[sc_paired["acquisition"].isin(mo.MODEL_FREE)]
             .groupby("dataset")["final_best_true_baseline"].mean())
    sc_gain = (opt_z - floor).dropna().to_dict()
    return sc_gain, gain, admissible


def arm_rows(arm: str, runs: pd.DataFrame, divisor: dict) -> list[dict]:
    runs = ob.add_fractions(runs, divisor)
    onsets = sorted(runs["onset"].unique())
    early_onset, late_onset = min(onsets), max(onsets)
    out = []
    for std in sorted(runs["jitter_std"].unique()):
        early = runs[(runs["jitter_std"] == std) & (runs["onset"] == early_onset)]
        late = runs[(runs["jitter_std"] == std) & (runs["onset"] == late_onset)]
        for onset, cell in ((early_onset, early), (late_onset, late)):
            for q, est, lo, hi in ob.cell_quantities(cell):
                out.append({"arm": arm, "error_model": ERROR_MODEL, "jitter_std": std, "onset": str(onset),
                            "quantity": q, "estimate": est, "lo": lo, "hi": hi,
                            "n_landscapes": cell["dataset"].nunique(), "n_runs": len(cell)})
        for q, est, lo, hi in ob.ratio_quantities(early, late):
            out.append({"arm": arm, "error_model": ERROR_MODEL, "jitter_std": std, "onset": "early/late",
                        "quantity": q, "estimate": est, "lo": lo, "hi": hi,
                        "n_landscapes": early["dataset"].nunique(), "n_runs": len(early) + len(late)})
    return out


def cross_arm_log_share(table: pd.DataFrame, arm: str = "multi-objective", ref: str = "scalar") -> pd.DataFrame:
    """Per magnitude, how much of the gap between two arms' raw onset ratios the bound ratio accounts for.

    raw = bound x normalised in each arm, so on a log scale the arm gap in raw
    ratio splits exactly into a bound part and a normalised part:

        share = log(bound_arm / bound_ref) / log(raw_arm / raw_ref)

    1 means the bound accounts for all of the gap, above 1 that the normalised
    ratios pull the other way. Point estimates of the published ratios only: the
    share is unstable where the two raw ratios are close, and where the late
    onset's excess interval spans zero (the multi-objective arm below 1 sigma).
    """
    r = (table[(table["onset"] == "early/late") & table["arm"].isin([arm, ref])]
         .pivot_table(index="jitter_std", columns=["arm", "quantity"], values="estimate"))
    out = pd.DataFrame({
        "raw_arm": r[(arm, "raw_ratio")], "raw_ref": r[(ref, "raw_ratio")],
        "bound_arm": r[(arm, "bound_ratio")], "bound_ref": r[(ref, "bound_ratio")],
        "normalised_arm": r[(arm, "normalised_ratio")], "normalised_ref": r[(ref, "normalised_ratio")],
    })
    out["log_raw_gap"] = np.log(out["raw_arm"] / out["raw_ref"])
    out["log_bound_gap"] = np.log(out["bound_arm"] / out["bound_ref"])
    out["log_normalised_gap"] = np.log(out["normalised_arm"] / out["normalised_ref"])
    out["bound_share"] = out["log_bound_gap"] / out["log_raw_gap"]
    return out


def check_published(table: pd.DataFrame, published: Path) -> float:
    """E per cell must be analyse_boba_mo.py's cross-arm cell mean; returns the largest difference."""
    pub = pd.read_csv(published)
    pub = pub.assign(onset=pub["jitter_iteration"].astype(int).astype(str))
    mine = table[table["quantity"] == "excess"]
    m = mine.merge(pub, on=["arm", "jitter_std", "onset"], how="inner", validate="one_to_one")
    if len(m) != len(pub):
        raise SystemExit(f"matched {len(m)} of {len(pub)} published cells")
    return float((m["estimate"] - m["mean"]).abs().max())


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    sc_gain, mo_gain, admissible = floor_gaps(args.scalar_dir, args.mo_dir)

    mo_all = mo_runs(args.mo_dir, args.workers)
    mo_all = mo_all[mo_all["error_model"] == ERROR_MODEL]
    checks = ob.crosscheck(mo_all.assign(arm="multi-objective"), {"multi-objective": args.mo_dir})
    mo_df = mo_all[mo_all["dataset"].isin(admissible)]

    sc = pd.read_parquet(args.scalar_runs)
    sc = sc[(sc["arm"] == "main") & (sc["error_model"] == ERROR_MODEL)]
    checks.update(ob.crosscheck(sc, {"main": args.scalar_dir}))

    rows = arm_rows("scalar", sc, sc_gain) + arm_rows("multi-objective", mo_df, mo_gain)
    table = pd.DataFrame(rows)
    diff = check_published(table, args.published)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"multi-objective problems passing the headroom screen: {admissible}")
    for arm, c in checks.items():
        print(f"{arm}: {c['n_matched']:,} of {c['n_mine']:,} pairs matched to the paired tables, max |diff| "
              f"search {c['max_abs_diff_search']:.1e}, deployed {c['max_abs_diff_deployed']:.1e}")
    for arm, block in table.groupby("arm", sort=False):
        cells = block[block["onset"] != "early/late"]
        viol = cells[cells["quantity"] == "n_bound_violations"]["estimate"].sum()
        prefix = cells[cells["quantity"] == "share_prefix_identical"]["estimate"].min()
        print(f"{arm}: bound violations {viol:.0f}, smallest share of pairs with an identical prefix {prefix:.3f}")
    print(f"E reproduces mo_vs_scalar.csv to {diff:.1e}\n")

    def fmt(r):
        return f"{r['estimate']:.2f}x [{r['lo']:.2f}, {r['hi']:.2f}]"

    def pct(r):
        return f"{100 * r['estimate']:.1f} [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"

    print("floor-gap units, gaussian error, learners; E = excess, B = bound (% of the floor gap), "
          "E/B = share of the bound lost (%); from trial 1 / from trial 21\n")
    for arm, block in table.groupby("arm", sort=False):
        print(f"{arm} ({block['n_landscapes'].max()} {'problems' if arm != 'scalar' else 'landscapes'})")
        for std in sorted(block["jitter_std"].unique()):
            b = block[block["jitter_std"] == std].set_index(["onset", "quantity"])
            e0, e1 = b.loc[("0", "excess")], b.loc[("20", "excess")]
            b0, b1 = b.loc[("0", "bound")], b.loc[("20", "bound")]
            n0, n1 = b.loc[("0", "normalised")], b.loc[("20", "normalised")]
            r = {q: b.loc[("early/late", q)] for q in QUANTITIES}
            print(f"  {std:>4g} sigma  E {pct(e0)} / {pct(e1)}  B {pct(b0)} / {pct(b1)}  E/B {pct(n0)} / {pct(n1)}")
            print(f"              raw {fmt(r['raw_ratio'])}  bound {fmt(r['bound_ratio'])}  "
                  f"normalised {fmt(r['normalised_ratio'])}")
        print()
    gap = cross_arm_log_share(table)
    print("multi-objective against scalar, raw onset ratio on a log scale = bound part + normalised part "
          "(point estimates)")
    for std, g in gap.iterrows():
        print(f"  {std:>4g} sigma  raw {g['raw_arm']:.2f}x / {g['raw_ref']:.2f}x  log gap {g['log_raw_gap']:+.3f} = "
              f"bound {g['log_bound_gap']:+.3f} + normalised {g['log_normalised_gap']:+.3f}; "
              f"share of the gap in the bound {g['bound_share']:.2f}")
    print()
    shown = args.out.resolve()
    print(f"Wrote {shown.relative_to(REPO).as_posix() if shown.is_relative_to(REPO) else shown}")
    return table


if __name__ == "__main__":
    main()
