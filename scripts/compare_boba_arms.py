"""Paired contrast between two arms of the known-function study.

The follow-up arms each change one thing about the optimiser and re-run a subset
of the main sweep's design: the GP is told the true observation variance rather
than fitting it, or the incumbent is the observed maximum rather than the
posterior mean. The question in both cases is not "how does this arm do" but
"how much of the measured cost of feedback error was actually this".

So the comparison is paired at the finest grain the two arms share --
(benchmark, acquisition, error model, magnitude, onset, seed) -- and the
statistic is the change in excess regret within each of those cells. Anything
coarser would mix the contrast with the enormous between-landscape variance the
main sweep already documents.

  python scripts/compare_boba_arms.py \\
    --reference output-boba --treatment output-boba-knownnoise \\
    --label-reference "learned noise" --label-treatment "known noise"
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import boba_benchmarks as bb  # noqa: E402

METRIC = "auc_simple_regret_excess_true_postonset_per_iter"
PAIR_KEYS = ["dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed"]
MODEL_FREE = ("random", "sobol")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--label-reference", type=str, default="reference")
    parser.add_argument("--label-treatment", type=str, default="treatment")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    parser.add_argument(
        "--relative-magnitudes", action="store_true",
        help="Map each dataset's own magnitudes to the grid 0.05/0.25/1/5 by rank before "
             "grouping. The fitted-oracle arm expresses the grid in each dataset's sigma_f, "
             "so raw magnitudes never coincide across datasets and every condition would "
             "otherwise hold a single dataset.",
    )
    parser.add_argument(
        "--metric", type=str, default=METRIC,
        help="Column of paired_excess_metrics.csv to contrast. The default is the "
             "post-onset excess regret; final_inference_simple_regret_excess_true scores "
             "the design the experimenter would deploy, which is what the re-rating arms "
             "change. Use a separate --output-dir per metric.",
    )
    return parser.parse_args(argv)


def load(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/evaluation/paired_excess_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No per-benchmark evaluation outputs under {root}.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[~df["acquisition"].isin(MODEL_FREE)]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = bb.load_stats(args.stats_path)

    metric = args.metric
    ref = load(args.reference)[PAIR_KEYS + [metric]].rename(columns={metric: "ref"})
    trt = load(args.treatment)[PAIR_KEYS + [metric]].rename(columns={metric: "trt"})
    paired = ref.merge(trt, on=PAIR_KEYS, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError(
            "The two arms share no cells. They must overlap on benchmark, acquisition, "
            "error model, magnitude, onset and seed for the contrast to be paired."
        )
    dropped_ref = len(ref) - len(paired)
    print(f"Paired {len(paired):,} cells; {dropped_ref:,} reference rows had no partner "
          f"(the treatment arm runs a subset by design).")

    if args.relative_magnitudes:
        grid = (0.05, 0.25, 1.0, 5.0)
        for dataset, block in paired.groupby("dataset"):
            stds = sorted(block["jitter_std"].unique())
            if len(stds) != len(grid):
                raise ValueError(f"{dataset}: {len(stds)} magnitudes, expected {len(grid)}")
            paired.loc[block.index, "jitter_std"] = block["jitter_std"].map(dict(zip(stds, grid)))

    # Divide by opt_z BEFORE any mean is taken. The paper reports every aggregate
    # as a ratio of landscape means in units of the achievable improvement, so
    # that no landscape outweighs another; opt_z spans 74x across the suite, and
    # on raw values shekel outweighed powell about 74 to 1 in both the numerator
    # and the denominator. The fitted-oracle datasets have no opt_z and keep
    # their own scale, which is why the contrast there is within one dataset.
    paired["opt_z"] = paired["dataset"].map(lambda n: stats.get(n, {}).get("opt_z", np.nan))
    scale = paired["opt_z"].fillna(1.0)
    paired["ref_raw"], paired["trt_raw"] = paired["ref"], paired["trt"]
    paired["ref"] = paired["ref"] / scale
    paired["trt"] = paired["trt"] / scale
    paired["delta"] = paired["trt"] - paired["ref"]
    # Share of the reference arm's measured cost that the treatment removes.
    paired["share_removed"] = np.where(
        paired["ref"] > 0, 1.0 - paired["trt"] / paired["ref"], np.nan
    )
    paired.to_csv(args.output_dir / "arm_contrast_pairs.csv", index=False)

    rows: list[dict] = []
    for keys, block in paired.groupby(["error_model", "jitter_std", "jitter_iteration"]):
        # Benchmark means first: the seeds inside a benchmark are replicates, and
        # the test's unit of replication is the landscape.
        per_benchmark = block.groupby("dataset")[["ref", "trt", "delta"]].mean()
        if len(per_benchmark) < 3:
            continue
        differences = per_benchmark["delta"].to_numpy()
        if np.allclose(differences, 0.0):
            # Identical arms on this condition: scipy divides by a zero rank
            # spread and returns nan, which reads as "missing" rather than
            # "no difference at all".
            p = 1.0
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p = wilcoxon(per_benchmark["trt"], per_benchmark["ref"])
            except ValueError:
                p = 1.0
        rows.append({
            "error_model": keys[0],
            "jitter_std": float(keys[1]),
            "jitter_iteration": int(keys[2]),
            "n_benchmarks": int(len(per_benchmark)),
            "mean_reference": float(per_benchmark["ref"].mean()),
            "mean_treatment": float(per_benchmark["trt"].mean()),
            "mean_delta": float(differences.mean()),
            "median_delta": float(np.median(differences)),
            "share_removed": float(1.0 - per_benchmark["trt"].mean() / per_benchmark["ref"].mean())
            if per_benchmark["ref"].mean() > 0 else np.nan,
            "cohens_dz": float(differences.mean() / differences.std(ddof=1))
            if differences.std(ddof=1) > 0 else np.nan,
            "wilcoxon_p": float(p),
        })
    summary = pd.DataFrame(rows)
    if len(summary):
        summary["wilcoxon_p_fdr"] = multipletests(summary["wilcoxon_p"], method="fdr_bh")[1]
    summary.to_csv(args.output_dir / "arm_contrast_summary.csv", index=False)

    by_benchmark = (
        paired.groupby("dataset")
        .agg(reference=("ref", "mean"), treatment=("trt", "mean"),
             delta=("delta", "mean"), opt_z=("opt_z", "first"), cells=("delta", "size"))
        .reset_index()
        .sort_values("delta")
    )
    by_benchmark["share_removed"] = 1.0 - by_benchmark["treatment"] / by_benchmark["reference"]
    by_benchmark.to_csv(args.output_dir / "arm_contrast_by_benchmark.csv", index=False)

    # Figure: per-benchmark before/after, ordered by how much the arm changes.
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(by_benchmark) + 2))
    y = np.arange(len(by_benchmark))
    ax.hlines(y, by_benchmark["reference"], by_benchmark["treatment"], color="0.7", zorder=1)
    ax.scatter(by_benchmark["reference"], y, s=28, label=args.label_reference, zorder=2)
    ax.scatter(by_benchmark["treatment"], y, s=28, label=args.label_treatment, zorder=2)
    ax.set_yticks(y, by_benchmark["dataset"])
    ax.set_xlabel("post-onset per-iteration excess regret (landscape SDs)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend()
    ax.set_title(f"{args.label_treatment} vs {args.label_reference}")
    fig.savefig(args.output_dir / "arm_contrast.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [f"ARM CONTRAST: {args.label_treatment} vs {args.label_reference}", "=" * 72, "",
             f"Paired on {', '.join(PAIR_KEYS)}; {len(paired):,} cells, "
             f"{paired['dataset'].nunique()} benchmarks, "
             f"{paired['acquisition'].nunique()} acquisitions, "
             f"{paired['seed'].nunique()} seeds.",
             "Positive delta = the treatment arm suffers MORE from feedback error.", ""]
    if len(summary):
        overall = paired.groupby("dataset")[["ref", "trt"]].mean()
        share = 1.0 - overall["trt"].mean() / overall["ref"].mean()
        lines += [
            f"OVERALL: {args.label_reference} {overall['ref'].mean():.4f} -> "
            f"{args.label_treatment} {overall['trt'].mean():.4f}  "
            f"({share:+.1%} of the measured cost)",
            "",
            f"  {'error':>9s} {'SD':>7s} {'onset':>6s} {'ref':>9s} {'trt':>9s} "
            f"{'delta':>9s} {'share':>8s} {'dz':>7s} {'p(FDR)':>9s}",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"  {row['error_model']:>9s} {row['jitter_std']:>7.3g} "
                f"{int(row['jitter_iteration']):>6d} {row['mean_reference']:>9.4f} "
                f"{row['mean_treatment']:>9.4f} {row['mean_delta']:>+9.4f} "
                f"{row['share_removed']:>+8.1%} {row['cohens_dz']:>7.2f} "
                f"{row['wilcoxon_p_fdr']:>9.4g}"
            )
        lines += ["", "Per benchmark (mean over every shared cell):", "",
                  f"  {'benchmark':<20s} {'opt_z':>8s} {'ref':>9s} {'trt':>9s} "
                  f"{'delta':>9s} {'share':>8s}"]
        for _, row in by_benchmark.iterrows():
            lines.append(
                f"  {row['dataset']:<20s} {row['opt_z']:>8.2f} {row['reference']:>9.4f} "
                f"{row['treatment']:>9.4f} {row['delta']:>+9.4f} {row['share_removed']:>+8.1%}"
            )
    report = "\n".join(lines)
    (args.output_dir / "arm_contrast_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    (args.output_dir / "arm_contrast_summary.json").write_text(json.dumps({
        "reference": str(args.reference),
        "treatment": str(args.treatment),
        "paired_cells": int(len(paired)),
        "benchmarks": int(paired["dataset"].nunique()),
        "mean_reference": float(paired["ref"].mean()),
        "mean_treatment": float(paired["trt"].mean()),
        "conditions_tested": int(len(summary)),
        "conditions_significant_fdr": int((summary["wilcoxon_p_fdr"] < 0.05).sum())
        if len(summary) else 0,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
