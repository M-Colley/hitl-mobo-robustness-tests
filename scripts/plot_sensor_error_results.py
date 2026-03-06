"""Plot simulation outputs and run paired baseline-vs-jitter analyses."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests


PAIRING_BASE_KEYS = ["objective", "acquisition", "seed", "oracle_model"]
CONDITION_KEYS = [
    "objective",
    "acquisition",
    "error_model",
    "jitter_std",
    "jitter_iteration",
    "oracle_model",
]
ANALYSIS_METRICS = [
    "objective_true",
    "objective_observed",
    "simple_regret_true",
    "regret_cum_true",
    "regret_avg_true",
]
OBJECTIVE_METRICS = ["objective_true", "objective_observed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/plots"))
    return parser.parse_args()


def load_iteration_logs(input_dir: Path) -> pd.DataFrame:
    files = list(input_dir.glob("bo_sensor_error_*_seed*_*.csv"))
    if not files:
        raise FileNotFoundError("No per-iteration logs found in input-dir.")
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def summarize_final_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = (
        df.sort_values("iteration")
        .groupby("run_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    final_rows["baseline"] = final_rows["error_model"] == "none"
    return final_rows


def build_paired_outcome_table(final_df: pd.DataFrame) -> pd.DataFrame:
    merge_keys = list(PAIRING_BASE_KEYS)
    if "dataset" in final_df.columns:
        merge_keys.insert(0, "dataset")

    baseline = final_df[final_df["baseline"]].copy()
    jittered = final_df[~final_df["baseline"]].copy()

    baseline_cols = merge_keys + ANALYSIS_METRICS
    paired = jittered.merge(
        baseline[baseline_cols],
        on=merge_keys,
        how="inner",
        suffixes=("_jitter", "_baseline"),
    )
    return paired


def _paired_test_from_diff(differences: np.ndarray) -> dict[str, float]:
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n_pairs = int(diffs.size)

    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "mean_abs_diff": np.nan,
            "std_diff": np.nan,
            "cohens_dz": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "p_value_t": np.nan,
            "wilcoxon_stat": np.nan,
            "p_value_wilcoxon": np.nan,
        }

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    mean_abs_diff = float(np.mean(np.abs(diffs)))

    if n_pairs > 1:
        std_diff = float(np.std(diffs, ddof=1))
    else:
        std_diff = 0.0

    cohens_dz = np.nan
    ci95_low = np.nan
    ci95_high = np.nan
    t_stat = np.nan
    p_value_t = np.nan

    if n_pairs >= 2 and not np.isclose(std_diff, 0.0):
        t_result = ttest_1samp(diffs, popmean=0.0)
        t_stat = float(t_result.statistic)
        p_value_t = float(t_result.pvalue)
        sem = std_diff / math.sqrt(n_pairs)
        critical = float(student_t.ppf(0.975, df=n_pairs - 1))
        margin = critical * sem
        ci95_low = mean_diff - margin
        ci95_high = mean_diff + margin
        cohens_dz = mean_diff / std_diff
    elif n_pairs >= 2 and np.allclose(diffs, 0.0):
        t_stat = 0.0
        p_value_t = 1.0
        ci95_low = 0.0
        ci95_high = 0.0

    wilcoxon_stat = np.nan
    p_value_wilcoxon = np.nan
    if n_pairs >= 2:
        if np.allclose(diffs, 0.0):
            wilcoxon_stat = 0.0
            p_value_wilcoxon = 1.0
        elif np.any(np.abs(diffs) > 0):
            try:
                wilcoxon_result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
                wilcoxon_stat = float(wilcoxon_result.statistic)
                p_value_wilcoxon = float(wilcoxon_result.pvalue)
            except ValueError:
                pass

    return {
        "n_pairs": n_pairs,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "mean_abs_diff": mean_abs_diff,
        "std_diff": std_diff,
        "cohens_dz": cohens_dz,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "t_stat": t_stat,
        "p_value_t": p_value_t,
        "wilcoxon_stat": wilcoxon_stat,
        "p_value_wilcoxon": p_value_wilcoxon,
    }


def _apply_fdr(df: pd.DataFrame, p_value_column: str) -> pd.DataFrame:
    adjusted = np.full(len(df), np.nan, dtype=float)
    rejected = np.zeros(len(df), dtype=bool)

    for metric in sorted(df["metric"].unique()):
        metric_index = df.index[df["metric"] == metric]
        valid_mask = df.loc[metric_index, p_value_column].notna()
        if not valid_mask.any():
            continue
        valid_index = metric_index[valid_mask]
        reject, adjusted_p, _, _ = multipletests(
            df.loc[valid_index, p_value_column].to_numpy(dtype=float),
            method="fdr_bh",
        )
        adjusted[valid_index.to_numpy(dtype=int)] = adjusted_p
        rejected[valid_index.to_numpy(dtype=int)] = reject

    df[f"{p_value_column}_fdr_bh"] = adjusted
    df[f"{p_value_column}_rejected"] = rejected
    return df


def evaluate_final_outcomes_improved(final_df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    paired = build_paired_outcome_table(final_df)
    if paired.empty:
        return {}

    condition_cols = list(CONDITION_KEYS)
    if "dataset" in paired.columns:
        condition_cols.insert(0, "dataset")

    rows: list[dict[str, object]] = []
    for condition_values, group in paired.groupby(condition_cols, dropna=False):
        if not isinstance(condition_values, tuple):
            condition_values = (condition_values,)
        condition_data = dict(zip(condition_cols, condition_values))
        for metric in ANALYSIS_METRICS:
            jitter_col = f"{metric}_jitter"
            baseline_col = f"{metric}_baseline"
            diffs = group[jitter_col].to_numpy(dtype=float) - group[baseline_col].to_numpy(dtype=float)
            stats = _paired_test_from_diff(diffs)
            rows.append(
                {
                    **condition_data,
                    "metric": metric,
                    "mean_jitter": float(group[jitter_col].mean()),
                    "mean_baseline": float(group[baseline_col].mean()),
                    **stats,
                }
            )

    paired_tests = pd.DataFrame(rows)
    paired_tests = _apply_fdr(paired_tests, "p_value_t")
    paired_tests = _apply_fdr(paired_tests, "p_value_wilcoxon")

    pair_differences = paired.copy()
    for metric in ANALYSIS_METRICS:
        pair_differences[f"{metric}_diff"] = (
            pair_differences[f"{metric}_jitter"] - pair_differences[f"{metric}_baseline"]
        )

    effect_sizes = paired_tests[paired_tests["metric"].isin(OBJECTIVE_METRICS)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_differences.to_csv(output_dir / "final_outcome_pair_differences.csv", index=False)
    paired_tests.to_csv(output_dir / "final_outcome_paired_tests.csv", index=False)
    effect_sizes.to_csv(output_dir / "effect_sizes_cohens_dz.csv", index=False)
    paired_tests[paired_tests["metric"].str.contains("regret")].to_csv(
        output_dir / "regret_paired_tests.csv",
        index=False,
    )

    return {
        "paired_outcomes": pair_differences,
        "paired_tests": paired_tests,
        "effect_sizes": effect_sizes,
    }


def generate_statistical_report(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    report_path = output_dir / "statistical_report.txt"
    paired_tests = results.get("paired_tests")

    def _mean_abs_effect(values: pd.Series) -> float:
        numeric = values.to_numpy(dtype=float)
        numeric = numeric[np.isfinite(numeric)]
        if numeric.size == 0:
            return float("nan")
        return float(np.mean(np.abs(numeric)))

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("PAIRED BASELINE VS JITTER ANALYSIS REPORT\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Method summary:\n")
        handle.write("- Tests are run within matched baseline/jitter condition pairs.\n")
        handle.write("- Each condition reports one-sample paired t-tests and Wilcoxon signed-rank tests.\n")
        handle.write("- Benjamini-Hochberg FDR correction is applied separately per metric.\n")
        handle.write("- Effect sizes use Cohen's dz on paired differences when variance is non-zero.\n\n")

        if paired_tests is None or paired_tests.empty:
            handle.write("No paired results were available.\n")
            return

        handle.write("Metric summary:\n")
        summary = (
            paired_tests.groupby("metric")
            .agg(
                conditions=("metric", "size"),
                significant_t=("p_value_t_rejected", "sum"),
                significant_wilcoxon=("p_value_wilcoxon_rejected", "sum"),
                mean_abs_effect=("cohens_dz", _mean_abs_effect),
            )
            .reset_index()
        )
        handle.write(summary.to_string(index=False))
        handle.write("\n\n")

        strongest = paired_tests.assign(abs_effect=np.abs(paired_tests["cohens_dz"])).sort_values(
            "abs_effect",
            ascending=False,
        )
        strongest = strongest[strongest["abs_effect"].notna()].head(10)
        handle.write("Largest paired effects:\n")
        if strongest.empty:
            handle.write("No non-zero variance effects were available.\n")
        else:
            columns = [
                "metric",
                "objective",
                "acquisition",
                "error_model",
                "jitter_std",
                "jitter_iteration",
                "oracle_model",
                "n_pairs",
                "mean_diff",
                "cohens_dz",
                "p_value_t_fdr_bh",
            ]
            if "dataset" in strongest.columns:
                columns.insert(0, "dataset")
            handle.write(strongest[columns].to_string(index=False))
        handle.write("\n\n")

        handle.write("Output files:\n")
        handle.write("- final_outcome_pair_differences.csv\n")
        handle.write("- final_outcome_paired_tests.csv\n")
        handle.write("- regret_paired_tests.csv\n")
        handle.write("- effect_sizes_cohens_dz.csv\n")

    print(f"Statistical report saved to: {report_path}")


def plot_final_outcome_significance(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    effect_sizes = results.get("effect_sizes")
    if effect_sizes is None or effect_sizes.empty:
        print("No paired effect-size results to plot")
        return

    for metric, metric_df in effect_sizes.groupby("metric"):
        metric_label = "true" if metric == "objective_true" else "observed"
        group_cols = ["objective", "error_model", "oracle_model", "jitter_iteration"]
        if "dataset" in metric_df.columns:
            group_cols.insert(0, "dataset")

        for group_values, data in metric_df.groupby(group_cols, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group_dict = dict(zip(group_cols, group_values))

            pivot = data.pivot_table(
                values="cohens_dz",
                index="acquisition",
                columns="jitter_std",
                aggfunc="mean",
            )
            if pivot.empty or pivot.isna().all().all():
                continue

            significance = data.pivot_table(
                values="p_value_t_fdr_bh",
                index="acquisition",
                columns="jitter_std",
                aggfunc="min",
            )

            plt.figure(figsize=(10, 5))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                center=0,
                cbar_kws={"label": "Cohen's dz"},
                mask=pivot.isna(),
            )

            for row_idx, acquisition in enumerate(pivot.index):
                for col_idx, jitter_std in enumerate(pivot.columns):
                    p_value = significance.loc[acquisition, jitter_std]
                    if pd.notna(p_value) and p_value < 0.05:
                        plt.text(col_idx + 0.5, row_idx + 0.2, "*", ha="center", va="center", color="black")

            title_parts = [
                f"metric={metric_label}",
                f"objective={group_dict['objective']}",
                f"error={group_dict['error_model']}",
                f"oracle={group_dict['oracle_model']}",
                f"jitter_iter={group_dict['jitter_iteration']}",
            ]
            if "dataset" in group_dict:
                title_parts.insert(0, f"dataset={group_dict['dataset']}")
            plt.title("Paired effect sizes (" + ", ".join(title_parts) + ")")
            plt.ylabel("Acquisition")
            plt.xlabel("Jitter std")
            plt.tight_layout()

            filename_parts = [
                "paired_effect_sizes",
                metric_label,
                str(group_dict["objective"]),
                str(group_dict["error_model"]),
                str(group_dict["oracle_model"]),
                f"jit{group_dict['jitter_iteration']}",
            ]
            if "dataset" in group_dict:
                filename_parts.insert(1, str(group_dict["dataset"]))
            filename = "_".join(filename_parts) + ".png"
            plt.savefig(output_dir / filename, dpi=200)
            plt.close()


def plot_objectives(df: pd.DataFrame, output_dir: Path) -> None:
    group_cols = [
        "objective",
        "acquisition",
        "error_model",
        "jitter_std",
        "jitter_iteration",
        "oracle_model",
        "iteration",
    ]
    grouped = (
        df.groupby(group_cols)[["objective_true", "objective_observed"]].mean().reset_index()
    )
    for (objective, acq, error_model, jitter_std, jitter_iteration, oracle_model), data in grouped.groupby(
        ["objective", "acquisition", "error_model", "jitter_std", "jitter_iteration", "oracle_model"]
    ):
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=data, x="iteration", y="objective_true", label="Objective (true)")
        sns.lineplot(data=data, x="iteration", y="objective_observed", label="Objective (observed)")
        plt.title(
            "Objective trajectory "
            f"({objective}, {acq}, {error_model}, {oracle_model}, jitter={jitter_iteration}, std={jitter_std})"
        )
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.tight_layout()
        filename = (
            "objective_trajectory_"
            f"{objective}_{acq}_{error_model}_{oracle_model}_jit{jitter_iteration}_std{jitter_std}.png"
        )
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def plot_adjustments(input_dir: Path, output_dir: Path) -> None:
    summary_stats = input_dir / "bo_sensor_error_summary_stats.csv"
    if not summary_stats.exists():
        return
    stats = pd.read_csv(summary_stats)
    for (objective, error_model, jitter_iteration), data in stats.groupby(
        ["objective", "error_model", "jitter_iteration"]
    ):
        plot = sns.catplot(
            data=data,
            x="acquisition",
            y="delta_l2_mean",
            hue="baseline",
            col="jitter_std",
            kind="bar",
            height=4,
            aspect=1.1,
        )
        plot.fig.suptitle(
            f"Mean parameter adjustment (L2 norm) - {objective} / {error_model} (jitter={jitter_iteration})"
        )
        plot.set_axis_labels("acquisition", "Mean delta L2")
        plot.tight_layout()
        filename = f"delta_l2_mean_{objective}_{error_model}_jit{jitter_iteration}.png"
        plot.savefig(output_dir / filename, dpi=200)
        plt.close(plot.fig)


def plot_excess_adjustments(input_dir: Path, output_dir: Path) -> None:
    excess_path = input_dir / "bo_sensor_error_excess_summary.csv"
    if not excess_path.exists():
        return
    excess = pd.read_csv(excess_path)
    summary = (
        excess.groupby(["objective", "acquisition", "error_model", "jitter_iteration", "jitter_std"])
        .agg(
            delta_excess_l2_mean=("delta_excess_l2_norm", "mean"),
            delta_excess_l2_std=("delta_excess_l2_norm", "std"),
            runs=("delta_excess_l2_norm", "count"),
        )
        .reset_index()
    )
    for (objective, jitter_iteration, jitter_std), data in summary.groupby(
        ["objective", "jitter_iteration", "jitter_std"]
    ):
        plt.figure(figsize=(8, 4))
        sns.barplot(
            data=data,
            x="acquisition",
            y="delta_excess_l2_mean",
            hue="error_model",
        )
        plt.title(
            "Mean excess adjustment (L2 norm) "
            f"- {objective} jitter={jitter_iteration}, std={jitter_std:.2f}"
        )
        plt.ylabel("Mean delta excess L2")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"delta_excess_l2_mean_{objective}_jit{jitter_iteration}_std{jitter_std}.png",
            dpi=200,
        )
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = load_iteration_logs(args.input_dir)
    plot_objectives(logs, args.output_dir)
    plot_adjustments(args.input_dir, args.output_dir)
    plot_excess_adjustments(args.input_dir, args.output_dir)
    final_outcomes = summarize_final_outcomes(logs)
    results_dict = evaluate_final_outcomes_improved(final_outcomes, args.output_dir)
    plot_final_outcome_significance(results_dict, args.output_dir)
    generate_statistical_report(results_dict, args.output_dir)
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
