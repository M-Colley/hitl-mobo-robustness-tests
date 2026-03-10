"""Evaluate robustness results against the research question.

This script focuses on the question:

    Which parts work how well for which dataset and which error level?

It uses the existing per-iteration logs to:

1. Recompute the first actually error-affected response step.
2. Pair each jittered run with its baseline counterpart.
3. Rank acquisition functions within each dataset / objective / error condition.
4. Run simple rank-based plausibility checks.
5. Write compact heatmaps for mean rank and excess regret.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


PRIMARY_METRIC = "auc_simple_regret_excess_true"
SECONDARY_METRIC = "final_simple_regret_excess_true"
REACTION_METRIC = "response_l2_excess"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation"))
    return parser.parse_args()


def load_iteration_logs(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("bo_sensor_error_*_seed*_*.csv"))
    if not files:
        raise FileNotFoundError("No per-iteration logs found in input-dir.")
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "dataset" not in df.columns:
        raise ValueError("Per-iteration logs must include a dataset column.")
    return df


def infer_param_columns(run_df: pd.DataFrame) -> list[str]:
    reserved = {
        "iteration",
        "objective_true",
        "objective_observed",
        "error_applied",
        "error_magnitude",
        "error_magnitude_l2",
        "acquisition",
        "fit_time_sec",
        "seed",
        "run_id",
        "error_model",
        "jitter_std",
        "jitter_iteration",
        "oracle_model",
        "objective",
        "y_opt",
        "best_true_so_far",
        "regret_inst_true",
        "regret_cum_true",
        "simple_regret_true",
        "regret_avg_true",
        "dataset",
    }
    param_columns: list[str] = []
    for column in run_df.columns:
        if column in reserved:
            continue
        if column.startswith("objective_true_") or column.startswith("objective_observed_"):
            continue
        if column.startswith("error_magnitude_"):
            continue
        if run_df[column].notna().any():
            param_columns.append(column)
    return param_columns


def build_response_table(logs: pd.DataFrame) -> pd.DataFrame:
    jitter_iterations = sorted(
        logs.loc[logs["error_model"] != "none", "jitter_iteration"].dropna().astype(int).unique().tolist()
    )
    rows: list[dict[str, object]] = []

    for _, run_df in logs.groupby("run_id", sort=False):
        run_df = run_df.sort_values("iteration").reset_index(drop=True)
        max_iter = int(run_df["iteration"].max())
        param_columns = infer_param_columns(run_df)
        base_row = run_df.iloc[-1]
        baseline = str(base_row["error_model"]) == "none"
        run_jitter_iteration = int(base_row["jitter_iteration"])
        target_iterations = jitter_iterations if baseline else [run_jitter_iteration]

        for jitter_iteration in target_iterations:
            # Noise starts affecting observations at iteration jitter_iteration + 1.
            # The first candidate that can react to that observation is iteration jitter_iteration + 2.
            start_iter = jitter_iteration + 1
            end_iter = jitter_iteration + 2
            if end_iter > max_iter:
                continue

            start_params = run_df.loc[run_df["iteration"] == start_iter, param_columns].iloc[0]
            end_params = run_df.loc[run_df["iteration"] == end_iter, param_columns].iloc[0]
            response = end_params - start_params
            response_l2 = float(np.linalg.norm(response.to_numpy(dtype=float)))

            row: dict[str, object] = {
                "run_id": str(base_row["run_id"]),
                "dataset": str(base_row["dataset"]),
                "objective": str(base_row["objective"]),
                "acquisition": str(base_row["acquisition"]),
                "seed": int(base_row["seed"]),
                "oracle_model": str(base_row["oracle_model"]),
                "baseline": baseline,
                "error_model": str(base_row["error_model"]),
                "jitter_std": float(base_row["jitter_std"]),
                "jitter_iteration": int(jitter_iteration),
                "response_start_iteration": int(start_iter),
                "response_end_iteration": int(end_iter),
                "response_l2": response_l2,
                "final_best_true": float(base_row["best_true_so_far"]),
                "final_simple_regret_true": float(base_row["simple_regret_true"]),
                "final_cum_regret_true": float(base_row["regret_cum_true"]),
                "final_avg_regret_true": float(base_row["regret_avg_true"]),
                "auc_simple_regret_true": float(
                    np.trapezoid(run_df["simple_regret_true"].to_numpy(dtype=float), dx=1.0)
                ),
                "param_columns": ",".join(param_columns),
            }
            for column in param_columns:
                row[f"response_{column}"] = float(response[column])
            rows.append(row)

    return pd.DataFrame(rows)


def build_paired_table(response_df: pd.DataFrame) -> pd.DataFrame:
    base_keys = ["dataset", "objective", "acquisition", "seed", "oracle_model", "jitter_iteration"]
    jittered = response_df[~response_df["baseline"]].copy()
    baseline = response_df[response_df["baseline"]].copy()

    metric_columns = [
        "response_l2",
        "final_best_true",
        "final_simple_regret_true",
        "final_cum_regret_true",
        "final_avg_regret_true",
        "auc_simple_regret_true",
    ]
    paired = jittered.merge(
        baseline[base_keys + metric_columns],
        on=base_keys,
        how="inner",
        suffixes=("_jitter", "_baseline"),
    )

    if paired.empty:
        raise ValueError("No baseline/jitter pairs could be formed from the run logs.")

    paired["response_l2_excess"] = paired["response_l2_jitter"] - paired["response_l2_baseline"]
    paired["final_best_true_excess"] = paired["final_best_true_jitter"] - paired["final_best_true_baseline"]
    paired["final_simple_regret_excess_true"] = (
        paired["final_simple_regret_true_jitter"] - paired["final_simple_regret_true_baseline"]
    )
    paired["final_cum_regret_excess_true"] = (
        paired["final_cum_regret_true_jitter"] - paired["final_cum_regret_true_baseline"]
    )
    paired["final_avg_regret_excess_true"] = (
        paired["final_avg_regret_true_jitter"] - paired["final_avg_regret_true_baseline"]
    )
    paired["auc_simple_regret_excess_true"] = (
        paired["auc_simple_regret_true_jitter"] - paired["auc_simple_regret_true_baseline"]
    )
    return paired


def summarize_conditions(paired: pd.DataFrame) -> pd.DataFrame:
    condition_cols = ["dataset", "objective", "error_model", "jitter_iteration", "jitter_std", "acquisition"]
    summary = (
        paired.groupby(condition_cols, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_response_l2_excess=("response_l2_excess", "mean"),
            median_response_l2_excess=("response_l2_excess", "median"),
            mean_final_simple_regret_excess_true=("final_simple_regret_excess_true", "mean"),
            median_final_simple_regret_excess_true=("final_simple_regret_excess_true", "median"),
            mean_auc_simple_regret_excess_true=(PRIMARY_METRIC, "mean"),
            median_auc_simple_regret_excess_true=(PRIMARY_METRIC, "median"),
            std_auc_simple_regret_excess_true=(PRIMARY_METRIC, "std"),
        )
        .reset_index()
    )
    return summary


def compute_condition_rankings(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    condition_cols = ["dataset", "objective", "error_model", "jitter_iteration", "jitter_std"]

    for condition_values, group in paired.groupby(condition_cols, dropna=False):
        if not isinstance(condition_values, tuple):
            condition_values = (condition_values,)
        condition = dict(zip(condition_cols, condition_values))

        pivot = group.pivot_table(
            index="seed",
            columns="acquisition",
            values=PRIMARY_METRIC,
            aggfunc="mean",
        )
        pivot = pivot.dropna(axis=0, how="any").sort_index()
        if pivot.empty or pivot.shape[1] < 2:
            continue

        ranks = pivot.rank(axis=1, method="average", ascending=True)
        mean_ranks = ranks.mean(axis=0).sort_values()

        metric_means = pivot.mean(axis=0)
        metric_medians = pivot.median(axis=0)
        response_means = (
            group.pivot_table(index="seed", columns="acquisition", values=REACTION_METRIC, aggfunc="mean")
            .reindex(index=pivot.index, columns=pivot.columns)
            .mean(axis=0)
        )

        best_mean_rank = float(mean_ranks.min())
        for acquisition in mean_ranks.index:
            ranking_rows.append(
                {
                    **condition,
                    "acquisition": str(acquisition),
                    "n_seeds": int(pivot.shape[0]),
                    "mean_rank": float(mean_ranks[acquisition]),
                    "mean_auc_simple_regret_excess_true": float(metric_means[acquisition]),
                    "median_auc_simple_regret_excess_true": float(metric_medians[acquisition]),
                    "mean_response_l2_excess": float(response_means[acquisition]),
                    "condition_win": math.isclose(float(mean_ranks[acquisition]), best_mean_rank, rel_tol=1e-12),
                }
            )

        row: dict[str, object] = {
            **condition,
            "n_seeds": int(pivot.shape[0]),
            "n_acquisitions": int(pivot.shape[1]),
            "best_acquisition": str(mean_ranks.index[0]),
            "best_mean_rank": float(mean_ranks.iloc[0]),
        }
        if pivot.shape[1] > 2:
            stat, p_value = friedmanchisquare(*[pivot[col].to_numpy(dtype=float) for col in pivot.columns])
            kendalls_w = float(stat / (pivot.shape[0] * (pivot.shape[1] - 1)))
            row.update(
                {
                    "test_name": "friedman",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "kendalls_w": kendalls_w,
                }
            )
        else:
            diff = pivot.iloc[:, 0] - pivot.iloc[:, 1]
            try:
                result = wilcoxon(diff.to_numpy(dtype=float), zero_method="wilcox", alternative="two-sided")
                stat = float(result.statistic)
                p_value = float(result.pvalue)
            except ValueError:
                stat = np.nan
                p_value = np.nan
            row.update(
                {
                    "test_name": "wilcoxon",
                    "statistic": stat,
                    "p_value": p_value,
                    "kendalls_w": np.nan,
                }
            )
        test_rows.append(row)

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        valid_mask = tests["p_value"].notna()
        adjusted = np.full(len(tests), np.nan, dtype=float)
        rejected = np.zeros(len(tests), dtype=bool)
        if valid_mask.any():
            reject, p_adj, _, _ = multipletests(tests.loc[valid_mask, "p_value"], method="fdr_bh")
            adjusted[valid_mask.to_numpy()] = p_adj
            rejected[valid_mask.to_numpy()] = reject
        tests["p_value_fdr_bh"] = adjusted
        tests["p_value_rejected"] = rejected

    return pd.DataFrame(ranking_rows), tests


def compute_overall_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    overall = (
        rankings.groupby(["dataset", "objective", "acquisition"], dropna=False)
        .agg(
            mean_rank=("mean_rank", "mean"),
            mean_auc_simple_regret_excess_true=("mean_auc_simple_regret_excess_true", "mean"),
            mean_response_l2_excess=("mean_response_l2_excess", "mean"),
            condition_wins=("condition_win", "sum"),
            conditions=("condition_win", "size"),
        )
        .reset_index()
        .sort_values(["dataset", "objective", "mean_rank", "mean_auc_simple_regret_excess_true"])
    )
    return overall


def plot_condition_heatmaps(rankings: pd.DataFrame, output_dir: Path) -> None:
    if rankings.empty:
        return

    sns.set_theme(style="whitegrid")
    acquisition_order = (
        rankings.groupby("acquisition")["mean_rank"].mean().sort_values().index.tolist()
    )

    for (dataset, objective, error_model), group in rankings.groupby(
        ["dataset", "objective", "error_model"],
        dropna=False,
    ):
        jitter_iterations = sorted(group["jitter_iteration"].unique().tolist())
        fig_rank, axes_rank = plt.subplots(
            1,
            len(jitter_iterations),
            figsize=(4.5 * len(jitter_iterations), 7),
            squeeze=False,
        )
        fig_auc, axes_auc = plt.subplots(
            1,
            len(jitter_iterations),
            figsize=(4.5 * len(jitter_iterations), 7),
            squeeze=False,
        )

        for idx, jitter_iteration in enumerate(jitter_iterations):
            subset = group[group["jitter_iteration"] == jitter_iteration]
            rank_pivot = subset.pivot_table(
                index="acquisition",
                columns="jitter_std",
                values="mean_rank",
                aggfunc="mean",
            ).reindex(acquisition_order)
            auc_pivot = subset.pivot_table(
                index="acquisition",
                columns="jitter_std",
                values="mean_auc_simple_regret_excess_true",
                aggfunc="mean",
            ).reindex(acquisition_order)

            ax_rank = axes_rank[0, idx]
            sns.heatmap(
                rank_pivot,
                ax=ax_rank,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu_r",
                cbar=idx == len(jitter_iterations) - 1,
                cbar_kws={"label": "Mean rank"},
                mask=rank_pivot.isna(),
            )
            ax_rank.set_title(f"jitter={jitter_iteration}")
            ax_rank.set_xlabel("Jitter std")
            ax_rank.set_ylabel("Acquisition" if idx == 0 else "")

            ax_auc = axes_auc[0, idx]
            sns.heatmap(
                auc_pivot,
                ax=ax_auc,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn_r",
                center=0,
                cbar=idx == len(jitter_iterations) - 1,
                cbar_kws={"label": "Mean excess AUC regret"},
                mask=auc_pivot.isna(),
            )
            ax_auc.set_title(f"jitter={jitter_iteration}")
            ax_auc.set_xlabel("Jitter std")
            ax_auc.set_ylabel("Acquisition" if idx == 0 else "")

        fig_rank.suptitle(f"Acquisition mean ranks: {dataset} / {objective} / {error_model}", y=0.98)
        fig_rank.tight_layout()
        fig_rank.savefig(
            output_dir / f"mean_rank_{dataset}_{objective}_{error_model}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_rank)

        fig_auc.suptitle(
            f"Excess AUC simple regret: {dataset} / {objective} / {error_model}",
            y=0.98,
        )
        fig_auc.tight_layout()
        fig_auc.savefig(
            output_dir / f"mean_excess_auc_{dataset}_{objective}_{error_model}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_auc)


def write_report(
    output_dir: Path,
    response_df: pd.DataFrame,
    paired: pd.DataFrame,
    rankings: pd.DataFrame,
    tests: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    report_path = output_dir / "evaluation_report.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("ROBUSTNESS EVALUATION REPORT\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Recommended primary evaluation\n")
        handle.write("- Lower excess AUC simple regret is better.\n")
        handle.write("- Lower excess final simple regret is better.\n")
        handle.write("- Response-step excess L2 is descriptive only: it shows how strongly the next\n")
        handle.write("  recommendation changes after the first noisy observation.\n\n")

        handle.write("Why the existing delta_excess_l2_norm should not be trusted as-is\n")
        handle.write("- In the current outputs, delta_excess_l2_norm is identically zero.\n")
        handle.write("- The simulator applies noise starting at iteration jitter_iteration + 1,\n")
        handle.write("  so the first candidate that can react is iteration jitter_iteration + 2.\n")
        handle.write("- This script therefore recomputes the response step as (t+2) - (t+1).\n\n")

        handle.write("Data coverage\n")
        handle.write(f"- Response rows: {len(response_df)}\n")
        handle.write(f"- Paired baseline/jitter rows: {len(paired)}\n")
        if not paired.empty:
            seed_counts = paired.groupby(
                ["dataset", "objective", "error_model", "jitter_iteration", "jitter_std"]
            )["seed"].nunique()
            handle.write(
                f"- Seeds per condition: min={int(seed_counts.min())}, "
                f"median={float(seed_counts.median()):.1f}, max={int(seed_counts.max())}\n"
            )
        handle.write("\n")

        handle.write("Statistical plausibility notes\n")
        handle.write("- With only 5 seeds per condition, p-values are low power.\n")
        handle.write("- Use mean rank, median excess regret, and sign consistency as the main evidence.\n")
        handle.write("- Friedman tests are appropriate for comparing >2 acquisitions within the same\n")
        handle.write("  condition because seeds act as matched blocks.\n")
        handle.write("- For 2-method conditions, paired Wilcoxon is used.\n")
        handle.write("- Dataset comparisons should stay stratified by dataset rather than pooling unless\n")
        handle.write("  you explicitly normalize outcomes across datasets.\n\n")

        if not tests.empty:
            handle.write("Condition-level omnibus tests\n")
            keep_cols = [
                "dataset",
                "objective",
                "error_model",
                "jitter_iteration",
                "jitter_std",
                "n_seeds",
                "n_acquisitions",
                "best_acquisition",
                "test_name",
                "statistic",
                "p_value",
                "p_value_fdr_bh",
                "kendalls_w",
            ]
            handle.write(tests[keep_cols].to_string(index=False))
            handle.write("\n\n")

        if not overall.empty:
            handle.write("Overall acquisition ranking by dataset/objective\n")
            for (dataset, objective), group in overall.groupby(["dataset", "objective"], dropna=False):
                handle.write(f"{dataset} / {objective}\n")
                handle.write(
                    group.head(5)[
                        [
                            "acquisition",
                            "mean_rank",
                            "mean_auc_simple_regret_excess_true",
                            "mean_response_l2_excess",
                            "condition_wins",
                            "conditions",
                        ]
                    ].to_string(index=False)
                )
                handle.write("\n\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logs = load_iteration_logs(args.input_dir)
    response_df = build_response_table(logs)
    paired = build_paired_table(response_df)
    condition_summary = summarize_conditions(paired)
    rankings, tests = compute_condition_rankings(paired)
    overall = compute_overall_rankings(rankings)

    response_df.to_csv(args.output_dir / "response_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_excess_metrics.csv", index=False)
    condition_summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
    rankings.to_csv(args.output_dir / "condition_rankings.csv", index=False)
    tests.to_csv(args.output_dir / "condition_tests.csv", index=False)
    overall.to_csv(args.output_dir / "overall_rankings.csv", index=False)

    plot_condition_heatmaps(rankings, args.output_dir)
    write_report(args.output_dir, response_df, paired, rankings, tests, overall)
    print(f"Evaluation outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
