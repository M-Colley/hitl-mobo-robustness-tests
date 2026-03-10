"""Prepare and analyze a confirmatory follow-up experiment."""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests

PRIMARY_METRIC = "auc_simple_regret_excess_true"
SECONDARY_METRIC = "final_simple_regret_excess_true"
REACTION_METRIC = "response_l2_excess"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input-dir", type=Path, default=Path("output"))
    parser.add_argument("--base-evaluation-dir", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--output-root", type=Path, default=Path("output/confirmatory"))
    parser.add_argument("--objective", type=str, default="composite")
    parser.add_argument("--acquisitions", type=str, default="pi,logpi")
    parser.add_argument("--seed-start", type=int, default=12)
    parser.add_argument("--num-new-seeds", type=int, default=20)
    parser.add_argument("--oracle-model", type=str, default="auto")
    parser.add_argument("--oracle-selection-path", type=Path, default=Path("output/best_oracle_models.json"))
    parser.add_argument("--error-models", type=str, default="gaussian,bias")
    parser.add_argument("--jitter-iterations", type=str, default="10,20,40")
    parser.add_argument("--jitter-stds", type=str, default="0.05,0.5,1,5")
    parser.add_argument("--run-simulation", action="store_true", default=False)
    parser.add_argument("--parallel", action="store_true", default=False)
    return parser.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_seed_list(seed_start: int, num_new_seeds: int) -> list[int]:
    return list(range(seed_start, seed_start + num_new_seeds))


def prepare_directories(output_root: Path) -> dict[str, Path]:
    paths = {
        "raw": output_root / "raw",
        "evaluation": output_root / "evaluation",
        "figures": output_root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def copy_existing_logs(base_input_dir: Path, raw_dir: Path, objective: str, acquisitions: list[str]) -> list[Path]:
    copied: list[Path] = []
    for acquisition in acquisitions:
        pattern = f"bo_sensor_error_*_{objective}_{acquisition}_seed*_*.csv"
        for source in sorted(base_input_dir.glob(pattern)):
            destination = raw_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
            copied.append(destination)
    return copied


def run_simulation(args: argparse.Namespace, raw_dir: Path, acquisitions: list[str], seeds: list[int]) -> None:
    command = [
        sys.executable,
        "scripts/bo_sensor_error_simulation.py",
        "--objective",
        args.objective,
        "--acq-list",
        ",".join(acquisitions),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--oracle-model",
        args.oracle_model,
        "--oracle-selection-path",
        str(args.oracle_selection_path),
        "--error-models",
        args.error_models,
        "--jitter-iterations",
        args.jitter_iterations,
        "--jitter-stds",
        args.jitter_stds,
        "--output-dir",
        str(raw_dir),
    ]
    if args.parallel:
        command.append("--parallel")
    subprocess.run(command, check=True)


def run_evaluation(raw_dir: Path, evaluation_dir: Path) -> None:
    command = [
        sys.executable,
        "scripts/evaluate_research_question.py",
        "--input-dir",
        str(raw_dir),
        "--output-dir",
        str(evaluation_dir),
    ]
    subprocess.run(command, check=True)


def paired_stats_from_diff(diff: pd.Series, ref: pd.Series, challenger: pd.Series) -> dict[str, float]:
    d = diff.to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    ref_values = ref.to_numpy(dtype=float)
    ref_values = ref_values[np.isfinite(ref_values)]
    challenger_values = challenger.to_numpy(dtype=float)
    challenger_values = challenger_values[np.isfinite(challenger_values)]
    n = int(d.size)
    result = {
        "n_pairs": n,
        "mean_reference": float(np.mean(ref_values)) if ref_values.size else np.nan,
        "mean_challenger": float(np.mean(challenger_values)) if challenger_values.size else np.nan,
        "mean_diff": float(np.mean(d)) if n else np.nan,
        "median_diff": float(np.median(d)) if n else np.nan,
        "std_diff": float(np.std(d, ddof=1)) if n > 1 else np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "cohens_dz": np.nan,
        "p_value_t": np.nan,
        "p_value_wilcoxon": np.nan,
    }
    if n >= 2:
        if np.nanstd(d, ddof=1) > 0:
            t_result = ttest_1samp(d, popmean=0.0)
            result["p_value_t"] = float(t_result.pvalue)
            std_diff = float(np.std(d, ddof=1))
            sem = std_diff / math.sqrt(n)
            critical = float(student_t.ppf(0.975, df=n - 1))
            margin = critical * sem
            result["ci95_low"] = result["mean_diff"] - margin
            result["ci95_high"] = result["mean_diff"] + margin
            result["cohens_dz"] = result["mean_diff"] / std_diff
        else:
            result["p_value_t"] = 1.0
            result["ci95_low"] = result["mean_diff"]
            result["ci95_high"] = result["mean_diff"]
        try:
            wilcoxon_result = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            result["p_value_wilcoxon"] = float(wilcoxon_result.pvalue)
        except ValueError:
            if np.allclose(d, 0.0):
                result["p_value_wilcoxon"] = 1.0
    return result


def build_head_to_head_table(paired: pd.DataFrame, reference: str, challenger: str) -> pd.DataFrame:
    subset = paired[paired["acquisition"].isin([reference, challenger])].copy()
    index_cols = ["dataset", "objective", "error_model", "jitter_iteration", "jitter_std", "seed"]
    value_cols = [PRIMARY_METRIC, SECONDARY_METRIC, REACTION_METRIC]
    pivot = subset.pivot_table(index=index_cols, columns="acquisition", values=value_cols, aggfunc="mean")
    pivot.columns = [f"{metric}_{acq}" for metric, acq in pivot.columns]
    pivot = pivot.reset_index()
    needed = [f"{PRIMARY_METRIC}_{reference}", f"{PRIMARY_METRIC}_{challenger}"]
    pivot = pivot.dropna(subset=needed)
    for metric in value_cols:
        pivot[f"{metric}_diff_{challenger}_minus_{reference}"] = (
            pivot[f"{metric}_{challenger}"] - pivot[f"{metric}_{reference}"]
        )
    return pivot


def summarize_condition_tests(head_to_head: pd.DataFrame, reference: str, challenger: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [PRIMARY_METRIC, SECONDARY_METRIC]
    group_cols = ["dataset", "objective", "error_model", "jitter_iteration", "jitter_std"]
    for group_values, group in head_to_head.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        condition = dict(zip(group_cols, group_values))
        for metric in metrics:
            diff_col = f"{metric}_diff_{challenger}_minus_{reference}"
            stats = paired_stats_from_diff(
                group[diff_col],
                group[f"{metric}_{reference}"],
                group[f"{metric}_{challenger}"],
            )
            rows.append({
                **condition,
                "metric": metric,
                "reference": reference,
                "challenger": challenger,
                **stats,
            })
    tests = pd.DataFrame(rows)
    for column in ["p_value_t", "p_value_wilcoxon"]:
        adjusted = np.full(len(tests), np.nan, dtype=float)
        rejected = np.zeros(len(tests), dtype=bool)
        for metric in tests["metric"].dropna().unique():
            metric_index = tests.index[tests["metric"] == metric]
            valid_mask = tests.loc[metric_index, column].notna()
            if not valid_mask.any():
                continue
            valid_index = metric_index[valid_mask]
            reject, p_adj, _, _ = multipletests(tests.loc[valid_index, column], method="fdr_bh")
            adjusted[valid_index.to_numpy(dtype=int)] = p_adj
            rejected[valid_index.to_numpy(dtype=int)] = reject
        tests[f"{column}_fdr_bh"] = adjusted
        tests[f"{column}_rejected"] = rejected
    return tests


def summarize_dataset_tests(head_to_head: pd.DataFrame, reference: str, challenger: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [PRIMARY_METRIC, SECONDARY_METRIC]
    seed_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []

    for (dataset, objective, seed), group in head_to_head.groupby(["dataset", "objective", "seed"], dropna=False):
        row = {"dataset": dataset, "objective": objective, "seed": seed}
        for metric in metrics:
            row[f"{metric}_{reference}"] = float(group[f"{metric}_{reference}"].mean())
            row[f"{metric}_{challenger}"] = float(group[f"{metric}_{challenger}"].mean())
            row[f"{metric}_diff_{challenger}_minus_{reference}"] = float(
                group[f"{metric}_diff_{challenger}_minus_{reference}"].mean()
            )
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)

    for (dataset, objective), group in seed_df.groupby(["dataset", "objective"], dropna=False):
        for metric in metrics:
            diff_col = f"{metric}_diff_{challenger}_minus_{reference}"
            stats = paired_stats_from_diff(
                group[diff_col],
                group[f"{metric}_{reference}"],
                group[f"{metric}_{challenger}"],
            )
            stats_rows.append({
                "dataset": dataset,
                "objective": objective,
                "metric": metric,
                "reference": reference,
                "challenger": challenger,
                **stats,
            })
    return seed_df, pd.DataFrame(stats_rows)


def build_main_confirmatory_table(
    dataset_tests: pd.DataFrame,
    reference: str,
    challenger: str,
) -> pd.DataFrame:
    table = dataset_tests[dataset_tests["metric"] == PRIMARY_METRIC].copy()
    if table.empty:
        return table

    table["winner"] = np.where(table["mean_diff"] < 0, challenger, reference)
    table["comparison"] = f"{reference} vs {challenger}"
    table["delta_label"] = f"{challenger} - {reference}"
    ordered_columns = [
        "dataset",
        "objective",
        "comparison",
        "n_pairs",
        "mean_reference",
        "mean_challenger",
        "mean_diff",
        "ci95_low",
        "ci95_high",
        "cohens_dz",
        "p_value_t",
        "p_value_wilcoxon",
        "winner",
        "delta_label",
    ]
    table = table[ordered_columns].sort_values(["dataset", "objective"]).reset_index(drop=True)
    return table


def _format_numeric(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _format_p_value(value: float) -> str:
    if pd.isna(value):
        return "--"
    if float(value) < 0.001:
        return "<0.001"
    return f"{float(value):.3f}"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_main_confirmatory_table(
    output_dir: Path,
    main_table: pd.DataFrame,
    reference: str,
    challenger: str,
) -> None:
    csv_path = output_dir / "main_confirmatory_table.csv"
    tex_path = output_dir / "main_confirmatory_table.tex"
    main_table.to_csv(csv_path, index=False)

    caption = (
        f"Confirmatory paired comparison for the primary robustness metric "
        f"({PRIMARY_METRIC}) using matched seeds."
    )
    label = "tab:confirmatory-primary"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llrrrrrrll}",
        r"\toprule",
        (
            "Dataset & Objective & $n$ & "
            f"Mean {_latex_escape(reference)} & Mean {_latex_escape(challenger)} & "
            rf"$\Delta$ ({_latex_escape(challenger)} - {_latex_escape(reference)}) & "
            r"95\% CI & $d_z$ & $p_t$ & $p_W$ \\"
        ),
        r"\midrule",
    ]

    if main_table.empty:
        lines.append(r"\multicolumn{10}{c}{No confirmatory results available yet.} \\")
    else:
        for _, row in main_table.iterrows():
            ci_text = f"[{_format_numeric(row['ci95_low'])}, {_format_numeric(row['ci95_high'])}]"
            lines.append(
                " & ".join(
                    [
                        _latex_escape(row["dataset"]),
                        _latex_escape(row["objective"]),
                        str(int(row["n_pairs"])),
                        _format_numeric(row["mean_reference"]),
                        _format_numeric(row["mean_challenger"]),
                        _format_numeric(row["mean_diff"]),
                        _latex_escape(ci_text),
                        _format_numeric(row["cohens_dz"]),
                        _latex_escape(_format_p_value(row["p_value_t"])),
                        _latex_escape(_format_p_value(row["p_value_wilcoxon"])),
                    ]
                )
                + r" \\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def plot_confirmatory_heatmaps(condition_summary: pd.DataFrame, output_dir: Path, acquisitions: list[str]) -> list[str]:
    paths: list[str] = []
    data = condition_summary[condition_summary["acquisition"].isin(acquisitions)].copy()
    for acquisition, subset in data.groupby("acquisition", dropna=False):
        datasets = sorted(subset["dataset"].dropna().unique().tolist())
        error_models = sorted(subset["error_model"].dropna().unique().tolist())
        fig, axes = plt.subplots(len(datasets), len(error_models), figsize=(4.6 * len(error_models), 3.8 * len(datasets)), squeeze=False)
        vmax = np.nanmax(np.abs(subset["mean_auc_simple_regret_excess_true"].to_numpy(dtype=float)))
        vmax = 1.0 if not np.isfinite(vmax) or np.isclose(vmax, 0.0) else vmax
        for row_idx, dataset in enumerate(datasets):
            for col_idx, error_model in enumerate(error_models):
                ax = axes[row_idx, col_idx]
                panel = subset[(subset["dataset"] == dataset) & (subset["error_model"] == error_model)]
                pivot = panel.pivot_table(index="jitter_iteration", columns="jitter_std", values="mean_auc_simple_regret_excess_true", aggfunc="mean")
                sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0, vmin=-vmax, vmax=vmax, cbar=row_idx == 0 and col_idx == len(error_models) - 1, cbar_kws={"label": "Mean excess AUC regret"}, mask=pivot.isna())
                ax.set_title(f"{dataset} / {error_model}")
                ax.set_xlabel("Jitter std")
                ax.set_ylabel("Jitter iteration")
        fig.suptitle(f"Confirmatory heatmap: {acquisition}", y=0.99)
        fig.tight_layout()
        path = output_dir / f"confirmatory_heatmap_{acquisition}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_winner_map(head_to_head: pd.DataFrame, reference: str, challenger: str, output_dir: Path) -> str:
    data = head_to_head.copy()
    diff_col = f"{PRIMARY_METRIC}_diff_{challenger}_minus_{reference}"
    data["winner"] = np.where(data[diff_col] < 0, challenger, reference)
    mean_data = data.groupby(["dataset", "error_model", "jitter_iteration", "jitter_std"], dropna=False).agg(mean_diff=(diff_col, "mean")).reset_index()
    mean_data["winner"] = np.where(mean_data["mean_diff"] < 0, challenger, reference)
    code_map = {reference: 0, challenger: 1}
    mean_data["winner_code"] = mean_data["winner"].map(code_map)
    datasets = sorted(mean_data["dataset"].dropna().unique().tolist())
    error_models = sorted(mean_data["error_model"].dropna().unique().tolist())
    fig, axes = plt.subplots(len(datasets), len(error_models), figsize=(4.8 * len(error_models), 3.8 * len(datasets)), squeeze=False)
    cmap = ListedColormap(sns.color_palette("Set2", n_colors=2))
    for row_idx, dataset in enumerate(datasets):
        for col_idx, error_model in enumerate(error_models):
            ax = axes[row_idx, col_idx]
            subset = mean_data[(mean_data["dataset"] == dataset) & (mean_data["error_model"] == error_model)]
            pivot = subset.pivot_table(index="jitter_iteration", columns="jitter_std", values="winner_code", aggfunc="mean")
            annot = subset.pivot_table(index="jitter_iteration", columns="jitter_std", values="winner", aggfunc="first")
            sns.heatmap(pivot, ax=ax, annot=annot, fmt="", cmap=cmap, cbar=False, mask=pivot.isna())
            ax.set_title(f"{dataset} / {error_model}")
            ax.set_xlabel("Jitter std")
            ax.set_ylabel("Jitter iteration")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(code), markersize=10, label=label) for label, code in code_map.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.suptitle(f"Winner map: {reference} vs {challenger}", y=1.02)
    fig.tight_layout()
    path = output_dir / f"winner_map_{reference}_vs_{challenger}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_head_to_head_lines(condition_summary: pd.DataFrame, output_dir: Path, acquisitions: list[str]) -> str:
    data = condition_summary[condition_summary["acquisition"].isin(acquisitions)].copy()
    plot = sns.relplot(
        data=data,
        x="jitter_std",
        y="mean_auc_simple_regret_excess_true",
        hue="acquisition",
        style="error_model",
        kind="line",
        col="jitter_iteration",
        row="dataset",
        marker="o",
        dashes=False,
        height=3.5,
        aspect=1.15,
    )
    for ax in plot.axes.flat:
        ax.axhline(0.0, color="grey", linewidth=1, alpha=0.5)
    plot.set_titles(row_template="{row_name}", col_template="jitter={col_name}")
    plot.set_axis_labels("Jitter std", "Mean excess AUC regret")
    plot.fig.suptitle("Head-to-head robustness lines", y=1.02)
    path = output_dir / "head_to_head_lines.png"
    plot.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(plot.fig)
    return str(path)


def plot_paired_points(seed_df: pd.DataFrame, output_dir: Path, reference: str, challenger: str) -> list[str]:
    paths: list[str] = []
    for metric, label in [(PRIMARY_METRIC, "primary"), (SECONDARY_METRIC, "secondary")]:
        rows: list[dict[str, object]] = []
        for _, row in seed_df.iterrows():
            rows.append({"dataset": row["dataset"], "seed": row["seed"], "acquisition": reference, "value": row[f"{metric}_{reference}"]})
            rows.append({"dataset": row["dataset"], "seed": row["seed"], "acquisition": challenger, "value": row[f"{metric}_{challenger}"]})
        tidy = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, len(tidy["dataset"].dropna().unique()), figsize=(5.0 * len(tidy["dataset"].dropna().unique()), 4.5), squeeze=False)
        for idx, dataset in enumerate(sorted(tidy["dataset"].dropna().unique().tolist())):
            ax = axes[0, idx]
            subset = tidy[tidy["dataset"] == dataset]
            pivot = subset.pivot_table(index="seed", columns="acquisition", values="value", aggfunc="mean").dropna()
            for _, values in pivot.iterrows():
                ax.plot([0, 1], [values[reference], values[challenger]], color="grey", alpha=0.35, linewidth=1)
            sns.boxplot(data=subset, x="acquisition", y="value", ax=ax, width=0.5, showfliers=False)
            sns.stripplot(data=subset, x="acquisition", y="value", ax=ax, color="black", size=4, alpha=0.7)
            ax.set_title(dataset)
            ax.set_xlabel("")
            ax.set_ylabel(metric)
        fig.suptitle(f"Paired confirmatory comparison: {metric}", y=1.02)
        fig.tight_layout()
        path = output_dir / f"paired_points_{label}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_summary(output_root: Path, base_overall: pd.DataFrame, condition_tests: pd.DataFrame, dataset_tests: pd.DataFrame, reference: str, challenger: str, copied_count: int, new_seed_count: int, run_simulation_flag: bool) -> None:
    path = output_root / "confirmatory_summary.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("CONFIRMATORY FOLLOW-UP SUMMARY\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Design\n")
        handle.write(f"- Broad-screen results were narrowed to {reference} vs {challenger}.\n")
        handle.write("- Objective kept: composite.\n")
        handle.write("- Error sweep kept: same jitter iterations, jitter stds, and error models.\n")
        handle.write(f"- Existing broad-screen logs reused: {copied_count} files.\n")
        handle.write(f"- Newly requested seeds: {new_seed_count}.\n")
        handle.write(f"- New simulation executed in this run: {run_simulation_flag}.\n\n")
        handle.write("Key exports\n")
        handle.write("- Main LaTeX table: evaluation/main_confirmatory_table.tex\n")
        handle.write("- Main CSV table: evaluation/main_confirmatory_table.csv\n\n")

        handle.write("Broad screening result\n")
        broad = base_overall[base_overall["acquisition"].isin([reference, challenger])].copy()
        if broad.empty:
            handle.write("- No broad-screen comparison available.\n\n")
        else:
            handle.write(broad.to_string(index=False))
            handle.write("\n\n")

        handle.write("Confirmatory dataset-level paired tests\n")
        if dataset_tests.empty:
            handle.write("- No confirmatory paired tests available yet.\n\n")
        else:
            handle.write(dataset_tests.to_string(index=False))
            handle.write("\n\n")

        handle.write("Condition-level paired tests\n")
        if condition_tests.empty:
            handle.write("- No condition-level tests available yet.\n")
        else:
            handle.write(condition_tests.to_string(index=False))


def main() -> None:
    args = parse_args()
    acquisitions = _parse_csv_list(args.acquisitions)
    if len(acquisitions) != 2:
        raise ValueError("Confirmatory follow-up currently expects exactly two acquisitions.")
    reference, challenger = acquisitions
    seeds = _parse_seed_list(args.seed_start, args.num_new_seeds)
    paths = prepare_directories(args.output_root)

    copied = copy_existing_logs(args.base_input_dir, paths["raw"], args.objective, acquisitions)

    if args.run_simulation:
        run_simulation(args, paths["raw"], acquisitions, seeds)

    run_evaluation(paths["raw"], paths["evaluation"])

    base_overall = pd.read_csv(args.base_evaluation_dir / "overall_rankings.csv")
    condition_summary = pd.read_csv(paths["evaluation"] / "condition_summary.csv")
    paired = pd.read_csv(paths["evaluation"] / "paired_excess_metrics.csv")

    head_to_head = build_head_to_head_table(paired, reference, challenger)
    condition_tests = summarize_condition_tests(head_to_head, reference, challenger)
    seed_df, dataset_tests = summarize_dataset_tests(head_to_head, reference, challenger)
    main_table = build_main_confirmatory_table(dataset_tests, reference, challenger)

    head_to_head.to_csv(paths["evaluation"] / "head_to_head_pairs.csv", index=False)
    condition_tests.to_csv(paths["evaluation"] / "head_to_head_condition_tests.csv", index=False)
    seed_df.to_csv(paths["evaluation"] / "head_to_head_seed_averages.csv", index=False)
    dataset_tests.to_csv(paths["evaluation"] / "head_to_head_dataset_tests.csv", index=False)
    write_main_confirmatory_table(paths["evaluation"], main_table, reference, challenger)

    figure_paths: list[str] = []
    figure_paths.extend(plot_confirmatory_heatmaps(condition_summary, paths["figures"], acquisitions))
    figure_paths.append(plot_winner_map(head_to_head, reference, challenger, paths["figures"]))
    figure_paths.append(plot_head_to_head_lines(condition_summary, paths["figures"], acquisitions))
    figure_paths.extend(plot_paired_points(seed_df, paths["figures"], reference, challenger))

    manifest = paths["figures"] / "figure_manifest.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("Confirmatory figure manifest\n")
        handle.write("=" * 60 + "\n\n")
        for figure_path in figure_paths:
            handle.write(Path(figure_path).name + "\n")

    write_summary(args.output_root, base_overall, condition_tests, dataset_tests, reference, challenger, len(copied), len(seeds), args.run_simulation)
    print(f"Confirmatory outputs saved to {args.output_root}")


if __name__ == "__main__":
    main()
