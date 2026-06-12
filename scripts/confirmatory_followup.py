"""Prepare and analyze a confirmatory follow-up experiment.

Seed handling (confirmatory vs. exploratory)
--------------------------------------------
The head-to-head hypothesis (e.g. pi vs. logpi) was *selected* on the
broad-screen seeds (seeds < ``--seed-start``). Reusing those seeds in the
confirmatory tests would double-dip and invalidate the p-values. Therefore:

- Everything labeled "confirmatory" (the condition-level and dataset-level
  paired tests, ``head_to_head_*.csv``, and the main LaTeX/CSV table) is
  computed from the NEW seeds only, i.e. ``seed >= --seed-start`` (seeds in
  ``range(seed_start, seed_start + num_new_seeds)``).
- The broad-screen logs are still copied next to the new runs so plots stay
  rich; analyses over the pooled (screening + new) seeds are written as
  clearly labeled ``exploratory_pooled_*`` outputs and figures, and must not
  be used for confirmatory claims.

Dataset-level metric choice
---------------------------
Different (jitter_std, jitter_iteration) cells produce mechanically different
excess-AUC magnitudes, so raw ``auc_simple_regret_excess_true`` must not be
averaged across conditions before testing. The dataset-level confirmatory test
therefore prefers the per-noisy-iteration normalized metric
``auc_simple_regret_excess_true_postonset_per_iter`` written by
``evaluate_research_question.py``. If that column is unavailable (older
evaluation outputs), the raw excess AUC is rescaled by the per-condition
standard deviation of the paired differences (scale-only standardization, no
centering) before averaging across conditions.

Multiple comparisons
--------------------
Benjamini-Hochberg FDR correction is applied across rows separately per test
family (t-tests as one family, Wilcoxon as another) and per metric; the main
table only declares a winner when the FDR-corrected primary test (paired t)
is significant, otherwise it reports "n.s.".
"""
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
# Per-noisy-iteration normalized excess AUC: comparable across error conditions,
# so it is the preferred metric when averaging across conditions before testing.
DATASET_PRIMARY_METRIC = "auc_simple_regret_excess_true_postonset_per_iter"
ALPHA = 0.05


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
    """Copy broad-screen logs next to the new runs.

    These screening-seed logs feed ONLY the pooled ``exploratory_pooled_*``
    outputs and figures; the confirmatory tests filter them out by seed
    (see module docstring).
    """
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
    index_cols = ["dataset", "objective", "oracle_model", "error_model", "jitter_iteration", "jitter_std", "seed"]
    if "oracle_model" not in subset.columns:
        # Older evaluation outputs: the duplicate check below still catches any
        # silent pooling across oracles.
        index_cols.remove("oracle_model")
    value_cols = [PRIMARY_METRIC, SECONDARY_METRIC, REACTION_METRIC]
    if DATASET_PRIMARY_METRIC in subset.columns:
        value_cols.append(DATASET_PRIMARY_METRIC)
    if subset.empty:
        columns = list(index_cols)
        columns += [f"{metric}_{acq}" for metric in value_cols for acq in (reference, challenger)]
        columns += [f"{metric}_diff_{challenger}_minus_{reference}" for metric in value_cols]
        return pd.DataFrame(columns=columns)
    cell_sizes = subset.groupby(index_cols + ["acquisition"], dropna=False).size()
    if (cell_sizes > 1).any():
        duplicated = cell_sizes[cell_sizes > 1]
        raise ValueError(
            "Duplicate paired rows for the same "
            f"{index_cols + ['acquisition']} cell; refusing to silently pool. "
            f"Offending cells:\n{duplicated.to_string()}"
        )
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


def apply_fdr_within_metric(tests: pd.DataFrame) -> pd.DataFrame:
    """Benjamini-Hochberg correction per test family (t / Wilcoxon) and per metric."""
    if tests.empty:
        return tests
    tests = tests.reset_index(drop=True)
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


def summarize_condition_tests(head_to_head: pd.DataFrame, reference: str, challenger: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [PRIMARY_METRIC, SECONDARY_METRIC]
    group_cols = ["dataset", "objective", "oracle_model", "error_model", "jitter_iteration", "jitter_std"]
    group_cols = [column for column in group_cols if column in head_to_head.columns]
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
    return apply_fdr_within_metric(tests)


def summarize_dataset_tests(head_to_head: pd.DataFrame, reference: str, challenger: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Seed-level averaging across conditions, then paired tests per dataset.

    Raw excess AUC is incommensurate across (jitter_std, jitter_iteration)
    cells, so averaging it across conditions before testing is invalid. This
    function prefers the per-noisy-iteration normalized metric
    ``DATASET_PRIMARY_METRIC``; when unavailable it rescales the raw excess
    AUC by the per-condition standard deviation of the paired differences
    (scale-only, no centering) before averaging. Returns the seed-level table,
    the FDR-corrected test table, and the name of the primary metric used.
    """
    diff_suffix = f"diff_{challenger}_minus_{reference}"
    if f"{DATASET_PRIMARY_METRIC}_{diff_suffix}" in head_to_head.columns:
        primary_metric = DATASET_PRIMARY_METRIC
    else:
        head_to_head = head_to_head.copy()
        primary_metric = f"{PRIMARY_METRIC}_per_condition_sd"
        condition_cols = ["dataset", "objective", "oracle_model", "error_model", "jitter_iteration", "jitter_std"]
        condition_cols = [column for column in condition_cols if column in head_to_head.columns]
        base_diff_col = f"{PRIMARY_METRIC}_{diff_suffix}"
        scale = head_to_head.groupby(condition_cols, dropna=False)[base_diff_col].transform(lambda s: s.std(ddof=1))
        scale = scale.where(scale > 0)
        head_to_head[f"{primary_metric}_{reference}"] = head_to_head[f"{PRIMARY_METRIC}_{reference}"] / scale
        head_to_head[f"{primary_metric}_{challenger}"] = head_to_head[f"{PRIMARY_METRIC}_{challenger}"] / scale
        head_to_head[f"{primary_metric}_{diff_suffix}"] = head_to_head[base_diff_col] / scale

    metrics = [primary_metric, SECONDARY_METRIC]
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
    if seed_df.empty:
        return seed_df, pd.DataFrame(), primary_metric

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
    return seed_df, apply_fdr_within_metric(pd.DataFrame(stats_rows)), primary_metric


def build_main_confirmatory_table(
    dataset_tests: pd.DataFrame,
    reference: str,
    challenger: str,
    primary_metric: str,
) -> pd.DataFrame:
    if dataset_tests.empty:
        return dataset_tests.copy()
    table = dataset_tests[dataset_tests["metric"] == primary_metric].copy()
    if table.empty:
        return table

    # The paired t-test on seed-averaged diffs is the pre-specified primary
    # test; a winner is only declared when its FDR-corrected p-value rejects.
    significant = table["p_value_t_fdr_bh"].notna() & (table["p_value_t_fdr_bh"] < ALPHA)
    direction = np.select(
        [table["mean_diff"] < 0, table["mean_diff"] > 0],
        [challenger, reference],
        default="n.s.",
    )
    table["winner"] = np.where(significant, direction, "n.s.")
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
        "p_value_t_fdr_bh",
        "p_value_wilcoxon_fdr_bh",
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
    primary_metric: str,
) -> None:
    csv_path = output_dir / "main_confirmatory_table.csv"
    tex_path = output_dir / "main_confirmatory_table.tex"
    main_table.to_csv(csv_path, index=False)

    caption = (
        f"Confirmatory paired comparison for the primary robustness metric "
        f"({primary_metric}) on the held-out confirmatory seeds only "
        f"(screening seeds excluded). p-values are Benjamini-Hochberg "
        f"FDR-corrected across rows; a winner is declared only when the "
        f"corrected paired t-test is significant."
    )
    label = "tab:confirmatory-primary"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llrrrrrrrrrrl}",
        r"\toprule",
        (
            "Dataset & Objective & $n$ & "
            f"Mean {_latex_escape(reference)} & Mean {_latex_escape(challenger)} & "
            rf"$\Delta$ ({_latex_escape(challenger)} - {_latex_escape(reference)}) & "
            r"95\% CI & $d_z$ & $p_t$ & $p_W$ & "
            r"$p_t^{\mathrm{BH}}$ & $p_W^{\mathrm{BH}}$ & Winner \\"
        ),
        r"\midrule",
    ]

    if main_table.empty:
        lines.append(r"\multicolumn{13}{c}{No confirmatory results available yet.} \\")
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
                        _latex_escape(_format_p_value(row["p_value_t_fdr_bh"])),
                        _latex_escape(_format_p_value(row["p_value_wilcoxon_fdr_bh"])),
                        _latex_escape(row["winner"]),
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
    """Exploratory heatmaps over the pooled (screening + confirmatory) seeds."""
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
        fig.suptitle(f"Exploratory heatmap (pooled seeds): {acquisition}", y=0.99)
        fig.tight_layout()
        path = output_dir / f"exploratory_pooled_heatmap_{acquisition}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_winner_map(head_to_head: pd.DataFrame, reference: str, challenger: str, output_dir: Path) -> str:
    """Exploratory winner map over pooled (screening + confirmatory) seeds.

    NaN mean diffs render as masked blanks and exact-zero diffs as explicit
    ties; neither is attributed to the reference acquisition.
    """
    data = head_to_head.copy()
    diff_col = f"{PRIMARY_METRIC}_diff_{challenger}_minus_{reference}"
    mean_data = data.groupby(["dataset", "error_model", "jitter_iteration", "jitter_std"], dropna=False).agg(mean_diff=(diff_col, "mean")).reset_index()
    diff = mean_data["mean_diff"].to_numpy(dtype=float)
    code_map = {reference: 0.0, "tie": 1.0, challenger: 2.0}
    mean_data["winner_code"] = np.select(
        [diff < 0, diff > 0],
        [code_map[challenger], code_map[reference]],
        default=code_map["tie"],
    )
    mean_data.loc[~np.isfinite(diff), "winner_code"] = np.nan
    code_labels = {code: label for label, code in code_map.items()}
    datasets = sorted(mean_data["dataset"].dropna().unique().tolist())
    error_models = sorted(mean_data["error_model"].dropna().unique().tolist())
    fig, axes = plt.subplots(len(datasets), len(error_models), figsize=(4.8 * len(error_models), 3.8 * len(datasets)), squeeze=False)
    palette = sns.color_palette("Set2", n_colors=2)
    cmap = ListedColormap([palette[0], "lightgrey", palette[1]])
    for row_idx, dataset in enumerate(datasets):
        for col_idx, error_model in enumerate(error_models):
            ax = axes[row_idx, col_idx]
            subset = mean_data[(mean_data["dataset"] == dataset) & (mean_data["error_model"] == error_model)]
            pivot = subset.pivot_table(index="jitter_iteration", columns="jitter_std", values="winner_code", aggfunc="mean")
            annot = pivot.astype(object)
            for code, label in code_labels.items():
                annot = annot.mask(pivot == code, label)
            annot = annot.where(pivot.notna(), "")
            sns.heatmap(pivot, ax=ax, annot=annot, fmt="", cmap=cmap, vmin=0.0, vmax=2.0, cbar=False, mask=pivot.isna())
            ax.set_title(f"{dataset} / {error_model}")
            ax.set_xlabel("Jitter std")
            ax.set_ylabel("Jitter iteration")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(code / 2.0), markersize=10, label=label) for label, code in code_map.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3)
    fig.suptitle(f"Winner map (exploratory, pooled seeds): {reference} vs {challenger}", y=1.02)
    fig.tight_layout()
    path = output_dir / f"exploratory_pooled_winner_map_{reference}_vs_{challenger}.png"
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
    plot.fig.suptitle("Head-to-head robustness lines (exploratory, pooled seeds)", y=1.02)
    path = output_dir / "exploratory_pooled_head_to_head_lines.png"
    plot.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(plot.fig)
    return str(path)


def plot_paired_points(seed_df: pd.DataFrame, output_dir: Path, reference: str, challenger: str, primary_metric: str) -> list[str]:
    paths: list[str] = []
    if seed_df.empty:
        return paths
    for metric, label in [(primary_metric, "primary"), (SECONDARY_METRIC, "secondary")]:
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
        fig.suptitle(f"Paired confirmatory comparison (new seeds only): {metric}", y=1.02)
        fig.tight_layout()
        path = output_dir / f"paired_points_{label}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_summary(output_root: Path, base_overall: pd.DataFrame, condition_tests: pd.DataFrame, dataset_tests: pd.DataFrame, reference: str, challenger: str, copied_count: int, new_seed_count: int, run_simulation_flag: bool, seed_start: int) -> None:
    path = output_root / "confirmatory_summary.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("CONFIRMATORY FOLLOW-UP SUMMARY\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Design\n")
        handle.write(f"- Broad-screen results were narrowed to {reference} vs {challenger}.\n")
        handle.write("- Objective kept: composite.\n")
        handle.write("- Error sweep kept: same jitter iterations, jitter stds, and error models.\n")
        handle.write(
            f"- Existing broad-screen logs reused for exploratory pooled outputs only: {copied_count} files.\n"
        )
        handle.write(f"- Newly requested seeds: {new_seed_count}.\n")
        handle.write(f"- New simulation executed in this run: {run_simulation_flag}.\n")
        handle.write(
            f"- IMPORTANT: all confirmatory tests below use ONLY new seeds (seed >= {seed_start}); "
            "the screening seeds that generated the hypothesis are excluded to avoid double-dipping.\n\n"
        )
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

        handle.write("Confirmatory dataset-level paired tests (new seeds only, BH-corrected)\n")
        if dataset_tests.empty:
            handle.write("- No confirmatory paired tests available yet (run with --run-simulation to generate new seeds).\n\n")
        else:
            handle.write(dataset_tests.to_string(index=False))
            handle.write("\n\n")

        handle.write("Confirmatory condition-level paired tests (new seeds only, BH-corrected)\n")
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

    # Pooled (screening + new seeds): exploratory only — the hypothesis was
    # selected on the screening seeds, so pooled tests would double-dip.
    exploratory_head_to_head = build_head_to_head_table(paired, reference, challenger)
    exploratory_condition_tests = summarize_condition_tests(exploratory_head_to_head, reference, challenger)

    # Confirmatory analysis: new seeds only (seed >= args.seed_start).
    confirmatory_paired = paired[paired["seed"] >= args.seed_start].copy()
    if confirmatory_paired.empty:
        print(
            f"WARNING: no runs with seed >= {args.seed_start} found; confirmatory "
            "tables will be empty. Run with --run-simulation to generate the new seeds."
        )
    head_to_head = build_head_to_head_table(confirmatory_paired, reference, challenger)
    condition_tests = summarize_condition_tests(head_to_head, reference, challenger)
    seed_df, dataset_tests, dataset_primary_metric = summarize_dataset_tests(head_to_head, reference, challenger)
    main_table = build_main_confirmatory_table(dataset_tests, reference, challenger, dataset_primary_metric)

    exploratory_head_to_head.to_csv(paths["evaluation"] / "exploratory_pooled_head_to_head_pairs.csv", index=False)
    exploratory_condition_tests.to_csv(paths["evaluation"] / "exploratory_pooled_condition_tests.csv", index=False)
    head_to_head.to_csv(paths["evaluation"] / "head_to_head_pairs.csv", index=False)
    condition_tests.to_csv(paths["evaluation"] / "head_to_head_condition_tests.csv", index=False)
    seed_df.to_csv(paths["evaluation"] / "head_to_head_seed_averages.csv", index=False)
    dataset_tests.to_csv(paths["evaluation"] / "head_to_head_dataset_tests.csv", index=False)
    write_main_confirmatory_table(paths["evaluation"], main_table, reference, challenger, dataset_primary_metric)

    figure_paths: list[str] = []
    figure_paths.extend(plot_confirmatory_heatmaps(condition_summary, paths["figures"], acquisitions))
    figure_paths.append(plot_winner_map(exploratory_head_to_head, reference, challenger, paths["figures"]))
    figure_paths.append(plot_head_to_head_lines(condition_summary, paths["figures"], acquisitions))
    figure_paths.extend(plot_paired_points(seed_df, paths["figures"], reference, challenger, dataset_primary_metric))

    manifest = paths["figures"] / "figure_manifest.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("Confirmatory figure manifest\n")
        handle.write("=" * 60 + "\n\n")
        for figure_path in figure_paths:
            handle.write(Path(figure_path).name + "\n")

    write_summary(args.output_root, base_overall, condition_tests, dataset_tests, reference, challenger, len(copied), len(seeds), args.run_simulation, args.seed_start)
    print(f"Confirmatory outputs saved to {args.output_root}")


if __name__ == "__main__":
    main()
