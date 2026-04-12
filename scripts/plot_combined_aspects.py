"""Create paper-friendly faceted views from evaluation outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation/figures"))
    parser.add_argument("--shortlist", type=str, default="pi,logpi,qucb")
    parser.add_argument(
        "--iteration-logs-dir",
        type=Path,
        default=None,
        help="Directory containing per-iteration CSV logs for regret trajectory plots "
             "(defaults to evaluation-dir parent).",
    )
    return parser.parse_args()


def load_inputs(evaluation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(evaluation_dir / "condition_summary.csv")
    rankings = pd.read_csv(evaluation_dir / "condition_rankings.csv")
    return summary, rankings


def _sorted_unique(values: pd.Series) -> list:
    return sorted(v for v in values.dropna().unique().tolist())


def plot_robustness_heatmaps(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    paths: list[str] = []
    sns.set_theme(style="whitegrid")

    for (objective, acquisition), data in summary.groupby(["objective", "acquisition"], dropna=False):
        datasets = _sorted_unique(data["dataset"])
        error_models = _sorted_unique(data["error_model"])
        fig, axes = plt.subplots(
            len(datasets),
            len(error_models),
            figsize=(4.8 * len(error_models), 3.8 * len(datasets)),
            squeeze=False,
        )

        vmax = np.nanmax(np.abs(data["mean_auc_simple_regret_excess_true"].to_numpy(dtype=float)))
        vmax = 1.0 if not np.isfinite(vmax) or np.isclose(vmax, 0.0) else vmax

        for row_idx, dataset in enumerate(datasets):
            for col_idx, error_model in enumerate(error_models):
                ax = axes[row_idx, col_idx]
                subset = data[(data["dataset"] == dataset) & (data["error_model"] == error_model)]
                pivot = subset.pivot_table(
                    index="jitter_iteration",
                    columns="jitter_std",
                    values="mean_auc_simple_regret_excess_true",
                    aggfunc="mean",
                )
                sns.heatmap(
                    pivot,
                    ax=ax,
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn_r",
                    center=0,
                    vmin=-vmax,
                    vmax=vmax,
                    cbar=row_idx == 0 and col_idx == len(error_models) - 1,
                    cbar_kws={"label": "Mean excess AUC regret"},
                    mask=pivot.isna(),
                )
                ax.set_title(f"{dataset} / {error_model}")
                ax.set_xlabel("Jitter std")
                ax.set_ylabel("Jitter iteration")

        fig.suptitle(f"Robustness loss heatmap: {objective} / {acquisition}", y=0.99)
        fig.tight_layout()
        path = output_dir / f"robustness_heatmap_{objective}_{acquisition}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_rank_progression(rankings: pd.DataFrame, output_dir: Path, shortlist: list[str]) -> list[str]:
    paths: list[str] = []
    filtered = rankings[rankings["acquisition"].isin(shortlist)].copy()
    if filtered.empty:
        return paths

    for objective, data in filtered.groupby("objective", dropna=False):
        plot = sns.relplot(
            data=data,
            x="jitter_std",
            y="mean_rank",
            hue="acquisition",
            style="error_model",
            kind="line",
            col="jitter_iteration",
            row="dataset",
            marker="o",
            dashes=False,
            height=3.4,
            aspect=1.2,
        )
        plot.set_titles(row_template="{row_name}", col_template="jitter={col_name}")
        plot.set_axis_labels("Jitter std", "Mean rank")
        plot.fig.suptitle(f"Rank progression: {objective}", y=1.02)
        path = output_dir / f"rank_progression_{objective}.png"
        plot.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(plot.fig)
        paths.append(str(path))
    return paths


def plot_winner_maps(rankings: pd.DataFrame, output_dir: Path) -> list[str]:
    paths: list[str] = []
    sns.set_theme(style="white")

    for objective, data in rankings.groupby("objective", dropna=False):
        winners = (
            data.sort_values(["dataset", "error_model", "jitter_iteration", "jitter_std", "mean_rank", "mean_auc_simple_regret_excess_true"])
            .groupby(["dataset", "error_model", "jitter_iteration", "jitter_std"], as_index=False)
            .first()
        )
        acquisitions = sorted(winners["acquisition"].unique().tolist())
        cmap = ListedColormap(sns.color_palette("Set2", n_colors=max(3, len(acquisitions))))
        acq_to_code = {acq: idx for idx, acq in enumerate(acquisitions)}
        winners["winner_code"] = winners["acquisition"].map(acq_to_code)

        datasets = _sorted_unique(winners["dataset"])
        error_models = _sorted_unique(winners["error_model"])
        fig, axes = plt.subplots(
            len(datasets),
            len(error_models),
            figsize=(4.8 * len(error_models), 3.8 * len(datasets)),
            squeeze=False,
        )

        for row_idx, dataset in enumerate(datasets):
            for col_idx, error_model in enumerate(error_models):
                ax = axes[row_idx, col_idx]
                subset = winners[(winners["dataset"] == dataset) & (winners["error_model"] == error_model)]
                pivot = subset.pivot_table(
                    index="jitter_iteration",
                    columns="jitter_std",
                    values="winner_code",
                    aggfunc="first",
                )
                annot = subset.pivot_table(
                    index="jitter_iteration",
                    columns="jitter_std",
                    values="acquisition",
                    aggfunc="first",
                )
                sns.heatmap(
                    pivot,
                    ax=ax,
                    annot=annot,
                    fmt="",
                    cmap=cmap,
                    cbar=False,
                    mask=pivot.isna(),
                )
                ax.set_title(f"{dataset} / {error_model}")
                ax.set_xlabel("Jitter std")
                ax.set_ylabel("Jitter iteration")

        handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(i), markersize=10, label=acq) for acq, i in acq_to_code.items()]
        fig.legend(handles=handles, loc="upper center", ncol=min(5, len(handles)))
        fig.suptitle(f"Winner map: {objective}", y=1.02)
        fig.tight_layout()
        path = output_dir / f"winner_map_{objective}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_response_scatter(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    paths: list[str] = []
    for objective, data in summary.groupby("objective", dropna=False):
        plot = sns.relplot(
            data=data,
            x="mean_response_l2_excess",
            y="mean_auc_simple_regret_excess_true",
            hue="acquisition",
            style="error_model",
            col="dataset",
            kind="scatter",
            height=4.2,
            aspect=1.0,
        )
        for ax in plot.axes.flat:
            ax.axhline(0.0, color="grey", linewidth=1, alpha=0.5)
            ax.axvline(0.0, color="grey", linewidth=1, alpha=0.5)
        plot.set_axis_labels("Mean response L2 excess", "Mean excess AUC regret")
        plot.fig.suptitle(f"Response vs robustness: {objective}", y=1.02)
        path = output_dir / f"response_vs_robustness_{objective}.png"
        plot.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(plot.fig)
        paths.append(str(path))
    return paths


def plot_regret_trajectories(
    iteration_logs_dir: Path,
    output_dir: Path,
    shortlist: list[str],
) -> list[str]:
    """Per-iteration regret curves: baseline vs jittered, mean ± 95 % CI.

    For each (objective, acquisition, error_model, jitter_std, jitter_iteration)
    combination a figure is produced showing how simple_regret_true evolves over
    BO iterations.  A shaded band represents the 95 % bootstrap CI across seeds.
    The vertical dashed line marks the first iteration where noise is applied.
    """
    files = sorted(iteration_logs_dir.glob("bo_sensor_error_*_seed*_*.csv"))
    if not files:
        return []

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "simple_regret_true" not in df.columns or "iteration" not in df.columns:
        return []

    df["baseline"] = df["error_model"] == "none"
    if shortlist:
        df = df[df["acquisition"].isin(shortlist)]

    paths: list[str] = []
    sns.set_theme(style="whitegrid")

    group_cols = ["objective", "error_model", "jitter_std", "jitter_iteration"]
    if "dataset" in df.columns:
        group_cols.insert(0, "dataset")

    for group_values, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        meta = dict(zip(group_cols, group_values))

        if meta["error_model"] == "none":
            continue

        jitter_iteration = int(meta["jitter_iteration"])
        acquisitions = sorted(group_df["acquisition"].dropna().unique().tolist())
        n_acq = len(acquisitions)
        if n_acq == 0:
            continue

        fig, axes = plt.subplots(
            1, n_acq,
            figsize=(4.5 * n_acq, 4.0),
            sharey=True,
            squeeze=False,
        )

        palette = sns.color_palette("Set2", 2)

        for col_idx, acq in enumerate(acquisitions):
            ax = axes[0, col_idx]
            acq_df = group_df[group_df["acquisition"] == acq]

            for is_baseline, label, color in [
                (True,  "Baseline (no noise)", palette[0]),
                (False, "Jittered",            palette[1]),
            ]:
                sub = acq_df[acq_df["baseline"] == is_baseline]
                if sub.empty:
                    continue
                stats = (
                    sub.groupby("iteration")["simple_regret_true"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                stats["se"] = stats["std"] / np.sqrt(stats["count"])
                # 95 % CI using t-distribution (df = count - 1)
                from scipy.stats import t as student_t_dist
                t_crit = student_t_dist.ppf(0.975, df=np.maximum(stats["count"] - 1, 1))
                stats["ci"] = t_crit * stats["se"]

                ax.plot(stats["iteration"], stats["mean"], label=label, color=color, linewidth=1.8)
                ax.fill_between(
                    stats["iteration"],
                    stats["mean"] - stats["ci"],
                    stats["mean"] + stats["ci"],
                    alpha=0.20,
                    color=color,
                )

            ax.axvline(x=jitter_iteration + 0.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_title(acq, fontsize=10)
            ax.set_xlabel("Iteration")
            if col_idx == 0:
                ax.set_ylabel("Simple regret (true)")
            if col_idx == n_acq - 1:
                ax.legend(fontsize=8)

        title_parts = [f"{k}={v}" for k, v in meta.items()]
        fig.suptitle("Regret trajectory  |  " + "  ·  ".join(title_parts), fontsize=9, y=1.01)
        fig.tight_layout()

        safe = lambda v: str(v).replace("/", "-").replace(".", "p")
        suffix = "_".join(safe(v) for v in group_values)
        path = output_dir / f"regret_trajectory_{suffix}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    return paths


def plot_effect_size_forest(
    evaluation_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Forest plot of Cohen's dz per acquisition function.

    Reads effect_sizes_cohens_dz.csv produced by plot_sensor_error_results.py and
    renders one forest plot per (objective, error_model, jitter_iteration) slice.
    Each row is one acquisition function; the horizontal bar is the 95 % CI.
    """
    es_path = evaluation_dir / "effect_sizes_cohens_dz.csv"
    if not es_path.exists():
        return []

    df = pd.read_csv(es_path)
    required = {"acquisition", "cohens_dz", "ci95_low", "ci95_high", "objective", "error_model", "jitter_iteration"}
    if not required.issubset(df.columns):
        return []

    paths: list[str] = []
    sns.set_theme(style="whitegrid")

    group_cols = ["objective", "error_model", "jitter_iteration"]
    if "dataset" in df.columns:
        group_cols.insert(0, "dataset")

    for group_values, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        meta = dict(zip(group_cols, group_values))

        plot_df = (
            group_df.groupby("acquisition")[["cohens_dz", "ci95_low", "ci95_high"]]
            .mean()
            .reset_index()
            .sort_values("cohens_dz")
        )
        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.0, max(2.5, 0.45 * len(plot_df))))
        y_pos = np.arange(len(plot_df))

        colors = ["#d62728" if v > 0 else "#2ca02c" for v in plot_df["cohens_dz"]]
        ax.barh(y_pos, plot_df["cohens_dz"], color=colors, alpha=0.75, height=0.5)
        ax.errorbar(
            plot_df["cohens_dz"],
            y_pos,
            xerr=[
                plot_df["cohens_dz"] - plot_df["ci95_low"],
                plot_df["ci95_high"] - plot_df["cohens_dz"],
            ],
            fmt="none",
            color="black",
            capsize=4,
            linewidth=1.2,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["acquisition"], fontsize=9)
        ax.set_xlabel("Cohen's dz  (jittered − baseline)")
        ax.set_title(
            "Effect sizes  |  " + "  ·  ".join(f"{k}={v}" for k, v in meta.items()),
            fontsize=9,
        )
        fig.tight_layout()

        safe = lambda v: str(v).replace("/", "-").replace(".", "p")
        suffix = "_".join(safe(v) for v in group_values)
        path = output_dir / f"forest_effect_sizes_{suffix}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    return paths


def write_manifest(output_dir: Path, paths: list[str], shortlist: list[str]) -> None:
    manifest = output_dir / "figure_manifest.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("Combined-aspects figure manifest\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Shortlist used for line plots: {', '.join(shortlist)}\n\n")
        for path in paths:
            handle.write(Path(path).name + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shortlist = [item.strip() for item in args.shortlist.split(",") if item.strip()]

    summary, rankings = load_inputs(args.evaluation_dir)
    paths: list[str] = []
    paths.extend(plot_robustness_heatmaps(summary, args.output_dir))
    paths.extend(plot_rank_progression(rankings, args.output_dir, shortlist))
    paths.extend(plot_winner_maps(rankings, args.output_dir))
    paths.extend(plot_response_scatter(summary, args.output_dir))
    logs_dir = args.iteration_logs_dir if args.iteration_logs_dir else args.evaluation_dir.parent
    paths.extend(plot_regret_trajectories(logs_dir, args.output_dir, shortlist))
    paths.extend(plot_effect_size_forest(args.evaluation_dir, args.output_dir))
    write_manifest(args.output_dir, paths, shortlist)
    print(f"Saved {len(paths)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
