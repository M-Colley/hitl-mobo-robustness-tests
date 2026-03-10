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
    write_manifest(args.output_dir, paths, shortlist)
    print(f"Saved {len(paths)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
