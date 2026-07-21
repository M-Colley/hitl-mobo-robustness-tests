"""Interactive robustness results dashboard for CHI.

Run with:
    streamlit run scripts/dashboard.py -- --output-dir output

The dashboard lets you explore simulation results interactively:
  - Regret trajectory curves (baseline vs jittered)
  - Acquisition function rank heatmaps
  - Effect size (Cohen's dz) forest plots
  - Raw paired statistics table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import streamlit as st
except ImportError as exc:
    sys.stderr.write(
        "The dashboard requires streamlit, which is not installed.\n"
        "Install it with `pip install streamlit` (or `pip install .[dashboard]`).\n"
    )
    raise SystemExit(1) from exc

from scipy.stats import t as student_t_dist

# ---------------------------------------------------------------------------
# Argument parsing (streamlit passes args after "--")
# ---------------------------------------------------------------------------

def _parse_output_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    # streamlit forwards everything after "--" as sys.argv
    args, _ = parser.parse_known_args()
    return args.output_dir


OUTPUT_DIR = _parse_output_dir()
EVAL_DIR = OUTPUT_DIR / "evaluation"

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading iteration logs…")
def load_iteration_logs(output_dir: Path) -> pd.DataFrame | None:
    files = sorted(output_dir.glob("bo_sensor_error_*_seed*_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["baseline"] = df["error_model"] == "none"
    return df


@st.cache_data(show_spinner="Loading evaluation outputs…")
def load_eval_outputs(eval_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in ["condition_rankings", "condition_summary", "condition_tests"]:
        path = eval_dir / f"{name}.csv"
        if path.exists():
            out[name] = pd.read_csv(path)
    es_path = eval_dir / "effect_sizes_cohens_dz.csv"
    if es_path.exists():
        out["effect_sizes"] = pd.read_csv(es_path)
    pt_path = eval_dir / "final_outcome_paired_tests.csv"
    if pt_path.exists():
        out["paired_tests"] = pd.read_csv(pt_path)
    return out


# ---------------------------------------------------------------------------
# Page: Regret trajectories
# ---------------------------------------------------------------------------

def page_trajectories(logs: pd.DataFrame) -> None:
    st.header("Regret Trajectories")
    st.caption(
        "Mean simple regret (true) over BO iterations, averaged across seeds. "
        "Shaded band = 95 % CI.  Dashed line = first noise-affected iteration."
    )

    has_dataset = "dataset" in logs.columns
    cols = st.columns(5 if has_dataset else 4)
    col_offset = 0
    if has_dataset:
        datasets = sorted(logs["dataset"].dropna().unique())
        dataset = cols[0].selectbox("Dataset", datasets)
        logs = logs[logs["dataset"] == dataset]
        col_offset = 1

    objectives = sorted(logs["objective"].dropna().unique())
    objective = cols[col_offset].selectbox("Objective", objectives)

    error_models = sorted(logs.loc[logs["error_model"] != "none", "error_model"].dropna().unique())
    if not error_models:
        st.info("No jittered runs found.")
        return
    error_model = cols[col_offset + 1].selectbox("Error model", error_models)

    jitter_stds = sorted(logs.loc[logs["error_model"] != "none", "jitter_std"].dropna().unique())
    jitter_std = cols[col_offset + 2].selectbox("Jitter std", jitter_stds)

    jitter_iterations = sorted(
        logs.loc[logs["error_model"] != "none", "jitter_iteration"].dropna().astype(int).unique()
    )
    jitter_iteration = cols[col_offset + 3].selectbox("Jitter iteration", jitter_iterations)

    acquisitions = sorted(logs["acquisition"].dropna().unique())
    selected_acqs = st.multiselect("Acquisitions", acquisitions, default=acquisitions)
    if not selected_acqs:
        st.warning("Select at least one acquisition function.")
        return

    mask_jit = (
        (logs["objective"] == objective)
        & (logs["error_model"] == error_model)
        & (logs["jitter_std"] == jitter_std)
        & (logs["jitter_iteration"] == jitter_iteration)
        & (logs["acquisition"].isin(selected_acqs))
    )
    mask_base = (
        (logs["objective"] == objective)
        & (logs["baseline"])
        & (logs["acquisition"].isin(selected_acqs))
    )
    plot_df = pd.concat([logs[mask_jit], logs[mask_base]], ignore_index=True)

    if plot_df.empty:
        st.info("No data for the selected filters.")
        return

    n_cols = min(3, len(selected_acqs))
    n_rows = int(np.ceil(len(selected_acqs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharey=True, squeeze=False)
    palette = sns.color_palette("Set2", 2)
    sns.set_theme(style="whitegrid")

    for idx, acq in enumerate(selected_acqs):
        ax = axes[idx // n_cols][idx % n_cols]
        acq_df = plot_df[plot_df["acquisition"] == acq]

        for is_base, label, color in [(True, "Baseline", palette[0]), (False, "Jittered", palette[1])]:
            sub = acq_df[acq_df["baseline"] == is_base]
            if sub.empty:
                continue
            stats = (
                sub.groupby("iteration")["simple_regret_true"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            stats["ci"] = (
                student_t_dist.ppf(0.975, df=np.maximum(stats["count"] - 1, 1))
                * stats["std"] / np.sqrt(stats["count"])
            )
            ax.plot(stats["iteration"], stats["mean"], label=label, color=color, lw=1.8)
            ax.fill_between(
                stats["iteration"],
                stats["mean"] - stats["ci"],
                stats["mean"] + stats["ci"],
                alpha=0.18,
                color=color,
            )

        ax.axvline(x=int(jitter_iteration) + 0.5, color="grey", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(acq, fontsize=10)
        ax.set_xlabel("Iteration")
        if idx % n_cols == 0:
            ax.set_ylabel("Simple regret (true)")
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(len(selected_acqs), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: Rank heatmap
# ---------------------------------------------------------------------------

def page_rank_heatmap(eval_outputs: dict[str, pd.DataFrame]) -> None:
    st.header("Acquisition Function Rankings")
    rankings = eval_outputs.get("condition_rankings")
    if rankings is None or rankings.empty:
        st.info("condition_rankings.csv not found in evaluation directory.")
        return

    objectives = sorted(rankings["objective"].dropna().unique())
    objective = st.selectbox("Objective", objectives)

    datasets = sorted(rankings["dataset"].dropna().unique()) if "dataset" in rankings.columns else ["all"]
    dataset = st.selectbox("Dataset", datasets) if len(datasets) > 1 else datasets[0]

    error_models = sorted(rankings["error_model"].dropna().unique())
    error_model = st.selectbox("Error model", error_models)

    sub = rankings[(rankings["objective"] == objective) & (rankings["error_model"] == error_model)]
    if "dataset" in rankings.columns:
        sub = sub[sub["dataset"] == dataset]

    if sub.empty:
        st.info("No data for the selected filters.")
        return

    jitter_iterations = sorted(sub["jitter_iteration"].dropna().astype(int).unique())
    n = len(jitter_iterations)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 7), squeeze=False)
    sns.set_theme(style="whitegrid")

    acq_order = sub.groupby("acquisition")["mean_rank"].mean().sort_values().index.tolist()

    for idx, jit_iter in enumerate(jitter_iterations):
        pivot = (
            sub[sub["jitter_iteration"] == jit_iter]
            .pivot_table(index="acquisition", columns="jitter_std", values="mean_rank", aggfunc="mean")
            .reindex(acq_order)
        )
        sns.heatmap(
            pivot,
            ax=axes[0, idx],
            annot=True,
            fmt=".2f",
            cmap="YlGnBu_r",
            cbar=idx == n - 1,
            cbar_kws={"label": "Mean rank"},
            mask=pivot.isna(),
        )
        axes[0, idx].set_title(f"jitter={jit_iter}")
        axes[0, idx].set_xlabel("Jitter std")
        axes[0, idx].set_ylabel("Acquisition" if idx == 0 else "")

    fig.suptitle(f"{dataset} · {objective} · {error_model}", y=1.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: Effect size forest plot
# ---------------------------------------------------------------------------

def page_forest_plot(eval_outputs: dict[str, pd.DataFrame]) -> None:
    st.header("Effect Sizes (Cohen's dz)")
    st.caption(
        "Cohen's dz = mean(jittered − baseline) / std(differences).  "
        "Positive = jittered is worse.  Error bars = 95 % CI."
    )

    es = eval_outputs.get("effect_sizes")
    if es is None or es.empty:
        st.info("effect_sizes_cohens_dz.csv not found in evaluation directory.")
        return

    has_dataset = "dataset" in es.columns
    cols = st.columns(5 if has_dataset else 4)
    col_offset = 0
    if has_dataset:
        datasets = sorted(es["dataset"].dropna().unique())
        dataset = cols[0].selectbox("Dataset", datasets)
        es = es[es["dataset"] == dataset]
        col_offset = 1

    objectives = sorted(es["objective"].dropna().unique())
    objective = cols[col_offset].selectbox("Objective", objectives)

    error_models = sorted(es["error_model"].dropna().unique())
    error_model = cols[col_offset + 1].selectbox("Error model", error_models)

    jitter_iterations = sorted(es["jitter_iteration"].dropna().astype(int).unique())
    jit_iter = cols[col_offset + 2].selectbox("Jitter iteration", jitter_iterations)

    # jitter_std is part of the condition: averaging dz (or CI endpoints)
    # across noise levels is meaningless, so it must be filtered, not pooled.
    jitter_stds = sorted(es["jitter_std"].dropna().unique())
    jit_std = cols[col_offset + 3].selectbox("Jitter std", jitter_stds)

    sub = es[
        (es["objective"] == objective)
        & (es["error_model"] == error_model)
        & (es["jitter_iteration"] == jit_iter)
        & (es["jitter_std"] == jit_std)
    ]
    if sub.empty:
        st.info("No data for the selected filters.")
        return

    # Prefer the exact noncentral-t dz CIs from the evaluation step; the old
    # rescaled mean-diff CI is not a valid dz CI and is kept only as a
    # legacy-output fallback.
    sub = sub.copy()
    if not {"dz_ci95_low", "dz_ci95_high"}.issubset(sub.columns):
        st.caption("Legacy evaluation output: dz CIs are approximate (rescaled mean-diff CI).")
        if "std_diff" in sub.columns:
            scale = sub["std_diff"].where(sub["std_diff"] > 0)
            sub["dz_ci95_low"] = sub["ci95_low"] / scale
            sub["dz_ci95_high"] = sub["ci95_high"] / scale
        else:
            sub["dz_ci95_low"] = sub["ci95_low"]
            sub["dz_ci95_high"] = sub["ci95_high"]

    plot_df = (
        sub[["acquisition", "cohens_dz", "dz_ci95_low", "dz_ci95_high"]]
        .dropna(subset=["cohens_dz"])
        .sort_values("cohens_dz")
    )

    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.5 * len(plot_df))))
    y_pos = np.arange(len(plot_df))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in plot_df["cohens_dz"]]
    ax.barh(y_pos, plot_df["cohens_dz"], color=colors, alpha=0.75, height=0.5)
    ax.errorbar(
        plot_df["cohens_dz"],
        y_pos,
        xerr=[
            plot_df["cohens_dz"] - plot_df["dz_ci95_low"],
            plot_df["dz_ci95_high"] - plot_df["cohens_dz"],
        ],
        fmt="none",
        color="black",
        capsize=4,
        lw=1.2,
    )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["acquisition"], fontsize=9)
    ax.set_xlabel("Cohen's dz  (jittered − baseline)")
    ax.set_title(f"{objective}  ·  {error_model}  ·  jitter={jit_iter}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: Paired statistics table
# ---------------------------------------------------------------------------

def page_stats_table(eval_outputs: dict[str, pd.DataFrame]) -> None:
    st.header("Paired Statistical Tests")
    pt = eval_outputs.get("paired_tests")
    if pt is None or pt.empty:
        st.info("final_outcome_paired_tests.csv not found in evaluation directory.")
        return

    metrics = sorted(pt["metric"].dropna().unique())
    metric = st.selectbox("Metric", metrics)

    sub = pt[pt["metric"] == metric].copy()

    display_cols = [
        c for c in [
            "dataset", "objective", "acquisition", "error_model",
            "jitter_std", "jitter_iteration", "oracle_model",
            "n_pairs", "mean_diff", "cohens_dz",
            "p_value_t", "p_value_t_fdr_bh",
            "p_value_wilcoxon", "p_value_wilcoxon_fdr_bh",
        ]
        if c in sub.columns
    ]
    sub = sub[display_cols].reset_index(drop=True)

    def _highlight_sig(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        fdr_col = "p_value_t_fdr_bh"
        if fdr_col in row.index:
            val = row[fdr_col]
            if pd.notna(val) and val < 0.05:
                idx = row.index.get_loc(fdr_col)
                styles[idx] = "background-color: #ffe0e0"
        return styles

    st.dataframe(sub.style.apply(_highlight_sig, axis=1), use_container_width=True, height=480)
    st.caption("Rows highlighted in red have FDR-adjusted p < 0.05 (t-test).")


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="HITL-MOBO Robustness Explorer",
        page_icon="🔬",
        layout="wide",
    )

    st.title("HITL-MOBO Robustness Results")
    st.caption(
        f"Results loaded from `{OUTPUT_DIR.resolve()}`.  "
        "Use the sidebar to navigate between views."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "View",
            [
                "Regret Trajectories",
                "Rank Heatmaps",
                "Effect Sizes",
                "Paired Statistics",
            ],
        )
        st.divider()
        st.caption(f"**Output dir:** `{OUTPUT_DIR}`")
        st.caption(f"**Eval dir:** `{EVAL_DIR}`")

    logs = load_iteration_logs(OUTPUT_DIR)
    eval_outputs = load_eval_outputs(EVAL_DIR)

    if page == "Regret Trajectories":
        if logs is None:
            st.warning(
                f"No per-iteration CSV logs found in `{OUTPUT_DIR}`.  "
                "Run the simulation first (`bo_sensor_error_simulation.py`)."
            )
        else:
            page_trajectories(logs)

    elif page == "Rank Heatmaps":
        page_rank_heatmap(eval_outputs)

    elif page == "Effect Sizes":
        page_forest_plot(eval_outputs)

    elif page == "Paired Statistics":
        page_stats_table(eval_outputs)


if __name__ == "__main__":
    main()
