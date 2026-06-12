"""Evaluate robustness results against the research question.

This script focuses on the question:

    Which parts work how well for which dataset and which error level?

It uses the existing per-iteration logs to:

1. Recompute the first actually error-affected response step.
2. Pair each jittered run with its baseline counterpart (same support enforced).
3. Rank acquisition functions within each dataset / objective / error condition,
   on both excess regret (robustness) and absolute noisy performance
   (the deployment question).
4. Run rank-based tests with per-family FDR correction and report paired
   effect sizes (Cohen's dz) with 95% CIs.
5. Write compact heatmaps for mean rank and excess regret.

Notes on metrics:

- ``auc_simple_regret_excess_true`` integrates the whole run, so it is only
  comparable across conditions with the same jitter onset. For any
  across-onset statement use ``auc_simple_regret_excess_true_postonset_per_iter``
  (post-onset window, normalized per noisy iteration).
- ``auc_simple_regret_true_jitter`` is the absolute noisy-run AUC: an
  acquisition that is uniformly bad has excess ~0 but a poor absolute value,
  so report both.
- ``*_inference_*`` metrics score the incumbent the experimenter would pick
  from OBSERVED (noisy) data by its TRUE value: the deployment-relevant
  recommendation quality.
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, ttest_1samp, wilcoxon
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests


PRIMARY_METRIC = "auc_simple_regret_excess_true"
POSTONSET_METRIC = "auc_simple_regret_excess_true_postonset_per_iter"
ABSOLUTE_METRIC = "auc_simple_regret_true_jitter"
SECONDARY_METRIC = "final_simple_regret_excess_true"
INFERENCE_METRIC = "final_inference_simple_regret_excess_true"
REACTION_METRIC = "response_l2_excess"

BOOTSTRAP_SEED = 12345
BOOTSTRAP_SAMPLES = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation"))
    parser.add_argument(
        "--allow-mixed-iterations",
        action="store_true",
        default=False,
        help="Allow per-iteration logs with different run lengths in the input dir. "
        "By default this is an error because stale runs silently corrupt pairing.",
    )
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
    # Preferred: the simulator writes the authoritative parameter-column list
    # into each log. Fall back to schema inference only for legacy logs.
    if "param_columns" in run_df.columns:
        raw = run_df["param_columns"].iloc[0]
        if isinstance(raw, str) and raw:
            columns = [col for col in raw.split(",") if col]
            missing = [col for col in columns if col not in run_df.columns]
            if missing:
                raise ValueError(
                    f"param_columns metadata names columns missing from the log: {missing}"
                )
            return columns

    reserved = {
        "iteration",
        "objective_true",
        "objective_observed",
        "error_applied",
        "error_magnitude",
        "error_magnitude_l2",
        "acquisition",
        "fit_time_sec",
        "acq_opt_failed",
        "seed",
        "run_id",
        "error_model",
        "jitter_std",
        "jitter_iteration",
        "oracle_model",
        "objective",
        "param_columns",
        "y_opt",
        "best_true_so_far",
        "regret_inst_true",
        "regret_cum_true",
        "simple_regret_true",
        "regret_avg_true",
        "inference_value_true",
        "inference_simple_regret_true",
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


def _auc(values: np.ndarray) -> float:
    return float(np.trapezoid(values, dx=1.0))


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

        simple_regret = run_df["simple_regret_true"].to_numpy(dtype=float)
        has_inference = "inference_simple_regret_true" in run_df.columns
        inference_regret = (
            run_df["inference_simple_regret_true"].to_numpy(dtype=float)
            if has_inference
            else None
        )
        acq_failures = (
            int(run_df["acq_opt_failed"].sum()) if "acq_opt_failed" in run_df.columns else 0
        )

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

            # Post-onset window: iterations jitter_iteration+1 .. max_iter.
            # Whole-run AUC mechanically shrinks for late onsets; the per-noisy-
            # iteration normalization makes values comparable across onsets.
            postonset_mask = run_df["iteration"].to_numpy() >= start_iter
            n_noisy_iters = int(postonset_mask.sum())
            auc_postonset = _auc(simple_regret[postonset_mask]) if n_noisy_iters > 1 else 0.0

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
                "n_iterations": max_iter,
                "acq_opt_failures": acq_failures,
                "response_start_iteration": int(start_iter),
                "response_end_iteration": int(end_iter),
                "response_l2": response_l2,
                "final_best_true": float(base_row["best_true_so_far"]),
                "final_simple_regret_true": float(base_row["simple_regret_true"]),
                "final_cum_regret_true": float(base_row["regret_cum_true"]),
                "final_avg_regret_true": float(base_row["regret_avg_true"]),
                "auc_simple_regret_true": _auc(simple_regret),
                "auc_simple_regret_true_postonset_per_iter": (
                    auc_postonset / max(1, n_noisy_iters - 1)
                ),
                "final_inference_simple_regret_true": (
                    float(inference_regret[-1]) if has_inference else float("nan")
                ),
                "auc_inference_simple_regret_true": (
                    _auc(inference_regret) if has_inference else float("nan")
                ),
                "param_columns": ",".join(param_columns),
            }
            for column in param_columns:
                row[f"response_{column}"] = float(response[column])
            rows.append(row)

    return pd.DataFrame(rows)


def build_paired_table(response_df: pd.DataFrame) -> pd.DataFrame:
    # n_iterations is a pairing key: pairing a 100-iteration baseline with a
    # 50-iteration jittered run would produce a meaningless excess AUC.
    base_keys = [
        "dataset",
        "objective",
        "acquisition",
        "seed",
        "oracle_model",
        "jitter_iteration",
        "n_iterations",
    ]
    jittered = response_df[~response_df["baseline"]].copy()
    baseline = response_df[response_df["baseline"]].copy()

    metric_columns = [
        "response_l2",
        "final_best_true",
        "final_simple_regret_true",
        "final_cum_regret_true",
        "final_avg_regret_true",
        "auc_simple_regret_true",
        "auc_simple_regret_true_postonset_per_iter",
        "final_inference_simple_regret_true",
        "auc_inference_simple_regret_true",
    ]
    dedup_baseline = baseline.drop_duplicates(subset=base_keys, keep="first")
    if len(dedup_baseline) != len(baseline):
        raise ValueError(
            "Duplicate baseline runs found for identical pairing keys. The input "
            "directory likely mixes runs from different invocations; clean it or "
            "separate the output directories."
        )
    paired = jittered.merge(
        dedup_baseline[base_keys + metric_columns],
        on=base_keys,
        how="inner",
        suffixes=("_jitter", "_baseline"),
        validate="many_to_one",
    )

    if paired.empty:
        raise ValueError("No baseline/jitter pairs could be formed from the run logs.")
    n_unpaired = len(jittered) - len(paired)
    if n_unpaired > 0:
        print(
            f"WARNING: {n_unpaired} jittered run(s) had no matching baseline "
            "(same dataset/objective/acquisition/seed/oracle/onset/run-length) "
            "and were dropped from the paired analysis.",
            file=sys.stderr,
        )

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
    paired["auc_simple_regret_excess_true_postonset_per_iter"] = (
        paired["auc_simple_regret_true_postonset_per_iter_jitter"]
        - paired["auc_simple_regret_true_postonset_per_iter_baseline"]
    )
    paired["final_inference_simple_regret_excess_true"] = (
        paired["final_inference_simple_regret_true_jitter"]
        - paired["final_inference_simple_regret_true_baseline"]
    )
    paired["auc_inference_simple_regret_excess_true"] = (
        paired["auc_inference_simple_regret_true_jitter"]
        - paired["auc_inference_simple_regret_true_baseline"]
    )
    return paired


def _bootstrap_ci_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    means = np.mean(
        rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True), axis=1
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


CONDITION_COLS = [
    "dataset",
    "objective",
    "oracle_model",
    "error_model",
    "jitter_iteration",
    "jitter_std",
]


def summarize_conditions(paired: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for keys, group in paired.groupby(CONDITION_COLS + ["acquisition"], dropna=False):
        condition = dict(zip(CONDITION_COLS + ["acquisition"], keys))
        primary = group[PRIMARY_METRIC].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci_mean(primary, rng)
        rows.append(
            {
                **condition,
                "n_seeds": int(group["seed"].nunique()),
                "mean_response_l2_excess": float(group[REACTION_METRIC].mean()),
                "median_response_l2_excess": float(group[REACTION_METRIC].median()),
                "mean_final_simple_regret_excess_true": float(group[SECONDARY_METRIC].mean()),
                "median_final_simple_regret_excess_true": float(group[SECONDARY_METRIC].median()),
                "mean_auc_simple_regret_excess_true": float(np.mean(primary)),
                "median_auc_simple_regret_excess_true": float(np.median(primary)),
                "std_auc_simple_regret_excess_true": float(np.std(primary, ddof=1))
                if len(primary) > 1
                else float("nan"),
                "ci95_low_auc_simple_regret_excess_true": ci_low,
                "ci95_high_auc_simple_regret_excess_true": ci_high,
                "mean_auc_excess_postonset_per_iter": float(group[POSTONSET_METRIC].mean()),
                "mean_auc_simple_regret_true_jitter": float(group[ABSOLUTE_METRIC].mean()),
                "mean_final_inference_simple_regret_excess_true": float(
                    group[INFERENCE_METRIC].mean()
                ),
                "acq_opt_failures": int(group["acq_opt_failures"].sum())
                if "acq_opt_failures" in group.columns
                else 0,
            }
        )
    return pd.DataFrame(rows)


def _assert_unique_cells(group: pd.DataFrame, condition: dict) -> None:
    cell_sizes = group.groupby(["seed", "acquisition"], dropna=False).size()
    if (cell_sizes > 1).any():
        raise ValueError(
            "Multiple paired rows per (seed, acquisition) cell in condition "
            f"{condition}. This indicates duplicated or mixed runs in the input "
            "directory; refusing to silently average them."
        )


def compute_condition_rankings(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for condition_values, group in paired.groupby(CONDITION_COLS, dropna=False):
        if not isinstance(condition_values, tuple):
            condition_values = (condition_values,)
        condition = dict(zip(CONDITION_COLS, condition_values))
        _assert_unique_cells(group, condition)

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
        n_acq = int(pivot.shape[1])

        # Absolute noisy-run performance answers the deployment question
        # ("which method gives the best design under noise"); excess regret
        # alone can crown uniformly bad methods.
        absolute_pivot = (
            group.pivot_table(index="seed", columns="acquisition", values=ABSOLUTE_METRIC, aggfunc="mean")
            .reindex(index=pivot.index, columns=pivot.columns)
        )
        absolute_ranks = absolute_pivot.rank(axis=1, method="average", ascending=True).mean(axis=0)

        metric_means = pivot.mean(axis=0)
        metric_medians = pivot.median(axis=0)
        postonset_means = (
            group.pivot_table(index="seed", columns="acquisition", values=POSTONSET_METRIC, aggfunc="mean")
            .reindex(index=pivot.index, columns=pivot.columns)
            .mean(axis=0)
        )
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
                    "n_acquisitions": n_acq,
                    "mean_rank": float(mean_ranks[acquisition]),
                    # Normalized to [0, 1] so ranks can be averaged across
                    # conditions with different acquisition counts.
                    "mean_rank_normalized": float((mean_ranks[acquisition] - 1.0) / max(1, n_acq - 1)),
                    "mean_rank_absolute": float(absolute_ranks[acquisition]),
                    "mean_auc_simple_regret_excess_true": float(metric_means[acquisition]),
                    "median_auc_simple_regret_excess_true": float(metric_medians[acquisition]),
                    "mean_auc_excess_postonset_per_iter": float(postonset_means[acquisition]),
                    "mean_auc_simple_regret_true_jitter": float(absolute_pivot.mean(axis=0)[acquisition]),
                    "mean_response_l2_excess": float(response_means[acquisition]),
                    "condition_win": math.isclose(float(mean_ranks[acquisition]), best_mean_rank, rel_tol=1e-12),
                }
            )

        row: dict[str, object] = {
            **condition,
            "n_seeds": int(pivot.shape[0]),
            "n_acquisitions": n_acq,
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
                    "min_attainable_p": float("nan"),
                    "underpowered": bool(pivot.shape[0] < 10),
                }
            )
        else:
            diff = pivot.iloc[:, 0] - pivot.iloc[:, 1]
            n = int(len(diff))
            # Exact two-sided Wilcoxon: the smallest attainable p is 2/2^n.
            min_attainable_p = 2.0 / (2.0**n) if n > 0 else float("nan")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
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
                    "min_attainable_p": min_attainable_p,
                    "underpowered": bool(min_attainable_p > 0.05),
                }
            )
        test_rows.append(row)

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        # FDR correction is applied per test family: omnibus Friedman tests and
        # pairwise Wilcoxon tests answer different hypotheses and must not share
        # one correction pool.
        tests["p_value_fdr_bh"] = np.nan
        tests["p_value_rejected"] = False
        for _, family_idx in tests.groupby("test_name").groups.items():
            family = tests.loc[family_idx]
            valid_mask = family["p_value"].notna()
            if valid_mask.any():
                reject, p_adj, _, _ = multipletests(
                    family.loc[valid_mask, "p_value"], method="fdr_bh"
                )
                tests.loc[family.index[valid_mask], "p_value_fdr_bh"] = p_adj
                tests.loc[family.index[valid_mask], "p_value_rejected"] = reject

    return pd.DataFrame(ranking_rows), tests


def compute_overall_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Pool conditions per dataset/objective.

    Pooled values use the normalized rank ([0, 1] within condition) and the
    per-noisy-iteration post-onset excess AUC, which are commensurate across
    error conditions; the raw whole-run excess AUC is not pooled because its
    magnitude scales with the onset window and noise level.
    """
    if rankings.empty:
        return pd.DataFrame()
    overall = (
        rankings.groupby(["dataset", "objective", "acquisition"], dropna=False)
        .agg(
            mean_rank_normalized=("mean_rank_normalized", "mean"),
            mean_rank_absolute=("mean_rank_absolute", "mean"),
            mean_auc_excess_postonset_per_iter=("mean_auc_excess_postonset_per_iter", "mean"),
            mean_auc_simple_regret_true_jitter=("mean_auc_simple_regret_true_jitter", "mean"),
            mean_response_l2_excess=("mean_response_l2_excess", "mean"),
            condition_wins=("condition_win", "sum"),
            conditions=("condition_win", "size"),
        )
        .reset_index()
        .sort_values(["dataset", "objective", "mean_rank_normalized", "mean_auc_excess_postonset_per_iter"])
    )
    return overall


def compute_effect_sizes(paired: pd.DataFrame) -> pd.DataFrame:
    """Paired baseline-vs-jittered effect sizes (Cohen's dz) per condition.

    Written to the evaluation dir so plot_combined_aspects.plot_effect_size_forest
    and the dashboard find it in the documented workflow.
    """
    rows: list[dict[str, object]] = []
    for keys, group in paired.groupby(CONDITION_COLS + ["acquisition"], dropna=False):
        condition = dict(zip(CONDITION_COLS + ["acquisition"], keys))
        diffs = group[PRIMARY_METRIC].to_numpy(dtype=float)
        n = int(len(diffs))
        row: dict[str, object] = {
            **condition,
            "metric": PRIMARY_METRIC,
            "n_pairs": n,
            "mean_diff": float(np.mean(diffs)) if n else float("nan"),
            "median_diff": float(np.median(diffs)) if n else float("nan"),
            "std_diff": float(np.std(diffs, ddof=1)) if n > 1 else float("nan"),
            "cohens_dz": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "t_stat": float("nan"),
            "p_value_t": float("nan"),
            "wilcoxon_stat": float("nan"),
            "p_value_wilcoxon": float("nan"),
        }
        std_diff = row["std_diff"]
        if n >= 2 and isinstance(std_diff, float) and std_diff > 0:
            row["cohens_dz"] = float(np.mean(diffs) / std_diff)
            t_result = ttest_1samp(diffs, popmean=0.0)
            row["t_stat"] = float(t_result.statistic)
            row["p_value_t"] = float(t_result.pvalue)
            sem = std_diff / math.sqrt(n)
            margin = float(student_t.ppf(0.975, df=n - 1)) * sem
            row["ci95_low"] = float(np.mean(diffs) - margin)
            row["ci95_high"] = float(np.mean(diffs) + margin)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    w_result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
                row["wilcoxon_stat"] = float(w_result.statistic)
                row["p_value_wilcoxon"] = float(w_result.pvalue)
            except ValueError:
                pass
        rows.append(row)

    effect_sizes = pd.DataFrame(rows)
    if not effect_sizes.empty:
        valid = effect_sizes["p_value_t"].notna()
        effect_sizes["p_value_t_fdr_bh"] = np.nan
        if valid.any():
            _, p_adj, _, _ = multipletests(effect_sizes.loc[valid, "p_value_t"], method="fdr_bh")
            effect_sizes.loc[valid, "p_value_t_fdr_bh"] = p_adj
    return effect_sizes


def plot_condition_heatmaps(rankings: pd.DataFrame, output_dir: Path) -> None:
    if rankings.empty:
        return

    sns.set_theme(style="whitegrid")
    acquisition_order = (
        rankings.groupby("acquisition")["mean_rank"].mean().sort_values().index.tolist()
    )

    for (dataset, objective, oracle_model, error_model), group in rankings.groupby(
        ["dataset", "objective", "oracle_model", "error_model"],
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

        slice_name = f"{dataset}_{objective}_{oracle_model}_{error_model}"
        fig_rank.suptitle(
            f"Acquisition mean ranks: {dataset} / {objective} / {oracle_model} / {error_model}",
            y=0.98,
        )
        fig_rank.tight_layout()
        fig_rank.savefig(
            output_dir / f"mean_rank_{slice_name}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_rank)

        fig_auc.suptitle(
            f"Excess AUC simple regret: {dataset} / {objective} / {oracle_model} / {error_model}",
            y=0.98,
        )
        fig_auc.tight_layout()
        fig_auc.savefig(
            output_dir / f"mean_excess_auc_{slice_name}.png",
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
        handle.write("- Report BOTH excess regret (robustness) and absolute noisy-run regret\n")
        handle.write("  (deployment quality): excess regret alone can crown acquisitions that\n")
        handle.write("  are uniformly bad in baseline and noisy conditions alike.\n")
        handle.write("- Lower excess AUC simple regret is better (within one condition).\n")
        handle.write("- For ANY comparison across jitter onsets use the post-onset,\n")
        handle.write("  per-noisy-iteration excess AUC; whole-run AUC scales with the window.\n")
        handle.write("- Inference regret (the *_inference_* metrics) scores the design the\n")
        handle.write("  experimenter would actually pick from noisy observations; it is the\n")
        handle.write("  deployment-relevant recommendation quality.\n")
        handle.write("- Response-step excess L2 is descriptive only.\n\n")

        handle.write("Data coverage\n")
        handle.write(f"- Response rows: {len(response_df)}\n")
        handle.write(f"- Paired baseline/jitter rows: {len(paired)}\n")
        if not paired.empty:
            seed_counts = paired.groupby(CONDITION_COLS)["seed"].nunique()
            handle.write(
                f"- Seeds per condition: min={int(seed_counts.min())}, "
                f"median={float(seed_counts.median()):.1f}, max={int(seed_counts.max())}\n"
            )
            if "acq_opt_failures" in paired.columns:
                n_failures = int(paired["acq_opt_failures"].sum())
                handle.write(f"- Acquisition-optimization fallbacks in jittered runs: {n_failures}\n")
        handle.write("\n")

        handle.write("Statistical notes\n")
        if not paired.empty:
            min_seeds = int(paired.groupby(CONDITION_COLS)["seed"].nunique().min())
            if min_seeds < 6:
                handle.write(
                    f"- WARNING: only {min_seeds} seeds in the smallest condition. Two-method\n"
                    f"  Wilcoxon cells cannot reach p<0.05 below n=6 (min attainable p at n=5\n"
                    f"  is 0.0625); treat per-condition p-values as descriptive and rely on\n"
                    f"  effect sizes, CIs and replication across conditions instead.\n"
                )
        handle.write("- FDR (Benjamini-Hochberg) correction is applied per test family\n")
        handle.write("  (Friedman omnibus and Wilcoxon pairwise are corrected separately).\n")
        handle.write("- Kendall's W accompanies every Friedman test as the effect size.\n")
        handle.write("- Dataset comparisons should stay stratified by dataset rather than pooling\n")
        handle.write("  unless outcomes are explicitly normalized across datasets.\n\n")

        if not tests.empty:
            handle.write("Condition-level omnibus tests\n")
            keep_cols = [
                "dataset",
                "objective",
                "oracle_model",
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
                "min_attainable_p",
                "underpowered",
                "kendalls_w",
            ]
            keep_cols = [col for col in keep_cols if col in tests.columns]
            handle.write(tests[keep_cols].to_string(index=False))
            handle.write("\n\n")

        if not overall.empty:
            handle.write("Overall acquisition ranking by dataset/objective\n")
            handle.write("(normalized ranks and per-noisy-iteration excess AUC; pooled across\n")
            handle.write("error conditions -- treat as a secondary, descriptive aggregate)\n\n")
            for (dataset, objective), group in overall.groupby(["dataset", "objective"], dropna=False):
                handle.write(f"{dataset} / {objective}\n")
                handle.write(
                    group.head(5)[
                        [
                            "acquisition",
                            "mean_rank_normalized",
                            "mean_rank_absolute",
                            "mean_auc_excess_postonset_per_iter",
                            "mean_auc_simple_regret_true_jitter",
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

    run_lengths = sorted(response_df["n_iterations"].unique().tolist())
    if len(run_lengths) > 1 and not args.allow_mixed_iterations:
        raise ValueError(
            f"Input dir contains runs with different lengths {run_lengths}. This "
            "usually means stale results from a previous invocation are mixed in. "
            "Clean the directory, or pass --allow-mixed-iterations to pair runs "
            "within each run length."
        )

    paired = build_paired_table(response_df)
    condition_summary = summarize_conditions(paired)
    rankings, tests = compute_condition_rankings(paired)
    overall = compute_overall_rankings(rankings)
    effect_sizes = compute_effect_sizes(paired)

    response_df.to_csv(args.output_dir / "response_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_excess_metrics.csv", index=False)
    condition_summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
    rankings.to_csv(args.output_dir / "condition_rankings.csv", index=False)
    tests.to_csv(args.output_dir / "condition_tests.csv", index=False)
    overall.to_csv(args.output_dir / "overall_rankings.csv", index=False)
    effect_sizes.to_csv(args.output_dir / "effect_sizes_cohens_dz.csv", index=False)
    # Same content under the name the dashboard's paired-statistics page expects.
    effect_sizes.to_csv(args.output_dir / "final_outcome_paired_tests.csv", index=False)

    plot_condition_heatmaps(rankings, args.output_dir)
    write_report(args.output_dir, response_df, paired, rankings, tests, overall)
    print(f"Evaluation outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
