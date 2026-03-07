from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_select_best_oracle_model_parses_all_including_catboost() -> None:
    select_mod = _load_module("select_best_oracle_model_mod", SCRIPTS_DIR / "select_best_oracle_model.py")
    models = select_mod.parse_oracle_models("all")
    assert "catboost" in models
    assert "xgboost" in models


def test_select_best_oracle_model_rejects_unknown_model() -> None:
    select_mod = _load_module("select_best_oracle_model_mod_unknown", SCRIPTS_DIR / "select_best_oracle_model.py")
    with pytest.raises(ValueError, match="Unknown oracle model"):
        select_mod.parse_oracle_models("unknown_model")


def test_simulation_parse_args_preserves_remote_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    sim_mod = _load_module("bo_sensor_error_simulation_args_mod", SCRIPTS_DIR / "bo_sensor_error_simulation.py")
    remote_url = "https://github.com/M-Colley/ehmi-optimization-chi25-data"
    monkeypatch.setattr(
        sys,
        "argv",
        ["bo_sensor_error_simulation.py", "--data-dir", remote_url],
    )
    args = sim_mod.parse_args()
    assert args.data_dir == remote_url


def test_select_best_oracle_model_parse_args_preserves_remote_data_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_mod = _load_module("select_best_oracle_model_args_mod", SCRIPTS_DIR / "select_best_oracle_model.py")
    remote_url = "https://github.com/M-Colley/opticarvis-data"
    monkeypatch.setattr(
        sys,
        "argv",
        ["select_best_oracle_model.py", "--data-dir", remote_url],
    )
    args = select_mod.parse_args()
    assert args.data_dir == remote_url


def test_simulation_parse_args_preserves_oracle_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    sim_mod = _load_module("bo_sensor_error_simulation_auto_args_mod", SCRIPTS_DIR / "bo_sensor_error_simulation.py")
    monkeypatch.setattr(
        sys,
        "argv",
        ["bo_sensor_error_simulation.py", "--oracle-model", "auto"],
    )
    args = sim_mod.parse_args()
    assert args.oracle_model == "auto"
    assert args.oracle_models is None


def test_select_best_oracle_model_uses_grouped_validation_for_user_ids() -> None:
    select_mod = _load_module("select_best_oracle_group_validation_mod", SCRIPTS_DIR / "select_best_oracle_model.py")
    df = pd.DataFrame(
        {
            "p1": [0.0, 0.1, 0.9, 1.0, 0.2, 0.3],
            "p2": [0.2, 0.3, 0.8, 0.7, 0.4, 0.5],
            "score": [0.1, 0.15, 0.9, 0.95, 0.2, 0.25],
            "User_ID": [1, 1, 2, 2, 3, 3],
        }
    )

    results = select_mod.evaluate_models_for_objective(
        df=df,
        objective="score",
        objective_columns=["score"],
        param_columns=["p1", "p2"],
        models=["extra_trees"],
        seed=7,
        cv_folds=5,
        normalize=False,
        weights=None,
        tree_scale=0.1,
        progress_desc=None,
    )

    assert results["validation"]["strategy"] == "group_kfold"
    assert results["validation"]["group_source"] == "User_ID"
    assert results["validation"]["effective_cv_folds"] == 3
    assert "extra_trees" in results["scores"]
    assert "extra_trees" in results["rmse"]


def test_plot_loader_and_final_row_summary(tmp_path: Path) -> None:
    plot_mod = _load_module("plot_sensor_error_results_mod", SCRIPTS_DIR / "plot_sensor_error_results.py")
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r2", "r2"],
            "iteration": [1, 2, 1, 2],
            "error_model": ["none", "none", "gaussian", "gaussian"],
        }
    )
    frame.to_csv(input_dir / "bo_sensor_error_demo_seed7_baseline.csv", index=False)

    loaded = plot_mod.load_iteration_logs(input_dir)
    assert loaded.shape[0] == 4

    final_rows = plot_mod.summarize_final_outcomes(loaded)
    assert final_rows.shape[0] == 2
    assert set(final_rows["baseline"].unique()) == {True, False}


def test_plot_analysis_builds_paired_outputs(tmp_path: Path) -> None:
    plot_mod = _load_module("plot_sensor_error_results_mod_analysis", SCRIPTS_DIR / "plot_sensor_error_results.py")
    final_df = pd.DataFrame(
        {
            "dataset": ["demo", "demo", "demo", "demo"],
            "objective": ["composite", "composite", "composite", "composite"],
            "acquisition": ["ei", "ei", "ei", "ei"],
            "seed": [1, 1, 2, 2],
            "oracle_model": ["xgboost", "xgboost", "xgboost", "xgboost"],
            "error_model": ["none", "gaussian", "none", "gaussian"],
            "jitter_std": [0.0, 0.1, 0.0, 0.1],
            "jitter_iteration": [20, 20, 20, 20],
            "objective_true": [1.0, 1.3, 2.0, 1.8],
            "objective_observed": [1.0, 1.4, 2.0, 1.7],
            "simple_regret_true": [0.5, 0.4, 0.6, 0.7],
            "regret_cum_true": [1.0, 1.2, 1.5, 1.7],
            "regret_avg_true": [0.5, 0.6, 0.75, 0.85],
            "baseline": [True, False, True, False],
        }
    )

    results = plot_mod.evaluate_final_outcomes_improved(final_df, tmp_path)
    paired = plot_mod.build_paired_outcome_table(final_df)

    assert paired.shape[0] == 2
    paired_tests = results["paired_tests"]
    assert set(paired_tests["metric"]) == {
        "objective_true",
        "objective_observed",
        "simple_regret_true",
        "regret_cum_true",
        "regret_avg_true",
    }

    true_row = paired_tests.loc[paired_tests["metric"] == "objective_true"].iloc[0]
    assert np.isclose(true_row["mean_diff"], 0.05)
    assert "p_value_t_fdr_bh" in paired_tests.columns
    assert "p_value_wilcoxon_fdr_bh" in paired_tests.columns

    assert (tmp_path / "final_outcome_pair_differences.csv").exists()
    assert (tmp_path / "final_outcome_paired_tests.csv").exists()
    assert (tmp_path / "effect_sizes_cohens_dz.csv").exists()
