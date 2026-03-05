from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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

