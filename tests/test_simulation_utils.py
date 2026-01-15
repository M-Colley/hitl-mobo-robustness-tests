from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bo_sensor_error_simulation.py"

spec = importlib.util.spec_from_file_location("bo_sim", MODULE_PATH)
bo_sim = importlib.util.module_from_spec(spec)
sys.modules["bo_sim"] = bo_sim
assert spec.loader is not None
spec.loader.exec_module(bo_sim)


def make_dummy_df(rows: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    data = {
        col: rng.normal(size=rows)
        for col in bo_sim.PARAM_COLUMNS + bo_sim.OBJECTIVE_MAP["composite"]
    }
    data["User_ID"] = rng.integers(1, 3, size=rows)
    data["Group_ID"] = rng.integers(1, 3, size=rows)
    return pd.DataFrame(data)


def test_parse_objective_list_defaults_to_composite_and_multi() -> None:
    objectives = bo_sim.parse_objective_list(None, None, bo_sim.OBJECTIVE_MAP)
    assert objectives == ["composite", "multi_objective"]


def test_parse_oracle_models_default_set() -> None:
    models = bo_sim.parse_oracle_models("xgboost", "xgboost,lightgbm,catboost,tabpfn")
    assert models == ["xgboost", "lightgbm", "catboost", "tabpfn"]


def test_filter_acquisitions_for_objective() -> None:
    single_acqs = ["ei", "ucb", "qei"]
    assert bo_sim.filter_acquisitions_for_objective(single_acqs, "composite") == single_acqs

    multi_acqs = ["qehvi", "qnehvi"]
    assert bo_sim.filter_acquisitions_for_objective(multi_acqs, "multi_objective") == multi_acqs


def test_compute_reference_point_multi_objective() -> None:
    df = make_dummy_df()
    ref = bo_sim.compute_reference_point(
        df,
        "multi_objective",
        bo_sim.OBJECTIVE_MAP["multi_objective"],
    )
    assert ref is not None
    assert ref.shape[0] == len(bo_sim.OBJECTIVE_MAP["multi_objective"])


def test_apply_sensor_error_vector_bias() -> None:
    config = bo_sim.SimulationConfig(
        iterations=5,
        jitter_iteration=2,
        jitter_std=0.1,
        single_error=False,
        initial_samples=1,
        candidate_pool=10,
        objective="composite",
        objective_columns=bo_sim.OBJECTIVE_MAP["composite"],
        param_columns=bo_sim.PARAM_COLUMNS,
        seed=1,
        error_model="bias",
        error_bias=0.5,
        error_spike_prob=0.1,
        error_spike_std=0.2,
        dropout_strategy="hold_last",
        normalize_objective=False,
        objective_weights=None,
        acq_num_restarts=2,
        acq_raw_samples=8,
        acq_maxiter=15,
        acq_mc_samples=32,
        ref_point=None,
    )
    rng = np.random.default_rng(0)
    true_value = np.array([1.0, 2.0])
    expected_rng = np.random.default_rng(0)
    jitter = expected_rng.normal(0.0, config.jitter_std, size=true_value.shape)
    expected_bias = np.full_like(true_value, config.error_bias, dtype=float)
    expected_error = expected_bias + jitter
    observed, error = bo_sim.apply_sensor_error(true_value, 3, config, rng, true_value)
    assert np.allclose(observed, true_value + expected_error)
    assert np.allclose(error, expected_error)


def test_oracle_builders_for_key_models() -> None:
    df = make_dummy_df()
    for model_name in ["xgboost", "lightgbm", "catboost"]:
        oracle = bo_sim.build_oracle(
            df=df,
            objective="composite",
            objective_columns=bo_sim.OBJECTIVE_MAP["composite"],
            param_columns=bo_sim.PARAM_COLUMNS,
            seed=7,
            normalize=False,
            weights=None,
            oracle_model=model_name,
            oracle_augmentation="none",
            oracle_augment_repeats=0,
            oracle_augment_std=0.0,
            oracle_fast=True,
        )
        pred = oracle.predict(df[bo_sim.PARAM_COLUMNS].iloc[0].to_numpy(dtype=float))
        assert pred.shape == (1,)


def test_oracle_builders_multi_objective_xgboost() -> None:
    df = make_dummy_df()
    oracle = bo_sim.build_oracle(
        df=df,
        objective="multi_objective",
        objective_columns=bo_sim.OBJECTIVE_MAP["multi_objective"],
        param_columns=bo_sim.PARAM_COLUMNS,
        seed=7,
        normalize=False,
        weights=None,
        oracle_model="xgboost",
        oracle_augmentation="none",
        oracle_augment_repeats=0,
        oracle_augment_std=0.0,
        oracle_fast=True,
    )
    pred = oracle.predict(df[bo_sim.PARAM_COLUMNS].iloc[0].to_numpy(dtype=float))
    assert pred.shape == (len(bo_sim.OBJECTIVE_MAP["multi_objective"]),)


def test_build_oracle_model_supports_tabpfn() -> None:
    model = bo_sim._build_oracle_model("tabpfn", seed=7, tree_scale=0.2)
    assert model.__class__.__name__ == "TabPFNRegressor"


def _make_error_config(error_model: str, jitter_std: float = 0.1) -> bo_sim.SimulationConfig:
    return bo_sim.SimulationConfig(
        iterations=5,
        jitter_iteration=2,
        jitter_std=jitter_std,
        single_error=False,
        initial_samples=1,
        candidate_pool=10,
        objective="composite",
        objective_columns=bo_sim.OBJECTIVE_MAP["composite"],
        param_columns=bo_sim.PARAM_COLUMNS,
        seed=1,
        error_model=error_model,
        error_bias=0.5,
        error_spike_prob=1.0,
        error_spike_std=0.2,
        dropout_strategy="hold_last",
        normalize_objective=False,
        objective_weights=None,
        acq_num_restarts=2,
        acq_raw_samples=8,
        acq_maxiter=15,
        acq_mc_samples=32,
        ref_point=None,
    )


def test_apply_sensor_error_uses_jitter_for_gaussian() -> None:
    config = _make_error_config("gaussian")
    true_value = np.array([1.0, 2.0])
    rng = np.random.default_rng(1)
    expected_rng = np.random.default_rng(1)
    jitter = expected_rng.normal(0.0, config.jitter_std, size=true_value.shape)
    observed, error = bo_sim.apply_sensor_error(true_value, 3, config, rng, true_value)
    assert np.allclose(observed, true_value + jitter)
    assert np.allclose(error, jitter)


def test_apply_sensor_error_uses_jitter_for_dropout() -> None:
    config = _make_error_config("dropout")
    true_value = np.array([1.0, 2.0])
    previous_observed = np.array([0.5, 1.5])
    rng = np.random.default_rng(2)
    expected_rng = np.random.default_rng(2)
    jitter = expected_rng.normal(0.0, config.jitter_std, size=true_value.shape)
    expected_observed = previous_observed + jitter
    observed, error = bo_sim.apply_sensor_error(true_value, 3, config, rng, previous_observed)
    assert np.allclose(observed, expected_observed)
    assert np.allclose(error, expected_observed - true_value)


def test_apply_sensor_error_uses_jitter_for_spike() -> None:
    config = _make_error_config("spike")
    true_value = np.array([1.0, 2.0])
    rng = np.random.default_rng(3)
    expected_rng = np.random.default_rng(3)
    jitter = expected_rng.normal(0.0, config.jitter_std, size=true_value.shape)
    _ = expected_rng.random()
    spike = expected_rng.normal(0.0, config.error_spike_std, size=true_value.shape)
    combined = spike + jitter
    observed, error = bo_sim.apply_sensor_error(true_value, 3, config, rng, true_value)
    assert np.allclose(observed, true_value + combined)
    assert np.allclose(error, combined)

def test_augment_oracle_data_jitter() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)
    X_aug, y_aug = bo_sim.augment_oracle_data(X, y, rng, "jitter", repeats=2, noise_std=0.01)
    assert X_aug.shape[0] == 30
    assert y_aug.shape[0] == 30


def test_parse_dataset_configs_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "datasets.json"
    config_path.write_text(
        """
        [
          {
            "name": "dataset_a",
            "data_dir": "data/a",
            "param_columns": ["p1", "p2"],
            "objective_map": {"score": ["s1", "s2"]},
            "observation_glob": "ObservationsPerEvaluation.csv"
          }
        ]
        """
    )
    datasets = bo_sim.parse_dataset_configs(Path("default"), config_path, tmp_path)
    assert len(datasets) == 1
    assert datasets[0].name == "dataset_a"
    assert datasets[0].param_columns == ["p1", "p2"]
    assert datasets[0].objective_map["score"] == ["s1", "s2"]


def test_combine_dataset_configs_intersection() -> None:
    dataset_a = bo_sim.DatasetConfig(
        name="a",
        data_dirs=[Path("a")],
        param_columns=["p1", "p2"],
        objective_map={"score": ["s1"], "alt": ["a1"]},
    )
    dataset_b = bo_sim.DatasetConfig(
        name="b",
        data_dirs=[Path("b")],
        param_columns=["p1", "p2"],
        objective_map={"score": ["s1"], "other": ["o1"]},
    )
    combined = bo_sim.combine_dataset_configs([dataset_a, dataset_b], name="combined")
    assert combined is not None
    assert combined.objective_map == {"score": ["s1"]}


def test_resolve_data_dirs_local(tmp_path: Path) -> None:
    local_dir = tmp_path / "data"
    local_dir.mkdir()
    resolved = bo_sim.resolve_data_dirs([str(local_dir)], tmp_path)
    assert resolved == [local_dir]


def test_is_remote_dataset_path() -> None:
    assert bo_sim.is_remote_dataset_path("https://github.com/M-Colley/opticarvis-data")
    assert bo_sim.is_remote_dataset_path("git@github.com:M-Colley/opticarvis-data.git")
    assert bo_sim.is_remote_dataset_path("https://github.com/M-Colley/opticarvis-data.git")
    assert not bo_sim.is_remote_dataset_path("/tmp/opticarvis-data")


def test_load_observations_multiple_dirs(tmp_path: Path) -> None:
    data_dir_a = tmp_path / "data_a"
    data_dir_b = tmp_path / "data_b"
    data_dir_a.mkdir()
    data_dir_b.mkdir()
    columns = bo_sim.PARAM_COLUMNS + bo_sim.OBJECTIVE_MAP["composite"] + ["User_ID", "Group_ID"]
    df = pd.DataFrame({col: [1.0, 2.0] for col in columns})
    df.to_csv(data_dir_a / "ObservationsPerEvaluation.csv", sep=";", index=False)
    df.to_csv(data_dir_b / "ObservationsPerEvaluation.csv", sep=";", index=False)

    dataset = bo_sim.DatasetConfig(
        name="multi",
        data_dirs=[data_dir_a, data_dir_b],
        param_columns=bo_sim.PARAM_COLUMNS,
        objective_map=bo_sim.OBJECTIVE_MAP,
    )
    loaded = bo_sim.load_observations(dataset, "composite")
    assert loaded.shape[0] == 4


def test_summarize_adjustment_includes_avg_regret() -> None:
    results = pd.DataFrame(
        {
            "iteration": [1, 2, 3],
            "p1": [0.1, 0.2, 0.3],
            "p2": [0.2, 0.3, 0.4],
            "best_true_so_far": [1.0, 1.5, 2.0],
            "simple_regret_true": [0.5, 0.4, 0.3],
            "regret_cum_true": [0.5, 0.9, 1.2],
            "regret_avg_true": [0.5, 0.45, 0.4],
        }
    )
    summary = bo_sim.summarize_adjustment(results, 1, ["p1", "p2"])
    assert np.isclose(summary["final_avg_regret_true"], 0.4)
