"""Select the best oracle model based on cross-validated performance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm

from bo_sensor_error_simulation import (
    DATA_DIR,
    ORACLE_MODEL_CHOICES,
    _build_oracle_model,
    compute_objective,
    compute_objective_matrix,
    fit_objective_normalization,
    infer_oracle_groups,
    load_observations,
    parse_dataset_configs,
    parse_objective_list,
    parse_objective_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional local dataset directory or remote Git repository URL.",
    )
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path(".dataset_cache"))
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--objectives", type=str, default=None)
    parser.add_argument("--oracle-models", type=str, default="all")
    parser.add_argument("--oracle-fast", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-path", type=Path, default=Path("output") / "best_oracle_models.json")
    parser.add_argument("--normalize-objective", action="store_true", default=False)
    parser.add_argument("--objective-weights", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def parse_oracle_models(oracle_models: str) -> list[str]:
    raw = oracle_models
    if raw == "all":
        return ORACLE_MODEL_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one oracle model must be specified.")
    unknown = [v for v in values if v not in ORACLE_MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unknown oracle model(s): {', '.join(unknown)}")
    return values


def _build_feature_frame(df: pd.DataFrame, param_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(df[param_columns].to_numpy(dtype=float), columns=param_columns)


def _build_cv_splits(
    df: pd.DataFrame,
    seed: int,
    cv_folds: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    groups, group_source = infer_oracle_groups(df)
    if groups is not None:
        unique_groups = np.unique(groups)
        if unique_groups.size >= 2:
            n_splits = min(cv_folds, int(unique_groups.size))
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(df, groups=groups))
            return splits, {
                "strategy": "group_kfold",
                "group_source": group_source,
                "effective_cv_folds": n_splits,
            }

    n_splits = min(cv_folds, len(df))
    if n_splits < 2:
        raise ValueError("Not enough rows to run cross-validation.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df)), {
        "strategy": "kfold",
        "group_source": None,
        "effective_cv_folds": n_splits,
    }


def _evaluate_model_single_objective(
    model_name: str,
    df: pd.DataFrame,
    objective_columns: list[str],
    param_columns: list[str],
    seed: int,
    tree_scale: float,
    normalize: bool,
    weights: np.ndarray | None,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    r2_scores: list[float] = []
    rmse_scores: list[float] = []
    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        normalization = (
            fit_objective_normalization(train_df, objective_columns)
            if normalize
            else None
        )
        y_train = compute_objective(
            train_df,
            objective_columns,
            normalize,
            weights,
            normalization=normalization,
        ).to_numpy(dtype=float)
        y_test = compute_objective(
            test_df,
            objective_columns,
            normalize,
            weights,
            normalization=normalization,
        ).to_numpy(dtype=float)
        model = _build_oracle_model(model_name, seed=seed, tree_scale=tree_scale)
        X_train = _build_feature_frame(train_df, param_columns)
        X_test = _build_feature_frame(test_df, param_columns)
        model.fit(X_train, y_train)
        preds = np.asarray(model.predict(X_test), dtype=float)
        r2_scores.append(float(r2_score(y_test, preds)))
        rmse_scores.append(float(np.sqrt(mean_squared_error(y_test, preds))))
    return {
        "r2": float(np.mean(r2_scores)),
        "rmse": float(np.mean(rmse_scores)),
    }


def _evaluate_model_multi_objective(
    model_name: str,
    df: pd.DataFrame,
    objective_columns: list[str],
    param_columns: list[str],
    seed: int,
    tree_scale: float,
    normalize: bool,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    r2_scores: list[float] = []
    rmse_scores: list[float] = []
    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        normalization = (
            fit_objective_normalization(train_df, objective_columns)
            if normalize
            else None
        )
        Y_train = compute_objective_matrix(
            train_df,
            objective_columns,
            normalize,
            normalization=normalization,
        )
        Y_test = compute_objective_matrix(
            test_df,
            objective_columns,
            normalize,
            normalization=normalization,
        )
        X_train = _build_feature_frame(train_df, param_columns)
        X_test = _build_feature_frame(test_df, param_columns)
        per_target_r2: list[float] = []
        per_target_rmse: list[float] = []
        for idx in range(Y_train.shape[1]):
            model = _build_oracle_model(model_name, seed=seed, tree_scale=tree_scale)
            model.fit(X_train, Y_train[:, idx])
            preds = np.asarray(model.predict(X_test), dtype=float)
            per_target_r2.append(float(r2_score(Y_test[:, idx], preds)))
            per_target_rmse.append(float(np.sqrt(mean_squared_error(Y_test[:, idx], preds))))
        r2_scores.append(float(np.mean(per_target_r2)))
        rmse_scores.append(float(np.mean(per_target_rmse)))
    return {
        "r2": float(np.mean(r2_scores)),
        "rmse": float(np.mean(rmse_scores)),
    }


def evaluate_models_for_objective(
    df,
    objective: str,
    objective_columns: list[str],
    param_columns: list[str],
    models: list[str],
    seed: int,
    cv_folds: int,
    normalize: bool,
    weights: np.ndarray | None,
    tree_scale: float,
    progress_desc: str | None = None,
) -> dict[str, float]:
    cv_splits, validation_info = _build_cv_splits(df, seed, cv_folds)
    scores: dict[str, dict[str, float]] = {}

    if objective == "multi_objective":
        for model_name in tqdm(models, desc=progress_desc, leave=False):
            scores[model_name] = _evaluate_model_multi_objective(
                model_name,
                df,
                objective_columns,
                param_columns,
                seed,
                tree_scale,
                normalize,
                cv_splits,
            )
        return {
            "scores": {name: metrics["r2"] for name, metrics in scores.items()},
            "rmse": {name: metrics["rmse"] for name, metrics in scores.items()},
            "validation": validation_info,
        }

    for model_name in tqdm(models, desc=progress_desc, leave=False):
        scores[model_name] = _evaluate_model_single_objective(
            model_name,
            df,
            objective_columns,
            param_columns,
            seed,
            tree_scale,
            normalize,
            weights,
            cv_splits,
        )
    return {
        "scores": {name: metrics["r2"] for name, metrics in scores.items()},
        "rmse": {name: metrics["rmse"] for name, metrics in scores.items()},
        "validation": validation_info,
    }


def main() -> None:
    args = parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")

    datasets = parse_dataset_configs(args.data_dir, args.dataset_config, args.dataset_cache_dir)

    model_list = parse_oracle_models(args.oracle_models)
    tree_scale = 0.7 if args.oracle_fast else 1.0

    results_payload = {
        "cv_folds_requested": args.cv_folds,
        "selection_metric": "r2",
        "tie_breaker_metric": "rmse",
        "oracle_models": model_list,
        "datasets": [],
    }

    for dataset in tqdm(datasets, desc="Datasets"):
        objective_names = parse_objective_list(args.objective, args.objectives, dataset.objective_map)
        dataset_entry = {
            "name": dataset.name,
            "objectives": {},
        }
        for objective_name in objective_names:
            objective_columns = dataset.objective_map[objective_name]
            weights = (
                parse_objective_weights(args.objective_weights, objective_name, objective_columns)
                if objective_name != "multi_objective"
                else None
            )
            df = load_observations(dataset, objective_name)
            if args.max_rows is not None:
                df = df.head(int(args.max_rows))
            evaluation = evaluate_models_for_objective(
                df,
                objective_name,
                objective_columns,
                dataset.param_columns,
                model_list,
                args.seed,
                args.cv_folds,
                args.normalize_objective,
                weights,
                tree_scale,
                progress_desc=f"{dataset.name}:{objective_name}",
            )
            scores = evaluation["scores"]
            rmse = evaluation["rmse"]
            validation = evaluation["validation"]
            best_model = max(
                scores.items(),
                key=lambda item: (item[1], -rmse[item[0]]),
            )[0]
            dataset_entry["objectives"][objective_name] = {
                "best_model": best_model,
                "scores": scores,
                "rmse": rmse,
                "validation_strategy": validation["strategy"],
                "group_source": validation["group_source"],
                "effective_cv_folds": validation["effective_cv_folds"],
            }
        results_payload["datasets"].append(dataset_entry)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(results_payload, indent=2))

    print("Best oracle model selection complete.")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
