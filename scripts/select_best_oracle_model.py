"""Select the best oracle model based on cross-validated performance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm

from bo_sensor_error_simulation import (
    DATA_DIR,
    ORACLE_MODEL_CHOICES,
    _build_oracle_model,
    compute_objective,
    compute_objective_matrix,
    load_observations,
    parse_dataset_configs,
    parse_objective_list,
    parse_objective_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path(".dataset_cache"))
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--objectives", type=str, default=None)
    parser.add_argument("--oracle-models", type=str, default="all")
    parser.add_argument("--oracle-fast", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cv-folds", type=int, default=3)
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


def _evaluate_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    cv_folds: int,
    tree_scale: float,
) -> float:
    model = _build_oracle_model(model_name, seed=seed, tree_scale=tree_scale)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


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
    X = df[param_columns].to_numpy(dtype=float)
    scores: dict[str, float] = {}

    if objective == "multi_objective":
        Y = compute_objective_matrix(df, objective_columns, normalize)
        for model_name in tqdm(models, desc=progress_desc, leave=False):
            per_target = []
            for idx in range(Y.shape[1]):
                per_target.append(
                    _evaluate_model(model_name, X, Y[:, idx], seed, cv_folds, tree_scale)
                )
            scores[model_name] = float(np.mean(per_target))
        return scores

    y = compute_objective(df, objective_columns, normalize, weights).to_numpy(dtype=float)
    for model_name in tqdm(models, desc=progress_desc, leave=False):
        scores[model_name] = _evaluate_model(model_name, X, y, seed, cv_folds, tree_scale)
    return scores


def main() -> None:
    args = parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")

    datasets = parse_dataset_configs(args.data_dir, args.dataset_config, args.dataset_cache_dir)

    model_list = parse_oracle_models(args.oracle_models)
    tree_scale = 0.35 if args.oracle_fast else 1.0

    results_payload = {
        "cv_folds": args.cv_folds,
        "metric": "r2",
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
            scores = evaluate_models_for_objective(
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
            best_model = max(scores.items(), key=lambda item: item[1])[0]
            dataset_entry["objectives"][objective_name] = {
                "best_model": best_model,
                "scores": scores,
            }
        results_payload["datasets"].append(dataset_entry)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(results_payload, indent=2))

    print("Best oracle model selection complete.")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
