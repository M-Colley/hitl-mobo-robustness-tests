"""Baseline reference: the deployed extra_trees oracle under both protocols."""
import harness


def main() -> None:
    rows = []
    for name in harness.DATASET_NAMES:
        data = harness.load_dataset(name)
        cold = harness.evaluate_cold(data, harness.baseline_factory)
        warm = harness.evaluate_warm_per_user(data, harness.baseline_factory)
        rows.append({"dataset": name, "model": "extra_trees", **cold})
        rows.append({"dataset": name, "model": "extra_trees", **warm})
    harness.print_results("baseline", rows)


if __name__ == "__main__":
    main()
