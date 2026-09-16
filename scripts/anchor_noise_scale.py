"""Put the real datasets on the synthetic arm's axis.

The known-function sweep measures error in units of each landscape's own
standard deviation, which makes twenty landscapes comparable but says nothing
about whether $1\\sigma$ is a lot. ``calibrate_noise_from_data.py`` measures the
opposite thing: the within-rater noise of real participants, in rating points.
Neither is interpretable without the other.

This script joins them. For each real dataset it builds the oracle exactly as
the simulator does, samples its design box, and reports the same two numbers the
synthetic arm reports for a benchmark:

  ``sigma_f``  the standard deviation of the objective over the box, which is
               the unit the synthetic sweep's error magnitudes are in;
  ``opt_z``    how many of those standard deviations the optimum stands above an
               average design, which is the scale the synthetic arm found to
               matter beyond ``sigma_f``.

Dividing the measured rating noise by ``sigma_f`` places each real study on the
synthetic sweep's x-axis, so "what does the noise in my study actually cost me"
can be read off the dose-response curve. Placing ``opt_z`` among the twenty
benchmarks says which of them the real study most resembles.

  python scripts/anchor_noise_scale.py --calibration output/noise_calibration.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bo_sensor_error_simulation as sim  # noqa: E402  (thread limits, dtype)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import boba_benchmarks as bb  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, default=Path("output/noise_calibration.csv"))
    parser.add_argument("--dataset-config", type=Path, default=Path("datasets.json"))
    parser.add_argument("--oracle-selection-path", type=Path,
                        default=Path("output/best_oracle_models.json"))
    parser.add_argument("--objective", type=str, default="composite")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=10_007)
    parser.add_argument("--output-path", type=Path, default=Path("output/noise_anchor.csv"))
    parser.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    calibration = pd.read_csv(args.calibration)
    calibration = calibration[calibration["objective"] == args.objective]
    selection = sim.load_oracle_selection(args.oracle_selection_path)
    datasets = sim.parse_dataset_configs(None, args.dataset_config, Path(".dataset_cache"))
    landscape = bb.load_stats(args.stats_path)

    rows: list[dict] = []
    for dataset in datasets:
        entry = selection.get((dataset.name, args.objective), {})
        model = entry.get("best_model")
        if model is None:
            print(f"no oracle selection for {dataset.name}/{args.objective}; skipping",
                  file=sys.stderr)
            continue
        frame = sim.load_observations(dataset, args.objective, None, None)
        columns = dataset.objective_map[args.objective]
        oracle = sim.build_oracle(
            df=frame, objective=args.objective, objective_columns=columns,
            param_columns=dataset.param_columns, seed=args.seed, normalize=False,
            weights=None, oracle_model=model, oracle_augmentation="jitter",
            oracle_augment_repeats=2, oracle_augment_std=0.02, oracle_fast=False,
            oracle_target=dataset.oracle_target,
        )
        bounds = sim.bounds_from_data(frame, dataset.param_columns)
        rng = np.random.default_rng(args.seed)
        X = rng.uniform(bounds.low, bounds.high,
                        size=(args.samples, len(dataset.param_columns)))
        y = oracle.predict_many(X).reshape(-1)
        mean, sd = float(np.mean(y)), float(np.std(y, ddof=1))
        # Same convention as the synthetic arm: anchored on the observed designs
        # so the estimate is never below the oracle's value at a real design.
        y_opt = sim.estimate_oracle_optimum(
            oracle=oracle, bounds=bounds, seed=args.seed, n=args.samples,
            batch_size=50_000,
            X_known=frame[dataset.param_columns].to_numpy(dtype=float),
        )

        noise = calibration[calibration["dataset"] == dataset.name]
        if noise.empty:
            continue
        noise_row = noise.iloc[0]
        # Prefer the close-neighbour estimator; it is the tighter of the two.
        sd_hat = noise_row.get("sd_nn_close")
        estimator = "close NN pairs"
        if not np.isfinite(sd_hat):
            sd_hat = noise_row.get("sd_nn_all")
            estimator = "all NN pairs (upper bound)"

        rows.append({
            "dataset": dataset.name,
            "objective": args.objective,
            "oracle_model": model,
            "sigma_f": sd,
            "mean_f": mean,
            "y_opt": y_opt,
            "opt_z": (y_opt - mean) / sd if sd > 0 else np.nan,
            "rating_noise_sd": float(sd_hat),
            "rating_noise_estimator": estimator,
            "noise_in_landscape_sd": float(sd_hat) / sd if sd > 0 else np.nan,
        })

    table = pd.DataFrame(rows)
    table.to_csv(args.output_path, index=False)

    suite = {k: v for k, v in landscape.items() if k in bb.DEFAULT_SUITE}
    opt_z_values = np.sort([v["opt_z"] for v in suite.values()])

    lines = ["REAL STUDIES ON THE SYNTHETIC ARM'S AXIS", "=" * 72, "",
             "sigma_f is the objective's spread over the design box, i.e. the unit the",
             "synthetic sweep's error magnitudes are in. The last column is the measured",
             "within-rater noise expressed in those units -- where each real study sits on",
             "the sweep's x-axis.", ""]
    lines.append(f"  {'dataset':<12s} {'oracle':<18s} {'sigma_f':>8s} {'opt_z':>7s} "
                 f"{'noise':>7s} {'noise/sigma_f':>14s} {'suite pctile':>13s}")
    for _, row in table.iterrows():
        pct = 100.0 * float(np.mean(opt_z_values <= row["opt_z"]))
        lines.append(
            f"  {row['dataset']:<12s} {row['oracle_model']:<18s} {row['sigma_f']:>8.3f} "
            f"{row['opt_z']:>7.2f} {row['rating_noise_sd']:>7.3f} "
            f"{row['noise_in_landscape_sd']:>14.3f} {pct:>12.0f}%"
        )
    lines += ["", "Read against the sweep's grid of 0.05, 0.25, 1.0 and 5.0 landscape SDs.",
              "'suite pctile' is where each study's opt_z falls among the twenty benchmarks."]
    report = "\n".join(lines)
    args.output_path.with_suffix(".txt").write_text(report, encoding="utf-8")
    print(report)

    args.output_path.with_suffix(".json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
