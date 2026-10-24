"""A confirmatory seed must never have appeared in the source screen."""
from argparse import Namespace

import pandas as pd
import pytest

import confirmatory_followup as cf


def screen_log(directory, filename_seed, logged_seed):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"bo_sensor_error_toy_composite_pi_seed{filename_seed}_baseline_extra_trees.csv"
    pd.DataFrame({"seed": [logged_seed, logged_seed]}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("run_simulation", [False, True])
def test_screening_overlap_fails_before_copy_simulation_or_evaluation(tmp_path, monkeypatch, run_simulation):
    # The logged identity is authoritative even if somebody renamed the file.
    screen_log(tmp_path / "screen", filename_seed=7, logged_seed=27)
    args = Namespace(acquisitions="pi,logpi", seed_start=27, num_new_seeds=2,
                     base_input_dir=tmp_path / "screen", objective="composite",
                     output_root=tmp_path / "result", run_simulation=run_simulation)
    monkeypatch.setattr(cf, "parse_args", lambda: args)

    def forbidden(*args, **kwargs):
        pytest.fail("Overlap must be rejected before any outputs or experiment execution")

    for name in ("prepare_directories", "copy_existing_logs", "run_simulation", "run_evaluation"):
        monkeypatch.setattr(cf, name, forbidden)
    with pytest.raises(SystemExit, match=r"confirmatory seeds \[27\].*broad-screen"):
        cf.main()
    assert not args.output_root.exists()


@pytest.mark.parametrize("start,count", [(-1, 2), (7, 0), (7, -1)])
def test_confirmatory_seed_block_must_be_valid(start, count):
    with pytest.raises(ValueError, match="non-negative start"):
        cf._parse_seed_list(start, count)


@pytest.mark.parametrize("values", [[], [7, 8], [float("nan")], [-1], [7.5]])
def test_screen_source_requires_one_valid_seed(tmp_path, values):
    path = tmp_path / "screen.csv"
    pd.DataFrame({"seed": values}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="screening seed"):
        cf.validate_screening_seeds([path], [27])


def test_disjoint_screen_allows_cached_fresh_runs_and_scores_only_new_seeds(tmp_path, monkeypatch):
    base = tmp_path / "screen"
    screen_log(base, 7, 7)
    screen_log(base, 8, 8)
    base_eval = tmp_path / "screen-evaluation"
    base_eval.mkdir()
    pd.DataFrame({"acquisition": ["pi", "logpi"]}).to_csv(
        base_eval / "overall_rankings.csv", index=False
    )
    output = tmp_path / "result"
    cached = screen_log(output / "raw", 27, 27)
    original = cached.read_bytes()
    args = Namespace(acquisitions="pi,logpi", seed_start=27, num_new_seeds=2,
                     base_input_dir=base, objective="composite", output_root=output,
                     run_simulation=False, allow_mixed_iterations=False,
                     base_evaluation_dir=base_eval)
    monkeypatch.setattr(cf, "parse_args", lambda: args)
    rows = []
    for seed in (7, 8, 27, 28):
        for acq in ("pi", "logpi"):
            row = dict(dataset="toy", objective="composite", oracle_model="extra_trees",
                       acquisition=acq, error_model="gaussian", jitter_std=1.,
                       jitter_iteration=0, seed=seed)
            for metric in (cf.PRIMARY_METRIC, cf.SECONDARY_METRIC,
                           cf.REACTION_METRIC, cf.DATASET_PRIMARY_METRIC):
                row[metric] = 1 + seed * (0.1 if acq == "pi" else 0.2)
            rows.append(row)

    def evaluate(raw, destination, allow_mixed):
        pd.DataFrame(rows).to_csv(destination / "paired_excess_metrics.csv", index=False)
        pd.DataFrame({"acquisition": []}).to_csv(destination / "condition_summary.csv", index=False)

    monkeypatch.setattr(cf, "run_evaluation", evaluate)
    monkeypatch.setattr(cf, "run_simulation", lambda *a: pytest.fail("Cached analysis must not run experiments"))
    monkeypatch.setattr(cf, "plot_confirmatory_heatmaps", lambda *a: [])
    monkeypatch.setattr(cf, "plot_winner_map", lambda *a: "winner.png")
    monkeypatch.setattr(cf, "plot_head_to_head_lines", lambda *a: "lines.png")
    monkeypatch.setattr(cf, "plot_paired_points", lambda *a: [])
    cf.main()
    pairs = pd.read_csv(output / "evaluation/head_to_head_pairs.csv")
    assert set(pairs.seed) == {27, 28}
    assert cached.read_bytes() == original
    assert len(list((output / "raw").glob("*.csv"))) == 3
