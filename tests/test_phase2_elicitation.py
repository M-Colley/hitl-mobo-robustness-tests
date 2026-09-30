"""The comparison loop: determinism, tie conventions and the standard-process estimand.

BoTorch's PairwiseGP starts every Laplace MAP search from the win counts plus a
draw from numpy's GLOBAL generator, so without a per-run seed a comparison-loop
run was a different run on every execution. These tests pin the fix (a run is
a function of its task alone), the uniform tie-break the summary reports beside
the logged first-index rule, and the standard-process estimand of AGENTS.md
(cost = rating noisy - rating clean, gain = rating noisy - comparison noisy,
price = comparison clean - rating clean).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402
import elicitation_compare as ec  # noqa: E402


def _task(**over):
    task = dict(dataset="branin", elicitation="pairwise", error_model="bias", magnitude=1.0, seed=7,
                apply_error=False, iterations=9, initial_samples=5, candidate_pool=32, spike_prob=0.15,
                spike_sd=5.0, ceiling_quantile=0.9, stats_path=str(bb.DEFAULT_STATS_PATH))
    task.update(over)
    return task


def test_a_comparison_run_does_not_depend_on_the_global_generators_it_inherits():
    out = []
    for state in (0, 12345):
        np.random.seed(state)
        torch.manual_seed(state)
        np.random.standard_normal(1000)
        torch.rand(1000)
        out.append(ec.run_cell(_task()))
    keys = ("shipped_true", "shipped_true_uniform_ties", "best_visited_true", "fit_failures", "n_fits")
    assert [out[0][k] for k in keys] == [out[1][k] for k in keys]


@pytest.mark.parametrize("model", ["bias", "drift"])
def test_the_invariance_holds_whatever_the_process_ran_before(model):
    """The by-construction result: a strictly monotone shared fault leaves the run unchanged."""
    np.random.seed(99)
    clean = ec.run_cell(_task(error_model=model))
    np.random.seed(3)
    noisy = ec.run_cell(_task(error_model=model, apply_error=True))
    assert noisy["shipped_true"] == clean["shipped_true"]
    assert noisy["fit_failures"] == clean["fit_failures"]


def test_the_loops_count_their_fits():
    rating = ec.run_cell(_task(elicitation="rating"))
    pairwise = ec.run_cell(_task())
    assert rating["n_fits"] == 9 - 5            # one fit per proposal
    assert pairwise["n_fits"] == 9 - 5 + 1      # and one more to choose the design to ship


def test_ship_reports_the_first_tied_design_and_the_uniform_expectation():
    score = np.array([1.0, 3.0, 2.0, 3.0, 3.0])
    truth = np.array([10.0, 4.0, 9.0, 7.0, 1.0])
    got = ec._ship(score, truth)
    assert got["shipped_true"] == 4.0                       # np.argmax: the earliest tied design
    assert got["shipped_true_uniform_ties"] == pytest.approx(4.0)   # mean of 4, 7, 1
    assert got["n_tied_top"] == 3
    untied = ec._ship(np.array([1.0, 2.0]), np.array([5.0, 6.0]))
    assert untied["shipped_true"] == untied["shipped_true_uniform_ties"] == 6.0


def _runs(values: dict[tuple[str, str, bool], list[float]], y_opt: float = 10.0) -> pd.DataFrame:
    """Runs on two landscapes and one seed; values are shipped_true per landscape."""
    rows = []
    for (elic, model, noisy), shipped in values.items():
        for dataset, s in zip(("a", "b"), shipped):
            rows.append({"dataset": dataset, "elicitation": elic, "error_model": model, "magnitude": 1.0,
                         "seed": 7, "apply_error": noisy, "shipped_true": s, "shipped_true_uniform_ties": s,
                         "n_tied_top": 1, "best_visited_true": s, "n_designs": 20, "fit_failures": 0,
                         "n_fits": 15, "y_opt": y_opt})
    return pd.DataFrame(rows)


def test_the_standard_process_estimand_is_scored_against_the_rating_loops_clean_run():
    # opt_z 1 and 2; regret = (10 - shipped) / opt_z.
    runs = _runs({("rating", "gaussian", False): [9.0, 8.0], ("rating", "gaussian", True): [7.0, 4.0],
                  ("pairwise", "gaussian", False): [8.0, 8.0], ("pairwise", "gaussian", True): [8.0, 6.0]})
    s = ec.summarise(runs, {"a": 1.0, "b": 2.0}).iloc[0]
    # regrets: rating clean 1, 1; rating noisy 3, 3; pairwise clean 2, 1; pairwise noisy 2, 2
    assert s["std_cost"] == pytest.approx(2.0)                   # mean of (3 - 1, 3 - 1)
    assert s["std_gain"] == pytest.approx(1.0)                   # mean of (3 - 2, 3 - 2)
    assert s["std_price"] == pytest.approx(0.5)                  # mean of (2 - 1, 1 - 1)
    assert s["std_recovered"] == pytest.approx(0.5)
    # The arm's own-twin share, kept beside it: rating excess 2, 2; pairwise excess 0, 1.
    assert s["removed_share"] == pytest.approx(1 - 0.5 / 2.0)


def test_a_dataset_without_opt_z_is_refused():
    runs = _runs({("rating", "bias", False): [9.0, 8.0], ("rating", "bias", True): [9.0, 8.0],
                  ("pairwise", "bias", False): [9.0, 8.0], ("pairwise", "bias", True): [9.0, 8.0]})
    with pytest.raises(KeyError):
        ec.summarise(runs, {"a": 1.0})


def test_suite_selects_the_papers_twenty_landscapes(tmp_path, monkeypatch):
    seen = {}

    def fake_tasks(args, names):
        seen["names"] = names
        return []

    monkeypatch.setattr(ec, "build_tasks", fake_tasks)
    monkeypatch.setattr(ec, "summarise", lambda runs, opt_z: pd.DataFrame(
        columns=["class", "error_model", "magnitude"]))
    monkeypatch.setattr(ec, "report", lambda summary: "")

    class _Pool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, tasks, chunksize=1):
            return iter(())

    monkeypatch.setattr(ec, "ProcessPoolExecutor", _Pool)
    ec.main(["--functions", "suite", "--output-dir", str(tmp_path), "--workers", "1"])
    assert seen["names"] == sorted(bb.DEFAULT_SUITE) and len(seen["names"]) == 20
    with pytest.raises(SystemExit):
        ec.main(["--functions", "branin,not_a_landscape", "--output-dir", str(tmp_path)])
