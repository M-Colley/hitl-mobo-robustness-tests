"""Error-model names are load-bearing twice over, and both uses broke once.

1. SEEDS. Every run's noise stream is seeded from ERROR_MODEL_CHOICES.index(...).
   Prepending "none" to that list shifted every existing model by one, so the
   published sweeps stopped reproducing -- and the full suite still passed,
   because nothing pinned the order. The tests below pin it.

2. LABELS. "none" is how the whole pipeline recognises a clean baseline. The
   first input-error arm labelled its corrupted runs "none", so the evaluator
   filed every one of them as a baseline, paired nothing, and failed three
   functions later with a KeyError that pointed nowhere near the cause.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bo_sensor_error_simulation as sim  # noqa: E402

# The order every published sweep ran on (git HEAD, 2026-09-10). Changing it
# changes every noise realisation. Append new models; never insert.
PUBLISHED_ORDER = ["gaussian", "bias", "dropout", "spike", "drift", "ar1"]


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def test_error_model_order_is_frozen():
    assert sim.ERROR_MODEL_CHOICES[: len(PUBLISHED_ORDER)] == PUBLISHED_ORDER


@pytest.mark.parametrize("index, name", list(enumerate(PUBLISHED_ORDER)))
def test_each_published_model_keeps_its_seed_index(index, name):
    """The quantity that actually enters the SeedSequence."""
    assert sim.ERROR_MODEL_CHOICES.index(name) == index


def test_none_is_not_an_error_model_choice():
    """Keeps "none" out of the seed table and out of what "all" expands to."""
    assert "none" not in sim.ERROR_MODEL_CHOICES


def test_all_expands_to_the_published_models():
    assert list(sim.parse_error_models(None, "all")) == PUBLISHED_ORDER


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response, inp, expected",
    [
        ("gaussian", "none", "gaussian"),   # every existing arm: unchanged
        ("ar1", "none", "ar1"),
        ("none", "slip", "slip"),           # the input-error arm
        ("none", "misclick", "misclick"),
        ("bias", "misclick", "bias+misclick"),  # both channels, named as both
    ],
)
def test_run_error_label(response, inp, expected):
    assert sim.run_error_label(response, inp) == expected


def test_a_corrupted_run_can_never_be_labelled_none():
    """Nothing corrupted means it is a baseline, and "none" belongs to those."""
    with pytest.raises(ValueError, match="no corruption"):
        sim.run_error_label("none", "none")


def test_input_error_arm_pairs_end_to_end(tmp_path):
    """The regression itself: an input-error run must reach the paired table.

    Runs the real driver, then the evaluator's own loading and pairing, and
    checks the three places the label lives -- filename, per-run column and
    paired row -- all say "slip", while the baseline still says "none".
    """
    out = tmp_path / "run"
    subprocess.run(
        [
            sys.executable, str(SCRIPTS / "bo_synthetic_error_simulation.py"),
            "--functions", "branin", "--acq-list", "logei",
            "--iterations", "8", "--initial-samples", "3",
            "--error-models", "gaussian", "--jitter-stds", "0.2",
            "--jitter-iterations", "0",
            "--input-error", "slip", "--input-error-from-sweep",
            "--seeds", "7", "--output-dir", str(out), "--n-jobs", "1",
        ],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    bench = out / "branin"
    jittered = sorted(bench.glob("*_jittered_*.csv"))
    assert len(jittered) == 1
    assert "_jittered_exact_slip_" in jittered[0].name, jittered[0].name
    assert not list(bench.glob("*_gaussian_*.csv")), "slip run filed under gaussian"

    import pandas as pd

    assert set(pd.read_csv(jittered[0])["error_model"]) == {"slip"}
    baseline = sorted(bench.glob("*_baseline_*.csv"))
    assert set(pd.read_csv(baseline[0])["error_model"]) == {"none"}

    import evaluate_research_question as erq

    logs = erq.load_iteration_logs(bench)
    response = erq.build_response_table(logs)
    assert len(response), "evaluator produced no rows -- every run was filed as a baseline"
    assert set(response.loc[~response["baseline"], "error_model"]) == {"slip"}
    paired = erq.build_paired_table(response)
    assert len(paired) == 1


# ---------------------------------------------------------------------------
# The floor check: a control under a rating error, a measurement under a design
# error. Getting that backwards either hides a real leak or reports a false one.
# ---------------------------------------------------------------------------


def test_analysis_knows_the_same_input_error_models():
    import analyse_boba_robustness as ab

    # The analysis list is the DESIGN errors, which a model-free floor feels. A
    # missing-rating process loses ratings, which a floor never reads, so its
    # floor excess is exactly zero and stays asserted like a rating error's.
    assert set(ab.INPUT_ERROR_MODELS) == (
        set(sim.INPUT_ERROR_CHOICES) - {"none"} - set(sim.MISSING_INPUT_ERROR_CHOICES)
    )


def _floor_frame(error_model: str, excess: float):
    import pandas as pd

    return pd.DataFrame({
        "acquisition": ["random", "sobol"],
        "dataset": ["branin", "branin"],
        "run_id": ["a", "b"],
        "error_model": [error_model, error_model],
        "auc_simple_regret_excess_true": [excess, -excess],
        "final_simple_regret_excess_true": [excess, -excess],
    })


def test_floor_check_still_catches_a_leak_under_rating_error(capsys):
    import analyse_boba_robustness as ab

    ab.floor_check(_floor_frame("gaussian", 0.3), tolerance=1e-12)
    assert "FLOOR CHECK FAILED" in capsys.readouterr().err


def test_floor_check_reports_rather_than_asserts_under_design_error(capsys):
    import analyse_boba_robustness as ab

    report = ab.floor_check(_floor_frame("slip", 0.3), tolerance=1e-12)
    out = capsys.readouterr()
    assert "FLOOR CHECK FAILED" not in out.err
    assert "reported, not asserted" in out.out
    assert len(report) == 2


def test_floor_check_passes_a_clean_rating_arm(capsys):
    import analyse_boba_robustness as ab

    ab.floor_check(_floor_frame("gaussian", 0.0), tolerance=1e-12)
    assert "FLOOR CHECK PASSED" in capsys.readouterr().out
