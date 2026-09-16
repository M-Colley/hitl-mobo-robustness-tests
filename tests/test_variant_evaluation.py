"""The variant suffix: several conditions in one output directory.

One arm directory may hold four spike sizes, two ceiling modes or a halo rho.
Nothing inside a run log distinguishes them -- run_metadata.json is overwritten
by the last invocation and the adaptation fields are not written per row -- so
the file name is the only record. These tests pin that it is read correctly,
that a variant separates CONDITIONS but not the BASELINE it pairs with, and
that a single-variant directory is untouched (variant "" everywhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_research_question as ev  # noqa: E402


@pytest.mark.parametrize("name,expected,oracle", [
    # A baseline's suffix is found only by stripping the LOGGED oracle name:
    # "extra_trees" carries an underscore, and guessing split it into oracle
    # "extra" and variant "trees".
    ("bo_sensor_error_ehmi_composite_ei_seed10_baseline_extra_trees.csv", "", "extra_trees"),
    ("bo_sensor_error_ackley_value_logei_seed7_baseline_exact_inc-observed_max.csv",
     "inc-observed_max", "exact"),
    ("bo_sensor_error_ackley_value_ei_seed10_baseline_exact.csv", "", "exact"),
    # Without the oracle a baseline never claims a variant: silence beats a guess.
    ("bo_sensor_error_ackley_value_logei_seed7_baseline_exact_inc-observed_max.csv", "", None),
])
def test_parse_variant_strips_the_logged_oracle(name, expected, oracle):
    assert ev.parse_variant(name, oracle) == expected


@pytest.mark.parametrize("name,expected", [
    # The four spike sizes of one arm directory.
    ("bo_sensor_error_ackley_value_logei_seed10_jittered_exact_spike_jit0_std0.25_sp0.05-5.csv", "sp0.05-5"),
    ("bo_sensor_error_ackley_value_logei_seed10_jittered_exact_spike_jit0_std0.25_sp0.15-20.csv", "sp0.15-20"),
    # Suffixes carrying their own underscores and dots.
    ("bo_sensor_error_branin_value_qnei_seed7_jittered_exact_gaussian_jit20_std1.0_sched-front10.csv",
     "sched-front10"),
    ("bo_sensor_error_branin_value_logei_seed7_jittered_exact_slip_jit0_std0.15_iu16-0.15.csv", "iu16-0.15"),
    ("bo_sensor_error_ackley_value_ucb_seed9_jittered_exact_missing_low_jit0_std0.3_miss-drop.csv", "miss-drop"),
    # No suffix: the standard sweep, which must keep behaving exactly as before.
    ("bo_sensor_error_ackley_value_ei_seed10_jittered_exact_ar1_jit0_std0.05.csv", ""),
    ("bo_sensor_error_ackley_value_ei_seed10_baseline_exact.csv", ""),
    # Not a per-run log at all.
    ("bo_sensor_error_dataset_effects.csv", ""),
])
def test_parse_variant_reads_the_suffix(name, expected):
    assert ev.parse_variant(name) == expected


def test_parse_variant_ignores_a_directory_prefix():
    assert ev.parse_variant("x_std0.25_sp0.05-5.csv") == ""  # no _jittered_ stem, no suffix claimed


def test_a_variant_separates_conditions():
    """Two variants of one condition must summarise separately, not be averaged."""
    assert "variant" in ev.CONDITION_COLS


def test_a_variant_does_not_separate_a_baseline():
    """The clean twin is shared: a variant that changes it needs its own directory.

    build_paired_table joins the jittered runs to the baselines on keys that must
    NOT include the variant, or a suffixed noisy run would find no clean twin.
    """
    import inspect
    source = inspect.getsource(ev.build_paired_table)
    base_keys_block = source.split("base_keys = [", 1)[1].split("]", 1)[0]
    assert "variant" not in base_keys_block


def test_the_standard_sweep_is_untouched(tmp_path):
    """Every file of a single-variant directory gets variant "", the old behaviour."""
    names = [
        "bo_sensor_error_ackley_value_ei_seed7_baseline_exact.csv",
        "bo_sensor_error_ackley_value_ei_seed7_jittered_exact_gaussian_jit0_std1.0.csv",
    ]
    for name in names:
        pd.DataFrame({"dataset": ["ackley"], "iteration": [1]}).to_csv(tmp_path / name, index=False)
    logs = ev.load_iteration_logs(tmp_path)
    assert set(logs["variant"]) == {""}


def test_a_multi_variant_directory_labels_each_run(tmp_path):
    """Each run keeps its own label, and the shared baseline keeps "" ."""
    expected = {
        "bo_sensor_error_ackley_value_logei_seed7_baseline_exact.csv": "",
        "bo_sensor_error_ackley_value_logei_seed7_jittered_exact_spike_jit0_std0.25_sp0.05-5.csv": "sp0.05-5",
        "bo_sensor_error_ackley_value_logei_seed7_jittered_exact_spike_jit0_std0.25_sp0.15-20.csv": "sp0.15-20",
    }
    for name in expected:
        pd.DataFrame({"dataset": ["ackley"], "iteration": [1], "oracle_model": ["exact"],
                      "run_id": [name]}).to_csv(tmp_path / name, index=False)
    logs = ev.load_iteration_logs(tmp_path)
    got = dict(zip(logs["run_id"], logs["variant"]))
    assert got == expected
