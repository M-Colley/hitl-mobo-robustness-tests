"""Acquisition names are load-bearing for the noise seed, and the order moved once.

Every jittered run seeds its noise from

    SeedSequence([seed, ACQUISITION_CHOICES.index(acq), onset,
                  round(std * 1e6), ERROR_MODEL_CHOICES.index(model)])

so an acquisition's POSITION in ACQUISITION_CHOICES decides the noise it sees.

What the order has been, established on 2026-09-10 by regenerating published
runs from their own recorded settings and diffing them bit for bit:

  * The ten single-objective acquisitions have held indices 0-9 throughout.
    Regenerated main-sweep runs for logei and qnei are identical on every
    column, so every headline number reproduces.
  * When qkg and replei were added they went in at 10-11, moving the
    hypervolume family from 10-13 to 12-15 and the model-free floors from 14-15
    to 16-17. Every arm run since uses that order, which is the one pinned here.
  * The main sweep predates that insertion, so its model-free floors' jittered
    runs do not reproduce in the observation-dependent columns
    (objective_observed, error_magnitude, and the inference metric). Their
    candidates and TRUE regret do reproduce -- a floor never reads an
    observation -- so no reported number moves. That is the one known
    exception. It is documented rather than "fixed", because restoring the old
    order would instead break every arm run since, including the robust and
    multi-objective arms' actual results.

Append new acquisitions at the END. Never insert.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bo_sensor_error_simulation as sim  # noqa: E402

PINNED_ORDER = [
    "logei", "logpi", "ei", "pi", "ucb", "qucb", "qei", "qpi", "qnei", "greedy",
    "qkg", "replei",
    "qehvi", "qnehvi", "qlogehvi", "qlognehvi",
    "random", "sobol",
]

# The acquisitions every headline number rests on, whose indices the main sweep
# shares with every later arm.
HEADLINE = PINNED_ORDER[:10]


def test_acquisition_order_is_frozen():
    assert sim.ACQUISITION_CHOICES[: len(PINNED_ORDER)] == PINNED_ORDER


@pytest.mark.parametrize("index, name", list(enumerate(HEADLINE)))
def test_headline_acquisitions_keep_their_main_sweep_index(index, name):
    assert sim.ACQUISITION_CHOICES.index(name) == index


def test_sensor_acq_all_excludes_the_robust_baselines():
    """--acq all was silently widened once; it must mean the published design."""
    assert "qkg" not in sim.DEFAULT_ACQUISITION_CHOICES
    assert "replei" not in sim.DEFAULT_ACQUISITION_CHOICES
    assert len(sim.DEFAULT_ACQUISITION_CHOICES) == 16
    # ...while staying nameable, which is what the robust arm does.
    assert set(sim.ROBUST_ACQUISITION_CHOICES) <= set(sim.ACQUISITION_CHOICES)


def test_synthetic_acq_all_is_the_twelve_the_preregistration_names():
    """HYPOTHESIS.md fixes the confirmatory design at 12 acquisitions."""
    import bo_synthetic_error_simulation as syn

    assert syn.DEFAULT_ACQUISITIONS == HEADLINE + ["random", "sobol"]
    assert {"qkg", "replei"} <= set(syn.SYNTHETIC_ACQUISITION_CHOICES)
