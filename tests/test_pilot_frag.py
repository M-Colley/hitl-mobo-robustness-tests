"""Tests for the pilot-frag estimator (experiment E7, scripts/analyse_pilot_frag.py).

The load-bearing property is that ``frag_k`` -- Equation eq:frag evaluated on a
GP posterior mean fitted to k pilot points -- converges to the exact frag as the
pilot grows. That is checked on a synthetic 1-D landscape where everything is
cheap. The other two tests pin the estimator to the study's definitions: the
replicated Monte Carlo loop must reproduce ``selection_fragility`` bit for bit,
and the measured-cost aggregation must use exactly the early-onset, model-based
cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyse_pilot_frag as apf  # noqa: E402
import boba_benchmarks as bb  # noqa: E402


def _landscape_1d(x: np.ndarray) -> np.ndarray:
    """A smooth 1-D objective on [0, 1] with two peaks of unequal height."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.exp(-((x - 0.25) / 0.12) ** 2) + 0.8 * np.exp(-((x - 0.7) / 0.08) ** 2) - 0.5 * x


def test_frag_k_converges_to_exact_frag_in_1d():
    """With a growing clean pilot the GP-mean frag approaches the exact frag."""
    rng = np.random.default_rng(3)
    # Standardise the objective the way the study does (mean/SD of a Sobol sample).
    reference = _landscape_1d(qmc.Sobol(d=1, scramble=True, seed=1).random_base2(m=12))
    mean, std = float(np.mean(reference)), float(np.std(reference, ddof=1))

    cand = qmc.Sobol(d=1, scramble=True, seed=bb.STATS_SEED + 7).random(apf.N_CANDIDATES)
    g_exact = (_landscape_1d(cand) - mean) / std
    levels = (0.25, 1.0)
    n_draws = 2000
    exact = apf.frag_from_values(g_exact, noise_levels=levels, n_draws=n_draws)

    pilot = rng.random((60, 1))
    y_pilot = (_landscape_1d(pilot) - mean) / std
    ks = (4, 12, 60)
    errors = {}
    for k in ks:
        frag_k, mu, _ = apf.pilot_frag(pilot[:k], y_pilot[:k], np.zeros(1), np.ones(1), cand,
                                       noise_levels=levels, n_draws=n_draws, fit_seed=k)
        errors[k] = {c: abs(frag_k[c] - exact[c]) for c in levels}
    # The GP mean at k = 60 is essentially the function, so its frag must match
    # the exact one closely; the tiny pilot must be worse.
    for c in levels:
        assert errors[ks[-1]][c] < 0.05, (c, errors)
        assert errors[ks[-1]][c] < errors[ks[0]][c], (c, errors)
    assert sum(errors[ks[-1]].values()) < sum(errors[ks[1]].values()) < sum(errors[ks[0]].values()), errors


@pytest.mark.parametrize("name", ["branin", "hartmann_3"])
def test_frag_from_values_replicates_selection_fragility(name):
    """The refactored loop returns the stored frag_* values exactly."""
    stats = bb.load_stats()
    entry = stats[name]
    cand = apf.frag_candidates(name)
    g = (bb.evaluate(name, cand) - entry["mean"]) / entry["std"]
    replica = apf.frag_from_values(g)
    reference = apf.exact_frag(name, stats)
    for c in apf.NOISE_LEVELS_ALL:
        assert replica[c] == reference[c]
        assert replica[c] == pytest.approx(entry[f"frag_{c:g}"], abs=1e-12)


def test_measured_cost_uses_early_onset_model_based_cells():
    rows = []
    for acq in ("ei", "logei", "random", "sobol"):
        for model in ("gaussian", "bias"):
            for onset in (0, 20):
                value = 1.0 if acq in ("ei", "logei") else 100.0
                if onset == 20:
                    value += 10.0
                rows.append({"dataset": "d", "error_model": model, "jitter_std": 1.0,
                             "jitter_iteration": onset, "acquisition": acq, "fragility": value})
    cost = apf.measured_cost(pd.DataFrame(rows))
    assert len(cost) == 1
    row = cost.iloc[0]
    assert row["sigma_e"] == 1.0
    assert row["cost_measured"] == pytest.approx(1.0)
    assert row["cost_cells"] == 4 and row["cost_acquisitions"] == 2 and row["cost_error_models"] == 2
