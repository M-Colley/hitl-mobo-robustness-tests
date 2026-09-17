"""The currency test's two restrictions, checked on data where the answer is known.

An earlier version regressed RAW excess regret on log10(sigma_e) and
log10(opt_z) and read GAIN as beta_z = 1 - beta_c. That restriction belongs to
the exponents of a power law in the mean, not to slopes on a semi-log scale, so
the old test could not return "gain" even on data built to satisfy GAIN. These
tests build such data, and data built to satisfy SPREAD, and check that each
verdict comes back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analyse_boba_robustness as abr

# The suite's real span, 0.76 on powell to 56.8 on shekel. The estimator is
# recovered from synthetic data here, so only the span matters, but a stale span
# invites the reader to quote it as the suite's.
OPT_Z = np.geomspace(0.76, 56.8, 20)
SIGMAS = [0.05, 0.25, 1.0, 5.0]
LEARNERS = ("ei", "ucb")
SEEDS = 3


def _frame(mean_fn, noise: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Per-run rows whose cell means follow mean_fn(sigma, opt_z) up to noise.

    A model-free arm rides along with zero excess, as in the real sweep, and
    must be ignored by the fit.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d, z in enumerate(OPT_Z):
        for s in SIGMAS:
            for acq in (*LEARNERS, "random"):
                for _ in range(SEEDS):
                    mean = 0.0 if acq == "random" else mean_fn(s, z)
                    rows.append({
                        "dataset": f"f{d:02d}", "acquisition": acq, "error_model": "gaussian",
                        "jitter_iteration": 0, "jitter_std": s, "opt_z": z,
                        "excess_sd": mean * (1.0 + noise * rng.standard_normal()),
                    })
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _fewer_bootstrap_reps(monkeypatch):
    monkeypatch.setattr(abr, "BOOTSTRAP_REPS", 400)


def _grid() -> tuple[np.ndarray, np.ndarray]:
    log_c = np.log(np.repeat(SIGMAS, len(OPT_Z)))
    log_z = np.log(np.tile(OPT_Z, len(SIGMAS)))
    return log_c, log_z


def test_power_law_fit_recovers_exponents_exactly():
    log_c, log_z = _grid()
    y = 0.3 * np.exp(0.7 * log_c + 1.2 * log_z)
    fit = abr._power_law_mean(log_c, log_z, y)
    assert fit is not None
    np.testing.assert_allclose(fit, [np.log(0.3), 0.7, 1.2], atol=1e-8)


def test_power_law_fit_keeps_cells_where_noise_helped():
    # OLS on log(y) has to drop these; the primary estimator must not.
    log_c, log_z = _grid()
    y = 0.3 * np.exp(0.7 * log_c + 0.5 * log_z)
    y[:5] = [-0.01, 0.0, -0.002, 0.0, -0.03]
    fit = abr._power_law_mean(log_c, log_z, y)
    assert fit is not None and np.all(np.isfinite(fit))
    assert abs(fit[1] - 0.7) < 0.05 and abs(fit[2] - 0.5) < 0.05


def test_data_built_for_gain_returns_gain(tmp_path):
    b = 0.7
    table = abr.noise_currency(_frame(lambda s, z: 0.2 * z ** (1 - b) * s ** b), tmp_path)
    row = table.iloc[0]
    assert row["verdict"] == "gain"
    assert row["loglog_verdict"] == "gain"
    assert abs(row["gain_gap"]) < 0.05
    assert row["beta_z_lo"] > 0.0
    assert row["n_cells"] == len(OPT_Z) * len(SIGMAS)
    # Only the learners enter: the model-free arm is excluded.
    assert row["n_runs"] == len(OPT_Z) * len(SIGMAS) * len(LEARNERS) * SEEDS
    assert (tmp_path / "noise_currency.csv").is_file()


def test_data_built_for_spread_returns_spread(tmp_path):
    table = abr.noise_currency(_frame(lambda s, z: 0.2 * s ** 0.7), tmp_path)
    row = table.iloc[0]
    assert row["verdict"] == "spread"
    assert abs(row["beta_z"]) < 0.05
    assert row["gain_gap_hi"] < 0.0


def test_growth_faster_than_gain_returns_neither(tmp_path):
    # The shape the main sweep shows: cost rises with opt_z faster than GAIN allows.
    table = abr.noise_currency(_frame(lambda s, z: 0.02 * z ** 1.1 * s ** 0.65), tmp_path)
    row = table.iloc[0]
    assert row["verdict"] == "neither"
    assert row["gain_gap_lo"] > 0.0


def test_verdict_needs_both_intervals():
    assert abr._currency_verdict(-0.1, 0.1, 0.2, 0.4) == "spread"
    assert abr._currency_verdict(0.2, 0.4, -0.1, 0.1) == "gain"
    assert abr._currency_verdict(-0.1, 0.4, -0.1, 0.1) == "both (underpowered)"
    assert abr._currency_verdict(0.2, 0.4, 0.2, 0.4) == "neither"
    assert abr._currency_verdict(np.nan, 0.4, 0.2, 0.4) == "not estimable"
