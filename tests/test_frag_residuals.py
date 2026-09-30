"""The out-of-sample fit of Appendix C.2 (scripts/review_checks/frag_residuals.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "review_checks"))

pytest.importorskip("statsmodels")
import frag_residuals as fr  # noqa: E402


def _cells(noise: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(8):
        for model in ("gaussian", "bias"):
            for onset in (0, 20):
                x = rng.normal()
                rows.append({"dataset": f"l{d}", "error_model": model, "jitter_iteration": onset,
                             "frag_at_c": x, "excess_sd": 2.0 * x + 0.5 * (onset == 20) + noise * rng.normal()})
    return pd.DataFrame(rows)


def test_leave_one_landscape_out_r2_is_one_for_an_exact_linear_relation():
    cells = _cells(noise=0.0)
    assert fr.loo_r2(cells, ["frag_at_c"]) == pytest.approx(1.0, abs=1e-9)


def test_leave_one_landscape_out_r2_is_below_the_in_sample_fit_under_noise():
    cells = _cells(noise=1.0, seed=3)
    model, _ = fr.fit(cells, ["frag_at_c"])
    assert fr.loo_r2(cells, ["frag_at_c"]) < model.rsquared


def test_the_indicator_only_model_predicts_about_nothing_out_of_sample():
    cells = _cells(noise=1.0, seed=5)
    assert fr.loo_r2(cells, []) < 0.2
