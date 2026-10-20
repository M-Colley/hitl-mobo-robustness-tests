"""Shared response timing for logged ratings and surrogate-driven proposals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def response_step(run: pd.DataFrame, onset: int) -> tuple[int, int] | None:
    """Last proposal before a rating-dependent response and that response.

    Observation onset+1 can affect the next proposal, once initialization has
    ended. Random and Sobol search never react to ratings. New logs name their
    initial design size; legacy logs can identify it by their first GP fit.
    Without either record the historical onset+1 -> onset+2 window is retained.
    """
    if "acquisition" in run and str(run["acquisition"].iloc[0]) in {"random", "sobol"}:
        return None
    initial = 0
    if "initial_samples" in run and run["initial_samples"].notna().any():
        counts = run["initial_samples"].dropna().unique()
        if len(counts) != 1 or float(counts[0]) < 0 or not float(counts[0]).is_integer():
            raise ValueError("A run must record one non-negative integer initial_samples value.")
        initial = int(counts[0])
    elif "fit_time_sec" in run:
        fit = pd.to_numeric(run["fit_time_sec"], errors="coerce").to_numpy(dtype=float)
        fitted = run.loc[np.isfinite(fit) & (fit > 0), "iteration"]
        if len(fitted):
            initial = int(fitted.min()) - 1
    end = max(int(onset) + 2, initial + 1)
    return end - 1, end
