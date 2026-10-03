"""The 22-32% headline with the noise anchor's own uncertainty propagated.

The headline interpolates the deployed-loss curve (error from the first rating,
ten model-based acquisitions, twenty landscapes) at each study's nugget. The
nugget has a participant-bootstrap interval (output-boba/analysis/review/
archival_error_processes.csv, the rows the paper uses), so the loss at the
measured noise has two sources of uncertainty: which landscapes, and how noisy
the raters are. This maps the nugget's interval through the curve, log-linearly
between the grid magnitudes and linearly from zero error to 0.05 SD, and joins
it with a landscape bootstrap of the curve: the reported interval runs from the
landscape-bootstrap lower bound at the nugget's lower end to the upper bound at
its upper end, a conservative union of the two.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import boba_benchmarks as bb  # noqa: E402

GRID = np.array([0.05, 0.25, 1.0, 5.0])
BOOT = 4000
# (dataset, data basis) of the anchor rows the paper uses
ROWS = {"ehmi": "pipeline", "opticarvis": "scale_consistent", "provoice": "sign_corrected"}


def curve_at(values: np.ndarray, sigma: float) -> float:
    """values: deployed loss at GRID for one landscape (or a mean)."""
    if sigma <= 0:
        return 0.0
    if sigma < GRID[0]:
        return float(values[0] * sigma / GRID[0])
    if sigma >= GRID[-1]:
        return float(values[-1])
    j = int(np.searchsorted(GRID, sigma))
    if GRID[j] == sigma:
        return float(values[j])
    lo, hi = GRID[j - 1], GRID[j]
    w = (np.log(sigma) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return float(values[j - 1] + w * (values[j] - values[j - 1]))


def main() -> None:
    stats = bb.load_stats()
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if isinstance(v, dict) and "opt_z" in v}
    cells = pd.read_csv(REPO / "output-boba" / "analysis" / "cell_means.csv")
    cells = cells[~cells.acquisition.isin(["random", "sobol"])]
    cells = cells.assign(deployed=cells.inference_excess / cells.dataset.map(opt_z))
    per = (cells[cells.jitter_iteration == 0].groupby(["dataset", "jitter_std"])["deployed"].mean()
           .unstack()[list(GRID)].to_numpy())

    arch = pd.read_csv(REPO / "output-boba" / "analysis" / "review" / "archival_error_processes.csv")
    arch = arch[arch.quantity == "noise_nn_close_over_sigma_f"]
    rng = np.random.default_rng(20260926)
    idx = rng.integers(0, len(per), (BOOT, len(per)))
    boot_curves = per[idx].mean(axis=1)            # BOOT x 4
    mean_curve = per.mean(axis=0)
    for name, basis in ROWS.items():
        row = arch[(arch.dataset == name) & (arch.data_basis == basis)].iloc[0]
        # express the nugget in the sigma_f of the oracle the headline uses (output/noise_anchor.csv);
        # only opticarvis differs (0.258 in the archival table, 0.156 for the reselected oracle)
        anchor = pd.read_csv(REPO / "output" / "noise_anchor.csv").set_index("dataset")
        factor = float(row.sigma_f_used) / float(anchor.loc[name, "sigma_f"])
        est, lo, hi = (float(row.estimate) * factor, float(row.ci_low) * factor, float(row.ci_high) * factor)
        at = lambda s, curves: np.array([curve_at(c, s) for c in curves])  # noqa: E731
        point = curve_at(mean_curve, est)
        land_lo, land_hi = np.percentile(at(est, boot_curves), [2.5, 97.5])
        low = np.percentile(at(lo, boot_curves), 2.5)
        high = np.percentile(at(hi, boot_curves), 97.5)
        print(f"{name:11s} nugget {est:.3f} [{lo:.3f}, {hi:.3f}]  deployed loss {point:.4f}  "
              f"landscapes only [{land_lo:.4f}, {land_hi:.4f}]  with the nugget's interval [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
