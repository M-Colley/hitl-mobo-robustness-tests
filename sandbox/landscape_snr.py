"""Characterize the optimization landscape each dataset presents to BO-under-noise.

The error-robustness experiment injects SYNTHETIC noise (jitter_std) onto the
oracle's prediction f(x). A dataset therefore contributes to the experiment only
through the SHAPE of f(x): its dynamic range (how far the optimum sits above a
typical design) relative to the noise levels applied. Where signal >> noise, BO
can converge and noise merely slows it; where signal ~ noise, the ranking of
candidates is dominated by noise and "robustness" becomes degenerate.

This script fits the same oracle family on every dataset's composite, samples the
design space, and reports the signal spread vs the empirically-calibrated noise SD,
so we can see whether the external datasets occupy a DIFFERENT signal-to-noise
regime than the automotive trio (which is what would make them contribute).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import bo_sensor_error_simulation as bo_sim  # noqa: E402
import calibrate_noise_from_data as cal  # noqa: E402

SEED = 7
CONFIG = REPO / "datasets-extended.json"
N_SAMPLE = 20_000


# Per-rater intra-/inter-rater noise SDs on each composite scale, from the
# rigorous calibration (output/noise_calibration.csv for the automotive trio;
# the external-dataset feasibility work for ashrae/nisqa, which used the correct
# subject key). The generic NN-nugget cannot be recomputed here for the
# single-file external datasets (no per-rater files -> all rows pool into one
# group), so we look these up rather than mis-estimating them.
NOISE_SD = {
    "ehmi": (0.269, "within-rater NN (composite 1-7)"),
    "opticarvis": (1.254, "within-rater NN (composite 1-9)"),
    "provoice": (0.651, "within-rater NN upper bound (composite)"),
    "ashrae": (0.98, "within-subject NN (ThermalComfort 1-6)"),
    "nisqa_sim": (0.74, "single-vote inter-rater SD (MOS 1-5)"),
}


def composite_noise_sd(dataset):
    return NOISE_SD[dataset.name]


def main() -> None:
    datasets = bo_sim.parse_dataset_configs(None, CONFIG, REPO / ".dataset_cache")
    rng = np.random.default_rng(SEED)

    print(
        f"{'dataset':<12}{'dims':>5}{'y_opt':>8}{'f_p50':>8}{'f_std':>8}"
        f"{'range95':>9}{'noiseSD':>8}{'SNR_std':>9}{'SNR_top':>9}  noise_basis"
    )
    print("-" * 96)
    for d in datasets:
        cols = d.objective_map["composite"]
        df = bo_sim.load_observations(d, "composite")
        X = df[d.param_columns].to_numpy(dtype=float)
        y = bo_sim.compute_objective(df, cols, False, None).to_numpy(dtype=float)

        model = ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=1
        )
        model.fit(X, y)

        low = X.min(axis=0)
        high = X.max(axis=0)
        Xs = rng.uniform(low, high, size=(N_SAMPLE, len(d.param_columns)))
        fs = model.predict(Xs)

        y_opt = float(fs.max())
        f_p50 = float(np.median(fs))
        f_std = float(fs.std())
        # range the optimizer climbs through, robust to tail outliers:
        range95 = float(np.percentile(fs, 95) - np.percentile(fs, 5))
        top_gap = float(y_opt - np.percentile(fs, 95))  # final-approach precision needed

        noise_sd, basis = composite_noise_sd(d)
        snr_std = f_std / noise_sd if noise_sd else float("nan")
        snr_top = top_gap / noise_sd if noise_sd else float("nan")

        print(
            f"{d.name:<12}{len(d.param_columns):>5}{y_opt:>8.2f}{f_p50:>8.2f}"
            f"{f_std:>8.3f}{range95:>9.3f}{noise_sd:>8.3f}{snr_std:>9.2f}{snr_top:>9.2f}"
            f"  {basis}"
        )

    print()
    print("SNR_std = std(f over design space) / intra-rater noise SD")
    print("SNR_top = (y_opt - 95th pct of f) / noise SD  [precision needed near the optimum]")
    print("Higher SNR => noise must be larger to corrupt candidate ranking => more 'room'")
    print("for robustness differences between acquisitions to appear.")


if __name__ == "__main__":
    main()
