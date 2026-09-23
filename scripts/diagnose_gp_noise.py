"""Does the GP actually know how noisy its observations are?

The whole study measures what feedback error costs Bayesian optimisation. That
number is only a property of the *information* the error destroys if the
surrogate is correctly told how much error there is. It is not:
``run_simulation`` builds ``SingleTaskGP(train_X, train_Y, ...)`` with no
``train_Yvar``, so the observation noise is a free hyperparameter fitted under
BoTorch's default prior -- a prior that pulls it hard toward zero.

This script measures the gap, from the sweep's own logs. For a sample of runs it
refits the identical GP on the recorded ``(X, objective_observed)`` and reads
back the fitted noise standard deviation, in the original objective units, next
to the error that was actually injected. It refits a second time with the noise
prior removed, which separates "the data cannot identify the noise" from "the
prior will not let it".

Why it matters for the conclusions
----------------------------------
* The ``gaussian`` arm is a correctly-specified likelihood only if the fitted
  noise is near the injected noise. If it is an order of magnitude low, the
  measured degradation is partly the cost of a mis-fitted hyperparameter rather
  than the cost of the information loss.
* ``qnei`` is the acquisition designed for noisy observations. It marginalises
  over the posterior it is given. If that posterior believes the noise is a
  tenth of its true size, a null result for ``qnei`` cannot distinguish "noise
  awareness does not help" from "noise awareness was never switched on".

Neither is a defect in this repository -- the same GP serves the data-driven arm,
so the two remain comparable -- but it has to be measured and reported rather
than assumed, and it is the reason to run a fixed-noise contrast arm (pass
``train_Yvar`` at the construction site in ``run_simulation``) before making any
claim about noise-aware acquisitions.

  python scripts/diagnose_gp_noise.py --input-dir output-boba --per-cell 3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bo_sensor_error_simulation as sim  # noqa: E402  (sets thread limits and dtype)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.transforms import Normalize, Standardize  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402
from gpytorch.priors.torch_priors import GammaPrior  # noqa: E402
from tqdm import tqdm  # noqa: E402

import boba_benchmarks as bb  # noqa: E402

warnings.filterwarnings("ignore")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("output-boba"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <input-dir>/analysis.")
    parser.add_argument("--acquisition", type=str, default="logei",
                        help="Only refit runs from this acquisition; the fitted noise is a "
                             "property of the data, and one arm keeps the sample balanced.")
    parser.add_argument("--per-cell", type=int, default=3,
                        help="Runs sampled per (benchmark, error model, magnitude, onset).")
    parser.add_argument("--error-models", type=str, default="gaussian",
                        help="Comma-separated; 'gaussian' is the one whose injected SD is "
                             "exactly known, so it is the only clean comparison.")
    parser.add_argument(
        "--at-iterations", type=str, default="10,20,50",
        help="Refit using only the first n observations, for each n. A surrogate that "
        "has the noise right by the end of a run may still have had it wrong while the "
        "trajectory was being decided, and the early iterations are the ones that "
        "matter.",
    )
    return parser.parse_args(argv)


def _fit_noise_sd(X: np.ndarray, y: np.ndarray, low: np.ndarray, high: np.ndarray,
                  flat_prior: bool) -> float:
    """Fitted observation-noise SD, in the objective's own units."""
    train_X = torch.tensor(X, dtype=torch.double)
    train_Y = torch.tensor(y.reshape(-1, 1), dtype=torch.double)
    bounds = torch.stack([torch.tensor(low, dtype=torch.double),
                          torch.tensor(high, dtype=torch.double)])
    outcome = Standardize(m=1)
    gp = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
        outcome_transform=outcome,
    )
    if flat_prior:
        # BoTorch's default noise prior puts its mass far below the noise levels
        # this study injects. Widening it is the cheapest test
        # of whether the prior, rather than the data, is doing the shrinking.
        gp.likelihood.noise_covar.register_prior(
            "noise_prior", GammaPrior(1.0, 0.01), lambda m: m.noise
        )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    # noise lives in the standardised output space; undo the Standardize scale.
    noise_var = float(gp.likelihood.noise.detach().reshape(-1)[0])
    scale = float(outcome.stdvs.detach().reshape(-1)[0])
    return float(np.sqrt(noise_var) * scale)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (args.input_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {m.strip() for m in args.error_models.split(",") if m.strip()}

    candidates: list[Path] = []
    for path in sorted(args.input_dir.glob("*/bo_sensor_error_*_jittered_*.csv")):
        if f"_{args.acquisition}_seed" not in path.name:
            continue
        if not any(f"_{model}_jit" in path.name for model in wanted):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No {args.acquisition} runs for {sorted(wanted)} under {args.input_dir}."
        )

    # Keep at most --per-cell runs per condition so no benchmark dominates.
    chosen: dict[tuple, list[Path]] = {}
    for path in candidates:
        frame = pd.read_csv(path, nrows=1)
        key = (str(frame["dataset"].iloc[0]), str(frame["error_model"].iloc[0]),
               float(frame["jitter_std"].iloc[0]), int(frame["jitter_iteration"].iloc[0]))
        bucket = chosen.setdefault(key, [])
        if len(bucket) < args.per_cell:
            bucket.append(path)
    sampled = [p for paths in chosen.values() for p in paths]
    print(f"Refitting {len(sampled)} runs across {len(chosen)} conditions.")

    at_iterations = sorted({int(v) for v in args.at_iterations.split(",") if v.strip()})
    rows: list[dict] = []
    for path in tqdm(sampled, desc="refit", unit="run", disable=not sys.stderr.isatty(),
                     mininterval=10.0):
        frame = pd.read_csv(path)
        name = str(frame["dataset"].iloc[0])
        spec = bb.BENCHMARKS[name]
        params = str(frame["param_columns"].iloc[0]).split(",")
        X = frame[params].to_numpy(dtype=float)
        y = frame["objective_observed"].to_numpy(dtype=float)

        onset = int(frame["jitter_iteration"].iloc[0])
        injected = float(frame["jitter_std"].iloc[0])

        for n in at_iterations:
            if n < 4 or n > len(frame):
                continue
            prefix = frame.iloc[:n]
            n_noisy = int((prefix["iteration"] > onset).sum())
            if n_noisy == 0:
                continue
            # Marginal SD of the error across the training set the GP actually
            # had at that point: the pre-onset observations are exact, so it
            # sees a mixture and a homoskedastic fit must compromise between them.
            marginal = injected * np.sqrt(n_noisy / n)
            empirical = float(np.std(prefix["error_magnitude"].to_numpy(dtype=float), ddof=1))
            Xn, yn = X[:n], y[:n]
            try:
                default = _fit_noise_sd(Xn, yn, spec.bounds_low, spec.bounds_high,
                                        flat_prior=False)
            except Exception:
                default = float("nan")
            try:
                flat = _fit_noise_sd(Xn, yn, spec.bounds_low, spec.bounds_high, flat_prior=True)
            except Exception:
                flat = float("nan")

            rows.append({
                "dataset": name,
                "dim": spec.dim,
                "error_model": str(frame["error_model"].iloc[0]),
                "jitter_std": injected,
                "jitter_iteration": onset,
                "seed": int(frame["seed"].iloc[0]),
                "n_observations": int(n),
                "noisy_fraction": n_noisy / n,
                "injected_marginal_sd": marginal,
                "empirical_error_sd": empirical,
                "fitted_noise_sd_default_prior": default,
                "fitted_noise_sd_flat_prior": flat,
                "ratio_default": default / marginal if marginal else np.nan,
                "ratio_flat": flat / marginal if marginal else np.nan,
            })

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "gp_noise_diagnostic_runs.csv", index=False)

    summary = (
        table.groupby(["error_model", "jitter_std", "jitter_iteration", "n_observations"])
        .agg(injected=("injected_marginal_sd", "mean"),
             empirical=("empirical_error_sd", "mean"),
             fitted_default=("fitted_noise_sd_default_prior", "median"),
             fitted_flat=("fitted_noise_sd_flat_prior", "median"),
             ratio_default=("ratio_default", "median"),
             ratio_flat=("ratio_flat", "median"),
             runs=("dataset", "count"))
        .reset_index()
    )
    summary.to_csv(output_dir / "gp_noise_diagnostic.csv", index=False)

    lines = ["GP NOISE DIAGNOSTIC", "=" * 72, "",
             "Fitted observation-noise SD against the error actually injected, in the",
             "objective's own (standardised) units. A ratio far below 1 means the",
             "surrogate does not know its observations are noisy.", ""]
    lines.append(f"  {'error SD':>9s} {'onset':>6s} {'obs':>5s} {'injected':>9s} "
                 f"{'fitted':>9s} {'ratio':>7s} {'flat':>8s} {'ratio':>7s} {'runs':>5s}")
    for _, row in summary.iterrows():
        lines.append(
            f"  {row['jitter_std']:>9.3g} {int(row['jitter_iteration']):>6d} "
            f"{int(row['n_observations']):>5d} {row['injected']:>9.3f} "
            f"{row['fitted_default']:>9.3f} {row['ratio_default']:>7.2f} "
            f"{row['fitted_flat']:>8.3f} {row['ratio_flat']:>7.2f} {int(row['runs']):>5d}"
        )
    lines += [
        "",
        "'fitted' uses BoTorch's shipped noise prior, as the sweep does; 'flat fit'",
        "re-runs the same fit with a much wider prior. If the flat fit recovers the",
        "injected level and the shipped one does not, the shrinkage is the prior's,",
        "not the data's -- and a fixed-noise arm (train_Yvar at the SingleTaskGP",
        "construction site in run_simulation) is the contrast needed before any",
        "claim about noise-aware acquisitions.",
    ]
    report = "\n".join(lines)
    (output_dir / "gp_noise_diagnostic.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
