"""Cross-benchmark synthesis for the known-function robustness sweep.

``evaluate_research_question.py`` answers "which acquisition is most robust on
THIS landscape". Run over 20 landscapes whose geometry is known in advance, a
second question opens up that the data-driven arm cannot ask at all:

    what property of a landscape, and what property of an error process,
    decides how much feedback error actually costs?

This script answers it. Four analyses:

1. **Floor check.** ``random`` and ``sobol`` choose candidates without looking at
   any observation, so a noisy run must visit exactly the same designs as its
   baseline and its excess regret must be identically zero. Any deviation means
   noise is leaking into the candidate stream, which would invalidate every
   other number here. This is a positive control, not a formality.

2. **The currency of "how big is this error".** An error of 0.5 landscape SDs is
   trivial on shekel, whose optimum stands 57 SDs above a random design, and
   catastrophic on branin, where it stands 1.25 SDs above. Two candidate scales
   therefore exist -- error relative to the landscape's spread, and error
   relative to the *achievable gain* -- and the suite spans a 45x range of the
   ratio between them, which is enough to tell them apart. Fitting the power
   law ``E[excess] = A * sigma_e**beta_c * opt_z**beta_z`` to landscape x
   magnitude cell means decides it: beta_z = 0 means spread is the currency,
   beta_c + beta_z = 1 means achievable gain is. Those restrictions belong to
   the exponents of a power law in the mean. An earlier version regressed raw
   excess on log10 predictors and read GAIN as beta_z = 1 - beta_c there, where
   it does not follow; see ``noise_currency``.

3. **Which landscapes are fragile.** Regression of the excess on the pre-measured
   descriptors (dimension, sparsity, ruggedness, skew, tail weight), with
   standard errors clustered by benchmark because the 20 seeds within a
   benchmark are not independent observations of anything.

4. **Which acquisitions are robust**, ranked per condition with the benchmarks as
   blocks -- a Friedman test over 20 landscapes is a far stronger design than the
   same test over a handful of seeds on one dataset.

Two response variables, and they are not interchangeable
--------------------------------------------------------
``excess_sd`` -- post-onset per-iteration excess simple regret in landscape
standard deviations. Every objective is standardised before any error is
injected, so this is already comparable across a suite whose raw outputs span
nearly seven orders of magnitude. **This is the response for every regression here.**

``fragility = excess_sd / opt_z`` -- the same quantity as a fraction of the gain
BO is trying to capture. Better for reporting a magnitude ("noise cost you a
third of the available improvement"), and used for the acquisition rankings,
where ranks within a benchmark are unaffected by a benchmark-constant divisor.
It must never appear as the response in a model containing opt_z or anything
collinear with it: dividing by opt_z and then regressing on log opt_z
manufactures the very coefficient the currency test is meant to estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as smapi  # noqa: E402
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import boba_benchmarks as bb  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

FRAGILITY_NUMERATOR = "auc_simple_regret_excess_true_postonset_per_iter"
ABSOLUTE_METRIC = "auc_simple_regret_true_postonset_per_iter_jitter"
INFERENCE_METRIC = "final_inference_simple_regret_excess_true"

# Model-free acquisitions: their candidate sequence cannot depend on what was
# observed, so they are the floor AND the internal control.
MODEL_FREE = ("random", "sobol")

# Pre-specified, deliberately small. log_sparsity is ~-0.95 correlated with
# log_opt_z across this suite (a landscape whose optimum stands many SDs up is
# necessarily one where almost nothing is near it) and skew is largely a
# restatement of tail weight, so both are dropped from the primary model and
# kept only in the sensitivity fit below. The VIF table records the choice.
PRIMARY_DESCRIPTORS = ["dim", "log_opt_z", "ruggedness", "log_tail_ratio"]
# log_sparsity is the optimum-anchored fraction, floored at half a sample so the
# needle landscapes (where it is exactly zero) stay loggable. It sits only in the
# sensitivity model: it correlates about -0.9 with log_opt_z across the suite.
ALL_DESCRIPTORS = PRIMARY_DESCRIPTORS + ["log_sparsity", "skew"]
DESCRIPTORS = PRIMARY_DESCRIPTORS
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260906


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("output-boba"))
    parser.add_argument("--output-dir", type=Path, default=Path("output-boba/analysis"))
    parser.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Restrict to these seeds (comma-separated). Use it to read a sweep that is "
        "still running: the task order fills one seed across every benchmark before "
        "starting the next, so a completed seed is already a balanced cross-benchmark "
        "panel while the directory as a whole is not.",
    )
    parser.add_argument(
        "--floor-tolerance",
        type=float,
        default=1e-9,
        help="Largest |excess regret| a model-free acquisition may show before the "
        "floor check is treated as failed.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_paired(input_dir: Path) -> pd.DataFrame:
    """Concatenate every benchmark's per-run paired metrics."""
    files = sorted(input_dir.glob("*/evaluation/paired_excess_metrics.csv"))
    if not files:
        raise FileNotFoundError(
            f"No per-benchmark evaluation outputs under {input_dir}. Run\n"
            f"  python scripts/evaluate_research_question.py --input-dir {input_dir}/<benchmark> "
            f"--output-dir {input_dir}/<benchmark>/evaluation\n"
            f"for each benchmark first (run_boba_pipeline.ps1 does this)."
        )
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} paired runs from {len(files)} benchmarks.")
    return df


def attach_landscape(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    table = pd.DataFrame(stats).T.reset_index(drop=True)
    keep = ["name", "dim", "opt_z", "sparsity_10pct", "sparsity_sampled", "ruggedness",
            "skew", "tail_ratio", "excess_kurtosis", "std", "mean", "y_opt", "y_best",
            "y_best_source"]
    table = table[[c for c in keep if c in table.columns]].rename(columns={"name": "dataset"})
    for column in ("dim", "opt_z", "sparsity_10pct", "sparsity_sampled", "ruggedness",
                   "skew", "tail_ratio", "excess_kurtosis", "std", "mean", "y_opt", "y_best"):
        if column in table:
            table[column] = pd.to_numeric(table[column])

    merged = df.merge(table, on="dataset", how="left", validate="many_to_one")
    missing = merged["opt_z"].isna()
    if missing.any():
        raise ValueError(
            f"No landscape statistics for: {sorted(merged.loc[missing, 'dataset'].unique())}"
        )

    # THE REGRESSION RESPONSE. Post-onset per-iteration excess simple regret, in
    # landscape standard deviations -- the objective is standardised, so this is
    # already on a common scale across the suite and needs no further division.
    merged["excess_sd"] = merged[FRAGILITY_NUMERATOR]

    # A REPORTING quantity only: the same thing as a fraction of the gain BO is
    # trying to capture. It must NEVER be the response in a model that has
    # opt_z (or anything ~collinear with it, such as sparsity or skew) on the
    # right-hand side: dividing by opt_z and then regressing on log opt_z
    # manufactures the coefficient the currency test is supposed to estimate. If
    # the raw excess were perfectly landscape-invariant -- the "spread" answer --
    # the normalised response alone would still deliver the "gain" answer, by
    # arithmetic. Ranks within a benchmark are unaffected (the divisor is
    # constant there), so the acquisition rankings may use either.
    merged["fragility"] = merged["excess_sd"] / merged["opt_z"]
    merged["absolute_loss"] = merged[ABSOLUTE_METRIC] / merged["opt_z"]
    merged["log_noise"] = np.log10(merged["jitter_std"])
    # sigma_e expressed in units of the achievable gain rather than the spread.
    merged["log_noise_per_gain"] = merged["log_noise"] - np.log10(merged["opt_z"])
    merged["log_opt_z"] = np.log10(merged["opt_z"])
    # A floor keeps the two exactly-zero sparsities (ackley, shekel, michalewicz,
    # hicks_law) finite; it is below the smallest non-zero value in the suite.
    # Optimum-anchored sparsity, floored at half a sample. The sample-anchored
    # variant avoids the exact zeros but measures the wrong thing on a needle
    # landscape: with no Sobol point near the spike it reports how broad the
    # BACKGROUND's maxima are, and moves in the opposite direction to the real
    # sparsity. A floor of 0.5/65536 reads as "fewer than one point in 65,536".
    merged["log_sparsity"] = np.log10(np.maximum(merged["sparsity_10pct"], 0.5 / 65536))
    merged["log_tail_ratio"] = np.log10(merged["tail_ratio"])

    # The mediator: one-shot selection loss at THIS condition's noise level,
    # computed from the landscape alone with no BO involved
    # (boba_benchmarks.selection_fragility). If the sequential result tracks it,
    # the landscape question has a mechanism rather than a correlation.
    def _frag(row: pd.Series) -> float:
        entry = stats.get(row["dataset"], {})
        return float(entry.get(f"frag_{row['jitter_std']:g}", np.nan))

    merged["frag_at_c"] = merged.apply(_frag, axis=1)
    if merged["frag_at_c"].isna().all():
        print("NOTE: no matching one-shot fragility entries; the mediator model is skipped.",
              file=sys.stderr)
    return merged


# ---------------------------------------------------------------------------
# 1. Floor check
# ---------------------------------------------------------------------------


# Input-error models corrupt the DESIGN rather than the rating. Kept in step
# with bo_sensor_error_simulation.INPUT_ERROR_CHOICES by
# tests/test_error_model_labels.py, and duplicated here so that the analysis
# does not have to import torch.
INPUT_ERROR_MODELS = ("slip", "misclick")


def _has_input_error(label: object) -> bool:
    """True for "slip", "misclick", or a combined label such as "bias+slip"."""
    return any(part in INPUT_ERROR_MODELS for part in str(label).split("+"))


def floor_check(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Model-free acquisitions must be exactly unaffected by observation noise.

    That holds for an error in the RATING, which a floor never reads. It does
    not hold for an error in the DESIGN: a floor evaluates the wrong points too,
    so its excess is the geometric cost of the slip -- the quantity the
    input-error arm exists to measure, not a leak. Those runs are reported and
    left out of the assertion, which is otherwise unchanged.
    """
    floor = df[df["acquisition"].isin(MODEL_FREE)]
    if floor.empty:
        print("FLOOR CHECK: skipped (no model-free acquisitions in the sweep).")
        return pd.DataFrame()
    report = (
        floor.groupby(["acquisition", "dataset"])
        .agg(
            max_abs_excess_auc=("auc_simple_regret_excess_true", lambda s: float(np.abs(s).max())),
            max_abs_excess_final=("final_simple_regret_excess_true", lambda s: float(np.abs(s).max())),
            runs=("run_id", "count"),
        )
        .reset_index()
    )
    design = (
        floor["error_model"].map(_has_input_error)
        if "error_model" in floor.columns
        else pd.Series(False, index=floor.index)
    )
    if design.any():
        cost = float(np.abs(floor.loc[design, "auc_simple_regret_excess_true"]).max())
        print(
            f"FLOOR CHECK: {int(design.sum()):,} model-free runs under an input error are "
            f"reported, not asserted. A floor evaluates the wrong points too, so their "
            f"excess (up to {cost:.3e}) is the geometric cost of the error, not a leak."
        )
    asserted = floor[~design]
    if asserted.empty:
        return report
    worst = float(np.abs(asserted["auc_simple_regret_excess_true"]).max())
    if worst > tolerance:
        offenders = report[report["max_abs_excess_auc"] > tolerance]
        # This is the negative control the whole design rests on: random and sobol
        # never read an observation, so a non-zero excess means observation noise
        # has reached the candidate stream and every paired number downstream is
        # suspect. Printing and continuing let a broken run produce a full set of
        # tables, so it raises.
        raise ValueError(
            f"FLOOR CHECK FAILED: model-free acquisitions show excess regret up to "
            f"{worst:.3e} (tolerance {tolerance:g}). Observation noise is reaching the "
            f"candidate stream.\n{offenders.to_string(index=False)}"
        )
    else:
        print(f"FLOOR CHECK PASSED: model-free excess regret <= {worst:.3e}.")
    return report


def balance_check(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Every cell must carry the same seeds.

    ``--resume`` matches on filename, and the filename does not encode the seed
    LIST, so an interrupted sweep relaunched with a different ``--seeds`` leaves
    the old seeds' runs on disk. The evaluator globs the directory
    unconditionally, so those extra runs are silently folded into some cells and
    not others -- and the ranking pivot then drops whole benchmarks with
    ``dropna(how="any")``. An unbalanced panel is not a small problem here: it
    biases exactly the cell means the rankings compare.
    """
    counts = (
        df.groupby(["dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration"])
        ["seed"].nunique().reset_index(name="n_seeds")
    )
    counts.to_csv(output_dir / "seed_balance.csv", index=False)
    modal = int(counts["n_seeds"].mode().iloc[0]) if len(counts) else 0
    offenders = counts[counts["n_seeds"] != modal]
    if len(offenders):
        worst = offenders.groupby("dataset")["n_seeds"].agg(["min", "max", "count"])
        print(
            f"WARNING: {len(offenders)} of {len(counts)} cells do not carry the modal "
            f"{modal} seeds. Cell means are not comparable across acquisitions until this "
            f"is resolved (delete the stale runs, or re-run the missing ones).\n"
            f"{worst.to_string()}",
            file=sys.stderr,
        )
    else:
        print(f"BALANCE CHECK PASSED: every cell carries {modal} seeds.")
    return counts


def beneficial_regime(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Where feedback error HELPED.

    Not a curiosity: on a multimodal landscape, noise after convergence is an
    escape mechanism, so excess regret can be reliably negative. Every test here
    is two-sided, but a stratum that moves the other way has to be named rather
    than averaged into a smaller positive mean.
    """
    learners = df[~df["acquisition"].isin(MODEL_FREE)]
    cells = (
        learners.groupby(["dataset", "error_model", "jitter_std", "jitter_iteration"])["excess_sd"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    cells["dz"] = cells["mean"] / cells["std"].replace(0.0, np.nan)
    helpful = cells[(cells["mean"] < 0) & (cells["dz"] < -0.2)].sort_values("dz")
    helpful.to_csv(output_dir / "beneficial_regime.csv", index=False)
    return helpful


def headroom_screen(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """How much a benchmark had to lose in the first place.

    On an easy landscape BO reaches the optimum in the noise-free baseline and
    stays there, so there is nothing for feedback error to destroy and a null is
    guaranteed by construction rather than earned. This measures that directly:
    the best learner's noise-free regret against the model-free floor's. A
    benchmark near zero headroom cannot produce a detectable effect at any error
    level, and a null there says nothing about robustness.
    """
    baseline = (
        df.groupby(["dataset", "acquisition"])["auc_simple_regret_true_baseline"]
        .mean().reset_index()
    )
    rows = []
    for name, block in baseline.groupby("dataset"):
        floors = block[block["acquisition"].isin(MODEL_FREE)]
        learners = block[~block["acquisition"].isin(MODEL_FREE)]
        if floors.empty or learners.empty:
            continue
        floor_auc = float(floors["auc_simple_regret_true_baseline"].mean())
        best_auc = float(learners["auc_simple_regret_true_baseline"].min())
        rows.append({
            "dataset": name,
            "floor_auc_baseline": floor_auc,
            "best_learner_auc_baseline": best_auc,
            "best_learner": learners.loc[
                learners["auc_simple_regret_true_baseline"].idxmin(), "acquisition"],
            "headroom": float(1.0 - best_auc / floor_auc) if floor_auc else np.nan,
        })
    if not rows:
        # An arm that runs only model-based acquisitions has no floor to measure
        # headroom against. That is fine -- the screen belongs to the main sweep,
        # and the arm inherits its verdict -- but it must not abort the analysis.
        print("HEADROOM: skipped (this arm runs no model-free acquisitions).")
        empty = pd.DataFrame(columns=["dataset", "floor_auc_baseline",
                                      "best_learner_auc_baseline", "best_learner",
                                      "headroom", "ceiling_free"])
        empty.to_csv(output_dir / "headroom_screen.csv", index=False)
        return empty
    table = pd.DataFrame(rows).sort_values("headroom")
    table["ceiling_free"] = table["headroom"] < 0.10
    table.to_csv(output_dir / "headroom_screen.csv", index=False)
    return table


def bias_onset_control(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """A constant offset present from the first observation must cost nothing.

    The GP standardises its training targets and the incumbent is a posterior
    mean, so adding the same constant to every observation leaves the
    standardised targets, the acquisition surface and the chosen candidate
    identical -- and regret is computed from TRUE values. The `bias` arm at onset
    0 should therefore differ from the `gaussian` arm at onset 0 only by its
    noise draw. If it differs systematically, something in the pipeline is
    reacting to the offset that should not be.
    """
    subset = df[(df["jitter_iteration"] == 0)
                & (df["error_model"].isin(["gaussian", "bias"]))
                & (~df["acquisition"].isin(MODEL_FREE))]
    if subset.empty:
        return pd.DataFrame()
    table = (
        subset.groupby(["error_model", "jitter_std"])["excess_sd"]
        .agg(["mean", "std", "count"]).reset_index()
        .pivot(index="jitter_std", columns="error_model")
    )
    table.columns = ["_".join(map(str, c)) for c in table.columns]
    table = table.reset_index()
    table["difference"] = table.get("mean_bias", np.nan) - table.get("mean_gaussian", np.nan)
    table.to_csv(output_dir / "bias_onset0_control.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# 2. Which currency measures "how big is this error"
# ---------------------------------------------------------------------------


def _currency_verdict(z_lo: float, z_hi: float, gap_lo: float, gap_hi: float) -> str:
    """Which restriction the bootstrap intervals are compatible with.

    SPREAD constrains beta_z to 0; GAIN constrains beta_c + beta_z - 1 to 0.
    Each is judged on the interval of the quantity it constrains, so the GAIN
    verdict carries the uncertainty in beta_c as well as in beta_z.
    """
    if not np.all(np.isfinite([z_lo, z_hi, gap_lo, gap_hi])):
        return "not estimable"
    spread_ok = z_lo <= 0.0 <= z_hi
    gain_ok = gap_lo <= 0.0 <= gap_hi
    if spread_ok and not gain_ok:
        return "spread"
    if gain_ok and not spread_ok:
        return "gain"
    if spread_ok and gain_ok:
        return "both (underpowered)"
    return "neither"


def _power_law_mean(log_c: np.ndarray, log_z: np.ndarray, y: np.ndarray,
                    max_iter: int = 200, tol: float = 1e-10) -> np.ndarray | None:
    """Fit E[y] = exp(a + b_c log c + b_z log z), returning (a, b_c, b_z).

    Quasi-Poisson estimating equations, X'(y - mu) = 0, solved by Newton's
    method with step-halving on the quasi-likelihood sum(y * eta - exp(eta)).
    That objective is concave in the coefficients, so the root is unique when
    it exists. Two properties make this the primary estimator:

    * It keeps cells whose mean excess is zero or negative. Noise helps a cell
      by chance often enough at the late onset that OLS on log(y), which has to
      drop those cells, loses up to 12 of 80 in the main sweep.
    * Its implied variance grows with the mean, so the landscapes with the
      tallest optima -- and the largest raw excess -- do not dominate the fit
      the way they would under unweighted least squares on the raw scale.

    Returns None when the solve fails or does not converge.
    """
    X = np.column_stack([np.ones_like(log_c), log_c, log_z])
    positive = y > 0
    if positive.sum() > X.shape[1]:
        beta = np.linalg.lstsq(X[positive], np.log(y[positive]), rcond=None)[0]
    else:
        beta = np.array([np.log(max(float(np.mean(y)), 1e-12)), 0.0, 0.0])

    def quasi(b: np.ndarray) -> float:
        eta = X @ b
        return float(np.sum(y * eta - np.exp(eta)))

    current = quasi(beta)
    for _ in range(max_iter):
        mu = np.exp(X @ beta)
        try:
            step = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        t = 1.0
        while True:
            trial = beta + t * step
            value = quasi(trial)
            if np.isfinite(value) and value >= current - 1e-12:
                break
            t *= 0.5
            if t < 1e-10:
                return None
        beta, current = trial, value
        if np.max(np.abs(t * step)) < tol:
            return beta
    return None


def _log_ols(log_c: np.ndarray, log_z: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """OLS of log(y) on the positive cells: the estimator check, not the primary."""
    positive = y > 0
    if positive.sum() <= 3:
        return None
    X = np.column_stack([np.ones(int(positive.sum())), log_c[positive], log_z[positive]])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    return np.linalg.lstsq(X, np.log(y[positive]), rcond=None)[0]


def _currency_fits(cells: pd.DataFrame) -> np.ndarray:
    """[b_c, b_z] from the power-law mean, then [b_c, b_z] from log-OLS; NaN if unfit."""
    log_c = np.log(cells["jitter_std"].to_numpy(dtype=float))
    log_z = np.log(cells["opt_z"].to_numpy(dtype=float))
    y = cells["excess_sd"].to_numpy(dtype=float)
    out: list[float] = []
    for fit in (_power_law_mean(log_c, log_z, y), _log_ols(log_c, log_z, y)):
        out.extend([np.nan, np.nan] if fit is None else [float(fit[1]), float(fit[2])])
    return np.array(out)


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 100:
        return np.nan, np.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _bca_interval(point: float, draws: np.ndarray, jack: np.ndarray,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval.

    The landscape bootstrap of the currency fit is strongly right-skewed and
    leans on one landscape (shekel is the only one with opt_z above 20), so
    the plain percentile interval sits well to the right of its point
    estimate. BCa corrects the median bias (z0, from the share of draws below
    the point) and the skew (a, from the jackknife over landscapes).
    """
    from scipy.stats import norm

    draws = draws[np.isfinite(draws)]
    jack = jack[np.isfinite(jack)]
    if len(draws) < 100 or len(jack) < 3 or not np.isfinite(point):
        return np.nan, np.nan
    below = np.clip((draws < point).mean(), 1.0 / len(draws), 1.0 - 1.0 / len(draws))
    z0 = norm.ppf(below)
    centred = jack.mean() - jack
    denom = 6.0 * (centred ** 2).sum() ** 1.5
    accel = 0.0 if denom == 0 else (centred ** 3).sum() / denom

    def adjusted(q: float) -> float:
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / (1.0 - accel * (z0 + z))))

    lo, hi = np.percentile(draws, [100 * adjusted(alpha / 2), 100 * adjusted(1 - alpha / 2)])
    return float(lo), float(hi)


def noise_currency(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Is the cost of an error set by its size relative to spread, or to gain?

    Fitted to landscape x magnitude cell means of the post-onset excess regret
    (landscape SDs), averaged over model-based acquisitions and seeds, as a
    power law in the MEAN:

        E[excess] = A * c**beta_c * opt_z**beta_z

    Two rival readings, stated as restrictions on the exponents:

      SPREAD  the landscape's own standard deviation is the whole story, so once
              the objective is standardised the same c costs the same anywhere:
              beta_z = 0.
      GAIN    what matters is the error relative to the achievable improvement:
              excess = opt_z * h(c / opt_z). For a power-law h that is
              beta_c + beta_z = 1.

    The GAIN restriction is a statement about exponents. An earlier version of
    this function regressed the RAW excess on log10(c) and log10(opt_z) and
    compared beta_z with 1 - beta_c; on that scale the coefficients are slopes,
    not exponents, and the restriction does not follow from GAIN, so its "gain"
    verdicts meant nothing. Intervals come from resampling benchmarks, and each
    restriction is judged on the interval of the quantity it constrains. OLS of
    log(excess) on the positive cells is reported alongside as a check that the
    verdict does not depend on the estimator.

    Intervals are BCa (see ``_bca_interval``): the plain percentile interval is
    kept in the ``*_pct_*`` columns. A leave-one-landscape-out jackknife also
    names the most influential landscape and the estimates without it, since
    one landscape (shekel) carries most of the leverage on the opt_z exponent.
    """
    learners = df[~df["acquisition"].isin(MODEL_FREE)]
    rows: list[dict] = []
    for (error_model, onset), block in learners.groupby(["error_model", "jitter_iteration"]):
        if block["dataset"].nunique() < 3 or block["jitter_std"].nunique() < 2:
            continue
        try:
            cells = (block.groupby(["dataset", "jitter_std"])
                     .agg(excess_sd=("excess_sd", "mean"), opt_z=("opt_z", "first"))
                     .reset_index())
            point = _currency_fits(cells)
            names = cells["dataset"].to_numpy()
            unique = np.unique(names)
            index_by_group = {g: np.flatnonzero(names == g) for g in unique}
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            draws = np.full((BOOTSTRAP_REPS, 4), np.nan)
            for rep in range(BOOTSTRAP_REPS):
                picked = rng.choice(unique, size=len(unique), replace=True)
                draws[rep] = _currency_fits(
                    cells.iloc[np.concatenate([index_by_group[g] for g in picked])])
            jack = np.full((len(unique), 4), np.nan)
            for j, g in enumerate(unique):
                jack[j] = _currency_fits(cells[names != g])
        except Exception as exc:  # noqa: BLE001 -- one bad condition must not sink the report
            print(f"WARNING: currency fit failed for {error_model}/onset {onset}: {exc!r}")
            continue
        estimates: dict[str, object] = {}
        for prefix, (ic, iz) in (("", (0, 1)), ("loglog_", (2, 3))):
            beta_c, beta_z = float(point[ic]), float(point[iz])
            gap = beta_c + beta_z - 1.0
            gap_draws, gap_jack = draws[:, ic] + draws[:, iz] - 1.0, jack[:, ic] + jack[:, iz] - 1.0
            z_lo, z_hi = _bca_interval(beta_z, draws[:, iz], jack[:, iz])
            gap_lo, gap_hi = _bca_interval(gap, gap_draws, gap_jack)
            z_plo, z_phi = _percentile_interval(draws[:, iz])
            gap_plo, gap_phi = _percentile_interval(gap_draws)
            # The landscape whose removal moves beta_z most, and the fit without it.
            influence = np.abs(jack[:, iz] - beta_z)
            lever = int(np.nanargmax(influence)) if np.isfinite(influence).any() else None
            estimates.update({
                f"{prefix}beta_c": beta_c,
                f"{prefix}beta_z": beta_z,
                f"{prefix}beta_z_lo": z_lo,
                f"{prefix}beta_z_hi": z_hi,
                f"{prefix}beta_z_pct_lo": z_plo,
                f"{prefix}beta_z_pct_hi": z_phi,
                f"{prefix}gain_gap": gap,
                f"{prefix}gain_gap_lo": gap_lo,
                f"{prefix}gain_gap_hi": gap_hi,
                f"{prefix}gain_gap_pct_lo": gap_plo,
                f"{prefix}gain_gap_pct_hi": gap_phi,
                f"{prefix}verdict": (_currency_verdict(z_lo, z_hi, gap_lo, gap_hi)
                                     if np.isfinite(beta_z) else "not estimable"),
                f"{prefix}leverage_dataset": (str(unique[lever]) if lever is not None else ""),
                f"{prefix}beta_z_without_leverage": (float(jack[lever, iz]) if lever is not None else np.nan),
                f"{prefix}gain_gap_without_leverage": (float(gap_jack[lever]) if lever is not None else np.nan),
            })
        rows.append({
            "error_model": error_model,
            "jitter_iteration": int(onset),
            "n_runs": int(len(block)),
            "n_benchmarks": int(block["dataset"].nunique()),
            "n_cells": int(len(cells)),
            "n_nonpositive_cells": int((cells["excess_sd"] <= 0).sum()),
            "gain_prediction": 1.0 - float(estimates["beta_c"]),
            "interval": "bca",
            **estimates,
            "bootstrap_reps_used": int(np.isfinite(draws[:, :2]).all(axis=1).sum()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "noise_currency.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# 3. Which landscapes are fragile
# ---------------------------------------------------------------------------


def _vif_table(design: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors, so the collinearity is on the record."""
    rows = []
    for column in design.columns:
        others = design.drop(columns=[column])
        X = smapi.add_constant(others, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = float(smapi.OLS(design[column], X).fit().rsquared)
        rows.append({"term": column, "vif": float(1.0 / max(1.0 - r2, 1e-12)), "r2_on_others": r2})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def _cluster_bootstrap(
    cells: pd.DataFrame, response: str, predictors: list[str], reps: int, seed: int
) -> pd.DataFrame:
    """Percentile CIs from resampling BENCHMARKS, not rows.

    The sandwich estimator that ``cov_type='cluster'`` uses is an asymptotic-in-
    the-number-of-clusters result, and this study has 20 clusters -- few enough
    that it is known to understate the standard errors badly, and few enough
    that with six predictors it can come out rank-deficient outright. Resampling
    whole benchmarks with replacement respects the same dependence structure
    without that assumption.
    """
    X = smapi.add_constant(cells[predictors].astype(float), has_constant="add")
    point = smapi.OLS(cells[response].astype(float), X).fit()

    groups = cells["dataset"].to_numpy()
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    draws = np.full((reps, len(point.params)), np.nan)
    for rep in range(reps):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_group[g] for g in picked])
        sample = cells.iloc[rows]
        Xb = smapi.add_constant(sample[predictors].astype(float), has_constant="add")
        if np.linalg.matrix_rank(Xb.to_numpy()) < Xb.shape[1]:
            continue
        draws[rep] = smapi.OLS(sample[response].astype(float), Xb).fit().params.to_numpy()

    valid = ~np.isnan(draws).any(axis=1)
    draws = draws[valid]
    if len(draws) < 100:
        # Too many resamples were rank-deficient for percentile intervals to
        # mean anything -- report the point estimates and say so rather than
        # emitting a confidence interval nobody should believe.
        return pd.DataFrame({
            "term": point.params.index,
            "coefficient": point.params.to_numpy(),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_bootstrap": np.nan,
            "bootstrap_reps_used": int(len(draws)),
            "r_squared": float(point.rsquared),
            "n_cells": int(len(cells)),
            "n_clusters": int(len(unique)),
        })
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    # Two-sided bootstrap p: the smallest alpha at which the CI excludes zero.
    p_boot = 2 * np.minimum((draws <= 0).mean(axis=0), (draws >= 0).mean(axis=0))
    return pd.DataFrame({
        "term": point.params.index,
        "coefficient": point.params.to_numpy(),
        "ci_low": lo,
        "ci_high": hi,
        "p_bootstrap": np.clip(p_boot, 1.0 / max(len(draws), 1), 1.0),
        "bootstrap_reps_used": int(len(draws)),
        "r_squared": float(point.rsquared),
        "n_cells": int(len(cells)),
        "n_clusters": int(len(unique)),
    })


def descriptor_regression(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Fragility on the pre-measured landscape descriptors.

    Fitted on (benchmark x condition) cell means rather than raw runs: the 20
    seeds inside a cell are replicates of one number, and pretending they are
    independent observations would inflate every test statistic. Predictors are
    z-scored so the coefficients are comparable, and inference is a
    benchmark-level cluster bootstrap (see ``_cluster_bootstrap``).
    """
    learners = df[~df["acquisition"].isin(MODEL_FREE)].copy()
    cells = (
        learners.groupby(["dataset", "error_model", "jitter_std", "jitter_iteration"])
        .agg(excess_sd=("excess_sd", "mean"),
             **{name: (name, "first") for name in ALL_DESCRIPTORS},
             log_noise=("log_noise", "first"))
        .reset_index()
    )

    raw = cells[["log_noise"] + ALL_DESCRIPTORS].astype(float)
    vif = _vif_table(raw)
    vif.to_csv(output_dir / "descriptor_vif.csv", index=False)
    raw.corr().to_csv(output_dir / "descriptor_correlations.csv")

    frames = []
    n_benchmarks = cells["dataset"].nunique()
    for label, descriptors in (("primary", PRIMARY_DESCRIPTORS), ("sensitivity", ALL_DESCRIPTORS)):
        # Landscape descriptors vary only BETWEEN benchmarks, so the effective
        # sample size for them is the number of benchmarks, not the number of
        # cells. Fitting k descriptors needs more than k distinct landscapes.
        if n_benchmarks < len(descriptors) + 2:
            print(
                f"  [{label}] skipped: {len(descriptors)} landscape descriptors cannot be "
                f"identified from {n_benchmarks} benchmarks.",
                file=sys.stderr,
            )
            continue
        design = cells[["log_noise"] + descriptors].astype(float)
        design = (design - design.mean()) / design.std(ddof=0).replace(0.0, 1.0)
        dummies = pd.get_dummies(
            pd.DataFrame({
                "error_model": cells["error_model"].to_numpy(),
                "onset": cells["jitter_iteration"].astype(str).to_numpy(),
            }),
            drop_first=True, dtype=float,
        )
        block = pd.concat([design.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        block["dataset"] = cells["dataset"].to_numpy()
        block["excess_sd"] = cells["excess_sd"].to_numpy()

        predictors = [c for c in block.columns if c not in {"dataset", "excess_sd"}]
        table = _cluster_bootstrap(block, "excess_sd", predictors, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        table.insert(0, "model", label)

        hypotheses = table["term"].isin(["log_noise"] + descriptors)
        table["p_fdr"] = np.nan
        if hypotheses.any():
            table.loc[hypotheses, "p_fdr"] = multipletests(
                table.loc[hypotheses, "p_bootstrap"], method="fdr_bh"
            )[1]
        frames.append(table)

    if not frames:
        empty = pd.DataFrame(columns=["model", "term", "coefficient", "ci_low", "ci_high",
                                      "p_bootstrap", "p_fdr", "r_squared", "n_cells",
                                      "n_clusters", "bootstrap_reps_used"])
        empty.to_csv(output_dir / "descriptor_regression.csv", index=False)
        return empty
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_dir / "descriptor_regression.csv", index=False)
    return result


# ---------------------------------------------------------------------------
# 4. Acquisition rankings, with the benchmarks as blocks
# ---------------------------------------------------------------------------


def mediator_model(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Does the sequential result reduce to one-shot selection loss?

    ``frag_at_c`` is what an error of exactly this size costs a single greedy
    pick on this landscape -- no GP, no sequence, no acquisition function. Two
    fits on the same cells: the landscape descriptors, and the single mediator.
    If the mediator explains as much as the descriptor set, the mechanism is
    identified and "which landscapes are fragile" has a one-line answer. If it
    explains much less, the gap IS the finding: a GP fitted to fifty
    observations averages noise down in a way one-shot selection cannot.
    """
    learners = df[~df["acquisition"].isin(MODEL_FREE)].copy()
    if learners["frag_at_c"].isna().all():
        empty = pd.DataFrame(columns=["model", "term", "coefficient", "ci_low", "ci_high",
                                      "p_bootstrap", "r_squared", "n_cells", "n_clusters"])
        empty.to_csv(output_dir / "mediator_model.csv", index=False)
        return empty

    cells = (
        learners.dropna(subset=["frag_at_c"])
        .groupby(["dataset", "error_model", "jitter_std", "jitter_iteration"])
        .agg(excess_sd=("excess_sd", "mean"),
             frag_at_c=("frag_at_c", "first"),
             log_noise=("log_noise", "first"),
             **{name: (name, "first") for name in PRIMARY_DESCRIPTORS})
        .reset_index()
    )
    if cells["dataset"].nunique() < len(PRIMARY_DESCRIPTORS) + 2:
        empty = pd.DataFrame(columns=["model", "term", "coefficient", "ci_low", "ci_high",
                                      "p_bootstrap", "r_squared", "n_cells", "n_clusters"])
        empty.to_csv(output_dir / "mediator_model.csv", index=False)
        return empty

    dummies = pd.get_dummies(
        pd.DataFrame({
            "error_model": cells["error_model"].to_numpy(),
            "onset": cells["jitter_iteration"].astype(str).to_numpy(),
        }),
        drop_first=True, dtype=float,
    )
    # The currency test finds the magnitude and opt_z interact (the cost is a
    # power law in both, and its exponents do not sum to one), so an additive
    # descriptor model is misspecified for the raw response. The two
    # "_interaction" models add that product and a quadratic in the magnitude;
    # whether frag still adds to THEM is the fair version of the mediation
    # question, and the answer differs from the additive one.
    cells["noise_x_opt_z"] = cells["log_noise"] * cells["log_opt_z"]
    cells["log_noise_sq"] = cells["log_noise"] ** 2
    INTERACTION = ["noise_x_opt_z", "log_noise_sq"]

    frames = []
    for label, terms in (
        ("mediator_only", ["frag_at_c"]),
        ("descriptors_only", ["log_noise"] + PRIMARY_DESCRIPTORS),
        ("both", ["frag_at_c", "log_noise"] + PRIMARY_DESCRIPTORS),
        ("descriptors_interaction", ["log_noise"] + PRIMARY_DESCRIPTORS + INTERACTION),
        ("both_interaction", ["frag_at_c", "log_noise"] + PRIMARY_DESCRIPTORS + INTERACTION),
    ):
        design = cells[terms].astype(float)
        design = (design - design.mean()) / design.std(ddof=0).replace(0.0, 1.0)
        block = pd.concat([design.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        block["dataset"] = cells["dataset"].to_numpy()
        block["excess_sd"] = cells["excess_sd"].to_numpy()
        predictors = [c for c in block.columns if c not in {"dataset", "excess_sd"}]
        table = _cluster_bootstrap(block, "excess_sd", predictors, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        table.insert(0, "model", label)
        frames.append(table)

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_dir / "mediator_model.csv", index=False)
    return result


def acquisition_rankings(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank acquisitions per condition, blocking on benchmark.

    A Friedman test over 20 landscapes is a much better-powered design than the
    same test over 5 seeds on one dataset, which is what the data-driven arm is
    limited to.
    """
    cell = (
        df.groupby(["error_model", "jitter_std", "jitter_iteration", "dataset", "acquisition"])
        .agg(fragility=("fragility", "mean"),
             absolute_loss=("absolute_loss", "mean"),
             inference_excess=(INFERENCE_METRIC, "mean"),
             acq_failures=("acq_opt_failures", "sum"),
             seeds=("seed", "nunique"))
        .reset_index()
    )
    cell.to_csv(output_dir / "cell_means.csv", index=False)

    # The floors are excluded from the RANKING. A model-free acquisition ignores
    # every observation, so its excess regret is identically zero and it wins any
    # robustness ranking outright while learning nothing -- the artifact the
    # README warns about, here in its most extreme form. They are reported
    # instead as the no-learning reference for absolute performance.
    floors = (
        cell[cell["acquisition"].isin(MODEL_FREE)]
        .groupby(["error_model", "jitter_std", "jitter_iteration", "acquisition"])
        .agg(mean_absolute_loss=("absolute_loss", "mean"),
             mean_fragility=("fragility", "mean"))
        .reset_index()
    )
    floors.to_csv(output_dir / "floor_reference.csv", index=False)
    cell = cell[~cell["acquisition"].isin(MODEL_FREE)]

    ranking_rows: list[dict] = []
    test_rows: list[dict] = []
    for (error_model, std, onset), block in cell.groupby(
        ["error_model", "jitter_std", "jitter_iteration"]
    ):
        wide = block.pivot_table(index="dataset", columns="acquisition", values="fragility")
        wide = wide.dropna(axis=0, how="any")
        if wide.shape[0] < 3 or wide.shape[1] < 2:
            continue
        ranks = wide.rank(axis=1, method="average")
        mean_rank = ranks.mean(axis=0).sort_values()

        absolute = (
            block.pivot_table(index="dataset", columns="acquisition", values="absolute_loss")
            .reindex(wide.index).mean(axis=0)
        )
        for acquisition, value in mean_rank.items():
            ranking_rows.append({
                "error_model": error_model,
                "jitter_std": float(std),
                "jitter_iteration": int(onset),
                "acquisition": acquisition,
                "mean_rank_fragility": float(value),
                "mean_fragility": float(wide[acquisition].mean()),
                "median_fragility": float(wide[acquisition].median()),
                "mean_absolute_loss": float(absolute.get(acquisition, np.nan)),
                "n_benchmarks": int(wide.shape[0]),
            })

        n_blocks, k = wide.shape
        if k >= 3:
            statistic, p_value = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
            kendall_w = float(statistic / (n_blocks * (k - 1))) if n_blocks else np.nan
        else:
            # An arm may legitimately run fewer than three acquisitions -- the
            # robust-baseline arm is qkg against replei -- and Friedman needs at
            # least three. The pairwise Wilcoxon below is the right test there
            # anyway, and Kendall's W is not defined for a two-way ranking, so
            # this reports no omnibus rather than crashing the whole analysis.
            statistic, p_value, kendall_w = np.nan, np.nan, np.nan

        best = mean_rank.index[0]
        raw_p: list[float] = []
        pairs: list[str] = []
        for acquisition in wide.columns:
            if acquisition == best:
                continue
            try:
                _, p = wilcoxon(wide[best], wide[acquisition])
            except ValueError:  # all differences zero
                p = 1.0
            raw_p.append(float(p))
            pairs.append(acquisition)
        corrected = multipletests(raw_p, method="fdr_bh")[1] if raw_p else []
        for acquisition, p, p_adj in zip(pairs, raw_p, corrected):
            test_rows.append({
                "error_model": error_model,
                "jitter_std": float(std),
                "jitter_iteration": int(onset),
                "best_acquisition": best,
                "compared_with": acquisition,
                "wilcoxon_p": p,
                "wilcoxon_p_fdr": float(p_adj),
                "n_benchmarks": int(n_blocks),
            })
        test_rows.append({
            "error_model": error_model,
            "jitter_std": float(std),
            "jitter_iteration": int(onset),
            "best_acquisition": best,
            "compared_with": "__friedman__",
            "wilcoxon_p": float(p_value),
            "wilcoxon_p_fdr": float(p_value),
            "kendall_w": kendall_w,
            "n_benchmarks": int(n_blocks),
        })

    rankings = pd.DataFrame(ranking_rows)
    tests = pd.DataFrame(test_rows)
    rankings.to_csv(output_dir / "acquisition_rankings.csv", index=False)
    tests.to_csv(output_dir / "acquisition_tests.csv", index=False)

    if rankings.empty:
        # A follow-up arm can run a single model-based acquisition, and one
        # acquisition cannot be ranked. Report its means so the rest of the
        # analysis and the report still have a row to read.
        overall = (
            cell.groupby("acquisition")
            .agg(mean_fragility=("fragility", "mean"),
                 mean_absolute_loss=("absolute_loss", "mean"),
                 conditions=("fragility", "size"))
            .reset_index()
        )
        overall.insert(1, "mean_rank", np.nan)
        overall.to_csv(output_dir / "overall_acquisition_rankings.csv", index=False)
        return rankings, overall

    overall = (
        rankings.groupby("acquisition")
        .agg(mean_rank=("mean_rank_fragility", "mean"),
             mean_fragility=("mean_fragility", "mean"),
             mean_absolute_loss=("mean_absolute_loss", "mean"),
             conditions=("mean_rank_fragility", "count"))
        .sort_values("mean_rank")
        .reset_index()
    )
    overall.to_csv(output_dir / "overall_acquisition_rankings.csv", index=False)
    return rankings, overall


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(df: pd.DataFrame, rankings: pd.DataFrame, output_dir: Path) -> None:
    learners = df[~df["acquisition"].isin(MODEL_FREE)]

    # (a) The collapse test: fragility against the two candidate currencies.
    per_cell = (
        learners.groupby(["dataset", "jitter_std", "opt_z"])["fragility"].mean().reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, xcol, label in (
        (axes[0], per_cell["jitter_std"], r"error SD / landscape SD"),
        (axes[1], per_cell["jitter_std"] / per_cell["opt_z"], r"error SD / achievable gain"),
    ):
        scatter = ax.scatter(xcol, per_cell["fragility"], c=np.log10(per_cell["opt_z"]),
                             cmap="viridis", s=22, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fragility\n(fraction of achievable gain lost)")
    fig.colorbar(scatter, ax=axes, label=r"$\log_{10}$ opt_z")
    axes[0].set_title("scaled by landscape spread")
    axes[1].set_title("scaled by achievable gain")
    fig.savefig(output_dir / "fragility_currency_collapse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # (b) Fragility vs error magnitude, one line per benchmark.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    order = per_cell.groupby("dataset")["opt_z"].first().sort_values()
    colours = plt.cm.viridis(np.linspace(0, 1, len(order)))
    for colour, name in zip(colours, order.index):
        sub = per_cell[per_cell["dataset"] == name].sort_values("jitter_std")
        ax.plot(sub["jitter_std"], sub["fragility"], marker="o", ms=4, color=colour, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("error SD (landscape SDs)")
    ax.set_ylabel("fragility")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.set_title("Cost of feedback error by landscape (ordered by achievable gain)")
    fig.savefig(output_dir / "fragility_by_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # (c) Acquisition x condition heatmap of mean rank.
    if not rankings.empty:
        pivot = rankings.pivot_table(
            index="acquisition",
            columns=["error_model", "jitter_iteration", "jitter_std"],
            values="mean_rank_fragility",
        )
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(max(8, 0.42 * pivot.shape[1]), 0.45 * pivot.shape[0] + 2))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn_r")
        ax.set_yticks(range(pivot.shape[0]), pivot.index)
        ax.set_xticks(
            range(pivot.shape[1]),
            ["/".join(str(part) for part in col) for col in pivot.columns],
            rotation=90, fontsize=6,
        )
        fig.colorbar(image, ax=ax, label="mean rank across benchmarks (1 = most robust)")
        ax.set_title("Acquisition robustness by condition")
        fig.savefig(output_dir / "acquisition_rank_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # (d) Onset contrast.
    onsets = sorted(learners["jitter_iteration"].unique())
    if len(onsets) > 1:
        onset_cell = (
            learners.groupby(["dataset", "jitter_std", "jitter_iteration"])["fragility"]
            .mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for onset in onsets:
            sub = onset_cell[onset_cell["jitter_iteration"] == onset]
            grouped = sub.groupby("jitter_std")["fragility"]
            ax.errorbar(grouped.mean().index, grouped.mean(), yerr=grouped.sem(),
                        marker="o", capsize=3, label=f"onset {onset}")
        ax.set_xscale("log")
        ax.set_xlabel("error SD (landscape SDs)")
        ax.set_ylabel("fragility (mean over benchmarks)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Does it matter when the error starts?")
        fig.savefig(output_dir / "onset_contrast.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    df: pd.DataFrame,
    floor: pd.DataFrame,
    headroom: pd.DataFrame,
    bias_control: pd.DataFrame,
    helpful: pd.DataFrame,
    currency: pd.DataFrame,
    descriptors: pd.DataFrame,
    mediator: pd.DataFrame,
    overall: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    output_dir: Path,
    tolerance: float,
) -> None:
    lines: list[str] = []
    add = lines.append

    add("KNOWN-FUNCTION ROBUSTNESS: CROSS-BENCHMARK SYNTHESIS")
    add("=" * 72)
    add("")
    add(f"Runs analysed      : {len(df):,}")
    add(f"Benchmarks         : {df['dataset'].nunique()}")
    add(f"Acquisitions       : {df['acquisition'].nunique()}")
    add(f"Seeds              : {df['seed'].nunique()}")
    add(f"Error models       : {sorted(df['error_model'].unique())}")
    add(f"Error magnitudes   : {sorted(df['jitter_std'].unique())} (landscape SDs)")
    add(f"Onsets             : {sorted(df['jitter_iteration'].unique())}")
    add("")

    add("1. FLOOR CHECK (model-free acquisitions must be noise-invariant)")
    add("-" * 72)
    floor_runs = df[df["acquisition"].isin(MODEL_FREE)]
    design = (
        floor_runs["error_model"].map(_has_input_error)
        if "error_model" in floor_runs.columns
        else pd.Series(False, index=floor_runs.index)
    )
    if floor.empty:
        add("  skipped: no model-free acquisitions in the sweep.")
    elif design.all():
        # Under an input error the floor evaluates the wrong points too, so its
        # excess is the measurement, not a failure. A PASSED/FAILED verdict here
        # would tell a reader that every result below is suspect, which is false.
        cost = float(np.abs(floor_runs["auc_simple_regret_excess_true"]).max())
        add("  not asserted: this arm corrupts the DESIGN, so a model-free floor evaluates")
        add("  the wrong points too. Its excess regret is the geometric cost of the error")
        add(f"  with no learning to corrupt (largest |excess AUC| = {cost:.3e}).")
    else:
        asserted = floor_runs[~design]
        worst = float(np.abs(asserted["auc_simple_regret_excess_true"]).max())
        verdict = "PASSED" if worst <= tolerance else "FAILED"
        add(f"  {verdict}: largest |excess AUC regret| = {worst:.3e} (tolerance {tolerance:g})")
        if design.any():
            add(f"  ({int(design.sum()):,} input-error floor runs reported separately, not asserted.)")
        if worst > tolerance:
            add("  Noise is reaching the candidate stream; every result below is suspect.")
    add("")

    add("1b. HEADROOM (could this benchmark have shown an effect at all?)")
    add("-" * 72)
    if headroom.empty:
        add("  not computed: this arm runs no model-free acquisitions, so there is no")
        add("  floor to measure headroom against. See the main sweep's screen.")
    else:
        ceiling_free = headroom[headroom["ceiling_free"]]
        add(f"  {'benchmark':<20s} {'headroom':>9s}   (noise-free advantage of the best")
        add(f"  {'':<20s} {'':>9s}    learner over the model-free floor)")
        for _, row in headroom.iterrows():
            mark = "  <- ceiling-free" if row["ceiling_free"] else ""
            add(f"  {row['dataset']:<20s} {row['headroom']:>9.3f}{mark}")
        if len(ceiling_free):
            add("")
            add(f"  {len(ceiling_free)} benchmark(s) have under 10% headroom: BO already reaches")
            add("  the optimum without noise, so a null there is guaranteed by construction")
            add("  and says nothing about robustness. Exclude them before interpreting any")
            add("  per-benchmark null.")
    add("")

    add("1c. CONTROL: a constant bias from iteration 1 should cost nothing")
    add("-" * 72)
    add("  The GP standardises its targets and the incumbent is a posterior mean, so")
    add("  an offset present in every observation is algebraically invisible. At")
    add("  onset 0 the bias arm should therefore match the gaussian arm up to its")
    add("  noise draw.")
    add("")
    if bias_control.empty:
        add("  not computed: this arm does not run both the gaussian and bias models")
        add("  at onset 0.")
    else:
        add(f"  {'error SD':>9s} {'gaussian':>11s} {'bias':>11s} {'difference':>11s}")
        for _, row in bias_control.iterrows():
            add(f"  {row['jitter_std']:>9.3g} {row.get('mean_gaussian', float('nan')):>11.4f} "
                f"{row.get('mean_bias', float('nan')):>11.4f} {row['difference']:>11.4f}")
    add("")

    add("1d. WHERE THE ERROR HELPED")
    add("-" * 72)
    add("  On a multimodal landscape, noise after convergence is an escape mechanism,")
    add("  so a negative excess regret is a real regime rather than sampling noise.")
    add("  Cells with mean excess < 0 and dz < -0.2:")
    add("")
    if helpful.empty:
        add("  none.")
    else:
        add(f"  {'benchmark':<20s} {'error':<10s} {'SD':>7s} {'onset':>6s} {'mean':>9s} {'dz':>7s}")
        for _, row in helpful.head(25).iterrows():
            add(f"  {row['dataset']:<20s} {row['error_model']:<10s} {row['jitter_std']:>7.3g} "
                f"{int(row['jitter_iteration']):>6d} {row['mean']:>9.4f} {row['dz']:>7.2f}")
        if len(helpful) > 25:
            add(f"  ... and {len(helpful) - 25} more (beneficial_regime.csv)")
    add("")

    add("2. WHAT MAKES AN ERROR 'BIG'?")
    add("-" * 72)
    add("  Power law E[excess] = A * sigma_e^b_c * opt_z^b_z on landscape x magnitude")
    add("  cell means: quasi-Poisson estimating equations, which keep cells whose mean")
    add("  excess is <= 0, with BCa intervals from resampling benchmarks. SPREAD predicts")
    add("  b_z = 0; GAIN predicts b_c + b_z - 1 = 0 (the 'gain gap'). log-OLS refits")
    add("  log(excess) on the positive cells only, as a check on the estimator.")
    add("")
    if currency.empty:
        add("  not estimable.")
    else:
        add(f"  {'error model':<12s} {'onset':>5s} {'b_c':>6s} {'b_z':>6s} {'95% CI':>15s} "
            f"{'gain gap':>8s} {'95% CI':>15s} {'verdict':>20s} {'log-OLS':>20s}")
        for _, row in currency.iterrows():
            add(f"  {row['error_model']:<12s} {int(row['jitter_iteration']):>5d} "
                f"{row['beta_c']:>6.2f} {row['beta_z']:>6.2f} "
                f"[{row['beta_z_lo']:>6.2f},{row['beta_z_hi']:>6.2f}] "
                f"{row['gain_gap']:>+8.2f} "
                f"[{row['gain_gap_lo']:>+6.2f},{row['gain_gap_hi']:>+6.2f}] "
                f"{row['verdict']:>20s} {row['loglog_verdict']:>20s}")
        add("")
        add(f"  Verdict across conditions: {currency['verdict'].value_counts().to_dict()}")
        add(f"  Cells with mean excess <= 0 (kept here, dropped by log-OLS): "
            f"{int(currency['n_nonpositive_cells'].sum())} of {int(currency['n_cells'].sum())}")
        lev = currency.iloc[0]
        add(f"  Leverage: dropping {lev['leverage_dataset']} moves b_z from {lev['beta_z']:.2f} to "
            f"{lev['beta_z_without_leverage']:.2f} in the first condition (see *_without_leverage columns)")
    add("")

    add("3. WHICH LANDSCAPES ARE FRAGILE (z-scored predictors, FDR over descriptors)")
    add("-" * 72)
    add("  Fitted on benchmark x condition cell means; 95% CIs and p-values from a")
    add("  2000-replicate bootstrap that resamples BENCHMARKS, not rows.")
    add("")
    if descriptors.empty:
        add("  not estimable: too few benchmarks to identify the landscape descriptors.")
        add("")
    for model in ("primary", "sensitivity"):
        block = descriptors[descriptors["model"] == model] if not descriptors.empty else descriptors
        if block.empty:
            continue
        terms = PRIMARY_DESCRIPTORS if model == "primary" else ALL_DESCRIPTORS
        interesting = block[block["term"].isin(["log_noise"] + terms)]
        add(f"  [{model}]  R^2 = {block['r_squared'].iloc[0]:.3f} over "
            f"{int(block['n_cells'].iloc[0]):,} cells in "
            f"{int(block['n_clusters'].iloc[0])} benchmarks")
        add(f"  {'term':<16s} {'beta':>9s} {'95% CI':>20s} {'p(FDR)':>9s}")
        for _, row in interesting.iterrows():
            add(f"  {row['term']:<16s} {row['coefficient']:>9.4f} "
                f"[{row['ci_low']:>8.3f},{row['ci_high']:>8.3f}] {row['p_fdr']:>9.4g}")
        add("")

    add("3b. DOES IT REDUCE TO ONE-SHOT SELECTION LOSS?")
    add("-" * 72)
    add("  frag_at_c is the expected loss from one noisy greedy pick on this")
    add("  landscape at this error size -- no GP, no sequence. Compare the variance")
    add("  it explains with that of the full descriptor set.")
    add("")
    if mediator.empty:
        add("  not estimable.")
    else:
        for model in ("mediator_only", "descriptors_only", "both"):
            block = mediator[mediator["model"] == model]
            if block.empty:
                continue
            row = block[block["term"] == "frag_at_c"]
            beta = f"{row['coefficient'].iloc[0]:+.4f}" if len(row) else "     -"
            add(f"  {model:<18s} R^2 = {block['r_squared'].iloc[0]:.3f}   "
                f"beta(frag_at_c) = {beta}")
    add("")

    add("4. ACQUISITION ROBUSTNESS, POOLED OVER CONDITIONS")
    add("-" * 72)
    add(f"  {'acquisition':<12s} {'mean rank':>10s} {'fragility':>11s} {'abs. loss':>11s}")
    for _, row in overall.iterrows():
        add(f"  {row['acquisition']:<12s} {row['mean_rank']:>10.3f} "
            f"{row['mean_fragility']:>11.4f} {row['mean_absolute_loss']:>11.4f}")
    add("")
    add("  Mean rank is over benchmarks within each condition, then averaged over")
    add("  conditions. Lower is more robust. 'abs. loss' is the deployment number:")
    add("  the post-onset per-iteration regret under noise as a fraction of the")
    add("  achievable gain -- 0 is perfect, 1 is no better than an average random")
    add("  design. random and sobol are EXCLUDED from the ranking: they ignore every")
    add("  observation, so their excess regret is exactly zero and they would win a")
    add("  robustness ranking while learning nothing. See floor_reference.csv for")
    add("  what they achieve in absolute terms.")
    add("")

    add("5. LANDSCAPE DESCRIPTORS")
    add("-" * 72)
    add(f"  {'benchmark':<20s} {'d':>3s} {'opt_z':>8s} {'sparsity':>9s} {'rugged':>7s} "
        f"{'skew':>7s} {'tail':>6s} {'frag@1':>7s}")
    for name, entry in sorted(stats.items(), key=lambda kv: kv[1]["opt_z"]):
        add(f"  {name:<20s} {int(entry['dim']):>3d} {entry['opt_z']:>8.2f} "
            f"{entry.get('sparsity_sampled', float('nan')):>9.5f} {entry['ruggedness']:>7.3f} "
            f"{entry['skew']:>7.2f} {entry['tail_ratio']:>6.2f} "
            f"{entry.get('frag_1', float('nan')):>7.3f}")
    add("  (sparsity is the sample-anchored fraction; frag@1 is the one-shot")
    add("   selection loss at an error of 1 landscape SD)")
    add("")

    (output_dir / "boba_robustness_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = bb.load_stats(args.stats_path)
    paired = load_paired(args.input_dir)
    if args.seeds:
        keep = {int(v.strip()) for v in args.seeds.split(",") if v.strip()}
        before = len(paired)
        paired = paired[paired["seed"].isin(keep)]
        print(f"Restricted to seeds {sorted(keep)}: {len(paired):,} of {before:,} runs.")
        if paired.empty:
            raise ValueError(f"No runs for seeds {sorted(keep)}.")
    df = attach_landscape(paired, stats)

    floor = floor_check(df, args.floor_tolerance)
    if not floor.empty:
        floor.to_csv(args.output_dir / "floor_check.csv", index=False)

    balance_check(df, args.output_dir)
    helpful = beneficial_regime(df, args.output_dir)
    headroom = headroom_screen(df, args.output_dir)
    bias_control = bias_onset_control(df, args.output_dir)
    currency = noise_currency(df, args.output_dir)
    descriptors = descriptor_regression(df, args.output_dir)
    mediator = mediator_model(df, args.output_dir)
    rankings, overall = acquisition_rankings(df, args.output_dir)
    make_figures(df, rankings, args.output_dir)
    write_report(df, floor, headroom, bias_control, helpful, currency, descriptors, mediator,
                 overall, stats, args.output_dir, args.floor_tolerance)

    summary = {
        "n_runs": int(len(df)),
        "benchmarks": sorted(df["dataset"].unique().tolist()),
        "floor_check_max_abs_excess": (
            float(floor["max_abs_excess_auc"].max()) if not floor.empty else None
        ),
        "currency_verdicts": (
            currency["verdict"].value_counts().to_dict() if not currency.empty else {}
        ),
        "most_robust_acquisition": overall["acquisition"].iloc[0] if len(overall) else None,
        "best_absolute_acquisition": (
            overall.sort_values("mean_absolute_loss")["acquisition"].iloc[0] if len(overall) else None
        ),
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2),
                                                           encoding="utf-8")
    print(f"\nWrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
