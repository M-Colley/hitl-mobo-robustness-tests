# E5: curvature, free-level and spline checks of the currency power law.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""E5 check. (1) Is log E[R] curved in log sigma_e inside the power law?
(2) Does the GAIN/SPREAD verdict survive when h is left free (a spline) instead
of a power law?  PPML = GLM Poisson (statsmodels), same cell means as
analyse_boba_robustness.noise_currency, landscape-cluster percentile bootstrap."""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import analyse_boba_robustness as ar  # noqa: E402
import boba_benchmarks as bb  # noqa: E402

warnings.filterwarnings("ignore")
REPS = 2000
rng_seed = 20260923


def ppml(y, X, offset=None, max_iter=300, tol=1e-10):
    """Quasi-Poisson estimating equations X'(y - mu) = 0 (PPML), Newton with
    step-halving, as analyse_boba_robustness._power_law_mean but for any X;
    keeps cells with y <= 0."""
    off = np.zeros(len(y)) if offset is None else offset
    pos = y > 0
    try:
        beta = np.linalg.lstsq(X[pos], np.log(y[pos]) - off[pos], rcond=None)[0]
    except Exception:
        beta = np.zeros(X.shape[1])
    q = lambda b: float(np.sum(y * (X @ b + off) - np.exp(X @ b + off)))
    cur = q(beta)
    for _ in range(max_iter):
        mu = np.exp(X @ beta + off)
        try:
            step = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        t = 1.0
        while True:
            tr = beta + t * step
            v = q(tr)
            if np.isfinite(v) and v >= cur - 1e-12:
                break
            t *= 0.5
            if t < 1e-12:
                return None
        beta, cur = tr, v
        if np.max(np.abs(t * step)) < tol:
            return beta
    return None


def fits(cells):
    y = cells["excess_sd"].to_numpy(float)
    lc = np.log(cells["jitter_std"].to_numpy(float))
    lz = np.log(cells["opt_z"].to_numpy(float))
    out = {}
    # (1) power law plus a quadratic in log sigma (centred)
    lcc = lc - lc.mean()
    X = np.column_stack([np.ones_like(lc), lcc, lcc ** 2, lz])
    p = ppml(y, X)
    out["quad"] = np.nan if p is None else p[2]
    X0 = np.column_stack([np.ones_like(lc), lc, lz])
    p0 = ppml(y, X0)
    out["beta_c"], out["beta_z"] = (np.nan, np.nan) if p0 is None else (p0[1], p0[2])
    # (2) flexible GAIN: E[R] = opt_z * exp(s(log(c/opt_z)) + gamma*log opt_z); GAIN <=> gamma = 0
    u = lc - lz
    B = np.asarray(dmatrix("bs(u, df=4, include_intercept=False)", {"u": u}, return_type="matrix"))
    Xg = np.column_stack([B, lz])
    pg = ppml(y, Xg, offset=lz)
    out["gain_gamma"] = np.nan if pg is None else pg[-1]
    # flexible SPREAD: E[R] = exp(s(log c) + beta*log opt_z); SPREAD <=> beta = 0
    # only 4 magnitudes: a saturated dummy per magnitude is the fully flexible s()
    D = pd.get_dummies(cells["jitter_std"].astype(str)).to_numpy(float)
    Xs = np.column_stack([D, lz])
    p = ppml(y, Xs)
    out["spread_beta"] = np.nan if p is None else p[-1]
    return out


def main():
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    df = ar.attach_landscape(ar.load_paired(REPO / "output-boba"), stats)
    learners = df[~df["acquisition"].isin(ar.MODEL_FREE)]
    rows = []
    for (model, onset), block in learners.groupby(["error_model", "jitter_iteration"]):
        cells = (block.groupby(["dataset", "jitter_std"])
                 .agg(excess_sd=("excess_sd", "mean"), opt_z=("opt_z", "first")).reset_index())
        point = fits(cells)
        names = cells["dataset"].to_numpy()
        uniq = np.unique(names)
        idx = {g: np.flatnonzero(names == g) for g in uniq}
        rng = np.random.default_rng(rng_seed)
        draws = []
        for _ in range(REPS):
            pick = rng.choice(uniq, len(uniq), replace=True)
            draws.append(fits(cells.iloc[np.concatenate([idx[g] for g in pick])]))
        draws = pd.DataFrame(draws)
        row = {"error_model": model, "onset": onset}
        for k, v in point.items():
            lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
            row[k] = f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "output-boba" / "analysis" / "review" / "register_checks" / "currency_curvature.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
