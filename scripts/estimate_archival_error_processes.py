"""Estimate the paper's rating-error processes from the three archival studies.

The simulation injects four error processes: gaussian noise, a constant offset,
a drift that ramps linearly from 0 to sigma_e over the session, and an AR(1)
with rho = 0.8. The paper anchors their magnitude with one number per study: a
close-nearest-neighbour nugget of within-rater noise, divided by sigma_f. The
three archival human-in-the-loop studies hold 11-28 ratings per participant in
presentation order, which is enough to estimate the processes themselves. For
each study this script estimates, with participant-bootstrap 95% intervals:

  1. NOISE. Reproduces the paper's nugget exactly, by calling
     calibrate_noise_from_data.calibrate_dataset, the function that produced
     output/noise_calibration.csv, which anchor_noise_scale.py divides by
     sigma_f. It adds a participant bootstrap, and reports how much of the
     nugget comes from consecutive-trial pairs, which serial dependence would
     shrink.
  2. DRIFT. The slope of the rating residual on trial position within a
     session. The residual removes a leave-one-participant-out (LOPO)
     prediction from the pipeline's selected oracle family, so search progress
     (the optimiser proposing better designs later) is removed without a
     participant's own ratings fitting their residuals away. The prediction is
     scaled by b, its coefficient in the pooled within-session regression of
     rating on [trial, prediction]. The LOPO predictions are over-dispersed
     (calibration slope < 1), and removing them at b = 1 subtracts noise that
     rises with search progress. b = 0 (naive) and b = 1 (full removal) are
     reported beside it. Three model-free checks need no design model: close
     design pairs rated k trials apart, the random-design sampling phase
     (ehmi), and the same sampling designs rated in both counterbalanced
     conditions (provoice).
  3. AR(1). Lag-one autocorrelation of the within-session residuals, per session
     (Kendall bias-corrected) and pooled. Beside each is the value the same
     estimator reads, at these session lengths, under the paper's AR(1) and
     under white noise, by simulation. Also reported: a split by consecutive
     design distance, where model misfit travels with distance and a rater's
     serial dependence does not, and a model-free temporal semivariogram of
     close design pairs.
  4. HETEROSCEDASTICITY. Residual variance against the predicted value, and the
     rating difference of close pairs against their rating level.
  5. SATURATION. Share of ratings at each instrument's minimum and maximum.
  6. OFFSETS. The variance share of a per-participant random intercept from a
     one-way random-effects decomposition, with a model-free check on the
     designs that several participants rated.

The composite is built exactly as the pipeline builds it, through
``sim.load_observations`` and ``sim.compute_objective`` with normalize=False:
the unweighted mean of the signed construct columns in
``objective_map["composite"]``.

OPTICARVIS SCALE MIXING. Its per-participant logs hold the ratings the optimiser
saw, normalised per construct to [-1, 1]. The exception is the first (Run 0)
row of warm-start groups B and E, which is logged on the raw 1-5, 1-7 and -3..3
scales. The pipeline averages both into one composite. The paper's opticarvis
numbers are reproduced on that "pipeline" basis. Every opticarvis
error-process estimate is computed on a "scale_consistent" basis instead, which
drops the raw-scale rows and has its own sigma_f, computed by the anchor
script's procedure.

PROVOICE PREDICTABILITY SIGN. The logged Predictability is lower-is-better (the
study's R scripts invert it), but datasets.json enters it as +Predictability.
Provoice is analysed on the pipeline composite (the paper's) and on a
"sign_corrected" composite with -Predictability, each with its own sigma_f.

  python scripts/estimate_archival_error_processes.py
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bo_sensor_error_simulation as sim  # noqa: E402  (sets thread limits first)
import calibrate_noise_from_data as cal  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402

S = sim.OBSERVATION_SOURCE_COLUMN
OBJECTIVE = "composite"
SEED = 10_007                 # anchor_noise_scale.py's seed and sample count
LANDSCAPE_SAMPLES = 100_000
CLOSE_PAIR_FRACTION = 0.05    # calibrate_noise_from_data.py's default
MIN_SESSION_TRIALS = 6        # shortest session used for slopes and lag-1
PAPER_RHO = 0.8
AR_SIMULATIONS = 4000
ALT_DESIGN_MODELS = ("extra_trees", "random_forest", "gradient_boosting")
EPS = 1e-9

# Instrument bounds of each construct as it appears in the logs. ehmi and
# provoice log raw item means. opticarvis logs (x - midpoint) / half-range, so
# every construct lies in [-1, 1] (verified: Trust takes the nine values
# -1, -0.75, ..., 1, i.e. (x - 3) / 2 of a two-item 1-5 mean).
INSTRUMENT_BOUNDS = {
    "ehmi": {
        "Trust": (1.0, 5.0), "Understanding": (1.0, 5.0),
        "PerceivedSafety": (-3.0, 3.0), "Aesthetics": (1.0, 7.0),
        "Acceptance": (1.0, 7.0),
    },
    "opticarvis": {c: (-1.0, 1.0) for c in
                   ("Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance")},
    "provoice": {
        "Predictability": (1.0, 5.0), "Percieved Usefulness": (1.0, 5.0),
        "Mental Demand": (1.0, 20.0),
    },
}
# provoice logs Predictability with LOWER = BETTER: the study's own analysis
# (ProVoiceData/MeanValueProgression.R, ParetoFront.R) inverts it as 6 - x
# "because predictability is maximised", its Pareto rows average 1.3 against 2.4
# for the rest, and it correlates -0.70 with Usefulness and +0.46 with Mental
# Demand. datasets.json enters it as +Predictability. The pipeline composite is
# kept as the paper's basis and this corrected composite is analysed beside it.
SIGN_CORRECTIONS = {
    "provoice": ["-Predictability", "Percieved Usefulness", "-Mental Demand"],
}
BOUND_PROVENANCE = {
    "ehmi": "1-5 (Trust, Understanding), -3..3 (PerceivedSafety), 1-7 (Aesthetics, "
            "Acceptance); every bound is attained in the data",
    "opticarvis": "logged normalised to [-1, 1] per construct by the study software",
    "provoice": "1-5 (Predictability, Usefulness, two-item means); Mental Demand assumed 1-20 "
                "(observed 1-17, so the floor is certain and the ceiling is never reached)",
}


# --------------------------------------------------------------------------- #
# Loading, exactly as the pipeline loads
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config", type=Path, default=Path("datasets.json"))
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path(".dataset_cache"))
    parser.add_argument("--oracle-selection-path", type=Path,
                        default=Path("output/best_oracle_models.json"))
    parser.add_argument("--manifest", type=Path, default=Path("output/per_dataset/manifest.json"))
    parser.add_argument("--noise-calibration", type=Path, default=Path("output/noise_calibration.csv"))
    parser.add_argument("--noise-anchor", type=Path, default=Path("output/noise_anchor.csv"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20_260_922)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--skip-model-comparison", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output-boba/analysis/review"))
    return parser.parse_args(argv)


def composite_columns(dataset: sim.DatasetConfig) -> tuple[list[str], list[str], np.ndarray]:
    cols = dataset.objective_map[OBJECTIVE]
    base = [sim._objective_base_column(c) for c in cols]
    signs = np.array([sim._objective_sign(c) for c in cols])
    return cols, base, signs


def check_presentation_order(df: pd.DataFrame) -> list[str]:
    """Row order within a file must be presentation order; list files where it is not."""
    problems = []
    for session, g in df.groupby(S, sort=False):
        if "Iteration" in g.columns:
            key = g["Iteration"].to_numpy(float)
        elif "Phase" in g.columns and "Run" in g.columns:
            key = (g["Phase"].astype(str).str.lower().eq("optimization").to_numpy() * 1000
                   + g["Run"].to_numpy(float))
        elif "Run" in g.columns:
            key = g["Run"].to_numpy(float)
        else:
            continue
        if np.any(np.diff(key) < 0):
            problems.append(Path(session).name)
    return problems


def annotate(df: pd.DataFrame, dataset: sim.DatasetConfig) -> pd.DataFrame:
    """Composite (pipeline definition), participant, session and trial position."""
    cols, _, _ = composite_columns(dataset)
    df = df.copy()
    df["composite"] = sim.compute_objective(df, cols, False, None).to_numpy(float)
    df["session"] = df[S].astype(str)
    df["participant"] = (df["User_ID"].astype(str) if "User_ID" in df.columns
                         else df["session"])
    # Presentation order is row order within the file (checked). Duplicates of
    # one presentation were already dropped, so this counts presentations.
    df["trial"] = df.groupby("session", sort=False).cumcount().astype(float)
    # Sitting position: a participant's sessions in chronological order (only
    # provoice has two sessions, one per counterbalanced condition).
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        first = ts.groupby(df["session"]).transform("min")
    else:
        first = pd.Series(0, index=df.index)
    df["_first"] = first
    offsets = {}
    for _, g in df.groupby("participant", sort=False):
        order = g.groupby("session")["_first"].min().sort_values(kind="stable").index.tolist()
        running = 0.0
        for session in order:
            offsets[session] = running
            running += float((g["session"] == session).sum())
    df["sitting_trial"] = df["trial"] + df["session"].map(offsets)
    return df.drop(columns="_first")


def logged_scale_violations(df: pd.DataFrame, dataset: sim.DatasetConfig) -> np.ndarray:
    bad = np.zeros(len(df), dtype=bool)
    for col, (lo, hi) in INSTRUMENT_BOUNDS[dataset.name].items():
        values = df[col].to_numpy(float)
        bad |= (values < lo - EPS) | (values > hi + EPS)
    return bad


def design_key(df: pd.DataFrame, params: list[str]) -> pd.Series:
    return df[params].round(10).astype(str).agg("|".join, axis=1)


# --------------------------------------------------------------------------- #
# Design model and landscape scale
# --------------------------------------------------------------------------- #

def _fit_predict(estimator, X_train, y_train, X_test):
    model = clone(estimator)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def lopo_predictions(df: pd.DataFrame, dataset: sim.DatasetConfig, model_name: str,
                     n_jobs: int) -> np.ndarray:
    """Leave-one-participant-out prediction of the composite from the design.

    Uses the pipeline's own estimator and hyperparameters
    (sim._build_oracle_model), fitted without the jitter augmentation, which
    lowers held-out R^2. It fits on design means when the dataset's
    oracle_target is "mean", as the pipeline does.
    """
    params = dataset.param_columns
    estimator = sim._build_oracle_model(model_name, SEED, 1.0)
    participants = df["participant"].to_numpy()
    jobs = []
    for pid in pd.unique(participants):
        test = np.flatnonzero(participants == pid)
        train = df[participants != pid]
        if dataset.oracle_target == "mean":
            train = train.groupby(params, as_index=False)["composite"].mean()
        jobs.append((test, train[params], train["composite"].to_numpy(float),
                     df.iloc[test][params]))
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(estimator, Xtr, ytr, Xte) for _, Xtr, ytr, Xte in jobs)
    out = np.full(len(df), np.nan)
    for (test, *_), p in zip(jobs, preds):
        out[test] = p
    return out


def landscape_sigma_f(frame: pd.DataFrame, dataset: sim.DatasetConfig, model_name: str,
                      n_jobs: int) -> float:
    """sigma_f by anchor_noise_scale.py's procedure (same oracle, box, seed, n)."""
    cols, _, _ = composite_columns(dataset)
    with contextlib.redirect_stdout(io.StringIO()):
        oracle = sim.build_oracle(
            df=frame, objective=OBJECTIVE, objective_columns=cols,
            param_columns=dataset.param_columns, seed=SEED, normalize=False,
            weights=None, oracle_model=model_name, oracle_augmentation="jitter",
            oracle_augment_repeats=2, oracle_augment_std=0.02, oracle_fast=False,
            oracle_target=dataset.oracle_target)
    if hasattr(oracle.model, "n_jobs"):
        oracle.model.n_jobs = n_jobs          # prediction only; values unchanged
    bounds = sim.bounds_from_data(frame, dataset.param_columns)
    rng = np.random.default_rng(SEED)
    X = rng.uniform(bounds.low, bounds.high, size=(LANDSCAPE_SAMPLES, len(dataset.param_columns)))
    y = oracle.predict_many(X).reshape(-1)
    return float(np.std(y, ddof=1))


# --------------------------------------------------------------------------- #
# Per-session sufficient statistics, so the participant bootstrap is a weighted
# sum and the prediction coefficient b can be re-estimated in every replicate.
# --------------------------------------------------------------------------- #

def _paper_drift_power(t: np.ndarray, alpha: float = 0.05) -> float:
    """Power of this session's slope test if the paper's DRIFT held exactly.

    DRIFT adds a ramp of total height sigma_e over the session to N(0, sigma_e^2)
    noise, so the slope's noncentrality, sqrt(Sxx) / span, does not depend on sigma_e.
    """
    n = len(t)
    tdm = t - t.mean()
    delta = math.sqrt(float(tdm @ tdm)) / float(t.max() - t.min())
    crit = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(stats.nct.sf(crit, n - 2, delta) + stats.nct.cdf(-crit, n - 2, delta))


def _unit(X: np.ndarray, bounds: sim.Bounds) -> np.ndarray:
    span = np.where(bounds.high > bounds.low, bounds.high - bounds.low, 1.0)
    return (X - bounds.low) / span


def build_sessions(df: pd.DataFrame, dataset: sim.DatasetConfig, sigma_f: float,
                   bounds: sim.Bounds, threshold: float, is_mixed: np.ndarray | None,
                   pred_terciles: np.ndarray, consec_cut: float, sampling_ok: bool
                   ) -> tuple[pd.DataFrame, list[str]]:
    """One row per session holding every sum the statistics need."""
    params = dataset.param_columns
    _, base, signs = composite_columns(dataset)
    bnds = INSTRUMENT_BOUNDS[dataset.name]
    participants = list(pd.unique(df["participant"]))
    pindex = {p: i for i, p in enumerate(participants)}
    rows = []
    for session, g in df.groupby("session", sort=False):
        g = g.sort_values("trial")
        n = len(g)
        t = g["trial"].to_numpy(float)
        y = g["composite"].to_numpy(float)
        p = g["pred"].to_numpy(float)
        rec: dict[str, float] = {"session": session, "pid": pindex[g["participant"].iloc[0]],
                                 "n": n, "span": float(t.max() - t.min())}
        # raw sums: design-model fit and participant-level offsets
        rec.update(sy=y.sum(), sp=p.sum(), syy=y @ y, spp=p @ p, syp=y @ p)

        # --- the pipeline's close NN pairs (the paper's nugget) ------------------
        pair_arr, dists = cal.nn_pairs(g.reset_index(drop=True), params, bounds)
        rec.update(nn_m=0, nn_s=0.0, nn_ss=0.0, nn_lag1=0, nn_ar=0.0)
        if len(pair_arr):
            keep = dists <= threshold
            a, b = pair_arr[keep, 0], pair_arr[keep, 1]
            d = y[a] - y[b]
            lag = np.abs(t[a] - t[b])
            rec.update(nn_m=len(d), nn_s=d.sum(), nn_ss=d @ d, nn_lag1=int(np.sum(lag == 1)),
                       nn_ar=float(np.sum(1 - PAPER_RHO ** lag)))
            if is_mixed is not None:
                mixed = is_mixed[g.index.to_numpy()]
                clean = ~(mixed[a] | mixed[b])
                rec.update(nnc_m=int(clean.sum()), nnc_s=d[clean].sum(),
                           nnc_ss=d[clean] @ d[clean])

        # --- all close design pairs within the session (model-free) --------------
        Xu = _unit(g[params].to_numpy(float), bounds)
        D = np.linalg.norm(Xu[:, None, :] - Xu[None, :, :], axis=2)
        i, j = np.triu_indices(n, 1)
        close = D[i, j] <= threshold
        i, j = i[close], j[close]              # j is later: rows are in trial order
        dd = y[j] - y[i]
        lag = t[j] - t[i]
        for name, mask in (("g1", lag == 1), ("g23", (lag >= 2) & (lag <= 3)), ("g4", lag >= 4)):
            rec[f"{name}_m"] = int(mask.sum())
            rec[f"{name}_ss"] = float(dd[mask] @ dd[mask])
        rec["g4_paper"] = float(np.sum(1 - PAPER_RHO ** lag[lag >= 4]))
        rec.update(cp_m=len(dd), cp_ld=float(lag @ dd), cp_ll=float(lag @ lag))
        level = (y[i] + y[j]) / 2.0 / sigma_f
        d2 = dd ** 2
        rec.update(ch_x=level.sum(), ch_xx=level @ level, ch_y=d2.sum(), ch_xy=level @ d2)

        # --- within-session sums, each residual quantity a polynomial in b -------
        tdm, ydm, pdm = t - t.mean(), y - y.mean(), p - p.mean()
        rec.update(tt=tdm @ tdm, ty=tdm @ ydm, tp=tdm @ pdm, yy=ydm @ ydm, yp=ydm @ pdm,
                   pp=pdm @ pdm)
        x = p / sigma_f
        rec.update(hx=x.sum(), hxx=x @ x, hxyy=x @ ydm ** 2, hxyp=x @ (ydm * pdm),
                   hxpp=x @ pdm ** 2)
        tercile = np.digitize(p, pred_terciles)
        for k in range(3):
            m = tercile == k
            rec[f"ter{k}_n"] = int(m.sum())
            rec[f"ter{k}_yy"] = float(ydm[m] @ ydm[m])
            rec[f"ter{k}_yp"] = float(ydm[m] @ pdm[m])
            rec[f"ter{k}_pp"] = float(pdm[m] @ pdm[m])

        eligible = n >= MIN_SESSION_TRIALS
        rec["eligible"] = float(eligible)
        rec["dr_power"] = _paper_drift_power(t) if eligible else 0.0
        # consecutive-trial lag sums (lag-1 autocorrelation)
        a = np.flatnonzero(np.diff(t) == 1)
        b = a + 1
        rec["ar_pairs"] = len(a)
        for u, uu in (("y", ydm), ("p", pdm), ("t", tdm)):
            for v, vv in (("y", ydm), ("p", pdm), ("t", tdm)):
                rec[f"L{u}{v}"] = float(uu[a] @ vv[b])
        step = np.linalg.norm(Xu[b] - Xu[a], axis=1)
        for name, m in (("near", step <= consec_cut), ("far", step > consec_cut)):
            aa, bb = a[m], b[m]
            rec[f"{name}_n"] = int(m.sum())
            rec[f"{name}_Lyy"] = float(ydm[aa] @ ydm[bb])
            rec[f"{name}_Lyp"] = float(ydm[aa] @ pdm[bb] + pdm[aa] @ ydm[bb])
            rec[f"{name}_Lpp"] = float(pdm[aa] @ pdm[bb])
            for side, idx in (("a", aa), ("b", bb)):
                rec[f"{name}_{side}yy"] = float(ydm[idx] @ ydm[idx])
                rec[f"{name}_{side}yp"] = float(ydm[idx] @ pdm[idx])
                rec[f"{name}_{side}pp"] = float(pdm[idx] @ pdm[idx])

        # --- the sampling phase: random designs, so no search progress -----------
        rec.update(sm_tt=0.0, sm_ty=0.0, sm_n=0)
        if sampling_ok:
            m = g["Phase"].astype(str).str.lower().eq("sampling").to_numpy()
            if m.sum() >= 3:
                ts, ys = t[m], y[m]
                rec.update(sm_tt=float((ts - ts.mean()) @ (ts - ts.mean())),
                           sm_ty=float((ts - ts.mean()) @ (ys - ys.mean())), sm_n=int(m.sum()))

        # --- saturation ----------------------------------------------------------
        late = t >= t.min() + 2.0 / 3.0 * rec["span"]
        best_all, worst_all = np.ones(n, bool), np.ones(n, bool)
        best_any, worst_any = np.zeros(n, bool), np.zeros(n, bool)
        for col, sign in zip(base, signs):
            lo, hi = bnds[col]
            v = g[col].to_numpy(float)
            at_min, at_max = np.abs(v - lo) < EPS, np.abs(v - hi) < EPS
            rec[f"sat_min_{col}"] = int(at_min.sum())
            rec[f"sat_max_{col}"] = int(at_max.sum())
            best, worst = (at_max, at_min) if sign > 0 else (at_min, at_max)
            best_all &= best
            worst_all &= worst
            best_any |= best
            worst_any |= worst
        rec.update(sat_best_all=int(best_all.sum()), sat_worst_all=int(worst_all.sum()),
                   sat_best_any=int(best_any.sum()), sat_worst_any=int(worst_any.sum()),
                   sat_late_n=int(late.sum()), sat_best_any_late=int(best_any[late].sum()),
                   sat_early_n=int((~late).sum()), sat_best_any_early=int(best_any[~late].sum()))
        rows.append(rec)
    return pd.DataFrame(rows).fillna(0.0), participants


def shared_design_residuals(df: pd.DataFrame, params: list[str], min_raters: int = 3) -> pd.Series:
    """Rating minus the mean of OTHER participants' ratings of the same design."""
    key = design_key(df, params)
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(key):
        if g["participant"].nunique() < min_raters:
            continue
        for idx, row in g.iterrows():
            others = g.loc[g["participant"] != row["participant"], "composite"]
            out[idx] = row["composite"] - others.mean()
    return out


def crossover_pairs(df: pd.DataFrame, params: list[str], participants: list[str],
                    value: str = "composite") -> pd.DataFrame:
    """provoice: one design rated once in each condition by the same participant."""
    if "Condition_ID" not in df.columns or df["Condition_ID"].nunique() != 2:
        return pd.DataFrame(columns=["pid", "dy", "dt"])
    c1, c2 = sorted(df["Condition_ID"].unique())
    df = df.assign(_key=design_key(df, params))
    rows = []
    for pid, g in df.groupby("participant"):
        a = g[g["Condition_ID"] == c1].drop_duplicates("_key", keep=False).set_index("_key")
        b = g[g["Condition_ID"] == c2].drop_duplicates("_key", keep=False).set_index("_key")
        for k in a.index.intersection(b.index):
            rows.append({"pid": participants.index(pid),
                         "dy": float(b.at[k, value] - a.at[k, value]),
                         "dt": float(b.at[k, "sitting_trial"] - a.at[k, "sitting_trial"])})
    return pd.DataFrame(rows, columns=["pid", "dy", "dt"])


def crossover_slope(cross: pd.DataFrame, w: np.ndarray) -> float:
    """Weighted OLS slope of the condition-2-minus-1 difference on the sitting-trial gap."""
    wc = w[cross["pid"].to_numpy(int)].astype(float)
    dt, dy = cross["dt"].to_numpy(float), cross["dy"].to_numpy(float)
    W = wc.sum()
    mdt, mdy = wc @ dt / W, wc @ dy / W
    sxx = wc @ (dt - mdt) ** 2
    return float(wc @ ((dt - mdt) * (dy - mdy)) / sxx) if sxx > 0 else np.nan


def simulate_ar_readings(sess: pd.DataFrame, rho: float, reps: int, rng: np.random.Generator) -> dict:
    """What each lag-1 estimator reads under a pure AR(1) at these session lengths."""
    lengths = sess.loc[sess["eligible"] > 0, "n"].astype(int).to_numpy()
    acc = {"mean_raw": [], "mean_corr": [], "pooled_raw": [], "pooled_corr": [], "detr": []}
    nbar = lengths.mean()
    for _ in range(reps):
        num = den = num_dt = den_dt = 0.0
        r1s, r1c = [], []
        for n in lengths:
            e = np.empty(n)
            e[0] = rng.normal()
            innov = rng.normal(0.0, math.sqrt(1 - rho ** 2), n)
            for k in range(1, n):
                e[k] = rho * e[k - 1] + innov[k]
            d = e - e.mean()
            nu, de = float(d[:-1] @ d[1:]), float(d @ d)
            num += nu
            den += de
            r1s.append(nu / de)
            r1c.append((n * nu / de + 1) / (n - 3))
            t = np.arange(n) - (n - 1) / 2
            dt = d - (t @ d) / (t @ t) * t
            num_dt += float(dt[:-1] @ dt[1:])
            den_dt += float(dt @ dt)
        acc["mean_raw"].append(np.mean(r1s))
        acc["mean_corr"].append(np.mean(r1c))
        acc["pooled_raw"].append(num / den)
        acc["pooled_corr"].append((nbar * num / den + 1) / (nbar - 3))
        acc["detr"].append(num_dt / den_dt)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# Statistics as functions of participant weights
# --------------------------------------------------------------------------- #

def _anova_share(n, s1, s2, w):
    keep = (w > 0) & (n > 0)
    n, s1, s2, w = n[keep], s1[keep], s2[keep], w[keep]
    a = w.sum()
    N = float(w @ n)
    if a < 3 or N - a < 2:
        return np.nan, np.nan, np.nan
    grand = float(w @ s1) / N
    msw = float(w @ (s2 - s1 ** 2 / n)) / (N - a)
    msb = float(w @ (n * (s1 / n - grand) ** 2)) / (a - 1)
    n0 = (N - float(w @ n ** 2) / N) / (a - 1)
    var_b = max(0.0, (msb - msw) / n0)
    return var_b / (var_b + msw), var_b, msw


def _wmedian(values: np.ndarray, weights: np.ndarray) -> float:
    rep = np.repeat(values, weights.astype(int))
    return float(np.median(rep)) if len(rep) else np.nan


def make_stat_function(sess: pd.DataFrame, part: pd.DataFrame, shared: pd.DataFrame,
                       cross: pd.DataFrame, sigma_f: float, detr_paper: float, detr_null: float):
    """Return f(w) -> dict of every statistic; w = participant bootstrap counts.

    detr_paper / detr_null: the detrended lag-1 estimator's reading under AR(1)
    rho = 0.8 and under white noise, at these session lengths.
    """
    spid = sess["pid"].to_numpy(int)
    c = {k: sess[k].to_numpy(float) for k in sess.columns if k != "session"}
    elig = c["eligible"] > 0
    n_s = c["n"]
    span = c["span"]
    median_span = float(np.median(span[elig]))
    cross_pid = cross["pid"].to_numpy(int)
    cdt, cdy = cross["dt"].to_numpy(float), cross["dy"].to_numpy(float)

    def resid_ss(b, yy, yp, pp):
        return yy - 2 * b * yp + b * b * pp

    def f(w: np.ndarray) -> dict[str, float]:
        ws = w[spid].astype(float)
        we = ws * elig
        T = lambda k, weights=ws: np.float64(weights @ c[k])  # noqa: E731
        out: dict[str, float] = {}

        # ---------------- design model ----------------
        n = T("n")
        sst = T("syy") - T("sy") ** 2 / n
        sse = T("syy") - 2 * T("syp") + T("spp")
        out["design_model_lopo_r2"] = 1 - sse / sst
        cal_slope = (T("syp") - T("sy") * T("sp") / n) / (T("spp") - T("sp") ** 2 / n)
        out["design_model_calibration_slope"] = cal_slope
        A = np.array([[T("tt", we), T("tp", we)], [T("tp", we), T("pp", we)]])
        beta_t, b = np.linalg.solve(A, np.array([T("ty", we), T("yp", we)]))
        out["design_model_within_session_coef"] = b

        # ---------------- noise ----------------
        m = T("nn_m")
        nug = (np.sqrt((T("nn_ss") - T("nn_s") ** 2 / m) / (m - 1)) / np.sqrt(2)
               if m >= 2 else np.nan)
        out["noise_nn_close_ratings"] = nug
        out["noise_nn_close_over_sigma_f"] = nug / sigma_f
        inv_nug = 1.0 / nug if np.isfinite(nug) and nug > 0 else np.nan
        out["noise_nn_close_share_lag1_pairs"] = T("nn_lag1") / m if m else np.nan
        out["noise_nn_close_attenuation_under_paper_ar1"] = np.sqrt(T("nn_ar") / m) if m else np.nan
        if "nnc_m" in c:
            mc = T("nnc_m")
            out["noise_nn_close_excl_mixed_pairs_over_sigma_f"] = (
                np.sqrt((T("nnc_ss") - T("nnc_s") ** 2 / mc) / (mc - 1)) / np.sqrt(2) / sigma_f
                if mc >= 2 else np.nan)
        out["noise_within_session_resid_sd_over_sigma_f"] = np.sqrt(
            float(ws @ resid_ss(b, c["yy"], c["yp"], c["pp"])) / float(ws @ (n_s - 1))) / sigma_f

        # ---------------- temporal semivariogram of close pairs ----------------
        g = {k: (T(f"{k}_ss") / (2 * T(f"{k}_m")) if T(f"{k}_m") > 0 else np.nan)
             for k in ("g1", "g23", "g4")}
        out["semivar_lag1_over_sigma_f2"] = g["g1"] / sigma_f ** 2
        out["semivar_lag2to3_over_sigma_f2"] = g["g23"] / sigma_f ** 2
        out["semivar_lag4plus_over_sigma_f2"] = g["g4"] / sigma_f ** 2
        out["semivar_ratio_lag1_to_lag4plus"] = (g["g1"] / g["g4"]
                                                 if np.isfinite(g["g4"]) and g["g4"] > 0 else np.nan)
        m4 = T("g4_m")
        out["semivar_ratio_paper_ar1_prediction"] = ((1 - PAPER_RHO) / (T("g4_paper") / m4)
                                                     if m4 > 0 else np.nan)

        # ---------------- drift ----------------
        de = elig & (ws > 0)
        wd = ws[de]
        wsum = wd.sum()
        tt, ty, tp = c["tt"][de], c["ty"][de], c["tp"][de]
        progress = tp / tt * span[de]

        def per_session(bb):
            slope = (ty - bb * tp) / tt
            ss = resid_ss(bb, c["yy"][de], c["yp"][de], c["pp"][de]) - slope ** 2 * tt
            se = np.sqrt(np.maximum(ss, 0) / (n_s[de] - 2) / tt)
            with np.errstate(divide="ignore", invalid="ignore"):
                pval = 2 * stats.t.sf(np.abs(slope / se), n_s[de] - 2)
            return slope * span[de], se * span[de], np.nan_to_num(pval, nan=1.0)

        totals, se_tot, pval = per_session(b)
        out["drift_resid_median_total_over_sigma_f"] = _wmedian(totals, wd) / sigma_f
        out["drift_resid_full_removal_median_total_over_sigma_f"] = _wmedian(per_session(1.0)[0], wd) / sigma_f
        out["drift_naive_median_total_over_sigma_f"] = _wmedian(per_session(0.0)[0], wd) / sigma_f
        out["search_progress_median_total_over_sigma_f"] = _wmedian(progress, wd) / sigma_f
        sig = pval < 0.05
        out["drift_share_sessions_significant"] = float(wd @ sig) / wsum
        out["drift_share_sessions_significant_positive"] = float(wd @ (sig & (totals > 0))) / wsum
        out["drift_share_sessions_significant_negative"] = float(wd @ (sig & (totals < 0))) / wsum
        out["drift_expected_share_significant_under_paper_drift"] = float(wd @ c["dr_power"][de]) / wsum
        mean_t = float(wd @ totals) / wsum
        var_t = float(wd @ (totals - mean_t) ** 2) / max(wsum - 1, 1)
        out["drift_between_session_sd_over_sigma_f"] = np.sqrt(
            max(0.0, var_t - float(wd @ se_tot ** 2) / wsum)) / sigma_f
        TT, TY, TP = T("tt", we), T("ty", we), T("tp", we)
        out["drift_resid_pooled_total_over_sigma_f"] = beta_t * median_span / sigma_f
        out["drift_resid_full_removal_pooled_total_over_sigma_f"] = (TY - TP) / TT * median_span / sigma_f
        out["drift_naive_pooled_total_over_sigma_f"] = TY / TT * median_span / sigma_f
        out["search_progress_pooled_total_over_sigma_f"] = TP / TT * median_span / sigma_f
        out["drift_resid_pooled_total_over_noise_sd"] = beta_t * median_span * inv_nug
        ll = T("cp_ll")
        out["drift_closepair_total_over_sigma_f"] = (T("cp_ld") / ll * median_span / sigma_f
                                                     if ll > 0 else np.nan)
        out["drift_closepair_total_over_noise_sd"] = (T("cp_ld") / ll * median_span * inv_nug
                                                      if ll > 0 else np.nan)
        smt = T("sm_tt")
        out["drift_sampling_phase_total_over_sigma_f"] = (T("sm_ty") / smt * median_span / sigma_f
                                                          if smt > 0 else np.nan)
        if len(cross):
            wc = w[cross_pid].astype(float)
            W = wc.sum()
            mdt, mdy = float(wc @ cdt) / W, float(wc @ cdy) / W
            sxx = float(wc @ (cdt - mdt) ** 2)
            slope = float(wc @ ((cdt - mdt) * (cdy - mdy))) / sxx if sxx > 0 else np.nan
            res = cdy - mdy - slope * (cdt - mdt)
            out["drift_crossover_total_over_sigma_f"] = slope * median_span / sigma_f
            out["drift_crossover_total_over_noise_sd"] = slope * median_span * inv_nug
            out["drift_crossover_condition_effect_over_sigma_f"] = (mdy - slope * mdt) / sigma_f
            out["noise_crossover_repeat_upper_bound_over_sigma_f"] = (
                np.sqrt(float(wc @ res ** 2) / (W - 2) / 2) / sigma_f)

        # ---------------- lag-one autocorrelation ----------------
        def ar_stats(bb):
            num = c["Lyy"] - bb * (c["Lyp"] + c["Lpy"]) + bb * bb * c["Lpp"]
            den = resid_ss(bb, c["yy"], c["yp"], c["pp"])
            with np.errstate(divide="ignore", invalid="ignore"):
                r1 = num / den
            r1c = (n_s * r1 + 1) / (n_s - 3)
            mean_raw = float(wd @ r1[de]) / wsum
            mean_corr = float(wd @ r1c[de]) / wsum
            pooled = float(we @ num) / float(we @ den)
            nbar = float(wd @ n_s[de]) / wsum
            # detrended: e = r - s t with s = (ty - b tp) / tt per session
            s = np.divide(c["ty"] - bb * c["tp"], c["tt"], out=np.zeros_like(c["tt"]),
                          where=c["tt"] > 0)
            cross_rt = (c["Lyt"] + c["Lty"]) - bb * (c["Lpt"] + c["Ltp"])
            num_dt = np.where(elig, num - s * cross_rt + s * s * c["Ltt"], 0.0)
            den_dt = np.where(elig, den - s * s * c["tt"], 0.0)
            detr = float(we @ num_dt) / float(we @ den_dt)
            return mean_raw, mean_corr, pooled, (nbar * pooled + 1) / (nbar - 3), detr

        mr, mc_, pr, pc, dt = ar_stats(b)
        out["ar1_mean_session_r1_raw"] = mr
        out["ar1_mean_session_r1_bias_corrected"] = mc_
        out["ar1_pooled_r1_raw"] = pr
        out["ar1_pooled_r1_bias_corrected"] = pc
        out["ar1_pooled_r1_detrended_raw"] = dt
        # Residual = AR(1) part + the rest. If the rest is white after detrending,
        # the detrended reading mixes the two readings in proportion to variance.
        out["ar1_rho08_share_of_residual_variance"] = (dt - detr_null) / (detr_paper - detr_null)
        out["ar1_full_removal_pooled_r1_bias_corrected"] = ar_stats(1.0)[3]
        out["ar1_raw_composite_pooled_r1_bias_corrected"] = ar_stats(0.0)[3]
        for name in ("near", "far"):
            num = float(we @ (c[f"{name}_Lyy"] - b * c[f"{name}_Lyp"] + b * b * c[f"{name}_Lpp"]))
            va = float(we @ resid_ss(b, c[f"{name}_ayy"], c[f"{name}_ayp"], c[f"{name}_app"]))
            vb = float(we @ resid_ss(b, c[f"{name}_byy"], c[f"{name}_byp"], c[f"{name}_bpp"]))
            out[f"ar1_consecutive_{name}_designs_r"] = num / np.sqrt(va * vb) if va * vb > 0 else np.nan

        # ---------------- heteroscedasticity ----------------
        e2 = resid_ss(b, c["yy"], c["yp"], c["pp"])            # per-session sum of e^2
        xe2 = c["hxyy"] - 2 * b * c["hxyp"] + b * b * c["hxpp"]
        hn, hx, hxx = T("n"), T("hx"), T("hxx")
        hy, hxy = float(ws @ e2), float(ws @ xe2)
        slope = (hxy - hx * hy / hn) / (hxx - hx ** 2 / hn)
        out["het_resid_var_rel_slope_per_sigma_f"] = slope / (hy / hn)
        ter = [float(ws @ resid_ss(b, c[f"ter{k}_yy"], c[f"ter{k}_yp"], c[f"ter{k}_pp"]))
               / T(f"ter{k}_n") for k in range(3)]
        out["het_resid_sd_ratio_top_to_bottom_tercile"] = np.sqrt(ter[2] / ter[0])
        cn, cx, cxx, cy, cxy = T("cp_m"), T("ch_x"), T("ch_xx"), T("ch_y"), T("ch_xy")
        out["het_closepair_var_rel_slope_per_sigma_f_level"] = (
            ((cxy - cx * cy / cn) / (cxx - cx ** 2 / cn)) / (cy / cn)
            if cn > 3 and cxx - cx ** 2 / cn > 0 else np.nan)

        # ---------------- saturation ----------------
        for k in [k for k in c if k.startswith(("sat_min_", "sat_max_"))]:
            out[f"saturation_share_{k[4:]}"] = T(k) / n
        for k in ("best_all", "worst_all", "best_any", "worst_any"):
            out[f"saturation_share_composite_{k}"] = T(f"sat_{k}") / n
        out["saturation_share_composite_best_any_last_third"] = T("sat_best_any_late") / T("sat_late_n")
        out["saturation_share_composite_best_any_first_two_thirds"] = (T("sat_best_any_early")
                                                                       / T("sat_early_n"))

        # ---------------- participant offsets ----------------
        # Residual y - cal_slope * pred: the design effect removed at the pooled
        # calibrated scale; b = 1 as a sensitivity.
        for label, bb in (("", cal_slope), ("_full_removal", 1.0)):
            s1 = part["sy"].to_numpy() - bb * part["sp"].to_numpy()
            s2 = part["syy"].to_numpy() - 2 * bb * part["syp"].to_numpy() + bb * bb * part["spp"].to_numpy()
            share, vb_, vw_ = _anova_share(part["n"].to_numpy(), s1, s2, w.astype(float))
            out[f"offset_variance_share{label}"] = share
            if not label:
                out["offset_sd_over_sigma_f"] = np.sqrt(vb_) / sigma_f if np.isfinite(vb_) else np.nan
                out["offset_sd_over_within_sd"] = np.sqrt(vb_ / vw_) if np.isfinite(vb_) else np.nan
        s1 = c["sy"] - cal_slope * c["sp"]
        s2 = c["syy"] - 2 * cal_slope * c["syp"] + cal_slope ** 2 * c["spp"]
        out["offset_variance_share_session_level"] = _anova_share(n_s, s1, s2, ws)[0]
        share, vb_, vw_ = _anova_share(shared["n"].to_numpy(), shared["s1"].to_numpy(),
                                       shared["s2"].to_numpy(), w.astype(float))
        out["offset_variance_share_shared_designs"] = share
        out["offset_sd_over_within_sd_shared_designs"] = (np.sqrt(vb_ / vw_)
                                                          if np.isfinite(vb_) else np.nan)
        return out

    return f, median_span


# --------------------------------------------------------------------------- #
# Descriptions, and the value each quantity takes under the paper's processes
# --------------------------------------------------------------------------- #

QUANTITY_INFO: dict[str, tuple[str, str, str]] = {
    # quantity: (units, method, what the simulation assumes)
    "design_model_lopo_r2": ("R^2", "pooled out-of-fold R^2 of the leave-one-participant-out design model (pipeline's oracle family and hyperparameters, no jitter)", ""),
    "design_model_calibration_slope": ("slope", "OLS of rating on the LOPO prediction; below 1 means over-dispersed predictions", ""),
    "design_model_within_session_coef": ("slope", "b: coefficient of the LOPO prediction in the pooled within-session regression of rating on [trial, prediction]", ""),
    "noise_nn_close_ratings": ("rating points", "SD of close within-rater NN pair differences / sqrt 2 (calibrate_noise_from_data.py)", ""),
    "noise_nn_close_over_sigma_f": ("sigma_f", "the paper's anchor: close-NN nugget / sigma_f", "sweep 0.05, 0.25, 1, 5; paper anchor 0.74-3.35"),
    "noise_nn_close_over_paper_sigma_f": ("sigma_f", "scale-consistent nugget over the paper's (mixed-scale) sigma_f", "paper anchor 3.35"),
    "noise_nn_close_share_lag1_pairs": ("share", "share of the nugget's close NN pairs rated on consecutive trials", ""),
    "noise_nn_close_attenuation_under_paper_ar1": ("ratio", "nugget / true SD if errors were AR(1) rho=0.8: sqrt(mean(1 - 0.8^lag)) over the nugget's pairs", "1 under GAUSSIAN"),
    "noise_nn_close_excl_mixed_pairs_over_sigma_f": ("sigma_f", "nugget without the pairs that involve a raw-scale row", ""),
    "noise_within_session_resid_sd_over_sigma_f": ("sigma_f", "SD of within-session residuals y - b*pred (includes design-model misfit, so an upper bound)", ""),
    "noise_crossover_repeat_upper_bound_over_sigma_f": ("sigma_f", "same sampling design rated in both conditions: residual SD / sqrt 2 (includes design x condition, so an upper bound)", ""),
    "semivar_lag1_over_sigma_f2": ("sigma_f^2", "semivariance of close design pairs rated 1 trial apart", "flat in lag"),
    "semivar_lag2to3_over_sigma_f2": ("sigma_f^2", "semivariance of close design pairs rated 2-3 trials apart", "flat in lag"),
    "semivar_lag4plus_over_sigma_f2": ("sigma_f^2", "semivariance of close design pairs rated >= 4 trials apart", "flat in lag"),
    "semivar_ratio_lag1_to_lag4plus": ("ratio", "lag-1 over lag>=4 semivariance of close design pairs (model-free serial dependence)", "1 (GAUSSIAN); paper_expected_value = AR(1) rho 0.8"),
    "semivar_ratio_paper_ar1_prediction": ("ratio", "the same ratio implied by AR(1) with rho = 0.8 at the observed lags", "AR(1)"),
    "drift_resid_median_total_over_sigma_f": ("sigma_f", "median over sessions of the slope of y - b*pred on trial x session span", "DRIFT: +sigma_e (paper_expected_value = the noise anchor)"),
    "drift_resid_full_removal_median_total_over_sigma_f": ("sigma_f", "as above with b = 1 (residual y - pred)", "DRIFT: +sigma_e"),
    "drift_naive_median_total_over_sigma_f": ("sigma_f", "as above with b = 0 (raw composite: drift + search progress)", ""),
    "search_progress_median_total_over_sigma_f": ("sigma_f", "as above on the LOPO prediction (what the model attributes to better designs)", ""),
    "drift_share_sessions_significant": ("share", "share of sessions whose slope of y - b*pred on trial has p < 0.05 (two-sided t)", "0.05 with no drift; paper_expected_value = power under exact DRIFT"),
    "drift_share_sessions_significant_positive": ("share", "significant and positive", "all significant drift positive"),
    "drift_share_sessions_significant_negative": ("share", "significant and negative", "0"),
    "drift_expected_share_significant_under_paper_drift": ("share", "power of the same test at these session lengths if DRIFT held exactly (noncentral t)", "DRIFT"),
    "drift_between_session_sd_over_sigma_f": ("sigma_f", "SD of true per-session total drift, method of moments (variance of totals minus mean SE^2)", "0 (one common ramp)"),
    "drift_resid_pooled_total_over_sigma_f": ("sigma_f", "pooled within-session regression of rating on [trial, prediction]: trial coefficient x median span", "DRIFT: +sigma_e"),
    "drift_resid_full_removal_pooled_total_over_sigma_f": ("sigma_f", "pooled within-session slope of y - pred on trial x median span", "DRIFT: +sigma_e"),
    "drift_naive_pooled_total_over_sigma_f": ("sigma_f", "pooled within-session slope of the raw composite on trial x median span", ""),
    "search_progress_pooled_total_over_sigma_f": ("sigma_f", "pooled within-session slope of the LOPO prediction on trial x median span", ""),
    "drift_resid_pooled_total_over_noise_sd": ("noise SD", "pooled drift over the session / close-NN noise SD", "DRIFT: 1"),
    "drift_closepair_total_over_sigma_f": ("sigma_f", "model-free: rating difference of close design pairs regressed on trial lag (through origin) x median span; biased low by the optimiser's selection of well-rated incumbents", "DRIFT: +sigma_e"),
    "drift_closepair_total_over_noise_sd": ("noise SD", "close-pair drift over the close-NN noise SD", "DRIFT: 1"),
    "drift_sampling_phase_total_over_sigma_f": ("sigma_f", "model-free: pooled within-session slope of the raw composite over the random-design sampling phase, extrapolated to the median span", "DRIFT: +sigma_e"),
    "drift_crossover_total_over_sigma_f": ("sigma_f", "model-free: same sampling design rated in both counterbalanced conditions; difference regressed on the sitting-trial gap, x median session span", "DRIFT: +sigma_e"),
    "drift_crossover_total_over_noise_sd": ("noise SD", "crossover drift over the close-NN noise SD", "DRIFT: 1"),
    "drift_crossover_condition_effect_over_sigma_f": ("sigma_f", "intercept of the crossover regression (condition 2 minus condition 1)", ""),
    "ar1_mean_session_r1_raw": ("r", "mean over sessions of the lag-1 autocorrelation of session-demeaned y - b*pred", "AR(1): 0.8 (paper_expected_value = estimator's reading under it)"),
    "ar1_mean_session_r1_bias_corrected": ("r", "as above with Kendall's correction (n r + 1)/(n - 3) per session", "AR(1): 0.8"),
    "ar1_pooled_r1_raw": ("r", "pooled lag-1: sum of e_t e_t+1 over sum of e_t^2 across sessions", "AR(1): 0.8"),
    "ar1_pooled_r1_bias_corrected": ("r", "pooled lag-1 with Kendall's correction at the mean session length", "AR(1): 0.8"),
    "ar1_pooled_r1_detrended_raw": ("r", "pooled lag-1 after removing each session's linear trend (uncorrected)", "AR(1): 0.8"),
    "ar1_rho08_share_of_residual_variance": ("share", "share of within-session residual variance an AR(1) with rho = 0.8 would carry, from the detrended lag-1, if the rest is white after detrending (approximately linear mix of the two simulated readings)", "AR(1): 1 if the residual were rating error alone"),
    "ar1_full_removal_pooled_r1_bias_corrected": ("r", "pooled, corrected, residual y - pred (b = 1)", "AR(1): 0.8"),
    "ar1_raw_composite_pooled_r1_bias_corrected": ("r", "pooled, corrected, raw composite (b = 0; search trajectory included)", ""),
    "ar1_consecutive_near_designs_r": ("r", "residual correlation of consecutive trials whose designs are closer than the median step", "AR(1): same at any distance"),
    "ar1_consecutive_far_designs_r": ("r", "same for consecutive designs farther apart than the median step", "AR(1): same at any distance"),
    "het_resid_var_rel_slope_per_sigma_f": ("relative variance per sigma_f", "OLS of squared within-session residual on the LOPO prediction, over the mean squared residual", "0 (homoscedastic)"),
    "het_resid_sd_ratio_top_to_bottom_tercile": ("ratio", "within-session residual SD in the top over the bottom tercile of LOPO prediction", "1 (homoscedastic)"),
    "het_breusch_pagan_p": ("p", "Breusch-Pagan test of the within-session residual on the LOPO prediction (ignores clustering)", ""),
    "het_closepair_var_rel_slope_per_sigma_f_level": ("relative variance per sigma_f", "model-free: squared close-pair difference on the pair's mean rating, over its mean", "0 (homoscedastic)"),
    "resid_excess_kurtosis": ("kurtosis", "excess kurtosis of within-session residuals (b fixed at its full-sample value)", "0 (gaussian)"),
    "resid_share_beyond_3sd": ("share", "share of within-session residuals beyond 3 SD", "0.0027 (gaussian)"),
    "saturation_share_composite_best_all": ("share", "every construct at its best-end bound (composite at its ceiling)", "0 in the main arms; the capped arm cuts the top 10% of the landscape"),
    "saturation_share_composite_worst_all": ("share", "every construct at its worst-end bound", "0"),
    "saturation_share_composite_best_any": ("share", "at least one construct at its best-end bound", "0"),
    "saturation_share_composite_worst_any": ("share", "at least one construct at its worst-end bound", "0"),
    "saturation_share_composite_best_any_last_third": ("share", "at least one construct at its best end, last third of each session", "0"),
    "saturation_share_composite_best_any_first_two_thirds": ("share", "same, first two thirds of each session", "0"),
    "offset_variance_share": ("share", "one-way random-effects (ANOVA) share of a per-participant intercept in y - c*pred, c the calibration slope", "BIAS: one rater offset by sigma_e; as a random effect SD sigma_e, share 0.5"),
    "offset_variance_share_full_model_removal": ("share", "the same with the residual y - pred", "0.5"),
    "offset_variance_share_full_removal": ("share", "the same with the residual y - pred", "0.5"),
    "offset_sd_over_sigma_f": ("sigma_f", "SD of the per-participant intercept", "BIAS: sigma_e"),
    "offset_sd_over_within_sd": ("ratio", "offset SD over within-participant residual SD", "BIAS: 1"),
    "offset_variance_share_session_level": ("share", "as offset_variance_share with sessions as the groups", "0.5"),
    "offset_variance_share_shared_designs": ("share", "model-free: same decomposition of rating minus the other participants' mean on designs rated by >= 3 participants", "0.5"),
    "offset_sd_over_within_sd_shared_designs": ("ratio", "offset SD over within SD on the shared designs", "1"),
}


def identification(q: str, info: dict, r2: float) -> str:
    if q.startswith(("saturation_", "design_model", "sigma_f", "pipeline_")):
        return "identified"
    def by_count(count: int, what: str, tag: str = "") -> str:
        if count < 10:
            return f"not identified: {count} {what}"
        if count < 30:
            return f"weak: {count} {what}"
        return "identified" + tag

    if q.startswith("noise_nn"):
        return by_count(info["nn_close_pairs"], "close NN pairs")
    if q.startswith("noise_crossover") or q.startswith("drift_crossover"):
        ok = (info["crossover_participants"] >= 10 and min(info["crossover_c1_first"],
                                                           info["crossover_c2_first"]) >= 4)
        return "identified (model-free)" if ok else "weak: too few crossover participants"
    if q.startswith("het_closepair"):
        return by_count(info["close_pairs_all"], "close pairs", " (model-free)")
    if q.startswith("semivar") or q.startswith("drift_closepair"):
        return by_count(info["close_pairs_lag4plus"], "close pairs at lag >= 4", " (model-free)")
    if q.startswith("drift_sampling_phase"):
        return "identified (model-free, random designs; imprecise, 5 trials per session)"
    if q.startswith("offset_") and "shared" in q:
        ok = info["shared_design_participants"] >= 10 and info["shared_design_rows"] >= 50
        return "identified (model-free)" if ok else "weak: few shared designs"
    if q.startswith("drift_expected"):
        return "identified"
    if q.startswith("drift_naive"):
        return "identified (raw trend; includes search progress)"
    if q == "noise_within_session_resid_sd_over_sigma_f":
        return "identified (upper bound)"
    # model-based: residual quantities depend on how well the design effect is removed
    if r2 <= 0:
        return f"not identified: design model held-out R^2 = {r2:.2f} <= 0"
    if r2 < 0.5:
        return f"weak: design model held-out R^2 = {r2:.2f}"
    return "identified"


def analyse_basis(label: str, dataset: sim.DatasetConfig, df: pd.DataFrame, model_name: str,
                  sigma_f: float, n_jobs: int, B: int, rng: np.random.Generator,
                  is_mixed: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    params = dataset.param_columns
    df = df.reset_index(drop=True).copy()
    df["pred"] = lopo_predictions(df, dataset, model_name, n_jobs)
    bounds = sim.bounds_from_data(df, params)
    threshold = CLOSE_PAIR_FRACTION * math.sqrt(len(params))
    pred_terciles = np.quantile(df["pred"], [1 / 3, 2 / 3])
    steps = []
    for _, g in df.groupby("session", sort=False):
        if len(g) < MIN_SESSION_TRIALS:
            continue
        g = g.sort_values("trial")
        ok = np.diff(g["trial"].to_numpy()) == 1
        steps.extend(np.linalg.norm(np.diff(_unit(g[params].to_numpy(float), bounds), axis=0),
                                    axis=1)[ok].tolist())
    consec_cut = float(np.median(steps))
    # The sampling phase is free of search progress only if its designs are
    # drawn afresh per participant rather than a fixed sequence shown to all.
    sampling_ok = False
    sampling_shared = np.nan
    if "Phase" in df.columns:
        sm_rows = df["Phase"].astype(str).str.lower().eq("sampling")
        if sm_rows.any():
            key = design_key(df, params)
            n_raters = df.groupby(key)["participant"].transform("nunique")
            sampling_shared = float((n_raters[sm_rows] > 1).mean())
            sampling_ok = sampling_shared < 0.1
    sess, participants = build_sessions(df, dataset, sigma_f, bounds, threshold, is_mixed,
                                        pred_terciles, consec_cut, sampling_ok)
    P = len(participants)
    pid_of_row = df["participant"].map({p: i for i, p in enumerate(participants)}).to_numpy()
    part = sess.groupby("pid")[["n", "sy", "sp", "syy", "spp", "syp"]].sum().reindex(range(P)).fillna(0.0)
    shared = shared_design_residuals(df, params)
    sh = pd.DataFrame({"pid": pid_of_row, "r": shared}).dropna()
    shared_sums = (sh.groupby("pid")["r"].agg(n="size", s1="sum", s2=lambda s: float(s @ s))
                   .reindex(range(P)).fillna(0.0))
    cross = crossover_pairs(df, params, participants)

    ar_paper = simulate_ar_readings(sess, PAPER_RHO, AR_SIMULATIONS, np.random.default_rng(SEED))
    ar_null = simulate_ar_readings(sess, 0.0, AR_SIMULATIONS, np.random.default_rng(SEED + 1))
    f, median_span = make_stat_function(sess, part, shared_sums, cross, sigma_f,
                                        ar_paper["detr"], ar_null["detr"])
    point = f(np.ones(P, dtype=int))
    counts = rng.multinomial(P, np.full(P, 1.0 / P), size=B)
    boot = pd.DataFrame([f(cnt) for cnt in counts])
    # provoice: which constructs drift across the sitting (raw logged direction)
    if len(cross) and label == "pipeline":
        _, base_cols, _ = composite_columns(dataset)
        for col in base_cols:
            cp = crossover_pairs(df, params, participants, value=col)
            q = f"drift_crossover_construct_{col.replace(' ', '_')}_points_per_session"
            point[q] = crossover_slope(cp, np.ones(P)) * median_span
            boot[q] = [crossover_slope(cp, cnt) * median_span for cnt in counts]
            QUANTITY_INFO.setdefault(q, (
                "rating points", f"model-free crossover drift of {col} as logged (not sign-adjusted), "
                "x median session span", ""))

    # full-sample only: Breusch-Pagan, kurtosis and tails at the fitted b
    b = point["design_model_within_session_coef"]
    e = (df["composite"] - b * df["pred"]).to_numpy()
    e = e - pd.Series(e).groupby(df["session"]).transform("mean").to_numpy()
    point["het_breusch_pagan_p"] = float(het_breuschpagan(e, sm.add_constant(df["pred"].to_numpy()))[1])
    z_sd = float(np.std(e, ddof=1))
    kurt_rows = pd.DataFrame({"pid": pid_of_row, "e": e})
    kboot = []
    for cnt in counts:
        idx = np.concatenate([np.flatnonzero(pid_of_row == k) for k in np.repeat(np.arange(P), cnt)])
        ee = e[idx]
        kboot.append((stats.kurtosis(ee), float(np.mean(np.abs(ee) > 3 * z_sd))))
    kboot = np.array(kboot)
    point["resid_excess_kurtosis"] = float(stats.kurtosis(e))
    point["resid_share_beyond_3sd"] = float(np.mean(np.abs(e) > 3 * z_sd))
    boot["resid_excess_kurtosis"] = kboot[:, 0]
    boot["resid_share_beyond_3sd"] = kboot[:, 1]
    del kurt_rows

    elig = sess["eligible"] > 0
    info = {
        "n_participants": P, "n_ratings": len(df), "n_sessions": len(sess),
        "n_eligible_sessions": int(elig.sum()),
        "n_eligible_participants": int(sess.loc[elig, "pid"].nunique()),
        "n_eligible_ratings": int(sess.loc[elig, "n"].sum()),
        "session_length_min_median_max": [int(sess["n"].min()), float(sess["n"].median()),
                                          int(sess["n"].max())],
        "median_session_span": median_span,
        "nn_close_pairs": int(sess["nn_m"].sum()), "close_pairs_all": int(sess["cp_m"].sum()),
        "close_pairs_lag1": int(sess["g1_m"].sum()), "close_pairs_lag2to3": int(sess["g23_m"].sum()),
        "close_pairs_lag4plus": int(sess["g4_m"].sum()),
        "shared_design_rows": int(sh.shape[0]),
        "shared_design_participants": int(sh["pid"].nunique()),
        "sampling_phase_rows_on_shared_designs": sampling_shared,
        "sampling_phase_drift_estimated": sampling_ok,
        "sampling_phase_rows": int(sess["sm_n"].sum()),
        "crossover_pairs": int(len(cross)),
        "crossover_participants": int(cross["pid"].nunique()) if len(cross) else 0,
        "crossover_c1_first": int((cross.groupby("pid")["dt"].mean() > 0).sum()) if len(cross) else 0,
        "crossover_c2_first": int((cross.groupby("pid")["dt"].mean() < 0).sum()) if len(cross) else 0,
        "consecutive_step_median": consec_cut,
        "ar_estimators_under_paper_ar1": ar_paper, "ar_estimators_under_white_noise": ar_null,
        "sigma_f": sigma_f, "bootstrap_nan_share": {},
    }
    r2 = point["design_model_lopo_r2"]
    nug_ratio = point["noise_nn_close_over_sigma_f"]
    paper_expected = {
        "noise_nn_close_attenuation_under_paper_ar1": 1.0,
        "semivar_ratio_lag1_to_lag4plus": point["semivar_ratio_paper_ar1_prediction"],
        "drift_resid_median_total_over_sigma_f": nug_ratio,
        "drift_resid_full_removal_median_total_over_sigma_f": nug_ratio,
        "drift_resid_pooled_total_over_sigma_f": nug_ratio,
        "drift_resid_full_removal_pooled_total_over_sigma_f": nug_ratio,
        "drift_closepair_total_over_sigma_f": nug_ratio,
        "drift_sampling_phase_total_over_sigma_f": nug_ratio,
        "drift_crossover_total_over_sigma_f": nug_ratio,
        "drift_resid_pooled_total_over_noise_sd": 1.0,
        "drift_closepair_total_over_noise_sd": 1.0,
        "drift_crossover_total_over_noise_sd": 1.0,
        "drift_share_sessions_significant": point["drift_expected_share_significant_under_paper_drift"],
        "drift_share_sessions_significant_negative": 0.0,
        "drift_between_session_sd_over_sigma_f": 0.0,
        "ar1_mean_session_r1_raw": ar_paper["mean_raw"],
        "ar1_mean_session_r1_bias_corrected": ar_paper["mean_corr"],
        "ar1_pooled_r1_raw": ar_paper["pooled_raw"],
        "ar1_pooled_r1_bias_corrected": ar_paper["pooled_corr"],
        "ar1_full_removal_pooled_r1_bias_corrected": ar_paper["pooled_corr"],
        "ar1_pooled_r1_detrended_raw": ar_paper["detr"],
        "ar1_rho08_share_of_residual_variance": 1.0,
        "ar1_consecutive_near_designs_r": ar_paper["pooled_raw"],
        "ar1_consecutive_far_designs_r": ar_paper["pooled_raw"],
        "het_resid_var_rel_slope_per_sigma_f": 0.0,
        "het_resid_sd_ratio_top_to_bottom_tercile": 1.0,
        "het_closepair_var_rel_slope_per_sigma_f_level": 0.0,
        "resid_excess_kurtosis": 0.0,
        "resid_share_beyond_3sd": 0.0027,
        "offset_variance_share": 0.5, "offset_variance_share_full_removal": 0.5,
        "offset_variance_share_session_level": 0.5, "offset_variance_share_shared_designs": 0.5,
        "offset_sd_over_sigma_f": nug_ratio, "offset_sd_over_within_sd": 1.0,
        "offset_sd_over_within_sd_shared_designs": 1.0,
    }
    null_expected = {
        "noise_nn_close_attenuation_under_paper_ar1": 1.0,
        "semivar_ratio_lag1_to_lag4plus": 1.0,
        "drift_share_sessions_significant": 0.05,
        "ar1_mean_session_r1_raw": ar_null["mean_raw"],
        "ar1_mean_session_r1_bias_corrected": ar_null["mean_corr"],
        "ar1_pooled_r1_raw": ar_null["pooled_raw"],
        "ar1_pooled_r1_bias_corrected": ar_null["pooled_corr"],
        "ar1_full_removal_pooled_r1_bias_corrected": ar_null["pooled_corr"],
        "ar1_pooled_r1_detrended_raw": ar_null["detr"],
        "ar1_rho08_share_of_residual_variance": 0.0,
        "ar1_consecutive_near_designs_r": ar_null["pooled_raw"],
        "ar1_consecutive_far_designs_r": ar_null["pooled_raw"],
    }
    for q in ("drift_resid_median_total_over_sigma_f", "drift_resid_pooled_total_over_sigma_f",
              "drift_resid_full_removal_median_total_over_sigma_f",
              "drift_resid_full_removal_pooled_total_over_sigma_f", "drift_closepair_total_over_sigma_f",
              "drift_sampling_phase_total_over_sigma_f", "drift_crossover_total_over_sigma_f",
              "drift_resid_pooled_total_over_noise_sd", "drift_closepair_total_over_noise_sd",
              "drift_crossover_total_over_noise_sd", "drift_between_session_sd_over_sigma_f",
              "het_resid_var_rel_slope_per_sigma_f", "het_closepair_var_rel_slope_per_sigma_f_level",
              "offset_variance_share", "offset_variance_share_shared_designs"):
        null_expected.setdefault(q, 0.0)
    null_expected["het_resid_sd_ratio_top_to_bottom_tercile"] = 1.0
    null_expected["resid_excess_kurtosis"] = 0.0
    null_expected["resid_share_beyond_3sd"] = 0.0027

    rows = []
    for q, est in point.items():
        if q in boot.columns:
            vals = boot[q].to_numpy(float)
            finite = vals[np.isfinite(vals)]
            if len(finite) < B:
                info["bootstrap_nan_share"][q] = 1 - len(finite) / B
            lo, hi = (np.percentile(finite, [2.5, 97.5]) if len(finite) >= 0.9 * B else (np.nan, np.nan))
        else:
            lo = hi = np.nan
        if not np.isfinite(est):
            continue
        if q.startswith(("drift_", "ar1_", "search_progress")) and not q.startswith(
                ("drift_closepair", "drift_crossover", "drift_sampling")):
            npart, nrat = info["n_eligible_participants"], info["n_eligible_ratings"]
        elif q.startswith("drift_sampling"):
            npart, nrat = int(sess.loc[sess["sm_n"] > 0, "pid"].nunique()), info["sampling_phase_rows"]
        elif q.startswith("offset_") and "shared" in q:
            npart, nrat = info["shared_design_participants"], info["shared_design_rows"]
        elif q.startswith("noise_nn"):
            npart, nrat = int(sess.loc[sess["nn_m"] > 0, "pid"].nunique()), info["nn_close_pairs"]
        elif q.startswith(("semivar", "drift_closepair", "het_closepair")):
            npart, nrat = int(sess.loc[sess["cp_m"] > 0, "pid"].nunique()), info["close_pairs_all"]
        elif q.startswith(("drift_crossover", "noise_crossover")):
            npart, nrat = info["crossover_participants"], info["crossover_pairs"]
        else:
            npart, nrat = P, len(df)
        units, method, paper = QUANTITY_INFO.get(q, ("", "", ""))
        if q.startswith(("saturation_share_min_", "saturation_share_max_")):
            col = q.split("_", 3)[3]
            lo_b, hi_b = INSTRUMENT_BOUNDS[dataset.name][col]
            end = "minimum" if "_min_" in q else "maximum"
            units = "share"
            method = f"share of {col} ratings at the instrument {end} (scale {lo_b:g} to {hi_b:g})"
            paper = "0 in the main arms"
            paper_expected[q] = 0.0
        if q.startswith(("noise_nn", "semivar", "het_closepair", "drift_closepair")):
            method += " [n_ratings counts pairs]"
        if q.startswith(("drift_crossover", "noise_crossover")):
            method += " [n_ratings counts design pairs]"
        rows.append({"dataset": dataset.name, "data_basis": label, "quantity": q,
                     "estimate": est, "ci_low": lo, "ci_high": hi,
                     "n_participants": npart, "n_ratings": nrat, "units": units,
                     "paper_assumption": paper,
                     "paper_expected_value": paper_expected.get(q, np.nan),
                     "null_expected_value": null_expected.get(q, np.nan),
                     "identification": identification(q, info, r2),
                     "sigma_f_used": sigma_f, "method": method})
    return pd.DataFrame(rows), info, df


def simple_row(dataset, basis, q, est, units, method, n_part, n_rat, sigma_f, lo=np.nan, hi=np.nan,
               paper="", ident="identified"):
    return {"dataset": dataset, "data_basis": basis, "quantity": q, "estimate": est, "ci_low": lo,
            "ci_high": hi, "n_participants": n_part, "n_ratings": n_rat, "units": units,
            "paper_assumption": paper, "paper_expected_value": np.nan, "null_expected_value": np.nan,
            "identification": ident, "sigma_f_used": sigma_f, "method": method}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    datasets = sim.parse_dataset_configs(None, args.dataset_config, args.dataset_cache_dir)
    selection = sim.load_oracle_selection(args.oracle_selection_path)
    manifest = json.loads(args.manifest.read_text())
    calib = pd.read_csv(args.noise_calibration)
    anchor = pd.read_csv(args.noise_anchor)

    all_rows: list[pd.DataFrame] = []
    infos: dict[str, dict] = {}
    for dataset in datasets:
        name = dataset.name
        model_name = selection[(name, OBJECTIVE)]["best_model"]
        cv_r2 = float(selection[(name, OBJECTIVE)]["scores"][model_name])
        _, base, _ = composite_columns(dataset)
        print(f"\n=== {name}: oracle family {model_name}, oracle_target {dataset.oracle_target}",
              flush=True)

        raw = sim.load_observations(dataset, OBJECTIVE)           # what the pipeline loads
        order_problems = check_presentation_order(raw)
        dedup = cal.drop_full_duplicates(raw, dataset.param_columns, base)
        df = annotate(dedup, dataset)
        mixed = logged_scale_violations(df, dataset)

        # Reproduce the paper's nugget with the function that produced it.
        cal_rows = pd.DataFrame(cal.calibrate_dataset(dataset, CLOSE_PAIR_FRACTION, False))
        repro = float(cal_rows.loc[cal_rows["objective"] == "composite", "sd_nn_close"].iloc[0])
        stored = float(calib.loc[(calib["dataset"] == name) & (calib["objective"] == "composite"),
                                 "sd_nn_close"].iloc[0])
        stored_ratio = float(anchor.loc[anchor["dataset"] == name, "noise_in_landscape_sd"].iloc[0])
        sigma_manifest = float(manifest[name]["sigma_f"])
        sigma_recomputed = landscape_sigma_f(raw, dataset, model_name, args.n_jobs)
        print(f"  nugget reproduced {repro:.6f} (stored {stored:.6f}); sigma_f manifest "
              f"{sigma_manifest:.6f}, recomputed {sigma_recomputed:.6f}; ratio "
              f"{repro / sigma_manifest:.4f} (stored {stored_ratio:.4f})", flush=True)

        info_ds = {"oracle_family": model_name, "pipeline_cv_r2": cv_r2,
                   "pipeline_cv_strategy": selection[(name, OBJECTIVE)].get("validation_strategy"),
                   "oracle_target": dataset.oracle_target,
                   "rows_loaded": len(raw), "rows_after_dedup": len(dedup),
                   "order_problem_files": order_problems,
                   "nugget_reproduced": repro, "nugget_stored": stored,
                   "sigma_f_manifest": sigma_manifest, "sigma_f_recomputed": sigma_recomputed,
                   "anchor_ratio_stored": stored_ratio, "raw_scale_rows": int(mixed.sum())}
        n_p, n_r = df["participant"].nunique(), len(df)
        rows_meta = [
            simple_row(name, "pipeline", "sigma_f_manifest", sigma_manifest, "composite points",
                       "output/per_dataset/manifest.json (used by the paper)", n_p, len(raw), sigma_manifest),
            simple_row(name, "pipeline", "sigma_f_recomputed", sigma_recomputed, "composite points",
                       "anchor_noise_scale.py procedure re-run: selected oracle, jitter, 100k uniform draws, seed 10007",
                       n_p, len(raw), sigma_manifest),
            simple_row(name, "pipeline", "pipeline_oracle_cv_r2", cv_r2, "R^2",
                       f"output/best_oracle_models.json ({info_ds['pipeline_cv_strategy']} CV, jitter, "
                       f"oracle_target {dataset.oracle_target})", n_p, len(raw), sigma_manifest),
        ]

        if mixed.any():
            runs = sorted(df.loc[mixed, "Run"].unique().tolist()) if "Run" in df else []
            groups = sorted(df.loc[mixed, "Group_ID"].unique().tolist()) if "Group_ID" in df else []
            print(f"  {int(mixed.sum())} of {len(df)} rows lie outside the logged [-1, 1] scale "
                  f"(runs {runs}; groups {groups})", flush=True)
            info_ds["raw_scale_runs"], info_ds["raw_scale_groups"] = runs, groups
            key = design_key(df, dataset.param_columns)
            raters = df.groupby(key)["participant"].transform("nunique")
            info_ds["raw_scale_rows_on_designs_rated_by_many"] = int((mixed & (raters > 1)).sum())
            info_ds["raw_scale_composite_mean"] = float(df.loc[mixed, "composite"].mean())
            info_ds["normalised_composite_mean"] = float(df.loc[~mixed, "composite"].mean())
            rows_mix, pipe_info, _ = analyse_basis("pipeline", dataset, df, model_name, sigma_manifest,
                                                   args.n_jobs, args.bootstrap, rng, is_mixed=mixed)
            keep = rows_mix["quantity"].isin([
                "noise_nn_close_ratings", "noise_nn_close_over_sigma_f",
                "noise_nn_close_excl_mixed_pairs_over_sigma_f", "noise_nn_close_share_lag1_pairs",
                "design_model_lopo_r2", "design_model_calibration_slope"])
            all_rows.append(rows_mix[keep])
            info_ds["pipeline_basis"] = pipe_info
            clean_raw = raw[~logged_scale_violations(raw, dataset)]
            sigma_basis = landscape_sigma_f(clean_raw, dataset, model_name, args.n_jobs)
            info_ds["sigma_f_scale_consistent"] = sigma_basis
            print(f"  scale-consistent sigma_f {sigma_basis:.6f}", flush=True)
            rows_meta.append(simple_row(
                name, "scale_consistent", "sigma_f_recomputed", sigma_basis,
                "composite points (normalised scale)",
                "anchor_noise_scale.py procedure on the rows logged on the [-1, 1] scale",
                df.loc[~mixed, "participant"].nunique(), len(clean_raw), sigma_basis))
            full_bases = [("scale_consistent", dataset, df[~mixed], sigma_basis)]
        else:
            full_bases = [("pipeline", dataset, df, sigma_manifest)]
        if name in SIGN_CORRECTIONS:
            fixed = dataclasses.replace(
                dataset, objective_map={**dataset.objective_map, OBJECTIVE: SIGN_CORRECTIONS[name]})
            sigma_fixed = landscape_sigma_f(raw, fixed, model_name, args.n_jobs)
            info_ds["sigma_f_sign_corrected"] = sigma_fixed
            info_ds["sign_corrected_composite"] = SIGN_CORRECTIONS[name]
            print(f"  sign-corrected composite {SIGN_CORRECTIONS[name]}: sigma_f {sigma_fixed:.6f}",
                  flush=True)
            rows_meta.append(simple_row(
                name, "sign_corrected", "sigma_f_recomputed", sigma_fixed, "composite points",
                f"anchor_noise_scale.py procedure on the composite {SIGN_CORRECTIONS[name]}",
                n_p, len(raw), sigma_fixed))
            full_bases.append(("sign_corrected", fixed, annotate(dedup, fixed), sigma_fixed))

        for basis, ds_b, frame, sigma_b in full_bases:
            rows_b, info_b, frame_pred = analyse_basis(basis, ds_b, frame, model_name, sigma_b,
                                                       args.n_jobs, args.bootstrap, rng)
            all_rows.append(rows_b)
            info_ds[f"{basis}_basis"] = info_b
            if basis == "scale_consistent":
                nug = rows_b.loc[rows_b["quantity"] == "noise_nn_close_ratings"].iloc[0]
                rows_meta.append(simple_row(
                    name, basis, "noise_nn_close_over_paper_sigma_f", nug["estimate"] / sigma_manifest,
                    "sigma_f", QUANTITY_INFO["noise_nn_close_over_paper_sigma_f"][1],
                    nug["n_participants"], nug["n_ratings"], sigma_manifest,
                    nug["ci_low"] / sigma_manifest, nug["ci_high"] / sigma_manifest,
                    "paper anchor 3.35", nug["identification"]))
            # Does another oracle family remove the design effect better?
            if not args.skip_model_comparison:
                y = frame_pred["composite"].to_numpy()
                comparison = {}
                for alt in ALT_DESIGN_MODELS:
                    pr = (frame_pred["pred"].to_numpy() if alt == model_name
                          else lopo_predictions(frame_pred, ds_b, alt, args.n_jobs))
                    comparison[alt] = float(1 - np.sum((y - pr) ** 2) / np.sum((y - y.mean()) ** 2))
                    rows_meta.append(simple_row(
                        name, basis, f"design_model_lopo_r2_{alt}", comparison[alt], "R^2",
                        f"LOPO held-out R^2 with the {alt} family (comparison; the analysis uses "
                        f"{model_name})", frame_pred["participant"].nunique(), len(frame_pred), sigma_b))
                info_ds[f"design_model_lopo_r2_by_family_{basis}"] = comparison
                print(f"  [{basis}] LOPO R^2 by family: "
                      + ", ".join(f"{k} {v:.3f}" for k, v in comparison.items()), flush=True)
        all_rows.append(pd.DataFrame(rows_meta))
        infos[name] = info_ds

    table = pd.concat(all_rows, ignore_index=True)
    csv_path = args.output_dir / "archival_error_processes.csv"
    table.to_csv(csv_path, index=False)
    (args.output_dir / "archival_error_processes.json").write_text(
        json.dumps(infos, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {csv_path} ({len(table)} rows)")
    write_summary(table, infos, args.output_dir / "archival_error_processes.md", args.bootstrap)


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #


COLUMNS = [("ehmi", "pipeline"), ("opticarvis", "scale_consistent"), ("provoice", "pipeline"),
           ("provoice", "sign_corrected")]
COLUMN_LABELS = ["ehmi", "opticarvis (scale-consistent)", "provoice (pipeline composite)",
                 "provoice (Predictability sign corrected)"]


def _get(table, ds, q, basis):
    row = table[(table["dataset"] == ds) & (table["data_basis"] == basis) & (table["quantity"] == q)]
    return None if row.empty else row.iloc[0]


def _fmt(row, digits=2, pct=False):
    if row is None or not np.isfinite(row["estimate"]):
        return "n/a"
    k = 100.0 if pct else 1.0
    est = f"{k * row['estimate']:.{digits}f}"
    if np.isfinite(row["ci_low"]):
        est = f"{est} [{k * row['ci_low']:.{digits}f}, {k * row['ci_high']:.{digits}f}]"
    if not str(row.get("identification", "identified")).startswith("identified"):
        est += " (" + str(row["identification"]).split(":")[0] + ")"
    return est


def _num(value, digits=2, pct=False):
    if value is None or not np.isfinite(value):
        return ""
    return f"{(100.0 if pct else 1.0) * value:.{digits}f}"


def _position(row, value) -> str:
    if row is None or not np.isfinite(row["ci_low"]) or value is None or not np.isfinite(value):
        return "no interval"
    if value < row["ci_low"]:
        return "paper below CI"
    if value > row["ci_high"]:
        return "paper above CI"
    return "paper inside CI"


def verdict_lines(table: pd.DataFrame, cols: list[tuple[str, str]]) -> list[str]:
    def v(d, b, q, digits=2, pct=False):
        r = _get(table, d, q, b)
        if r is None or not np.isfinite(r["estimate"]):
            return "n/a"
        k = 100.0 if pct else 1.0
        s = f"{k * r['estimate']:.{digits}f}"
        if np.isfinite(r["ci_low"]):
            s += f" [{k * r['ci_low']:.{digits}f}, {k * r['ci_high']:.{digits}f}]"
        return s

    def ref(d, b, q, key="paper_expected_value", digits=2):
        r = _get(table, d, q, b)
        return _num(r[key], digits) if r is not None else "n/a"

    def hi(d, b, q, pct=True):
        r = _get(table, d, q, b)
        return _num(r["ci_high"], 0, pct) if r is not None else "n/a"

    def outside(q):
        rows = [_get(table, d, q, b) for d, b in cols]
        return sum(1 for r in rows if r is not None and _position(r, r["paper_expected_value"])
                   in ("paper above CI", "paper below CI")), len([r for r in rows if r is not None])

    tag = {c: f"{c[0]}" + ("" if c[1] in ("pipeline", "scale_consistent") else " (sign-corrected)")
           for c in cols}
    E, O, P, PS = cols[0], cols[1], cols[2], cols[3] if len(cols) > 3 else cols[2]
    ar_out, ar_n = outside("ar1_pooled_r1_bias_corrected")
    het_out, het_n = outside("het_resid_sd_ratio_top_to_bottom_tercile")
    off_out, off_n = outside("offset_variance_share_shared_designs")
    return [
        "## Verdict against the simulation's assumptions", "",
        "- **Noise anchor.** " + "; ".join(
            f"{tag[c]} {v(*c, 'noise_nn_close_over_sigma_f')}" for c in cols)
        + f" sigma_f. The paper's 3.35 for opticarvis is "
        f"{v('opticarvis', 'pipeline', 'noise_nn_close_over_sigma_f')} on the mixed-scale "
        f"composite, and "
        f"{v('opticarvis', 'scale_consistent', 'noise_nn_close_over_paper_sigma_f')} on the "
        "scale-consistent rows in the paper's sigma_f. The provoice value rests on 16 pairs. On "
        "consistent composites, rating noise is roughly 0.5-1 sigma_f, not 0.74-3.35. If "
        "errors were serially dependent the nugget would understate it, because most of its "
        "pairs are consecutive trials.",
        f"- **AR(1), rho = 0.8: outside.** The pooled, bias-corrected lag-1 of the residuals is "
        + ", ".join(f"{tag[c]} {v(*c, 'ar1_pooled_r1_bias_corrected')}" for c in cols)
        + f". The same estimator reads {ref(*E, 'ar1_pooled_r1_bias_corrected')}-"
        f"{ref(*O, 'ar1_pooled_r1_bias_corrected')} under the paper's AR(1), and "
        f"{ar_out} of {ar_n} intervals exclude that. After each session's linear trend is "
        "removed, the lag-1 is near the white-noise reading ("
        + ", ".join(f"{_num(_get(table, c[0], 'ar1_pooled_r1_detrended_raw', c[1])['estimate'])} vs "
                    f"{ref(*c, 'ar1_pooled_r1_detrended_raw', 'null_expected_value')}" for c in cols)
        + "), so the correlation is mostly session-specific trend. If the rest of the residual "
        "were white, an AR(1) with rho = 0.8 could carry at most "
        + "/".join(f"{hi(*c, 'ar1_rho08_share_of_residual_variance')}" for c in cols)
        + "% of the residual variance (upper 95% bounds). Model-free, ehmi's close designs "
        "differ more on consecutive trials than far apart (semivariance ratio "
        f"{v(*E, 'semivar_ratio_lag1_to_lag4plus')}, against "
        f"{ref(*E, 'semivar_ratio_lag1_to_lag4plus')} under AR(1)), the opposite of carry-over. "
        "Opticarvis's ratio is uninformative, and provoice has 2 long-lag close pairs. An AR(1) "
        "with rho = 0.8 as the whole error process is outside the data. A minority "
        "serially-correlated component is not excluded.",
        "- **DRIFT, a common linear ramp to +sigma_e: not identified, except in provoice.** "
        "Residual drift reverses sign when the model's predicted search progress is removed in "
        f"full instead of at the fitted b: {tag[E]} {v(*E, 'drift_resid_median_total_over_sigma_f')} "
        f"vs {v(*E, 'drift_resid_full_removal_median_total_over_sigma_f')}, and {tag[P]} "
        f"{v(*P, 'drift_resid_median_total_over_sigma_f')} vs "
        f"{v(*P, 'drift_resid_full_removal_median_total_over_sigma_f')} sigma_f per session. The "
        "one identified estimate is provoice's counterbalanced crossover: "
        f"{v(*P, 'drift_crossover_total_over_sigma_f')} sigma_f per session on the pipeline "
        f"composite (paper {ref(*P, 'drift_crossover_total_over_sigma_f')}; "
        f"{v(*P, 'drift_crossover_total_over_noise_sd')} noise SDs against 1) and "
        f"{v(*PS, 'drift_crossover_total_over_sigma_f')} on the sign-corrected one (paper "
        f"{ref(*PS, 'drift_crossover_total_over_sigma_f')}). It is construct-specific. In "
        "logged rating points per session, Mental Demand changes by "
        f"{v('provoice', 'pipeline', 'drift_crossover_construct_Mental_Demand_points_per_session')} "
        "(it eases), Predictability by "
        f"{v('provoice', 'pipeline', 'drift_crossover_construct_Predictability_points_per_session')} "
        "(lower is better, so it worsens, borderline), and Usefulness by "
        f"{v('provoice', 'pipeline', 'drift_crossover_construct_Percieved_Usefulness_points_per_session')}. "
        "Session slopes take both signs ("
        + ", ".join(f"{tag[c]} {v(*c, 'drift_share_sessions_significant_negative', 0, True)}%"
                    for c in cols)
        + " significantly negative), and the between-session SD of drift is "
        + ", ".join(f"{v(*c, 'drift_between_session_sd_over_sigma_f')}" for c in cols)
        + " sigma_f. That points to rater-specific drift rather than one common positive ramp, "
        "though search progress the model misses can contribute. Provoice's drift magnitude is "
        "about one to two noise SDs per session, within reach of the paper's, but not its shape.",
        "- **BIAS, a constant per-rater offset: inside, as a between-rater spread.** The offset "
        "SD over the within SD (BIAS = 1) is "
        + ", ".join(f"{tag[c]} {v(*c, 'offset_sd_over_within_sd')}" for c in cols)
        + ". The model-free variance share on the fixed shared designs is "
        + ", ".join(f"{v(*c, 'offset_variance_share_shared_designs')}" for c in cols)
        + f", with 0.5 inside {off_n - off_out} of {off_n} intervals. Raters differ by about "
        "one noise SD in level, so one rater offset by sigma_e is a plausible draw. Whether "
        "every rater is shifted the same way is unidentifiable.",
        "- **Homoscedastic gaussian noise: outside.** The residual SD in the top prediction "
        "tercile relative to the bottom one is "
        + ", ".join(f"{tag[c]} {v(*c, 'het_resid_sd_ratio_top_to_bottom_tercile')}" for c in cols)
        + f", with 1 outside {het_out} of {het_n} intervals. The model-free close-pair slope "
        "agrees in sign, significantly in ehmi "
        f"({v(*E, 'het_closepair_var_rel_slope_per_sigma_f_level')}). High-rated designs are "
        "rated more consistently, as a ceiling compresses them. The excess kurtosis is "
        + ", ".join(f"{v(*c, 'resid_excess_kurtosis', 1)}" for c in cols)
        + ", and "
        + ", ".join(f"{v(*c, 'resid_share_beyond_3sd', 1, True)}%" for c in cols)
        + " of residuals lie beyond 3 SD, against 0.3% under a gaussian.",
        "- **Saturation: the unbounded main arms are unrealistic.** At least one construct is at "
        "its best-end bound in "
        + ", ".join(f"{tag[c]} {v(*c, 'saturation_share_composite_best_any', 0, True)}%" for c in cols)
        + " of ratings, and in "
        + ", ".join(f"{v(*c, 'saturation_share_composite_best_any_last_third', 0, True)}%" for c in cols)
        + " in the last third of a session. The whole composite is at its ceiling in "
        + ", ".join(f"{v(*c, 'saturation_share_composite_best_all', 1, True)}%" for c in cols)
        + ". The capped-scale arm, not the uncapped main sweep, is the realistic case for the "
        "top of the landscape.", ""]


def write_summary(table: pd.DataFrame, infos: dict, path: Path, B: int) -> None:
    cols = [(d, b) for d, b in COLUMNS if f"{b}_basis" in infos.get(d, {})]
    labels = [COLUMN_LABELS[COLUMNS.index(c)] for c in cols]

    def line(label, q, digits=2, pct=False, paper=True, null=False):
        cells = []
        for d, b in cols:
            r = _get(table, d, q, b)
            cell = _fmt(r, digits, pct)
            if r is not None and paper and np.isfinite(r["paper_expected_value"]):
                cell += f"; paper {_num(r['paper_expected_value'], digits, pct)}"
            if r is not None and null and np.isfinite(r["null_expected_value"]):
                cell += f"; white noise {_num(r['null_expected_value'], digits, pct)}"
            cells.append(cell)
        return f"| {label} | " + " | ".join(cells) + " |"

    header = ["| quantity | " + " | ".join(labels) + " |", "|---" * (len(cols) + 1) + "|"]
    oc = infos["opticarvis"]
    pv = infos["provoice"]
    L = ["# Error processes estimated from the three archival studies", "",
         "Generated by `scripts/estimate_archival_error_processes.py`. Intervals are participant "
         f"bootstraps ({B} resamples of participants with replacement) and are conditional on the "
         "leave-one-participant-out (LOPO) design model. \"paper x\" is the value the estimator "
         "would read if the paper's process held at the measured noise level. For the lag-1 rows "
         "it is the estimator's simulated reading under AR(1) with rho = 0.8 at these session "
         "lengths, and \"white noise\" is its reading with no serial dependence. A parenthesised "
         "flag marks a quantity that the design does not identify, or identifies only weakly "
         "(the `identification` column of the CSV). The full table is "
         "`archival_error_processes.csv`, and the counts and checks are in "
         "`archival_error_processes.json`.", ""]

    L += ["## Reproduction of the paper's anchor, with intervals", ""]
    for d in ("ehmi", "opticarvis", "provoice"):
        i = infos[d]
        L.append(f"- **{d}**: the nugget is {i['nugget_reproduced']:.4f} rating points (stored "
                 f"{i['nugget_stored']:.4f}). sigma_f is {i['sigma_f_manifest']:.4f} in the manifest "
                 f"and {i['sigma_f_recomputed']:.4f} recomputed. The ratio "
                 f"{i['nugget_reproduced'] / i['sigma_f_manifest']:.3f} equals the paper's "
                 f"{i['anchor_ratio_stored']:.3f}. Its participant-bootstrap 95% interval is "
                 f"{_fmt(_get(table, d, 'noise_nn_close_over_sigma_f', 'pipeline'))}, from "
                 f"{i['pipeline_basis']['nn_close_pairs']} close pairs.")
    L += ["", "The paper's range of 0.74-3.35 therefore has intervals of its own. The top of the "
          "range is an artefact of the opticarvis scale mixing described below.", ""]

    L += ["## Two defects in the pipeline's composites", "",
          f"**Opticarvis mixes two rating scales.** {oc['raw_scale_rows']} of the "
          f"{oc['rows_after_dedup']} deduplicated rows (Run {oc.get('raw_scale_runs')}, groups "
          f"{oc.get('raw_scale_groups')}) log the raw instrument values, on the 1-5, 1-7 and -3..3 "
          "scales. Every other row logs the value the optimiser saw, rescaled to [-1, 1] as "
          "(x - midpoint) / half-range. Their composite means are "
          f"{oc.get('raw_scale_composite_mean', float('nan')):.2f} and "
          f"{oc.get('normalised_composite_mean', float('nan')):.2f}. "
          f"{oc.get('raw_scale_rows_on_designs_rated_by_many', 0)} raw-scale rows sit on the "
          "GroupE standard design, which every GroupE participant rated. The pipeline averages "
          "both scales into one composite, so that design is a spike of about 4 on a surface "
          "that otherwise lies between -1 and 1. The spike sets the oracle's optimum "
          "(y_opt = 5.36, above the normalised maximum of 1, hence opt_z = 13.4), carries its "
          "held-out R^2, and inflates the nugget.", "",
          f"- **Pipeline composite.** The nugget is "
          f"{_fmt(_get(table, 'opticarvis', 'noise_nn_close_over_sigma_f', 'pipeline'))} sigma_f, "
          "and "
          f"{_fmt(_get(table, 'opticarvis', 'noise_nn_close_excl_mixed_pairs_over_sigma_f', 'pipeline'))} "
          "without the close pairs that involve a raw-scale row.",
          f"- **Scale-consistent rows.** The nugget is "
          f"{_fmt(_get(table, 'opticarvis', 'noise_nn_close_over_sigma_f', 'scale_consistent'))} "
          f"in this basis's own sigma_f ({oc.get('sigma_f_scale_consistent', float('nan')):.4f}), "
          "and "
          f"{_fmt(_get(table, 'opticarvis', 'noise_nn_close_over_paper_sigma_f', 'scale_consistent'))} "
          "in the paper's sigma_f.",
          f"- **Held-out (LOPO) R^2 of the pipeline's gradient-boosting model.** It is "
          f"{_fmt(_get(table, 'opticarvis', 'design_model_lopo_r2', 'pipeline'))} on the pipeline "
          "composite and "
          f"{_fmt(_get(table, 'opticarvis', 'design_model_lopo_r2', 'scale_consistent'))} on the "
          "scale-consistent rows, where no family tried reaches zero. The pipeline's own group-CV "
          f"R^2 is {oc['pipeline_cv_r2']:.3f}. The opticarvis oracle's apparent fidelity comes "
          "from the spike.", "",
          "**Provoice enters Predictability with the wrong sign.** The logged Predictability is "
          "lower-is-better. The study's own R scripts invert it (6 - x) \"because predictability "
          "is maximised\", its Pareto rows average 1.3 against 2.4 for the rest, and it "
          "correlates -0.70 with Usefulness and +0.46 with Mental Demand. `datasets.json` enters "
          "it as +Predictability, so the pipeline composite partly cancels two constructs that "
          "agree. The sign-corrected composite "
          f"{pv.get('sign_corrected_composite')} has sigma_f "
          f"{pv.get('sigma_f_sign_corrected', float('nan')):.4f}, against "
          f"{pv['sigma_f_manifest']:.4f}. Its nugget is "
          f"{_fmt(_get(table, 'provoice', 'noise_nn_close_over_sigma_f', 'sign_corrected'))} "
          "sigma_f, and its LOPO R^2 is "
          f"{_fmt(_get(table, 'provoice', 'design_model_lopo_r2', 'sign_corrected'))}, against "
          f"{_fmt(_get(table, 'provoice', 'design_model_lopo_r2', 'pipeline'))} for the pipeline "
          "composite.", ""]

    def family_cells():
        out = []
        for d, b in cols:
            comp = infos[d].get(f"design_model_lopo_r2_by_family_{b}", {})
            out.append(" / ".join(_num(comp.get(m)) or "n/a" for m in ALT_DESIGN_MODELS))
        return out

    L += ["## How well search progress can be removed", ""] + header + [
        line("LOPO held-out R^2 (pipeline family)", "design_model_lopo_r2", paper=False),
        "| LOPO R^2: extra trees / random forest / gradient boosting | " + " | ".join(family_cells()) + " |",
        "| pipeline's own CV R^2 (row k-fold for ehmi, group k-fold otherwise) | " + " | ".join(
            f"{infos[d]['pipeline_cv_r2']:.2f}" if b == "pipeline" else "n/a" for d, b in cols) + " |",
        line("calibration slope of rating on the LOPO prediction", "design_model_calibration_slope", paper=False),
        line("b, within-session coefficient on the prediction", "design_model_within_session_coef", paper=False),
        ""]

    L += ["## Estimates", ""] + header + [
        "| **noise** | " + " | " * len(cols),
        line("close-NN nugget / sigma_f (the paper's anchor)", "noise_nn_close_over_sigma_f", paper=False),
        line("share of the nugget's pairs on consecutive trials, %", "noise_nn_close_share_lag1_pairs", 0, True, paper=False),
        line("nugget / true SD if AR(1) rho = 0.8 held", "noise_nn_close_attenuation_under_paper_ar1", paper=False),
        line("upper bound: within-session residual SD / sigma_f", "noise_within_session_resid_sd_over_sigma_f", paper=False),
        line("residual excess kurtosis", "resid_excess_kurtosis"),
        line("residuals beyond 3 SD, %", "resid_share_beyond_3sd", 1, True),
        "| **drift over one session** | " + " | " * len(cols),
        line("naive trend / sigma_f (median session)", "drift_naive_median_total_over_sigma_f", paper=False),
        line("search progress the model sees / sigma_f (median)", "search_progress_median_total_over_sigma_f", paper=False),
        line("drift / sigma_f, y - b pred (median session)", "drift_resid_median_total_over_sigma_f"),
        line("drift / sigma_f, y - b pred (pooled within-session)", "drift_resid_pooled_total_over_sigma_f"),
        line("drift / sigma_f, y - pred (b = 1, median)", "drift_resid_full_removal_median_total_over_sigma_f"),
        line("drift / noise SD, pooled", "drift_resid_pooled_total_over_noise_sd"),
        line("model-free, close design pairs / sigma_f", "drift_closepair_total_over_sigma_f"),
        line("**model-free, counterbalanced crossover / sigma_f**", "drift_crossover_total_over_sigma_f"),
        line("**model-free, crossover / noise SD**", "drift_crossover_total_over_noise_sd"),
        line("sessions with a significant slope, %", "drift_share_sessions_significant", 0, True),
        line("  significant and negative, %", "drift_share_sessions_significant_negative", 0, True),
        line("between-session SD of drift / sigma_f", "drift_between_session_sd_over_sigma_f"),
        "| **serial dependence** | " + " | " * len(cols),
        line("lag-1, per-session mean, bias-corrected", "ar1_mean_session_r1_bias_corrected", null=True),
        line("lag-1, pooled, bias-corrected", "ar1_pooled_r1_bias_corrected", null=True),
        line("lag-1, pooled, per-session trend removed", "ar1_pooled_r1_detrended_raw", null=True),
        line("share of residual variance an AR(1) rho = 0.8 would carry, if the rest is white", "ar1_rho08_share_of_residual_variance", null=True),
        line("lag-1, consecutive designs nearer than the median step", "ar1_consecutive_near_designs_r", null=True),
        line("lag-1, consecutive designs farther than the median step", "ar1_consecutive_far_designs_r", null=True),
        line("model-free: close-pair semivariance, lag 1 / lag >= 4", "semivar_ratio_lag1_to_lag4plus", null=True),
        "| **heteroscedasticity** | " + " | " * len(cols),
        line("residual SD, top / bottom prediction tercile", "het_resid_sd_ratio_top_to_bottom_tercile"),
        line("relative residual variance per sigma_f of prediction", "het_resid_var_rel_slope_per_sigma_f"),
        line("model-free: close-pair variance per sigma_f of rating level", "het_closepair_var_rel_slope_per_sigma_f_level"),
        "| **saturation** | " + " | " * len(cols),
        line("composite at its ceiling (every construct at best end), %", "saturation_share_composite_best_all", 1, True, paper=False),
        line("at least one construct at its best end, %", "saturation_share_composite_best_any", 0, True, paper=False),
        line("  in the last third of the session, %", "saturation_share_composite_best_any_last_third", 0, True, paper=False),
        line("at least one construct at its worst end, %", "saturation_share_composite_worst_any", 0, True, paper=False),
        "| **participant offsets** | " + " | " * len(cols),
        line("random-intercept variance share", "offset_variance_share"),
        line("offset SD / within SD", "offset_sd_over_within_sd"),
        line("offset SD / sigma_f", "offset_sd_over_sigma_f"),
        line("model-free, on shared designs: variance share", "offset_variance_share_shared_designs"),
    ]
    pvb = pv["pipeline_basis"]
    L += ["", f"Provoice's crossover: {pvb['crossover_c1_first']} participants did condition 1 "
          f"first and {pvb['crossover_c2_first']} did condition 2 first, over "
          f"{pvb['crossover_pairs']} design pairs. It bounds the noise on the pipeline composite "
          f"at {_fmt(_get(table, 'provoice', 'noise_crossover_repeat_upper_bound_over_sigma_f', 'pipeline'))} "
          "sigma_f, and on the sign-corrected composite at "
          f"{_fmt(_get(table, 'provoice', 'noise_crossover_repeat_upper_bound_over_sigma_f', 'sign_corrected'))}. "
          "The drift-over-noise ratio has a long upper tail, because provoice's nugget rests on "
          f"only {pvb['nn_close_pairs']} close pairs. By construct, as logged, in rating points "
          "per session: "
          + "; ".join(f"{c} {_fmt(_get(table, 'provoice', f'drift_crossover_construct_{c.replace(chr(32), chr(95))}_points_per_session', 'pipeline'))}"
                      for c in INSTRUMENT_BOUNDS["provoice"])
          + ". Predictability and Mental Demand are lower-is-better, Usefulness higher-is-better.", ""]
    L += verdict_lines(table, cols)

    sat = table[table["quantity"].str.startswith(("saturation_share_min_", "saturation_share_max_"))]
    L += ["## Saturation by construct (% of raw ratings at the instrument bound)", "",
          "| dataset | construct | at minimum | at maximum |", "|---|---|---|---|"]
    for d, b in (("ehmi", "pipeline"), ("opticarvis", "scale_consistent"), ("provoice", "pipeline")):
        sub = sat[(sat["dataset"] == d) & (sat["data_basis"] == b)]
        for col in INSTRUMENT_BOUNDS[d]:
            lo = sub[sub["quantity"] == f"saturation_share_min_{col}"]
            hi = sub[sub["quantity"] == f"saturation_share_max_{col}"]
            if not lo.empty:
                L.append(f"| {d} | {col} | {_fmt(lo.iloc[0], 1, True)} | {_fmt(hi.iloc[0], 1, True)} |")
    L += ["", "Provoice's Predictability floor is its best end (see the sign defect above). "
          "Instrument bounds: " + "; ".join(f"{d}: {BOUND_PROVENANCE[d]}"
                                           for d in ("ehmi", "opticarvis", "provoice")) + ".", ""]

    r2s = {f"{d}/{b}": float(_get(table, d, "design_model_lopo_r2", b)["estimate"]) for d, b in cols}
    flips = [f"{d}/{b}" for d, b in cols
             if np.sign(_get(table, d, "drift_resid_median_total_over_sigma_f", b)["estimate"])
             != np.sign(_get(table, d, "drift_resid_full_removal_median_total_over_sigma_f", b)["estimate"])]
    flip_text = (("and the two differ in sign for " + ", ".join(flips)) if flips
                 else "and the two agree in sign everywhere")
    lag4 = {f"{d}/{b}": infos[d][f"{b}_basis"]["close_pairs_lag4plus"] for d, b in cols}
    L += ["## What the data can and cannot identify", "",
          "- **Search progress confounds trial order.** Later designs are better, so a naive "
          "rating-versus-trial slope mixes drift with progress. A LOPO design model removes "
          "progress only as well as it predicts held-out participants. Its R^2 is "
          + ", ".join(f"{v:.2f} ({k})" for k, v in r2s.items()) + ". Within a session, "
          "trial and predicted quality rise together, so a weak model cannot divide the trend "
          "between them. The model-based drift moves from the fitted b to b = 1 as more of the "
          f"model's progress is removed, {flip_text}. Model-based drift is therefore flagged weak "
          "or not identified.",
          "- **Fixed sampling sequences.** ehmi (36 of 37 participants) and provoice show every "
          "participant the same initial designs in the same order. Within a session, the "
          "sampling phase cannot separate drift from design. The LOPO prediction of those rows "
          "is the other participants' mean at the same position, which absorbs any drift common "
          "to all participants.",
          "- **The model-free estimates.** Provoice's counterbalanced crossover (the same design "
          "rated in the first and in the second condition of one sitting) is free of both "
          "confounds, and it is the one drift estimate the design identifies. It assumes no "
          "carryover between conditions. Close-design-pair drift is biased low: a pair exists "
          "because the optimiser returned near an earlier design, which it does when that design "
          "was rated well, so the earlier rating is selected upward. It rests on "
          + ", ".join(f"{v} ({k})" for k, v in lag4.items()) + " close pairs at lag >= 4.",
          "- **Serial dependence.** The lag-1 correlation of residuals is not a clean estimate of "
          "a rater's serial dependence. Design-model misfit that consecutive close designs share "
          "inflates it, and session-to-session differences in drift inflate it too. Misfit that is "
          "white along the trajectory dilutes it. Removing each session's linear trend deals with "
          "the drift, and the near/far split checks the shared misfit. The share row converts "
          "the detrended lag-1 into the largest share of residual variance an AR(1) with rho = 0.8 "
          "could carry. The close-pair semivariogram needs no model: under AR(1), close designs "
          "rated on consecutive trials would differ far less than close designs rated far apart, "
          "unless the design difference within a close pair exceeds the noise. The nugget's close pairs are mostly consecutive trials, so "
          "if errors were AR(1) the nugget would understate the noise (the attenuation row).",
          "- **A constant offset shared by every rater cannot be identified**, because nothing "
          "measures a design's value except the ratings. What can be identified is how much "
          "raters differ from one another: the random-intercept share. Because each session is "
          "personalised, that share also absorbs design-model misfit in each participant's region "
          "of the design space. The shared-design row is its model-free check, on the fixed "
          "sampling designs.",
          f"- **Short sessions.** Slopes and lag-1 use sessions of at least {MIN_SESSION_TRIALS} "
          "trials. At 10-20 trials, a per-session slope test has little power even under the "
          "paper's exact DRIFT, which is the paper value shown in that row. The test is also "
          "anti-conservative if residuals are autocorrelated.", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
