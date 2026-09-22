"""Reviewer response: the smooth, low-modal subset, per-landscape results, and GP fit at n = 50.

A reviewer made three connected points about the 20-landscape BOBA arm:

  (i)   the thirteen classical functions are multimodal global-optimisation
        benchmarks, while HITL objectives are smooth with few modes;
  (ii)  pooled results average a unit (fractions of opt_z) that differs across
        landscapes, so per-landscape distributions should be primary;
  (iii) the posterior-mean ship rule (pm) losing to the single best rating is
        more likely surrogate misspecification (a stationary RBF GP on 50 points
        of a multimodal function) than the corrupted-posterior story. Question 6:
        does pm beat the best single rating on the smooth landscapes?

This script answers all three from existing logs, adding no BO run.

1. The smooth, low-modal subset is fixed A PRIORI, from the landscape
   definitions alone, before any result is read: the six human-performance laws
   of boba_benchmarks (yerkes_dodson, stevens, hicks_law, weber_fechner,
   power_law_practice, steering_law). Each is unimodal or monotone in every
   input by construction. The thirteen classical functions are the comparison;
   moving_peaks belongs to neither and appears only in "all 20". A secondary,
   equally a-priori split of the classical functions separates the three whose
   definitions make them convex or effectively unimodal on BOBA's box (powell,
   rosenbrock, griewank) from the other ten; it is reported as a sensitivity
   check, never as the answer. The definition is written to
   smooth_subset_definition.csv before anything else is computed.

2. Headline by group (smooth_subset.csv). From analysis/cell_means.csv without
   the model-free floors: the deployed cost (inference_excess / opt_z) and the
   search cost (fragility, the post-onset per-iteration excess, already / opt_z)
   per magnitude x onset x error process, as means of landscape means with a
   95% landscape bootstrap. From analysis/ship_rules_per_run.csv, through
   decompose_regret's own functions: the final-trial search loss
   (regret_best_visited) and the selection share of the deployed excess. The
   two pipelines must agree on the deployed excess to 5e-4 in every cell (the
   same check make_boba_paper_tables applies) or the script stops.

3. Per-landscape distribution (per_landscape_headline.csv): the same costs per
   landscape and cell, with a crossed (pigeonhole) bootstrap over acquisitions
   and seeds within the landscape, from the per-run paired metrics; the per-run
   deployed excess of the two pipelines must agree to 1e-8.

4. Ship rules on the subset (ship_rules_subset.csv): pm, lcb1, lcb2, best_mean
   and the oracle best_visited against best_observed with the paper's estimand,
   through analyse_ship_rules' own loader and pairing and
   analyse_boba_adaptations.summarise (recovered = gain / cost, ratio of
   landscape means, landscape bootstrap, Wilcoxon over landscapes). Because a
   six-landscape bootstrap is coarse, a second interval conditions on the
   landscapes and resamples acquisitions and seeds (crossed). The paper's other
   ship-rule number, the share of the SELECTION loss a rule removes
   (decompose_regret.rule_recovery), is given beside it.

5. GP fit at n = 50 (gp_diagnostics.csv, gp_diagnostics_per_run.csv,
   gp_fit_vs_pm.csv): for every landscape, LogEI and qNEI, seeds 7-16, gaussian
   error at 1 SD from the first rating and the matching clean runs, the loop's
   own GP is refitted with rescore_ship_rules.fit_loop_gp (identical to the
   simulator, same torch seed as the ship-rule rescoring, so its pm pick must
   reproduce the table's). Reported: closed-form leave-one-out predictive log
   likelihood (fixed hyperparameters, Rasmussen & Williams 5.4.2) and its gain
   over a trivial mean/variance predictor; the RMSE of the posterior mean
   against the TRUE objective at the visited designs and at 256 fresh scrambled
   Sobol points, in landscape-SD units (the objective is standardised over its
   box, so predicting 0 scores about 1); the fitted lengthscales as a fraction
   of the box width (Normalize maps the box to the unit cube); the fitted noise
   SD in landscape-SD units (gp.likelihood.noise is the variance of the
   STANDARDISED targets, multiplied here by outcome_transform.stdvs**2) against
   the injected SD. Whether fit quality explains pm's result is tested across
   landscapes and within landscapes across runs.

    python scripts/smooth_subset_and_gp_diagnostics.py --workers 4
    python scripts/smooth_subset_and_gp_diagnostics.py --reuse-gp     # re-aggregate only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import analyse_boba_adaptations as aba  # noqa: E402
import analyse_ship_rules as asr  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
import decompose_regret as dr  # noqa: E402
from rescore_ship_rules import MODEL_FREE, file_seed, fit_loop_gp, latent_posterior  # noqa: E402

# ---------------------------------------------------------------------------
# 1. The a-priori subset. Written from the landscape DEFINITIONS in
#    boba_benchmarks.py; no result of this project enters it.
# ---------------------------------------------------------------------------

HUMAN_LAWS = ("yerkes_dodson", "stevens", "hicks_law", "weber_fechner", "power_law_practice", "steering_law")
CLASSICAL = ("schwefel", "powell", "eggholder", "ackley", "shekel", "griewank", "hartmann_3", "hartmann_6",
             "branin", "rosenbrock", "rastrigin", "michalewicz", "levy_10")
OTHER = ("moving_peaks",)
# Secondary, also a priori: classical functions whose definition makes them
# convex or effectively unimodal on BOBA's box.
SMOOTH_CLASSICAL = ("powell", "rosenbrock", "griewank")
RUGGED_CLASSICAL = tuple(n for n in CLASSICAL if n not in SMOOTH_CLASSICAL)
ALL20 = HUMAN_LAWS + CLASSICAL + OTHER

GROUPS = {
    "all20": ALL20,
    "laws6": HUMAN_LAWS,
    "classical13": CLASSICAL,
    "smooth_classical3": SMOOTH_CLASSICAL,
    "rugged_classical10": RUGGED_CLASSICAL,
}
PRIMARY_GROUPS = ("all20", "laws6", "classical13")

RATIONALE = {
    "yerkes_dodson": "mean of 1-D Gaussian bumps centred at 0.5: separable, one interior peak",
    "stevens": "mean of x^0.67: monotone increasing, optimum at the upper corner (infinite slope at x=0)",
    "hicks_law": "minus mean of log2(x+1): monotone decreasing, optimum at the lower corner",
    "weber_fechner": "mean of log(1+x/0.01): monotone increasing, optimum at the upper corner (steep near 0)",
    "power_law_practice": "-(B(m) N(t)^-alpha(m)): monotone in the time axis, corner optimum",
    "steering_law": "-(a + bA/W + lambda A W): smooth, single interior supremum",
    "powell": "convex quartic (sum of squares and fourth powers of linear forms): one minimum",
    "rosenbrock": "curved valley, effectively unimodal (spec note)",
    "griewank": "near-quadratic at the +-600 box; cosine ripple negligible (spec note)",
    "moving_peaks": "five cones, piecewise-linear ridges: neither a law nor a classical function",
}

ERROR_MODELS = ("ar1", "bias", "drift", "gaussian")
EM_LABELS = ("pooled",) + ERROR_MODELS
GRID = (0.05, 0.25, 1.0, 5.0)
ONSETS = (0, 20)
POOL_MIN_STD = 0.25
BOOT = 2000
BOOT_SEED = 20260922
COMPARE_RULES = ("pm", "lcb1", "lcb2", "best_mean", "best_visited")
PER_LANDSCAPE_RULES = ("pm", "lcb1", "lcb2")

# GP diagnostic sample.
GP_ACQS = ("logei", "qnei")
GP_SEEDS = tuple(range(7, 17))
GP_NOISY_SUFFIX = "jittered_exact_gaussian_jit0_std1.0"
GP_CLEAN_SUFFIX = "baseline_exact"
N_SOBOL = 256


def group_of(name: str) -> str:
    return "law" if name in HUMAN_LAWS else "classical" if name in CLASSICAL else "other"


def write_definition(out_dir: Path, stats: dict) -> pd.DataFrame:
    missing = set(ALL20) ^ set(bb.DEFAULT_SUITE)
    if missing:
        raise ValueError(f"the twenty landscapes here differ from bb.DEFAULT_SUITE by {sorted(missing)}")
    rows = []
    for name in bb.DEFAULT_SUITE:
        spec = bb.BENCHMARKS[name]
        rows.append({
            "landscape": name, "group": group_of(name),
            "primary_smooth_subset": name in HUMAN_LAWS,
            "secondary_smooth_classical": name in SMOOTH_CLASSICAL,
            "dim": spec.dim, "optimum": spec.optimum, "opt_z": float(stats[name]["opt_z"]),
            "ruggedness": float(stats[name]["ruggedness"]), "spec_note": spec.notes,
            "rationale": RATIONALE.get(name, "classical global-optimisation test function"),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "smooth_subset_definition.csv", index=False)
    return frame


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def boot_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    draws = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = a[rng.integers(0, len(a), (BOOT, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (BOOT, len(b)))].mean(axis=1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_ratio_diff(na, da, nb, db, rng) -> tuple[float, float, float]:
    """Difference of two ratios of landscape means, landscapes resampled within each group."""
    na, da, nb, db = (np.asarray(x, float) for x in (na, da, nb, db))
    ia = rng.integers(0, len(na), (BOOT, len(na)))
    ib = rng.integers(0, len(nb), (BOOT, len(nb)))
    ra = na[ia].mean(1) / da[ia].mean(1)
    rb = nb[ib].mean(1) / db[ib].mean(1)
    d = ra - rb
    ok = np.isfinite(d)
    return float(na.mean() / da.mean() - nb.mean() / db.mean()), float(np.percentile(d[ok], 2.5)), \
        float(np.percentile(d[ok], 97.5))


def crossed_weights(n_a: int, n_s: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pigeonhole bootstrap: acquisitions and seeds resampled independently (Owen 2007)."""
    wa = np.zeros((BOOT, n_a))
    ws = np.zeros((BOOT, n_s))
    np.add.at(wa, (np.arange(BOOT)[:, None], rng.integers(0, n_a, (BOOT, n_a))), 1.0)
    np.add.at(ws, (np.arange(BOOT)[:, None], rng.integers(0, n_s, (BOOT, n_s))), 1.0)
    return wa, ws


def cube(frame: pd.DataFrame, value: str) -> tuple[np.ndarray, list, list, list]:
    """(landscape, acquisition, seed) array of a per-run value already averaged over other keys."""
    per = frame.groupby(["dataset", "acquisition", "seed"])[value].mean()
    lands = sorted(per.index.get_level_values(0).unique())
    acqs = sorted(per.index.get_level_values(1).unique())
    seeds = sorted(per.index.get_level_values(2).unique())
    full = pd.MultiIndex.from_product([lands, acqs, seeds], names=["dataset", "acquisition", "seed"])
    arr = per.reindex(full).to_numpy().reshape(len(lands), len(acqs), len(seeds))
    return arr, lands, acqs, seeds


def crossed_ratio_ci(num: np.ndarray, den: np.ndarray, rng) -> tuple[float, float]:
    """Interval for mean_l(num) / mean_l(den), acquisitions and seeds resampled, landscapes fixed."""
    if np.isnan(num).any() or np.isnan(den).any():
        return np.nan, np.nan
    _, n_a, n_s = num.shape
    wa, ws = crossed_weights(n_a, n_s, rng)
    nr = np.einsum("ra,las,rs->rl", wa, num, ws).mean(1)
    dr_ = np.einsum("ra,las,rs->rl", wa, den, ws).mean(1)
    ratio = np.where(dr_ > 0, nr / np.where(dr_ > 0, dr_, 1.0), np.nan)
    ok = np.isfinite(ratio)
    if ok.mean() < 0.95:
        return np.nan, np.nan
    return float(np.percentile(ratio[ok], 2.5)), float(np.percentile(ratio[ok], 97.5))


def crossed_mean_ci(values: np.ndarray, rng) -> tuple[float, float]:
    if np.isnan(values).any():
        return np.nan, np.nan
    _, n_a, n_s = values.shape
    wa, ws = crossed_weights(n_a, n_s, rng)
    m = np.einsum("ra,las,rs->rl", wa, values, ws).mean(1) / (n_a * n_s)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------------------
# 2. Headline by group
# ---------------------------------------------------------------------------


def load_cells(analysis: Path, opt_z: dict) -> pd.DataFrame:
    cells = pd.read_csv(analysis / "cell_means.csv")
    cells = cells[~cells["acquisition"].isin(MODEL_FREE)].copy()
    lacking = sorted(set(cells["dataset"]) - set(opt_z))
    if lacking:
        raise KeyError(f"no opt_z for {lacking}")
    if set(cells["dataset"]) != set(ALL20):
        raise ValueError(f"cell_means landscapes {sorted(set(cells['dataset']))} are not the twenty")
    cells["deployed"] = cells["inference_excess"] / cells["dataset"].map(opt_z)
    cells["search_traj"] = cells["fragility"]
    return cells


def load_decomposition(analysis: Path, opt_z: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(analysis / "ship_rules_per_run.csv")
    runs["baseline"] = runs["baseline"].astype(str).str.lower().isin(("true", "1"))
    runs = runs[~runs["acquisition"].isin(MODEL_FREE)]
    if not runs["best_observed_reproduces"].astype(str).str.lower().isin(("true", "1")).all():
        raise ValueError("best_observed does not reproduce the logged deployed regret in some run")
    if set(runs["logged_rule"].unique()) != {"best_observed"}:
        raise ValueError(f"output-boba logged rules {runs['logged_rule'].unique()}; expected best_observed only")
    table = dr.decompose(runs, opt_z)
    clean, noisy = table[table["baseline"]], table[~table["baseline"]].copy()
    dr.assert_clean_runs_have_no_selection_loss(clean)
    noisy["jitter_iteration"] = noisy["jitter_iteration"].astype(int)
    return noisy, clean


def headline(cells: pd.DataFrame, noisy: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for em in EM_LABELS:
        c = cells if em == "pooled" else cells[cells["error_model"] == em]
        n = noisy if em == "pooled" else noisy[noisy["error_model"] == em]
        per = c.groupby(["jitter_std", "jitter_iteration", "dataset"])[["deployed", "search_traj"]].mean()
        for (std, onset), block in per.groupby(level=[0, 1]):
            block = block.droplevel([0, 1])
            nb = n[(np.isclose(n["jitter_std"], std)) & (n["jitter_iteration"] == onset)]
            for gname, lands in GROUPS.items():
                g = block.loc[[x for x in lands if x in block.index]]
                ng = nb[nb["dataset"].isin(lands)]
                rng = np.random.default_rng(BOOT_SEED)
                d, dlo, dhi = boot_mean(g["deployed"].to_numpy(), rng)
                s, slo, shi = boot_mean(g["search_traj"].to_numpy(), rng)
                dec = dr.summarise(ng, clean, np.random.default_rng(dr.BOOTSTRAP_SEED))
                if abs(dec["mean_excess_deployed"] - d) > 5e-4:
                    raise ValueError(f"cell_means and the decomposition disagree on the deployed excess "
                                     f"({em}, {std}, {onset}, {gname}): {d:.5f} vs {dec['mean_excess_deployed']:.5f}")
                # Final-trial search loss: its own landscape bootstrap.
                pairs = ng.merge(clean.groupby(dr.PAIR_KEYS)[["search"]].mean()
                                 .rename(columns={"search": "search_clean"}), on=dr.PAIR_KEYS)
                perx = (pairs["search"] - pairs["search_clean"]).groupby(pairs["dataset"]).mean()
                sf, sflo, sfhi = boot_mean(perx.to_numpy(), rng)
                rows.append({
                    "error_model": em, "jitter_std": float(std), "jitter_iteration": int(onset), "group": gname,
                    "n_landscapes": len(g),
                    "deployed": d, "deployed_lo": dlo, "deployed_hi": dhi,
                    "search_traj": s, "search_traj_lo": slo, "search_traj_hi": shi,
                    "deployed_over_search_traj": d / s if s > 0 else np.nan,
                    "search_final": sf, "search_final_lo": sflo, "search_final_hi": sfhi,
                    "selection_excess": dec["mean_excess_selection"],
                    "selection_share": dec["excess_selection_share"],
                    "selection_share_lo": dec["excess_selection_share_lo"],
                    "selection_share_hi": dec["excess_selection_share_hi"],
                    "deployed_decomposition_check": dec["mean_excess_deployed"],
                })
    return pd.DataFrame(rows)


def headline_differences(cells: pd.DataFrame, noisy: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """laws6 minus classical13, landscapes resampled within each group."""
    rows = []
    base = clean.groupby(dr.PAIR_KEYS)[["deployed", "search"]].mean().rename(
        columns={"deployed": "deployed_clean", "search": "search_clean"})
    for em in ("pooled", "gaussian"):
        c = cells if em == "pooled" else cells[cells["error_model"] == em]
        n = noisy if em == "pooled" else noisy[noisy["error_model"] == em]
        per = c.groupby(["jitter_std", "jitter_iteration", "dataset"])[["deployed", "search_traj"]].mean()
        m = n.merge(base, on=dr.PAIR_KEYS)
        m = m.assign(ex_dep=m["deployed"] - m["deployed_clean"], ex_sel=m["selection"])
        perx = m.groupby(["jitter_std", "jitter_iteration", "dataset"])[["ex_dep", "ex_sel"]].mean()
        for (std, onset), block in per.groupby(level=[0, 1]):
            block = block.droplevel([0, 1])
            px = perx.loc[(std, onset)]
            a, b = block.loc[list(HUMAN_LAWS)], block.loc[list(CLASSICAL)]
            rng = np.random.default_rng(BOOT_SEED)
            row = {"error_model": em, "jitter_std": float(std), "jitter_iteration": int(onset),
                   "contrast": "laws6_minus_classical13"}
            for col in ("deployed", "search_traj"):
                v, lo, hi = boot_diff(a[col].to_numpy(), b[col].to_numpy(), rng)
                row.update({col: v, f"{col}_lo": lo, f"{col}_hi": hi})
            pa, pb = px.loc[list(HUMAN_LAWS)], px.loc[list(CLASSICAL)]
            v, lo, hi = boot_ratio_diff(pa["ex_sel"], pa["ex_dep"], pb["ex_sel"], pb["ex_dep"], rng)
            row.update({"selection_share": v, "selection_share_lo": lo, "selection_share_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Per-landscape distribution
# ---------------------------------------------------------------------------


def per_run_costs(input_dir: Path, noisy: pd.DataFrame, clean: pd.DataFrame, opt_z: dict) -> pd.DataFrame:
    files = sorted(input_dir.glob("*/evaluation/paired_excess_metrics.csv"))
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration",
            "auc_simple_regret_excess_true_postonset_per_iter", "final_inference_simple_regret_excess_true"]
    pm = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    pm = pm[~pm["acquisition"].isin(MODEL_FREE) & pm["dataset"].isin(ALL20)].copy()
    pm["jitter_iteration"] = pm["jitter_iteration"].astype(int)
    z = pm["dataset"].map(opt_z)
    pm["search_traj"] = pm["auc_simple_regret_excess_true_postonset_per_iter"] / z
    pm["deployed_eval"] = pm["final_inference_simple_regret_excess_true"] / z
    base = clean.groupby(dr.PAIR_KEYS)[["deployed", "search"]].mean().rename(
        columns={"deployed": "deployed_clean", "search": "search_clean"})
    m = noisy.merge(base, on=dr.PAIR_KEYS, how="left", validate="many_to_one")
    m = m.assign(deployed_ex=m["deployed"] - m["deployed_clean"], search_final=m["search"] - m["search_clean"])
    keys = aba.PAIR_KEYS
    out = pm.merge(m[keys + ["deployed_ex", "search_final", "selection"]], on=keys, how="inner",
                   validate="one_to_one")
    if len(out) != len(pm) or len(out) != len(m):
        raise ValueError(f"per-run tables do not line up: {len(pm)} paired metrics, {len(m)} rescored, "
                         f"{len(out)} matched")
    worst = float((out["deployed_eval"] - out["deployed_ex"]).abs().max())
    if worst > 1e-8:
        raise ValueError(f"per-run deployed excess differs between the pipelines by up to {worst:.2e}")
    return out.rename(columns={"deployed_ex": "deployed"})


def per_landscape(runs: pd.DataFrame, stats: dict) -> pd.DataFrame:
    rows = []
    for em in EM_LABELS:
        r = runs if em == "pooled" else runs[runs["error_model"] == em]
        for (std, onset), block in r.groupby(["jitter_std", "jitter_iteration"]):
            for land, lb in block.groupby("dataset"):
                rng = np.random.default_rng(BOOT_SEED + zlib.crc32(land.encode()) % 1000)
                row = {"landscape": land, "group": group_of(land), "error_model": em, "jitter_std": float(std),
                       "jitter_iteration": int(onset),
                       "headline_cell": bool(em == "pooled" and np.isclose(std, 1.0) and onset == 0),
                       "opt_z": float(stats[land]["opt_z"]), "n_runs": len(lb),
                       "n_acquisitions": lb["acquisition"].nunique(), "n_seeds": lb["seed"].nunique()}
                cubes = {}
                for col in ("deployed", "search_traj", "search_final", "selection"):
                    arr, *_ = cube(lb, col)
                    cubes[col] = arr
                    lo, hi = crossed_mean_ci(arr, rng)
                    row.update({col: float(np.nanmean(arr)), f"{col}_lo": lo, f"{col}_hi": hi})
                row["selection_share"] = row["selection"] / row["deployed"] if row["deployed"] > 0 else np.nan
                lo, hi = crossed_ratio_ci(cubes["selection"], cubes["deployed"], rng)
                row.update({"selection_share_lo": lo, "selection_share_hi": hi})
                row["deployed_over_search_traj"] = (row["deployed"] / row["search_traj"]
                                                    if row["search_traj"] > 0 else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Ship rules on the subset
# ---------------------------------------------------------------------------


def ship_cells(frame: pd.DataFrame):
    """(cell label, std, onset, block): condition cells, both onsets per std, pooled >= 0.25 SD."""
    for (std, onset), block in frame.groupby(["jitter_std", "jitter_iteration"]):
        yield "condition", float(std), float(onset), block
    for std, block in frame.groupby("jitter_std"):
        yield "both_onsets", float(std), np.nan, block
    big = frame[frame["jitter_std"] >= POOL_MIN_STD - 1e-12]
    for onset, block in big.groupby("jitter_iteration"):
        yield f"pooled_std_ge_{POOL_MIN_STD:g}", np.nan, float(onset), block
    yield f"pooled_std_ge_{POOL_MIN_STD:g}", np.nan, np.nan, big


def ship_rules_subset(analysis: Path, opt_z: dict, noisy_dec: pd.DataFrame) -> pd.DataFrame:
    runs = asr.load_per_run(analysis / f"{asr.OUTPUT_NAME}.csv")
    ref = asr.twin_frame(runs, asr.STANDARD)
    raw = runs[~runs["baseline"]].copy()
    raw["jitter_iteration"] = raw["jitter_iteration"].astype(int)
    rows = []
    for rule in COMPARE_RULES:
        trt = asr.twin_frame(runs, rule)
        parts = []
        for (model, variant), ref_g in ref.groupby(["error_model", "variant"]):
            trt_g = trt[(trt["error_model"] == model) & (trt["variant"] == variant)]
            parts.append(aba.paired_frame(ref_g, trt_g, asr.RESPONSE, opt_z, pool=False))
        paired = pd.concat(parts, ignore_index=True)
        paired["cost_run"] = paired["ref_noisy"] - paired["ref_clean"]
        paired["gain_run"] = paired["ref_noisy"] - paired["trt_noisy"]
        targets = [(g, GROUPS[g]) for g in GROUPS]
        if rule in PER_LANDSCAPE_RULES:
            targets += [(f"landscape:{x}", (x,)) for x in ALL20]
        for em in EM_LABELS:
            if em not in ("pooled", "gaussian") and rule not in PER_LANDSCAPE_RULES:
                continue
            P = paired if em == "pooled" else paired[paired["error_model"] == em]
            R = raw if em == "pooled" else raw[raw["error_model"] == em]
            for gname, lands in targets:
                if gname.startswith("landscape:") and em not in ("pooled", "gaussian"):
                    continue
                G = P[P["dataset"].isin(lands)]
                RG = R[R["dataset"].isin(lands)]
                cond_rows = []
                for cell, std, onset, block in ship_cells(G):
                    s = aba.summarise(block, np.random.default_rng(aba.BOOTSTRAP_SEED))
                    if s["n_landscapes"] < 3:
                        s["recovered_lo"] = s["recovered_hi"] = np.nan
                        s["wilcoxon_p"] = np.nan
                    num, *_ = cube(block, "gain_run")
                    den, *_ = cube(block, "cost_run")
                    clo, chi = crossed_ratio_ci(num, den, np.random.default_rng(BOOT_SEED))
                    per_l = block.groupby("dataset")[["gain_run"]].mean()["gain_run"]
                    sel = RG
                    if np.isfinite(std):
                        sel = sel[np.isclose(sel["jitter_std"], std)]
                    else:
                        sel = sel[sel["jitter_std"] >= POOL_MIN_STD - 1e-12]
                    if np.isfinite(onset):
                        sel = sel[sel["jitter_iteration"] == int(onset)]
                    diff = sel[f"regret_{rule}"] - sel["regret_best_observed"]
                    row = {"estimand": "share_of_deployed_cost", "rule": rule, "reference": asr.STANDARD,
                           "error_model": em, "group": gname, "cell": cell, "jitter_std": std,
                           "jitter_iteration": onset, **s,
                           "recovered_cond_lo": clo, "recovered_cond_hi": chi,
                           "n_landscapes_rule_better": int((per_l > 0).sum()),
                           "n_landscapes_rule_worse": int((per_l < 0).sum()),
                           "n_runs": int(len(sel)),
                           "share_runs_same_design": float((sel[f"idx_{rule}"] == sel["idx_best_observed"]).mean()),
                           "share_runs_rule_better": float((diff < -1e-12).mean()),
                           "share_runs_rule_worse": float((diff > 1e-12).mean())}
                    if cell == "condition":
                        cond_rows.append(row)
                    rows.append(row)
                ps = [r["wilcoxon_p"] for r in cond_rows]
                if cond_rows and all(np.isfinite(ps)):
                    for r, q in zip(cond_rows, multipletests(ps, method="fdr_bh")[1]):
                        r["wilcoxon_p_fdr"] = float(q)
    # The decomposition's estimand: share of the SELECTION loss a rule removes.
    for em in ("pooled", "gaussian"):
        n = noisy_dec if em == "pooled" else noisy_dec[noisy_dec["error_model"] == em]
        for gname, lands in GROUPS.items():
            G = n[n["dataset"].isin(lands)]
            # The last cell pools every magnitude, as the paper's -8.7% (regret_decomposition.csv, all pooled) does.
            for cell, std, onset, block in list(ship_cells(G)) + [("pooled_all_magnitudes", np.nan, np.nan, G)]:
                rec = dr.rule_recovery(block, np.random.default_rng(dr.BOOTSTRAP_SEED))
                for rule in ("pm", "lcb1", "lcb2", "best_mean"):
                    rows.append({"estimand": "share_of_selection_loss", "rule": rule, "reference": asr.STANDARD,
                                 "error_model": em, "group": gname, "cell": cell, "jitter_std": std,
                                 "jitter_iteration": onset, "n_landscapes": block["dataset"].nunique(),
                                 "recovered": rec[f"{rule}_selection_recovered"],
                                 "recovered_lo": rec[f"{rule}_selection_recovered_lo"],
                                 "recovered_hi": rec[f"{rule}_selection_recovered_hi"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. GP diagnostics
# ---------------------------------------------------------------------------


def gp_tasks(input_dir: Path) -> list[dict]:
    tasks = []
    for land in ALL20:
        for acq in GP_ACQS:
            for seed in GP_SEEDS:
                for cond, suffix in (("clean", GP_CLEAN_SUFFIX), ("gaussian_1sd_onset0", GP_NOISY_SUFFIX)):
                    name = f"bo_sensor_error_{land}_value_{acq}_seed{seed}_{suffix}.csv"
                    path = input_dir / land / name
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    tasks.append({"landscape": land, "acquisition": acq, "seed": seed, "condition": cond,
                                  "injected_sd": 0.0 if cond == "clean" else 1.0,
                                  "path": str(path), "file": f"{land}/{name}"})
    return tasks


def _kernel_parts(gp):
    k = gp.covar_module
    outputscale = np.nan
    if hasattr(k, "base_kernel"):
        outputscale = float(k.outputscale.detach().reshape(-1)[0])
        k = k.base_kernel
    return k, outputscale


def gp_diagnose(task: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    land = task["landscape"]
    spec = bb.BENCHMARKS[land]
    df = pd.read_csv(task["path"], float_precision="round_trip")
    cols = str(df["param_columns"].iloc[0]).split(",")
    if cols != spec.param_columns:
        raise ValueError(f"{task['file']}: param columns {cols}")
    if "observation_noise" in df.columns and str(df["observation_noise"].iloc[0]) == "known":
        raise ValueError(f"{task['file']}: known-noise run; the loop GP here learns its noise")
    X = df[cols].to_numpy(float)
    obs = df["objective_observed"].to_numpy(float)
    true = df["objective_true"].to_numpy(float)
    y_opt = float(df["y_opt"].iloc[0])
    if not np.all(np.isfinite(obs)):
        raise ValueError(f"{task['file']}: non-finite ratings")
    n, d = X.shape
    lo, hi = spec.bounds_low, spec.bounds_high

    gp, diag = fit_loop_gp(X, obs, lo, hi, yvar=None, seed=file_seed(task["file"]))
    mu, sd = latent_posterior(gp, X)

    # --- closed-form LOO with the fitted hyperparameters, in standardised space.
    kern, outputscale = _kernel_parts(gp)
    Xn = torch.tensor((X - lo) / (hi - lo), dtype=torch.double)
    with torch.no_grad():
        Kf = gp.covar_module(Xn).to_dense().numpy()
        noise_s = float(gp.likelihood.noise.detach().reshape(-1)[0])
        c = float(gp.mean_module.constant.detach().reshape(-1)[0])
        ys = gp.train_targets.detach().numpy().reshape(-1)
        means = float(gp.outcome_transform.means.detach().reshape(-1)[0])
        stdv = float(gp.outcome_transform.stdvs.detach().reshape(-1)[0])
        ls = kern.lengthscale.detach().numpy().reshape(-1)
    if np.max(np.abs(ys - (obs - means) / stdv)) > 1e-8:
        raise ValueError(f"{task['file']}: train targets are not the standardised ratings")
    Ky = Kf + noise_s * np.eye(n)
    cf = sla.cho_factor(Ky, lower=True)
    Kinv = sla.cho_solve(cf, np.eye(n))
    alpha = sla.cho_solve(cf, ys - c)
    # Check the hand-built posterior against the model's own before trusting it
    # (relative to the target scale; asserted in main, recorded here so one
    # ill-conditioned clean fit cannot kill the pool).
    mu_check = (c + Kf @ alpha) * stdv + means
    var_check = np.diag(Kf - Kf @ sla.cho_solve(cf, Kf)) * stdv ** 2
    err_mu = float(np.max(np.abs(mu_check - mu))) / stdv
    err_var = float(np.max(np.abs(var_check - sd ** 2))) / stdv ** 2
    kii = np.diag(Kinv)
    loo_mu = (ys - alpha / kii) * stdv + means
    loo_var = (1.0 / kii) * stdv ** 2
    noise_var_obj = noise_s * stdv ** 2
    loo_lat_var = np.clip(loo_var - noise_var_obj, 1e-12, None)
    lpd = sps.norm.logpdf(obs, loo_mu, np.sqrt(loo_var))
    lpd_true_lat = sps.norm.logpdf(true, loo_mu, np.sqrt(loo_lat_var))
    z = (obs - loo_mu) / np.sqrt(loo_var)
    # Trivial LOO predictor: mean and SD of the other 49 ratings.
    tot, tot2 = obs.sum(), (obs ** 2).sum()
    m_i = (tot - obs) / (n - 1)
    v_i = ((tot2 - obs ** 2) - (n - 1) * m_i ** 2) / (n - 2)
    lpd_triv = sps.norm.logpdf(obs, m_i, np.sqrt(v_i))
    lpd_oracle = (float(np.mean(sps.norm.logpdf(obs, true, task["injected_sd"])))
                  if task["injected_sd"] > 0 else np.nan)

    # --- fresh Sobol points, truth from the oracle the simulator used.
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    oracle = bb.SyntheticOracle.from_stats(land, stats)
    true_check = float(np.max(np.abs(oracle.predict_many(X).reshape(-1) - true)))
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=zlib.crc32(land.encode()) & 0x7FFFFFFF)
    U = eng.draw(N_SOBOL, dtype=torch.double).numpy()
    Xs = lo + U * (hi - lo)
    fs = oracle.predict_many(Xs).reshape(-1)
    mus, sds = latent_posterior(gp, Xs)

    # --- ship picks among the visited designs, and ranks of the truly best one.
    i_pm, i_bo, i_bv = int(np.argmax(mu)), int(np.argmax(obs)), int(np.argmax(true))
    rank_mu = int((mu > mu[i_bv]).sum()) + 1
    rank_obs = int((obs > obs[i_bv]).sum()) + 1
    opt_z = float(stats[land]["opt_z"])
    rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))  # noqa: E731
    var_true_v = float(np.var(true))
    var_fs = float(np.var(fs))
    return {
        "landscape": land, "group": group_of(land), "acquisition": task["acquisition"], "seed": task["seed"],
        "condition": task["condition"], "file": task["file"], "dim": d, "n_train": n, "opt_z": opt_z,
        "injected_sd": task["injected_sd"],
        "realized_noise_sd": float(np.sqrt(np.mean((obs - true) ** 2))),
        "gp_fit_ok": diag["gp_fit_ok"], "gp_n_warnings": diag["gp_n_warnings"], "gp_kernel": diag["gp_kernel"],
        "noise_sd_fit": float(np.sqrt(noise_var_obj)),
        "noise_sd_ratio": float(np.sqrt(noise_var_obj)) / task["injected_sd"] if task["injected_sd"] > 0 else np.nan,
        "noise_var_standardised": noise_s, "target_stdv": stdv, "outputscale": outputscale,
        "ls_geomean": float(np.exp(np.mean(np.log(ls)))), "ls_min": float(ls.min()), "ls_max": float(ls.max()),
        "ls_median": float(np.median(ls)), "n_ls_above_1": int((ls > 1.0).sum()),
        "loo_lpd": float(lpd.mean()), "loo_lpd_trivial": float(lpd_triv.mean()),
        "loo_lpd_gain": float(lpd.mean() - lpd_triv.mean()), "loo_lpd_oracle": lpd_oracle,
        "loo_lpd_frac_achievable": (float((lpd.mean() - lpd_triv.mean()) / (lpd_oracle - lpd_triv.mean()))
                                    if np.isfinite(lpd_oracle) else np.nan),
        "loo_lpd_true_latent": float(lpd_true_lat.mean()),
        "loo_mean_z2": float(np.mean(z ** 2)), "loo_cov95": float(np.mean(np.abs(z) < 1.96)),
        "loo_rmse_rating": rmse(loo_mu, obs), "loo_rmse_true": rmse(loo_mu, true),
        "visited_rmse_pm": rmse(mu, true), "visited_rmse_obs": rmse(obs, true),
        "visited_shrink_ratio": rmse(mu, true) / rmse(obs, true) if task["injected_sd"] > 0 else np.nan,
        "visited_r2_pm": 1.0 - np.mean((mu - true) ** 2) / var_true_v if var_true_v > 0 else np.nan,
        "visited_spearman_pm": float(sps.spearmanr(mu, true)[0]),
        "visited_spearman_obs": float(sps.spearmanr(obs, true)[0]),
        "visited_true_sd": float(np.sqrt(var_true_v)),
        "rank_best_visited_under_pm": rank_mu, "rank_best_visited_under_obs": rank_obs,
        "pm_bias_at_best_visited": float(mu[i_bv] - true[i_bv]), "pm_mean_bias": float(np.mean(mu - true)),
        "sobol_rmse": rmse(mus, fs), "sobol_r2": 1.0 - np.mean((mus - fs) ** 2) / var_fs,
        "sobol_spearman": float(sps.spearmanr(mus, fs)[0]),
        "sobol_cov95": float(np.mean(np.abs(fs - mus) < 1.96 * np.maximum(sds, 1e-12))),
        "sobol_mean_post_sd": float(np.mean(sds)), "sobol_true_sd": float(np.sqrt(var_fs)),
        "regret_pm": y_opt - float(true[i_pm]), "regret_best_observed": y_opt - float(true[i_bo]),
        "regret_best_visited": y_opt - float(true[i_bv]),
        # Positive = pm ships the truly better design (regret_best_observed - regret_pm).
        "pm_gain_over_opt_z": (float(true[i_pm]) - float(true[i_bo])) / opt_z,
        "idx_pm": i_pm, "idx_best_observed": i_bo, "idx_best_visited": i_bv,
        "log_final_simple_regret_true": float(df["simple_regret_true"].iloc[-1]),
        "log_final_inference_regret_true": float(df["inference_simple_regret_true"].iloc[-1]),
        "check_posterior_mean_err": err_mu, "check_posterior_var_err": err_var, "check_oracle_err": true_check,
    }


def run_gp(tasks: list[dict], workers: int) -> pd.DataFrame:
    t0 = time.time()
    out = []
    if workers <= 1:
        for i, t in enumerate(tasks):
            out.append(gp_diagnose(t))
    else:
        from multiprocessing import get_context

        from rescore_ship_rules import _init_worker

        with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            for i, rec in enumerate(pool.imap_unordered(gp_diagnose, tasks, chunksize=4)):
                out.append(rec)
                if (i + 1) % 100 == 0 or i + 1 == len(tasks):
                    print(f"  GP {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(out).sort_values(["landscape", "condition", "acquisition", "seed"]).reset_index(drop=True)


def check_gp_against_table(gp_runs: pd.DataFrame, analysis: Path) -> dict:
    table = pd.read_csv(analysis / "ship_rules_per_run.csv")
    table["key"] = table["file"]
    m = gp_runs.merge(table[["key", "regret_best_observed", "regret_pm", "regret_best_visited", "idx_pm",
                             "gp_noise_sd"]].rename(columns=lambda c: c if c == "key" else f"tab_{c}"),
                      left_on="file", right_on="key", how="left", validate="one_to_one")
    if m["tab_regret_pm"].isna().any():
        raise ValueError("some GP-sample runs are missing from ship_rules_per_run.csv")
    res = {
        "n": len(m),
        "bo_regret_matches": int((m["regret_best_observed"] - m["tab_regret_best_observed"]).abs().le(1e-9).sum()),
        "bo_matches_log": int((m["regret_best_observed"] - m["log_final_inference_regret_true"]).abs().le(1e-9).sum()),
        "bv_regret_matches": int((m["regret_best_visited"] - m["tab_regret_best_visited"]).abs().le(1e-9).sum()),
        "bv_matches_log_simple_regret": int((m["regret_best_visited"] - m["log_final_simple_regret_true"])
                                            .abs().le(1e-9).sum()),
        "pm_idx_matches": int((m["idx_pm"] == m["tab_idx_pm"]).sum()),
        "pm_regret_matches": int((m["regret_pm"] - m["tab_regret_pm"]).abs().le(1e-9).sum()),
        "noise_sd_max_abs_diff": float((m["noise_sd_fit"] - m["tab_gp_noise_sd"]).abs().max()),
        "max_oracle_err": float(m["check_oracle_err"].max()),
        "max_posterior_mean_err": float(m["check_posterior_mean_err"].max()),
        "max_posterior_var_err": float(m["check_posterior_var_err"].max()),
        "fit_failures": int((~m["gp_fit_ok"].astype(bool)).sum()),
    }
    if res["max_posterior_mean_err"] > 1e-5 or res["max_posterior_var_err"] > 1e-5:
        raise ValueError(f"hand-built GP posterior differs from gp.posterior (mean {res['max_posterior_mean_err']:.2e}, "
                         f"var {res['max_posterior_var_err']:.2e} of the target scale); the LOO would be wrong")
    for k in ("bo_regret_matches", "bo_matches_log", "bv_regret_matches", "bv_matches_log_simple_regret"):
        if res[k] != res["n"]:
            raise ValueError(f"GP sample check {k}: {res[k]} of {res['n']}")
    if res["max_oracle_err"] > 1e-8:
        raise ValueError(f"oracle does not reproduce the logged objective_true (max err {res['max_oracle_err']:.2e})")
    return res


GP_METRICS = ["realized_noise_sd", "noise_sd_fit", "noise_sd_ratio", "ls_geomean", "ls_min", "ls_max",
              "n_ls_above_1", "loo_lpd", "loo_lpd_trivial", "loo_lpd_gain", "loo_lpd_oracle",
              "loo_lpd_frac_achievable", "loo_lpd_true_latent", "loo_mean_z2", "loo_cov95", "loo_rmse_rating",
              "loo_rmse_true", "visited_rmse_pm", "visited_rmse_obs", "visited_shrink_ratio", "visited_r2_pm",
              "visited_spearman_pm", "visited_spearman_obs", "rank_best_visited_under_pm",
              "rank_best_visited_under_obs", "pm_bias_at_best_visited", "sobol_rmse", "sobol_r2",
              "sobol_spearman", "sobol_cov95", "sobol_mean_post_sd", "pm_gain_over_opt_z", "gp_n_warnings"]
GP_KEY = ["noise_sd_ratio", "ls_geomean", "loo_lpd_gain", "loo_lpd_frac_achievable", "loo_mean_z2",
          "visited_rmse_pm", "visited_shrink_ratio", "visited_r2_pm", "visited_spearman_pm", "visited_spearman_obs",
          "sobol_rmse", "sobol_r2", "sobol_cov95", "pm_gain_over_opt_z"]


def gp_summary(gp_runs: pd.DataFrame) -> pd.DataFrame:
    per = gp_runs.groupby(["landscape", "condition"])[GP_METRICS].mean()
    counts = gp_runs.groupby(["landscape", "condition"]).size()
    rows = []
    for (land, cond), r in per.iterrows():
        rows.append({"level": "landscape", "name": land, "group": group_of(land), "condition": cond,
                     "n_landscapes": 1, "n_runs": int(counts[(land, cond)]),
                     "dim": bb.BENCHMARKS[land].dim, "optimum": bb.BENCHMARKS[land].optimum, **r.to_dict()})
    for gname, lands in GROUPS.items():
        for cond in sorted(gp_runs["condition"].unique()):
            block = per.xs(cond, level="condition").loc[list(lands)]
            row = {"level": "group", "name": gname, "group": gname, "condition": cond,
                   "n_landscapes": len(lands), "n_runs": int(counts.xs(cond, level="condition").loc[list(lands)].sum())}
            rng = np.random.default_rng(BOOT_SEED)
            for mcol in GP_METRICS:
                vals = block[mcol].to_numpy(float)
                if np.all(np.isnan(vals)):
                    row[mcol] = np.nan
                    continue
                v, lo, hi = boot_mean(vals[np.isfinite(vals)], rng)
                row[mcol] = v
                if mcol in GP_KEY:
                    row[f"{mcol}_lo"], row[f"{mcol}_hi"] = lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def gp_fit_vs_pm(gp_runs: pd.DataFrame, ship: pd.DataFrame) -> pd.DataFrame:
    """Does GP fit quality explain whether pm beats best_observed?"""
    noisy = gp_runs[gp_runs["condition"] == "gaussian_1sd_onset0"]
    per = noisy.groupby("landscape")[GP_METRICS].mean()
    # pm's result on the full table (ten acquisitions, seeds 7-16) at the same condition.
    full = ship[(ship["estimand"] == "share_of_deployed_cost") & (ship["rule"] == "pm")
                & (ship["error_model"] == "gaussian") & (ship["cell"] == "condition")
                & (ship["jitter_std"] == 1.0) & (ship["jitter_iteration"] == 0.0)
                & ship["group"].str.startswith("landscape:")].copy()
    full["landscape"] = full["group"].str.slice(len("landscape:"))
    full = full.set_index("landscape")
    per["pm_gain_full"] = full["gain"]
    per["pm_recovered_full"] = full["recovered"]
    rows = []
    predictors = ["sobol_r2", "sobol_rmse", "visited_r2_pm", "visited_rmse_pm", "visited_shrink_ratio",
                  "visited_spearman_pm", "loo_lpd_gain", "loo_lpd_frac_achievable", "loo_mean_z2", "ls_geomean",
                  "noise_sd_ratio"]
    for outcome in ("pm_gain_full", "pm_gain_over_opt_z"):
        for p in predictors:
            for scope, lands in (("all20", ALL20), ("classical13", CLASSICAL)):
                sub = per.loc[list(lands), [p, outcome]].dropna()
                rho, pv = sps.spearmanr(sub[p], sub[outcome])
                rows.append({"analysis": "across_landscapes", "scope": scope, "outcome": outcome, "predictor": p,
                             "n": len(sub), "spearman": float(rho), "p": float(pv)})
    # Within landscapes: rank both within each landscape, then correlate the ranks.
    for p in predictors:
        for scope, lands in (("all20", ALL20), ("laws6", HUMAN_LAWS), ("classical13", CLASSICAL)):
            sub = noisy[noisy["landscape"].isin(lands)][["landscape", p, "pm_gain_over_opt_z"]].dropna()
            rp = sub.groupby("landscape")[p].rank()
            ro = sub.groupby("landscape")["pm_gain_over_opt_z"].rank()
            rho, pv = sps.pearsonr(rp - rp.groupby(sub["landscape"]).transform("mean"),
                                   ro - ro.groupby(sub["landscape"]).transform("mean"))
            rows.append({"analysis": "within_landscapes", "scope": scope, "outcome": "pm_gain_over_opt_z",
                         "predictor": p, "n": len(sub), "spearman": float(rho), "p": float(pv)})
    return pd.DataFrame(rows), per


def pm_by_fit_stratum(gp_runs: pd.DataFrame) -> pd.DataFrame:
    """pm minus best_observed in the noisy sample, split by fit quality (tertiles over all runs)."""
    noisy = gp_runs[gp_runs["condition"] == "gaussian_1sd_onset0"].copy()
    rows = []
    for p in ("sobol_r2", "visited_r2_pm", "loo_lpd_frac_achievable"):
        noisy["stratum"] = pd.qcut(noisy[p], 3, labels=["low", "mid", "high"])
        for (grp, st), b in noisy.groupby(["group", "stratum"], observed=True):
            rows.append({"predictor": p, "group": grp, "stratum": str(st), "n_runs": len(b),
                         "n_landscapes": b["landscape"].nunique(),
                         "predictor_mean": float(b[p].mean()),
                         "pm_gain_over_opt_z": float(b["pm_gain_over_opt_z"].mean()),
                         "share_pm_better": float((b["pm_gain_over_opt_z"] > 1e-12).mean()),
                         "share_pm_worse": float((b["pm_gain_over_opt_z"] < -1e-12).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(v, lo=None, hi=None, digits=1) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    s = f"{v * 100:+.{digits}f}%"
    if lo is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{lo * 100:+.0f}, {hi * 100:+.0f}]"
    return s


def _num(v, lo=None, hi=None, fmt="{:.3f}") -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    s = fmt.format(v)
    if lo is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{fmt.format(lo)}, {fmt.format(hi)}]"
    return s


def write_markdown(path: Path, definition: pd.DataFrame, head: pd.DataFrame, diffs: pd.DataFrame,
                   perl: pd.DataFrame, ship: pd.DataFrame, gps: pd.DataFrame | None, fitpm: pd.DataFrame | None,
                   strata: pd.DataFrame | None, checks: dict | None, defined_at: str) -> None:
    L = []
    L.append("# Smooth subset, per-landscape headline, ship rules and GP fit at n = 50\n")
    L.append(f"Generated by `scripts/smooth_subset_and_gp_diagnostics.py` on "
             f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}. Every number below is computed by that script; "
             "nothing is typed in by hand.\n")

    L.append("## 1. The a-priori smooth, low-modal subset\n")
    L.append(f"Written to `smooth_subset_definition.csv` at {defined_at}, before any result was computed, "
             "from the landscape definitions in `scripts/boba_benchmarks.py` alone.\n")
    L.append("**Primary subset: the six human-performance laws.**\n")
    L.append("| landscape | d | optimum | opt_z | why smooth and low-modal |")
    L.append("|---|---|---|---|---|")
    for _, r in definition[definition["primary_smooth_subset"]].iterrows():
        L.append(f"| {r['landscape']} | {r['dim']} | {r['optimum']} | {r['opt_z']:.2f} | {r['rationale']} |")
    L.append("")
    L.append("Comparison: the thirteen classical functions (" + ", ".join(CLASSICAL) + "). moving_peaks is in "
             "neither group and enters only 'all 20'. Secondary, also a priori and reported only as a sensitivity "
             "check: the classical functions whose definitions make them convex or effectively unimodal on BOBA's "
             "box (" + ", ".join(SMOOTH_CLASSICAL) + ") against the other ten.\n")
    L.append("Caveats that come with the definition, not from any result: all six laws are 4-D (the classical "
             "functions span d = 2 to 11); four of the six have a corner optimum (stevens, hicks_law, "
             "weber_fechner, power_law_practice), which no classical function except eggholder (one face) has; "
             "stevens and weber_fechner have unbounded or very steep slopes at the lower edge. So the laws differ "
             "from the classical functions in more than modality, and the secondary split exists to separate "
             "smoothness from those other differences.\n")

    L.append("## 2. Headline by group\n")
    L.append("Means of landscape means, 95% landscape bootstrap (2,000 draws). Deployed = excess regret of the "
             "shipped design at the final trial / opt_z. Search (trajectory) = the paper's search-loss row, "
             "post-onset per-iteration excess / opt_z. Search (final) = excess of the best VISITED design at the "
             "final trial / opt_z. Selection share = share of the deployed excess that is selection loss (ratio of "
             "landscape means). Pooled over the four error processes, model-free floors excluded. The deployed "
             "column is computed from cell_means.csv and from the decomposition independently; they agree to 5e-4 "
             "in every cell.\n")
    for std, onset in ((1.0, 0), (1.0, 20), (5.0, 0), (5.0, 20), (0.25, 0)):
        L.append(f"**{std:g} SD, error from trial {onset + 1}**\n")
        L.append("| group | n | deployed | search (trajectory) | search (final) | selection share |")
        L.append("|---|---|---|---|---|---|")
        blk = head[(head["error_model"] == "pooled") & np.isclose(head["jitter_std"], std)
                   & (head["jitter_iteration"] == onset)]
        for g in GROUPS:
            r = blk[blk["group"] == g].iloc[0]
            L.append(f"| {g} | {r['n_landscapes']} | {_pct(r['deployed'], r['deployed_lo'], r['deployed_hi'])} | "
                     f"{_pct(r['search_traj'], r['search_traj_lo'], r['search_traj_hi'])} | "
                     f"{_pct(r['search_final'], r['search_final_lo'], r['search_final_hi'])} | "
                     f"{_pct(r['selection_share'], r['selection_share_lo'], r['selection_share_hi'], 0)} |")
        d = diffs[(diffs["error_model"] == "pooled") & np.isclose(diffs["jitter_std"], std)
                  & (diffs["jitter_iteration"] == onset)].iloc[0]
        L.append(f"| laws6 - classical13 | | {_pct(d['deployed'], d['deployed_lo'], d['deployed_hi'])} | "
                 f"{_pct(d['search_traj'], d['search_traj_lo'], d['search_traj_hi'])} | | "
                 f"{_pct(d['selection_share'], d['selection_share_lo'], d['selection_share_hi'], 0)} |")
        L.append("")
    L.append("All magnitudes, onsets and each error process separately are in `smooth_subset.csv`; the "
             "laws-minus-classical contrasts (pooled and gaussian) in `smooth_subset_differences.csv`.\n")

    L.append("## 3. Per-landscape distribution at 1 SD from the first rating\n")
    L.append("Pooled over the four error processes; 95% crossed bootstrap over the ten acquisitions and ten seeds "
             "within each landscape. Sorted by deployed cost. `per_landscape_headline.csv` has every cell, "
             "magnitude and error process in the same tidy form.\n")
    h = perl[perl["headline_cell"]].sort_values("deployed")
    L.append("| landscape | group | opt_z | deployed | search (trajectory) | search (final) | selection share |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in h.iterrows():
        L.append(f"| {r['landscape']} | {r['group']} | {r['opt_z']:.2f} | "
                 f"{_pct(r['deployed'], r['deployed_lo'], r['deployed_hi'])} | "
                 f"{_pct(r['search_traj'], r['search_traj_lo'], r['search_traj_hi'])} | "
                 f"{_pct(r['search_final'], r['search_final_lo'], r['search_final_hi'])} | "
                 f"{_pct(r['selection_share'], r['selection_share_lo'], r['selection_share_hi'], 0)} |")
    L.append("")
    dep = h["deployed"].to_numpy()
    L.append(f"Deployed cost across the twenty landscapes: median {np.median(dep) * 100:.1f}%, interquartile range "
             f"{np.percentile(dep, 25) * 100:.1f}% to {np.percentile(dep, 75) * 100:.1f}%, range "
             f"{dep.min() * 100:.1f}% to {dep.max() * 100:.1f}%; "
             f"{int((h['deployed_lo'] > 0).sum())} of 20 have an interval above zero. Selection share: median "
             f"{np.nanmedian(h['selection_share']) * 100:.0f}%, range {np.nanmin(h['selection_share']) * 100:.0f}% "
             f"to {np.nanmax(h['selection_share']) * 100:.0f}%; above 50% in "
             f"{int((h['selection_share'] > 0.5).sum())} of 20.\n")

    L.append("## 4. Ship rules on the subset (question 6)\n")
    L.append("Share of the standard process's deployed cost of error recovered by switching the ship rule from "
             "the single best rating (best_observed) to the rule named, the estimand of "
             "`scripts/analyse_ship_rules.py`: recovered = gain / cost, cost = std_noisy - std_clean, gain = "
             "std_noisy - rule_noisy, ratio of landscape means in opt_z units. First interval: landscape bootstrap "
             "(coarse with six landscapes); second: landscapes fixed, acquisitions and seeds resampled (crossed). "
             "'better/worse' = landscapes whose mean gain is positive/negative. 'price' = rule_clean - std_clean "
             "(opt_z units), what the rule costs without error.\n")

    def ship_row(rule, em, group, cell, std, onset):
        s = ship[(ship["estimand"] == "share_of_deployed_cost") & (ship["rule"] == rule)
                 & (ship["error_model"] == em) & (ship["group"] == group) & (ship["cell"] == cell)]
        s = s[(np.isclose(s["jitter_std"], std) if np.isfinite(std) else s["jitter_std"].isna())]
        s = s[(np.isclose(s["jitter_iteration"], onset) if np.isfinite(onset) else s["jitter_iteration"].isna())]
        return s.iloc[0] if len(s) else None

    cells_q6 = [("condition", 1.0, 0.0, "1 SD, trial 1"), ("condition", 1.0, 20.0, "1 SD, trial 21"),
                ("both_onsets", 1.0, np.nan, "1 SD, both onsets"), ("condition", 5.0, 0.0, "5 SD, trial 1"),
                ("condition", 5.0, 20.0, "5 SD, trial 21"), ("both_onsets", 5.0, np.nan, "5 SD, both onsets"),
                (f"pooled_std_ge_{POOL_MIN_STD:g}", np.nan, np.nan, ">= 0.25 SD pooled")]
    for em in ("gaussian", "pooled"):
        for rule in ("pm", "lcb1", "lcb2"):
            L.append(f"**{rule} vs best_observed, {'gaussian error' if em == 'gaussian' else 'pooled over the four error processes'}**\n")
            L.append("| cell | group | cost | recovered (landscape CI) | (run CI) | better/worse | price | runs same design |")
            L.append("|---|---|---|---|---|---|---|---|")
            for cell, std, onset, label in cells_q6:
                for g in PRIMARY_GROUPS + ("smooth_classical3",):
                    r = ship_row(rule, em, g, cell, std, onset)
                    if r is None:
                        continue
                    L.append(f"| {label} | {g} | {r['cost']:.3f} | "
                             f"{_pct(r['recovered'], r['recovered_lo'], r['recovered_hi'])} | "
                             f"{_pct(np.nan) if not np.isfinite(r['recovered_cond_lo']) else '[' + format(r['recovered_cond_lo'] * 100, '+.0f') + ', ' + format(r['recovered_cond_hi'] * 100, '+.0f') + ']'} | "
                             f"{int(r['n_landscapes_rule_better'])}/{int(r['n_landscapes_rule_worse'])} | "
                             f"{r['price']:+.4f} | {r['share_runs_same_design'] * 100:.0f}% |")
            L.append("")
    # Per-landscape pm on the laws.
    L.append("**pm vs best_observed per law landscape, gaussian error** (recovered, run-level interval)\n")
    L.append("| landscape | 1 SD trial 1 | 1 SD trial 21 | 5 SD trial 1 | 5 SD trial 21 | >= 0.25 SD pooled |")
    L.append("|---|---|---|---|---|---|")
    for land in HUMAN_LAWS + SMOOTH_CLASSICAL:
        cells_txt = []
        for cell, std, onset in (("condition", 1.0, 0.0), ("condition", 1.0, 20.0), ("condition", 5.0, 0.0),
                                 ("condition", 5.0, 20.0), (f"pooled_std_ge_{POOL_MIN_STD:g}", np.nan, np.nan)):
            r = ship_row("pm", "gaussian", f"landscape:{land}", cell, std, onset)
            cells_txt.append(_pct(r["recovered"], r["recovered_cond_lo"], r["recovered_cond_hi"], 0)
                             if r is not None else "n/a")
        L.append(f"| {land} | " + " | ".join(cells_txt) + " |")
    L.append("")
    # pm by error process: where the GP's i.i.d.-noise likelihood is wrong.
    L.append("**pm vs best_observed by error process** (recovered, landscape interval). Only gaussian error "
             "matches the GP's i.i.d. noise model; ar1, bias and drift are temporally structured.\n")
    L.append("| error process | group | 1 SD, trial 1 | 5 SD, trial 21 | >= 0.25 SD pooled |")
    L.append("|---|---|---|---|---|")
    for em in ERROR_MODELS:
        for g in ("laws6", "smooth_classical3", "rugged_classical10"):
            vals = []
            for cell, std, onset in (("condition", 1.0, 0.0), ("condition", 5.0, 20.0),
                                     (f"pooled_std_ge_{POOL_MIN_STD:g}", np.nan, np.nan)):
                r = ship_row("pm", em, g, cell, std, onset)
                vals.append(_pct(r["recovered"], r["recovered_lo"], r["recovered_hi"]) if r is not None else "n/a")
            L.append(f"| {em} | {g} | " + " | ".join(vals) + " |")
    L.append("")
    # The selection-loss estimand for comparison with the paper's -8.7%.
    L.append("**The paper's other ship-rule number: share of the SELECTION loss removed** "
             "(`decompose_regret.rule_recovery`, pooled over error processes)\n")
    L.append("| rule | group | 1 SD, trial 1 | 5 SD, trial 1 | all cells >= 0.25 SD | every cell (the paper's pool) |")
    L.append("|---|---|---|---|---|---|")
    for rule in ("pm", "lcb1", "lcb2"):
        for g in PRIMARY_GROUPS:
            vals = []
            for cell, std, onset in (("condition", 1.0, 0.0), ("condition", 5.0, 0.0),
                                     (f"pooled_std_ge_{POOL_MIN_STD:g}", np.nan, np.nan),
                                     ("pooled_all_magnitudes", np.nan, np.nan)):
                s = ship[(ship["estimand"] == "share_of_selection_loss") & (ship["rule"] == rule)
                         & (ship["error_model"] == "pooled") & (ship["group"] == g) & (ship["cell"] == cell)]
                s = s[(np.isclose(s["jitter_std"], std) if np.isfinite(std) else s["jitter_std"].isna())]
                s = s[(np.isclose(s["jitter_iteration"], onset) if np.isfinite(onset) else s["jitter_iteration"].isna())]
                vals.append(_pct(s.iloc[0]["recovered"], s.iloc[0]["recovered_lo"], s.iloc[0]["recovered_hi"])
                            if len(s) else "n/a")
            L.append(f"| {rule} | {g} | " + " | ".join(vals) + " |")
    L.append("")
    # Direct answer.
    L.append("**Answer to question 6.**\n")
    for em in ("gaussian", "pooled"):
        parts = []
        for cell, std, onset, label in cells_q6:
            r = ship_row("pm", em, "laws6", cell, std, onset)
            if r is None:
                continue
            parts.append(f"{label}: {_pct(r['recovered'], r['recovered_lo'], r['recovered_hi'])}, run-level "
                         f"[{r['recovered_cond_lo'] * 100:+.0f}, {r['recovered_cond_hi'] * 100:+.0f}], "
                         f"{int(r['n_landscapes_rule_better'])} of 6 laws better")
        L.append(f"- pm on the six laws, {em}: " + "; ".join(parts) + ".")
    for g in ("classical13", "all20"):
        r1 = ship_row("pm", "gaussian", g, "condition", 1.0, 0.0)
        r5 = ship_row("pm", "gaussian", g, "condition", 5.0, 0.0)
        L.append(f"- for comparison, pm on {g}, gaussian: 1 SD trial 1 {_pct(r1['recovered'], r1['recovered_lo'], r1['recovered_hi'])}; "
                 f"5 SD trial 1 {_pct(r5['recovered'], r5['recovered_lo'], r5['recovered_hi'])}.")
    L.append("")

    if gps is not None:
        L.append("## 5. GP fit at n = 50\n")
        L.append(f"{len(GP_ACQS)} acquisitions ({', '.join(GP_ACQS)}) x seeds {GP_SEEDS[0]}-{GP_SEEDS[-1]} x 20 "
                 "landscapes, gaussian error at 1 SD from trial 1 and the matching clean runs: "
                 f"{int(gps[gps['level'] == 'landscape']['n_runs'].sum())} refits of the loop's own GP "
                 "(`rescore_ship_rules.fit_loop_gp`). Group values are means of landscape means with a landscape "
                 "bootstrap. Units: landscape SDs (the objective is standardised over its box, so a predictor of 0 "
                 "has RMSE about 1 at uniform points). LOO: closed form with the fitted hyperparameters, "
                 "nats per point in objective units; 'gain' is over a trivial predictor (mean and SD of the other "
                 "49 ratings); 'fraction achievable' divides that gain by the gain of an oracle that knows f and the "
                 "injected SD. Lengthscales: fraction of the box width. Noise: sqrt(likelihood.noise x stdvs^2).\n")
        if checks:
            L.append(f"Checks on the {checks['n']} refits: best_observed regret reproduces the table and the log "
                     f"in all; the best-visited regret reproduces the table and the log's final simple regret in all; "
                     f"the oracle reproduces the logged objective_true to {checks['max_oracle_err']:.1e}; the hand-built "
                     f"LOO posterior reproduces gp.posterior to {checks['max_posterior_mean_err']:.1e}; the refitted "
                     f"GP ships the same pm design as the ship-rule table in {checks['pm_idx_matches']} of {checks['n']} "
                     f"runs (pm regret identical in {checks['pm_regret_matches']}); fitted noise SD differs from the "
                     f"table's by at most {checks['noise_sd_max_abs_diff']:.1e}; fit failures {checks['fit_failures']}.\n")
        show = [("noise_sd_fit", "fitted noise SD", "{:.2f}"), ("noise_sd_ratio", "fitted / injected SD", "{:.2f}"),
                ("ls_geomean", "lengthscale (geo. mean, box widths)", "{:.2f}"),
                ("ls_min", "shortest lengthscale", "{:.2f}"),
                ("loo_lpd", "LOO log lik. / point", "{:.2f}"), ("loo_lpd_gain", "LOO gain over trivial", "{:.2f}"),
                ("loo_lpd_frac_achievable", "LOO fraction achievable", "{:.2f}"),
                ("loo_mean_z2", "LOO mean z^2 (1 = calibrated)", "{:.2f}"),
                ("visited_rmse_pm", "RMSE(pm, f) at visited", "{:.3f}"),
                ("visited_rmse_obs", "RMSE(rating, f) at visited", "{:.3f}"),
                ("visited_shrink_ratio", "RMSE ratio pm / rating", "{:.2f}"),
                ("visited_r2_pm", "R^2 of pm at visited", "{:.2f}"),
                ("visited_spearman_pm", "rank corr. pm vs f, visited", "{:.2f}"),
                ("visited_spearman_obs", "rank corr. rating vs f, visited", "{:.2f}"),
                ("rank_best_visited_under_pm", "rank of best visited under pm", "{:.1f}"),
                ("rank_best_visited_under_obs", "rank of best visited under rating", "{:.1f}"),
                ("sobol_rmse", "RMSE(pm, f) at 256 Sobol", "{:.3f}"),
                ("sobol_r2", "R^2 at 256 Sobol", "{:.2f}"),
                ("sobol_cov95", "95% band coverage, Sobol", "{:.2f}"),
                ("pm_gain_over_opt_z", "pm gain over best_obs (true value / opt_z; + = pm better)", "{:+.4f}")]
        for cond in ("gaussian_1sd_onset0", "clean"):
            L.append(f"**{cond}**\n")
            L.append("| metric | " + " | ".join(PRIMARY_GROUPS + ("smooth_classical3", "rugged_classical10")) + " |")
            L.append("|---|" + "---|" * 5)
            for col, label, fmt in show:
                vals = []
                for g in PRIMARY_GROUPS + ("smooth_classical3", "rugged_classical10"):
                    r = gps[(gps["level"] == "group") & (gps["name"] == g) & (gps["condition"] == cond)].iloc[0]
                    lo, hi = r.get(f"{col}_lo", np.nan), r.get(f"{col}_hi", np.nan)
                    vals.append(_num(r[col], lo, hi, fmt))
                L.append(f"| {label} | " + " | ".join(vals) + " |")
            L.append("")
        L.append("**Per landscape, gaussian 1 SD from trial 1** (means over 20 runs)\n")
        L.append("| landscape | group | d | noise ratio | lengthscale | LOO frac. achiev. | RMSE pm/rating | "
                 "R^2 Sobol | rank corr. pm | rank corr. rating | pm - best_obs |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        lr = gps[(gps["level"] == "landscape") & (gps["condition"] == "gaussian_1sd_onset0")].sort_values(
            ["group", "sobol_r2"], ascending=[False, False])
        for _, r in lr.iterrows():
            L.append(f"| {r['name']} | {r['group']} | {int(r['dim'])} | {r['noise_sd_ratio']:.2f} | "
                     f"{r['ls_geomean']:.2f} | {r['loo_lpd_frac_achievable']:.2f} | {r['visited_shrink_ratio']:.2f} | "
                     f"{r['sobol_r2']:.2f} | {r['visited_spearman_pm']:.2f} | {r['visited_spearman_obs']:.2f} | "
                     f"{r['pm_gain_over_opt_z']:+.4f} |")
        L.append("")
        if fitpm is not None:
            L.append("**Does fit quality explain pm's result?** Spearman correlations (`gp_fit_vs_pm.csv`). "
                     "'pm_gain_full' is pm's per-landscape gain over best_observed on the full table (ten "
                     "acquisitions, gaussian, 1 SD, trial 1); 'pm_gain_over_opt_z' is the same contrast in the "
                     "refitted sample. Both are positive when pm ships the better design. Within-landscape "
                     "correlations rank runs inside each landscape.\n")
            L.append("| analysis | scope | outcome | predictor | n | Spearman | p |")
            L.append("|---|---|---|---|---|---|---|")
            for _, r in fitpm[fitpm["predictor"].isin(["sobol_r2", "visited_r2_pm", "visited_shrink_ratio",
                                                        "loo_lpd_frac_achievable", "ls_geomean",
                                                        "noise_sd_ratio"])].iterrows():
                L.append(f"| {r['analysis']} | {r['scope']} | {r['outcome']} | {r['predictor']} | {r['n']} | "
                         f"{r['spearman']:+.2f} | {r['p']:.3f} |")
            L.append("")
        if strata is not None:
            L.append("**pm's gain over best_observed by fit tertile** (tertiles over all 400 noisy refits; true "
                     "value / opt_z; positive = pm better)\n")
            L.append("| predictor | group | tertile | runs | landscapes | predictor mean | pm gain | pm better | pm worse |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for _, r in strata.iterrows():
                L.append(f"| {r['predictor']} | {r['group']} | {r['stratum']} | {r['n_runs']} | {r['n_landscapes']} | "
                         f"{r['predictor_mean']:.2f} | {r['pm_gain_over_opt_z']:+.4f} | "
                         f"{r['share_pm_better'] * 100:.0f}% | {r['share_pm_worse'] * 100:.0f}% |")
            L.append("")
    if gps is not None and fitpm is not None:
        L.extend(interpretation(head, diffs, perl, ship, gps, fitpm))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def interpretation(head, diffs, perl, ship, gps, fitpm) -> list[str]:
    """Section 6: what the tables say about the reviewer's three points. Numbers are read, not typed."""
    def hrow(group, std=1.0, onset=0):
        b = head[(head["error_model"] == "pooled") & np.isclose(head["jitter_std"], std)
                 & (head["jitter_iteration"] == onset) & (head["group"] == group)]
        return b.iloc[0]

    def srow(rule, em, group, cell, std, onset, estimand="share_of_deployed_cost"):
        s = ship[(ship["estimand"] == estimand) & (ship["rule"] == rule) & (ship["error_model"] == em)
                 & (ship["group"] == group) & (ship["cell"] == cell)]
        s = s[(np.isclose(s["jitter_std"], std) if np.isfinite(std) else s["jitter_std"].isna())]
        s = s[(np.isclose(s["jitter_iteration"], onset) if np.isfinite(onset) else s["jitter_iteration"].isna())]
        return s.iloc[0]

    def grow(group, cond):
        return gps[(gps["level"] == "group") & (gps["name"] == group) & (gps["condition"] == cond)].iloc[0]

    def corr(analysis, scope, outcome, predictor):
        r = fitpm[(fitpm["analysis"] == analysis) & (fitpm["scope"] == scope) & (fitpm["outcome"] == outcome)
                  & (fitpm["predictor"] == predictor)].iloc[0]
        return f"rho = {r['spearman']:+.2f} (p = {r['p']:.3f}, n = {int(r['n'])})"

    P = f"pooled_std_ge_{POOL_MIN_STD:g}"
    a, l, c = hrow("all20"), hrow("laws6"), hrow("classical13")
    d = diffs[(diffs["error_model"] == "pooled") & np.isclose(diffs["jitter_std"], 1.0)
              & (diffs["jitter_iteration"] == 0)].iloc[0]
    h = perl[perl["headline_cell"]]
    lg, cg = srow("pm", "gaussian", "laws6", "condition", 1.0, 0.0), srow("pm", "gaussian", "classical13", "condition", 1.0, 0.0)
    l5, c5 = srow("pm", "gaussian", "laws6", "condition", 5.0, 0.0), srow("pm", "gaussian", "classical13", "condition", 5.0, 0.0)
    sm, ru = srow("pm", "gaussian", "smooth_classical3", P, np.nan, np.nan), srow("pm", "gaussian", "rugged_classical10", P, np.nan, np.nan)
    lp = srow("pm", "gaussian", "laws6", P, np.nan, np.nan)
    rugp = srow("pm", "pooled", "rugged_classical10", P, np.nan, np.nan)
    ldr, lbi = srow("pm", "drift", "laws6", "condition", 5.0, 20.0), srow("pm", "bias", "laws6", "condition", 5.0, 20.0)
    sel_all = srow("pm", "pooled", "all20", "pooled_all_magnitudes", np.nan, np.nan, "share_of_selection_loss")
    sel_l = srow("pm", "pooled", "laws6", "pooled_all_magnitudes", np.nan, np.nan, "share_of_selection_loss")
    sel_c = srow("pm", "pooled", "classical13", "pooled_all_magnitudes", np.nan, np.nan, "share_of_selection_loss")
    gl, gc, gr = grow("laws6", "gaussian_1sd_onset0"), grow("classical13", "gaussian_1sd_onset0"), grow("rugged_classical10", "gaussian_1sd_onset0")
    kl, kc, kr = grow("laws6", "clean"), grow("classical13", "clean"), grow("rugged_classical10", "clean")
    ks = grow("smooth_classical3", "clean")
    L = ["## 6. What the tables say about the three points\n"]
    L.append(f"**(i) The headline is not an artefact of multimodal test functions.** At 1 SD from the first rating "
             f"the deployed cost is {_pct(l['deployed'], l['deployed_lo'], l['deployed_hi'])} of opt_z on the six "
             f"laws and {_pct(c['deployed'], c['deployed_lo'], c['deployed_hi'])} on the thirteen classical "
             f"functions (all 20: {_pct(a['deployed'], a['deployed_lo'], a['deployed_hi'])}; laws minus classical "
             f"{_pct(d['deployed'], d['deployed_lo'], d['deployed_hi'])}). The search cost is larger on the laws "
             f"({_pct(l['search_traj'])} vs {_pct(c['search_traj'])}), and the selection share is the same "
             f"({_pct(l['selection_share'], l['selection_share_lo'], l['selection_share_hi'], 0)} vs "
             f"{_pct(c['selection_share'], c['selection_share_lo'], c['selection_share_hi'], 0)}). With error from "
             "trial 21 the laws lose less in total but a larger share of it is selection (section 2).\n")
    L.append(f"**(ii) Per landscape.** Every one of the twenty landscapes has a deployed cost above zero at 1 SD "
             f"from trial 1 (median {np.median(h['deployed']) * 100:.1f}%, range {h['deployed'].min() * 100:.1f}% "
             f"to {h['deployed'].max() * 100:.1f}%), so the pooled mean describes a consistent effect. The selection "
             f"share is heterogeneous (range {np.nanmin(h['selection_share']) * 100:.0f}% to "
             f"{np.nanmax(h['selection_share']) * 100:.0f}%; lowest on "
             f"{', '.join(h.sort_values('selection_share')['landscape'].head(3))}, where the search itself fails "
             "most), so a pooled share needs the per-landscape figure beside "
             "it. `per_landscape_headline.csv` is the tidy table for that figure.\n")
    L.append(f"**(iii) / question 6: yes, on the smooth landscapes pm beats the single best rating, and the "
             f"reviewer's misspecification explanation is largely right for gaussian error.** Under gaussian error "
             f"pm recovers {_pct(lg['recovered'], lg['recovered_lo'], lg['recovered_hi'])} of the deployed cost on "
             f"the six laws at 1 SD from trial 1 ({int(lg['n_landscapes_rule_better'])} of 6 better) and "
             f"{_pct(l5['recovered'], l5['recovered_lo'], l5['recovered_hi'])} at 5 SD "
             f"({int(l5['n_landscapes_rule_better'])} of 6), against "
             f"{_pct(cg['recovered'], cg['recovered_lo'], cg['recovered_hi'])} and "
             f"{_pct(c5['recovered'], c5['recovered_lo'], c5['recovered_hi'])} on the classical functions. The "
             f"secondary split puts the line at smoothness, not at 'law': pooled over >= 0.25 SD, pm recovers "
             f"{_pct(lp['recovered'], lp['recovered_lo'], lp['recovered_hi'])} on the laws, "
             f"{_pct(sm['recovered'], sm['recovered_lo'], sm['recovered_hi'])} on the three smooth classical "
             f"functions and {_pct(ru['recovered'], ru['recovered_lo'], ru['recovered_hi'])} on the ten rugged ones. "
             f"With six landscapes the smallest attainable two-sided Wilcoxon p is 0.031, which the 1 SD cells reach.\n")
    L.append("The GP diagnostics say why.\n")
    L.append(f"- The noise estimate is NOT what separates the groups: at n = 50 the fitted noise SD is "
             f"{_num(gl['noise_sd_ratio'], gl['noise_sd_ratio_lo'], gl['noise_sd_ratio_hi'], '{:.2f}')} of the "
             f"injected SD on the laws and {_num(gc['noise_sd_ratio'], gc['noise_sd_ratio_lo'], gc['noise_sd_ratio_hi'], '{:.2f}')} "
             f"on the classical functions: largely recovered in both (short by "
             f"{(1 - gl['noise_sd_ratio']) * 100:.0f}% and {(1 - gc['noise_sd_ratio']) * 100:.0f}%), as the reviewer says.")
    L.append(f"- The posterior mean does shrink the ratings everywhere (RMSE against the truth at the visited "
             f"designs is {_num(gl['visited_shrink_ratio'], None, None, '{:.2f}')} of the raw ratings' on the laws, "
             f"{_num(gc['visited_shrink_ratio'], None, None, '{:.2f}')} on the classical, "
             f"{_num(gr['visited_shrink_ratio'], None, None, '{:.2f}')} on the rugged ten), but less where the fit "
             "is worse, and its rank advantage over the ratings among the visited designs is "
             f"{gl['visited_spearman_pm'] - gl['visited_spearman_obs']:+.2f} (rank correlation) on the laws against "
             f"{gc['visited_spearman_pm'] - gc['visited_spearman_obs']:+.2f} on the classical functions.")
    L.append(f"- Misspecification is visible with NO error at all. On clean runs best_observed is the best visited "
             f"design, so pm's price there is pure surrogate error: {lg['price']:.4f} of opt_z on the laws, "
             f"{sm['price']:.4f} on the smooth classical three and {ru['price']:.4f} on the rugged ten. The clean "
             f"refits agree: RMSE of the posterior mean against the exact values it was fitted to is "
             f"{kl['visited_rmse_pm']:.3f} landscape SDs on the laws against {kr['visited_rmse_pm']:.3f} on the "
             f"rugged ten; the GP explains unmodelled roughness as noise (fitted 'noise' SD {kl['noise_sd_fit']:.2f} "
             f"on the laws, {ks['noise_sd_fit']:.2f} on the smooth classical, {kr['noise_sd_fit']:.2f} on the rugged "
             f"ten, where the true noise is zero); lengthscales are {kl['ls_geomean']:.2f} against "
             f"{kr['ls_geomean']:.2f} box widths; held-out R^2 at fresh Sobol points is {kl['sobol_r2']:.2f} against "
             f"{kr['sobol_r2']:.2f}.")
    L.append(f"- Across landscapes fit quality predicts pm's gain: with the LOO fraction of achievable predictive "
             f"gain, {corr('across_landscapes', 'all20', 'pm_gain_full', 'loo_lpd_frac_achievable')} over all 20 and "
             f"{corr('across_landscapes', 'classical13', 'pm_gain_full', 'loo_lpd_frac_achievable')} within the "
             f"classical functions; with the shrinkage ratio "
             f"{corr('across_landscapes', 'all20', 'pm_gain_full', 'visited_shrink_ratio')}. Within a landscape, "
             f"run-to-run variation in fit does not predict the run's outcome "
             f"({corr('within_landscapes', 'all20', 'pm_gain_over_opt_z', 'loo_lpd_frac_achievable')}), so fit acts "
             "as a property of the landscape, and a per-run fit diagnostic would not tell a practitioner when to "
             "trust pm.")
    lar = srow("pm", "ar1", "laws6", "condition", 5.0, 20.0)
    L.append(f"- The corrupted-posterior explanation survives where the LIKELIHOOD is wrong. Under drift and bias, "
             f"which the GP's i.i.d. noise model cannot represent, pm loses even on the laws late in a study: at "
             f"5 SD from trial 21, {_pct(ldr['recovered'], ldr['recovered_lo'], ldr['recovered_hi'])} under drift "
             f"and {_pct(lbi['recovered'], lbi['recovered_lo'], lbi['recovered_hi'])} under bias (under ar1 "
             f"{_pct(lar['recovered'], lar['recovered_lo'], lar['recovered_hi'])}). Pooled over the four "
             f"processes the rugged ten give {_pct(rugp['recovered'], rugp['recovered_lo'], rugp['recovered_hi'])}.")
    L.append(f"- The paper's pooled selection-loss figure for pm (every cell pooled) reproduces here as "
             f"{_pct(sel_all['recovered'], sel_all['recovered_lo'], sel_all['recovered_hi'])} over all 20; it splits "
             f"into {_pct(sel_l['recovered'], sel_l['recovered_lo'], sel_l['recovered_hi'])} on the laws and "
             f"{_pct(sel_c['recovered'], sel_c['recovered_lo'], sel_c['recovered_hi'])} on the classical functions.\n")
    L.append("So the sentence 'the posterior ... inherits the corrupted ratings' is too broad. On smooth "
             "landscapes under i.i.d. error the posterior mean shrinks the noise and pm beats the single best "
             "rating. pm's pooled loss comes from two sources: surrogate misspecification on the rugged classical "
             "functions, which already costs pm a price on clean runs, and likelihood misspecification under "
             "drift and bias, which costs it even on smooth landscapes once the error starts late. The cautious "
             "rules are "
             f"comparable to pm on the laws (gaussian, >= 0.25 SD pooled: lcb1 "
             f"{_pct(srow('lcb1', 'gaussian', 'laws6', P, np.nan, np.nan)['recovered'])}, lcb2 "
             f"{_pct(srow('lcb2', 'gaussian', 'laws6', P, np.nan, np.nan)['recovered'])}, pm {_pct(lp['recovered'])}) "
             f"and hold up on the rugged ten, where pm does not (lcb2 "
             f"{_pct(srow('lcb2', 'gaussian', 'rugged_classical10', P, np.nan, np.nan)['recovered'])}, "
             f"pm {_pct(ru['recovered'])}).\n")
    return L


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("output-boba"))
    p.add_argument("--out-dir", type=Path, default=Path("output-boba/analysis/review"))
    p.add_argument("--workers", type=int, default=4, help="GP refit processes (at most 4 here)")
    p.add_argument("--reuse-gp", action="store_true", help="read gp_diagnostics_per_run.csv instead of refitting")
    p.add_argument("--skip-gp", action="store_true", help="steps 1-4 only")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    analysis = args.input_dir / "analysis"
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}

    # 1. Definition first.
    definition_path = out / "smooth_subset_definition.csv"
    if definition_path.is_file():
        defined_at = _dt.datetime.fromtimestamp(definition_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        old = pd.read_csv(definition_path)
        if sorted(old.loc[old["primary_smooth_subset"], "landscape"]) != sorted(HUMAN_LAWS):
            raise ValueError("the recorded a-priori subset differs from HUMAN_LAWS; it must not change")
        definition = old
    else:
        definition = write_definition(out, stats)
        defined_at = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"1. subset: laws6 = {', '.join(HUMAN_LAWS)} (defined {defined_at})", flush=True)

    # 2. Headline.
    cells = load_cells(analysis, opt_z)
    noisy, clean = load_decomposition(analysis, opt_z)
    head = headline(cells, noisy, clean)
    head.to_csv(out / "smooth_subset.csv", index=False)
    diffs = headline_differences(cells, noisy, clean)
    diffs.to_csv(out / "smooth_subset_differences.csv", index=False)
    print("2. headline written", flush=True)

    # 3. Per landscape.
    runs = per_run_costs(args.input_dir, noisy, clean, opt_z)
    perl = per_landscape(runs, stats)
    perl.to_csv(out / "per_landscape_headline.csv", index=False)
    print("3. per-landscape written", flush=True)

    # 4. Ship rules.
    ship = ship_rules_subset(analysis, opt_z, noisy)
    ship.to_csv(out / "ship_rules_subset.csv", index=False)
    print("4. ship rules written", flush=True)

    # 5. GP diagnostics.
    gps = fitpm = strata = checks = None
    if not args.skip_gp:
        per_run_path = out / "gp_diagnostics_per_run.csv"
        if args.reuse_gp and per_run_path.is_file():
            gp_runs = pd.read_csv(per_run_path)
        else:
            tasks = gp_tasks(args.input_dir)
            workers = max(1, min(args.workers, 4, os.cpu_count() or 1))
            print(f"5. refitting {len(tasks)} GPs with {workers} workers", flush=True)
            gp_runs = run_gp(tasks, workers)
            gp_runs.to_csv(per_run_path, index=False)
        checks = check_gp_against_table(gp_runs, analysis)
        gps = gp_summary(gp_runs)
        gps.to_csv(out / "gp_diagnostics.csv", index=False)
        fitpm, _ = gp_fit_vs_pm(gp_runs, ship)
        fitpm.to_csv(out / "gp_fit_vs_pm.csv", index=False)
        strata = pm_by_fit_stratum(gp_runs)
        strata.to_csv(out / "gp_fit_strata.csv", index=False)
        print(f"5. GP diagnostics written; checks {checks}", flush=True)

    write_markdown(out / "smooth_subset_gp.md", definition, head, diffs, perl, ship, gps, fitpm, strata, checks,
                   defined_at)
    print(f"wrote {out / 'smooth_subset_gp.md'}")


if __name__ == "__main__":
    main()
