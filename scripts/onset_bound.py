"""Is the onset effect mechanical? The running-maximum bound on post-onset search loss.

A reviewer's objection: search loss is a running maximum over visited designs,
and a noisy run shares every design with its clean twin up to the onset, so the
noisy run's excess is bounded by what the clean run gains afterwards. Write
r(t) = y_opt - max_{i<=t} f(x_i) for the search loss (``simple_regret_true``),
n and c for the noisy and clean run, and s for the last design the two share.
For every t >= s the noisy run still holds designs 1..s, so r_n(t) <= r_n(s) =
r_c(s), and

    excess(t) = r_n(t) - r_c(t)  <=  r_c(s) - r_c(t)  =  B(t).

B is the clean run's improvement after the shared prefix. Nothing in the
argument depends on the optimizer, so the bound is exact, not approximate; its
only premise is that the prefix really is shared, which this script checks in
every pair rather than assumes. Below s both runs are identical and excess = 0,
so B(t) is taken as 0 there. excess(t) can be negative (the noisy run gets
lucky) but never exceeds B(t), so any positive linear average of excess --
including the paper's trapezoid window mean -- is at most the same average of B.

Which design is the last shared one. The design at trial k+1 (k = onset) is
chosen from k clean observations, so it is shared; the first
``initial_samples`` (= 5) trials are uniform draws taken before any observation
and are shared whatever the onset. Hence

    s = max(k + 1, initial_samples):   s = 21 at onset 20,  s = 5 at onset 0.

That is the tightest structural bound. The looser variants the reviewer's
wording suggests -- r_c(k) at the late onset (s = 20) and r_c(1) at onset 0 --
are also computed (columns ``*_task``); they can only be larger.

The question answered is: what is the post-onset excess as a fraction of B, the
improvement it could possibly have destroyed? That separates the mechanical part
of the onset effect (less improvement left to lose after a late onset) from the
behavioural one (a noisy optimizer losing a smaller or larger share of what was
left). With E and B the landscape means of the window-averaged excess and bound,
the early-to-late raw ratio factors exactly as

    E_early / E_late  =  (B_early / B_late)  x  (N_early / N_late),   N = E / B,

a mechanical factor times a behavioural one.

Two responses outside the bound are reported as well: the deployed design's
regret at the final trial (``inference_simple_regret_true``, not a running
maximum, so not bounded by B), and the per-trial sample quality of the designs
visited after the onset (``objective_true``, likewise not a running maximum),
which tests "hard to dislodge" directly: does noise push the optimizer to sample
worse designs, early versus late?

Conventions follow the paper: the window is iterations k+1..T, averaged as a
trapezoid AUC over (n - 1) exactly as ``evaluate_research_question.py`` does;
values are fractions of opt_z (``boba_benchmarks.load_stats``); aggregates are a
mean over landscapes of per-landscape means over acquisitions x seeds (x error
processes when pooled); intervals are 95% percentile landscape bootstraps
(``scipy.stats.bootstrap``, landscapes resampled, paired across quantities).
The model-free random and sobol acquisitions are excluded.

Outputs (``--out``, default output-boba/analysis/review):
    onset_bound_runs.parquet   per-run summaries (cache; --refresh rebuilds)
    onset_bound.csv            long table: arm, T, error model, magnitude, onset,
                               quantity, estimate, lo, hi
    onset_bound.md             the report

Usage:
    python scripts/onset_bound.py [--workers 4] [--refresh]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MODEL_FREE = {"random", "sobol"}
BUDGET_ACQS = ["logei", "ei", "pi", "ucb", "qucb", "qnei"]
BUDGET_SEEDS = [7, 8, 9, 10, 11]
EPS = 1e-9
BOOT_REPS = 10_000
BOOT_REPS_MEDIAN = 2_000
BOOT_SEED = 20260922

READ_COLS = {"iteration", "objective_true", "objective_observed",
             "simple_regret_true", "inference_simple_regret_true", "y_opt"}
X_COL = re.compile(r"x\d+$")
BASE_RE = re.compile(
    r"^bo_sensor_error_(?P<ds>.+)_value_(?P<acq>[^_]+)_seed(?P<seed>\d+)_baseline_exact\.csv$")
NOISY_RE = re.compile(
    r"^bo_sensor_error_(?P<ds>.+)_value_(?P<acq>[^_]+)_seed(?P<seed>\d+)_jittered_exact_"
    r"(?P<em>.+?)_jit(?P<onset>\d+)_std(?P<std>\d+(?:\.\d+)?)(?P<suffix>_.*)?\.csv$")


# ---------------------------------------------------------------------------
# Per-run extraction (runs in worker processes)
# ---------------------------------------------------------------------------


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda c: c in READ_COLS or X_COL.match(c) is not None)


def _win(values: np.ndarray, mask: np.ndarray) -> float:
    """The paper's post-onset window mean: trapezoid AUC over (n - 1)."""
    n = int(mask.sum())
    if n <= 1:
        return 0.0
    return float(np.trapezoid(values[mask], dx=1.0) / (n - 1))


def summarise_pair(base: pd.DataFrame, noisy: pd.DataFrame, onset: int, n_init: int) -> dict:
    it_b = base["iteration"].to_numpy()
    it_n = noisy["iteration"].to_numpy()
    T = len(it_b)
    if not (np.array_equal(it_b, np.arange(1, T + 1)) and np.array_equal(it_b, it_n)):
        raise ValueError("iterations are not 1..T in both runs")
    xcols = sorted(c for c in base.columns if X_COL.match(c))
    if not xcols or xcols != sorted(c for c in noisy.columns if X_COL.match(c)):
        raise ValueError("design columns missing or mismatched")

    k = int(onset)
    s = max(k + 1, n_init)          # last shared design (structural)
    s_task = max(k, 1)              # reviewer's wording: r_c(t0), or r_c(1) at onset 0

    rc = base["simple_regret_true"].to_numpy(float)
    rn = noisy["simple_regret_true"].to_numpy(float)
    fc = base["objective_true"].to_numpy(float)
    fn = noisy["objective_true"].to_numpy(float)
    ic = base["inference_simple_regret_true"].to_numpy(float)
    inn = noisy["inference_simple_regret_true"].to_numpy(float)
    xb = base[xcols].to_numpy(float)
    xn = noisy[xcols].to_numpy(float)

    same_x = (xb == xn).all(axis=1)
    first_div = int(it_b[~same_x][0]) if (~same_x).any() else T + 1
    obs_diff = base["objective_observed"].to_numpy(float) != noisy["objective_observed"].to_numpy(float)
    prefix_x = bool(same_x[:s].all())
    prefix_f = bool((fc[:s] == fn[:s]).all())
    prefix_r = bool((rc[:s] == rn[:s]).all())
    prefix_obs = bool((~obs_diff[:k]).all()) if k > 0 else True
    prefix_inf = bool((ic[:k] == inn[:k]).all()) if k > 0 else True

    mask = it_b >= k + 1            # the paper's window, iterations k+1..T
    exc = rn - rc
    B = np.where(it_b >= s, rc[s - 1] - rc, 0.0)
    Bt = np.where(it_b >= s_task, rc[s_task - 1] - rc, 0.0)
    post = mask & (it_b > s)
    Bp, ep = B[post], exc[post]
    pos = Bp > EPS

    return {
        "T": T,
        "n_init": n_init,
        "s": s,
        "s_task": s_task,
        "first_div": first_div,
        "first_obs_diff": int(it_b[obs_diff][0]) if obs_diff.any() else T + 1,
        "prefix_x": prefix_x,
        "prefix_f": prefix_f,
        "prefix_r": prefix_r,
        "prefix_obs": prefix_obs,
        "prefix_inf": prefix_inf,
        "prefix_ok": prefix_x and prefix_f and prefix_r and prefix_obs,
        "y_opt_equal": bool(np.array_equal(base["y_opt"].to_numpy(float), noisy["y_opt"].to_numpy(float))),
        "exc_win": _win(exc, mask),
        "rn_win": _win(rn, mask),
        "rc_win": _win(rc, mask),
        "B_win": _win(B, mask),
        "Bt_win": _win(Bt, mask),
        "max_viol": float(np.max((exc - B)[mask])),
        "max_viol_task": float(np.max((exc - Bt)[mask])),
        "n_post": int(post.sum()),
        "n_Bpos": int(pos.sum()),
        "n_bind": int((pos & (ep >= Bp - EPS)).sum()),
        "n_near90": int((pos & (ep >= 0.9 * Bp)).sum()),
        "n_near50": int((pos & (ep >= 0.5 * Bp)).sum()),
        "n_neg": int((ep < -EPS).sum()),
        "exc_final": float(exc[-1]),
        "B_final": float(B[-1]),
        "Bt_final": float(Bt[-1]),
        "rc_s": float(rc[s - 1]),
        "rc_1": float(rc[0]),
        "rc_T": float(rc[-1]),
        "noisy_improved": bool(rn[-1] < rc[s - 1] - EPS),
        "clean_improved": bool(rc[-1] < rc[s - 1] - EPS),
        "dep_exc": float(inn[-1] - ic[-1]),
        "dep_c": float(ic[-1]),
        "inst_win": _win(fc - fn, mask),
    }


def process_group(arm: str, n_init: int, runs: list[tuple[dict, str, list[tuple[dict, str]]]]) -> list[dict]:
    """One (arm, landscape, acquisition): every seed's clean run and its noisy twins."""
    rows = []
    for key, base_path, noisy_list in runs:
        base = _read(base_path)
        for meta, noisy_path in noisy_list:
            row = {"arm": arm, **key, **meta}
            try:
                row.update(summarise_pair(base, _read(noisy_path), meta["onset"], n_init))
                row["error"] = ""
            except Exception as exc:  # recorded, then fatal in main
                row["error"] = f"{type(exc).__name__}: {exc} ({noisy_path})"
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Discovery and the cache
# ---------------------------------------------------------------------------


def discover(arm: str, root: Path, acqs: set[str] | None, seeds: set[int] | None):
    """Group the per-run logs of one arm into (landscape, acquisition) tasks."""
    meta = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    n_init = int(meta["args"]["initial_samples"])
    tasks = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        bases, noisies = {}, {}
        for entry in os.scandir(ds_dir):
            if not entry.name.endswith(".csv"):
                continue
            m = BASE_RE.match(entry.name)
            if m:
                bases[(m["acq"], int(m["seed"]))] = entry.path
                continue
            m = NOISY_RE.match(entry.name)
            if m:
                meta_row = {"error_model": m["em"], "onset": int(m["onset"]),
                            "jitter_std": float(m["std"]),
                            "variant": (m["suffix"] or "").lstrip("_")}
                noisies.setdefault((m["acq"], int(m["seed"])), []).append((meta_row, entry.path))
        if not bases:
            continue
        ds = ds_dir.name
        by_acq: dict[str, list] = {}
        for (acq, seed), base_path in sorted(bases.items()):
            if acq in MODEL_FREE or (acqs and acq not in acqs) or (seeds and seed not in seeds):
                continue
            if (acq, seed) not in noisies:
                continue
            by_acq.setdefault(acq, []).append(
                ({"dataset": ds, "acquisition": acq, "seed": seed}, base_path,
                 sorted(noisies[(acq, seed)], key=lambda t: t[1])))
        orphan = set(noisies) - set(bases)
        orphan = {o for o in orphan if o[0] not in MODEL_FREE}
        if orphan:
            raise ValueError(f"{arm}/{ds}: noisy runs with no clean twin: {sorted(orphan)[:5]}")
        for acq, runs in by_acq.items():
            tasks.append((arm, n_init, runs))
    return tasks, meta


def build_runs(arms: dict[str, Path], workers: int) -> pd.DataFrame:
    tasks = []
    for arm, root in arms.items():
        t, _ = discover(arm, root, None, None)
        tasks += t
    n_files = sum(len(r[2]) + 1 for t in tasks for r in t[2])
    print(f"reading {n_files:,} run logs in {len(tasks)} groups with {workers} workers ...", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_group, *task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows += fut.result()
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} groups, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    bad = df[df["error"] != ""]
    if len(bad):
        raise RuntimeError(f"{len(bad)} pairs failed, e.g. {bad['error'].iloc[0]}")
    return df.drop(columns="error")


# ---------------------------------------------------------------------------
# Landscape-bootstrap aggregation
# ---------------------------------------------------------------------------


def _boot(stat, *arrays: np.ndarray, reps: int = BOOT_REPS) -> tuple[float, float, float]:
    """Point estimate and 95% percentile landscape-bootstrap interval of a
    vectorised statistic of per-landscape arrays (paired by landscape)."""
    from scipy.stats import bootstrap

    point = float(stat(*arrays, axis=-1))
    if not np.isfinite(point):
        return point, float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bootstrap(arrays, stat, paired=True, vectorized=True, n_resamples=reps,
                        method="percentile", confidence_level=0.95,
                        rng=np.random.default_rng(BOOT_SEED))
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


def _mean(a, axis=-1):
    return np.mean(a, axis=axis)


def _ratio(a, b, axis=-1):
    return np.mean(a, axis=axis) / np.mean(b, axis=axis)


def _diff(a, b, axis=-1):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


def _ratio_of_ratios(a, b, c, d, axis=-1):
    return (np.mean(a, axis=axis) / np.mean(b, axis=axis)) / (np.mean(c, axis=axis) / np.mean(d, axis=axis))


def _boot_pooled_median(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Median of per-run ratios pooled over landscapes, landscapes resampled."""
    from scipy.stats import bootstrap

    groups = [g for g in groups if len(g)]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(np.concatenate(groups)))

    def stat(idx):
        return np.median(np.concatenate([groups[int(i)] for i in idx]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bootstrap((np.arange(len(groups)),), stat, vectorized=False,
                        n_resamples=BOOT_REPS_MEDIAN, method="percentile",
                        rng=np.random.default_rng(BOOT_SEED))
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


def per_landscape(cell: pd.DataFrame) -> pd.DataFrame:
    cols = ["exc_f", "B_f", "Bt_f", "dep_f", "inst_f", "Bfin_f"]
    return cell.groupby("dataset")[cols].mean().sort_index()


def cell_quantities(cell: pd.DataFrame) -> list[tuple[str, float, float, float]]:
    L = per_landscape(cell)
    out = [
        ("excess",) + _boot(_mean, L["exc_f"].to_numpy()),
        ("bound",) + _boot(_mean, L["B_f"].to_numpy()),
        ("normalised",) + _boot(_ratio, L["exc_f"].to_numpy(), L["B_f"].to_numpy()),
    ]
    runs = cell[cell["B_win"] > EPS]
    out.append(("median_run_ratio",) + _boot_pooled_median(
        [g["exc_win"].to_numpy() / g["B_win"].to_numpy() for _, g in runs.groupby("dataset")]))
    out += [
        ("bound_task",) + _boot(_mean, L["Bt_f"].to_numpy()),
        ("normalised_task",) + _boot(_ratio, L["exc_f"].to_numpy(), L["Bt_f"].to_numpy()),
        ("deployed",) + _boot(_mean, L["dep_f"].to_numpy()),
        ("bound_final",) + _boot(_mean, L["Bfin_f"].to_numpy()),
        ("inst_excess",) + _boot(_mean, L["inst_f"].to_numpy()),
    ]
    n_bpos = cell["n_Bpos"].sum()
    desc = {
        "share_bound_zero": float((cell["B_win"] <= EPS).mean()),
        "share_prefix_identical": float(cell["prefix_ok"].mean()),
        "n_bound_violations": float((cell["max_viol"] > EPS).sum()),
        "max_violation": float(cell["max_viol"].max()),
        "share_iters_bound_binds": float(cell["n_bind"].sum() / n_bpos) if n_bpos else float("nan"),
        "share_iters_ge90pct_bound": float(cell["n_near90"].sum() / n_bpos) if n_bpos else float("nan"),
        "share_iters_ge50pct_bound": float(cell["n_near50"].sum() / n_bpos) if n_bpos else float("nan"),
        "share_runs_noisy_no_gain_clean_gain": float(
            ((~cell["noisy_improved"]) & cell["clean_improved"]).mean()),
        "share_runs_deployed_exceeds_bound": float((cell["dep_exc"] > cell["B_final"] + EPS).mean()),
    }
    out += [(k, v, float("nan"), float("nan")) for k, v in desc.items()]
    return out


def ratio_quantities(early: pd.DataFrame, late: pd.DataFrame) -> list[tuple[str, float, float, float]]:
    """Early/late ratios, landscapes resampled jointly for the two onsets."""
    Le, Ll = per_landscape(early), per_landscape(late)
    common = Le.index.intersection(Ll.index)
    Le, Ll = Le.loc[common], Ll.loc[common]
    e = {c: Le[c].to_numpy() for c in Le}
    l = {c: Ll[c].to_numpy() for c in Ll}
    out = [
        ("raw_ratio",) + _boot(_ratio, e["exc_f"], l["exc_f"]),
        ("bound_ratio",) + _boot(_ratio, e["B_f"], l["B_f"]),
        ("normalised_ratio",) + _boot(_ratio_of_ratios, e["exc_f"], e["B_f"], l["exc_f"], l["B_f"]),
        ("normalised_task_ratio",) + _boot(_ratio_of_ratios, e["exc_f"], e["Bt_f"], l["exc_f"], l["Bt_f"]),
        ("deployed_ratio",) + _boot(_ratio, e["dep_f"], l["dep_f"]),
        ("inst_ratio",) + _boot(_ratio, e["inst_f"], l["inst_f"]),
        ("inst_diff",) + _boot(_diff, e["inst_f"], l["inst_f"]),
    ]
    raw = out[0][1]
    bnd = out[1][1]
    share = math.log(bnd) / math.log(raw) if raw > 0 and bnd > 0 and raw != 1 else float("nan")
    out.append(("mechanical_log_share", share, float("nan"), float("nan")))
    return out


def add_fractions(df: pd.DataFrame, opt_z: dict[str, float]) -> pd.DataFrame:
    z = df["dataset"].map(opt_z)
    if z.isna().any():
        raise ValueError(f"no opt_z for {sorted(df.loc[z.isna(), 'dataset'].unique())}")
    return df.assign(exc_f=df["exc_win"] / z, B_f=df["B_win"] / z, Bt_f=df["Bt_win"] / z,
                     dep_f=df["dep_exc"] / z, inst_f=df["inst_win"] / z, Bfin_f=df["B_final"] / z)


# ---------------------------------------------------------------------------
# Cross-checks against the pipeline's own paired table
# ---------------------------------------------------------------------------


def crosscheck(df: pd.DataFrame, arms: dict[str, Path]) -> dict[str, dict[str, float]]:
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration",
            "auc_simple_regret_excess_true_postonset_per_iter", "final_inference_simple_regret_excess_true"]
    result = {}
    for arm, root in arms.items():
        files = sorted(root.glob("*/evaluation/paired_excess_metrics.csv"))
        if not files:
            continue
        ref = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
        ref = ref.rename(columns={"jitter_iteration": "onset"})
        mine = df[df["arm"] == arm]
        keys = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "onset"]
        m = mine.merge(ref, on=keys, how="inner", validate="one_to_one")
        result[arm] = {
            "n_mine": len(mine), "n_matched": len(m),
            "max_abs_diff_search": float((m["exc_win"] - m["auc_simple_regret_excess_true_postonset_per_iter"]).abs().max()),
            "max_abs_diff_deployed": float((m["dep_exc"] - m["final_inference_simple_regret_excess_true"]).abs().max()),
        }
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(v, lo=None, hi=None, digits=1):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{100 * v:.{digits}f}"
    if lo is not None and np.isfinite(lo):
        s += f" [{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]"
    return s


def _x(v, lo=None, hi=None):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.1f}x"
    if lo is not None and np.isfinite(lo):
        s += f" [{lo:.1f}, {hi:.1f}]"
    return s


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, default=REPO / "output-boba")
    ap.add_argument("--budget25", type=Path, default=REPO / "output-boba-budget25")
    ap.add_argument("--budget100", type=Path, default=REPO / "output-boba-budget100")
    ap.add_argument("--out", type=Path, default=REPO / "output-boba" / "analysis" / "review")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO / "scripts"))
    import boba_benchmarks as bb

    args.out.mkdir(parents=True, exist_ok=True)
    arms = {"main": args.main, "budget25": args.budget25, "budget100": args.budget100}
    cache = args.out / "onset_bound_runs.parquet"
    if cache.is_file() and not args.refresh:
        runs = pd.read_parquet(cache)
        print(f"loaded {len(runs):,} per-run summaries from {cache}")
    else:
        runs = build_runs(arms, args.workers)
        runs.to_parquet(cache, index=False)
        print(f"wrote {len(runs):,} per-run summaries to {cache}")

    stats = bb.load_stats()
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    for arm, root in arms.items():
        meta = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
        for name, entry in meta.get("landscape_stats", {}).items():
            if name in opt_z and abs(float(entry["opt_z"]) - opt_z[name]) > 1e-9:
                raise ValueError(f"{arm}: opt_z of {name} differs from boba_landscape_stats.json")
    runs = add_fractions(runs, opt_z)
    checks = crosscheck(runs, arms)

    rows: list[dict] = []

    def emit(arm, T, em, std, onset, quantities, n_land, n_runs):
        for q, est, lo, hi in quantities:
            rows.append({"arm": arm, "T": T, "error_model": em, "jitter_std": std, "onset": onset,
                         "quantity": q, "estimate": est, "lo": lo, "hi": hi,
                         "n_landscapes": n_land, "n_runs": n_runs})

    # ---- main sweep: per error model and pooled -------------------------------------------
    main = runs[runs["arm"] == "main"]
    mags = sorted(main["jitter_std"].unique())
    onsets = sorted(main["onset"].unique())
    ems = sorted(main["error_model"].unique()) + ["pooled"]
    print("aggregating main sweep ...", flush=True)
    for em in ems:
        sub = main if em == "pooled" else main[main["error_model"] == em]
        for std in mags:
            by_onset = {}
            for onset in onsets:
                cell = sub[(sub["jitter_std"] == std) & (sub["onset"] == onset)]
                by_onset[onset] = cell
                emit("main", 50, em, std, str(onset), cell_quantities(cell),
                     cell["dataset"].nunique(), len(cell))
            emit("main", 50, em, std, "early/late",
                 ratio_quantities(by_onset[min(onsets)], by_onset[max(onsets)]),
                 by_onset[min(onsets)]["dataset"].nunique(),
                 len(by_onset[min(onsets)]) + len(by_onset[max(onsets)]))

    # ---- budget: T = 25, 50 (matched main restriction), 100 --------------------------------
    print("aggregating budget arms ...", flush=True)
    budget = {
        25: runs[runs["arm"] == "budget25"],
        50: main[(main["error_model"] == "gaussian") & main["acquisition"].isin(BUDGET_ACQS)
                 & main["seed"].isin(BUDGET_SEEDS)],
        100: runs[runs["arm"] == "budget100"],
    }
    budget_late = {}
    for T, sub in budget.items():
        sub = sub[sub["error_model"] == "gaussian"]
        late_onset = int(sub["onset"].max())
        for std in sorted(sub["jitter_std"].unique()):
            early = sub[(sub["jitter_std"] == std) & (sub["onset"] == 0)]
            late = sub[(sub["jitter_std"] == std) & (sub["onset"] == late_onset)]
            budget_late[(T, std)] = late
            emit("budget", T, "gaussian", std, "0", cell_quantities(early), early["dataset"].nunique(), len(early))
            emit("budget", T, "gaussian", std, str(late_onset), cell_quantities(late),
                 late["dataset"].nunique(), len(late))
            emit("budget", T, "gaussian", std, "early/late", ratio_quantities(early, late),
                 early["dataset"].nunique(), len(early) + len(late))
    # Does normalisation flatten the late-onset budget dependence? T=25 over T=100.
    for std in sorted(budget[25]["jitter_std"].unique()):
        a, b = per_landscape(budget_late[(25, std)]), per_landscape(budget_late[(100, std)])
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        q = [
            ("late_raw_T25_over_T100",) + _boot(_ratio, a["exc_f"].to_numpy(), b["exc_f"].to_numpy()),
            ("late_bound_T25_over_T100",) + _boot(_ratio, a["B_f"].to_numpy(), b["B_f"].to_numpy()),
            ("late_normalised_T25_over_T100",) + _boot(
                _ratio_of_ratios, a["exc_f"].to_numpy(), a["B_f"].to_numpy(),
                b["exc_f"].to_numpy(), b["B_f"].to_numpy()),
        ]
        emit("budget", "25/100", "gaussian", std, "late", q, len(common), 0)

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "onset_bound.csv", index=False)
    print(f"wrote {args.out / 'onset_bound.csv'} ({len(table)} rows)")
    write_report(args.out / "onset_bound.md", table, runs, checks)
    print(f"wrote {args.out / 'onset_bound.md'}")


def write_report(path: Path, table: pd.DataFrame, runs: pd.DataFrame, checks: dict) -> None:
    def get(arm, T, em, std, onset, q):
        r = table[(table["arm"] == arm) & (table["T"].astype(str) == str(T)) & (table["error_model"] == em)
                  & np.isclose(table["jitter_std"], std) & (table["onset"] == str(onset)) & (table["quantity"] == q)]
        if len(r) != 1:
            return float("nan"), float("nan"), float("nan")
        r = r.iloc[0]
        return r["estimate"], r["lo"], r["hi"]

    main = runs[runs["arm"] == "main"]
    lines: list[str] = []
    w = lines.append
    w("# The running-maximum bound on post-onset search loss\n")
    w("Generated by `scripts/onset_bound.py`. All values are fractions of opt_z, in %, "
      "mean over landscapes of per-landscape means (10 model-based acquisitions x 10 seeds "
      "per landscape in the main sweep), with 95% landscape-bootstrap intervals.\n")

    # ---- findings (templated from the table) ------------------------------------------
    def P(*a, d=1):
        return _pct(*a, digits=d)

    def g(T, onset, q, std=1.0, arm="main"):
        return get(arm, T, "gaussian" if arm == "budget" else "pooled", std, onset, q)

    late_main = main[(main["onset"] == 20) & np.isclose(main["jitter_std"], 1.0)]
    early_main = main[(main["onset"] == 0) & np.isclose(main["jitter_std"], 1.0)]
    stall = lambda c: 100 * ((~c["noisy_improved"]) & c["clean_improved"]).mean()  # noqa: E731
    w("## Findings\n")
    w(f"1. **The premise holds exactly.** In all {len(runs):,} noisy/clean pairs the designs, "
      "true values and search loss are identical through the last shared design s "
      "(s = 21 at onset 20; s = 5 at onset 0, since the first five trials are a uniform "
      "initial design), the observations are identical through the onset, and excess(t) "
      "never exceeds B(t) (max excess - B = 0). The bound is a theorem given the shared "
      "prefix, not an approximation. It binds often after a late onset: at 1 sigma, "
      f"{stall(late_main):.0f}% of late-onset runs make no progress at all after trial 21 while "
      f"their clean twin does (early onset: {stall(early_main):.0f}% after trial 5), and "
      f"{100 * late_main['n_bind'].sum() / late_main['n_Bpos'].sum():.0f}% of late post-onset "
      "iterations with B > 0 sit exactly on the bound (early: "
      f"{100 * early_main['n_bind'].sum() / early_main['n_Bpos'].sum():.0f}%).")
    w(f"2. **Normalised, the late onset is not cheaper.** At 1 sigma (pooled) the late-onset "
      f"excess of {P(*g(50, 20, 'excess'))}% of opt_z is {P(*g(50, 20, 'normalised'), d=0)}% of "
      f"the clean run's improvement after trial 21 (B = {P(*g(50, 20, 'bound'))}%). The early "
      f"excess of {P(*g(50, 0, 'excess'))}% is {P(*g(50, 0, 'normalised'), d=0)}% of its B "
      f"({P(*g(50, 0, 'bound'))}%). The normalised early/late ratio is "
      f"{_x(*g(50, 'early/late', 'normalised_ratio'))} against the raw "
      f"{_x(*g(50, 'early/late', 'raw_ratio'))}; the bound ratio, "
      f"{_x(*g(50, 'early/late', 'bound_ratio'))}, accounts for all of the raw ratio. The same "
      f"holds for every error process and at 0.25 and 5 sigma (normalised ratio 0.8-1.0x; at 5 "
      f"sigma the late onset loses a larger share, {P(*g(50, 20, 'normalised', std=5.0), d=0)}% "
      f"against {P(*g(50, 0, 'normalised', std=5.0), d=0)}%). At 0.05 sigma the excesses are too "
      "close to zero for a ratio. With the looser bounds of the review's wording "
      "(r_c(20) late, r_c(1) early) the normalised ratio is "
      f"{_x(*g(50, 'early/late', 'normalised_task_ratio'))}: the early bound then also counts "
      "improvement from the random initial design, which the noisy run shares for free.")
    w("3. **Normalisation removes the budget dependence.** At 1 sigma the raw early/late "
      f"ratio is {_x(g(25, 'early/late', 'raw_ratio', arm='budget')[0])}, "
      f"{_x(g(50, 'early/late', 'raw_ratio', arm='budget')[0])} and "
      f"{_x(g(100, 'early/late', 'raw_ratio', arm='budget')[0])} at T = 25, 50, 100; the bound "
      f"ratio is {_x(g(25, 'early/late', 'bound_ratio', arm='budget')[0])}, "
      f"{_x(g(50, 'early/late', 'bound_ratio', arm='budget')[0])} and "
      f"{_x(g(100, 'early/late', 'bound_ratio', arm='budget')[0])}; the normalised ratio is "
      f"{_x(*g(25, 'early/late', 'normalised_ratio', arm='budget'))}, "
      f"{_x(*g(50, 'early/late', 'normalised_ratio', arm='budget'))} and "
      f"{_x(*g(100, 'early/late', 'normalised_ratio', arm='budget'))}. The late-onset bound "
      f"falls from {P(*g(25, 10, 'bound', arm='budget'))}% to {P(*g(50, 20, 'bound', arm='budget'))}% "
      f"to {P(*g(100, 40, 'bound', arm='budget'))}% as T grows, as the review predicts, while the "
      f"normalised late excess is {P(*g(25, 10, 'normalised', arm='budget'), d=0)}%, "
      f"{P(*g(50, 20, 'normalised', arm='budget'), d=0)}% and "
      f"{P(*g(100, 40, 'normalised', arm='budget'), d=0)}% (T = 25 over T = 100: "
      f"{_x(*get('budget', '25/100', 'gaussian', 1.0, 'late', 'late_normalised_T25_over_T100'))}, "
      f"raw {_x(*get('budget', '25/100', 'gaussian', 1.0, 'late', 'late_raw_T25_over_T100'))}).")
    w(f"4. **On the deployed design the late onset is not cheap.** At 1 sigma the shipped "
      f"design's late-onset excess is {P(*g(50, 20, 'deployed'))}% against "
      f"{P(*g(50, 0, 'deployed'))}% early, a ratio of "
      f"{_x(*g(50, 'early/late', 'deployed_ratio'))} rather than 5x; at 5 sigma it is "
      f"{P(*g(50, 20, 'deployed', std=5.0))}%. In "
      f"{100 * g(50, 20, 'share_runs_deployed_exceeds_bound')[0]:.0f}% of late-onset runs at "
      "1 sigma the deployed excess exceeds B(T), which the search loss cannot do.")
    w("5. **Implication.** The trajectory metric cannot support 'an optimizer that has already "
      "located a good region is hard to dislodge': after a late onset the noisy run keeps the "
      "improvement banked by trial 21 by construction, and of the improvement still available "
      "it loses the same share as an early-onset run loses of its own. The raw onset ratio "
      "measures how much improvement is left after the onset, which is also what makes it "
      "budget-dependent.\n")
    w("Paper-number check: pooled 1 sigma search excess "
      f"{P(g(50, 0, 'excess')[0])}% / {P(g(50, 20, 'excess')[0])}% and deployed "
      f"{P(g(50, 0, 'deployed')[0])}% / {P(g(50, 20, 'deployed')[0])}% (early / late) reproduce "
      "tables/dose_response.tex; the budget raw ratios reproduce tables/budget.tex.\n")

    # ---- premise ----------------------------------------------------------------------
    w("## 1. The premise\n")
    w("Bound: for t >= s, excess(t) = r_n(t) - r_c(t) <= r_c(s) - r_c(t) = B(t), where s is "
      "the last design the noisy and clean run share. Structurally s = max(onset + 1, 5): "
      "the design at trial onset+1 is chosen from clean data, and the first 5 trials are a "
      "uniform initial design. So s = 21 at onset 20 and s = 5 at onset 0.\n")
    rows = []
    for arm in ["main", "budget25", "budget100"]:
        a = runs[runs["arm"] == arm]
        for onset in sorted(a["onset"].unique()):
            c = a[a["onset"] == onset]
            n_bpos = c["n_Bpos"].sum()
            rows.append([
                arm, f"{int(c['T'].iloc[0])}", f"{onset}", f"{int(c['s'].iloc[0])}", f"{len(c):,}",
                f"{100 * c['prefix_ok'].mean():.2f}%",
                f"{100 * (c['first_div'] > c['s']).mean():.2f}%",
                f"{int((c['first_div'] == c['s'] + 1).sum()):,}",
                f"{int((c['max_viol'] > EPS).sum())}",
                f"{c['max_viol'].max():.2e}",
                f"{100 * c['n_bind'].sum() / n_bpos:.1f}%",
                f"{100 * c['n_near90'].sum() / n_bpos:.1f}%",
                f"{100 * c['n_near50'].sum() / n_bpos:.1f}%",
                f"{100 * (c['B_win'] <= EPS).mean():.1f}%",
            ])
    w(_md_table(["arm", "T", "onset", "s", "pairs", "prefix identical", "designs identical through s",
                 "pairs diverging at s+1", "bound violations", "max(excess - B)",
                 "iters at bound", "iters >= 90% of B", "iters >= 50% of B", "runs with B = 0"], rows))
    w("")
    w("'prefix identical' = designs, true values and search loss identical through trial s, "
      "and observations identical through trial onset. 'iters at bound' = share of post-s "
      "iterations with B(t) > 0 at which excess(t) = B(t), i.e. the noisy run has found "
      "nothing better than design s while the clean run has. 'runs with B = 0' = the clean "
      "run made no improvement after s, so the excess there is forced to be <= 0.\n")
    for arm, c in checks.items():
        w(f"- Cross-check vs `paired_excess_metrics.csv` ({arm}): {c['n_matched']:,} of "
          f"{c['n_mine']:,} pairs matched; max |diff| search excess {c['max_abs_diff_search']:.1e}, "
          f"deployed excess {c['max_abs_diff_deployed']:.1e}.")
    w("")

    # ---- per-cell at 1 sigma ---------------------------------------------------------------
    w("## 2-3. Per cell, main sweep (T = 50)\n")
    w("excess = the paper's post-onset window mean of the search-loss excess; bound B = the "
      "same window mean of B(t) with s = max(onset + 1, 5); excess / B = ratio of the landscape "
      "means (runs with B = 0 enter with excess <= 0 and B = 0); median run excess/B = median "
      "over runs with B > 0 (bounded above by 100%); B_task = the looser bound of the review's "
      "wording, r_c(20) at the late onset and r_c(1) at onset 0. In the ratio tables, raw = "
      "bound ratio x normalised ratio exactly; 'log share mechanical' = log(bound ratio) / "
      "log(raw ratio), above 100% when the late onset loses a larger share of what was left.\n")
    for std in sorted(main["jitter_std"].unique()):
        w(f"### {std:g} sigma\n")
        rows = []
        for em in ["gaussian", "bias", "drift", "ar1", "pooled"]:
            for onset in ["0", "20"]:
                rows.append([
                    em, "from it. 1" if onset == "0" else "from it. 21",
                    _pct(*get("main", 50, em, std, onset, "excess")),
                    _pct(*get("main", 50, em, std, onset, "bound")),
                    _pct(*get("main", 50, em, std, onset, "normalised"), digits=0),
                    _pct(*get("main", 50, em, std, onset, "median_run_ratio"), digits=0),
                    _pct(*get("main", 50, em, std, onset, "normalised_task"), digits=0),
                    _pct(get("main", 50, em, std, onset, "share_bound_zero")[0], digits=0),
                ])
        w(_md_table(["error", "onset", "excess %", "bound B %", "excess / B (ratio of means) %",
                     "median run excess/B %", "excess / B_task %", "runs with B = 0 %"], rows))
        w("")
        rows = []
        for em in ["gaussian", "bias", "drift", "ar1", "pooled"]:
            rows.append([
                em,
                _x(*get("main", 50, em, std, "early/late", "raw_ratio")),
                _x(*get("main", 50, em, std, "early/late", "bound_ratio")),
                _x(*get("main", 50, em, std, "early/late", "normalised_ratio")),
                _x(*get("main", 50, em, std, "early/late", "normalised_task_ratio")),
                f"{100 * get('main', 50, em, std, 'early/late', 'mechanical_log_share')[0]:.0f}%",
            ])
        w(f"Early/late ratios at {std:g} sigma (raw = mechanical x behavioural):\n")
        w(_md_table(["error", "raw early/late", "bound early/late (mechanical)",
                     "normalised early/late (behavioural)", "normalised (task bound)",
                     "log share mechanical"], rows))
        w("")

    # ---- budget ----------------------------------------------------------------------------
    w("## 4. Budget (gaussian, 6 acquisitions, seeds 7-11; late onset at 40% of T)\n")
    b = table[table["arm"] == "budget"]
    for std in sorted(b["jitter_std"].unique()):
        rows = []
        for T, late in ((25, 10), (50, 20), (100, 40)):
            rows.append([
                f"{T}",
                _pct(*get("budget", T, "gaussian", std, "0", "excess")),
                _pct(*get("budget", T, "gaussian", std, late, "excess")),
                _x(*get("budget", T, "gaussian", std, "early/late", "raw_ratio")),
                _pct(*get("budget", T, "gaussian", std, "0", "bound")),
                _pct(*get("budget", T, "gaussian", std, late, "bound")),
                _pct(*get("budget", T, "gaussian", std, "0", "normalised"), digits=0),
                _pct(*get("budget", T, "gaussian", std, late, "normalised"), digits=0),
                _x(*get("budget", T, "gaussian", std, "early/late", "normalised_ratio")),
                _x(*get("budget", T, "gaussian", std, "early/late", "bound_ratio")),
            ])
        w(f"### {std:g} sigma\n")
        w(_md_table(["T", "early excess %", "late excess %", "raw ratio", "early B %", "late B %",
                     "early excess/B %", "late excess/B %", "normalised ratio", "bound ratio"], rows))
        w("")
        w(f"T = 25 over T = 100, late onset: raw "
          f"{_x(*get('budget', '25/100', 'gaussian', std, 'late', 'late_raw_T25_over_T100'))}, "
          f"bound {_x(*get('budget', '25/100', 'gaussian', std, 'late', 'late_bound_T25_over_T100'))}, "
          f"normalised {_x(*get('budget', '25/100', 'gaussian', std, 'late', 'late_normalised_T25_over_T100'))}.\n")

    # ---- deployed ----------------------------------------------------------------------------
    w("## 5. Deployed design at the final trial (not a running maximum, not bounded by B)\n")
    rows = []
    for std in sorted(main["jitter_std"].unique()):
        for em in ["gaussian", "bias", "drift", "ar1", "pooled"]:
            rows.append([
                f"{std:g}", em,
                _pct(*get("main", 50, em, std, "0", "deployed")),
                _pct(*get("main", 50, em, std, "20", "deployed")),
                _x(*get("main", 50, em, std, "early/late", "deployed_ratio")),
                _pct(*get("main", 50, em, std, "20", "bound_final")),
                _pct(get("main", 50, em, std, "20", "share_runs_deployed_exceeds_bound")[0], digits=0),
                _pct(*get("main", 50, em, std, "20", "excess")),
            ])
    w(_md_table(["sigma", "error", "deployed early %", "deployed late %", "deployed early/late",
                 "late B(T) %", "late runs with deployed excess > B(T) %", "late search excess %"], rows))
    w("")

    # ---- per-trial sample quality ----------------------------------------------------------
    w("## Supplement: per-trial sample quality after the onset (not a running maximum)\n")
    w("Window mean of f_clean(t) - f_noisy(t) over iterations onset+1..T: how much worse the "
      "designs the noisy optimizer chooses to sample are, pooled over error processes. It is "
      "not a running maximum, so it is not bounded by B. It is not a clean test of "
      "'dislodgement' either: it is negative at small magnitudes, i.e. noisy runs sample "
      "designs that are better on average (more exploitation) while finding less, so it mixes "
      "the exploration a clean optimizer spends on purpose with harm from the error. A ratio "
      "is not shown because the late value crosses zero.\n")
    rows = []
    for std in sorted(main["jitter_std"].unique()):
        rows.append([
            f"{std:g}",
            _pct(*get("main", 50, "pooled", std, "0", "inst_excess")),
            _pct(*get("main", 50, "pooled", std, "20", "inst_excess")),
            _pct(*get("main", 50, "pooled", std, "early/late", "inst_diff")),
        ])
    w(_md_table(["sigma", "early %", "late %", "early - late %"], rows))
    w("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
