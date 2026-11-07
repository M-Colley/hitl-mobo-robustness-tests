"""Three 95% intervals for every headline number: landscape bootstrap, t, and landscape x seed bootstrap.

Three mock TMLR reviewers objected to the paper's intervals. A 95% percentile bootstrap over 20 landscape clusters
covers only 92-93% (the paper's own simulation, scripts/bootstrap_coverage.py), and it ignores the seed, a second
clustering factor crossed with the landscape: the landscapes share their seeds, and with them their noise streams,
by design (AGENTS.md), and the seed halves 7-11 and 12-16 differ by about 7 points of recovery. For each headline
quantity this script recomputes the point estimate from the tracked per-run tables, through the producer's own
loaders where the producer exposes them, and prints three 95% intervals side by side:

  percentile  the paper's interval: 2000 resamples of the 20 landscapes, the 2.5% and 97.5% quantiles of the
              replicate ratios of landscape means (or landscape means), bootstrap_coverage.intervals.
  t           estimate +/- t(19, 0.975) x the delta-method (linearisation) standard error of a ratio of landscape
              means, or the standard error of a mean (bootstrap_coverage.intervals' 't').
  two-way     landscapes and seeds resampled independently, the pattern of sitting_by_magnitude.two_way_ci extended
              to ratios of means: each replicate draws 20 landscapes and as many seeds as the scope holds, with
              replacement, averages each drawn landscape's per-(landscape, seed) cell means over the drawn seeds,
              and forms the ratio of the landscape means (or their mean). 2000 resamples. The landscape draws are
              the percentile interval's own, so the two intervals differ only by the seed draws.

Each conclusion is whether the interval excludes 0 and, for a share, whether it excludes 1 (100%); a verdict that
differs from the percentile interval's is named. A changed verdict whose deciding bootstrap endpoint lies within
two Monte Carlo SEs of the threshold is marked '(MC)'.

The run stops unless every point estimate matches its producer's published value to four decimals and the paper's
printed rounding, and every percentile interval (and the two published two-way intervals, sitting_by_magnitude's
test_two_way_lo/hi) matches the published one to within four Monte Carlo SEs of the difference of the endpoints.
Producers: bootstrap_coverage.py (a; the 20,000-resample interval), decompose_regret.py (b),
sitting_by_magnitude.py (c, d; its selection JSON), analyse_ship_rules.py (e), replay_hitl_remedies.py (f).

Diagnostic, not an interval: the seed main effect. The per-(landscape, seed) cell values are linearised (for a
ratio R, (num - R den) / mean(den)) and split into landscape, seed and residual mean squares. 'seed p' is a
permutation test of the seed main effect: each landscape's seed labels are permuted independently (2000
permutations), which keeps every landscape's values and breaks only the alignment of seeds across landscapes,
the dependence a shared noise stream creates; it is unadjusted. 'crossed SE ratio' is the moment estimate of the
standard error under a crossed random-effects model, sqrt((MS_L + MS_S - MS_E) / MS_L), relative to the
landscape-only SE behind the t interval. The pigeonhole two-way bootstrap is not an unbiased estimate of that
variance: for the mean of an R x C table its variance is sum(a_l^2)/R^2 + sum(b_s^2)/C^2 + sum(r_ls^2)/(RC)^2
(landscape effects a, seed effects b, landscape x seed residual r), whose expectation counts the residual's
variance (3RC - 2R - 2C + 1)/(RC) times, 2.5 at 20 landscapes and 5 seeds and 2.7 at 10 seeds, and shrinks the
seed term by (C - 1)/C (McCullagh 2000, Bernoulli 6:285; Owen 2007, Ann. Appl. Stat. 1:386). It widens even
without a seed effect; the seed p and the crossed SE ratio say how much of its extra width a seed effect explains.

heldout_remedies.py and replay_hitl_remedies.py import the simulator and torch at module level; their constants
(PAPER_RANK_RULE, recovery_table's resample count) are read from their source text instead of imported.

    python scripts/checks/headline_intervals.py
writes output-boba/analysis/review/headline_intervals.csv and
output-boba/analysis/review/register_checks/headline_intervals.txt (exactly what it prints).
"""
from __future__ import annotations

import os

# Indexing and reductions only; a threaded BLAS buys nothing and other jobs share the machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyse_boba_adaptations as aba  # noqa: E402
import analyse_ship_rules as asr  # noqa: E402
import bootstrap_coverage as bc  # noqa: E402
import decompose_regret as dr  # noqa: E402
import sitting_by_magnitude as sbm  # noqa: E402

ALPHA = bc.ALPHA
BOOT = 2000
PERMS = 2000
SEED = 20261107
POINT_TOL = 5e-5          # "matches to four decimals"
MC_TOL = 4.0              # Monte Carlo SEs allowed between an endpoint and its published value
MC_FLAG = 2.0             # a changed verdict this close (in MC SEs) to its threshold is marked (MC)
MIN_KEPT = 0.95           # as analyse_boba_adaptations.summarise: refuse a ratio interval if > 5% of draws lack one

ANALYSIS = REPO / "output-boba" / "analysis"
SPIKE = REPO / "output-boba-spike" / "analysis"
STATS = REPO / "boba_landscape_stats.json"
OUT_CSV = ANALYSIS / "review" / "headline_intervals.csv"
OUT_TXT = ANALYSIS / "review" / "register_checks" / "headline_intervals.txt"

PROCESSES = sbm.PROCESSES
ALL_SEEDS = sbm.TRAIN_SEEDS + sbm.TEST_SEEDS
KINDS = ("share", "loss", "gain", "price")


# ---------------------------------------------------------------------------
# Constants of scripts that must not be imported here
# ---------------------------------------------------------------------------


def source_constant(path: Path, name: str):
    """A module-level literal of a script, read from its source text without importing the script."""
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} is not a module-level literal of {path}")


def source_default(path: Path, func: str, arg: str):
    """The literal default of argument ``arg`` of function ``func`` in a script's source text."""
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            pos = node.args.args
            defaults = dict(zip([a.arg for a in pos[len(pos) - len(node.args.defaults):]], node.args.defaults))
            if arg in defaults:
                return ast.literal_eval(defaults[arg])
    raise KeyError(f"no literal default for {func}({arg}=...) in {path}")


# ---------------------------------------------------------------------------
# A headline quantity
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Quantity:
    key: str
    label: str
    scope: str
    kind: str                       # share (a ratio, in %), loss (a mean, in % of opt_z), gain or price (opt_z units)
    runs: pd.DataFrame              # one row per run: dataset, seed, num and, for a share, den
    seeds: tuple[int, ...]
    published: float
    published_lo: float = np.nan
    published_hi: float = np.nan
    published_boot: int = 0
    source: str = ""
    paper: str = ""                 # as the paper prints it
    paper_value: float = np.nan     # the printed point estimate, as a fraction (shares, losses) or in opt_z units
    paper_digits: int = 3           # its decimals in those units
    published_two_way: tuple[float, float] | None = None

    @property
    def is_ratio(self) -> bool:
        return self.kind == "share"

    @property
    def thresholds(self) -> tuple[float, ...]:
        return (0.0, 1.0) if self.is_ratio else (0.0,)


def frame(df: pd.DataFrame, num, den=None) -> pd.DataFrame:
    """The per-run frame of a quantity: dataset, seed, num[, den]."""
    out = pd.DataFrame({"dataset": df["dataset"].to_numpy(), "seed": df["seed"].astype(int).to_numpy(),
                        "num": np.asarray(num, dtype=float)})
    if den is not None:
        out["den"] = np.asarray(den, dtype=float)
    return out


# ---------------------------------------------------------------------------
# Estimands
# ---------------------------------------------------------------------------


def landscape_estimate(runs: pd.DataFrame) -> float:
    """Ratio of landscape means (mean over landscapes of each landscape's mean over its runs), or a landscape mean."""
    if "den" in runs:
        per = runs.groupby("dataset")[["num", "den"]].mean()
        return float(per["num"].mean() / per["den"].mean())
    return float(runs.groupby("dataset")["num"].mean().mean())


def landscape_seed_tables(runs: pd.DataFrame, seeds) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Per-(landscape, seed) cell means of num (and den), landscapes sorted by name, seeds ascending.

    The table must be complete, and every seed of a landscape must hold the same number of runs, so that a
    landscape's mean over its seed cells is its mean over its runs (the producers' landscape mean); either
    failure raises rather than reweighting the runs.
    """
    seeds = sorted(int(s) for s in seeds)
    present = sorted(int(s) for s in runs["seed"].unique())
    if present != seeds:
        raise ValueError(f"the runs hold seeds {present}, the scope names {seeds}")
    counts = runs.groupby(["dataset", "seed"]).size().unstack("seed").reindex(columns=seeds)
    if counts.isna().to_numpy().any():
        missing = [(d, s) for d in counts.index for s in seeds if pd.isna(counts.loc[d, s])]
        raise ValueError(f"{len(missing)} (landscape, seed) cells hold no run, e.g. {missing[:3]}")
    uneven = counts.nunique(axis=1) > 1
    if uneven.any():
        raise ValueError(f"unequal run counts across the seeds of {list(counts.index[uneven])[:3]}: a mean over "
                         "seed cells would not be the landscape's mean over its runs")
    num = runs.groupby(["dataset", "seed"])["num"].mean().unstack("seed").reindex(columns=seeds)
    den = None
    if "den" in runs:
        den = runs.groupby(["dataset", "seed"])["den"].mean().unstack("seed").reindex(columns=seeds).to_numpy(float)
    return num.to_numpy(float), den, list(num.index)


# ---------------------------------------------------------------------------
# Interval machinery
# ---------------------------------------------------------------------------


def draw_indices(rng: np.random.Generator, n_land: int, n_seed: int, boot: int = BOOT
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Landscape draws (boot, n_land) and, after them, seed draws (boot, n_seed), all with replacement."""
    li = rng.integers(0, n_land, (boot, n_land))
    si = rng.integers(0, n_seed, (boot, n_seed))
    return li, si


def two_way_draws(num: np.ndarray, den: np.ndarray | None, li: np.ndarray, si: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Replicate estimates of the landscape x seed bootstrap and which of them are defined.

    num, den  (n_land, n_seed) per-cell means; li (B, n_land) and si (B, n_seed) resampled positions. Each
    replicate averages every drawn landscape's cells over the drawn seeds, then forms the ratio of those landscape
    means (or their mean). A ratio replicate whose mean denominator is not positive is undefined (NaN, ok False).
    """
    rows, cols = li[:, :, None], si[:, None, :]
    num_l = num[rows, cols].mean(axis=2)
    if den is None:
        return num_l.mean(axis=1), np.ones(len(li), dtype=bool)
    den_bar = den[rows, cols].mean(axis=2).mean(axis=1)
    ok = den_bar > 0
    theta = np.full(len(li), np.nan)
    theta[ok] = num_l.mean(axis=1)[ok] / den_bar[ok]
    return theta, ok


def percentile(draws: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """Equal-tailed percentile interval, numpy's linear rule (the producers' np.percentile)."""
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def quantile_mcse(draws: np.ndarray, q: float) -> float:
    """Monte Carlo SE of the q-quantile of ``draws``: half the spread of the order statistics one binomial SD
    either side of it, sqrt(q (1 - q) / B) / density in the limit."""
    x = np.sort(np.asarray(draws, dtype=float))
    b = len(x)
    d = int(np.ceil(np.sqrt(b * q * (1 - q))))
    j = q * (b - 1)
    lo, hi = int(max(0, np.floor(j) - d)), int(min(b - 1, np.ceil(j) + d))
    return float((x[hi] - x[lo]) / 2.0)


def three_intervals(num: np.ndarray, den: np.ndarray | None, li: np.ndarray, si: np.ndarray,
                    alpha: float = ALPHA) -> dict:
    """The percentile, t and two-way intervals of a quantity from its per-(landscape, seed) cell means.

    The landscape means are the row means of the table (equal to the means over runs for a balanced table). The
    percentile and t intervals are bootstrap_coverage.intervals' on those means with the landscape draws ``li``;
    the two-way interval uses the same ``li`` and the seed draws ``si``.
    """
    num_l = num.mean(axis=1)
    den_l = None if den is None else den.mean(axis=1)
    if den_l is not None and (den_l <= 0).any():
        raise ValueError("a landscape has a non-positive denominator: the share is not a share there")
    n = len(num_l)
    res = bc.intervals(num_l[None, :], None if den_l is None else den_l[None, :], li[None, :, :], alpha)
    theta_l, _ = bc.estimate(num_l[li], None if den_l is None else den_l[li])
    pct = (float(res["percentile"][0, 0]), float(res["percentile"][0, 1]))
    if max(abs(a - b) for a, b in zip(pct, percentile(theta_l, alpha))) > 1e-12:
        raise AssertionError("bootstrap_coverage's percentile interval differs from the replicates' quantiles")
    _, se = bc.estimate(num_l[None, :], None if den_l is None else den_l[None, :])
    tw, ok = two_way_draws(num, den, li, si)
    kept = float(ok.mean())
    if kept < MIN_KEPT:
        raise ValueError(f"only {kept:.1%} of the two-way replicates have a positive denominator")
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    return {
        "estimate": float(res["_theta"][0]),
        "percentile": pct,
        "t": (float(res["t"][0, 0]), float(res["t"][0, 1])),
        "two_way": percentile(tw[ok], alpha),
        "se_t": float(se[0] * np.sqrt(n / (n - 1))),
        "pct_mcse": (quantile_mcse(theta_l, lo_q), quantile_mcse(theta_l, hi_q)),
        "two_way_mcse": (quantile_mcse(tw[ok], lo_q), quantile_mcse(tw[ok], hi_q)),
        "two_way_kept": kept,
    }


# ---------------------------------------------------------------------------
# The seed main effect (diagnostic)
# ---------------------------------------------------------------------------


def linearised(num: np.ndarray, den: np.ndarray | None) -> np.ndarray:
    """Per-cell influence values: (num - R den) / mean(den) for a ratio R, num itself for a mean."""
    if den is None:
        return np.asarray(num, dtype=float)
    theta = num.mean() / den.mean()
    return (num - theta * den) / den.mean()


def crossed_anova(u: np.ndarray) -> dict:
    """Two-way (landscape x seed, one value per cell) mean squares and the variances of the grand mean.

    var_landscape  MS_L / (RC), the landscape-cluster variance behind the t interval;
    var_crossed    (MS_L + MS_S - MS_E) / (RC), the moment estimate under crossed random effects;
    var_pigeonhole the exact variance of the grand mean under the two-way (pigeonhole) bootstrap,
                   sum(a^2)/R^2 + sum(b^2)/C^2 + sum(r^2)/(RC)^2.
    """
    r_, c_ = u.shape
    g = u.mean()
    a = u.mean(axis=1) - g
    b = u.mean(axis=0) - g
    resid = u - g - a[:, None] - b[None, :]
    ms_l = c_ * float(a @ a) / (r_ - 1)
    ms_s = r_ * float(b @ b) / (c_ - 1) if c_ > 1 else np.nan
    ms_e = float((resid ** 2).sum()) / ((r_ - 1) * (c_ - 1)) if c_ > 1 else np.nan
    return {"ms_landscape": ms_l, "ms_seed": ms_s, "ms_residual": ms_e,
            "var_landscape": ms_l / (r_ * c_), "var_crossed": (ms_l + ms_s - ms_e) / (r_ * c_),
            "var_pigeonhole": float(a @ a) / r_ ** 2 + float(b @ b) / c_ ** 2 + float((resid ** 2).sum()) / (r_ * c_) ** 2}


def seed_effect_p(u: np.ndarray, rng: np.random.Generator, perms: int = PERMS) -> float:
    """Permutation p of the seed main effect: each landscape's seed labels permuted independently."""
    r_, c_ = u.shape
    if c_ < 2:
        return float("nan")

    def spread(v: np.ndarray) -> np.ndarray:
        col = v.mean(axis=-2)
        return ((col - col.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)

    observed = float(spread(u))
    order = rng.random((perms, r_, c_)).argsort(axis=-1)
    null = spread(np.take_along_axis(u[None, :, :], order, axis=-1))
    return float((1 + np.sum(null >= observed * (1 - 1e-12))) / (1 + perms))


# ---------------------------------------------------------------------------
# Conclusions and checks
# ---------------------------------------------------------------------------


def excludes(lo: float, hi: float, threshold: float) -> bool:
    return bool(lo > threshold or hi < threshold)


def conclusion_changes(iv: dict[str, tuple[float, float]], thresholds, mcse: dict | None = None) -> list[str]:
    """Verdicts (excludes the threshold or not) of the t and two-way intervals that differ from the percentile's.

    ``mcse`` maps a bootstrap method to the Monte Carlo SEs of its two endpoints; a change whose deciding endpoint
    (the reference's or the method's) lies within MC_FLAG of them of the threshold is marked '(MC)'.
    """
    out = []
    ref = iv["percentile"]
    for method, label in (("t", "t"), ("two_way", "two-way")):
        for th in thresholds:
            before, after = excludes(*ref, th), excludes(*iv[method], th)
            if before == after:
                continue
            near = False
            if mcse is not None:
                for m in ("percentile", method):
                    if m in mcse:
                        near |= any(abs(e - th) <= MC_FLAG * s for e, s in zip(iv[m], mcse[m]))
            out.append(f"{label} {'excludes' if after else 'includes'} {th:g}" + (" (MC)" if near else ""))
    return out


def check_point(q: Quantity, estimate: float) -> None:
    """Stop unless the estimate matches the producer's published value to four decimals and the paper's rounding."""
    if not abs(estimate - q.published) < POINT_TOL:
        raise SystemExit(f"{q.key} {q.label}: estimate {estimate:.6f} does not match the published "
                         f"{q.published:.6f} ({q.source}) to four decimals")
    if np.isfinite(q.paper_value) and abs(round(estimate, q.paper_digits) - q.paper_value) > 1e-9:
        raise SystemExit(f"{q.key} {q.label}: estimate {estimate:.6f} does not round to the paper's {q.paper}")


def check_interval(label: str, mine: tuple[float, float], mcse: tuple[float, float], published: tuple[float, float],
                   boot: int, published_boot: int) -> float:
    """Largest endpoint gap in Monte Carlo SEs of the difference; stops beyond MC_TOL."""
    worst = 0.0
    for e, s, p in zip(mine, mcse, published):
        se = s * np.sqrt(1.0 + boot / published_boot)
        z = abs(e - p) / se if se > 0 else (0.0 if e == p else np.inf)
        worst = max(worst, z)
    if worst > MC_TOL:
        raise SystemExit(f"{label}: interval [{mine[0]:.4f}, {mine[1]:.4f}] differs from the published "
                         f"[{published[0]:.4f}, {published[1]:.4f}] by {worst:.1f} Monte Carlo SEs")
    return worst


# ---------------------------------------------------------------------------
# Loaders, one per producer
# ---------------------------------------------------------------------------


def _require(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not p.is_file()]


def _expect(df: pd.DataFrame, what: str, n_land: int = 20, acqs=None, processes=None, seeds=None) -> None:
    if df["dataset"].nunique() != n_land:
        raise ValueError(f"{what}: {df['dataset'].nunique()} landscapes, expected {n_land}")
    if acqs is not None and sorted(df["acquisition"].unique()) != sorted(acqs):
        raise ValueError(f"{what}: acquisitions {sorted(df['acquisition'].unique())}, expected {sorted(acqs)}")
    if processes is not None and sorted(df["error_model"].unique()) != sorted(processes):
        raise ValueError(f"{what}: error processes {sorted(df['error_model'].unique())}, expected {sorted(processes)}")
    if seeds is not None and sorted(int(s) for s in df["seed"].unique()) != sorted(seeds):
        raise ValueError(f"{what}: seeds {sorted(df['seed'].unique())}, expected {sorted(seeds)}")


def quantities_deployed(opt_z: dict[str, float]) -> list[Quantity]:
    """(a) The deployed design's excess over the clean twin / opt_z, 1 sigma from trial 1."""
    files = sorted((REPO / "output-boba").glob("*/evaluation/paired_excess_metrics.csv"))
    published = ANALYSIS / "review" / "bootstrap_coverage.csv"
    if not files or _require([published, ANALYSIS / "cell_means.csv"]):
        raise FileNotFoundError("output-boba/*/evaluation/paired_excess_metrics.csv, review/bootstrap_coverage.csv "
                                "or cell_means.csv")
    col = "final_inference_simple_regret_excess_true"
    use = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", col]
    df = pd.concat([pd.read_csv(f, usecols=use) for f in files], ignore_index=True)
    cell = df[(df["jitter_std"] == 1.0) & (df["jitter_iteration"] == 0) & ~df["acquisition"].isin(bc.MODEL_FREE)]
    _expect(cell, "deployed loss", acqs=aba.TEN.split(","), processes=PROCESSES, seeds=ALL_SEEDS)
    if cell.duplicated(["dataset", "acquisition", "seed", "error_model"]).any():
        raise ValueError("deployed loss: a run appears twice")
    runs = frame(cell, cell[col] / bc.landscape_scale(cell["dataset"], opt_z))
    # The same per-landscape means from cell_means.csv, bootstrap_coverage's independent route.
    other = bc.deployed_cost_inputs(REPO / "output-boba", opt_z)["num"]
    gap = float((runs.groupby("dataset")["num"].mean() - other).abs().max())
    if not gap < 1e-9:
        raise ValueError(f"paired_excess_metrics and cell_means disagree by {gap:g}")
    pub = pd.read_csv(published).set_index(["estimand", "method"]).loc[("deployed_cost", "percentile")]
    return [Quantity(
        "a", "deployed loss, 1 sigma from trial 1",
        "deployed design at the final trial, excess over the clean twin / opt_z; 1 sigma from trial 1; ten "
        "model-based acquisitions; gaussian, bias, drift and ar1 pooled; seeds 7-16",
        "loss", runs, ALL_SEEDS, float(pub["estimate"]), float(pub["actual_lo"]), float(pub["actual_hi"]),
        int(pub["boot_actual"]), "review/bootstrap_coverage.csv (deployed_cost, percentile)",
        "29.7% [26.6, 32.9]", 0.297, 3)]


def quantities_selection(opt_z: dict[str, float], ship: pd.DataFrame) -> list[Quantity]:
    """(b) Selection loss as a share of the deployed excess over the clean twin (decompose_regret.py)."""
    published = ANALYSIS / "regret_decomposition.csv"
    if _require([published]):
        raise FileNotFoundError(str(published))
    table = dr.decompose(ship, opt_z)
    table = table.assign(baseline=table["baseline"].astype(str).str.lower().isin(("true", "1")))
    clean, noisy = table[table["baseline"]], table[~table["baseline"]]
    dr.assert_clean_runs_have_no_selection_loss(clean)
    base = clean.groupby(dr.PAIR_KEYS)[["deployed"]].mean().rename(columns={"deployed": "deployed_clean"})
    pub = pd.read_csv(published, dtype={"jitter_std": str, "jitter_iteration": str})
    pub = pub[pub["error_model"] == "pooled"].assign(
        s=lambda f: pd.to_numeric(f["jitter_std"], errors="coerce"),
        o=lambda f: pd.to_numeric(f["jitter_iteration"], errors="coerce"))
    out = []
    for key, sigma, onset, paper, value in (("b1", 1.0, 0, "41.8% [34.1, 50.4]", 0.418),
                                            ("b2", 5.0, 20, "83.1% [75, 90]", 0.831)):
        cell = noisy[(noisy["jitter_std"] == sigma) & (noisy["jitter_iteration"] == onset)]
        _expect(cell, f"selection share {sigma:g} sigma", acqs=aba.TEN.split(","), processes=PROCESSES,
                seeds=ALL_SEEDS)
        m = cell.merge(base, on=dr.PAIR_KEYS, how="inner", validate="many_to_one")
        if len(m) != len(cell):
            raise ValueError(f"{len(cell) - len(m)} noisy runs lack a clean twin")
        runs = frame(m, m["selection"], m["deployed"] - m["deployed_clean"])
        if key == "b1":
            ref = bc.selection_share_inputs(REPO / "output-boba", opt_z)
            per = runs.groupby("dataset")[["num", "den"]].mean()
            gap = float(max((per["num"] - ref["num"]).abs().max(), (per["den"] - ref["den"]).abs().max()))
            if not gap < 1e-12:
                raise ValueError(f"the selection-share inputs differ from bootstrap_coverage's by {gap:g}")
        row = pub[(pub["s"] == sigma) & (pub["o"] == onset)].iloc[0]
        out.append(Quantity(
            key, f"selection share, {sigma:g} sigma from trial {onset + 1}",
            f"selection loss / deployed excess over the clean twin, final trial; {sigma:g} sigma from trial "
            f"{onset + 1}; ten model-based acquisitions; four processes pooled; seeds 7-16",
            "share", runs, ALL_SEEDS, float(row["excess_selection_share"]), float(row["excess_selection_share_lo"]),
            float(row["excess_selection_share_hi"]), dr.BOOTSTRAP_REPS, "regret_decomposition.csv (pooled row)",
            paper, value, 3))
    return out


def _cell(selection: list[dict], name: str) -> dict:
    hit = [s for s in selection if s.get("cell") == name]
    if len(hit) != 1:
        raise KeyError(f"{name} not found once in the selection JSON")
    return hit[0]


def quantities_end_of_study(ship: pd.DataFrame) -> list[Quantity]:
    """(c, d) The zero-trial rules and the final sitting on the held-out seeds (sitting_by_magnitude.py)."""
    replays = [ANALYSIS / name / "end_of_study_per_run.csv.gz" for name in sbm.REPLAY_DIRS]
    published = ANALYSIS / "review" / "sitting_by_magnitude_selection.json"
    lacking = _require(replays + [published])
    if lacking:
        raise FileNotFoundError(", ".join(str(p.relative_to(REPO)) for p in lacking))
    m = sbm.attach_ship_rules(sbm.load(ANALYSIS, STATS), ship)
    selection = json.loads(published.read_text(encoding="utf-8"))
    prices = _cell(selection, "clean_twin_prices")
    test = sbm.TEST_SEEDS
    src = "review/sitting_by_magnitude_selection.json"
    acqs = list(sbm.ACQS)
    out: list[Quantity] = []

    # (c) LCB1 and LCB2 after all 50 trials, 1 sigma from trial 1, held out.
    s1 = _cell(selection, "1sigma_from_trial_1")
    runs = m[(m["sigma"] == 1.0) & (m["onset"] == 0)].drop_duplicates("file")
    runs = runs[runs["seed"].isin(test)]
    _expect(runs, "LCB rules", acqs=acqs, processes=PROCESSES, seeds=test)
    scope = ("{what}; 1 sigma from trial 1; LogEI and qNEI; four processes pooled; held-out seeds 12-16")
    for rule, gain_txt, share_txt, price_txt in (("lcb1", "+0.027 [+0.006, +0.048]", "10% [2, 18]", "0.021"),
                                                 ("lcb2", "+0.026 [+0.004, +0.046]", "10% [1, 18]", "0.023")):
        pub = s1["ship_rules"][rule]["test"]
        name = rule.upper()
        out.append(Quantity(
            f"c-{rule}-gain", f"{name} gain, 1 sigma from trial 1",
            scope.format(what=f"{name} after 50 trials, gain over the standard process, deployed design / opt_z"),
            "gain", frame(runs, runs[f"gain_{rule}"]), test, pub["gain"], pub["gain_lo"], pub["gain_hi"], sbm.REPS,
            src, gain_txt, float(gain_txt.split()[0]), 3))
        out.append(Quantity(
            f"c-{rule}-share", f"{name} share of the cost, 1 sigma from trial 1",
            scope.format(what=f"{name} gain / the standard process's cost of error, deployed design"),
            "share", frame(runs, runs[f"gain_{rule}"], runs["cost"]), test, pub["share"], pub["share_lo"],
            pub["share_hi"], sbm.REPS, src, share_txt, float(share_txt.split("%")[0]) / 100, 2))
        pp = prices["ship_rules"][rule]["test"]
        out.append(Quantity(
            f"c-{rule}-price", f"{name} price",
            f"{name} on the clean twin, ({name} - standard) / opt_z, deployed design; LogEI and qNEI; seeds 12-16",
            "price", frame(runs, runs[f"price_{rule}"]), test, pp["mean"], pp["lo"], pp["hi"], sbm.REPS,
            src + " (clean_twin_prices)", price_txt, float(price_txt), 3))

    # (d) The final sitting: k = 2 at 1 sigma from trial 1, k = 16 at 5 sigma from trial 21, held out.
    for cell_name, sigma, onset, k, texts in (
            ("1sigma_from_trial_1", 1.0, 0, 2, {"gain": "+0.030 [+0.012, +0.050]", "share": "12% [5, 18]",
                                                "lcb1": "+0.003 [-0.003, +0.009]", "price": "0.018"}),
            ("5sigma_from_trial_21", 5.0, 20, 16, {"gain": "+0.227 [+0.147, +0.321]", "share": "41% [30, 50]",
                                                   "lcb1": "+0.249 [+0.164, +0.339]",
                                                   "lcb2": "+0.218 [+0.146, +0.291]", "price": "0.033"})):
        s = _cell(selection, cell_name)
        if int(s["k_chosen_on_7_11"]) != k:
            raise ValueError(f"{cell_name}: seeds 7-11 chose k = {s['k_chosen_on_7_11']}, not {k}")
        x = m[(m["sigma"] == sigma) & (m["onset"] == onset) & (m["k"] == k) & m["seed"].isin(test)]
        _expect(x, f"sitting k = {k}", acqs=acqs, processes=PROCESSES, seeds=test)
        if x["file"].duplicated().any():
            raise ValueError(f"sitting k = {k}: a run appears twice")
        where = f"{sigma:g} sigma from trial {onset + 1}"
        what = (f"sitting of {k} (tournament, LCB1 candidates, rho 1, best look ships) after {50 - k} trials")
        scope = f"{{what}}; {where}; LogEI and qNEI; four processes pooled; held-out seeds 12-16"
        out.append(Quantity(
            f"d-k{k}-gain", f"sitting of {k} gain, {where}",
            scope.format(what=f"{what}, gain over the standard process, deployed design / opt_z"),
            "gain", frame(x, x["gain"]), test, s["test_gain"], s["test_lo"], s["test_hi"], sbm.REPS, src,
            texts["gain"], float(texts["gain"].split()[0]), 3,
            published_two_way=(s["test_two_way_lo"], s["test_two_way_hi"])))
        out.append(Quantity(
            f"d-k{k}-share", f"sitting of {k} share of the cost, {where}",
            scope.format(what=f"{what}, gain / the standard process's cost of error"),
            "share", frame(x, x["gain"], x["cost"]), test, s["test_recovered_share"], s["test_recovered_lo"],
            s["test_recovered_hi"], sbm.REPS, src, texts["share"], float(texts["share"].split("%")[0]) / 100, 2))
        for rule in ("lcb1", "lcb2"):
            if rule not in texts:
                continue
            inc = s[f"increment_over_{rule}"]
            out.append(Quantity(
                f"d-k{k}-over-{rule}", f"sitting of {k} over {rule.upper()}, {where}",
                scope.format(what=f"{what}, paired increment over {rule.upper()} after 50 trials / opt_z"),
                "gain", frame(x, x[f"inc_{rule}"]), test, inc["test"], inc["test_lo"], inc["test_hi"], sbm.REPS,
                src, texts[rule], float(texts[rule].split()[0]), 3))
        pp = prices["sitting"][str(k)]["test"]
        if abs(pp["mean"] - s["price_test"]) > 1e-12:
            raise ValueError(f"k = {k}: the cell's price differs from clean_twin_prices")
        out.append(Quantity(
            f"d-k{k}-price", f"sitting of {k} price",
            f"sitting of {k} on the clean twin, (sitting - standard) / opt_z, deployed design; LogEI and qNEI; "
            "seeds 12-16",
            "price", frame(x, x["price"]), test, pp["mean"], pp["lo"], pp["hi"], sbm.REPS,
            src + " (clean_twin_prices)", texts["price"], float(texts["price"]), 3))
    return out


def quantities_cautious(opt_z: dict[str, float]) -> list[Quantity]:
    """(e) The cautious rule LCB2 under gaussian error, pooled over 0.25 sigma and above (analyse_ship_rules.py)."""
    per_run = asr.per_run_path(REPO / "output-boba")
    published = ANALYSIS / "ship_rules_recovery.csv"
    lacking = _require([per_run, published])
    if lacking:
        raise FileNotFoundError(", ".join(str(p.relative_to(REPO)) for p in lacking))
    pool_min = asr.parse_args([]).pool_min_std          # the producer's default threshold, 0.25
    runs = asr.load_per_run(per_run)
    ref, trt = asr.twin_frame(runs, asr.STANDARD), asr.twin_frame(runs, "lcb2")

    def gaussian(f: pd.DataFrame) -> pd.DataFrame:
        return f[(f["error_model"] == "gaussian") & (f["variant"] == "")]

    paired = aba.paired_frame(gaussian(ref), gaussian(trt), asr.RESPONSE, opt_z, pool=False)
    block = paired[paired["jitter_std"] >= pool_min - 1e-12]
    _expect(block, "cautious rule", acqs=aba.TEN.split(","), seeds=ALL_SEEDS)
    pub = pd.read_csv(published)
    row = pub[(pub["mode"] == "within") & (pub["arm"] == "output-boba") & (pub["rule"] == "lcb2")
              & (pub["error_model"] == "gaussian") & (pub["cell"] == f"pooled_std_ge_{pool_min:g}")
              & pub["jitter_iteration"].isna()]
    if len(row) != 1:
        raise KeyError("the cautious rule's pooled row is not in ship_rules_recovery.csv once")
    row = row.iloc[0]
    scope = (f"{{what}}; gaussian; {pool_min:g}, 1 and 5 sigma, both onsets pooled; ten model-based acquisitions; "
             "seeds 7-16")
    src = "ship_rules_recovery.csv (lcb2, gaussian, pooled_std_ge_0.25)"
    return [
        Quantity("e-lcb2-recovery", "cautious rule (LCB2) recovery",
                 scope.format(what="LCB2 after 50 trials, gain / the standard process's cost of error, deployed design"),
                 "share", frame(block, block["ref_noisy"] - block["trt_noisy"], block["ref_noisy"] - block["ref_clean"]),
                 ALL_SEEDS, float(row["recovered"]), float(row["recovered_lo"]), float(row["recovered_hi"]),
                 aba.BOOTSTRAP_REPS, src, "10.1% [4.6, 15.9]", 0.101, 3),
        Quantity("e-lcb2-price", "cautious rule (LCB2) price",
                 scope.format(what="LCB2 on the clean twin, (LCB2 - standard) / opt_z, deployed design"),
                 "price", frame(block, block["trt_clean"] - block["ref_clean"]), ALL_SEEDS, float(row["price"]),
                 source=src, paper="0.015", paper_value=0.015, paper_digits=3),
    ]


def quantities_rank_refit(opt_z: dict[str, float]) -> list[Quantity]:
    """(f) The rank refit under gross faults (replay_hitl_remedies.py), seeds 7-11 and held-out seeds 12-16."""
    rule = source_constant(SCRIPTS / "heldout_remedies.py", "PAPER_RANK_RULE")
    reps = source_default(SCRIPTS / "replay_hitl_remedies.py", "recovery_table", "reps")
    out = []
    for name, seeds, rec_txt, price_txt, rec_val in (
            ("hitl_remedies", sbm.TRAIN_SEEDS, "51% [36, 63]", "0.029", 0.51),
            ("hitl_remedies_heldout", sbm.TEST_SEEDS, "49% [34, 63]", "0.024", 0.49)):
        per_run, published = SPIKE / name / "hitl_remedies_per_run.csv.gz", SPIKE / name / "hitl_remedies_recovery.csv"
        lacking = _require([per_run, published])
        if lacking:
            raise FileNotFoundError(", ".join(str(p.relative_to(REPO)) for p in lacking))
        h = pd.read_csv(per_run, low_memory=False)
        g = h[h["procedure"] == rule]
        _expect(g, f"rank refit {name}", acqs=list(sbm.ACQS), processes=["spike"], seeds=seeds)
        cells = g[["variant", "jitter_std", "jitter_iteration"]].drop_duplicates()
        if len(cells) != 1 or tuple(cells.iloc[0]) != ("sp0.15-20", 0.25, 0):
            raise ValueError(f"rank refit {name}: cells {cells.to_dict('records')}, expected sp0.15-20 at 0.25 "
                             "sigma from trial 1")
        if g["file"].duplicated().any():
            raise ValueError(f"rank refit {name}: a run appears twice")
        z = bc.landscape_scale(g["dataset"], opt_z)
        pub = pd.read_csv(published)
        row = pub[(pub["procedure"] == rule) & (pub["error_model"] == "pooled")].iloc[0]
        seeds_txt = "seeds 7-11" if seeds == sbm.TRAIN_SEEDS else "held-out seeds 12-16"
        scope = (f"{{what}}; spike arm, 15% of trials spiking at 20 SD (sp0.15-20), 0.25 sigma from trial 1; "
                 f"LogEI and qNEI; {seeds_txt}")
        tag = "7-11" if seeds == sbm.TRAIN_SEEDS else "12-16"
        src = f"output-boba-spike/analysis/{name}/hitl_remedies_recovery.csv ({rule}, pooled)"
        out.append(Quantity(
            f"f-rank-{tag}", f"rank refit recovery, seeds {tag}",
            scope.format(what=f"rank refit ({rule}), gain / the standard process's cost of error, deployed design"),
            "share", frame(g, (g["ref_noisy"] - g["regret_noisy"]) / z, (g["ref_noisy"] - g["ref_clean"]) / z),
            seeds, float(row["recovered"]), float(row["recovered_lo"]), float(row["recovered_hi"]), int(reps), src,
            rec_txt, rec_val, 2))
        out.append(Quantity(
            f"f-rank-{tag}-price", f"rank refit price, seeds {tag}",
            scope.format(what=f"rank refit on the clean twin, (refit - standard) / opt_z, deployed design"),
            "price", frame(g, (g["regret_clean"] - g["ref_clean"]) / z), seeds, float(row["price"]),
            source=src, paper=price_txt, paper_value=float(price_txt), paper_digits=3))
    return out


# ---------------------------------------------------------------------------
# Scoring one quantity
# ---------------------------------------------------------------------------


def score(q: Quantity, boot: int = BOOT, perms: int = PERMS, seed: int = SEED) -> dict:
    """Every number of one quantity's row; stops if the point or the published intervals do not reproduce."""
    if q.kind not in KINDS:
        raise ValueError(f"{q.key}: unknown kind {q.kind}")
    num, den, lands = landscape_seed_tables(q.runs, q.seeds)
    est = landscape_estimate(q.runs)
    rng = np.random.default_rng([seed, zlib.crc32(q.key.encode())])
    li, si = draw_indices(rng, len(lands), len(q.seeds), boot)
    iv = three_intervals(num, den, li, si)
    if abs(iv["estimate"] - est) > 1e-12:
        raise AssertionError(f"{q.key}: the table's estimate differs from the ratio of landscape means over runs")
    check_point(q, est)
    gap_pct = gap_tw = np.nan
    if np.isfinite(q.published_lo):
        gap_pct = check_interval(f"{q.key} percentile", iv["percentile"], iv["pct_mcse"],
                                 (q.published_lo, q.published_hi), boot, q.published_boot)
    if q.published_two_way is not None:
        gap_tw = check_interval(f"{q.key} two-way", iv["two_way"], iv["two_way_mcse"], q.published_two_way,
                                boot, q.published_boot)
    u = linearised(num, den)
    anova = crossed_anova(u)
    p_seed = seed_effect_p(u, rng, perms)
    intervals = {m: iv[m] for m in ("percentile", "t", "two_way")}
    changes = conclusion_changes(intervals, q.thresholds, {"percentile": iv["pct_mcse"], "two_way": iv["two_way_mcse"]})
    width = {m: iv[m][1] - iv[m][0] for m in intervals}
    row = {
        "key": q.key, "quantity": q.label, "scope": q.scope, "kind": q.kind,
        "n_landscapes": len(lands), "n_seeds": len(q.seeds), "seeds": ",".join(str(s) for s in q.seeds),
        "n_runs": len(q.runs), "estimate": est, "published": q.published, "published_source": q.source,
        "paper": q.paper, "published_lo": q.published_lo, "published_hi": q.published_hi,
        "published_boot": q.published_boot or np.nan,
        "pct_lo": iv["percentile"][0], "pct_hi": iv["percentile"][1],
        "pct_lo_mcse": iv["pct_mcse"][0], "pct_hi_mcse": iv["pct_mcse"][1], "pct_gap_mcse": gap_pct,
        "t_lo": iv["t"][0], "t_hi": iv["t"][1], "t_se": iv["se_t"],
        "two_way_lo": iv["two_way"][0], "two_way_hi": iv["two_way"][1],
        "two_way_lo_mcse": iv["two_way_mcse"][0], "two_way_hi_mcse": iv["two_way_mcse"][1],
        "two_way_kept": iv["two_way_kept"], "two_way_published_gap_mcse": gap_tw,
        "width_t_over_pct": width["t"] / width["percentile"],
        "width_two_way_over_pct": width["two_way"] / width["percentile"],
        "seed_effect_p": p_seed,
        "se_ratio_crossed": float(np.sqrt(max(anova["var_crossed"], 0.0) / anova["var_landscape"])),
        "se_ratio_pigeonhole": float(np.sqrt(anova["var_pigeonhole"] / anova["var_landscape"])),
        "ms_landscape": anova["ms_landscape"], "ms_seed": anova["ms_seed"], "ms_residual": anova["ms_residual"],
        "boot": boot, "perms": perms,
    }
    for m in intervals:
        for th in q.thresholds:
            row[f"{m}_excludes_{th:g}"] = excludes(*iv[m], th)
    row["conclusion_changes"] = "; ".join(changes) if changes else "none"
    return row


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _signed(v: float, digits: int = 4) -> str:
    s = f"{v:+.{digits}f}"
    return f"{0:.{digits}f}" if s in (f"+{0:.{digits}f}", f"-{0:.{digits}f}") else s


# Gains and prices get four decimals, one more than the paper prints: at three, an endpoint of +0.0003 prints as
# 0.000 and an interval that excludes 0 would read as one that touches it.
def fmt_value(kind: str, v: float) -> str:
    if kind in ("share", "loss"):
        return f"{100 * v:.1f}%"
    return _signed(v) if kind == "gain" else f"{v:.4f}"


def fmt_ci(kind: str, lo: float, hi: float) -> str:
    if kind in ("share", "loss"):
        return f"[{100 * lo:.1f}, {100 * hi:.1f}]"
    if kind == "gain":
        return f"[{_signed(lo)}, {_signed(hi)}]"
    return f"[{lo:.4f}, {hi:.4f}]"


def verdict(row: dict) -> str:
    """The percentile interval's verdicts: 'excl 0', 'incl 0', and for a share 'excl 1' / 'incl 1'."""
    parts = [f"{'excl' if row['percentile_excludes_0'] else 'incl'} 0"]
    if "percentile_excludes_1" in row:
        parts.append(f"{'excl' if row['percentile_excludes_1'] else 'incl'} 1")
    return ", ".join(parts)


def report(rows: list[dict], skipped: list[str], boot: int, perms: int) -> list[str]:
    out = [
        "Headline intervals three ways (scripts/checks/headline_intervals.py)",
        "",
        f"percentile = the paper's landscape bootstrap ({boot} resamples of the 20 landscapes);",
        "t          = estimate +/- t(19, 0.975) x delta-method SE of the ratio of landscape means (SE of the mean for a mean);",
        f"two-way    = landscapes and seeds resampled independently ({boot} resamples, the percentile interval's landscape",
        "             draws), each drawn landscape averaged over the drawn seeds, then the ratio of landscape means.",
        "Shares and losses in %; gains, increments and prices in units of the achievable improvement (opt_z).",
        "Every estimate matches its producer's published value to 4 decimals and the paper's rounding, and every",
        "percentile interval (and sitting_by_magnitude's two published two-way intervals) matches the published one",
        f"to within {MC_TOL:g} Monte Carlo SEs; otherwise the script stops.",
        "",
    ]
    head = (f"{'key':<19}{'estimate':>9}  {'percentile':<20}{'t (19 df)':<20}{'two-way':<20}"
            f"{'w t':>5}{'w 2w':>6}{'seed p':>8}  {'percentile verdict':<19}change")
    out += [head, "-" * len(head)]
    for r in rows:
        out.append(f"{r['key']:<19}{fmt_value(r['kind'], r['estimate']):>9}  "
                   f"{fmt_ci(r['kind'], r['pct_lo'], r['pct_hi']):<20}{fmt_ci(r['kind'], r['t_lo'], r['t_hi']):<20}"
                   f"{fmt_ci(r['kind'], r['two_way_lo'], r['two_way_hi']):<20}"
                   f"{r['width_t_over_pct']:>5.2f}{r['width_two_way_over_pct']:>6.2f}"
                   f"{r['seed_effect_p']:>8.3f}  {verdict(r):<19}{r['conclusion_changes']}")
    out += ["",
            "w t, w 2w: width of the t and two-way intervals over the percentile interval's. seed p: permutation test",
            f"of the seed main effect ({perms} within-landscape permutations of the seed labels, unadjusted).",
            "SE ratio crossed: the moment estimate of the SE under crossed landscape and seed effects over the",
            "landscape-only SE of the t interval; below 1 where the seed mean square falls below the residual one (no",
            "seed effect). SE ratio pigeonhole: the two-way bootstrap's own SE over that SE; it exceeds the crossed",
            "ratio because it counts the residual 2.5 (5 seeds) to 2.7 (10 seeds) times (see the module docstring).",
            "",
            "Scopes, producers and checks (estimate to 4 decimals; published and paper values; percentile-interval gap",
            "to the published one in Monte Carlo SEs; crossed-model and pigeonhole SE over the landscape-only SE):"]
    for r in rows:
        pub_ci = ("" if not np.isfinite(r["published_lo"]) else
                  f" {fmt_ci(r['kind'], r['published_lo'], r['published_hi'])} (B = {int(r['published_boot'])})")
        gap = "" if not np.isfinite(r["pct_gap_mcse"]) else f"; gap {r['pct_gap_mcse']:.1f} MC SE"
        tw = ("" if not np.isfinite(r["two_way_published_gap_mcse"])
              else f"; published two-way gap {r['two_way_published_gap_mcse']:.1f} MC SE")
        out += [f"{r['key']}: {r['quantity']}. {r['scope']}; {r['n_runs']} runs, {r['n_landscapes']} landscapes x "
                f"{r['n_seeds']} seeds.",
                f"    estimate {r['estimate']:.4f}; published {r['published']:.4f}{pub_ci}, {r['published_source']}; "
                f"paper {r['paper']}{gap}{tw}; SE ratio crossed {r['se_ratio_crossed']:.2f}, pigeonhole "
                f"{r['se_ratio_pigeonhole']:.2f}."]
    changed = [r for r in rows if r["conclusion_changes"] != "none"]
    out += ["", "Conclusions that change against the percentile interval: "
            + ("none." if not changed else "; ".join(f"{r['key']} ({r['conclusion_changes']})" for r in changed))]
    flagged = [r["key"] for r in rows if r["seed_effect_p"] < 0.05]
    out.append("Seed main effect at p < 0.05 (unadjusted): " + (", ".join(flagged) if flagged else "none") + ".")
    if skipped:
        out += ["", "Skipped (per-run inputs missing on this machine):"] + [f"  {s}" for s in skipped]
    return out


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boot", type=int, default=BOOT, help="resamples of each bootstrap (the paper's 2000)")
    p.add_argument("--perms", type=int, default=PERMS, help="permutations of the seed-effect test")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-csv", type=Path, default=OUT_CSV)
    p.add_argument("--out-txt", type=Path, default=OUT_TXT)
    args = p.parse_args(argv)

    opt_z = aba.dataset_scale("opt_z")
    ship_path = ANALYSIS / "ship_rules_per_run.csv"
    ship = pd.read_csv(ship_path, low_memory=False) if ship_path.is_file() else None
    quantities: list[Quantity] = []
    skipped: list[str] = []
    families = [("a (deployed loss)", lambda: quantities_deployed(opt_z)),
                ("b (selection share)", lambda: quantities_selection(opt_z, ship)),
                ("c, d (zero-trial rules and sittings)", lambda: quantities_end_of_study(ship)),
                ("e (cautious rule)", lambda: quantities_cautious(opt_z)),
                ("f (rank refit)", lambda: quantities_rank_refit(opt_z))]
    for label, build in families:
        if ship is None and label[0] in "bcd":
            skipped.append(f"{label}: {ship_path.relative_to(REPO)} is missing")
            continue
        try:
            quantities += build()
        except FileNotFoundError as exc:
            skipped.append(f"{label}: {exc}")
    rows = [score(q, args.boot, args.perms, args.seed) for q in quantities]
    lines = report(rows, skipped, args.boot, args.perms)
    text = "\n".join(lines) + "\n"
    print(text, end="")
    table = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text(text, encoding="utf-8")
    return table


if __name__ == "__main__":
    main()
