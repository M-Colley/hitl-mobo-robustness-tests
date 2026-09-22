"""Does a 95% landscape-cluster bootstrap interval cover 95% of the time at 20 clusters?

Every headline interval in the paper resamples the 20 landscapes: a ratio of
landscape means (a share of the cost of error, a recovery = mean gain / mean
cost) or a plain landscape mean, with a percentile interval in most producers
and a BCa interval in the currency fit. Twenty clusters is few, so the actual
coverage of those procedures is an empirical question. This script answers it by
simulation grounded in the paper's own per-landscape data.

The check is the standard one for a bootstrap procedure when the population is
unknown (the outer level of an iterated, or double, bootstrap; Hall 1986, Beran
1987, Davison & Hinkley 1997 ch. 5): the empirical distribution of the 20
observed landscapes plays the population. The true value is the estimand
computed on all 20. Each of R simulated studies draws 20 landscapes with
replacement, builds its intervals from B cluster-bootstrap replicates of ITS OWN
20, and scores whether each interval covers the true value.

Intervals, all at nominal 95% and all from the same B replicates of a study:

  percentile  2.5% and 97.5% quantiles of the replicate estimates (what
              decompose_regret.py and replay_hitl_remedies.py print);
  BCa         bias-corrected and accelerated, z0 from the share of replicates
              below the point estimate, acceleration from the leave-one-cluster-
              out jackknife -- the formula of analyse_boba_robustness._bca_interval;
  boot-t      studentised bootstrap, t* = (theta* - theta) / se*, with se the
              linearisation (delta-method) standard error of a ratio of means;
  t           theta +/- t_{19, 0.975} * se, the normal-theory reference.

Limitation, stated in the report as well: replacing the population by the
empirical distribution can only see coverage error that comes from having 20
clusters. It cannot see bias that the empirical distribution shares -- an
unrepresentative set of landscapes -- nor dependence between clusters (the
landscapes share their seeds by design; positive correlation between them would
make every cluster interval too narrow and is invisible here).

The three estimands (per-landscape inputs reproduce the paper exactly; the
script asserts it):

  selection_share   selection loss / deployed excess over the clean twin at
                    1 sigma, onset 0, pooled over the four error processes and
                    the ten model-based acquisitions (decompose_regret.py):
                    paper 41.8% [34, 50].
  shortlist_m3      recovery of the standard process's cost of error by
                    shipping three designs, main sweep, pooled over everything
                    (replay_hitl_remedies.py): paper 24% [18, 30].
  deployed_cost     deployed excess / opt_z at 1 sigma, onset 0, model-based
                    acquisitions, from cell_means.csv: paper 29.7%.

    python scripts/bootstrap_coverage.py
    python scripts/bootstrap_coverage.py --reps 500 --boot 1000
"""
from __future__ import annotations

import os

# Other jobs share this machine; the work here is indexing and reductions, which
# gain nothing from a threaded BLAS.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402

ALPHA = 0.05
SEED = 20260922
MODEL_FREE = ("random", "sobol")
METHODS = ("percentile", "bca", "boot_t", "t")
PAIR_KEYS = ["dataset", "acquisition", "seed"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("output-boba"))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <input-dir>/analysis/review")
    # 10,000 rather than 1,000: vectorised, the whole run takes under a minute,
    # and the Monte Carlo SE of a coverage near 95% falls from 0.7 to 0.2 points.
    p.add_argument("--reps", type=int, default=10000, help="simulated studies R")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap replicates B per study")
    p.add_argument("--boot-actual", type=int, default=20000,
                   help="replicates for the intervals on the actual data")
    p.add_argument("--chunk", type=int, default=50, help="studies per vectorised batch")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# The per-landscape inputs, following each producer exactly
# ---------------------------------------------------------------------------


def selection_share_inputs(input_dir: Path, opt_z: dict[str, float]) -> pd.DataFrame:
    """decompose_regret.summarise, row (pooled, 1.0, 0): per-run division by opt_z,
    noisy runs paired with the clean twin on (dataset, acquisition, seed), then
    landscape means of the selection loss and of the deployed excess."""
    df = pd.read_csv(input_dir / "analysis" / "ship_rules_per_run.csv")
    z = df["dataset"].map(lambda d: opt_z.get(d, 1.0))
    df = df.assign(deployed=df["regret_best_observed"] / z,
                   search=df["regret_best_visited"] / z)
    df = df.assign(selection=(df["deployed"] - df["search"]).clip(lower=0.0))
    clean, noisy = df[df["baseline"]], df[~df["baseline"]]
    if float(clean["selection"].abs().max()) > 1e-6:
        raise ValueError("a clean run shows selection loss; the pairing is wrong")
    cell = noisy[(noisy["jitter_std"] == 1.0) & (noisy["jitter_iteration"] == 0)
                 & ~noisy["acquisition"].isin(MODEL_FREE)]
    models = sorted(cell["error_model"].unique())
    if models != ["ar1", "bias", "drift", "gaussian"]:
        raise ValueError(f"unexpected error processes {models}")
    base = clean.groupby(PAIR_KEYS)[["deployed"]].mean().rename(columns={"deployed": "deployed_clean"})
    m = cell.merge(base, on=PAIR_KEYS, how="inner")
    if len(m) != len(cell):
        raise ValueError(f"{len(cell) - len(m)} noisy runs lack a clean twin")
    m = m.assign(excess_deployed=m["deployed"] - m["deployed_clean"])
    per = m.groupby("dataset")[["selection", "excess_deployed"]].mean()
    per.attrs["n_runs"] = len(m)
    return per.rename(columns={"selection": "num", "excess_deployed": "den"})


def shortlist_inputs(input_dir: Path, opt_z: dict[str, float], procedure: str = "shortlist_m3") -> pd.DataFrame:
    """replay_hitl_remedies.recovery_table, pooled row of `procedure`."""
    cols = ["dataset", "procedure", "regret_noisy", "regret_clean", "ref_noisy", "ref_clean"]
    path = input_dir / "analysis" / "hitl_remedies" / "hitl_remedies_per_run.csv.gz"
    h = pd.read_csv(path, usecols=cols)
    g = h[h["procedure"] == procedure]
    z = g["dataset"].map(lambda d: opt_z.get(d, 1.0))
    g = g.assign(**{c: g[c] / z for c in cols[2:]})
    per = g.groupby("dataset")[["ref_noisy", "ref_clean", "regret_noisy"]].mean()
    out = pd.DataFrame({"num": per["ref_noisy"] - per["regret_noisy"],   # gain
                        "den": per["ref_noisy"] - per["ref_clean"]})     # cost
    out.attrs["n_runs"] = len(g)
    return out


def deployed_cost_inputs(input_dir: Path, opt_z: dict[str, float]) -> pd.DataFrame:
    """Mean over landscapes of the per-landscape mean of inference_excess / opt_z
    at 1 sigma, onset 0, over the model-based acquisitions and four processes."""
    cm = pd.read_csv(input_dir / "analysis" / "cell_means.csv")
    c = cm[(cm["jitter_std"] == 1.0) & (cm["jitter_iteration"] == 0)
           & ~cm["acquisition"].isin(MODEL_FREE)]
    c = c.assign(x=c["inference_excess"] / c["dataset"].map(opt_z))
    per = c.groupby("dataset")[["x"]].mean().rename(columns={"x": "num"})
    per.attrs["n_runs"] = int(c["seeds"].sum())
    return per


# ---------------------------------------------------------------------------
# Vectorised interval machinery. The last axis always indexes clusters.
# ---------------------------------------------------------------------------


def estimate(num: np.ndarray, den: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate and plug-in linearisation standard error.

    For a ratio of means R = mean(num) / mean(den) the influence of cluster i is
    (num_i - R den_i) / mean(den); for a mean it is x_i - mean(x). The plug-in SE
    is sqrt(sum u_i^2) / n (no n - 1: it only enters the bootstrap-t pivot, where
    a constant factor cancels, and the t interval rescales it)."""
    n = num.shape[-1]
    if den is None:
        theta = num.mean(-1)
        u = num - theta[..., None]
    else:
        dbar = den.mean(-1)
        theta = num.mean(-1) / dbar
        u = (num - theta[..., None] * den) / dbar[..., None]
    se = np.sqrt((u * u).sum(-1)) / n
    return theta, se


def jackknife(num: np.ndarray, den: np.ndarray | None) -> np.ndarray:
    """Leave-one-cluster-out estimates, shape (..., n), without a loop."""
    n = num.shape[-1]
    s_num = num.sum(-1, keepdims=True)
    if den is None:
        return (s_num - num) / (n - 1)
    return (s_num - num) / (den.sum(-1, keepdims=True) - den)


def row_quantile(sorted_x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row quantile of pre-sorted rows at a per-row level, numpy's default
    'linear' rule, so the percentile columns equal np.percentile exactly."""
    s, b = sorted_x.shape
    h = (b - 1) * np.clip(np.broadcast_to(np.asarray(q, dtype=float), (s,)), 0.0, 1.0)
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, b - 1)
    rows = np.arange(s)
    return sorted_x[rows, lo] + (h - lo) * (sorted_x[rows, hi] - sorted_x[rows, lo])


def intervals(num: np.ndarray, den: np.ndarray | None, boot_idx: np.ndarray,
              alpha: float = ALPHA) -> dict[str, np.ndarray]:
    """All four intervals for S studies at once.

    num, den   (S, n) per-cluster inputs of each study
    boot_idx   (S, B, n) resampled cluster positions, shared by every method
    returns    {method: (S, 2)} plus the BCa constants z0 and a
    """
    s, n = num.shape
    b = boot_idx.shape[1]
    theta, se = estimate(num, den)

    rows = np.arange(s)[:, None, None]
    bnum = num[rows, boot_idx]
    bden = None if den is None else den[rows, boot_idx]
    theta_b, se_b = estimate(bnum, bden)                       # (S, B)
    sorted_b = np.sort(theta_b, axis=1)
    out: dict[str, np.ndarray] = {}

    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    out["percentile"] = np.stack([row_quantile(sorted_b, lo_q), row_quantile(sorted_b, hi_q)], 1)

    # BCa, as analyse_boba_robustness._bca_interval: z0 from the share strictly
    # below the point (clipped to [1/B, 1 - 1/B]), a from the jackknife.
    below = np.clip((theta_b < theta[:, None]).mean(1), 1.0 / b, 1.0 - 1.0 / b)
    z0 = stats.norm.ppf(below)
    jack = jackknife(num, den)
    centred = jack.mean(1, keepdims=True) - jack
    denom = 6.0 * ((centred ** 2).sum(1)) ** 1.5
    accel = np.where(denom > 0, (centred ** 3).sum(1) / np.where(denom > 0, denom, 1.0), 0.0)

    def adjusted(q: float) -> np.ndarray:
        z = stats.norm.ppf(q)
        return stats.norm.cdf(z0 + (z0 + z) / (1.0 - accel * (z0 + z)))

    out["bca"] = np.stack([row_quantile(sorted_b, adjusted(lo_q)),
                           row_quantile(sorted_b, adjusted(hi_q))], 1)

    # Studentised bootstrap. A replicate made of one repeated cluster has se* = 0;
    # at n = 20 that has probability ~20^-19, but it is dropped rather than divided.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_b = (theta_b - theta[:, None]) / se_b
    t_b[~np.isfinite(t_b)] = np.nan
    if np.isnan(t_b).any():
        t_lo, t_hi = np.nanquantile(t_b, [lo_q, hi_q], axis=1)
    else:
        t_sorted = np.sort(t_b, axis=1)
        t_lo, t_hi = row_quantile(t_sorted, lo_q), row_quantile(t_sorted, hi_q)
    out["boot_t"] = np.stack([theta - t_hi * se, theta - t_lo * se], 1)

    half = stats.t.ppf(hi_q, n - 1) * se * np.sqrt(n / (n - 1))
    out["t"] = np.stack([theta - half, theta + half], 1)

    out["_theta"], out["_z0"], out["_accel"] = theta, z0, accel
    return out


# ---------------------------------------------------------------------------
# Actual data and coverage simulation
# ---------------------------------------------------------------------------


def actual_intervals(num: np.ndarray, den: np.ndarray | None, boot: int,
                     rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(num)
    idx = rng.integers(0, n, (1, boot, n))
    return intervals(num[None, :], None if den is None else den[None, :], idx)


def scipy_check(num: np.ndarray, den: np.ndarray | None, boot: int, seed: int) -> dict[str, tuple]:
    """The same percentile and BCa intervals from scipy.stats.bootstrap, as an
    independent implementation (paired resampling of the cluster rows)."""
    if den is None:
        data, stat = (num,), (lambda x, axis=-1: np.mean(x, axis=axis))
    else:
        data = (num, den)

        def stat(x, y, axis=-1):
            return np.mean(x, axis=axis) / np.mean(y, axis=axis)
    out = {}
    for method in ("percentile", "BCa"):
        res = stats.bootstrap(data, stat, paired=True, vectorized=True, n_resamples=boot,
                              confidence_level=1 - ALPHA, method=method,
                              rng=np.random.default_rng(seed))
        out[method.lower()] = (float(res.confidence_interval.low), float(res.confidence_interval.high))
    return out


def coverage(num: np.ndarray, den: np.ndarray | None, study_idx: np.ndarray, boot: int,
             chunk: int, rng: np.random.Generator) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
    """Intervals for every simulated study; returns the population value, the
    (R, 2) interval per method, and each study's point estimate."""
    truth = float(num.mean() / den.mean()) if den is not None else float(num.mean())
    r, n = study_idx.shape
    bounds = {m: np.empty((r, 2)) for m in METHODS}
    points = np.empty(r)
    for start in range(0, r, chunk):
        sl = slice(start, min(start + chunk, r))
        sidx = study_idx[sl]
        s_num = num[sidx]
        s_den = None if den is None else den[sidx]
        bidx = rng.integers(0, n, (len(sidx), boot, n))
        res = intervals(s_num, s_den, bidx)
        for m in METHODS:
            bounds[m][sl] = res[m]
        points[sl] = res["_theta"]
    return truth, bounds, points


def known_truth_check(rng: np.random.Generator, reps: int = 4000, boot: int = 2000,
                      n: int = 20, chunk: int = 100) -> dict[str, float]:
    """The machinery on a case with a textbook answer: the mean of 20 standard
    normals. The t interval is exact (95%); the percentile interval is, to first
    order, xbar +/- 1.96 * s * sqrt((n-1)/n) / sqrt(n), whose coverage is
    P(|T_19| <= 1.96 sqrt(19/20)), about 92.9%."""
    x = rng.normal(size=(reps, n))
    hits = {m: 0 for m in METHODS}
    for start in range(0, reps, chunk):
        block = x[start:start + chunk]
        res = intervals(block, None, rng.integers(0, n, (len(block), boot, n)))
        for m in METHODS:
            hits[m] += int(((res[m][:, 0] <= 0) & (0 <= res[m][:, 1])).sum())
    out = {m: hits[m] / reps for m in METHODS}
    out["percentile_theory"] = float(2 * stats.t.cdf(stats.norm.ppf(0.975) * np.sqrt((n - 1) / n), n - 1) - 1)
    out["reps"] = reps
    return out


def score(truth: float, bounds: np.ndarray) -> dict[str, float]:
    lo, hi = bounds[:, 0], bounds[:, 1]
    covered = (lo <= truth) & (truth <= hi)
    r = len(covered)
    p = covered.mean()
    width = hi - lo
    return {
        "coverage": float(p),
        "coverage_mcse": float(np.sqrt(p * (1 - p) / r)),
        # "below": the whole interval lies below the true value (it misses low);
        # "above": it lies above. Equal-tailed nominal is 2.5% each.
        "miss_below": float((hi < truth).mean()),
        "miss_above": float((lo > truth).mean()),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
    }


def paired_test(truth: float, a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Studies covered by a but not b, by b but not a, and the exact McNemar p."""
    ca = (a[:, 0] <= truth) & (truth <= a[:, 1])
    cb = (b[:, 0] <= truth) & (truth <= b[:, 1])
    only_a, only_b = int((ca & ~cb).sum()), int((cb & ~ca).sum())
    p = 1.0 if only_a + only_b == 0 else stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
    return only_a, only_b, float(p)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


ESTIMANDS = {
    "selection_share": {
        "label": "Selection share of the deployed excess, 1 sigma, onset 0",
        "paper": "41.8% [34, 50] (percentile, decompose_regret.py)",
        "num": "selection loss", "den": "deployed excess over the clean twin",
    },
    "shortlist_m3": {
        "label": "Shortlist m = 3 recovery, main sweep, pooled",
        "paper": "24% [18, 30] (percentile, replay_hitl_remedies.py)",
        "num": "gain = ref_noisy - regret_noisy", "den": "cost = ref_noisy - ref_clean",
    },
    "deployed_cost": {
        "label": "Deployed cost (excess / opt_z), 1 sigma, onset 0",
        "paper": "29.7%",
        "num": "deployed excess / opt_z", "den": None,
    },
}

METHOD_LABEL = {"percentile": "percentile", "bca": "BCa", "boot_t": "bootstrap-t", "t": "t (delta method)"}


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def ci(lo: float, hi: float, digits: int = 1) -> str:
    return f"[{pct(lo, digits)}, {pct(hi, digits)}]"


def write_markdown(path: Path, frame: pd.DataFrame, inputs: dict[str, pd.DataFrame],
                   extras: dict[str, dict], args: argparse.Namespace, runtime: float) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Coverage of the landscape-cluster bootstrap at 20 clusters")
    add("")
    add(f"Produced by `scripts/bootstrap_coverage.py` on {pd.Timestamp.now():%Y-%m-%d}; "
        f"R = {args.reps} simulated studies, B = {args.boot} replicates per study, "
        f"B = {args.boot_actual} for the intervals on the actual data; seed {args.seed}; "
        f"{runtime:.0f} s.")
    add("")
    add("## Question")
    add("")
    add("A reviewer asked whether a 95% interval that resamples 20 landscape clusters "
        "covers 95% of the time, and asked for percentile intervals beside BCa. Every "
        "headline interval in the paper is a landscape-cluster bootstrap of a ratio of "
        "landscape means or of a landscape mean. The selection share and the remedy "
        "recoveries are printed with percentile intervals (`decompose_regret.py`, "
        "`replay_hitl_remedies.py`); BCa is used in the currency fit "
        "(`analyse_boba_robustness.py`), which this check does not cover.")
    add("")
    add("## Method")
    add("")
    add("The 20 observed landscapes are treated as the population, so the true value is "
        "the estimand computed on all 20. Each simulated study draws 20 landscapes with "
        "replacement, builds all four intervals from the same B cluster-bootstrap "
        "replicates of its own 20, and is scored on whether each interval covers the "
        "true value. This is the standard way to check a bootstrap procedure's coverage "
        "when the population is unknown: it is the outer level of an iterated (double) "
        "bootstrap (Hall 1986; Beran 1987; Davison & Hinkley 1997, ch. 5).")
    add("")
    add("- **percentile**: 2.5% and 97.5% quantiles of the replicate estimates.")
    add("- **BCa**: z0 from the share of replicates below the point estimate, "
        "acceleration from the leave-one-cluster-out jackknife "
        "(the formula in `analyse_boba_robustness._bca_interval`).")
    add("- **bootstrap-t**: studentised with the linearisation (delta-method) standard "
        "error of a ratio of means, recomputed in every replicate.")
    add("- **t**: estimate +/- t(19, 0.975) x delta-method SE (with the n - 1 correction), "
        "the standard cluster-robust interval with G - 1 degrees of freedom "
        "(Cameron & Miller 2015).")
    add("")
    mcse = np.sqrt(0.95 * 0.05 / args.reps)
    add(f"Monte Carlo standard error of a coverage near 95% is {100 * mcse:.2f} points at "
        f"R = {args.reps}. The four methods share each study's replicates, so their "
        "coverages are compared pairwise (exact McNemar test on the studies one covers and "
        "the other misses).")
    add("")
    ck = extras["_check"]
    add(f"**Machinery check.** On {ck['reps']:,} samples of 20 standard normals, where the answer "
        f"is known, the same code gives t {pct(ck['t'])}% (exact: 95.0%), percentile "
        f"{pct(ck['percentile'])}% (first-order theory: {pct(ck['percentile_theory'])}%), "
        f"BCa {pct(ck['bca'])}% and bootstrap-t {pct(ck['boot_t'])}%. The percentile and "
        "BCa intervals of a mean undercover at n = 20 even for normal data, because they "
        "use normal rather than t quantiles and a plug-in (divide-by-n) spread.")
    add("")
    add("**What this cannot detect.** Replacing the population by the empirical "
        "distribution measures only the coverage error that comes from having 20 clusters. "
        "It cannot detect bias that the empirical distribution itself carries: if the 20 "
        "landscapes are unrepresentative of the landscapes the claim is about, every "
        "simulated study inherits that. It also treats landscapes as independent clusters. "
        "The landscapes share their random seeds by design; positive correlation between "
        "landscapes would make every cluster interval too narrow, and this simulation "
        "cannot see it. Finally, a study resampled from 20 points holds on average 12.8 "
        "distinct landscapes, so the simulated studies are less diverse than real draws "
        "from a continuous population; the estimate is first-order accurate for the true "
        "coverage, not exact.")
    add("")
    add("## Results")
    add("")
    for key, meta in ESTIMANDS.items():
        sub = frame[frame["estimand"] == key].set_index("method")
        ex = extras[key]
        add(f"### {meta['label']}")
        add("")
        add(f"Paper: {meta['paper']}. Point estimate here: **{pct(ex['truth'])}%** "
            f"({ex['n_runs']:,} runs over 20 landscapes). On the actual data the BCa "
            f"constants are z0 = {ex['z0']:+.3f} and a = {ex['accel']:+.4f}; the estimator's "
            f"bias across simulated studies is {100 * ex['bias']:+.2f} points.")
        add("")
        add("| interval | on the actual data | simulated coverage | interval wholly below / above truth | mean width | actual width |")
        add("|---|---|---|---|---|---|")
        for m in METHODS:
            r = sub.loc[m]
            add(f"| {METHOD_LABEL[m]} | {ci(r['actual_lo'], r['actual_hi'])} | "
                f"{pct(r['coverage'])}% (+/- {pct(r['coverage_mcse'])}) | "
                f"{pct(r['miss_below'])}% / {pct(r['miss_above'])}% | "
                f"{pct(r['mean_width'])} | {pct(r['actual_hi'] - r['actual_lo'])} |")
        add("")
        for (a, b), (only_a, only_b, p) in ex["paired"].items():
            add(f"- {METHOD_LABEL[a]} vs {METHOD_LABEL[b]}: {only_a} studies covered only by "
                f"{METHOD_LABEL[a]}, {only_b} only by {METHOD_LABEL[b]} (McNemar p = {p:.3g}).")
        sc = ex["scipy"]
        add(f"- Cross-check with `scipy.stats.bootstrap` (independent implementation, "
            f"B = {args.boot_actual}): percentile {ci(*sc['percentile'])}, BCa {ci(*sc['bca'])}.")
        add("")
    add("## Recommendation")
    add("")
    for line in extras["_recommendation"]:
        add(line)
    add("")
    add("## Per-landscape inputs")
    add("")
    add("All in units of the landscape's achievable improvement (opt_z). The deployed-cost "
        "column is computed from `cell_means.csv` independently and equals the selection "
        "share's denominator to within "
        f"{extras['_den_match']:.1e}.")
    add("")
    add("| landscape | selection loss | deployed excess | shortlist gain | shortlist cost |")
    add("|---|---|---|---|---|")
    a, b = inputs["selection_share"], inputs["shortlist_m3"]
    for d in a.index:
        add(f"| {d} | {a.loc[d, 'num']:.4f} | {a.loc[d, 'den']:.4f} | "
            f"{b.loc[d, 'num']:.4f} | {b.loc[d, 'den']:.4f} |")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


def recommend(frame: pd.DataFrame) -> list[str]:
    """The verdict, computed from the table rather than written ahead of it."""
    cov = frame.pivot(index="estimand", columns="method", values="coverage").loc[list(ESTIMANDS)]
    width = frame.pivot(index="estimand", columns="method", values="mean_width").loc[list(ESTIMANDS)]
    worst_gap = (cov - 0.95).abs().max()
    best = str(worst_gap.idxmin())

    def span(m: str) -> str:
        return f"{pct(cov[m].min())}-{pct(cov[m].max())}%"

    extra = (width[best] / width["percentile"] - 1)
    z0 = frame.groupby("estimand")["bca_z0_actual"].first().abs().max()
    acc = frame.groupby("estimand")["bca_accel_actual"].first().abs().max()
    shift = []
    for key in ESTIMANDS:
        sub = frame[frame["estimand"] == key].set_index("method")
        shift.append(max(abs(sub.loc["percentile", "actual_lo"] - sub.loc["bca", "actual_lo"]),
                         abs(sub.loc["percentile", "actual_hi"] - sub.loc["bca", "actual_hi"])))
    out = [
        f"At nominal 95% and 20 clusters the percentile interval covers {span('percentile')}, "
        f"BCa {span('bca')}, the studentised bootstrap {span('boot_t')} and the t interval "
        f"{span('t')}. The method closest to nominal on every estimand is "
        f"**{METHOD_LABEL[best]}** (worst miss {pct(worst_gap[best])} points), at "
        f"{pct(extra.min(), 0)}-{pct(extra.max(), 0)}% more width than the percentile interval.",
        "",
        f"BCa buys almost nothing here: its constants are small on every estimand "
        f"(|z0| <= {z0:.3f}, |a| <= {acc:.3f}), its endpoints on the actual data move by at most "
        f"{pct(max(shift), 1)} points from the percentile ones, and it still undercovers. The "
        "shortfall of both is the textbook small-sample narrowness of any interval built on "
        "normal quantiles and a plug-in spread (the machinery check shows the same 2-point "
        "shortfall on normal data), not skew that BCa could correct.",
        "",
        "Per estimand, what to print (headline interval first, percentile alongside):",
        "",
    ]
    for key, meta in ESTIMANDS.items():
        sub = frame[frame["estimand"] == key].set_index("method")
        b, p = sub.loc[best], sub.loc["percentile"]
        out.append(f"- {meta['label']}: {pct(b['estimate'])}% {ci(b['actual_lo'], b['actual_hi'])} "
                   f"({METHOD_LABEL[best]}, simulated coverage {pct(b['coverage'])}%); percentile "
                   f"{ci(p['actual_lo'], p['actual_hi'])} ({pct(p['coverage'])}%).")
    return out


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    t0 = time.time()
    out_dir = args.output_dir or (args.input_dir / "analysis" / "review")
    out_dir.mkdir(parents=True, exist_ok=True)
    opt_z = {k: float(v["opt_z"]) for k, v in bb.load_stats(args.stats_path).items() if "opt_z" in v}

    inputs = {
        "selection_share": selection_share_inputs(args.input_dir, opt_z),
        "shortlist_m3": shortlist_inputs(args.input_dir, opt_z),
        "deployed_cost": deployed_cost_inputs(args.input_dir, opt_z),
    }
    order = inputs["selection_share"].index
    for key in inputs:
        if len(inputs[key]) != 20 or not inputs[key].index.equals(order):
            raise ValueError(f"{key}: expected the same 20 landscapes")

    # The inputs must reproduce the paper before their intervals mean anything.
    a, b, c = inputs["selection_share"], inputs["shortlist_m3"], inputs["deployed_cost"]
    reproduced = {
        "selection_share": (a["num"].mean() / a["den"].mean(), 0.418),
        "shortlist_m3": (b["num"].mean() / b["den"].mean(), 0.240),
        "deployed_cost": (c["num"].mean(), 0.297),
    }
    for key, (value, paper) in reproduced.items():
        if abs(value - paper) > 0.0006:
            raise ValueError(f"{key} = {value:.4f} does not reproduce the paper's {paper}")
        print(f"{key:16s} reproduces the paper: {100 * value:.2f}% (paper {100 * paper:.1f}%)")
    den_match = float((c["num"] - a["den"]).abs().max())
    if den_match > 1e-9:
        raise ValueError("cell_means deployed cost differs from the decomposition's denominator")

    ss = np.random.SeedSequence(args.seed)
    s_study, s_boot, s_actual, s_check = ss.spawn(4)
    check = known_truth_check(np.random.default_rng(s_check))
    print("known-truth check, mean of 20 N(0,1): " + ", ".join(
        f"{METHOD_LABEL[m]} {pct(check[m])}%" for m in METHODS)
        + f" (theory: t 95.0%, percentile {pct(check['percentile_theory'])}%)")
    if abs(check["t"] - 0.95) > 4 * np.sqrt(0.95 * 0.05 / check["reps"]):
        raise ValueError("the t interval misses its exact coverage: the machinery is wrong")
    # One set of simulated studies (landscape draws) shared by the three estimands.
    study_idx = np.random.default_rng(s_study).integers(0, 20, (args.reps, 20))
    boot_rng = np.random.default_rng(s_boot)
    actual_rng = np.random.default_rng(s_actual)

    rows, extras = [], {}
    for key in ESTIMANDS:
        num = inputs[key]["num"].to_numpy(float)
        den = inputs[key]["den"].to_numpy(float) if "den" in inputs[key] else None
        if den is not None and (den <= 0).any():
            raise ValueError(f"{key}: a landscape has a non-positive denominator")

        act = actual_intervals(num, den, args.boot_actual, actual_rng)
        truth, bounds, points = coverage(num, den, study_idx, args.boot, args.chunk, boot_rng)
        sc = scipy_check(num, den, args.boot_actual, args.seed + 1)
        for m, (lo, hi) in (("percentile", sc["percentile"]), ("bca", sc["bca"])):
            drift = max(abs(lo - act[m][0, 0]), abs(hi - act[m][0, 1]))
            if drift > 0.01:
                raise ValueError(f"{key} {m}: scipy and this implementation differ by {drift:.4f}")

        pairs = {}
        for x, y in (("percentile", "bca"), ("percentile", "boot_t"), ("bca", "boot_t")):
            pairs[(x, y)] = paired_test(truth, bounds[x], bounds[y])
        extras[key] = {"truth": truth, "n_runs": int(inputs[key].attrs.get("n_runs", 0)),
                       "z0": float(act["_z0"][0]), "accel": float(act["_accel"][0]),
                       "bias": float(points.mean() - truth), "scipy": sc, "paired": pairs}
        for m in METHODS:
            rows.append({
                "estimand": key, "method": m, "estimate": truth,
                "actual_lo": float(act[m][0, 0]), "actual_hi": float(act[m][0, 1]),
                "population_value": truth, **score(truth, bounds[m]),
                "estimator_bias": extras[key]["bias"],
                "bca_z0_actual": extras[key]["z0"], "bca_accel_actual": extras[key]["accel"],
                "scipy_lo": sc[m][0] if m in sc else np.nan,
                "scipy_hi": sc[m][1] if m in sc else np.nan,
                "n_clusters": 20, "reps": args.reps, "boot": args.boot,
                "boot_actual": args.boot_actual, "nominal": 1 - ALPHA,
            })
        print(f"{key:16s} done ({time.time() - t0:.0f} s)")

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "bootstrap_coverage.csv", index=False)
    extras["_recommendation"] = recommend(frame)
    extras["_den_match"] = den_match
    extras["_check"] = check
    write_markdown(out_dir / "bootstrap_coverage.md", frame, inputs, extras, args, time.time() - t0)

    print()
    for key in ESTIMANDS:
        print(ESTIMANDS[key]["label"])
        for _, r in frame[frame["estimand"] == key].iterrows():
            print(f"  {METHOD_LABEL[r['method']]:17s} actual {ci(r['actual_lo'], r['actual_hi'])}"
                  f"  coverage {pct(r['coverage'])}% (+/-{pct(r['coverage_mcse'])})"
                  f"  miss below/above {pct(r['miss_below'])}/{pct(r['miss_above'])}"
                  f"  mean width {pct(r['mean_width'])}")
    print(f"\nWrote {out_dir / 'bootstrap_coverage.csv'} and bootstrap_coverage.md")
    return frame


if __name__ == "__main__":
    main()
