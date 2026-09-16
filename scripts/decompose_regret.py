"""Split the regret of a deployed design into a SEARCH loss and a SELECTION loss.

Almost every Bayesian-optimisation paper reports simple regret at the best
OBSERVED point. Under exact observation that is the best visited point, so the
metric is the search's. Under a noisy rater it is not: the run ships whichever
design happened to be rated highest, which need not be the best one it visited.
The standard metric then silently folds two different failures into one number.

Write x* for the optimum, V for the designs a run visited, and d for the design
it deploys. With f the exact objective and R(x) = f(x*) - f(x),

    R(d)  =  [ f(x*) - max_{x in V} f(x) ]  +  [ max_{x in V} f(x) - f(d) ]
          =        search loss             +        selection loss

The first term is what the search failed to find; it is the only term an
acquisition function can act on. The second is what the ratings failed to
identify among what WAS found; no acquisition function can act on it, and it is
exactly zero whenever the ratings order the visited designs correctly -- which is
why it never appears in a noiseless benchmark.

This script computes both terms per run from the table of
scripts/rescore_ship_rules.py, which already scores every visited design under
the loop's own refitted surrogate:

    search loss     regret_best_visited   (the oracle over visited designs)
    selection loss  regret_best_observed - regret_best_visited
    deployed regret regret_best_observed  (the standard ship rule)

and reports, per error process, magnitude and onset:

  * the selection SHARE of deployed regret, share = selection / deployed;
  * the same split of the EXCESS over the identically seeded clean run, which is
    the part attributable to the error rather than to the problem's difficulty.
    A clean run of an exact objective has zero selection loss by construction, so
    the excess selection loss is the whole selection loss -- the script asserts
    this on the clean rows rather than assuming it;
  * how much of the selection loss each ship rule of the rescoring recovers, so
    the decomposition and the remedies are on one scale.

Regret is divided by the landscape's opt_z (boba_landscape_stats.json) so that
landscapes with a 45x spread of achievable improvement can be averaged. Shares
are ratios of landscape means with a landscape bootstrap, as everywhere else in
this project.

    python scripts/decompose_regret.py
    python scripts/decompose_regret.py --input-dir output-boba --acquisitions logei,qnei
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260916

# The ship rules scored by rescore_ship_rules.py, best_observed being the
# standard one the decomposition is written against.
RULES = ["best_observed", "best_mean", "pm", "lcb1", "lcb2"]
PAIR_KEYS = ["dataset", "acquisition", "seed"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("output-boba"),
                   help="an arm directory holding analysis/ship_rules_per_run.csv")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <input-dir>/analysis")
    p.add_argument("--acquisitions", type=str, default=None,
                   help="comma-separated; default every acquisition in the table")
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    return p.parse_args(argv)


def load_runs(path: Path, acquisitions: set[str] | None) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"{path} not found: run scripts/rescore_ship_rules.py first")
    df = pd.read_csv(path)
    missing = [c for c in ("regret_best_observed", "regret_best_visited") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} lacks {missing}; re-run rescore_ship_rules.py")
    if acquisitions:
        df = df[df["acquisition"].isin(acquisitions)]
    if df.empty:
        raise SystemExit("no runs left after filtering")
    return df


def decompose(df: pd.DataFrame, opt_z: dict[str, float]) -> pd.DataFrame:
    """Per run: the two terms, in units of the landscape's achievable improvement."""
    z = df["dataset"].map(lambda d: opt_z.get(d, 1.0))
    out = df.copy()
    out["deployed"] = df["regret_best_observed"] / z
    out["search"] = df["regret_best_visited"] / z
    out["selection"] = out["deployed"] - out["search"]
    for rule in RULES:
        column = f"regret_{rule}"
        if column in df.columns:
            out[f"deployed_{rule}"] = df[column] / z
    # A rule that ships a visited design can never beat the oracle over visited
    # designs, so selection loss cannot be negative. Floating point can make it
    # -1e-16; anything larger is a bug in the rescoring, not in the data.
    worst = float(out["selection"].min())
    if worst < -1e-9:
        raise ValueError(
            f"negative selection loss ({worst:.3e}): the best-observed design scores better "
            "than the best visited one, so the two are not being scored on the same objective"
        )
    out["selection"] = out["selection"].clip(lower=0.0)
    return out


def assert_clean_runs_have_no_selection_loss(clean: pd.DataFrame) -> float:
    """A clean run of an exact objective rates every design exactly, so its ship
    rule cannot be wrong. Report the largest violation instead of assuming it."""
    if clean.empty:
        return float("nan")
    worst = float(clean["selection"].abs().max())
    if worst > 1e-6:
        raise ValueError(
            f"a clean run shows selection loss up to {worst:.3e}. With an exact objective the "
            "best-rated visited design IS the best visited design, so this means the clean "
            "baselines are not clean or are being paired with the wrong runs."
        )
    return worst


def _boot_ratio(num: np.ndarray, den: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Ratio of landscape means, with a landscape bootstrap."""
    n = len(num)
    if n == 0 or den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(num.mean() / den.mean())
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, n)
        d = den[idx].mean()
        if d != 0:
            draws.append(num[idx].mean() / d)
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarise(noisy: pd.DataFrame, clean: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Shares of the deployed regret, and of the excess over the clean twin."""
    per = noisy.groupby("dataset")[["deployed", "search", "selection"]].mean()
    row: dict[str, object] = {"n_landscapes": len(per), "n_runs": len(noisy)}

    share, lo, hi = _boot_ratio(per["selection"].to_numpy(), per["deployed"].to_numpy(), rng)
    row.update(selection_share=share, selection_share_lo=lo, selection_share_hi=hi)
    row.update(mean_deployed=float(per["deployed"].mean()),
               mean_search=float(per["search"].mean()),
               mean_selection=float(per["selection"].mean()))

    # The excess over the identically seeded clean run: the part the error caused.
    # Pair on (landscape, acquisition, seed); the clean twin is shared by every
    # magnitude and onset, exactly as the recovery analyses pair them.
    if not clean.empty:
        base = clean.groupby(PAIR_KEYS)[["deployed", "search"]].mean().rename(
            columns={"deployed": "deployed_clean", "search": "search_clean"})
        m = noisy.merge(base, on=PAIR_KEYS, how="inner")
        if not m.empty:
            m = m.assign(excess_deployed=m["deployed"] - m["deployed_clean"],
                         excess_search=m["search"] - m["search_clean"])
            # Selection loss is zero in the clean twin, so the excess selection
            # loss is the selection loss itself.
            m = m.assign(excess_selection=m["selection"])
            perx = m.groupby("dataset")[["excess_deployed", "excess_search", "excess_selection"]].mean()
            share, lo, hi = _boot_ratio(perx["excess_selection"].to_numpy(),
                                        perx["excess_deployed"].to_numpy(), rng)
            row.update(excess_selection_share=share, excess_selection_share_lo=lo,
                       excess_selection_share_hi=hi,
                       mean_excess_deployed=float(perx["excess_deployed"].mean()),
                       mean_excess_search=float(perx["excess_search"].mean()),
                       mean_excess_selection=float(perx["excess_selection"].mean()),
                       n_paired=len(m))
    return row


def rule_recovery(noisy: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Share of the SELECTION loss each ship rule removes.

    The denominator is the selection loss of the standard rule, so 100% means the
    rule ships the best visited design every time and 0% means it ships what the
    standard rule ships. It can go negative: a rule may be worse.
    """
    out: dict[str, object] = {}
    per = noisy.groupby("dataset")[["deployed", "search", "selection"]].mean()
    den = per["selection"].to_numpy()
    for rule in RULES:
        column = f"deployed_{rule}"
        if column not in noisy.columns or rule == "best_observed":
            continue
        rule_sel = noisy.groupby("dataset")[column].mean().to_numpy() - per["search"].to_numpy()
        num = den - np.clip(rule_sel, 0.0, None)
        share, lo, hi = _boot_ratio(num, den, rng)
        out[f"{rule}_selection_recovered"] = share
        out[f"{rule}_selection_recovered_lo"] = lo
        out[f"{rule}_selection_recovered_hi"] = hi
    return out


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    out_dir = args.output_dir or (args.input_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    acqs = {a.strip() for a in args.acquisitions.split(",")} if args.acquisitions else None
    runs = load_runs(args.input_dir / "analysis" / "ship_rules_per_run.csv", acqs)
    stats = bb.load_stats(args.stats_path)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}

    table = decompose(runs, opt_z)
    clean = table[table["baseline"]]
    noisy = table[~table["baseline"]]
    worst_clean = assert_clean_runs_have_no_selection_loss(clean)
    print(f"{len(noisy):,} noisy runs, {len(clean):,} clean, "
          f"{noisy['dataset'].nunique()} landscapes, "
          f"{noisy['acquisition'].nunique()} acquisitions; "
          f"largest selection loss on a clean run {worst_clean:.2e} (must be ~0)\n")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (model, std, onset), group in noisy.groupby(["error_model", "jitter_std", "jitter_iteration"],
                                                    dropna=False):
        row = {"error_model": model, "jitter_std": std, "jitter_iteration": onset}
        row.update(summarise(group, clean, rng))
        row.update(rule_recovery(group, rng))
        rows.append(row)
    for model, group in noisy.groupby("error_model", dropna=False):
        row = {"error_model": model, "jitter_std": "pooled", "jitter_iteration": "pooled"}
        row.update(summarise(group, clean, rng))
        row.update(rule_recovery(group, rng))
        rows.append(row)
    row = {"error_model": "pooled", "jitter_std": "pooled", "jitter_iteration": "pooled"}
    row.update(summarise(noisy, clean, rng))
    row.update(rule_recovery(noisy, rng))
    rows.append(row)

    frame = pd.DataFrame(rows)
    path = out_dir / "regret_decomposition.csv"
    frame.to_csv(path, index=False)

    print("Share of the DEPLOYED design's regret that is selection, not search")
    print("(the part no acquisition function can reach):\n")
    for _, r in frame.iterrows():
        if r["jitter_std"] == "pooled" and r["error_model"] != "pooled":
            continue
        label = (f"{r['error_model']:<9} std {str(r['jitter_std']):>6} onset {str(r['jitter_iteration']):>5}"
                 if r["error_model"] != "pooled" else f"{'ALL':<9} {'':>10} {'':>11}")
        share = r.get("selection_share", float("nan"))
        ex = r.get("excess_selection_share", float("nan"))
        print(f"  {label}  selection {share * 100:5.1f}% [{r.get('selection_share_lo', np.nan) * 100:4.0f},"
              f"{r.get('selection_share_hi', np.nan) * 100:4.0f}]   "
              f"of the EXCESS {ex * 100:5.1f}% [{r.get('excess_selection_share_lo', np.nan) * 100:4.0f},"
              f"{r.get('excess_selection_share_hi', np.nan) * 100:4.0f}]")

    print("\nShare of the SELECTION loss each ship rule removes (pooled over everything):")
    last = frame.iloc[-1]
    for rule in RULES:
        key = f"{rule}_selection_recovered"
        if key in last and pd.notna(last[key]):
            print(f"  {rule:<14} {last[key] * 100:+6.1f}% "
                  f"[{last[f'{rule}_selection_recovered_lo'] * 100:+.0f},"
                  f"{last[f'{rule}_selection_recovered_hi'] * 100:+.0f}]")

    print(f"\nWrote {path}")
    return frame


if __name__ == "__main__":
    main()
