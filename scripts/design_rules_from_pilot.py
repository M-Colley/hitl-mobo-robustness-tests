"""Two decisions a practitioner makes BEFORE the study, from a ten-trial pilot.

Everything else in this project changes the loop or the analysis. These two
change the study's design, and both are decidable in advance from quantities a
pilot yields, so they cost nothing once the pilot exists.

  sizing      Trials are the scarce resource, and landscapes differ 74x in what
              is there to win. Given a portfolio of studies and a fixed total
              budget, is it better to give every study T = 50, or to give a
              third of them 100 and the rest 25, chosen by the one-shot
              selection loss of a pilot? Those two allocations cost exactly the
              same (25n + 75(n/3) = 50n), which is what makes the contrast fair.

  instrument  A saturating rating scale costs far more than the same noise on a
              scale that does not saturate. The question is whether a pilot can
              see it coming. The screen is the headroom above the cap: how far
              the best design a pilot finds sits above the instrument's ceiling,
              in units of the rating scale's own spread. If that predicts the
              damage, an experimenter can reject an instrument before spending a
              participant on it.

    python scripts/design_rules_from_pilot.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import boba_benchmarks as bb  # noqa: E402

MODEL_FREE = ("random", "sobol")
BOOT = 2000
SEED = 20260921


def boot_mean(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(x)
    draws = x[rng.integers(0, n, (BOOT, n))].mean(axis=1)
    return float(x.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_cells(root: Path, opt_z: dict[str, float]) -> pd.DataFrame:
    """Per (landscape, magnitude) deployed cost of error, as a fraction of opt_z."""
    d = pd.read_csv(root / "analysis" / "cell_means.csv")
    d = d[(~d.acquisition.isin(MODEL_FREE)) & (d.error_model == "gaussian") & (d.jitter_iteration == 0)]
    d = d.assign(deployed=d.inference_excess / d.dataset.map(opt_z))
    return d.groupby(["dataset", "jitter_std"], as_index=False).agg(
        deployed=("deployed", "mean"), trajectory=("fragility", "mean"), n=("deployed", "size"))


# ---------------------------------------------------------------------------
# 1. Sizing: spend the same trials, spread by predicted fragility
# ---------------------------------------------------------------------------


def sizing(arms: dict[int, Path], stats: dict, opt_z: dict[str, float],
           rng: np.random.Generator) -> pd.DataFrame:
    """A budget-neutral reallocation against a flat one, per magnitude.

    A third of the studies get the large budget and the rest the small one, so
    the total is exactly what a flat middle budget costs. The only question is
    who gets which, and the rule may consult the pilot but not the outcome.
    """
    cells = {T: load_cells(root, opt_z).set_index(["dataset", "jitter_std"]) for T, root in arms.items()}
    small, mid, large = sorted(arms)
    shared = set.intersection(*(set(c.index) for c in cells.values()))
    rows = []
    for std in sorted({s for _, s in shared}):
        keys = sorted(k for k in shared if k[1] == std)
        names = [d for d, _ in keys]
        frag = np.array([float(stats[d].get(f"frag_{std:g}", np.nan)) for d in names])
        if not np.all(np.isfinite(frag)):
            continue
        cost = {T: np.array([cells[T].loc[k, "deployed"] for k in keys]) for T in arms}
        n = len(keys)
        n_large = max(1, round(n / 3))

        def allocate(order: np.ndarray) -> np.ndarray:
            """Deployed cost per study when `order`'s first n_large get the large budget."""
            out = cost[small].copy()
            out[order[:n_large]] = cost[large][order[:n_large]]
            return out

        flat = cost[mid]
        by_frag = allocate(np.argsort(-frag, kind="stable"))
        # The ceiling: choose in hindsight the studies that gain most from the
        # large budget. Nothing a practitioner can do, but it bounds the rule.
        gain = cost[small] - cost[large]
        oracle = allocate(np.argsort(-gain, kind="stable"))
        draws_random = []
        for _ in range(200):
            draws_random.append(allocate(rng.permutation(n)).mean())
        for label, value in (("flat", flat), ("by_frag", by_frag), ("oracle", oracle)):
            mean, lo, hi = boot_mean(value, rng)
            rows.append({"rule": label, "jitter_std": std, "n_studies": n, "n_large": n_large,
                         "mean_deployed_cost": mean, "lo": lo, "hi": hi,
                         "gain_vs_flat": float(flat.mean() - value.mean())})
        rows.append({"rule": "random", "jitter_std": std, "n_studies": n, "n_large": n_large,
                     "mean_deployed_cost": float(np.mean(draws_random)),
                     "lo": float(np.percentile(draws_random, 2.5)),
                     "hi": float(np.percentile(draws_random, 97.5)),
                     "gain_vs_flat": float(flat.mean() - np.mean(draws_random))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Instrument: can a pilot see a ceiling coming?
# ---------------------------------------------------------------------------


def instrument(ceiling_root: Path, base_root: Path, stats: dict, opt_z: dict[str, float],
               quantile: float, rng: np.random.Generator) -> pd.DataFrame:
    """Damage done by a capped scale against the headroom a pilot would see.

    The cap sits at a quantile of the landscape, so how OFTEN it binds is fixed
    by construction. What varies, and what a pilot can measure, is how far the
    reachable optimum sits above it: that is the part of the signal the
    instrument cannot represent.
    """
    # The capped arm holds two variants in one directory, so it has no pooled
    # cell_means; the per-dataset paired metrics keep the variants apart, which
    # is exactly what this comparison needs.
    parts = []
    for path in sorted(ceiling_root.glob("*/evaluation/paired_excess_metrics.csv")):
        block = pd.read_csv(path)
        block["dataset"] = path.parent.parent.name
        parts.append(block)
    if not parts:
        raise SystemExit(f"no paired metrics under {ceiling_root}")
    cap_cells = pd.concat(parts, ignore_index=True)
    cap_cells = cap_cells[~cap_cells.acquisition.isin(MODEL_FREE)]
    cap_cells = cap_cells.rename(columns={"final_inference_simple_regret_excess_true": "inference_excess"})
    cap_cells["variant"] = cap_cells["variant"].fillna("")
    base = load_cells(base_root, opt_z).set_index(["dataset", "jitter_std"])
    rows = []
    cap_cells = cap_cells[cap_cells.jitter_iteration == 0]
    for (dataset, std, variant), block in cap_cells.groupby(["dataset", "jitter_std", "variant"]):
        if dataset not in opt_z or (dataset, std) not in base.index:
            continue
        capped = float((block.inference_excess / opt_z[dataset]).mean())
        uncapped = float(base.loc[(dataset, std), "deployed"])
        s = stats[dataset]
        # Headroom in units of the landscape's own spread: the objective is
        # z-scored, so the cap at quantile q of a standard normal is its z-score
        # and the reachable optimum is opt_z above the mean.
        from scipy.stats import norm
        cap_z = float(norm.ppf(quantile))
        rows.append({"dataset": dataset, "jitter_std": float(std), "variant": str(variant),
                     "opt_z": float(opt_z[dataset]), "cap_z": cap_z,
                     "headroom_above_cap": float(opt_z[dataset]) - cap_z,
                     "frag": float(s.get(f"frag_{std:g}", np.nan)),
                     "cost_capped": capped, "cost_uncapped": uncapped,
                     "extra_cost": capped - uncapped})
    out = pd.DataFrame(rows)
    return out


def screen_fit(out: pd.DataFrame) -> pd.DataFrame:
    """Does the pilot-visible headroom predict the extra cost the cap imposes?"""
    from scipy.stats import spearmanr
    rows = []
    for variant, block in out.groupby("variant"):
        b = block.dropna(subset=["extra_cost", "headroom_above_cap"])
        if len(b) < 5:
            continue
        rows.append({
            "variant": variant, "n_cells": len(b),
            "median_extra_cost": float(b.extra_cost.median()),
            "rho_headroom": float(spearmanr(b.headroom_above_cap, b.extra_cost).statistic),
            "rho_frag": float(spearmanr(b.frag, b.extra_cost).statistic)
            if b.frag.notna().all() else float("nan"),
            "extra_cost_low_headroom": float(b[b.headroom_above_cap <= b.headroom_above_cap.median()].extra_cost.mean()),
            "extra_cost_high_headroom": float(b[b.headroom_above_cap > b.headroom_above_cap.median()].extra_cost.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--small", type=Path, default=Path("output-boba-budget25"))
    p.add_argument("--mid", type=Path, default=Path("output-boba"))
    p.add_argument("--large", type=Path, default=Path("output-boba-budget100"))
    p.add_argument("--ceiling", type=Path, default=Path("output-boba-ceiling"))
    p.add_argument("--cap-quantile", type=float, default=0.9)
    p.add_argument("--output-dir", type=Path, default=Path("output-boba/analysis"))
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = bb.load_stats(args.stats_path)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    rng = np.random.default_rng(SEED)

    size = sizing({25: args.small, 50: args.mid, 100: args.large}, stats, opt_z, rng)
    size.to_csv(args.output_dir / "design_rule_sizing.csv", index=False)
    print("SIZING: mean deployed cost of error per study, same total trial budget")
    print("  (a third of the studies get T=100 and the rest T=25, against everyone at T=50)")
    for std, block in size.groupby("jitter_std"):
        flat = float(block[block.rule == "flat"].mean_deployed_cost.iloc[0])
        parts = []
        for rule in ("by_frag", "random", "oracle"):
            r = block[block.rule == rule]
            if len(r):
                parts.append(f"{rule} {float(r.mean_deployed_cost.iloc[0]):.4f}")
        print(f"  sigma_e {std:>5g}: flat {flat:.4f} | " + " | ".join(parts))

    inst = instrument(args.ceiling, args.mid, stats, opt_z, args.cap_quantile, rng)
    inst.to_csv(args.output_dir / "design_rule_instrument.csv", index=False)
    fit = screen_fit(inst)
    fit.to_csv(args.output_dir / "design_rule_instrument_screen.csv", index=False)
    print("\nINSTRUMENT: extra deployed cost of a capped scale, against the headroom a pilot sees")
    for _, r in fit.iterrows():
        print(f"  {r.variant:22s} n={r.n_cells:3d}  median extra cost {r.median_extra_cost:+.3f}  "
              f"rho(headroom) {r.rho_headroom:+.2f}  low-headroom {r.extra_cost_low_headroom:+.3f} "
              f"vs high {r.extra_cost_high_headroom:+.3f}")
    print(f"\nwrote {args.output_dir / 'design_rule_sizing.csv'} and two instrument files")


if __name__ == "__main__":
    main()
