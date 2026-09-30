"""Each archival instrument's step and span, in units of its study's sigma_f.

The instrument arm (run_boba_gaps2.ps1, output-boba-instrument/) clips the
standardised rating to +-8 sigma and rounds it to 0.55 sigma. Its design note
derived those numbers from the three archival studies before the 2026-09-23 data
fix, which dropped the opticarvis rows logged on another scale (datasets.json
column_ranges) and reversed the sign of provoice's Predictability; both changed
sigma_f (0.374 -> 0.156 and 0.298 -> 0.449, tables/noise_anchor.tex). This
recomputes the instrument's geometry from the corrected composites and the
current sigma_f (output/noise_anchor.csv), so the paper can say what the
archival instruments are and treat the arm's clip and step as the design choice
they are.

The composite is the unweighted mean of the items in datasets.json's
``composite`` objective, signs applied, exactly as the simulator builds it
(bo_sensor_error_simulation.load_observations and compute_objective, which
also apply the column ranges). Per study it reports, in raw rating units and in
sigma_f:

  composite_step     the grid the logged composite lives on: the greatest
                     common step of its logged values, the smallest change one
                     rating can register;
  coarsest_item_step one step of the coarsest item divided by the item count,
                     the change in the composite when that item moves by one
                     of its own steps (the design note's convention: one point
                     of a seven-point item on a five-item composite);
  span_items         mean of the item maxima minus mean of the item minima, the
                     range a composite can take between its items' scale ends
                     (below);
  span_logged        the logged composite's own range;
  ceiling_above_mean the highest attainable composite (mean of item maxima)
                     above the oracle's average design (mean_f), which is how
                     far a rating can rise above an average design;
  floor_below_mean   the same distance down to the lowest attainable composite.

Items' scale ends are the column range datasets.json declares (opticarvis:
[-1, 1] on every item), otherwise the extremes the logged ratings reach (ehmi
and provoice declare none). A nominal scale wider than any rating reached would
widen span_items and floor_below_mean, and would also raise ceiling_above_mean
if an item's better end lay beyond its logged extreme; it never changes the
steps. The printout gives the number of ratings at each logged end, which says
which ends are the scale's own. ehmi's items reach 1 to 5, -3 to 3 and 1 to 7,
the ends of five- and seven-point scales, with a pile-up at the better end of
each. provoice's Mental Demand is logged on 1 to 17 while the study's own
ParetoFront.R plots it on 0 to 20: 237 of its 532 ratings sit at 1, its better
end, which reads as the scale's floor, so provoice's ceiling stands as logged,
but only one sits at 17, so its span_items and floor_below_mean are lower
bounds.

The printout also says where the arm's own clip meets the twenty landscapes.
The clip is +-8 in each landscape's standardised units, so it is centred on the
landscape mean, not placed at the optimum: it lies below a landscape's optimum
only where opt_z > 8, and above its sampled minimum only where that is more
than 8 below the mean (output-boba-instrument/run_metadata.json, whose logged
--response-clip and --response-round must be the arm settings used here).

    python scripts/review_checks/instrument_scale.py
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import bo_sensor_error_simulation as sim  # noqa: E402

DATASETS = REPO / "datasets.json"
ANCHOR = REPO / "output" / "noise_anchor.csv"
CACHE = REPO / ".dataset_cache"
OUT = REPO / "output-boba" / "analysis" / "review" / "instrument_scale.csv"
ARM_META = REPO / "output-boba-instrument" / "run_metadata.json"
OBJECTIVE = "composite"
# The instrument arm's own settings (run_boba_gaps2.ps1: --response-clip=-8,8 --response-round 0.55).
ARM_CLIP = 8.0
ARM_STEP = 0.55
MAX_DENOMINATOR = 10_000
TOL = 1e-6


def as_fraction(x: float) -> Fraction:
    f = Fraction(float(x)).limit_denominator(MAX_DENOMINATOR)
    if abs(float(f) - float(x)) > TOL:
        raise ValueError(f"{x!r} is not on a rational grid with denominator <= {MAX_DENOMINATOR}")
    return f


def fraction_gcd(values: list[Fraction]) -> Fraction:
    """The largest g with every value an integer multiple of g (0 for no non-zero value)."""
    values = [abs(v) for v in values if v != 0]
    if not values:
        return Fraction(0)
    den = reduce(lambda a, b: a * b // gcd(a, b), (v.denominator for v in values))
    return Fraction(reduce(gcd, (int(v * den) for v in values)), den)


def grid_step(values: np.ndarray) -> float:
    """The greatest common step of a set of values on a rational grid (0 for a constant)."""
    fracs = sorted({as_fraction(v) for v in np.asarray(values, dtype=float)})
    return float(fraction_gcd([b - a for a, b in zip(fracs, fracs[1:])]))


def end_counts(values: np.ndarray) -> tuple[float, float, int, int]:
    """(logged minimum, logged maximum, ratings at the minimum, ratings at the maximum)."""
    v = np.asarray(values, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    return lo, hi, int(np.sum(v == lo)), int(np.sum(v == hi))


def item_ends(datasets_path: Path = DATASETS, cache_dir: Path = CACHE) -> pd.DataFrame:
    """Per study and item (signs applied, so the maximum is the better end): the
    logged range, the declared range if any, and the ratings at each logged end."""
    rows = []
    for ds in sim.parse_dataset_configs(None, datasets_path, cache_dir):
        if OBJECTIVE not in ds.objective_map:
            continue
        frame = sim.load_observations(ds, OBJECTIVE, None, None)
        cols = ds.objective_map[OBJECTIVE]
        items = sim._extract_objective_values(frame, cols)
        for j, col in enumerate(cols):
            lo, hi, n_lo, n_hi = end_counts(items[:, j])
            base = col.lstrip("-")
            declared = ds.column_ranges.get(base) if ds.column_ranges else None
            if declared is not None:
                sign = -1.0 if col.startswith("-") else 1.0
                declared = tuple(sorted(sign * float(v) for v in declared))
            rows.append({"dataset": ds.name, "item": col, "n_ratings": int(len(frame)),
                         "logged_min": lo, "logged_max": hi, "n_at_min": n_lo, "n_at_max": n_hi,
                         "declared": declared})
    return pd.DataFrame(rows)


def instrument_rows(datasets_path: Path = DATASETS, anchor_path: Path = ANCHOR,
                    cache_dir: Path = CACHE) -> pd.DataFrame:
    anchor = pd.read_csv(anchor_path)
    anchor = anchor[anchor["objective"] == OBJECTIVE].set_index("dataset")
    rows = []
    for ds in sim.parse_dataset_configs(None, datasets_path, cache_dir):
        if ds.name not in anchor.index or OBJECTIVE not in ds.objective_map:
            continue
        frame = sim.load_observations(ds, OBJECTIVE, None, None)
        cols = ds.objective_map[OBJECTIVE]
        items = sim._extract_objective_values(frame, cols)          # signs applied, one column per item
        composite = sim.compute_objective(frame, cols, normalize=False, weights=None).to_numpy(float)
        n = len(cols)
        item_lo, item_hi = items.min(axis=0), items.max(axis=0)
        # A declared column range is the item's nominal scale; it overrides the logged extremes.
        for j, col in enumerate(cols):
            base = col.lstrip("-")
            if base in ds.column_ranges:
                sign = -1.0 if col.startswith("-") else 1.0
                lo, hi = sorted(sign * float(v) for v in ds.column_ranges[base])
                item_lo[j], item_hi[j] = min(lo, item_lo[j]), max(hi, item_hi[j])
        item_steps = np.array([grid_step(items[:, j]) for j in range(n)])
        step = grid_step(composite)
        # The composite grid must be the items' common step over n, or the composite is not their mean.
        implied = float(fraction_gcd([as_fraction(s) / n for s in item_steps]))
        a = anchor.loc[ds.name]
        sigma_f, mean_f = float(a["sigma_f"]), float(a["mean_f"])
        top, bottom = float(item_hi.mean()), float(item_lo.mean())
        raw = {
            "composite_step": step,
            "coarsest_item_step": float(item_steps.max() / n),
            "finest_item_step": float(item_steps.min() / n),
            "span_items": top - bottom,
            "span_logged": float(composite.max() - composite.min()),
            "ceiling_above_mean": top - mean_f,
            "floor_below_mean": mean_f - bottom,
        }
        rows.append({
            "dataset": ds.name, "n_ratings": int(len(frame)), "n_items": n,
            "items": ";".join(cols),
            "item_steps": ";".join(f"{s:.6g}" for s in item_steps),
            "item_min": ";".join(f"{v:.6g}" for v in item_lo),
            "item_max": ";".join(f"{v:.6g}" for v in item_hi),
            "composite_min_attainable": bottom, "composite_max_attainable": top,
            "composite_min_logged": float(composite.min()), "composite_max_logged": float(composite.max()),
            "composite_step_implied_by_items": implied,
            "sigma_f": sigma_f, "mean_f": mean_f, "y_opt": float(a["y_opt"]), "opt_z": float(a["opt_z"]),
            **{f"{k}_raw": v for k, v in raw.items()},
            **{f"{k}_sigma": v / sigma_f for k, v in raw.items()},
            "arm_clip_sigma": ARM_CLIP, "arm_step_sigma": ARM_STEP,
        })
    return pd.DataFrame(rows)


def clip_reach(meta_path: Path = ARM_META, clip: float = ARM_CLIP, step: float = ARM_STEP) -> dict:
    """Where the arm's +-clip, in each landscape's standardised units, meets the true landscape.

    Returns the landscapes whose optimum lies above +clip (opt_z > clip) and whose
    sampled minimum lies below -clip. The arm's run_metadata.json must have run
    with this clip and step, or the arm is not the one described.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    args = meta.get("args", {})
    logged_clip = [float(v) for v in str(args.get("response_clip", "")).split(",") if v.strip()]
    logged_step = args.get("response_round")
    if logged_clip != [-clip, clip] or logged_step is None or abs(float(logged_step) - step) > 1e-12:
        raise SystemExit(f"{meta_path} ran --response-clip {args.get('response_clip')!r} --response-round "
                         f"{logged_step!r}, not +-{clip:g} and {step:g}")
    stats = meta["landscape_stats"]
    above = sorted((k for k, v in stats.items() if float(v["opt_z"]) > clip), key=lambda k: float(stats[k]["opt_z"]))
    below = sorted(k for k, v in stats.items()
                   if (float(v["min"]) - float(v["mean"])) / float(v["std"]) < -clip)
    return {"n_landscapes": len(stats), "optimum_above_clip": above,
            "opt_z_above_clip": [float(stats[k]["opt_z"]) for k in above], "minimum_below_clip": below}


def main() -> None:
    table = instrument_rows()
    ends = item_ends()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"composites: datasets.json '{OBJECTIVE}' (column ranges applied); sigma_f: {ANCHOR.relative_to(REPO).as_posix()}\n")
    for r in table.itertuples():
        mine = ends[ends["dataset"] == r.dataset]
        print(f"{r.dataset}: {r.n_ratings} ratings, {r.n_items} items ({r.items})")
        print(f"  item steps {r.item_steps}; item ranges [{r.item_min}] to [{r.item_max}]")
        for e in mine.itertuples():
            declared = f"declared [{e.declared[0]:g}, {e.declared[1]:g}]" if e.declared is not None else "no declared range"
            print(f"    {e.item}: logged [{e.logged_min:g}, {e.logged_max:g}], {declared}; ratings at the logged "
                  f"worse end {e.n_at_min}, at the better end {e.n_at_max} of {e.n_ratings}")
        print(f"  sigma_f {r.sigma_f:.4f}, mean_f {r.mean_f:.4f}, y_opt {r.y_opt:.4f}, opt_z {r.opt_z:.3f}")
        print(f"  composite step          {r.composite_step_raw:.6g} raw = {r.composite_step_sigma:.3f} sigma "
              f"(implied by the items {r.composite_step_implied_by_items:.6g})")
        print(f"  coarsest-item step      {r.coarsest_item_step_raw:.6g} raw = {r.coarsest_item_step_sigma:.3f} sigma "
              f"(finest {r.finest_item_step_sigma:.3f} sigma)")
        print(f"  span over item extremes {r.span_items_raw:.4g} raw = {r.span_items_sigma:.2f} sigma "
              f"({r.composite_min_attainable:.4g} to {r.composite_max_attainable:.4g})")
        print(f"  span of logged composite {r.span_logged_raw:.4g} raw = {r.span_logged_sigma:.2f} sigma")
        print(f"  ceiling above the average design {r.ceiling_above_mean_sigma:.2f} sigma, "
              f"floor below it {r.floor_below_mean_sigma:.2f} sigma\n")
    print(f"the arm: clip +-{ARM_CLIP:g} sigma about the landscape mean (span {2 * ARM_CLIP:g} sigma), "
          f"step {ARM_STEP:g} sigma")
    reach = clip_reach()
    n, above = reach["n_landscapes"], reach["optimum_above_clip"]
    shown = ", ".join(f"{k} {z:.2f}" for k, z in zip(above, reach["opt_z_above_clip"]))
    print(f"  its top lies above the optimum on {n - len(above)} of {n} landscapes; "
          f"below it on {len(above)} (opt_z: {shown})")
    print(f"  its bottom lies above the sampled minimum on {len(reach['minimum_below_clip'])} of {n} "
          f"({', '.join(reach['minimum_below_clip']) or 'none'}); {ARM_META.relative_to(REPO).as_posix()}")
    print(f"\nWrote {OUT.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
