"""Freeze a choice among the human-feedback methods on the screening seeds, then score it held out.

The human-feedback benchmark (scripts/human_feedback_benchmark.py, docs/HUMAN_FEEDBACK.md) runs every
method and ship rule on seeds 7-11 (configs/feedback_screen.json) and, separately, on seeds 12-16
(configs/feedback_heldout.json). This script makes the choice on the screen and writes it, with the
sha256 of the inputs it read, BEFORE any held-out run exists; `score` then applies the frozen choice to
the held-out output unchanged.

Estimand (the repository's, analyse_boba_adaptations): per run in units of opt_z, the deployed design's
regret at the final judgment; gain = reference noisy - treatment noisy, price = treatment clean -
reference clean, where the reference is the standard process (bo_logei shipping the best observed
rating); landscape means, then cell means over landscapes.

Candidates: every (method, ship rule) pair the benchmark reports, the reference included. Two choices,
both fixed here and made on seeds 7-11 only:

    single  the one procedure with the largest V = (sum over the cells of the landscape-mean gain - the
            price) / (cells + 1): the cells and the error-free case with equal weight, as in
            checks/decision_map.py. Ties go to the reference, then to the earlier candidate.
    map     per cell (error model x magnitude x onset), the procedure with the largest landscape-mean
            gain, or the reference when none gains (it pays no price without error, since a study that
            knows the cell knows when there is none).

Held-out scoring (seeds 12-16): per cell, the chosen procedure's landscape-mean gain with a
2000-resample landscape bootstrap (generator seed 20261105) and a two-sided Wilcoxon signed-rank test
over the twenty landscapes, Holm over the cells where the choice is not the reference; V of the single
procedure and of the map with bootstrap intervals; and the map's V less the single procedure's, paired
by landscape. These families are defined by this file and by no other analysis.

    python scripts/feedback_select.py freeze --screen output-feedback-screen --out configs/feedback_frozen_selection.json
    python scripts/feedback_select.py score --heldout output-feedback-heldout --selection configs/feedback_frozen_selection.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
RESPONSE = "final_deployed_regret_true"
CELL = ["error_model", "magnitude", "onset"]
SEED = 20261105
REPS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(output: Path) -> pd.DataFrame:
    """Per-landscape gains and prices of every (method, ship rule) on the deployed design."""
    p = pd.read_csv(Path(output) / "comparison_per_landscape.csv")
    p = p[p["response"] == RESPONSE].copy()
    if p.empty:
        raise ValueError(f"no {RESPONSE} rows in {output}")
    p["procedure"] = p["method"] + ":" + p["ship_rule"]
    return p


def reference_of(p: pd.DataFrame) -> str:
    refs = p["reference"].unique()
    if len(refs) != 1:
        raise ValueError(f"more than one reference: {refs}")
    return str(refs[0])


def cell_key(row) -> str:
    return f"{row['error_model']}|{float(row['magnitude']):g}|{int(row['onset'])}"


def tables(p: pd.DataFrame):
    """gain[procedure][cell] -> per-landscape Series; price[procedure] -> per-landscape Series."""
    p = p.assign(cell=p.apply(cell_key, axis=1))
    gain = {proc: {c: g.set_index("dataset")["gain"].sort_index() for c, g in block.groupby("cell")}
            for proc, block in p.groupby("procedure")}
    price = {}
    for proc, block in p.groupby("procedure"):
        per = block.groupby("dataset")["price"]
        if (per.max() - per.min()).abs().max() > 1e-12:
            raise ValueError(f"{proc}: the price differs between cells of one landscape")
        price[proc] = per.first().sort_index()
    cells = sorted({c for g in gain.values() for c in g})
    return gain, price, cells


def value(gain, price, proc, cells) -> pd.Series:
    """Per-landscape V: (sum of the cells' gains - the price) / (cells + 1)."""
    return (sum(gain[proc][c] for c in cells) - price[proc]) / (len(cells) + 1)


def map_value(gain, choice, cells) -> pd.Series:
    """The map pays no price: without error it ships by the reference."""
    return sum(gain[choice[c]][c] for c in cells) / (len(cells) + 1)


def freeze(screen: Path, out: Path) -> dict:
    p = load(screen)
    ref = reference_of(p)
    gain, price, cells = tables(p)
    procs = [ref] + sorted(x for x in gain if x != ref)
    v = {proc: float(value(gain, price, proc, cells).mean()) for proc in procs}
    single = max(procs, key=lambda x: (round(v[x], 12), x == ref, -procs.index(x)))
    choice = {}
    for c in cells:
        best, best_gain = ref, 0.0
        for proc in procs:
            g = float(gain[proc][c].mean())
            if g > best_gain + 1e-12:
                best, best_gain = proc, g
        choice[c] = best
    inputs = {name: sha256(Path(screen) / name) for name in ("comparison_per_landscape.csv", "comparison_summary.csv",
                                                             "per_run_metrics.csv", "manifest.json")}
    record = {
        "producer": "scripts/feedback_select.py freeze",
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "screen": str(Path(screen).as_posix()), "screen_inputs_sha256": inputs,
        "producer_sha256": sha256(Path(__file__)),
        "response": RESPONSE, "reference": ref, "cells": cells,
        "candidates": procs, "screen_V": v, "single": single,
        "map": choice, "map_screen_gain": {c: float(gain[choice[c]][c].mean()) for c in cells},
        "scoring": {"seeds": "12-16 (configs/feedback_heldout.json)", "bootstrap_resamples": REPS,
                    "bootstrap_seed": SEED, "test": "two-sided Wilcoxon signed-rank over landscapes",
                    "holm": "over the cells where the chosen procedure is not the reference, separately for "
                            "the single procedure and the map"},
    }
    Path(out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def boot_ci(values: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(values), (REPS, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if np.allclose(values, 0):
        return float("nan")
    return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(ps: list[float]) -> list[float]:
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = np.isfinite(ps)
    if ok.any():
        out[ok] = multipletests(ps[ok], method="holm")[1]
    return out.tolist()


def score(heldout: Path, selection: Path) -> tuple[pd.DataFrame, list[str]]:
    sel = json.loads(Path(selection).read_text(encoding="utf-8"))
    p = load(heldout)
    if reference_of(p) != sel["reference"]:
        raise ValueError("the held-out reference differs from the frozen one")
    gain, price, cells = tables(p)
    if cells != sel["cells"]:
        raise ValueError("the held-out cells differ from the frozen ones")
    missing = [x for x in {sel["single"], *sel["map"].values()} if x not in gain]
    if missing:
        raise ValueError(f"the held-out run lacks the chosen procedures {missing}")
    rng = np.random.default_rng(SEED)
    rows = []
    for policy, pick in (("single", {c: sel["single"] for c in cells}), ("map", sel["map"])):
        block = []
        for c in cells:
            g = gain[pick[c]][c]
            acts = pick[c] != sel["reference"]
            lo, hi = boot_ci(g.to_numpy(), rng)
            block.append({"policy": policy, "cell": c, "procedure": pick[c], "gain": float(g.mean()), "gain_lo": lo,
                          "gain_hi": hi, "landscapes_gaining": int((g > 0).sum()), "n_landscapes": int(len(g)),
                          "p": wilcoxon_p(g.to_numpy()) if acts else float("nan"), "acts": acts,
                          "price": float(price[pick[c]].mean())})
        adj = holm([b["p"] for b in block if b["acts"]])
        it = iter(adj)
        for b in block:
            b["p_holm"] = next(it) if b["acts"] else float("nan")
        rows += block
    frame = pd.DataFrame(rows)
    v_single = value(gain, price, sel["single"], cells)
    v_map = map_value(gain, sel["map"], cells)
    lines = [f"Frozen human-feedback selection scored held out ({Path(heldout).as_posix()}); frozen at {sel['frozen_at']}",
             f"reference {sel['reference']}; response {RESPONSE}, fraction of opt_z; {len(cells)} cells; "
             f"landscape bootstrap {REPS} resamples, seed {SEED}; Holm over acting cells per policy",
             f"single procedure chosen on seeds 7-11: {sel['single']} (screen V {sel['screen_V'][sel['single']]:+.4f})"]
    for name, v in (("single", v_single), ("map", v_map)):
        lo, hi = boot_ci(v.to_numpy(), rng)
        lines.append(f"  V {name:6s} {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  landscapes ahead of the reference "
                     f"{int((v > 0).sum())}/{len(v)}")
    d = v_map - v_single
    lo, hi = boot_ci(d.to_numpy(), rng)
    lines.append(f"  map - single {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], ahead on {int((d > 0).sum())}/{len(d)}, "
                 f"Wilcoxon p {wilcoxon_p(d.to_numpy()):.3g}")
    lines.append(f"  price of the single procedure {price[sel['single']].mean():.4f} (the map pays none)")
    for policy in ("single", "map"):
        lines.append(f"-- {policy}: per cell (gain [95% CI], landscapes gaining, Wilcoxon p, Holm p, price)")
        for _, r in frame[frame.policy == policy].iterrows():
            lines.append(f"   {r['cell']:16s} {r['procedure']:26s} {r['gain']:+.4f} [{r['gain_lo']:+.4f}, "
                         f"{r['gain_hi']:+.4f}]  {r['landscapes_gaining']:2d}/{r['n_landscapes']}  "
                         f"p {r['p']:.3g}  Holm {r['p_holm']:.3g}  price {r['price']:.4f}")
    return frame, lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--screen", default="output-feedback-screen")
    f.add_argument("--out", default="configs/feedback_frozen_selection.json")
    f.add_argument("--force", action="store_true", help="overwrite an existing frozen selection")
    s = sub.add_parser("score")
    s.add_argument("--heldout", default="output-feedback-heldout")
    s.add_argument("--selection", default="configs/feedback_frozen_selection.json")
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        out = Path(args.out)
        if out.exists() and not args.force:
            raise SystemExit(f"{out} exists; a frozen selection is not overwritten without --force")
        rec = freeze(Path(args.screen), out)
        print(f"froze {out}: single {rec['single']}; map " +
              ", ".join(f"{c} -> {x}" for c, x in rec["map"].items() if x != rec["reference"]))
        return
    frame, lines = score(Path(args.heldout), Path(args.selection))
    outdir = Path(args.heldout) / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "frozen_selection_scores.csv", index=False)
    text = "\n".join(lines) + "\n"
    (outdir / "frozen_selection_scores.txt").write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
