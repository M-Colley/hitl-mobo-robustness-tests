"""The final sitting against the ship rules that buy no trial, side by side.

The sitting ranks its candidates by the posterior mean less one latent SD
(LCB1) and ships the best look. Shipping the LCB1 design after all 50 trials
costs no trial at all, so the sitting's gain over the standard process has to be
read beside LCB1's on the same runs. This report puts the two side by side, in
every scope the paper quotes, for the four error processes pooled and for
gaussian error alone:

1. 1 sigma from the first rating, k = 2, on seeds 7-11, 12-16, 7-16 and the
   untouched seeds 27-36 (gaussian only; the replication arm ran gaussian error);
2. 5 sigma from trial 21, k = 16 (and k = 12, the largest k replayed on 27-36),
   with the increment by error process;
3. the other cells the text quotes (5 sigma from trial 1, k = 8; 1 sigma from
   trial 21);
4. the pooled k-curve, and k = 12 pooled without the 5 sigma, trial-21 cell;
5. Holm's correction for the increment over LCB1: over the nine k within a cell,
   over the eight (cell, k chosen on seeds 7-11) and over all 72 (cell, k);
6. the clean-twin prices of the sittings and of the rules;
7. the look model: the increment over LCB1 in the cells above when the drift
   ramp and the AR(1) state move on through the sitting (the sequential
   replays, looks in random order or in rank order), by process;
8. the other two zero-trial rules: the sitting's increment over LCB2 (the
   cautious rule) and over PM in the same cells, with the same Holm families,
   and over the rule with the largest gain on seeds 7-11 (the seeds that choose
   k), on the main sweep, on seeds 27-36 and under the sequential look models.

It computes nothing of its own on the main sweep: it reads the outputs of
``scripts/sitting_by_magnitude.py`` (the four processes), of
``scripts/sitting_by_magnitude.py --processes gaussian --tag gaussian`` and, for
section 7, of the ``--tag seq`` and ``--tag seqrank`` rescorings (the commands
are in that script's docstring), so every number equals its producer's. The seeds 27-36 values are
``fresh_seed_replication.reference_table``, which uses its own fixed generator,
so they equal the lines of ``register_checks/fresh_seed_replication.txt``.
Estimands: gains over the standard process in units of opt_z; the increment is
the per-run paired difference (regret of LCB1 - regret of the sitting) / opt_z,
averaged per landscape, landscape bootstrap; price = (trt_clean - ref_clean) /
opt_z. Writes output-boba/analysis/review/register_checks/sitting_vs_shiprule.txt.

    python scripts/sitting_by_magnitude.py
    python scripts/sitting_by_magnitude.py --processes gaussian --tag gaussian
    python scripts/sitting_by_magnitude.py --replay-dirs end_of_study_ksweep_seq,end_of_study_kwide_seq --tag seq
    python scripts/sitting_by_magnitude.py --tag seqrank --replay-dirs end_of_study_ksweep_seqrank=drift+ar1,end_of_study_kwide_seqrank=drift+ar1,end_of_study_ksweep=gaussian+bias,end_of_study_kwide=gaussian+bias
    python scripts/review_checks/sitting_vs_shiprule.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REVIEW = REPO / "output-boba" / "analysis" / "review"
OUT = REVIEW / "register_checks" / "sitting_vs_shiprule.txt"
SCOPES = (("four processes", ""), ("gaussian", "_gaussian"))
SEED_LABEL = {"train": "7-11", "test": "12-16", "all": "7-16"}


def _load_fresh_module():
    spec = importlib.util.spec_from_file_location("fresh_seed_replication",
                                                  Path(__file__).resolve().parent / "fresh_seed_replication.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_scope(suffix: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    base = REVIEW / f"sitting_by_magnitude{suffix}"
    paths = [Path(f"{base}.csv"), Path(f"{base}_ship_rules.csv"), Path(f"{base}_selection.json")]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"missing {p}; run scripts/sitting_by_magnitude.py"
                             + (" --processes gaussian --tag gaussian" if suffix else "") + " first")
    frame = pd.read_csv(paths[0])
    if "inc_lcb1_test" not in frame:
        raise SystemExit(f"{paths[0]} predates the ship-rule reference; rerun scripts/sitting_by_magnitude.py")
    return frame, pd.read_csv(paths[1]), json.loads(paths[2].read_text())


def f4(v: float, lo: float | None = None, hi: float | None = None) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:+.4f}" + ("" if lo is None else f" [{lo:+.4f}, {hi:+.4f}]")


def pct(v: float) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{100 * v:+.1f}%"


def row_of(frame: pd.DataFrame, cell: str, k: int) -> pd.Series:
    r = frame[(frame["cell"] == cell) & (frame["k"] == k)]
    if len(r) != 1:
        raise KeyError(f"no row for {cell}, k = {k}")
    return r.iloc[0]


def rule_of(rules: pd.DataFrame, cell: str, rule: str, seeds: str) -> pd.Series:
    r = rules[(rules["cell"] == cell) & (rules["rule"] == rule) & (rules["seeds"] == seeds)]
    if len(r) != 1:
        raise KeyError(f"no rule row for {cell}, {rule}, {seeds}")
    return r.iloc[0]


def main_sweep_block(frame: pd.DataFrame, rules: pd.DataFrame, cell: str, k: int, scope: str,
                     sel: list[dict] | None = None) -> list[str]:
    r = row_of(frame, cell, k)
    # Where k is the cell's chosen k, the held-out gain and its interval are the selection summary's, which is
    # what tables/sitting_by_magnitude.tex prints (the per-k rows draw their own interval for the same mean).
    s = next((x for x in (sel or []) if x.get("cell") == cell and x.get("k_chosen_on_7_11") == k), None)
    out = []
    for seeds in ("train", "test", "all"):
        lcb1, pm, lcb2 = (rule_of(rules, cell, name, seeds) for name in ("lcb1", "pm", "lcb2"))
        if seeds == "train":
            sit, inc = f4(r["gain_train"]), f4(r["inc_lcb1_train"], r["inc_lcb1_train_lo"], r["inc_lcb1_train_hi"])
            extra = "(seeds 7-11 chose k; the sitting's gain there is a mean only)"
        elif seeds == "test":
            sit = (f4(s["test_gain"], s["test_lo"], s["test_hi"]) + " (the table's interval)" if s is not None
                   else f4(r["gain_test"], r["gain_test_lo"], r["gain_test_hi"]))
            inc = (f"{f4(r['inc_lcb1_test'], r['inc_lcb1_test_lo'], r['inc_lcb1_test_hi'])} ahead "
                   f"{int(r['inc_lcb1_test_landscapes_ahead'])}/20 p {r['inc_lcb1_p_test']:.3g} "
                   f"(Holm over k {r['inc_lcb1_p_test_holm_over_k']:.3g})")
            # looks alone - increment = LCB1 after all T trials less the top candidate of the T - k prefix
            extra = (f"looks alone {f4(r['inc_top_test'], r['inc_top_test_lo'], r['inc_top_test_hi'])} (so the k search "
                     f"trials add {r['inc_top_test'] - r['inc_lcb1_test']:+.4f} to LCB1); "
                     f"price {r['price_test']:.4f}, LCB1 {r['lcb1_price_test']:.4f}")
        else:
            sit = f4(r["gain_all"], r["gain_all_lo"], r["gain_all_hi"])
            inc = (f"{f4(r['inc_lcb1_all'], r['inc_lcb1_all_lo'], r['inc_lcb1_all_hi'])} ahead "
                   f"{int(r['inc_lcb1_all_landscapes_ahead'])}/20 p {r['inc_lcb1_p_all']:.3g}")
            extra = f"looks alone {f4(r['inc_top_all'])}; price {r['price_all']:.4f}, LCB1 {r['lcb1_price_all']:.4f}"
        share = r["recovered_test"] if seeds == "test" else (r["recovered_all"] if seeds == "all" else np.nan)
        share_txt = f" ({pct(share)} of the cost)"
        if seeds == "test" and s is not None:
            share_txt = (f" ({pct(s['test_recovered_share'])} [{100 * s['test_recovered_lo']:.1f}, "
                         f"{100 * s['test_recovered_hi']:.1f}] of the cost)")
        out.append(f"   {scope:15s} seeds {SEED_LABEL[seeds]:5s}  cost {lcb1['cost']:.4f}  sitting {sit}"
                   + (share_txt if np.isfinite(share) and lcb1["cost_positive_everywhere"] else ""))
        lcb1_share = pct(lcb1["share"])
        if np.isfinite(lcb1["share"]):
            lcb1_share += f" [{100 * lcb1['share_lo']:.1f}, {100 * lcb1['share_hi']:.1f}]"
        out.append(f"   {'':15s}              LCB1 {f4(lcb1['gain'], lcb1['gain_lo'], lcb1['gain_hi'])} "
                   f"({lcb1_share}; price {lcb1['price']:.4f})  PM {f4(pm['gain'])}  LCB2 {f4(lcb2['gain'])}")
        out.append(f"   {'':15s}              sitting - LCB1 {inc}")
        out.append(f"   {'':15s}              {extra}")
    return out


def fresh_block(inc: pd.DataFrame, rules: pd.DataFrame, cell: str, k: int) -> list[str]:
    r = inc[(inc["cell"] == cell) & (inc["k"] == k)]
    if not len(r):
        return [f"   gaussian        seeds 27-36  k = {k} was not replayed on these seeds"]
    r = r.iloc[0]
    lcb1, pm, lcb2 = (rules[(rules["cell"] == cell) & (rules["rule"] == n)].iloc[0] for n in ("lcb1", "pm", "lcb2"))
    return [f"   {'gaussian':15s} seeds 27-36  cost {lcb1['cost']:.4f}  sitting {f4(r['gain'], r['gain_lo'], r['gain_hi'])}"
            + (f" ({pct(r['share'])} of the cost)" if np.isfinite(r["share"]) else ""),
            f"   {'':15s}              LCB1 {f4(lcb1['gain'], lcb1['gain_lo'], lcb1['gain_hi'])} "
            f"({pct(lcb1['share'])})  PM {f4(pm['gain'])}  LCB2 {f4(lcb2['gain'])}",
            f"   {'':15s}              sitting - LCB1 {f4(r['inc_lcb1'], r['inc_lcb1_lo'], r['inc_lcb1_hi'])} ahead "
            f"{int(r['inc_lcb1_landscapes_ahead'])}/20 p {r['inc_lcb1_p']:.3g} (Holm over the three k "
            f"{r['inc_lcb1_p_holm_over_k']:.3g})",
            f"   {'':15s}              looks alone {f4(r['inc_top'], r['inc_top_lo'], r['inc_top_hi'])}; "
            f"price {r['price']:.4f}, LCB1 {lcb1['price']:.4f}"]


def report() -> list[str]:
    fresh_mod = _load_fresh_module()
    scopes = {name: read_scope(suffix) for name, suffix in SCOPES}
    oz = fresh_mod.opt_z()
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file", "family", "k",
            "candidates", "rho", "winner", "regret_noisy", "ref_noisy", "ref_clean", "regret_clean",
            "top_candidate_regret_noisy"]
    fresh = pd.read_csv(fresh_mod.FRESH / "end_of_study_fresh" / "end_of_study_per_run.csv.gz", usecols=cols,
                        low_memory=False)
    ship = pd.read_csv(fresh_mod.FRESH / "ship_rules_fresh" / "ship_rules_per_run.csv", low_memory=False)
    f_rules, f_inc, f_prices = fresh_mod.reference_table(fresh, ship, oz)

    L = ["The final sitting against the ship rules that buy no trial (LogEI and qNEI, looks at the full SD, rho = 1)",
         "Gains over the standard process in units of opt_z, landscape means with a landscape bootstrap; sitting - LCB1",
         "is the paired per-run difference; 'looks alone' is the sitting over the top candidate of its own T - k prefix;",
         "price is the regret added to the clean twin. LCB1, PM, LCB2 ship after all 50 trials with no extra trial.",
         "Two sitting gains below are also printed by fresh_seed_replication.txt's first block, with intervals from",
         "its own generator: gaussian error at 1 sigma from the first rating, k = 2, on seeds 12-16 and on seeds",
         "27-36. The means are the same; the paper quotes that block's intervals for those two gains.",
         ""]
    L.append("== 1. 1 sigma from the first rating, k = 2")
    for name, (frame, rules, sel) in scopes.items():
        L += main_sweep_block(frame, rules, "1sigma_from_trial_1", 2, name, sel)
    L += fresh_block(f_inc, f_rules, "1sigma_from_trial_1", 2)
    L.append("   same cell, other k on seeds 27-36 (gaussian):")
    for k in (5, 12):
        r = f_inc[(f_inc["cell"] == "1sigma_from_trial_1") & (f_inc["k"] == k)].iloc[0]
        L.append(f"      k = {k:2d}  sitting {f4(r['gain'], r['gain_lo'], r['gain_hi'])}  sitting - LCB1 "
                 f"{f4(r['inc_lcb1'], r['inc_lcb1_lo'], r['inc_lcb1_hi'])} ahead {int(r['inc_lcb1_landscapes_ahead'])}/20")

    L.append("")
    L.append("== 2. 5 sigma from trial 21, k = 16 (the k chosen on seeds 7-11) and k = 12")
    for k in (16, 12):
        L.append(f"-- k = {k}")
        for name, (frame, rules, sel) in scopes.items():
            L += main_sweep_block(frame, rules, "5sigma_from_trial_21", k, name, sel)
        L += fresh_block(f_inc, f_rules, "5sigma_from_trial_21", k)
    for name, (_, _, sel) in scopes.items():
        s = next(x for x in sel if x.get("cell") == "5sigma_from_trial_21")
        L.append(f"   {name}: k chosen on seeds 7-11 = {s['k_chosen_on_7_11']}; on seeds 12-16, by process, "
                 f"sitting - LCB1 {({p: round(v, 4) for p, v in s['per_process_increment_over_lcb1_test'].items()})}, "
                 f"LCB1 gain {({p: round(v, 4) for p, v in s['per_process_lcb1_gain_test'].items()})}, "
                 f"sitting gain {({p: round(v, 4) for p, v in s['per_process_test'].items()})}")

    L.append("")
    L.append("== 3. other cells the text quotes")
    for cell, ks in (("5sigma_from_trial_1", (8,)), ("1sigma_from_trial_21", (3, 5, 8, 12, 16))):
        for k in ks:
            L.append(f"-- {cell}, k = {k}")
            for name, (frame, rules, sel) in scopes.items():
                L += main_sweep_block(frame, rules, cell, k, name, sel)

    L.append("")
    L.append("== 4. the pooled k-curve (every magnitude and onset)")
    L.append("   scope           k   gain 7-16                  gain 12-16                 LCB1 7-16 / 12-16   "
             "sitting - LCB1, 7-16          sitting - LCB1, 12-16")
    for name, (frame, rules, _) in scopes.items():
        lall, lte = rule_of(rules, "pooled", "lcb1", "all"), rule_of(rules, "pooled", "lcb1", "test")
        for k in sorted(frame.loc[frame["cell"] == "pooled", "k"].unique()):
            r = row_of(frame, "pooled", int(k))
            L.append(f"   {name:15s} {int(k):2d}  {f4(r['gain_all'], r['gain_all_lo'], r['gain_all_hi'])}  "
                     f"{f4(r['gain_test'], r['gain_test_lo'], r['gain_test_hi'])}  {lall['gain']:+.4f} / {lte['gain']:+.4f}  "
                     f"{f4(r['inc_lcb1_all'], r['inc_lcb1_all_lo'], r['inc_lcb1_all_hi'])}  "
                     f"{f4(r['inc_lcb1_test'], r['inc_lcb1_test_lo'], r['inc_lcb1_test_hi'])}")
    for k in sorted(f_inc.loc[f_inc["cell"] == "pooled", "k"].unique()):
        r = f_inc[(f_inc["cell"] == "pooled") & (f_inc["k"] == k)].iloc[0]
        lr = f_rules[(f_rules["cell"] == "pooled") & (f_rules["rule"] == "lcb1")].iloc[0]
        L.append(f"   gaussian 27-36  {int(k):2d}  {f4(r['gain'], r['gain_lo'], r['gain_hi'])}  LCB1 {lr['gain']:+.4f}  "
                 f"sitting - LCB1 {f4(r['inc_lcb1'], r['inc_lcb1_lo'], r['inc_lcb1_hi'])}")
    for name, (_, _, sel) in scopes.items():
        w = next(x for x in sel if x.get("cell") == "pooled_k12_without_5sigma_from_trial_21")
        for seeds in ("all_seeds", "seeds_12_16"):
            x = w[seeds]
            i = x["increment_over_lcb1"]
            L.append(f"   {name}, k = 12 pooled without 5 sigma from trial 21, {seeds}: gain {f4(x['gain'], x['lo'], x['hi'])}, "
                     f"LCB1 {x['lcb1_gain']:+.4f}, sitting - LCB1 {f4(i['gain'], i['lo'], i['hi'])} "
                     f"ahead {i['landscapes_ahead']}/20 p {i['p']:.3g}")

    L.append("")
    L.append("== 5. Holm's correction for the increment over LCB1 on seeds 12-16 (two-sided Wilcoxon over 20 landscapes)")
    for name, (frame, _, sel) in scopes.items():
        L.append(f"-- {name}")
        for s in sel:
            if "k_chosen_on_7_11" not in s or s["cell"] == "pooled":
                continue
            i = s["increment_over_lcb1"]
            L.append(f"   {s['cell']:24s} k = {i['k']:2d}  sitting - LCB1 {f4(i['test'], i['test_lo'], i['test_hi'])}  "
                     f"p {i['test_p']:.3g}  Holm over k {i['test_p_holm_over_k']:.3g}  over the 8 chosen "
                     f"{i['test_p_holm_chosen_cells']:.3g}  over all 72 {i['test_p_holm_all_cells']:.3g}   "
                     f"(gain over the standard process: Holm over k {s['test_p_holm_over_k']:.3g})")
        for cell, k in (("1sigma_from_trial_1", 2), ("5sigma_from_trial_21", 16)):
            r = row_of(frame, cell, k)
            L.append(f"   fixed {cell}, k = {k}: p {r['inc_lcb1_p_test']:.3g}, Holm over k "
                     f"{r['inc_lcb1_p_test_holm_over_k']:.3g}, over all 72 {r['inc_lcb1_p_test_holm_all_cells']:.3g}")

    L.append("")
    L.append("== 6. clean-twin prices, (trt_clean - ref_clean) / opt_z, landscape mean (the clean twin is the same for")
    L.append("   every error process, so the two scopes give one price)")
    _, _, sel = scopes["four processes"]
    prices = next(x for x in sel if x.get("cell") == "clean_twin_prices")
    for k, v in prices["sitting"].items():
        L.append(f"   sitting k = {int(k):2d}  seeds 7-16 {f4(v['all']['mean'], v['all']['lo'], v['all']['hi'])}  "
                 f"seeds 7-11 {f4(v['train']['mean'])}  seeds 12-16 {f4(v['test']['mean'], v['test']['lo'], v['test']['hi'])}")
    for rule, v in prices["ship_rules"].items():
        L.append(f"   {rule.upper():12s}  seeds 7-16 {f4(v['all']['mean'], v['all']['lo'], v['all']['hi'])}  "
                 f"seeds 7-11 {f4(v['train']['mean'])}  seeds 12-16 {f4(v['test']['mean'], v['test']['lo'], v['test']['hi'])}")
    L.append("   seeds 27-36:")
    for _, r in f_prices.iterrows():
        L.append(f"   {r['procedure']:14s} {f4(r['price'], r['lo'], r['hi'])}")
    g = scopes["gaussian"][2]
    gp = next(x for x in g if x.get("cell") == "clean_twin_prices")
    same = all(abs(gp["sitting"][k]["all"]["mean"] - prices["sitting"][k]["all"]["mean"]) < 1e-12 for k in gp["sitting"])
    L.append(f"   the gaussian-only outputs give the same prices: {same}")
    L.append("")
    L += look_model_block(scopes["four processes"][0])
    L.append("")
    L += other_rules_block(scopes, fresh_mod.rule_increments(fresh, ship, oz))
    return L


OTHER_CELLS = (("1sigma_from_trial_1", (2, 12, 16)), ("5sigma_from_trial_21", (16, 12)), ("5sigma_from_trial_1", (8,)),
               ("1sigma_from_trial_21", (3, 5, 8, 12)), ("pooled", (12,)))


def _inc_line(r: pd.Series, rule: str) -> str:
    c = f"inc_{rule}"
    return (f"sitting - {rule.upper():4s} 7-11 {f4(r[f'{c}_train'], r[f'{c}_train_lo'], r[f'{c}_train_hi'])}  "
            f"12-16 {f4(r[f'{c}_test'], r[f'{c}_test_lo'], r[f'{c}_test_hi'])} ahead "
            f"{int(r[f'{c}_test_landscapes_ahead'])}/20 p {r[f'{c}_p_test']:.3g} (Holm over k "
            f"{r[f'{c}_p_test_holm_over_k']:.3g}, over the 8 chosen {r[f'{c}_p_test_holm_chosen_cells']:.3g}, over all 72 "
            f"{r[f'{c}_p_test_holm_all_cells']:.3g})  7-16 {f4(r[f'{c}_all'], r[f'{c}_all_lo'], r[f'{c}_all_hi'])}")


def other_rules_block(scopes: dict, f_other: pd.DataFrame) -> list[str]:
    """Section 8: the sitting over LCB2 and PM, and over the zero-trial rule chosen on seeds 7-11."""
    L = ["== 8. the other two zero-trial rules. LCB1 ranks the sitting's candidates, but it is not always the strongest",
         "   rule that buys no trial: LCB2 (the cautious rule, posterior mean less two latent SDs) gains more in some cells.",
         "   The sitting's increment over LCB2 and over PM, the same paired estimand and Holm families as over LCB1; then",
         "   the rule with the largest gain on seeds 7-11, the seeds that choose k, and the sitting over it on seeds 12-16."]
    for cell, ks in OTHER_CELLS:
        L.append(f"-- {cell}")
        for name, (frame, rules, _) in scopes.items():
            gains = "  ".join(f"{rule.upper()} " + " / ".join(f"{rule_of(rules, cell, rule, s)['gain']:+.4f}"
                                                              for s in ("train", "test", "all"))
                              for rule in ("lcb1", "lcb2", "pm"))
            L.append(f"   {name:15s} rule gains 7-11 / 12-16 / 7-16: {gains}")
            for k in ks:
                r = row_of(frame, cell, k)
                for rule in ("lcb1",) + tuple(x for x in ("lcb2", "pm")):
                    if rule == "lcb1":
                        line = (f"sitting - LCB1 7-11 {f4(r['inc_lcb1_train'], r['inc_lcb1_train_lo'], r['inc_lcb1_train_hi'])}"
                                f"  12-16 {f4(r['inc_lcb1_test'], r['inc_lcb1_test_lo'], r['inc_lcb1_test_hi'])}")
                    else:
                        line = _inc_line(r, rule)
                    L.append(f"   {'':15s} k = {k:2d}  {line}")
        f = f_other[f_other["cell"] == cell]
        for k in ks:
            r = f[f["k"] == k]
            if not len(r):
                L.append(f"   {'gaussian 27-36':15s} k = {k:2d}  not replayed on these seeds")
                continue
            r = r.iloc[0]
            L.append(f"   {'gaussian 27-36':15s} k = {k:2d}  " + "  ".join(
                f"sitting - {rule.upper()} ({rule.upper()} gain {r[f'{rule}_gain']:+.4f}) "
                f"{f4(r[f'inc_{rule}'], r[f'inc_{rule}_lo'], r[f'inc_{rule}_hi'])} ahead "
                f"{int(r[f'inc_{rule}_landscapes_ahead'])}/20 p {r[f'inc_{rule}_p']:.3g} (Holm over the three k "
                f"{r[f'inc_{rule}_p_holm_over_k']:.3g}, over the 24 {r[f'inc_{rule}_p_holm_all_cells']:.3g})"
                for rule in ("lcb2", "pm")))
    L.append("-- the zero-trial rule chosen on seeds 7-11, and the chosen sitting over it on seeds 12-16")
    models = [(name, sel) for name, (_, _, sel) in scopes.items()]
    for name, suffix in LOOK_MODELS:
        path = REVIEW / f"sitting_by_magnitude{suffix}_selection.json"
        if path.is_file():
            models.append((f"four processes, {name}", json.loads(path.read_text())))
    for name, sel in models:
        L.append(f"   {name}")
        for s in sel:
            c = s.get("increment_over_chosen_rule")
            if c is None:
                continue
            holm8 = c.get("test_p_holm_chosen_cells")
            L.append(f"      {s['cell']:24s} k = {c['k']:2d}  rule {c['rule'].upper():4s} (gain on 7-11 "
                     f"{c['rule_gain_train']:+.4f})  sitting - rule {f4(c['test'], c['test_lo'], c['test_hi'])} ahead "
                     f"{c['test_landscapes_ahead']}/20 p {c['test_p']:.3g}"
                     + (f" (Holm over the 8 cells {holm8:.3g})" if holm8 is not None else ""))
    return L


LOOK_MODELS = (("sequential, random order", "_seq"), ("sequential, rank order", "_seqrank"))
LOOK_CELLS = (("1sigma_from_trial_1", (2, 12, 16)), ("5sigma_from_trial_21", (8, 12, 16)),
              ("5sigma_from_trial_1", (8,)), ("1sigma_from_trial_21", (3, 5, 8, 12, 16)), ("pooled", (5, 8, 12, 16)))


def look_model_block(shared: pd.DataFrame) -> list[str]:
    """Section 7: the increment over LCB1 when the drift ramp and the AR(1) state move on from look to look.

    Reads the tagged rescorings of the sequential replays (sitting_by_magnitude.py --tag seq / seqrank); each is
    the four processes pooled, and its gaussian and bias runs equal the shared replay's.
    """
    L = ["== 7. the look model: the replays above cancel the error one sitting shares, including the drift ramp and",
         "   the AR(1) state; the sequential replays let that state move on from look to look (random order, or the",
         "   candidates in rank order), drift and AR(1) only change. Four processes pooled; seeds 12-16 unless stated;",
         "   intervals are the per-k rows' (for a cell's chosen k the table's interval of the gain is in sections 1-3)."]
    models = {"shared": (shared, None)}
    for name, suffix in LOOK_MODELS:
        base = REVIEW / f"sitting_by_magnitude{suffix}"
        if Path(f"{base}.csv").is_file() and Path(f"{base}_selection.json").is_file():
            models[name] = (pd.read_csv(f"{base}.csv"), json.loads(Path(f"{base}_selection.json").read_text()))
        else:
            L.append(f"   {name}: {base.name}.csv is missing (python scripts/sitting_by_magnitude.py --tag "
                     f"{suffix[1:]} ...; see its docstring)")
    for name, (_, sel) in models.items():
        if sel is None:
            continue
        prov = next((x for x in sel if x.get("cell") == "provenance"), {})
        chosen = {x["cell"]: x["k_chosen_on_7_11"] for x in sel if "k_chosen_on_7_11" in x}
        L.append(f"   {name}: replays {', '.join(prov.get('replay_dirs', ['?']))}; k chosen on seeds 7-11 by cell {chosen}")
    for cell, ks in LOOK_CELLS:
        for k in ks:
            L.append(f"-- {cell}, k = {k}")
            for name, (frame, _) in models.items():
                r = frame[(frame["cell"] == cell) & (frame["k"] == k)]
                if not len(r):
                    L.append(f"   {name:25s} k = {k} not replayed")
                    continue
                r = r.iloc[0]
                by = "  ".join(f"{p} {r.get(f'inc_lcb1_test_{p}', np.nan):+.4f}" for p in ("drift", "ar1"))
                L.append(f"   {name:25s} gain {f4(r['gain_test'], r['gain_test_lo'], r['gain_test_hi'])}  sitting - LCB1 "
                         f"{f4(r['inc_lcb1_test'], r['inc_lcb1_test_lo'], r['inc_lcb1_test_hi'])} ahead "
                         f"{int(r['inc_lcb1_test_landscapes_ahead'])}/20 p {r['inc_lcb1_p_test']:.3g} (Holm over k "
                         f"{r['inc_lcb1_p_test_holm_over_k']:.3g}); seeds 7-16 "
                         f"{f4(r['inc_lcb1_all'], r['inc_lcb1_all_lo'], r['inc_lcb1_all_hi'])}; by process {by}")
    return L


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    lines = report()
    text = "\n".join(lines) + "\n"
    print(text, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
