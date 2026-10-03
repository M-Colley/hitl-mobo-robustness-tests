"""The end-of-run procedures on seeds 27-36, which no choice in the paper has seen.

Every procedure of Section 7 was chosen on seeds 7-11 and scored on seeds 12-16.
The replication arm (output-boba-confirmatory, gaussian error only, seeds 27-36,
all twelve acquisitions) was run for a different purpose and never used to choose
or score a remedy, so replaying the procedures on its logs is a third, untouched
test. The replays are

    python scripts/replay_end_of_study.py --arms output-boba-confirmatory
        --output-dir output-boba-confirmatory/analysis/end_of_study_fresh
        --acquisitions logei,qnei,ucb --seeds 27-36 --error-models gaussian
        --stds 0.05,0.25,1.0,5.0 --onsets 0,20 --tournament-k 2,5,12 --confirm-k 2
        --rho 1 --rerate-dirs none
    python scripts/rescore_ship_rules.py --input-dir output-boba-confirmatory
        --output-dir output-boba-confirmatory/analysis/ship_rules_fresh
        --acquisitions <the ten model-based> --seeds 27-36 --error-models gaussian

This prints, with the paper's estimands: the sitting's gain over the standard
process (candidates by the posterior mean less one latent SD, the best look
ships, looks at the full SD; LogEI and qNEI) at 1 sigma from the first rating for
k = 2, and pooled over the cells for k = 12, beside the same gaussian-only
quantity on seeds 12-16 and 7-16 from the main sweep's replay; the confirmation
test's false-claim rate and power (LogEI, qNEI, UCB); and the cautious ship
rule's recovery of the deployed cost at 0.25 sigma and above (ten acquisitions).

It then scores, on the same untouched runs, the three ship rules that buy no
trial (after all 50 trials: PM, the highest posterior mean; LCB1, the posterior
mean less one latent SD, the rule that ranks the sitting's candidates; LCB2, less
two), and for every cell and every replayed k (2, 5, 12) the sitting's increment
over LCB1, per run (regret of LCB1 - regret of the sitting) / opt_z, averaged per
landscape, with a landscape bootstrap, a two-sided Wilcoxon test over the twenty
per-landscape increments and Holm's correction over the three k in the cell and
over the 24 (cell, k); the increment over the top candidate of the sitting's own
prefix (the value of the looks alone); and the clean-twin price of each
procedure, (trt_clean - ref_clean) / opt_z. The join and the estimands are
sitting_by_magnitude.attach_ship_rules. That block draws from its own generator
(REFERENCE_SEED), created inside reference_table, so the lines above it are
unchanged and any caller of reference_table gets the same numbers.

A last block gives the sitting's increment over the other two zero-trial rules,
LCB2 (the cautious rule of Section 7) and PM, per cell and k, with the same
estimand and Holm families (rule_increments, its own generator
REFERENCE_SEED + 1, so every line above it is unchanged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
import sitting_by_magnitude as sbm  # noqa: E402

FRESH = REPO / "output-boba-confirmatory" / "analysis"
MAIN = REPO / "output-boba" / "analysis"
REPS = 4000
REFERENCE_SEED = 20260929


def opt_z() -> dict[str, float]:
    stats = json.loads((REPO / "boba_landscape_stats.json").read_text())["functions"]
    return {k: float(v["opt_z"]) for k, v in stats.items()}


def sitting(frame: pd.DataFrame, oz: dict[str, float]) -> pd.DataFrame:
    d = frame[(frame["family"] == "tournament") & (frame["candidates"] == "lcb") & (frame["winner"] == "look")
              & (frame["rho"] == 1.0) & frame["acquisition"].isin(["logei", "qnei"])
              & (frame["error_model"] == "gaussian")].copy()
    d = d.drop_duplicates(subset=["file", "k"], keep="first")
    z = d["dataset"].map(oz)
    d["gain"] = (d["ref_noisy"] - d["regret_noisy"]) / z
    d["cost"] = (d["ref_noisy"] - d["ref_clean"]) / z
    d["k"] = d["k"].astype(int)
    return d


def summarise(sub: pd.DataFrame, rng: np.random.Generator) -> str:
    per = sub.groupby("dataset")[["gain", "cost"]].mean()
    g, c = per["gain"].to_numpy(), per["cost"].to_numpy()
    idx = rng.integers(0, len(g), (REPS, len(g)))
    lo, hi = np.percentile(g[idx].mean(axis=1), [2.5, 97.5])
    share = g.mean() / c.mean()
    s_lo, s_hi = np.percentile(g[idx].mean(axis=1) / c[idx].mean(axis=1), [2.5, 97.5])
    p = wilcoxon(g).pvalue if np.any(g != 0) else float("nan")
    return (f"gain {g.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  share of cost {share:+.3f} [{s_lo:+.3f}, {s_hi:+.3f}]  "
            f"landscapes gaining {int((g > 0).sum())}/{len(g)}  Wilcoxon p {p:.3g}  ({len(sub)} runs)")


def _mean_ci(v, rng: np.random.Generator) -> tuple[float, float, float]:
    v = np.asarray(v, dtype=float)
    draws = v[rng.integers(0, len(v), (REPS, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _ratio_ci(g, c, rng: np.random.Generator) -> tuple[float, float, float]:
    g, c = np.asarray(g, dtype=float), np.asarray(c, dtype=float)
    idx = rng.integers(0, len(g), (REPS, len(g)))
    draws = g[idx].mean(axis=1) / c[idx].mean(axis=1)
    if not sbm.share_defined(c):
        # A ratio on a reference cost that is not positive on every landscape is not reported (AGENTS.md).
        return float("nan"), float("nan"), float("nan")
    return float(g.mean() / c.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _share(v: float, lo: float | None = None, hi: float | None = None) -> str:
    if not np.isfinite(v):
        return "share n/a (cost not positive on every landscape)"
    return f"share {v:+.3f}" + ("" if lo is None else f" [{lo:+.3f}, {hi:+.3f}]")


def _p3(p: float) -> str:
    return "n/a" if not np.isfinite(p) else f"{p:.3g}"


def _cells(m: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    m = m.assign(sigma=m["jitter_std"].astype(float), onset=m["jitter_iteration"].astype(int))
    return [("pooled", m)] + [(f"{sig:g}sigma_from_trial_{ons + 1}", sub)
                              for (sig, ons), sub in sorted(m.groupby(["sigma", "onset"]))]


def reference_table(frame: pd.DataFrame, ship: pd.DataFrame, oz: dict[str, float]
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The zero-trial ship rules, the sitting's increments over LCB1 and the prices, on one set of runs.

    ``frame`` is an end_of_study_per_run.csv.gz (gaussian error, LogEI and qNEI
    are kept), ``ship`` the ship_rules_per_run.csv of the same runs. Returns the
    rules per cell, the sitting per cell and k, and the clean-twin prices. Uses its
    own generator, so every caller gets the same numbers.
    """
    rng = np.random.default_rng(REFERENCE_SEED)
    d = sitting(frame, oz)
    d["opt_z"] = d["dataset"].map(oz)
    m = sbm.attach_ship_rules(d, ship)
    rule_rows, inc_rows = [], []
    for label, sub in _cells(m):
        runs = sub.drop_duplicates("file")
        per = runs.groupby("dataset")
        cost = per["cost"].mean()
        for rule in sbm.SHIP_RULES:
            g = per[f"gain_{rule}"].mean()
            mean, lo, hi = _mean_ci(g, rng)
            share, s_lo, s_hi = _ratio_ci(g, cost.reindex(g.index), rng)
            rule_rows.append({"cell": label, "rule": rule, "gain": mean, "gain_lo": lo, "gain_hi": hi,
                              "share": share, "share_lo": s_lo, "share_hi": s_hi, "cost": float(cost.mean()),
                              "landscapes_gaining": int((g > 0).sum()), "p": sbm.wilcoxon_p(g.to_numpy()),
                              "price": float(per[f"price_{rule}"].mean().mean()), "n_runs": len(runs)})
        for k in sorted(sub["k"].unique()):
            x = sub[sub["k"] == k].groupby("dataset")
            g, inc, top = x["gain"].mean(), x["inc_lcb1"].mean(), x["inc_top"].mean()
            gm, glo, ghi = _mean_ci(g, rng)
            share, s_lo, s_hi = _ratio_ci(g, x["cost"].mean().reindex(g.index), rng)
            im, ilo, ihi = _mean_ci(inc, rng)
            tm, tlo, thi = _mean_ci(top, rng)
            inc_rows.append({"cell": label, "k": int(k), "gain": gm, "gain_lo": glo, "gain_hi": ghi, "share": share,
                             "share_lo": s_lo, "share_hi": s_hi, "lcb1_gain": float(per["gain_lcb1"].mean().mean()),
                             "inc_lcb1": im, "inc_lcb1_lo": ilo, "inc_lcb1_hi": ihi,
                             "inc_lcb1_landscapes_ahead": int((inc > 0).sum()), "inc_lcb1_p": sbm.wilcoxon_p(inc),
                             "inc_top": tm, "inc_top_lo": tlo, "inc_top_hi": thi,
                             "price": float(x["price"].mean().mean()), "n_runs": int(x.size().sum())})
    rules, inc = pd.DataFrame(rule_rows), pd.DataFrame(inc_rows)
    inc["inc_lcb1_p_holm_over_k"] = np.nan
    for label in inc["cell"].unique():
        at = inc["cell"] == label
        inc.loc[at, "inc_lcb1_p_holm_over_k"] = sbm.holm(inc.loc[at, "inc_lcb1_p"].to_numpy())
    in_cells = inc["cell"] != "pooled"
    inc["inc_lcb1_p_holm_all_cells"] = np.nan
    inc.loc[in_cells, "inc_lcb1_p_holm_all_cells"] = sbm.holm(inc.loc[in_cells, "inc_lcb1_p"].to_numpy())
    # Prices: the clean twin is shared by every cell, so one row per (landscape, acquisition, seed[, k]).
    price_rows = []
    stems = m.drop_duplicates(["dataset", "acquisition", "seed", "k"])
    for k in sorted(stems["k"].unique()):
        mean, lo, hi = _mean_ci(stems[stems["k"] == k].groupby("dataset")["price"].mean(), rng)
        price_rows.append({"procedure": f"sitting k = {int(k)}", "price": mean, "lo": lo, "hi": hi})
    runs = stems.drop_duplicates(["dataset", "acquisition", "seed"])
    for rule in sbm.SHIP_RULES:
        mean, lo, hi = _mean_ci(runs.groupby("dataset")[f"price_{rule}"].mean(), rng)
        price_rows.append({"procedure": rule, "price": mean, "lo": lo, "hi": hi})
    return rules, inc, pd.DataFrame(price_rows)


def rule_increments(frame: pd.DataFrame, ship: pd.DataFrame, oz: dict[str, float]) -> pd.DataFrame:
    """The sitting's increment over LCB2 and over PM per cell and k, the estimand of reference_table.

    Per run (regret of the rule after all 50 trials - regret of the sitting) / opt_z,
    averaged per landscape, landscape bootstrap, two-sided Wilcoxon over the twenty
    landscapes, Holm over the k of a cell and over the 24 (cell, k). Its own
    generator, so every caller gets the same numbers and reference_table is unchanged.
    """
    rng = np.random.default_rng(REFERENCE_SEED + 1)
    d = sitting(frame, oz)
    d["opt_z"] = d["dataset"].map(oz)
    m = sbm.attach_ship_rules(d, ship)
    rows = []
    for label, sub in _cells(m):
        runs = sub.drop_duplicates("file").groupby("dataset")
        for k in sorted(sub["k"].unique()):
            x = sub[sub["k"] == k].groupby("dataset")
            row = {"cell": label, "k": int(k)}
            for rule in sbm.OTHER_RULES:
                v = x[f"inc_{rule}"].mean()
                mean, lo, hi = _mean_ci(v, rng)
                row.update({f"{rule}_gain": float(runs[f"gain_{rule}"].mean().mean()), f"inc_{rule}": mean,
                            f"inc_{rule}_lo": lo, f"inc_{rule}_hi": hi,
                            f"inc_{rule}_landscapes_ahead": int((v > 0).sum()), f"inc_{rule}_p": sbm.wilcoxon_p(v)})
            rows.append(row)
    out = pd.DataFrame(rows)
    in_cells = out["cell"] != "pooled"
    for rule in sbm.OTHER_RULES:
        out[f"inc_{rule}_p_holm_over_k"] = np.nan
        for label in out["cell"].unique():
            at = out["cell"] == label
            out.loc[at, f"inc_{rule}_p_holm_over_k"] = sbm.holm(out.loc[at, f"inc_{rule}_p"].to_numpy())
        out[f"inc_{rule}_p_holm_all_cells"] = np.nan
        out.loc[in_cells, f"inc_{rule}_p_holm_all_cells"] = sbm.holm(out.loc[in_cells, f"inc_{rule}_p"].to_numpy())
    return out


def format_rule_increments(out: pd.DataFrame) -> list[str]:
    lines = []
    for label in out["cell"].unique():
        lines.append(f"cell {label}")
        for _, r in out[out["cell"] == label].iterrows():
            parts = []
            for rule in sbm.OTHER_RULES:
                parts.append(f"over {rule.upper()} ({rule.upper()} gain {r[f'{rule}_gain']:+.4f}) "
                             f"{r[f'inc_{rule}']:+.4f} [{r[f'inc_{rule}_lo']:+.4f}, {r[f'inc_{rule}_hi']:+.4f}] "
                             f"ahead {int(r[f'inc_{rule}_landscapes_ahead'])}/20 p {_p3(r[f'inc_{rule}_p'])} "
                             f"(Holm over k {_p3(r[f'inc_{rule}_p_holm_over_k'])}, over the 24 "
                             f"{_p3(r[f'inc_{rule}_p_holm_all_cells'])})")
            lines.append(f"   sitting k = {int(r['k']):2d}  " + "  ".join(parts))
    return lines


def format_reference(rules: pd.DataFrame, inc: pd.DataFrame, prices: pd.DataFrame) -> list[str]:
    lines = []
    for label in inc["cell"].unique():
        lines.append(f"cell {label}  (cost of error {rules.loc[rules.cell == label, 'cost'].iloc[0]:.4f})")
        for _, r in rules[rules["cell"] == label].iterrows():
            lines.append(f"   {r['rule'].upper():5s} no extra trial   gain {r['gain']:+.4f} [{r['gain_lo']:+.4f}, "
                         f"{r['gain_hi']:+.4f}]  {_share(r['share'], r['share_lo'], r['share_hi'])}  "
                         f"gaining {int(r['landscapes_gaining'])}/20  p {r['p']:.3g}  price {r['price']:.4f}")
        for _, r in inc[inc["cell"] == label].iterrows():
            lines.append(f"   sitting k = {int(r['k']):2d}    gain {r['gain']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}]  "
                         f"{_share(r['share'])}  over LCB1 {r['inc_lcb1']:+.4f} [{r['inc_lcb1_lo']:+.4f}, "
                         f"{r['inc_lcb1_hi']:+.4f}] ahead {int(r['inc_lcb1_landscapes_ahead'])}/20 p {r['inc_lcb1_p']:.3g} "
                         f"(Holm over k {_p3(r['inc_lcb1_p_holm_over_k'])}, over the 24 {_p3(r['inc_lcb1_p_holm_all_cells'])})"
                         f"  over its top candidate {r['inc_top']:+.4f} [{r['inc_top_lo']:+.4f}, {r['inc_top_hi']:+.4f}]"
                         f"  price {r['price']:.4f}  ({int(r['n_runs'])} runs)")
    lines.append("clean-twin price (trt_clean - ref_clean) / opt_z, landscape mean:")
    for _, r in prices.iterrows():
        lines.append(f"   {r['procedure']:14s} {r['price']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    return lines


def main() -> None:
    oz = opt_z()
    rng = np.random.default_rng(20260926)
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file", "family", "k",
            "candidates", "rho", "winner", "regret_noisy", "ref_noisy", "ref_clean", "claim_noisy",
            "truly_better_noisy", "false_claim_noisy", "regret_clean", "top_candidate_regret_noisy"]
    fresh = pd.read_csv(FRESH / "end_of_study_fresh" / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
    main_ = pd.concat([pd.read_csv(MAIN / d / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
                       for d in ("end_of_study_ksweep",)], ignore_index=True)
    fs, ms = sitting(fresh, oz), sitting(main_, oz)
    cell = lambda d: d[(d["jitter_std"].astype(float) == 1.0) & (d["jitter_iteration"].astype(int) == 0)]  # noqa: E731

    print("== the sitting, gaussian error only, LogEI and qNEI")
    for k in (2, 5, 12):
        print(f"k = {k:2d}, 1 sigma from the first rating")
        print(f"   seeds 27-36 (fresh)   {summarise(cell(fs[fs.k == k]), rng)}")
        m = cell(ms[ms.k == k])
        if len(m):
            print(f"   seeds 12-16           {summarise(m[m.seed.between(12, 16)], rng)}")
            print(f"   seeds 7-16            {summarise(m, rng)}")
    for k in (2, 12):
        print(f"k = {k:2d}, pooled over magnitudes and onsets")
        print(f"   seeds 27-36 (fresh)   {summarise(fs[fs.k == k], rng)}")
        m = ms[ms.k == k]
        if len(m):
            print(f"   seeds 12-16           {summarise(m[m.seed.between(12, 16)], rng)}")

    print("\n== confirmation (four trials), gaussian error, LogEI, qNEI, UCB")
    for label, d in (("seeds 27-36 (fresh)", fresh), ):
        c = d[(d["error_model"] == "gaussian") & d["acquisition"].isin(["logei", "qnei", "ucb"])]
        std = c[c["family"] == "standard"].drop_duplicates("file")
        con = c[(c["family"] == "confirmation") & (c["k"].astype(float) == 2)].drop_duplicates("file")
        for name, block in (("standard", std), ("confirm k=2", con)):
            claim = block["claim_noisy"].astype(bool)
            better = block["truly_better_noisy"].astype(bool)
            false_claim = block["false_claim_noisy"].astype(bool)
            print(f"   {label} {name:12s} claims {claim.mean():.3f}  false claims {false_claim.mean():.4f}  "
                  f"power {(claim & better).sum() / max(better.sum(), 1):.3f}  ({len(block)} runs)")
        one = con[(con["jitter_std"].astype(float) == 1.0) & (con["jitter_iteration"].astype(int) == 0)]
        claim, better = one["claim_noisy"].astype(bool), one["truly_better_noisy"].astype(bool)
        print(f"   {label} confirm k=2 at 1 sigma from the first rating: power "
              f"{(claim & better).sum() / max(better.sum(), 1):.3f}, false claims {one['false_claim_noisy'].astype(bool).mean():.4f}")

    ship = FRESH / "ship_rules_fresh" / "ship_rules_per_run.csv"
    if ship.is_file():
        r = pd.read_csv(ship)
        r = r[~r["acquisition"].isin(["random", "sobol"])]
        r["baseline"] = r["baseline"].astype(str).str.lower().isin(("true", "1"))
        clean = r[r["baseline"]].set_index(["dataset", "acquisition", "seed"])
        noisy = r[~r["baseline"] & (r["error_model"] == "gaussian") & (r["jitter_std"].astype(float) >= 0.25)]
        noisy = noisy.join(clean[["regret_best_observed", "regret_lcb2"]], on=["dataset", "acquisition", "seed"],
                           rsuffix="_clean")
        z = noisy["dataset"].map(oz)
        noisy = noisy.assign(cost=(noisy["regret_best_observed"] - noisy["regret_best_observed_clean"]) / z,
                             gain=(noisy["regret_best_observed"] - noisy["regret_lcb2"]) / z)
        acqs = sorted(noisy["acquisition"].unique())
        print(f"\n== cautious ship rule (posterior mean less two latent SDs), gaussian >= 0.25 sigma, "
              f"{len(acqs)} acquisitions: {', '.join(acqs)}")
        print(f"   seeds 27-36 (fresh)   {summarise(noisy, rng)}")

    if ship.is_file():
        rules, inc, prices = reference_table(fresh, pd.read_csv(ship, low_memory=False), oz)
        print("\n== the ship rules that buy no trial, on the same runs, and the sitting's increment over LCB1")
        print("   seeds 27-36 (fresh), gaussian error, LogEI and qNEI; gains over the standard process in units of")
        print("   opt_z; 'over LCB1' is the paired per-run difference (regret of LCB1 after 50 trials less the")
        print("   sitting's regret), averaged per landscape; 'over its top candidate' is the value of the looks alone")
        print("   (the sitting over shipping the top LCB1 candidate of its own T - k prefix). This block draws")
        print("   its intervals from its own generator; where the first block above reports the same sitting gain")
        print("   (1 sigma from the first rating, k = 2, 5, 12; pooled, k = 2, 12), the paper quotes the first block.")
        for line in format_reference(rules, inc, prices):
            print(f"   {line}")
        print("\n== the sitting's increment over the other two ship rules that buy no trial, on the same runs")
        print("   seeds 27-36 (fresh), gaussian error, LogEI and qNEI; LCB2 is the cautious rule (posterior mean less two")
        print("   latent SDs), PM the posterior-mean maximiser, each after all 50 trials; the same paired estimand as")
        print("   over LCB1 above, from its own generator.")
        for line in format_rule_increments(rule_increments(fresh, pd.read_csv(ship, low_memory=False), oz)):
            print(f"   {line}")


if __name__ == "__main__":
    sys.exit(main())
