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
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
FRESH = REPO / "output-boba-confirmatory" / "analysis"
MAIN = REPO / "output-boba" / "analysis"
REPS = 4000


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


def main() -> None:
    oz = opt_z()
    rng = np.random.default_rng(20260926)
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file", "family", "k",
            "candidates", "rho", "winner", "regret_noisy", "ref_noisy", "ref_clean", "claim_noisy",
            "truly_better_noisy", "false_claim_noisy"]
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


if __name__ == "__main__":
    sys.exit(main())
