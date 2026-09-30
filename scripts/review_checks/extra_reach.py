"""App. B.7: the trial at which a noisy run reaches its clean twin's k-trial regret.

analyse_extra_runs.extra_trials counts ``extra`` from ``origin``, the trial at which
the CLEAN twin itself first came within the tolerance of its own k-trial regret,
not from k. The noisy run's reach trial is therefore origin + extra, and its
budget as a multiple of a clean k-trial study's is reach / k. The per-run table's
``multiplier`` column is (k + extra) / k, a different quantity.

This recomputes origin and reach from the budget-100 arm's logs with
analyse_extra_runs' own functions (index_runs, regret_curve, load_opt_z,
extra_trials, clean_origin: the same pairing, the same truncation to the shorter
log, the same tolerance), checks that extra, the censored flag, the origin
(clean_origin) and the reach trial reproduce extra_runs_per_run.csv exactly, and prints the medians the
appendix quotes (GAUSSIAN error from the first rating, tolerance 0.01 of opt_z,
k = 25 and 50) at every magnitude.

    python scripts/review_checks/extra_reach.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import analyse_extra_runs as aer  # noqa: E402

ARM = REPO / "output-boba-budget100"
TOL = 0.01
KS = (25, 50)


def recompute() -> pd.DataFrame:
    baselines, jittered = aer.index_runs(ARM)
    opt_z = aer.load_opt_z(ARM)
    rows, cache = [], {}
    for run in jittered:
        if run["acquisition"] in aer.MODEL_FREE or run["error_model"] != "gaussian":
            continue
        if run["variant"] or run["jitter_iteration"] != 0:
            continue
        base = baselines.get(run["stem"])
        if base is None:
            continue
        if base not in cache:
            cache[base] = aer.regret_curve(base)
        clean, noisy = cache[base], aer.regret_curve(run["path"])
        T = min(len(clean), len(noisy))
        clean, noisy = clean[:T], noisy[:T]
        slack = TOL * opt_z[run["dataset"]]
        for k in KS:
            extra, censored = aer.extra_trials(clean, noisy, k, slack)
            origin = aer.clean_origin(clean, k, slack)
            rows.append({"dataset": run["dataset"], "acquisition": run["acquisition"], "seed": run["seed"],
                         "jitter_std": run["jitter_std"], "k": k, "budget": T, "origin": origin, "extra": extra,
                         "censored": censored, "reach": origin + extra})
    return pd.DataFrame(rows)


def main() -> None:
    df = recompute()
    pub = pd.read_csv(ARM / "analysis" / "extra_runs_per_run.csv")
    pub = pub[(pub["error_model"] == "gaussian") & (pub["tolerance"] == TOL) & (pub["jitter_iteration"] == 0)
              & (pub["variant"].fillna("") == "") & pub["k"].isin(KS)]
    keys = ["dataset", "acquisition", "seed", "jitter_std", "k"]
    m = df.merge(pub, on=keys, suffixes=("", "_pub"), validate="one_to_one")
    if not (len(m) == len(df) == len(pub)):
        raise SystemExit(f"pairing mismatch: {len(m)} merged, {len(df)} recomputed, {len(pub)} published")
    same = {c: bool((m[c] == m[f"{c}_pub"]).all()) for c in ("extra", "censored")}
    if "reach_trial" in m:
        same["clean_origin"] = bool((m["origin"] == m["clean_origin"]).all())
        same["reach_trial"] = bool((m["reach"] == m["reach_trial"]).all())
    print(f"reproduces output-boba-budget100/analysis/extra_runs_per_run.csv: {len(m)} rows; equal: "
          + ", ".join(f"{c} {v}" for c, v in same.items()))
    if not all(same.values()):
        raise SystemExit("the recomputation does not reproduce the published per-run table")

    print(f"\nGAUSSIAN error from the first rating, tolerance {TOL} of opt_z, six acquisitions, "
          f"{df['seed'].nunique()} seeds, {df['dataset'].nunique()} landscapes, budget {int(df['budget'].max())}")
    print("medians over runs; a censored run's extra is budget - origin and its reach is the budget, so a median "
          "is exact only while fewer than half the runs are censored")
    for (k, std), b in df.groupby(["k", "jitter_std"]):
        cens = b["censored"].mean()
        exact = "exact" if cens < 0.5 else "a lower bound"
        reached = b[~b["censored"]]
        print(f"\nk = {k}, {std:g} sigma: {len(b)} runs, {cens:.1%} censored (medians {exact})")
        print(f"  median extra {b['extra'].median():g} (counted from origin); origin: median trial "
              f"{b['origin'].median():g}, before trial {k} in {np.mean(b['origin'] < k):.1%} of runs")
        print(f"  reach trial (origin + extra): median {b['reach'].median():g}, median reach / {k} = "
              f"{np.median(b['reach'] / k):.3f}")
        print(f"  the per-run 'multiplier' (k + extra) / k, for contrast: median {np.median((k + b['extra']) / k):.3f}")
        if len(reached):
            print(f"  runs that reach it ({len(reached)}): median reach trial {reached['reach'].median():g}, "
                  f"median extra {reached['extra'].median():g}")


if __name__ == "__main__":
    main()
