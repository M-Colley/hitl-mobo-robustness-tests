"""Test-retest pairs whose second rating remembers the first: how the known-variance rule degrades.

known_variance_ship.py estimates the rater's noise variance from three test-retest pairs and assumes the
second rating of a pair is drawn afresh. A rater who remembers the first rating gives a second rating whose
error correlates with the first at rho_m. For normal errors a pair's difference is then distributed exactly as
sqrt(1 - rho_m) times a memoryless one (its variance is 2 s^2 (1 - rho_m) instead of 2 s^2), so the
three-pair estimate is (1 - rho_m) times known_variance_ship's own draw for that run: the same draw, scaled,
which couples every rho_m to the published level. This check refits the loop's GP with that estimate as
train_Yvar for rho_m in 0.5, 0.75 and 0.9, and rho_m = 0 as the control, which must reproduce the published
retest3 regrets of known_variance_ship_per_run.csv exactly. A clean twin's pairs differ by nothing, so its
estimate stays zero and the price stays zero by construction, as for every level of known_variance_ship.

Scope: the main sweep, LogEI and qNEI, the four processes, seeds 7-16 (12,800 noisy runs). The rule is LCB2,
the one rule seeds 7-11 chose for the three-pair level in known_variance_ship.py; it is not re-chosen here.
Scores on seeds 12-16 use known_variance_ship's own estimand and summary (analyse_boba_adaptations), so the
control's rows equal that check's retest3 rows. Post hoc, outside the paper's Holm families.

    python scripts/checks/retest_memory.py --workers 8      # refits what the cache lacks (needs the run logs), then scores
    python scripts/checks/retest_memory.py --stage score    # from the tracked per-run table only

Writes review/retest_memory_per_run.csv.gz, review/retest_memory_cells.csv and
register_checks/retest_memory.txt; caches the refits in review/retest_memory/per_run/ (git-ignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import known_variance_ship as kvs  # noqa: E402  (imports the simulator first, as the drivers do)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MEMORY = (0.0, 0.5, 0.75, 0.9)
RULE = "lcb2"
CACHE_VERSION = 1
CACHE = kvs.REVIEW / "retest_memory" / "per_run" / "output-boba.jsonl"
PER_RUN = kvs.REVIEW / "retest_memory_per_run.csv.gz"
CELLS = kvs.REVIEW / "retest_memory_cells.csv"
REGISTER = kvs.REVIEW / "register_checks" / "retest_memory.txt"


def level(rho_m: float) -> str:
    return f"retest3_mem{rho_m:g}"


def refit_memory(task: dict) -> dict:
    """Worker: one noisy log, every rho_m."""
    df = pd.read_csv(task["path"], float_precision="round_trip")
    name = Path(task["file"]).name
    seed = kvs.rs.file_seed(task["file"])
    a = kvs.run_arrays(df, task)
    T = a["T"]
    cfg = kvs.error_config(task["run_args"], task, T)
    v_hat = kvs.retest_estimate(task["error_model"], float(task["jitter_std"]), float(cfg.error_ar1_rho), name)
    rec = {"cache_version": CACHE_VERSION, "file": task["file"], "sigma2_retest3": v_hat, "n_fit_failures": 0}
    for rho_m in MEMORY:
        vec = np.full(T, max((1.0 - rho_m) * v_hat, kvs.FLOOR))
        picks, diag = kvs.posterior_picks(a["X"], a["y"], a["lo"], a["hi"], vec, seed)
        rec["n_fit_failures"] += int(not diag["gp_fit_ok"])
        for r, i in picks.items():
            rec[f"regret_{level(rho_m)}_{r}"] = float(a["y_opt"] - a["score"][i]) if i >= 0 else float("nan")
    return rec


def tasks() -> list[dict]:
    out = [t for t in kvs.build_tasks(["output-boba"], None)
           if not t["baseline"] and t["acquisition"] in kvs.PAIR_ACQS and t["error_model"] in kvs.PROCESSES]
    if len(out) != 12_800:
        raise ValueError(f"expected 12,800 noisy LogEI and qNEI runs, found {len(out):,}")
    return out


def refit_stage(workers: int) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # a line cut off by a killed run is refitted
            if rec.get("cache_version") == CACHE_VERSION:
                done.add(rec["file"])
    todo = [t for t in tasks() if t["file"] not in done]
    print(f"{len(done):,} cached, {len(todo):,} to refit", flush=True)
    if not todo:
        return
    n = max(1, min(workers, os.cpu_count() or 1, len(todo)))
    t0 = time.time()
    from multiprocessing import get_context

    with CACHE.open("a", encoding="utf-8") as fh, get_context("spawn").Pool(n, initializer=kvs._init_worker) as pool:
        for i, rec in enumerate(pool.imap_unordered(refit_memory, todo, chunksize=4)):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 1000 == 0 or i + 1 == len(todo):
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  {i + 1:,}/{len(todo):,}  {rate:.1f} runs/s", flush=True)


def assemble() -> pd.DataFrame:
    """The published per-run table's runs of this scope, with the memory levels joined; clean twins keep their
    published retest3 regrets (their estimate is zero whatever the memory)."""
    recs = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("cache_version") == CACHE_VERSION:
            recs[r["file"]] = r
    pr = pd.read_csv(kvs.PER_RUN, float_precision="round_trip", low_memory=False)
    pr["baseline"] = pr["baseline"].astype(str).str.lower().isin(("true", "1"))
    pr["variant"] = pr["variant"].fillna("").astype(str)
    pr = pr[(pr["arm"] == "output-boba") & pr["acquisition"].isin(kvs.PAIR_ACQS)
            & (pr["baseline"] | pr["error_model"].isin(kvs.PROCESSES))].reset_index(drop=True)
    noisy = ~pr["baseline"]
    missing = sorted(set(pr.loc[noisy, "file"]) - set(recs))
    if missing:
        raise SystemExit(f"{len(missing):,} noisy runs have no refit (e.g. {missing[0]}); run the refit stage")
    keep = ["file", "dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed", "baseline",
            "variant", "arm", "y_opt", "regret_best_observed"] + [f"regret_retest3_{r}" for r in kvs.RULES]
    out = pr[keep].copy()
    for rho_m in MEMORY:
        for r in kvs.RULES:
            col = f"regret_{level(rho_m)}_{r}"
            out[col] = np.where(out["baseline"], out[f"regret_retest3_{r}"],
                                out["file"].map(lambda f: recs.get(f, {}).get(col, np.nan)))
    out["sigma2_retest3_refit"] = out["file"].map(lambda f: recs.get(f, {}).get("sigma2_retest3", np.nan))
    return out


def control(per_run: pd.DataFrame) -> str:
    """rho_m = 0 is the published three-pair level: every regret must match exactly."""
    gaps = [float(np.nanmax(np.abs(per_run[f"regret_{level(0.0)}_{r}"] - per_run[f"regret_retest3_{r}"])))
            for r in kvs.RULES]
    if max(gaps) != 0.0:
        raise ValueError(f"rho_m = 0 does not reproduce the published retest3 regrets (largest gap {max(gaps):g})")
    return (f"control: rho_m = 0 reproduces known_variance_ship's retest3 regrets exactly in all "
            f"{int((~per_run['baseline']).sum()):,} noisy runs (PM, LCB1, LCB2)")


def score(per_run: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    opt_z = kvs.aba.dataset_scale("opt_z")
    levels = [level(m) for m in MEMORY]
    pf = kvs.paired_frames(per_run, levels, opt_z)
    rows = []
    for rho_m in MEMORY:
        frame = pf[(level(rho_m), RULE)]
        sub = frame[frame["seed"].isin(kvs.TEST_SEEDS)]
        for cell, sig, ons, block in kvs.blocks(sub):
            rows.append({"memory": rho_m, "rule": RULE, "cell": cell, "n_runs": int(len(block)), **kvs.summary(block)})
    cells = pd.DataFrame(rows)
    lines = ["Test-retest pairs with memory: the known-variance rule (LCB2, three pairs) when the second rating's",
             "error correlates with the first at rho_m (scripts/checks/retest_memory.py). Gains over the standard process",
             "on the deployed design at T = 50, units of opt_z; LogEI and qNEI, four processes, seeds 12-16 (held out);",
             "price zero by construction (a clean twin's estimate is zero). Post hoc, outside the paper's Holm families.",
             control(per_run)]
    ratio = per_run.loc[~per_run["baseline"], "sigma2_retest3_refit"] / per_run.loc[~per_run["baseline"], "jitter_std"] ** 2
    lines.append(f"memoryless estimate / sigma_e^2: median {ratio.median():.2f} (10th {ratio.quantile(0.1):.2f}, "
                 f"90th {ratio.quantile(0.9):.2f}); with memory rho_m it is (1 - rho_m) times that")
    show = ["0.05sigma_from_trial_1", "0.05sigma_from_trial_21", "0.25sigma_from_trial_1", "0.25sigma_from_trial_21",
            "1sigma_from_trial_1", "1sigma_from_trial_21", "5sigma_from_trial_1", "5sigma_from_trial_21",
            f"pooled_ge_{kvs.POOL_MIN:g}sigma"]
    for rho_m in MEMORY:
        lines.append(f"-- rho_m = {rho_m:g}" + ("  (the published three-pair level)" if rho_m == 0 else ""))
        c = cells[cells["memory"] == rho_m].set_index("cell")
        for cell in show:
            r = c.loc[cell]
            share = ("no share" if isinstance(r.get("recovered_suppressed"), str) and r["recovered_suppressed"]
                     else f"{100 * r['recovered']:+.1f}% [{100 * r['recovered_lo']:+.0f}, {100 * r['recovered_hi']:+.0f}]")
            lines.append(f"   {cell:<24s} gain {r['gain']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}]  {share}  "
                         f"gaining {int(r['landscapes_gaining'])}/20  price {r['price']:.4f}")
    return cells, lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("all", "score"), default="all")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.stage == "all":
        refit_stage(args.workers)
        per_run = assemble()
        per_run.to_csv(PER_RUN, index=False, compression={"method": "gzip", "mtime": 0})
    per_run = pd.read_csv(PER_RUN, float_precision="round_trip", low_memory=False)
    per_run["baseline"] = per_run["baseline"].astype(str).str.lower().isin(("true", "1"))
    per_run["variant"] = per_run["variant"].fillna("").astype(str)
    cells, lines = score(per_run)
    cells.to_csv(CELLS, index=False)
    text = "\n".join(lines) + "\n"
    REGISTER.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
