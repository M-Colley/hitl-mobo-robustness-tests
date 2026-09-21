"""Three remedies that need no new simulation, replayed from the logged runs.

The decomposition says the deployed cost is mostly SELECTION: the run visits a
better design than the one it ships. Every acquisition-side change tested so far
fails, and what works is either spending trials on identification or modelling
how the rating was produced. These three follow that grain and cost nothing to
evaluate, because each is a different reading of runs that already exist.

  shortlist_m    Ship m designs instead of one and let a later, cheaper decision
                 pick among them; scored as the regret of the best of the m.
                 m = 1 is exactly the cautious ship rule, so the m-curve prices
                 the whole idea against its own baseline.

  ordinal        Refit the surrogate on the RANKS of the ratings (their normal
                 scores) rather than their values, and ship its argmax. Ranks are
                 invariant to any strictly monotone distortion of the scale and
                 are robust to a single wild rating, so this buys part of what a
                 comparison loop buys without spending one extra human trial.
                 It should help against spikes and a saturating cap and do
                 nothing, or a little harm, against pure gaussian noise, where
                 discarding the magnitudes throws information away.

  lucb_sitting   The final comparative sitting of k looks, allocated
                 sequentially to the empirical leader and its closest challenger
                 (LUCB, the standard best-arm identification rule
                 \\citep{audibert2010best, kim2006selecting}) instead of one look
                 each to the top k. Blind replication of the first ten trials
                 fails because it spends trials where the ranking is not in
                 doubt; this spends the same trials where it is. Scored against
                 the fixed-allocation tournament at the same k, on the same runs.

Every procedure is also replayed on the identically seeded CLEAN twin, so the
output feeds the standard-process estimand of analyse_boba_adaptations.py
directly: cost = ref_noisy - ref_clean, gain = ref_noisy - trt_noisy,
price = trt_clean - ref_clean.

    python scripts/replay_hitl_remedies.py --input-dir output-boba --workers 8
    python scripts/replay_hitl_remedies.py --input-dir output-boba-spike --error-models spike
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import boba_benchmarks as bb  # noqa: E402
import replay_end_of_study as eos  # noqa: E402

OUTPUT_NAME = "hitl_remedies"
SHORTLIST_SIZES = (1, 2, 3, 5)
PROC_LUCB = 91          # a stream of its own, disjoint from the tournament's
PROC_ORDINAL = 92


# ---------------------------------------------------------------------------
# The three procedures
# ---------------------------------------------------------------------------


def shortlist(run: eos.RunLog, state: eos.SearchState) -> dict:
    """Ship the top m designs by the cautious rule; score the best of them.

    No extra trial is spent: the same run, the same rule, a wider deliverable.
    m = 1 is the cautious rule itself, which is what makes the curve readable.
    """
    order = np.argsort(-state.lcb, kind="stable")
    out = {}
    for m in SHORTLIST_SIZES:
        rows = state.first[order[:m]]
        truth = run.deployed[rows]
        out[f"shortlist_m{m}"] = {
            "family": "shortlist", "k": 0, "m": int(len(rows)),
            "regret": run.y_opt - float(truth.max()),
            "n_candidates": int(len(rows)),
            "pick_changed": bool(int(np.argmax(truth)) != 0),
            "gp_failed": state.gp_failed,
        }
    return out


def normal_scores(values: np.ndarray) -> np.ndarray:
    """Van der Waerden scores: the ranks mapped through the normal quantile.

    Any strictly increasing map of the ratings leaves these unchanged, which is
    the whole point; ties share their average rank, so a saturating cap collapses
    to a tie rather than to an arbitrary order.
    """
    from scipy.stats import norm, rankdata

    v = np.asarray(values, dtype=float)
    ranks = rankdata(v, method="average")
    return norm.ppf(ranks / (len(v) + 1.0))


def ordinal_rule(run: eos.RunLog, state: eos.SearchState, bounds: torch.Tensor, beta: float) -> dict:
    """Refit on the ratings' normal scores and ship the argmax of that fit."""
    n = state.n
    X = run.X[:n]
    z = normal_scores(run.observed[:n])
    train_X = torch.tensor(X, dtype=torch.double)
    train_Y = torch.tensor(z.reshape(-1, 1), dtype=torch.double)
    failed = False
    try:
        gp = eos.fit_loop_gp(train_X, train_Y, bounds, eos.torch_seed(run.name, n + PROC_ORDINAL))
        mean, sd = eos.latent_mean_sd(gp, X[state.first])
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sd)):
            raise FloatingPointError("non-finite ordinal posterior")
    except Exception as exc:  # noqa: BLE001 - recorded per run, never silently absorbed
        print(f"[ordinal-fallback] {run.name} n={n}: {exc}", file=sys.stderr)
        mean, sd, failed = state.obs_mean.copy(), np.zeros_like(state.obs_mean), True
    out = {}
    for label, crit in (("ordinal_pm", mean), (f"ordinal_lcb{beta:g}", mean - beta * sd)):
        pick = int(np.argmax(crit))
        out[label] = {
            "family": "ordinal", "k": 0, "m": int(len(state.first)),
            "regret": run.y_opt - float(run.deployed[state.first[pick]]),
            "n_candidates": int(len(state.first)),
            "pick_changed": bool(pick != int(np.argmax(state.lcb))),
            "gp_failed": bool(state.gp_failed or failed),
        }
    return out


def lucb_sitting(run: eos.RunLog, state: eos.SearchState, k: int, rho: float,
                 sd_of_trial, T: int, pool_factor: int = 3) -> dict:
    """Spend k looks on identification, allocated where the ranking is in doubt.

    The fixed tournament gives one look to each of the top k. This keeps a wider
    pool and gives each successive look to the empirical leader or to the
    challenger with the highest upper bound, alternating -- the single-look form
    of LUCB. With the same k it can look at one pair four times instead of four
    designs once, which is what "replicate only where it matters" means.
    """
    pool = max(2, min(pool_factor * k, len(state.first)))
    rows = state.first[np.argsort(-state.lcb, kind="stable")[:pool]]
    truth = run.deployed[rows]
    m = len(rows)
    sds = np.array([sd_of_trial(t) for t in range(T - k + 1, T + 1)], dtype=float)
    rng = np.random.default_rng(eos.noise_seed(run.name, PROC_LUCB, k, int(rho * 1000)))
    sums = np.zeros(m)
    counts = np.zeros(m, dtype=int)
    for step in range(k):
        if step < 2 and m >= 2:
            j = step                      # one look each to the top two, to start
        else:
            seen = counts > 0
            means = np.where(seen, sums / np.maximum(counts, 1), -np.inf)
            leader = int(np.argmax(means))
            # The challenger is the most plausible alternative: highest upper
            # bound among the rest, with anything unseen ranked first.
            width = np.where(seen, 1.0 / np.sqrt(np.maximum(counts, 1)), np.inf)
            upper = means + width
            upper[leader] = -np.inf
            challenger = int(np.argmax(upper))
            j = leader if step % 2 == 0 else challenger
        sums[j] += truth[j] + rho * sds[step] * rng.standard_normal()
        counts[j] += 1
    seen = counts > 0
    means = np.where(seen, sums / np.maximum(counts, 1), -np.inf)
    pick = int(np.argmax(means))
    return {
        f"lucb_k{k}_rho{rho:g}": {
            "family": "lucb", "k": int(k), "m": int(m),
            "regret": run.y_opt - float(truth[pick]),
            "n_candidates": int(m),
            "pick_changed": bool(pick != 0),
            "gp_failed": state.gp_failed,
            "n_distinct_looked": int(seen.sum()),
            "max_looks_on_one": int(counts.max()),
            # what the same k looks buy with one each, for a like-for-like read
            "top_candidate_regret": run.y_opt - float(truth[0]),
            "best_candidate_regret": run.y_opt - float(truth.max()),
        }
    }


# ---------------------------------------------------------------------------
# Per-run assembly
# ---------------------------------------------------------------------------

META = ("family", "k", "m")


def replay_run(run: eos.RunLog, arm: eos.ArmInfo, settings: eos.Settings, bounds: torch.Tensor,
               sd_of_trial, ks: tuple[int, ...]) -> dict:
    """Every procedure on one run, plus the standard process as the reference."""
    T = arm.iterations
    eos.check_logged_inference(run, [T])
    out = {"standard": {"family": "standard", "k": 0, "m": 0,
                        "regret": float(run.logged_inference[-1]), "gp_failed": False}}
    # No extra trial: the state after the whole budget.
    full = eos.search_state(run, T, bounds, settings.lcb_beta)
    out.update(shortlist(run, full))
    out.update(ordinal_rule(run, full, bounds, settings.lcb_beta))
    # k trials spent on the sitting: the state before them, as the tournament does.
    for k in ks:
        state = eos.search_state(run, T - k, bounds, settings.lcb_beta)
        for rho in settings.rhos:
            out.update(lucb_sitting(run, state, k, rho, sd_of_trial, T))
        # The fixed-allocation comparator, from the production replay, on the
        # same state and the same schedule.
        out.update(eos.tournament(run, state, k, settings.rhos, sd_of_trial, T))
    return out


def replay_stem_rows(task: dict) -> pd.DataFrame:
    arm, settings, ks = task["arm"], task["settings"], tuple(task["ks"])
    spec = bb.BENCHMARKS[task["dataset"]]
    bounds = torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)
    clean = eos.read_run(Path(task["clean_path"]), arm.iterations)
    clean_out = replay_run(clean, arm, settings, bounds,
                           eos.sitting_sd_fn(arm, "none", 0.0, 0, settings), ks)
    rows: list[dict] = []
    for rec in task["runs"]:
        path = Path(rec["path"])
        noisy = eos.read_run(path, arm.iterations)
        sd_fn = eos.sitting_sd_fn(arm, rec["error_model"], rec["jitter_std"], rec["jitter_iteration"], settings)
        noisy_out = replay_run(noisy, arm, settings, bounds, sd_fn, ks)
        base = {"arm": arm.name, "dataset": task["dataset"], "acquisition": task["acquisition"],
                "seed": task["seed"], "error_model": rec["error_model"], "jitter_std": rec["jitter_std"],
                "jitter_iteration": rec["jitter_iteration"], "variant": rec["variant"], "file": path.name}
        ref_noisy = noisy_out["standard"]["regret"]
        ref_clean = clean_out["standard"]["regret"]
        for proc, n_res in noisy_out.items():
            c_res = clean_out.get(proc)
            if c_res is None:
                continue
            row = {**base, "procedure": proc, **{m: n_res.get(m) for m in META},
                   "regret_noisy": n_res["regret"], "regret_clean": c_res["regret"],
                   "ref_noisy": ref_noisy, "ref_clean": ref_clean}
            for suffix, res in (("noisy", n_res), ("clean", c_res)):
                for key, value in res.items():
                    if key not in META and key != "regret":
                        row[f"{key}_{suffix}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def replay_stem(task: dict) -> dict:
    start = time.perf_counter()
    frame = replay_stem_rows(task)
    out_path = Path(task["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out_path)
    return {"stem": task["stem"], "rows": len(frame), "runs": len(task["runs"]),
            "seconds": time.perf_counter() - start}


def _worker_init() -> None:
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def recovery_table(frame: pd.DataFrame, opt_z: dict[str, float], reps: int = 2000,
                   seed: int = 20260921) -> pd.DataFrame:
    """The standard-process estimand, per procedure and condition.

    Identical in form to analyse_boba_adaptations.summarise: a ratio of landscape
    means, resampled over landscapes, with the denominator refused when it is not
    positive rather than conditioned away.
    """
    rng = np.random.default_rng(seed)
    z = frame["dataset"].map(lambda d: opt_z.get(d, 1.0))
    f = frame.assign(**{c: frame[c] / z for c in ("regret_noisy", "regret_clean", "ref_noisy", "ref_clean")})
    rows = []
    keys = ["procedure", "error_model", "jitter_std", "jitter_iteration"]
    for scope, block in [("pooled", f)] + [(k, g) for k, g in f.groupby(keys, sort=True)]:
        if scope == "pooled":
            for proc, g in f.groupby("procedure", sort=True):
                rows.append(_summarise(g, {"procedure": proc, "error_model": "pooled",
                                           "jitter_std": np.nan, "jitter_iteration": np.nan}, rng, reps))
        else:
            proc, em, std, onset = scope
            rows.append(_summarise(block, {"procedure": proc, "error_model": em,
                                           "jitter_std": std, "jitter_iteration": onset}, rng, reps))
    return pd.DataFrame(rows)


def _summarise(block: pd.DataFrame, ident: dict, rng: np.random.Generator, reps: int) -> dict:
    per = block.groupby("dataset")[["ref_noisy", "ref_clean", "regret_noisy", "regret_clean"]].mean()
    cost_l = (per.ref_noisy - per.ref_clean).to_numpy()
    gain_l = (per.ref_noisy - per.regret_noisy).to_numpy()
    price_l = (per.regret_clean - per.ref_clean).to_numpy()
    n = len(per)
    draws = np.full(reps, np.nan)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        c = cost_l[idx].mean()
        if c > 0:
            draws[i] = gain_l[idx].mean() / c
    ok = np.isfinite(draws)
    usable = bool(cost_l.mean() > 0 and ok.mean() >= 0.95)
    return {
        **ident, "n_landscapes": int(n), "n_cells": int(len(block)),
        "cost": float(cost_l.mean()), "gain": float(gain_l.mean()), "price": float(price_l.mean()),
        "recovered": float(gain_l.mean() / cost_l.mean()) if cost_l.mean() > 0 else np.nan,
        "bootstrap_draws_kept": float(ok.mean()),
        "recovered_lo": float(np.nanpercentile(draws[ok], 2.5)) if usable else np.nan,
        "recovered_hi": float(np.nanpercentile(draws[ok], 97.5)) if usable else np.nan,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("output-boba"))
    p.add_argument("--output-dir", type=Path, default=None, help="default: <input-dir>/analysis/hitl_remedies")
    p.add_argument("--acquisitions", type=str, default="logei,qnei")
    p.add_argument("--seeds", type=str, default="7,8,9,10,11,12,13,14,15,16")
    p.add_argument("--functions", type=str, default="all")
    p.add_argument("--error-models", type=str, default=None)
    p.add_argument("--stds", type=str, default=None)
    p.add_argument("--onsets", type=str, default=None)
    p.add_argument("--variants", type=str, default=None,
                   help="Comma-separated run-name variants to replay, for a directory that holds "
                        "several arms (a spike size, a cap mode). Without it only the standard "
                        "process is taken, which is what a single-arm directory has.")
    p.add_argument("--sitting-k", type=str, default="4,8",
                   help="trials spent on the final sitting; 8 is where the fixed curve peaks")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    return p.parse_args(argv)


def _set(value: str | None, cast=str):
    if value is None or value.lower() == "all":
        return None
    return frozenset(cast(v.strip()) for v in value.split(",") if v.strip())


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = args.output_dir or (args.input_dir / "analysis" / "hitl_remedies")
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = eos.load_arm(args.input_dir)
    settings = eos.Settings()
    eos.validate_settings(settings)
    eos.validate_arm(arm, settings)
    ks = tuple(int(k) for k in args.sitting_k.split(","))
    stds = _set(args.stds, float)
    filters = eos.Filters(
        functions=_set(args.functions), acquisitions=_set(args.acquisitions),
        seeds=_set(args.seeds, int), error_models=_set(args.error_models),
        stds=None if stds is None else tuple(sorted(stds)), onsets=_set(args.onsets, int),
        variants=_set(args.variants),
    )
    tasks, counts = eos.build_tasks(arm, filters, settings, out_dir, resume=args.resume)
    for task in tasks:
        task["ks"] = ks
    print(f"arm {arm.name}: {len(tasks)} stems, {counts['noisy runs']} noisy runs, k = {ks}")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    if not tasks:
        raise SystemExit("nothing to replay")

    failures, done, total = [], 0, sum(len(t["runs"]) for t in tasks)
    start = time.perf_counter()

    def report(info: dict) -> None:
        nonlocal done
        done += info["runs"]
        el = time.perf_counter() - start
        print(f"  [{done}/{total}] {info['stem']}: {info['seconds']:.1f}s "
              f"(elapsed {el / 60:.1f} min, eta {el / max(done, 1) * (total - done) / 60:.1f} min)", flush=True)

    if args.workers <= 1:
        _worker_init()
        for task in tasks:
            try:
                report(replay_stem(task))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{task['stem']}: {exc!r}")
                print(f"  FAILED {task['stem']}: {exc!r}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            futures = {pool.submit(replay_stem, t): t["stem"] for t in tasks}
            for fut in as_completed(futures):
                try:
                    report(fut.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{futures[fut]}: {exc!r}")
                    print(f"  FAILED {futures[fut]}: {exc!r}", file=sys.stderr)
    if failures:
        (out_dir / "failures.log").write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"{len(failures)} stems failed; see {out_dir / 'failures.log'}", file=sys.stderr)

    parts = sorted((out_dir / "per_run" / arm.name).rglob("*.csv"))
    if not parts:
        raise SystemExit("no per-run output")
    frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    frame.to_csv(out_dir / f"{OUTPUT_NAME}_per_run.csv.gz", index=False)
    stats = bb.load_stats(args.stats_path)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    table = recovery_table(frame, opt_z)
    table.to_csv(out_dir / f"{OUTPUT_NAME}_recovery.csv", index=False)

    print(f"\nwrote {out_dir / (OUTPUT_NAME + '_per_run.csv.gz')} ({len(frame)} rows)")
    pooled = table[table.error_model == "pooled"].sort_values("recovered", ascending=False)
    print("\npooled recovery of the standard process's cost of error, deployed design:")
    for _, r in pooled.iterrows():
        if r.procedure == "standard":
            continue
        lo = "" if not np.isfinite(r.recovered_lo) else f" [{r.recovered_lo:+.0%}, {r.recovered_hi:+.0%}]"
        print(f"  {r.procedure:26s} {r.recovered:+7.1%}{lo}   price {r.price:+.4f}")


if __name__ == "__main__":
    main()
