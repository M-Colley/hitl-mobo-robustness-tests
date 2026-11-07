"""The three experiments of the 2026-11-07 review round, scored as configs/review_round_plan.json fixes.

Three mock TMLR reviews asked for (E1) an untouched test of the end-of-study claims under all four error
processes, (E2) a drift of the other sign and (E3) a second standard GP surrogate. run_review_experiments.sh ran
them; the plan, frozen before any run, names every estimand, cell, seed set and test, and its sha256 is checked
here before anything is scored (PLAN_SHA256). All three are post hoc with respect to the paper's procedures and
lie outside the paper's Holm families; E1's own family (H1-H4) is Holm-corrected here as the plan declares.

Estimand (the repository's, analyse_boba_adaptations): deployed design at the final trial, regret as a fraction
of opt_z; gain = standard process noisy regret - procedure noisy regret; price = procedure clean - standard clean;
cost = standard noisy - standard clean; shares are ratios of landscape means. The sitting rows and the
zero-trial ship rules are joined run by run with sitting_by_magnitude's own functions (load, attach_ship_rules),
which stop unless the rescored best-observed regret reproduces the replay's standard process to 1e-9.

Intervals: the paper's percentile landscape bootstrap (2,000 resamples), a t interval with 19 degrees of freedom
over the landscape means, and a two-way landscape x seed bootstrap (landscapes and seeds resampled independently,
as sitting_by_magnitude.two_way_ci does). Tests: two-sided Wilcoxon over the twenty landscape means.

    python scripts/checks/review_experiments.py verify-plan
    python scripts/checks/review_experiments.py replay --workers 4     # rescoring and sitting replays of the new arms
    python scripts/checks/review_experiments.py score > output-boba/analysis/review/register_checks/review_experiments.txt

Writes review/review_experiments_{e1,e2,e3}.csv beside the register.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyse_boba_adaptations as aba  # noqa: E402
import sitting_by_magnitude as sbm  # noqa: E402

PLAN = REPO / "configs" / "review_round_plan.json"
PLAN_SHA256 = "75ccf817d949cc702ff2716049354511c126b7e348e590d0b86e043efbbe7cd2"
STATS = REPO / "boba_landscape_stats.json"
REVIEW = REPO / "output-boba" / "analysis" / "review"

CELLS = {"A": (1.0, 0), "B": (5.0, 20)}
CELL_LABEL = {"A": "1 sigma from the first rating", "B": "5 sigma from trial 21"}
SITTING_K = {"A": 2, "B": 16}
ACQS = ("logei", "qnei")
FRESH_SEEDS = tuple(range(27, 37))
MAIN_SEEDS = tuple(range(7, 17))
E1_DIRS = {"A": "output-boba-confirm4p-1s-t1", "B": "output-boba-confirm4p-5s-t21"}
E2_DIRS = {"A": "output-boba-driftdown-1s-t1", "B": "output-boba-driftdown-5s-t21"}
E3_DIRS = {"A": "output-boba-gplegacy-1s-t1", "B": "output-boba-gplegacy-5s-t21"}
E1_MODELS = ("bias", "drift", "ar1")
PROCESSES = ("gaussian", "bias", "drift", "ar1")
REPLAY = "end_of_study_review"
FRESH = REPO / "output-boba-confirmatory" / "analysis"
FRESH_REPLAYS = {"A": "end_of_study_fresh", "B": "end_of_study_fresh_k16"}
FRESH_SHIP = FRESH / "ship_rules_fresh" / "ship_rules_per_run.csv"
MAIN = REPO / "output-boba" / "analysis"
MAIN_REPLAYS = {"A": "end_of_study_ksweep", "B": "end_of_study_kwide"}
MAIN_SHIP = MAIN / "ship_rules_per_run.csv"
REPS = 2000
SEED = 20261107


# ---------------------------------------------------------------------------
# the plan and the replays
# ---------------------------------------------------------------------------


def verify_plan() -> str:
    digest = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    if digest != PLAN_SHA256:
        raise SystemExit(f"{PLAN.name} has sha256 {digest}, not the {PLAN_SHA256} recorded before the runs: "
                         "the plan was edited after it was frozen")
    return digest


def replay(workers: int, only: tuple[str, ...] = ("E1", "E2", "E3")) -> None:
    py = sys.executable
    jobs = []
    for exp, dirs, models, seeds in (("E1", E1_DIRS, ",".join(E1_MODELS), FRESH_SEEDS),
                                     ("E2", E2_DIRS, "drift_down", MAIN_SEEDS),
                                     ("E3", E3_DIRS, ",".join(PROCESSES), MAIN_SEEDS)):
        if exp not in only:
            continue
        for cell, d in dirs.items():
            arm = REPO / d
            if not (arm / "ARM_COMPLETE").is_file():
                raise SystemExit(f"{d} is not complete (no ARM_COMPLETE); run run_review_experiments.sh first")
            seed_arg = ",".join(str(s) for s in seeds)
            jobs.append([py, str(REPO / "scripts" / "rescore_ship_rules.py"), "--input-dir", str(arm),
                         "--acquisitions", ",".join(ACQS), "--seeds", seed_arg, "--workers", str(workers), "--resume"])
            if exp in ("E1", "E2"):   # the sitting replay refits only the default GP, so E3 has none
                std, onset = CELLS[cell]
                jobs.append([py, str(REPO / "scripts" / "replay_end_of_study.py"), "--arms", str(arm),
                             "--output-dir", str(arm / "analysis" / REPLAY), "--acquisitions", ",".join(ACQS),
                             "--seeds", seed_arg, "--error-models", models, "--stds", f"{std}",
                             "--onsets", str(onset), "--tournament-k", str(SITTING_K[cell]), "--confirm-k", "",
                             "--rho", "1", "--rerate-dirs", "none", "--workers", str(workers), "--resume"])
    for cmd in jobs:
        print("[replay]", " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def sitting_rows(analysis: Path, replay_dir: str, ship_csv: Path, cell: str, seeds: tuple[int, ...],
                 processes: tuple[str, ...]) -> pd.DataFrame:
    """One row per noisy run of the cell: the sitting of SITTING_K[cell] joined to the zero-trial ship rules."""
    d = sbm.load(analysis, STATS, processes=processes, replay_dirs=(replay_dir,))
    std, onset = CELLS[cell]
    d = d[(d["k"] == SITTING_K[cell]) & np.isclose(d["sigma"], std) & (d["onset"] == onset)
          & d["seed"].isin(seeds) & d["acquisition"].isin(ACQS)]
    if d.empty:
        raise ValueError(f"{analysis / replay_dir}: no sitting rows for cell {cell}")
    ship = pd.read_csv(ship_csv, low_memory=False)
    ship = ship[ship["acquisition"].isin(ACQS) & ship["seed"].isin(seeds)]
    return sbm.attach_ship_rules(d, ship)


def ship_rows(arm: Path, ship_csv: Path, cell: str, seeds: tuple[int, ...], processes: tuple[str, ...]) -> pd.DataFrame:
    """One row per noisy run of the cell: the standard process and the zero-trial rules, and the search loss."""
    ship = pd.read_csv(ship_csv, low_memory=False)
    ship["baseline"] = ship["baseline"].astype(str).str.lower().isin(("true", "1"))
    ship = ship[ship["acquisition"].isin(ACQS) & ship["seed"].isin(seeds)]
    std, onset = CELLS[cell]
    noisy = ship[~ship["baseline"]]   # a clean twin has no magnitude or onset
    noisy = noisy[np.isclose(noisy["jitter_std"].astype(float), std)
                  & (noisy["jitter_iteration"].astype(float) == onset) & noisy["error_model"].isin(processes)]
    clean = ship[ship["baseline"]][["dataset", "acquisition", "seed", "regret_best_observed", "regret_lcb1",
                                     "regret_lcb2", "regret_pm"]]
    m = noisy.merge(clean, on=["dataset", "acquisition", "seed"], suffixes=("", "_clean"), validate="many_to_one")
    opt_z = sbm.read_opt_z(STATS)
    z = m["dataset"].map(opt_z)
    m = m.assign(opt_z=z, cost=(m["regret_best_observed"] - m["regret_best_observed_clean"]) / z)
    for r in ("lcb1", "lcb2", "pm"):
        m[f"gain_{r}"] = (m["regret_best_observed"] - m[f"regret_{r}"]) / z
        m[f"price_{r}"] = (m[f"regret_{r}_clean"] - m["regret_best_observed_clean"]) / z
    # The search loss of the same runs, from the arm's evaluation (final trial, excess over the clean twin).
    ev = pd.concat([pd.read_csv(f, usecols=["dataset", "acquisition", "seed", "error_model", "jitter_std",
                                             "jitter_iteration", "final_simple_regret_excess_true",
                                             "final_inference_simple_regret_excess_true",
                                             "final_inference_simple_regret_true_baseline"])
                    for f in sorted(arm.glob("*/evaluation/paired_excess_metrics.csv"))], ignore_index=True)
    ev = ev[ev["acquisition"].isin(ACQS) & ev["seed"].isin(seeds) & ev["error_model"].isin(processes)
            & np.isclose(ev["jitter_std"].astype(float), std) & (ev["jitter_iteration"].astype(int) == onset)]
    keys = ["dataset", "acquisition", "seed", "error_model"]
    m = m.merge(ev[keys + ["final_simple_regret_excess_true", "final_inference_simple_regret_excess_true",
                           "final_inference_simple_regret_true_baseline"]], on=keys, validate="one_to_one")
    gap = float(((m["final_inference_simple_regret_excess_true"] / z) - m["cost"]).abs().max())
    if gap > 1e-9:
        raise ValueError(f"{arm.name}: the rescored standard process does not reproduce the evaluation's deployed excess ({gap:g})")
    m["search"] = m["final_simple_regret_excess_true"] / m["opt_z"]
    m["selection"] = m["cost"] - m["search"]
    m["clean_deployed"] = m["final_inference_simple_regret_true_baseline"] / m["opt_z"]
    return m


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def landscape_means(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("dataset")[col].mean()


def two_way(df: pd.DataFrame, num: str, den: str | None, rng: np.random.Generator) -> tuple[float, float]:
    """Resample landscapes and seeds independently; per-landscape means over the resampled seeds, then the
    mean over landscapes (or the ratio of landscape means when ``den`` is given)."""
    cols = [num] + ([den] if den else [])
    table = {c: df.pivot_table(index="dataset", columns="seed", values=c, aggfunc="mean") for c in cols}
    a = table[num].to_numpy()
    b = table[den].to_numpy() if den else None
    nl, ns = a.shape
    draws = np.empty(REPS)
    for r in range(REPS):
        li, si = rng.integers(0, nl, nl), rng.integers(0, ns, ns)
        x = np.nanmean(a[np.ix_(li, si)], axis=1)
        draws[r] = np.nanmean(x) / np.nanmean(np.nanmean(b[np.ix_(li, si)], axis=1)) if den else np.nanmean(x)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def t_interval(x: np.ndarray, y: np.ndarray | None = None) -> tuple[float, float]:
    """t interval with n - 1 df over landscape means; for a ratio of means a delta-method standard error."""
    n = len(x)
    q = sps.t.ppf(0.975, n - 1)
    if y is None:
        m, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(n))
    else:
        m = float(np.mean(x) / np.mean(y))
        resid = (x - m * y) / np.mean(y)
        se = float(np.std(resid, ddof=1) / np.sqrt(n))
    return m - q * se, m + q * se


def summarise(df: pd.DataFrame, col: str, rng: np.random.Generator, cost: str | None = "cost") -> dict:
    """Mean over landscapes of ``col`` with the three intervals and Wilcoxon, plus its share of ``cost``."""
    v = landscape_means(df, col)
    lo, hi = sbm.boot_ci(v.to_numpy(), rng)
    tlo, thi = t_interval(v.to_numpy())
    wlo, whi = two_way(df, col, None, rng)
    out = {"mean": float(v.mean()), "lo": lo, "hi": hi, "t_lo": tlo, "t_hi": thi, "tw_lo": wlo, "tw_hi": whi,
           "landscapes_positive": int((v > 0).sum()), "n_landscapes": int(len(v)), "wilcoxon_p": sbm.wilcoxon_p(v.to_numpy())}
    if cost is not None:
        c = landscape_means(df, cost).reindex(v.index)
        frame = pd.DataFrame({"ref": c.to_numpy(), "trt": (c - v).to_numpy()})
        share = aba.share_of_reference(frame)
        out["share"] = share["share_removed"]
        out["share_suppressed"] = share["share_suppressed"]
        if share["share_suppressed"] == "":
            slo, shi = sbm.ratio_ci(v.to_numpy(), c.to_numpy(), rng)
            out.update(share_lo=slo, share_hi=shi)
    return out


def fmt(s: dict, share: bool = True) -> str:
    text = (f"{s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]  t [{s['t_lo']:+.3f}, {s['t_hi']:+.3f}]  "
            f"two-way [{s['tw_lo']:+.3f}, {s['tw_hi']:+.3f}]  {s['landscapes_positive']}/{s['n_landscapes']} "
            f"positive, p {s['wilcoxon_p']:.2g}")
    if share and "share" in s:
        text += (f"  share {s['share']:.1%} [{s.get('share_lo', np.nan):.0%}, {s.get('share_hi', np.nan):.0%}]"
                 if s["share_suppressed"] == "" else f"  (no share: {s['share_suppressed']})")
    return text


def holm(p: list[float]) -> list[float]:
    return list(sbm.holm(np.asarray(p, dtype=float)))


# ---------------------------------------------------------------------------
# E1, E2, E3
# ---------------------------------------------------------------------------


def clean_twin_control() -> str:
    """The plan's control: E1's clean twins equal output-boba-confirmatory's, run for run, in every column the
    scores read. They were simulated by today's code, the confirmatory ones months earlier."""
    cols = ["iteration", "objective_true", "objective_observed", "inference_simple_regret_true", "simple_regret_true"]
    n = same = 0
    worst = 0.0
    for cell, d in E1_DIRS.items():
        for f in sorted((REPO / d).glob("*/bo_sensor_error_*_baseline_exact.csv")):
            ref = REPO / "output-boba-confirmatory" / f.parent.name / f.name
            if not ref.is_file():
                raise FileNotFoundError(f"no confirmatory clean twin for {d}/{f.parent.name}/{f.name}")
            a = pd.read_csv(f, usecols=lambda c: c in cols, float_precision="round_trip")
            b = pd.read_csv(ref, usecols=lambda c: c in cols, float_precision="round_trip")
            n += 1
            if list(a.columns) != list(b.columns) or len(a) != len(b):
                continue
            diff = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))
            worst = max(worst, diff)
            same += diff == 0.0
    if n == 0:
        # The run logs are git-ignored (AGENTS.md), so a checkout without them would otherwise pass the
        # control with 0 of 0 twins compared.
        raise FileNotFoundError(f"control: no clean run logs under {', '.join(E1_DIRS.values())}; they stay on "
                                "the simulation machine, where this check must run")
    if same != n:
        raise ValueError(f"control failed: {n - same} of {n} clean twins differ from output-boba-confirmatory's")
    return f"control: {same} of {n} clean twins equal output-boba-confirmatory's in {cols} (largest difference {worst:g})"


def score_e1(rng: np.random.Generator) -> pd.DataFrame:
    print("== E1: seeds 27-36, LogEI and qNEI, four processes pooled (gaussian from output-boba-confirmatory)")
    print("   " + clean_twin_control())
    rows, primary, fresh = [], {}, {}
    for cell in ("A", "B"):
        new = sitting_rows(REPO / E1_DIRS[cell] / "analysis", REPLAY, REPO / E1_DIRS[cell] / "analysis" /
                           "ship_rules_per_run.csv", cell, FRESH_SEEDS, E1_MODELS)
        gau = sitting_rows(FRESH, FRESH_REPLAYS[cell], FRESH_SHIP, cell, FRESH_SEEDS, ("gaussian",))
        both = pd.concat([gau, new], ignore_index=True)
        if set(both["error_model"]) != set(PROCESSES) or both.duplicated(subset=["file"]).any():
            raise ValueError(f"E1 cell {cell}: expected each run once under the four processes")
        print(f"-- cell {cell}: {CELL_LABEL[cell]}, sitting of {SITTING_K[cell]}; {len(both)} noisy runs, "
              f"standard process's cost {landscape_means(both, 'cost').mean():.3f} of opt_z")
        quantities = [("sitting", "gain"), ("LCB1", "gain_lcb1"), ("LCB2", "gain_lcb2"),
                      ("sitting over LCB1", "inc_lcb1"), ("sitting over LCB2", "inc_lcb2")]
        for name, col in quantities:
            s = summarise(both, col, rng, cost=None if name.startswith("sitting over") else "cost")
            print(f"   {name:<18s} {fmt(s)}")
            rows.append({"experiment": "E1", "cell": cell, "process": "pooled", "quantity": name, **s})
            primary[(cell, name)] = s
        for name, col in (("sitting price", "price"), ("LCB1 price", "price_lcb1"), ("LCB2 price", "price_lcb2")):
            v = landscape_means(both, col)
            print(f"   {name:<18s} {v.mean():.3f}")
            rows.append({"experiment": "E1", "cell": cell, "process": "pooled", "quantity": name, "mean": float(v.mean())})
        for proc in PROCESSES:
            sub = both[both["error_model"] == proc]
            parts = []
            for name, col in quantities:
                v = landscape_means(sub, col)
                parts.append(f"{name} {v.mean():+.3f}")
                rows.append({"experiment": "E1", "cell": cell, "process": proc, "quantity": name, "mean": float(v.mean())})
            print(f"     {proc:<9s} " + ", ".join(parts))
        fresh[cell] = both[both["error_model"].isin(E1_MODELS)]
    # The family the plan declares.
    family = [("H1", "A", "LCB2"), ("H2", "B", "sitting"), ("H3", "B", "sitting over LCB1"), ("H4", "B", "sitting over LCB2")]
    adj = holm([primary[(c, q)]["wilcoxon_p"] for _, c, q in family])
    print("-- primary family (configs/review_round_plan.json), Holm over H1-H4:")
    for (h, c, q), a in zip(family, adj):
        s = primary[(c, q)]
        print(f"   {h}: {q} in cell {c}: {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}], Wilcoxon p {s['wilcoxon_p']:.2g}, "
              f"Holm {a:.2g} -> {'positive' if (a < 0.05 and s['mean'] > 0) else 'not shown'}")
        rows.append({"experiment": "E1", "cell": c, "process": "pooled", "quantity": f"{h} Holm", "mean": s["mean"],
                     "wilcoxon_p": s["wilcoxon_p"], "holm_p": a})
    # Beyond the plan (descriptive): the gaussian runs of seeds 27-36 had been scored before the plan was fixed,
    # so the same quantities on the three processes whose runs are new to this replication.
    print("-- beyond the plan: the three processes new to this replication (bias, drift, ar1), pooled:")
    for cell, names in (("A", ("LCB2", "LCB1", "sitting", "sitting over LCB1")),
                        ("B", ("sitting", "sitting over LCB1", "sitting over LCB2"))):
        cols = {"sitting": "gain", "LCB1": "gain_lcb1", "LCB2": "gain_lcb2", "sitting over LCB1": "inc_lcb1",
                "sitting over LCB2": "inc_lcb2"}
        for name in names:
            s = summarise(fresh[cell], cols[name], rng, cost=None if name.startswith("sitting over") else "cost")
            print(f"   cell {cell} {name:<18s} {fmt(s)}")
            rows.append({"experiment": "E1", "cell": cell, "process": "bias+drift+ar1", "quantity": name, **s})
    return pd.DataFrame(rows)


def score_e2(rng: np.random.Generator) -> pd.DataFrame:
    print("== E2: drift_down against the main sweep's drift, seeds 7-16, LogEI and qNEI (paired noise draws)")
    rows = []
    for cell in ("A", "B"):
        down_dir = REPO / E2_DIRS[cell]
        down = ship_rows(down_dir, down_dir / "analysis" / "ship_rules_per_run.csv", cell, MAIN_SEEDS, ("drift_down",))
        up = ship_rows(REPO / "output-boba", MAIN_SHIP, cell, MAIN_SEEDS, ("drift",))
        sit_down = sitting_rows(down_dir / "analysis", REPLAY, down_dir / "analysis" / "ship_rules_per_run.csv",
                                cell, MAIN_SEEDS, ("drift_down",))
        sit_up = sitting_rows(MAIN, MAIN_REPLAYS[cell], MAIN_SHIP, cell, MAIN_SEEDS, ("drift",))
        print(f"-- cell {cell}: {CELL_LABEL[cell]}")
        for label, a, b, col, cost in (("cost of error", down, up, "cost", None),
                                       ("final search loss", down, up, "search", None),
                                       ("LCB1 gain", down, up, "gain_lcb1", "cost"),
                                       ("LCB2 gain", down, up, "gain_lcb2", "cost"),
                                       (f"sitting of {SITTING_K[cell]} gain", sit_down, sit_up, "gain", "cost")):
            sd, su = summarise(a, col, rng, cost=cost), summarise(b, col, rng, cost=cost)
            diff = (landscape_means(a, col) - landscape_means(b, col)).dropna()
            dlo, dhi = sbm.boot_ci(diff.to_numpy(), rng)
            print(f"   {label:<20s} down {fmt(sd)}\n   {'':<20s} up   {fmt(su)}\n"
                  f"   {'':<20s} down - up {diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}], p {sbm.wilcoxon_p(diff.to_numpy()):.2g}")
            rows.append({"experiment": "E2", "cell": cell, "quantity": label, "down": sd["mean"], "up": su["mean"],
                         "diff": float(diff.mean()), "diff_lo": dlo, "diff_hi": dhi,
                         "down_share": sd.get("share"), "up_share": su.get("share")})
        for name, d in (("down", down), ("up", up)):
            share = landscape_means(d, "selection").mean() / landscape_means(d, "cost").mean()
            print(f"   selection share of the cost, {name}: {share:.1%}")
            rows.append({"experiment": "E2", "cell": cell, "quantity": f"selection share {name}", "mean": share})
        # Beyond the plan's list (descriptive): the sitting's paired increments over the zero-trial rules.
        for name, d in (("down", sit_down), ("up", sit_up)):
            for col, label in (("inc_lcb1", "over LCB1"), ("inc_lcb2", "over LCB2")):
                s = summarise(d, col, rng, cost=None)
                print(f"   sitting of {SITTING_K[cell]} {label}, {name} (beyond the plan): {fmt(s, share=False)}")
                rows.append({"experiment": "E2", "cell": cell, "quantity": f"sitting {label} {name}", "mean": s["mean"],
                             "lo": s["lo"], "hi": s["hi"], "wilcoxon_p": s["wilcoxon_p"]})
    return pd.DataFrame(rows)


def score_e3(rng: np.random.Generator) -> pd.DataFrame:
    print("== E3: BoTorch's pre-0.12 default GP (legacy) against its current default, seeds 7-16, LogEI and qNEI, "
          "four processes (paired noise draws)")
    rows = []
    for cell in ("A", "B"):
        leg_dir = REPO / E3_DIRS[cell]
        leg = ship_rows(leg_dir, leg_dir / "analysis" / "ship_rules_per_run.csv", cell, MAIN_SEEDS, PROCESSES)
        dft = ship_rows(REPO / "output-boba", MAIN_SHIP, cell, MAIN_SEEDS, PROCESSES)
        print(f"-- cell {cell}: {CELL_LABEL[cell]}")
        for label, col, cost in (("cost of error", "cost", None), ("final search loss", "search", None),
                                 ("clean twin's deployed regret", "clean_deployed", None),
                                 ("LCB1 gain", "gain_lcb1", "cost"), ("LCB2 gain", "gain_lcb2", "cost"),
                                 ("posterior-mean gain", "gain_pm", "cost")):
            sl, sd = summarise(leg, col, rng, cost=cost), summarise(dft, col, rng, cost=cost)
            diff = (landscape_means(leg, col) - landscape_means(dft, col)).dropna()
            dlo, dhi = sbm.boot_ci(diff.to_numpy(), rng)
            print(f"   {label:<29s} legacy  {fmt(sl)}\n   {'':<29s} default {fmt(sd)}\n"
                  f"   {'':<29s} legacy - default {diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}], "
                  f"p {sbm.wilcoxon_p(diff.to_numpy()):.2g}")
            rows.append({"experiment": "E3", "cell": cell, "quantity": label, "legacy": sl["mean"], "default": sd["mean"],
                         "diff": float(diff.mean()), "diff_lo": dlo, "diff_hi": dhi,
                         "legacy_share": sl.get("share"), "default_share": sd.get("share")})
        for name, d in (("legacy", leg), ("default", dft)):
            sel = landscape_means(d, "selection")
            share = sel.mean() / landscape_means(d, "cost").mean()
            slo, shi = sbm.ratio_ci(sel.to_numpy(), landscape_means(d, "cost").reindex(sel.index).to_numpy(), rng)
            print(f"   selection share of the cost, {name}: {share:.1%} [{slo:.0%}, {shi:.0%}]")
            rows.append({"experiment": "E3", "cell": cell, "quantity": f"selection share {name}", "mean": share,
                         "lo": slo, "hi": shi})
        for name, col in (("LCB1 price", "price_lcb1"), ("LCB2 price", "price_lcb2")):
            print(f"   {name}: legacy {landscape_means(leg, col).mean():.3f}, default {landscape_means(dft, col).mean():.3f}")
        # Beyond the plan (descriptive point estimates, no resampling, so no earlier interval moves): where the
        # differences sit, by acquisition and by error process.
        for by, levels in (("acquisition", ACQS), ("error_model", PROCESSES)):
            for level in levels:
                a, b = leg[leg[by] == level], dft[dft[by] == level]
                parts = []
                for label, col in (("cost", "cost"), ("search", "search"), ("selection", "selection"),
                                   ("LCB2 gain", "gain_lcb2")):
                    va, vb = landscape_means(a, col).mean(), landscape_means(b, col).mean()
                    parts.append(f"{label} {va:.3f} / {vb:.3f}")
                    rows.append({"experiment": "E3", "cell": cell, "quantity": f"{label} by {by} {level} (beyond the plan)",
                                 "legacy": float(va), "default": float(vb), "diff": float(va - vb)})
                print(f"   beyond the plan, {by} {level:<9s} legacy / default: " + ", ".join(parts))
    return pd.DataFrame(rows)


def score(only: tuple[str, ...] = ("E1", "E2", "E3")) -> None:
    digest = verify_plan()
    print(f"Review-round experiments (configs/review_round_plan.json, sha256 {digest[:16]}..., frozen before the runs).")
    print("Deployed design at the final trial, fractions of opt_z; [percentile landscape bootstrap], t (19 df), "
          "two-way landscape x seed bootstrap; Wilcoxon over the twenty landscape means. Post hoc with respect to "
          "the paper's procedures.\n")
    REVIEW.mkdir(parents=True, exist_ok=True)
    for i, (name, fn) in enumerate((("e1", score_e1), ("e2", score_e2), ("e3", score_e3))):
        if name.upper() not in only:
            continue
        # One generator per experiment, so an experiment's intervals do not depend on which others are scored.
        fn(np.random.default_rng([SEED, i])).to_csv(REVIEW / f"review_experiments_{name}.csv", index=False)
        print()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-plan")
    r = sub.add_parser("replay")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--only", default="E1,E2,E3", help="comma-separated experiments to replay")
    s = sub.add_parser("score")
    s.add_argument("--only", default="E1,E2,E3", help="comma-separated experiments to score")
    args = p.parse_args(argv)
    only = tuple(x.strip().upper() for x in args.only.split(",")) if hasattr(args, "only") else ()
    if args.cmd == "verify-plan":
        print(f"plan sha256 {verify_plan()} matches the value recorded before the runs")
    elif args.cmd == "replay":
        verify_plan()
        replay(args.workers, only)
    else:
        score(only)


if __name__ == "__main__":
    main()
