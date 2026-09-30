"""The final sitting, cell by cell: where the pooled k-curve's gain comes from.

Companion to ``heldout_remedies.py`` and ``budget_split.py``. The paper's
k-curve (Figure 3) pools four error processes, four magnitudes and both onsets.
This script re-scores the same replays inside each (magnitude, onset) cell and,
in every cell, re-makes the choice of k on seeds 7-11 and scores it on seeds
12-16 (and the reverse), on random 10/10 landscape splits, with a Wilcoxon test
over the twenty per-landscape held-out gains and Holm's correction over the nine
values of k. It also gives the held-out gain as a share of the standard
process's cost of error in the cell, and a two-way (landscape x seed) bootstrap
for the selected k.

Inputs are the replays the k-curve is built from: ``end_of_study_ksweep`` and
``end_of_study_kwide`` (tournament, candidates by the posterior mean less one
latent SD, the best look ships, looks at the full idiosyncratic SD), LogEI and
qNEI, the four main error processes. Regret is divided by the landscape's
achievable improvement opt_z (``boba_landscape_stats.json``). Needs pandas,
numpy, scipy and statsmodels only; no simulation is run.

The zero-trial reference. The sitting's candidates are ranked by the posterior
mean less one latent SD, so the sitting is also compared with shipping by that
rule, and by its two neighbours, after all 50 trials with no extra trial, on the
same runs (``ship_rules_per_run.csv`` of ``rescore_ship_rules.py``): PM, the
visited design with the highest posterior mean; LCB1, the highest posterior mean
less one latent SD; LCB2, less two. Each run is joined by its file name, and the
join stops unless the rescoring's best-observed regret reproduces the replay's
standard process to 1e-9. The sitting's increment over LCB1 is, per run,
(regret of LCB1 - regret of the sitting) / opt_z, a paired difference, averaged
per landscape, with a landscape bootstrap, a two-sided Wilcoxon test over the
twenty per-landscape increments and Holm's correction over the nine k within the
cell, over the eight (cell, k chosen on seeds 7-11) pairs, and over all 72
(cell, k). The increment over the top candidate of the sitting's own prefix
(the LCB1 design after T - k trials, shipped without a look) is the value of the
looks alone. Because LCB1 does not depend on k and every k covers the same runs,
the k chosen on seeds 7-11 by gain is also the k with the largest increment.

The price. A procedure's price is the regret it adds to the clean twin,
(trt_clean - ref_clean) / opt_z, the estimand of analyse_boba_adaptations.py
(the sitting's looks are exact there). The clean twin is shared by every cell and
every error process, so a price depends on k and the seeds only.

The new quantities draw from their own generator, after every existing quantity
has drawn, so the existing columns and intervals are unchanged by them.

The other two zero-trial rules. LCB1 is the rule the sitting builds on, not
always the strongest rule that buys no trial: at 5 sigma from trial 21 the
cautious rule LCB2 gains more. The sitting's increments over LCB2 and over PM are
given with the same estimand, seeds and Holm families as over LCB1 (columns
inc_lcb2_* and inc_pm_*; summary keys increment_over_lcb2 and increment_over_pm),
from a fourth generator, so every earlier column is unchanged. Each cell's
summary also names the zero-trial rule with the largest gain on seeds 7-11, the
seeds that choose k, and gives the sitting's increment over that rule on seeds
12-16, with Holm's correction over the eight cells (increment_over_chosen_rule).

The look model. The replays above cancel the error that one sitting shares,
including the drift ramp and the AR(1) state. ``--replay-dirs`` rescores the
sequential replays of replay_end_of_study.py, in which that state moves on from
look to look (random order: end_of_study_ksweep_seq and _kwide_seq; rank order,
drift and AR(1) only: end_of_study_ksweep_seqrank and _kwide_seqrank, beside the
shared replay's gaussian and bias runs, which the sequential process leaves
unchanged). Those runs write only tagged outputs and no table.

    python scripts/sitting_by_magnitude.py
    python scripts/sitting_by_magnitude.py --processes gaussian --tag gaussian
    python scripts/sitting_by_magnitude.py --replay-dirs end_of_study_ksweep_seq,end_of_study_kwide_seq --tag seq
    python scripts/sitting_by_magnitude.py --tag seqrank --replay-dirs
        end_of_study_ksweep_seqrank=drift+ar1,end_of_study_kwide_seqrank=drift+ar1,
        end_of_study_ksweep=gaussian+bias,end_of_study_kwide=gaussian+bias
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

TRAIN_SEEDS = (7, 8, 9, 10, 11)
TEST_SEEDS = (12, 13, 14, 15, 16)
K_GRID = (2, 3, 5, 8, 12, 16, 20, 25, 30)
ACQS = ("logei", "qnei")
PROCESSES = ("gaussian", "bias", "drift", "ar1")
BOOTSTRAP_SEED = 20260925
REPS = 2000
N_SPLITS = 1000
# The zero-trial ship rules of rescore_ship_rules.py, and the seed sets they are scored on.
SHIP_RULES = ("pm", "lcb1", "lcb2")
INCREMENT_BOOTSTRAP_SEED = BOOTSTRAP_SEED + 2
# The rules other than LCB1 over which the sitting's increment is also given, and their generator.
OTHER_RULES = ("lcb2", "pm")
OTHER_RULES_BOOTSTRAP_SEED = BOOTSTRAP_SEED + 3
SEED_SETS = {"all": TRAIN_SEEDS + TEST_SEEDS, "train": TRAIN_SEEDS, "test": TEST_SEEDS}
REF_TOL = 1e-9


def read_opt_z(stats: Path) -> dict[str, float]:
    return {k: float(v["opt_z"]) for k, v in json.loads(Path(stats).read_text())["functions"].items()}


REPLAY_DIRS = ("end_of_study_ksweep", "end_of_study_kwide")


def parse_replay_dir(spec: str) -> tuple[str, tuple[str, ...] | None]:
    """'dir' or 'dir=drift+ar1': a replay directory, optionally read for those error processes only."""
    name, sep, only = spec.partition("=")
    if not sep:
        return name.strip(), None
    keep = tuple(p.strip() for p in only.split("+") if p.strip())
    if not keep:
        raise ValueError(f"no error process after '=' in {spec!r}")
    return name.strip(), keep


def load(analysis: Path, stats: Path, processes: tuple[str, ...] = PROCESSES,
         replay_dirs: tuple[str, ...] = REPLAY_DIRS) -> pd.DataFrame:
    """The sitting runs of the replays, one row per (run, k), gains and costs in opt_z units.

    An entry of ``replay_dirs`` may name the error processes to read from it
    ('end_of_study_ksweep_seqrank=drift+ar1'). Entries with such a filter must not
    overlap: a (run, k) found in two filtered entries is an error, not a choice.
    Without filters a (run, k) found twice keeps its first occurrence.
    """
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file", "family",
            "k", "candidates", "rho", "winner", "regret_noisy", "ref_noisy", "ref_clean", "regret_clean",
            "top_candidate_regret_noisy"]
    frames, filtered = [], False
    for spec in replay_dirs:
        name, only = parse_replay_dir(spec)
        f = pd.read_csv(analysis / name / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
        if only is not None:
            filtered = True
            f = f[f["error_model"].isin(only)]
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["family"] == "tournament") & (d["candidates"] == "lcb") & (d["winner"] == "look")
          & (d["rho"] == 1.0) & d["acquisition"].isin(ACQS) & d["error_model"].isin(processes)]
    if filtered and d.duplicated(subset=["file", "k"]).any():
        raise ValueError(f"{int(d.duplicated(subset=['file', 'k']).sum())} (run, k) rows appear in more than one "
                         "filtered replay directory; the error-process filters must partition the runs")
    d = d.drop_duplicates(subset=["file", "k"], keep="first")
    opt_z = read_opt_z(stats)
    z = d["dataset"].map(opt_z)
    if z.isna().any():
        raise KeyError(f"no opt_z for {sorted(d.loc[z.isna(), 'dataset'].unique())}")
    d = d.assign(gain=(d["ref_noisy"] - d["regret_noisy"]) / z, cost=(d["ref_noisy"] - d["ref_clean"]) / z,
                 k=d["k"].astype(int), opt_z=z)
    d["onset"] = d["jitter_iteration"].astype(int)
    d["sigma"] = d["jitter_std"].astype(float)
    return d


def attach_ship_rules(d: pd.DataFrame, ship: pd.DataFrame) -> pd.DataFrame:
    """Join every sitting run to the zero-trial ship rules of the same run and of its clean twin.

    ``d`` is a frame from ``load`` (it needs file, dataset, acquisition, seed, opt_z,
    regret_noisy, regret_clean, ref_noisy, ref_clean, top_candidate_regret_noisy);
    ``ship`` is ``ship_rules_per_run.csv``. Adds, in opt_z units, the gain of each
    rule over the standard process (gain_pm, gain_lcb1, gain_lcb2), each rule's price
    (price_pm, ...), the sitting's price, and the sitting's increments over LCB1
    (inc_lcb1), over the other two rules (inc_lcb2, inc_pm) and over the top
    candidate of its own prefix (inc_top).
    """
    s = ship.copy()
    s["baseline"] = s["baseline"].astype(str).str.lower().isin(("true", "1"))
    s["fname"] = s["file"].map(lambda f: Path(str(f)).name)
    rule_cols = ["regret_best_observed"] + [f"regret_{r}" for r in SHIP_RULES]
    noisy = s.loc[~s["baseline"], ["fname"] + rule_cols]
    clean = s.loc[s["baseline"], ["dataset", "acquisition", "seed"] + rule_cols].rename(
        columns={c: f"{c}_clean" for c in rule_cols})
    m = d.merge(noisy, left_on="file", right_on="fname", how="left", validate="many_to_one").drop(columns="fname")
    missing = m["regret_best_observed"].isna()
    if missing.any():
        raise KeyError(f"{int(missing.sum())} sitting rows have no ship-rule rescoring, "
                       f"e.g. {m.loc[missing, 'file'].iloc[0]}")
    gap = float((m["ref_noisy"] - m["regret_best_observed"]).abs().max())
    if gap > REF_TOL:
        raise ValueError(f"the rescored best-observed regret differs from the replay's standard process by {gap:g}")
    m = m.merge(clean, on=["dataset", "acquisition", "seed"], how="left", validate="many_to_one")
    missing = m["regret_best_observed_clean"].isna()
    if missing.any():
        raise KeyError(f"{int(missing.sum())} sitting rows have no rescored clean twin, "
                       f"e.g. {m.loc[missing, 'file'].iloc[0]}")
    gap = float((m["ref_clean"] - m["regret_best_observed_clean"]).abs().max())
    if gap > REF_TOL:
        raise ValueError(f"the rescored clean twin differs from the replay's standard clean run by {gap:g}")
    z = m["opt_z"]
    for r in SHIP_RULES:
        m[f"gain_{r}"] = (m["regret_best_observed"] - m[f"regret_{r}"]) / z
        m[f"price_{r}"] = (m[f"regret_{r}_clean"] - m["regret_best_observed_clean"]) / z
    m["inc_lcb1"] = (m["regret_lcb1"] - m["regret_noisy"]) / z
    for r in OTHER_RULES:
        m[f"inc_{r}"] = (m[f"regret_{r}"] - m["regret_noisy"]) / z
    m["inc_top"]= (m["top_candidate_regret_noisy"] - m["regret_noisy"]) / z
    m["price"] = (m["regret_clean"] - m["ref_clean"]) / z
    return m


def per_landscape(sub: pd.DataFrame, value: str = "gain") -> pd.DataFrame:
    """Landscape x k table of mean gains (or costs) over runs."""
    return sub.pivot_table(index="dataset", columns="k", values=value, aggfunc="mean")


def boot_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(v)
    draws = v[rng.integers(0, n, (REPS, n))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm's step-down adjustment (statsmodels); a NaN p is left out of the family and stays NaN."""
    p = np.asarray(p, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        adj[ok] = multipletests(p[ok], method="holm")[1]
    return adj


def wilcoxon_p(v: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p over per-landscape values; NaN when every value is zero."""
    v = np.asarray(v, dtype=float)
    return float(wilcoxon(v).pvalue) if np.any(v != 0) else float("nan")


def ratio_ci(g: np.ndarray, c: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Landscape-bootstrap interval of a ratio of landscape means."""
    idx = rng.integers(0, len(g), (REPS, len(g)))
    draws = g[idx].mean(axis=1) / c[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def share_defined(c: np.ndarray) -> bool:
    """A share of the cost is reported only where the reference cost is positive on every landscape.

    AGENTS.md: never report a recovery ratio whose reference cost is near zero or
    changes sign; the absolute gain is reported instead.
    """
    c = np.asarray(c, dtype=float)
    return bool(len(c) and np.all(c > 0))


def two_way_ci(sub: pd.DataFrame, k: int, seeds: tuple[int, ...], rng: np.random.Generator) -> tuple[float, float]:
    """Resample landscapes and seeds independently; a seed is shared by every landscape."""
    x = sub[(sub["k"] == k) & sub["seed"].isin(seeds)]
    table = x.pivot_table(index="dataset", columns="seed", values="gain", aggfunc="mean")
    arr = table.to_numpy()
    nl, ns = arr.shape
    draws = np.empty(REPS)
    for r in range(REPS):
        li = rng.integers(0, nl, nl)
        si = rng.integers(0, ns, ns)
        draws[r] = np.nanmean(np.nanmean(arr[np.ix_(li, si)], axis=1))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def analyse_cell(sub: pd.DataFrame, label: str, rng: np.random.Generator,
                 processes: tuple[str, ...] = PROCESSES) -> tuple[list[dict], dict]:
    ks = [k for k in K_GRID if k in set(sub["k"])]
    all_l = per_landscape(sub)[ks]
    tr_l = per_landscape(sub[sub["seed"].isin(TRAIN_SEEDS)])[ks]
    te_l = per_landscape(sub[sub["seed"].isin(TEST_SEEDS)])[ks]
    cost = sub[sub["k"] == ks[0]].groupby("dataset")["cost"].mean()
    cost_te = sub[(sub["k"] == ks[0]) & sub["seed"].isin(TEST_SEEDS)].groupby("dataset")["cost"].mean()
    p_te = np.array([wilcoxon(te_l[k].to_numpy()).pvalue for k in ks])
    p_holm = holm(p_te)
    rows = []
    for i, k in enumerate(ks):
        lo, hi = boot_ci(all_l[k].to_numpy(), rng)
        tlo, thi = boot_ci(te_l[k].to_numpy(), rng)
        rows.append({"cell": label, "k": k, "gain_all": all_l[k].mean(), "gain_all_lo": lo, "gain_all_hi": hi,
                     "gain_train": tr_l[k].mean(), "gain_test": te_l[k].mean(), "gain_test_lo": tlo,
                     "gain_test_hi": thi, "test_landscapes_gaining": int((te_l[k] > 0).sum()),
                     "p_test": p_te[i], "p_test_holm_over_k": p_holm[i],
                     "recovered_all": all_l[k].mean() / cost.mean(),
                     "recovered_test": te_l[k].mean() / cost_te.mean()})
    k_fwd = int(tr_l.mean().idxmax())
    k_rev = int(te_l.mean().idxmax())
    # landscape splits: choose on ten landscapes (all seeds), score on the other ten
    names = np.array(all_l.index)
    held, chosen = [], []
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(names))
        a, b = names[perm[:10]], names[perm[10:]]
        for trn, tst in ((a, b), (b, a)):
            kk = int(all_l.loc[trn].mean().idxmax())
            chosen.append(kk)
            held.append(all_l.loc[tst, kk].mean())
    held = np.array(held)
    tw_lo, tw_hi = two_way_ci(sub, k_fwd, TEST_SEEDS, rng)
    fwd = te_l[k_fwd]
    rev = tr_l[k_rev]
    rlo, rhi = boot_ci(rev.to_numpy(), rng)
    flo, fhi = boot_ci(fwd.to_numpy(), rng)
    per_process = {pr: float(sub[(sub["k"] == k_fwd) & sub["seed"].isin(TEST_SEEDS) & (sub["error_model"] == pr)]
                             .groupby("dataset")["gain"].mean().mean()) for pr in processes}
    per_acq = {a: float(sub[(sub["k"] == k_fwd) & sub["seed"].isin(TEST_SEEDS) & (sub["acquisition"] == a)]
                        .groupby("dataset")["gain"].mean().mean()) for a in ACQS}
    summary = {
        "cell": label, "k_chosen_on_7_11": k_fwd, "test_gain": float(fwd.mean()), "test_lo": flo, "test_hi": fhi,
        "test_two_way_lo": tw_lo, "test_two_way_hi": tw_hi,
        "test_landscapes_gaining": int((fwd > 0).sum()),
        "test_p": float(p_te[ks.index(k_fwd)]), "test_p_holm_over_k": float(p_holm[ks.index(k_fwd)]),
        "test_recovered_share": float(fwd.mean() / cost_te.mean()), "cost_of_error_test": float(cost_te.mean()),
        "k_chosen_on_12_16": k_rev, "reverse_test_gain": float(rev.mean()), "reverse_lo": rlo, "reverse_hi": rhi,
        "split_median": float(np.median(held)), "split_p2_5": float(np.percentile(held, 2.5)),
        "split_p97_5": float(np.percentile(held, 97.5)),
        "split_k_share": {int(k): float(np.mean(np.array(chosen) == k)) for k in sorted(set(chosen))},
        "k12_gain_all": float(all_l[12].mean()) if 12 in ks else float("nan"),
        "per_process_test": per_process, "per_acquisition_test": per_acq,
    }
    return rows, summary


def landscape_means(sub: pd.DataFrame, value: str, seeds: tuple[int, ...]) -> pd.Series:
    """Per-landscape mean of one per-run column over the given seeds."""
    return sub[sub["seed"].isin(seeds)].groupby("dataset")[value].mean()


def mean_ci(v: pd.Series | np.ndarray, rng: np.random.Generator) -> dict:
    v = np.asarray(v, dtype=float)
    lo, hi = boot_ci(v, rng)
    return {"mean": float(v.mean()), "lo": lo, "hi": hi}


def rule_reference(sub: pd.DataFrame, label: str, rng: np.random.Generator) -> list[dict]:
    """The zero-trial ship rules on the cell's runs: gain over the standard process, share of the cost, price."""
    runs = sub.drop_duplicates("file")
    rows = []
    for rule in SHIP_RULES:
        for name, seeds in SEED_SETS.items():
            g = landscape_means(runs, f"gain_{rule}", seeds)
            c = landscape_means(runs, "cost", seeds).reindex(g.index)
            price = landscape_means(runs, f"price_{rule}", seeds)
            ci = mean_ci(g, rng)
            # The bootstrap draws either way, so a suppressed share leaves the later draws unchanged.
            slo, shi = ratio_ci(g.to_numpy(), c.to_numpy(), rng)
            share = float(g.mean() / c.mean())
            if not share_defined(c.to_numpy()):
                share, slo, shi = np.nan, np.nan, np.nan
            rows.append({"cell": label, "rule": rule, "seeds": name, "gain": ci["mean"], "gain_lo": ci["lo"],
                         "gain_hi": ci["hi"], "cost": float(c.mean()), "cost_positive_everywhere":
                         share_defined(c.to_numpy()), "share": share,
                         "share_lo": slo, "share_hi": shi, "landscapes_gaining": int((g > 0).sum()),
                         "p": wilcoxon_p(g.to_numpy()), "price": float(price.mean()), "n_runs": int(
                             runs["seed"].isin(seeds).sum()), "n_landscapes": int(len(g))})
    return rows


def increment_cell(sub: pd.DataFrame, label: str, rng: np.random.Generator) -> list[dict]:
    """Per k: the sitting's increment over LCB1 and over its own top candidate, and its price."""
    ks = [k for k in K_GRID if k in set(sub["k"])]
    runs = sub.drop_duplicates("file")
    lcb1 = {n: landscape_means(runs, "gain_lcb1", s) for n, s in SEED_SETS.items()}
    tab = {n: {v: per_landscape(sub[sub["seed"].isin(s)], v)[ks] for v in ("inc_lcb1", "inc_top", "price")}
           for n, s in SEED_SETS.items()}
    p_te = np.array([wilcoxon_p(tab["test"]["inc_lcb1"][k].to_numpy()) for k in ks])
    p_all = np.array([wilcoxon_p(tab["all"]["inc_lcb1"][k].to_numpy()) for k in ks])
    p_holm = holm(p_te)
    rows = []
    for i, k in enumerate(ks):
        inc_all = mean_ci(tab["all"]["inc_lcb1"][k], rng)
        inc_te = mean_ci(tab["test"]["inc_lcb1"][k], rng)
        top_te = mean_ci(tab["test"]["inc_top"][k], rng)
        inc_tr = mean_ci(tab["train"]["inc_lcb1"][k], rng)
        row = {"cell": label, "k": k}
        for rule in SHIP_RULES:
            for n in ("all", "test"):
                row[f"{rule}_gain_{n}"] = float(landscape_means(runs, f"gain_{rule}", SEED_SETS[n]).mean())
        row.update({
            "inc_lcb1_all": inc_all["mean"], "inc_lcb1_all_lo": inc_all["lo"], "inc_lcb1_all_hi": inc_all["hi"],
            "inc_lcb1_all_landscapes_ahead": int((tab["all"]["inc_lcb1"][k] > 0).sum()),
            "inc_lcb1_p_all": p_all[i],
            "inc_lcb1_train": inc_tr["mean"], "inc_lcb1_train_lo": inc_tr["lo"], "inc_lcb1_train_hi": inc_tr["hi"],
            "inc_lcb1_test": inc_te["mean"], "inc_lcb1_test_lo": inc_te["lo"], "inc_lcb1_test_hi": inc_te["hi"],
            "inc_lcb1_test_landscapes_ahead": int((tab["test"]["inc_lcb1"][k] > 0).sum()),
            "inc_lcb1_p_test": p_te[i], "inc_lcb1_p_test_holm_over_k": p_holm[i],
            "inc_top_all": float(tab["all"]["inc_top"][k].mean()),
            "inc_top_test": top_te["mean"], "inc_top_test_lo": top_te["lo"], "inc_top_test_hi": top_te["hi"],
            "price_all": float(tab["all"]["price"][k].mean()), "price_train": float(tab["train"]["price"][k].mean()),
            "price_test": float(tab["test"]["price"][k].mean()),
            "lcb1_price_all": float(landscape_means(runs, "price_lcb1", SEED_SETS["all"]).mean()),
            "lcb1_price_test": float(landscape_means(runs, "price_lcb1", SEED_SETS["test"]).mean()),
        })
        # By error process (means only, no draws): where the increment comes from, and what LCB1 gains there.
        for pr in PROCESSES:
            at = sub[(sub["k"] == k) & (sub["error_model"] == pr)]
            if not len(at):
                continue
            at_runs = runs[runs["error_model"] == pr]
            for n in ("test", "all"):
                row[f"inc_lcb1_{n}_{pr}"] = float(landscape_means(at, "inc_lcb1", SEED_SETS[n]).mean())
                row[f"lcb1_gain_{n}_{pr}"] = float(landscape_means(at_runs, "gain_lcb1", SEED_SETS[n]).mean())
        # A paired difference of landscape means is the difference of the two landscape means.
        gain_all = float(per_landscape(sub, "gain")[k].mean())
        if abs(row["inc_lcb1_all"] - (gain_all - float(lcb1["all"].mean()))) > 1e-12:
            raise AssertionError(f"{label}, k = {k}: the increment is not the gain less LCB1's gain on the same runs")
        rows.append(row)
    return rows


def rule_increment_cell(sub: pd.DataFrame, label: str, rng: np.random.Generator, rule: str) -> list[dict]:
    """Per k: the sitting's increment over one other zero-trial rule (LCB2 or PM), as over LCB1.

    Columns inc_<rule>_{all,train,test} with landscape-bootstrap intervals, landscapes
    ahead, two-sided Wilcoxon p on seeds 12-16 and on all seeds, Holm over the k of
    the cell, and the increment by error process (means only).
    """
    ks = [k for k in K_GRID if k in set(sub["k"])]
    runs = sub.drop_duplicates("file")
    col = f"inc_{rule}"
    tab = {n: per_landscape(sub[sub["seed"].isin(s)], col)[ks] for n, s in SEED_SETS.items()}
    p_te = np.array([wilcoxon_p(tab["test"][k].to_numpy()) for k in ks])
    p_all = np.array([wilcoxon_p(tab["all"][k].to_numpy()) for k in ks])
    p_holm = holm(p_te)
    rule_all = float(landscape_means(runs, f"gain_{rule}", SEED_SETS["all"]).mean())
    rows = []
    for i, k in enumerate(ks):
        row = {"cell": label, "k": k}
        for n in ("all", "train", "test"):
            ci = mean_ci(tab[n][k], rng)
            row[f"{col}_{n}"], row[f"{col}_{n}_lo"], row[f"{col}_{n}_hi"] = ci["mean"], ci["lo"], ci["hi"]
        row[f"{col}_all_landscapes_ahead"] = int((tab["all"][k] > 0).sum())
        row[f"{col}_test_landscapes_ahead"] = int((tab["test"][k] > 0).sum())
        row[f"{col}_p_all"], row[f"{col}_p_test"], row[f"{col}_p_test_holm_over_k"] = p_all[i], p_te[i], p_holm[i]
        for pr in PROCESSES:
            at = sub[(sub["k"] == k) & (sub["error_model"] == pr)]
            if len(at):
                for n in ("test", "all"):
                    row[f"{col}_{n}_{pr}"] = float(landscape_means(at, col, SEED_SETS[n]).mean())
        gain_all = float(per_landscape(sub, "gain")[k].mean())
        if abs(row[f"{col}_all"] - (gain_all - rule_all)) > 1e-12:
            raise AssertionError(f"{label}, k = {k}: the increment is not the gain less {rule.upper()}'s gain")
        rows.append(row)
    return rows


def add_rule_increments(mcells: list[tuple[str, pd.DataFrame]], frame: pd.DataFrame,
                        summaries: list[dict]) -> pd.DataFrame:
    """The sitting's increments over LCB2 and PM, merged into the frame and the cell summaries.

    Draws from its own generator (OTHER_RULES_BOOTSTRAP_SEED), after every other
    quantity, so every earlier column and summary value is unchanged. Holm families
    as over LCB1: the nine k of a cell, the eight (cell, k chosen on seeds 7-11)
    and all 72 (cell, k). Each cell summary also names the zero-trial rule with the
    largest gain on seeds 7-11 (increment_over_chosen_rule), with Holm over the
    eight cells for that increment.
    """
    rng4 = np.random.default_rng(OTHER_RULES_BOOTSTRAP_SEED)
    chosen = {s["cell"]: s["k_chosen_on_7_11"] for s in summaries if "k_chosen_on_7_11" in s}
    for rule in OTHER_RULES:
        rows = []
        for label, sub in mcells:
            rows += rule_increment_cell(sub, label, rng4, rule)
        inc = pd.DataFrame(rows)
        col = f"inc_{rule}"
        inc[f"{col}_p_test_holm_all_cells"] = np.nan
        inc[f"{col}_p_test_holm_chosen_cells"] = np.nan
        in_cells = inc["cell"] != "pooled"
        inc.loc[in_cells, f"{col}_p_test_holm_all_cells"] = holm(inc.loc[in_cells, f"{col}_p_test"].to_numpy())
        is_chosen = inc.apply(lambda r: r["cell"] != "pooled" and chosen.get(r["cell"]) == r["k"], axis=1)
        inc.loc[is_chosen, f"{col}_p_test_holm_chosen_cells"] = holm(inc.loc[is_chosen, f"{col}_p_test"].to_numpy())
        frame = frame.merge(inc, on=["cell", "k"], how="left", validate="one_to_one")
    picked = []
    for s in summaries:
        if "k_chosen_on_7_11" not in s:
            continue
        k = s["k_chosen_on_7_11"]
        r = frame[(frame["cell"] == s["cell"]) & (frame["k"] == k)].iloc[0]
        for rule in OTHER_RULES:
            col = f"inc_{rule}"
            s[f"increment_over_{rule}"] = {
                "k": int(k), **{f"{n}{sfx}": float(r[f"{col}_{n}{sfx}"]) for n in ("test", "train", "all")
                                for sfx in ("", "_lo", "_hi")},
                "test_landscapes_ahead": int(r[f"{col}_test_landscapes_ahead"]),
                "all_landscapes_ahead": int(r[f"{col}_all_landscapes_ahead"]),
                "test_p": float(r[f"{col}_p_test"]), "all_p": float(r[f"{col}_p_all"]),
                "test_p_holm_over_k": float(r[f"{col}_p_test_holm_over_k"]),
                "test_p_holm_chosen_cells": float(r[f"{col}_p_test_holm_chosen_cells"]),
                "test_p_holm_all_cells": float(r[f"{col}_p_test_holm_all_cells"])}
            s[f"per_process_increment_over_{rule}_test"] = {
                pr: float(r[f"{col}_test_{pr}"]) for pr in PROCESSES
                if f"{col}_test_{pr}" in r.index and pd.notna(r[f"{col}_test_{pr}"])}
        # The rule a study would pick on the seeds that choose k: the largest gain on seeds 7-11.
        train = {rule: s["ship_rules"][rule]["train"]["gain"] for rule in SHIP_RULES}
        best = max(SHIP_RULES, key=lambda rule: train[rule])
        over = s["increment_over_lcb1"] if best == "lcb1" else s[f"increment_over_{best}"]
        s["zero_trial_rule_chosen_on_7_11"] = best
        s["increment_over_chosen_rule"] = {"rule": best, "rule_gain_train": float(train[best]),
                                           **{c: over[c] for c in ("k", "test", "test_lo", "test_hi",
                                                                   "test_landscapes_ahead", "test_p")}}
        if s["cell"] != "pooled":
            picked.append(s)
    adj = holm(np.array([s["increment_over_chosen_rule"]["test_p"] for s in picked]))
    for s, a in zip(picked, adj):
        s["increment_over_chosen_rule"]["test_p_holm_chosen_cells"] = float(a)
    return frame


def price_table(d: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Clean-twin prices by k and by ship rule; the clean twin is shared by every cell and process."""
    stems = d.drop_duplicates(["dataset", "acquisition", "seed", "k"])
    out: dict = {"sitting": {}, "ship_rules": {}}
    for k in sorted(stems["k"].unique()):
        x = stems[stems["k"] == k]
        out["sitting"][int(k)] = {n: mean_ci(landscape_means(x, "price", s), rng) for n, s in SEED_SETS.items()}
    runs = stems.drop_duplicates(["dataset", "acquisition", "seed"])
    for rule in SHIP_RULES:
        out["ship_rules"][rule] = {n: mean_ci(landscape_means(runs, f"price_{rule}", s), rng)
                                   for n, s in SEED_SETS.items()}
    return out


def _cell_label(cell: str) -> str:
    if cell == "pooled":
        return "pooled"
    sig, rest = cell.split("sigma_from_trial_")
    return f"${sig}\\sigma$, from trial {rest}"


def _p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.2f}"


def _signed(v: float) -> str:
    """A signed value to three decimals; one that rounds to zero prints as 0.000, never -0.000 or +0.000."""
    s = f"{v:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s


def share_ci(sub: pd.DataFrame, k: int, seeds: tuple[int, ...], rng: np.random.Generator) -> tuple[float, float]:
    """Held-out gain as a share of the standard process's cost of error, a ratio of landscape means."""
    x = sub[(sub["k"] == k) & sub["seed"].isin(seeds)]
    g = x.groupby("dataset")["gain"].mean().to_numpy()
    c = x.groupby("dataset")["cost"].mean().to_numpy()
    idx = rng.integers(0, len(g), (REPS, len(g)))
    draws = g[idx].mean(axis=1) / c[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_table(summaries: list[dict], path: Path) -> None:
    """The per-cell selection as the LaTeX table the appendix inputs.

    Beside the held-out gain over the standard process, the sitting's held-out
    increment over LCB1 after all 50 trials (no extra trial) on the same runs, and
    the chosen k's clean-twin price on the held-out seeds. Nine columns fit the
    TMLR text width (469.8pt) only at a column separation of 3pt (469.6pt), so the
    table sets it itself; it sits inside the table float, which scopes it. (A
    column spec with @{} would save the outer padding, but check_paper.py reads
    the spec up to the first closing brace.)
    """
    def gain(v, lo, hi):
        return f"${_signed(v)}$ {{\\scriptsize $[{_signed(lo)}, {_signed(hi)}]$}}"
    lines = [r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrlrrlrrr}", r"\toprule",
             r" & $k$ & gain & & Holm & over LCB1 & price & $k$ & $k = 12$ \\",
             r"cell & 7--11 & seeds 12--16 & gaining & $p$ & seeds 12--16 & 12--16 & 12--16 & 7--16 \\",
             r"\midrule"]
    for s in summaries:
        if "k_chosen_on_7_11" not in s:
            continue
        inc = s["increment_over_lcb1"]
        lines.append(f"{_cell_label(s['cell'])} & {s['k_chosen_on_7_11']} & "
                     f"{gain(s['test_gain'], s['test_lo'], s['test_hi'])} & {s['test_landscapes_gaining']} & "
                     f"{_p(s['test_p_holm_over_k'])} & {gain(inc['test'], inc['test_lo'], inc['test_hi'])} & "
                     f"${s['price_test']:.3f}$ & {s['k_chosen_on_12_16']} & ${_signed(s['k12_gain_all'])}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def add_increments(d: pd.DataFrame, ship: pd.DataFrame, cells: list[tuple[str, pd.DataFrame]],
                   frame: pd.DataFrame, summaries: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The zero-trial references, increments and prices, merged into the existing outputs.

    Draws from its own generator, so it leaves every existing number unchanged.
    """
    rng3 = np.random.default_rng(INCREMENT_BOOTSTRAP_SEED)
    m = attach_ship_rules(d, ship)
    mcells = [("pooled", m)] + [(f"{sig:g}sigma_from_trial_{ons + 1}", sub)
                                for (sig, ons), sub in sorted(m.groupby(["sigma", "onset"]))]
    if [c for c, _ in mcells] != [c for c, _ in cells]:
        raise AssertionError("the cells of the joined frame differ from the cells analysed")
    inc_rows, rule_rows = [], []
    for label, sub in mcells:
        inc_rows += increment_cell(sub, label, rng3)
        rule_rows += rule_reference(sub, label, rng3)
    inc = pd.DataFrame(inc_rows)
    inc["inc_lcb1_p_test_holm_all_cells"] = np.nan
    inc["inc_lcb1_p_test_holm_chosen_cells"] = np.nan
    in_cells = inc["cell"] != "pooled"
    inc.loc[in_cells, "inc_lcb1_p_test_holm_all_cells"] = holm(inc.loc[in_cells, "inc_lcb1_p_test"].to_numpy())
    chosen = {s["cell"]: s["k_chosen_on_7_11"] for s in summaries if "k_chosen_on_7_11" in s}
    is_chosen = inc.apply(lambda r: r["cell"] != "pooled" and chosen.get(r["cell"]) == r["k"], axis=1)
    inc.loc[is_chosen, "inc_lcb1_p_test_holm_chosen_cells"] = holm(inc.loc[is_chosen, "inc_lcb1_p_test"].to_numpy())
    frame = frame.merge(inc, on=["cell", "k"], how="left", validate="one_to_one")
    rules = pd.DataFrame(rule_rows)
    mframes = dict(mcells)
    for s in summaries:
        if "k_chosen_on_7_11" not in s:
            continue
        k = s["k_chosen_on_7_11"]
        r = frame[(frame["cell"] == s["cell"]) & (frame["k"] == k)].iloc[0]
        # By gain on seeds 7-11 is by increment over LCB1 on seeds 7-11: LCB1 does not depend on k.
        tr = frame[frame["cell"] == s["cell"]].set_index("k")["inc_lcb1_train"]
        if int(tr.idxmax()) != k:
            raise AssertionError(f"{s['cell']}: the k chosen by gain is not the k chosen by increment")
        s["increment_over_lcb1"] = {
            "k": int(k), "test": float(r["inc_lcb1_test"]), "test_lo": float(r["inc_lcb1_test_lo"]),
            "test_hi": float(r["inc_lcb1_test_hi"]),
            "test_landscapes_ahead": int(r["inc_lcb1_test_landscapes_ahead"]),
            "test_p": float(r["inc_lcb1_p_test"]), "test_p_holm_over_k": float(r["inc_lcb1_p_test_holm_over_k"]),
            "test_p_holm_chosen_cells": float(r["inc_lcb1_p_test_holm_chosen_cells"]),
            "test_p_holm_all_cells": float(r["inc_lcb1_p_test_holm_all_cells"]),
            "train": float(r["inc_lcb1_train"]), "train_lo": float(r["inc_lcb1_train_lo"]),
            "train_hi": float(r["inc_lcb1_train_hi"]), "all": float(r["inc_lcb1_all"]),
            "all_lo": float(r["inc_lcb1_all_lo"]), "all_hi": float(r["inc_lcb1_all_hi"]),
            "all_landscapes_ahead": int(r["inc_lcb1_all_landscapes_ahead"]), "all_p": float(r["inc_lcb1_p_all"])}
        s["increment_over_top_candidate"] = {"test": float(r["inc_top_test"]), "test_lo": float(r["inc_top_test_lo"]),
                                             "test_hi": float(r["inc_top_test_hi"]), "all": float(r["inc_top_all"])}
        x = mframes[s["cell"]]
        x = x[(x["k"] == k) & x["seed"].isin(TEST_SEEDS)]
        s["per_process_increment_over_lcb1_test"] = {
            pr: float(x[x["error_model"] == pr].groupby("dataset")["inc_lcb1"].mean().mean())
            for pr in PROCESSES if pr in set(x["error_model"])}
        s["per_process_lcb1_gain_test"] = {
            pr: float(x[x["error_model"] == pr].groupby("dataset")["gain_lcb1"].mean().mean())
            for pr in PROCESSES if pr in set(x["error_model"])}
        s["price_test"], s["price_all"] = float(r["price_test"]), float(r["price_all"])
        s["lcb1_price_test"], s["lcb1_price_all"] = float(r["lcb1_price_test"]), float(r["lcb1_price_all"])
        cell_rules = rules[rules["cell"] == s["cell"]]
        s["ship_rules"] = {rule: {row["seeds"]: {**{c: float(row[c]) for c in ("gain", "gain_lo", "gain_hi", "share",
                                                                               "share_lo", "share_hi", "p", "price")},
                                                 "landscapes_gaining": int(row["landscapes_gaining"])}
                                  for _, row in cell_rules[cell_rules["rule"] == rule].iterrows()}
                           for rule in SHIP_RULES}
    # k = 12 pooled without the 5 sigma, trial-21 cell, against LCB1 on the same runs.
    rest = m[(m["k"] == 12) & ~((m["sigma"] == 5.0) & (m["onset"] == 20))]
    without = next(s for s in summaries if s["cell"] == "pooled_k12_without_5sigma_from_trial_21")
    for name, seeds in (("all_seeds", SEED_SETS["all"]), ("seeds_12_16", TEST_SEEDS)):
        v = landscape_means(rest, "inc_lcb1", seeds)
        if not len(v):  # a replay set without k = 12
            continue
        ci = mean_ci(v, rng3)
        without[name]["lcb1_gain"] = float(landscape_means(rest.drop_duplicates("file"), "gain_lcb1", seeds).mean())
        without[name]["increment_over_lcb1"] = {"gain": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                                                "landscapes_ahead": int((v > 0).sum()), "p": wilcoxon_p(v.to_numpy())}
    prices = price_table(m, rng3)
    # The increments over LCB2 and PM (a fourth generator; nothing above changes).
    frame = add_rule_increments(mcells, frame, summaries)
    return frame, rules, prices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, default=Path("output-boba/analysis"))
    ap.add_argument("--stats", type=Path, default=Path("boba_landscape_stats.json"))
    ap.add_argument("--ship-rules", type=Path, default=None,
                    help="ship_rules_per_run.csv (default: <analysis>/ship_rules_per_run.csv)")
    ap.add_argument("--out", type=Path, default=Path("output-boba/analysis/review"))
    ap.add_argument("--processes", default=",".join(PROCESSES),
                    help="comma-separated error processes (default: the four the paper's table pools)")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. 'gaussian'")
    ap.add_argument("--replay-dirs", default=",".join(REPLAY_DIRS),
                    help="comma-separated replay directories under --analysis (default: the k-sweep's two); "
                         "'dir=drift+ar1' reads only those error processes from dir; a different set needs --tag")
    ap.add_argument("--table", type=Path, default=None,
                    help="LaTeX table to write (default: paper/tables/sitting_by_magnitude.tex, and only for the "
                         "four processes; 'none' writes no table)")
    args = ap.parse_args()
    processes = tuple(p.strip() for p in args.processes.split(",") if p.strip())
    unknown = sorted(set(processes) - set(PROCESSES))
    if unknown:
        raise SystemExit(f"unknown error processes {unknown}; choose from {PROCESSES}")
    table = args.table
    if table is None:
        table = Path("paper/tables/sitting_by_magnitude.tex") if processes == PROCESSES else None
    elif str(table).lower() == "none":
        table = None
    replay_dirs = tuple(x.strip() for x in args.replay_dirs.split(",") if x.strip())
    suffix = f"_{args.tag}" if args.tag else ""
    if (processes != PROCESSES or replay_dirs != REPLAY_DIRS) and not suffix:
        raise SystemExit("a subset of the processes or other replays need --tag, so that they do not overwrite "
                         "the paper's outputs")
    if replay_dirs != REPLAY_DIRS and args.table is None:
        table = None
    d = load(args.analysis, args.stats, processes, replay_dirs)
    ship = pd.read_csv(args.ship_rules or args.analysis / "ship_rules_per_run.csv", low_memory=False)
    frame, rules, summaries = run(d, ship, processes)
    counts = d.groupby(["error_model", "k"]).size().unstack("k")
    summaries.append({"cell": "provenance", "replay_dirs": list(replay_dirs), "processes": list(processes),
                      "runs_by_process_and_k": {p: {int(k): int(n) for k, n in row.items() if pd.notna(n)}
                                                for p, row in counts.iterrows()}})
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / f"sitting_by_magnitude{suffix}.csv", index=False)
    rules.to_csv(args.out / f"sitting_by_magnitude{suffix}_ship_rules.csv", index=False)
    (args.out / f"sitting_by_magnitude{suffix}_selection.json").write_text(json.dumps(summaries, indent=2))
    if table is not None:
        write_table(summaries, table)
    show = pd.DataFrame([{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries
                         if "k_chosen_on_7_11" in s])
    print(f"error processes: {', '.join(processes)}; replays: {', '.join(replay_dirs)}")
    print("sitting runs by error process and k:")
    print(counts.to_string())
    print(show.round(4).to_string(index=False))
    without = next(s for s in summaries if s["cell"] == "pooled_k12_without_5sigma_from_trial_21")
    print("pooled k = 12 without the 5 sigma, trial-21 cell:", without)
    prices = next(s for s in summaries if s["cell"] == "clean_twin_prices")
    for s in summaries:
        if "k_chosen_on_7_11" not in s:
            continue
        inc = s["increment_over_lcb1"]
        print(s["cell"], "per process", {k: round(v, 4) for k, v in s["per_process_test"].items()},
              "per acquisition", {k: round(v, 4) for k, v in s["per_acquisition_test"].items()},
              "split k share", s["split_k_share"])
        print(f"   k = {s['k_chosen_on_7_11']}: over LCB1 on seeds 12-16 {inc['test']:+.4f} "
              f"[{inc['test_lo']:+.4f}, {inc['test_hi']:+.4f}], ahead on {inc['test_landscapes_ahead']}/20, "
              f"p {inc['test_p']:.3g} (Holm over k {inc['test_p_holm_over_k']:.3g}, over the chosen cells "
              f"{inc['test_p_holm_chosen_cells']:.3g}); LCB1 gain {s['ship_rules']['lcb1']['test']['gain']:+.4f}; "
              f"price {s['price_test']:.4f} (LCB1 {s['lcb1_price_test']:.4f})")
        for rule in OTHER_RULES:
            o = s[f"increment_over_{rule}"]
            print(f"   over {rule.upper()} on seeds 12-16 {o['test']:+.4f} [{o['test_lo']:+.4f}, {o['test_hi']:+.4f}], "
                  f"ahead on {o['test_landscapes_ahead']}/20, p {o['test_p']:.3g} (Holm over k "
                  f"{o['test_p_holm_over_k']:.3g}); {rule.upper()} gain {s['ship_rules'][rule]['test']['gain']:+.4f}")
        c = s["increment_over_chosen_rule"]
        print(f"   rule chosen on seeds 7-11: {c['rule'].upper()}; over it on seeds 12-16 {c['test']:+.4f} "
              f"[{c['test_lo']:+.4f}, {c['test_hi']:+.4f}], p {c['test_p']:.3g}"
              + (f" (Holm over the 8 cells {c['test_p_holm_chosen_cells']:.3g})" if "test_p_holm_chosen_cells" in c
                 else ""))
    print("clean-twin price by k, seeds 7-16:",
          {k: round(v["all"]["mean"], 4) for k, v in prices["sitting"].items()},
          "ship rules:", {r: round(v["all"]["mean"], 4) for r, v in prices["ship_rules"].items()})


def run(d: pd.DataFrame, ship: pd.DataFrame, processes: tuple[str, ...] = PROCESSES
        ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Every output of the script from the loaded replays ``d`` and the ship-rule rescoring ``ship``.

    Returns the per-(cell, k) frame, the per-(cell, rule, seed set) ship-rule frame
    and the list of summaries written to the selection JSON.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cells = [("pooled", d)]
    for (sig, ons), sub in sorted(d.groupby(["sigma", "onset"])):
        cells.append((f"{sig:g}sigma_from_trial_{ons + 1}", sub))
    rows, summaries = [], []
    for label, sub in cells:
        r, s = analyse_cell(sub, label, rng, processes)
        rows += r
        summaries.append(s)
    # Extra quantities the text quotes, on a separate generator so the cell results keep their draws.
    rng2 = np.random.default_rng(BOOTSTRAP_SEED + 1)
    frame = pd.DataFrame(rows)
    cells_only = frame[frame["cell"] != "pooled"].copy()
    cells_only["p_test_holm_all_cells"] = holm(cells_only["p_test"].to_numpy())
    frame = frame.merge(cells_only[["cell", "k", "p_test_holm_all_cells"]], on=["cell", "k"], how="left")
    by_label = dict(cells)
    for s in summaries:
        lo, hi = share_ci(by_label[s["cell"]], s["k_chosen_on_7_11"], TEST_SEEDS, rng2)
        s["test_recovered_lo"], s["test_recovered_hi"] = lo, hi
        s["test_p_holm_all_cells"] = float(frame.loc[(frame["cell"] == s["cell"]) & (frame["k"] == s["k_chosen_on_7_11"]),
                                                     "p_test_holm_all_cells"].iloc[0]) if s["cell"] != "pooled" else float("nan")
    rest = d[(d["k"] == 12) & ~((d["sigma"] == 5.0) & (d["onset"] == 20))]
    without = {}
    for name, seeds in (("all_seeds", TRAIN_SEEDS + TEST_SEEDS), ("seeds_12_16", TEST_SEEDS)):
        v = rest[rest["seed"].isin(seeds)].groupby("dataset")["gain"].mean().to_numpy()
        if not len(v):  # a replay set without k = 12
            without[name] = {"gain": float("nan"), "lo": float("nan"), "hi": float("nan")}
            continue
        lo, hi = boot_ci(v, rng2)
        without[name] = {"gain": float(v.mean()), "lo": lo, "hi": hi}
    summaries.append({"cell": "pooled_k12_without_5sigma_from_trial_21", **without})
    # The zero-trial references, the increments over LCB1 and the prices (their own generator).
    frame, rules, prices = add_increments(d, ship, cells, frame, summaries)
    summaries.append({"cell": "clean_twin_prices", "processes": list(processes), **prices})
    return frame, rules, summaries


if __name__ == "__main__":
    main()
