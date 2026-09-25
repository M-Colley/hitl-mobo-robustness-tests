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
numpy and scipy only; no simulation is run.

    python scripts/sitting_by_magnitude.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TRAIN_SEEDS = (7, 8, 9, 10, 11)
TEST_SEEDS = (12, 13, 14, 15, 16)
K_GRID = (2, 3, 5, 8, 12, 16, 20, 25, 30)
ACQS = ("logei", "qnei")
PROCESSES = ("gaussian", "bias", "drift", "ar1")
BOOTSTRAP_SEED = 20260925
REPS = 2000
N_SPLITS = 1000


def load(analysis: Path, stats: Path) -> pd.DataFrame:
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file", "family",
            "k", "candidates", "rho", "winner", "regret_noisy", "ref_noisy", "ref_clean"]
    frames = [pd.read_csv(analysis / d / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
              for d in ("end_of_study_ksweep", "end_of_study_kwide")]
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["family"] == "tournament") & (d["candidates"] == "lcb") & (d["winner"] == "look")
          & (d["rho"] == 1.0) & d["acquisition"].isin(ACQS) & d["error_model"].isin(PROCESSES)]
    d = d.drop_duplicates(subset=["file", "k"], keep="first")
    opt_z = {k: v["opt_z"] for k, v in json.loads(stats.read_text())["functions"].items()}
    z = d["dataset"].map(opt_z)
    if z.isna().any():
        raise KeyError(f"no opt_z for {sorted(d.loc[z.isna(), 'dataset'].unique())}")
    d = d.assign(gain=(d["ref_noisy"] - d["regret_noisy"]) / z, cost=(d["ref_noisy"] - d["ref_clean"]) / z,
                 k=d["k"].astype(int))
    d["onset"] = d["jitter_iteration"].astype(int)
    d["sigma"] = d["jitter_std"].astype(float)
    return d


def per_landscape(sub: pd.DataFrame, value: str = "gain") -> pd.DataFrame:
    """Landscape x k table of mean gains (or costs) over runs."""
    return sub.pivot_table(index="dataset", columns="k", values=value, aggfunc="mean")


def boot_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(v)
    draws = v[rng.integers(0, n, (REPS, n))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


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


def analyse_cell(sub: pd.DataFrame, label: str, rng: np.random.Generator) -> tuple[list[dict], dict]:
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
                             .groupby("dataset")["gain"].mean().mean()) for pr in PROCESSES}
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


def _cell_label(cell: str) -> str:
    if cell == "pooled":
        return "pooled"
    sig, rest = cell.split("sigma_from_trial_")
    return f"${sig}\\sigma$, from trial {rest}"


def _p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.2f}"


def share_ci(sub: pd.DataFrame, k: int, seeds: tuple[int, ...], rng: np.random.Generator) -> tuple[float, float]:
    """Held-out gain as a share of the standard process's cost of error, a ratio of landscape means."""
    x = sub[(sub["k"] == k) & sub["seed"].isin(seeds)]
    g = x.groupby("dataset")["gain"].mean().to_numpy()
    c = x.groupby("dataset")["cost"].mean().to_numpy()
    idx = rng.integers(0, len(g), (REPS, len(g)))
    draws = g[idx].mean(axis=1) / c[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_table(summaries: list[dict], path: Path) -> None:
    """The per-cell selection as the LaTeX table the appendix inputs."""
    def gain(v, lo, hi):
        return f"${v:+.3f}$ {{\\scriptsize $[{lo:+.3f}, {hi:+.3f}]$}}"
    lines = [r"\begin{tabular}{lrlrrrr}", r"\toprule",
             r"cell & $k$, 7--11 & gain on seeds 12--16 & landscapes & Holm $p$ & $k$, 12--16 & $k = 12$ \\",
             r"\midrule"]
    for s in summaries:
        if "k_chosen_on_7_11" not in s:
            continue
        lines.append(f"{_cell_label(s['cell'])} & {s['k_chosen_on_7_11']} & "
                     f"{gain(s['test_gain'], s['test_lo'], s['test_hi'])} & {s['test_landscapes_gaining']} & "
                     f"{_p(s['test_p_holm_over_k'])} & {s['k_chosen_on_12_16']} & ${s['k12_gain_all']:+.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, default=Path("output-boba/analysis"))
    ap.add_argument("--stats", type=Path, default=Path("boba_landscape_stats.json"))
    ap.add_argument("--out", type=Path, default=Path("output-boba/analysis/review"))
    ap.add_argument("--table", type=Path, default=Path("paper/tables/sitting_by_magnitude.tex"))
    args = ap.parse_args()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    d = load(args.analysis, args.stats)
    cells = [("pooled", d)]
    for (sig, ons), sub in sorted(d.groupby(["sigma", "onset"])):
        cells.append((f"{sig:g}sigma_from_trial_{ons + 1}", sub))
    rows, summaries = [], []
    for label, sub in cells:
        r, s = analyse_cell(sub, label, rng)
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
        lo, hi = boot_ci(v, rng2)
        without[name] = {"gain": float(v.mean()), "lo": lo, "hi": hi}
    summaries.append({"cell": "pooled_k12_without_5sigma_from_trial_21", **without})
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "sitting_by_magnitude.csv", index=False)
    (args.out / "sitting_by_magnitude_selection.json").write_text(json.dumps(summaries, indent=2))
    write_table(summaries, args.table)
    show = pd.DataFrame([{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries
                         if "k_chosen_on_7_11" in s])
    print(show.round(4).to_string(index=False))
    print("pooled k = 12 without the 5 sigma, trial-21 cell:", summaries[-1])
    for s in summaries[:-1]:
        print(s["cell"], "per process", {k: round(v, 4) for k, v in s["per_process_test"].items()},
              "per acquisition", {k: round(v, 4) for k, v in s["per_acquisition_test"].items()},
              "split k share", s["split_k_share"])


if __name__ == "__main__":
    main()
