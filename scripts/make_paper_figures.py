"""The paper's figures, drawn from the analysis CSVs and nothing else.

Every figure here is a picture of a table the paper already prints, so the
number a reader takes from the figure can be checked against the table, and
the command that produced it is this file. Four figures:

  dose_response.pdf   search loss and deployed loss against the error magnitude,
                      one panel per onset, landscape-bootstrap bands
  decomposition.pdf   the deployed excess split into search and selection loss,
                      per magnitude and onset, the selection share on each bar
  kcurve.pdf          gain over the standard process from a final comparative
                      sitting of k trials, with intervals, against the oracle
  frag_scatter.pdf    the one-shot selection loss of a landscape against the
                      cost the sweep measured on it, one point per landscape
                      and magnitude

Colour is assigned by the job it does, in the fixed order of the reference
palette: two categorical hues for the two responses, one sequential hue for the
ordered magnitudes. Both were run through the palette validator before use.

    python scripts/make_paper_figures.py
    python scripts/make_paper_figures.py --analysis output-boba/analysis --out paper/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import boba_benchmarks as bb  # noqa: E402

MODEL_FREE = {"random", "sobol"}
BOOT = 2000
SEED = 20260918

# Reference palette (validated): categorical slots 1 and 2, sequential blue.
SEARCH, DEPLOYED = "#2a78d6", "#eb6834"
SEQ_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # 0.05, 0.25, 1, 5 sigma
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
WIDTH = 5.5   # inches; the ICLR text width is 5.5in

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42,
})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", type=Path, default=Path("output-boba/analysis"))
    p.add_argument("--out", type=Path, default=Path("paper/figures"))
    p.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    p.add_argument("--policies", type=Path, default=None,
                   help="budget_split_policies.csv for the k-curve; the default is the one "
                        "in --analysis. Point it at the rho=1 run to draw the sitting in "
                        "which only the shared error cancels.")
    return p.parse_args(argv)


def boot_mean(per_landscape: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean over landscapes with a landscape bootstrap; the paper's aggregate."""
    n = len(per_landscape)
    draws = per_landscape[rng.integers(0, n, (BOOT, n))].mean(axis=1)
    return float(per_landscape.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _save(fig, out: Path, name: str) -> None:
    """PDF for the paper, PNG beside it so the figure can be looked at."""
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _style(ax, ylabel: str, xlabel: str | None = None) -> None:
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


# ---------------------------------------------------------------------------


def fig_dose_response(cells: pd.DataFrame, out: Path) -> None:
    rng = np.random.default_rng(SEED)
    stds = sorted(cells["jitter_std"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), sharey=True)
    for ax, onset, title in zip(axes, (0, 20), ("error from trial 1", "error from trial 21")):
        for value, colour, label in (("fragility", SEARCH, "search loss, trajectory average"),
                                     ("deployed", DEPLOYED, "deployed design, final trial")):
            mean, lo, hi = [], [], []
            for s in stds:
                per = (cells[(cells.jitter_std == s) & (cells.jitter_iteration == onset)]
                       .groupby("dataset")[value].mean().to_numpy())
                m, l, h = boot_mean(per, rng)
                mean.append(m); lo.append(l); hi.append(h)
            ax.fill_between(stds, lo, hi, color=colour, alpha=0.18, linewidth=0)
            ax.plot(stds, mean, color=colour, linewidth=1.4, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, label=label)
            # Direct label at the right end; the legend carries the full name.
            ax.annotate(f"{mean[-1] * 100:.0f}%", (stds[-1], mean[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks(stds)
        ax.set_xticklabels([f"{s:g}" for s in stds])
        ax.set_title(title, loc="left", fontweight="normal")
        _style(ax, "fraction of achievable\nimprovement lost" if onset == 0 else "",
               r"error magnitude $\sigma_e$ (landscape SDs)")
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.5)
    _save(fig, out, "dose_response")


def fig_decomposition(decomp: pd.DataFrame, out: Path) -> None:
    d = decomp[(decomp.error_model == "pooled") & (decomp.jitter_std != "pooled")].copy()
    d["std"] = d.jitter_std.astype(float)
    d["onset"] = d.jitter_iteration.astype(float)
    d = d.sort_values(["onset", "std"])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    width, gap = 0.7, 0.9
    xs, labels = [], []
    for g, (onset, block) in enumerate(d.groupby("onset", sort=True)):
        for i, (_, r) in enumerate(block.iterrows()):
            x = g * (len(block) + gap) + i
            xs.append(x); labels.append(f"{r['std']:g}")
            # The two segments must sum to the drawn total, so the gap between them is a
            # surface-coloured edge, never an offset added to the bottom: an offset once made
            # the 0.05 sigma bars half again too tall.
            ax.bar(x, r.mean_excess_search, width, color=SEARCH,
                   edgecolor="white", linewidth=0.9)
            ax.bar(x, r.mean_excess_selection, width, bottom=r.mean_excess_search,
                   color=DEPLOYED, edgecolor="white", linewidth=0.9)
            top = r.mean_excess_search + r.mean_excess_selection
            ax.annotate(f"{r.excess_selection_share * 100:.0f}%", (x, top), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
        mid = g * (len(block) + gap) + (len(block) - 1) / 2
        ax.annotate("error from trial 1" if onset == 0 else "error from trial 21", (mid, 0),
                    xytext=(0, -24), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    _style(ax, "deployed excess over\nthe clean twin", r"$\sigma_e$ (landscape SDs)")
    ax.xaxis.labelpad = 14
    ax.bar([np.nan], [np.nan], color=SEARCH, label="search loss")
    ax.bar([np.nan], [np.nan], color=DEPLOYED, label="selection loss (share on bar)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, out, "decomposition")


def fig_kcurve(policies: pd.DataFrame, out: Path) -> None:
    fixed = policies[policies.policy.str.startswith("fixed_k")].copy()
    fixed["k"] = fixed.policy.str.replace("fixed_k", "").astype(float)
    fixed = fixed.sort_values("k")
    oracle = policies[policies.policy == "oracle"].iloc[0]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axhline(oracle.gain_vs_standard, color=INK2, linewidth=0.8, linestyle="--")
    ax.annotate("oracle k per run", (fixed.k.max(), oracle.gain_vs_standard), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.annotate("standard process", (fixed.k.min(), 0), xytext=(0, -3),
                textcoords="offset points", ha="left", va="top", fontsize=7.5, color=INK2)
    ax.errorbar(fixed.k, fixed.gain_vs_standard,
                yerr=[fixed.gain_vs_standard - fixed.gain_lo, fixed.gain_hi - fixed.gain_vs_standard],
                color=SEARCH, linewidth=1.4, elinewidth=0.8, capsize=2.5, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.8, label="fixed k, all runs")
    # The argmax is not resolved: several k share overlapping intervals, so the
    # label names the flat region rather than a winner.
    best = fixed.loc[fixed.gain_vs_standard.idxmax()]
    flat = fixed[fixed.gain_hi >= best.gain_vs_standard]
    ax.annotate(f"flat from k = {flat.k.min():g} to {flat.k.max():g},"
                f" best +{best.gain_vs_standard:.3f}",
                (best.k, best.gain_vs_standard), xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(fixed.k)
    ax.set_xticklabels([f"{k:g}" for k in fixed.k])
    _style(ax, "gain over the standard process\n(fraction of achievable improvement)",
           "trials k spent on the final sitting, of T = 50")
    fig.tight_layout()
    _save(fig, out, "kcurve")


def fig_frag_scatter(cells: pd.DataFrame, stats: dict, out: Path) -> float:
    """One point per (landscape, magnitude), error from the first rating."""
    rows = []
    early = cells[cells.jitter_iteration == 0]
    for (dataset, s), block in early.groupby(["dataset", "jitter_std"]):
        key = f"frag_{s:g}"
        if dataset not in stats or key not in stats[dataset]:
            continue
        rows.append({"dataset": dataset, "std": float(s), "frag": float(stats[dataset][key]),
                     "cost": float(block.fragility.mean())})
    f = pd.DataFrame(rows)
    # Log axes cannot place a zero one-shot loss or a negative measured cost, and
    # those cells are the cheap end, so dropping them from the CORRELATION too
    # would flatter it (0.87 against 0.84). The rank correlation is therefore
    # taken over every cell and only the drawing is restricted.
    drawable = f[(f.frag > 0) & (f.cost > 0)]
    stds = sorted(f["std"].unique())
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    # The four magnitudes are ordered, so one hue light->dark; adjacent steps of
    # a ramp sit under the normal-vision separation floor, so the marker shape
    # carries identity as well (the validator's "secondary encoding" clause).
    for colour, marker, s in zip(SEQ_BLUE, ("o", "s", "^", "D"), stds):
        b = drawable[drawable["std"] == s]
        ax.scatter(b.frag, b.cost, s=20, color=colour, marker=marker, edgecolor="white",
                   linewidth=0.5, label=rf"$\sigma_e = {s:g}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    _style(ax, "measured cost\n(fraction of achievable improvement)",
           r"one-shot selection loss $\mathrm{frag}(\sigma_e)$ of the landscape")
    ax.grid(True, axis="x")
    from scipy.stats import spearmanr
    rho = float(spearmanr(f.frag, f.cost).statistic)
    by_std = [float(spearmanr(g.frag, g.cost).statistic) for _, g in f.groupby("std")]
    # Pooled over the magnitudes, the magnitude alone already reaches 0.85, so the
    # pooled number says little. The within-magnitude range is the cross-landscape
    # claim the section actually makes.
    ax.annotate(rf"Spearman $\rho$ = {rho:.2f} over all {len(f)} cells, "
                rf"{min(by_std):.2f} to {max(by_std):.2f} within a magnitude",
                (0.02, 0.97), xycoords="axes fraction", va="top", fontsize=7.5, color=INK2)
    ax.legend(frameon=False, loc="lower right", title="error magnitude", title_fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, "frag_scatter")
    return rho


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = bb.load_stats(args.stats_path)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}

    cells = pd.read_csv(args.analysis / "cell_means.csv")
    cells = cells[~cells.acquisition.isin(MODEL_FREE)].copy()
    missing = sorted(set(cells.dataset) - set(opt_z))
    if missing:
        raise SystemExit(f"no opt_z for {missing}")
    cells["deployed"] = cells.inference_excess / cells.dataset.map(opt_z)

    fig_dose_response(cells, args.out)
    fig_decomposition(pd.read_csv(args.analysis / "regret_decomposition.csv"), args.out)
    fig_kcurve(pd.read_csv(args.policies or args.analysis / "budget_split_policies.csv"), args.out)
    rho = fig_frag_scatter(cells, stats, args.out)
    for name in ("dose_response", "decomposition", "kcurve", "frag_scatter"):
        print(f"wrote {args.out / (name + '.pdf')}")
    print(f"frag scatter: Spearman rho on log-log = {rho:.3f}")


if __name__ == "__main__":
    main()
