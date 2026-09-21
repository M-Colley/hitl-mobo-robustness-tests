"""Generate the paper's LaTeX tables from the analysis outputs.

Nothing here is typed by hand. Every number in `paper/tables/*.tex` is read from
the CSVs `analyse_boba_robustness.py` and `compare_boba_arms.py` write, so the
paper cannot drift from the sweep, and re-running after more seeds land updates
the paper by rebuilding rather than by editing.

  python scripts/make_boba_paper_tables.py --analysis output-boba/analysis
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import boba_benchmarks as bb  # noqa: E402

PRETTY_ACQ = {
    "logei": r"LogEI", "logpi": r"LogPI", "ei": r"EI", "pi": r"PI", "ucb": r"UCB",
    "qucb": r"qUCB", "qei": r"qEI", "qpi": r"qPI", "qnei": r"qNEI",
    "greedy": r"Greedy", "random": r"Random", "sobol": r"Sobol",
}
PRETTY_ERROR = {
    "gaussian": r"\textsc{gaussian}", "bias": r"\textsc{bias}",
    "drift": r"\textsc{drift}", "ar1": r"\textsc{ar}(1)",
}
ERROR_ORDER = ["gaussian", "bias", "drift", "ar1"]

PRETTY_TERM = {
    "log_noise": r"$\log\sigma_e$",
    "log_opt_z": r"$\log\mathrm{opt}_z$",
    "log_tail_ratio": r"$\log$ tail",
    "log_sparsity": r"$\log$ sparsity",
    "ruggedness": "rugged.",
    "skew": "skew",
    "dim": r"$d$",
    "frag_at_c": r"$\mathrm{frag}(\sigma_e)$",
    "noise_x_opt_z": r"$\log\sigma_e{\times}\log\mathrm{opt}_z$",
}


def _tex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return r"$^{***}$" if p < 0.001 else r"$^{**}$" if p < 0.01 else r"$^{*}$" if p < 0.05 else ""


def _sparsity(value: float) -> str:
    """A Sobol sample of 65,536 points cannot resolve below one point."""
    floor = 1.0 / 65536
    return f"$<${floor:.1e}" if value < floor else f"{value:.1e}"


def _centred(header: str) -> str:
    """A header centred over a value-plus-interval cell.

    Those cells are right-aligned, so a plain header would sit over the
    interval box rather than over the value it names.
    """
    return f"\\multicolumn{{1}}{{c}}{{{header}}}"


def _p_fdr(p: float) -> str:
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(path: Path, body: str) -> None:
    # A negative number in a text-mode cell prints with a hyphen, which is
    # shorter than the plus sign in the same column. Only a minus directly
    # before a digit and directly after a cell boundary, a bracket or a space
    # is touched, so exponents (7.2e-01), ranges (2-3) and "--" are left alone.
    body = re.sub(r"(?<=[&\[ ])-(?=\d)", "$-$", body)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path}")


def table_dose_response(analysis: Path, out: Path, stats_path: Path | None = None) -> None:
    """Search loss and deployed loss, one above the other.

    `fragility` is the post-onset per-iteration excess of SEARCH loss, a time
    average over the optimizer's trajectory. `inference_excess` is the excess of
    the design the study would SHIP, at the final trial; it is not divided by
    opt_z in cell_means, so that is done here. Reporting only the first invites
    the reader to take a trajectory average for the cost of a deployment, which
    is between 1.6 and 9.1 times smaller.
    """
    cells = pd.read_csv(analysis / "cell_means.csv")
    cells = cells[~cells["acquisition"].isin(["random", "sobol"])]
    stats = bb.load_stats(stats_path or bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    missing = sorted(set(cells["dataset"]) - set(opt_z))
    if missing:
        raise ValueError(f"no opt_z for {missing}; the deployed row would be on another scale")
    cells = cells.assign(deployed=cells["inference_excess"] / cells["dataset"].map(opt_z))

    labels = {0: "from it.\\ 1", 20: "from it.\\ 21"}
    blocks, columns = [], None
    for title, value in (("search loss, post-onset per-iteration average", "fragility"),
                         ("deployed design, final trial", "deployed")):
        grid = cells.pivot_table(index="jitter_iteration", columns="jitter_std", values=value)
        columns = " & ".join(f"${c:g}\\sigma$" for c in grid.columns)
        rows = [f"\\multicolumn{{{len(grid.columns) + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for onset, row in grid.iterrows():
            cells_tex = " & ".join(f"{v * 100:.1f}\\%" for v in row)
            rows.append(f"\\quad {labels.get(int(onset), f'onset {int(onset)}')} & {cells_tex} \\\\")
        blocks.append(chr(10).join(rows))

    # Third block: how much of the deployed excess above is selection loss. It
    # comes from the decomposition rather than cell_means, and the two pipelines
    # are checked against each other here rather than trusted: the deployed
    # excess is computed independently by both.
    decomp = analysis / "regret_decomposition.csv"
    if decomp.is_file():
        dec = pd.read_csv(decomp)
        dec = dec[(dec["error_model"] == "pooled") & (dec["jitter_std"] != "pooled")].copy()
        dec["jitter_std"] = dec["jitter_std"].astype(float)
        dec["jitter_iteration"] = dec["jitter_iteration"].astype(float)
        check = cells.pivot_table(index="jitter_iteration", columns="jitter_std", values="deployed")
        for _, r in dec.iterrows():
            mine = check.loc[r["jitter_iteration"], r["jitter_std"]]
            if abs(mine - r["mean_excess_deployed"]) > 5e-4:
                raise ValueError(
                    f"cell_means and the decomposition disagree on the deployed excess at "
                    f"{r['jitter_std']}sigma onset {r['jitter_iteration']:.0f}: "
                    f"{mine:.4f} against {r['mean_excess_deployed']:.4f}"
                )
        grid = dec.pivot_table(index="jitter_iteration", columns="jitter_std",
                               values="excess_selection_share")
        lo = dec.pivot_table(index="jitter_iteration", columns="jitter_std",
                             values="excess_selection_share_lo")
        hi = dec.pivot_table(index="jitter_iteration", columns="jitter_std",
                             values="excess_selection_share_hi")
        rows = [f"\\multicolumn{{{len(grid.columns) + 1}}}{{l}}"
                f"{{\\emph{{of that deployed excess, the share that is selection loss}}}} \\\\"]
        for onset in grid.index:
            cells_tex = " & ".join(
                f"{grid.loc[onset, c] * 100:.0f}\\% {{\\scriptsize $[{lo.loc[onset, c] * 100:.0f},"
                f"{hi.loc[onset, c] * 100:.0f}]$}}" for c in grid.columns)
            rows.append(f"\\quad {labels.get(int(onset), f'onset {int(onset)}')} & {cells_tex} \\\\")
        blocks.append(chr(10).join(rows))

    body = "\n\\addlinespace\n".join(blocks)
    write(out / "dose_response.tex", f"""\\begin{{tabular}}{{l{'r' * 4}}}
\\toprule
error present & {columns} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def table_mediation(analysis: Path, out: Path) -> None:
    mediator = pd.read_csv(analysis / "mediator_model.csv")
    if mediator.empty:
        return
    order = ["frag_at_c", "log_noise", "log_opt_z", "noise_x_opt_z", "log_tail_ratio",
             "ruggedness", "dim"]
    # The two interaction models also carry a quadratic in the magnitude,
    # which the table does not show (the caption says so).
    # The two baselines come first. Without them a reader cannot tell how much of
    # the mediator's R^2 is the onset and error-process indicators, and how much is
    # simply knowing the error magnitude.
    models = [("indicators_only", "indicators only"), ("magnitude_only", r"$\log\sigma_e$ only"),
              ("mediator_only", r"$\mathrm{frag}$ only"), ("descriptors_only", "descriptors only"),
              ("both", "both"), ("descriptors_interaction", "descriptors + int."),
              ("both_interaction", "both + int.")]
    header = " & ".join(PRETTY_TERM.get(t, t) for t in order)
    rows = []
    for key, label in models:
        block = mediator[mediator["model"] == key].set_index("term")
        if block.empty:
            continue
        cells = []
        for term in order:
            if term not in block.index:
                cells.append("--")
                continue
            row = block.loc[term]
            cells.append(f"{row['coefficient']:+.2f}{_stars(row['p_bootstrap'])}")
        r2 = block["r_squared"].iloc[0]
        rows.append(f"{label} & {' & '.join(cells)} & {r2:.3f} \\\\")
    write(out / "mediation.tex", f"""\\begin{{tabular}}{{l{'r' * len(order)}r}}
\\toprule
model & {header} & $R^2$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_currency(analysis: Path, out: Path) -> None:
    """Which scale the cost of an error follows.

    beta_c and beta_z are the exponents of E[excess] = A sigma_e^beta_c opt_z^beta_z.
    SPREAD predicts beta_z = 0 and GAIN predicts beta_c + beta_z - 1 = 0, so each
    restriction sits next to the interval of the quantity it constrains.
    """
    currency = pd.read_csv(analysis / "noise_currency.csv")
    rows = []
    currency = currency.assign(_k=currency["error_model"].map(ERROR_ORDER.index))
    for _, row in currency.sort_values(["jitter_iteration", "_k"]).iterrows():
        rows.append(
            f"{PRETTY_ERROR.get(row['error_model'], _tex_escape(row['error_model']))} & "
            f"{int(row['jitter_iteration'])} & "
            f"{row['beta_c']:.2f} & {row['beta_z']:.2f} & "
            f"[{row['beta_z_lo']:.2f}, {row['beta_z_hi']:.2f}] & "
            f"{row['gain_gap']:+.2f} & [{row['gain_gap_lo']:+.2f}, {row['gain_gap_hi']:+.2f}] & "
            f"{_tex_escape(row['verdict'])} \\\\"
        )
    write(out / "currency.tex", f"""\\begin{{tabular}}{{llrrcrcl}}
\\toprule
& & & \\multicolumn{{2}}{{c}}{{\\textsc{{spread}}: $\\beta_z = 0$}} & \\multicolumn{{2}}{{c}}{{\\textsc{{gain}}: $\\beta_c + \\beta_z = 1$}} & \\\\
\\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
error process & $t_0$ & $\\beta_c$ & $\\beta_z$ & 95\\% CI & $\\beta_c + \\beta_z - 1$ & 95\\% CI & verdict \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_acquisitions(analysis: Path, out: Path) -> None:
    overall = pd.read_csv(analysis / "overall_acquisition_rankings.csv")
    floor = pd.read_csv(analysis / "floor_reference.csv")
    rows = [
        f"{PRETTY_ACQ.get(r['acquisition'], r['acquisition'])} & {r['mean_rank']:.2f} & "
        f"{r['mean_fragility'] * 100:.2f}\\% & {r['mean_absolute_loss'] * 100:.1f}\\% \\\\"
        for _, r in overall.iterrows()
    ]
    rows.append(r"\midrule")
    # The floor's excess regret was typed as 0.00% here. It IS zero, but a table
    # that asserts its own negative control cannot check it, so the value is read
    # from the control's own output and refused if it is not zero.
    check = pd.read_csv(analysis / "floor_check.csv")
    worst = float(check["max_abs_excess_auc"].abs().max())
    if worst > 1e-9:
        raise ValueError(
            f"the model-free floor shows excess regret up to {worst:.3e}; it must be zero, "
            "and the table may not print 0.00% over a failed control"
        )
    rows.append(
        f"model-free floor & -- & {worst * 100:.2f}\\% & "
        f"{floor['mean_absolute_loss'].mean() * 100:.1f}\\% \\\\"
    )
    write(out / "acquisitions.tex", f"""\\begin{{tabular}}{{lrrr}}
\\toprule
acquisition & mean rank & excess regret & absolute loss \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_benchmarks(out: Path, stats_path: Path) -> None:
    stats = bb.load_stats(stats_path)
    rows = []
    for name, entry in sorted(stats.items(), key=lambda kv: kv[1]["opt_z"]):
        if name not in bb.DEFAULT_SUITE:
            continue
        rows.append(
            f"\\texttt{{{_tex_escape(name)}}} & {int(entry['dim'])} & {entry['opt_z']:.2f} & "
            f"{_sparsity(entry['sparsity_10pct'])} & {entry['ruggedness']:.2f} & "
            f"{entry['skew']:+.2f} & {entry['tail_ratio']:.2f} & {entry.get('frag_1', float('nan')):.2f} \\\\"
        )
    write(out / "benchmarks.tex", f"""\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
benchmark & $d$ & $\\mathrm{{opt}}_z$ & sparsity & rugged. & skew & tail & $\\mathrm{{frag}}(1)$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_noise_diagnostic(analysis: Path, out: Path) -> None:
    """Fitted / injected observation-noise SD as a function of how much data the
    surrogate has seen. The point of the table is the top-left corner."""
    path = analysis / "gp_noise_diagnostic.csv"
    if not path.exists():
        print(f"skipping noise diagnostic: {path} not present")
        return
    table = pd.read_csv(path)
    table = table[table["jitter_iteration"] == 0]
    grid = table.pivot_table(index="n_observations", columns="jitter_std",
                             values="ratio_default")
    columns = " & ".join(f"${c:g}\\sigma$" for c in grid.columns)
    rows = []
    for n, row in grid.iterrows():
        cells = " & ".join("--" if pd.isna(v) else f"{v:.2f}" for v in row)
        rows.append(f"{int(n)} & {cells} \\\\")
    write(out / "noise_diagnostic.tex", f"""\\begin{{tabular}}{{l{'r' * len(grid.columns)}}}
\\toprule
observations & {columns} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_noise_anchor(path: Path, out: Path) -> None:
    """Where three real human studies sit on the synthetic arm's x-axis."""
    if not path.exists():
        print(f"skipping noise anchor: {path} not present")
        return
    table = pd.read_csv(path)
    rows = []
    for _, row in table.iterrows():
        rows.append(
            f"\\texttt{{{_tex_escape(row['dataset'])}}} & {row['sigma_f']:.3f} & "
            f"{row['opt_z']:.2f} & {row['rating_noise_sd']:.3f} & "
            f"{row['noise_in_landscape_sd']:.2f} \\\\"
        )
    write(out / "noise_anchor.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
study & $\\sigma_f$ & $\\mathrm{{opt}}_z$ & rating noise & noise / $\\sigma_f$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_manipulation(extensions: Path, main: Path, out: Path, stats_path: Path) -> None:
    """The purpose-built families, on the regression's own response.

    Raw post-onset excess regret in landscape SDs at each swept magnitude, for
    gaussian error from the first observation. The bump grid varies amplitude
    (which moves opt_z) against width (which moves sparsity); the Levy rows are
    one function family at three dimensions.
    """
    import glob
    import json

    metric = "auc_simple_regret_excess_true_postonset_per_iter"
    files = sorted(glob.glob(str(extensions / "*" / "evaluation" / "paired_excess_metrics.csv")))
    if not files:
        print(f"skipping manipulation table: nothing under {extensions}")
        return
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ladder = sorted(glob.glob(str(main / "levy_10" / "evaluation" / "paired_excess_metrics.csv")))
    if ladder:
        top = pd.read_csv(ladder[0])
        frame = pd.concat([frame, top[top["seed"] <= frame["seed"].max()]], ignore_index=True)
    # The bump ladder (opt_z and spike volume held fixed across dimensions)
    # lives in its own run directory; the paragraph that rests on it quotes it,
    # so the table carries it too.
    fixed = sorted(glob.glob(str(Path("output-boba-ladder") / "*" / "evaluation" / "paired_excess_metrics.csv")))
    if fixed:
        frame = pd.concat([frame] + [pd.read_csv(f) for f in fixed], ignore_index=True)
    frame = frame[(~frame["acquisition"].isin(["random", "sobol"]))
                  & (frame["error_model"] == "gaussian")
                  & (frame["jitter_iteration"] == 0)]
    grid = frame.pivot_table(index="dataset", columns="jitter_std", values=metric)

    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))["functions"]
    order = [n for n in ["bump_a4_w0.05", "bump_a16_w0.05", "bump_a4_w0.15",
                         "bump_a16_w0.15", "levy_4d", "levy_7d", "levy_10",
                         "bump_d4", "bump_d7", "bump_d11"]
             if n in grid.index]
    columns = " & ".join(f"${c:g}\\sigma$" for c in grid.columns)
    rows = []
    for name in order:
        entry = stats[name]
        cells = " & ".join(f"{v:.3f}" for v in grid.loc[name])
        rows.append(
            f"\\texttt{{{_tex_escape(name)}}} & {int(entry['dim'])} & "
            f"{entry['opt_z']:.1f} & {_sparsity(entry['sparsity_10pct'])} & {cells} \\\\"
        )
        if name == "bump_a16_w0.05":
            rows.append(r"\addlinespace")
        if name in ("bump_a16_w0.15", "levy_10"):
            rows.append(r"\midrule")
    write(out / "manipulation.tex", f"""\\begin{{tabular}}{{lrrr{'r' * len(grid.columns)}}}
\\toprule
landscape & $d$ & $\\mathrm{{opt}}_z$ & sparsity & {columns} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_arm_by_acquisition(path: Path, out: Path, name: str) -> None:
    """Which acquisitions the arm actually moves.

    For the incumbent arm this is the whole finding: UCB and qNEI never read
    best_f, so they must move by exactly zero, and how far the others move is
    how much of their measured fragility was the incumbent's rather than theirs.
    """
    pairs = path / "arm_contrast_pairs.csv"
    if not pairs.exists():
        print(f"skipping {name} by-acquisition: {pairs} not present")
        return
    frame = pd.read_csv(pairs)
    grouped = (
        frame.groupby("acquisition")
        .agg(ref=("ref", "mean"), trt=("trt", "mean"))
        .assign(delta=lambda d: d["trt"] - d["ref"],
                share=lambda d: 1.0 - d["trt"] / d["ref"])
        .sort_values("delta")
    )
    rows = [
        f"{PRETTY_ACQ.get(acq, acq)} & {row['ref']:.3f} & {row['trt']:.3f} & "
        f"{row['delta']:+.3f} & {row['share'] * 100:+.0f}\\% \\\\"
        for acq, row in grouped.iterrows()
    ]
    write(out / f"{name}_by_acquisition.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
acquisition & reference & treatment & $\\Delta$ & share removed \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_arm_contrast(path: Path, out: Path, name: str) -> None:
    summary_path = path / "arm_contrast_summary.csv"
    if not summary_path.exists():
        print(f"skipping {name}: {summary_path} not present yet")
        return
    summary = pd.read_csv(summary_path)
    rows = []
    for _, row in summary.sort_values(["jitter_iteration", "jitter_std"]).iterrows():
        rows.append(
            f"{row['jitter_std']:g} & {int(row['jitter_iteration'])} & "
            f"{row['mean_reference']:.3f} & {row['mean_treatment']:.3f} & "
            f"{row['mean_delta']:+.3f} & {row['share_removed'] * 100:+.0f}\\% & "
            f"{row['cohens_dz']:.2f} & {_p_fdr(row['wilcoxon_p_fdr'])} \\\\"
        )
    write(out / f"{name}.tex", f"""\\begin{{tabular}}{{rrrrrrrr}}
\\toprule
$\\sigma_e$ & $t_0$ & reference & treatment & $\\Delta$ & share & $d_z$ & $p_{{\\text{{FDR}}}}$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_multiobjective(mo_analysis: Path, out: Path) -> None:
    """The two arms side by side, on the one denominator that makes them comparable.

    Both columns are excess divided by the gap between the optimum and a
    same-budget model-free floor. That is NOT the normalisation the scalar arm
    reports elsewhere in the paper (which divides by $\\optz$), and the two must
    not be mixed -- see analyse_boba_mo.py. Intervals are the reason the table
    exists: six problems against twenty means a factor-of-two gap in the point
    estimate can still overlap.
    """
    path = mo_analysis / "mo_vs_scalar.csv"
    if not path.is_file():
        print(f"skipped multiobjective table: {path} not found")
        return
    cross = pd.read_csv(path)
    onsets = sorted(cross["jitter_iteration"].unique())
    early, late = onsets[0], onsets[-1]
    rows = []
    for std in sorted(cross["jitter_std"].unique()):
        cells = []
        for arm in ("scalar", "multi-objective"):
            for onset in (early, late):
                r = cross[(cross.arm == arm) & (cross.jitter_std == std)
                          & (cross.jitter_iteration == onset)]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                # A bootstrap lower bound of -0.4% renders as "-0" under %.0f,
                # which reads as a typo rather than as "indistinguishable from
                # zero". Round first, then add 0.0 to normalise the negative
                # zero away.
                lo = round(r["ci_low"] * 100) + 0.0
                hi = round(r["ci_high"] * 100) + 0.0
                cells.append(f"{r['mean'] * 100:.1f} \\makebox[3.4em][l]{{\\scriptsize [{lo:.0f}, {hi:.0f}]}}")
        rows.append(f"${std:g}\\sigma$ & " + " & ".join(cells) + r" \\")
    n_sc = int(cross[cross.arm == "scalar"]["n_problems"].iloc[0])
    n_mo = int(cross[cross.arm == "multi-objective"]["n_problems"].iloc[0])
    # Keep the spanning header on ONE source line: check_paper.py counts cells
    # per line, and a row split across two lines reads to it as a 2-cell row in
    # a 5-column tabular.
    write(out / "multiobjective.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{scalar ({n_sc} landscapes)}} & \\multicolumn{{2}}{{c}}{{multi-objective ({n_mo} problems)}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
error & \\multicolumn{{1}}{{c}}{{from it.\\ 1}} & \\multicolumn{{1}}{{c}}{{from it.\\ {int(late) + 1}}} & \\multicolumn{{1}}{{c}}{{from it.\\ 1}} & \\multicolumn{{1}}{{c}}{{from it.\\ {int(late) + 1}}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_multiobjective_acquisitions(mo_analysis: Path, out: Path) -> None:
    path = mo_analysis / "mo_acquisitions.csv"
    if not path.is_file():
        print(f"skipped multiobjective acquisition table: {path} not found")
        return
    acq = pd.read_csv(path)
    pretty = {"qehvi": "qEHVI", "qnehvi": "qNEHVI",
              "qlogehvi": "qLogEHVI", "qlognehvi": "qLogNEHVI"}
    rows = [
        f"{pretty.get(r.acquisition, r.acquisition)} & {r.mean_rank:.2f} "
        f"& {r.mean_frac * 100:.1f}\\% \\\\"
        for r in acq.itertuples()
    ]
    write(out / "multiobjective_acquisitions.tex", f"""\\begin{{tabular}}{{lrr}}
\\toprule
acquisition & mean rank & absolute loss \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def _matched_dose_grid(runs: Path, error_model: str, acquisitions: list[str],
                       seeds: list[int]) -> pd.DataFrame | None:
    """The main sweep restricted to what a follow-up arm actually ran.

    The budget, instrument and robust arms ran one error process, a subset of
    the acquisitions and five seeds. Setting them beside the full sweep
    (four processes, ten acquisitions, ten seeds) changed three things at once
    besides the one under test, and at 1 sigma made a discretised rating scale
    look 17% better than a continuous one when the like-for-like difference is
    zero. This reads the per-run pairs and applies the arm's own restriction.
    """
    import glob
    import json

    meta = runs / "run_metadata.json"
    files = sorted(glob.glob(str(runs / "*" / "evaluation" / "paired_excess_metrics.csv")))
    if not meta.is_file() or not files:
        return None
    opt_z = {k: float(v["opt_z"]) for k, v in
             json.loads(meta.read_text(encoding="utf-8"))["landscape_stats"].items()}
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame = frame[(frame["error_model"] == error_model)
                  & frame["acquisition"].isin(acquisitions)
                  & frame["seed"].isin(seeds)]
    frame = frame.assign(fragility=frame["auc_simple_regret_excess_true_postonset_per_iter"]
                         / frame["dataset"].map(opt_z))
    cells = (frame.groupby(["dataset", "acquisition", "jitter_std", "jitter_iteration"])
             ["fragility"].mean().reset_index())
    return cells.pivot_table(index="jitter_iteration", columns="jitter_std", values="fragility")


def _dose_grid(analysis) -> pd.DataFrame | None:
    """Model-based fragility as onset x magnitude.

    ``analysis`` is either an arm's analysis directory (its own cell means) or
    a dict {runs, error_model, acquisitions, seeds} naming a restriction of the
    main sweep, for a reference row matched to the arm it sits beside.
    """
    if isinstance(analysis, dict):
        return _matched_dose_grid(**analysis)
    path = analysis / "cell_means.csv"
    if not path.is_file():
        return None
    cells = pd.read_csv(path)
    cells = cells[~cells["acquisition"].isin(["random", "sobol"])]
    return cells.pivot_table(
        index="jitter_iteration", columns="jitter_std", values="fragility"
    )


def table_budget(arms: dict[str, Path], out: Path) -> None:
    """Onset held at the same FRACTION of the budget, so budget is the only change."""
    rows, columns = [], None
    for label, analysis in arms.items():
        grid = _dose_grid(analysis)
        if grid is None:
            continue
        columns = columns or [f"${c:g}\\sigma$" for c in grid.columns]
        early, late = grid.index.min(), grid.index.max()
        rows.append(
            f"{label} & from it.\\ 1 & "
            + " & ".join(f"{v * 100:.1f}" for v in grid.loc[early]) + r" \\"
        )
        rows.append(
            f" & from it.\\ {int(late) + 1} & "
            + " & ".join(f"{v * 100:.1f}" for v in grid.loc[late]) + r" \\"
        )
        rows.append(
            " & ratio & "
            + " & ".join(f"{v:.1f}$\\times$" for v in grid.loc[early] / grid.loc[late])
            + r" \\"
        )
        rows.append(r"\midrule")
    if not rows:
        print("skipped budget table: no arms found")
        return
    rows = rows[:-1]
    write(out / "budget.tex", f"""\\begin{{tabular}}{{ll{'r' * len(columns)}}}
\\toprule
budget & error present & {' & '.join(columns)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_arm_dose(arms: dict[str, Path], out: Path, name: str, first_column: str) -> None:
    """Several arms' dose-response side by side, one row block each."""
    rows, columns = [], None
    for label, analysis in arms.items():
        grid = _dose_grid(analysis)
        if grid is None:
            continue
        columns = columns or [f"${c:g}\\sigma$" for c in grid.columns]
        early, late = grid.index.min(), grid.index.max()
        rows.append(
            f"{label} & from it.\\ {int(early) + 1} & "
            + " & ".join(f"{v * 100:.1f}" for v in grid.loc[early]) + r" \\"
        )
        rows.append(
            f" & from it.\\ {int(late) + 1} & " + " & ".join(f"{v * 100:.1f}" for v in grid.loc[late]) + r" \\"
        )
    if not rows:
        print(f"skipped {name} table: no arms found")
        return
    write(out / f"{name}.tex", f"""\\begin{{tabular}}{{ll{'r' * len(columns)}}}
\\toprule
{first_column} & error present & {' & '.join(columns)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_fitted_companion(path: Path, out: Path) -> None:
    if not path.is_file():
        print(f"skipped fitted companion table: {path} not found")
        return
    cross = pd.read_csv(path)
    order = ["synthetic", "fitted", "fitted no-aug"]
    pretty = {"synthetic": "known functions (20)", "fitted": "fitted oracle (3)",
              "fitted no-aug": "fitted, no augmentation (3)"}
    onsets = sorted(cross["jitter_iteration"].unique())
    early, late = onsets[0], onsets[-1]
    rows = []
    for arm in order:
        block = cross[cross.arm == arm]
        if block.empty:
            continue
        cells = []
        for mag in sorted(block["sigma_multiple"].unique()):
            r = block[(block.sigma_multiple == mag) & (block.jitter_iteration == early)]
            cells.append(f"{r['mean'].iloc[0] * 100:.0f}" if len(r) else "--")
        for mag in sorted(block["sigma_multiple"].unique()):
            r = block[(block.sigma_multiple == mag) & (block.jitter_iteration == late)]
            cells.append(f"{r['mean'].iloc[0] * 100:.0f}" if len(r) else "--")
        rows.append(f"{pretty.get(arm, arm)} & " + " & ".join(cells) + r" \\")
    mags = sorted(cross["sigma_multiple"].unique())
    header = " & ".join(f"${m:g}\\sigma$" for m in mags)
    write(out / "fitted_companion.tex", f"""\\begin{{tabular}}{{l{'r' * (2 * len(mags))}}}
\\toprule
& \\multicolumn{{{len(mags)}}}{{c}}{{error from it.\\ 1}} & \\multicolumn{{{len(mags)}}}{{c}}{{error from it.\\ 21}} \\\\
\\cmidrule(lr){{2-{1 + len(mags)}}} \\cmidrule(lr){{{2 + len(mags)}-{1 + 2 * len(mags)}}}
arm & {header} & {header} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_confirmatory(analysis: Path, out: Path) -> None:
    """Preregistered claims, screening estimate beside the fresh-seed estimate.

    The verdict column reports the SENSITIVITY analysis, because the
    preregistered run is void on a control failure and a void run has no
    per-claim verdict to report. The caption has to say so; the table cannot.
    """
    path = analysis / "confirmatory_results.json"
    if not path.is_file():
        print(f"skipped confirmatory table: {path} not found")
        return
    result = json.loads(path.read_text(encoding="utf-8"))["results"]

    def cell(block: dict, keys: tuple[str, ...], fmt: str) -> str:
        for key in keys:
            if key in block:
                return format(block[key], fmt)
        return "--"

    spec = [
        ("H1", r"H1 $\beta(\frag)$", ("beta_frag",), "+.2f"),
        ("H1", r"H1 $\beta(\log_{10}\sigma_e)$", ("beta_noise",), "+.2f"),
        ("H2", "H2 onset ratio", ("ratio",), ".2f"),
        ("H3", "H3 robust mean rank", ("robust_mean_rank",), ".2f"),
        ("H3", "H3 fragile mean rank", ("fragile_mean_rank",), ".2f"),
    ]
    rows = []
    for key, label, fields, fmt in spec:
        block = result.get(key, {})
        screening = cell(block.get("screening", {}), fields, fmt)
        confirmatory = cell(block.get("confirmatory", {}), fields, fmt)
        rows.append(f"{label} & {screening} & {confirmatory} \\\\")
    h4 = result.get("H4", {})
    yes_no = lambda block: "yes" if block.get("confirmed") else "no"
    rows.append(f"H4 dose-response monotone & {yes_no(h4.get('screening', {}))} & "
                f"{yes_no(h4.get('confirmatory', {}))} \\\\")
    verdicts = " ".join(
        f"{k}: {'yes' if result.get(k, {}).get('confirmatory', {}).get('confirmed') else 'no'};"
        for k in ("H1", "H2", "H3", "H4")
    )
    rows.append(r"\midrule")
    rows.append(f"\\multicolumn{{3}}{{l}}{{replicates --- {verdicts.rstrip(';')}}} \\\\")
    write(out / "confirmatory.tex", f"""\\begin{{tabular}}{{lrr}}
\\toprule
preregistered quantity & screening & fresh seeds \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_descriptor_regression(analysis: Path, out: Path) -> None:
    """The full landscape-descriptor fit, with the collinearity that limits it."""
    reg = analysis / "descriptor_regression.csv"
    vif = analysis / "descriptor_vif.csv"
    if not reg.is_file() or not vif.is_file():
        print("skipped descriptor table: analysis outputs not found")
        return
    fit = pd.read_csv(reg)
    fit = fit[fit["model"] == "primary"]
    inflation = pd.read_csv(vif).set_index("term")["vif"].to_dict()
    order = ["log_noise", "log_opt_z", "log_tail_ratio", "ruggedness", "dim"]
    rows = []
    for term in order:
        block = fit[fit["term"] == term]
        if block.empty:
            continue
        r = block.iloc[0]
        p = r.get("p_fdr")
        p = r["p_bootstrap"] if pd.isna(p) else p
        rows.append(
            f"{PRETTY_TERM.get(term, term)} & {r['coefficient']:+.2f} & "
            f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] & {p:.3f}{_stars(p)} & "
            f"{inflation.get(term, float('nan')):.1f} \\\\"
        )
    r2 = fit["r_squared"].iloc[0]
    n_cells = int(fit["n_cells"].iloc[0])
    n_clusters = int(fit["n_clusters"].iloc[0])
    write(out / "descriptors.tex", f"""\\begin{{tabular}}{{lrcrr}}
\\toprule
term & $\\beta$ & 95\\% CI & $p_{{\\mathrm{{FDR}}}}$ & VIF \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
\\multicolumn{{5}}{{l}}{{$R^2 = {r2:.3f}$, {n_cells} cells, {n_clusters} landscape clusters}} \\\\
\\bottomrule
\\end{{tabular}}""")


def _pct_ci(row: pd.DataFrame, width: str = "3.4em") -> str:
    """A cell mean as a percentage with its bootstrap interval, or -- if absent.

    round() before adding 0.0 so that a lower bound of -0.4% prints as "0",
    not as a "-0" that reads like a typo.
    """
    if row.empty:
        return "--"
    r = row.iloc[0]
    lo = round(r["ci_low"] * 100) + 0.0
    hi = round(r["ci_high"] * 100) + 0.0
    mean = round(r["mean"] * 100, 1) + 0.0
    # A fixed-width box for the interval keeps the values aligned on the
    # decimal point in a right-aligned column; 3.4em fits "[77, 232]".
    return f"{mean:.1f} \\makebox[{width}][l]{{\\scriptsize [{lo:.0f}, {hi:.0f}]}}"


def table_inputerror_dose(analysis: Path, out: Path) -> None:
    """How much a slip and a misclick destroy, early and late, with intervals.

    The magnitude column is a fraction of each coordinate's range for a slip
    and a probability for a misclick. Neither is in landscape standard
    deviations, and the caption must say so: these rows are not comparable
    with the response-error tables row for row, only on the damage axis.
    """
    path = analysis / "inputerror_dose.csv"
    if not path.is_file():
        print(f"skipped input-error dose table: {path} not found")
        return
    dose = pd.read_csv(path)
    onsets = sorted(dose["jitter_iteration"].unique())
    early, late = onsets[0], onsets[-1]
    arms = [a for a in ("slip", "misclick") if a in set(dose["arm"])]
    rows = []
    for mag in sorted(dose["jitter_std"].unique()):
        cells = []
        for arm in arms:
            for onset in (early, late):
                cells.append(_pct_ci(dose[(dose.arm == arm) & (dose.jitter_std == mag)
                                          & (dose.jitter_iteration == onset)]))
        rows.append(f"{mag * 100:g}\\% & " + " & ".join(cells) + r" \\")
    groups = " & ".join(f"\\multicolumn{{2}}{{c}}{{{a}}}" for a in arms)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(arms)))
    sub = " & ".join(f"from it.\\ 1 & from it.\\ {int(late) + 1}" for _ in arms)
    write(out / "inputerror_dose.tex", f"""\\begin{{tabular}}{{l{'rr' * len(arms)}}}
\\toprule
& {groups} \\\\
{rules}
magnitude & {sub} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_inputerror_mechanism(analysis: Path, out: Path) -> None:
    """Where a slip's cost comes from, at both onsets.

    floor: model-free, so the pure geometric cost of evaluating the wrong
    points. logged: a learner with true data that cannot place its
    evaluations. unnoticed: the same plus a mislabelled surrogate. The last
    column is the paired difference of those two -- the cost of mislabelling.
    One block of rows per onset.
    """
    mech_path = analysis / "inputerror_mechanism.csv"
    if not mech_path.is_file():
        print(f"skipped input-error mechanism table: {mech_path} not found")
        return
    mech = pd.read_csv(mech_path)
    mis_path = analysis / "inputerror_mislabel.csv"
    mis = pd.read_csv(mis_path) if mis_path.is_file() else mech.iloc[0:0]
    columns = [("floor", mech, "floor"), ("actual", mech, "logged"),
               ("proposed", mech, "unnoticed"), ("proposed - actual", mis, "mislabelling")]
    present = [(arm, src, label) for arm, src, label in columns if arm in set(src["arm"])]
    rows = []
    for onset in sorted(mech["jitter_iteration"].unique()):
        rows.append(f"\\multicolumn{{{len(present) + 1}}}{{l}}{{\\textit{{error from it.\\ {int(onset) + 1}}}}} \\\\")
        for mag in sorted(mech["jitter_std"].unique()):
            cells = [_pct_ci(src[(src.arm == arm) & (src.jitter_std == mag)
                                 & (src.jitter_iteration == onset)])
                     for arm, src, _ in present]
            rows.append(f"{mag * 100:g}\\% & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    header = " & ".join(_centred(label) for _, _, label in present)
    write(out / "inputerror_mechanism.tex", f"""\\begin{{tabular}}{{l{'r' * len(present)}}}
\\toprule
\\textsc{{slip}} & {header} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_inputerror_deployed(analysis: Path, out: Path) -> None:
    """Under an unnoticed slip: the best design touched against the design the log names.

    evaluated: final excess simple regret of the best true value among the
    points actually evaluated. deployed: the same at the recorded location of
    the best-rated trial, which is where a practitioner would go.
    """
    path = analysis / "inputerror_deployed.csv"
    if not path.is_file():
        print(f"skipped input-error deployed table: {path} not found")
        return
    dep = pd.read_csv(path)
    onsets = sorted(dep["jitter_iteration"].unique())
    early, late = onsets[0], onsets[-1]
    rows = []
    for mag in sorted(dep["jitter_std"].unique()):
        cells = [_pct_ci(dep[(dep.arm == arm) & (dep.jitter_std == mag) & (dep.jitter_iteration == onset)])
                 for onset in (early, late) for arm in ("evaluated", "deployed")]
        rows.append(f"{mag * 100:g}\\% & " + " & ".join(cells) + r" \\")
    write(out / "inputerror_deployed.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{from it.\\ 1}} & \\multicolumn{{2}}{{c}}{{from it.\\ {int(late) + 1}}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
\\textsc{{slip}} & \\multicolumn{{1}}{{c}}{{evaluated}} & \\multicolumn{{1}}{{c}}{{deployed}} & \\multicolumn{{1}}{{c}}{{evaluated}} & \\multicolumn{{1}}{{c}}{{deployed}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_inputerror_main(analysis: Path, out: Path) -> None:
    """The main-text input-error table: dose-response plus the early-onset slip decomposed.

    The mechanism table's "unnoticed" column is the dose table's early-slip
    column, so the two can share one table without repeating it: floor, logged
    and the paired mislabelling cost sit beside the slip and misclick doses.
    """
    paths = {k: analysis / f"inputerror_{k}.csv" for k in ("dose", "mechanism", "mislabel")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print(f"skipped input-error main table: {missing[0]} not found")
        return
    dose, mech, mis = (pd.read_csv(paths[k]) for k in ("dose", "mechanism", "mislabel"))
    onsets = sorted(dose["jitter_iteration"].unique())
    early, late = onsets[0], onsets[-1]

    def pick(frame, arm, mag, onset):
        return frame[(frame.arm == arm) & (frame.jitter_std == mag) & (frame.jitter_iteration == onset)]

    rows = []
    # A narrower interval box than the appendix tables use: this table's widest
    # interval is "[16, 31]", and eight columns have to share the text width.
    narrow = lambda r: _pct_ci(r, "3.0em")
    for mag in sorted(dose["jitter_std"].unique()):
        cells = [narrow(pick(dose, "slip", mag, early)), narrow(pick(dose, "slip", mag, late)),
                 narrow(pick(dose, "misclick", mag, early)), narrow(pick(dose, "misclick", mag, late)),
                 narrow(pick(mech, "floor", mag, early)), narrow(pick(mech, "actual", mag, early)),
                 narrow(pick(mis, "proposed - actual", mag, early))]
        rows.append(f"{mag * 100:g}\\% & " + " & ".join(cells) + r" \\")
    heads = " & ".join(_centred(h) for h in (f"from it.\\ 1", f"from it.\\ {int(late) + 1}", "from it.\\ 1",
                                              f"from it.\\ {int(late) + 1}", "floor", "logged", "mislabelling"))
    write(out / "inputerror_main.tex", f"""\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textsc{{slip}}}} & \\multicolumn{{2}}{{c}}{{\\textsc{{misclick}}}} & \\multicolumn{{3}}{{c}}{{\\textsc{{slip}} from it.\\ 1, decomposed}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-8}}
magnitude & {heads} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def _extra_cell(row: pd.DataFrame, budget_left: float | None = None) -> str:
    """Median extra trials and the share that never reach the target.

    The median is exact while fewer than half the runs are censored; past
    that it is only a bound, printed as such.
    """
    if row.empty:
        return "--"
    r = row.iloc[0]
    # Round half up: a median of 40.5 trials (an even run count) and a share of
    # 38.5% both sit on the boundary, and Python's banker's rounding sent them
    # the wrong way.
    half_up = lambda x: int(math.floor(x + 0.5))
    never = f"{half_up(r['censored_fraction'] * 100)}\\%"
    if r["censored_fraction"] >= 0.5:
        return f"$>${half_up(r['median_extra'])} ({never})"
    return f"{half_up(r['median_extra'])} ({never})"


def table_extra_runs(analysis: Path, out: Path, tolerance: float = 0.01) -> None:
    """Extra trials a noisy run needs to match a clean one, from the budget-100 arm.

    Early onset against a clean 25-trial study, late onset (error from trial
    41) against a clean 50-trial study; the target is the clean run's regret
    within `tolerance` of the achievable improvement. Cells are the median
    extra trials with the share of runs that never got there in 100 trials.
    """
    path = analysis / "extra_runs.csv"
    if not path.is_file():
        print(f"skipped extra-runs table: {path} not found")
        return
    e = pd.read_csv(path)
    e = e[(e.tolerance == tolerance) & (e.error_model == "gaussian") & (e.variant.fillna("") == "")]
    onsets = sorted(e.jitter_iteration.unique())
    early, late = onsets[0], onsets[-1]
    k_early, k_late = 25, 50
    rows = []
    for mag in sorted(e.jitter_std.unique()):
        a = e[(e.jitter_std == mag) & (e.jitter_iteration == early) & (e.k == k_early)]
        b = e[(e.jitter_std == mag) & (e.jitter_iteration == late) & (e.k == k_late)]
        rows.append(f"${mag:g}\\sigma$ & {_extra_cell(a)} & {_extra_cell(b)} \\\\")
    write(out / "extra_runs.tex", f"""\\begin{{tabular}}{{lrr}}
\\toprule
& error from it.\\ 1, & error from it.\\ {int(late) + 1}, \\\\
error & match a clean {k_early}-trial study & match a clean {k_late}-trial study \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_extra_runs_full(arms: dict[str, tuple[Path, str, str]], out: Path,
                          k: int = 10, tolerance: float = 0.01) -> None:
    """Every arm at the early onset: extra trials to match a clean k-trial study."""
    blocks = []
    for label, (analysis, model, variant) in arms.items():
        path = analysis / "extra_runs.csv"
        if not path.is_file():
            print(f"skipped extra-runs row {label}: {path} not found")
            continue
        e = pd.read_csv(path)
        # The bias runs carry their offset in the variant ("bias0.25"), one per
        # magnitude; every other model's variant is empty or the recorded-point flag.
        v = e.variant.fillna("")
        wanted = (v == variant) if variant else (v == "") | v.str.startswith(model)
        e = e[(e.tolerance == tolerance) & (e.error_model == model) & wanted
              & (e.k == k) & (e.jitter_iteration == e.jitter_iteration.min())]
        if e.empty:
            continue
        cells = [_extra_cell(e[e.jitter_std == mag]) for mag in sorted(e.jitter_std.unique())]
        unit = "sigma" if model in PRETTY_ERROR else "box"
        blocks.append((unit, PRETTY_ERROR.get(model, f"\\textsc{{{label}}}") if variant == "" else f"\\textsc{{{model}}}, logged",
                       sorted(e.jitter_std.unique()), cells))
    if not blocks:
        print("skipped full extra-runs table: nothing found")
        return
    rows = []
    for unit in ("sigma", "box"):
        group = [b for b in blocks if b[0] == unit]
        if not group:
            continue
        mags = group[0][2]
        head = " & ".join(f"${m:g}\\sigma$" if unit == "sigma" else f"{m * 100:g}\\%" for m in mags)
        rows.append(f"\\midrule\nerror & {head} \\\\\n\\midrule" if rows else f"error & {head} \\\\\n\\midrule")
        for _, label, _, cells in group:
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write(out / "extra_runs_full.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


# (group label, [(arm key, row label), ...]) in the order the table prints them.
ADAPTATION_GROUPS = [
    ("rating error (\\textsc{gaussian})", [
        ("incumbent", "observed-max incumbent"),
        ("knownnoise", "GP given the true noise variance"),
        ("rep10", "first ten proposals rated twice"),
        ("bundle", "\\quad + observed-max incumbent (LogEI)"),
        ("rerate", "last six trials re-rate the top three"),
        ("qkg", "knowledge gradient"),
        ("replei", "re-rate the incumbent every second trial"),
    ]),
    ("unnoticed \\textsc{slip}", [
        ("nigp", "noisy-input GP"),
        ("rerate-slip", "last six trials re-rate the top three"),
    ]),
    ("\\textsc{misclick}", [
        ("studentt", "Student-$t$ surrogate"),
    ]),
    ("changing what the rater is asked or when", [
        ("idea-selfreport", "the rater reports their own precision"),
        ("idea-anchor-gaussian", "each proposal judged beside the incumbent"),
        ("idea-anchors-gaussian", "every fifth trial rates a fixed anchor"),
        ("idea-hold", "the first five proposals rated late"),
        ("idea-shiplcb", "an acquisition aimed at the ship rule"),
    ]),
    ("fitted oracles, rating error", [
        ("fitted-rep10", "first ten proposals rated twice"),
    ]),
]


def _recovered_cell(r: pd.Series) -> str:
    """Pooled share recovered with its landscape-bootstrap interval, in percent."""
    if pd.isna(r["pooled_recovered"]):
        return "--"
    lo, hi = r["pooled_recovered_lo"], r["pooled_recovered_hi"]
    # Star on the test the analysis actually ran, a BH-corrected Wilcoxon over
    # the per-landscape gains, NOT on "the interval excludes zero". The two
    # disagreed on two rows, one of them resting on a bootstrap lower bound of
    # +0.0004 from 2000 resamples of 20 landscapes, inside its own Monte Carlo
    # error.
    q = r.get("pooled_wilcoxon_p_fdr", r.get("pooled_wilcoxon_p", float("nan")))
    mark = r"$^{*}$" if (not pd.isna(q) and q < 0.05) else ""
    # round() then + 0.0, so a value of -0.3% prints as "0", not a "-0" that reads as a typo.
    pct = lambda v: round(v * 100) + 0.0

    def bound(v: float) -> str:
        # A bound that is not zero but rounds to it keeps one significant figure,
        # so a starred interval never prints as touching zero ([0.04, 9], not [0, 9]).
        p = v * 100
        if p != 0 and round(p) == 0:
            exponent = int(f"{abs(p):.0e}".split("e")[1])
            return f"{p:.{min(2, max(1, -exponent))}f}"
        return f"{pct(v):.0f}"

    interval = "" if pd.isna(lo) else f" {{\\scriptsize [{bound(lo)}, {bound(hi)}]}}"
    value = pct(r["pooled_recovered"])
    return f"{value:+.0f}{mark}{interval}" if value != 0 else f"0{mark}{interval}"


SHORTLIST_ARMS = [
    ("output-boba", "the main sweep"),
    ("output-boba-spike", "$15\\%$ of trials spiking at $20$ SD"),
    ("output-boba-ceiling", "a rating scale capped at its $0.9$ quantile"),
]


def table_shortlist(analysis: Path, out: Path, root: Path | None = None) -> None:
    """What a shortlist recovers, and what the same runs recover by rank.

    Both are read from replay_hitl_remedies.py's own recovery files, one per
    arm, so the table cannot drift from the replay that produced it. The
    shortlist row at m = 1 IS the cautious ship rule, which is what makes the
    column readable: everything above it is what the wider deliverable buys.
    """
    root = root or Path(".")
    NL = chr(10)
    procedures = [("shortlist_m1", "ship 1 design (the cautious rule)"),
                  ("shortlist_m2", "ship 2"),
                  ("shortlist_m3", "ship 3"),
                  ("shortlist_m5", "ship 5"),
                  ("ordinal_lcb1", "ship 1, chosen on the ratings' ranks")]
    columns, blocks = [], {}
    for arm_dir, label in SHORTLIST_ARMS:
        path = root / arm_dir / "analysis" / "hitl_remedies" / "hitl_remedies_recovery.csv"
        if not path.is_file():
            print(f"skipped shortlist table: {path} not found")
            return
        frame = pd.read_csv(path)
        frame = frame[frame["error_model"] == "pooled"].set_index("procedure")
        columns.append(label)
        blocks[label] = frame
    rows = []
    for proc, label in procedures:
        cells = []
        for column in columns:
            block = blocks[column]
            if proc not in block.index:
                cells.append("--")
                continue
            r = block.loc[proc]
            lo, hi = r["recovered_lo"], r["recovered_hi"]
            interval = "" if pd.isna(lo) else f" {{\\scriptsize $[{lo * 100:.0f},{hi * 100:.0f}]$}}"
            cells.append(f"${r['recovered'] * 100:.0f}\\%$" + interval)
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    header = " & ".join(columns)
    body = NL.join(rows)
    tex = (f"\\begin{{tabular}}{{l{'r' * len(columns)}}}{NL}\\toprule{NL}"
           f"deliverable & {header} \\\\{NL}\\midrule{NL}{body}{NL}\\bottomrule{NL}"
           f"\\end{{tabular}}{NL}")
    (out / "shortlist.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {out / 'shortlist.tex'}")


def table_adaptations(analysis: Path, out: Path) -> None:
    """The process adaptations, scored against the STANDARD process.

    For each arm and response (trajectory regret; regret of the deployed
    design): the pooled share of the standard process's cost of error that the
    adaptation recovers, with its landscape-bootstrap interval (a star when the
    interval excludes zero), and the price the adaptation pays when there is no
    error, in percent of the achievable improvement. Negative recovery means
    the adapted process does worse under error than the standard one. The
    fitted-oracle datasets have no achievable-improvement scale, so no price.
    """
    path = analysis / "adaptations_recovery.csv"
    if not path.is_file():
        print(f"skipped adaptations table: {path} not found")
        return
    c = pd.read_csv(path)
    rows = []
    for group, members in ADAPTATION_GROUPS:
        body = []
        for arm, label in members:
            cells, prices = [], []
            for response in ("trajectory", "deployed"):
                block = c[(c.arm == arm) & (c.response == response)]
                if block.empty:
                    cells.append("--")
                    prices.append("--")
                    continue
                r = block.iloc[0]
                cells.append(_recovered_cell(r))
                prices.append("--" if arm.startswith("fitted") else f"{r['pooled_price'] * 100:+.1f}")
            if cells == ["--", "--"]:
                continue
            body.append(f"{label} & " + " & ".join(cells + prices) + r" \\")
        if body:
            # Upright, not italic: the labels carry small caps, and Times has no
            # italic small caps (LaTeX substitutes a font and warns).
            rows.append(f"\\multicolumn{{5}}{{l}}{{{group}}} \\\\")
            rows.extend(body)
    if not rows:
        print("skipped adaptations table: no completed arm")
        return
    write(out / "adaptations.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{recovered (\\%)}} & \\multicolumn{{2}}{{c}}{{price without error}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
adaptation & trajectory & deployed & trajectory & deployed \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def table_pilot_frag(analysis: Path, out: Path) -> None:
    """Can frag be computed from a pilot? Correlations across landscapes.

    Rows: frag computed on the posterior mean of a GP fitted to the first k
    trials of a clean run (seed mean), and the exact frag as the ceiling.
    Columns: Spearman correlation with the exact frag (pooled over the four
    magnitudes), and with the measured early-onset cost, pooled and at 1 and 5
    landscape SDs.
    """
    path = analysis / "pilot_frag_summary.csv"
    if not path.is_file():
        print(f"skipped pilot-frag table: {path} not found")
        return
    s = pd.read_csv(path, dtype={"k": str, "sigma_e": str})

    def rho(k: str, sigma: str, target: str) -> str:
        r = s[(s.k == k) & (s.sigma_e == sigma) & (s.target == target)]
        if r.empty or pd.isna(r.iloc[0]["spearman"]):
            return "--"
        return f"{r.iloc[0]['spearman']:.2f}"

    rows = []
    ks = [k for k in s["k"].unique() if k != "exact"]
    for k in sorted(ks, key=int) + ["exact"]:
        label = f"first {k} trials" if k != "exact" else "exact objective"
        rows.append(f"{label} & {rho(k, 'pooled', 'frag_exact')} & {rho(k, 'pooled', 'cost_measured')} & "
                    f"{rho(k, '1', 'cost_measured')} & {rho(k, '5', 'cost_measured')} \\\\")
    write(out / "pilot_frag.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{1}}{{c}}{{with exact frag}} & \\multicolumn{{3}}{{c}}{{with measured cost}} \\\\
\\cmidrule(lr){{2-2}} \\cmidrule(lr){{3-5}}
frag computed from & pooled & pooled & $1\\sigma$ & $5\\sigma$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=Path("output-boba/analysis"))
    parser.add_argument("--known-noise", type=Path,
                        default=Path("output-boba-knownnoise/analysis"))
    parser.add_argument("--incumbent", type=Path, default=Path("output-boba-incumbent/analysis"))
    parser.add_argument("--extensions", type=Path, default=Path("output-boba-extensions"))
    parser.add_argument("--main-runs", type=Path, default=Path("output-boba"))
    parser.add_argument("--noise-anchor", type=Path, default=Path("output/noise_anchor.csv"))
    parser.add_argument("--multiobjective", type=Path,
                        default=Path("output-boba-mo/analysis"))
    parser.add_argument("--out", type=Path, default=Path("paper/tables"))
    parser.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    table_dose_response(args.analysis, args.out)
    table_mediation(args.analysis, args.out)
    table_currency(args.analysis, args.out)
    table_acquisitions(args.analysis, args.out)
    table_benchmarks(args.out, args.stats_path)
    table_noise_diagnostic(args.analysis, args.out)
    table_noise_anchor(args.noise_anchor, args.out)
    table_manipulation(args.extensions, args.main_runs, args.out, args.stats_path)
    table_arm_contrast(args.known_noise, args.out, "known_noise")
    table_arm_contrast(args.incumbent, args.out, "incumbent")
    table_arm_by_acquisition(args.known_noise, args.out, "known_noise")
    table_arm_by_acquisition(args.incumbent, args.out, "incumbent")
    table_multiobjective(args.multiobjective, args.out)
    table_multiobjective_acquisitions(args.multiobjective, args.out)
    # Reference rows are the main sweep restricted to what each arm ran: one
    # error process, its acquisition list, its seeds (see _matched_dose_grid).
    six = ["logei", "ei", "pi", "ucb", "qucb", "qnei"]
    ten = six + ["logpi", "qei", "qpi", "greedy"]
    five_seeds = [7, 8, 9, 10, 11]
    table_budget({
        "$T=25$": Path("output-boba-budget25/analysis"),
        "$T=50$": {"runs": args.main_runs, "error_model": "gaussian",
                   "acquisitions": six, "seeds": five_seeds},
        "$T=100$": Path("output-boba-budget100/analysis"),
    }, args.out)
    table_arm_dose({
        "unbounded, continuous": {"runs": args.main_runs, "error_model": "gaussian",
                                  "acquisitions": six, "seeds": five_seeds},
        "bounded, discrete": Path("output-boba-instrument/analysis"),
    }, args.out, "instrument", "response scale")
    table_arm_dose({
        "ten standard": {"runs": args.main_runs, "error_model": "gaussian",
                         "acquisitions": ten, "seeds": five_seeds},
        "qKG and replication": Path("output-boba-robust/analysis"),
    }, args.out, "robust", "acquisitions")
    table_extra_runs(Path("output-boba-budget100/analysis"), args.out)
    table_adaptations(args.analysis, args.out)
    table_shortlist(args.analysis, args.out)
    table_pilot_frag(args.analysis, args.out)
    table_extra_runs_full({
        "gaussian": (Path("output-boba/analysis"), "gaussian", ""),
        "bias": (Path("output-boba/analysis"), "bias", ""),
        "drift": (Path("output-boba/analysis"), "drift", ""),
        "ar1": (Path("output-boba/analysis"), "ar1", ""),
        "slip": (Path("output-boba-slip/analysis"), "slip", ""),
        "slip, logged": (Path("output-boba-slip-actual/analysis"), "slip", "rec-actual"),
        "misclick": (Path("output-boba-misclick/analysis"), "misclick", ""),
    }, args.out)
    table_fitted_companion(Path("output-fitted/analysis/fitted_vs_synthetic.csv"), args.out)
    # The SENSITIVITY run, not the preregistered one: the preregistered run is
    # void on a control failure, so it has no per-claim verdict. Section text
    # and caption both say so.
    table_confirmatory(Path("output-boba-confirmatory/analysis-sensitivity"), args.out)
    table_descriptor_regression(args.analysis, args.out)
    # Written by analyse_boba_inputerror.py; each skips with a message until then,
    # so no partial input-error numbers can reach paper/tables.
    table_inputerror_dose(Path("output-boba-slip/analysis"), args.out)
    table_inputerror_mechanism(Path("output-boba-slip/analysis"), args.out)
    table_inputerror_deployed(Path("output-boba-slip/analysis"), args.out)
    table_inputerror_main(Path("output-boba-slip/analysis"), args.out)


if __name__ == "__main__":
    main()
