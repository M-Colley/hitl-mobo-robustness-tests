"""Test the preregistered hypotheses on the fresh confirmatory seeds.

Written BEFORE the confirmatory seeds existed, for the same reason
``output-boba-confirmatory/HYPOTHESIS.md`` was: a test chosen after seeing the
data is not a test. Every threshold, every statistic and every decision rule
here is transcribed from that file, and the script refuses to run on any seed
that appears in the screening sweep.

It reports the screening and confirmatory estimates side by side for each claim
and prints a verdict of CONFIRMED, NOT CONFIRMED or VOID. Void is reserved for
the control failures: if the model-free floor is not exactly zero, or the panel
is unbalanced, or a benchmark has no headroom, the run says nothing either way
and must not be read as a negative result.

  python scripts/test_confirmatory_hypotheses.py \\
    --screening output-boba --confirmatory output-boba-confirmatory
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import boba_benchmarks as bb  # noqa: E402
from analyse_boba_robustness import (  # noqa: E402
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    MODEL_FREE,
    PRIMARY_DESCRIPTORS,
    _cluster_bootstrap,
    attach_landscape,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCREENING_SEEDS = set(range(7, 27))     # 7-16 screening, 17-26 reserved
ROBUST_FAMILY = ("qucb", "qnei", "ucb")
FRAGILE_FAMILY = ("pi", "logpi", "qpi")


def load(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "*" / "evaluation" / "paired_excess_metrics.csv")))
    if not files:
        raise FileNotFoundError(f"No per-benchmark evaluation outputs under {root}.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Controls. A failure here voids the run rather than refuting anything.
# ---------------------------------------------------------------------------


def controls(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    floor = df[df["acquisition"].isin(MODEL_FREE)]
    worst = float(np.abs(floor["auc_simple_regret_excess_true"]).max()) if len(floor) else np.nan
    checks.append(("model-free floor exactly zero", bool(len(floor)) and worst == 0.0,
                   f"max |excess| = {worst:.3e}"))

    counts = (df.groupby(["dataset", "acquisition", "error_model", "jitter_std",
                          "jitter_iteration"])["seed"].nunique())
    balanced = counts.nunique() == 1
    checks.append(("seed panel balanced", bool(balanced),
                   f"seeds per cell: {sorted(counts.unique().tolist())}"))

    base = (df.groupby(["dataset", "acquisition"])["auc_simple_regret_true_baseline"]
            .mean().reset_index())
    headrooms = {}
    for name, block in base.groupby("dataset"):
        fl = block[block["acquisition"].isin(MODEL_FREE)]["auc_simple_regret_true_baseline"]
        le = block[~block["acquisition"].isin(MODEL_FREE)]["auc_simple_regret_true_baseline"]
        if len(fl) and len(le) and fl.mean() > 0:
            headrooms[name] = 1.0 - le.min() / fl.mean()
    low = {k: round(v, 3) for k, v in headrooms.items() if v < 0.10}
    checks.append(("no benchmark below 0.10 headroom", not low, f"below threshold: {low or 'none'}"))

    failures = int(df["acq_opt_failures"].sum())
    checks.append(("zero acquisition fallbacks", failures == 0, f"{failures} fallbacks"))
    return checks


# ---------------------------------------------------------------------------
# The hypotheses, exactly as preregistered
# ---------------------------------------------------------------------------


def h1_mediation(df: pd.DataFrame) -> dict:
    """H1a beta(frag) > 0; H1b beta(log sigma_e) interval contains 0; H1c dR2 >= 0.10."""
    learners = df[~df["acquisition"].isin(MODEL_FREE)].copy()
    cells = (
        learners.dropna(subset=["frag_at_c"])
        .groupby(["dataset", "error_model", "jitter_std", "jitter_iteration"])
        .agg(excess_sd=("excess_sd", "mean"), frag_at_c=("frag_at_c", "first"),
             log_noise=("log_noise", "first"),
             **{name: (name, "first") for name in PRIMARY_DESCRIPTORS})
        .reset_index()
    )
    out = {}
    for label, terms in (("descriptors", ["log_noise"] + PRIMARY_DESCRIPTORS),
                         ("both", ["frag_at_c", "log_noise"] + PRIMARY_DESCRIPTORS)):
        design = cells[terms].astype(float)
        design = (design - design.mean()) / design.std(ddof=0).replace(0.0, 1.0)
        design = design.assign(dataset=cells["dataset"].to_numpy(),
                               excess_sd=cells["excess_sd"].to_numpy())
        fit = _cluster_bootstrap(design, "excess_sd", terms, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        out[label] = fit.set_index("term")

    frag = out["both"].loc["frag_at_c"]
    noise = out["both"].loc["log_noise"]
    delta_r2 = float(out["both"]["r_squared"].iloc[0] - out["descriptors"]["r_squared"].iloc[0])
    h1a = bool(frag["ci_low"] > 0)
    h1b = bool(noise["ci_low"] <= 0 <= noise["ci_high"])
    h1c = bool(delta_r2 >= 0.10)
    return {
        "beta_frag": float(frag["coefficient"]),
        "beta_frag_ci": [float(frag["ci_low"]), float(frag["ci_high"])],
        "beta_noise": float(noise["coefficient"]),
        "beta_noise_ci": [float(noise["ci_low"]), float(noise["ci_high"])],
        "r2_descriptors": float(out["descriptors"]["r_squared"].iloc[0]),
        "r2_both": float(out["both"]["r_squared"].iloc[0]),
        "delta_r2": delta_r2,
        "H1a": h1a, "H1b": h1b, "H1c": h1c,
        "confirmed": h1a and h1b and h1c,
    }


def h2_onset(df: pd.DataFrame) -> dict:
    """Ratio >= 3 at 1 sigma, and a two-sided Wilcoxon over the 20 benchmarks."""
    learners = df[(~df["acquisition"].isin(MODEL_FREE)) & (df["jitter_std"] == 1.0)]
    wide = (learners.groupby(["dataset", "jitter_iteration"])["fragility"].mean()
            .unstack("jitter_iteration"))
    early, late = wide[0], wide[20]
    ratio = float(early.mean() / late.mean())
    _, p = wilcoxon(early, late)
    return {"onset0": float(early.mean()), "onset20": float(late.mean()), "ratio": ratio,
            "p": float(p), "n_benchmarks": int(len(wide)),
            "confirmed": bool(ratio >= 3.0 and p < 0.05)}


def h3_families(df: pd.DataFrame) -> dict:
    """H3a robust family ranks lower; H3b Kendall's W < 0.40. BH over the two."""
    learners = df[~df["acquisition"].isin(MODEL_FREE)]
    cell = (learners.groupby(["error_model", "jitter_std", "jitter_iteration",
                              "dataset", "acquisition"])["fragility"].mean().reset_index())
    ranks, ws = [], []
    for _, block in cell.groupby(["error_model", "jitter_std", "jitter_iteration"]):
        wide = block.pivot_table(index="dataset", columns="acquisition",
                                 values="fragility").dropna()
        if wide.shape[0] < 3 or wide.shape[1] < 2:
            continue
        r = wide.rank(axis=1, method="average")
        ranks.append(r.mean(axis=0))
        from scipy.stats import friedmanchisquare
        stat, _ = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
        n, k = wide.shape
        ws.append(stat / (n * (k - 1)))
    rank_table = pd.DataFrame(ranks)
    robust = rank_table[[c for c in ROBUST_FAMILY if c in rank_table]].mean(axis=1)
    fragile = rank_table[[c for c in FRAGILE_FAMILY if c in rank_table]].mean(axis=1)
    _, p_a = wilcoxon(robust, fragile)
    kendall_w = float(np.mean(ws))
    return {"robust_mean_rank": float(robust.mean()), "fragile_mean_rank": float(fragile.mean()),
            "p_raw": float(p_a), "kendall_w": kendall_w,
            "H3a_direction": bool(robust.mean() < fragile.mean()),
            "H3b": bool(kendall_w < 0.40)}


def h4_monotone(df: pd.DataFrame) -> dict:
    learners = df[~df["acquisition"].isin(MODEL_FREE)]
    grid = learners.pivot_table(index="jitter_iteration", columns="jitter_std",
                                values="fragility")
    ok = all(bool(np.all(np.diff(row.to_numpy()) > 0)) for _, row in grid.iterrows())
    return {"grid": grid.round(4).to_dict(), "confirmed": ok}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screening", type=Path, default=Path("output-boba"))
    parser.add_argument("--confirmatory", type=Path, default=Path("output-boba-confirmatory"))
    parser.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--exclude-benchmarks",
        type=str,
        default="",
        help="Comma-separated benchmarks to drop from BOTH arms before testing. "
        "This is a POST-HOC DEVIATION from the preregistration and the output "
        "says so: use it only to report a sensitivity analysis ALONGSIDE the "
        "preregistered verdict, never in place of it. The preregistration voids "
        "the run if any benchmark falls below 0.10 headroom, and voiding is not "
        "something an exclusion can undo after the fact.",
    )
    args = parser.parse_args(argv)
    excluded = [b.strip() for b in args.exclude_benchmarks.split(",") if b.strip()]
    output_dir = args.output_dir or (args.confirmatory / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    prereg = args.confirmatory / "HYPOTHESIS.md"
    if not prereg.exists():
        raise SystemExit(f"No preregistration at {prereg}. Refusing to test hypotheses that "
                         f"were never written down.")

    stats = bb.load_stats(args.stats_path)
    conf = attach_landscape(load(args.confirmatory), stats)
    overlap = sorted(set(conf["seed"].unique()) & SCREENING_SEEDS)
    if overlap:
        raise SystemExit(
            f"Confirmatory run contains screening seeds {overlap}. That is double-dipping; "
            f"the confirmatory arm must use seeds outside {min(SCREENING_SEEDS)}-"
            f"{max(SCREENING_SEEDS)}."
        )
    screen = attach_landscape(load(args.screening), stats)
    screen = screen[screen["error_model"] == "gaussian"]
    if excluded:
        missing = [b for b in excluded if b not in set(conf["dataset"])]
        if missing:
            raise SystemExit(f"--exclude-benchmarks names unknown benchmarks: {missing}")
        conf = conf[~conf["dataset"].isin(excluded)]
        screen = screen[~screen["dataset"].isin(excluded)]
        print("*" * 72)
        print("POST-HOC SENSITIVITY ANALYSIS -- NOT THE PREREGISTERED TEST.")
        print(f"Excluded from both arms: {excluded}")
        print("The preregistration voids the run on a control failure; an exclusion")
        print("applied after seeing the data cannot un-void it. Read the verdict")
        print("below as 'what the data would have said', not as confirmation.")
        print("*" * 72)
    conf_g = conf[conf["error_model"] == "gaussian"]

    lines = ["CONFIRMATORY TEST OF THE PREREGISTERED HYPOTHESES", "=" * 72, "",
             f"Screening : {args.screening}  seeds {sorted(screen['seed'].unique())}",
             f"Confirmatory: {args.confirmatory}  seeds {sorted(conf['seed'].unique())}",
             f"Preregistration: {prereg}", "",
             "CONTROLS (a failure VOIDS the run; it is not a negative result)", "-" * 72]
    checks = controls(conf_g)
    for name, ok, detail in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name:<38s} {detail}")
    void = not all(ok for _, ok, _ in checks)
    lines.append("")

    results = {}
    for key, fn, title in (("H1", h1_mediation, "H1 mediation"),
                           ("H2", h2_onset, "H2 onset"),
                           ("H3", h3_families, "H3 acquisition families"),
                           ("H4", h4_monotone, "H4 dose-response monotone")):
        results[key] = {"confirmatory": fn(conf_g), "screening": fn(screen)}

    # H3's two parts share a correction, as preregistered.
    p_a = results["H3"]["confirmatory"]["p_raw"]
    corrected = multipletests([p_a, p_a], method="fdr_bh")[1][0]
    results["H3"]["confirmatory"]["p_fdr"] = float(corrected)
    results["H3"]["confirmatory"]["H3a"] = bool(
        results["H3"]["confirmatory"]["H3a_direction"] and corrected < 0.05)
    results["H3"]["confirmatory"]["confirmed"] = bool(
        results["H3"]["confirmatory"]["H3a"] and results["H3"]["confirmatory"]["H3b"])

    def verdict(key: str) -> str:
        if void:
            return "VOID"
        return "CONFIRMED" if results[key]["confirmatory"].get("confirmed") else "NOT CONFIRMED"

    c, s = results["H1"]["confirmatory"], results["H1"]["screening"]
    lines += ["H1  MEDIATION", "-" * 72,
              f"  {'':<22s} {'screening':>22s} {'confirmatory':>22s}",
              f"  {'beta(frag)':<22s} "
              f"{s['beta_frag']:>+8.3f} [{s['beta_frag_ci'][0]:+.2f},{s['beta_frag_ci'][1]:+.2f}] "
              f"{c['beta_frag']:>+8.3f} [{c['beta_frag_ci'][0]:+.2f},{c['beta_frag_ci'][1]:+.2f}]",
              f"  {'beta(log sigma_e)':<22s} "
              f"{s['beta_noise']:>+8.3f} [{s['beta_noise_ci'][0]:+.2f},{s['beta_noise_ci'][1]:+.2f}] "
              f"{c['beta_noise']:>+8.3f} [{c['beta_noise_ci'][0]:+.2f},{c['beta_noise_ci'][1]:+.2f}]",
              f"  {'delta R^2':<22s} {s['delta_r2']:>22.3f} {c['delta_r2']:>22.3f}",
              f"  H1a {c['H1a']}   H1b {c['H1b']}   H1c {c['H1c']}   -> {verdict('H1')}", ""]

    c, s = results["H2"]["confirmatory"], results["H2"]["screening"]
    lines += ["H2  ONSET", "-" * 72,
              f"  screening   onset0 {s['onset0']:.4f}  onset20 {s['onset20']:.4f}  "
              f"ratio {s['ratio']:.2f}  p {s['p']:.3g}",
              f"  confirmatory onset0 {c['onset0']:.4f}  onset20 {c['onset20']:.4f}  "
              f"ratio {c['ratio']:.2f}  p {c['p']:.3g}",
              f"  -> {verdict('H2')}", ""]

    c, s = results["H3"]["confirmatory"], results["H3"]["screening"]
    lines += ["H3  ACQUISITION FAMILIES", "-" * 72,
              f"  screening   robust {s['robust_mean_rank']:.2f} vs fragile "
              f"{s['fragile_mean_rank']:.2f}   W {s['kendall_w']:.3f}",
              f"  confirmatory robust {c['robust_mean_rank']:.2f} vs fragile "
              f"{c['fragile_mean_rank']:.2f}   W {c['kendall_w']:.3f}  p(FDR) {c['p_fdr']:.3g}",
              f"  H3a {c['H3a']}   H3b {c['H3b']}   -> {verdict('H3')}", ""]

    lines += ["H4  DOSE-RESPONSE MONOTONE", "-" * 72,
              f"  screening {results['H4']['screening']['confirmed']}   "
              f"confirmatory {results['H4']['confirmatory']['confirmed']}   -> {verdict('H4')}",
              ""]

    if void:
        lines += ["OVERALL: VOID. A control failed, so the confirmatory run says nothing "
                  "either way.", ""]
    else:
        n_ok = sum(1 for k in results if results[k]["confirmatory"].get("confirmed"))
        lines += [f"OVERALL: {n_ok} of 4 preregistered hypotheses confirmed.", ""]

    report = "\n".join(lines)
    (output_dir / "confirmatory_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "confirmatory_results.json").write_text(
        json.dumps({"void": void, "controls": [(n, ok, d) for n, ok, d in checks],
                    "results": results}, indent=2, default=float), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
