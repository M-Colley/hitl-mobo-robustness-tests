"""Ship rules: how much of the deployed cost of error does a better ship rule recover?

Reads the per-run tables of scripts/rescore_ship_rules.py and scores each ship
rule with the process-change estimand of analyse_boba_adaptations.py, whose
paired_frame and summarise functions it imports, so the two cannot drift:

    cost       = std_noisy  - std_clean    what the error costs the standard process
    gain       = std_noisy  - rule_noisy   how much better the rule does under error
    price      = rule_clean - std_clean    what the rule costs when there is no error
    recovered  = gain / cost               share of the standard process's cost recovered

The STANDARD is the best_observed rule (ship the single best rating). Regret is
divided by the landscape's opt_z (boba_landscape_stats.json), so cost, gain and
price are fractions of the achievable improvement; aggregates are ratios of
landscape means, intervals resample landscapes, the test is a Wilcoxon over the
per-landscape gains, BH-corrected within arm x rule x error process.

Two modes.

Within an arm (--arms): rule R against the same arm's best_observed, paired at
(landscape, acquisition, error model, magnitude, onset, seed), per error model
and variant, per magnitude x onset, and pooled over magnitudes >= 0.25 SD. The
0.05 SD cells are pooled separately: their cost is tiny, so a ratio over them is
mostly noise and would swamp a pooled share. The input-error arms (slip,
misclick) express magnitude as a fraction of the box, not in SDs; their pooling
threshold is --input-pool-min-std (default 0.05, which leaves out only the
smallest, 0.01-of-the-range cell).

Across arms (--cross): rule R of arm B against the best_observed standard of a
reference arm A, using the arm definitions of analyse_boba_adaptations.ARMS
(or --cross-spec). Pairing is on acquisition, or -- for the robust baselines
qKG and replication, which are different acquisitions -- on the mean over the
reference's ten standard acquisitions at (landscape, error model, magnitude,
onset, seed). Rule best_observed of arm B reproduces the adaptation analysis's
"deployed" number for that arm where the arm ships by best_observed.

    python scripts/analyse_ship_rules.py
    python scripts/analyse_ship_rules.py --arms output-boba --cross qkg,replei
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import analyse_boba_adaptations as aba  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
from rescore_ship_rules import MODEL_FREE, OUTPUT_NAME, RULES, parse_seeds  # noqa: E402

RESPONSE = "ship"
STANDARD = "best_observed"
TWIN_KEYS = ["dataset", "acquisition", "seed"]
# Input-error models: their magnitude is a fraction of the box, not an SD.
INPUT_ERROR_MODELS = ("slip", "misclick")
DEFAULT_ARMS = (
    "output-boba", "output-boba-incumbent", "output-boba-robust", "output-boba-knownnoise",
    "output-boba-slip", "output-boba-slip-actual", "output-boba-misclick", "output-boba-instrument",
    "output-boba-budget25", "output-boba-budget100", "output-boba-adapt-rerate", "output-boba-adapt-rep10",
)
SUMMARY_KEYS = ["n_landscapes", "n_cells", "cost", "gain", "price", "recovered", "recovered_lo",
                "recovered_hi", "wilcoxon_p"]


# ---------------------------------------------------------------------------
# Loading and pairing
# ---------------------------------------------------------------------------


def per_run_path(arm: str | Path, root: Path | None = None) -> Path:
    arm = Path(arm)
    return (root / arm.name / f"{OUTPUT_NAME}.csv") if root else arm / "analysis" / f"{OUTPUT_NAME}.csv"


def load_per_run(path: Path, acquisitions: set[str] | None = None, seeds: set[int] | None = None,
                 error_model: str | None = None) -> pd.DataFrame:
    """A per-run table, restricted; clean baselines survive the error-model filter."""
    df = pd.read_csv(path)
    missing = [c for c in ["dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed",
                           "baseline", "variant", "file"] + [f"regret_{r}" for r in RULES] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}; is it a rescore_ship_rules.py table?")
    df["variant"] = df["variant"].fillna("").astype(str)
    df["baseline"] = df["baseline"].astype(str).str.lower().isin(("true", "1"))
    keep = ~df["acquisition"].isin(MODEL_FREE)
    if acquisitions is not None:
        keep &= df["acquisition"].isin(acquisitions)
    if seeds is not None:
        keep &= df["seed"].isin(seeds)
    if error_model is not None:
        keep &= df["baseline"] | (df["error_model"] == error_model)
    return df[keep].reset_index(drop=True)


def twin_frame(runs: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Each noisy run with its clean twin, as paired_frame's jitter / baseline columns.

    The twin is the clean run of the same landscape, acquisition and seed in the
    same arm (one per arm, as in analyse_extra_runs); a missing or duplicated
    twin is an error, not a silent drop.
    """
    col = f"regret_{rule}"
    clean = runs[runs["baseline"]]
    dup = clean.duplicated(TWIN_KEYS, keep=False)
    if dup.any():
        raise ValueError(f"several clean runs for one twin key, e.g. {clean[dup]['file'].head(2).tolist()}")
    noisy = runs[~runs["baseline"]]
    if noisy.empty:
        raise ValueError("no noisy runs to pair")
    m = noisy.merge(clean[TWIN_KEYS + [col]].rename(columns={col: "ship_baseline"}), on=TWIN_KEYS,
                    how="left", validate="many_to_one", indicator=True)
    lost = m["_merge"] == "left_only"
    if lost.any():
        raise ValueError(f"{int(lost.sum())} noisy runs have no clean twin, e.g. {m.loc[lost, 'file'].head(2).tolist()}; "
                         "re-score the baselines too")
    undecided = m[col].isna() | m["ship_baseline"].isna()
    if undecided.any():
        raise ValueError(f"rule {rule}: {int(undecided.sum())} paired runs have no regret (GP refit could not "
                         f"rank designs), e.g. {m.loc[undecided, 'file'].head(2).tolist()}")
    m["jitter_iteration"] = m["jitter_iteration"].astype(int)
    return m.assign(ship_jitter=m[col])[aba.PAIR_KEYS + ["variant", "ship_jitter", "ship_baseline"]]


def pick_variant(runs: pd.DataFrame, variant: str, label: str) -> pd.DataFrame:
    """Restrict the noisy runs to one variant ("auto": the only one present)."""
    present = sorted(runs.loc[~runs["baseline"], "variant"].unique())
    if variant == "auto":
        if len(present) != 1:
            raise ValueError(f"{label}: noisy variants {present}; name one with variant= / ref_variant=")
        variant = present[0]
    elif variant not in present:
        raise ValueError(f"{label}: variant {variant!r} not among {present}")
    return runs[runs["baseline"] | (runs["variant"] == variant)]


def check_pool_complete(frame: pd.DataFrame, acquisitions: list[str], label: str) -> None:
    """A pooled mean must average the same acquisitions in every cell."""
    got = frame.groupby(aba.POOLED_KEYS)["acquisition"].nunique()
    short = got[got != len(acquisitions)]
    if len(short):
        raise ValueError(f"{label}: {len(short)} of {len(got)} pooled cells lack some of the "
                         f"{len(acquisitions)} acquisitions {acquisitions}; re-score them or narrow the list")


def landscape_opt_z(datasets) -> dict[str, float]:
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    lacking = sorted(set(datasets) - set(opt_z))
    if lacking:
        # paired_frame would silently use 1.0, mixing units across landscapes.
        raise KeyError(f"no opt_z for {lacking} in {bb.DEFAULT_STATS_PATH}")
    return opt_z


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _summary(block: pd.DataFrame) -> dict:
    # A fresh generator per summary, so a cell's interval does not depend on
    # which cells were summarised before it.
    return aba.summarise(block, np.random.default_rng(aba.BOOTSTRAP_SEED))


def cell_rows(paired: pd.DataFrame, pool_min_std: float) -> list[dict]:
    """Per magnitude x onset, then pooled above and below the magnitude threshold."""
    rows = []
    for (std, onset), block in paired.groupby(["jitter_std", "jitter_iteration"]):
        rows.append({"cell": "condition", "jitter_std": float(std), "jitter_iteration": float(onset),
                     **_summary(block)})
    if rows:
        for r, q in zip(rows, multipletests([r["wilcoxon_p"] for r in rows], method="fdr_bh")[1]):
            r["wilcoxon_p_fdr"] = float(q)
    big = paired["jitter_std"] >= pool_min_std - 1e-12
    for label, part in ((f"pooled_std_ge_{pool_min_std:g}", paired[big]),
                        (f"pooled_std_lt_{pool_min_std:g}", paired[~big])):
        if part.empty:
            continue
        for onset, block in part.groupby("jitter_iteration"):
            rows.append({"cell": label, "jitter_std": np.nan, "jitter_iteration": float(onset), **_summary(block)})
        # jitter_iteration NaN: both onsets together.
        rows.append({"cell": label, "jitter_std": np.nan, "jitter_iteration": np.nan, **_summary(part)})
    return rows


def pool_threshold(error_model: str, args: argparse.Namespace) -> float:
    return args.input_pool_min_std if error_model in INPUT_ERROR_MODELS else args.pool_min_std


def within_arm(arm: str, runs: pd.DataFrame, opt_z: dict[str, float], rules: list[str],
               args: argparse.Namespace) -> list[dict]:
    rows = []
    ref = twin_frame(runs, STANDARD)
    for rule in rules:
        if rule == STANDARD:
            continue
        trt = twin_frame(runs, rule)
        for (model, variant), ref_g in ref.groupby(["error_model", "variant"]):
            trt_g = trt[(trt["error_model"] == model) & (trt["variant"] == variant)]
            paired = aba.paired_frame(ref_g, trt_g, RESPONSE, opt_z, pool=False)
            for r in cell_rows(paired, pool_threshold(model, args)):
                rows.append({"mode": "within", "arm": arm, "reference": arm, "rule": rule, "error_model": model,
                             "variant": variant, "reference_variant": variant, "pooled_acquisitions": False, **r})
    return rows


def cross_specs(names: list[str], custom: list[str]) -> list[dict]:
    specs = []
    for name in names:
        s = aba.ARMS[name]
        if s["relative"]:
            print(f"cross {name}: fitted-oracle arm, not re-scorable here; skipped")
            continue
        specs.append({"name": name, "dir": s["dir"], "ref": s["ref"], "acqs": s["acqs"], "ref_acqs": s["ref_acqs"],
                      "seeds": s["seeds"], "pool": bool(s["pool"]), "error_model": s["error_model"],
                      "variant": "auto", "ref_variant": "auto"})
    for text in custom:
        kv = dict(part.split("=", 1) for part in text.split(";") if part.strip())
        need = [k for k in ("name", "dir", "ref", "acqs", "error_model") if k not in kv]
        if need:
            raise ValueError(f"--cross-spec {text!r} lacks {need}")
        specs.append({"name": kv["name"], "dir": kv["dir"], "ref": kv["ref"], "acqs": kv["acqs"],
                      "ref_acqs": kv.get("ref_acqs", kv["acqs"]), "seeds": kv.get("seeds", ""),
                      "pool": kv.get("pool", "0").lower() in ("1", "true", "yes"), "error_model": kv["error_model"],
                      "variant": kv.get("variant", "auto"), "ref_variant": kv.get("ref_variant", "auto")})
    return specs


def cross_arm(spec: dict, rules: list[str], args: argparse.Namespace) -> list[dict]:
    arm_path, ref_path = per_run_path(spec["dir"], args.per_run_root), per_run_path(spec["ref"], args.per_run_root)
    for p in (arm_path, ref_path):
        if not p.is_file():
            print(f"cross {spec['name']}: no per-run table {p}; skipped")
            return []
    seeds = parse_seeds(args.seeds or spec["seeds"])
    acqs, ref_acqs = spec["acqs"].split(","), spec["ref_acqs"].split(",")
    label = f"cross {spec['name']}"
    trt_runs = pick_variant(load_per_run(arm_path, set(acqs), seeds, spec["error_model"]), spec["variant"], label)
    ref_runs = pick_variant(load_per_run(ref_path, set(ref_acqs), seeds, spec["error_model"]),
                            spec["ref_variant"], label + " (reference)")
    opt_z = landscape_opt_z(set(trt_runs["dataset"]) | set(ref_runs["dataset"]))
    ref = twin_frame(ref_runs, STANDARD)
    if spec["pool"]:
        check_pool_complete(ref, ref_acqs, label + " (reference)")
    trt_variant = trt_runs.loc[~trt_runs["baseline"], "variant"].iloc[0]
    ref_variant = ref_runs.loc[~ref_runs["baseline"], "variant"].iloc[0]
    rows = []
    for rule in rules:
        trt = twin_frame(trt_runs, rule)
        if spec["pool"]:
            check_pool_complete(trt, acqs, label)
        paired = aba.paired_frame(ref, trt, RESPONSE, opt_z, spec["pool"])
        for r in cell_rows(paired, pool_threshold(spec["error_model"], args)):
            rows.append({"mode": "cross", "name": spec["name"], "arm": spec["dir"], "reference": spec["ref"],
                         "rule": rule, "error_model": spec["error_model"], "variant": trt_variant,
                         "reference_variant": ref_variant, "pooled_acquisitions": spec["pool"], **r})
    return rows


def arm_means(arm: str, runs: pd.DataFrame, opt_z: dict[str, float]) -> list[dict]:
    """Per arm, clean vs noisy: mean normalised regret per rule, and how often it ships the standard's design."""
    z = runs["dataset"].map(opt_z)
    rows = []
    for baseline, block in runs.groupby("baseline"):
        zb = z[block.index]
        for rule in RULES:
            row = {"arm": arm, "baseline": bool(baseline), "rule": rule, "n_runs": int(len(block)),
                   "mean_regret_over_opt_z": float((block[f"regret_{rule}"] / zb).mean()),
                   "logged_rule": ",".join(sorted(block["logged_rule"].astype(str).unique()))
                   if "logged_rule" in block else ""}
            if f"idx_{rule}" in block and "idx_best_observed" in block:
                row["share_same_design_as_best_observed"] = float(
                    (block[f"idx_{rule}"] == block["idx_best_observed"]).mean())
            if "gp_fit_ok" in block:
                row["gp_fit_failures"] = int((~block["gp_fit_ok"].astype(str).str.lower().isin(("true", "1"))).sum())
            rows.append(row)
    return rows


def _print_pooled(rows: list[dict], header: str) -> None:
    pooled = [r for r in rows if r["cell"] != "condition" and np.isnan(r["jitter_iteration"])]
    if not pooled:
        return
    print(header)
    for r in pooled:
        tag = f"{r['error_model']}{' (' + r['variant'] + ')' if r['variant'] else ''}"
        print(f"  {r['rule']:<13} {tag:<24} {r['cell']:<20} cost {r['cost']:.3f}  gain {r['gain']:+.3f}  "
              f"price {r['price']:+.3f}  recovered {r['recovered']:+6.1%} "
              f"[{r['recovered_lo']:+.0%}, {r['recovered_hi']:+.0%}]  landscapes {r['n_landscapes']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS),
                   help="arm directories for the within-arm analysis ('none' to skip)")
    non_relative = [n for n, s in aba.ARMS.items() if not s["relative"]]
    p.add_argument("--cross", type=str, default=",".join(non_relative),
                   help="analyse_boba_adaptations.ARMS entries for the cross-arm analysis ('none' to skip)")
    p.add_argument("--cross-spec", action="append", default=[],
                   help="a custom cross-arm pairing: 'name=N;dir=B;ref=A;acqs=a,b;ref_acqs=...;seeds=7-11;"
                        "pool=0|1;error_model=gaussian[;variant=..;ref_variant=..]'")
    p.add_argument("--per-run-root", type=Path, default=None,
                   help="read <root>/<arm name>/ship_rules_per_run.csv instead of <arm>/analysis/")
    p.add_argument("--seeds", type=str, default=None, help="restrict every table to these seeds (e.g. 7-11)")
    p.add_argument("--rules", type=str, default=",".join(RULES))
    p.add_argument("--pool-min-std", type=float, default=0.25,
                   help="response-error magnitudes pooled into the headline share (SDs)")
    p.add_argument("--input-pool-min-std", type=float, default=0.05,
                   help="the same threshold for the input-error arms (fraction of the box)")
    p.add_argument("--output-dir", type=Path, default=Path("output-boba/analysis"))
    args = p.parse_args(argv)
    unknown = [r for r in args.rules.split(",") if r and r not in RULES]
    if unknown:
        p.error(f"unknown rules {unknown}; choose from {list(RULES)}")
    bad = [n for n in args.cross.split(",") if n and n != "none" and n not in aba.ARMS]
    if bad:
        p.error(f"unknown --cross names {bad}; choose from {list(aba.ARMS)}")
    return args


def main(argv=None) -> dict[str, pd.DataFrame]:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = [r for r in args.rules.split(",") if r]
    seeds = parse_seeds(args.seeds)

    within, means = [], []
    arms = [] if args.arms.strip() == "none" else [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        path = per_run_path(arm, args.per_run_root)
        if not path.is_file():
            print(f"{arm}: no per-run table {path}; skipped")
            continue
        runs = load_per_run(path, seeds=seeds)
        opt_z = landscape_opt_z(runs["dataset"].unique())
        rows = within_arm(arm, runs, opt_z, rules, args)
        within.extend(rows)
        means.extend(arm_means(arm, runs, opt_z))
        _print_pooled(rows, f"\n=== {arm}: rule vs this arm's best_observed ({len(runs):,} runs) ===")

    cross = []
    names = [] if args.cross.strip() == "none" else [n for n in args.cross.split(",") if n]
    for spec in cross_specs(names, args.cross_spec):
        rows = cross_arm(spec, rules, args)
        cross.extend(rows)
        _print_pooled(rows, f"\n=== cross {spec['name']}: rule of {spec['dir']} vs best_observed of {spec['ref']}"
                            f"{' (mean over its acquisitions)' if spec['pool'] else ''} ===")

    out = {"ship_rules_recovery": pd.DataFrame(within), "ship_rules_cross_arm": pd.DataFrame(cross),
           "ship_rules_arm_means": pd.DataFrame(means)}
    for name, frame in out.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    print(f"\nWrote {', '.join(n + '.csv' for n in out)} to {args.output_dir}")
    return out


if __name__ == "__main__":
    main()
