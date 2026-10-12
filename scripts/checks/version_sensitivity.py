"""Do the headline numbers depend on the library versions?

The main sweep ran on the environment that requirements-eval.txt records (torch
2.13.0, pandas 3.0.5, scikit-learn 1.9.0, BoTorch 0.18.2.dev23, ...). Newer
releases change the floating-point path of the GP fit and the acquisition
optimiser, so a rerun is not bit-identical (scripts/check_provenance.py run under
torch 2.14.1 differs from trial 8 of a branin run by at most 1e-6, and later
designs can diverge). This check asks whether the conclusions move.

It reruns two cells of the main sweep with whatever interpreter runs it -- for
the paper, a separate venv with torch 2.14.1, pandas 3.0.6, scikit-learn 1.9.1,
matplotlib 3.11.2 and pyarrow 26.0.0 over the published environment -- using
the settings recorded in output-boba/run_metadata.json, and compares the logged
and rerun runs on exactly the same tasks:

  * the headline cell, gaussian error at 1 sigma from the first rating, and
  * the stress cell, gaussian error at 5 sigma from trial 21,

each for LogEI and qNEI on the twenty landscapes and seeds 7-16 (400 noisy runs
and 400 clean twins; a clean twin is shared by the two cells of its landscape,
acquisition and seed). `run --seeds` reruns a subset; the paper's check reran
seeds 7-11 and then 12-16 into the same directory. Per cell it gives the deployed
excess, the final-trial search excess and the selection share as ratios of
landscape means (fractions of opt_z), for the logged and the rerun runs, and the
paired difference with a landscape bootstrap.

    <venv>/Scripts/python scripts/checks/version_sensitivity.py run --out-dir output-version-check --workers 12
    <venv>/Scripts/python scripts/checks/version_sensitivity.py run --out-dir output-version-check --workers 16 --seeds 12,13,14,15,16
    <venv>/Scripts/python scripts/checks/version_sensitivity.py evaluate --out-dir output-version-check
    python scripts/checks/version_sensitivity.py compare --out-dir output-version-check

``compare`` prints the report the register file keeps
(output-boba/analysis/review/register_checks/version_sensitivity.txt) and writes
version_sensitivity.csv beside it. The run logs stay in --out-dir.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MAIN = REPO / "output-boba"
ACQS = ["logei", "qnei"]
SEEDS = [7, 8, 9, 10, 11]
ALL_SEEDS = list(range(7, 17))
CELLS = [("headline: gaussian, 1 sigma from the first rating", 1.0, 0),
         ("stress: gaussian, 5 sigma from trial 21", 5.0, 20)]
PACKAGES = ["torch", "botorch", "gpytorch", "linear_operator", "numpy", "scipy", "pandas",
            "scikit-learn", "statsmodels", "matplotlib", "pyarrow"]
REGISTER = REPO / "output-boba" / "analysis" / "review" / "register_checks"
BOOT = 2000
SEED = 20261009


def versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "absent"
    try:
        import torch
        out["torch"] = torch.__version__
    except ImportError:
        pass
    return out


def _task(job: tuple[str, int, str]) -> str:
    """One landscape and seed: both cells, both acquisitions, with the clean twins."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    function, seed, out_dir = job
    import bo_synthetic_error_simulation as synth
    import check_provenance as cp

    meta = json.loads((MAIN / "run_metadata.json").read_text(encoding="utf-8"))
    parser, defaults = cp._capture_parser(lambda: synth.parse_args([]))
    ns = cp.namespace_from_metadata(parser, defaults, meta.get("args") or {})
    ns.baseline_run = True
    ns.resume = True          # the second cell reuses the clean twins the first one wrote
    stats = synth.bb.load_stats(REPO / "boba_landscape_stats.json")
    target = Path(out_dir) / function
    target.mkdir(parents=True, exist_ok=True)
    for _, std, onset in CELLS:
        synth.run_task(synth.Task(function=function, seed=seed), ns, stats[function],
                       ACQS, ["gaussian"], [std], [onset], target, None)
    return f"{function} seed {seed}"


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((MAIN / "run_metadata.json").read_text(encoding="utf-8"))
    functions = list(meta["functions"])
    (out / "versions.json").write_text(json.dumps(versions(), indent=1), encoding="utf-8")
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(f, s, str(out)) for f in functions for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, done in enumerate(pool.map(_task, jobs), 1):
            print(f"[{i}/{len(jobs)}] {done}", flush=True)


def cmd_evaluate(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        subprocess.run([sys.executable, str(SCRIPTS / "evaluate_research_question.py"),
                        "--input-dir", str(d), "--output-dir", str(d / "evaluation")],
                       check=True, stdout=subprocess.DEVNULL)
        print("evaluated", d.name, flush=True)


def _cell_frame(root: Path, std: float, onset: int, seeds: list[int] | None = None) -> pd.DataFrame:
    seeds = SEEDS if seeds is None else seeds
    rows = []
    for path in sorted(root.glob("*/evaluation/paired_excess_metrics.csv")):
        d = pd.read_csv(path)
        d = d[(d["error_model"] == "gaussian") & np.isclose(d["jitter_std"], std)
              & (d["jitter_iteration"] == onset) & d["acquisition"].isin(ACQS) & d["seed"].isin(seeds)]
        rows.append(d)
    frame = pd.concat(rows, ignore_index=True)
    opt = frame["final_best_true_baseline"] + frame["final_simple_regret_true_baseline"]
    frame = frame.assign(
        deployed=frame["final_inference_simple_regret_excess_true"] / opt,
        search=frame["final_simple_regret_excess_true"] / opt,
    )
    frame["selection"] = frame["deployed"] - frame["search"]
    return frame[["dataset", "acquisition", "seed", "deployed", "search", "selection"]]


def _summaries(per: pd.DataFrame) -> dict[str, float]:
    m = per.mean()
    return {"deployed": m["deployed"], "search": m["search"], "selection_share": m["selection"] / m["deployed"]}


def compare_cell(old: pd.DataFrame, new: pd.DataFrame, rng: np.random.Generator) -> dict:
    keys = ["dataset", "acquisition", "seed"]
    joined = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    if len(joined) != len(old) or len(joined) != len(new):
        raise SystemExit(f"pairing failed: {len(old)} logged, {len(new)} rerun, {len(joined)} paired")
    cols = ["deployed", "search", "selection"]
    per_old = joined.groupby("dataset")[[c + "_old" for c in cols]].mean()
    per_new = joined.groupby("dataset")[[c + "_new" for c in cols]].mean()
    per_old.columns = cols
    per_new.columns = cols
    point_old, point_new = _summaries(per_old), _summaries(per_new)
    n = len(per_old)
    draws = {k: [] for k in point_old}
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        a, b = _summaries(per_old.iloc[idx]), _summaries(per_new.iloc[idx])
        for k in draws:
            draws[k].append(b[k] - a[k])
    out = {"runs": len(joined), "landscapes": n,
           "identical_runs": int((np.isclose(joined["deployed_old"], joined["deployed_new"], rtol=0, atol=1e-12)
                                  & np.isclose(joined["search_old"], joined["search_new"], rtol=0, atol=1e-12)).sum()),
           "spearman_deployed_per_landscape": float(per_old["deployed"].corr(per_new["deployed"], method="spearman")),
           "spearman_deployed_per_run": float(joined["deployed_old"].corr(joined["deployed_new"], method="spearman"))}
    for k in point_old:
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        out.update({f"{k}_logged": point_old[k], f"{k}_rerun": point_new[k],
                    f"{k}_diff": point_new[k] - point_old[k], f"{k}_diff_lo": lo, f"{k}_diff_hi": hi})
    return out


def cmd_compare(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    vers_new = json.loads((out / "versions.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    rows = []
    lines = ["Version sensitivity of the main sweep (scripts/checks/version_sensitivity.py)",
             f"rerun environment: {', '.join(f'{k} {v}' for k, v in vers_new.items())}",
             "logged environment: requirements-eval.txt (torch 2.13.0, pandas 3.0.5, scikit-learn 1.9.0, "
             "BoTorch 0.18.2.dev23, numpy 2.5.3, scipy 1.18.1)",
             "LogEI and qNEI, the seeds rerun (see the run counts), twenty landscapes; fractions of opt_z at the final trial;",
             f"differences rerun minus logged with {BOOT} landscape-bootstrap resamples", ""]
    for label, std, onset in CELLS:
        new = _cell_frame(out, std, onset, ALL_SEEDS)
        seeds = sorted(int(s) for s in new["seed"].unique())
        old = _cell_frame(MAIN, std, onset, seeds)
        r = compare_cell(old, new, rng)
        r["cell"] = label
        rows.append(r)
        lines.append(f"{label}: {r['runs']} paired runs, {r['identical_runs']} identical to 1e-12")
        for k, name in (("deployed", "deployed excess"), ("search", "search excess"),
                        ("selection_share", "selection share")):
            lines.append(f"  {name:16s} logged {r[k + '_logged']:.4f}  rerun {r[k + '_rerun']:.4f}  "
                         f"difference {r[k + '_diff']:+.4f} [{r[k + '_diff_lo']:+.4f}, {r[k + '_diff_hi']:+.4f}]")
        lines.append(f"  Spearman of the deployed excess, per landscape {r['spearman_deployed_per_landscape']:.3f}, "
                     f"per run {r['spearman_deployed_per_run']:.3f}")
        lines.append("")
    REGISTER.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REGISTER.parent / "version_sensitivity.csv", index=False)
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--seeds", default="7,8,9,10,11", help="comma-separated seeds to rerun")
    e = sub.add_parser("evaluate")
    e.add_argument("--out-dir", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    {"run": cmd_run, "evaluate": cmd_evaluate, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
