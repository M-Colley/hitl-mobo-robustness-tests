# I5/D8: share of extra-trial zeros that are structural.
# Moved from the 2026-09-23 register triage; run all checks with
#   python scripts/run_review_checks.py
"""Share of extra-trial zeros that are structural (clean run reached its target
inside the prefix the noisy twin shares with it), for Table extra (budget-100)
and Table extrafull (main sweep). Reuses analyse_extra_runs' own functions."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import analyse_extra_runs as ae  # noqa: E402


def origins(input_dir: Path, k: int, tol: float, acqs=None, seeds=None):
    baselines, jittered = ae.index_runs(input_dir)
    opt_z = ae.load_opt_z(input_dir)
    rows = []
    stems = {r["stem"]: r for r in jittered if r["acquisition"] not in ae.MODEL_FREE}
    for stem, rec in stems.items():
        if acqs and rec["acquisition"] not in acqs:
            continue
        if seeds and rec["seed"] not in seeds:
            continue
        base = baselines.get(stem)
        if base is None:
            continue
        clean = ae.regret_curve(base)
        slack = tol * opt_z.get(rec["dataset"], 0.0)
        target = clean[k - 1] + 1e-12 + slack
        origin = int(np.flatnonzero(clean <= target)[0]) + 1
        rows.append({"stem": stem, "dataset": rec["dataset"], "acquisition": rec["acquisition"],
                     "seed": rec["seed"], "origin": origin, "T": len(clean)})
    return pd.DataFrame(rows)


def report(name, input_dir, k, tol, onset, per_run_csv, **kw):
    o = origins(input_dir, k, tol, **kw)
    s = max(onset + 1, 5)  # last trial the noisy run shares with its clean twin
    o["structural"] = o["origin"] <= s
    print(f"\n== {name}: k={k}, tol={tol}, onset t0={onset} (shared prefix through trial {s}) ==")
    print(f"clean runs: {len(o)}; structural (origin <= {s}): {o.structural.sum()} = {o.structural.mean():.1%}")
    pr = pd.read_csv(per_run_csv)
    pr = pr[(pr.k == k) & (pr.tolerance == tol) & (pr.jitter_iteration == onset)
            & (pr.variant.fillna("") == "")]
    if "error_model" in kw:
        pass
    pr["stem"] = pr.apply(lambda r: f"bo_sensor_error_{r.dataset}_value_{r.acquisition}_seed{r.seed}", axis=1)
    pr = pr.merge(o[["stem", "structural", "origin"]], on="stem", how="left")
    miss = pr.structural.isna().sum()
    if miss:
        print(f"  WARNING {miss} per-run rows without a matched clean run")
    for (model, mag), cell in pr.groupby(["error_model", "jitter_std"]):
        zero = cell.extra == 0
        nonstruct = cell[~cell.structural.astype(bool)]
        med_ns = nonstruct.extra.median() if len(nonstruct) else np.nan
        cens_ns = nonstruct.censored.mean() if len(nonstruct) else np.nan
        print(f"  {model:8s} {mag:5g}: n={len(cell):5d} median={cell.extra.median():5.1f} "
              f"zeros={zero.mean():5.1%} structural={cell.structural.mean():5.1%} "
              f"share of zeros structural={(zero & cell.structural.astype(bool)).sum() / max(zero.sum(), 1):5.1%} | "
              f"non-structural: n={len(nonstruct)} median={med_ns:5.1f} never={cens_ns:5.1%}")
        # multiplier check (A4): tau/k versus (k+extra)/k
    return o, pr


if __name__ == "__main__":
    b100 = REPO / "output-boba-budget100"
    o_late, pr_late = report("budget100 late onset (Table extra, right column)", b100, 50, 0.01, 40,
                             b100 / "analysis" / "extra_runs_per_run.csv")
    o_early, pr_early = report("budget100 early onset (Table extra, left column)", b100, 25, 0.01, 0,
                               b100 / "analysis" / "extra_runs_per_run.csv")
    # A4: the '2.6x' is (k+extra)/k with extra from origin; tau/k = (origin+extra)/k
    cell = pr_early[(pr_early.error_model == "gaussian") & (pr_early.jitter_std == 1.0)].copy()
    cell["tau"] = cell["origin"] + cell["extra"]
    print("\nA4 check, budget100, 1 sigma, onset 0, k=25, tol 0.01:")
    print(f"  median extra={cell.extra.median():.1f}; (k+median extra)/k={(25 + cell.extra.median()) / 25:.2f}; "
          f"median origin={cell.origin.median():.1f}; median tau={cell.tau.median():.1f}; "
          f"median tau/k={cell.tau.median() / 25:.2f}; median tau/origin={(cell.tau / cell.origin).median():.2f}; "
          f"censored={cell.censored.mean():.1%}")
    main = REPO / "output-boba"
    report("main sweep early onset (Table extrafull)", main, 10, 0.01, 0,
           main / "analysis" / "extra_runs_per_run.csv")
