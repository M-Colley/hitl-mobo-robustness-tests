"""Does a fitted oracle understate the cost of feedback error, holding the landscape fixed?

The paper motivates its analytic design by the claim that the usual fitted-oracle
design understates the cost of feedback error. Its own companion arm cannot
settle that, because the three archival datasets are different objectives from
the twenty analytic landscapes: a gap between the two arms is as consistent with
"the oracle understates" as with "the analytic suite is harder". This experiment
removes the confound by holding the landscape fixed and changing only the oracle.

For each analytic landscape it builds a synthetic ARCHIVAL dataset the way an
archival HITL-BO study arises: 37 "participants" (eHMI's count), each the first
20 designs of a real BO trajectory on that landscape (clean main-sweep runs,
so the designs cluster where BO goes, as archival designs do), rated once with
gaussian noise at the archival scale (1.0 sigma_f, the median of the measured
anchors). It then runs the fitted-oracle pipeline on that dataset exactly as the
companion arm ran it on the real data: oracle family chosen by the same grouped
cross-validation over the same seven model families, the oracle fitted with the
same augmentation, the box taken from the data, the optimum estimated by random
search, the error grid scaled to the FITTED oracle's own sigma_f, and the same
fraction-of-the-floor-gap metric. The exact pipeline on the same landscape, same
acquisitions, seeds, magnitudes and onset, is read from the main sweep.

If the fitted fraction is well below the exact fraction on the same landscape,
the fitted design understates the cost. If the two agree, the threefold gap
between the paper's two arms is a difference between landscapes, not oracles.

    python scripts/oracle_isolation.py make-data
    python scripts/oracle_isolation.py select
    python scripts/oracle_isolation.py calibrate
    (run run_oracle_isolation.ps1)
    python scripts/oracle_isolation.py analyse
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import boba_benchmarks as bb  # noqa: E402

ROOT = Path("output-oracle-iso")
N_PARTICIPANTS = 37          # eHMI's participant count
TRIALS_PER_PARTICIPANT = 20  # eHMI's trials per participant
ARCHIVAL_NOISE = 1.0         # in sigma_f; the measured anchors run 0.74 to 3.35, median about 1
ACQS = ("logei", "qnei", "ucb", "ei")
FLOORS = ("random", "sobol")
SEEDS = (7, 8, 9, 10, 11)
SIGMA_GRID = (0.25, 1.0, 5.0)
MODELS = "xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting"
DATA_SEED = 20260922
RESPONSE = "auc_simple_regret_excess_true_postonset_per_iter"
BASELINE_BEST = "final_best_true_baseline"
BOOT = 2000


def landscapes() -> list[str]:
    main = Path("output-boba")
    return sorted(p.name for p in main.iterdir() if p.is_dir() and p.name in bb.BENCHMARKS)


# ---------------------------------------------------------------------------
# 1. the synthetic archival datasets
# ---------------------------------------------------------------------------


def make_data(args) -> None:
    rng = np.random.default_rng(DATA_SEED)
    (ROOT / "configs").mkdir(parents=True, exist_ok=True)
    for name in landscapes():
        spec = bb.BENCHMARKS[name]
        clean = sorted((Path("output-boba") / name).glob(f"bo_sensor_error_{name}_value_*_baseline_exact.csv"))
        clean = [p for p in clean if not any(f"_{m}_seed" in p.name for m in FLOORS)]
        if len(clean) < N_PARTICIPANTS:
            raise SystemExit(f"{name}: only {len(clean)} clean trajectories for {N_PARTICIPANTS} participants")
        order = rng.permutation(len(clean))[:N_PARTICIPANTS]
        out = ROOT / "data" / name
        cols = spec.param_columns
        for i, idx in enumerate(order, start=1):
            run = pd.read_csv(clean[idx]).head(TRIALS_PER_PARTICIPANT)
            X = run[cols].to_numpy(dtype=float)
            truth = run["objective_true"].to_numpy(dtype=float)  # standardised: sigma_f(true) = 1
            rating = truth + rng.normal(0.0, ARCHIVAL_NOISE, size=len(truth))
            frame = pd.DataFrame(X, columns=cols)
            frame.insert(0, "value", rating)
            frame.insert(0, "Phase", ["sampling" if t < 5 else "optimization" for t in range(len(frame))])
            frame.insert(0, "Run", np.arange(1, len(frame) + 1))
            frame.insert(0, "User_ID", i)
            (out / f"u_{i}").mkdir(parents=True, exist_ok=True)
            frame.to_csv(out / f"u_{i}" / "ObservationsPerEvaluation.csv", sep=";", index=False)
        config = [{
            "name": f"iso_{name}",
            "data_dir": str(out.resolve()),
            "oracle_target": "individual",
            "param_columns": cols,
            "objective_map": {"composite": ["value"]},
        }]
        (ROOT / "configs" / f"datasets-iso_{name}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"{name:<22s} d={len(cols):2d}  {N_PARTICIPANTS} participants x {TRIALS_PER_PARTICIPANT} ratings")


# ---------------------------------------------------------------------------
# 2. the same oracle selection the companion arm used
# ---------------------------------------------------------------------------


def select(args) -> None:
    (ROOT / "selection").mkdir(parents=True, exist_ok=True)
    for name in landscapes():
        out = ROOT / "selection" / f"iso_{name}.json"
        if out.is_file() and not args.force:
            continue
        cmd = [sys.executable, str(SCRIPT_DIR / "select_best_oracle_model.py"),
               "--dataset-config", str(ROOT / "configs" / f"datasets-iso_{name}.json"),
               "--objective", "composite", "--oracle-models", MODELS,
               "--cv-folds", "5", "--output-path", str(out)]
        print(f"[select] {name}", flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 3. sigma_f of the fitted oracle, exactly as the anchor computes it
# ---------------------------------------------------------------------------


def calibrate(args) -> None:
    import bo_sensor_error_simulation as sim
    rows = []
    for name in landscapes():
        cfg = ROOT / "configs" / f"datasets-iso_{name}.json"
        selection = sim.load_oracle_selection(ROOT / "selection" / f"iso_{name}.json")
        dataset = sim.parse_dataset_configs(None, cfg, Path(".dataset_cache"))[0]
        model = selection[(dataset.name, "composite")]["best_model"]
        r2 = selection[(dataset.name, "composite")].get("score", np.nan)
        frame = sim.load_observations(dataset, "composite", None, None)
        oracle = sim.build_oracle(
            df=frame, objective="composite", objective_columns=["value"],
            param_columns=dataset.param_columns, seed=7, normalize=False, weights=None,
            oracle_model=model, oracle_augmentation="jitter", oracle_augment_repeats=2,
            oracle_augment_std=0.02, oracle_fast=False, oracle_target=dataset.oracle_target)
        bounds = sim.bounds_from_data(frame, dataset.param_columns)
        X = np.random.default_rng(7).uniform(bounds.low, bounds.high, size=(20_000, len(dataset.param_columns)))
        y = oracle.predict_many(X).reshape(-1)
        sigma_f = float(np.std(y, ddof=1))
        # What the fitted oracle says against what the landscape is: its values
        # at fresh points in the data box, compared with the true function there.
        truth = bb.evaluate(name, X)
        stats = bb.load_stats()[name]
        truth = (truth - stats["mean"]) / stats["std"]
        corr = float(np.corrcoef(y, truth)[0, 1])
        rows.append({"landscape": name, "dataset": dataset.name, "oracle_model": model,
                     "cv_r2": r2, "sigma_f_fitted": sigma_f, "sigma_f_true": 1.0,
                     "corr_oracle_truth": corr,
                     "jitter_stds": ",".join(f"{s * sigma_f:.6f}" for s in SIGMA_GRID)})
        print(f"{name:<22s} {model:<22s} sigma_f(fitted)={sigma_f:.3f}  corr(oracle, truth)={corr:+.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "manifest.csv", index=False)
    print(f"\nwrote {ROOT / 'manifest.csv'}")


# ---------------------------------------------------------------------------
# 4. the comparison
# ---------------------------------------------------------------------------


def _paired(directory: Path) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in directory.glob("*/evaluation/paired_excess_metrics.csv")]
    if not parts:
        parts = [pd.read_csv(p) for p in directory.glob("evaluation/paired_excess_metrics.csv")]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _fraction(frame: pd.DataFrame, optimum: float) -> pd.DataFrame:
    floor = frame[frame.acquisition.isin(FLOORS)][BASELINE_BEST].mean()
    learners = frame[frame.acquisition.isin(ACQS)].copy()
    learners["frac"] = learners[RESPONSE] / (optimum - floor)
    return learners


def analyse(args) -> None:
    manifest = pd.read_csv(ROOT / "manifest.csv").set_index("landscape")
    stats = bb.load_stats()
    rows = []
    for name in manifest.index:
        fitted = _paired(ROOT / "runs" / f"iso_{name}")
        if fitted.empty:
            print(f"  {name}: no fitted runs yet")
            continue
        fitted = fitted[(fitted.jitter_iteration == 0) & fitted.seed.isin(SEEDS)]
        sigma_f = float(manifest.loc[name, "sigma_f_fitted"])
        fitted["sigma_multiple"] = (fitted["jitter_std"] / sigma_f).round(2)
        # The fitted design's own optimum: the best any arm reached.
        opt_fit = float(fitted[BASELINE_BEST].max())
        f_fit = _fraction(fitted, opt_fit)

        exact = _paired(Path("output-boba") / name)
        exact = exact[(exact.error_model.isin(["gaussian", "none"]) | exact.acquisition.isin(FLOORS))
                      & (exact.jitter_iteration == 0) & exact.seed.isin(SEEDS)
                      & exact.jitter_std.round(2).isin(SIGMA_GRID)
                      & (exact.error_model == "gaussian")]
        exact = exact.assign(sigma_multiple=exact["jitter_std"].round(2))
        f_exact = _fraction(exact, float(stats[name]["opt_z"]))

        for s in SIGMA_GRID:
            a = f_fit[f_fit.sigma_multiple == s]["frac"]
            b = f_exact[f_exact.sigma_multiple == s]["frac"]
            if len(a) and len(b):
                rows.append({"landscape": name, "sigma_multiple": s,
                             "frac_fitted": float(a.mean()), "frac_exact": float(b.mean()),
                             "n_fitted": len(a), "n_exact": len(b),
                             "oracle_model": manifest.loc[name, "oracle_model"],
                             "corr_oracle_truth": manifest.loc[name, "corr_oracle_truth"]})
    per = pd.DataFrame(rows)
    per.to_csv(ROOT / "oracle_isolation_per_landscape.csv", index=False)
    if per.empty:
        raise SystemExit("nothing to compare yet")

    rng = np.random.default_rng(DATA_SEED)
    summary = []
    for s, block in per.groupby("sigma_multiple"):
        fit, ex = block["frac_fitted"].to_numpy(), block["frac_exact"].to_numpy()
        n = len(block)
        idx = rng.integers(0, n, (BOOT, n))
        ratio = fit[idx].mean(axis=1) / ex[idx].mean(axis=1)
        diff = fit[idx].mean(axis=1) - ex[idx].mean(axis=1)
        from scipy.stats import spearmanr, wilcoxon
        summary.append({
            "sigma_multiple": s, "n_landscapes": n,
            "frac_fitted": fit.mean(), "frac_exact": ex.mean(),
            "ratio_fitted_over_exact": fit.mean() / ex.mean(),
            "ratio_lo": float(np.percentile(ratio, 2.5)), "ratio_hi": float(np.percentile(ratio, 97.5)),
            "diff": fit.mean() - ex.mean(),
            "diff_lo": float(np.percentile(diff, 2.5)), "diff_hi": float(np.percentile(diff, 97.5)),
            "wilcoxon_p": float(wilcoxon(fit, ex).pvalue) if n >= 5 else np.nan,
            "spearman_per_landscape": float(spearmanr(fit, ex).statistic),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(ROOT / "oracle_isolation_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["make-data", "select", "calibrate", "analyse"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    {"make-data": make_data, "select": select, "calibrate": calibrate, "analyse": analyse}[args.step](args)


if __name__ == "__main__":
    main()
