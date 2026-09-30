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
same augmentation, the box taken from the data, the error grid scaled to the
FITTED oracle's own sigma_f, and the same fraction-of-the-floor-gap metric, on
the trajectory and on the deployed design. The exact pipeline on the same
landscape, same acquisitions, seeds, magnitudes and onset, is read from the main
sweep.

EVERY RUN SEED REFITS THE ORACLE (bo_sensor_error_simulation.py builds it with
seed=seed), so each landscape has five fitted oracles, one per seed 7-11, with
five optima, five floors and five sigma_f. The floor gap is therefore formed
within each seed's oracle, gap_s = optimum_s - floor_s (floor_s the mean clean
best of that seed's random and Sobol runs), and a landscape's fraction is the
ratio of seed means, mean_s(excess_s) / mean_s(gap_s). Three optima are reported:

  oracle      (primary) the logged y_opt of that seed's oracle, the simulator's
              estimate of its maximum (200,000 uniform points plus the training
              designs, fixed before any run). This is the analogue of the exact
              arm's opt_z: a property of the objective, not of the runs.
  best_clean  (sensitivity) the best value any clean run of that seed reached,
              the companion arm's estimator. It is attainable by construction,
              so it shrinks the gap and biases the fitted fraction upward.
  visited     (check) y_opt is a random-search estimate that a BO run can beat;
              this raises it to the best value any run of the seed visited
              (clean or noisy) where one did, the tightest lower bound on the
              oracle's supremum in the logs. The per-landscape file records how
              many seeds were raised and by how much.

y_opt is primary because the simulator measures every regret of a run, the
numerator's included, from it, as the exact arm measures them from opt_z.

The exact arm has one function, so its gap is opt_z less the floor mean pooled
over seeds, as before. Up to 2026-09-28 the fitted gap was the maximum over all
five oracles' clean bests less the floor pooled over all five, which mixes five
functions.

The magnitudes and the fidelity figure (corr_oracle_truth) come from the seed-7
oracle only (calibrate()); scripts/review_checks/oracle_sigmaf_seeds.py reports
how far the other seeds' sigma_f are from it.

If the fitted fraction is well below the exact fraction on the same landscape,
the fitted design understates the cost. If the two agree, the threefold gap
between the paper's two arms is a difference between landscapes, not oracles.

THE NORMALISER MATTERS. The floor gap depends on how hard the objective is for
random search as well as on its scale: it is 1-2% of opt_z for the exact
objective on Branin, Powell and Rosenbrock, and a same-budget random run leaves
more of the improvement unreached on a tree-ensemble oracle than on the exact
landscape. scripts/review_checks/oracle_achievable.py divides the same excess by
the paper's headline unit instead, the achievable improvement (opt_z; for the
fitted arm y_opt_s less seed s's oracle mean over the box, formed within each
seed), and scripts/review_checks/oracle_families.py prints both. The paper's
comparison is the achievable one; the floor gap is kept as tab:fitted's unit.
Each family also sets its own achievable improvement (a tree oracle's
y_opt - mean is a median 1.27 times sigma_f * opt_z, a GP's or MLP's about 1),
so oracle_achievable.py reports three alternatives beside it (each seed's best
clean value as the optimum; sigma_f,s * opt_z; both arms on their best clean
value), and a claim about the families should hold under all four
(the closing table of oracle_families.py).

    python scripts/oracle_isolation.py make-data
    python scripts/oracle_isolation.py select
    python scripts/oracle_isolation.py calibrate
    (run run_oracle_isolation.ps1)
    python scripts/oracle_isolation.py analyse

The pipeline's model selection picks tree ensembles on all twenty datasets. To
ask whether the result depends on that, --family forces one smooth oracle
family on the same synthetic datasets, with no selection and no jitter
augmentation (a tree-ensemble heuristic; a GP refitted on the tripled data every
run is also too slow), and writes to output-oracle-iso-<family>:

    python scripts/oracle_isolation.py select --family gaussian_process
    python scripts/oracle_isolation.py calibrate --family gaussian_process
    (run run_oracle_isolation.ps1 -Family gaussian_process)
    python scripts/oracle_isolation.py analyse --family gaussian_process
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import boba_benchmarks as bb  # noqa: E402

ROOT = Path("output-oracle-iso")
DATA_ROOT = ROOT  # the synthetic datasets and their configs, shared by every family
FAMILY: str | None = None
AUGMENTATION = "jitter"
N_PARTICIPANTS = 37          # eHMI's participant count
TRIALS_PER_PARTICIPANT = 20  # eHMI's trials per participant
ARCHIVAL_NOISE = 1.0         # in sigma_f; the measured anchors run 0.67 to 1.02 once a logging defect is fixed
ACQS = ("logei", "qnei", "ucb", "ei")
FLOORS = ("random", "sobol")
SEEDS = (7, 8, 9, 10, 11)
SIGMA_GRID = (0.25, 1.0, 5.0)
MODELS = "xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting"
DATA_SEED = 20260922
RESPONSE = "auc_simple_regret_excess_true_postonset_per_iter"
BASELINE_BEST = "final_best_true_baseline"
BASELINE_REGRET = "final_simple_regret_true_baseline"  # y_opt - BASELINE_BEST, so the two give y_opt
NOISY_BEST = "final_best_true_jitter"  # the best true value the noisy twin visited
BOOT = 2000
# How the fitted oracle's optimum is estimated, per seed; the first is primary.
# Each writes its own per-landscape column and summary file (suffix).
OPTIMA = {"oracle": "", "best_clean": "_best_clean", "visited": "_visited"}
# Landscapes whose exact floor gap (opt_z less the model-free floor) is below
# this share of opt_z: random search nearly reaches the optimum there, so the
# exact fraction's denominator is near zero and the landscape dominates a ratio
# of landscape means. The summaries also report the ratio without them.
SMALL_GAP = 0.02


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
            "data_dir": out.as_posix(),  # relative to the repository root, where every step runs
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
               "--dataset-config", str(DATA_ROOT / "configs" / f"datasets-iso_{name}.json"),
               "--objective", "composite", "--oracle-models", FAMILY or MODELS,
               "--oracle-augmentation", AUGMENTATION,
               "--cv-folds", "5", "--output-path", str(out)]
        print(f"[select] {name}", flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 3. sigma_f of the fitted oracle, exactly as the anchor computes it
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def fit_outside_repo():
    """Run an in-process oracle fit with the working directory in a temporary folder.

    CatBoost, which grouped cross-validation selects on two landscapes, writes its
    training logs to catboost_info/ in the working directory, and that folder is
    tracked in the repository, so every refit at the repository root rewrote it.
    build_oracle reads nothing from disk (the data are passed in), so the fit and
    its numbers are unchanged; read every input before entering.
    """
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="oracle_fit_", ignore_cleanup_errors=True) as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(previous)


def oracle_stats(name: str, model: str, augmentation: str, seed: int = 7) -> tuple[float, float, float]:
    """sigma_f of the oracle a run with this seed fits, its correlation with the
    truth, and its mean.

    The oracle is built as the simulator builds it for a run seed; sigma_f and
    the mean are over 20,000 uniform points in the data box (fixed, rng 7), and
    the correlation compares it there with the standardised true landscape. The
    manifest holds seed 7; scripts/review_checks/oracle_sigmaf_seeds.py the rest.
    The fit runs in a temporary working directory (fit_outside_repo), so that a
    CatBoost oracle's training logs do not land in the repository.
    """
    import bo_sensor_error_simulation as sim
    cfg = DATA_ROOT / "configs" / f"datasets-iso_{name}.json"
    dataset = sim.parse_dataset_configs(None, cfg, Path(".dataset_cache"))[0]
    frame = sim.load_observations(dataset, "composite", None, None)
    with fit_outside_repo():
        oracle = sim.build_oracle(
            df=frame, objective="composite", objective_columns=["value"],
            param_columns=dataset.param_columns, seed=seed, normalize=False, weights=None,
            oracle_model=model, oracle_augmentation=augmentation, oracle_augment_repeats=2,
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
    return sigma_f, float(np.corrcoef(y, truth)[0, 1]), float(np.mean(y))


def calibrate(args) -> None:
    import bo_sensor_error_simulation as sim
    rows = []
    for name in landscapes():
        cfg = DATA_ROOT / "configs" / f"datasets-iso_{name}.json"
        selection = sim.load_oracle_selection(ROOT / "selection" / f"iso_{name}.json")
        dataset = sim.parse_dataset_configs(None, cfg, Path(".dataset_cache"))[0]
        model = selection[(dataset.name, "composite")]["best_model"]
        r2 = selection[(dataset.name, "composite")].get("score", np.nan)
        # The noise grid of every run seed is scaled to the seed-7 oracle.
        sigma_f, corr, _ = oracle_stats(name, model, AUGMENTATION, seed=7)
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


def exact_runs(name: str, main: Path = Path("output-boba")) -> pd.DataFrame:
    """The exact arm's paired runs on one landscape: the main sweep under gaussian
    error at the three magnitudes from the first rating, seeds 7-11, with the
    magnitude in sigma_multiple (the exact objective is standardised, sigma = 1)."""
    exact = _paired(main / name)
    exact = exact[(exact.error_model.isin(["gaussian", "none"]) | exact.acquisition.isin(FLOORS))
                  & (exact.jitter_iteration == 0) & exact.seed.isin(SEEDS)
                  & exact.jitter_std.round(2).isin(SIGMA_GRID)
                  & (exact.error_model == "gaussian")]
    return exact.assign(sigma_multiple=exact["jitter_std"].round(2))


def seed_optima(frame: pd.DataFrame, how: str = "oracle") -> pd.Series:
    """The optimum of the oracle each run seed fitted, one value per seed.

    ``oracle`` is the logged y_opt (BASELINE_BEST + BASELINE_REGRET of any run of
    that seed; it is constant within a seed, which is checked). ``best_clean``
    is the best value any clean run of that seed reached. ``visited`` is y_opt
    raised to the best value any run of the seed visited, clean or noisy, where
    a run beat it: the tightest lower bound on the oracle's supremum the logs give.
    """
    if how == "oracle":
        y_opt = (frame[BASELINE_BEST] + frame[BASELINE_REGRET]).groupby(frame["seed"])
        spread = float((y_opt.max() - y_opt.min()).max())
        scale = max(1.0, float(y_opt.max().abs().max()))
        if spread > 1e-9 * scale:
            raise ValueError(f"y_opt differs between runs of one seed by {spread:g}")
        return y_opt.mean()
    if how == "best_clean":
        return frame.groupby("seed")[BASELINE_BEST].max()
    if how == "visited":
        visited = frame.groupby("seed")[[BASELINE_BEST, NOISY_BEST]].max().max(axis=1)
        return np.maximum(seed_optima(frame, "oracle"), visited)
    raise ValueError(f"unknown optimum estimator {how!r}; choose from {list(OPTIMA)}")


def floor_gap(frame: pd.DataFrame, optimum) -> float:
    """The gap from the model-free floor to the optimum.

    A scalar optimum is one function (the exact arm): the gap is the optimum less
    the floor mean pooled over every floor run. A per-seed Series is one oracle
    per seed (the fitted arm): the gap is formed within each seed and averaged
    over seeds, mean_s(optimum_s - floor_s).
    """
    floors = frame[frame.acquisition.isin(FLOORS)]
    if np.ndim(optimum) == 0:
        return float(optimum) - float(floors[BASELINE_BEST].mean())
    floor = floors.groupby("seed")[BASELINE_BEST].mean()
    learner_seeds = set(frame.loc[frame.acquisition.isin(ACQS), "seed"])
    if not learner_seeds <= set(floor.index):
        raise ValueError(f"seeds {sorted(learner_seeds - set(floor.index))} have learners but no floor runs")
    opt = pd.Series(optimum).reindex(floor.index)
    if opt.isna().any():
        raise ValueError(f"no optimum for seeds {list(opt.index[opt.isna()])}")
    return float((opt - floor).mean())


def _fraction(frame: pd.DataFrame, optimum, response: str = RESPONSE) -> pd.DataFrame:
    """Each learner's response over the floor gap (see floor_gap).

    With a per-seed optimum the gap is mean_s(gap_s), so the mean of ``frac``
    over a cell balanced across seeds is mean_s(excess_s) / mean_s(gap_s), the
    ratio of seed means.
    """
    gap = floor_gap(frame, optimum)
    learners = frame[frame.acquisition.isin(ACQS)].copy()
    learners["frac"] = learners[response] / gap
    return learners


def _check_balanced(block: pd.DataFrame, what: str) -> None:
    """A cell's mean over runs is the mean over seeds only if every seed has as many runs."""
    if "seed" in block and block.groupby("seed").size().nunique() > 1:
        raise ValueError(f"{what}: runs per seed differ {block.groupby('seed').size().to_dict()}")


# The trajectory response is the companion arm's; the deployed one is the
# paper's primary estimand, the excess regret of the design the run would ship
# at the final trial. Both are differences of two regrets on one oracle, so the
# oracle's own optimum estimate cancels in the numerator either way.
RESPONSES = {"": RESPONSE, "_deployed": "final_inference_simple_regret_excess_true"}


def analyse(args) -> None:
    for suffix, response in RESPONSES.items():
        print(f"\n== {response}")
        _analyse_one(response, suffix)


def _analyse_one(response: str, suffix: str) -> None:
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
        # One oracle per run seed: the gap is formed within each seed's oracle,
        # for the primary optimum (the logged y_opt) and the best-clean one.
        optima = {how: seed_optima(fitted, how) for how in OPTIMA}
        f_fit = {how: _fraction(fitted, opt, response) for how, opt in optima.items()}
        gap_fit = {how: floor_gap(fitted, opt) for how, opt in optima.items()}
        raised = optima["visited"] - optima["oracle"]

        exact = exact_runs(name)
        opt_z = float(stats[name]["opt_z"])
        f_exact = _fraction(exact, opt_z, response)
        gap_exact = floor_gap(exact, opt_z)

        for s in SIGMA_GRID:
            cells = {how: f[f.sigma_multiple == s] for how, f in f_fit.items()}
            a = cells["oracle"]["frac"]
            b = f_exact[f_exact.sigma_multiple == s]
            if len(a) and len(b):
                for how, cell in cells.items():
                    _check_balanced(cell, f"{name} {s} sigma fitted ({how})")
                _check_balanced(b, f"{name} {s} sigma exact")
                row = {"landscape": name, "sigma_multiple": s,
                       "frac_fitted": float(a.mean()), "frac_exact": float(b["frac"].mean()),
                       "n_fitted": len(a), "n_exact": len(b),
                       "oracle_model": manifest.loc[name, "oracle_model"],
                       "corr_oracle_truth": manifest.loc[name, "corr_oracle_truth"]}
                for how, osuffix in OPTIMA.items():
                    if osuffix:
                        row[f"frac_fitted{osuffix}"] = float(cells[how]["frac"].mean())
                row.update({"excess_fitted": float(cells["oracle"][response].mean()),
                            **{f"gap_fitted{osuffix}": gap_fit[how] for how, osuffix in OPTIMA.items()},
                            "excess_exact": float(b[response].mean()), "gap_exact": gap_exact,
                            "gap_exact_over_opt_z": gap_exact / opt_z,
                            # seeds on which some run visited a value above the logged y_opt
                            # (by more than rounding: a run can revisit the design y_opt came from)
                            "n_seeds_above_y_opt": int((raised > 1e-9).sum()),
                            "max_visited_over_y_opt": float(raised.max())})
                rows.append(row)
    per = pd.DataFrame(rows)
    per.to_csv(ROOT / f"oracle_isolation_per_landscape{suffix}.csv", index=False)
    if per.empty:
        raise SystemExit("nothing to compare yet")

    for how, osuffix in OPTIMA.items():
        summary = summarise(per, f"frac_fitted{osuffix}")
        summary.to_csv(ROOT / f"oracle_isolation_summary{suffix}{osuffix}.csv", index=False)
        print(f"\n-- optimum: {how}")
        print(summary.drop(columns=["small_gap_landscapes"], errors="ignore").to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"))


def bootstrap_indices(sizes, seed: int = DATA_SEED) -> list[np.ndarray]:
    """The landscape resamples of every summary, one array per magnitude.

    One generator, one (BOOT, n) draw per magnitude in order, as the summaries
    have drawn them since 2026-09-22 (seed DATA_SEED; the subset without the
    small-gap landscapes uses DATA_SEED + 1). Every family has the same twenty
    landscapes in the same order, so the same call gives the same resamples to
    all three, which is what the paired family contrast needs.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, (BOOT, n)) for n in sizes]


def summarise(per: pd.DataFrame, column: str = "frac_fitted", exact_column: str = "frac_exact") -> pd.DataFrame:
    """Per magnitude: the ratio of landscape means of ``column`` over the exact
    fraction ``exact_column`` with a landscape-bootstrap interval, the median
    per-landscape ratio, the landscapes where the fitted design reports less, the
    Wilcoxon p, the Spearman correlation of the per-landscape costs, and the ratio
    without the landscapes whose exact floor gap is below SMALL_GAP of opt_z.
    ``exact_column`` changes only when a sensitivity re-estimates the exact arm's
    normaliser too (oracle_achievable.py, both arms on their best clean value);
    the output keeps the name frac_exact for it."""
    from scipy.stats import spearmanr, wilcoxon
    blocks = [(s, b.sort_values("landscape")) for s, b in per.groupby("sigma_multiple")]
    indices = bootstrap_indices([len(b) for _, b in blocks])
    has_gap = "gap_exact_over_opt_z" in per
    keeps = [b["gap_exact_over_opt_z"].to_numpy() >= SMALL_GAP if has_gap else None for _, b in blocks]
    sub_indices = bootstrap_indices([int(k.sum()) for k in keeps], seed=DATA_SEED + 1) if has_gap else [None] * len(blocks)
    summary = []
    for (s, block), idx, keep, sub in zip(blocks, indices, keeps, sub_indices):
        fit, ex = block[column].to_numpy(), block[exact_column].to_numpy()
        n = len(block)
        ratio = fit[idx].mean(axis=1) / ex[idx].mean(axis=1)
        diff = fit[idx].mean(axis=1) - ex[idx].mean(axis=1)
        row = {
            "sigma_multiple": s, "n_landscapes": n,
            "frac_fitted": fit.mean(), "frac_exact": ex.mean(),
            "ratio_fitted_over_exact": fit.mean() / ex.mean(),
            "ratio_lo": float(np.percentile(ratio, 2.5)), "ratio_hi": float(np.percentile(ratio, 97.5)),
            "diff": fit.mean() - ex.mean(),
            "diff_lo": float(np.percentile(diff, 2.5)), "diff_hi": float(np.percentile(diff, 97.5)),
            "wilcoxon_p": float(wilcoxon(fit, ex).pvalue) if n >= 5 else np.nan,
            "spearman_per_landscape": float(spearmanr(fit, ex).statistic),
            "median_ratio": float(np.median(fit / ex)),
            "n_lower": int((fit < ex).sum()),
        }
        if has_gap:
            fk, ek = fit[keep], ex[keep]
            rsub = fk[sub].mean(axis=1) / ek[sub].mean(axis=1)
            row.update({"n_wo_small_gap": int(keep.sum()),
                        "ratio_wo_small_gap": fk.mean() / ek.mean(),
                        "ratio_wo_small_gap_lo": float(np.percentile(rsub, 2.5)),
                        "ratio_wo_small_gap_hi": float(np.percentile(rsub, 97.5)),
                        "n_lower_wo_small_gap": int((fk < ek).sum()),
                        "small_gap_landscapes": ";".join(block["landscape"][~keep])})
        summary.append(row)
    return pd.DataFrame(summary)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["make-data", "select", "calibrate", "analyse"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--family", default=None, help="force one oracle family (gaussian_process, mlp)")
    args = p.parse_args(argv)
    global ROOT, FAMILY, AUGMENTATION
    if args.family:
        FAMILY, AUGMENTATION = args.family, "none"
        ROOT = Path(f"output-oracle-iso-{args.family}")
    {"make-data": make_data, "select": select, "calibrate": calibrate, "analyse": analyse}[args.step](args)


if __name__ == "__main__":
    main()
