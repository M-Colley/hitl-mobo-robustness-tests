"""How many extra trials does feedback error cost?

Every other number in this project is a regret. A practitioner planning a human
study asks a different question: if my raters are this noisy, how many more
trials do I have to budget to end up where a clean study would have ended up?
This script answers it directly from the paired trajectories.

For a noisy run and its identically seeded clean run, with R(t) the best-so-far
TRUE simple regret after t trials:

    tau(k) = min{ t : R_noisy(t) <= R_clean(k) }        extra(k) = tau(k) - k

is the number of extra trials the noisy run needs to reach the point the clean
run had reached after k trials. R is the true regret of the best point actually
evaluated, so this is the optimizer's real progress, not what it believes.

A noisy run that never gets there within its budget T is CENSORED. Its extra is
counted as T - k, so every mean reported here is a lower bound, and the censored
fraction is reported beside it. Use a k well inside the budget (k = 25 of 50, or
k = 50 of 100) so that censoring stays rare.

Model-free floors are excluded: in the response-error arms their extra is zero
by construction, and in the input-error arms they are not a learner.

    python scripts/analyse_extra_runs.py --input-dir output-boba --k 10,25
    python scripts/analyse_extra_runs.py --input-dir output-boba-budget100 --k 25,50
    python scripts/analyse_extra_runs.py --input-dir output-fitted --k 10,25 --relative-magnitudes
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_FREE = ("random", "sobol")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260912
# The magnitudes of the response-error grid, in landscape SDs. The fitted arm
# expresses the same four in each dataset's own sigma_f, so its file names
# carry different numbers; --relative-magnitudes maps them back by rank.
GRID = (0.05, 0.25, 1.0, 5.0)

STEM = re.compile(r"^bo_sensor_error_(?P<dataset>.+)_(?P<objective>[a-z]+)_(?P<acq>[a-z]+)_seed(?P<seed>\d+)$")
# The two missing-rating labels carry an underscore, which [a-z0-9]+ would split
# into oracle "exact_missing" and model "mcar"; they are named first. No other
# label is affected, so existing directories parse exactly as before.
JITTERED = re.compile(
    r"^(?P<oracle>[a-z_]+?)_(?P<model>missing_mcar|missing_low|[a-z0-9]+)_jit(?P<onset>\d+)_std(?P<std>[0-9.]+)(?P<suffix>.*)$"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <input-dir>/analysis")
    p.add_argument("--k", type=str, default="10,25", help="comma-separated reference trial counts")
    p.add_argument("--relative-magnitudes", action="store_true",
                   help="map each dataset's four magnitudes to the grid by rank (fitted-oracle arm)")
    p.add_argument("--tolerance", type=str, default="0",
                   help="comma-separated slack on the target, as a fraction of opt_z (the achievable "
                        "improvement): 0 means the noisy run must match the clean run's regret exactly, "
                        "0.01 means it may stop one percent of the achievable improvement short")
    p.add_argument("--acquisitions", type=str, default=None,
                   help="comma-separated subset of acquisitions (a reference matched to a follow-up arm)")
    p.add_argument("--seeds", type=str, default=None, help="comma-separated subset of seeds, likewise")
    p.add_argument("--output-name", type=str, default="extra_runs",
                   help="stem of the two output files; use a distinct name for a restricted reference")
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="take the CLEAN runs from this directory instead of --input-dir. A process "
                        "adaptation (replication, re-rating) also changes its own clean twin, so the "
                        "question 'how many trials to match a clean standard study' needs the "
                        "standard arm's clean runs as the target.")
    return p.parse_args(argv)


def load_opt_z(input_dir: Path) -> dict[str, float]:
    meta = input_dir / "run_metadata.json"
    if not meta.is_file():
        return {}
    import json
    stats = json.loads(meta.read_text(encoding="utf-8")).get("landscape_stats", {})
    return {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}


def index_runs(input_dir: Path) -> tuple[dict, list]:
    """Baselines keyed by stem (landscape, objective, acquisition, seed); jittered runs as records.

    One clean run per stem per directory. The part after "_baseline_" is the
    oracle plus any variant suffix ("exact_inc-observed_max"), and oracle names
    themselves contain underscores ("extra_trees"), so it cannot be split
    reliably; the stem alone identifies the clean twin, and a second clean run
    for the same stem is an error rather than a silent overwrite.
    """
    baselines: dict[str, Path] = {}
    jittered: list[dict] = []
    for path in sorted(input_dir.glob("*/bo_sensor_error_*.csv")):
        name = path.stem
        if "_baseline_" in name:
            stem, _ = name.split("_baseline_", 1)
            if stem in baselines:
                raise ValueError(f"two clean runs for {stem} under {input_dir}: {baselines[stem].name}, {path.name}")
            baselines[stem] = path
        elif "_jittered_" in name:
            stem, rest = name.split("_jittered_", 1)
            m = JITTERED.match(rest)
            s = STEM.match(stem)
            if not m or not s:
                continue
            jittered.append({
                "stem": stem, "oracle": m["oracle"], "error_model": m["model"],
                "jitter_iteration": int(m["onset"]), "jitter_std": float(m["std"]),
                "variant": m["suffix"].lstrip("_"), "dataset": s["dataset"],
                "acquisition": s["acq"], "seed": int(s["seed"]), "path": path,
            })
    return baselines, jittered


def regret_curve(path: Path) -> np.ndarray:
    df = pd.read_csv(path, usecols=["iteration", "simple_regret_true"])
    return df.sort_values("iteration")["simple_regret_true"].to_numpy(dtype=float)


def extra_trials(clean: np.ndarray, noisy: np.ndarray, k: int, slack: float = 0.0) -> tuple[float, bool]:
    """(extra, censored).

    Measured from the trial at which the CLEAN run itself first reached its
    k-trial regret, not from k: best-so-far regret sits on plateaus, so a run
    identical to the clean one (every late-onset run before its onset) would
    otherwise read as "ahead" by the length of the plateau. With this origin an
    identical run scores exactly zero. ``slack`` (in regret units) lets the
    noisy run stop that far short of the target; ties count as reached.
    """
    target = clean[k - 1] + 1e-12 + slack
    origin = int(np.flatnonzero(clean <= target)[0]) + 1
    hit = np.flatnonzero(noisy <= target)
    if len(hit) == 0:
        return float(len(noisy) - origin), True
    return float(hit[0] + 1 - origin), False


def per_run_table(input_dir: Path, ks: list[int], tolerances: list[float],
                  acquisitions: set[str] | None = None, seeds: set[int] | None = None,
                  baseline_dir: Path | None = None) -> pd.DataFrame:
    baselines, jittered = index_runs(input_dir)
    if baseline_dir is not None:
        baselines, _ = index_runs(baseline_dir)
    # Restrict before reading: a matched reference on the main sweep would
    # otherwise load all of its runs to keep a few percent of them.
    if acquisitions:
        jittered = [r for r in jittered if r["acquisition"] in acquisitions]
    if seeds:
        jittered = [r for r in jittered if r["seed"] in seeds]
    opt_z = load_opt_z(input_dir)
    if any(t > 0 for t in tolerances) and not opt_z:
        raise SystemExit(f"a tolerance needs opt_z, and {input_dir}/run_metadata.json has no landscape_stats")
    rows = []
    cache: dict[Path, np.ndarray] = {}
    for run in jittered:
        if run["acquisition"] in MODEL_FREE:
            continue
        base = baselines.get(run["stem"])
        if base is None:
            continue
        if base not in cache:
            cache[base] = regret_curve(base)
        clean, noisy = cache[base], regret_curve(run["path"])
        T = min(len(clean), len(noisy))
        for k in ks:
            if k >= T:
                continue
            for tol in tolerances:
                slack = tol * opt_z.get(run["dataset"], 0.0)
                extra, censored = extra_trials(clean[:T], noisy[:T], k, slack)
                rows.append({**{c: run[c] for c in ("dataset", "acquisition", "seed", "error_model",
                                                       "jitter_std", "jitter_iteration", "variant")},
                             "budget": T, "k": k, "tolerance": tol, "extra": extra,
                             "censored": censored, "multiplier": (k + extra) / k})
    if not rows:
        raise SystemExit(f"no paired runs found under {input_dir}")
    return pd.DataFrame(rows)


def rank_magnitudes(runs: pd.DataFrame) -> pd.DataFrame:
    """Replace each dataset's own magnitudes with the grid, by rank."""
    out = runs.copy()
    for dataset, block in runs.groupby("dataset"):
        stds = sorted(block["jitter_std"].unique())
        if len(stds) != len(GRID):
            raise ValueError(f"{dataset}: {len(stds)} magnitudes, expected {len(GRID)}")
        out.loc[block.index, "jitter_std"] = block["jitter_std"].map(dict(zip(stds, GRID)))
    return out


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Per condition: mean extra (censored at T - k, so a lower bound), its
    landscape-bootstrap interval, the median, and the censored fraction."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    keys = ["error_model", "variant", "jitter_std", "jitter_iteration", "k", "tolerance"]
    for key, cell in runs.groupby(keys):
        per = cell.groupby("dataset")["extra"].mean()
        values = per.to_numpy()
        draws = np.array([values[rng.integers(0, len(values), len(values))].mean()
                          for _ in range(BOOTSTRAP_REPS)])
        rows.append({
            **dict(zip(keys, key)),
            "budget": int(cell["budget"].iloc[0]),
            "n_runs": int(len(cell)),
            "n_landscapes": int(len(values)),
            "mean_extra": float(values.mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            # The median is exact as long as fewer than half the runs are
            # censored, since every censored value sits above every reached one.
            "median_extra": float(cell["extra"].median()),
            "mean_extra_reached": float(cell.loc[~cell["censored"], "extra"].mean())
            if (~cell["censored"]).any() else np.nan,
            "mean_multiplier": float(cell["multiplier"].mean()),
            "censored_fraction": float(cell["censored"].mean()),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    ks = [int(v) for v in args.k.split(",") if v.strip()]
    tolerances = [float(v) for v in args.tolerance.split(",") if v.strip()]
    out = args.output_dir or (args.input_dir / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    acqs = {a.strip() for a in args.acquisitions.split(",")} if args.acquisitions else None
    seed_set = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    runs = per_run_table(args.input_dir, ks, tolerances, acqs, seed_set, args.baseline_dir)
    if args.relative_magnitudes:
        runs = rank_magnitudes(runs)
    runs.to_csv(out / f"{args.output_name}_per_run.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(out / f"{args.output_name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(runs):,} paired learner runs from {runs['dataset'].nunique()} landscapes; "
          f"budget T = {sorted(runs['budget'].unique())}")
    print("\nextra trials the noisy run needs to reach the clean run's regret after k trials")
    print("(mean over landscapes, censored at T - k; 95% landscape-bootstrap interval; censored fraction)\n")
    show = summary.assign(
        cell=lambda d: d.apply(lambda r: f"{r.mean_extra:5.1f} [{r.ci_low:4.1f}, {r.ci_high:4.1f}]  "
                                         f"x{r.mean_multiplier:.2f}  cens {r.censored_fraction:.0%}", axis=1))
    for (model, variant, tol), block in show.groupby(["error_model", "variant", "tolerance"]):
        label = model + (f" ({variant})" if variant else "") + f"  tolerance {tol:g} of opt_z"
        print(f"--- {label} ---")
        print(block.pivot_table(index=["k", "jitter_std"], columns="jitter_iteration", values="cell",
                                aggfunc="first").to_string())
        print()
    print(f"Wrote {out / (args.output_name + '.csv')} and {args.output_name}_per_run.csv")


if __name__ == "__main__":
    main()
