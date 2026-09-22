"""Search and selection split of every process-change arm, and acquisition robustness by family.

Two answers to one reviewer.

A. The search/selection split of each arm (reviewer question 3)
---------------------------------------------------------------
The paper says none of the six acquisition-side changes lowers the deployed
regret, and reads that as "a better search cannot fix selection". The reviewer's
objection: the paper reports deployed regret only, so an arm that LOWERS the
selection loss while RAISING the search loss by more would look like a null.
This part measures both terms for every arm.

With f the exact objective, V the designs a run evaluated and d the design it
deploys (the best-rated one),

    R(d) = [ y_opt - max_{x in V} f(x) ] + [ max_{x in V} f(x) - f(d) ]
         =        search loss            +        selection loss

At the final trial T = 50, per run, from the evaluation tables:

    deployed   final_inference_simple_regret_true    (y_opt - f(d))
    search     final_simple_regret_true              (y_opt - best true value visited)
    selection  deployed - search

Every arm is paired with the STANDARD process on the same landscape, seed,
magnitude, onset and error model, using the like-for-like reference of
analyse_boba_adaptations.ARMS (LogEI where the arm replaces the acquisition), and
reports, as fractions of opt_z (divided per landscape BEFORE any mean):

    delta_<term>   arm noisy - standard noisy      negative = the arm lowers that loss
    price_<term>   arm clean - standard clean      what the change costs without error
    std_<term>     the standard process's own noisy split in the same cells
    cost_<term>    standard noisy - standard clean (the cost of error, split)

Aggregates are means over landscapes of per-landscape means; intervals are 95%
percentile intervals from 2000 landscape-bootstrap resamples, drawn jointly for
every quantity. The identity deployed = search + selection is exact by
construction on the evaluation columns, so it is checked where it is NOT
circular: on a sample of the raw per-run logs, where the simulator writes the
deployed design's true value (inference_value_true) and the best true value
visited (best_true_so_far) as separate columns. The same sample verifies the
evaluation table's final-trial values against the logs' last row and that every
directory uses one y_opt per landscape.

An arm that runs two acquisitions is also split by acquisition, because pooling
them can hide the very pattern in question. Wherever a selection interval lies
below zero, the raw logs are read once more to ask how tightly the visited
designs crowd the best one (the gap to the 10th-best true value, the number
within 0.05 opt_z of the best): a mis-pick among near-equal designs costs little,
so a search that re-visits its incumbent's neighbourhood lowers selection loss
without identifying anything better.

Under an unnoticed input slip the deployed design is the one written down while
the rating came from the slipped design, so there f(d) can exceed the best
evaluated true value and the selection term can be negative. The minimum
per-run selection loss is reported for that reason.

B. Acquisition robustness by strategy family
--------------------------------------------
The ten model-based acquisitions are five strategies, three of them in several
forms. Win counts and Kendall's W are therefore recomputed with the forms merged:
EI (ei, logei, qei), PI (pi, logpi, qpi), UCB (ucb, qucb), NEI (qnei), Greedy.
The per-acquisition ranking is first reproduced exactly as
analyse_boba_robustness.acquisition_rankings does it: per condition (error model
x magnitude x onset), the post-onset per-iteration excess simple regret divided
by opt_z ("fragility") is averaged over seeds per landscape, ranked across
acquisitions within each landscape, and the condition's winner is the lowest MEAN
RANK over the twenty landscapes; W = Friedman chi2 / (n (k - 1)). A family's
value in a (condition, landscape) cell is the mean of its members' cell means.
As a sensitivity check, the same ranking is repeated for each of the 18 ways of
picking ONE member per family.

    python scripts/remedy_decomposition.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import boba_benchmarks as bb  # noqa: E402
import analyse_boba_adaptations as ada  # noqa: E402

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260922
T_FINAL = 50
TERMS = ("deployed", "search", "selection")
STATES = ("noisy", "clean")

# The six acquisition-side families of the paper, then the 2026-09-21 idea round.
ARM_GROUPS = {
    "q-aei": ("acquisition-side", "augmented EI"),
    "q-ts": ("acquisition-side", "Thompson sampling"),
    "q-inclcb": ("acquisition-side", "LCB incumbent"),
    "q-mind": ("acquisition-side", "minimum distance"),
    "q-iu-0.05": ("acquisition-side", "slip-aware"),
    "q-iu-0.15": ("acquisition-side", "slip-aware"),
    "q-iu-0.4": ("acquisition-side", "slip-aware"),
    "sched-front10": ("acquisition-side", "rating-effort schedule"),
    "sched-front20": ("acquisition-side", "rating-effort schedule"),
    "sched-U": ("acquisition-side", "rating-effort schedule"),
    "sched-back10": ("acquisition-side", "rating-effort schedule"),
    "idea-selfreport": ("idea round", "self-reported precision"),
    "idea-anchor-gaussian": ("idea round", "anchored rating"),
    "idea-anchors-gaussian": ("idea round", "fixed anchors"),
    "idea-hold": ("idea round", "held ratings"),
    "idea-shiplcb": ("idea round", "ship-rule acquisition"),
}

FAMILIES = {
    "EI": ["ei", "logei", "qei"],
    "PI": ["pi", "logpi", "qpi"],
    "UCB": ["ucb", "qucb"],
    "NEI": ["qnei"],
    "Greedy": ["greedy"],
}
FRAGILITY_NUMERATOR = "auc_simple_regret_excess_true_postonset_per_iter"
CONDITION = ["error_model", "jitter_std", "jitter_iteration"]
# The paper's numbers (main.tex, section "Which acquisition function?"); the
# reproduction below must match them before the family level means anything.
PAPER_WINS = {"qucb": 10, "ucb": 6, "qnei": 6, "logei": 5, "ei": 3, "qei": 2}
PAPER_MEDIAN_W = 0.247


# ---------------------------------------------------------------------------
# A. search / selection split
# ---------------------------------------------------------------------------


def per_run_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Deployed, search and selection loss at the final trial, noisy and clean."""
    bad = df["n_iterations"] != T_FINAL
    if bad.any():
        raise ValueError(f"{int(bad.sum())} runs do not have {T_FINAL} trials")
    out = df.copy()
    for state, suffix in (("noisy", "jitter"), ("clean", "baseline")):
        out[f"deployed_{state}"] = df[f"final_inference_simple_regret_true_{suffix}"]
        out[f"search_{state}"] = df[f"final_simple_regret_true_{suffix}"]
        out[f"selection_{state}"] = out[f"deployed_{state}"] - out[f"search_{state}"]
    return out


VALUE_COLS = [f"{t}_{s}" for t in TERMS for s in STATES]


def pair(ref: pd.DataFrame, trt: pd.DataFrame, pool: bool, opt_z: dict[str, float]) -> pd.DataFrame:
    keys = ada.POOLED_KEYS if pool else ada.PAIR_KEYS
    if pool:
        ref = ref.groupby(keys, as_index=False)[VALUE_COLS].mean()
        trt = trt.groupby(keys, as_index=False)[VALUE_COLS].mean()
    m = ref[keys + VALUE_COLS].merge(trt[keys + VALUE_COLS], on=keys, suffixes=("_ref", "_trt"),
                                     validate="one_to_one")
    if m.empty:
        raise ValueError("the two arms share no paired cells")
    missing = sorted(set(m["dataset"]) - set(opt_z))
    if missing:
        raise ValueError(f"no opt_z for {missing}")
    z = m["dataset"].map(opt_z)
    for c in VALUE_COLS:
        m[f"{c}_ref"] = m[f"{c}_ref"] / z
        m[f"{c}_trt"] = m[f"{c}_trt"] / z
    return m


def landscape_table(block: pd.DataFrame) -> pd.DataFrame:
    """Per-landscape means of every reported quantity (linear, so differences commute)."""
    per = block.groupby("dataset")[[f"{c}_{side}" for c in VALUE_COLS for side in ("ref", "trt")]].mean()
    out = pd.DataFrame(index=per.index)
    for t in TERMS:
        out[f"delta_{t}"] = per[f"{t}_noisy_trt"] - per[f"{t}_noisy_ref"]
        out[f"price_{t}"] = per[f"{t}_clean_trt"] - per[f"{t}_clean_ref"]
        out[f"std_{t}"] = per[f"{t}_noisy_ref"]
        out[f"std_clean_{t}"] = per[f"{t}_clean_ref"]
        out[f"arm_{t}"] = per[f"{t}_noisy_trt"]
        out[f"cost_{t}"] = per[f"{t}_noisy_ref"] - per[f"{t}_clean_ref"]
    return out


def summarise(block: pd.DataFrame) -> dict:
    per = landscape_table(block)
    arr = per.to_numpy()
    n = arr.shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
    draws = arr[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    point = arr.mean(axis=0)
    row = {"n_landscapes": int(n), "n_pairs": int(len(block))}
    for j, name in enumerate(per.columns):
        row[name] = float(point[j])
        row[f"{name}_lo"] = float(lo[j])
        row[f"{name}_hi"] = float(hi[j])
    for name in [f"delta_{t}" for t in TERMS] + [f"price_{t}" for t in TERMS]:
        v = per[name].to_numpy()
        if np.allclose(v, 0.0):
            p = 1.0
        else:
            try:
                p = float(wilcoxon(v).pvalue)
            except ValueError:
                p = 1.0
        row[f"{name}_wilcoxon_p"] = p
    # deployed = search + selection, on the aggregates and in every resample.
    row["identity_max_abs_err"] = float(max(
        abs(point[per.columns.get_loc("delta_deployed")] - point[per.columns.get_loc("delta_search")]
            - point[per.columns.get_loc("delta_selection")]),
        np.abs(draws[:, per.columns.get_loc("delta_deployed")]
               - draws[:, per.columns.get_loc("delta_search")]
               - draws[:, per.columns.get_loc("delta_selection")]).max(),
    ))
    # Selection loss per run, before any mean: negative only where the deployed
    # design differs from the evaluated one (an unnoticed input slip).
    row["min_run_selection_arm_noisy"] = float(block["selection_noisy_trt"].min())
    row["min_run_selection_std_noisy"] = float(block["selection_noisy_ref"].min())
    row["max_abs_run_selection_std_clean"] = float(block["selection_clean_ref"].abs().max())
    row["max_abs_run_selection_arm_clean"] = float(block["selection_clean_trt"].abs().max())
    return row


def masking_flags(row: dict) -> dict:
    """The reviewer's scenario: selection loss down, search loss up by more."""
    sel_down = row["delta_selection_hi"] < 0
    search_up = row["delta_search_lo"] > 0
    point = (row["delta_selection"] < 0) and (row["delta_search"] > -row["delta_selection"])
    return {
        "selection_lowered_ci": bool(sel_down),
        "search_raised_ci": bool(search_up),
        "masking_ci": bool(sel_down and search_up),
        "masking_point": bool(point),
    }


# ---------------------------------------------------------------------------
# Raw-log verification
# ---------------------------------------------------------------------------


def _read_last(path: Path) -> dict:
    log = pd.read_csv(path)
    last = log.iloc[-1]
    return {
        "iteration": int(last["iteration"]),
        "run_id": str(last["run_id"]),
        "y_opt": float(last["y_opt"]),
        "best_true_so_far": float(last["best_true_so_far"]),
        "max_objective_true": float(log["objective_true"].max()),
        "simple_regret_true": float(last["simple_regret_true"]),
        "inference_value_true": float(last["inference_value_true"]),
        "inference_simple_regret_true": float(last["inference_simple_regret_true"]),
    }


def verify_raw(frame: pd.DataFrame, root: Path, n_sample: int, rng_seed: int) -> tuple[list[dict], dict]:
    """Compare evaluation rows with the last row of their raw logs, and check the identity there."""
    sample = frame.sample(n=min(n_sample, len(frame)), random_state=rng_seed)
    rows, y_opts = [], {}
    for r in sample.itertuples(index=False):
        folder = root / r.dataset
        stem = f"bo_sensor_error_{r.dataset}_value_{r.acquisition}_seed{int(r.seed)}"
        noisy = [p for p in folder.glob(
            f"{stem}_jittered_exact_{r.error_model}_jit{int(r.jitter_iteration)}_std{float(r.jitter_std)}*.csv")]
        noisy_hit = None
        for p in noisy:
            got = _read_last(p)
            if got["run_id"] == str(r.run_id):
                noisy_hit = got
                break
        clean = sorted(folder.glob(f"{stem}_baseline_exact*.csv"))
        clean_hit = _read_last(clean[0]) if len(clean) == 1 else None
        rec = {"root": str(root), "dataset": r.dataset, "acquisition": r.acquisition, "seed": int(r.seed),
               "noisy_found": noisy_hit is not None, "clean_found": clean_hit is not None}
        for label, hit, suffix in (("noisy", noisy_hit, "jitter"), ("clean", clean_hit, "baseline")):
            if hit is None:
                continue
            y_opts.setdefault(r.dataset, set()).add(round(hit["y_opt"], 12))
            search_raw = hit["y_opt"] - hit["best_true_so_far"]
            selection_raw = hit["best_true_so_far"] - hit["inference_value_true"]
            rec[f"{label}_last_iteration"] = hit["iteration"]
            rec[f"{label}_eval_search_err"] = abs(getattr(r, f"final_simple_regret_true_{suffix}")
                                                  - hit["simple_regret_true"])
            rec[f"{label}_eval_deployed_err"] = abs(
                getattr(r, f"final_inference_simple_regret_true_{suffix}") - hit["inference_simple_regret_true"])
            rec[f"{label}_search_def_err"] = abs(search_raw - hit["simple_regret_true"])
            rec[f"{label}_best_is_max_err"] = abs(hit["best_true_so_far"] - hit["max_objective_true"])
            rec[f"{label}_identity_err"] = abs(hit["inference_simple_regret_true"]
                                               - (search_raw + selection_raw))
        rows.append(rec)
    return rows, y_opts


# ---------------------------------------------------------------------------
# Mechanism check for an arm that lowers selection loss
# ---------------------------------------------------------------------------


NEAR_TOP = 0.05  # of opt_z


def visited_set_concentration(frame: pd.DataFrame, root: Path, opt_z: dict[str, float]) -> pd.DataFrame:
    """How tightly the visited designs crowd the best one, per noisy run.

    Selection loss is the cost of mis-picking among the visited designs, so it
    shrinks when the designs near the top are close in true value, whatever the
    ratings do. A search that re-visits the neighbourhood of its incumbent can
    lower selection loss that way while finding less.
    """
    out = []
    for r in frame.itertuples(index=False):
        stem = f"bo_sensor_error_{r.dataset}_value_{r.acquisition}_seed{int(r.seed)}"
        pattern = f"{stem}_jittered_exact_{r.error_model}_jit{int(r.jitter_iteration)}_std{float(r.jitter_std)}*.csv"
        for path in (root / r.dataset).glob(pattern):
            log = pd.read_csv(path, usecols=["run_id", "objective_true"])
            if str(log["run_id"].iloc[-1]) != str(r.run_id):
                continue
            y = np.sort(log["objective_true"].to_numpy())[::-1]
            z = opt_z[r.dataset]
            out.append({k: getattr(r, k) for k in ada.PAIR_KEYS}
                       | {"gap10": (y[0] - y[9]) / z, "n_near_top": float((y >= y[0] - NEAR_TOP * z).sum())})
            break
    got = pd.DataFrame(out)
    if len(got) != len(frame):
        raise ValueError(f"{root}: matched {len(got)} of {len(frame)} noisy logs")
    return got


def concentration_contrast(ref: pd.DataFrame, trt: pd.DataFrame, pool: bool) -> dict:
    cols = ["gap10", "n_near_top"]
    keys = ada.POOLED_KEYS if pool else ada.PAIR_KEYS
    if pool:
        ref = ref.groupby(keys, as_index=False)[cols].mean()
        trt = trt.groupby(keys, as_index=False)[cols].mean()
    m = ref.merge(trt, on=keys, suffixes=("_ref", "_trt"), validate="one_to_one")
    per = m.groupby("dataset")[[f"{c}_{s}" for c in cols for s in ("ref", "trt")]].mean()
    arr = np.column_stack([per[f"{c}_trt"] - per[f"{c}_ref"] for c in cols] + [per[f"{c}_ref"] for c in cols])
    names = [f"conc_delta_{c}" for c in cols] + [f"conc_std_{c}" for c in cols]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.shape[0], size=(BOOTSTRAP_REPS, arr.shape[0]))
    draws = arr[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    row = {}
    for j, nm in enumerate(names):
        row[nm], row[f"{nm}_lo"], row[f"{nm}_hi"] = float(arr[:, j].mean()), float(lo[j]), float(hi[j])
    return row


# ---------------------------------------------------------------------------
# B. families
# ---------------------------------------------------------------------------


def load_cells(input_dir: Path, opt_z: dict[str, float]) -> pd.DataFrame:
    files = sorted(input_dir.glob("*/evaluation/paired_excess_metrics.csv"))
    use = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration",
           FRAGILITY_NUMERATOR]
    df = pd.concat([pd.read_csv(f, usecols=use) for f in files], ignore_index=True)
    df = df[~df["acquisition"].isin(ada.MODEL_FREE)]
    df["fragility"] = df[FRAGILITY_NUMERATOR] / df["dataset"].map(opt_z)
    return (df.groupby(CONDITION + ["dataset", "acquisition"])
            .agg(fragility=("fragility", "mean"), seeds=("seed", "nunique")).reset_index())


def to_units(cell: pd.DataFrame, units: dict[str, list[str]]) -> pd.DataFrame:
    """A unit's value in a (condition, landscape) cell is the mean of its members' cell means."""
    parts = []
    for unit, members in units.items():
        blk = cell[cell["acquisition"].isin(members)]
        found = set(blk["acquisition"])
        if found != set(members):
            raise ValueError(f"{unit}: members {sorted(set(members) - found)} missing")
        g = blk.groupby(CONDITION + ["dataset"])
        agg = g["fragility"].mean().rename("fragility").to_frame()
        agg["n_members"] = g["acquisition"].nunique()
        parts.append(agg.reset_index().assign(unit=unit))
    out = pd.concat(parts, ignore_index=True)
    # A missing member in one cell would make the unit an average of fewer forms there.
    short = out["n_members"] != out["unit"].map({u: len(m) for u, m in units.items()})
    if short.any():
        raise ValueError(f"{int(short.sum())} unit cells average fewer members than the unit has")
    return out


def rank_conditions(units_cell: pd.DataFrame, level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exactly analyse_boba_robustness.acquisition_rankings, on arbitrary units."""
    unit_rows, cond_rows = [], []
    for (em, std, onset), block in units_cell.groupby(CONDITION):
        wide = block.pivot_table(index="dataset", columns="unit", values="fragility").dropna(how="any")
        n, k = wide.shape
        ranks = wide.rank(axis=1, method="average")
        mean_rank = ranks.mean(axis=0).sort_values()
        means = wide.mean(axis=0)
        stat, p = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
        w = float(stat / (n * (k - 1)))
        win_rank, win_mean = mean_rank.index[0], means.idxmin()
        cond_rows.append({"level": level, "error_model": em, "jitter_std": float(std),
                          "jitter_iteration": int(onset), "n_landscapes": int(n), "k": int(k),
                          "winner_by_mean_rank": win_rank, "winner_by_mean_fragility": win_mean,
                          "kendall_w": w, "friedman_p": float(p)})
        for u in wide.columns:
            unit_rows.append({"level": level, "error_model": em, "jitter_std": float(std),
                              "jitter_iteration": int(onset), "unit": u,
                              "mean_rank": float(mean_rank[u]), "mean_fragility": float(means[u]),
                              "is_winner_by_mean_rank": u == win_rank,
                              "is_winner_by_mean_fragility": u == win_mean,
                              "kendall_w": w, "friedman_p": float(p), "n_landscapes": int(n), "k": int(k)})
    return pd.DataFrame(unit_rows), pd.DataFrame(cond_rows)


def win_counts(cond: pd.DataFrame, col: str, order: list[str]) -> dict[str, int]:
    vc = cond[col].value_counts()
    return {u: int(vc.get(u, 0)) for u in order}


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def pct(v: float) -> str:
    return f"{100 * v:+.1f}"


def ci(row: pd.Series, name: str) -> str:
    return f"{pct(row[name])} [{pct(row[name + '_lo'])}, {pct(row[name + '_hi'])}]"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", type=str, default=",".join(ARM_GROUPS))
    p.add_argument("--main-dir", type=Path, default=Path("output-boba"))
    p.add_argument("--output-dir", type=Path, default=Path("output-boba/analysis/review"))
    p.add_argument("--verify-per-dir", type=int, default=40,
                   help="raw logs sampled per arm and per reference for the verification")
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}

    # ------------------------------------------------------------------ A
    rows, verify_rows, y_opt_by_dir = [], [], {}
    verified_frames: set[tuple] = set()
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        spec = ada.ARMS[name]
        group, family = ARM_GROUPS[name]
        arm_dir, ref_dir = Path(spec["dir"]), Path(spec["ref"])
        seeds = {int(s) for s in spec["seeds"].split(",")}
        ref_raw = ada.load(ref_dir, set(spec["ref_acqs"].split(",")), seeds, spec["error_model"],
                           spec.get("ref_variant"))
        arm_raw = ada.load(arm_dir, set(spec["acqs"].split(",")), seeds, spec["error_model"],
                           spec.get("variant"))
        ref_df, arm_df = per_run_terms(ref_raw), per_run_terms(arm_raw)
        paired = pair(ref_df, arm_df, spec["pool"], opt_z)

        # Raw-log verification: each (directory, filter) once.
        for root, frame, tag in ((arm_dir, arm_raw, name), (ref_dir, ref_raw, f"ref:{name}")):
            key = (str(root), spec["error_model"], spec["acqs"] if root == arm_dir else spec["ref_acqs"],
                   spec.get("variant") if root == arm_dir else spec.get("ref_variant"))
            if key in verified_frames:
                continue
            verified_frames.add(key)
            vr, yo = verify_raw(frame, root, args.verify_per_dir, rng_seed=len(verified_frames))
            for v in vr:
                v["arm_filter"] = tag
            verify_rows.extend(vr)
            for d, s in yo.items():
                y_opt_by_dir.setdefault(d, {}).setdefault(str(root), set()).update(s)

        common = {"arm": name, "group": group, "family": family, "what": spec["what"],
                  "reference_dir": str(ref_dir), "reference_acqs": spec["ref_acqs"],
                  "arm_acqs": spec["acqs"], "error_model": spec["error_model"],
                  "pooled_over_acquisitions": bool(spec["pool"])}
        r = {**common, "scope": "pooled", "acquisition": "all", "jitter_std": np.nan,
             "jitter_iteration": np.nan, **summarise(paired)}
        pooled = {**r, **masking_flags(r)}
        arm_rows = [pooled]
        for (std, onset), block in paired.groupby(["jitter_std", "jitter_iteration"]):
            r = {**common, "scope": "cell", "acquisition": "all", "jitter_std": float(std),
                 "jitter_iteration": int(onset), **summarise(block)}
            arm_rows.append({**r, **masking_flags(r)})
        # Pooling two acquisitions can hide the same pattern the reviewer worries
        # about, so a multi-acquisition arm is also split by acquisition.
        if not spec["pool"] and paired["acquisition"].nunique() > 1:
            for acq, block in paired.groupby("acquisition"):
                r = {**common, "scope": "acquisition", "acquisition": acq, "jitter_std": np.nan,
                     "jitter_iteration": np.nan, **summarise(block)}
                arm_rows.append({**r, **masking_flags(r)})
        # Mechanism check wherever the selection loss goes down.
        if any(x["selection_lowered_ci"] for x in arm_rows if x["scope"] != "cell"):
            conc_ref = visited_set_concentration(ref_raw, ref_dir, opt_z)
            conc_arm = visited_set_concentration(arm_raw, arm_dir, opt_z)
            for x in arm_rows:
                if x["scope"] == "cell":
                    continue
                sel = (lambda f: f) if x["acquisition"] == "all" else (
                    lambda f, a=x["acquisition"]: f[f["acquisition"] == a])
                x.update(concentration_contrast(sel(conc_ref), sel(conc_arm), spec["pool"]))
        rows.extend(arm_rows)
        print(f"{name:22s} d_dep {pct(pooled['delta_deployed'])}  d_search {pct(pooled['delta_search'])}  "
              f"d_sel {pct(pooled['delta_selection'])}  price {pct(pooled['price_deployed'])}  "
              f"(pairs {pooled['n_pairs']}, landscapes {pooled['n_landscapes']})")

    res = pd.DataFrame(rows)
    res.to_csv(args.output_dir / "remedy_decomposition.csv", index=False)

    # Cross-check against the recovery table the paper quotes: delta_deployed = -gain.
    cross = []
    rec_path = Path("output-boba/analysis/adaptations_recovery.csv")
    if rec_path.is_file():
        rec = pd.read_csv(rec_path)
        rec = rec[rec["response"] == "deployed"].drop_duplicates("arm").set_index("arm")
        for r in res[res["scope"] == "pooled"].itertuples():
            if r.arm in rec.index:
                cross.append((r.arm, r.delta_deployed, -float(rec.loc[r.arm, "pooled_gain"]),
                              r.price_deployed, float(rec.loc[r.arm, "pooled_price"])))
    cross_df = pd.DataFrame(cross, columns=["arm", "delta_deployed", "minus_recovery_gain",
                                            "price_deployed", "recovery_price"])
    ver = pd.DataFrame(verify_rows)

    # ------------------------------------------------------------------ B
    cell = load_cells(args.main_dir, opt_z)
    ten = [a for fam in FAMILIES.values() for a in fam]
    acq_units, acq_cond = rank_conditions(to_units(cell, {a: [a] for a in ten}), "acquisition")
    fam_units, fam_cond = rank_conditions(to_units(cell, FAMILIES), "family")
    acq_wins = win_counts(acq_cond, "winner_by_mean_rank", ten)
    acq_wins_mean = win_counts(acq_cond, "winner_by_mean_fragility", ten)
    fam_order = list(FAMILIES)
    fam_wins = win_counts(fam_cond, "winner_by_mean_rank", fam_order)
    fam_wins_mean = win_counts(fam_cond, "winner_by_mean_fragility", fam_order)
    collapsed = {f: sum(acq_wins[a] for a in m) for f, m in FAMILIES.items()}
    reproduced = ({a: c for a, c in acq_wins.items() if c} == PAPER_WINS
                  and abs(acq_cond["kendall_w"].median() - PAPER_MEDIAN_W) < 5e-4)

    rep_rows, rep_units = [], []
    for combo in itertools.product(*FAMILIES.values()):
        units = dict(zip(fam_order, [[a] for a in combo]))
        u, c = rank_conditions(to_units(cell, units), "representatives:" + "/".join(combo))
        rep_units.append(u)
        wins = win_counts(c, "winner_by_mean_rank", fam_order)
        rep_rows.append({"representatives": "/".join(combo), "median_w": float(c["kendall_w"].median()),
                         "friedman_sig": int((c["friedman_p"] < 0.05).sum()),
                         **{f"wins_{f}": wins[f] for f in fam_order}})
    rep = pd.DataFrame(rep_rows)
    fam_out = pd.concat([acq_units, fam_units] + rep_units, ignore_index=True)
    fam_out.to_csv(args.output_dir / "acquisition_families.csv", index=False)

    # ------------------------------------------------------------------ report
    pooled = res[res["scope"] == "pooled"].set_index("arm")
    cells = res[res["scope"] == "cell"]
    L = []
    L.append("# Search/selection split of the process-change arms, and acquisition families\n")
    L.append("Generated by `scripts/remedy_decomposition.py`. Units: percent of opt_z (the achievable "
             "improvement per landscape), divided per landscape before averaging; mean over landscapes of "
             "per-landscape means; 95% landscape-bootstrap percentile intervals, 2000 resamples. "
             "Final trial T = 50. delta = arm minus the like-for-like standard process on the same "
             "landscape, seed, magnitude, onset and error model; **negative = the arm lowers that loss**. "
             "Price = the same difference on the clean (error-free) runs.\n")
    L.append("## A. Noisy runs: change in deployed, search and selection loss\n")
    L.append("| arm | family | reference | pairs | delta deployed | delta search | delta selection | "
             "price deployed | price search | price selection |")
    L.append("|---|---|---|---:|---|---|---|---|---|---|")
    for name in pooled.index:
        r = pooled.loc[name]
        L.append(f"| {name} | {r['family']} | {Path(r['reference_dir']).name} ({r['reference_acqs']}) | "
                 f"{int(r['n_pairs'])} | {ci(r, 'delta_deployed')} | {ci(r, 'delta_search')} | "
                 f"{ci(r, 'delta_selection')} | {ci(r, 'price_deployed')} | {ci(r, 'price_search')} | "
                 f"{ci(r, 'price_selection')} |")
    L.append("\n## The standard process's own split in the same cells (noisy, and its cost of error)\n")
    L.append("| arm cells | std deployed | std search | std selection | selection share | "
             "cost deployed | cost search | cost selection |")
    L.append("|---|---|---|---|---:|---|---|---|")
    for name in pooled.index:
        r = pooled.loc[name]
        share = r["std_selection"] / r["std_deployed"] if r["std_deployed"] > 0 else np.nan
        L.append(f"| {name} | {ci(r, 'std_deployed')} | {ci(r, 'std_search')} | {ci(r, 'std_selection')} | "
                 f"{100 * share:.0f}% | {ci(r, 'cost_deployed')} | {ci(r, 'cost_search')} | "
                 f"{ci(r, 'cost_selection')} |")

    L.append("\n## Does the reviewer's masking scenario occur?\n")
    L.append("Masking = selection loss lowered (delta selection CI entirely below 0) while search loss is "
             "raised (delta search CI entirely above 0). The point-estimate version asks only for delta "
             "selection < 0 and delta search > |delta selection|.\n")
    L.append(f"A masked NULL additionally needs the delta-deployed interval to cover zero. Cell-level flags "
             f"are uncorrected for the {len(cells)} cells and are exploratory.\n")
    L.append("| arm | selection lowered (CI) | search raised (CI) | masking (CI) | masking (point) | "
             "cells with masking (CI) / point | of which masked null (CI) | cells |")
    L.append("|---|---|---|---|---|---|---|---:|")
    for name in pooled.index:
        r = pooled.loc[name]
        c = cells[cells["arm"] == name]
        null_cells = c[c["masking_ci"] & (c["delta_deployed_lo"] <= 0) & (c["delta_deployed_hi"] >= 0)]
        where = ", ".join(f"{s.jitter_std:g} sigma / trial {int(s.jitter_iteration) + 1}"
                          for s in null_cells.itertuples())
        L.append(f"| {name} | {r['selection_lowered_ci']} | {r['search_raised_ci']} | {r['masking_ci']} | "
                 f"{r['masking_point']} | {int(c['masking_ci'].sum())} / {int(c['masking_point'].sum())} | "
                 f"{len(null_cells)}{' (' + where + ')' if where else ''} | {len(c)} |")
    acq_side = pooled[pooled["group"] == "acquisition-side"]
    L.append(f"\nAcquisition-side arms with the pooled masking pattern (CI): "
             f"{', '.join(acq_side.index[acq_side['masking_ci']]) or 'none'}; "
             f"(point): {', '.join(acq_side.index[acq_side['masking_point']]) or 'none'}. "
             f"Acquisition-side arms that lower selection loss at all (CI): "
             f"{', '.join(acq_side.index[acq_side['selection_lowered_ci']]) or 'none'}. "
             f"Masked null (delta deployed interval covering zero while selection falls and search "
             f"rises, CI): "
             f"{', '.join(acq_side.index[acq_side['masking_ci'] & (acq_side['delta_deployed_lo'] <= 0) & (acq_side['delta_deployed_hi'] >= 0)]) or 'none'}.\n")

    by_acq = res[res["scope"] == "acquisition"]
    if len(by_acq):
        L.append("## Per acquisition within the multi-acquisition arms (noisy, pooled over cells)\n")
        L.append("| arm | acquisition | pairs | delta deployed | delta search | delta selection | price deployed | "
                 "std selection | masking (CI) | masking (point) |")
        L.append("|---|---|---:|---|---|---|---|---|---|---|")
        for r in by_acq.itertuples():
            s = pd.Series(r._asdict())
            L.append(f"| {r.arm} | {r.acquisition} | {r.n_pairs} | {ci(s, 'delta_deployed')} | "
                     f"{ci(s, 'delta_search')} | {ci(s, 'delta_selection')} | {ci(s, 'price_deployed')} | "
                     f"{ci(s, 'std_selection')} | {r.masking_ci} | {r.masking_point} |")
        L.append("")
    if "conc_delta_gap10" in res.columns:
        conc = res[res["conc_delta_gap10"].notna()]
        L.append("## Mechanism where selection loss falls: how tightly the visited designs crowd the best\n")
        L.append(f"Computed from the raw noisy logs of every arm whose pooled or per-acquisition selection "
                 f"interval lies below zero. gap10 = (best - 10th-best true value visited) / opt_z, in percent; "
                 f"n near top = visited designs within {NEAR_TOP:g} opt_z of the best visited. A smaller gap "
                 f"or more designs near the top means a mis-pick costs less whatever the ratings do.\n")
        L.append("| arm | acquisition | std gap10 | delta gap10 | std n near top | delta n near top |")
        L.append("|---|---|---|---|---|---|")
        for r in conc.itertuples():
            s = pd.Series(r._asdict())
            L.append(f"| {r.arm} | {r.acquisition} | {ci(s, 'conc_std_gap10')} | {ci(s, 'conc_delta_gap10')} | "
                     f"{r.conc_std_n_near_top:.1f} | {r.conc_delta_n_near_top:+.1f} "
                     f"[{r.conc_delta_n_near_top_lo:+.1f}, {r.conc_delta_n_near_top_hi:+.1f}] |")
        L.append("")

    L.append("## Per cell (magnitude x onset), noisy\n")
    L.append("| arm | sigma | onset (trial) | delta deployed | delta search | delta selection | price deployed |")
    L.append("|---|---:|---:|---|---|---|---|")
    for r in cells.itertuples():
        s = pd.Series(r._asdict())
        L.append(f"| {r.arm} | {r.jitter_std:g} | {int(r.jitter_iteration) + 1} | {ci(s, 'delta_deployed')} | "
                 f"{ci(s, 'delta_search')} | {ci(s, 'delta_selection')} | {ci(s, 'price_deployed')} |")

    L.append("\n## Checks\n")
    L.append(f"* Identity deployed = search + selection on the aggregates and in every bootstrap resample: "
             f"max abs error {res['identity_max_abs_err'].max():.1e} (exact by construction on the "
             f"evaluation columns).")
    if len(ver):
        def mx(col):
            return float(ver[col].max()) if col in ver else float("nan")
        n_noisy, n_clean = int(ver["noisy_found"].sum()), int(ver["clean_found"].sum())
        iters = pd.concat([ver[c] for c in ("noisy_last_iteration", "clean_last_iteration") if c in ver])
        all_final = bool((iters.dropna() == T_FINAL).all())
        L.append(f"* Raw logs, where the identity is NOT circular (selection = best_true_so_far - "
                 f"inference_value_true, search = y_opt - best_true_so_far, deployed = "
                 f"inference_simple_regret_true): {n_noisy} noisy and {n_clean} clean logs matched out of "
                 f"{len(ver)} sampled runs across {ver['root'].nunique()} directories. Max identity error "
                 f"{max(mx('noisy_identity_err'), mx('clean_identity_err')):.1e}; max |evaluation - log| "
                 f"search {max(mx('noisy_eval_search_err'), mx('clean_eval_search_err')):.1e}, deployed "
                 f"{max(mx('noisy_eval_deployed_err'), mx('clean_eval_deployed_err')):.1e}; "
                 f"best_true_so_far = max objective_true within "
                 f"{max(mx('noisy_best_is_max_err'), mx('clean_best_is_max_err')):.1e}; last iteration "
                 f"always {T_FINAL}: {all_final}.")
    multi = {d: {k: v for k, v in dirs.items()} for d, dirs in y_opt_by_dir.items()}
    n_distinct = {d: len(set().union(*dirs.values())) for d, dirs in multi.items()}
    L.append(f"* y_opt identical across every sampled directory for each landscape: "
             f"{all(v == 1 for v in n_distinct.values())} ({len(n_distinct)} landscapes).")
    L.append(f"* Minimum per-run noisy selection loss, arm / standard: "
             + "; ".join(f"{a} {pooled.loc[a, 'min_run_selection_arm_noisy']:.2g} / "
                         f"{pooled.loc[a, 'min_run_selection_std_noisy']:.2g}" for a in pooled.index)
             + " (fractions of opt_z; negative only under the slip error model, where the deployed "
               "design is the logged one and the rating came from the slipped one).")
    L.append(f"* Max |clean selection loss| per run, standard / arm: "
             + "; ".join(f"{a} {pooled.loc[a, 'max_abs_run_selection_std_clean']:.1g} / "
                         f"{pooled.loc[a, 'max_abs_run_selection_arm_clean']:.1g}" for a in pooled.index))
    if len(cross_df):
        L.append(f"* Against `adaptations_recovery.csv` (deployed response): max |delta deployed - (-gain)| "
                 f"{(cross_df['delta_deployed'] - cross_df['minus_recovery_gain']).abs().max():.1e}, max "
                 f"|price - price| {(cross_df['price_deployed'] - cross_df['recovery_price']).abs().max():.1e} "
                 f"over {len(cross_df)} arms.")

    L.append("\n## B. Acquisition robustness by strategy family\n")
    L.append("Criterion reproduced from `analyse_boba_robustness.acquisition_rankings`: per condition, "
             "fragility (post-onset per-iteration excess simple regret / opt_z) averaged over seeds per "
             "landscape, ranked within landscape; winner = lowest **mean rank** over the 20 landscapes; "
             "W = Friedman chi2 / (n(k-1)). Note that this is the lowest mean rank, not the lowest mean "
             "excess; the latter is reported beside it.\n")
    L.append(f"Reproduction of the paper (qUCB 10, UCB 6, qNEI 6, LogEI 5, EI 3, qEI 2; median W 0.247): "
             f"**{'matched' if reproduced else 'NOT matched'}**.\n")
    L.append("| unit | wins by mean rank | wins by mean excess |")
    L.append("|---|---:|---:|")
    for a in ten:
        L.append(f"| {a} | {acq_wins[a]} | {acq_wins_mean[a]} |")
    L.append(f"\n10 acquisitions: median W {acq_cond['kendall_w'].median():.3f} (range "
             f"{acq_cond['kendall_w'].min():.3f}-{acq_cond['kendall_w'].max():.3f}), Friedman p < 0.05 in "
             f"{int((acq_cond['friedman_p'] < 0.05).sum())} of {len(acq_cond)}.\n")
    L.append("| family | members | wins by mean rank (family mean) | wins by mean excess | "
             "naive sum of member wins | range over the 18 one-member choices |")
    L.append("|---|---|---:|---:|---:|---|")
    for f in fam_order:
        L.append(f"| {f} | {', '.join(FAMILIES[f])} | {fam_wins[f]} | {fam_wins_mean[f]} | {collapsed[f]} | "
                 f"{rep[f'wins_{f}'].min()}-{rep[f'wins_{f}'].max()} |")
    L.append(f"\nFamilies (member means): median W {fam_cond['kendall_w'].median():.3f} (range "
             f"{fam_cond['kendall_w'].min():.3f}-{fam_cond['kendall_w'].max():.3f}), Friedman p < 0.05 in "
             f"{int((fam_cond['friedman_p'] < 0.05).sum())} of {len(fam_cond)}. Over the 18 one-member "
             f"choices the median W runs {rep['median_w'].min():.3f}-{rep['median_w'].max():.3f}. Under the "
             f"null E[W] = 1/n = {1 / 20:.3f} for n = 20 landscapes, whatever k.\n")
    L.append("| representatives (EI/PI/UCB/NEI/Greedy) | median W | Friedman sig. | "
             + " | ".join(f"wins {f}" for f in fam_order) + " |")
    L.append("|---|---:|---:|" + "---:|" * len(fam_order))
    for r in rep.itertuples():
        L.append(f"| {r.representatives} | {r.median_w:.3f} | {r.friedman_sig} | "
                 + " | ".join(str(getattr(r, f"wins_{f}")) for f in fam_order) + " |")
    (args.output_dir / "remedy_decomposition.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n10-arm wins (mean rank): { {a: c for a, c in acq_wins.items() if c} }, "
          f"median W {acq_cond['kendall_w'].median():.3f}; reproduced: {reproduced}")
    print(f"family wins (mean rank): {fam_wins}, median W {fam_cond['kendall_w'].median():.3f}; "
          f"naive collapse {collapsed}")
    print(f"family wins (mean excess): {fam_wins_mean}; 10-arm by mean excess "
          f"{ {a: c for a, c in acq_wins_mean.items() if c} }")
    print(f"representatives median W {rep['median_w'].min():.3f}-{rep['median_w'].max():.3f}")
    if len(ver):
        print(f"verified {int(ver['noisy_found'].sum())} noisy / {int(ver['clean_found'].sum())} clean raw logs")
    print(f"Wrote {args.output_dir}/remedy_decomposition.csv, acquisition_families.csv, remedy_decomposition.md")


if __name__ == "__main__":
    main()
