"""Every selected remedy, re-selected on half the data and scored on the other half.

The remedy section picks winners from families of replayed procedures: a k for
the final comparative sitting, one of 19 sitting variants, one of four ship rules
(and a magnitude threshold for it). Each winner's full-data value went through
that selection, so it is optimistic by an unknown amount. This script repeats
every selection on training data and scores the chosen procedure on data the
selection never saw. It runs no new simulation: the replays already cover seeds
7-16 of the main sweep.

Protocol
--------
1. Seed split. Select on seeds 7-11, evaluate on seeds 12-16. The reverse
   direction is reported as a check.
   (a) k: the fixed k of the sitting that maximises the pooled gain over the
       standard process (scripts/budget_split.py scoring, rho = 1);
   (b) sitting variant: the best of the 19 end-of-study procedures by the share
       of GAUSSIAN error's deployed cost recovered (replay_end_of_study.py), once
       over all 19 as the paper selected and once over the 11 at rho = 1;
   (c) ship rule: the best of best_mean / pm / lcb1 / lcb2 by the recovery of
       gaussian error's cost at >= 0.25 sigma (analyse_ship_rules.py);
   (d) shortlist m = 2, 3, 5 and the rank rule involve no selection on outcomes
       and are reported on the test seeds only.
2. Landscape split. 1000 random 10/10 partitions of the 20 landscapes, each used
   in both directions (2000 folds): select on 10 landscapes, evaluate on the other
   10. Once with every seed on both sides ("landscape") and once crossed ("crossed":
   seeds 7-11 of the training landscapes against seeds 12-16 of the test
   landscapes), the strictest hold-out these data allow.
3. Optimism = (train-selected choice's value on train) - (its value on test),
   averaged over splits. On random landscape splits a FIXED choice has expected
   optimism zero, so the mean is the selection bias itself.
4. Multiplicity. Two-sided Wilcoxon signed-rank tests of the 20 per-landscape
   gains over the standard process, on the TEST seeds only, each in the scope its
   headline number is reported in, Holm-adjusted over (i) the family the review
   names (sitting variants, ship rules, shortlist sizes, rank rule) and (ii) that
   family plus every other replayed arm the paper reports (the nine fixed k, the
   derived budget rule, the four LUCB sittings).
5. The derived budget rule against the best fixed k at rho = 1, with a paired
   landscape bootstrap on the difference; and against the TRAIN-selected k on the
   test seeds, which is the comparison that does not favour the fixed rule by
   letting it choose k on the data it is scored on.

Estimands are imported, not re-implemented, wherever the paper's scripts expose
them: budget_split.score_policies builds the k-curve's per-run frame,
analyse_boba_adaptations.summarise scores every recovery, and
analyse_ship_rules.twin_frame pairs every ship-rule run with its clean twin. The
full-data numbers must reproduce the published tables or the script stops.

    python scripts/heldout_remedies.py
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import analyse_boba_adaptations as aba  # noqa: E402
import analyse_ship_rules as asr  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
import budget_split as bs  # noqa: E402
import replay_end_of_study as eos  # noqa: E402

TRAIN_SEEDS = (7, 8, 9, 10, 11)
TEST_SEEDS = (12, 13, 14, 15, 16)
ALL_SEEDS = TRAIN_SEEDS + TEST_SEEDS
K_GRID = (2, 3, 5, 8, 12, 16, 20, 25, 30)
KCURVE_ACQS = ("logei", "qnei")
RESPONSE_MODELS = ("gaussian", "bias", "drift", "ar1")
SHIP_RULES = ("best_mean", "pm", "lcb1", "lcb2")
SHIP_POOL_MIN_STD = 0.25
# The rows the paper quotes.
PAPER_K = "fixed_k8"                                   # "+0.022 [+0.008, +0.034]"
PAPER_SITTING = "tournament_k5_lcb_rho0.5_look"        # "19.7% [13, 27]"
PAPER_SITTING_RHO1 = "tournament_k5_lcb_rho1_look"     # the same procedure at rho = 1
PAPER_SHIP_RULE = "lcb2"                               # "10.1% [4.6, 15.9]"
PAPER_RANK_RULE = "ordinal_lcb1"                       # "ship 1, chosen on ranks"
N_PARTITIONS = 1000
SPLIT_SEED = 20260922
VALUE_COLS = ("ref_noisy", "ref_clean", "trt_noisy", "trt_clean")
ATOL = 1e-9


# ---------------------------------------------------------------------------
# Loading: each family as one long frame of per-run values in opt_z units
# ---------------------------------------------------------------------------


def _normalise(df: pd.DataFrame, opt_z: dict[str, float]) -> pd.DataFrame:
    z = df["dataset"].map(opt_z).astype(float)
    if z.isna().any():
        raise KeyError(f"no opt_z for {sorted(df.loc[z.isna(), 'dataset'].unique())}")
    return df.assign(ref_noisy=df["ref_noisy"] / z, ref_clean=df["ref_clean"] / z,
                     trt_noisy=df["regret_noisy"] / z, trt_clean=df["regret_clean"] / z)


def load_kcurve(analysis: Path, rho: float, derived_name: str, opt_z: dict[str, float]):
    """budget_split.main's filters and join, then its own score_policies."""
    cols = ["dataset", "acquisition", "seed", "error_model", "file", "procedure", "family", "k",
            "candidates", "rho", "winner", "regret_noisy", "ref_noisy"]
    frames = [pd.read_csv(analysis / d / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
              for d in ("end_of_study_ksweep", "end_of_study_kwide")]
    sweep = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["file", "procedure"], keep="first")
    sweep = sweep[(sweep["family"] == "tournament") & (sweep["candidates"] == "lcb")
                  & (sweep["winner"] == "look") & (sweep["rho"] == rho)
                  & sweep["acquisition"].isin(KCURVE_ACQS) & sweep["error_model"].isin(RESPONSE_MODELS)]
    derived = pd.read_csv(analysis / derived_name)
    joined = sweep.merge(derived[["file", "k_hat", "gp_failed"]], on="file", how="inner")
    joined = joined[~eos._as_bool(joined["gp_failed"])]
    no_trial = pd.read_csv(analysis / "ship_rules_per_run.csv", usecols=["file", "regret_pm", "regret_lcb1"])
    no_trial["file"] = no_trial["file"].map(lambda f: Path(f).name)
    summary, frame, k_hat = bs.score_policies(joined, no_trial, opt_z, np.random.default_rng(bs.BOOTSTRAP_SEED))
    # Policy names as the published table spells them ("fixed_k8.0") and as this script does ("fixed_k8").
    rename = {c: f"fixed_k{int(float(c[len('fixed_k'):]))}" for c in frame.columns if c.startswith("fixed_k")}
    frame = frame.rename(columns=rename).reset_index()
    seeds = joined.drop_duplicates("file").set_index("file")["seed"]
    frame["seed"] = frame["file"].map(seeds).astype(int)
    summary = summary.assign(policy_short=summary["policy"].replace(rename))
    return summary, frame, k_hat


def kcurve_long(frame: pd.DataFrame) -> pd.DataFrame:
    """The k-curve's per-run frame as (procedure, standard, policy) rows; gain needs no clean twin."""
    policies = [c for c in frame.columns if c.startswith("fixed_k")] + ["derived", "always_lcb", "always_pm"]
    parts = [pd.DataFrame({"procedure": p, "dataset": frame["dataset"], "seed": frame["seed"],
                           "ref_noisy": frame["standard"], "ref_clean": np.nan,
                           "trt_noisy": frame[p], "trt_clean": np.nan}) for p in policies]
    return pd.concat(parts, ignore_index=True)


def load_sittings(analysis: Path, opt_z: dict[str, float]) -> pd.DataFrame:
    """The 19 end-of-study procedures, gaussian error, as replay_end_of_study.recovery_table sees them."""
    cols = ["arm", "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file",
            "procedure", "family", "k", "candidates", "rho", "winner", "regret_noisy", "regret_clean",
            "ref_noisy", "ref_clean"]
    df = pd.read_csv(analysis / "end_of_study" / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
    df = df[(df["arm"] == "output-boba") & df["family"].isin(eos.TREATMENT_FAMILIES)
            & (df["error_model"] == "gaussian")]
    return _normalise(df, opt_z)


def load_remedies(path: Path, opt_z: dict[str, float]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return _normalise(df[df["procedure"] != "standard"], opt_z)


def load_ship_rules(analysis: Path, opt_z: dict[str, float]) -> pd.DataFrame:
    """Each gaussian run's rule and standard regret with its clean twin, via analyse_ship_rules."""
    runs = asr.load_per_run(analysis / "ship_rules_per_run.csv")
    ref = asr.twin_frame(runs, asr.STANDARD)
    ref = ref[(ref["error_model"] == "gaussian") & (ref["variant"] == "")]
    parts = []
    for rule in SHIP_RULES:
        trt = asr.twin_frame(runs, rule)
        trt = trt[(trt["error_model"] == "gaussian") & (trt["variant"] == "")]
        parts.append(aba.paired_frame(ref, trt, asr.RESPONSE, opt_z, pool=False).assign(procedure=rule))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# A cube of per-(procedure, landscape, seed) sums, so any split is a cheap slice
# ---------------------------------------------------------------------------


class Cube:
    def __init__(self, long: pd.DataFrame):
        self.procs = sorted(long["procedure"].unique())
        self.lands = sorted(long["dataset"].unique())
        self.seeds = sorted(int(s) for s in long["seed"].unique())
        shape = (len(VALUE_COLS), len(self.procs), len(self.lands), len(self.seeds))
        self.S, self.N = np.zeros(shape), np.zeros(shape)
        g = long.groupby(["procedure", "dataset", "seed"])[list(VALUE_COLS)]
        sums, counts = g.sum(), g.count()
        ip = sums.index.get_level_values(0).map({p: i for i, p in enumerate(self.procs)}).to_numpy()
        il = sums.index.get_level_values(1).map({d: i for i, d in enumerate(self.lands)}).to_numpy()
        ik = sums.index.get_level_values(2).map({s: i for i, s in enumerate(self.seeds)}).to_numpy()
        for q, col in enumerate(VALUE_COLS):
            self.S[q, ip, il, ik] = sums[col].to_numpy()
            self.N[q, ip, il, ik] = counts[col].to_numpy()

    def landscape_means(self, seeds) -> np.ndarray:
        """[value column, procedure, landscape] means over the runs of the given seeds."""
        mask = np.isin(self.seeds, list(seeds))
        s, n = self.S[..., mask].sum(-1), self.N[..., mask].sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)

    def gain_cost(self, seeds) -> tuple[np.ndarray, np.ndarray]:
        m = self.landscape_means(seeds)
        return m[0] - m[2], m[0] - m[1]

    def balance(self, proc: str) -> str:
        """Runs per landscape and seed for one procedure: a hold-out needs every seed filled."""
        n = self.N[2, self.procs.index(proc)]
        present = [s for j, s in enumerate(self.seeds) if n[:, j].sum() > 0]
        vals = np.unique(n[:, [self.seeds.index(s) for s in present]])
        return f"seeds {present[0]}-{present[-1]} ({len(present)}), runs per landscape x seed {vals.tolist()}"


@dataclasses.dataclass
class Problem:
    """One selection a paper made: a candidate set, a metric, and the choice it reported."""
    name: str
    label: str
    cube: Cube
    candidates: list[str]
    metric: str                     # "gain" (k-curve units) or "recovery" (share of cost)
    paper_choice: str | None
    schemes: tuple[str, ...] = ("landscape", "crossed")

    def __post_init__(self):
        self.idx = [self.cube.procs.index(c) for c in self.candidates]
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def matrices(self, seeds):
        key = tuple(sorted(seeds))
        if key not in self._cache:
            g, c = self.cube.gain_cost(key)
            self._cache[key] = (g[self.idx], c[self.idx])
        return self._cache[key]

    def values(self, seeds, lands=None) -> np.ndarray:
        g, c = self.matrices(seeds)
        if lands is not None:
            g, c = g[:, lands], c[:, lands]
        if self.metric == "gain":
            return g.mean(axis=1)
        cost = c.mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cost > 0, g.mean(axis=1) / cost, np.nan)

    def pick(self, values: np.ndarray) -> int:
        if not np.isfinite(values).any():
            raise ValueError(f"{self.name}: no candidate has a value on the training data")
        return int(np.nanargmax(values))


# ---------------------------------------------------------------------------
# Intervals, with the paper's own conventions
# ---------------------------------------------------------------------------


def recovery_summary(long: pd.DataFrame, proc: str, seeds) -> dict:
    """analyse_boba_adaptations.summarise on one procedure's runs of the given seeds."""
    block = long[(long["procedure"] == proc) & long["seed"].isin(list(seeds))]
    if block.empty:
        return {"value": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n_landscapes": 0, "n_runs": 0}
    s = aba.summarise(block, np.random.default_rng(aba.BOOTSTRAP_SEED))
    return {"value": s["recovered"], "lo": s["recovered_lo"], "hi": s["recovered_hi"], "p": s["wilcoxon_p"],
            "gain": s["gain"], "cost": s["cost"], "n_landscapes": s["n_landscapes"], "n_runs": s["n_cells"]}


def gain_summary(problem: Problem, cand: str, seeds) -> dict:
    """budget_split._boot_mean_diff on the landscape means of one policy, with a fresh generator."""
    m = problem.cube.landscape_means(seeds)[:, problem.cube.procs.index(cand)]
    base, pol = m[0], m[2]
    v, lo, hi = bs._boot_mean_diff(base, pol, np.random.default_rng(bs.BOOTSTRAP_SEED))
    d = base - pol
    return {"value": v, "lo": lo, "hi": hi, "p": _wilcoxon_p(d), "n_landscapes": int(np.isfinite(d).sum())}


def summary_for(problem: Problem, long: pd.DataFrame | None, cand: str, seeds) -> dict:
    return gain_summary(problem, cand, seeds) if problem.metric == "gain" else recovery_summary(long, cand, seeds)


def _wilcoxon_p(gains: np.ndarray) -> float:
    g = np.asarray(gains, dtype=float)
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return np.nan
    if np.allclose(g, 0.0):
        return 1.0
    try:
        return float(wilcoxon(g).pvalue)
    except ValueError:
        return 1.0


# ---------------------------------------------------------------------------
# Reproduction of the published tables
# ---------------------------------------------------------------------------


def check_kcurve(summary: pd.DataFrame, published: Path, label: str) -> list[str]:
    pub = pd.read_csv(published).set_index("policy")
    mine = summary.set_index("policy")
    cols = ["gain_vs_standard", "gain_lo", "gain_hi", "mean_regret"]
    diff = (mine.loc[pub.index, cols] - pub[cols]).abs().to_numpy().max()
    if diff > ATOL:
        raise SystemExit(f"{label}: the k-curve does not reproduce {published} (max abs diff {diff:.2e})")
    return [f"{label}: {len(pub)} policies of {published.name} reproduced, max abs difference {diff:.1e}"]


def check_sittings(long: pd.DataFrame, published: Path) -> list[str]:
    pub = pd.read_csv(published)
    pub = pub[(pub["arm"] == "output-boba") & (pub["error_model"] == "gaussian") & (pub["scope"] == "pooled")]
    worst = 0.0
    for _, row in pub.iterrows():
        s = recovery_summary(long, row["procedure"], ALL_SEEDS)
        worst = max(worst, abs(s["value"] - row["recovered"]), abs(s["lo"] - row["recovered_lo"]),
                    abs(s["hi"] - row["recovered_hi"]))
    if worst > ATOL:
        raise SystemExit(f"sitting variants do not reproduce {published} (max abs diff {worst:.2e})")
    return [f"sitting variants: {len(pub)} pooled gaussian rows of {published.name} reproduced "
            f"(recovery and interval), max abs difference {worst:.1e}"]


def check_remedies(cube: Cube, published: Path, label: str) -> list[str]:
    pub = pd.read_csv(published)
    pub = pub[(pub["error_model"] == "pooled") & (pub["procedure"] != "standard")]
    g, c = cube.gain_cost(cube.seeds)
    worst = 0.0
    for _, row in pub.iterrows():
        i = cube.procs.index(row["procedure"])
        worst = max(worst, abs(g[i].mean() / c[i].mean() - row["recovered"]))
    if worst > ATOL:
        raise SystemExit(f"{label}: remedies do not reproduce {published} (max abs diff {worst:.2e})")
    # Its intervals come from one generator shared across rows, so only the point estimates are compared.
    return [f"{label}: {len(pub)} pooled recoveries of {published.name} reproduced, max abs difference {worst:.1e}"]


def check_ship(long_ge: pd.DataFrame, published: Path) -> list[str]:
    pub = pd.read_csv(published)
    pub = pub[(pub["arm"] == "output-boba") & (pub["error_model"] == "gaussian")
              & (pub["cell"] == f"pooled_std_ge_{SHIP_POOL_MIN_STD:g}") & pub["jitter_iteration"].isna()
              & pub["rule"].isin(SHIP_RULES)]
    worst = 0.0
    for _, row in pub.iterrows():
        s = recovery_summary(long_ge, row["rule"], ALL_SEEDS)
        worst = max(worst, abs(s["value"] - row["recovered"]), abs(s["lo"] - row["recovered_lo"]),
                    abs(s["hi"] - row["recovered_hi"]))
    if worst > ATOL:
        raise SystemExit(f"ship rules do not reproduce {published} (max abs diff {worst:.2e})")
    return [f"ship rules: {len(pub)} gaussian >= {SHIP_POOL_MIN_STD:g} sigma rows of {published.name} reproduced "
            f"(recovery and interval), max abs difference {worst:.1e}"]


# ---------------------------------------------------------------------------
# 1. The seed split
# ---------------------------------------------------------------------------


def seed_split(problem: Problem, long: pd.DataFrame | None, train, test, direction: str,
               full: dict[str, dict]) -> tuple[list[dict], dict]:
    tr, te = problem.values(train), problem.values(test)
    j = problem.pick(tr)
    j_test_best = problem.pick(te)
    rows = []
    for i, cand in enumerate(problem.candidates):
        s = summary_for(problem, long, cand, test)
        role = ",".join(r for r, ok in (("selected", i == j), ("paper_choice", cand == problem.paper_choice),
                                        ("full_best", cand == full["_best"]), ("test_best", i == j_test_best)) if ok)
        rows.append({"section": "1_seed_split", "problem": problem.name, "scheme": "seed", "direction": direction,
                     "candidate": cand, "role": role or "candidate", "metric": problem.metric,
                     "full_value": full[cand]["value"], "full_lo": full[cand]["lo"], "full_hi": full[cand]["hi"],
                     "train_value": tr[i], "test_value": s["value"], "test_lo": s["lo"], "test_hi": s["hi"],
                     "p_test": s["p"], "n_landscapes_test": s["n_landscapes"]})
    sel = problem.candidates[j]
    shift = float(np.nanmean(tr - te))
    head = {"problem": problem.name, "direction": direction, "selected": sel, "train_value": tr[j],
            "test": summary_for(problem, long, sel, test),
            "paper": summary_for(problem, long, problem.paper_choice, test) if problem.paper_choice else None,
            "test_best": problem.candidates[j_test_best], "test_best_value": te[j_test_best],
            "optimism": tr[j] - te[j], "shift": shift, "rank_on_test": int((te > te[j]).sum()) + 1,
            "n_candidates": int(np.isfinite(te).sum())}
    rows.append({"section": "1_seed_split", "problem": problem.name, "scheme": "seed", "direction": direction,
                 "candidate": sel, "role": "summary_selected", "metric": problem.metric,
                 "full_value": full[sel]["value"], "full_lo": full[sel]["lo"], "full_hi": full[sel]["hi"],
                 "train_value": tr[j], "test_value": head["test"]["value"], "test_lo": head["test"]["lo"],
                 "test_hi": head["test"]["hi"], "p_test": head["test"]["p"], "optimism": head["optimism"],
                 "seed_shift_mean_over_candidates": shift, "test_rank_of_selected": head["rank_on_test"],
                 "test_best_candidate": head["test_best"], "test_best_value": head["test_best_value"]})
    return rows, head


# ---------------------------------------------------------------------------
# 2-3. Landscape splits and optimism
# ---------------------------------------------------------------------------


def landscape_splits(problems: list[Problem], full_best: dict[str, str], n_partitions: int,
                     seed: int) -> pd.DataFrame:
    n_lands = {len(p.cube.lands) for p in problems}
    if n_lands != {20}:
        raise ValueError(f"every problem must have the 20 landscapes, found {n_lands}")
    lands0 = problems[0].cube.lands
    if any(p.cube.lands != lands0 for p in problems):
        raise ValueError("problems order their landscapes differently")
    rng = np.random.default_rng(seed)
    schemes = {"landscape": (ALL_SEEDS, ALL_SEEDS), "crossed": (TRAIN_SEEDS, TEST_SEEDS)}
    recs = []
    for part in range(n_partitions):
        perm = rng.permutation(20)
        halves = (np.sort(perm[:10]), np.sort(perm[10:]))
        for fold, (tr_l, te_l) in enumerate((halves, halves[::-1])):
            for prob in problems:
                fb = prob.candidates.index(full_best[prob.name])
                pc = prob.candidates.index(prob.paper_choice) if prob.paper_choice else None
                for scheme in prob.schemes:
                    tr_s, te_s = schemes[scheme]
                    tr, te = prob.values(tr_s, tr_l), prob.values(te_s, te_l)
                    tr_on_test_lands = prob.values(tr_s, te_l)
                    j = prob.pick(tr)
                    jt = prob.pick(te)
                    recs.append({
                        "problem": prob.name, "scheme": scheme, "partition": part, "fold": fold,
                        "train_landscapes": ",".join(lands0[i] for i in tr_l),
                        "selected": prob.candidates[j], "train_value": tr[j], "test_value": te[j],
                        # What train minus test is for a choice made WITHOUT looking: the
                        # baseline the selected choice's optimism is compared with.
                        "shift_mean_over_candidates": float(np.nanmean(tr - te)),
                        "full_best_train_value": tr[fb], "full_best_test_value": te[fb],
                        "paper_choice_test_value": te[pc] if pc is not None else np.nan,
                        "test_best": prob.candidates[jt], "test_best_value": te[jt],
                        # The same choice on the training SEEDS of the test landscapes: separates
                        # the landscape hold-out from the seed hold-out in the crossed scheme.
                        "selected_train_seeds_test_landscapes": tr_on_test_lands[j],
                    })
    return pd.DataFrame(recs)


def summarise_splits(splits: pd.DataFrame, full_best: dict[str, str], full_values: dict[str, dict],
                     paper_choice: dict[str, str | None]) -> tuple[list[dict], dict]:
    rows, heads = [], {}
    for (prob, scheme), g in splits.groupby(["problem", "scheme"], sort=False):
        opt = g["train_value"] - g["test_value"]
        net = opt - g["shift_mean_over_candidates"]
        fb_opt = g["full_best_train_value"] - g["full_best_test_value"]
        counts = g["selected"].value_counts(normalize=True)
        dist = ", ".join(f"{k}: {v:.1%}" for k, v in counts.items())
        fb = full_best[prob]
        h = {"problem": prob, "scheme": scheme, "n_folds": len(g),
             "test_median": g["test_value"].median(), "test_p2_5": g["test_value"].quantile(0.025),
             "test_p97_5": g["test_value"].quantile(0.975), "test_mean": g["test_value"].mean(),
             "train_mean": g["train_value"].mean(),
             "optimism_mean": opt.mean(), "optimism_p2_5": opt.quantile(0.025), "optimism_p97_5": opt.quantile(0.975),
             "optimism_net_of_shift_mean": net.mean(), "optimism_net_p2_5": net.quantile(0.025),
             "optimism_net_p97_5": net.quantile(0.975),
             "fixed_choice_optimism_mean": fb_opt.mean(),
             "full_best_test_p2_5": g["full_best_test_value"].quantile(0.025),
             "full_best_test_p97_5": g["full_best_test_value"].quantile(0.975),
             "share_selected_k5_to_16": (float(g["selected"].isin(["fixed_k5", "fixed_k8", "fixed_k12", "fixed_k16"]).mean())
                                         if prob == "k" else np.nan),
             "share_selected_full_best": float((g["selected"] == fb).mean()),
             "share_selected_paper_choice": (float((g["selected"] == paper_choice[prob]).mean())
                                             if paper_choice[prob] else np.nan),
             "full_best": fb, "full_best_value": full_values[prob][fb]["value"],
             "full_best_test_median": g["full_best_test_value"].median(),
             "selection_regret_on_test_mean": float((g["test_best_value"] - g["test_value"]).mean()),
             "selected_distribution": dist}
        heads[(prob, scheme)] = h
        rows.append({"section": "2_landscape_split", "problem": prob, "scheme": scheme, "role": "summary",
                     "candidate": fb, **{k: v for k, v in h.items() if k not in ("problem", "scheme")}})
    return rows, heads


# ---------------------------------------------------------------------------
# 4. Multiplicity on the test seeds
# ---------------------------------------------------------------------------


def holm_rows(tests: list[dict]) -> list[dict]:
    df = pd.DataFrame(tests)
    for fam, mask in (("core", df["family"] == "core"), ("all", df["family"].isin(("core", "extra")))):
        for split, col in (("test", "p_test"), ("full", "p_full")):
            ok = mask & df[col].notna()
            adj = np.full(len(df), np.nan)
            adj[ok.to_numpy()] = multipletests(df.loc[ok, col].to_numpy(), method="holm")[1]
            df[f"p_holm_{fam}_{split}"] = adj
    df["section"] = "4_holm"
    return df.to_dict("records")


def per_landscape_gains(cube: Cube, proc: str, seeds) -> np.ndarray:
    g, _ = cube.gain_cost(seeds)
    return g[cube.procs.index(proc)]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def f_gain(v, lo=None, hi=None) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    s = f"{v:+.3f}"
    if lo is not None and np.isfinite(lo):
        s += f" [{lo:+.3f}, {hi:+.3f}]"
    return s


def f_pct(v, lo=None, hi=None) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    s = f"{100 * v:+.1f}%"
    if lo is not None and np.isfinite(lo):
        s += f" [{100 * lo:+.0f}, {100 * hi:+.0f}]"
    return s


def f_val(metric: str, v, lo=None, hi=None) -> str:
    return f_gain(v, lo, hi) if metric == "gain" else f_pct(v, lo, hi)


def f_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def short(proc: str) -> str:
    if not proc.startswith("tournament_"):
        return proc
    return (proc.replace("tournament_", "T ").replace("_rho", " rho").replace("_look", " look")
            .replace("_post", " post").replace("_lcb", " lcb").replace("_obs", " obs"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=Path("output-boba/analysis"))
    p.add_argument("--output-dir", type=Path, default=None, help="default: <analysis-dir>/review")
    p.add_argument("--partitions", type=int, default=N_PARTITIONS,
                   help="random 10/10 landscape partitions, each used in both directions")
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    A = args.analysis_dir
    out_dir = args.output_dir or (A / "review")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    checks: list[str] = []

    # ---- load and reproduce -------------------------------------------------
    print("loading the k-sweep (rho = 1 and rho = 0.5) ...", flush=True)
    k_summary, k_frame, k_hat = load_kcurve(A, 1.0, "budget_split_derived.csv", opt_z)
    checks += check_kcurve(k_summary, A / "budget_split_policies.csv", "k-curve, rho = 1")
    k_summary05, k_frame05, k_hat05 = load_kcurve(A, 0.5, "budget_split_derived_rho0.5.csv", opt_z)
    checks += check_kcurve(k_summary05, A / "budget_split_policies_rho0.5.csv", "k-curve, rho = 0.5")
    k_cube = Cube(kcurve_long(k_frame))

    print("loading the sitting variants, remedies and ship rules ...", flush=True)
    sit_long = load_sittings(A, opt_z)
    checks += check_sittings(sit_long, A / "end_of_study" / "end_of_study_recovery.csv")
    sit_cube = Cube(sit_long)

    rem_long = load_remedies(A / "hitl_remedies" / "hitl_remedies_per_run.csv.gz", opt_z)
    rem_cube = Cube(rem_long)
    checks += check_remedies(rem_cube, A / "hitl_remedies" / "hitl_remedies_recovery.csv", "main sweep")
    extra_arms = {}
    for arm, label in (("output-boba-spike", "gross faults (spike)"), ("output-boba-ceiling", "capped scale")):
        path = Path(arm) / "analysis" / "hitl_remedies" / "hitl_remedies_per_run.csv.gz"
        if path.is_file():
            long = load_remedies(path, opt_z)
            cube = Cube(long)
            checks += check_remedies(cube, path.parent / "hitl_remedies_recovery.csv", label)
            extra_arms[arm] = (label, long, cube)

    ship_long_all = load_ship_rules(A, opt_z)
    ship_long = ship_long_all[ship_long_all["jitter_std"] >= SHIP_POOL_MIN_STD - 1e-12]
    checks += check_ship(ship_long, A / "ship_rules_recovery.csv")
    ship_cube, ship_cube_all = Cube(ship_long), Cube(ship_long_all)
    for c in checks:
        print("  check:", c)

    balance = {
        "k-curve (fixed k = 8)": k_cube.balance("fixed_k8"),
        "sitting variants (T k5 lcb rho1 look)": sit_cube.balance(PAPER_SITTING_RHO1),
        "sitting variant rerate3x2": sit_cube.balance("rerate3x2"),
        "ship rules (lcb2, gaussian >= 0.25 sigma)": ship_cube.balance("lcb2"),
        "remedies (shortlist_m3)": rem_cube.balance("shortlist_m3"),
    }
    for arm, (label, _, cube) in extra_arms.items():
        balance[f"{label} (ordinal_lcb1)"] = cube.balance(PAPER_RANK_RULE)

    # ---- the selection problems --------------------------------------------
    k_cands = [f"fixed_k{k}" for k in K_GRID]
    sit_all = [p for p in sit_cube.procs]
    sit_rho1 = [p for p in sit_all if "rho0.5" not in p]
    problems = [
        Problem("k", "k of the final sitting (gain, rho = 1)", k_cube, k_cands, "gain", PAPER_K),
        Problem("sitting19", "sitting variant, all 19 (gaussian recovery)", sit_cube, sit_all, "recovery",
                PAPER_SITTING),
        Problem("sitting_rho1", "sitting variant, the 11 at rho = 1 (gaussian recovery)", sit_cube, sit_rho1,
                "recovery", PAPER_SITTING_RHO1),
        Problem("ship_rule", "ship rule (gaussian recovery at >= 0.25 sigma)", ship_cube, list(SHIP_RULES),
                "recovery", PAPER_SHIP_RULE),
    ]
    longs = {"k": None, "sitting19": sit_long, "sitting_rho1": sit_long, "ship_rule": ship_long}
    for arm, (label, long, cube) in extra_arms.items():
        name = "rank_" + arm.replace("output-boba-", "")
        problems.append(Problem(name, f"rank-rule variant under {label}", cube, ["ordinal_lcb1", "ordinal_pm"],
                                "recovery", PAPER_RANK_RULE, schemes=("landscape",)))
        longs[name] = long

    # Full-data values of every candidate, and each problem's full-data winner.
    full: dict[str, dict] = {}
    kpub = k_summary.set_index("policy_short")
    for prob in problems:
        vals = {}
        for cand in prob.candidates:
            if prob.name == "k":
                r = kpub.loc[cand]
                vals[cand] = {"value": r["gain_vs_standard"], "lo": r["gain_lo"], "hi": r["gain_hi"]}
            else:
                vals[cand] = recovery_summary(longs[prob.name], cand, ALL_SEEDS)
        v = np.array([vals[c]["value"] for c in prob.candidates])
        vals["_best"] = prob.candidates[prob.pick(v)]
        full[prob.name] = vals
    full_best = {p.name: full[p.name]["_best"] for p in problems}
    print("full-data winners:", full_best, flush=True)

    rows: list[dict] = [{"section": "0_checks", "note": c} for c in checks]
    rows += [{"section": "0_balance", "problem": k, "note": v} for k, v in balance.items()]

    # ---- 1. seed split ------------------------------------------------------
    seed_heads = {}
    for prob in problems:
        if "crossed" not in prob.schemes:   # arms with seeds 7-11 only
            continue
        for direction, (tr, te) in (("7-11 -> 12-16", (TRAIN_SEEDS, TEST_SEEDS)),
                                    ("12-16 -> 7-11", (TEST_SEEDS, TRAIN_SEEDS))):
            r, h = seed_split(prob, longs[prob.name], tr, te, direction, full[prob.name])
            rows += r
            seed_heads[(prob.name, direction)] = h
    # (d) procedures with no outcome-based selection: test seeds only.
    fixed_rows = []
    for proc in ("shortlist_m1", "shortlist_m2", "shortlist_m3", "shortlist_m5", "ordinal_lcb1", "ordinal_pm",
                 "lucb_k4_rho1", "lucb_k8_rho1", "lucb_k4_rho0.5", "lucb_k8_rho0.5"):
        f_ = recovery_summary(rem_long, proc, ALL_SEEDS)
        t_ = recovery_summary(rem_long, proc, TEST_SEEDS)
        tr_ = recovery_summary(rem_long, proc, TRAIN_SEEDS)
        fixed_rows.append((proc, f_, tr_, t_))
        rows.append({"section": "1_seed_split", "problem": "no_selection_remedies", "scheme": "seed",
                     "direction": "test seeds 12-16 only", "candidate": proc, "role": "fixed", "metric": "recovery",
                     "full_value": f_["value"], "full_lo": f_["lo"], "full_hi": f_["hi"],
                     "train_value": tr_["value"], "test_value": t_["value"], "test_lo": t_["lo"],
                     "test_hi": t_["hi"], "p_test": t_["p"], "n_landscapes_test": t_["n_landscapes"]})
    # The ship rule's threshold: the same rules pooled over every magnitude, on the test seeds.
    ship_all_rows = []
    for rule in SHIP_RULES:
        f_ = recovery_summary(ship_long_all, rule, ALL_SEEDS)
        t_ = recovery_summary(ship_long_all, rule, TEST_SEEDS)
        ship_all_rows.append((rule, f_, t_))
        rows.append({"section": "1_seed_split", "problem": "ship_rule_all_magnitudes", "scheme": "seed",
                     "direction": "test seeds 12-16 only", "candidate": rule, "role": "fixed", "metric": "recovery",
                     "full_value": f_["value"], "full_lo": f_["lo"], "full_hi": f_["hi"], "test_value": t_["value"],
                     "test_lo": t_["lo"], "test_hi": t_["hi"], "p_test": t_["p"]})

    # ---- 2-3. landscape splits ----------------------------------------------
    print(f"landscape splits: {args.partitions} partitions x 2 folds ...", flush=True)
    splits = landscape_splits(problems, full_best, args.partitions, args.split_seed)
    split_rows, split_heads = summarise_splits(splits, full_best, full,
                                               {p.name: p.paper_choice for p in problems})
    rows += split_rows
    splits.to_csv(out_dir / "heldout_remedies_splits.csv.gz", index=False)

    # ---- 4. Holm on the test seeds -------------------------------------------
    tests = []

    def add_test(family, group, proc, cube, scope, label=None):
        g_test = per_landscape_gains(cube, proc, TEST_SEEDS)
        g_full = per_landscape_gains(cube, proc, cube.seeds)
        tests.append({"family": family, "group": group, "candidate": proc, "label": label or proc, "scope": scope,
                      "n_landscapes": int(np.isfinite(g_test).sum()),
                      "mean_gain_test": float(np.nanmean(g_test)) if np.isfinite(g_test).any() else np.nan,
                      "landscapes_gaining_test": int((g_test > 0).sum()),
                      "p_test": _wilcoxon_p(g_test), "p_full": _wilcoxon_p(g_full)})

    sit_scope = "gaussian, LogEI/qNEI/UCB, all magnitudes and onsets"
    for proc in sit_all:
        if not np.isfinite(per_landscape_gains(sit_cube, proc, TEST_SEEDS)).any():
            continue   # rerate3x2 was run on seeds 7-11 only
        add_test("core", "sitting variant", proc, sit_cube, sit_scope)
    for rule in SHIP_RULES:
        add_test("core", "ship rule", rule, ship_cube, "gaussian >= 0.25 sigma, ten acquisitions")
    rem_scope = "four error processes, LogEI/qNEI, all magnitudes and onsets"
    for proc in ("shortlist_m2", "shortlist_m3", "shortlist_m5"):
        add_test("core", "shortlist size", proc, rem_cube, rem_scope)
    for proc in ("ordinal_lcb1", "ordinal_pm"):
        add_test("core", "rank rule", proc, rem_cube, rem_scope)
    k_scope = "four error processes, LogEI/qNEI, all magnitudes and onsets, rho = 1"
    for cand in k_cands + ["derived"]:
        add_test("extra", "fixed k" if cand != "derived" else "derived budget rule", cand, k_cube, k_scope)
    for proc in ("lucb_k4_rho1", "lucb_k8_rho1", "lucb_k4_rho0.5", "lucb_k8_rho0.5"):
        add_test("extra", "LUCB sitting", proc, rem_cube, rem_scope)
    holm = holm_rows(tests)
    rows += holm

    # ---- 5. the derived budget rule -----------------------------------------
    d_rows = []

    def paired(policy_a: str, policy_b: str, seeds, cube=k_cube) -> tuple[float, float, float]:
        """Gain of a minus gain of b, landscape bootstrap on the paired difference."""
        m = cube.landscape_means(seeds)
        a, b = m[2, cube.procs.index(policy_a)], m[2, cube.procs.index(policy_b)]
        return bs._boot_mean_diff(b, a, np.random.default_rng(bs.BOOTSTRAP_SEED))

    kpub05 = k_summary05.set_index("policy_short")
    best_k = full_best["k"]
    k_sel_train = seed_heads[("k", "7-11 -> 12-16")]["selected"]
    comparisons = [
        ("full data, best fixed k vs derived", best_k, "derived", ALL_SEEDS),
        ("full data, fixed k = 16 vs derived", "fixed_k16", "derived", ALL_SEEDS),
        ("test seeds, train-selected k vs derived", k_sel_train, "derived", TEST_SEEDS),
        ("test seeds, fixed k = 8 vs derived", "fixed_k8", "derived", TEST_SEEDS),
    ]
    for label, a, b, seeds in comparisons:
        v, lo, hi = paired(a, b, seeds)
        d_rows.append((label, a, b, v, lo, hi))
        rows.append({"section": "5_derived_rule", "problem": "derived_vs_fixed", "candidate": a, "note": label,
                     "metric": "gain difference", "test_value": v, "test_lo": lo, "test_hi": hi})
    derived_test = gain_summary(Problem("derived", "derived budget rule", k_cube, ["derived"], "gain", None),
                                "derived", TEST_SEEDS)
    share_pm = {1.0: float((k_hat.astype(str) == "pm").mean()), 0.5: float((k_hat05.astype(str) == "pm").mean())}
    for rho, s in share_pm.items():
        rows.append({"section": "5_derived_rule", "problem": "derived_share_pm", "note": f"rho = {rho:g}",
                     "test_value": s})
    for pol in ("derived", "fixed_k8", "fixed_k16", "always_pm"):
        for rho, tab in ((1.0, kpub), (0.5, kpub05)):
            r = tab.loc[pol]
            rows.append({"section": "5_derived_rule", "problem": "published_rows", "candidate": pol,
                         "note": f"rho = {rho:g}", "full_value": r["gain_vs_standard"], "full_lo": r["gain_lo"],
                         "full_hi": r["gain_hi"]})

    # The rest of the sitting paragraph, at the rho it was printed at and at rho = 1.
    rec = pd.read_csv(A / "end_of_study" / "end_of_study_recovery.csv")
    para = []
    for label, arm, proc_tmpl, model, scope, std, onset in (
            ("gaussian, pooled", "output-boba", "tournament_k5_lcb_rho{r}_look", "gaussian", "pooled", None, None),
            ("gaussian, 5 sigma from trial 21", "output-boba", "tournament_k5_lcb_rho{r}_look", "gaussian", "cell",
             5.0, 20),
            ("slip, pooled", "output-boba-slip", "tournament_k5_lcb_rho{r}_look", "slip", "pooled", None, None),
            ("posterior winner, gaussian", "output-boba", "tournament_k5_lcb_rho{r}_post", "gaussian", "pooled",
             None, None),
            ("misclick, pooled", "output-boba-misclick", "tournament_k5_lcb_rho{r}_look", "misclick", "pooled",
             None, None)):
        vals = []
        for r in ("0.5", "1"):
            q = rec[(rec["arm"] == arm) & (rec["procedure"] == proc_tmpl.format(r=r)) & (rec["error_model"] == model)
                    & (rec["scope"] == scope)]
            if std is not None:
                q = q[(q["jitter_std"] == std) & (q["jitter_iteration"] == onset)]
            vals.append(q.iloc[0] if len(q) == 1 else None)
        para.append((label, *vals))

    # ---- write ---------------------------------------------------------------
    table = pd.DataFrame(rows)
    front = ["section", "problem", "scheme", "direction", "candidate", "role", "metric", "full_value", "full_lo",
             "full_hi", "train_value", "test_value", "test_lo", "test_hi", "p_test", "optimism"]
    table = table[[c for c in front if c in table] + [c for c in table.columns if c not in front]]
    table.to_csv(out_dir / "heldout_remedies.csv", index=False)

    md = render(problems, longs, full, full_best, seed_heads, fixed_rows, ship_all_rows, split_heads, holm,
                d_rows, derived_test, share_pm, kpub, kpub05, para, checks, balance, args)
    (out_dir / "heldout_remedies.md").write_text(md, encoding="utf-8")
    print(f"\nwrote {out_dir / 'heldout_remedies.csv'}, {out_dir / 'heldout_remedies.md'}, "
          f"{out_dir / 'heldout_remedies_splits.csv.gz'}")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def render(problems, longs, full, full_best, seed_heads, fixed_rows, ship_all_rows, split_heads, holm, d_rows,
           derived_test, share_pm, kpub, kpub05, para, checks, balance, args) -> str:
    P = {p.name: p for p in problems}
    sit_long, ship_long = longs["sitting19"], longs["ship_rule"]
    L = []
    w = L.append
    w("# Held-out evaluation of the selected remedies\n")
    w("Generated by `scripts/heldout_remedies.py`. Every procedure the remedy section *selects* from a "
      "family of replayed arms is re-selected on training data and scored on data the selection never saw. "
      "Seed split: select on seeds 7-11, evaluate on seeds 12-16 (the reverse direction is a check). "
      f"Landscape split: {args.partitions} random 10/10 partitions of the 20 landscapes, each used in both "
      f"directions ({2 * args.partitions} folds), once with all seeds on both sides and once crossed "
      "(seeds 7-11 of the training landscapes, seeds 12-16 of the test landscapes). Gains are in units of the "
      "achievable improvement (opt_z); recoveries are shares of the standard process's deployed cost of error, "
      "ratios of landscape means. Intervals are 95% landscape bootstraps with the paper's own code and seeds. "
      "No new simulation was run.\n")

    w("## Reproduction of the published full-data numbers\n")
    for c in checks:
        w(f"- {c}")
    w("\nRuns per landscape and seed (a seed hold-out needs seeds 12-16 filled):\n")
    for k, v in balance.items():
        w(f"- {k}: {v}")
    w("")

    # ---- 1a
    h = seed_heads[("k", "7-11 -> 12-16")]
    hr = seed_heads[("k", "12-16 -> 7-11")]
    pk = P["k"]
    tr = pk.values(TRAIN_SEEDS)
    te = pk.values(TEST_SEEDS)
    w("## 1a. The choice of k (seed split)\n")
    w(f"Selected on seeds 7-11: **k = {h['selected'][7:]}** (train gain {f_gain(h['train_value'])}). "
      f"On the held-out seeds 12-16 it gains **{f_gain(h['test']['value'], h['test']['lo'], h['test']['hi'])}**, "
      f"rank {h['rank_on_test']} of {h['n_candidates']} there. The full-data peak is k = {full_best['k'][7:]} at "
      f"{f_gain(full['k'][full_best['k']]['value'], full['k'][full_best['k']]['lo'], full['k'][full_best['k']]['hi'])}. "
      f"Optimism of the selection: {f_gain(h['optimism'])} (train minus test), against a seed-set shift of "
      f"{f_gain(h['shift'])} averaged over all nine k. Reverse direction (select on 12-16, test on 7-11): "
      f"k = {hr['selected'][7:]}, test gain {f_gain(hr['test']['value'], hr['test']['lo'], hr['test']['hi'])}.\n")
    trows = []
    for i, c in enumerate(pk.candidates):
        s_te = gain_summary(pk, c, TEST_SEEDS)
        mark = " (selected)" if c == h["selected"] else ""
        trows.append([c[7:] + mark, f_gain(full["k"][c]["value"], full["k"][c]["lo"], full["k"][c]["hi"]),
                      f_gain(tr[i]), f_gain(s_te["value"], s_te["lo"], s_te["hi"])])
    w(md_table(["k", "full data (published)", "train, seeds 7-11", "test, seeds 12-16"], trows))
    w("")

    # ---- 1b
    w("## 1b. The sitting variant (seed split)\n")
    for name in ("sitting19", "sitting_rho1"):
        h = seed_heads[(name, "7-11 -> 12-16")]
        hr = seed_heads[(name, "12-16 -> 7-11")]
        pr = P[name]
        paper = h["paper"]
        w(f"**{pr.label}.** Selected on seeds 7-11: `{h['selected']}` (train {f_pct(h['train_value'])}); on seeds "
          f"12-16 it recovers **{f_pct(h['test']['value'], h['test']['lo'], h['test']['hi'])}** of gaussian "
          f"error's cost (rank {h['rank_on_test']} of {h['n_candidates']} on test). The paper's variant "
          f"`{pr.paper_choice}` recovers {f_pct(paper['value'], paper['lo'], paper['hi'])} on the test seeds "
          f"(full data {f_pct(full[name][pr.paper_choice]['value'], full[name][pr.paper_choice]['lo'], full[name][pr.paper_choice]['hi'])}). "
          f"Optimism {100 * h['optimism']:+.1f} points, of which the seed-set shift is {100 * h['shift']:+.1f} "
          f"(net of it, {100 * (h['optimism'] - h['shift']):+.1f}). Reverse "
          f"direction: `{hr['selected']}`, test {f_pct(hr['test']['value'], hr['test']['lo'], hr['test']['hi'])}.\n")
    pr = P["sitting19"]
    tr, te = pr.values(TRAIN_SEEDS), pr.values(TEST_SEEDS)
    order = np.argsort(-np.array([full["sitting19"][c]["value"] for c in pr.candidates]))
    trows = []
    for i in order:
        c = pr.candidates[i]
        s_te = recovery_summary(sit_long, c, TEST_SEEDS)
        f = full["sitting19"][c]
        trows.append([short(c), f_pct(f["value"], f["lo"], f["hi"]), f_pct(tr[i]),
                      f_pct(s_te["value"], s_te["lo"], s_te["hi"])])
    w(md_table(["variant", "full data (published)", "train, seeds 7-11", "test, seeds 12-16"], trows))
    w("\nThe published 19.7% [13, 27] is the rho = 0.5 row (the sitting's looks carry half the idiosyncratic "
      "SD), pooled over LogEI, qNEI and UCB. rerate3x2 is a simulated arm run on seeds 7-11 only, so it has no "
      "test value.\n")

    # ---- 1c
    h = seed_heads[("ship_rule", "7-11 -> 12-16")]
    pr = P["ship_rule"]
    tr = pr.values(TRAIN_SEEDS)
    w("## 1c. The ship rule and its magnitude threshold (seed split)\n")
    w(f"Selected on seeds 7-11 (gaussian, >= 0.25 sigma, ten acquisitions): `{h['selected']}` "
      f"(train {f_pct(h['train_value'])}); test recovery "
      f"**{f_pct(h['test']['value'], h['test']['lo'], h['test']['hi'])}**. The published `lcb2` 10.1% [4.6, 15.9] "
      f"is {f_pct(h['paper']['value'], h['paper']['lo'], h['paper']['hi'])} on the test seeds.\n")
    trows = []
    all_map = {r: (f_, t_) for r, f_, t_ in ship_all_rows}
    for i, c in enumerate(pr.candidates):
        s_te = recovery_summary(ship_long, c, TEST_SEEDS)
        f = full["ship_rule"][c]
        fa, ta = all_map[c]
        trows.append([c, f_pct(f["value"], f["lo"], f["hi"]), f_pct(tr[i]), f_pct(s_te["value"], s_te["lo"], s_te["hi"]),
                      f_pct(fa["value"], fa["lo"], fa["hi"]), f_pct(ta["value"], ta["lo"], ta["hi"])])
    w(md_table(["rule", ">= 0.25 sigma, full", "train", "test", "all magnitudes, full", "all magnitudes, test"], trows))
    w("")

    # ---- 1d
    w("## 1d. Shortlist, rank rule and LUCB (no outcome-based selection): test seeds only\n")
    trows = []
    for proc, f_, tr_, t_ in fixed_rows:
        trows.append([proc, f_pct(f_["value"], f_["lo"], f_["hi"]), f_pct(tr_["value"]),
                      f_pct(t_["value"], t_["lo"], t_["hi"]), f_p(t_["p"])])
    w(md_table(["procedure", "full data (published)", "seeds 7-11", "test, seeds 12-16", "Wilcoxon p, test"], trows))
    w("\nMain sweep: four error processes, four magnitudes, both onsets, LogEI and qNEI. The rank rule the paper "
      "tables is `ordinal_lcb1` (\"ship 1, chosen on ranks\"). The gross-fault (spike) and capped-scale arms were "
      "run on seeds 7-11 only, so their rank-rule numbers (51%, 36%) cannot be held out by seed; section 2 holds "
      "them out by landscape.\n")

    # ---- 2
    w("## 2. Landscape splits\n")
    w("Distribution over folds of the TEST value of the train-selected choice (median and 2.5/97.5 "
      "percentiles), how often the train-selected choice equals the full-data winner, and the mean optimism. "
      "Test values are over 10 landscapes, so their spread includes the sampling noise of 10 landscapes, not "
      "just the selection.\n")
    trows = []
    for (prob, scheme), hh in split_heads.items():
        m = P[prob].metric
        fmt = f_gain if m == "gain" else f_pct
        trows.append([P[prob].label, scheme, f"{fmt(hh['test_median'])} [{fmt(hh['test_p2_5'])}, {fmt(hh['test_p97_5'])}]",
                      f"{hh['full_best']}: {fmt(hh['full_best_value'])}",
                      f"{fmt(hh['full_best_test_median'])} [{fmt(hh['full_best_test_p2_5'])}, {fmt(hh['full_best_test_p97_5'])}]",
                      f"{hh['share_selected_full_best']:.0%}",
                      "" if not np.isfinite(hh["share_selected_paper_choice"]) else f"{hh['share_selected_paper_choice']:.0%}",
                      fmt(hh["optimism_mean"]), fmt(hh["optimism_net_of_shift_mean"])])
    w(md_table(["selection", "scheme", "train-selected choice, test value: median [2.5, 97.5]", "full-data winner",
                "full-data winner, test value: median [2.5, 97.5]", "selects full-data winner",
                "selects paper's choice", "mean optimism", "mean optimism net of seed shift"], trows))
    w("\nThe fifth column is the same statistic for the full-data winner held fixed, so the gap between the "
      "third and fifth columns is what choosing on 10 landscapes costs on held-out ones.\n")
    w("Which choice the training half selects:\n")
    for (prob, scheme), hh in split_heads.items():
        extra = (f" (within k = 5-16, the flat top of the curve: {hh['share_selected_k5_to_16']:.1%})"
                 if prob == "k" else "")
        w(f"- {P[prob].label}, {scheme}: {hh['selected_distribution']}{extra}")
    w("")

    # ---- 3
    w("## 3. Selection bias (optimism)\n")
    w("Optimism = value of the train-selected choice on the training data minus its value on the test data. "
      "Part of any train-test gap is not selection: seeds 12-16 simply score differently from seeds 7-11 for "
      "every candidate (the seed-set shift, the mean of train minus test over all candidates). The net column "
      "subtracts it. On random landscape splits the shift averages to zero by symmetry, so the landscape "
      "scheme's mean optimism is the selection bias of picking the best of the family on 10 landscapes; "
      "picking on 20, as the paper did, is biased by less.\n")
    trows = []
    for name in ("k", "sitting19", "sitting_rho1", "ship_rule"):
        m = P[name].metric
        fmt = f_gain if m == "gain" else f_pct
        hs = seed_heads[(name, "7-11 -> 12-16")]
        hsr = seed_heads[(name, "12-16 -> 7-11")]
        hl = split_heads[(name, "landscape")]
        hc = split_heads[(name, "crossed")]
        trows.append([P[name].label, f"{fmt(hs['optimism'])} (net {fmt(hs['optimism'] - hs['shift'])})",
                      f"{fmt(hsr['optimism'])} (net {fmt(hsr['optimism'] - hsr['shift'])})",
                      f"{fmt(hl['optimism_mean'])} [{fmt(hl['optimism_p2_5'])}, {fmt(hl['optimism_p97_5'])}]",
                      f"{fmt(hc['optimism_mean'])} (net {fmt(hc['optimism_net_of_shift_mean'])})"])
    for (prob, scheme), hh in split_heads.items():
        if prob.startswith("rank_"):
            trows.append([P[prob].label, "n/a", "n/a",
                          f"{f_pct(hh['optimism_mean'])} [{f_pct(hh['optimism_p2_5'])}, {f_pct(hh['optimism_p97_5'])}]",
                          "n/a"])
    w(md_table(["selection", "seed split 7-11 -> 12-16", "reverse 12-16 -> 7-11",
                "landscape split, mean [2.5, 97.5]", "crossed split, mean"], trows))
    w("")

    # ---- 4
    hdf = pd.DataFrame(holm)
    w("## 4. Multiplicity: Holm over the family, test seeds only\n")
    w("Two-sided Wilcoxon signed-rank test of the 20 per-landscape gains over the standard process, each "
      "procedure in the scope its headline number uses, on seeds 12-16 only. Holm over the **core** family "
      f"the review names ({int((hdf['family'] == 'core').sum())} tests: the sitting variants that have test "
      "seeds, four ship rules, shortlist m = 2, 3, 5, two rank rules) and over **all** replayed arms the paper "
      f"reports ({len(hdf)} tests: adds the nine fixed k, the derived rule and four LUCB sittings). The last "
      "column repeats the all-family Holm on the full data for contrast.\n")
    trows = []
    for _, r in hdf.iterrows():
        fmt = f_gain
        trows.append([r["group"], short(r["candidate"]), fmt(r["mean_gain_test"]), f"{int(r['landscapes_gaining_test'])}/20",
                      f_p(r["p_test"]), f_p(r["p_holm_core_test"]) if r["family"] == "core" else "",
                      f_p(r["p_holm_all_test"]), f_p(r["p_holm_all_full"])])
    w(md_table(["group", "procedure", "mean gain, test", "landscapes gaining", "p, test", "Holm core, test",
                "Holm all, test", "Holm all, full data"], trows))
    surv_core = hdf[(hdf["family"] == "core") & (hdf["p_holm_core_test"] < 0.05)]
    surv_all = hdf[hdf["p_holm_all_test"] < 0.05]

    def _named(block):
        return ", ".join(f"`{r.candidate}`" + (" (a significant LOSS)" if r.mean_gain_test < 0 else "")
                         for r in block.itertuples()) or "none"

    w(f"\nSurvive Holm at 0.05 on the test seeds, core family ({len(surv_core)}): {_named(surv_core)}.")
    w(f"Survive Holm at 0.05 on the test seeds, all replayed arms ({len(surv_all)}): {_named(surv_all)}.")
    n_core, n_all = int((hdf["family"] == "core").sum()), len(hdf)
    w(f"With 20 landscapes the smallest two-sided Wilcoxon p is 1.9e-06, and Holm's first threshold is "
      f"0.05/{n_core} = {0.05 / n_core:.4f} (core) or 0.05/{n_all} = {0.05 / n_all:.4f} (all), so a procedure "
      "must gain on nearly every landscape to survive.\n")

    # ---- 5
    w("## 5. The derived budget rule against a fixed k\n")
    w("The paper's sentence compares the derived rule's \"+0.004 [-0.013, +0.021]\" with \"the fixed rule's "
      "+0.051\". Both are the rho = 0.5 numbers (`budget_split_derived_rho0.5.csv`, "
      "`budget_split_policies_rho0.5.csv`: derived "
      f"{f_gain(kpub05.loc['derived', 'gain_vs_standard'], kpub05.loc['derived', 'gain_lo'], kpub05.loc['derived', 'gain_hi'])}, "
      f"fixed k = 16 {f_gain(kpub05.loc['fixed_k16', 'gain_vs_standard'], kpub05.loc['fixed_k16', 'gain_lo'], kpub05.loc['fixed_k16', 'gain_hi'])}), "
      "left over from before the sitting moved to rho = 1; the rest of the section is at rho = 1, which is why "
      "+0.051 lies outside the k-curve peak's interval. The \"65% of runs\" in which the rule picks the "
      f"posterior-mean design is also rho = 0.5 ({share_pm[0.5]:.1%}); at rho = 1 it is {share_pm[1.0]:.1%}. "
      f"`always_pm` ({f_gain(kpub.loc['always_pm', 'gain_vs_standard'])}) does not depend on rho.\n")
    trows = [["derived rule, rho = 1, full data",
              f_gain(kpub.loc['derived', 'gain_vs_standard'], kpub.loc['derived', 'gain_lo'], kpub.loc['derived', 'gain_hi'])],
             ["best fixed k (k = 8), rho = 1, full data",
              f_gain(kpub.loc['fixed_k8', 'gain_vs_standard'], kpub.loc['fixed_k8', 'gain_lo'], kpub.loc['fixed_k8', 'gain_hi'])],
             ["fixed k = 16, rho = 1, full data",
              f_gain(kpub.loc['fixed_k16', 'gain_vs_standard'], kpub.loc['fixed_k16', 'gain_lo'], kpub.loc['fixed_k16', 'gain_hi'])],
             ["derived rule, rho = 1, test seeds 12-16", f_gain(derived_test["value"], derived_test["lo"], derived_test["hi"])]]
    for label, a, b, v, lo, hi in d_rows:
        trows.append([f"difference: {label} ({a} minus {b})", f_gain(v, lo, hi)])
    w(md_table(["quantity", "gain over the standard process [95% CI]"], trows))
    w("\nThe paired interval resamples landscapes and takes the difference within each, so it is narrower than "
      "the difference of the two marginal intervals. The test-seed row compares the derived rule with the k "
      "chosen on seeds 7-11, which removes the fixed rule's advantage of choosing k on the data it is scored "
      "on.\n")

    w("### The rest of the sitting paragraph at rho = 1\n")
    w("Every number in the \"Ship the fresh look\" paragraph is the rho = 0.5 row. The same rows at rho = 1 "
      "(from `end_of_study_recovery.csv`, full data):\n")
    trows = []
    for label, r05, r1 in para:
        trows.append([label, f_pct(r05["recovered"], r05["recovered_lo"], r05["recovered_hi"]) if r05 is not None else "n/a",
                      f_pct(r1["recovered"], r1["recovered_lo"], r1["recovered_hi"]) if r1 is not None else "n/a"])
    w(md_table(["top five by the cautious rule, ship the fresh look", "rho = 0.5 (printed)", "rho = 1"], trows))
    w("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
