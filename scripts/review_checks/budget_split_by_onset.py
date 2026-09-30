"""The derived budget rule by onset, and with a decision point after the onset.

scripts/budget_split.py derives k (the size of a final comparative sitting) per
run. Its decide-once form decides once, at n0 = T - max(k grid). The paper's grid
runs to k = 30 of T = 50, so n0 = 20, and the simulator corrupts trial t only
when t > onset: at the late onset (error from trial 21) the rule decides on
twenty exact ratings, before any error exists, and cannot tell the four
magnitudes apart. Its pooled deficit against a fixed k mixes those blind cells
with the cells where it can see the error (onset 0).

This check reports, from the rule's own outputs and the replays it is scored
against (no refitting, no simulation):

  1. where the decide-once rule decides, and that it is blind at onset 20: the
     sitting SD it assumes, the policy it picks and the size of the sittings
     it buys, by onset and magnitude;
  2. the deployed-design gain over the standard process of the derived rule,
     of fixed k = 8 and k = 12 and of the two no-trial ship rules, and the
     derived rule's gain minus each, by onset, on all seeds and on the seed
     halves (seeds 12-16 are the held-out seeds of heldout_remedies.py, whose
     train-selected k is 12), and (rho = 1, all seeds) within each onset x
     magnitude cell; (rho = 1) the same pooled comparison without the 5 sigma
     cell from trial 21, which carries the pooled gain of k = 12, and every
     policy's gain at 0.05 sigma, where the error costs least (not a price:
     the price, from the replays' clean twins, is item 9);
  3. the same for the two recomputations with a decision point after the
     onset: the grid capped at k = 25 (one decision at n0 = 25) and the
     sequential rule (a decision at each T - k, budget_split.py --decision
     sequential --output-suffix _decision_sequential), where their outputs
     exist;
  4. the same split by landscape group (the six laws, on which the loop's GP
     fits; the ten rugged classical functions, on which the paper finds it
     misspecified; the four others), and the Spearman correlation across the
     20 landscapes of the rule's paired gain with two measures of the GP's fit
     (review/gp_diagnostics.csv), to test whether the deficit is where the
     posterior is misspecified;
  5. that the pooled held-out numbers reproduce heldout_remedies.csv;
  6. what the rule's search term reads at each decision point: the run's recent
     gain per trial in its best rating, against the same estimator on the true
     value of the same visited designs (the run logs of output-boba), by onset
     and magnitude, to test whether rating error inflates the progress the rule
     credits to searching on;
  7. (rho = 1) a counterfactual that replaces every posterior-mean choice by
     the lcb ship rule, which the rule can never choose (the posterior-mean
     pick always scores at least as high under the posterior it is scored
     by), to size what that one comparison costs; and, for the two
     fixed-decision rules, a clairvoyant counterfactual that re-ranks the
     stored scores with the search credit read from the best true value, to
     test whether the inflated rate of 6 drives the late-onset choices;
  8. the same clairvoyant credit refitted (budget_split.py --rate-source
     truth) for the capped grid, which validates the re-ranking of 7, and for
     the sequential rule, whose path cannot be re-ranked from stored scores.
     These are diagnostics, never rules: they read the true objective;
  9. the price of the decide-once and the sequential rule, trt_clean - ref_clean:
     each rule derived on the clean twins and charged the clean twin's replayed
     regret of what it chose (budget_split.py --twin clean, suffixes _clean and
     _decision_sequential_clean), beside the same price for fixed k and the
     no-trial rules.

Units: regret / opt_z, averaged within a landscape and then over the 20
landscapes (budget_split.score_policies); intervals are 2000-draw landscape
bootstraps, each from a fresh generator (budget_split.score_by_onset). The
landscapes share random numbers by design and the bootstrap treats them as
independent clusters. The metric is the deployed design's regret at T = 50.

Inputs (read-only), all under output-boba/analysis:
  end_of_study_ksweep/, end_of_study_kwide/   the replays (per-run regret of every (run, k))
  ship_rules_per_run.csv                      the no-trial policies
  budget_split_derived[_<variant>][_rho0.5].csv  the rule's choices
  budget_split_price[_decision_sequential]_clean.csv  the rules' prices (9)
  review/heldout_remedies.csv, .md            for the reproduction check
  review/smooth_subset_definition.csv         the landscape groups
  review/gp_diagnostics.csv                   the GP's fit per landscape
and output-boba/<landscape>/<run>.csv, the run logs (for 6 only).

    python scripts/review_checks/budget_split_by_onset.py

writes output-boba/analysis/review/register_checks/budget_split_by_onset.txt and
prints the same text.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import budget_split as bs  # noqa: E402

A = REPO / "output-boba" / "analysis"
OUT_TXT = A / "review" / "register_checks" / "budget_split_by_onset.txt"
DIRS = ",".join(str(A / d) for d in ("end_of_study_ksweep", "end_of_study_kwide"))
T = 50
SEED_SETS = bs.SEED_SETS
# (label, derived file, rho). Missing files are reported and skipped.
VARIANTS = (
    ("decide-once rule: one decision at n0 = T - 30 = 20", "budget_split_derived.csv", 1.0),
    ("grid capped at k = 25: one decision at n0 = 25", "budget_split_derived_kmax25.csv", 1.0),
    ("sequential rule: a decision at each T - k", "budget_split_derived_decision_sequential.csv", 1.0),
    ("decide-once rule, rho = 0.5", "budget_split_derived_rho0.5.csv", 0.5),
    ("grid capped at k = 25, rho = 0.5", "budget_split_derived_kmax25_rho0.5.csv", 0.5),
    ("sequential rule, rho = 0.5", "budget_split_derived_decision_sequential_rho0.5.csv", 0.5),
    # Clairvoyant diagnostics (budget_split.py --rate-source truth), never rules.
    ("DIAGNOSTIC grid capped at k = 25, search credit from the true value",
     "budget_split_derived_kmax25_truerate.csv", 1.0),
    ("DIAGNOSTIC sequential rule, search credit from the true value",
     "budget_split_derived_decision_sequential_truerate.csv", 1.0),
)
# Which rule's stored scores each diagnostic re-ranks, to validate clairvoyant_search_credit.
RERANK_OF = {"budget_split_derived_kmax25_truerate.csv": "budget_split_derived_kmax25.csv"}
COMPARATORS = bs.COMPARATORS


def fmt(v, lo=None, hi=None) -> str:
    # Five decimals, so a number the paper rounds to three is never ambiguous.
    if lo is None:
        return f"{v:+.5f}"
    return f"{v:+.5f} [{lo:+.5f}, {hi:+.5f}]"


def load(derived_name: str, rho: float, sweeps: dict, no_trial, opt_z):
    if rho not in sweeps:
        sweeps[rho] = bs.load_sweep(DIRS, rho)
    derived = pd.read_csv(A / derived_name)
    joined, n_failed = bs.join_derived(sweeps[rho], derived)
    return derived, joined, n_failed


def grid_of(derived: pd.DataFrame) -> list[int]:
    """The sitting sizes the rule could choose. The fixed rule scores the whole grid in every
    row; the sequential rule records only the sizes still open where it committed, so the
    grid is the union over runs."""
    keys = set()
    for text in derived["scores"].dropna():
        keys.update(json.loads(text).keys())
    return sorted(int(k) for k in keys if k not in bs.NO_TRIAL_POLICIES)


def base_key(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per run with the fields that name its cell; magnitude separated from the rest."""
    runs = joined.drop_duplicates("file")[["file", "dataset", "acquisition", "seed", "error_model",
                                          "jitter_std", bs.ONSET_COLUMN, "k_hat"]].copy()
    runs["k_hat"] = runs["k_hat"].map(bs.choice_name)
    return runs


def blindness(derived: pd.DataFrame, joined: pd.DataFrame) -> None:
    runs = base_key(joined).merge(derived[["file", "sitting_sd_used"]], on="file", how="left")
    print("  median sitting SD the rule assumes (objective units), by onset x magnitude:")
    med = runs.pivot_table(index=bs.ONSET_COLUMN, columns="jitter_std", values="sitting_sd_used", aggfunc="median")
    print("    " + med.to_string(float_format=lambda x: f"{x:.4f}").replace("\n", "\n    "))
    print("  share of runs that ship the posterior-mean design (k_hat = pm), by onset x magnitude:")
    pm = runs.assign(pm=runs["k_hat"] == "pm").pivot_table(index=bs.ONSET_COLUMN, columns="jitter_std",
                                                             values="pm", aggfunc="mean")
    print("    " + pm.to_string(float_format=lambda x: f"{x:.3f}").replace("\n", "\n    "))
    k_num = pd.to_numeric(runs["k_hat"], errors="coerce")
    sizes = runs.assign(k_num=k_num, sitting=k_num.notna()).groupby([bs.ONSET_COLUMN, "jitter_std"])
    summary = pd.DataFrame({
        "share_sitting": sizes["sitting"].mean(),
        "median_k_of_sittings": sizes["k_num"].median(),
        "share_of_sittings_k_ge_8": sizes["k_num"].apply(lambda s: float((s.dropna() >= 8).mean())
                                                         if s.notna().any() else float("nan")),
    })
    print("  the sittings it buys, by onset x magnitude (share of runs that buy one; among those, the "
          "median k and the share with k >= 8):")
    print("    " + summary.to_string(float_format=lambda x: f"{x:.3f}").replace("\n", "\n    "))
    keys = ["dataset", "acquisition", "seed", "error_model", bs.ONSET_COLUMN]
    same = runs.groupby(keys)["k_hat"].agg(lambda s: s.nunique() == 1 and len(s) == 4)
    by_onset = same.groupby(level=bs.ONSET_COLUMN).mean()
    print("  share of base runs (landscape x acquisition x seed x process x onset) whose choice is the same "
          "at all four magnitudes: " + ", ".join(f"onset {int(o)} {v:.3f}" for o, v in by_onset.items()))
    # The no-trial scores and the sitting SD do not depend on the Monte Carlo stream: if they agree at
    # all four magnitudes the rule saw the same fit, and a differing choice comes from the file-seeded
    # draws of the sitting utilities alone.
    scores = derived.set_index("file")["scores"].dropna().map(json.loads)
    fit = runs.assign(
        pm_score=runs["file"].map(lambda x: scores[x]["pm"] if x in scores.index else np.nan),
        lcb_score=runs["file"].map(lambda x: scores[x]["lcb"] if x in scores.index else np.nan))
    spread = fit.groupby(keys)[["sitting_sd_used", "pm_score", "lcb_score"]].agg(
        lambda s: float(np.ptp(s.to_numpy())) if len(s) == 4 else np.nan)
    identical = (spread.max(axis=1) < 1e-9).groupby(level=bs.ONSET_COLUMN).mean()
    print("  share of base runs whose fit is identical at all four magnitudes (same sitting SD and no-trial "
          "scores to 1e-9): " + ", ".join(f"onset {int(o)} {v:.3f}" for o, v in identical.items()))


def sequential_detail(derived: pd.DataFrame, joined: pd.DataFrame) -> None:
    runs = base_key(joined).merge(derived[["file", "decided_at", "n_fit_failures"]], on="file", how="left")
    print(f"  fits that failed at some decision point: {int(runs['n_fit_failures'].sum())} "
          f"(runs dropped for failing at every point: see above)")
    tab = runs.pivot_table(index=bs.ONSET_COLUMN, columns="decided_at", values="file", aggfunc="count", fill_value=0)
    share = tab.div(tab.sum(axis=1), axis=0)
    print("  where the rule decided (trial n = T - k at which it committed; T - 2 = 48 also when it shipped "
          "with no sitting), share of runs by onset:")
    print("    " + share.to_string(float_format=lambda x: f"{x:.3f}").replace("\n", "\n    "))
    for onset in sorted(runs[bs.ONSET_COLUMN].unique()):
        sub = runs[runs[bs.ONSET_COLUMN] == onset]
        print(f"  onset {int(onset)}: {float((sub['decided_at'] <= onset).mean()):.3f} of runs committed at a "
              f"trial n <= onset, i.e. before any error had been rated")


def by_onset_tables(joined: pd.DataFrame, no_trial, opt_z) -> pd.DataFrame:
    # The same table budget_split.py writes to budget_split_by_onset<suffix>.csv.
    return bs.by_onset_tables(joined, no_trial, opt_z, comparators=COMPARATORS, seed_sets=SEED_SETS)


def print_by_onset(table: pd.DataFrame) -> None:
    for seeds_label, _ in SEED_SETS:
        print(f"  seeds {seeds_label}:")
        for onset in ("all", "0", "20"):
            b = table[(table["seeds"] == seeds_label) & (table["onset"] == onset)]
            if b.empty:
                continue
            g = b[b["kind"] == "gain"].set_index("policy")
            d = b[b["kind"] == "derived_minus"].set_index("policy")
            share = b[b["kind"] == "choice_share"].set_index("policy")["estimate"]
            n_runs = int(b["n_runs"].iloc[0])
            print(f"    onset {onset:>3} ({n_runs:,} runs, {int(b['n_landscapes'].iloc[0])} landscapes): "
                  f"pm share {share.get('pm', 0.0):.3f}")
            for pol in ("derived", "fixed_k8", "fixed_k12", "fixed_k16", "always_pm", "always_lcb", "oracle"):
                if pol in g.index:
                    r = g.loc[pol]
                    print(f"      gain {pol:<11} {fmt(r['estimate'], r['lo'], r['hi'])}")
            for pol in COMPARATORS:
                if pol in d.index:
                    r = d.loc[pol]
                    print(f"      derived minus {pol:<11} {fmt(r['estimate'], r['lo'], r['hi'])}")


HELD_OUT = (12, 13, 14, 15, 16)


def without_late_5sigma(joined: pd.DataFrame, no_trial, opt_z) -> None:
    """The comparison without the one cell, 5 sigma from trial 21, that carries the pooled gain
    of k = 12 (Section 6 quotes k = 12's pooled gain without it). The cell is dropped before
    the landscape means are taken, so each remaining cell keeps its weight."""
    late_5 = (joined[bs.ONSET_COLUMN] == 20) & np.isclose(joined["jitter_std"], 5.0)
    sub = joined[~late_5]
    for label, seeds in (("all", None), ("12-16", HELD_OUT)):
        t = bs.score_by_onset(sub, no_trial, opt_z, comparators=("fixed_k8", "fixed_k12"), seeds=seeds,
                              seeds_label=label)
        for onset in ("all", "20"):
            b = t[t["onset"] == onset]
            g = b[b["kind"] == "gain"].set_index("policy")
            d = b[b["kind"] == "derived_minus"]
            gains = "; ".join(f"{pol} {fmt(g.loc[pol, 'estimate'], g.loc[pol, 'lo'], g.loc[pol, 'hi'])}"
                              for pol in ("derived", "fixed_k8", "fixed_k12") if pol in g.index)
            text = "; ".join(f"derived minus {r.policy} {fmt(r.estimate, r.lo, r.hi)}" for r in d.itertuples(index=False))
            where = "both onsets" if onset == "all" else "onset 20 (0.05, 0.25, 1 sigma)"
            print(f"    seeds {label:>5}, {where} ({int(b['n_runs'].iloc[0]):,} runs): gain {gains}; {text}")


def near_clean(joined: pd.DataFrame, no_trial, opt_z, sd: float = 0.05) -> None:
    """Every policy's gain over the standard process at the smallest magnitude, where the
    error costs least. Not a price: the price, trt_clean - ref_clean, is item 9 of the module
    docstring (budget_split.py --twin clean). Pooled over both onsets and by onset."""
    sub = joined[np.isclose(joined["jitter_std"], sd)]
    for label, seeds in (("all", None), ("12-16", HELD_OUT)):
        t = bs.score_by_onset(sub, no_trial, opt_z, comparators=("fixed_k8", "fixed_k12"), seeds=seeds,
                              seeds_label=label)
        for onset in ("all", "0", "20"):
            b = t[t["onset"] == onset]
            g = b[b["kind"] == "gain"].set_index("policy")
            gains = "; ".join(f"{pol} {fmt(g.loc[pol, 'estimate'], g.loc[pol, 'lo'], g.loc[pol, 'hi'])}"
                              for pol in ("derived", "fixed_k8", "fixed_k12", "always_pm", "always_lcb")
                              if pol in g.index)
            print(f"    seeds {label:>5}, {sd:g} sigma, onset {onset:>3}: gain {gains}")


def by_magnitude(joined: pd.DataFrame, no_trial, opt_z) -> None:
    """The rule's gain and its gain minus fixed k within each onset x magnitude cell (all seeds)."""
    for sd in sorted(joined["jitter_std"].unique()):
        t = bs.score_by_onset(joined[joined["jitter_std"] == sd], no_trial, opt_z,
                              comparators=("fixed_k8", "fixed_k12", "always_pm"))
        for onset in ("0", "20"):
            b = t[t["onset"] == onset]
            g = b[b["kind"] == "gain"].set_index("policy")
            d = b[b["kind"] == "derived_minus"]
            share = b[b["kind"] == "choice_share"].set_index("policy")["estimate"].get("pm", 0.0)
            text = "; ".join(f"minus {r.policy} {fmt(r.estimate, r.lo, r.hi)}" for r in d.itertuples(index=False))
            r = g.loc["derived"]
            print(f"    {sd:g} sigma, onset {onset:>2}: derived {fmt(r['estimate'], r['lo'], r['hi'])} "
                  f"(k=12 {g.loc['fixed_k12', 'estimate']:+.4f}), pm share {share:.3f}; {text}")


# Every decision point n = T - k of the paper's grid (budget_split.decision_points).
SEARCH_POINTS = tuple(n for _, n in bs.decision_points((2, 3, 5, 8, 12, 16, 20, 25, 30), T))
_RATES: dict = {}


def run_rates(joined: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Per run and decision point n: the recent gain per trial in the best rating (exactly the
    rule's budget_split.recent_improvement_rate on the log's ratings) and the same estimator on
    the true values of the same visited designs, in objective units. Read once from the logs."""
    import replay_end_of_study as eos

    key = (window, tuple(sorted(joined["file"].unique())))
    if key in _RATES:
        return _RATES[key]
    rows = []
    for r in base_key(joined).itertuples(index=False):
        run = eos.read_run(REPO / "output-boba" / r.dataset / Path(r.file).name, T)
        for n in SEARCH_POINTS:
            rows.append({"file": r.file, "dataset": r.dataset, "process": r.error_model,
                         "onset": int(getattr(r, bs.ONSET_COLUMN)), "sd": float(r.jitter_std), "n": n,
                         "rated": bs.recent_improvement_rate(run.observed, n, window),
                         "true": bs.recent_improvement_rate(run.deployed, n, window)})
    _RATES[key] = pd.DataFrame(rows)
    return _RATES[key]


def rerank_choices(derived: pd.DataFrame, joined: pd.DataFrame, window: int = 10) -> dict:
    """Each fixed-decision run's choice with the search credit read from the true value."""
    rates = run_rates(joined, window).set_index(["file", "n"])
    new_choice = {}
    for rec in derived.dropna(subset=["scores"]).itertuples(index=False):
        scores = json.loads(rec.scores)
        k_max = max(int(k) for k in scores if k not in bs.NO_TRIAL_POLICIES)
        n0 = T - k_max
        if (rec.file, n0) not in rates.index:
            continue
        rated, true = rates.loc[(rec.file, n0), ["rated", "true"]]
        shift = true - rated
        adjusted = {key: value + (k_max if key in bs.NO_TRIAL_POLICIES else k_max - int(key)) * shift
                    for key, value in scores.items()}
        new_choice[rec.file] = max(adjusted, key=lambda key: adjusted[key])
    return new_choice


def rerank_agreement(truth_derived: pd.DataFrame, rating_derived: pd.DataFrame, joined: pd.DataFrame) -> None:
    """The re-ranking of the rating rule's stored scores against the rule refitted with
    --rate-source truth: they must choose alike, which validates clairvoyant_search_credit."""
    reranked = rerank_choices(rating_derived, joined)
    refit = truth_derived.set_index("file")["k_hat"].map(bs.choice_name)
    both = [f for f in reranked if f in refit.index]
    agree = np.mean([bs.choice_name(reranked[f]) == refit[f] for f in both]) if both else float("nan")
    print(f"  validation: re-ranking {RERANK_OF.get('budget_split_derived_kmax25_truerate.csv')} with the "
          f"true-value rate picks the same policy as this refit in {agree:.4f} of {len(both):,} runs")


def clairvoyant_search_credit(derived: pd.DataFrame, joined: pd.DataFrame, no_trial, opt_z,
                              window: int = 10) -> None:
    """A diagnostic, not a rule: the fixed-decision rule with its search credit read from the
    best TRUE value instead of the best rating, every other input unchanged.

    derive_k scores the no-trial rules as mu + k_max * rate and a sitting of k as U(k) +
    (k_max - k) * rate, with rate the recent gain per trial in best rating at n0. The stored
    scores keep U and mu, so swapping in the true-value rate re-ranks the options exactly
    (ties keep derive_k's order, pm first) without refitting. If the choice then tracks the
    error and the late-onset gap to fixed k shrinks, the inflated rate is what drives it.
    """
    new_choice = rerank_choices(derived, joined, window)
    swapped = joined.assign(k_hat=joined["file"].map(new_choice))
    swapped = swapped[swapped["k_hat"].notna()]
    runs = base_key(swapped)
    pm = runs.assign(pm=runs["k_hat"] == "pm").pivot_table(index=bs.ONSET_COLUMN, columns="jitter_std",
                                                             values="pm", aggfunc="mean")
    print(f"  {runs['file'].nunique():,} runs; share that ship the posterior-mean design, by onset x magnitude:")
    print("    " + pm.to_string(float_format=lambda x: f"{x:.3f}").replace("\n", "\n    "))
    for label, seeds in (("all", None), ("12-16", (12, 13, 14, 15, 16))):
        t = bs.score_by_onset(swapped, no_trial, opt_z, comparators=("fixed_k8", "fixed_k12"), seeds=seeds,
                              seeds_label=label)
        for onset in ("all", "0", "20"):
            b = t[t["onset"] == onset]
            g = b[(b["kind"] == "gain") & (b["policy"] == "derived")].iloc[0]
            d = b[b["kind"] == "derived_minus"]
            text = "; ".join(f"minus {r.policy} {fmt(r.estimate, r.lo, r.hi)}" for r in d.itertuples(index=False))
            print(f"    seeds {label:>5}, onset {onset:>3}: {fmt(g['estimate'], g['lo'], g['hi'])}; {text}")


def search_term(joined: pd.DataFrame, opt_z: dict[str, float], window: int = 10) -> None:
    """What the rule's search term reads at each decision point, against what the search found.

    The rule credits every trial it keeps searching with the run's recent gain in its best
    RATING per trial (budget_split.recent_improvement_rate over the last `window` trials).
    The same estimator applied to the true value of the same visited designs (the log's
    objective_true) is the search progress the run actually made. Their ratio says how much
    of the credited progress is rating error. Both are divided by opt_z and averaged within
    a landscape and then over landscapes; the ratio is a ratio of those means. No refitting:
    the logs of output-boba only.
    """
    rates = run_rates(joined, window)
    runs = rates["file"].unique()
    z = rates["dataset"].map(lambda d: opt_z.get(d, 1.0))
    frame = rates.assign(rated=rates["rated"] / z, true=rates["true"] / z)
    means = frame.groupby(["onset", "sd", "n", "dataset"])[["rated", "true"]].mean().groupby(
        level=["onset", "sd", "n"]).mean()
    print(f"  {len(runs):,} runs; rate per trial in units of opt_z (x 1000), landscape means; "
          f"'x' = rated / true (ratio of landscape means):")
    header = "    onset  sigma " + "".join(f"   n={n:<2} rated/true (x)  " for n in SEARCH_POINTS)
    print(header)
    for (onset, sd), block in means.groupby(level=["onset", "sd"]):
        cells = []
        for n in SEARCH_POINTS:
            rated, true = block.loc[(onset, sd, n), ["rated", "true"]]
            cells.append(f"{1000 * rated:6.2f}/{1000 * true:5.2f} ({rated / true if true > 0 else float('nan'):4.1f})")
        print(f"    {onset:>5} {sd:>6g}  " + "  ".join(cells))
    by_process = frame.groupby(["process", "onset", "sd", "n", "dataset"])[["rated", "true"]].mean().groupby(
        level=["process", "onset", "sd", "n"]).mean()
    ratio = (by_process["rated"] / by_process["true"]).unstack("n")
    print("  the same ratio (rated / true) by process:")
    print("    " + ratio.to_string(float_format=lambda x: f"{x:.1f}").replace("\n", "\n    "))


def pm_for_lcb(joined: pd.DataFrame, no_trial, opt_z) -> None:
    """How much of the deficit is the rule's no-trial choice? A counterfactual, not a rule.

    Both no-trial policies are scored by the posterior mean of the design they would ship plus
    the same search credit, so the posterior-mean pick always scores at least as high as the
    lcb pick and the rule never chooses lcb. Replacing every pm choice by lcb (the cautious ship
    rule, the better of the two in realised regret) and keeping every sitting choice shows how
    much of the gap to fixed k that one comparison accounts for.
    """
    swapped = joined.assign(k_hat=joined["k_hat"].map(lambda k: "lcb" if bs.choice_name(k) == "pm" else k))
    for label, seeds in (("all", None), ("12-16", (12, 13, 14, 15, 16))):
        t = bs.score_by_onset(swapped, no_trial, opt_z, comparators=("fixed_k8", "fixed_k12", "always_lcb"),
                              seeds=seeds, seeds_label=label)
        for onset in ("all", "0", "20"):
            b = t[t["onset"] == onset]
            g = b[(b["kind"] == "gain") & (b["policy"] == "derived")].iloc[0]
            d = b[b["kind"] == "derived_minus"]
            text = "; ".join(f"minus {r.policy} {fmt(r.estimate, r.lo, r.hi)}" for r in d.itertuples(index=False))
            print(f"    seeds {label:>5}, onset {onset:>3}: {fmt(g['estimate'], g['lo'], g['hi'])}; {text}")


def landscape_groups() -> dict[str, set[str]] | None:
    """The paper's landscape groups (review/smooth_subset_definition.csv): the six laws, on
    which the loop's GP fits; the ten rugged classical functions, on which it reads roughness
    as noise (Section 4); and the other four (three smooth classical functions, moving peaks)."""
    path = A / "review" / "smooth_subset_definition.csv"
    if not path.is_file():
        return None
    spec = pd.read_csv(path)
    laws = set(spec.loc[spec["primary_smooth_subset"].astype(bool), "landscape"])
    rugged = set(spec.loc[(spec["group"] == "classical") & ~spec["secondary_smooth_classical"].astype(bool),
                          "landscape"])
    others = set(spec["landscape"]) - laws - rugged
    return {"six laws": laws, "ten rugged classical": rugged, "four others": others}


def by_group(joined: pd.DataFrame, no_trial, opt_z) -> None:
    """Is the deficit where the surrogate is misspecified? The rule by landscape group."""
    groups = landscape_groups()
    if groups is None:
        print("  smooth_subset_definition.csv not found; skipped")
        return
    for label, members in groups.items():
        sub = joined[joined["dataset"].isin(members)]
        t = bs.score_by_onset(sub, no_trial, opt_z, comparators=("fixed_k8", "fixed_k12", "always_pm"))
        print(f"  {label} ({sub['dataset'].nunique()} landscapes; all seeds; with this few clusters the "
              f"intervals are rough):")
        for onset in ("0", "20"):
            b = t[t["onset"] == onset]
            g = b[(b["kind"] == "gain") & (b["policy"] == "derived")].iloc[0]
            share = b[b["kind"] == "choice_share"].set_index("policy")["estimate"].get("pm", 0.0)
            diffs = b[b["kind"] == "derived_minus"]
            text = "; ".join(f"minus {r.policy} {fmt(r.estimate, r.lo, r.hi)}" for r in diffs.itertuples(index=False))
            print(f"    onset {onset:>2}: derived {fmt(g['estimate'], g['lo'], g['hi'])}, pm share {share:.3f}; {text}")


def fit_association(joined: pd.DataFrame, no_trial, opt_z) -> None:
    """Across the 20 landscapes, does the surrogate's fit predict the rule's deficit?

    Per landscape (all seeds): the derived rule's regret-based gain minus fixed k's, by onset,
    against two fit measures from review/gp_diagnostics.csv (smooth_subset_and_gp_diagnostics.py):
    the clean GP's R^2 at fresh Sobol points (Section 4's 'R^2 at fresh points') and the
    leave-one-out log predictive density as a fraction of the achievable, under 1 sigma
    gaussian error from the first rating (the predictor Section 4 correlates with the
    posterior-mean rule's gain). Spearman's rho, scipy.stats.spearmanr.
    """
    from scipy.stats import spearmanr

    path = A / "review" / "gp_diagnostics.csv"
    if not path.is_file():
        print("  gp_diagnostics.csv not found; skipped")
        return
    g = pd.read_csv(path)
    g = g[g["level"] == "landscape"]
    fits = {
        "clean sobol_r2": g[g["condition"] == "clean"].set_index("name")["sobol_r2"],
        "1sd onset-0 loo_lpd_frac_achievable": g[g["condition"] == "gaussian_1sd_onset0"]
        .set_index("name")["loo_lpd_frac_achievable"],
    }
    for onset in (0, 20):
        _, frame, _ = bs.score_policies(joined[joined[bs.ONSET_COLUMN] == onset], no_trial, opt_z,
                                        np.random.default_rng(bs.BOOTSTRAP_SEED))
        means = frame.rename(columns=bs.policy_name).groupby(level="dataset").mean()
        for comp in ("fixed_k8", "fixed_k12", "always_pm"):
            diff = means[comp] - means["derived"]      # derived gain minus comparator gain
            parts = []
            for name, fit in fits.items():
                both = pd.concat([diff.rename("d"), fit.rename("f")], axis=1, join="inner").dropna()
                rho, p = spearmanr(both["f"], both["d"])
                parts.append(f"vs {name} rho {rho:+.2f} (p {p:.3f}, n {len(both)})")
            print(f"    onset {onset:>2}, derived minus {comp:<10}: " + "; ".join(parts))


def reproduction(table: pd.DataFrame) -> None:
    path = A / "review" / "heldout_remedies.csv"
    if not path.is_file():
        print("  heldout_remedies.csv not found; skipped")
        return
    hr = pd.read_csv(path, low_memory=False)
    test = table[(table["seeds"] == "12-16") & (table["onset"] == "all")]
    g = test[(test["kind"] == "gain") & (test["policy"] == "derived")].iloc[0]
    holm = hr[(hr["section"] == "4_holm") & (hr["candidate"] == "derived")]
    published = float(holm["mean_gain_test"].iloc[0]) if not holm.empty else float("nan")
    md = A / "review" / "heldout_remedies.md"
    md_line = next((ln.strip() for ln in md.read_text(encoding="utf-8").splitlines()
                    if ln.startswith("| derived rule, rho = 1, test seeds 12-16")), "not found") if md.is_file() else "not found"
    ok = np.isclose(g["estimate"], published, atol=1e-12)
    print(f"  pooled held-out derived gain here {g['estimate']:+.6f} [{g['lo']:+.6f}, {g['hi']:+.6f}]; "
          f"heldout_remedies.csv mean_gain_test {published:+.6f} ({'reproduced' if ok else 'MISMATCH'}); "
          f"heldout_remedies.md: {md_line}")
    row = hr[(hr["section"] == "5_derived_rule") & (hr["problem"] == "derived_vs_fixed")
             & (hr["note"] == "test seeds, train-selected k vs derived")]
    if not row.empty:
        d = test[(test["kind"] == "derived_minus") & (test["policy"] == row["candidate"].iloc[0])].iloc[0]
        r = row.iloc[0]
        ok = (np.isclose(-d["estimate"], r["test_value"], atol=1e-12) and np.isclose(-d["hi"], r["test_lo"], atol=1e-12)
              and np.isclose(-d["lo"], r["test_hi"], atol=1e-12))
        print(f"  {r['candidate']} minus derived on seeds 12-16: here {-d['estimate']:+.6f} [{-d['hi']:+.6f}, {-d['lo']:+.6f}], "
              f"heldout_remedies.csv {r['test_value']:+.6f} [{r['test_lo']:+.6f}, {r['test_hi']:+.6f}]: "
              f"{'reproduced' if ok else 'MISMATCH'}")


# (label, price table, the rule's choices on the clean twins) from budget_split.py --twin clean.
PRICES = (
    ("decide-once rule", "budget_split_price_clean.csv", "budget_split_derived_clean.csv"),
    ("sequential rule", "budget_split_price_decision_sequential_clean.csv",
     "budget_split_derived_decision_sequential_clean.csv"),
)


def price_section() -> None:
    """The price of each rule, trt_clean - ref_clean: the regret it adds to a run without
    error, derived on the clean twin and charged the clean twin's replayed regret of what it
    chose (budget_split.py --twin clean), beside the same price for fixed k and the no-trial
    rules. Read from the price tables; nothing is refitted here."""
    for label, price_name, derived_name in PRICES:
        if not (A / price_name).is_file():
            print(f"  {label}: {price_name} not found; skipped")
            continue
        table = pd.read_csv(A / price_name)
        print(f"  {label} [{price_name}]:")
        for seeds, block in table.groupby("seeds", sort=False):
            price = block[block["kind"] == "price"].set_index("policy")
            text = "; ".join(f"{pol} {fmt(price.loc[pol, 'estimate'], price.loc[pol, 'lo'], price.loc[pol, 'hi'])}"
                             for pol in ("derived", "fixed_k8", "fixed_k12", "fixed_k16", "always_pm", "always_lcb")
                             if pol in price.index)
            diffs = block[block["kind"] == "derived_minus"]
            dtext = "; ".join(f"derived minus {r.policy} {fmt(r.estimate, r.lo, r.hi)}"
                              for r in diffs.itertuples(index=False))
            share = block[block["kind"] == "choice_share"].set_index("policy")["estimate"]
            print(f"    seeds {seeds:>5} ({int(block['n_twins'].iloc[0])} clean twins, "
                  f"{int(block['n_failed'].iloc[0])} failed fits; pm share {share.get('pm', 0.0):.3f}, "
                  f"lcb share {share.get('lcb', 0.0):.3f}): price {text}")
            print(f"      {dtext}")
        derived = pd.read_csv(A / derived_name) if (A / derived_name).is_file() else None
        if derived is not None and "decided_at" in derived.columns:
            where = derived["decided_at"].value_counts(normalize=True).sort_index()
            print("    where it committed on the clean twins (trial n; 48 also when it shipped with no sitting): "
                  + ", ".join(f"{int(n)} {v:.3f}" for n, v in where.items()))


def run() -> None:
    no_trial = bs.load_no_trial(REPO / "output-boba")
    opt_z = bs.load_opt_z()
    sweeps: dict = {}
    print(__doc__.split("\n\n")[0])
    print("\nT = 50; the simulator corrupts trial t when t > onset (onsets 0 and 20). LogEI and qNEI, the four "
          "response-error processes (gaussian, bias, drift, ar1), four magnitudes, seeds 7-16; 20 landscapes.")
    for label, name, rho in VARIANTS:
        print("\n" + "=" * 100)
        print(f"{label}   [{name}, rho = {rho:g}]")
        if not (A / name).is_file():
            print("  not found; skipped")
            continue
        derived, joined, n_failed = load(name, rho, sweeps, no_trial, opt_z)
        grid = grid_of(derived)
        sequential = "decided_at" in derived.columns
        how = "a decision at each T - k" if sequential else f"one decision at n0 = T - {max(grid)} = {T - max(grid)}"
        print(f"  grid {','.join(map(str, grid))}; {how}; {joined['file'].nunique():,} runs scored, "
              f"{n_failed} dropped for a failed fit")
        if sequential:
            sequential_detail(derived, joined)
        # For the sequential rule the SD is the one read where it committed.
        blindness(derived, joined)
        table = by_onset_tables(joined, no_trial, opt_z)
        print("  deployed-design gain over the standard process (regret / opt_z, landscape means, "
              "landscape bootstrap), by seeds and onset:")
        print_by_onset(table)
        diagnostic = label.startswith("DIAGNOSTIC")
        if rho == 1.0 and diagnostic:
            print("  by magnitude and onset (all seeds; deployed-design gain over the standard process):")
            by_magnitude(joined, no_trial, opt_z)
            if name in RERANK_OF and (A / RERANK_OF[name]).is_file():
                rerank_agreement(derived, pd.read_csv(A / RERANK_OF[name]), joined)
        if rho == 1.0 and not diagnostic:
            print("  by magnitude and onset (all seeds; deployed-design gain over the standard process):")
            by_magnitude(joined, no_trial, opt_z)
            print("  without the 5 sigma cell from trial 21 (the cell that carries the pooled gain of k = 12; "
                  "deployed-design gain over the standard process):")
            without_late_5sigma(joined, no_trial, opt_z)
            print("  gains at 0.05 sigma, where the error costs least (deployed-design gain over the standard "
                  "process; not a price, the clean-twin prices are at the end):")
            near_clean(joined, no_trial, opt_z)
            print("  by landscape group (is the deficit where the surrogate is misspecified?):")
            by_group(joined, no_trial, opt_z)
            print("  across landscapes: Spearman correlation of the rule's per-landscape paired gain "
                  "with the surrogate's fit (positive: the rule does better where the GP fits):")
            fit_association(joined, no_trial, opt_z)
            print("  counterfactual: every pm choice replaced by lcb, sitting choices kept (deployed-design "
                  "gain over the standard process, and minus each comparator):")
            pm_for_lcb(joined, no_trial, opt_z)
            if not sequential:
                print("  counterfactual: the search credit read from the best true value instead of the best "
                      "rating (clairvoyant; the stored scores re-ranked, nothing refitted):")
                clairvoyant_search_credit(derived, joined, no_trial, opt_z)
        if name == "budget_split_derived.csv":
            print("  reproduction of the held-out numbers of heldout_remedies.csv:")
            reproduction(table)
            print("  the search term at each decision point n: the recent gain per trial in the best rating "
                  "(what the rule credits) against the same gain in the best true value (what the search "
                  "found), by onset x magnitude, at every n = T - k of the grid (the sequential rule's "
                  "decision points; n = 20 is this rule's, 25 the capped grid's):")
            search_term(joined, opt_z)
    print("\n" + "=" * 100)
    print("the price of each rule without error: trt_clean - ref_clean on the clean twins (budget_split.py "
          "--twin clean; regret / opt_z, landscape means, landscape bootstrap; positive = regret added)")
    price_section()


def main() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
