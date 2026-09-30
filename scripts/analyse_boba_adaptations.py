"""The process-adaptation arms: how much of the cost of error does each recover?

Why not the usual excess contrast
---------------------------------
Every other arm is scored by its excess regret, noisy minus its own identically
seeded clean run. That is the wrong estimand for a PROCESS change. Rating the
first ten designs twice, or spending trials re-rating, also changes the clean
run -- a repeated rating of an exact objective carries no information, so the
arm's clean twin is itself handicapped -- and "excess over a handicapped twin"
mixes the benefit under error with the price paid without it. What a
practitioner wants to know is whether the adapted process, under error, gets
closer to what the STANDARD process achieves without error. So, per paired cell:

    cost       = ref_noisy - ref_clean     what the error costs the standard process
    gain       = ref_noisy - trt_noisy     how much better the adapted process does under error
    price      = trt_clean - ref_clean     what the adaptation costs when there is no error
    recovered  = gain / cost               share of the standard process's cost recovered

on two responses: the post-onset per-iteration true simple regret (the
trajectory), and the final regret of the design the experimenter would deploy.
Aggregates are ratios of landscape means; intervals resample landscapes; the
test is a Wilcoxon over per-landscape gains, BH-corrected within arm and
response.

Regret is divided by a per-dataset scale BEFORE any mean is taken, so that cost,
gain and price are fractions of what there is to win and no landscape outweighs
another. Each arm names its scale (SCALES), and a dataset without one raises;
until 2026-09-29 a missing opt_z silently became 1.0, which scored the
multi-objective halo arms in raw hypervolume (VehicleSafety carried 91% of the
summed cost) and the fitted-oracle arm in raw rating units:

    opt_z              the analytic landscapes: the achievable improvement
    hv_floor_gap       the multi-objective problems, which have no scalar optimum:
                       the published maximum hypervolume less what a same-budget
                       model-free design reaches, the normaliser of the
                       multi-objective table (analyse_boba_mo.achievable_gain)
    fitted_achievable  the fitted-oracle datasets: mean over seeds of the seed's
                       oracle optimum less its mean over the search box, the
                       achievable-improvement analogue
                       (analyse_fitted_companion.fitted_achievable)

A dataset's own recovery gain/cost does not depend on its scale; the scale only
weights the datasets in a pooled ratio. adaptations_per_dataset.csv holds every
arm's per-dataset cost, gain, price and recovery pooled over its cells, which is
what an arm with three or four clusters should be reported by: with n clusters
the exact two-sided Wilcoxon cannot go below 2 / 2**n (0.25 at three, 0.125 at
four), and with three the percentile bootstrap returns the range of the three
per-dataset values (each extreme dataset is the whole resample with probability
1/27 > 2.5%).

A recovery is a share only when its denominator is. Every row of
adaptations_recovery.csv keeps its recovered value, but recovered_suppressed
(and pooled_recovered_suppressed) gives the reason when the cell's cost is below
NEAR_ZERO_REFERENCE = 0.01 of the scale or negative on any landscape
(n_cost_negative counts those); there the gain, in the same units, is what may
be quoted. The rule is share_of_reference, which compare_boba_arms.py imports.

Most arms change the process for the same acquisitions and pair at
(landscape, acquisition, magnitude, onset, seed). The robust-baseline arms
(qKG, replication) are different acquisitions, so there the standard process is
the mean over the ten standard model-based acquisitions and pairing is at
(landscape, magnitude, onset, seed). The same flaw applies to them: replication
re-rates the incumbent in its clean run too, and the paper's earlier "recovers
a third" was an excess-over-own-twin comparison.

Extra trials are counted against the STANDARD clean run (analyse_extra_runs.py
--baseline-dir), beside the reference's own count, for the arms that pair on
acquisition.

    python scripts/analyse_boba_adaptations.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import boba_benchmarks as bb  # noqa: E402

PY = sys.executable
PAIR_KEYS = ["dataset", "acquisition", "error_model", "jitter_std", "jitter_iteration", "seed"]
POOLED_KEYS = ["dataset", "error_model", "jitter_std", "jitter_iteration", "seed"]
MODEL_FREE = ("random", "sobol")
RESPONSES = {
    "trajectory": "auc_simple_regret_true_postonset_per_iter",
    "deployed": "final_inference_simple_regret_true",
}
GRID = (0.05, 0.25, 1.0, 5.0)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260913
S5 = "7,8,9,10,11"
S10 = "7,8,9,10,11,12,13,14,15,16"
TEN = "logei,ei,pi,ucb,qucb,qnei,logpi,qei,qpi,greedy"
# The per-dataset scales an arm can name (see the module docstring), and where
# the two that are not landscape statistics are computed from. The halo arms run
# the multi-objective arm's budget (50 trials, 5 initial) on four of its
# problems with the same reference points and maximum hypervolumes, so its
# floor is theirs; the fitted-oracle arm refits the oracle per seed, identically
# in output-fitted and output-fitted-adapt-rep10.
SCALES = ("opt_z", "hv_floor_gap", "fitted_achievable")
MO_FLOOR_DIR = Path("output-boba-mo")
FITTED_DIR = Path("output-fitted")
# Below this pooled reference cost, in units of the dataset's scale, a share of
# it is not a number worth quoting. compare_boba_arms.py's docstring gives the
# choice; the rule is defined once, here, and imported there.
NEAR_ZERO_REFERENCE = 0.01


def share_of_reference(per_landscape: pd.DataFrame, near_zero: float | None = NEAR_ZERO_REFERENCE) -> dict:
    """The share of the reference's cost removed, or why it is not reported.

    ``per_landscape`` holds one row per landscape with the landscape means ``ref``
    (the reference cost) and ``trt`` (what is left of it). Returns share_removed,
    1 - mean(trt) / mean(ref) or NaN when suppressed; the reason (share_suppressed,
    empty when reported); and the counts of landscapes whose reference cost is
    negative or exactly zero. The share is suppressed when the pooled reference is
    below ``near_zero`` or negative on any landscape (AGENTS.md: never a recovery
    ratio whose reference cost is near zero or changes sign). An exactly zero
    landscape (solved before a mid-run onset) is counted but does not suppress.
    ``near_zero=None`` skips the magnitude test, for values that are not in units
    of an achievable improvement, where 0.01 means nothing.
    """
    ref, trt = per_landscape["ref"], per_landscape["trt"]
    mean_ref = float(ref.mean())
    negative, zero = int((ref < 0).sum()), int((ref == 0).sum())
    reasons = []
    if near_zero is not None and not mean_ref >= near_zero:
        reasons.append(f"reference {mean_ref:.4f} below {near_zero:g}")
    if negative:
        reasons.append(f"reference negative on {negative} of {len(ref)} landscapes")
    if not mean_ref > 0 and not reasons:
        reasons.append(f"reference {mean_ref:.4f} is not positive")
    share = np.nan if reasons else float(1.0 - trt.mean() / mean_ref)
    return {"share_removed": share, "share_suppressed": "; ".join(reasons),
            "n_reference_negative": negative, "n_reference_zero": zero}


def arm(dir_, ref, acqs, what, seeds=S5, ref_acqs=None, pool=False, relative=False, error_model=None,
        variant=None, ref_variant=None, scale="opt_z"):
    # variant/ref_variant select one condition out of a directory that holds
    # several (four spike sizes, two ceiling modes, a halo rho). None means "the
    # whole directory", which is every arm that predates the variant column.
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose from {SCALES}")
    return dict(dir=dir_, ref=ref, acqs=acqs, ref_acqs=ref_acqs or acqs, seeds=seeds, what=what,
                pool=pool, relative=relative, error_model=error_model,
                variant=variant, ref_variant=ref_variant, scale=scale)


ARMS = {
    # The two ablation arms of the paper change the surrogate or the incumbent,
    # which changes the clean run as well; re-scored here against the standard
    # process so their headline shares can be checked on the same footing.
    "incumbent": arm("output-boba-incumbent", "output-boba", "logei,ei,pi,logpi,qei,ucb,qnei",
                     "observed-max incumbent instead of the posterior-mean one", error_model="gaussian"),
    "knownnoise": arm("output-boba-knownnoise", "output-boba", "logei,ei,pi,ucb,qucb,qnei",
                      "the GP is given the true observation variance", error_model="gaussian"),
    "rep10": arm("output-boba-adapt-rep10", "output-boba", "logei,qnei", "first ten proposals rated twice",
                 error_model="gaussian"),
    "bundle": arm("output-boba-adapt-rep10-obs", "output-boba", "logei",
                  "first ten rated twice + observed-max incumbent (LogEI), against the standard process",
                  error_model="gaussian"),
    "rep10-obs": arm("output-boba-adapt-rep10-obs", "output-boba-incumbent", "logei",
                     "replication on top of the observed-max incumbent (against the incumbent arm)",
                     error_model="gaussian"),
    "rerate": arm("output-boba-adapt-rerate", "output-boba", "logei,qnei",
                  "last six trials re-rate the top three designs", error_model="gaussian"),
    # A single acquisition scored against the MEAN OF TEN, four of which the paper
    # ranks bottom, is not a like-for-like contrast: the reference carries the weak
    # arms' cost, so the treatment "recovers" part of a cost it never had. Augmented
    # EI reached +373% of the cost in one cell that way, which is impossible for a
    # valid estimand. The primary reference is therefore LogEI, the suite's default
    # and the acquisition an experimenter would otherwise have run; the ten-mean
    # version stays beside it as "*-ten" so the difference is visible.
    "qkg": arm("output-boba-robust", "output-boba", "qkg",
               "knowledge gradient, against LogEI",
               ref_acqs="logei", pool=True, error_model="gaussian"),
    "qkg-ten": arm("output-boba-robust", "output-boba", "qkg",
                   "knowledge gradient, against the mean of the ten standard acquisitions",
                   ref_acqs=TEN, pool=True, error_model="gaussian"),
    "replei": arm("output-boba-robust", "output-boba", "replei",
                  "every second trial re-rates the incumbent, against LogEI",
                  ref_acqs="logei", pool=True, error_model="gaussian"),
    "replei-ten": arm("output-boba-robust", "output-boba", "replei",
                      "every second trial re-rates the incumbent, against the mean of the ten",
                      ref_acqs=TEN, pool=True, error_model="gaussian"),
    "rerate-slip": arm("output-boba-adapt-rerate-slip", "output-boba-slip", "logei,qnei",
                       "re-rating under an unnoticed slip", error_model="slip"),
    "nigp": arm("output-boba-adapt-nigp", "output-boba-slip", "logei,qnei",
                "noisy-input GP under an unnoticed slip", error_model="slip"),
    "studentt": arm("output-boba-adapt-studentt", "output-boba-misclick", "logei,qnei",
                    "Student-t surrogate under misclicks", error_model="misclick"),
    # Register item B2: the arm above also swaps the RBF kernel for Matern-5/2.
    # This one keeps the standard kernel, so the likelihood is the only change.
    "studentt-rbf": arm("output-boba-adapt-studentt-rbf", "output-boba-misclick", "logei,qnei",
                        "Student-t likelihood with the standard RBF kernel, under misclicks",
                        error_model="misclick"),
    "fitted-rep10": arm("output-fitted-adapt-rep10", "output-fitted", "logei,qnei",
                        "first ten rated twice, on the three fitted-oracle datasets",
                        seeds="7,8,9,10,11,12,13,14,15,16", relative=True, error_model="gaussian",
                        scale="fitted_achievable"),

    # ---------------------------------------------------------------- budget-neutral
    # The 2026-09-14/16 sweep (run_boba_budget_neutral.ps1). Every arm here keeps
    # the number of human trials equal to the standard process, or lowers it, so
    # each is scored against the standard process on the same trial budget. The
    # arms whose variants differ only in a filename SUFFIX (spike, spike-clip,
    # spike-rrp, relay, ceiling, mo-halo, mo-halo-backfit) share one directory
    # per family, so each names its variant (and its reference's) explicitly;
    # they are defined further down.
    # ------------------------------------------------- the 2026-09-21 idea round
    # Five process changes aimed at the selection term. run_boba_hitl_ideas.ps1
    # owns their sweep; handover/hitl-ideas-2026-09-21.md states each prediction
    # BEFORE the numbers, which is the only way a prediction is worth anything.
    "idea-anchor-gaussian": arm("output-boba-idea-anchor", "output-boba", "logei,qnei",
                               "the proposal judged beside the incumbent, so the shared error cancels",
                               error_model="gaussian"),
    "idea-anchor-bias": arm("output-boba-idea-anchor", "output-boba", "logei,qnei",
                           "the anchored rating under a constant offset, which it should remove",
                           error_model="bias"),
    "idea-anchor-drift": arm("output-boba-idea-anchor", "output-boba", "logei,qnei",
                            "the anchored rating under drift, which it should remove",
                            error_model="drift"),
    "idea-selfreport": arm("output-boba-idea-selfreport", "output-boba", "logei,qnei",
                          "the rater reports their own precision and the GP uses it per trial",
                          error_model="gaussian"),
    # The 2026-09-22 review: the arm above was chosen on seeds 7-11, so it is
    # re-scored on seeds 12-16, which were run for this check afterwards; and its
    # report model (log-normal multiplier, SD 0.5) gives the rater a rank
    # correlation of about 0.96 with their own realised error, so three arms name
    # that correlation instead, at 0.6, 0.3 and 0 (gaussian copula,
    # --confidence-corr), in the two cells where the arm helped most.
    "idea-selfreport-heldout": arm("output-boba-idea-selfreport", "output-boba", "logei,qnei",
                                  "self-reported precision, scored on held-out seeds 12-16",
                                  seeds="12,13,14,15,16", error_model="gaussian"),
    "idea-selfreport-r0.6": arm("output-boba-idea-selfreport-r0.6", "output-boba", "logei,qnei",
                               "self-reported precision, rank correlation 0.6 with the realised error",
                               error_model="gaussian"),
    "idea-selfreport-r0.3": arm("output-boba-idea-selfreport-r0.3", "output-boba", "logei,qnei",
                               "self-reported precision, rank correlation 0.3 with the realised error",
                               error_model="gaussian"),
    "idea-selfreport-r0": arm("output-boba-idea-selfreport-r0", "output-boba", "logei,qnei",
                             "self-reported precision, uninformative (rank correlation 0)",
                             error_model="gaussian"),
    "idea-anchors-gaussian": arm("output-boba-idea-anchors", "output-boba", "logei,qnei",
                                "every fifth trial rates a fixed anchor; the anchors detrend the rest",
                                error_model="gaussian"),
    "idea-anchors-drift": arm("output-boba-idea-anchors", "output-boba", "logei,qnei",
                             "anchors under drift, the fault they are meant to identify",
                             error_model="drift"),
    "idea-hold": arm("output-boba-idea-hold", "output-boba", "logei,qnei",
                    "the first five proposals rated late instead of early",
                    error_model="gaussian"),
    # An acquisition, so it is scored against LogEI like every other acquisition
    # arm, with the ten-mean version kept beside it.
    "idea-shiplcb": arm("output-boba-idea-shiplcb", "output-boba", "shiplcb",
                       "an acquisition that values a rating by what it does to the ship rule",
                       ref_acqs="logei", pool=True, error_model="gaussian"),
    "idea-shiplcb-ten": arm("output-boba-idea-shiplcb", "output-boba", "shiplcb",
                           "the same, against the mean of the ten standard acquisitions",
                           ref_acqs=TEN, pool=True, error_model="gaussian"),
    "q-inclcb": arm("output-boba-q-inclcb", "output-boba", "logei,logpi",
                    "a lower-confidence-bound incumbent instead of the posterior mean",
                    error_model="gaussian"),
    "q-aei": arm("output-boba-q-aei", "output-boba", "aei",
                 "augmented expected improvement, against LogEI",
                 ref_acqs="logei", pool=True, error_model="gaussian"),
    "q-aei-ten": arm("output-boba-q-aei", "output-boba", "aei",
                     "augmented expected improvement, against the mean of the ten",
                     ref_acqs=TEN, pool=True, error_model="gaussian"),
    "q-ts": arm("output-boba-q-ts", "output-boba", "ts",
                "Thompson sampling, against LogEI",
                ref_acqs="logei", pool=True, error_model="gaussian"),
    "q-ts-ten": arm("output-boba-q-ts", "output-boba", "ts",
                    "Thompson sampling, against the mean of the ten",
                    ref_acqs=TEN, pool=True, error_model="gaussian"),
    "sched-front10": arm("output-boba-sched-front10", "output-boba", "logei,qnei",
                         "the same total rating effort, concentrated on the first ten trials",
                         error_model="gaussian"),
    "sched-front20": arm("output-boba-sched-front20", "output-boba", "logei,qnei",
                         "the same total rating effort, concentrated on the first twenty trials",
                         error_model="gaussian"),
    "sched-U": arm("output-boba-sched-U", "output-boba", "logei,qnei",
                   "the same total rating effort, spent on the first and last trials",
                   error_model="gaussian"),
    "sched-back10": arm("output-boba-sched-back10", "output-boba", "logei,qnei",
                        "the same total rating effort, concentrated on the last ten trials",
                        error_model="gaussian"),
    "q-mind": arm("output-boba-q-mind", "output-boba", "logei,pi",
                  "a proposal must stay 0.05 of the box away from every visited design",
                  error_model="gaussian"),
    "q-iu-0.05": arm("output-boba-q-iu-0.05", "output-boba-slip", "logei,qnei",
                     "slip-aware acquisition, 0.05 slip", error_model="slip"),
    "q-iu-0.15": arm("output-boba-q-iu-0.15", "output-boba-slip", "logei,qnei",
                     "slip-aware acquisition, 0.15 slip", error_model="slip"),
    "q-iu-0.4": arm("output-boba-q-iu-0.4", "output-boba-slip", "logei,qnei",
                    "slip-aware acquisition, 0.4 slip", error_model="slip"),
    "missing-impute-mcar": arm("output-boba-missing-impute", "output-boba-missing-drop", "logei,ucb",
                               "a rating lost at random, imputed low instead of dropped",
                               error_model="missing_mcar"),
    "missing-impute-low": arm("output-boba-missing-impute", "output-boba-missing-drop", "logei,ucb",
                              "a rating lost because the design was bad, imputed low instead of dropped",
                              error_model="missing_low"),

    # A saturating rating scale: does re-anchoring the cap to the best design so
    # far beat a cap fixed at the 0.9 quantile of the landscape?
    "ceiling-anchored": arm("output-boba-ceiling", "output-boba-ceiling", "logei,qnei",
                            "a rating cap re-anchored to the best design so far, against a fixed cap",
                            error_model="gaussian", variant="ceil0.9-anchored",
                            ref_variant="ceil0.9-fixed"),
    "ceiling-cost": arm("output-boba-ceiling", "output-boba", "logei,qnei",
                        "what a rating scale that saturates costs, against a scale that does not",
                        error_model="gaussian", variant="ceil0.9-fixed"),

    # Correlated rating error across objectives (halo), and the backfit that
    # removes the shared factor. The reference for the remedy is the SAME halo
    # with no model; the no-halo pair prices the model when there is nothing to
    # remove.
    "mo-halo-cost": arm("output-boba-mo-halo", "output-boba-mo-halo", "qlognehvi",
                        "rating error shared across objectives, against independent error",
                        seeds=S10, error_model="gaussian", variant="xc0.85", ref_variant="",
                        scale="hv_floor_gap"),
    "mo-halo-backfit": arm("output-boba-mo-halo-backfit", "output-boba-mo-halo", "qlognehvi",
                           "the shared factor estimated and removed, under halo error",
                           seeds=S10, error_model="gaussian", variant="xc0.85_halo-backfit",
                           ref_variant="xc0.85", scale="hv_floor_gap"),
    "mo-halo-backfit-price": arm("output-boba-mo-halo-backfit", "output-boba-mo-halo", "qlognehvi",
                                 "the same model where there is no shared factor to remove",
                                 seeds=S10, error_model="gaussian", variant="halo-backfit",
                                 ref_variant="", scale="hv_floor_gap"),
}

# The gross-fault family. Four spike sizes share one directory each, so every
# comparison names its own: both remedies are scored against the SAME spike size
# with no remedy, never against a different one.
for _sp in ("sp0.05-5", "sp0.05-20", "sp0.15-5", "sp0.15-20"):
    _prob, _size = _sp[2:].split("-")
    ARMS[f"spike-clip-{_sp}"] = arm(
        "output-boba-spike-clip", "output-boba-spike", "logei,qnei",
        f"the response clipped to the landscape's known range, {_prob} of trials spiking at {_size} SD",
        error_model="spike", variant=_sp, ref_variant=_sp)
    ARMS[f"spike-rrp-{_sp}"] = arm(
        "output-boba-spike-rrp", "output-boba-spike", "logei,qnei",
        f"a relevance-pursuit GP, {_prob} of trials spiking at {_size} SD",
        error_model="spike", variant=f"{_sp}_relevancepursuit", ref_variant=_sp)

# Relay raters: the per-rater offset model against the same handover with none.
# The backfit changes the clean run, so each assignment has its own directory
# (one variant, hence no `variant` here) and its reference is the matching
# handover inside the shared relay directory.
for _tag, _relay in (("block", "rater-block10-tau2"), ("rr", "rater-roundrobin5-tau2")):
    ARMS[f"relay-backfit-{_tag}"] = arm(
        f"output-boba-relay-backfit-{_tag}", "output-boba-relay", "logei,qnei",
        f"per-rater offsets estimated inside the GP ({_relay})",
        error_model="gaussian", ref_variant=_relay)


def load(root: Path, acqs: set[str], seeds: set[int], error_model: str | None,
         variant: str | None = None) -> pd.DataFrame:
    files = sorted(root.glob("*/evaluation/paired_excess_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"no evaluation outputs under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # An empty variant is written as an empty field and read back as NaN, so it
    # is normalised here rather than in every caller. A directory that predates
    # the variant column has none at all.
    df["variant"] = df["variant"].fillna("") if "variant" in df.columns else ""
    keep = ~df["acquisition"].isin(MODEL_FREE) & df["acquisition"].isin(acqs) & df["seed"].isin(seeds)
    if error_model is not None:
        keep &= df["error_model"] == error_model
    if variant is not None:
        present = set(df["variant"])
        if variant not in present:
            raise ValueError(f"{root} holds no runs with variant {variant!r}; it has {sorted(present)}")
        keep &= df["variant"] == variant
    return df[keep]


def rank_magnitudes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for dataset, block in frame.groupby("dataset"):
        stds = sorted(block["jitter_std"].unique())
        if len(stds) != len(GRID):
            raise ValueError(f"{dataset}: {len(stds)} magnitudes, expected {len(GRID)}")
        frame.loc[block.index, "jitter_std"] = block["jitter_std"].map(dict(zip(stds, GRID)))
    return frame


def dataset_scale(kind: str) -> dict[str, float]:
    """The per-dataset scale ``kind`` (one of SCALES), as {dataset: scale}.

    Computed from the recorded outputs each time it is asked for, with the same
    functions that produce the paper's other tables in that unit, so the two
    cannot drift. Every value is checked to be finite and positive.
    """
    if kind == "opt_z":
        stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
        out = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}
    elif kind == "hv_floor_gap":
        import analyse_boba_mo as abm
        import analyse_boba_robustness as ab
        gain = abm.achievable_gain(MO_FLOOR_DIR, ab.load_paired(MO_FLOOR_DIR))
        out = {str(d): float(g) for d, g in zip(gain["dataset"], gain["gain"]) if pd.notna(g)}
    elif kind == "fitted_achievable":
        import analyse_boba_robustness as ab
        import analyse_fitted_companion as afc
        raw = ab.load_paired(FITTED_DIR)
        out = {str(d): float(g) for d, g in
               afc.fitted_achievable(raw, afc.box_means(FITTED_DIR, raw), "oracle").items()}
    else:
        raise ValueError(f"unknown scale {kind!r}; choose from {SCALES}")
    bad = {d: v for d, v in out.items() if not (np.isfinite(v) and v > 0)}
    if bad:
        raise ValueError(f"scale {kind!r} is not a positive number for {bad}")
    return out


def paired_frame(ref: pd.DataFrame, trt: pd.DataFrame, response: str, scale: dict[str, float],
                 pool: bool) -> pd.DataFrame:
    """Pair the two arms and divide every regret by its dataset's ``scale``.

    A dataset missing from ``scale`` raises. The old fallback to 1.0 mixed units
    across datasets without a word (analyse_ship_rules.landscape_opt_z named it
    as the bug it guards against).
    """
    cols = [f"{response}_jitter", f"{response}_baseline"]
    keys = POOLED_KEYS if pool else PAIR_KEYS
    if pool:
        ref = ref.groupby(POOLED_KEYS, as_index=False)[cols].mean()
        trt = trt.groupby(POOLED_KEYS, as_index=False)[cols].mean()
    m = ref[keys + cols].merge(trt[keys + cols], on=keys, suffixes=("_ref", "_trt"), validate="one_to_one")
    if m.empty:
        raise ValueError("the two arms share no paired cells")
    lacking = sorted(set(m["dataset"]) - set(scale))
    if lacking:
        raise KeyError(f"no scale for {lacking}: every dataset needs one (opt_z, a floor gap or an "
                       f"achievable improvement) before regrets from different datasets are pooled")
    z = m["dataset"].map(scale).astype(float)
    if not (np.isfinite(z) & (z > 0)).all():
        raise ValueError(f"non-positive scale for {sorted(set(m.loc[~(z > 0), 'dataset']))}")
    return m.assign(
        ref_noisy=m[f"{response}_jitter_ref"] / z, ref_clean=m[f"{response}_baseline_ref"] / z,
        trt_noisy=m[f"{response}_jitter_trt"] / z, trt_clean=m[f"{response}_baseline_trt"] / z,
    )


def per_dataset(block: pd.DataFrame) -> pd.DataFrame:
    """Each dataset's cost, gain, price and recovery over the cells of ``block``.

    The means are the ones summarise() pools, so the pooled recovery is the
    cost-weighted mean of these recoveries; cost_share is each dataset's weight.
    """
    per = block.groupby("dataset")[["ref_noisy", "ref_clean", "trt_noisy", "trt_clean"]].mean()
    out = pd.DataFrame({
        "n_cells": block.groupby("dataset").size(),
        "cost": per.ref_noisy - per.ref_clean,
        "gain": per.ref_noisy - per.trt_noisy,
        "price": per.trt_clean - per.ref_clean,
    })
    out["recovered"] = np.where(out["cost"] > 0, out["gain"] / out["cost"].where(out["cost"] > 0), np.nan)
    total = out["cost"].sum()
    out["cost_share"] = out["cost"] / total if total > 0 else np.nan
    return out.reset_index()


def summarise(block: pd.DataFrame, rng: np.random.Generator) -> dict:
    per = block.groupby("dataset")[["ref_noisy", "ref_clean", "trt_noisy", "trt_clean"]].mean()
    cost_l = (per.ref_noisy - per.ref_clean).to_numpy()
    gain_l = (per.ref_noisy - per.trt_noisy).to_numpy()
    price_l = (per.trt_clean - per.ref_clean).to_numpy()
    cost, gain, price = cost_l.mean(), gain_l.mean(), price_l.mean()
    n = len(per)
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, n)
        c = cost_l[idx].mean()
        draws.append(gain_l[idx].mean() / c if c > 0 else np.nan)
    draws = np.array(draws)
    # A resample whose denominator is <= 0 has no ratio. Dropping those and
    # taking percentiles over the survivors prints a CONDITIONAL interval as a
    # 95% one, so the discard share is recorded and an interval is refused when
    # more than a twentieth of the draws are gone or the point estimate itself
    # is undefined.
    ok = np.isfinite(draws)
    usable = bool(cost > 0 and ok.mean() >= 0.95)
    if np.allclose(gain_l, 0.0):
        p = 1.0
    else:
        try:
            p = float(wilcoxon(gain_l).pvalue)
        except ValueError:
            p = 1.0
    return {
        "n_landscapes": int(n), "n_cells": int(len(block)),
        "cost": float(cost), "gain": float(gain), "price": float(price),
        "recovered": float(gain / cost) if cost > 0 else np.nan,
        "bootstrap_draws_kept": float(ok.mean()),
        "recovered_lo": float(np.percentile(draws[ok], 2.5)) if usable else np.nan,
        "recovered_hi": float(np.percentile(draws[ok], 97.5)) if usable else np.nan,
        "wilcoxon_p": p,
    }


def share_flags(block: pd.DataFrame) -> dict:
    """Whether the recovery of ``block`` is a share to quote (share_of_reference).

    recovered stays in the output (the tables and the replays read it), but where
    its denominator, the landscape-mean cost, is near zero or negative on any
    landscape it is not a share: recovered_suppressed says why, and the gain is
    the result. Kept apart from summarise(), whose dict other scripts spread into
    their own outputs.
    """
    per = block.groupby("dataset")[["ref_noisy", "ref_clean", "trt_noisy"]].mean()
    cost = per.ref_noisy - per.ref_clean
    guard = share_of_reference(pd.DataFrame({"ref": cost, "trt": per.trt_noisy - per.ref_clean}))
    return {"n_cost_negative": guard["n_reference_negative"], "recovered_suppressed": guard["share_suppressed"]}


def run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd[1:]))
    subprocess.run(cmd, check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--output-dir", type=Path, default=Path("output-boba/analysis"))
    parser.add_argument("--no-extra-trials", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scales: dict[str, dict[str, float]] = {}

    rows, extra_rows, dataset_rows = [], [], []
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        spec = ARMS[name]
        arm_dir, ref_dir = Path(spec["dir"]), Path(spec["ref"])
        if not any(arm_dir.glob("*/evaluation/paired_excess_metrics.csv")):
            print(f"{name}: no results yet, skipped")
            continue
        seed_set = {int(s) for s in spec["seeds"].split(",")}
        ref_df = load(ref_dir, set(spec["ref_acqs"].split(",")), seed_set, spec["error_model"],
                      spec.get("ref_variant"))
        arm_df = load(arm_dir, set(spec["acqs"].split(",")), seed_set, spec["error_model"],
                      spec.get("variant"))
        if spec["relative"]:
            ref_df, arm_df = rank_magnitudes(ref_df), rank_magnitudes(arm_df)
        kind = spec.get("scale", "opt_z")
        if kind not in scales:
            scales[kind] = dataset_scale(kind)
        print(f"\n=== {name}: {spec['what']} (reference {ref_dir}; scale {kind}) ===")
        for label, response in RESPONSES.items():
            paired = paired_frame(ref_df, arm_df, response, scales[kind], spec["pool"])
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            cond_rows = []
            for (std, onset), block in paired.groupby(["jitter_std", "jitter_iteration"]):
                cond_rows.append({"arm": name, "reference": str(ref_dir), "response": label,
                                  "jitter_std": float(std), "jitter_iteration": int(onset),
                                  **summarise(block, rng), **share_flags(block)})
            pooled = {**summarise(paired, np.random.default_rng(BOOTSTRAP_SEED)), **share_flags(paired)}
            per = per_dataset(paired)
            valid = per["recovered"].dropna()
            ps = [r["wilcoxon_p"] for r in cond_rows]
            for r, q in zip(cond_rows, multipletests(ps, method="fdr_bh")[1]):
                r["wilcoxon_p_fdr"] = float(q)
                for key in ("recovered", "recovered_lo", "recovered_hi", "price", "cost", "gain"):
                    r[f"pooled_{key}"] = pooled[key]
                r["pooled_wilcoxon_p"] = pooled["wilcoxon_p"]
                # With three or four clusters the pooled row is reported by its
                # per-dataset values; these give their range and the count.
                r["pooled_n_landscapes"] = pooled["n_landscapes"]
                r["pooled_recovered_range_lo"] = float(valid.min()) if len(valid) else np.nan
                r["pooled_recovered_range_hi"] = float(valid.max()) if len(valid) else np.nan
                r["scale"] = kind
                r["pooled_n_cost_negative"] = pooled["n_cost_negative"]
                r["pooled_recovered_suppressed"] = pooled["recovered_suppressed"]
            rows.extend(cond_rows)
            dataset_rows.append(per.assign(arm=name, reference=str(ref_dir), response=label, scale=kind,
                                           scale_value=per["dataset"].map(scales[kind])))
            print(f"  {label}: pooled recovered {pooled['recovered']:+.0%} "
                  f"[{pooled['recovered_lo']:+.0%}, {pooled['recovered_hi']:+.0%}], "
                  f"cost {pooled['cost']:.3f}, gain {pooled['gain']:+.3f}, price without error {pooled['price']:+.3f}")
            if pooled["n_landscapes"] <= 4:
                print(f"    only {pooled['n_landscapes']} clusters: the smallest exact two-sided Wilcoxon p "
                      f"is {2 / 2 ** pooled['n_landscapes']:.3g}; per dataset: "
                      + ", ".join(f"{d} {v:+.1%} (cost share {s:.0%}, price {p:+.3f})" for d, v, s, p in
                                  per[["dataset", "recovered", "cost_share", "price"]].itertuples(index=False)))
            if pooled["recovered_suppressed"]:
                print(f"    pooled share not to be quoted ({pooled['recovered_suppressed']}); the gain is the result")
            for r in cond_rows:
                star = "*" if r["wilcoxon_p_fdr"] < 0.05 else " "
                print(f"     {r['jitter_std']:>5g} / it.{r['jitter_iteration'] + 1:<3d} cost {r['cost']:7.3f}  "
                      f"gain {r['gain']:+7.3f}  price {r['price']:+7.3f}  recovered {r['recovered']:+6.0%} "
                      f"[{r['recovered_lo']:+.0%}, {r['recovered_hi']:+.0%}]{star}"
                      + (f"  (no share: {r['recovered_suppressed']})" if r["recovered_suppressed"] else ""))

        # analyse_extra_runs.py reads a whole directory, so an arm that is one
        # variant among several in its directory (spike sizes, cap modes) would
        # mix them; those arms have no extra-trial rows.
        if args.no_extra_trials or spec["relative"] or spec["pool"] or spec.get("variant"):
            continue
        # Extra trials against the STANDARD clean run: the arm's noisy runs and the
        # reference's own noisy runs, both targeting the reference's clean runs.
        tag = name.replace("-", "_")
        vs_std = arm_dir / "analysis" / f"extra_runs_vs_{tag}_reference.csv"
        if not vs_std.is_file():
            run([PY, str(SCRIPT_DIR / "analyse_extra_runs.py"), "--input-dir", str(arm_dir), "--baseline-dir",
                 str(ref_dir), "--k", "10,25", "--tolerance", "0.01", "--acquisitions", spec["acqs"],
                 "--seeds", spec["seeds"], "--output-name", f"extra_runs_vs_{tag}_reference"])
        ref_out = ref_dir / "analysis" / f"extra_runs_ref_{tag}.csv"
        if not ref_out.is_file():
            run([PY, str(SCRIPT_DIR / "analyse_extra_runs.py"), "--input-dir", str(ref_dir), "--k", "10,25",
                 "--tolerance", "0.01", "--acquisitions", spec["acqs"], "--seeds", spec["seeds"],
                 "--output-name", f"extra_runs_ref_{tag}"])
        e_arm, e_ref = pd.read_csv(vs_std), pd.read_csv(ref_out)
        e_arm = e_arm[e_arm["error_model"] == spec["error_model"]]
        # The reference directory can hold several error processes and bias
        # variants; the arm's own process, with no variant, is the matched row.
        e_ref = e_ref[(e_ref["error_model"] == spec["error_model"]) & (e_ref["variant"].fillna("") == "")]
        key = ["jitter_std", "jitter_iteration", "k", "tolerance"]
        merged = e_arm.merge(e_ref, on=key, suffixes=("_arm", "_ref"), validate="one_to_one")
        for r in merged.itertuples():
            extra_rows.append({"arm": name, "jitter_std": r.jitter_std, "jitter_iteration": r.jitter_iteration,
                               "k": r.k, "tolerance": r.tolerance,
                               "median_extra_arm": r.median_extra_arm, "never_arm": r.censored_fraction_arm,
                               "mean_extra_arm": r.mean_extra_arm, "median_extra_ref": r.median_extra_ref,
                               "never_ref": r.censored_fraction_ref, "mean_extra_ref": r.mean_extra_ref})
        early = merged[(merged.jitter_iteration == merged.jitter_iteration.min()) & (merged.k == 10)]
        print("  extra trials to match a clean STANDARD 10-trial study, error from it. 1 "
              "(median / never, arm vs reference): "
              + "; ".join(f"{r.jitter_std:g}: {r.median_extra_arm:.0f}/{r.censored_fraction_arm:.0%} vs "
                          f"{r.median_extra_ref:.0f}/{r.censored_fraction_ref:.0%}" for r in early.itertuples()))

    out = pd.DataFrame(rows)
    # Merge, never overwrite. A call with --arms used to truncate the file to the
    # arms it ran, which is how the budget-neutral numbers came to exist in no
    # artefact at all. Rows for the arms in THIS call replace their old selves;
    # every other arm is kept exactly as it was.
    recovery_path = args.output_dir / "adaptations_recovery.csv"
    if not len(out):
        print("\nNone of the requested arms has results yet; nothing written.")
        return
    if recovery_path.is_file():
        previous = pd.read_csv(recovery_path)
        kept = previous[~previous["arm"].isin(set(out["arm"]))]
        if len(kept):
            print(f"  keeping {kept['arm'].nunique()} arm(s) "
                  f"already in {recovery_path.name}")
        out = pd.concat([kept, out], ignore_index=True)
    # The pooled p was never corrected anywhere, yet the paper's table stars on it.
    # One BH family per response over every arm in the merged file, taking each
    # arm's pooled p once; correcting only the arms of this call made a one-arm
    # call's q equal its p.
    out["pooled_wilcoxon_p_fdr"] = np.nan
    for response, blk in out.groupby("response"):
        one = blk.drop_duplicates(subset=["arm"])[["arm", "pooled_wilcoxon_p"]].dropna()
        if len(one):
            q = dict(zip(one["arm"], multipletests(one["pooled_wilcoxon_p"], method="fdr_bh")[1]))
            sel = out["response"] == response
            out.loc[sel, "pooled_wilcoxon_p_fdr"] = out.loc[sel, "arm"].map(q).astype(float)
    out = out.sort_values(["arm", "response", "jitter_std", "jitter_iteration"], kind="stable")
    out.to_csv(recovery_path, index=False)
    # Per-dataset values, merged on arm the same way.
    per_path = args.output_dir / "adaptations_per_dataset.csv"
    per_out = pd.concat(dataset_rows, ignore_index=True)[
        ["arm", "reference", "response", "scale", "dataset", "scale_value", "n_cells",
         "cost", "gain", "price", "recovered", "cost_share"]]
    if per_path.is_file():
        previous = pd.read_csv(per_path)
        per_out = pd.concat([previous[~previous["arm"].isin(set(per_out["arm"]))], per_out], ignore_index=True)
    per_out.sort_values(["arm", "response", "dataset"], kind="stable").to_csv(per_path, index=False)
    # The extra-trials file is merged the same way, and a call that computed no
    # extra trials (--no-extra-trials, or only relative/pooled arms) leaves it
    # alone: on 2026-09-22 such calls truncated it to an empty frame, and the
    # extra-trial numbers of the process-adaptations appendix went with it.
    extra_path = args.output_dir / "adaptations_extra_runs.csv"
    extra = pd.DataFrame(extra_rows)
    if len(extra):
        if extra_path.is_file() and extra_path.stat().st_size > 10:
            previous = pd.read_csv(extra_path)
            if "arm" in previous.columns:
                extra = pd.concat([previous[~previous["arm"].isin(set(extra["arm"]))], extra],
                                  ignore_index=True)
        extra.to_csv(extra_path, index=False)
    print(f"\nWrote {args.output_dir / 'adaptations_recovery.csv'}, adaptations_per_dataset.csv "
          f"and adaptations_extra_runs.csv")


if __name__ == "__main__":
    main()
