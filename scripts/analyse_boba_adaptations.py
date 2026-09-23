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
response. Regret is divided by opt_z where the landscape has one, so cost, gain
and price are fractions of the achievable improvement.

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


def arm(dir_, ref, acqs, what, seeds=S5, ref_acqs=None, pool=False, relative=False, error_model=None,
        variant=None, ref_variant=None):
    # variant/ref_variant select one condition out of a directory that holds
    # several (four spike sizes, two ceiling modes, a halo rho). None means "the
    # whole directory", which is every arm that predates the variant column.
    return dict(dir=dir_, ref=ref, acqs=acqs, ref_acqs=ref_acqs or acqs, seeds=seeds, what=what,
                pool=pool, relative=relative, error_model=error_model,
                variant=variant, ref_variant=ref_variant)


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
    "fitted-rep10": arm("output-fitted-adapt-rep10", "output-fitted", "logei,qnei",
                        "first ten rated twice, on the three fitted-oracle datasets",
                        seeds="7,8,9,10,11,12,13,14,15,16", relative=True, error_model="gaussian"),

    # ---------------------------------------------------------------- budget-neutral
    # The 2026-09-14/16 sweep (run_boba_budget_neutral.ps1). Every arm here keeps
    # the number of human trials equal to the standard process, or lowers it, so
    # each is scored against the standard process on the same trial budget. The
    # arms whose variants differ only in a filename SUFFIX (spike, spike-clip,
    # spike-rrp, relay, ceiling, mo-halo, mo-halo-backfit) are absent: one
    # directory holds several conditions and evaluate_research_question.py
    # refuses to average them, correctly. They need a variant-aware evaluation
    # before they can appear here.
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
                        seeds=S10, error_model="gaussian", variant="xc0.85", ref_variant=""),
    "mo-halo-backfit": arm("output-boba-mo-halo-backfit", "output-boba-mo-halo", "qlognehvi",
                           "the shared factor estimated and removed, under halo error",
                           seeds=S10, error_model="gaussian", variant="xc0.85_halo-backfit",
                           ref_variant="xc0.85"),
    "mo-halo-backfit-price": arm("output-boba-mo-halo-backfit", "output-boba-mo-halo", "qlognehvi",
                                 "the same model where there is no shared factor to remove",
                                 seeds=S10, error_model="gaussian", variant="halo-backfit",
                                 ref_variant=""),
}

# The gross-fault family. Four spike sizes share one directory each, so every
# comparison names its own: both remedies are scored against the SAME spike size
# with no remedy, never against a different one.
for _sp in ("sp0.05-5", "sp0.05-20", "sp0.15-5", "sp0.15-20"):
    _prob, _size = _sp[2:].split("-")
    ARMS[f"spike-clip-{_sp}"] = arm(
        "output-boba-spike-clip", "output-boba-spike", "logei,qnei",
        f"the response clipped to the observed range, {_prob} of trials spiking at {_size} SD",
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


def paired_frame(ref: pd.DataFrame, trt: pd.DataFrame, response: str, opt_z: dict[str, float],
                 pool: bool) -> pd.DataFrame:
    cols = [f"{response}_jitter", f"{response}_baseline"]
    keys = POOLED_KEYS if pool else PAIR_KEYS
    if pool:
        ref = ref.groupby(POOLED_KEYS, as_index=False)[cols].mean()
        trt = trt.groupby(POOLED_KEYS, as_index=False)[cols].mean()
    m = ref[keys + cols].merge(trt[keys + cols], on=keys, suffixes=("_ref", "_trt"), validate="one_to_one")
    if m.empty:
        raise ValueError("the two arms share no paired cells")
    z = m["dataset"].map(lambda d: opt_z.get(d, 1.0))
    return m.assign(
        ref_noisy=m[f"{response}_jitter_ref"] / z, ref_clean=m[f"{response}_baseline_ref"] / z,
        trt_noisy=m[f"{response}_jitter_trt"] / z, trt_clean=m[f"{response}_baseline_trt"] / z,
    )


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
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    opt_z = {k: float(v["opt_z"]) for k, v in stats.items() if "opt_z" in v}

    rows, extra_rows = [], []
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
        print(f"\n=== {name}: {spec['what']} (reference {ref_dir}) ===")
        for label, response in RESPONSES.items():
            paired = paired_frame(ref_df, arm_df, response, {} if spec["relative"] else opt_z, spec["pool"])
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            cond_rows = []
            for (std, onset), block in paired.groupby(["jitter_std", "jitter_iteration"]):
                cond_rows.append({"arm": name, "reference": str(ref_dir), "response": label,
                                  "jitter_std": float(std), "jitter_iteration": int(onset),
                                  **summarise(block, rng)})
            pooled = summarise(paired, np.random.default_rng(BOOTSTRAP_SEED))
            ps = [r["wilcoxon_p"] for r in cond_rows]
            for r, q in zip(cond_rows, multipletests(ps, method="fdr_bh")[1]):
                r["wilcoxon_p_fdr"] = float(q)
                for key in ("recovered", "recovered_lo", "recovered_hi", "price", "cost", "gain"):
                    r[f"pooled_{key}"] = pooled[key]
                r["pooled_wilcoxon_p"] = pooled["wilcoxon_p"]
            rows.extend(cond_rows)
            print(f"  {label}: pooled recovered {pooled['recovered']:+.0%} "
                  f"[{pooled['recovered_lo']:+.0%}, {pooled['recovered_hi']:+.0%}], "
                  f"cost {pooled['cost']:.3f}, gain {pooled['gain']:+.3f}, price without error {pooled['price']:+.3f}")
            for r in cond_rows:
                star = "*" if r["wilcoxon_p_fdr"] < 0.05 else " "
                print(f"     {r['jitter_std']:>5g} / it.{r['jitter_iteration'] + 1:<3d} cost {r['cost']:7.3f}  "
                      f"gain {r['gain']:+7.3f}  price {r['price']:+7.3f}  recovered {r['recovered']:+6.0%} "
                      f"[{r['recovered_lo']:+.0%}, {r['recovered_hi']:+.0%}]{star}")

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
    # The pooled p was never corrected anywhere, yet the paper's table stars on it.
    # One BH family per response over the arms, taking each arm's pooled p once.
    if len(out):
        out["pooled_wilcoxon_p_fdr"] = np.nan
        for response, blk in out.groupby("response"):
            one = blk.drop_duplicates(subset=["arm"])[["arm", "pooled_wilcoxon_p"]].dropna()
            if len(one):
                q = dict(zip(one["arm"], multipletests(one["pooled_wilcoxon_p"], method="fdr_bh")[1]))
                sel = out["response"] == response
                out.loc[sel, "pooled_wilcoxon_p_fdr"] = out.loc[sel, "arm"].map(q).astype(float)
    # Merge, never overwrite. A call with --arms used to truncate the file to the
    # arms it ran, which is how the budget-neutral numbers came to exist in no
    # artefact at all. Rows for the arms in THIS call replace their old selves;
    # every other arm is kept exactly as it was.
    recovery_path = args.output_dir / "adaptations_recovery.csv"
    if recovery_path.is_file() and len(out):
        previous = pd.read_csv(recovery_path)
        kept = previous[~previous["arm"].isin(set(out["arm"]))]
        if len(kept):
            print(f"  keeping {kept['arm'].nunique()} arm(s) "
                  f"already in {recovery_path.name}")
        out = pd.concat([kept, out], ignore_index=True)
    out = out.sort_values(["arm", "response", "jitter_std", "jitter_iteration"], kind="stable")
    out.to_csv(recovery_path, index=False)
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
    print(f"\nWrote {args.output_dir / 'adaptations_recovery.csv'} and adaptations_extra_runs.csv")


if __name__ == "__main__":
    main()
