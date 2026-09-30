"""Is the CUSUM the right detector for a rater who changes mid-study?

scripts/replay_stopping.py detects a mid-study change with a one-sided CUSUM on
the squared one-step-ahead standardised residual of the loop's own GP. That rule
works -- it recovers 42 to 64 per cent of the deployed cost when a large fault
arrives at trial 21 -- but a CUSUM is a choice, not a derivation, and it was
never compared with the alternatives. This script compares three detectors on
exactly the same residual streams, so any difference is the detector's.

The signal. With the GP's hyperparameters and standardisation frozen on the first
n0 trials, the one-step-ahead residual

    z_t = (y_t - mu_{t-1}(x_t)) / sqrt(var_{t-1}(x_t) + noise)

is standard normal while the rater is behaving, whatever the landscape and
whatever the acquisition. A rater who becomes noisier inflates its VARIANCE, so
every detector here tests N(0,1) against N(0, sigma^2), sigma > 1.

    cusum   S_t = max(0, S_{t-1} + z_t^2 - 1 - kappa), alarm on S_t > h.
            The incumbent rule. One accumulator, one threshold, no model of when
            the change happened.
    glr     the generalised likelihood ratio for the same alternative, maximised
            over the unknown change point: for each candidate tau the maximised
            log-likelihood ratio has the closed form
                (n/2)(sigma_hat^2 - 1 - log sigma_hat^2),  sigma_hat^2 = mean of z^2 after tau
            and the statistic is its maximum over tau, alarm on max > h. This is
            the classical answer to "a variance change at an unknown time", and
            it is what a CUSUM approximates.
    bocpd   Bayesian online change-point detection (Adams & MacKay, 2007) with a
            constant hazard and a normal--inverse-gamma model of the post-change
            variance, alarming when the posterior probability that the current
            run is short, P(r_t <= r_max), exceeds a threshold. Unlike the other
            two it carries a distribution over WHEN the change happened, which is
            what a freeze rule needs in order to decide how much to discard.

Matched false alarms. Comparing detection rates at whatever threshold each
happens to use would say nothing, so every detector's threshold is tuned on the
TUNING seeds to a common false-alarm budget and then scored on held-out seeds. An
alarm is false when the rater never changed (a clean run, or a run whose error
was there from trial 1 -- in both, nothing changes after n0) or when it fires at
or before the onset of a late change.

Reads the residual cache that replay_stopping.py already wrote, so it refits
nothing.

The freeze rule's own operating point. The comparison above re-tunes the CUSUM
from its own kappa grid, so the CUSUM it compares is not the one the freeze rule
runs: replay_stopping.py tunes (h, kappa, w) to deployed regret and lands on
kappa = 16, h = 2000, w = 3 (stopping_tuned_A.json). w only chooses which prefix
the shipped design is refitted on, so the detector is (kappa, h). The script
therefore also writes changepoint_freeze_point.csv:

    freeze_rule_fixed   that CUSUM at its own (kappa, h), tuning and held-out
                        seeds, with the false alarms split into clean runs and
                        runs with error from trial 1;
    freeze_rule_budget  every detector (the re-tuned CUSUM with the freeze
                        kappa added to its grid, the CUSUM at the freeze kappa,
                        the GLR, BOCPD) re-thresholded to the freeze rule's own
                        pooled false-alarm rate on the tuning seeds and scored on
                        the held-out seeds, pooled and by magnitude;
    budget              the CUSUM at the freeze kappa re-thresholded to each
                        common budget, beside the kappa-grid CUSUM and the GLR.

and checks that the fixed operating point reproduces the alarm trial rule A
logged for every run in stopping_per_run.csv. changepoint_metadata.json records
the arguments.

    python scripts/changepoint_compare.py
    python scripts/changepoint_compare.py --false-alarm 0.05,0.10,0.20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rescore_ship_rules import parse_run_name  # noqa: E402

DETECTORS = ("cusum", "glr", "bocpd")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path,
                   default=Path("output-boba/analysis/stopping/cache_n015_rep3"),
                   help="the residual cache written by replay_stopping.py")
    p.add_argument("--output-dir", type=Path, default=Path("output-boba/analysis/changepoint"))
    p.add_argument("--n0", type=int, default=15, help="trials the frozen GP was fitted on")
    p.add_argument("--tune-seeds", type=str, default="7,8,9,10,11")
    p.add_argument("--score-seeds", type=str, default="12,13,14,15,16")
    p.add_argument("--false-alarm", type=str, default="0.05,0.10,0.20")
    p.add_argument("--kappa-grid", type=str, default="0.25,0.5,1.0,2.0")
    p.add_argument("--hazard-grid", type=str, default="0.01,0.02,0.05")
    p.add_argument("--rmax", type=int, default=5,
                   help="bocpd alarms when P(run length <= rmax) exceeds its threshold")
    p.add_argument("--freeze-params", type=str,
                   default="output-boba/analysis/stopping/stopping_tuned_A.json",
                   help="the freeze rule's tuned (h, kappa, w) written by replay_stopping.py; its CUSUM "
                        "is scored at that fixed operating point and every detector is re-tuned to its "
                        "false-alarm rate ('none' skips this)")
    p.add_argument("--per-run", type=str,
                   default="output-boba/analysis/stopping/stopping_per_run.csv",
                   help="replay_stopping.py's per-run table: the fixed operating point must reproduce "
                        "rule A's logged alarm trial on every run ('none' skips the check)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Detectors. Each maps a (runs x monitored trials) matrix of z to a statistic
# path of the same shape; an alarm is the first column whose value exceeds h.
# ---------------------------------------------------------------------------


def cusum_path(z: np.ndarray, kappa: float) -> np.ndarray:
    z2 = z ** 2
    out = np.zeros_like(z2)
    s = np.zeros(z2.shape[0])
    for j in range(z2.shape[1]):
        s = np.maximum(0.0, s + z2[:, j] - 1.0 - kappa)
        out[:, j] = s
    return out


def glr_path(z: np.ndarray) -> np.ndarray:
    """max over tau of the maximised log-likelihood ratio for a variance change.

    For observations after tau, with sigma_hat^2 their mean square, the maximised
    log ratio of N(0, sigma^2) to N(0, 1) is (n/2)(sigma_hat^2 - 1 - log sigma_hat^2),
    which is zero at sigma_hat^2 = 1 and increasing away from it. Only inflation
    is of interest, so sigma_hat^2 below 1 contributes nothing.
    """
    z2 = z ** 2
    n_runs, n_cols = z2.shape
    csum = np.concatenate([np.zeros((n_runs, 1)), np.cumsum(z2, axis=1)], axis=1)
    out = np.zeros_like(z2)
    for j in range(n_cols):
        best = np.zeros(n_runs)
        for tau in range(j + 1):
            n = j - tau + 1
            s2 = (csum[:, j + 1] - csum[:, tau]) / n
            s2 = np.maximum(s2, 1.0)          # one-sided: only an increase counts
            stat = 0.5 * n * (s2 - 1.0 - np.log(s2))
            best = np.maximum(best, stat)
        out[:, j] = best
    return out


def _student_t_logpdf(x: np.ndarray, nu: np.ndarray, scale2: np.ndarray) -> np.ndarray:
    """log density of a zero-mean Student-t, the NIG posterior predictive."""
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu * scale2)
            - (nu + 1) / 2 * np.log1p(x ** 2 / (nu * scale2)))


def bocpd_path(z: np.ndarray, hazard: float, rmax: int, a0: float = 1.0,
               b0: float = 1.0) -> np.ndarray:
    """P(run length <= rmax) after each observation, per run.

    Adams & MacKay's recursion with a constant hazard and a zero-mean normal
    observation whose variance has an inverse-gamma prior, so the predictive is a
    Student-t. A short run means the detector believes a change happened
    recently, which is exactly what a freeze rule acts on.
    """
    n_runs, n_cols = z.shape
    out = np.zeros_like(z)
    # Run-length posterior, growing by one column per step.
    rl = np.ones((n_runs, 1))
    a = np.full((n_runs, 1), a0)
    b = np.full((n_runs, 1), b0)
    for j in range(n_cols):
        x = z[:, j : j + 1]
        nu = 2 * a
        scale2 = b / a
        pred = np.exp(_student_t_logpdf(x, nu, scale2))
        growth = rl * pred * (1.0 - hazard)
        change = (rl * pred * hazard).sum(axis=1, keepdims=True)
        rl = np.concatenate([change, growth], axis=1)
        total = rl.sum(axis=1, keepdims=True)
        rl = rl / np.where(total > 0, total, 1.0)
        # Sufficient statistics: a new run starts at the prior, an old one absorbs x.
        a = np.concatenate([np.full((n_runs, 1), a0), a + 0.5], axis=1)
        b = np.concatenate([np.full((n_runs, 1), b0), b + 0.5 * x ** 2], axis=1)
        # Before the run-length support reaches rmax every run IS short, so the
        # statistic is identically 1 and says nothing; reporting it would put the
        # tuned threshold at 1 and silence the detector entirely.
        out[:, j] = rl[:, : rmax + 1].sum(axis=1) if j >= rmax else 0.0
    return out


def first_alarm(path: np.ndarray, h: float, n0: int) -> np.ndarray:
    """Trial of the first value above h, 0 for none. Column j is trial n0+1+j."""
    if not np.isfinite(h):
        return np.zeros(path.shape[0], dtype=int)
    hit = path > h
    return np.where(hit.any(axis=1), n0 + 1 + hit.argmax(axis=1), 0).astype(int)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_cache(root: Path) -> pd.DataFrame:
    if not root.is_dir():
        raise SystemExit(f"{root} not found: run scripts/replay_stopping.py first")
    rows = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            z = rec.get("z")
            if not z or not np.all(np.isfinite(z)) or not rec.get("z_fit_ok", True):
                continue
            # The cache stores the residuals and the run's NAME; the condition
            # lives in the name, and rescore_ship_rules already knows how to read
            # it, so that parser is reused rather than a second one written.
            info = parse_run_name(Path(rec["file"]).name)
            if info is None:
                continue
            rows.append({"file": rec["file"], "z": np.asarray(z, dtype=float),
                         "dataset": info["dataset"], "acquisition": info["acquisition"],
                         "seed": info["seed"], "error_model": info["error_model"],
                         "jitter_std": info["jitter_std"],
                         "jitter_iteration": info["jitter_iteration"]})
    if not rows:
        raise SystemExit(f"no usable residual streams in {root}")
    frame = pd.DataFrame(rows)
    # One stream length, or the matrix below is ragged.
    lengths = {len(z) for z in frame["z"]}
    if len(lengths) != 1:
        keep = max(lengths, key=lambda L: (frame["z"].map(len) == L).sum())
        frame = frame[frame["z"].map(len) == keep]
    return frame.reset_index(drop=True)


def label_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """changed: the rater really does change after n0, so an alarm can be true."""
    onset = frame["jitter_iteration"].fillna(0).astype(int)
    clean = frame["error_model"].astype(str).eq("none")
    frame = frame.assign(onset=onset, clean=clean, changed=(~clean) & (onset > 0))
    return frame


def rates(alarms: np.ndarray, block: pd.DataFrame) -> dict:
    """Detection and false-alarm rates for one detector at one threshold."""
    changed = block["changed"].to_numpy()
    onset = block["onset"].to_numpy()
    fired = alarms > 0
    # False: fired although nothing changes after n0, or fired at/before the onset.
    false = (fired & ~changed) | (fired & changed & (alarms <= onset))
    true = fired & changed & (alarms > onset)
    delay = np.where(true, alarms - onset, np.nan)
    n_steady = int((~changed).sum())
    return {
        "false_alarm_rate": float(false[~changed].mean()) if n_steady else float("nan"),
        "detection_rate": float(true[changed].mean()) if changed.any() else float("nan"),
        "median_delay": float(np.nanmedian(delay)) if np.isfinite(delay).any() else float("nan"),
        "n_changed": int(changed.sum()), "n_steady": n_steady,
    }


def threshold_for(path: np.ndarray, block: pd.DataFrame, target: float, n0: int) -> float:
    """The smallest threshold whose false-alarm rate on steady runs is <= target.

    Taken as a quantile of each steady run's running maximum: a run alarms iff
    that maximum exceeds the threshold, so the (1 - target) quantile is exactly
    the tightest threshold meeting the budget.
    """
    steady = ~block["changed"].to_numpy()
    if not steady.any():
        return float("inf")
    peaks = path[steady].max(axis=1)
    return float(np.quantile(peaks, 1.0 - target))


# ---------------------------------------------------------------------------
# The freeze rule's own operating point
# ---------------------------------------------------------------------------


def load_freeze_params(raw: str | None) -> dict | None:
    """(h, kappa, w) of the freeze rule, from replay_stopping.py's stopping_tuned_A.json."""
    if raw is None or str(raw).strip().lower() in ("", "none"):
        return None
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"{path} not found: run scripts/replay_stopping.py first, or pass --freeze-params none")
    got = json.loads(path.read_text(encoding="utf-8"))
    h, kappa, w = float(got["h"]), float(got["kappa"]), int(got["w"])
    if not np.isfinite(h):
        raise SystemExit(f"{path} holds the never-stop rule (h = inf); there is no operating point to score")
    return {"h": h, "kappa": kappa, "w": w, "source": path.as_posix()}


def steady_split(alarms: np.ndarray, block: pd.DataFrame) -> dict:
    """False alarms split as replay_stopping.py constrains them, plus pre-onset alarms on late runs.

    Clean runs and runs with error from trial 1 are the two kinds of steady run;
    the freeze rule's tuning caps each at 10% separately, while the comparison
    above budgets their pool.
    """
    clean = block["clean"].to_numpy(bool)
    changed = block["changed"].to_numpy(bool)
    onset0 = ~clean & ~changed
    fired = alarms > 0
    early = fired & changed & (alarms <= block["onset"].to_numpy())
    return {
        "false_alarm_clean": float(fired[clean].mean()) if clean.any() else float("nan"),
        "false_alarm_onset0": float(fired[onset0].mean()) if onset0.any() else float("nan"),
        "pre_onset_alarm_late": float(early[changed].mean()) if changed.any() else float("nan"),
        "n_clean": int(clean.sum()), "n_onset0": int(onset0.sum()),
    }


def _point_row(comparison: str, detector: str, params: dict, h: float, target: float, split: str,
               alarms: np.ndarray, block: pd.DataFrame, magnitude="all") -> dict:
    return {"comparison": comparison, "detector": detector, "params": json.dumps(params),
            "threshold": h, "target_false_alarm": target, "split": split, "jitter_std": magnitude,
            **rates(alarms, block), **steady_split(alarms, block)}


def freeze_point_table(tune: pd.DataFrame, score: pd.DataFrame, Z_tune: np.ndarray, Z_score: np.ndarray,
                       freeze: dict, kappas: list[float], hazards: list[float], targets: list[float],
                       rmax: int, n0: int) -> pd.DataFrame:
    """The freeze rule's CUSUM at its own (kappa, h), and every detector at its false-alarm rate.

    Three blocks (column ``comparison``):

    freeze_rule_fixed   the CUSUM at the freeze rule's (kappa, h), not re-tuned, on
                        both seed splits;
    freeze_rule_budget  each detector's threshold tuned on the tuning seeds to the
                        freeze rule's own pooled steady false-alarm rate there,
                        then scored on the held-out seeds (pooled, and by
                        magnitude, where the false-alarm rate is that of the runs
                        with error from trial 1 at that magnitude);
    budget              at each common budget, the CUSUM at the freeze kappa
                        beside the kappa-grid CUSUM and the GLR, as scored in
                        changepoint_detectors.csv.

    Within a family (the kappa grid, the hazard grid) the variant with the highest
    tuning-seed detection rate is kept, exactly as in the main comparison.
    """
    kf, hf = freeze["kappa"], freeze["h"]
    fixed = {"kappa": kf}
    rows = []
    fixed_tune = first_alarm(cusum_path(Z_tune, kf), hf, n0)
    fixed_score = first_alarm(cusum_path(Z_score, kf), hf, n0)
    params_fixed = {"kappa": kf, "h": hf, "w": freeze["w"]}
    rows.append(_point_row("freeze_rule_fixed", "cusum_freeze_rule", params_fixed, hf, float("nan"),
                           "tune", fixed_tune, tune))
    rows.append(_point_row("freeze_rule_fixed", "cusum_freeze_rule", params_fixed, hf, float("nan"),
                           "score", fixed_score, score))
    budget_freeze = rates(fixed_tune, tune)["false_alarm_rate"]

    grid_plus = sorted(set(kappas) | {kf})
    families = {
        "cusum": [("cusum", {"kappa": k}) for k in grid_plus],
        "cusum_freeze_kappa": [("cusum", fixed)],
        "glr": [("glr", {})],
        "bocpd": [("bocpd", {"hazard": h}) for h in hazards],
    }

    def build(name: str, params: dict, Z: np.ndarray) -> np.ndarray:
        if name == "cusum":
            return cusum_path(Z, params["kappa"])
        if name == "glr":
            return glr_path(Z)
        return bocpd_path(Z, params["hazard"], rmax)

    def tuned(variants, target):
        best = None
        for name, params in variants:
            p_tune = build(name, params, Z_tune)
            h = threshold_for(p_tune, tune, target, n0)
            got = rates(first_alarm(p_tune, h, n0), tune)
            if best is None or got["detection_rate"] > best["tune_detection"]:
                best = {"name": name, "params": params, "h": h, "tune_detection": got["detection_rate"],
                        "tune_false_alarm": got["false_alarm_rate"]}
        return best

    magnitudes = sorted(score.loc[score["changed"], "jitter_std"].dropna().unique())
    for label, variants in families.items():
        best = tuned(variants, budget_freeze)
        alarms = first_alarm(build(best["name"], best["params"], Z_score), best["h"], n0)
        extra = {"tune_detection": best["tune_detection"], "tune_false_alarm": best["tune_false_alarm"]}
        rows.append({**_point_row("freeze_rule_budget", label, best["params"], best["h"], budget_freeze,
                                  "score", alarms, score), **extra})
        for m in magnitudes:
            sel = (score["jitter_std"] == m).to_numpy()
            rows.append({**_point_row("freeze_rule_budget", label, best["params"], best["h"], budget_freeze,
                                      "score", alarms[sel], score[sel].reset_index(drop=True),
                                      magnitude=float(m)), **extra})

    budget_families = {"cusum": [("cusum", {"kappa": k}) for k in kappas],
                       "cusum_freeze_kappa": [("cusum", fixed)], "glr": [("glr", {})]}
    for target in targets:
        for label, variants in budget_families.items():
            best = tuned(variants, target)
            alarms = first_alarm(build(best["name"], best["params"], Z_score), best["h"], n0)
            rows.append({**_point_row("budget", label, best["params"], best["h"], target, "score", alarms, score),
                         "tune_detection": best["tune_detection"], "tune_false_alarm": best["tune_false_alarm"]})
    return pd.DataFrame(rows)


def check_against_per_run(path: Path, frame: pd.DataFrame, freeze: dict, n0: int) -> dict:
    """The fixed operating point must reproduce rule A's logged alarm trial on every run.

    stopping_per_run.csv logs rule A's stop trial (empty when it never alarmed)
    and the (h, kappa, w) it ran with. A mismatch means the cache and the
    per-run table are from different analyses, and the fixed point is not the
    freeze rule's detector.
    """
    per = pd.read_csv(path, usecols=["file", "rule", "stop_t", "h", "kappa", "w"])
    per = per[per["rule"] == "A"]
    logged_params = {(float(h), float(k), int(w)) for h, k, w in per[["h", "kappa", "w"]].itertuples(index=False)}
    if logged_params != {(freeze["h"], freeze["kappa"], freeze["w"])}:
        raise SystemExit(f"{path} ran rule A at {sorted(logged_params)}, not at the freeze parameters {freeze}")
    logged = dict(zip(per["file"].map(lambda f: Path(f).name), per["stop_t"].fillna(0).astype(int)))
    alarms = first_alarm(cusum_path(np.stack(frame["z"].to_numpy()), freeze["kappa"]), freeze["h"], n0)
    names = frame["file"].map(lambda f: Path(f).name)
    missing = int((~names.isin(logged.keys())).sum())
    mismatches = int(sum(logged.get(n, -1) != int(a) for n, a in zip(names, alarms)))
    return {"per_run_path": Path(path).as_posix(), "per_run_rows_rule_A": int(len(per)),
            "streams_checked": int(len(frame)), "streams_missing_from_per_run": missing,
            "alarm_mismatches": mismatches}


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = label_runs(load_cache(args.cache))

    tune_seeds = {int(s) for s in args.tune_seeds.split(",")}
    score_seeds = {int(s) for s in args.score_seeds.split(",")}
    targets = [float(t) for t in args.false_alarm.split(",")]
    kappas = [float(k) for k in args.kappa_grid.split(",")]
    hazards = [float(h) for h in args.hazard_grid.split(",")]

    tune = frame[frame["seed"].isin(tune_seeds)].reset_index(drop=True)
    score = frame[frame["seed"].isin(score_seeds)].reset_index(drop=True)
    if tune.empty or score.empty:
        raise SystemExit("the cache holds no runs for one of the seed splits")
    Z_tune = np.stack(tune["z"].to_numpy())
    Z_score = np.stack(score["z"].to_numpy())
    print(f"{len(frame):,} residual streams of length {Z_tune.shape[1]}: "
          f"{len(tune):,} tuning, {len(score):,} scoring; "
          f"{int(frame['changed'].sum()):,} with a real mid-study change")

    variants = ([("cusum", {"kappa": k}) for k in kappas]
                + [("glr", {})]
                + [("bocpd", {"hazard": h}) for h in hazards])

    def build(name: str, params: dict, Z: np.ndarray) -> np.ndarray:
        if name == "cusum":
            return cusum_path(Z, params["kappa"])
        if name == "glr":
            return glr_path(Z)
        return bocpd_path(Z, params["hazard"], args.rmax)

    rows = []
    for target in targets:
        best: dict[str, dict] = {}
        for name, params in variants:
            p_tune = build(name, params, Z_tune)
            h = threshold_for(p_tune, tune, target, args.n0)
            got = rates(first_alarm(p_tune, h, args.n0), tune)
            key = name
            if key not in best or got["detection_rate"] > best[key]["tune"]["detection_rate"]:
                best[key] = {"params": params, "h": h, "tune": got}
        for name, chosen in best.items():
            p_score = build(name, chosen["params"], Z_score)
            got = rates(first_alarm(p_score, chosen["h"], args.n0), score)
            rows.append({"detector": name, "target_false_alarm": target,
                         "params": json.dumps(chosen["params"]), "threshold": chosen["h"],
                         "tune_detection": chosen["tune"]["detection_rate"],
                         "tune_false_alarm": chosen["tune"]["false_alarm_rate"], **got})

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "changepoint_detectors.csv", index=False)

    print("\nHeld-out seeds, thresholds tuned to a common false-alarm budget:\n")
    print(f"  {'budget':>7}  {'detector':<8} {'params':<18} {'detected':>9} {'false':>7} {'delay':>6}")
    for target in targets:
        for _, r in summary[summary["target_false_alarm"] == target].sort_values(
                "detection_rate", ascending=False).iterrows():
            print(f"  {target * 100:6.0f}%  {r['detector']:<8} {r['params']:<18} "
                  f"{r['detection_rate'] * 100:8.1f}% {r['false_alarm_rate'] * 100:6.1f}% "
                  f"{r['median_delay']:6.1f}")
    print(f"\nWrote {args.output_dir / 'changepoint_detectors.csv'}")

    metadata = {"args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "n_streams": int(len(frame)), "stream_length": int(Z_tune.shape[1]),
                "n_tune": int(len(tune)), "n_score": int(len(score)),
                "n_changed": int(frame["changed"].sum())}
    freeze = load_freeze_params(args.freeze_params)
    if freeze is not None:
        metadata["freeze_params"] = freeze
        if args.per_run and str(args.per_run).strip().lower() not in ("", "none") and Path(args.per_run).is_file():
            check = check_against_per_run(Path(args.per_run), frame, freeze, args.n0)
            metadata["per_run_check"] = check
            if check["alarm_mismatches"] or check["streams_missing_from_per_run"]:
                raise SystemExit(f"the freeze rule's operating point does not reproduce rule A's logged "
                                 f"alarms: {check}")
            print(f"\nfreeze rule CUSUM (kappa = {freeze['kappa']:g}, h = {freeze['h']:g}) reproduces rule A's "
                  f"logged alarm trial on all {check['streams_checked']:,} streams")
        point = freeze_point_table(tune, score, Z_tune, Z_score, freeze, kappas, hazards, targets,
                                   args.rmax, args.n0)
        point.to_csv(args.output_dir / "changepoint_freeze_point.csv", index=False)
        pooled = point[point["jitter_std"].astype(str) == "all"]
        print("\nThe freeze rule's operating point (w only picks the refit prefix, so the detector is (kappa, h)):\n")
        print(f"  {'block':<19} {'detector':<19} {'split':<6} {'budget':>7} {'detected':>9} {'false':>7} "
              f"{'clean':>6} {'onset0':>7} {'delay':>6}")
        for _, r in pooled.iterrows():
            budget = "" if not np.isfinite(r["target_false_alarm"]) else f"{r['target_false_alarm'] * 100:6.1f}%"
            print(f"  {r['comparison']:<19} {r['detector']:<19} {r['split']:<6} {budget:>7} "
                  f"{r['detection_rate'] * 100:8.1f}% {r['false_alarm_rate'] * 100:6.1f}% "
                  f"{r['false_alarm_clean'] * 100:5.1f}% {r['false_alarm_onset0'] * 100:6.1f}% "
                  f"{r['median_delay']:6.1f}")
        print(f"\nWrote {args.output_dir / 'changepoint_freeze_point.csv'}")
    (args.output_dir / "changepoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
