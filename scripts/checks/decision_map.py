"""A practitioner's decision map over the end-of-study procedures, and the freeze rule without its fallback.

Post hoc, like gated_sitting.py: the map is assembled from procedures the paper
already reports (and from the post hoc F1 and gated sittings), after their
per-cell results were known. No test here belongs to a Holm family of the paper.

Procedures. LogEI and qNEI, the four response processes, the main sweep's seeds
7-16. Every procedure is replayed exactly from the run logs or re-scored from
them; nothing is simulated here.

    standard      ship the best observed rating (the reference: gain 0, price 0)
    PM, LCB1, LCB2  the zero-trial ship rules after all 50 trials (ship_rules_per_run.csv)
    sit<k>        a final sitting of k, k in 2, 3, 5, 8, 12, 16, 20, 25, 30 (end_of_study_ksweep and
                  _kwide: candidates by LCB1, each shown once, the best look ships, rho 1, the shared sitting)
    freeze        the CUSUM detect-and-freeze rule as published (replay_stopping.py rule A, tuned on seeds
                  7-11: h 2000, kappa 16, w 3): at an alarm stop and ship LCB1 refitted on trials 1..t-w;
                  without an alarm ship LCB1 after all 50 trials
    F1            the same detector at the same operating point, freezing only on an alarm: without an alarm
                  ship the best observed rating, so a run without an alarm is the logged standard process
                  exactly. Its price is only that of false alarms on the clean twin.
    G16, G2       the gated sittings of gated_sitting.py, with the thresholds its rule chooses on seeds 7-11
    KV*-LCB1, KV*-LCB2  the ship rules of known_variance_ship.py, whose final refit is given the rater's noise
                  variance, at each level of knowledge it scores (only if review/known_variance_ship_per_run.csv.gz
                  covers every run and reproduces the standard process):
                    KV    the magnitude and the onset (exact before the onset, sigma_e^2 after): needs the cell
                    KVm   the magnitude (sigma_e^2 on every trial): needs the magnitude
                    KVc   the magnitude, the onset placed at the freeze rule's CUSUM alarm: needs the magnitude
                    KVr   three test-retest pairs rated after the study (six ratings outside the budget)
                    KVrc  three test-retest pairs, the onset at the CUSUM alarm (six ratings outside the budget)
                    KVi   trials 48-50 re-rate the top three by LCB1 at trial 47 (inside the budget)
                  A policy may use a level only if it has that knowledge: M1 (the cell) any level, M2 (the
                  magnitude) any but KV, a single procedure (nothing) only KVr, KVrc and KVi. They enter only
                  the fourth candidate set below, so the first three, and the appendix table, are unchanged.

Self-gating combinations, used only in the last sensitivity (post hoc, every threshold and the detector
inherited, no new choice): G16+G2 (G16 at trial 34; if it does not fire, G2 at trial 48), F1+G2 (an alarm
at t <= 48 freezes; else G2 at 48; else an alarm at 49 or 50 freezes; else the best rating) and
F1+G16+G2 (an alarm at t <= 34 freezes; else G16; else F1+G2). Each is exact: until a procedure acts, the
run is the logged one, and the alarm times before a gate fires are the logged run's.

Estimand (analyse_boba_adaptations.py): per run in units of opt_z, cost = ref_noisy - ref_clean,
gain = ref_noisy - trt_noisy, price = trt_clean - ref_clean (the clean twin follows the procedure, so a
false alarm or a gate that fires without error is price); landscape means; landscape bootstrap; two-sided
Wilcoxon over the twenty landscapes.

Maps, every choice made on seeds 7-11 and scored on seeds 12-16:

    M1  cell known: in each (magnitude, onset) cell, the procedure with the largest landscape-mean gain on
        seeds 7-11 (ties to the earlier procedure in the order above, the standard rule first). Without
        error the map knows it and ships by the standard rule (gain 0, price 0).
    M2  onset unknown: per magnitude, the largest mean gain over the two onsets on seeds 7-11.
    M3  magnitude misjudged by one grid step: M1's choice for the neighbouring magnitude, scored in the
        true cell.
    best single procedure: the procedure with the largest V on seeds 7-11, where V = (sum over the eight
        cells of the landscape-mean gain - the price) / 9 averages the eight cells and the error-free case
        (gain = -price) with equal weight; the map's V counts the error-free case as 0.

The value of the map over the best single procedure is the paired per-landscape difference of V, with a
landscape bootstrap and the count of landscapes ahead. V weighs the cells equally on the scale of the
achievable improvement, so the 5 sigma, trial-21 cell (where the gains are largest) dominates it; V is
also given without that cell. Four candidate sets: the published procedures only (standard, PM, LCB1,
LCB2, the nine sittings, freeze), the main set (those plus F1, G16 and G2), the main set plus the self-gating
combinations, and, if their per-run file is complete, the main set plus the known-variance rules at the
levels each policy may use ('main+variance'; no M3 there, since a misjudged magnitude would also misstate the
variance the rule is given, and no run scores that).

The map table (decision_map_table.csv, one row per cell plus the error-free row: the chosen procedure,
its held-out gain with interval, share of the cost of error where the cost is positive on every
landscape and at least 0.01, Wilcoxon p with Holm over the cells where the map does not ship by the
standard rule, and the chosen procedure's price, which the map pays only if the cell is misjudged) is
what the paper's appendix table prints; this script writes it as paper/tables/decision_map.tex.

F1 against the published freeze rule: per cell held out, the price and its decomposition (the published
rule's price splits into the LCB1 it ships at T on clean twins without an alarm and the false alarms;
F1 pays only the second), alarm rates, and the share of the cost recovered per error process at 5 sigma
from trial 21.

Inputs: the replays and the ship-rule rescoring above (tracked), the stopping replay's per-run file
output-boba/analysis/stopping/stopping_per_run.csv (git-ignored; replay_stopping.py writes it) and
gated_sitting.py's refit cache review/gated_sitting_noise.csv.gz. Writes review/decision_map_*.csv.

    python scripts/checks/decision_map.py > output-boba/analysis/review/register_checks/decision_map.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import gated_sitting as gs  # noqa: E402
import sitting_by_magnitude as sbm  # noqa: E402

K_GRID = sbm.K_GRID
TRAIN_SEEDS, TEST_SEEDS = sbm.TRAIN_SEEDS, sbm.TEST_SEEDS
SIGMAS = (0.05, 0.25, 1.0, 5.0)
ONSETS = (0, 20)
CELLS = [(s, o) for s in SIGMAS for o in ONSETS]
STOPPING = gs.MAIN_ANALYSIS / "stopping" / "stopping_per_run.csv"
REVIEW = gs.REVIEW
MAP_SEED = 20261010
REPS = sbm.REPS
PUBLISHED = ("standard", "PM", "LCB1", "LCB2", *(f"sit{k}" for k in K_GRID), "freeze")
POST_HOC = ("F1", "G16", "G2")
COMBINATIONS = ("G16+G2", "F1+G2", "F1+G16+G2")
ATOL = 1e-9


def cell_label(sigma: float, onset: int) -> str:
    return gs.cell_label(sigma, onset)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def attach_stopping(frame: pd.DataFrame, stopping: pd.DataFrame) -> pd.DataFrame:
    """Rule A's alarm, alarm trial and shipped regret (opt_z units) for each noisy run and its clean twin.

    Rule A's standard-process regret must reproduce the replays' ref_noisy / ref_clean, and without an
    alarm rule A ships LCB1 after all 50 trials, which must reproduce the rescored LCB1 (both to 1e-9 in
    raw units), or the join stops.
    """
    a = stopping[stopping["rule"] == "A"].copy()
    a["stopped"] = a["stopped"].astype(str).str.lower().isin(("true", "1"))
    a["stop_t"] = pd.to_numeric(a["stop_t"], errors="coerce")
    noisy = a[a["error_model"] != "none"].set_index("file")
    clean = a[a["error_model"] == "none"].set_index(["dataset", "acquisition", "seed"])
    out = frame.copy()
    key = pd.MultiIndex.from_frame(out[["dataset", "acquisition", "seed"]])
    for col in ("stopped", "stop_t", "regret", "regret_standard_T"):
        out[f"A_{col}"] = out["file"].map(noisy[col])
        out[f"A_{col}_clean"] = clean[col].reindex(key).to_numpy()
    missing = out["A_regret"].isna() | out["A_regret_clean"].isna()
    if missing.any():
        raise KeyError(f"{int(missing.sum())} runs have no rule-A replay (e.g. {out.loc[missing, 'file'].iloc[0]})")
    z = out["opt_z"].to_numpy()
    for sfx, ref in (("", "ref_noisy"), ("_clean", "ref_clean")):
        gap = np.abs(out[f"A_regret_standard_T{sfx}"].to_numpy() - out[ref].to_numpy() * z)
        if gap.max() > ATOL * max(1.0, float(np.abs(z).max())):
            raise ValueError(f"rule A's standard regret{sfx} differs from the replays' by {gap.max():g}")
        lcb1 = out[f"regret_lcb1{sfx}"].to_numpy() * z
        no_alarm = ~out[f"A_stopped{sfx}"].astype(bool).to_numpy()
        gap = np.abs(out[f"A_regret{sfx}"].to_numpy()[no_alarm] - lcb1[no_alarm])
        if len(gap) and gap.max() > ATOL * max(1.0, float(np.abs(z).max())):
            raise ValueError(f"rule A without an alarm{sfx} does not ship the rescored LCB1 (gap {gap.max():g})")
    out["freeze"] = out["A_regret"].to_numpy() / z
    out["freeze_clean"] = out["A_regret_clean"].to_numpy() / z
    out["alarm"] = out["A_stopped"].astype(bool).to_numpy()
    out["alarm_clean"] = out["A_stopped_clean"].astype(bool).to_numpy()
    out["alarm_t"] = out["A_stop_t"].fillna(np.inf).to_numpy(dtype=float)
    out["alarm_t_clean"] = out["A_stop_t_clean"].fillna(np.inf).to_numpy(dtype=float)
    return out


KNOWN_FILE = "known_variance_ship_per_run.csv.gz"
# The known-variance rules (known_variance_ship.py: the rater's noise variance given to the final refit), by
# level: name -> (per-run column, what a policy must know to use it). 'cell' is the magnitude and the onset,
# 'magnitude' the magnitude alone, 'none' nothing (the variance is estimated from the run's own repeats).
# The 'mag_onset' pair is required for the file to be used; the other levels join when their columns exist.
KNOWN_LEVELS = {}
for _tag, _level, _needs in (("KV", "mag_onset", "cell"), ("KVm", "mag", "magnitude"),
                             ("KVc", "cusum", "magnitude"), ("KVr", "retest3", "none"),
                             ("KVrc", "retest3_cusum", "none"), ("KVi", "inside_top", "none")):
    for _rule in ("lcb1", "lcb2"):
        KNOWN_LEVELS[f"{_tag}-{_rule.upper()}"] = (f"regret_{_level}_{_rule}", _needs)
KNOWN_COLUMNS = {name: col for name, (col, needs) in KNOWN_LEVELS.items() if needs == "cell"}
KNOWLEDGE = {"M1": {"cell", "magnitude", "none"}, "M2": {"magnitude", "none"}, "single": {"none"}}


def needs(name: str) -> str:
    """What a policy must know to use procedure ``name``: 'cell', 'magnitude' or 'none'."""
    return KNOWN_LEVELS[name][1] if name in KNOWN_LEVELS else "none"


def allowed(cands, policy: str) -> list:
    """The candidates a policy ('M1', 'M2' or 'single') may use, in order."""
    return [c for c in cands if needs(c) in KNOWLEDGE[policy]]


def attach_known_variance(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """The known-variance ship rules' regret per noisy run and clean twin, if their per-run file covers the runs.

    Reads review/known_variance_ship_per_run.csv.gz (known_variance_ship.py), output-boba rows. Each rule
    must reproduce the standard process (its regret_best_observed equals ref_noisy / ref_clean to 1e-9)
    and cover every noisy run and clean twin of ``frame``; otherwise it is left out with a note. Returns
    the frame with KV-* columns and the file name, or the frame unchanged and None.
    """
    path = REVIEW / KNOWN_FILE
    if not path.is_file():
        return frame, None
    kv = pd.read_csv(path, low_memory=False)
    if "arm" in kv:
        kv = kv[kv["arm"] == gs.MAIN_ARM]
    missing_cols = [c for c in ["file", "baseline", "regret_best_observed", *KNOWN_COLUMNS.values()] if c not in kv]
    if missing_cols:
        print(f"[known-variance] {path.name} lacks {missing_cols}; not used")
        return frame, None
    kv = kv.assign(fname=kv["file"].map(lambda f: Path(str(f)).name),
                   baseline=kv["baseline"].astype(str).str.lower().isin(("true", "1")))
    noisy = kv[~kv["baseline"]].drop_duplicates("fname").set_index("fname")
    clean = kv[kv["baseline"]].drop_duplicates(["dataset", "acquisition", "seed"]).set_index(
        ["dataset", "acquisition", "seed"])
    out = frame.copy()
    z = out["opt_z"].to_numpy()
    key = pd.MultiIndex.from_frame(out[["dataset", "acquisition", "seed"]])
    std_n = out["file"].map(noisy["regret_best_observed"]).to_numpy(dtype=float)
    std_c = clean["regret_best_observed"].reindex(key).to_numpy(dtype=float)
    covered = np.isfinite(std_n) & np.isfinite(std_c)
    if not covered.all():
        print(f"[known-variance] {path.name} covers {int(covered.sum())} of {len(out)} noisy runs and their clean "
              f"twins; not used")
        return frame, None
    for sfx, std, ref in (("", std_n, "ref_noisy"), ("_clean", std_c, "ref_clean")):
        gap = float(np.abs(std - out[ref].to_numpy() * z).max())
        if gap > ATOL * max(1.0, float(np.abs(z).max())):
            raise ValueError(f"{path.name}: the standard process{sfx} differs from the replays' by {gap:g}")
    for name, (col, _) in KNOWN_LEVELS.items():
        if col not in kv:
            continue
        out[name] = out["file"].map(noisy[col]).to_numpy(dtype=float) / z
        out[f"{name}_clean"] = clean[col].reindex(key).to_numpy(dtype=float) / z
        if out[[name, f"{name}_clean"]].isna().any().any():
            if name in KNOWN_COLUMNS:
                print(f"[known-variance] {path.name}: {col} is missing for some runs; not used")
                return frame, None
            print(f"[known-variance] {path.name}: {col} is missing for some runs; {name} left out")
            out = out.drop(columns=[name, f"{name}_clean"])
    return out, path.name


def known_names(frame: pd.DataFrame) -> list:
    """The known-variance procedures attached to ``frame``, in KNOWN_LEVELS order."""
    return [n for n in KNOWN_LEVELS if n in frame and f"{n}_clean" in frame]


def load(noise_path: Path = gs.NOISE_CACHE, stopping_path: Path = STOPPING) -> tuple[pd.DataFrame, dict, str | None]:
    """The main sweep's per-run frame with every procedure's inputs, and the gate thresholds."""
    noise = gs.read_noise(noise_path)
    frame = gs.run_frame(gs.MAIN_ARM, noise, ks=K_GRID)
    cols = ["dataset", "acquisition", "seed", "error_model", "file", "rule", "stopped", "stop_t", "regret",
            "regret_standard_T"]
    frame = attach_stopping(frame, pd.read_csv(stopping_path, usecols=cols, low_memory=False))
    frame, known = attach_known_variance(frame)
    train = frame[frame["seed"].isin(TRAIN_SEEDS)].reset_index(drop=True)
    taus = {gate: gs.choose_threshold(train, k, gs.TAU_GRID, "noise_sd")[0] for gate, k in gs.GATES.items()}
    return frame, taus, known


# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------


def procedures(frame: pd.DataFrame, taus: dict, known: bool = False, combinations: bool = False) -> dict:
    """name -> (deployed regret of the noisy run, of its clean twin), opt_z units, aligned with frame rows."""
    ref = (frame["ref_noisy"].to_numpy(), frame["ref_clean"].to_numpy())
    P = {"standard": ref}
    for rule in ("pm", "lcb1", "lcb2"):
        P[rule.upper()] = (frame[f"regret_{rule}"].to_numpy(), frame[f"regret_{rule}_clean"].to_numpy())
    for k in K_GRID:
        if f"sit{k}" in frame:
            P[f"sit{k}"] = (frame[f"sit{k}"].to_numpy(), frame[f"sit{k}_clean"].to_numpy())
    fr = (frame["freeze"].to_numpy(), frame["freeze_clean"].to_numpy())
    alarm = (frame["alarm"].to_numpy(dtype=bool), frame["alarm_clean"].to_numpy(dtype=bool))
    at = (frame["alarm_t"].to_numpy(dtype=float), frame["alarm_t_clean"].to_numpy(dtype=float))
    P["freeze"] = fr
    P["F1"] = tuple(np.where(alarm[i], fr[i], ref[i]) for i in (0, 1))
    for gate, k in gs.GATES.items():
        P[gate] = gs.gated(frame, k, taus[gate])
    if known:
        for name in known_names(frame):
            P[name] = (frame[name].to_numpy(), frame[f"{name}_clean"].to_numpy())
    if combinations:
        P["G16+G2"] = gs.two_stage(frame, taus["G16"], taus["G2"])
        f16 = gs.gate_fires(frame, 16, taus["G16"])
        f2 = gs.gate_fires(frame, 2, taus["G2"])
        s16 = (frame["sit16"].to_numpy(), frame["sit16_clean"].to_numpy())
        s2 = (frame["sit2"].to_numpy(), frame["sit2_clean"].to_numpy())
        P["F1+G2"] = tuple(np.where(alarm[i] & (at[i] <= 48), fr[i],
                                    np.where(f2[i], s2[i], np.where(alarm[i], fr[i], ref[i]))) for i in (0, 1))
        P["F1+G16+G2"] = tuple(np.where(alarm[i] & (at[i] <= 34), fr[i],
                                        np.where(f16[i], s16[i], P["F1+G2"][i])) for i in (0, 1))
    return P


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


class Scores:
    """Per-landscape cell gains, costs and prices of every procedure on one seed set."""

    def __init__(self, frame: pd.DataFrame, P: dict, seeds):
        at = frame["seed"].isin(seeds).to_numpy()
        self.frame = frame[at].reset_index(drop=True)
        f = self.frame
        self.lands = sorted(f["dataset"].unique())
        ref_n, ref_c = f["ref_noisy"].to_numpy(), f["ref_clean"].to_numpy()
        self.gain, self.price = {}, {}
        groups = [f["sigma"], f["onset"], f["dataset"]]
        stems = f.drop_duplicates(["dataset", "acquisition", "seed"]).index
        for name, (tn, tc) in P.items():
            g = pd.Series(ref_n - np.asarray(tn)[at]).groupby(groups).mean()
            self.gain[name] = g.unstack("dataset").reindex(columns=self.lands)       # (sigma, onset) x landscape
            pr = pd.Series(np.asarray(tc)[at] - ref_c)
            self.price[name] = pr.loc[stems].groupby(f.loc[stems, "dataset"].to_numpy()).mean().reindex(self.lands)
        self.cost = pd.Series(ref_n - ref_c).groupby(groups).mean().unstack("dataset").reindex(columns=self.lands)
        self.P = P

    def cell(self, name: str, sigma: float, onset: int) -> np.ndarray:
        return self.gain[name].loc[(sigma, onset)].to_numpy()

    def cell_mean(self, name: str, sigma: float, onset: int) -> float:
        return float(self.cell(name, sigma, onset).mean())

    def value_l(self, name: str, cells=CELLS, pay_price: bool = True) -> np.ndarray:
        """Per-landscape V: (sum of the cells' gains - the price) / (cells + 1)."""
        tot = sum(self.cell(name, s, o) for s, o in cells)
        price = self.price[name].to_numpy() if pay_price else 0.0
        return (tot - price) / (len(cells) + 1)

    def map_value_l(self, choice: dict, cells=CELLS) -> np.ndarray:
        """A map's per-landscape V: each cell scored with its own choice; no price (no error, standard rule)."""
        return sum(self.cell(choice[c], *c) for c in cells) / (len(cells) + 1)


def choose_map(train: Scores, candidates) -> dict:
    """M1: per cell, the largest training gain; ties to the earlier candidate (standard first)."""
    out = {}
    for c in CELLS:
        best, val = None, None
        for name in candidates:
            v = train.cell_mean(name, *c)
            if val is None or v > val + 1e-15:
                best, val = name, v
        out[c] = best
    return out


def choose_map_onset_unknown(train: Scores, candidates) -> dict:
    out = {}
    for s in SIGMAS:
        best, val = None, None
        for name in candidates:
            v = np.mean([train.cell_mean(name, s, o) for o in ONSETS])
            if val is None or v > val + 1e-15:
                best, val = name, v
        for o in ONSETS:
            out[(s, o)] = best
    return out


def choose_single(train: Scores, candidates) -> str:
    best, val = None, None
    for name in candidates:
        v = float(train.value_l(name).mean())
        if val is None or v > val + 1e-15:
            best, val = name, v
    return best


def diff_ci(a: np.ndarray, b: np.ndarray, rng) -> dict:
    d = np.asarray(a) - np.asarray(b)
    lo, hi = sbm.boot_ci(d, rng)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "ahead": int((d > 0).sum()), "behind": int((d < 0).sum()),
            "p": sbm.wilcoxon_p(d)}


def map_table(train: Scores, test: Scores, choice: dict, rng) -> pd.DataFrame:
    """The map's cells held out: choice, gain [CI], share of the cost [CI], p (Holm over the non-standard
    cells), the chosen procedure's price; plus the error-free row."""
    rows = []
    for c in CELLS:
        name = choice[c]
        g = test.cell(name, *c)
        cost = test.cost.loc[c].to_numpy()
        lo, hi = sbm.boot_ci(g, rng)
        slo, shi = sbm.ratio_ci(g, cost, rng)
        why = sbm.share_suppressed(cost)
        runner = sorted(((train.cell_mean(n, *c), n) for n in train.P if n != name), reverse=True)[0]
        best_test = max(train.P, key=lambda n: test.cell_mean(n, *c))
        rows.append({
            "cell": cell_label(*c), "sigma": c[0], "onset": c[1], "procedure": name,
            "train_gain": train.cell_mean(name, *c), "runner_up_train": runner[1], "runner_up_train_gain": runner[0],
            "gain": float(g.mean()), "gain_lo": lo, "gain_hi": hi,
            "landscapes_gaining": int((g > 0).sum()), "p": sbm.wilcoxon_p(g) if name != "standard" else np.nan,
            "cost": float(cost.mean()), "share": np.nan if why else float(g.mean() / cost.mean()),
            "share_lo": np.nan if why else slo, "share_hi": np.nan if why else shi, "share_suppressed": why,
            "price": float(test.price[name].mean()), "best_on_test": best_test,
            "best_on_test_gain": test.cell_mean(best_test, *c)})
    t = pd.DataFrame(rows)
    acts = t["procedure"] != "standard"
    t["p_holm_map_cells"] = np.nan
    t.loc[acts, "p_holm_map_cells"] = sbm.holm(t.loc[acts, "p"].to_numpy())
    t = pd.concat([t, pd.DataFrame([{"cell": "no error", "sigma": 0.0, "onset": np.nan, "procedure": "standard",
                                     "gain": 0.0, "gain_lo": 0.0, "gain_hi": 0.0, "price": 0.0, "cost": 0.0,
                                     "share_suppressed": "no cost without error"}])], ignore_index=True)
    return t


def policy_rows(train: Scores, test: Scores, maps: dict, single_train: str, rng, label: str) -> pd.DataFrame:
    rows = []
    without = [c for c in CELLS if c != (5.0, 20)]
    for name in train.P:
        v = test.value_l(name)
        lo, hi = sbm.boot_ci(v, rng)
        rows.append({"candidates": label, "policy": f"always {name}", "single_procedure": needs(name) == "none",
                     "needs": needs(name),
                     "value_test": float(v.mean()), "lo": lo,
                     "hi": hi, "value_train": float(train.value_l(name).mean()),
                     "value_test_without_5s21": float(test.value_l(name, without).mean()),
                     "price_test": float(test.price[name].mean()),
                     "worst_cell_test": min(test.cell_mean(name, *c) for c in CELLS)})
    for mname, choice in maps.items():
        v = test.map_value_l(choice)
        lo, hi = sbm.boot_ci(v, rng)
        rows.append({"candidates": label, "policy": mname, "single_procedure": False,
                     "needs": "cell" if mname.startswith("M1") else "magnitude",
                     "value_test": float(v.mean()), "lo": lo, "hi": hi,
                     "value_train": float(train.map_value_l(choice).mean()),
                     "value_test_without_5s21": float(test.map_value_l(choice, without).mean()), "price_test": 0.0,
                     "worst_cell_test": min(test.cell_mean(choice[c], *c) for c in CELLS)})
    out = pd.DataFrame(rows).sort_values("value_test", ascending=False).reset_index(drop=True)
    out["chosen_single_on_7_11"] = out["policy"] == f"always {single_train}"
    return out


def misjudged(test: Scores, m1: dict) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(SIGMAS):
        for o in ONSETS:
            for step in (-1, 1):
                j = i + step
                if 0 <= j < len(SIGMAS):
                    name = m1[(SIGMAS[j], o)]
                    rows.append({"true_cell": cell_label(s, o), "assumed_sigma": SIGMAS[j], "procedure": name,
                                 "gain": test.cell_mean(name, s, o), "map_gain": test.cell_mean(m1[(s, o)], s, o)})
            rows.append({"true_cell": cell_label(s, o), "assumed_sigma": 0.0, "procedure": "standard", "gain": 0.0,
                         "map_gain": test.cell_mean(m1[(s, o)], s, o)})
    return pd.DataFrame(rows)


def f1_block(frame: pd.DataFrame, P: dict, rng) -> tuple[pd.DataFrame, list[str]]:
    """F1 against the published freeze rule on seeds 12-16: cells, price decomposition, alarms, processes."""
    te = frame["seed"].isin(TEST_SEEDS).to_numpy()
    f = frame[te].reset_index(drop=True)
    lines, rows = [], []
    stems = f.drop_duplicates(["dataset", "acquisition", "seed"])
    for name in ("freeze", "F1", "LCB1"):
        tn, tc = (np.asarray(x)[te] for x in P[name])
        s = pd.DataFrame({"dataset": f["dataset"], "sigma": f["sigma"], "onset": f["onset"],
                          "error_model": f["error_model"], "gain": f["ref_noisy"].to_numpy() - tn,
                          "cost": (f["ref_noisy"] - f["ref_clean"]).to_numpy(), "price": tc - f["ref_clean"].to_numpy()})
        price_l = s.loc[stems.index].groupby("dataset")["price"].mean()
        plo, phi = sbm.boot_ci(price_l.to_numpy(), rng)
        for (sig, ons), sub in s.groupby(["sigma", "onset"]):
            per = sub.groupby("dataset")[["gain", "cost"]].mean()
            g, c = per["gain"].to_numpy(), per["cost"].to_numpy()
            lo, hi = sbm.boot_ci(g, rng)
            why = sbm.share_suppressed(c)
            row = {"procedure": name, "cell": cell_label(sig, ons), "gain": float(g.mean()), "gain_lo": lo,
                   "gain_hi": hi, "p": sbm.wilcoxon_p(g), "share": np.nan if why else float(g.mean() / c.mean()),
                   "price": float(price_l.mean()), "price_lo": plo, "price_hi": phi,
                   "alarm_rate": float(f.loc[(f["sigma"] == sig) & (f["onset"] == ons), "alarm"].mean())}
            for pr, sp in sub.groupby("error_model"):
                pp = sp.groupby("dataset")[["gain", "cost"]].mean()
                row[f"share_{pr}"] = float(pp["gain"].mean() / pp["cost"].mean()) if not sbm.share_suppressed(
                    pp["cost"].to_numpy()) else np.nan
                row[f"gain_{pr}"] = float(pp["gain"].mean())
            rows.append(row)
    t = pd.DataFrame(rows)
    # The published rule's price, split: LCB1 shipped at T without an alarm, and the false alarms.
    z_ref = stems["ref_clean"].to_numpy()
    alarm_c = stems["alarm_clean"].to_numpy(dtype=bool)
    part_lcb1 = np.where(~alarm_c, stems["regret_lcb1_clean"].to_numpy() - z_ref, 0.0)
    part_alarm = np.where(alarm_c, stems["freeze_clean"].to_numpy() - z_ref, 0.0)
    by = stems["dataset"].to_numpy()
    p1 = pd.Series(part_lcb1).groupby(by).mean().mean()
    p2 = pd.Series(part_alarm).groupby(by).mean().mean()
    lines.append(f"   the published freeze rule's price on seeds 12-16, {p1 + p2:.4f} of opt_z, splits into {p1:.4f} from "
                 f"the LCB1 it ships at T on clean twins without an alarm and {p2:.4f} from false alarms (stopping "
                 f"and shipping the refitted LCB1); F1 pays only the second")
    lines.append(f"   alarms on clean twins, seeds 12-16: {100 * alarm_c.mean():.1f}% "
                 f"({int(alarm_c.sum())} of {len(alarm_c)}); on noisy runs with error from trial 1: "
                 f"{100 * f.loc[f['onset'] == 0, 'alarm'].mean():.1f}%, from trial 21: "
                 f"{100 * f.loc[f['onset'] == 20, 'alarm'].mean():.1f}% (by magnitude: "
                 + ", ".join(f"{s:g} sigma {100 * f.loc[f['sigma'] == s, 'alarm'].mean():.0f}%" for s in SIGMAS) + ")")
    return t, lines


def _fp(p) -> str:
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.2g}"


TABLE = REPO / "paper" / "tables" / "decision_map.tex"
_WORDS = {2: "two", 3: "three", 5: "five", 8: "eight", 12: "twelve", 16: "sixteen", 20: "twenty", 25: "twenty-five",
          30: "thirty"}


def display_name(name: str) -> str:
    """A procedure's name in the paper's words."""
    if name.startswith("sit") and name[3:].isdigit():
        return f"sitting of {_WORDS[int(name[3:])]}"
    if name.startswith("G") and name[1:].isdigit():
        return f"gated, $k = {int(name[1:])}$"
    return {"standard": "standard rule", "freeze": "freeze rule", "F1": "freeze on an alarm"}.get(name, name)


def _cell_words(cell: str) -> str:
    sigma, onset = cell.split("sigma_from_trial_")
    return f"${sigma}\\sigma$, trial {onset}"


def latex_table(t: pd.DataFrame) -> str:
    """paper/tables/decision_map.tex from the 'main' map table: per cell the procedure chosen on seeds 7-11, its gain
    on seeds 12-16 with interval, share of the cost, Holm p over the cells where the map acts, its price, and the
    runner-up on seeds 7-11. The paper's tab:decisionmap inputs it; this script is its producer."""
    def num(v, d=3):
        s = f"{v:+.{d}f}"
        return "0" if float(s) == 0 else s
    lines = [r"\begin{tabular}{llrrrrl}", r"\toprule",
             r"cell & chosen on 7--11 & gain on 12--16 & share & Holm $p$ & price & runner-up on 7--11 \\",
             r"\midrule"]
    for _, x in t.iterrows():
        if x["cell"] == "no error":
            lines.append(r"no error & standard rule & $0$ & -- & -- & $0$ & \\")
            continue
        acts = x["procedure"] != "standard"
        gain = (f"${num(x['gain'])}$ $[{num(x['gain_lo'])}, {num(x['gain_hi'])}]$" if acts else "$0$")
        share = (f"${100 * x['share']:.0f}\\%$ $[{100 * x['share_lo']:.0f}, {100 * x['share_hi']:.0f}]$"
                 if acts and np.isfinite(x["share"]) else "--")
        p = x["p_holm_map_cells"]
        holm = "--" if not acts or not np.isfinite(p) else ("$<0.001$" if p < 0.001 else f"${p:.2g}$")
        price = "$0$" if x["price"] == 0 else f"${x['price']:.3f}$"
        runner = f"{display_name(x['runner_up_train'])} (${num(x['runner_up_train_gain'], 4)}$)"
        lines.append(f"{_cell_words(x['cell'])} & {display_name(x['procedure'])} & {gain} & {share} & {holm} & "
                     f"{price} & {runner} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def analyse(frame: pd.DataFrame, taus: dict, known: str | None, rng) -> dict:
    """Every output, for the three candidate sets."""
    out = {}
    kv = known_names(frame) if known else []
    sets = {"published": list(PUBLISHED),
            "main": list(PUBLISHED) + list(POST_HOC),
            "main+combinations": list(PUBLISHED) + list(POST_HOC) + list(COMBINATIONS)}
    if kv:
        sets["main+variance"] = list(PUBLISHED) + list(POST_HOC) + kv
    P_all = procedures(frame, taus, known=bool(known), combinations=True)
    for label, cands in sets.items():
        P = {n: P_all[n] for n in cands}
        train, test = Scores(frame, P, TRAIN_SEEDS), Scores(frame, P, TEST_SEEDS)
        # Each policy uses only the levels of knowledge it has: M1 the cell, M2 the magnitude, a single
        # procedure nothing (KNOWLEDGE); outside 'main+variance' every candidate needs nothing.
        m1 = choose_map(train, allowed(cands, "M1"))
        m2 = choose_map_onset_unknown(train, allowed(cands, "M2"))
        singles = allowed(cands, "single")
        single = choose_single(train, singles)
        single_test = max(singles, key=lambda n: float(test.value_l(n).mean()))
        table = map_table(train, test, m1, rng)
        pol = policy_rows(train, test, {"M1 map (cell known)": m1, "M2 map (onset unknown)": m2}, single, rng, label)
        v_m1, v_m2 = test.map_value_l(m1), test.map_value_l(m2)
        without = [c for c in CELLS if c != (5.0, 20)]
        comps = {
            "M1 - best single (chosen on 7-11)": diff_ci(v_m1, test.value_l(single), rng),
            "M1 - best single on 12-16 (optimistic)": diff_ci(v_m1, test.value_l(single_test), rng),
            "M1 - M2 (knowing the onset)": diff_ci(v_m1, v_m2, rng),
            "M1 - best single, without the 5 sigma trial-21 cell": diff_ci(
                test.map_value_l(m1, without), test.value_l(single, without), rng),
        }
        out[label] = {"train": train, "test": test, "m1": m1, "m2": m2, "single": single,
                      "single_test": single_test, "table": table, "policies": pol, "comparisons": comps,
                      "misjudged": misjudged(test, m1)}
    return out


def gains_table(scores: Scores, seeds_label: str) -> pd.DataFrame:
    rows = []
    for name in scores.P:
        row = {"procedure": name, "seeds": seeds_label, "price": float(scores.price[name].mean()),
               "value": float(scores.value_l(name).mean())}
        row.update({cell_label(*c): scores.cell_mean(name, *c) for c in CELLS})
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise", default=str(gs.NOISE_CACHE))
    ap.add_argument("--stopping", default=str(STOPPING))
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    frame, taus, known = load(Path(args.noise), Path(args.stopping))
    rng = np.random.default_rng(MAP_SEED)
    res = analyse(frame, taus, known, rng)
    print("Decision map over the end-of-study procedures: LogEI and qNEI, four processes pooled, main sweep.")
    print("Choices on seeds 7-11, scores on seeds 12-16; gains over the standard process and prices in units of")
    print("the achievable improvement opt_z; post hoc and outside every Holm family of the paper.")
    print(f"runs: {len(frame)} noisy runs ({frame['seed'].isin(TRAIN_SEEDS).sum()} on seeds 7-11); gate thresholds "
          f"(gated_sitting.py's rule on seeds 7-11): " + ", ".join(f"{g} {t:g} landscape SDs" for g, t in taus.items()))
    print("known-variance ship rules (only in the candidate set 'main+variance', at the levels each policy may "
          "use): " + (f"{', '.join(known_names(frame))} from review/{known}" if known else
                      f"not included (review/{KNOWN_FILE} absent or incomplete when this ran)"))

    main_ = res.get("main+variance", res["main"])
    gains = pd.concat([gains_table(main_["train"], "7-11"), gains_table(main_["test"], "12-16")])
    combo = res["main+combinations"]
    gains = pd.concat([gains, gains_table(combo["train"], "7-11").query("procedure in @COMBINATIONS"),
                       gains_table(combo["test"], "12-16").query("procedure in @COMBINATIONS")])
    REVIEW.mkdir(parents=True, exist_ok=True)
    gains.to_csv(REVIEW / "decision_map_gains.csv", index=False)
    print("\n== held-out gain over the standard process per cell (seeds 12-16), price and V (eight cells and no error)")
    show = gains[gains["seeds"] == "12-16"].set_index("procedure")
    cols = [cell_label(*c) for c in CELLS]
    short = {c: c.replace("sigma_from_trial_", "s/t") for c in cols}
    print(show[cols + ["price", "value"]].rename(columns=short).round(4).to_string())

    tables, policies, comps_rows = [], [], []
    for label, r in res.items():
        t = r["table"].assign(candidates=label)
        tables.append(t)
        policies.append(r["policies"])
        print(f"\n== decision map, candidate set '{label}' ({len(r['train'].P)} procedures)")
        print("   cell                     chosen on 7-11    held-out gain [95% CI]        share of cost       p (Holm)"
              "      price   runner-up on 7-11; best on 12-16")
        for _, x in t.iterrows():
            if x["cell"] == "no error":
                print(f"   {'no error':24s} standard          gain 0, price 0 (the map ships by the standard rule)")
                continue
            share = ("n/a" if not np.isfinite(x["share"]) else
                     f"{100 * x['share']:.0f}% [{100 * x['share_lo']:.0f}, {100 * x['share_hi']:.0f}]")
            print(f"   {x['cell']:24s} {x['procedure']:16s} {x['gain']:+.4f} [{x['gain_lo']:+.4f}, {x['gain_hi']:+.4f}]"
                  f"  {share:18s} {_fp(x['p']):>7s} ({_fp(x['p_holm_map_cells'])})  {x['price']:.4f}   "
                  f"{x['runner_up_train']} ({x['runner_up_train_gain']:+.4f}); {x['best_on_test']} "
                  f"({x['best_on_test_gain']:+.4f})")
        print(f"   best single procedure chosen on seeds 7-11: {r['single']}; best on seeds 12-16: {r['single_test']}")
        pol = r["policies"]
        top = pol.head(8)
        print("   V on seeds 12-16 (eight cells and no error, equal weight) [95% CI], V on 7-11, V without the 5 sigma "
              "trial-21 cell, price, worst cell:")
        for _, x in top.iterrows():
            tag = {"cell": "  (needs the cell)", "magnitude": "  (needs the magnitude)"}.get(x["needs"], "")
            print(f"     {x['policy']:28s} {x['value_test']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]  train "
                  f"{x['value_train']:+.4f}  without 5s/t21 {x['value_test_without_5s21']:+.4f}  price "
                  f"{x['price_test']:.4f}  worst {x['worst_cell_test']:+.4f}"
                  + ("  <- single chosen on 7-11" if x["chosen_single_on_7_11"] else "")
                  + (tag if x["policy"].startswith("always") else ""))
        for name, c in r["comparisons"].items():
            print(f"   {name}: {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}], ahead on {c['ahead']}/20 landscapes, "
                  f"Wilcoxon p {_fp(c['p'])}")
            comps_rows.append({"candidates": label, "comparison": name, **c})
        if label == "main+variance":
            m2 = r["m2"]
            print("   M2 (the magnitude known, not the onset; no KV level) chooses: "
                  + ", ".join(f"{s:g} sigma -> {m2[(s, 0)]}" for s in SIGMAS))
            print("   a single procedure may use only the levels that estimate the variance from the run's own "
                  "repeats (KVr, KVrc: six ratings after the study; KVi: trials 48-50); no M3 in this set")
        if label == "main":
            m2 = r["m2"]
            print("   M2 (onset unknown) chooses: " + ", ".join(f"{s:g} sigma -> {m2[(s, 0)]}" for s in SIGMAS))
            mj = r["misjudged"]
            print("   M3 (magnitude misjudged by one step; assumed 0 = no error assumed), held-out gain in the true cell:")
            for cell, g in mj.groupby("true_cell", sort=False):
                print(f"     {cell:24s} map {g['map_gain'].iloc[0]:+.4f}; "
                      + "; ".join(f"assumed {a:g}: {p} {v:+.4f}" for a, p, v in
                                  zip(g["assumed_sigma"], g["procedure"], g["gain"])))
            mj.to_csv(REVIEW / "decision_map_misjudged.csv", index=False)
    pd.concat(tables).to_csv(REVIEW / "decision_map_table_all.csv", index=False)
    res["main"]["table"].to_csv(REVIEW / "decision_map_table.csv", index=False)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(latex_table(res["main"]["table"]), encoding="utf-8")
    pd.concat(policies).to_csv(REVIEW / "decision_map_policies.csv", index=False)
    pd.DataFrame(comps_rows).to_csv(REVIEW / "decision_map_comparisons.csv", index=False)

    P = procedures(frame, taus, known=bool(known), combinations=False)
    f1, lines = f1_block(frame, P, rng)
    f1.to_csv(REVIEW / "decision_map_f1.csv", index=False)
    print("\n== F1 (freeze only on an alarm, otherwise the standard rule) against the published freeze rule, seeds 12-16")
    for line in lines:
        print(line)
    for name in ("freeze", "F1", "LCB1"):
        b = f1[f1["procedure"] == name]
        print(f"   {name}: price {b['price'].iloc[0]:.4f} [{b['price_lo'].iloc[0]:.4f}, {b['price_hi'].iloc[0]:.4f}]")
        for _, x in b.iterrows():
            share = "n/a" if not np.isfinite(x["share"]) else f"{100 * x['share']:.0f}%"
            per = ", ".join(f"{pr} {100 * x[f'share_{pr}']:.0f}%" for pr in gs.PROCESSES
                            if np.isfinite(x.get(f"share_{pr}", np.nan)))
            print(f"     {x['cell']:24s} gain {x['gain']:+.4f} [{x['gain_lo']:+.4f}, {x['gain_hi']:+.4f}] p {_fp(x['p'])} "
                  f"share {share:5s} alarms {100 * x['alarm_rate']:.0f}%" + (f"  by process: {per}" if per else ""))
    print(f"\nwrote review/decision_map_table.csv (the appendix table; paper/tables/decision_map.tex), "
          f"decision_map_table_all.csv, decision_map_gains.csv, decision_map_policies.csv, decision_map_comparisons.csv, "
          f"decision_map_misjudged.csv, decision_map_f1.csv")


if __name__ == "__main__":
    main()
