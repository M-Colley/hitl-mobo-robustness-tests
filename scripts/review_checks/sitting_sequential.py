# Register checks code-estimands-6 and code-sim-repro-8 (review of 2026-09-28): the final
# sitting when the drift ramp and the AR(1) state move through it.
#   python scripts/review_checks/sitting_sequential.py > output-boba/analysis/review/register_checks/sitting_sequential.txt
"""The final sitting under the shared and the sequential rating process.

Every sitting number in the paper models the sitting as simultaneous: the drift
ramp and the AR(1) state are held fixed across its looks, so they cancel
(replay_end_of_study.py --sitting-process shared). Under the simulator's own
per-trial definitions they do not stay fixed over k trials. The sequential
replays (--sitting-process sequential) continue both through trials T-k+1..T,
the AR(1) state from the run's logged error at trial T-k, with the candidates
shown in a random order per run (and, as a sensitivity, best-ranked first).
Gaussian and bias are unchanged by construction, which this checks row by row.

Inputs (all read-only): the shared replays end_of_study_ksweep and
end_of_study_kwide, the sequential end_of_study_ksweep_seq and
end_of_study_kwide_seq, and the rank-order end_of_study_ksweep_seqrank and
end_of_study_kwide_seqrank (drift and ar1 only), all under output-boba/analysis;
for the confirmation claims end_of_study (shared) against end_of_study_ksweep_seq;
for code-estimands-10 end_of_study_slipsd{0,0.1,0.25,0.5}; for code-estimands-9
output-boba-spike/analysis/{hitl_remedies,end_of_study_spikesd,hitl_remedies_spikesd}
and their _heldout twins. The block "Every sitting value the paper prints" re-runs
the paper's own producers in-process on each replay set: sitting_by_magnitude.run
(tab:sittingcell and App. H; ship_rules_per_run.csv for the LCB1 reference) and
heldout_remedies' k-curve scoring (budget_split.score_policies with
budget_split_derived*.csv, the derived budget rule, the tab:heldout k row with its
landscape-split column, the optimism of the choice of k, and the Holm family of
App. H). Its shared side must reproduce
review/sitting_by_magnitude_selection.json, review/sitting_by_magnitude_gaussian_selection.json
and review/heldout_remedies.csv before any sequential value is printed.
The rows are those of sitting_by_magnitude.py and the k-curve: tournament,
candidates by the posterior mean less one latent SD, the best look ships, looks
at the full idiosyncratic SD (rho = 1), LogEI and qNEI, the four main processes.
Gains are (ref_noisy - regret_noisy) / opt_z: the deployed design's regret at the
final trial, standard process minus sitting, in units of the achievable
improvement. A gain is a mean over landscapes of landscape means, with a
landscape bootstrap (2,000 draws; the landscapes share their random numbers,
so the bootstrap treats correlated clusters as independent). A share of the cost
is a ratio of landscape means. Differences between processes are paired: the
same runs, the same fresh draws. The shared replays hold LogEI and qNEI (and
the input arms), the sequential ones also UCB; every comparison here is on the
runs both hold.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))
import sitting_by_magnitude as sbm  # noqa: E402

ANALYSIS = REPO / "output-boba" / "analysis"
STATS = REPO / "boba_landscape_stats.json"
PROCESSES = ("gaussian", "bias", "drift", "ar1")
MOVING = ("drift", "ar1")
DIRS = {
    "shared": ("end_of_study_ksweep", "end_of_study_kwide"),
    "sequential": ("end_of_study_ksweep_seq", "end_of_study_kwide_seq"),
    "sequential_rank": ("end_of_study_ksweep_seqrank", "end_of_study_kwide_seqrank"),
}
CELLS = {
    "1sigma_from_trial_1": (1.0, 0, (2, 3, 5, 8, 12, 16, 20, 25, 30)),
    "5sigma_from_trial_21": (5.0, 20, (8, 12, 16, 20)),
    # the other two cells the text discusses (App. H): sizes that pay there
    "1sigma_from_trial_21": (1.0, 20, (3, 5, 8, 12, 16)),
    "5sigma_from_trial_1": (5.0, 0, (5, 8, 12)),
}
KEY = ["file", "k"]
REPS = 2000
SEED = 20260928
LABELS = ("shared", "random", "rank")
# The same runs as DIRS, spelled as sitting_by_magnitude.load reads them (rank order:
# drift and ar1 from the rank-order replay, gaussian and bias from the random-order one,
# which the process leaves identical to the shared replay).
SPECS = {
    "shared": ("end_of_study_ksweep", "end_of_study_kwide"),
    "random": ("end_of_study_ksweep_seq", "end_of_study_kwide_seq"),
    "rank": ("end_of_study_ksweep_seq=gaussian+bias", "end_of_study_kwide_seq=gaussian+bias",
             "end_of_study_ksweep_seqrank=drift+ar1", "end_of_study_kwide_seqrank=drift+ar1"),
}
PAPER_CELLS = ("pooled", "0.05sigma_from_trial_1", "0.05sigma_from_trial_21", "0.25sigma_from_trial_1",
               "0.25sigma_from_trial_21", "1sigma_from_trial_1", "1sigma_from_trial_21", "5sigma_from_trial_1",
               "5sigma_from_trial_21")


def load(dirs: tuple[str, ...], rho: float = 1.0) -> pd.DataFrame:
    """sitting_by_magnitude.load, for any pair of replay directories and look-noise multiple."""
    cols = ["dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file", "family",
            "k", "candidates", "rho", "winner", "regret_noisy", "regret_clean", "ref_noisy", "ref_clean"]
    frames = [pd.read_csv(ANALYSIS / d / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
              for d in dirs]
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["family"] == "tournament") & (d["candidates"] == "lcb") & (d["winner"] == "look")
          & np.isclose(d["rho"].astype(float), rho) & d["acquisition"].isin(sbm.ACQS)
          & d["error_model"].isin(PROCESSES)]
    d = d.drop_duplicates(subset=KEY, keep="first")
    opt_z = {k: v["opt_z"] for k, v in json.loads(STATS.read_text())["functions"].items()}
    z = d["dataset"].map(opt_z)
    if z.isna().any():
        raise KeyError(f"no opt_z for {sorted(d.loc[z.isna(), 'dataset'].unique())}")
    d = d.assign(gain=(d["ref_noisy"] - d["regret_noisy"]) / z, cost=(d["ref_noisy"] - d["ref_clean"]) / z,
                 price=(d["regret_clean"] - d["ref_clean"]) / z, k=d["k"].astype(int))
    d["onset"] = d["jitter_iteration"].astype(int)
    d["sigma"] = d["jitter_std"].astype(float)
    return d.reset_index(drop=True)


def check_unchanged(shared: pd.DataFrame, seq: pd.DataFrame, label: str) -> pd.DataFrame:
    """Gaussian and bias rows must be identical, and every clean-side column too; returns the paired frame."""
    m = shared.merge(seq, on=KEY, suffixes=("", "_seq"), validate="one_to_one")
    if len(m) != len(seq):
        raise SystemExit(f"{label}: {len(seq) - len(m)} sequential rows have no shared twin")
    fixed = m[m["error_model"].isin(("gaussian", "bias"))]
    same_fixed = bool((fixed["regret_noisy"] == fixed["regret_noisy_seq"]).all())
    same_clean = bool(((m["ref_noisy"] == m["ref_noisy_seq"]) & (m["ref_clean"] == m["ref_clean_seq"])
                       & (m["regret_clean"] == m["regret_clean_seq"])).all())
    moving = m[m["error_model"].isin(MOVING)]
    changed = float((moving["regret_noisy"] != moving["regret_noisy_seq"]).mean()) if len(moving) else float("nan")
    print(f"{label}: {len(m):,} paired rows. gaussian and bias rows identical: {same_fixed}; "
          f"standard process and clean twin identical on every row: {same_clean}; "
          f"drift and ar1 rows whose shipped design changed: {changed:.1%}")
    if not (same_fixed and same_clean):
        raise SystemExit(f"{label}: a row that the sequential process must not touch differs")
    return m


def boot(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(v), (REPS, len(v)))
    draws = v[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_share(g: np.ndarray, c: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(g), (REPS, len(g)))
    draws = g[idx].mean(axis=1) / c[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fmt(v: float, lo: float, hi: float) -> str:
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def gain_line(block: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Shared, sequential and their paired difference, per landscape then over landscapes."""
    per = block.groupby("dataset")[["gain", "gain_seq", "cost"]].mean()
    diff = (per["gain_seq"] - per["gain"]).to_numpy()
    g, s, c = per["gain"].to_numpy(), per["gain_seq"].to_numpy(), per["cost"].to_numpy()
    return {"n_land": len(per), "n_runs": int(block["file"].nunique()),
            "shared": (g.mean(), *boot(g, rng)), "seq": (s.mean(), *boot(s, rng)),
            "diff": (diff.mean(), *boot(diff, rng)),
            "land_shared": int((g > 0).sum()), "land_seq": int((s > 0).sum()),
            "share_shared": (g.mean() / c.mean(), *boot_share(g, c, rng)),
            "share_seq": (s.mean() / c.mean(), *boot_share(s, c, rng)), "cost": c.mean()}


def print_cells(m: pd.DataFrame, label: str) -> None:
    rng = np.random.default_rng(SEED)
    for cell, (sigma, onset, ks) in CELLS.items():
        sub = m[np.isclose(m["sigma"], sigma) & (m["onset"] == onset)]
        for seeds_label, seeds in (("all seeds 7-16", None), ("held-out seeds 12-16", sbm.TEST_SEEDS)):
            s = sub if seeds is None else sub[sub["seed"].isin(seeds)]
            print(f"\n{cell}, {seeds_label} ({label}); gain in opt_z units [95% landscape bootstrap]; "
                  f"landscapes gaining out of 20")
            print(f"  {'k':>3s} {'process':10s} {'shared':>26s} {'sequential':>26s} {'seq - shared':>26s} "
                  f"{'lands sh/seq':>12s}")
            for k in ks:
                sk = s[s["k"] == k]
                for proc in PROCESSES + ("pooled",):
                    b = sk if proc == "pooled" else sk[sk["error_model"] == proc]
                    r = gain_line(b, rng)
                    print(f"  {k:3d} {proc:10s} {fmt(*r['shared']):>26s} {fmt(*r['seq']):>26s} "
                          f"{fmt(*r['diff']):>26s} {r['land_shared']:5d}/{r['land_seq']:<5d}")
                r = gain_line(sk, rng)
                print(f"  {k:3d} {'share of the cost of error, pooled':38s} shared "
                      f"{r['share_shared'][0]:+.1%} [{r['share_shared'][1]:+.0%}, {r['share_shared'][2]:+.0%}]"
                      f", sequential {r['share_seq'][0]:+.1%} [{r['share_seq'][1]:+.0%}, {r['share_seq'][2]:+.0%}]"
                      f" (cost {r['cost']:.3f})")


def print_pooled_curve(m: pd.DataFrame, label: str, rho_label: str) -> None:
    rng = np.random.default_rng(SEED)
    print(f"\nPooled over the four magnitudes, both onsets and the four processes, all seeds ({label}, {rho_label})")
    without = ~(np.isclose(m["sigma"], 5.0) & (m["onset"] == 20))
    for k in sorted(m["k"].unique()):
        r = gain_line(m[m["k"] == k], rng)
        w = gain_line(m[(m["k"] == k) & without], rng)
        print(f"  k={k:2d}  shared {fmt(*r['shared'])}  sequential {fmt(*r['seq'])}  diff {fmt(*r['diff'])}"
              f"   | without 5sigma from trial 21: shared {fmt(*w['shared'])}  sequential {fmt(*w['seq'])}")


def print_selection(frame: pd.DataFrame, label: str) -> dict[str, dict]:
    """sitting_by_magnitude's per-cell choice of k (seeds 7-11) and held-out score (12-16), re-made.

    The cells are analysed in sitting_by_magnitude's order from its bootstrap seed, so the
    shared frame reproduces tab:sittingcell interval for interval and the sequential frames
    get the same draws.
    """
    rng = np.random.default_rng(sbm.BOOTSTRAP_SEED)
    print(f"\nPer-cell choice of k re-made on seeds 7-11 and scored on 12-16 ({label}; "
          f"sitting_by_magnitude.analyse_cell)")
    cells = [("pooled", frame)] + [
        (f"{s:g}sigma_from_trial_{o + 1}", frame[np.isclose(frame["sigma"], s) & (frame["onset"] == o)])
        for s in (0.05, 0.25, 1.0, 5.0) for o in (0, 20)]
    out = {}
    for name, sub in cells:
        _, sm = sbm.analyse_cell(sub, name, rng)
        out[name] = sm
        pp = ", ".join(f"{p} {v:+.3f}" for p, v in sm["per_process_test"].items())
        print(f"  {name:24s} k={sm['k_chosen_on_7_11']:2d} held-out gain {fmt(sm['test_gain'], sm['test_lo'], sm['test_hi'])}"
              f" on {sm['test_landscapes_gaining']:2d}/20, Holm p over k {sm['test_p_holm_over_k']:.3f}, "
              f"share {sm['test_recovered_share']:+.1%}; per process {pp}; k=12 all seeds {sm['k12_gain_all']:+.3f}; "
              f"k chosen on 12-16 {sm['k_chosen_on_12_16']}")
    return out


def with_seq(shared: pd.DataFrame, seq: pd.DataFrame) -> pd.DataFrame:
    """The sequential gains as the frame sitting_by_magnitude analyses (every shared row must have a twin)."""
    m = shared.merge(seq[KEY + ["gain"]], on=KEY, suffixes=("_shared", ""), validate="one_to_one")
    if len(m) != len(shared):
        raise SystemExit(f"{len(shared) - len(m)} shared rows have no sequential twin")
    return m.drop(columns=["gain_shared"])


def largest_moves(m: pd.DataFrame, label: str) -> None:
    """The largest |sequential - shared| pooled gain over the cells and sizes listed in CELLS."""
    rows = []
    for cell, (sigma, onset, ks) in CELLS.items():
        sub = m[np.isclose(m["sigma"], sigma) & (m["onset"] == onset)]
        for seeds_label, seeds in (("all seeds", None), ("seeds 12-16", sbm.TEST_SEEDS)):
            s = sub if seeds is None else sub[sub["seed"].isin(seeds)]
            for k in ks:
                sk = s[s["k"] == k]
                for proc in ("pooled",) + MOVING:
                    b = sk if proc == "pooled" else sk[sk["error_model"] == proc]
                    pb = b.groupby("dataset")[["gain", "gain_seq"]].mean()
                    rows.append({"cell": cell, "seeds": seeds_label, "k": k, "process": proc,
                                 "shared": pb["gain"].mean(), "seq": pb["gain_seq"].mean(),
                                 "diff": pb["gain_seq"].mean() - pb["gain"].mean()})
    t = pd.DataFrame(rows)
    print(f"\nLargest move of a gain, sequential ({label}) minus shared, over the cells and sizes above:")
    for proc in ("pooled",) + MOVING:
        x = t[t["process"] == proc]
        r = x.loc[x["diff"].abs().idxmax()]
        sign_flips = int(((x["shared"] > 0) != (x["seq"] > 0)).sum())
        print(f"  {proc:7s} max |diff| {abs(r['diff']):.3f} ({r['cell']}, {r['seeds']}, k={r['k']}: "
              f"{r['shared']:+.3f} -> {r['seq']:+.3f}); sign changes {sign_flips} of {len(x)}")


def claims(dirs: dict[str, str]) -> None:
    """Confirmation's claims per process, by replay_end_of_study.claims_table on the same runs.

    end_of_study (the shared process, the paper's confirmation numbers) and the sequential
    k-sweep both hold LogEI, qNEI and UCB; they are re-tabulated on the (run, procedure) pairs
    both hold, and the shared side is checked against end_of_study_claims.csv.
    """
    import replay_end_of_study as eos  # noqa: E402 - heavy (torch), only needed here
    cols = ["arm", "dataset", "acquisition", "seed", "error_model", "jitter_std", "jitter_iteration", "file",
            "procedure", "family", "k", "claim_noisy", "truly_better_noisy", "claim_clean"]
    frames = {}
    for label, d in dirs.items():
        f = pd.read_csv(ANALYSIS / d / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
        frames[label] = f[(f["arm"] == "output-boba") & f["error_model"].isin(PROCESSES)
                          & f["family"].isin(("confirmation", "standard"))]
    common = set.intersection(*(set(zip(f["file"], f["procedure"])) for f in frames.values()))
    print(f"\nConfirmation claims per process, pooled over the eight cells, on the {len(common):,} "
          f"(run, procedure) pairs both replays hold ({', '.join(f'{k} = {v}' for k, v in dirs.items())}; "
          f"LogEI, qNEI and UCB; replay_end_of_study.claims_table):")
    tables = {}
    for label, f in frames.items():
        f = f[[pair in common for pair in zip(f["file"], f["procedure"])]]
        c = eos.claims_table(f)
        tables[label] = c[c["scope"] == "pooled"].set_index(["error_model", "procedure"])
    published = pd.read_csv(ANALYSIS / dirs["shared"] / "end_of_study_claims.csv")
    published = published[(published["arm"] == "output-boba") & (published["scope"] == "pooled")].set_index(
        ["error_model", "procedure"])
    for idx in sorted(tables["shared"].index):
        a, b = tables["shared"].loc[idx], tables["sequential"].loc[idx]
        same = (idx in published.index and np.isclose(published.loc[idx, "false_claim_rate"], a["false_claim_rate"],
                                                      rtol=0, atol=1e-12))
        print(f"  {idx[0]:8s} {idx[1]:11s} false-claim rate shared {a['false_claim_rate']:.2%} "
              f"[{a['false_claim_rate_lo']:.2%}, {a['false_claim_rate_hi']:.2%}] power {a['power']:.1%}"
              f" ({'= ' + dirs['shared'] + '_claims' if same else 'NOT the published value'});"
              f"  sequential {b['false_claim_rate']:.2%} [{b['false_claim_rate_lo']:.2%}, "
              f"{b['false_claim_rate_hi']:.2%}] power {b['power']:.1%}")


def check_against_published(m: pd.DataFrame, half: pd.DataFrame) -> int:
    """The shared side must reproduce the numbers the paper prints from these replays."""
    rng = np.random.default_rng(SEED)
    rows = []
    held = m[m["seed"].isin(sbm.TEST_SEEDS)]
    for (sigma, onset, k), paper in {(1.0, 0, 2): 0.030, (5.0, 20, 16): 0.227}.items():
        b = held[np.isclose(held["sigma"], sigma) & (held["onset"] == onset) & (held["k"] == k)]
        rows.append((f"{sigma:g}sigma onset {onset} k={k}, seeds 12-16 (tab:sittingcell)", gain_line(b, rng)["shared"][0],
                     paper, 5e-4))
    for rho, frame, fname in ((1.0, m, "budget_split_policies.csv"), (0.5, half, "budget_split_policies_rho0.5.csv")):
        pol = pd.read_csv(ANALYSIS / fname).set_index("policy")["gain_vs_standard"]
        for k in (5, 8, 12, 16):
            rows.append((f"pooled fixed k={k}, rho={rho:g}, all seeds ({fname})",
                         gain_line(frame[frame["k"] == k], rng)["shared"][0], float(pol[f"fixed_k{k}.0"]), 1e-9))
    print("\nThe shared side against the published values:")
    bad = 0
    for label, mine, published, tol in rows:
        ok = abs(mine - published) <= tol
        bad += not ok
        print(f"  {label:62s} here {mine:+.4f}  published {published:+.4f}  {'ok' if ok else 'MISMATCH'}")
    return bad


RECOVERY_COLS = ["procedure", "cost", "gain", "price", "recovered", "recovered_lo", "recovered_hi"]
SLIP_SDS = ("0", "0.1", "0.25", "0.5")
SLIP_PROC = "tournament_k5_lcb_rho1_look"


def _pct(r: pd.Series) -> str:
    return f"{r['recovered']:+.1%} [{r['recovered_lo']:+.1%}, {r['recovered_hi']:+.1%}]"


def slip_sensitivity() -> None:
    """code-estimands-10: the input arms' sitting over the assumed rating-noise SD of a look.

    Under a slip or a misclick the rating itself is exact; the replay gives each look
    --slip-look-sd landscape SDs of rating noise in the noisy and the clean twin alike
    (0.25 by default, every published number). Replayed at 0, 0.1, 0.25 and 0.5
    (end_of_study_slipsd<sd>, top five by the posterior mean less one latent SD, best
    look, rho = 1). Recovery is replay_end_of_study's: a ratio of landscape means of the
    deployed design's regret at the final trial, standard process against the sitting,
    with a landscape bootstrap; price is the same sitting's cost on the clean twin.
    """
    print("\n=== code-estimands-10: the input arms' sitting over the assumed look SD "
          f"({SLIP_PROC}, pooled over cells) ===")
    base = pd.read_csv(ANALYSIS / "end_of_study" / "end_of_study_recovery.csv")
    base = base[(base["scope"] == "pooled") & (base["procedure"] == SLIP_PROC)].set_index("arm")
    for sd in SLIP_SDS:
        path = ANALYSIS / f"end_of_study_slipsd{sd}" / "end_of_study_recovery.csv"
        if not path.is_file():
            print(f"  look SD {sd}: {path.parent.name} missing")
            continue
        r = pd.read_csv(path)
        r = r[(r["scope"] == "pooled") & (r["procedure"] == SLIP_PROC)]
        for _, x in r.sort_values("arm").iterrows():
            note = ""
            if sd == "0.25" and x["arm"] in base.index:
                same = np.isclose(x["recovered"], base.loc[x["arm"], "recovered"], rtol=0, atol=1e-12)
                note = f"   (end_of_study, published: {base.loc[x['arm'], 'recovered']:+.1%}; " \
                       f"{'reproduced' if same else 'MISMATCH'})"
            print(f"  look SD {sd:>4s} {x['arm']:22s} cost {x['cost']:.4f}  gain {x['gain']:+.4f}  "
                  f"price {x['price']:+.4f}  recovered {_pct(x)}{note}")


def spike_corrected() -> None:
    """code-estimands-9: the gross-fault arm's look-buying rows at the run's own spikes.

    hitl_remedies{,_heldout} (replay_hitl_remedies.py) gave every look the SD
    sigma * sqrt(1 + 0.15) = 0.268, the scaled-spike assumption; the arm ran spikes of SD
    20 with probability 0.15 (variant sp0.15-20), a marginal SD of 7.75. Two corrected
    replays: end_of_study_spikesd{,_heldout} (replay_end_of_study.py --variants sp0.15-20)
    draw a spike per look at the run's rate and size in the fixed-allocation sitting, and
    hitl_remedies_spikesd{,_heldout} re-run the whole remedy replay (replay_hitl_remedies.py,
    unchanged: build_tasks now gives each task the stem's spike variant, which
    sitting_sd_fn falls back on), so its tournament rows are the same draws and its LUCB
    looks, which take only an SD, get Gaussian noise at the marginal SD 7.75. The LUCB
    rows are therefore corrected in scale, not in shape. The capped-scale arm's looks
    are not clipped at the cap either; its look-buying rows are optimistic in the same
    way, and only its rank-rule rows, which buy no look, are quoted. No number of either
    corrected replay is quoted in the paper.

    hitl_remedies{,_heldout} were themselves rerun in place with the corrected look model
    (paper/COMMANDS.md), so the rows at the old look SD of 0.268 are no longer on disk and
    the left-hand side below is the logged directory as it now stands: the block checks
    that the in-place rerun and the two corrected replays agree, not the size of the fix.
    """
    print("\n=== code-estimands-9: gross faults (spike sp0.15-20), look-buying rows as now logged "
          "(rerun in place at the run's spikes) against the two corrected replays (pooled, recovered share "
          "of the deployed cost) ===")
    print("  tournament rows: a spike drawn per look at the run's rate and size; LUCB rows: Gaussian looks at "
          "the run's marginal SD sqrt(sigma^2 + p s^2) = 7.75, since lucb_sitting takes only an SD")
    root = REPO / "output-boba-spike" / "analysis"
    for old, new, full, seeds in (("hitl_remedies", "end_of_study_spikesd", "hitl_remedies_spikesd", "7-11"),
                                  ("hitl_remedies_heldout", "end_of_study_spikesd_heldout",
                                   "hitl_remedies_spikesd_heldout", "12-16")):
        if not (root / new / "end_of_study_recovery.csv").is_file():
            print(f"  {new} missing")
            continue
        o = pd.read_csv(root / old / "hitl_remedies_recovery.csv")
        o = o[(o["error_model"] == "pooled") & o["procedure"].str.startswith(("tournament", "lucb"))]
        n = pd.read_csv(root / new / "end_of_study_recovery.csv")
        n = n[(n["scope"] == "pooled") & n["procedure"].str.startswith("tournament")].set_index("procedure")
        f = None
        if (root / full / "hitl_remedies_recovery.csv").is_file():
            f = pd.read_csv(root / full / "hitl_remedies_recovery.csv")
            f = f[f["error_model"] == "pooled"].set_index("procedure")
        print(f"  seeds {seeds}: {old} (as logged, corrected in place) -> {new} (tournament) / {full} (LUCB)")
        for _, x in o.sort_values("procedure").iterrows():
            p = x["procedure"]
            y = n.loc[p] if p in n.index else (f.loc[p] if f is not None and p in f.index else None)
            if y is None:
                print(f"    {p:32s} {_pct(x):>26s} -> not re-run")
                continue
            how = "spikes per look" if p.startswith("tournament") else "Gaussian at SD 7.75"
            same_cost = np.isclose(x["cost"], y["cost"], rtol=0, atol=1e-12)
            agree = ""
            if f is not None and p in n.index and p in f.index:
                agree = ("  (both corrected replays agree)" if np.isclose(f.loc[p, "recovered"], y["recovered"],
                                                                          rtol=0, atol=1e-12)
                         else "  (CORRECTED REPLAYS DISAGREE)")
            print(f"    {p:32s} {_pct(x):>26s} -> {_pct(y):>26s} ({how})  gain {x['gain']:+.4f} -> {y['gain']:+.4f}"
                  f"  price {x['price']:+.4f} -> {y['price']:+.4f}  cost {y['cost']:.4f}"
                  f"{'' if same_cost else '  COST DIFFERS'}{agree}")
        if f is not None:
            oa = pd.read_csv(root / old / "hitl_remedies_recovery.csv")
            oa = oa[oa["error_model"] == "pooled"].set_index("procedure")
            quiet = [p for p in oa.index if not p.startswith(("tournament", "lucb")) and p in f.index]
            same = all(np.isclose(oa.loc[p, "recovered"], f.loc[p, "recovered"], rtol=0, atol=1e-12,
                                  equal_nan=True) for p in quiet)
            print(f"    the {len(quiet)} rows that buy no look (standard, shortlist, ordinal) identical in "
                  f"{full}: {same}")


# ---------------------------------------------------------------------------
# Every sitting value the paper prints, by the paper's own producers
# ---------------------------------------------------------------------------


def _plain(x):
    """A summary as the selection JSON stores it (int keys become strings, tuples lists)."""
    return json.loads(json.dumps(x))


def _diff_leaves(a, b, path: str = "") -> list[str]:
    """Paths at which two JSON-like trees differ (numbers to 1e-12, NaN equal to NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += _diff_leaves(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: lengths {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff_leaves(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if np.isclose(float(a), float(b), rtol=0, atol=1e-12, equal_nan=True) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def sbm_results() -> tuple[dict[str, dict], int]:
    """sitting_by_magnitude.run on each replay set, and on gaussian error alone (shared).

    Returns, per label, the per-(cell, k) frame and the summaries by cell; and the
    number of shared-side summaries that do not reproduce the published selection JSON.
    """
    ship = pd.read_csv(ANALYSIS / "ship_rules_per_run.csv", low_memory=False)
    out: dict[str, dict] = {}
    for label, specs in SPECS.items():
        d = sbm.load(ANALYSIS, STATS, sbm.PROCESSES, specs)
        frame, _, summaries = sbm.run(d, ship)
        out[label] = {"frame": frame, "cells": {s["cell"]: _plain(s) for s in summaries}, "n_runs": d["file"].nunique()}
    dg = sbm.load(ANALYSIS, STATS, ("gaussian",))
    frame_g, _, sg = sbm.run(dg, ship, ("gaussian",))
    out["gaussian"] = {"frame": frame_g, "cells": {s["cell"]: _plain(s) for s in sg}, "n_runs": dg["file"].nunique()}
    bad = 0
    print("\nsitting_by_magnitude.run re-run in-process; the shared side against its published outputs:")
    for label, fname in (("shared", "sitting_by_magnitude_selection.json"),
                         ("gaussian", "sitting_by_magnitude_gaussian_selection.json")):
        path = ANALYSIS / "review" / fname
        if not path.is_file():
            print(f"  {fname} missing")
            bad += 1
            continue
        pub = {s["cell"]: s for s in json.loads(path.read_text())}
        diffs = []
        for cell, s in out[label]["cells"].items():
            if cell not in pub:
                diffs.append(f"{cell}: not published")
                continue
            diffs += [f"{cell}{d}" for d in _diff_leaves(s, pub[cell])]
        bad += bool(diffs)
        print(f"  {label:8s} vs review/{fname}: {len(out[label]['cells'])} summaries, "
              f"{'identical' if not diffs else f'{len(diffs)} differences, e.g. {diffs[:3]}'}")
    return out, bad


def kcurve(specs: tuple[str, ...], rho: float, derived_name: str, opt_z: dict[str, float]):
    """heldout_remedies.load_kcurve for any replay set (that function reads the shared replays only)."""
    import heldout_remedies as hr  # noqa: E402 - heavy (torch), only needed here
    import budget_split as bs  # noqa: E402
    import replay_end_of_study as eos  # noqa: E402
    cols = ["dataset", "acquisition", "seed", "error_model", "file", "procedure", "family", "k",
            "candidates", "rho", "winner", "regret_noisy", "ref_noisy"]
    frames = []
    for spec in specs:
        name, only = sbm.parse_replay_dir(spec)
        f = pd.read_csv(ANALYSIS / name / "end_of_study_per_run.csv.gz", usecols=cols, low_memory=False)
        frames.append(f if only is None else f[f["error_model"].isin(only)])
    sweep = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["file", "procedure"], keep="first")
    sweep = sweep[(sweep["family"] == "tournament") & (sweep["candidates"] == "lcb")
                  & (sweep["winner"] == "look") & (sweep["rho"] == rho)
                  & sweep["acquisition"].isin(hr.KCURVE_ACQS) & sweep["error_model"].isin(hr.RESPONSE_MODELS)]
    derived = pd.read_csv(ANALYSIS / derived_name)
    joined = sweep.merge(derived[["file", "k_hat", "gp_failed"]], on="file", how="inner")
    joined = joined[~eos._as_bool(joined["gp_failed"])]
    no_trial = pd.read_csv(ANALYSIS / "ship_rules_per_run.csv", usecols=["file", "regret_pm", "regret_lcb1"])
    no_trial["file"] = no_trial["file"].map(lambda f: Path(f).name)
    summary, frame, _ = bs.score_policies(joined, no_trial, opt_z, np.random.default_rng(bs.BOOTSTRAP_SEED))
    rename = {c: f"fixed_k{int(float(c[len('fixed_k'):]))}" for c in frame.columns if c.startswith("fixed_k")}
    frame = frame.rename(columns=rename).reset_index()
    seeds = joined.drop_duplicates("file").set_index("file")["seed"]
    frame["seed"] = frame["file"].map(seeds).astype(int)
    summary = summary.assign(policy_short=summary["policy"].replace(rename))
    return summary, frame


def kcurve_results() -> tuple[dict[str, dict], int]:
    """The pooled k-curve, the tab:heldout k row, the derived budget rule and App. H's Holm family.

    Scored exactly as heldout_remedies.py scores them (budget_split.score_policies,
    Cube, gain_summary, the paired landscape bootstrap of the derived rule against the
    k chosen on seeds 7-11, seed_split and landscape_splits/summarise_splits for the
    landscape-split column and the optimism, Holm over the 41 replayed procedures). In the Holm family
    only the nine fixed k and the derived rule are re-scored; the LUCB sittings come
    from replay_hitl_remedies, which has no sequential process, and keep their shared
    p-values; every other member buys no look or is gaussian and is unchanged.
    """
    import heldout_remedies as hr  # noqa: E402
    import budget_split as bs  # noqa: E402
    from statsmodels.stats.multitest import multipletests  # noqa: E402
    opt_z = {k: float(v["opt_z"]) for k, v in json.loads(STATS.read_text())["functions"].items() if "opt_z" in v}
    pub = pd.read_csv(ANALYSIS / "review" / "heldout_remedies.csv", low_memory=False)
    holm_pub = pub[pub["section"] == "4_holm"].reset_index(drop=True)
    derived_pub = pub[pub["section"] == "5_derived_rule"]
    split_pub = pub[(pub["section"] == "1_seed_split") & (pub["problem"] == "k")
                    & (pub["direction"] == "7-11 -> 12-16")]
    kc = [f"fixed_k{k}" for k in hr.K_GRID]
    out, bad = {}, 0
    for label, specs in SPECS.items():
        s1, f1 = kcurve(specs, 1.0, "budget_split_derived.csv", opt_z)
        s05, _ = kcurve(specs, 0.5, "budget_split_derived_rho0.5.csv", opt_z)
        if label == "shared":
            ref_s, ref_f, _ = hr.load_kcurve(ANALYSIS, 1.0, "budget_split_derived.csv", opt_z)
            same = ref_f.equals(f1) and ref_s.equals(s1)
            bad += not same
            print(f"  k-curve frame, shared, against heldout_remedies.load_kcurve: {'identical' if same else 'DIFFERS'}")
        cube = hr.Cube(hr.kcurve_long(f1))
        prob = hr.Problem("k", "fixed k", cube, kc, "gain", None)
        fwd = kc[prob.pick(prob.values(hr.TRAIN_SEEDS))]
        rev = kc[prob.pick(prob.values(hr.TEST_SEEDS))]
        # tab:heldout's landscape-split column and App. H's optimism, as heldout_remedies.main
        # makes them: the full-data values are the k-curve summary's, the partitions come from
        # SPLIT_SEED (the permutation of each partition does not depend on the other problems).
        pol = s1.set_index("policy_short")
        full = {c: {"value": float(pol.loc[c, "gain_vs_standard"]), "lo": float(pol.loc[c, "gain_lo"]),
                    "hi": float(pol.loc[c, "gain_hi"])} for c in kc}
        full["_best"] = kc[prob.pick(np.array([full[c]["value"] for c in kc]))]
        kprob = hr.Problem("k", "k of the final sitting (gain, rho = 1)", cube, kc, "gain", hr.PAPER_K)
        seed_heads = {d: hr.seed_split(kprob, None, tr, te, d, full)[1]
                      for d, (tr, te) in (("7-11 -> 12-16", (hr.TRAIN_SEEDS, hr.TEST_SEEDS)),
                                          ("12-16 -> 7-11", (hr.TEST_SEEDS, hr.TRAIN_SEEDS)))}
        split_frame = hr.landscape_splits([kprob], {"k": full["_best"]}, hr.N_PARTITIONS, hr.SPLIT_SEED)
        _, split_heads = hr.summarise_splits(split_frame, {"k": full["_best"]}, {"k": full}, {"k": hr.PAPER_K})
        splits = {scheme: split_heads[("k", scheme)] for scheme in ("landscape", "crossed")}
        sel = hr.gain_summary(prob, fwd, hr.TEST_SEEDS)
        der = hr.gain_summary(hr.Problem("derived", "derived", cube, ["derived"], "gain", None), "derived",
                              hr.TEST_SEEDS)
        m = cube.landscape_means(hr.TEST_SEEDS)
        gap = bs._boot_mean_diff(m[2, cube.procs.index("derived")], m[2, cube.procs.index(fwd)],
                                 np.random.default_rng(bs.BOOTSTRAP_SEED))
        p = holm_pub["p_test"].to_numpy(dtype=float).copy()
        gaining = holm_pub["landscapes_gaining_test"].to_numpy(dtype=float).copy()
        mean_gain = holm_pub["mean_gain_test"].to_numpy(dtype=float).copy()
        for i, cand in enumerate(holm_pub["candidate"]):
            if holm_pub.loc[i, "family"] == "extra" and (str(cand).startswith("fixed_k") or cand == "derived"):
                g = hr.per_landscape_gains(cube, cand, hr.TEST_SEEDS)
                p[i], gaining[i], mean_gain[i] = hr._wilcoxon_p(g), (g > 0).sum(), np.nanmean(g)
        adj = np.full(len(p), np.nan)
        ok = np.isfinite(p)
        adj[ok] = multipletests(p[ok], method="holm")[1]
        out[label] = {"rho1": s1.set_index("policy_short")["gain_vs_standard"],
                      "rho05": s05.set_index("policy_short")["gain_vs_standard"],
                      "k_fwd": fwd, "k_rev": rev, "sel": sel, "derived": der, "gap": gap,
                      "full_best": full["_best"], "splits": splits, "seed_heads": seed_heads,
                      "holm": pd.DataFrame({"candidate": holm_pub["candidate"], "family": holm_pub["family"],
                                            "p_test": p, "p_holm_all_test": adj, "gaining": gaining,
                                            "mean_gain_test": mean_gain})}
    sh = out["shared"]
    checks = [
        ("derived rule, seeds 12-16", sh["derived"]["value"],
         float(holm_pub.loc[holm_pub["candidate"] == "derived", "mean_gain_test"].iloc[0])),
        ("k chosen on 7-11 vs derived, seeds 12-16",
         sh["gap"][0], float(derived_pub.loc[derived_pub["note"] == "test seeds, train-selected k vs derived",
                                             "test_value"].iloc[0])),
        ("  its interval, low", sh["gap"][1], float(derived_pub.loc[derived_pub["note"] == "test seeds, train-selected k vs derived", "test_lo"].iloc[0])),
        ("  its interval, high", sh["gap"][2], float(derived_pub.loc[derived_pub["note"] == "test seeds, train-selected k vs derived", "test_hi"].iloc[0])),
        ("tab:heldout k row, seeds 12-16", sh["sel"]["value"],
         float(split_pub.loc[split_pub["role"] == "summary_selected", "test_value"].iloc[0])),
        ("  its interval, low", sh["sel"]["lo"], float(split_pub.loc[split_pub["role"] == "summary_selected", "test_lo"].iloc[0])),
        ("  its interval, high", sh["sel"]["hi"], float(split_pub.loc[split_pub["role"] == "summary_selected", "test_hi"].iloc[0])),
        ("Holm p over the 41 procedures, max |diff|",
         float(np.max(np.abs(sh["holm"]["p_holm_all_test"] - holm_pub["p_holm_all_test"]))), 0.0),
        ("raw p on the test seeds, max |diff|",
         float(np.max(np.abs(sh["holm"]["p_test"] - holm_pub["p_test"]))), 0.0),
    ]
    lsplit = pub[(pub["section"] == "2_landscape_split") & (pub["problem"] == "k")].set_index("scheme")
    for scheme in ("landscape", "crossed"):
        for col in ("test_median", "test_p2_5", "test_p97_5", "share_selected_k5_to_16", "optimism_mean",
                    "optimism_net_of_shift_mean"):
            checks.append((f"k, {scheme} splits: {col}", float(sh["splits"][scheme][col]),
                           float(lsplit.loc[scheme, col])))
        if sh["splits"][scheme]["full_best"] != lsplit.loc[scheme, "full_best"]:
            bad += 1
            print(f"    k, {scheme} splits: full-data winner {sh['splits'][scheme]['full_best']}, published "
                  f"{lsplit.loc[scheme, 'full_best']}  MISMATCH")
    sseed = pub[(pub["section"] == "1_seed_split") & (pub["problem"] == "k")
                & (pub["role"] == "summary_selected")].set_index("direction")
    for direction, h in sh["seed_heads"].items():
        checks.append((f"k, seed split {direction}: optimism", float(h["optimism"]),
                       float(sseed.loc[direction, "optimism"])))
        checks.append((f"k, seed split {direction}: seed shift", float(h["shift"]),
                       float(sseed.loc[direction, "seed_shift_mean_over_candidates"])))
    print("  the shared side against review/heldout_remedies.csv:")
    for name, mine, published in checks:
        ok = abs(mine - published) <= 1e-9
        bad += not ok
        print(f"    {name:44s} here {mine:+.6f}  published {published:+.6f}  {'ok' if ok else 'MISMATCH'}")
    if sh["k_fwd"] != "fixed_k12" or sh["k_rev"] != "fixed_k8":
        bad += 1
        print(f"    pooled k chosen: {sh['k_fwd']} / {sh['k_rev']}, not the published fixed_k12 / fixed_k8  MISMATCH")
    return out, bad


def printed_values(sb: dict[str, dict], kr: dict[str, dict], label: str) -> list[dict]:
    """Every pooled sitting gain the paper prints from these replays, under one replay set.

    Gains of the deployed design at the final trial over the standard process, in
    units of the achievable improvement (increments over LCB1 likewise). ``k`` is the
    sitting size the value is at (0 for a value at no single size), so a value at a
    re-chosen k is not compared like for like with its shared counterpart.
    """
    c, f, k = sb[label]["cells"], sb[label]["frame"], kr[label]
    rows = []

    def add(where: str, what: str, value: float, size) -> None:
        rows.append({"where": where, "what": what, "value": float(value), "k": 0 if size is None else int(size)})

    for cell in PAPER_CELLS:
        s = c[cell]
        kf = s["k_chosen_on_7_11"]
        add("tab:sittingcell", f"{cell}: held-out gain at the k chosen on seeds 7-11", s["test_gain"], kf)
        add("tab:sittingcell", f"{cell}: held-out increment over LCB1 at that k", s["increment_over_lcb1"]["test"], kf)
        add("tab:sittingcell", f"{cell}: k = 12, all seeds", s["k12_gain_all"], 12)
    s = c["1sigma_from_trial_1"]
    add("Sec. 7 / App. H", "1sigma_from_trial_1: gain on seeds 7-11 at the k chosen on 12-16", s["reverse_test_gain"],
        s["k_chosen_on_12_16"])
    add("App. H", "1sigma_from_trial_1: median held-out gain over landscape splits", s["split_median"], None)
    for kk in sbm.K_GRID:
        row = f[(f["cell"] == "1sigma_from_trial_1") & (f["k"] == kk)]
        add("fig:kcurve (blue) / Sec. 7", f"1sigma_from_trial_1: k = {kk}, all seeds", row["gain_all"].iloc[0], kk)
    for kk in (3, 5, 8, 12):
        row = f[(f["cell"] == "1sigma_from_trial_21") & (f["k"] == kk)]
        add("App. H", f"1sigma_from_trial_21: k = {kk}, all seeds", row["gain_all"].iloc[0], kk)
    w = c["pooled_k12_without_5sigma_from_trial_21"]
    add("Sec. 7 / App. H", "pooled k = 12 without 5sigma_from_trial_21, all seeds", w["all_seeds"]["gain"], 12)
    add("App. H", "pooled k = 12 without 5sigma_from_trial_21, seeds 12-16", w["seeds_12_16"]["gain"], 12)
    for kk in sbm.K_GRID:
        add("fig:kcurve (grey) / Sec. 7", f"pooled fixed k = {kk}, all seeds, rho = 1", k["rho1"][f"fixed_k{kk}"], kk)
    add("App. E.2", "pooled fixed k = 16, all seeds, rho = 0.5", k["rho05"]["fixed_k16"], 16)
    add("tab:heldout", "pooled k chosen on seeds 7-11, held-out gain", k["sel"]["value"], int(k["k_fwd"][7:]))
    add("tab:heldout", "pooled k chosen on ten landscapes: median held-out gain over splits",
        k["splits"]["landscape"]["test_median"], None)
    add("Sec. 7 / App. H", "derived budget rule, held-out gain", k["derived"]["value"], None)
    add("Sec. 7 / App. H", "k chosen on seeds 7-11 minus the derived rule, held-out", k["gap"][0], int(k["k_fwd"][7:]))
    return rows


def paper_values(sb: dict[str, dict], kr: dict[str, dict]) -> dict:
    """Print every value, the moves under the sequential process, and where the chosen k changes."""
    print("\n=== Every sitting value the paper prints, shared vs sequential (random order) vs sequential "
          "(rank order) ===")
    print("Gains of the deployed design at the final trial over the standard process, in units of the "
          "achievable improvement; LogEI and qNEI, the four processes, looks at the full idiosyncratic SD. "
          "'*' marks a value at a k that differs from the shared one.")
    rows = {label: printed_values(sb, kr, label) for label in LABELS}
    table = pd.DataFrame(rows["shared"]).rename(columns={"value": "shared", "k": "k_shared"})
    for label in ("random", "rank"):
        other = pd.DataFrame(rows[label])
        table[label] = other["value"].to_numpy()
        table[f"k_{label}"] = other["k"].to_numpy()
    moves = {}
    for _, r in table.iterrows():
        marks = {lab: ("*" if r[f"k_{lab}"] != r["k_shared"] else " ") for lab in ("random", "rank")}
        ks = "" if r["k_shared"] == 0 else f"k={int(r['k_shared'])}"
        print(f"  {r['where']:27s} {r['what']:68s} {ks:5s} shared {r['shared']:+.4f}  random {r['random']:+.4f}"
              f"{marks['random']} ({r['random'] - r['shared']:+.4f})  rank {r['rank']:+.4f}{marks['rank']} "
              f"({r['rank'] - r['shared']:+.4f})")
    for lab in ("random", "rank"):
        same_k = table[table[f"k_{lab}"] == table["k_shared"]]
        other_k = table[~table.index.isin(same_k.index)]
        d = (same_k[lab] - same_k["shared"]).abs()
        i = d.idxmax()
        flips = int(((same_k["shared"] > 0) != (same_k[lab] > 0)).sum())
        moves[lab] = {"max": float(d.max()), "where": same_k.loc[i, "what"], "shared": same_k.loc[i, "shared"],
                      "seq": same_k.loc[i, lab], "n": len(same_k), "flips": flips,
                      "other_k": [(r["what"], int(r["k_shared"]), int(r[f"k_{lab}"]), r["shared"], r[lab])
                                  for _, r in other_k.iterrows()]}
        print(f"  {lab} order: over the {len(same_k)} values at the same k, the largest move is {d.max():.5f} "
              f"({same_k.loc[i, 'what']}: {same_k.loc[i, 'shared']:+.5f} -> {same_k.loc[i, lab]:+.5f}); "
              f"sign changes {flips}; values at a re-chosen k: {len(other_k)}")
        top = d.sort_values(ascending=False).head(5)
        print("      the five largest: " + "; ".join(
            f"{same_k.loc[j, 'what']} {same_k.loc[j, 'shared']:+.4f} -> {same_k.loc[j, lab]:+.4f}" for j in top.index))
        for _, r in same_k[(same_k["shared"] > 0) != (same_k[lab] > 0)].iterrows():
            print(f"      sign change: {r['what']} {r['shared']:+.4f} -> {r[lab]:+.4f}")
        for what, k0, k1, v0, v1 in moves[lab]["other_k"]:
            print(f"      re-chosen k: {what}: k={k0} {v0:+.4f} -> k={k1} {v1:+.4f} ({v1 - v0:+.4f})")

    print("\nOther quantities the text quotes (shared / random / rank):")
    for cell in ("1sigma_from_trial_1", "5sigma_from_trial_21", "5sigma_from_trial_1", "1sigma_from_trial_21"):
        for lab in LABELS:
            s = sb[lab]["cells"][cell]
            pp = ", ".join(f"{p} {v:+.3f}" for p, v in s["per_process_test"].items())
            pa = ", ".join(f"{a} {v:+.3f}" for a, v in s["per_acquisition_test"].items())
            share_k = s["split_k_share"].get(str(s["k_chosen_on_7_11"]), 0.0)
            print(f"  {cell:22s} {lab:6s} k {s['k_chosen_on_7_11']:2d}/{s['k_chosen_on_12_16']:2d}: held-out "
                  f"{fmt(s['test_gain'], s['test_lo'], s['test_hi'])} on {s['test_landscapes_gaining']}/20, share "
                  f"{s['test_recovered_share']:.1%} [{s['test_recovered_lo']:.0%}, {s['test_recovered_hi']:.0%}], "
                  f"Holm p over k {s['test_p_holm_over_k']:.3f}, over all cells {s['test_p_holm_all_cells']:.3f}; "
                  f"two-way [{s['test_two_way_lo']:+.3f}, {s['test_two_way_hi']:+.3f}]; reverse "
                  f"{fmt(s['reverse_test_gain'], s['reverse_lo'], s['reverse_hi'])}; splits choose k="
                  f"{s['k_chosen_on_7_11']} in {share_k:.1%}, median {s['split_median']:+.3f} "
                  f"[{s['split_p2_5']:+.3f}, {s['split_p97_5']:+.3f}]; per process {pp}; per acquisition {pa}; "
                  f"over LCB1 {fmt(s['increment_over_lcb1']['test'], s['increment_over_lcb1']['test_lo'], s['increment_over_lcb1']['test_hi'])}"
                  f", price {s['price_test']:.3f}")
    for lab in LABELS:
        w = sb[lab]["cells"]["pooled_k12_without_5sigma_from_trial_21"]
        print(f"  pooled k = 12 without 5sigma_from_trial_21 ({lab}): all seeds {fmt(w['all_seeds']['gain'], w['all_seeds']['lo'], w['all_seeds']['hi'])}"
              f", seeds 12-16 {fmt(w['seeds_12_16']['gain'], w['seeds_12_16']['lo'], w['seeds_12_16']['hi'])}")
    for lab in LABELS:
        s = sb[lab]["cells"]["pooled"]
        share = sum(v for kk, v in s["split_k_share"].items() if 5 <= int(kk) <= 16)
        print(f"  pooled ({lab}): landscape splits (sitting_by_magnitude's 2,000 folds) choose k in 5..16 in "
              f"{share:.1%}; two-way interval of the held-out k={s['k_chosen_on_7_11']} "
              f"[{s['test_two_way_lo']:+.3f}, {s['test_two_way_hi']:+.3f}]")
    g = sb["gaussian"]["cells"]["5sigma_from_trial_21"]
    print(f"  5sigma_from_trial_21, gaussian error alone (sitting_by_magnitude --processes gaussian; identical under "
          f"every process model): k {g['k_chosen_on_7_11']} held-out {fmt(g['test_gain'], g['test_lo'], g['test_hi'])} "
          f"on {g['test_landscapes_gaining']}/20")
    for lab in LABELS:
        k = kr[lab]
        print(f"  k-curve ({lab}): pooled k chosen {k['k_fwd']} on 7-11 / {k['k_rev']} on 12-16; held-out gain "
              f"{fmt(k['sel']['value'], k['sel']['lo'], k['sel']['hi'])}; derived rule held-out "
              f"{fmt(k['derived']['value'], k['derived']['lo'], k['derived']['hi'])}, "
              f"{fmt(*k['gap'])} below the chosen k")
        net = [k["seed_heads"][d]["optimism"] - k["seed_heads"][d]["shift"] for d in k["seed_heads"]]
        net += [k["splits"][sc]["optimism_net_of_shift_mean"] for sc in ("landscape", "crossed")]
        print("      landscape splits (heldout_remedies, 1000 partitions x 2 folds, full-data winner "
              f"{k['full_best']}): " + "; ".join(
                  f"{sc} median {fmt(k['splits'][sc]['test_median'], k['splits'][sc]['test_p2_5'], k['splits'][sc]['test_p97_5'])}"
                  f", chosen k in 5..16 in {k['splits'][sc]['share_selected_k5_to_16']:.2%}"
                  f" (outside in {1 - k['splits'][sc]['share_selected_k5_to_16']:.2%})" for sc in ("landscape", "crossed"))
              + f"; optimism net of the seed shift, seed split both ways and the two split schemes: "
              + ", ".join(f"{v:+.4f}" for v in net) + f" (range {min(net):+.3f} to {max(net):+.3f})")
        h = k["holm"]
        extra = h[h["candidate"].astype(str).str.startswith("fixed_k") | (h["candidate"] == "derived")]
        print("      Holm over the 41 procedures, test seeds: " + "; ".join(
            f"{r['candidate']} {int(r['gaining'])}/20 p {r['p_test']:.4f} adj {r['p_holm_all_test']:.3f}"
            for _, r in extra.iterrows()))
        survive = h[h["p_holm_all_test"] < 0.05]
        print(f"      surviving at 0.05 over the 41: {', '.join(f'{c} ({m:+.3f})' for c, m in zip(survive['candidate'], survive['mean_gain_test']))}")
    return moves


def k_changes(sb: dict[str, dict], kr: dict[str, dict]) -> list[tuple]:
    """Every cell and seed direction in which the sequential process changes the chosen k."""
    print("\nThe chosen k, shared / random / rank (seeds 7-11 choose and 12-16 score, and the reverse):")
    changes = []
    for cell in PAPER_CELLS:
        fk = {lab: sb[lab]["frame"][sb[lab]["frame"]["cell"] == cell].set_index("k") for lab in LABELS}
        for direction, key, seeds_col in (("7-11", "k_chosen_on_7_11", "gain_train"),
                                          ("12-16", "k_chosen_on_12_16", "gain_test")):
            ks = {lab: sb[lab]["cells"][cell][key] for lab in LABELS}
            line = f"  {cell:24s} chosen on {direction:5s}: {ks['shared']:2d} / {ks['random']:2d} / {ks['rank']:2d}"
            for lab in ("random", "rank"):
                if ks[lab] != ks["shared"]:
                    t = fk[lab]
                    holm_new = float(t.loc[ks[lab], "p_test_holm_over_k"])
                    best_all = float(t["gain_all"].max())
                    tie = float(t.loc[ks[lab], seeds_col] - t.loc[ks["shared"], seeds_col])
                    changes.append((cell, direction, lab, ks["shared"], ks[lab], holm_new, best_all, tie))
                    line += (f"   [{lab}: k {ks['shared']} -> {ks[lab]}; on the choosing seeds k={ks[lab]} leads "
                             f"k={ks['shared']} by {tie:+.4f}; best all-seed gain of any size in the cell "
                             f"{best_all:+.4f}]")
            print(line)
    for direction, key in (("7-11", "k_fwd"), ("12-16", "k_rev")):
        ks = {lab: int(kr[lab][key][7:]) for lab in LABELS}
        print(f"  {'pooled, heldout_remedies':24s} chosen on {direction:5s}: {ks['shared']:2d} / {ks['random']:2d} / "
              f"{ks['rank']:2d}")
    return changes


def main() -> None:
    print(__doc__)
    shared = load(DIRS["shared"])
    seq = load(DIRS["sequential"])
    m = check_unchanged(shared, seq, "shared vs sequential (random order)")
    print("\nReproduction of the paper's shared-model cells (seeds 12-16, pooled over processes):")
    for cell, (sigma, onset, _) in CELLS.items():
        for k in ((2, 16) if onset == 0 else (16,)):
            b = m[np.isclose(m["sigma"], sigma) & (m["onset"] == onset) & (m["k"] == k) & m["seed"].isin(sbm.TEST_SEEDS)]
            per = b.groupby(["error_model", "dataset"])["gain"].mean().groupby("error_model").mean()
            pooled = b.groupby("dataset")["gain"].mean().mean()
            print(f"  {cell} k={k}: pooled {pooled:+.4f}; " + ", ".join(f"{p} {per[p]:+.4f}" for p in PROCESSES))

    print("\n=== Sequential process, candidates shown in a random order per run ===")
    print_cells(m, "sequential = random order")
    largest_moves(m, "random order")
    print_pooled_curve(m, "sequential = random order", "rho = 1")
    sel = {"shared": print_selection(shared, "shared"),
           "random": print_selection(with_seq(shared, seq), "sequential, random order")}

    rank_moving = load(DIRS["sequential_rank"])
    rank = pd.concat([seq[seq["error_model"].isin(("gaussian", "bias"))], rank_moving], ignore_index=True)
    mr = shared.merge(rank, on=KEY, suffixes=("", "_seq"), validate="one_to_one")
    if len(mr) != len(m):
        raise SystemExit(f"rank order: {len(m) - len(mr)} rows have no rank-order twin")
    print("\n=== Sequential process, candidates shown best-ranked first (drift and ar1 replayed; "
          "gaussian and bias from the random-order replay, identical by construction) ===")
    print_cells(mr, "sequential = rank order")
    largest_moves(mr, "rank order")
    print_pooled_curve(mr, "sequential = rank order", "rho = 1")
    sel["rank"] = print_selection(with_seq(shared, rank), "sequential, rank order")

    half_shared, half_seq = load(DIRS["shared"], rho=0.5), load(DIRS["sequential"], rho=0.5)
    mh = check_unchanged(half_shared, half_seq, "rho = 0.5: shared vs sequential (random order)")
    print_pooled_curve(mh, "sequential = random order", "rho = 0.5, the half-SD sensitivity")
    half_rank = pd.concat([half_seq[half_seq["error_model"].isin(("gaussian", "bias"))],
                           load(DIRS["sequential_rank"], rho=0.5)], ignore_index=True)
    mhr = half_shared.merge(half_rank, on=KEY, suffixes=("", "_seq"), validate="one_to_one")
    if len(mhr) != len(mh):
        raise SystemExit(f"rho = 0.5, rank order: {len(mh) - len(mhr)} rows have no rank-order twin")
    print_pooled_curve(mhr, "sequential = rank order", "rho = 0.5, the half-SD sensitivity")
    bad = check_against_published(m, mh)
    claims({"shared": "end_of_study", "sequential": "end_of_study_ksweep_seq"})
    slip_sensitivity()
    spike_corrected()
    print("\n=== The paper's own producers re-run on each replay set (shared = the published outputs) ===")
    sb, bad_sbm = sbm_results()
    print("heldout_remedies' k-curve scoring re-run in-process:")
    kr, bad_kc = kcurve_results()
    bad += bad_sbm + bad_kc
    for label in LABELS:
        for cell in PAPER_CELLS:
            a, b = sel[label][cell], sb[label]["cells"][cell]
            if _diff_leaves(_plain({k: a[k] for k in ("k_chosen_on_7_11", "test_gain", "test_lo", "test_hi")}),
                            {k: b[k] for k in ("k_chosen_on_7_11", "test_gain", "test_lo", "test_hi")}):
                bad += 1
                print(f"  {label} {cell}: print_selection and sitting_by_magnitude.run disagree  MISMATCH")
    moves = paper_values(sb, kr)
    changes = k_changes(sb, kr)
    headline(sel, m, mr, mh, mhr, sb, kr, moves, changes)
    if bad:
        raise SystemExit(f"{bad} shared-model values do not reproduce the published ones")


def headline(sel: dict[str, dict], m: pd.DataFrame, mr: pd.DataFrame, mh: pd.DataFrame,
             mhr: pd.DataFrame, sb: dict[str, dict], kr: dict[str, dict], moves: dict, changes: list) -> None:
    """The values the paper quotes, simultaneous against sequential, in one place."""
    print("\n=== The paper's sitting numbers, simultaneous (shared) vs sequential ===")
    print("Held-out gain (seeds 12-16) at the k chosen on seeds 7-11, sitting_by_magnitude.analyse_cell "
          "(same bootstrap draws as tab:sittingcell):")
    for cell in ("1sigma_from_trial_1", "5sigma_from_trial_21", "1sigma_from_trial_21", "5sigma_from_trial_1"):
        for label in ("shared", "random", "rank"):
            s = sel[label][cell]
            pp = ", ".join(f"{p} {v:+.3f}" for p, v in s["per_process_test"].items())
            print(f"  {cell:22s} {label:6s} k={s['k_chosen_on_7_11']:2d}/{s['k_chosen_on_12_16']:2d} "
                  f"{fmt(s['test_gain'], s['test_lo'], s['test_hi'])}"
                  f" on {s['test_landscapes_gaining']}/20, Holm p {s['test_p_holm_over_k']:.3f}, share of the cost "
                  f"{s['test_recovered_share']:.1%}; {pp}")
    g = sb["gaussian"]["cells"]["5sigma_from_trial_21"]
    print(f"  5sigma_from_trial_21 k={g['k_chosen_on_7_11']}, gaussian error alone, seeds 12-16 (either process; "
          f"sitting_by_magnitude --processes gaussian, its draws): {fmt(g['test_gain'], g['test_lo'], g['test_hi'])}")
    for lab in ("random", "rank"):
        mv = moves[lab]
        print(f"  {lab} order: largest move of a gain the paper prints, at the same k, {mv['max']:.4f} "
              f"({mv['where']}: {mv['shared']:+.4f} -> {mv['seq']:+.4f}), sign changes {mv['flips']} of {mv['n']}; "
              f"the chosen k changes in {sum(1 for c in changes if c[2] == lab)} of {2 * len(PAPER_CELLS)} "
              f"(cell, seed direction) choices, the new size leading the old by at most "
              f"{max((c[7] for c in changes if c[2] == lab), default=0.0):.4f} on the seeds that chose it: "
              + "; ".join(f"{c[0]} on {c[1]}: {c[3]} -> {c[4]}" for c in changes if c[2] == lab))
    for lab in LABELS:
        k = kr[lab]
        print(f"  derived budget rule, held-out ({lab}): {fmt(k['derived']['value'], k['derived']['lo'], k['derived']['hi'])}"
              f", {fmt(*k['gap'])} below the pooled k chosen on seeds 7-11 ({k['k_fwd']})")
    print("Pooled k-curve over magnitudes, onsets and the four processes, all seeds (the shared values are "
          "budget_split_policies*.csv's, checked above):")
    for k in (5, 8, 12, 16):
        a = gain_line(m[m["k"] == k], np.random.default_rng(SEED))
        b = gain_line(mr[mr["k"] == k], np.random.default_rng(SEED))
        h = gain_line(mh[mh["k"] == k], np.random.default_rng(SEED))
        hr = gain_line(mhr[mhr["k"] == k], np.random.default_rng(SEED))
        print(f"  k={k:2d}  rho=1: shared {a['shared'][0]:+.3f}, sequential random {a['seq'][0]:+.3f}, "
              f"rank {b['seq'][0]:+.3f};  rho=0.5: shared {h['shared'][0]:+.3f}, sequential random "
              f"{h['seq'][0]:+.3f}, rank {hr['seq'][0]:+.3f}")
    for lab in LABELS:
        w = sb[lab]["cells"]["pooled_k12_without_5sigma_from_trial_21"]
        print(f"  k=12 without 5sigma from trial 21 ({lab}; sitting_by_magnitude's draws): all seeds "
              f"{fmt(w['all_seeds']['gain'], w['all_seeds']['lo'], w['all_seeds']['hi'])}, seeds 12-16 "
              f"{fmt(w['seeds_12_16']['gain'], w['seeds_12_16']['lo'], w['seeds_12_16']['hi'])}")


if __name__ == "__main__":
    main()
