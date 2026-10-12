"""End-of-study replays (scripts/replay_end_of_study.py).

The replay is only valid if (1) it never looks past the trial where the changed
ending starts, (2) it reproduces what the loop itself logged, (3) the GP it
conditions on the looks is the loop's GP with the transforms right, and (4) the
noise, ship and claim rules do what the design says. Each test pins one of these
on a tiny run (16 trials, a 64-point candidate pool).
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bo_sensor_error_simulation as sim  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
import replay_end_of_study as eos  # noqa: E402

T, INIT, FN, SEED = 16, 5, "branin", 7
STEM = f"bo_sensor_error_{FN}_value_logei_seed{SEED}"
CLEAN = f"{STEM}_baseline_exact.csv"
NOISY = f"{STEM}_jittered_exact_gaussian_jit0_std1.0.csv"


def _metadata(root: Path, **extra) -> None:
    args = {"iterations": T, "initial_samples": INIT, "error_ar1_rho": 0.8, "input_error": "none", **extra}
    (root / "run_metadata.json").write_text(json.dumps({"args": args}), encoding="utf-8")


def _simulate(apply_error: bool) -> pd.DataFrame:
    stats = bb.load_stats()
    entry = stats[FN]
    oracle = bb.SyntheticOracle.from_stats(FN, stats)
    spec = bb.BENCHMARKS[FN]
    config = sim.SimulationConfig(
        iterations=T, jitter_iteration=0, jitter_std=1.0 if apply_error else 0.0, single_error=False,
        initial_samples=INIT, candidate_pool=64, objective="value", objective_columns=["value"],
        param_columns=spec.param_columns, seed=SEED, error_model="gaussian" if apply_error else "none",
        error_bias=0.0, error_spike_prob=0.1, error_spike_std=0.5, dropout_strategy="hold_last",
        normalize_objective=False, objective_weights=None, acq_num_restarts=2, acq_raw_samples=32,
        acq_maxiter=50, acq_mc_samples=32, ref_point=None,
    )
    torch.manual_seed(SEED)
    jitter_rng = np.random.default_rng(np.random.SeedSequence([SEED, 0, 0, 1_000_000, 0])) if apply_error else None
    frame = sim.run_simulation(
        oracle=oracle, bounds=sim.Bounds(low=spec.bounds_low, high=spec.bounds_high), config=config,
        acq=sim.AcquisitionConfig(name="logei"), rng=np.random.default_rng(SEED), jitter_rng=jitter_rng,
        run_id="t", apply_error=apply_error, oracle_model="exact",
        y_opt=(entry["y_opt"] - entry["mean"]) / entry["std"],
    )
    frame["dataset"] = FN
    return frame


@pytest.fixture(scope="module")
def tiny_arm(tmp_path_factory) -> Path:
    """One clean and one gaussian 1 SD LogEI run, named the way the synthetic driver names them."""
    root = tmp_path_factory.mktemp("arm") / "output-tiny"
    (root / FN).mkdir(parents=True)
    _metadata(root)
    _simulate(False).to_csv(root / FN / CLEAN, index=False)
    _simulate(True).to_csv(root / FN / NOISY, index=False)
    return root


def _bounds() -> torch.Tensor:
    spec = bb.BENCHMARKS[FN]
    return torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)


# --- noise model -------------------------------------------------------------


def test_idiosyncratic_sd_keeps_only_the_part_that_does_not_cancel():
    assert eos.idiosyncratic_sd("none", 5.0) == 0.0
    for model in ("gaussian", "bias", "drift"):
        assert eos.idiosyncratic_sd(model, 2.0) == 2.0
    assert eos.idiosyncratic_sd("ar1", 1.0, 0.8) == pytest.approx(0.6)
    # A spike is drawn afresh per rating, so all of it is idiosyncratic and the
    # marginal SD carries the spike probability as well as the jitter.
    assert eos.idiosyncratic_sd("spike", 1.0) == pytest.approx((1.0 + eos.SPIKE_PROB_DEFAULT) ** 0.5)
    with pytest.raises(ValueError, match="no idiosyncratic-noise model"):
        eos.idiosyncratic_sd("dropout", 1.0)


def test_sitting_noise_follows_the_onset_except_for_rendered_designs():
    settings = eos.Settings()
    response = eos.ArmInfo("a", Path("a"), 50, 5, 0.8, input_arm=False)
    sd = eos.sitting_sd_fn(response, "gaussian", 1.0, 20, settings)
    assert sd(20) == 0.0 and sd(21) == 1.0
    assert eos.sitting_sd_fn(response, "none", 0.0, 0, settings)(49) == 0.0
    slip = dataclasses.replace(response, input_arm=True)
    # the rendered design's rating noise is part of the procedure: same in the clean twin
    assert eos.sitting_sd_fn(slip, "slip", 0.4, 20, settings)(3) == 0.25
    assert eos.sitting_sd_fn(slip, "none", 0.0, 0, settings)(49) == 0.25


def test_noise_is_seeded_by_file_name_and_procedure():
    draw = lambda *a: np.random.default_rng(eos.noise_seed(*a)).standard_normal(3)  # noqa: E731
    np.testing.assert_array_equal(draw("a.csv", 1, 3), draw("a.csv", 1, 3))
    assert not np.allclose(draw("a.csv", 1, 3), draw("b.csv", 1, 3))
    assert not np.allclose(draw("a.csv", 1, 3), draw("a.csv", 2, 3))


# --- GP -------------------------------------------------------------------------


def test_conditioned_mean_matches_botorch():
    torch.manual_seed(0)
    bounds = _bounds()
    X = bounds[0] + torch.rand(12, 2, dtype=torch.double) * (bounds[1] - bounds[0])
    Y = torch.sin(X[:, :1]) + 0.1 * torch.randn(12, 1, dtype=torch.double)
    gp = eos.fit_loop_gp(X, Y, bounds, seed=0)
    Xq = bounds[0] + torch.rand(5, 2, dtype=torch.double) * (bounds[1] - bounds[0])
    no_extra = eos.conditioned_mean(gp, X, Y, X[:0], Y[:0], Xq, bounds)
    with torch.no_grad():
        np.testing.assert_allclose(no_extra, gp.posterior(Xq).mean.reshape(-1).numpy(), atol=1e-6)
    X_new, Y_new = X[:3], Y[:3] + 0.5   # looks at visited designs, as in a tournament
    mine = eos.conditioned_mean(gp, X, Y, X_new, Y_new, Xq, bounds)
    conditioned = gp.condition_on_observations(X=X_new, Y=Y_new)
    with torch.no_grad():
        np.testing.assert_allclose(mine, conditioned.posterior(Xq).mean.reshape(-1).numpy(), atol=1e-6)


# --- decision rules -----------------------------------------------------------------


def test_paired_p_value():
    assert eos.paired_one_sided_p([2.0, 2.0], [1.0, 1.0]) == 0.0      # exact ratings, D better
    assert eos.paired_one_sided_p([1.0, 1.0], [1.0, 1.0]) == 1.0      # exact ratings, a tie
    better, worse = np.array([1.0, 2.5, 1.7]), np.array([0.2, 0.9, 1.5])
    from scipy import stats
    assert eos.paired_one_sided_p(better, worse) == pytest.approx(
        stats.ttest_rel(better, worse, alternative="greater").pvalue)
    with pytest.raises(ValueError):
        eos.paired_one_sided_p([1.0], [0.0])


def test_confirmation_ships_and_claims_by_the_ratings_but_is_scored_by_the_truth():
    # D truly worse, but its ratings are clearly higher: ships D, claims, falsely
    d = eos.confirmation_decision(f_d=0.0, f_c=1.0, r_d=np.array([3.0, 3.1, 2.9]), r_c=np.array([0.0, 0.2, 0.1]),
                                  alpha=0.05)
    assert d["shipped_d"] and d["claim"] and d["false_claim"] and d["shipped_value"] == 0.0
    # equal mean ratings ship D (at least C's), but a tie is no claim
    d = eos.confirmation_decision(1.0, 0.0, np.array([1.0, 2.0]), np.array([2.0, 1.0]), 0.05)
    assert d["shipped_d"] and not d["claim"] and not d["false_claim"]
    d = eos.confirmation_decision(1.0, 0.0, np.array([0.0, 0.1]), np.array([2.0, 1.0]), 0.05)
    assert not d["shipped_d"] and d["shipped_value"] == 0.0


# --- the replay on a real (tiny) run ----------------------------------------------------


def test_replay_reproduces_the_logged_standard_process(tiny_arm, tmp_path):
    arm = eos.load_arm(tiny_arm, "none")
    settings = eos.Settings()
    tasks, counts = eos.build_tasks(arm, eos.Filters(), settings, tmp_path)
    assert len(tasks) == 1 and counts["noisy runs"] == 1
    rows = eos.replay_stem_rows(tasks[0])
    noisy, clean = pd.read_csv(tiny_arm / FN / NOISY), pd.read_csv(tiny_arm / FN / CLEAN)
    std = rows[rows["procedure"] == "standard"].iloc[0]
    assert std["regret_noisy"] == noisy["inference_simple_regret_true"].iloc[-1]
    assert std["ref_clean"] == clean["inference_simple_regret_true"].iloc[-1]
    # 2 candidate rules x 2 rho x 2 winner rules per k, two k; two confirmations; the standard process
    assert rows["procedure"].nunique() == 2 * 8 + 2 + 1
    assert np.isfinite(rows[["regret_noisy", "regret_clean"]].to_numpy()).all()
    assert not rows.filter(like="gp_failed").fillna(False).astype(bool).to_numpy().any()
    # exact looks in the clean twin: the best look is the best candidate
    look = rows[rows["winner"] == "look"]
    np.testing.assert_allclose(look["regret_clean"], look["best_candidate_regret_clean"])
    # a replay is deterministic
    pd.testing.assert_frame_equal(rows, eos.replay_stem_rows(tasks[0]))


def test_replay_never_reads_past_the_changed_ending(tiny_arm):
    log = eos.read_run(tiny_arm / FN / NOISY, T)
    arm = eos.load_arm(tiny_arm, "none")
    sd = eos.sitting_sd_fn(arm, "gaussian", 1.0, 0, eos.Settings())
    bounds = _bounds()

    def spoil(run, start):
        bad = dataclasses.replace(run, X=run.X.copy(), observed=run.observed.copy(), deployed=run.deployed.copy())
        bad.X[start:] = 1.0
        bad.observed[start:] = 99.0
        bad.deployed[start:] = -99.0
        return bad

    def tournament(run):   # search 1..T-3, one sitting in T-2..T
        return eos.tournament(run, eos.search_state(run, T - 3, bounds, 1.0), 3, (1.0, 0.5), sd, T)

    def confirm(run):      # search 1..T-4, ratings in T-3..T
        return eos.confirmation(run, eos.search_state(run, T - 4, bounds, 1.0), 2, INIT, sd, T, 0.05)

    assert tournament(spoil(log, T - 3)) == tournament(log)
    assert confirm(spoil(log, T - 4)) == confirm(log)
    # the check has teeth: spoiling the last SEARCHED trial does change the tournament
    assert tournament(spoil(log, T - 4)) != tournament(log)


def test_look_noise_scales_with_rho():
    run = eos.RunLog(name="r.csv", X=np.arange(6, dtype=float).reshape(-1, 1), observed=np.zeros(6),
                     deployed=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), logged_inference=np.zeros(6), y_opt=1.0)
    state = eos.SearchState(n=6, first=np.arange(6), obs_mean=run.deployed.copy(), lcb=run.deployed.copy(), gp=None,
                            train_X=torch.zeros(6, 1), train_Y=torch.zeros(6, 1),
                            bounds=torch.tensor([[0.0], [5.0]]), gp_failed=True)
    out = eos.tournament(run, state, 3, (0.0, 1.0), lambda trial: 1000.0, T=10)
    # rho = 0 silences the sitting whatever its SD: the best look is the best candidate
    assert out["tournament_k3_lcb_rho0_look"]["regret"] == pytest.approx(0.5)
    # at an SD of 1000 the pick is the noise's argmax, not the truth's
    z = np.random.default_rng(eos.noise_seed("r.csv", eos.PROC_TOURNAMENT, 3)).standard_normal(3)
    truth = np.array([0.5, 0.4, 0.3])   # the candidates, best ship-rule score first
    assert out["tournament_k3_lcb_rho1_look"]["regret"] == pytest.approx(1.0 - truth[np.argmax(truth + 1000 * z)])


def test_logged_inference_mismatch_is_an_error(tiny_arm):
    log = eos.read_run(tiny_arm / FN / NOISY, T)
    log = dataclasses.replace(log, logged_inference=log.logged_inference.copy())
    log.logged_inference[T - 1] += 1e-3
    with pytest.raises(eos.ReproductionError):
        eos.check_logged_inference(log, [T])


# --- validation and indexing -------------------------------------------------------


def test_invalid_settings_and_arms_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="paired t-test"):
        eos.validate_settings(eos.Settings(confirm_k=(1,)))
    with pytest.raises(ValueError, match="two candidates"):
        eos.validate_settings(eos.Settings(tournament_k=(1,)))
    with pytest.raises(ValueError, match="repeats"):
        eos.validate_settings(eos.Settings(rhos=(1.0, 1.0)))
    with pytest.raises(ValueError, match="no model-based search"):
        eos.validate_arm(eos.ArmInfo("a", tmp_path, 12, 5, 0.8, False), eos.Settings())
    _metadata(tmp_path, final_rerate="3,2")
    with pytest.raises(ValueError, match="re-rates"):
        eos.load_arm(tmp_path)
    _metadata(tmp_path, input_error="slip", input_error_from_sweep=False)
    with pytest.raises(ValueError, match="fixed input error"):
        eos.load_arm(tmp_path)
    _metadata(tmp_path)
    with pytest.raises(ValueError, match="unsupported error model"):
        eos.build_tasks(eos.load_arm(tmp_path), eos.Filters(error_models=frozenset({"dropout"})),
                        eos.Settings(), tmp_path / "out")


def test_only_the_standard_variants_are_replayed(tmp_path):
    root = tmp_path / "arm"
    (root / FN).mkdir(parents=True)
    _metadata(root)
    names = [CLEAN, NOISY,
             f"{STEM}_jittered_exact_bias_jit0_std1.0_bias1.csv",      # the arm's scaled bias: kept
             f"{STEM}_jittered_exact_bias_jit0_std1.0_bias0.5.csv",    # another bias: skipped
             f"{STEM}_jittered_exact_gaussian_jit0_std1.0_single.csv",  # another arm: skipped
             f"{STEM}_jittered_exact_dropout_jit0_std1.0.csv"]          # unsupported: skipped
    for name in names:
        (root / FN / name).write_text("iteration\n", encoding="utf-8")
    tasks, counts = eos.build_tasks(eos.load_arm(root, "none"), eos.Filters(), eos.Settings(), tmp_path / "o")
    kept = sorted((r["error_model"], r["variant"]) for r in tasks[0]["runs"])
    assert kept == [("bias", "bias1"), ("gaussian", "")]
    assert counts["skipped: variant bias0.5"] == 1 and counts["skipped: variant single"] == 1
    assert counts["skipped: error model dropout"] == 1


# --- summaries -------------------------------------------------------------------------------


def test_recovery_and_claim_tables():
    rows = []
    for i, dataset in enumerate(["a", "b", "c"]):
        for seed in (1, 2):
            base = {"arm": "arm", "dataset": dataset, "acquisition": "logei", "seed": seed,
                    "error_model": "gaussian", "jitter_std": 1.0, "jitter_iteration": 0, "file": f"{dataset}{seed}",
                    "ref_noisy": 2.0, "ref_clean": 1.0}
            rows.append({**base, "procedure": "confirm_k2", "family": "confirmation", "k": 2, "candidates": None,
                         "rho": None, "winner": None, "regret_noisy": 1.5, "regret_clean": 1.1,
                         "claim_noisy": True, "truly_better_noisy": seed == 1, "false_claim_noisy": seed == 2,
                         "claim_clean": True, "shipped_d_noisy": True, "d_is_c_noisy": False})
            rows.append({**base, "procedure": "standard", "family": "standard", "k": None, "candidates": None,
                         "rho": None, "winner": None, "regret_noisy": 2.0, "regret_clean": 1.0,
                         "claim_noisy": i == 0, "truly_better_noisy": True, "false_claim_noisy": False,
                         "claim_clean": True})
    frame = pd.DataFrame(rows)
    with pytest.raises(KeyError, match="no opt_z"):   # a landscape without a scale is refused, not given 1.0
        eos.recovery_table(frame, {"a": 2.0})
    rec = eos.recovery_table(frame, {"a": 2.0, "b": 1.0, "c": 1.0})
    assert set(rec["procedure"]) == {"confirm_k2"}
    cell = rec[rec["scope"] == "cell"].iloc[0]
    # cost 1, gain 0.5, price 0.1 on every landscape; landscape a is in units of opt_z = 2
    assert cell["cost"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert cell["recovered"] == pytest.approx(0.5)
    assert cell["price"] == pytest.approx((0.05 + 0.1 + 0.1) / 3)
    claims = eos.claims_table(frame).set_index(["procedure", "scope"])
    confirm = claims.loc[("confirm_k2", "cell")]
    assert confirm["claim_rate"] == 1.0 and confirm["false_claim_rate"] == 0.5
    assert confirm["power"] == 1.0 and confirm["size"] == 1.0 and confirm["false_discovery_share"] == 0.5
    standard = claims.loc[("standard", "cell")]
    assert standard["claim_rate"] == pytest.approx(1 / 3) and standard["power"] == pytest.approx(1 / 3)
    assert np.isnan(standard["size"])


def test_fdr_leaves_a_missing_p_out_of_the_family():
    from statsmodels.stats.multitest import multipletests
    q = eos.fdr_bh([0.01, np.nan, 0.04, 0.03])
    assert np.isnan(q[1])
    np.testing.assert_allclose(q[[0, 2, 3]], multipletests([0.01, 0.04, 0.03], method="fdr_bh")[1])
    np.testing.assert_allclose(eos.fdr_bh([0.2, 0.01]), multipletests([0.2, 0.01], method="fdr_bh")[1])
    assert np.isnan(eos.fdr_bh([np.nan])).all()


# --- a sitting that cannot fill its trials ---------------------------------------------------


def test_a_sitting_short_of_distinct_designs_records_the_looks_it_made():
    run = eos.RunLog(name="r.csv", X=np.arange(6, dtype=float).reshape(-1, 1), observed=np.zeros(6),
                     deployed=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), logged_inference=np.zeros(6), y_opt=1.0)
    state = eos.SearchState(n=6, first=np.arange(6), obs_mean=run.deployed.copy(), lcb=run.deployed.copy(), gp=None,
                            train_X=torch.zeros(6, 1), train_Y=torch.zeros(6, 1),
                            bounds=torch.tensor([[0.0], [5.0]]), gp_failed=True)
    for k, looks in ((3, 3), (8, 6)):
        for proc, res in eos.tournament(run, state, k, (1.0,), lambda trial: 1.0, T=20).items():
            # one look per candidate shown: six distinct designs fill a sitting of three, not one of eight
            assert res["n_looks"] == res["n_candidates"] == looks, proc
            assert res["unused_trials"] == k - looks, proc


def test_old_per_run_files_get_their_looks_and_the_short_share_is_tabulated():
    rows = []
    for i, (n_shown, k) in enumerate([(8, 8), (6, 8), (5, 5), (8, 8)]):
        for proc in ("lcb_rho1_look", "obs_rho1_look"):
            rows.append({"arm": "arm", "dataset": "a", "acquisition": "logei", "seed": i, "file": f"f{i}_k{k}",
                         "family": "tournament", "k": k, "procedure": f"tournament_k{k}_{proc}",
                         "n_candidates_noisy": n_shown, "n_candidates_clean": k})
    rows.append({"arm": "arm", "dataset": "a", "acquisition": "logei", "seed": 0, "file": "f0_k8",
                 "family": "standard", "k": np.nan, "procedure": "standard"})
    frame = eos.derive_looks(pd.DataFrame(rows))
    sit = frame[frame["family"] == "tournament"]
    np.testing.assert_array_equal(sit["n_looks_noisy"], sit["n_candidates_noisy"])
    np.testing.assert_array_equal(sit["unused_trials_noisy"], sit["k"] - sit["n_candidates_noisy"])
    assert frame.loc[frame["family"] == "standard", "n_looks_noisy"].isna().all()
    looks = eos.looks_table(frame).set_index("k")
    assert looks.loc[8, "n_runs"] == 3 and looks.loc[8, "share_short_noisy"] == pytest.approx(1 / 3)
    assert looks.loc[8, "min_looks_noisy"] == 6 and looks.loc[8, "mean_unused_noisy"] == pytest.approx(2 / 3)
    assert looks.loc[5, "share_short_noisy"] == 0.0 and looks.loc[8, "share_short_clean"] == 0.0


# --- improvement claims without the test, and with the chosen design equal to the start -----------


def _old_claim_rates(block: pd.DataFrame, rng: np.random.Generator) -> dict:
    """claim_rates as committed before the fresh-mean and D = C definitions were added."""
    claim = eos._as_bool(block["claim_noisy"])
    better = eos._as_bool(block["truly_better_noisy"])
    false = claim & ~better
    parts = pd.DataFrame({"dataset": block["dataset"].to_numpy(), "n": 1, "claim": claim.to_numpy(),
                          "false": false.to_numpy(), "better": better.to_numpy(),
                          "hit": (claim & better).to_numpy(), "not_better": (~better).to_numpy()})
    per = parts.groupby("dataset")[["n", "claim", "false", "better", "hit", "not_better"]].sum()
    cols = {c: i for i, c in enumerate(per.columns)}
    counts = per.to_numpy(dtype=float)
    boot = counts[rng.integers(0, len(per), size=(eos.aba.BOOTSTRAP_REPS, len(per)))].sum(axis=1)
    total = counts.sum(axis=0)
    out = {"n_runs": int(total[cols["n"]]), "n_landscapes": int(len(per))}
    for name, (num, den) in {"claim_rate": ("claim", "n"), "false_claim_rate": ("false", "n"),
                             "false_discovery_share": ("false", "claim"), "power": ("hit", "better"),
                             "size": ("false", "not_better"), "truly_better_share": ("better", "n")}.items():
        out[name] = float(total[cols[num]] / total[cols[den]]) if total[cols[den]] > 0 else np.nan
        ok = boot[:, cols[den]] > 0
        ratio = boot[ok, cols[num]] / boot[ok, cols[den]]
        out[f"{name}_lo"] = float(np.percentile(ratio, 2.5)) if ok.sum() >= 100 else np.nan
        out[f"{name}_hi"] = float(np.percentile(ratio, 97.5)) if ok.sum() >= 100 else np.nan
    out["clean_claim_rate"] = float(eos._as_bool(block["claim_clean"]).mean())
    if "shipped_d_noisy" in block and block["shipped_d_noisy"].notna().all():
        out["shipped_d_share"] = float(eos._as_bool(block["shipped_d_noisy"]).mean())
        out["d_is_c_share"] = float(eos._as_bool(block["d_is_c_noisy"]).mean())
    return out


def _confirmation_block() -> pd.DataFrame:
    # (truly better, test claims, fresh mean higher, D is C), the same five runs on two landscapes
    runs = [(True, True, True, False),    # a hit under every definition
            (False, False, True, False),  # a false fresh-mean claim
            (False, False, True, True),   # D is C: the fresh means differ by noise alone, a false claim
            (False, True, True, True),    # D is C and the test claims too
            (True, False, False, False)]  # a miss
    rows = [{"dataset": ds, "truly_better_noisy": b, "claim_noisy": c, "fresh_mean_claim_noisy": f,
             "shipped_d_noisy": f, "d_is_c_noisy": dc, "claim_clean": True}
            for ds in ("a", "b") for b, c, f, dc in runs]
    return pd.DataFrame(rows)


def test_claims_without_the_test_and_without_the_runs_whose_choice_is_the_start():
    block = _confirmation_block()
    out = eos.claim_rates(block, np.random.default_rng(0))
    assert out["fresh_mean_claim_rate"] == pytest.approx(4 / 5)
    assert out["fresh_mean_false_claim_rate"] == pytest.approx(3 / 5)
    assert out["fresh_mean_excl_dc_claim_rate"] == pytest.approx(2 / 5)
    assert out["fresh_mean_excl_dc_false_claim_rate"] == pytest.approx(1 / 5)
    assert out["fresh_mean_power"] == out["fresh_mean_excl_dc_power"] == pytest.approx(1 / 2)
    assert out["false_claim_rate"] == pytest.approx(1 / 5) and out["false_claim_rate_excl_dc"] == 0.0
    assert out["claim_rate_excl_dc"] == pytest.approx(1 / 5) and out["power_excl_dc"] == out["power"] == 0.5
    # the columns that existed before, and their intervals, do not move
    old = _old_claim_rates(block, np.random.default_rng(0))
    assert {k: out[k] for k in old} == old
    assert list(out)[:len(old)] == list(old)
    # a per-run file without the fresh-mean column: in a noisy run the ship decision is the same event
    legacy = eos.claim_rates(block.drop(columns="fresh_mean_claim_noisy"), np.random.default_rng(0))
    assert legacy == out
    mixed = block.astype({"fresh_mean_claim_noisy": object})
    mixed.loc[0, "fresh_mean_claim_noisy"] = None
    with pytest.raises(ValueError, match="mixes"):
        eos.claim_rates(mixed, np.random.default_rng(0))
    # the standard process has no D, so no extra columns
    std = block.drop(columns=["fresh_mean_claim_noisy", "shipped_d_noisy", "d_is_c_noisy"])
    assert eos.claim_rates(std, np.random.default_rng(0)) == _old_claim_rates(std, np.random.default_rng(0))


def test_the_fresh_mean_claim_is_strict_and_the_ship_rule_is_not():
    tie = eos.confirmation_decision(1.0, 1.0, np.array([1.0, 1.0]), np.array([1.0, 1.0]), 0.05)
    assert tie["shipped_d"] and not tie["fresh_mean_claim"] and not tie["claim"]
    up = eos.confirmation_decision(1.0, 0.0, np.array([1.2, 0.9]), np.array([0.1, 0.3]), 0.05)
    assert up["fresh_mean_claim"] and up["shipped_d"]
