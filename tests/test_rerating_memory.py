"""Re-rating with memory (scripts/checks/rerating_memory.py).

What must hold: (1) the look error is rho_m * e_orig + sqrt(1 - rho_m^2) * e_fresh, exactly e_fresh at
rho_m = 0 and e_orig at rho_m = 1, and under Gaussian errors it keeps the SD and correlates with the
original error at rho_m; (2) the rho_m = 0 control: the sitting ships the published look winner and the
confirmation reproduces replay_end_of_study's decision bit for bit, on random arrays and on toy runs
through tournament() and confirmation() themselves, and on the cached inputs when they are present;
(3) a clean twin (exact ratings) is untouched at every rho_m; (4) a design rated more than once takes its
highest rating, ties to the earliest; (5) the confirmation control stops on any difference from the
published replay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import rerating_memory as rm  # noqa: E402
import sitting_order_effects as soe  # noqa: E402


# --- the error formula ------------------------------------------------------------------------------


def test_the_look_error_is_the_stated_mixture():
    e_orig = np.array([1.0, -2.0, 0.5])
    e_fresh = np.array([0.3, 0.4, -0.1])
    assert np.allclose(rm.memory_errors(e_orig, e_fresh, 0.5), 0.5 * e_orig + np.sqrt(0.75) * e_fresh)
    assert np.allclose(rm.memory_errors(e_orig, e_fresh, 0.25), 0.25 * e_orig + np.sqrt(1 - 0.0625) * e_fresh)
    assert np.array_equal(rm.memory_errors(e_orig, e_fresh, 1.0), e_orig)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="rho_m"):
            rm.memory_errors(e_orig, e_fresh, bad)
    with pytest.raises(ValueError, match="shape"):
        rm.memory_errors(e_orig[:2], e_fresh, 0.5)
    with pytest.raises(ValueError, match="finite"):
        rm.memory_errors(np.array([np.nan, 0.0, 0.0]), e_fresh, 0.5)


def test_rho_zero_returns_the_fresh_draws_bit_for_bit():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 17))
        e_orig = rng.normal(size=n) * 10 ** rng.uniform(-3, 2)
        e_fresh = rng.normal(size=n) * 10 ** rng.uniform(-3, 2)
        truth = rng.normal(size=n)
        assert np.array_equal(rm.memory_errors(e_orig, e_fresh, 0.0), e_fresh)
        # the published look is truth + e_fresh; the memory look at rho_m = 0 is the same float
        assert np.array_equal(truth + rm.memory_errors(e_orig, e_fresh, 0.0), truth + e_fresh)


def test_gaussian_memory_keeps_the_sd_and_correlates_at_rho_m():
    rng = np.random.default_rng(2)
    s, n = 2.5, 400_000
    e_orig, e_fresh = rng.normal(0, s, n), rng.normal(0, s, n)
    for rho in rm.MEMORY:
        e = rm.memory_errors(e_orig, e_fresh, rho)
        assert abs(e.std() / s - 1.0) < 0.01
        assert abs(np.corrcoef(e, e_orig)[0, 1] - rho) < 0.01


# --- the sitting ------------------------------------------------------------------------------------


def test_the_sitting_at_rho_zero_ships_the_published_look_winner():
    rng = np.random.default_rng(3)
    for _ in range(500):
        m = int(rng.integers(1, 17))
        truth, error, e_orig = rng.normal(size=m), rng.normal(size=m), rng.normal(size=m) * 3
        pub = soe.published_pick(truth, error)
        assert rm.sitting_pick(truth, error, e_orig, 0.0) == pub
    # ties go to the earliest candidate, as the published argmax does
    assert rm.sitting_pick(np.array([1.0, 1.0]), np.zeros(2), np.zeros(2), 0.5) == 0


def test_memory_carries_the_original_error_into_the_look():
    truth = np.array([1.0, 1.1])
    fresh = np.zeros(2)
    e_orig = np.array([0.5, 0.0])
    assert rm.sitting_pick(truth, fresh, e_orig, 0.0) == 1          # the better design
    assert rm.sitting_pick(truth, fresh, e_orig, 0.25) == 0         # 1 + 0.125 > 1.1: the remembered error wins
    # a design first rated before the onset (exact rating) keeps only sqrt(1 - rho^2) of its fresh error
    e = rm.memory_errors(np.array([0.0]), np.array([2.0]), 0.75)
    assert np.isclose(e[0], 2.0 * np.sqrt(1 - 0.75 ** 2))


def test_a_clean_twin_is_untouched_at_every_rho():
    rng = np.random.default_rng(4)
    for _ in range(200):
        m = int(rng.integers(1, 17))
        truth = rng.normal(size=m)
        error = -0.0 * np.abs(rng.normal(size=m))     # the cached clean looks are signed zeros
        for rho in rm.MEMORY:
            assert rm.sitting_pick(truth, error, np.zeros(m), rho) == int(np.argmax(truth))
            assert np.array_equal(rm.confirmation_ratings(0.7, np.zeros(2), 0.0, rho), np.full(2, 0.7))


# --- the original rating ------------------------------------------------------------------------------


def test_a_design_rated_more_than_once_takes_its_highest_rating_ties_to_the_earliest():
    keys = ["a", "b", "c", "b", "b", "b"]
    observed = np.array([1.0, 5.0, 2.0, 5.0, 3.0, 9.0])
    assert rm.rating_row(keys, observed, 1, 5) == (1, 1, 3)        # 5 at rows 1 and 3: the earliest
    assert rm.rating_row(keys, observed, 3, 5) == (1, 1, 3)        # any row of the design names it
    assert rm.rating_row(keys, observed, 1, 6) == (5, 1, 4)        # row 5 counts only inside the prefix
    assert rm.rating_row(keys, observed, 2, 5) == (2, 2, 1)
    with pytest.raises(ValueError, match="first"):
        rm.rating_row(keys, observed, 5, 5)


# --- toy runs through replay_end_of_study itself (imports the simulator) ------------------------------


def _toy(name: str, n: int = 50, repeat: bool = False, seed: int = 0):
    import torch

    import replay_end_of_study as eos

    rng = np.random.default_rng(seed)
    truth = np.linspace(1.0, 0.0, n) + 0.01 * rng.normal(size=n)
    X = np.arange(n, dtype=float).reshape(-1, 1)
    if repeat:
        X[30] = X[3]               # design 3 rated again at trial 31
        truth[30] = truth[3]
    err = rng.normal(size=n)
    observed = truth + err
    run = eos.RunLog(name=name, X=X, observed=observed, deployed=truth.copy(), logged_inference=np.zeros(n),
                     y_opt=2.0, error=err)
    return run, eos, torch


def _state(run, eos, torch, n: int, rng: np.random.Generator):
    first, mean = eos.distinct_designs(run.X[:n], run.observed[:n])
    lcb = np.empty(len(first))
    lcb[rng.permutation(len(first))] = -np.arange(len(first), dtype=float)    # a random ranking of the designs
    return eos.SearchState(n=n, first=first, obs_mean=mean, lcb=lcb, gp=None, train_X=torch.zeros(n, 1),
                           train_Y=torch.zeros(n, 1), bounds=torch.tensor([[0.0], [float(n)]]), gp_failed=True)


def test_the_toy_sitting_reproduces_tournament_and_reads_the_original_errors():
    T = 50
    for k, model in ((2, "gaussian"), (16, "bias"), (16, "ar1")):
        run, eos, torch = _toy(f"toy_sit_{model}_{k}.csv", repeat=True, seed=k)
        arm = eos.ArmInfo("a", Path("a"), T, 5, 0.8, input_arm=False)
        n = T - k
        state = _state(run, eos, torch, n, np.random.default_rng(10 + k))
        state.lcb[int(np.flatnonzero(state.first == 3)[0])] = state.lcb.max() + 1.0   # the re-rated design leads
        sd_fn = eos.sitting_sd_fn(arm, model, 1.0, 0, eos.Settings())
        rec = soe.sitting_inputs(run, state, k, sd_fn, T)
        pub = eos.tournament(run, state, k, (1.0,), sd_fn, T)[soe.procedure_name(k)]
        truth, error = soe.parse_vec(rec["truth"]), soe.parse_vec(rec["error"])
        rows = state.first[np.argsort(-state.lcb, kind="stable")[:k]]
        orig = rm.original_errors(run, rows, n)
        assert np.array_equal(orig["rows_first"], rows)
        c = rm.sitting_pick(truth, error, orig["e_orig"], 0.0)
        assert run.y_opt - truth[c] == pub["regret"]
        assert rows[0] == 3
        for j, r in enumerate(rows):
            if r == 3:     # rated at rows 3 and 30: the higher rating is the original one
                best = 3 if run.observed[3] >= run.observed[30] else 30
                assert orig["rows_rating"][j] == best and orig["n_ratings"][j] == 2
                assert orig["e_orig"][j] == run.observed[best] - run.deployed[best]
            else:
                assert orig["rows_rating"][j] == r and orig["n_ratings"][j] == 1
                assert orig["e_orig"][j] == run.observed[r] - run.deployed[r]


def test_the_toy_confirmation_reproduces_confirmation_and_memory_moves_only_noisy_ratings():
    T, k = 50, 2
    for i in range(12):
        run, eos, torch = _toy(f"toy_conf_{i}.csv", repeat=bool(i % 2), seed=100 + i)
        arm = eos.ArmInfo("a", Path("a"), T, 5, 0.8, input_arm=False)
        n = T - 2 * k
        state = _state(run, eos, torch, n, np.random.default_rng(i))
        sd_fn = eos.sitting_sd_fn(arm, "gaussian", 1.0, 0, eos.Settings())
        rec = rm.confirmation_inputs(run, state, k, arm.initial_samples, sd_fn, T, 0.05)   # raises on a mismatch
        pub = eos.confirmation(run, state, k, arm.initial_samples, sd_fn, T, 0.05)
        cache = pd.DataFrame([{"file": run.name, **rec}])
        dec0 = rm.confirm_decisions(cache, 0.0).iloc[0]
        assert dec0["p_value"] == pub["p_value"] and dec0["claim"] == pub["claim"]
        assert dec0["regret"] == pub["regret"] and dec0["shipped_d"] == pub["shipped_d"]
        for who in ("d", "c"):
            r = rec[f"{who}_rating_row"]
            assert rec[f"e_{who}"] == run.observed[r] - run.deployed[r]
            assert run.observed[r] == max(run.observed[q] for q in range(n)
                                          if run.X[q, 0] == run.X[rec[f"{who}_row"], 0])
        # the clean twin's ratings are exact: memory leaves its decision where it was
        clean = cache.assign(e_d=0.0, e_c=0.0, err_d=rm._vec(np.zeros(k)), err_c=rm._vec(np.zeros(k)))
        base = rm.confirm_decisions(clean, 0.0).iloc[0]
        for rho in rm.MEMORY[1:]:
            moved = rm.confirm_decisions(clean, rho).iloc[0]
            assert (moved["claim"], moved["regret"], moved["p_value"]) == (base["claim"], base["regret"],
                                                                         base["p_value"])


def test_memory_cannot_move_a_claim_when_d_is_c():
    """D = C: equal original errors cancel, and the t statistic does not change with a common scale."""
    import replay_end_of_study as eos

    rng = np.random.default_rng(5)
    for _ in range(200):
        f, e, err_d, err_c = 0.3, float(rng.normal()), rng.normal(size=2), rng.normal(size=2)
        base = eos.confirmation_decision(f, f, rm.confirmation_ratings(f, err_d, e, 0.0),
                                         rm.confirmation_ratings(f, err_c, e, 0.0), 0.05)
        for rho in rm.MEMORY[1:]:
            dec = eos.confirmation_decision(f, f, rm.confirmation_ratings(f, err_d, e, rho),
                                            rm.confirmation_ratings(f, err_c, e, rho), 0.05)
            assert dec["claim"] == base["claim"] and np.isclose(dec["p_value"], base["p_value"], rtol=1e-9)


# --- the confirmation control ----------------------------------------------------------------------------


def _control_frames():
    cache = pd.DataFrame({"file": ["a.csv", "b.csv"], "regret": [0.1, 0.2], "p_value": [0.01, 0.4],
                          "claim": [True, False], "shipped_d": [True, False], "truly_better": [True, False],
                          "d_is_c": [False, False]})
    pub = pd.DataFrame({"file": ["a.csv", "b.csv"], "regret_noisy": [0.1, 0.2], "p_value_noisy": [0.01, 0.4],
                        "claim_noisy": [True, False], "shipped_d_noisy": [True, False],
                        "truly_better_noisy": [True, False], "d_is_c_noisy": [False, False],
                        "false_claim_noisy": [False, False]})
    return cache, pub


def test_the_confirmation_control_stops_on_any_difference():
    cache, pub = _control_frames()
    assert rm.check_confirm_control(cache, pub)["n_runs"] == 2
    off = pub.copy()
    off.loc[0, "p_value_noisy"] = np.nextafter(0.01, 1.0)
    with pytest.raises(rm.ControlError, match="p_value"):
        rm.check_confirm_control(cache, off)
    off = pub.copy()
    off.loc[1, "claim_noisy"] = True
    with pytest.raises(rm.ControlError, match="claim"):
        rm.check_confirm_control(cache, off)
    with pytest.raises(rm.ControlError, match="no published row"):
        rm.check_confirm_control(cache, pub.iloc[:1])


# --- the cached inputs (when the replay has run) -----------------------------------------------------------


@pytest.mark.skipif(not (rm.SITTING_INPUTS.is_file() and soe.INPUTS.is_file()), reason="no cached sitting inputs")
def test_the_cached_sitting_at_rho_zero_is_the_published_sitting_and_clean_twins_never_move():
    inputs = soe.load_inputs(soe.INPUTS)
    reg = rm.sitting_regrets(inputs, rm.load_sitting_memory())
    m = inputs[["file", "k", "published_regret", "baseline"]].merge(reg, on=["file", "k"], validate="one_to_one")
    assert np.array_equal(m["reg_mem0"].to_numpy(), m["published_regret"].to_numpy())
    clean = m[m["baseline"]]
    for rho in rm.MEMORY:
        assert np.array_equal(clean[f"reg_{rm.mem_label(rho)}"].to_numpy(), clean["published_regret"].to_numpy())
    assert len(m) == 4000 and int((~m["baseline"]).sum()) == 3200


@pytest.mark.skipif(not rm.CONFIRM_INPUTS.is_file(), reason="no cached confirmation inputs")
def test_the_cached_confirmation_at_rho_zero_reproduces_the_replays_decisions():
    cache = rm.load_confirm_memory()
    dec = rm.confirm_decisions(cache, 0.0)
    for c in ("regret", "p_value", "claim", "shipped_d", "truly_better", "fresh_mean_claim"):
        assert np.array_equal(dec[c].to_numpy(), cache[c].to_numpy()), c
    assert len(cache) == 4800
