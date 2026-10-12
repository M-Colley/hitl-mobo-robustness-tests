"""The best-arm (LUCB) sitting of scripts/replay_hitl_remedies.py.

Before 2026-10-09 an unseen candidate's upper bound was -inf + inf = NaN, which
np.argmax picks first, so odd looks always went to the next unseen candidate and
even looks to the leader: every run looked at exactly k / 2 + 1 designs and the
bound was never used. These tests pin the corrected rule: every bound is finite,
a seen challenger can be looked at again, the GP prior decides whether to explore,
the allocation is invariant to error shared by the sitting, and a look the
one-look-each sitting also makes gets the same noise draw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import replay_end_of_study as eos  # noqa: E402
import replay_hitl_remedies as rhr  # noqa: E402


def _random_prior(rng: np.random.Generator, m: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    a = rng.normal(size=(m, m)) * scale
    return rng.normal(size=m) * scale, a @ a.T / m + 1e-3 * scale ** 2 * np.eye(m)


def _old_allocation(truth: np.ndarray, look_sd: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """The committed rule before the fix, kept here to show what changed."""
    m, k = len(truth), len(look_sd)
    sums, counts = np.zeros(m), np.zeros(m, dtype=int)
    for step in range(k):
        if step < 2 and m >= 2:
            j = step
        else:
            seen = counts > 0
            means = np.where(seen, sums / np.maximum(counts, 1), -np.inf)
            leader = int(np.argmax(means))
            with np.errstate(invalid="ignore"):
                upper = means + np.where(seen, 1.0 / np.sqrt(np.maximum(counts, 1)), np.inf)
            upper[leader] = -np.inf
            challenger = int(np.argmax(upper))
            j = leader if step % 2 == 0 else challenger
        sums[j] += truth[j] + look_sd[step] * draws[step]
        counts[j] += 1
    return counts


# --- the anchored posterior -------------------------------------------------------------


def test_anchored_posterior_matches_conditioning_with_a_vague_offset():
    rng = np.random.default_rng(0)
    m = 6
    mu, cov = _random_prior(rng, m)
    noise_var = 0.3
    counts = np.array([2, 1, 0, 3, 0, 0])
    sums = rng.normal(size=m) * counts
    mean, sd = rhr.anchored_posterior(mu, cov, noise_var, counts, sums)
    # Brute force: g = f + d with d ~ N(0, V), V large, observed through the mean look of each seen candidate.
    V = 1e7
    seen = np.flatnonzero(counts)
    prior = cov + V
    A = prior[np.ix_(seen, seen)] + np.diag(noise_var / counts[seen])
    y = sums[seen] / counts[seen]
    gain = np.linalg.solve(A, prior[seen, :]).T
    np.testing.assert_allclose(mean, mu + gain @ (y - mu[seen]), atol=1e-6)
    np.testing.assert_allclose(sd, np.sqrt(np.diag(prior - gain @ prior[seen, :])), atol=1e-6)


def test_one_seen_candidate_anchors_the_rest_by_the_gp_difference():
    rng = np.random.default_rng(1)
    mu, cov = _random_prior(rng, 4)
    counts, sums = np.array([0, 2, 0, 0]), np.array([0.0, 3.0, 0.0, 0.0])
    mean, sd = rhr.anchored_posterior(mu, cov, 0.5, counts, sums)
    np.testing.assert_allclose(mean, 1.5 + mu - mu[1], atol=1e-10)
    var = np.diag(cov) + cov[1, 1] - 2 * cov[:, 1] + 0.5 / 2
    np.testing.assert_allclose(sd, np.sqrt(var), atol=1e-10)
    with pytest.raises(ValueError, match="at least one look"):
        rhr.anchored_posterior(mu, cov, 0.5, np.zeros(4), np.zeros(4))


# --- the allocation ---------------------------------------------------------------------------


def test_every_bound_is_finite_and_no_nan_is_ever_chosen():
    rng = np.random.default_rng(2)
    for trial in range(200):
        m, k = int(rng.integers(2, 13)), int(rng.integers(1, 9))
        mu, cov = _random_prior(rng, m, scale=float(rng.choice([1e-3, 1.0, 50.0])))
        if trial % 5 == 0:
            cov[1] = cov[0]           # two indistinguishable candidates: a singular prior
            cov[:, 1] = cov[:, 0]
        noise_var = float(rng.choice([0.0, 1e-12, 0.5, 400.0]))
        out = rhr.lucb_allocation(rng.normal(size=m), mu, cov, noise_var, np.abs(rng.normal(size=k)),
                                  rng.normal(size=k), rng.normal(size=(m, k)))
        assert out["counts"].sum() == k and 0 <= out["pick"] < m and out["counts"][out["pick"]] > 0
        assert all(0 <= j < m for j in out["order"])


def test_a_bound_that_is_not_finite_is_an_error_not_a_choice(monkeypatch):
    rng = np.random.default_rng(3)
    mu, cov = _random_prior(rng, 5)
    monkeypatch.setattr(rhr, "anchored_posterior", lambda *a, **k: (np.full(5, np.nan), np.full(5, np.nan)))
    with pytest.raises(FloatingPointError):
        rhr.lucb_allocation(np.zeros(5), mu, cov, 1.0, np.ones(4), np.zeros(4), np.zeros((5, 4)))


def _two_close_and_the_rest_far_below(m: int = 6):
    truth = np.array([0.0, -0.01] + [-5.0] * (m - 2))
    cov = np.eye(m) * 1e-4          # the GP is sure of every candidate
    return truth, truth.copy(), cov


def test_a_seen_challenger_is_looked_at_again():
    truth, mu, cov = _two_close_and_the_rest_far_below()
    k = 4
    out = rhr.lucb_allocation(truth, mu, cov, 1.0, np.ones(k), np.zeros(k), np.zeros((6, k)))
    # look 1 the top candidate; then the close second (unseen, then seen with as many looks as the
    # leader, so its bound is no narrower); then the leader, whose bound is now the wider one
    assert out["order"] == [0, 1, 1, 0]
    np.testing.assert_array_equal(out["counts"], [2, 2, 0, 0, 0, 0])
    # The committed rule looked at k / 2 + 1 designs whatever the data said.
    assert (_old_allocation(truth, np.ones(k), np.zeros(k)) > 0).sum() == 3


def test_the_old_rule_ignored_its_bound_and_the_new_one_uses_it():
    rng = np.random.default_rng(4)
    old_distinct, new_distinct = set(), set()
    for _ in range(300):
        m, k = 24, 8
        truth = np.sort(rng.normal(size=m))[::-1]
        mu, cov = truth + rng.normal(scale=0.3, size=m), np.eye(m) * float(rng.choice([1e-3, 0.1, 4.0]))
        draws = rng.normal(size=(m, k))
        old_distinct.add(int((_old_allocation(truth, np.ones(k), draws[0]) > 0).sum()))
        new = rhr.lucb_allocation(truth, mu, cov, 1.0, np.ones(k), draws[0], draws)
        new_distinct.add(int((new["counts"] > 0).sum()))
    assert old_distinct == {5}
    assert len(new_distinct) > 3 and min(new_distinct) < 5 < max(new_distinct)


def test_an_uncertain_candidate_outside_the_top_k_is_explored():
    m, k = 6, 3
    truth = np.array([0.0, -0.01, -0.02, -0.03, -0.04, -0.05])
    mu, cov = truth.copy(), np.eye(m) * 1e-4
    cov[5, 5] = 25.0                # the GP knows nothing about the sixth design
    out = rhr.lucb_allocation(truth, mu, cov, 0.01, np.ones(k) * 0.1, np.zeros(k), np.zeros((m, k)))
    assert out["order"][1] == 5 and out["counts"][3:].sum() >= 1


def test_the_allocation_ignores_error_shared_by_the_sitting():
    rng = np.random.default_rng(5)
    for _ in range(100):
        m, k = 12, 8
        mu, cov = _random_prior(rng, m)
        truth, draws, sd = rng.normal(size=m), rng.normal(size=(m, k)), np.abs(rng.normal(size=k))
        a = rhr.lucb_allocation(truth, mu, cov, 0.7, sd, draws[0], draws)
        b = rhr.lucb_allocation(truth + 3.25, mu, cov, 0.7, sd, draws[0], draws)
        assert a["order"] == b["order"] and a["pick"] == b["pick"]


def test_a_look_one_look_each_also_makes_gets_its_draw():
    rng = np.random.default_rng(6)
    m, k = 10, 4
    mu, cov = _random_prior(rng, m)
    z_first, extra = rng.normal(size=k), rng.normal(size=(m, k))
    out = rhr.lucb_allocation(np.zeros(m), mu, cov, 1.0, np.ones(k), z_first, extra)
    for j in np.flatnonzero(out["counts"] == 1):
        assert out["sums"][j] == pytest.approx(z_first[j] if j < k else extra[j, 0])


# --- the sitting on a run ------------------------------------------------------------------------


def _run_and_state(T: int = 30, k: int = 4, gp: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(T, 2))
    f = -((X - 0.6) ** 2).sum(axis=1)
    observed = f + 0.05 * rng.normal(size=T)
    inference = np.array([f.max() - f[int(np.argmax(observed[:t + 1]))] for t in range(T)])
    run = eos.RunLog(name="r.csv", X=X, observed=observed, deployed=f, logged_inference=inference,
                     y_opt=float(f.max()))
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    if gp:
        return run, eos.search_state(run, T - k, bounds, 1.0)
    n = T - k
    first, mean = eos.distinct_designs(X[:n], observed[:n])
    state = eos.SearchState(n=n, first=first, obs_mean=mean, lcb=mean.copy(), gp=None,
                            train_X=torch.tensor(X[:n]), train_Y=torch.tensor(observed[:n, None]),
                            bounds=bounds, gp_failed=True)
    return run, state


@pytest.mark.parametrize("gp", [True, False])
def test_lucb_sitting_on_a_run(gp):
    T, k = 30, 4
    run, state = _run_and_state(T, k, gp)
    noise = eos.SittingNoise(base=0.3)
    out = rhr.lucb_sitting(run, state, k, 1.0, noise, T)["lucb_k4_rho1"]
    assert out["family"] == "lucb" and out["m"] == 3 * k and out["gp_failed"] is (not gp)
    assert 1 <= out["n_distinct_looked"] <= k and out["n_relooks"] == k - out["n_distinct_looked"]
    assert np.isfinite(out["regret"]) and out["regret"] >= out["best_candidate_regret"]
    assert out["look_sd_assumed"] > 0
    assert rhr.lucb_sitting(run, state, k, 1.0, noise, T) == rhr.lucb_sitting(run, state, k, 1.0, noise, T)


def test_the_sittings_draws_are_the_tournaments_where_they_share_a_look():
    T, k = 30, 4
    run, state = _run_and_state(T, k)
    noise = eos.SittingNoise(base=0.3)
    z = np.random.default_rng(eos.noise_seed(run.name, eos.PROC_TOURNAMENT, k)).standard_normal(k)
    rows = state.first[np.argsort(-state.lcb, kind="stable")]
    # With rho = 0 the looks are exact; with the shared draws the first look at the top candidate is the
    # tournament's look at it, truth + rho * SD * z[0], in both procedures.
    tour = eos.tournament(run, state, k, (1.0,), noise, T)["tournament_k4_lcb_rho1_look"]
    assert tour["top_candidate_regret"] == pytest.approx(run.y_opt - run.deployed[rows[0]])
    truth = run.deployed[rows[:3 * k]]
    res = rhr.lucb_allocation(truth, *rhr.lucb_prior(run, state, np.argsort(-state.lcb, kind="stable")[:3 * k]),
                              0.3 * np.ones(k), z, np.zeros((3 * k, k)))
    # every later look draws zero here, so what is left of the top candidate's sum is its first look
    first_look = res["sums"][0] - (res["counts"][0] - 1) * truth[0]
    assert res["order"][0] == 0 and first_look == pytest.approx(truth[0] + 0.3 * z[0])


def test_the_recovery_table_refuses_a_landscape_without_a_scale():
    frame = pd.DataFrame({"dataset": ["a", "b"], "procedure": ["p", "p"], "error_model": ["gaussian"] * 2,
                          "jitter_std": [1.0, 1.0], "jitter_iteration": [0, 0], "regret_noisy": [1.0, 1.0],
                          "regret_clean": [0.5, 0.5], "ref_noisy": [2.0, 2.0], "ref_clean": [0.5, 0.5]})
    with pytest.raises(KeyError, match="no opt_z"):
        rhr.recovery_table(frame, {"a": 2.0}, reps=10)
    with pytest.raises(ValueError, match="non-positive"):
        rhr.recovery_table(frame, {"a": 2.0, "b": 0.0}, reps=10)
    table = rhr.recovery_table(frame, {"a": 2.0, "b": 1.0}, reps=10)
    # in opt_z units: cost 0.75 and 1.5, gain 0.5 and 1.0, a ratio of landscape means
    assert table.loc[table["error_model"] == "pooled", "recovered"].iloc[0] == pytest.approx(0.75 / 1.125)
