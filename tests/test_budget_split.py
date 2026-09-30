"""The rule that decides how many trials to spend identifying rather than searching.

The tests pin the two forces the identification term is built to trade off --
coverage (a larger candidate set can only contain a better design) against
discrimination (a larger candidate set gives a worse design more chances to win
the sitting on a lucky look) -- pin the search term's estimator, and pin that
the rule reads nothing but the log and the surrogate: the true objective enters
only when the result is scored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import budget_split as bs  # noqa: E402
import replay_end_of_study as eos  # noqa: E402


# ---------------------------------------------------------------------------
# The identification term
# ---------------------------------------------------------------------------


def test_a_noiseless_sitting_ships_the_best_candidate():
    rng = np.random.default_rng(0)
    mu, sd = np.array([1.0, 0.5, 0.0]), np.full(3, 1e-9)
    assert bs.expected_shipped_value(mu, sd, 1e-9, rng) == pytest.approx(1.0, abs=1e-6)


def test_an_uninformative_sitting_ships_at_random():
    """With look noise swamping every gap the winner is uniform over candidates."""
    rng = np.random.default_rng(0)
    mu, sd = np.array([1.0, 0.0]), np.full(2, 1e-9)
    assert bs.expected_shipped_value(mu, sd, 1e6, rng) == pytest.approx(0.5, abs=0.02)


def test_coverage_a_noiseless_sitting_never_loses_by_adding_a_candidate():
    rng = np.random.default_rng(1)
    mu = np.array([0.0, 1.0])          # the SECOND candidate is the better one
    sd = np.full(2, 1e-9)
    one = bs.expected_shipped_value(mu[:1], sd[:1], 1e-9, rng)
    two = bs.expected_shipped_value(mu, sd, 1e-9, rng)
    assert two > one


def test_discrimination_a_noisy_sitting_loses_by_adding_a_worse_candidate():
    """The force that stops k from growing without bound.

    The padding has to sit within reach of the look noise: a candidate five SDs
    below the leader never wins a sitting, so adding it changes nothing.
    """
    rng = np.random.default_rng(2)
    good = bs.expected_shipped_value(np.array([1.0, 0.9]), np.full(2, 1e-9), 1.0, rng)
    padded = bs.expected_shipped_value(np.array([1.0, 0.9, 0.2, 0.2]), np.full(4, 1e-9), 1.0, rng)
    assert padded < good


def test_a_single_candidate_needs_no_sitting():
    rng = np.random.default_rng(0)
    assert bs.expected_shipped_value(np.array([0.7]), np.array([0.1]), 1.0, rng) == pytest.approx(0.7)


def test_posterior_uncertainty_is_carried_not_ignored():
    """The candidates' latent values are drawn from the posterior, so a wide
    posterior makes the best MEAN less certain to be the best VALUE."""
    rng = np.random.default_rng(3)
    tight = bs.expected_shipped_value(np.array([1.0, 0.9]), np.full(2, 1e-9), 1e-9, rng)
    wide = bs.expected_shipped_value(np.array([1.0, 0.9]), np.full(2, 3.0), 1e-9, rng)
    assert wide > tight  # a wide posterior sometimes offers something better than mu_1


# ---------------------------------------------------------------------------
# The search term
# ---------------------------------------------------------------------------


def test_the_rate_is_the_recent_gain_per_trial():
    assert bs.recent_improvement_rate(np.array([0.0, 1.0, 2.0, 3.0]), 4, 10) == pytest.approx(1.0)


def test_the_rate_only_looks_at_the_window():
    # A big early gain, then flat: a short window must not see the early gain.
    obs = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    assert bs.recent_improvement_rate(obs, 6, window=3) == pytest.approx(0.0)
    assert bs.recent_improvement_rate(obs, 6, window=10) > 0.0


def test_the_rate_is_floored_at_zero():
    """A run whose best rating fell back (a noisy rating can do that) forgoes
    nothing by stopping, so the search term must never be negative."""
    assert bs.recent_improvement_rate(np.array([5.0, 4.0, 3.0]), 3, 10) == 0.0


def test_the_rate_of_a_run_with_no_history_is_zero():
    assert bs.recent_improvement_rate(np.array([1.0]), 1, 10) == 0.0


# ---------------------------------------------------------------------------
# The rule end to end
# ---------------------------------------------------------------------------


def _run(observed: np.ndarray, seed: int = 0) -> "eos.RunLog":
    rng = np.random.default_rng(seed)
    T = len(observed)
    X = rng.uniform(0.0, 1.0, size=(T, 2))
    return eos.RunLog(name="test", X=X, observed=observed, deployed=observed.copy(),
                      logged_inference=np.maximum.accumulate(observed), y_opt=float(observed.max()) + 1.0)


BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
GRID = (2, 3, 5, 8)


def test_a_search_still_improving_fast_keeps_its_trials():
    """When every extra trial is still worth a lot, the rule spends none on a sitting."""
    observed = np.linspace(0.0, 40.0, 40)          # a steep, unbroken climb
    out = bs.derive_k(_run(observed), BOUNDS, GRID, beta=1.0, window=10,
                      sitting_sd=0.05, rng=np.random.default_rng(0), T=40)
    assert out["k_hat"] in bs.NO_TRIAL_POLICIES


def test_a_plateaued_search_under_a_precise_sitting_buys_identification():
    """Nothing left to find and a sitting that can tell candidates apart: spend."""
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    observed += np.random.default_rng(4).normal(0.0, 0.05, size=40)
    out = bs.derive_k(_run(observed), BOUNDS, GRID, beta=1.0, window=10,
                      sitting_sd=1e-6, rng=np.random.default_rng(0), T=40)
    assert out["k_hat"] in GRID


def test_a_sitting_too_noisy_to_discriminate_is_not_bought():
    """Same plateaued run, but a sitting that cannot tell anything apart.

    A useless sitting ships a uniformly random candidate, whose expected value
    cannot beat the best posterior mean already in hand. Comparing only against
    the lcb pick made this case buy a tournament whenever lcb happened to rank
    below the posterior mean -- a reason to change the ship rule, not to spend
    trials -- which is why the rule weighs both no-trial policies.
    """
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    observed += np.random.default_rng(4).normal(0.0, 0.05, size=40)
    out = bs.derive_k(_run(observed), BOUNDS, GRID, beta=1.0, window=10,
                      sitting_sd=1e6, rng=np.random.default_rng(0), T=40)
    assert out["k_hat"] in bs.NO_TRIAL_POLICIES


def test_every_k_on_the_grid_is_scored():
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    out = bs.derive_k(_run(observed), BOUNDS, GRID, beta=1.0, window=10,
                      sitting_sd=0.1, rng=np.random.default_rng(0), T=40)
    assert set(out["scores"]) == set(bs.NO_TRIAL_POLICIES) | set(GRID)


def test_the_rule_never_reads_the_true_objective():
    """Two runs with identical ratings but different truths must choose the same k.

    This is the property that makes the rule usable: an experimenter has the
    ratings and the surrogate, never f.
    """
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    a, b = _run(observed), _run(observed)
    b.deployed = b.deployed * -3.0 + 7.0
    b.y_opt = 1e6
    ka = bs.derive_k(a, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)
    kb = bs.derive_k(b, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)
    assert ka["k_hat"] == kb["k_hat"]


# ---------------------------------------------------------------------------
# When the rule decides
# ---------------------------------------------------------------------------


def _plateau(seed: int = 4) -> np.ndarray:
    observed = np.concatenate([np.linspace(0.0, 5.0, 10), np.full(30, 5.0)])
    return observed + np.random.default_rng(seed).normal(0.0, 0.05, size=40)


def test_the_fixed_rule_is_blind_to_every_rating_after_its_decision_point():
    """n0 = T - max(k): with the published grid (max 30 of 50) that is trial 20, so
    at the late onset (error on trials 21 on) the rule sees no error at all. Here
    the ratings after n0 = 32 are replaced by garbage and nothing changes."""
    a = _run(_plateau())
    b = _run(_plateau())
    b.observed[32:] = 1e3 * np.random.default_rng(9).standard_normal(8)
    ka = bs.derive_k(a, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)
    kb = bs.derive_k(b, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)
    assert ka["k_hat"] == kb["k_hat"] and ka["scores"] == kb["scores"]


def test_a_precomputed_state_gives_the_same_decision_as_a_fresh_fit():
    run = _run(_plateau())
    state = eos.search_state(run, 40 - max(GRID), BOUNDS, 1.0)
    fresh = bs.derive_k(run, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)
    given = bs.derive_k(run, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40, state=state)
    assert fresh["k_hat"] == given["k_hat"] and fresh["scores"] == given["scores"]


def test_a_state_from_the_wrong_trial_is_refused():
    run = _run(_plateau())
    state = eos.search_state(run, 30, BOUNDS, 1.0)
    with pytest.raises(ValueError):
        bs.derive_k(run, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40, state=state)


def test_decision_points_run_from_the_largest_sitting_down():
    assert bs.decision_points((2, 3, 5, 8, 12, 16, 20, 25, 30), 50) == [
        (30, 20), (25, 25), (20, 30), (16, 34), (12, 38), (8, 42), (5, 45), (3, 47), (2, 48)]
    # A sitting that would start before two trials exist has no decision point.
    assert bs.decision_points((2, 9), 10) == [(2, 8)]


def _sequential(run, grid=GRID, sitting_sd=0.1, T=40):
    return bs.derive_k_sequential(run, BOUNDS, grid, 1.0, 10, np.random.default_rng(0), T,
                                  lambda state: sitting_sd)


def test_the_sequential_rule_commits_only_to_the_sitting_that_expires_or_ships_at_T():
    for observed, sd in ((_plateau(), 1e-6), (_plateau(), 1e6), (np.linspace(0.0, 40.0, 40), 0.05)):
        out = _sequential(_run(observed), sitting_sd=sd)
        assert not out["gp_failed"]
        if out["k_hat"] in bs.NO_TRIAL_POLICIES:
            assert out["decided_at"] == 40 - min(GRID)
        else:
            assert out["decided_at"] == 40 - out["k_hat"]


def test_the_sequential_rule_never_reads_past_where_it_decided():
    run = _run(_plateau())
    out = _sequential(run, sitting_sd=1e-6)
    garbled = _run(_plateau())
    n = out["decided_at"]
    garbled.observed[n:] = 1e3 * np.random.default_rng(9).standard_normal(40 - n)
    again = _sequential(garbled, sitting_sd=1e-6)
    assert (again["k_hat"], again["decided_at"], again["scores"]) == (out["k_hat"], out["decided_at"], out["scores"])


def test_the_sequential_rule_with_one_sitting_size_is_the_fixed_rule():
    run = _run(_plateau())
    fixed = bs.derive_k(run, BOUNDS, (5,), 1.0, 10, 1e-6, np.random.default_rng(0), 40)
    seq = _sequential(run, grid=(5,), sitting_sd=1e-6)
    assert seq["k_hat"] == fixed["k_hat"] and seq["scores"] == fixed["scores"]


def test_the_sequential_rule_agrees_with_the_fixed_rule_when_the_largest_sitting_wins_at_once():
    """The first decision point IS the fixed rule's n0: if the fixed rule picks the
    largest k there, the sequential rule commits to it there, on the same scores."""
    for seed in range(6):
        run = _run(_plateau(seed), seed)
        fixed = bs.derive_k(run, BOUNDS, GRID, 1.0, 10, 1e-6, np.random.default_rng(0), 40)
        seq = _sequential(run, sitting_sd=1e-6)
        if fixed["k_hat"] == max(GRID):
            assert seq["k_hat"] == max(GRID) and seq["decided_at"] == 40 - max(GRID)
            assert seq["scores"] == fixed["scores"]
        assert seq["path"].split(";")[0] == f"{max(GRID)}:{fixed['k_hat']}"


def test_a_failed_fit_at_one_decision_point_is_skipped_not_fatal(monkeypatch):
    real = eos.search_state

    def flaky(run, n, bounds, beta):
        state = real(run, n, bounds, beta)
        if n == 40 - max(GRID):
            state.gp = None
        return state

    monkeypatch.setattr(eos, "search_state", flaky)
    out = _sequential(_run(_plateau()), sitting_sd=0.1)
    assert not out["gp_failed"] and out["n_fit_failures"] == 1
    assert out["path"].startswith(f"{max(GRID)}:fail")


def test_the_sequential_rule_fails_only_when_every_fit_fails(monkeypatch):
    real = eos.search_state

    def broken(run, n, bounds, beta):
        state = real(run, n, bounds, beta)
        state.gp = None
        return state

    monkeypatch.setattr(eos, "search_state", broken)
    out = _sequential(_run(_plateau()))
    assert out["gp_failed"] and out["k_hat"] is None and out["n_fit_failures"] == len(GRID)


def test_derive_for_run_passes_the_decision_through(tmp_path):
    """The task dict carries the decision; the sequential record says where it decided,
    and the fixed record keeps exactly the published columns."""
    import pandas as pd
    observed = _plateau()
    run = _run(observed)
    path = tmp_path / "run.csv"
    T = len(observed)
    pd.DataFrame({"iteration": np.arange(1, T + 1), "param_columns": "x0,x1",
                  "x0": run.X[:, 0], "x1": run.X[:, 1], "objective_observed": observed,
                  "objective_true": observed, "inference_simple_regret_true": 0.0,
                  "y_opt": run.y_opt}).to_csv(path, index=False)
    task = {"file": "run.csv", "path": str(path), "iterations": T, "bounds_low": np.zeros(2),
            "bounds_high": np.ones(2), "k_grid": GRID, "beta": 1.0, "window": 10, "rho": 1.0,
            "noise_source": "truth", "true_sitting_sd": 0.1}
    fixed = bs.derive_for_run({**task, "decision": "fixed"})
    seq = bs.derive_for_run({**task, "decision": "sequential"})
    assert set(fixed) == {"file", "k_hat", "gp_failed", "sitting_sd_used", "scores"}
    assert {"decided_at", "path", "n_fit_failures"} <= set(seq)
    assert seq["decided_at"] in {T - k for k in GRID}


def test_the_clairvoyant_rate_reads_the_true_values_and_nothing_else():
    """--rate-source truth swaps only the series the search rate is read from."""
    climbing = np.linspace(0.0, 5.0, 40)
    run = _run(climbing.copy())
    run.deployed = np.full(40, 2.0)                  # the truth never improved
    rating = bs.derive_k(run, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)
    truth = bs.derive_k(run, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40, rate_source="truth")
    assert all(d["extra_search"] > 0 for k, d in rating["detail"].items() if k != max(GRID))
    assert all(d["extra_search"] == 0 for d in truth["detail"].values())
    # The identification term is untouched: same posterior, same Monte Carlo stream.
    for k in GRID:
        assert truth["detail"][k]["utility"] == rating["detail"][k]["utility"]
    # Where the truth moved exactly as the ratings did, the two agree.
    same = _run(climbing.copy())
    assert bs.derive_k(same, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40, rate_source="truth")[
        "scores"] == bs.derive_k(same, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40)["scores"]
    with pytest.raises(ValueError):
        bs.derive_k(run, BOUNDS, GRID, 1.0, 10, 0.1, np.random.default_rng(0), 40, rate_source="oracle")


def test_the_clairvoyant_rate_needs_its_own_output_names():
    assert bs.parse_args([]).rate_source == "rating"
    with pytest.raises(SystemExit):
        bs.parse_args(["--rate-source", "truth"])
    assert bs.parse_args(["--rate-source", "truth", "--output-suffix", "_truerate"]).rate_source == "truth"


def test_derive_for_run_passes_the_rate_source_through(tmp_path):
    import pandas as pd
    observed = np.linspace(0.0, 5.0, 40)
    run = _run(observed)
    path = tmp_path / "run.csv"
    pd.DataFrame({"iteration": np.arange(1, 41), "param_columns": "x0,x1", "x0": run.X[:, 0], "x1": run.X[:, 1],
                  "objective_observed": observed, "objective_true": np.full(40, 2.0),
                  "inference_simple_regret_true": 0.0, "y_opt": run.y_opt}).to_csv(path, index=False)
    task = {"file": "run.csv", "path": str(path), "iterations": 40, "bounds_low": np.zeros(2),
            "bounds_high": np.ones(2), "k_grid": GRID, "beta": 1.0, "window": 10, "rho": 1.0,
            "noise_source": "truth", "true_sitting_sd": 0.1, "decision": "fixed"}
    import json
    rating = json.loads(bs.derive_for_run(task)["scores"])
    truth = json.loads(bs.derive_for_run({**task, "rate_source": "truth"})["scores"])
    # With a flat truth the search credit vanishes: each score drops by (k_max - k) x rate.
    assert truth["pm"] < rating["pm"] and truth[str(max(GRID))] == rating[str(max(GRID))]
    seq = json.loads(bs.derive_for_run({**task, "rate_source": "truth", "decision": "sequential"})["scores"])
    assert seq


# ---------------------------------------------------------------------------
# Outputs and the by-onset table
# ---------------------------------------------------------------------------


def test_the_default_invocation_keeps_the_published_names():
    paths = bs.output_paths(Path("out"), "")
    assert {k: p.name for k, p in paths.items()} == {
        "derived": "budget_split_derived.csv", "policies": "budget_split_policies.csv",
        "by_onset": "budget_split_by_onset.csv"}


def test_a_suffix_names_every_output():
    """The rho = 0.5 files the held-out analysis reads have a producer, not a rename."""
    paths = bs.output_paths(Path("out"), "_rho0.5")
    assert paths["derived"].name == "budget_split_derived_rho0.5.csv"
    assert paths["policies"].name == "budget_split_policies_rho0.5.csv"
    assert paths["by_onset"].name == "budget_split_by_onset_rho0.5.csv"


def test_a_suffix_cannot_leave_the_output_directory():
    with pytest.raises(SystemExit):
        bs.parse_args(["--output-suffix", "/../x"])


def test_the_default_decision_is_the_published_fixed_rule():
    assert bs.parse_args([]).decision == "fixed"
    assert bs.parse_args(["--decision", "sequential"]).decision == "sequential"


def _sweep_rows(k: float, regret: float, **extra) -> dict:
    row = {"file": "r.csv", "family": "tournament", "candidates": "lcb", "winner": "look", "rho": 1.0,
           "acquisition": "logei", "error_model": "gaussian", "k": k, "regret_noisy": regret, **extra}
    # The replay's procedure name, e.g. tournament_k16_lcb_rho0.5_look.
    row["procedure"] = f"tournament_k{int(k)}_{row['candidates']}_rho{row['rho']:g}_{row['winner']}"
    return row


def test_two_replay_directories_are_pooled_and_a_duplicate_k_is_scored_once(tmp_path):
    """The paper's grid lives in two replays (k-sweep 2..12, k-wide 16..30); a k that
    both hold must count once, from the first directory named."""
    import pandas as pd
    a, b = tmp_path / "ksweep", tmp_path / "kwide"
    a.mkdir(), b.mkdir()
    pd.DataFrame([_sweep_rows(2.0, 0.1), _sweep_rows(12.0, 0.2),
                  _sweep_rows(2.0, 0.9, rho=0.5),
                  _sweep_rows(2.0, 0.9, acquisition="ucb", file="u.csv"),
                  _sweep_rows(2.0, 0.9, error_model="slip", file="s.csv"),
                  _sweep_rows(2.0, 0.9, candidates="obs")]
                 ).to_csv(a / "end_of_study_per_run.csv.gz", index=False)
    pd.DataFrame([_sweep_rows(12.0, 0.7), _sweep_rows(30.0, 0.3)]).to_csv(b / "end_of_study_per_run.csv.gz",
                                                                            index=False)
    sweep = bs.load_sweep(f"{a},{b}", 1.0)
    assert sorted(sweep["k"]) == [2.0, 12.0, 30.0]
    assert float(sweep.loc[sweep["k"] == 12.0, "regret_noisy"].iloc[0]) == 0.2
    assert set(sweep["acquisition"]) == {"logei"} and set(sweep["error_model"]) == {"gaussian"}
    assert sorted(bs.load_sweep(f"{a},{b}", 0.5)["k"]) == [2.0]
    with pytest.raises(SystemExit):
        bs.load_sweep(str(tmp_path / "missing"), 1.0)


def test_runs_whose_fit_failed_are_dropped_and_counted():
    import pandas as pd
    sweep = pd.DataFrame([{"file": f, "k": k} for f in ("a.csv", "b.csv", "c.csv") for k in (2.0, 8.0)])
    derived = pd.DataFrame({"file": ["a.csv", "b.csv", "c.csv"], "k_hat": ["2", None, "pm"],
                            "gp_failed": [False, True, "False"]})
    joined, n_failed = bs.join_derived(sweep, derived)
    assert n_failed == 1 and set(joined["file"]) == {"a.csv", "c.csv"} and len(joined) == 4
    with pytest.raises(SystemExit):
        bs.join_derived(sweep, derived.assign(gp_failed=True))


def test_policy_and_choice_names_are_normalised():
    assert bs.policy_name("fixed_k8.0") == bs.policy_name("fixed_k8") == "fixed_k8"
    assert bs.policy_name("derived") == "derived"
    assert bs.choice_name(8) == bs.choice_name("8") == bs.choice_name("8.0") == bs.choice_name(8.0) == "8"
    assert bs.choice_name("pm") == "pm"


def _toy_joined():
    """Two landscapes x two onsets x two runs, k in {2, 8}; opt_z 1 and 10."""
    import pandas as pd
    rows, no_trial = [], []
    rng = np.random.default_rng(0)
    for d, z in (("a", 1.0), ("b", 10.0)):
        for onset in (0, 20):
            for r in range(2):
                f = f"{d}_{onset}_{r}.csv"
                ref = z * rng.uniform(0.3, 0.6)
                k_hat = "pm" if onset == 20 else "2"
                for k in (2.0, 8.0):
                    rows.append({"dataset": d, "file": f, "k": k, "jitter_iteration": onset,
                                 "regret_noisy": z * rng.uniform(0.1, 0.6), "ref_noisy": ref,
                                 "k_hat": k_hat})
                no_trial.append({"file": f, "regret_pm": z * rng.uniform(0.2, 0.7),
                                 "regret_lcb1": z * rng.uniform(0.2, 0.7)})
    return pd.DataFrame(rows), pd.DataFrame(no_trial), {"a": 1.0, "b": 10.0}


def test_the_by_onset_table_reproduces_the_pooled_policies_table():
    """Same point estimates as the policies table; each interval from its own generator."""
    joined, no_trial, opt_z = _toy_joined()
    summary, frame, _ = bs.score_policies(joined, no_trial, opt_z, np.random.default_rng(bs.BOOTSTRAP_SEED))
    table = bs.score_by_onset(joined, no_trial, opt_z)
    pooled = table[(table["onset"] == "all") & (table["kind"] == "gain")].set_index("policy")
    means = frame.groupby(level="dataset").mean()
    for r in summary.itertuples(index=False):
        row = pooled.loc[bs.policy_name(r.policy)]
        assert row["estimate"] == pytest.approx(r.gain_vs_standard)
        assert row["mean_regret"] == pytest.approx(r.mean_regret)
        expected = bs._boot_mean_diff(means["standard"].to_numpy(), means[r.policy].to_numpy(),
                                      np.random.default_rng(bs.BOOTSTRAP_SEED))
        assert (row["estimate"], row["lo"], row["hi"]) == expected
    assert set(table["onset"]) == {"all", "0", "20"}
    assert set(table["seeds"]) == {"all"}


def test_the_by_onset_table_can_be_restricted_to_held_out_seeds():
    joined, no_trial, opt_z = _toy_joined()
    joined["seed"] = np.where(joined["file"].str.endswith("_0.csv"), 7, 12)
    table = bs.score_by_onset(joined, no_trial, opt_z, seeds=(12,), seeds_label="12")
    assert set(table["seeds"]) == {"12"}
    assert table["n_runs"].max() == joined[joined["seed"] == 12]["file"].nunique()


def test_the_written_by_onset_table_carries_all_seeds_and_each_seed_half():
    """budget_split_by_onset<suffix>.csv holds the held-out numbers the paper quotes; each
    block equals score_by_onset on those seeds, and a half with no runs is left out."""
    joined, no_trial, opt_z = _toy_joined()
    joined["seed"] = np.where(joined["file"].str.endswith("_0.csv"), 7, 12)
    table = bs.by_onset_tables(joined, no_trial, opt_z)
    assert list(dict.fromkeys(table["seeds"])) == ["all", "7-11", "12-16"]
    for label, seeds in bs.SEED_SETS:
        alone = bs.score_by_onset(joined, no_trial, opt_z, seeds=seeds, seeds_label=label)
        block = table[table["seeds"] == label].reset_index(drop=True)
        assert block.equals(alone)
    only_train = joined[joined["seed"] == 7]
    assert set(bs.by_onset_tables(only_train, no_trial, opt_z)["seeds"]) == {"all", "7-11"}
    # Without a seed column only the pooled block can be formed.
    assert set(bs.by_onset_tables(joined.drop(columns="seed"), no_trial, opt_z)["seeds"]) == {"all"}


def test_the_by_onset_differences_are_ratios_of_landscape_means_not_run_means():
    joined, no_trial, opt_z = _toy_joined()
    table = bs.score_by_onset(joined, no_trial, opt_z, comparators=("fixed_k8", "always_pm"))
    sub = joined[joined["jitter_iteration"] == 20].assign(z=lambda f: f["dataset"].map(opt_z))
    # Onset 20 chose pm everywhere: derived regret = pm regret, so derived minus pm is zero.
    diff_pm = table[(table["onset"] == "20") & (table["kind"] == "derived_minus")
                    & (table["policy"] == "always_pm")].iloc[0]
    assert diff_pm["estimate"] == pytest.approx(0.0, abs=1e-12)
    # derived minus fixed_k8, by hand: per-run regrets / opt_z, landscape means, then the mean.
    k8 = sub[sub["k"] == 8.0].set_index("file")
    pm = no_trial.set_index("file")["regret_pm"]
    per_run = (k8["regret_noisy"] - pm.reindex(k8.index)) / k8["z"]
    expected = per_run.groupby(k8["dataset"]).mean().mean()
    diff_k8 = table[(table["onset"] == "20") & (table["kind"] == "derived_minus")
                    & (table["policy"] == "fixed_k8")].iloc[0]
    assert diff_k8["estimate"] == pytest.approx(expected)
    shares = table[(table["onset"] == "20") & (table["kind"] == "choice_share")]
    assert dict(zip(shares["policy"], shares["estimate"])) == {"pm": 1.0}


# ---------------------------------------------------------------------------
# The price: the rule derived on the clean twins
# ---------------------------------------------------------------------------


def test_a_noisy_run_and_its_clean_twin_share_a_stem():
    noisy = "bo_sensor_error_ackley_value_logei_seed10_jittered_exact_ar1_jit0_std0.05.csv"
    clean = "ackley/bo_sensor_error_ackley_value_logei_seed10_baseline_exact.csv"
    assert bs.twin_stem(noisy) == bs.twin_stem(clean) == "bo_sensor_error_ackley_value_logei_seed10"
    with pytest.raises(ValueError):
        bs.twin_stem("summary.csv")


def test_the_clean_twin_needs_its_own_output_names():
    assert bs.parse_args([]).twin == "noisy"
    with pytest.raises(SystemExit):
        bs.parse_args(["--twin", "clean"])
    assert bs.parse_args(["--twin", "clean", "--output-suffix", "_clean"]).twin == "clean"
    assert bs.price_path(Path("out"), "_clean").name == "budget_split_price_clean.csv"


def _toy_price():
    """Two landscapes (opt_z 1 and 10) x two seeds; each clean twin has two noisy runs
    whose replay rows repeat the twin's clean regrets, as the replay writes them."""
    import pandas as pd
    rows, no_trial, derived = [], [], []
    choice = {("a", 7): "2", ("a", 12): "pm", ("b", 7): "8", ("b", 12): "lcb"}
    for d, z in (("a", 1.0), ("b", 10.0)):
        for seed in (7, 12):
            stem = f"bo_sensor_error_{d}_value_logei_seed{seed}"
            ref = z * (0.1 + 0.01 * seed)
            clean_regret = {2.0: ref + z * 0.02, 8.0: ref + z * 0.05}
            for model in ("gaussian", "drift"):
                noisy = f"{stem}_jittered_exact_{model}_jit0_std1.csv"
                for k in (2.0, 8.0):
                    rows.append({"dataset": d, "seed": seed, "file": noisy, "k": k, "regret_clean": clean_regret[k],
                                 "ref_clean": ref, "regret_noisy": 9.0, "ref_noisy": 9.0})
                no_trial.append({"file": noisy, "regret_pm": 99.0, "regret_lcb1": 99.0})
            clean = f"{stem}_baseline_exact.csv"
            no_trial.append({"file": clean, "regret_pm": ref + z * 0.01, "regret_lcb1": ref + z * 0.03})
            derived.append({"file": clean, "k_hat": choice[(d, seed)], "gp_failed": False})
    return pd.DataFrame(rows), pd.DataFrame(derived), pd.DataFrame(no_trial), {"a": 1.0, "b": 10.0}


def test_the_price_is_trt_clean_less_ref_clean_in_opt_z_units_and_counts_each_twin_once():
    sweep, derived, no_trial, opt_z = _toy_price()
    frame, choices, n_failed = bs.price_frame(sweep, derived, no_trial, opt_z)
    assert n_failed == 0 and len(frame) == 4          # one row per clean twin, not per noisy run
    assert frame["fixed_k2"].to_numpy() == pytest.approx(np.full(4, 0.02))
    assert frame["fixed_k8"].to_numpy() == pytest.approx(np.full(4, 0.05))
    # The no-trial rules are read from the clean twin's rows, never the noisy runs'.
    assert frame["always_pm"].to_numpy() == pytest.approx(np.full(4, 0.01))
    assert frame["always_lcb"].to_numpy() == pytest.approx(np.full(4, 0.03))
    # The derived rule is charged the clean price of what it chose on the clean twin.
    expected = {("a", 7): 0.02, ("a", 12): 0.01, ("b", 7): 0.05, ("b", 12): 0.03}
    for (d, stem, seed), value in frame["derived"].items():
        assert value == pytest.approx(expected[(d, seed)])
    table = bs.score_price(sweep, derived, no_trial, opt_z, comparators=("fixed_k8",))
    pooled = table[(table["seeds"] == "all") & (table["kind"] == "price")].set_index("policy")
    assert pooled.loc["derived", "estimate"] == pytest.approx(np.mean([np.mean([0.02, 0.01]), np.mean([0.05, 0.03])]))
    diff = table[(table["seeds"] == "all") & (table["kind"] == "derived_minus")].iloc[0]
    assert diff["policy"] == "fixed_k8" and diff["estimate"] == pytest.approx(0.0275 - 0.05)
    held_out = table[(table["seeds"] == "12-16") & (table["kind"] == "price")].set_index("policy")
    assert held_out.loc["derived", "estimate"] == pytest.approx(np.mean([0.01, 0.03]))
    assert int(held_out["n_twins"].iloc[0]) == 2
    shares = table[(table["seeds"] == "all") & (table["kind"] == "choice_share")]
    assert dict(zip(shares["policy"], shares["estimate"])) == {"2": 0.25, "pm": 0.25, "8": 0.25, "lcb": 0.25}


def test_the_price_drops_a_failed_fit_and_refuses_inconsistent_twins():
    sweep, derived, no_trial, opt_z = _toy_price()
    derived.loc[0, "gp_failed"] = True
    frame, _, n_failed = bs.price_frame(sweep, derived, no_trial, opt_z)
    assert n_failed == 1 and len(frame) == 3
    broken = sweep.copy()
    broken.loc[0, "regret_clean"] += 1.0             # two noisy runs disagree about their twin
    with pytest.raises(ValueError):
        bs.price_frame(broken, derived, no_trial, opt_z)


def test_clean_tasks_are_one_per_twin_with_an_exact_sitting(tmp_path):
    sweep, _, _, _ = _toy_price()
    sweep["dataset"] = "branin"
    sweep["file"] = sweep["file"].str.replace("_a_", "_branin_").str.replace("_b_", "_branin_")
    stems = {bs.twin_stem(f) for f in sweep["file"]}
    baselines = {s: tmp_path / "branin" / f"{s}_baseline_exact.csv" for s in stems}

    class Arm:
        iterations = 50

    settings = {"k_grid": (2, 8), "beta": 1.0, "window": 10, "rho": 1.0, "noise_source": "truth",
                "stats_path": bs.bb.DEFAULT_STATS_PATH, "decision": "fixed", "rate_source": "rating"}
    tasks = bs.build_clean_tasks(sweep, baselines, Arm(), settings)
    assert len(tasks) == len(stems)
    assert {t["file"] for t in tasks} == {p.name for p in baselines.values()}
    assert all(t["true_sitting_sd"] == 0.0 for t in tasks)
    with pytest.raises(SystemExit):
        bs.build_clean_tasks(sweep, {}, Arm(), settings)
