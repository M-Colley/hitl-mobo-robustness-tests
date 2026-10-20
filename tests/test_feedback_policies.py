"""Cheap protocol and numerical checks, without running any benchmark cell."""
from __future__ import annotations

import builtins

import numpy as np
import pytest

import feedback_policies as fp


X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])


@pytest.mark.parametrize("method", fp.METHODS)
def test_external_feedback_is_required_and_recommendation_is_seen(method):
    policy = fp.make_policy(method, X, 17)
    with pytest.raises(RuntimeError, match="purchased feedback"):
        policy.recommend()
    if fp.FEEDBACK[method] == "rating":
        policy.tell((3,), 2.0)
    else:
        policy.tell((3, 1), 1.0)
    assert policy.recommend() in policy.seen
    assert set(policy.seen) <= {1, 3}


@pytest.mark.parametrize("method", [m for m in fp.METHODS if m != "bo_logei"])
def test_protocol_determinism_distinct_pairs_and_finite_scores(method):
    left = fp.make_policy(method, X, 31)
    right = fp.make_policy(method, X.copy(), 31)
    for t in range(8):
        indices = left.ask()
        assert indices == right.ask()
        assert isinstance(indices, tuple)
        assert len(set(indices)) == len(indices)
        assert all(0 <= i < len(X) for i in indices)
        # Handwritten feedback fixtures, independent of an oracle or benchmark.
        value = (0.5 if t % 3 == 0 else float(t % 2)) if len(indices) == 2 else float(t % 3)
        left.tell(indices, value)
        right.tell(indices, value)
        assert left.recommend() in left.seen
        for actual in left.scores():
            assert actual.shape == (len(X),)
            assert np.isfinite(actual).all()
        assert np.all(left.scores()[1] >= 0)


@pytest.mark.parametrize("method", ["random", "sobol"])
def test_observation_ignorant_paths_are_identical_under_changed_feedback(method):
    clean = fp.make_policy(method, X, 81)
    corrupted = fp.make_policy(method, X, 81)
    for t in range(12):
        indices = clean.ask()
        assert indices == corrupted.ask()
        clean.tell(indices, float(t))
        corrupted.tell(indices, -float(t))


@pytest.mark.parametrize("method", ["random", "sobol", "cem"])
def test_standard_rating_recommendation_uses_first_highest_observation(method):
    policy = fp.make_policy(method, X, 9)
    policy.tell((4,), 3.0)
    policy.tell((0,), 3.0)
    policy.tell((4,), -10.0)
    assert policy.recommend() == 4


def test_policy_owns_the_catalog_and_does_not_mutate_global_rng():
    supplied = X.copy()
    np.random.seed(901)
    before = np.random.get_state()
    policy = fp.make_policy("reward_policy", supplied, 22)
    supplied[:] = 0.1
    assert np.array_equal(policy.X, X)
    policy.tell((4, 0), 1)
    policy.ask()
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert not policy.X.flags.writeable


@pytest.mark.parametrize("bad", [[], [[np.nan]], [[np.inf]], [[-0.1]], [[1.1]], [0, 1]])
def test_invalid_catalog_is_rejected(bad):
    with pytest.raises(ValueError, match="matrix"):
        fp.make_policy("random", bad, 1)


@pytest.mark.parametrize("indices,value", [
    ((0, 0), 1), ((0, 5), 1), ((0, -1), 1), ((0.5, 1), 1),
    ((True, 1), 1), ((0,), 1), ((0, 1), 0.2), ((0, 1), np.nan),
])
def test_invalid_pair_feedback_does_not_change_policy(indices, value):
    policy = fp.make_policy("dpo", X, 0)
    with pytest.raises(ValueError):
        policy.tell(indices, value)
    assert policy.seen == []
    assert policy.feedback == []


@pytest.mark.parametrize("method", ["dueling_ucb", "reward_policy", "dpo"])
def test_pair_models_use_label_direction_and_do_not_recommend_unseen(method):
    forward = fp.make_policy(method, X, 5)
    reversed_labels = fp.make_policy(method, X, 5)
    for _ in range(4):
        forward.tell((1, 3), 1.0)
        reversed_labels.tell((1, 3), 0.0)
    assert forward.recommend() == 1
    assert reversed_labels.recommend() == 3
    assert set(forward.seen) == {1, 3}


@pytest.mark.parametrize("method", ["dueling_ucb", "reward_policy", "dpo"])
def test_preference_ties_preserve_first_exposure_recommendation(method):
    policy = fp.make_policy(method, X, 0)
    policy.tell((3, 1), 0.5)
    assert policy.recommend() == 3


def test_dueling_ucb_reuses_feedback_leader_before_catalog_exhaustion():
    catalog = np.linspace(0, 1, 100)[:, None]
    first_wins = fp.make_policy("dueling_ucb", catalog, 8)
    second_wins = fp.make_policy("dueling_ucb", catalog, 8)
    first_pair = first_wins.ask()
    assert first_pair == second_wins.ask()
    first_wins.tell(first_pair, 1)
    second_wins.tell(first_pair, 0)
    next_first = first_wins.ask()
    next_second = second_wins.ask()
    assert len(first_wins.seen) == 2 < len(catalog)
    assert next_first[0] == first_pair[0]
    assert next_second[0] == first_pair[1]
    assert next_first != next_second
    # A fresh opponent is exposed while the known winner is tested again.
    assert next_first[1] not in first_pair
    for _ in range(4):
        indices = first_wins.ask()
        assert indices[0] == first_pair[0]
        first_wins.tell(indices, 1)
    assert first_wins.recommend() == first_pair[0]


def test_ridge_update_matches_batch_ridge_and_shrinks_uncertainty():
    policy = fp.make_policy("ridge_ucb", X, 4, regularization=0.3)
    _, prior_sd = policy.scores()
    for index, rating in [(0, -1), (4, 2), (0, 0.5)]:
        policy.tell((index,), rating)
    features = policy.features[[0, 4, 0]]
    precision = 0.3 * np.eye(features.shape[1]) + features.T @ features
    weights = np.linalg.solve(precision, features.T @ np.array([-1, 2, 0.5]))
    mean, sd = policy.scores()
    np.testing.assert_allclose(mean, policy.features @ weights, atol=1e-12)
    assert np.all(sd <= prior_sd + 1e-12)
    assert mean[4] > mean[0]


def test_reward_policy_is_the_exp_reward_over_beta_distribution():
    policy = fp.make_policy("reward_policy", X, 7, beta=0.4)
    policy.tell((4, 0), 1)
    means, _ = policy.scores()
    probs = policy._distribution()
    assert np.log(probs[4] / probs[0]) == pytest.approx((means[4] - means[0]) / 0.4)


def test_dpo_loss_has_explicit_beta_and_logit_regularization():
    beta, precision = 0.3, 0.07
    policy = fp.make_policy("dpo", X, 7, beta=beta, regularization=precision)
    policy.tell((4, 0), 1)
    probability = policy._distribution()
    logits = policy._logits
    delta = beta * (logits[4] - logits[0])
    # First-order condition of log(1+exp(-beta*(theta_w-theta_l)))
    # + precision * ||theta||^2 / 2, with the uniform reference fixed.
    residual = beta / (1 + np.exp(delta))
    assert precision * logits[4] == pytest.approx(residual, abs=1e-6)
    assert precision * logits[0] == pytest.approx(-residual, abs=1e-6)
    assert probability[4] > probability[0]
    assert np.sum(logits) == pytest.approx(0.0, abs=1e-8)


def test_preferences_never_sample_duplicate_pair_when_softmax_underflows():
    policy = fp.make_policy("dpo", X, 4, epsilon=0)
    policy._logits[:] = [-1e5, -1e5, 1e5, -1e5, -1e5]
    policy._fitted_at = 0
    first, second = policy.ask()
    assert first == 2
    assert second != first


def test_optional_bo_imports_are_lazy_and_missing_dependency_is_explicit(monkeypatch):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("dependency unavailable in unit test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    policy = fp.make_policy("bo_logei", X, 3, n_initial=2)
    assert len(policy.ask()) == 1  # Construction and initialization need no torch.
    policy.tell((0,), -1.0)
    policy.tell((4,), 1.0)
    with pytest.raises(RuntimeError, match="optional torch"):
        policy.ask()


def test_unknown_settings_fail_instead_of_being_silently_ignored():
    with pytest.raises(ValueError, match="Unknown method"):
        fp.make_policy("rlfh", X, 0)
    with pytest.raises(TypeError):
        fp.make_policy("ridge_ucb", X, 0, exploraton=2)
    for method, kwargs in [
        ("reward_policy", {"beta": 0}), ("dpo", {"regularization": 0}),
        ("cem", {"epsilon": 1.1}), ("cem", {"n_initial": 1.5}),
        ("ridge_ucb", {"lengthscale": np.nan}), ("dueling_ucb", {"exploration": -1}),
    ]:
        with pytest.raises(ValueError):
            fp.make_policy(method, X, 0, **kwargs)


@pytest.mark.parametrize("method", fp.METHODS)
def test_policy_options_preflight_does_not_construct_models(method, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Option validation must not construct a policy")

    monkeypatch.setattr(fp, "make_policy", forbidden)
    fp.validate_policy_options(method, {})
    with pytest.raises(ValueError, match="Unsupported options"):
        fp.validate_policy_options(method, {"typo": 1})


@pytest.mark.parametrize("method,options", [
    ("bo_logei", {"n_initial": 1}), ("reward_policy", {"beta": True}),
    ("reward_policy", {"beta": "0.2"}), ("dpo", {"regularization": 0}),
    ("cem", {"elite_fraction": 2}), ("ridge_ucb", {"n_features": 2.5}),
])
def test_policy_options_preflight_rejects_invalid_values(method, options):
    with pytest.raises(ValueError):
        fp.validate_policy_options(method, options)


def test_real_bo_fits_once_and_reuses_exact_marginals_for_logei(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from botorch import fit
    from botorch.acquisition.analytic import LogExpectedImprovement

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    original_fit = fit.fit_gpytorch_mll
    calls = []

    def recorded_fit(*args, **kwargs):
        calls.append(1)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(fit, "fit_gpytorch_mll", recorded_fit)
    try:
        policy = fp.make_policy("bo_logei", X, 30, n_initial=2)
        policy.tell((0,), -1)
        policy.tell((4,), 1)
        state = torch.random.get_rng_state().clone()
        assert policy.recommend() == 4
        assert torch.equal(state, torch.random.get_rng_state())
        assert len(calls) == 1
        # Recommending early should evaluate only seen candidates, not all X.
        assert policy._cached.tolist() == [True, False, False, False, True]
        original_posterior = policy._gp.posterior
        posterior_queries = []

        def recorded_posterior(query=None, **kwargs):
            query = kwargs.pop("X", query)
            posterior_queries.append(query.shape)
            return original_posterior(query, **kwargs)

        monkeypatch.setattr(policy._gp, "posterior", recorded_posterior)
        mean, sd = policy.scores()
        assert len(posterior_queries) == 1
        assert posterior_queries[0][-2] == 1
        available = policy._available()
        acq = LogExpectedImprovement(policy._gp, best_f=float(mean[policy.seen].max()))
        with torch.no_grad():
            expected_values = acq(torch.tensor(X[available], dtype=torch.double).unsqueeze(-2))
        expected = int(available[expected_values.argmax().item()])
        query_count = len(posterior_queries)
        assert policy.ask() == (expected,)
        assert policy.recommend() == 4
        np.testing.assert_array_equal(policy.scores()[0], mean)
        np.testing.assert_array_equal(policy.scores()[1], sd)
        assert len(posterior_queries) == query_count
        assert len(calls) == 1
        policy.tell((2,), 0.25)
        assert policy.recommend() in policy.seen
        assert len(calls) == 2
    finally:
        torch.set_num_threads(previous_threads)
