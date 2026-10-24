import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import replay_schmitt_decisions as replay


@pytest.fixture
def selector_path(monkeypatch):
    """Use an installed selector or an explicitly configured OpenBO source root."""
    configured = os.environ.get("OPENBO_SRC")
    if configured is not None:
        path = Path(configured).expanduser()
        if (
            not configured.strip()
            or not (path / "openbo/decision_making/__init__.py").is_file()
        ):
            pytest.fail(
                "OPENBO_SRC must point to an OpenBO source root containing "
                "openbo/decision_making/__init__.py; correct it or unset it to use "
                "an installed OpenBO package.",
                pytrace=False,
            )
        monkeypatch.syspath_prepend(str(path.resolve()))
    try:
        spec = importlib.util.find_spec("openbo.decision_making")
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        pytest.skip(
            "OpenBO selector integration requires an installed OpenBO package "
            "or OPENBO_SRC pointing to its src directory; pure replay tests still run."
        )


def test_normalization_excludes_dominated_before_range():
    Y = np.array([[0.0, -1.0], [-0.3, -0.3], [-1.0, 0.0], [-99.0, -99.0]])
    Z, mask = replay.normalized_costs(Y)
    assert mask.tolist() == [True, True, True, False]
    np.testing.assert_allclose(Z[:3], -Y[:3])


def test_preferences_reproducible_and_corruption_separate():
    P, labels = replay.preference_bank(3, 12, 4)
    np.testing.assert_array_equal(P, replay.preference_bank(3, 12, 4)[0])
    np.testing.assert_array_equal(P, replay.corrupt_preferences(P, 0, 12))
    Q = replay.corrupt_preferences(P, 1, 12)
    np.testing.assert_allclose(Q.sum(1), 1)
    np.testing.assert_array_equal(Q[1:4], np.eye(3))
    assert not np.array_equal(P[-4:], Q[-4:])
    assert labels == ["equal", "axis", "axis", "axis"] + ["dirichlet"] * 4


def test_exact_matching_reference_zero_and_knee_price_real(selector_path):
    Y = -np.array([[0.0, 1.0], [0.3, 0.3], [1.0, 0.0]])
    frame = replay.archive_decisions(Y, Y, seed=12, n_random=2)
    exact = frame[(frame.observation_mode == "exact") & (frame.preference_noise == 0)]
    for family, reference in replay.REFERENCE.items():
        assert np.allclose(
            exact.loc[
                (exact.utility_family == family) & (exact.rule == reference), "regret"
            ],
            0,
        )
    axis_knee = exact[
        (exact.utility_family == "linear")
        & (exact.rule == "knee_weighted")
        & (exact.preference_type == "axis")
    ]
    assert (axis_knee.regret > 0).all()
    np.testing.assert_allclose(exact.regret, exact.clean_regret)


def test_recorded_error_uses_same_candidates_and_exact_counterfactual(selector_path):
    truth = -np.array([[0.0, 1.0], [0.3, 0.3], [1.0, 0.0], [0.8, 0.8]])
    observed = truth.copy()
    observed[3] = [10.0, 10.0]  # lucky observation of truly dominated design
    frame = replay.archive_decisions(truth, observed, seed=12, n_random=2)
    for _, block in frame.groupby(["rule", "utility_family", "preference_id"]):
        assert block.clean_index.nunique() == 1
        assert block.clean_regret.nunique() == 1
    bad = frame[(frame.observation_mode == "recorded") & (frame.preference_noise == 0)]
    assert (bad.chosen_index == 3).all()
    assert (bad.chosen_true_dominated == 1).all()
    assert frame.chosen_index.between(0, len(truth) - 1).all()


def test_comparison_reports_clean_price_against_matching_reference(monkeypatch):
    monkeypatch.setattr(replay.aba, "BOOTSTRAP_REPS", 20)
    rows = []
    for dataset in replay.PROBLEMS:
        for rule, noisy, clean in [
            ("weighted", 0.2, 0),
            ("nearest_ideal", 0.25, 0.05),
            ("knee_weighted", 0.1, 0.04),
        ]:
            rows.append(
                {
                    "dataset": dataset,
                    "acquisition": "qlognehvi",
                    "seed": 12,
                    "error_model": "gaussian",
                    "jitter_std": 1.0,
                    "jitter_iteration": 0,
                    "source_kind": "noisy",
                    "measurement_setting": "cell",
                    "observation_mode": "recorded",
                    "preference_noise": 0.0,
                    "utility_family": "linear",
                    "rule": rule,
                    "utility_regret_jitter": noisy,
                    "utility_regret_baseline": clean,
                }
            )
    summary, _ = replay.comparisons(pd.DataFrame(rows))
    knee = summary[
        (summary.measurement_setting == "cell") & (summary.rule == "knee_weighted")
    ].iloc[0]
    assert knee.gain == pytest.approx(0.1)
    assert knee.price == pytest.approx(0.04)
    assert knee.cost == pytest.approx(0.2)
    assert "wilcoxon_p" not in summary


@pytest.mark.parametrize(
    "costs",
    [
        (-0.005,) * 4,
        (0.0,) * 4,
        (0.005,) * 4,
        (0.01,) * 4,
        (0.02,) * 4,
        (0.005, 0.1, 0.1, 0.1),
    ],
)
def test_per_problem_recovery_suppression_preserves_absolute_results(monkeypatch, costs):
    monkeypatch.setattr(replay.aba, "BOOTSTRAP_REPS", 20)
    rows = []
    for dataset, cost in zip(replay.PROBLEMS, costs, strict=True):
        for rule in replay.RULES:
            # Positive regret keeps negative reference costs physically valid.
            is_reference = rule == "weighted"
            rows.append({
                "dataset": dataset, "acquisition": "qlognehvi", "seed": 12,
                "error_model": "gaussian", "jitter_std": 1.0, "jitter_iteration": 0,
                "source_kind": "noisy", "measurement_setting": "cell",
                "observation_mode": "recorded", "preference_noise": 0.0,
                "utility_family": "linear", "rule": rule,
                "utility_regret_jitter": 0.125 + cost - (0.0 if is_reference else 0.003),
                "utility_regret_baseline": 0.125 if is_reference else 0.145,
            })
    summary, per_problem = replay.comparisons(pd.DataFrame(rows))
    knee = per_problem[
        (per_problem.measurement_setting == "cell")
        & (per_problem.rule == "knee_weighted")
    ].set_index("dataset")
    for dataset, cost in zip(replay.PROBLEMS, costs, strict=True):
        row = knee.loc[dataset]
        assert row.cost == pytest.approx(cost)
        assert row.gain == pytest.approx(0.003)
        assert row.price == pytest.approx(0.02)
        assert bool(row.recovery_suppressed) == (cost < 0.01)
        if cost < 0.01:
            assert np.isnan(row.recovered)
        else:
            assert row.recovered == pytest.approx(0.003 / cost)
    # One tiny problem must be guarded even when the pooled denominator is safe.
    if min(costs) < 0.01 and np.mean(costs) >= 0.01:
        pooled_knee = summary[
            (summary.measurement_setting == "cell")
            & (summary.rule == "knee_weighted")
        ].iloc[0]
        assert not pooled_knee.recovery_suppressed
