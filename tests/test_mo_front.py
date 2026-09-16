"""Tests for scripts/replay_mo_front.py, the multi-objective deployment replay.

All fast: hand-built fronts, a toy estimand frame, and one 12-iteration driver
run with the model-free acquisition (the replay does not care how the designs
were proposed, only that it reads back what the simulator deployed).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import replay_mo_front as rmf

REPO = Path(__file__).resolve().parents[1]


# --- fronts and hypervolume -------------------------------------------------


def test_hypervolume_matches_the_simulator():
    import bo_sensor_error_simulation as sim

    rng = np.random.default_rng(0)
    for m in (2, 3):
        Y = rng.normal(size=(25, m))
        ref = np.full(m, -1.5)
        assert rmf.hypervolume(Y, ref) == sim._compute_hypervolume(list(Y), ref)


def test_strictly_dominated_mask():
    Y = np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [3.0, 0.0]])
    assert rmf.strictly_dominated(Y).tolist() == [True, False, False, False]


def test_trim_removes_the_smallest_contribution_at_every_step():
    ref = np.zeros(2)
    # Exclusive contributions: 0.1, 1.9, 2.0, 0.8, 0.02. Drop (4.1, 0.2) first;
    # (1, 5) then still adds only 0.1 and goes next.
    Y = np.array([[1.0, 5.0], [2.0, 4.9], [3.0, 3.0], [4.0, 1.0], [4.1, 0.2]])
    front = rmf.non_dominated(Y)
    assert len(front) == 5
    kept = rmf.trim_to_cardinality(Y, front, 3, ref)
    assert kept.tolist() == [1, 2, 3]

    members = list(front)
    while len(members) > 3:
        total = rmf.hypervolume(Y[members], ref)
        contrib = [total - rmf.hypervolume(np.delete(Y[members], j, axis=0), ref) for j in range(len(members))]
        members.pop(int(np.argmin(contrib)))
    assert sorted(members) == kept.tolist()
    assert rmf.trim_to_cardinality(Y, front, 7, ref).tolist() == front.tolist()


def test_trim_drops_points_outside_the_reference_box_farthest_first():
    ref = np.zeros(2)
    Y = np.array([[3.0, 3.0], [-2.0, 6.0], [-0.5, 5.0], [5.0, 1.0]])
    front = rmf.non_dominated(Y)
    assert len(front) == 4
    assert rmf.trim_to_cardinality(Y, front, 3, ref).tolist() == [0, 2, 3]
    assert rmf.trim_to_cardinality(Y, front, 2, ref).tolist() == [0, 3]
    with pytest.raises(ValueError):
        rmf.trim_to_cardinality(Y, front, 0, ref)


# --- the decision-maker ------------------------------------------------------


def test_weights_are_common_random_numbers():
    a, b = rmf.dirichlet_weights(7, 2), rmf.dirichlet_weights(7, 2)
    assert a.shape == (rmf.N_WEIGHTS, 2)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    assert not np.array_equal(a, rmf.dirichlet_weights(8, 2))


def test_chebyshev_utility_is_monotone_in_every_objective():
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(20, 3))
    better = Y + rng.uniform(0.0, 1.0, Y.shape)
    W = rmf.dirichlet_weights(7, 3, 10)
    utopia = np.full(3, 3.0)
    assert (rmf.chebyshev_utility(better, W, utopia) >= rmf.chebyshev_utility(Y, W, utopia)).all()


def test_decision_maker_has_no_regret_when_it_sees_the_truth():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(30, 3))
    regret, shortfall = rmf.decision_maker(Y, Y, rmf.non_dominated(Y), rmf.dirichlet_weights(7, 3),
                                           Y.max(axis=0) + 0.5)
    assert regret == 0.0 and shortfall > 0.0


def test_decision_maker_pays_for_a_misleading_estimate():
    Y_true = np.array([[0.0, 2.0], [2.0, 0.0], [1.5, 1.5]])
    Y_est = Y_true.copy()
    Y_est[2] = [-5.0, -5.0]  # the balanced design looks terrible
    regret, shortfall = rmf.decision_maker(Y_est, Y_true, np.arange(3), np.array([[0.5, 0.5]]),
                                           np.array([2.0, 2.0]))
    # True utilities -1, -1, -0.25; the pick is design 0.
    assert regret == pytest.approx(0.75) and shortfall == pytest.approx(1.0)


# --- finding runs ------------------------------------------------------------


def test_parse_run_names_including_a_variant_suffix():
    b = rmf.parse_run_name("bo_sensor_error_zdt1_multi_objective_qlognehvi_seed7_baseline_exact.csv")
    assert (b.kind, b.dataset, b.acquisition, b.seed, b.suffix) == ("clean", "zdt1", "qlognehvi", 7, "")
    n = rmf.parse_run_name(
        "bo_sensor_error_zdt1_multi_objective_qlognehvi_seed12_jittered_exact_gaussian_jit20_std1.0_xcorr0.85.csv"
    )
    assert (n.kind, n.channel, n.jitter_iteration, n.jitter_std, n.suffix) == ("noisy", "gaussian", 20, 1.0, "_xcorr0.85")
    plain = rmf.parse_run_name(
        "bo_sensor_error_branincurrin_multi_objective_qehvi_seed9_jittered_exact_gaussian_jit0_std0.05.csv"
    )
    assert (plain.jitter_std, plain.suffix) == (0.05, "")
    assert rmf.parse_run_name("bo_sensor_error_ackley_value_logei_seed7_baseline_exact.csv") is None


def test_discovery_pairs_a_variant_with_the_unsuffixed_clean_run(tmp_path):
    folder = tmp_path / "zdt1"
    folder.mkdir()
    stem = "bo_sensor_error_zdt1_multi_objective_qlognehvi"
    names = [
        f"{stem}_seed7_baseline_exact.csv",
        f"{stem}_seed7_jittered_exact_gaussian_jit0_std1.0.csv",
        f"{stem}_seed7_jittered_exact_gaussian_jit0_std1.0_xcorr0.85.csv",
        f"{stem}_seed7_jittered_exact_gaussian_jit20_std1.0_xcorr0.85.csv",
        f"{stem}_seed8_baseline_exact.csv",
        f"{stem}_seed9_jittered_exact_gaussian_jit0_std1.0_xcorr0.85.csv",
    ]
    for name in names:
        (folder / name).write_text("")

    groups, orphans = rmf.discover_groups(tmp_path, acquisitions={"qlognehvi"}, error_model="gaussian",
                                          suffix_for=lambda em, s: "_xcorr0.85")
    assert [(g.seed, [(s, o) for s, o, _ in g.noisy]) for g in groups] == [(7, [(1.0, 0), (1.0, 20)])]
    assert groups[0].baseline.name == names[0]
    assert orphans == [names[5]]

    standard, orphans = rmf.discover_groups(tmp_path, acquisitions={"qlognehvi"}, error_model="gaussian",
                                            suffix_for=lambda em, s: "")
    assert [(g.seed, len(g.noisy)) for g in standard] == [(7, 1)] and not orphans
    assert rmf.available_suffixes(tmp_path) == ["", "_xcorr0.85"]


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--n-jobs", "7"], "--n-jobs"),
        (["--n-jobs", "0"], "--n-jobs"),
        (["--error-cross-corr", "0.85", "--variant-suffix", "_x"], "not both"),
        (["--error-cross-corr", "1.5"], "[-1, 1]"),
        (["--variant-suffix", "xcorr0.85"], "start with '_'"),
        (["--lower-bound-sds", "0"], "--lower-bound-sds"),
        (["--acquisitions", "qlognehvi,sobol"], "model-free"),
        (["--admissible", "zdt1", "--include-inadmissible"], "--include-inadmissible"),
        (["--error-model", "none"], "clean-run marker"),
        (["--n-weights", "0"], "--n-weights"),
    ],
)
def test_incompatible_settings_are_refused(argv, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        rmf.validate_args(rmf.parse_args(argv))
    rmf.validate_args(rmf.parse_args([]))


def test_cross_corr_suffix_is_asked_of_the_driver(monkeypatch):
    import bo_synthetic_error_simulation as drv

    assert rmf.cross_corr_suffix(0.0, "gaussian", 1.0) == ""

    def refuse(argv=None):
        raise SystemExit(2)

    monkeypatch.setattr(drv, "parse_args", refuse)
    with pytest.raises(ValueError, match="--variant-suffix"):
        rmf.cross_corr_suffix(0.85, "gaussian", 1.0)

    seen = {}

    def accept(argv=None):
        seen["argv"] = argv
        return argparse.Namespace(error_bias_mode="scaled", error_bias=0.2,
                                  error_spike_std_mode="scaled", error_spike_std=0.5)

    monkeypatch.setattr(drv, "parse_args", accept)
    monkeypatch.setattr(drv, "_variant_suffix", lambda args, em, bias, spike: "_xcorr0.85")
    assert rmf.cross_corr_suffix(0.85, "gaussian", 1.0) == "_xcorr0.85"
    assert "--error-cross-corr=0.85" in seen["argv"]

    monkeypatch.setattr(drv, "_variant_suffix", lambda *a: "")
    with pytest.raises(ValueError, match="does not name"):
        rmf.cross_corr_suffix(0.85, "gaussian", 1.0)


def test_cross_corr_suffix_with_the_real_driver():
    import bo_synthetic_error_simulation as drv

    try:
        drv.parse_args(["--multi-objective", "--error-cross-corr=0.85"])
    except SystemExit:
        pytest.skip("the driver has no --error-cross-corr flag yet")
    suffix = rmf.cross_corr_suffix(0.85, "gaussian", 1.0)
    assert suffix.startswith("_") and "/" not in suffix and "\\" not in suffix
    parsed = rmf.parse_run_name(
        f"bo_sensor_error_zdt1_multi_objective_qlognehvi_seed7_jittered_exact_gaussian_jit0_std1.0{suffix}.csv"
    )
    assert parsed is not None and parsed.suffix == suffix and parsed.jitter_std == 1.0


# --- the estimand -------------------------------------------------------------


TOY = pd.DataFrame({
    "dataset": ["a", "a", "b", "b", "c"],
    "ref_noisy": [1.0, 3.0, 2.0, 2.0, 4.0],
    "ref_clean": [0.5, 0.5, 1.0, 1.0, 1.0],
    "trt_noisy": [1.0, 1.0, 1.5, 1.5, 2.0],
    "trt_clean": [0.6, 0.6, 1.0, 1.0, 1.5],
})


def test_recovery_is_a_ratio_of_problem_means():
    out = rmf.summarise(TOY, np.random.default_rng(rmf.BOOTSTRAP_SEED))
    # Per problem: cost 1.5, 1, 3; gain 1, 0.5, 2; price 0.1, 0, 0.5.
    assert out["n_problems"] == 3 and out["n_cells"] == 5
    assert out["cost"] == pytest.approx(5.5 / 3)
    assert out["gain"] == pytest.approx(3.5 / 3)
    assert out["price"] == pytest.approx(0.2)
    assert out["recovered"] == pytest.approx(3.5 / 5.5)
    assert out["recovered_lo"] <= out["recovered"] <= out["recovered_hi"]


def test_summarise_agrees_with_the_adaptations_analysis():
    try:
        import analyse_boba_adaptations as aba
    except Exception as exc:  # pragma: no cover - that module is being edited elsewhere
        pytest.skip(f"analyse_boba_adaptations not importable: {exc}")
    ours = rmf.summarise(TOY, np.random.default_rng(rmf.BOOTSTRAP_SEED))
    theirs = aba.summarise(TOY, np.random.default_rng(rmf.BOOTSTRAP_SEED))
    for key in ("cost", "gain", "price", "recovered", "recovered_lo", "recovered_hi", "wilcoxon_p"):
        assert ours[key] == pytest.approx(theirs[key]), key


def _toy_runs() -> pd.DataFrame:
    rows = []
    hv = {("clean", "observed"): 9.0, ("clean", "posterior_mean"): 8.5, ("clean", "lower_bound"): 8.0,
          ("noisy", "observed"): 6.0, ("noisy", "posterior_mean"): 7.0, ("noisy", "lower_bound"): 7.5}
    for (condition, rule), value in hv.items():
        onsets = [-1] if condition == "clean" else [0, 20]
        for onset in onsets:
            rows.append({"dataset": "p", "acquisition": "qlognehvi", "seed": 7, "condition": condition,
                         "error_model": "gaussian" if condition == "noisy" else "none",
                         "jitter_std": 1.0 if condition == "noisy" else 0.0, "jitter_iteration": onset,
                         "rule": rule, "hv_true": value - (onset == 20), "max_hv": 10.0,
                         "dm_regret": 0.0 if condition == "clean" and rule == "observed" else 0.1,
                         "dm_shortfall": 1.0})
    return pd.DataFrame(rows)


def test_cells_score_every_rule_against_the_standard_clean_run():
    cells = rmf.paired_cells(_toy_runs(), {"p": 2.0})
    assert len(cells) == 2 * 2 * len(rmf.RESPONSES)  # two rules x two onsets x responses
    hv = cells[(cells.response == "hv") & (cells.rule == "lower_bound")].set_index("jitter_iteration")
    assert hv.loc[0, "ref_clean"] == pytest.approx((10 - 9.0) / 2)
    assert hv.loc[0, "trt_clean"] == pytest.approx((10 - 8.0) / 2)
    assert hv.loc[0, "ref_noisy"] == pytest.approx((10 - 6.0) / 2)
    assert hv.loc[20, "trt_noisy"] == pytest.approx((10 - 6.5) / 2)
    table = rmf.recovery_table(cells)
    row = table[(table.response == "hv") & (table.rule == "lower_bound") & (table.jitter_iteration == 0)].iloc[0]
    assert row.cost == pytest.approx(1.5) and row.gain == pytest.approx(0.75) and row.price == pytest.approx(0.5)
    with pytest.raises(ValueError, match="floor gap"):
        rmf.paired_cells(_toy_runs(), {})


# --- end to end on a real (tiny) driver run -----------------------------------


def test_replay_reproduces_the_logged_front_of_a_driver_run(tmp_path):
    out = tmp_path / "mo"
    cmd = [
        sys.executable, str(REPO / "scripts" / "bo_synthetic_error_simulation.py"), "--multi-objective",
        "--functions", "branincurrin", "--acq-list", "random", "--seeds", "7", "--iterations", "12",
        "--initial-samples", "5", "--candidate-pool", "64", "--acq-num-restarts", "2",
        "--acq-raw-samples", "32", "--acq-mc-samples", "32", "--error-models", "gaussian",
        "--jitter-stds", "1.0", "--jitter-iterations", "0,5", "--n-jobs", "1", "--output-dir", str(out),
    ]
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]

    stats = rmf.load_run_stats(out)
    groups, orphans = rmf.discover_groups(out, acquisitions={"random"}, error_model="gaussian",
                                          suffix_for=lambda em, s: "")
    assert len(groups) == 1 and len(groups[0].noisy) == 2 and not orphans
    context = rmf.build_context(stats, ["branincurrin"], n_weights=8, utopia_log2=8)
    frame = rmf.replay_group(groups[0], context)

    assert len(frame) == 3 * len(rmf.RULES) and set(frame.rule) == set(rmf.RULES)
    assert (frame.observed_check_abs_error <= 1e-9).all()
    observed = frame[frame.rule == "observed"]
    assert (observed.n_deployed == observed.k_observed).all()
    assert (frame.n_deployed <= frame.k_observed).all()
    clean = frame[(frame.condition == "clean") & (frame.rule == "observed")].iloc[0]
    assert clean.dm_regret == 0.0 and clean.truly_dominated_share == 0.0

    # A logged value the replayed front does not reproduce must stop the replay.
    path = groups[0].noisy[0][2]
    bad = pd.read_csv(path)
    bad.loc[bad.index[-1], "inference_value_true"] += 1e-3
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inference_value_true"):
        rmf.replay_group(groups[0], context)
