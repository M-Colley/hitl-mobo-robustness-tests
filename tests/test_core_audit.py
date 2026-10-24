"""Small deterministic checks for the human-error helpers; no benchmark runs."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bo_sensor_error_simulation as sim


def _config(**overrides):
    values = dict(
        iterations=8, jitter_iteration=2, jitter_std=1.0, single_error=False,
        initial_samples=2, candidate_pool=8, objective="value",
        objective_columns=["value"], param_columns=["x"], seed=7,
        error_model="dropout", error_bias=0.0, error_spike_prob=0.1,
        error_spike_std=2.0, dropout_strategy="hold_last",
        normalize_objective=False, objective_weights=None, acq_num_restarts=1,
        acq_raw_samples=8, acq_maxiter=10, acq_mc_samples=8, ref_point=None,
    )
    values.update(overrides)
    return sim.SimulationConfig(**values)


def test_dropout_metadata_does_not_reject_the_learned_noise_model():
    config = _config()
    assert sim.logged_noise_variance(2, config, True) == 1e-6
    assert np.isnan(sim.logged_noise_variance(3, config, True))
    assert sim.logged_noise_variance(3, config, False) == 1e-6
    single = replace(config, single_error=True)
    assert np.isnan(sim.logged_noise_variance(3, single, True))
    assert sim.logged_noise_variance(4, single, True) == 1e-6


def test_dropout_cannot_claim_a_known_additive_variance():
    config = _config(observation_noise="known")
    with pytest.raises(ValueError, match="dropout"):
        sim.logged_noise_variance(3, config, True)
    with pytest.raises(ValueError, match="dropout"):
        sim.known_noise_variance(3, replace(config, observation_noise="learned"), True)


def test_logging_the_variance_preserves_the_other_error_processes():
    for process in ("gaussian", "bias", "spike", "drift", "ar1", "none"):
        config = _config(error_model=process)
        for iteration in (1, 3, 8):
            assert sim.logged_noise_variance(iteration, config, True) == sim.known_noise_variance(
                iteration, config, True)


def test_two_interleaved_anchors_identify_a_linear_drift_without_attenuation():
    # Anchor levels differ by ten. Their schedule is correlated with anchor
    # identity, which must not attenuate an otherwise exact slope of 2/trial.
    true = np.asarray([-10.0, -20.0, 4.0, 2.0, -10.0, -20.0, 3.0])
    observed = true + 2.0 * np.arange(len(true))
    rows = [0, 1, 4, 5]
    identities = [0, 1, 0, 1]
    correction = sim.anchor_detrend(rows, identities, observed.tolist(), len(true))
    assert correction == pytest.approx(2.0 * (np.arange(len(true)) - np.mean(rows)))
    corrected = sim.anchor_corrected_ratings(rows, identities, observed.tolist())
    assert np.asarray(corrected) - true == pytest.approx(np.full(len(true), 5.0))
    # Raw ratings ship the later, worse design. The corrected ratings used by
    # deployment recover the better design without access to its true value.
    assert int(np.argmax(observed)) == 6
    assert int(np.argmax(corrected)) == 2


def test_detrending_does_not_invent_drift_without_repeat_anchors():
    observed = [2.0, -5.0, 9.0]
    assert sim.anchor_corrected_ratings([0, 1, 2], [0, 1, 2], observed) == observed
    assert sim.anchor_corrected_ratings([], [], observed) == observed


def test_constant_anchor_ratings_do_not_move_the_deployment_values():
    observed = [2.0, -5.0, 4.0, 2.0, -5.0]
    corrected = sim.anchor_corrected_ratings([0, 1, 3, 4], [0, 1, 0, 1], observed)
    assert corrected == observed


def _fingerprint(config=None, acquisition=None, context=None):
    return sim.run_config_fingerprint(
        config or _config(), acquisition or sim.AcquisitionConfig(name="logei", xi=0.01, kappa=2),
        bounds=sim.Bounds(low=np.asarray([0.0]), high=np.asarray([1.0])),
        apply_error=True, oracle_model="exact", y_opt=3.0, context=context,
    )


@pytest.mark.parametrize("change", [
    {"initial_samples": 3}, {"candidate_pool": 16}, {"inference_rule": "best_mean"},
    {"replicate_first": 2}, {"likelihood": "student_t"}, {"acq_maxiter": 11},
])
def test_resume_identity_includes_settings_missing_from_the_filename(change):
    assert _fingerprint(replace(_config(), **change)) != _fingerprint()


def test_resume_identity_includes_acquisition_and_objective_context():
    assert _fingerprint(acquisition=sim.AcquisitionConfig(name="logei", xi=0.2, kappa=2)) != _fingerprint()
    assert _fingerprint(context={"statistics": {"mean": 0, "std": 1}}) != _fingerprint(
        context={"statistics": {"mean": 0, "std": 2}})
    assert _fingerprint(context={"a": [1, 2], "b": 3}) == _fingerprint(context={"b": 3, "a": [1, 2]})


@pytest.mark.parametrize("likelihood", ["student_t", "student_t_rbf"])
def test_student_t_resume_identity_tracks_the_external_surrogate(monkeypatch, tmp_path, likelihood):
    config = replace(_config(), likelihood=likelihood)
    expected = _fingerprint(config=config)
    gaussian = _fingerprint()
    original = sim.source_sha256
    robust_source = Path(sim.__file__).with_name("robust_gp.py").resolve()
    changed = "changed-surrogate-implementation"
    monkeypatch.setattr(
        sim, "source_sha256",
        lambda path: changed if Path(path).resolve() == robust_source else original(path),
    )
    actual = _fingerprint(config=config)
    assert actual != expected
    assert _fingerprint() == gaussian
    assert "robust_gp_sha256" not in sim._resume_environment()
    path = tmp_path / "student_t.csv"
    _complete_log(expected).to_csv(path, index=False)
    with pytest.raises(ValueError, match="different configuration"):
        sim._load_resumable_run(path, 3, expected_fingerprint=actual)


@pytest.mark.parametrize("observation_noise", ["known", "self_report"])
@pytest.mark.parametrize("apply_error", [False, True])
def test_multi_objective_simulation_rejects_unsupported_noise_modes(observation_noise, apply_error):
    config = replace(
        _config(), objective="multi_objective", objective_columns=["a", "b"],
        observation_noise=observation_noise, ref_point=np.asarray([-1.0, -1.0]),
    )
    with pytest.raises(ValueError, match="single-objective only"):
        sim.run_simulation(
            oracle=None,
            bounds=sim.Bounds(low=np.asarray([0.0]), high=np.asarray([1.0])),
            config=config, acq=sim.AcquisitionConfig(name="qehvi"),
            rng=np.random.default_rng(7), jitter_rng=np.random.default_rng(8),
            run_id="unsupported", apply_error=apply_error, oracle_model="exact", y_opt=4.0,
        )


@pytest.mark.parametrize("observation_noise", ["known", "self_report"])
def test_multi_objective_driver_rejects_unsupported_noise_modes_before_loading_stats(
    monkeypatch, tmp_path, observation_noise,
):
    import bo_synthetic_error_simulation as synthetic

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid noise mode must fail before loading objectives")

    monkeypatch.setattr(synthetic.mob, "load_mo_stats", forbidden)
    with pytest.raises(ValueError, match="single-objective only"):
        synthetic.main([
            "--multi-objective", "--functions", "branincurrin", "--acq", "qehvi",
            "--observation-noise", observation_noise, "--dry-run",
            "--output-dir", str(tmp_path / "unused"),
        ])
    assert not (tmp_path / "unused").exists()


def _complete_log(fingerprint=None):
    frame = pd.DataFrame({
        "iteration": [1, 2, 3], "run_id": "r", "jitter_std": 1.0, "error_model": "gaussian",
        "simple_regret_true": [2.0, 1.0, 0.5], "inference_simple_regret_true": [2.0, 1.0, 0.5],
        "regret_cum_true": [2.0, 3.0, 3.5], "regret_avg_true": [2.0, 1.5, 3.5 / 3],
        "best_true_so_far": [1.0, 2.0, 2.5], "acq_opt_failed": False, "dataset": "toy",
    })
    if fingerprint is not None:
        frame["run_config_fingerprint"] = fingerprint
    return frame


def test_resume_requires_the_matching_fingerprint_even_with_legacy_override(tmp_path):
    path = tmp_path / "run.csv"
    _complete_log("expected").to_csv(path, index=False)
    assert sim._load_resumable_run(path, 3, expected_fingerprint="expected") is not None
    for allow_legacy in (False, True):
        with pytest.raises(ValueError, match="different configuration"):
            sim._load_resumable_run(path, 3, expected_fingerprint="other", allow_legacy=allow_legacy)


def test_legacy_resume_requires_an_explicit_override(tmp_path):
    path = tmp_path / "run.csv"
    _complete_log().to_csv(path, index=False)
    with pytest.raises(ValueError, match="allow-legacy-resume"):
        sim._load_resumable_run(path, 3, expected_fingerprint="expected")
    with pytest.warns(RuntimeWarning, match="unverified legacy"):
        assert sim._load_resumable_run(path, 3, expected_fingerprint="expected", allow_legacy=True) is not None
    # Analysis and other legacy readers can still load historical files.
    assert sim._load_resumable_run(path, 3) is not None


def _timing_log(acquisition="logei", onset=0):
    iteration = np.arange(1, 9)
    return pd.DataFrame({
        "iteration": iteration, "x": iteration.astype(float) ** 2, "param_columns": "x",
        "run_id": "r", "dataset": "toy", "objective": "value", "acquisition": acquisition,
        "seed": 7, "oracle_model": "exact", "error_model": "gaussian", "jitter_std": 1.0,
        "jitter_iteration": onset, "initial_samples": 5, "best_true_so_far": iteration.astype(float),
        "simple_regret_true": 9.0 - iteration, "inference_simple_regret_true": 9.0 - iteration,
        "regret_cum_true": iteration.astype(float), "regret_avg_true": 1.0, "acq_opt_failed": False,
    })


def test_onset_zero_reaction_waits_for_a_model_based_proposal_without_moving_auc():
    import evaluate_research_question as evaluation

    log = _timing_log()
    row = evaluation.build_response_table(log).iloc[0]
    assert (row["response_start_iteration"], row["response_end_iteration"]) == (5, 6)
    assert row["response_l2"] == 11.0
    assert sim.summarize_adjustment(log, 0, ["x"])["delta_l2_norm"] == 11.0
    assert row["auc_simple_regret_true_postonset_per_iter"] == pytest.approx(4.5)


def test_legacy_gp_fit_timing_identifies_the_end_of_initialization():
    from _feedback_timing import response_step

    log = _timing_log().drop(columns="initial_samples")
    log["fit_time_sec"] = [0, 0, 0, 0, 0, 0.2, 0.2, 0.2]
    assert response_step(log, 0) == (5, 6)
    assert response_step(log, 5) == (6, 7)


@pytest.mark.parametrize("acquisition", ["random", "sobol"])
def test_model_free_search_has_no_reaction_but_keeps_its_regret_rows(acquisition):
    import evaluate_research_question as evaluation

    log = _timing_log(acquisition)
    row = evaluation.build_response_table(log).iloc[0]
    assert np.isnan(row["response_l2"])
    assert np.isnan(sim.summarize_adjustment(log, 0, ["x"])["delta_l2_norm"])
    assert row["final_simple_regret_true"] == 1.0
    assert row["auc_simple_regret_true_postonset_per_iter"] == pytest.approx(4.5)


def test_error_on_the_last_rating_keeps_the_deployment_result():
    import evaluate_research_question as evaluation

    row = evaluation.build_response_table(_timing_log(onset=7)).iloc[0]
    assert row["final_inference_simple_regret_true"] == 1.0
    assert np.isnan(row["response_l2"])
    assert row["auc_simple_regret_true_postonset_per_iter"] == 0.0


def test_fitted_oracle_library_versions_are_part_of_resume_identity(monkeypatch):
    environment = sim._resume_environment()
    libraries = {"scikit-learn", "xgboost", "lightgbm", "catboost", "tabpfn", "linear_operator"}
    assert libraries <= environment["packages"].keys()
    assert environment["platform"] and environment["machine"]
    assert len(environment["timing_sha256"]) == 64
    original = _fingerprint()
    for library in libraries:
        updated = {**environment, "packages": {**environment["packages"], library: "changed-version"}}
        monkeypatch.setattr(sim, "_resume_environment", lambda value=updated: value)
        assert _fingerprint() != original, library


def test_fingerprint_keeps_undefined_diagnostics_without_accepting_invalid_run_settings():
    diagnostics = {"statistics": {"y_best": -float("inf"), "opt_gap": float("nan"),
                                  "nested": [np.asarray([float("inf"), float("nan")])]}}
    same = {"statistics": {"nested": [[float("inf"), np.float64("nan")]],
                           "opt_gap": np.float64("nan"), "y_best": np.float64("-inf")}}
    assert _fingerprint(context=diagnostics) == _fingerprint(context=same)
    assert _fingerprint(context=diagnostics) != _fingerprint(
        context={"statistics": {**diagnostics["statistics"], "y_best": float("inf")}})
    assert _fingerprint(context=diagnostics) != _fingerprint(
        context={"statistics": {**diagnostics["statistics"], "opt_gap": None}})
    with pytest.raises(ValueError, match="Out of range float"):
        _fingerprint(config=replace(_config(), jitter_std=float("nan")))


def _tiny_sweep_args(directory):
    return ["--functions", "branin", "--acq", "random", "--iterations", "4",
            "--initial-samples", "2", "--num-seeds", "1", "--error-models", "gaussian",
            "--jitter-stds", "1", "--jitter-iterations", "0", "--n-jobs", "1",
            "--output-dir", str(directory)]


@pytest.mark.parametrize("failure_stage", ["guard", "task"])
def test_failed_sweep_cannot_retain_a_previous_completion_marker(tmp_path, monkeypatch, failure_stage):
    import bo_synthetic_error_simulation as synthetic

    marker = tmp_path / "SWEEP_COMPLETE"
    marker.write_text("previous successful invocation", encoding="utf-8")

    def fail(*args, **kwargs):
        raise ValueError("deliberate failed invocation")

    target = "_guard_clean_run_settings" if failure_stage == "guard" else "run_task"
    monkeypatch.setattr(synthetic, target, fail)
    expected = ValueError if failure_stage == "guard" else SystemExit
    with pytest.raises(expected):
        synthetic.main(_tiny_sweep_args(tmp_path))
    assert not marker.exists()


def test_dry_run_preserves_the_previous_completion_marker(tmp_path):
    import bo_synthetic_error_simulation as synthetic

    marker = tmp_path / "SWEEP_COMPLETE"
    marker.write_text("previous successful invocation", encoding="utf-8")
    synthetic.main([*_tiny_sweep_args(tmp_path), "--dry-run"])
    assert marker.read_text(encoding="utf-8") == "previous successful invocation"


def test_bump_with_undefined_diagnostics_runs_and_resumes_four_random_trials(tmp_path):
    import bo_synthetic_error_simulation as synthetic

    function = "bump_a4_w0.05"
    stats = synthetic.bb.load_stats(synthetic.bb.DEFAULT_STATS_PATH)[function]
    assert np.isneginf(stats["y_best"]) and np.isnan(stats["opt_gap"])
    args = synthetic.parse_args(_tiny_sweep_args(tmp_path))
    task = synthetic.Task(function=function, seed=7)
    summaries, diagnostics = synthetic.run_task(task, args, stats, ["random"], ["gaussian"], [1.0], [0], tmp_path)
    assert len(summaries) == len(diagnostics) == 2
    files = sorted(tmp_path.glob("bo_sensor_error_*.csv"))
    assert len(files) == 2
    before = {path: path.read_bytes() for path in files}
    frames = [pd.read_csv(path) for path in files]
    for frame in frames:
        assert len(frame) == 4
        assert frame["run_config_fingerprint"].str.fullmatch("[0-9a-f]{64}").all()
    np.testing.assert_array_equal(frames[0]["simple_regret_true"], frames[1]["simple_regret_true"])
    args.resume = True
    resumed, _ = synthetic.run_task(task, args, stats, ["random"], ["gaussian"], [1.0], [0], tmp_path)
    assert {row["run_id"] for row in resumed} == {row["run_id"] for row in summaries}
    assert {path: path.read_bytes() for path in files} == before


def test_historical_fixture_accepts_only_valid_new_timing_metadata():
    import _reference_fixtures as reference

    record = reference.backup_fixture()["runs"]["mo_default_run"]
    frame = pd.DataFrame(record["data"], columns=record["columns"])
    for column, kind in record["kinds"].items():
        frame[column] = frame[column].astype({"float": float, "int": int, "bool": bool, "str": str}[kind])
    frame["initial_samples"] = 5
    reference.assert_matches_backup_run(frame)
    invalid = frame.copy()
    invalid["initial_samples"] = 4
    with pytest.raises(AssertionError, match="recorded recipe"):
        reference.assert_matches_backup_run(invalid)
    invalid = frame.copy()
    invalid["unexpected_metadata"] = "cannot silently ignore this"
    with pytest.raises(AssertionError):
        reference.assert_matches_backup_run(invalid)
    invalid = frame.copy()
    invalid.loc[0, "iteration"] += 1
    with pytest.raises(AssertionError):
        reference.assert_matches_backup_run(invalid)
