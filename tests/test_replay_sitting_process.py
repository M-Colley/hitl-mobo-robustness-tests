"""The rating process of the end-of-study sitting (scripts/replay_end_of_study.py).

The shared model (the default) holds the drift ramp and the AR(1) state fixed
across a sitting, so they cancel. The sequential process continues them through
trials T-k+1..T exactly as the simulator defines them. These tests pin (1) that
the shared model is the historic per-trial SD, bit for bit, (2) that the
sequential process leaves gaussian and bias alone and adds exactly the
simulator's ramp and AR(1) recursion, started from the run's logged state, and
(3) that a spike run's looks carry that run's own spike probability and size.
"""
from __future__ import annotations

import dataclasses
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
import replay_end_of_study as eos  # noqa: E402

T = 50


def _arm(**kw) -> eos.ArmInfo:
    return eos.ArmInfo("a", Path("a"), T, 5, 0.8, input_arm=False, **kw)


def _noise(model: str, sigma: float = 1.0, onset: int = 20, process: str = "sequential", order: str = "rank",
           variant: str | None = None) -> eos.SittingNoise:
    settings = eos.Settings(sitting_process=process, sitting_order=order)
    return eos.sitting_sd_fn(_arm(), model, sigma, onset, settings, variant=variant)


def _toy_run(name: str = "r.csv", n: int = 40, error: np.ndarray | None = None) -> tuple[eos.RunLog, eos.SearchState]:
    """n distinct designs whose truth falls with their rank, and no GP (the look winner only)."""
    truth = np.linspace(1.0, 0.0, n)
    run = eos.RunLog(name=name, X=np.arange(n, dtype=float).reshape(-1, 1), observed=truth.copy(),
                     deployed=truth.copy(), logged_inference=np.zeros(n), y_opt=2.0,
                     error=np.zeros(n) if error is None else error)
    state = eos.SearchState(n=n, first=np.arange(n), obs_mean=truth.copy(), lcb=truth.copy(), gp=None,
                            train_X=torch.zeros(n, 1), train_Y=torch.zeros(n, 1),
                            bounds=torch.tensor([[0.0], [float(n)]]), gp_failed=True)
    return run, state


def _z(name: str, k: int) -> np.ndarray:
    return np.random.default_rng(eos.noise_seed(name, eos.PROC_TOURNAMENT, k)).standard_normal(k)


# --- settings ------------------------------------------------------------------------------


def test_settings_record_is_the_historic_one_for_the_shared_process():
    historic = {"tournament_k", "confirm_k", "rhos", "slip_look_sd", "alpha", "lcb_beta"}
    assert set(eos.settings_record(eos.Settings())) == historic
    seq = eos.settings_record(eos.Settings(sitting_process="sequential", sitting_order="rank"))
    assert set(seq) == historic | {"sitting_process", "sitting_order"}
    assert seq["sitting_process"] == "sequential" and seq["sitting_order"] == "rank"
    with pytest.raises(ValueError, match="sitting-process"):
        eos.validate_settings(eos.Settings(sitting_process="simultaneous"))
    with pytest.raises(ValueError, match="sitting-order"):
        eos.validate_settings(eos.Settings(sitting_process="sequential", sitting_order="alphabetical"))


def test_cli_passes_the_process_order_and_variants():
    args = eos.parse_args(["--sitting-process", "sequential", "--sitting-order", "rank", "--variants", "sp0.15-20"])
    assert args.sitting_process == "sequential" and args.sitting_order == "rank" and args.variants == "sp0.15-20"
    default = eos.parse_args([])
    assert default.sitting_process == "shared" and default.variants == ""


# --- the shared model is the historic one ----------------------------------------------------


@pytest.mark.parametrize("model", ["gaussian", "bias", "drift", "ar1"])
def test_shared_process_is_the_plain_per_trial_sd(model):
    run, state = _toy_run()
    noise = _noise(model, sigma=0.7, process="shared")
    plain = lambda trial: noise(trial)  # noqa: E731 - what the replay passed before the process existed
    for k in (3, 12):
        assert eos.tournament(run, state, k, (1.0, 0.5), noise, T) == eos.tournament(run, state, k, (1.0, 0.5),
                                                                                       plain, T)
        assert (eos.confirmation(run, state, 2, 5, noise, T, 0.05)
                == eos.confirmation(run, state, 2, 5, plain, T, 0.05))


@pytest.mark.parametrize("model", ["gaussian", "bias"])
def test_sequential_process_leaves_gaussian_and_bias_unchanged(model):
    run, state = _toy_run()
    for order in eos.SITTING_ORDERS:
        seq, shared = _noise(model, order=order), _noise(model, process="shared")
        assert not seq.moves
        for k in (2, 8, 16):
            assert eos.tournament(run, state, k, (1.0, 0.5), seq, T) == eos.tournament(run, state, k, (1.0, 0.5),
                                                                                         shared, T)
        assert eos.confirmation(run, state, 4, 5, seq, T, 0.05) == eos.confirmation(run, state, 4, 5, shared, T, 0.05)


def test_input_arms_are_unchanged_by_the_process():
    slip = dataclasses.replace(_arm(), input_arm=True)
    noise = eos.sitting_sd_fn(slip, "slip", 0.4, 20, eos.Settings(sitting_process="sequential"))
    assert noise.always_on and not noise.moves and noise(3) == 0.25


# --- drift -----------------------------------------------------------------------------------


def test_ramp_is_the_simulators():
    sigma, onset = 5.0, 20
    config = sim.SimulationConfig(
        iterations=T, jitter_iteration=onset, jitter_std=sigma, single_error=False, initial_samples=5,
        candidate_pool=64, objective="value", objective_columns=["value"], param_columns=["x"], seed=0,
        error_model="drift", error_bias=0.0, error_spike_prob=0.1, error_spike_std=0.5,
        dropout_strategy="hold_last", normalize_objective=False, objective_weights=None, acq_num_restarts=2,
        acq_raw_samples=32, acq_maxiter=50, acq_mc_samples=32, ref_point=None,
    )
    noise = _noise("drift", sigma=sigma, onset=onset)
    for t in (onset, onset + 1, 35, T):
        _, err = sim.apply_sensor_error(np.zeros(1), t, config, np.random.default_rng(t), np.zeros(1))
        jitter = 0.0 if t <= onset else np.random.default_rng(t).normal(0.0, sigma, size=1)[0]
        assert err[0] - jitter == pytest.approx(noise.ramp(t), abs=1e-12)
    # the finding's arithmetic: 5 sigma from trial 21, the ramp is 2.5 at trial 35 and 5.0 at trial 50
    assert noise.ramp(35) == pytest.approx(2.5) and noise.ramp(T) == pytest.approx(5.0)


@pytest.mark.parametrize("order", eos.SITTING_ORDERS)
def test_sequential_drift_adds_the_ramp_at_each_candidates_presentation_trial(order):
    run, _ = _toy_run()
    k, rho = 16, 0.5
    seq, shared = _noise("drift", sigma=5.0, order=order), _noise("drift", sigma=5.0, process="shared")
    z = _z(run.name, k)
    e_seq = eos.tournament_look_errors(run, seq, k, k, z, rho, T)
    e_shared = rho * np.array([shared(t) for t in range(T - k + 1, T + 1)]) * z
    if order == "rank":
        position = np.arange(k)
    else:
        position = np.random.default_rng(eos.noise_seed(run.name, eos.PROC_TOURNAMENT, k,
                                                        eos.ORDER_STREAM)).permutation(k)
        assert not np.array_equal(position, np.arange(k))
    ramp = np.array([seq.ramp(T - k + 1 + p) for p in position])
    # the same fresh draw per candidate, scaled by rho; the ramp is not scaled
    np.testing.assert_allclose(e_seq - e_shared, ramp, atol=1e-12)


def test_counterbalanced_confirmation_cancels_a_linear_ramp():
    run, state = _toy_run()
    noise = _noise("drift", sigma=5.0)
    for k in (2, 4, 6):
        start = T - 2 * k + 1
        d_trials = [start + 2 * j + (j % 2) for j in range(k)]
        c_trials = [start + 2 * j + 1 - (j % 2) for j in range(k)]
        z = np.random.default_rng(0).standard_normal(2 * k)
        e = eos.process_errors(noise, np.array(d_trials + c_trials), z, 1.0, 0.0)
        # D and C see the same ramp on average, so the difference of the means is the jitter's alone
        assert e[:k].mean() - e[k:].mean() == pytest.approx(5.0 * (z[:k].mean() - z[k:].mean()), abs=1e-12)


# --- ar(1) ------------------------------------------------------------------------------------


def test_sequential_ar1_continues_the_logged_state():
    k, sigma = 8, 2.0
    error = np.zeros(T)
    error[T - k - 1] = 3.0            # the loop's error at trial T - k, the last searched trial
    run, _ = _toy_run(error=error, n=T)
    z = _z(run.name, k)
    for rho in (1.0, 0.5):
        e = eos.tournament_look_errors(run, _noise("ar1", sigma=sigma), k, k, z, rho, T)
        prev = 3.0
        for j in range(k):          # rank order: candidate j is shown at trial T - k + 1 + j
            prev = 0.8 * prev + rho * sigma * 0.6 * z[j]
            assert e[j] == pytest.approx(prev, abs=1e-12)
    # random order: the candidate shown at position p carries the state carried to that position
    noise = _noise("ar1", sigma=sigma, order="random")
    e = eos.tournament_look_errors(run, noise, k, k, z, 1.0, T)
    pos = np.random.default_rng(eos.noise_seed(run.name, eos.PROC_TOURNAMENT, k, eos.ORDER_STREAM)).permutation(k)
    shown = np.argsort(pos)         # candidate shown at each position
    prev = 3.0
    for p in range(k):
        prev = 0.8 * prev + sigma * 0.6 * z[shown[p]]
        assert e[shown[p]] == pytest.approx(prev, abs=1e-12)


def test_ar1_recursion_is_the_simulators():
    sigma, onset, rho = 1.5, 20, 0.8
    config = sim.SimulationConfig(
        iterations=T, jitter_iteration=onset, jitter_std=sigma, single_error=False, initial_samples=5,
        candidate_pool=64, objective="value", objective_columns=["value"], param_columns=["x"], seed=0,
        error_model="ar1", error_bias=0.0, error_spike_prob=0.1, error_spike_std=0.5,
        dropout_strategy="hold_last", normalize_objective=False, objective_weights=None, acq_num_restarts=2,
        acq_raw_samples=32, acq_maxiter=50, acq_mc_samples=32, ref_point=None, error_ar1_rho=rho,
    )
    noise = _noise("ar1", sigma=sigma, onset=onset)
    rng, twin = np.random.default_rng(5), np.random.default_rng(5)
    prev, sim_errors, z = None, [], []
    trials = np.arange(onset + 1, onset + 11)   # the first post-onset trial is a stationary draw
    for t in trials:
        _, err = sim.apply_sensor_error(np.zeros(1), int(t), config, rng, np.zeros(1), previous_error=prev)
        prev = err
        sim_errors.append(err[0])
        jitter = twin.normal(0.0, sigma)
        if t == onset + 1:
            z.append(jitter / sigma)
        else:
            z.append(twin.normal(0.0, sigma * np.sqrt(1 - rho ** 2)) / (sigma * np.sqrt(1 - rho ** 2)))
    ours = eos.process_errors(noise, trials, np.array(z), 1.0, prev_error=0.0)
    np.testing.assert_allclose(ours, sim_errors, atol=1e-12)


def test_ar1_look_differences_have_the_processes_variance():
    # Var(e_i - e_{i+L}) = 2 sigma^2 (1 - 0.8^L) under the process, 0.72 sigma^2 at every lag when shared
    rng = np.random.default_rng(1)
    k, n = 16, 20000
    noise = _noise("ar1", sigma=1.0, onset=0)
    trials = np.arange(T - k + 1, T + 1)
    draws = np.array([eos.process_errors(noise, trials, rng.standard_normal(k), 1.0, rng.standard_normal())
                      for _ in range(n)])
    for lag in (1, 5, 15):
        var = np.var(draws[:, lag] - draws[:, 0])
        assert var == pytest.approx(2 * (1 - 0.8 ** lag), rel=0.05)
    assert 2 * (1 - 0.8 ** 15) == pytest.approx(1.93, abs=0.01)


def test_logged_error_is_the_state_the_loop_carried(tmp_path):
    n = 6
    df = pd.DataFrame({"iteration": np.arange(1, n + 1), "x": np.arange(n, dtype=float), "param_columns": "x",
                       "objective_true": np.linspace(0, 1, n), "objective_observed": np.linspace(0, 1, n) + 0.25,
                       "error_magnitude": 0.25, "inference_simple_regret_true": 0.0, "y_opt": 2.0})
    path = tmp_path / "run.csv"
    df.to_csv(path, index=False)
    run = eos.read_run(path, n)
    np.testing.assert_allclose(run.error, 0.25)
    eos.check_logged_error(run, n)
    bad = dataclasses.replace(run, error=run.error + 1e-3)
    with pytest.raises(eos.ReproductionError, match="differs"):
        eos.check_logged_error(bad, n)
    with pytest.raises(ValueError, match="no error_magnitude"):
        eos._prefix_error(dataclasses.replace(run, error=None), 3)


def test_sitting_that_starts_at_the_onset_gets_a_stationary_first_look():
    # k = 30 from trial 21: the prefix is error-free and the first look is the first corrupted trial
    run, _ = _toy_run(n=T)
    k = 30
    z = _z(run.name, k)
    e = eos.tournament_look_errors(run, _noise("ar1", sigma=1.0, onset=20), k, k, z, 1.0, T)
    assert e[0] == pytest.approx(z[0])
    assert e[1] == pytest.approx(0.8 * z[0] + 0.6 * z[1])


# --- spikes -----------------------------------------------------------------------------------


def test_spike_spec_is_read_from_the_variant():
    assert eos.spike_spec("sp0.15-20") == (0.15, 20.0)
    assert eos.spike_spec("sp0.05-5_single") == (0.05, 5.0)
    assert eos.spike_spec("sp0.1-0.25") == (0.1, 0.25)       # a scaled spike names the magnitude
    for bad in (None, "", "bias1", "single"):
        with pytest.raises(ValueError):
            eos.spike_spec(bad)


def test_spike_look_sd_is_the_runs_marginal_sd():
    noise = _noise("spike", sigma=0.25, onset=0, process="shared", variant="sp0.15-20")
    assert noise(1) == pytest.approx(np.sqrt(0.25 ** 2 + 0.15 * 20.0 ** 2))
    assert noise(1) == pytest.approx(7.75, abs=0.01)          # the finding's value, not 0.268
    with pytest.raises(ValueError, match="variant"):
        eos.sitting_sd_fn(_arm(), "spike", 0.25, 0, eos.Settings())
    # without the run's parameters idiosyncratic_sd keeps its old assumption (scaled, p = 0.15)
    assert eos.idiosyncratic_sd("spike", 1.0) == pytest.approx((1.0 + eos.SPIKE_PROB_DEFAULT) ** 0.5)


def test_a_stem_carries_its_one_spike_variant_for_callers_that_pass_none(monkeypatch, tmp_path):
    # replay_hitl_remedies.replay_stem_rows calls sitting_sd_fn without variant=; the
    # task's settings carry the stem's spike variant, so its looks still get the run's spikes.
    def rec(stem: str, variant: str, model: str = "spike") -> dict:
        return {"dataset": "d", "acquisition": "logei", "seed": 12, "jitter_iteration": 0, "jitter_std": 0.25,
                "error_model": model, "variant": variant, "stem": stem, "path": tmp_path / f"{stem}_{variant}.csv"}
    jittered = [rec("s1", "sp0.15-20"), rec("s2", "sp0.15-20"), rec("s2", "sp0.05-5"), rec("s3", "", "gaussian")]
    baselines = {s: tmp_path / f"{s}_clean.csv" for s in ("s1", "s2", "s3")}
    monkeypatch.setattr(eos.aer, "index_runs", lambda root: (baselines, jittered))
    arm, settings = _arm(), eos.Settings()
    tasks, _ = eos.build_tasks(arm, eos.Filters(variants=frozenset({"sp0.15-20", "sp0.05-5", ""})), settings,
                               tmp_path / "out")
    by = {t["stem"]: t["settings"] for t in tasks}
    assert by["s1"].spike_variant == "sp0.15-20"
    assert by["s2"].spike_variant is None and by["s3"].spike_variant is None   # two variants; none
    assert by["s3"] == settings
    fallback = eos.sitting_sd_fn(arm, "spike", 0.25, 0, by["s1"])
    assert (fallback.spike_prob, fallback.spike_std) == (0.15, 20.0)
    assert fallback == eos.sitting_sd_fn(arm, "spike", 0.25, 0, settings, variant="sp0.15-20")
    with pytest.raises(ValueError, match="variant"):
        eos.sitting_sd_fn(arm, "spike", 0.25, 0, by["s2"])
    explicit = eos.sitting_sd_fn(arm, "spike", 0.25, 0, by["s1"], variant="sp0.05-5")   # an explicit variant wins
    assert (explicit.spike_prob, explicit.spike_std) == (0.05, 5.0)
    assert eos.settings_record(by["s1"]) == eos.settings_record(settings)   # never written to settings.json


def test_a_directory_mixing_spike_variants_in_a_cell_is_refused(tmp_path):
    # The summaries do not split by variant, so two spike sizes in one cell would be averaged.
    base = {"arm": "a", "dataset": "d", "acquisition": "logei", "error_model": "spike", "jitter_std": 0.25,
            "jitter_iteration": 0, "procedure": "standard", "regret_noisy": 1.0}
    for i, variant in enumerate(("sp0.15-20", "sp0.05-5")):
        path = tmp_path / "per_run" / "a" / "d" / f"run{i}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{**base, "seed": i, "variant": variant, "file": f"run{i}.csv"}]).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="mixes run variants"):
        eos.load_per_run(tmp_path, ["a"], eos.Filters())
    one = eos.load_per_run(tmp_path, ["a"], eos.Filters(variants=frozenset({"sp0.15-20"})))
    assert one["variant"].tolist() == ["sp0.15-20"]


def test_spikes_are_drawn_per_look_at_the_runs_rate_and_size():
    run, _ = _toy_run()
    noise = _noise("spike", sigma=0.25, onset=0, process="shared", variant="sp0.15-20")
    k, errors = 8, []
    for i in range(3000):
        r = dataclasses.replace(run, name=f"r{i}.csv")
        errors.append(eos.tournament_look_errors(r, noise, k, k, _z(r.name, k), 1.0, T))
    errors = np.concatenate(errors)
    assert np.mean(np.abs(errors) > 2.0) == pytest.approx(0.15 * 0.92, abs=0.02)   # |N(0, 20)| > 2 w.p. 0.92
    assert np.var(errors) == pytest.approx(0.25 ** 2 + 0.15 * 400.0, rel=0.08)
    # the spike stream is its own: the Gaussian part is the shared model's draw
    quiet = eos.tournament_look_errors(run, dataclasses.replace(noise, spike_prob=0.0), k, k, _z(run.name, k), 1.0, T)
    np.testing.assert_allclose(quiet, 0.25 * _z(run.name, k))
