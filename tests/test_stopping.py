"""The stopping-rule replay (scripts/replay_stopping.py).

The replay is only worth anything if (1) a stopped study really is the logged
run truncated, (2) the frozen-hyperparameter residuals are exact GP
conditioning, (3) the rules stop where their definitions say and never spend
more than T trials, and (4) tuning honours the false-alarm cap. Each is pinned
here on tiny inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bo_sensor_error_simulation as sim  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
import replay_stopping as rs  # noqa: E402

STATS = bb.load_stats()


def _config(iterations: int, seed: int = 7, **overrides) -> sim.SimulationConfig:
    spec = bb.BENCHMARKS["branin"]
    base = dict(
        iterations=iterations, jitter_iteration=0, jitter_std=1.0, single_error=False, initial_samples=4,
        candidate_pool=64, objective="value", objective_columns=["value"], param_columns=spec.param_columns,
        seed=seed, error_model="gaussian", error_bias=1.0, error_spike_prob=0.1, error_spike_std=1.0,
        dropout_strategy="hold_last", normalize_objective=False, objective_weights=None, acq_num_restarts=2,
        acq_raw_samples=32, acq_maxiter=30, acq_mc_samples=32, ref_point=None,
    )
    base.update(overrides)
    return sim.SimulationConfig(**base)


def _simulate(iterations: int, seed: int = 7, **overrides) -> pd.DataFrame:
    spec = bb.BENCHMARKS["branin"]
    torch.manual_seed(seed)
    return sim.run_simulation(
        oracle=bb.SyntheticOracle.from_stats("branin", STATS),
        bounds=sim.Bounds(low=spec.bounds_low, high=spec.bounds_high),
        config=_config(iterations, seed, **overrides), acq=sim.AcquisitionConfig(name="logei"),
        rng=np.random.default_rng(seed), jitter_rng=np.random.default_rng(99), run_id="t", apply_error=True,
        oracle_model="exact", y_opt=(STATS["branin"]["y_opt"] - STATS["branin"]["mean"]) / STATS["branin"]["std"],
    )


def _branin_data(n: int, seed: int = 0, noise: float = 0.0):
    spec = bb.BENCHMARKS["branin"]
    rng = np.random.default_rng(seed)
    X = spec.bounds_low + rng.random((n, 2)) * (spec.bounds_high - spec.bounds_low)
    y = bb.SyntheticOracle.from_stats("branin", STATS).predict_many(X)[:, 0] + rng.normal(0, noise, n)
    bounds = sim.Bounds(low=spec.bounds_low, high=spec.bounds_high).tensor
    return torch.tensor(X), torch.tensor(y.reshape(-1, 1)), y, bounds


# --- (1) a stopped study is the logged run truncated -------------------------


def test_a_shorter_budget_reproduces_the_prefix_of_a_longer_run():
    """The premise of the whole replay: stopping at t changes nothing before t."""
    long, short = _simulate(9), _simulate(6)
    cols = ["x0", "x1", "objective_true", "objective_observed", "inference_simple_regret_true"]
    pd.testing.assert_frame_equal(long[cols].iloc[:6].reset_index(drop=True), short[cols])


# --- (2) the frozen-hyperparameter residuals ----------------------------------


def test_one_step_residuals_are_exact_gp_conditioning():
    X, Y, _, bounds = _branin_data(22, noise=0.1)
    n0 = 12
    gp, ok, _ = rs.fit_loop_gp(X[:n0], Y[:n0], bounds, "t")
    assert ok
    z = rs.one_step_residuals(gp, X, Y, n0)
    assert z.shape == (22 - n0,)
    with torch.no_grad():
        gp.posterior(X[:1])  # BoTorch needs its prediction caches before conditioning
        noise_raw = gp.likelihood.noise.reshape(-1)[0] * gp.outcome_transform.stdvs.reshape(-1)[0] ** 2
        for t in (n0, n0 + 1, 17, 21):  # 0-based: trial t + 1 given trials 1..t
            model = gp.condition_on_observations(X[n0:t], Y[n0:t]) if t > n0 else gp
            post = model.posterior(X[t:t + 1])
            ref = (Y[t, 0] - post.mean.reshape(())) / torch.sqrt(post.variance.reshape(()) + noise_raw)
            assert z[t - n0] == pytest.approx(float(ref), rel=1e-5, abs=1e-6)


def test_residuals_ignore_everything_after_the_trial_they_score():
    X, Y, _, bounds = _branin_data(20, noise=0.1)
    gp, _, _ = rs.fit_loop_gp(X[:10], Y[:10], bounds, "t")
    z = rs.one_step_residuals(gp, X, Y, 10)
    Y2 = Y.clone()
    Y2[15:] += 50.0
    z2 = rs.one_step_residuals(gp, X, Y2, 10)
    np.testing.assert_allclose(z[:5], z2[:5])
    assert abs(z2[5]) > 10 * abs(z[5])


def test_ship_rule_is_argmax_of_mean_minus_one_latent_sd():
    X, Y, y, bounds = _branin_data(15, noise=0.3)
    gp, ok, n_warn = rs.fit_loop_gp(X, Y, bounds, "t")
    rec = rs.ship_record(gp, X, 15, y, y_opt=5.0, ok=ok, n_warn=n_warn)
    with torch.no_grad():
        post = gp.posterior(X)
    lcb = post.mean.reshape(-1) - post.variance.reshape(-1).sqrt()
    assert rec["ship_idx"] == int(torch.argmax(lcb))
    assert rec["regret"] == pytest.approx(5.0 - y[rec["ship_idx"]])
    assert rec["max_ucb"] >= rec["lcb_ship"]


# --- (3) the rules ---------------------------------------------------------------


def test_cusum_resets_at_zero_and_alarms_strictly_above_h():
    z2 = np.array([[0.0, 5.0, 0.0, 0.0, 4.0, 4.0]])  # increments with kappa 1: -2, 3, -2, -2, 2, 2
    S = rs.cusum_paths(z2, kappa=1.0)
    np.testing.assert_allclose(S[0], [0, 3, 1, 0, 2, 4])
    assert rs.first_alarm(S, h=3.0, n0=15)[0] == 15 + 6  # S = 3 is not > 3; S = 4 at column 5
    assert rs.first_alarm(S, h=2.5, n0=15)[0] == 15 + 2
    assert rs.first_alarm(S, h=10.0, n0=15)[0] == 0
    assert rs.first_alarm(S, h=rs.NEVER, n0=15)[0] == 0


def _fit(design, max_ucb, lcb, y_std=2.0):
    return dict(ship_design=design, max_ucb=max_ucb, lcb_ship=lcb, y_std=y_std)


def test_rule_b_needs_an_unchanged_design_and_a_small_gap():
    cps = [20, 30, 40]
    same = [_fit([1.0], 1.0, 0.8), _fit([1.0], 1.0, 0.8), _fit([1.0], 1.0, 0.8)]
    assert rs.rule_b_stop(same, cps, delta=0.25, units="landscape") == 30  # gap 0.2 < 0.25
    assert rs.rule_b_stop(same, cps, delta=0.15, units="landscape") is None
    assert rs.rule_b_stop(same, cps, delta=0.15, units="model") == 30  # 0.2 / 2 = 0.1
    moved = [_fit([1.0], 1.0, 0.9), _fit([2.0], 1.0, 0.9), _fit([2.0], 1.0, 0.9)]
    assert rs.rule_b_stop(moved, cps, delta=0.25, units="landscape") == 40


def _log_frame(T: int, dataset: str, seed: int, noisy: bool, rng: np.random.Generator) -> pd.DataFrame:
    """A log with the simulator's schema. The replay never asks how designs were chosen."""
    spec = bb.BENCHMARKS[dataset]
    oracle = bb.SyntheticOracle.from_stats(dataset, STATS)
    X = spec.bounds_low + rng.random((T, spec.dim)) * (spec.bounds_high - spec.bounds_low)
    true = oracle.predict_many(X)[:, 0]
    err = np.where(np.arange(1, T + 1) > 0, rng.normal(0, 1.0, T), 0.0) if noisy else np.zeros(T)
    obs = true + err
    y_opt = oracle.y_opt
    frame = pd.DataFrame(X, columns=spec.param_columns)
    frame.insert(0, "iteration", np.arange(1, T + 1))
    frame["objective_true"], frame["objective_observed"], frame["error_magnitude"] = true, obs, err
    frame["y_opt"] = y_opt
    frame["inference_simple_regret_true"] = [y_opt - true[int(np.argmax(obs[:t]))] for t in range(1, T + 1)]
    frame["param_columns"] = ",".join(spec.param_columns)
    return frame


def _write_logs(root: Path, T: int, seeds: list[int]) -> None:
    rng = np.random.default_rng(3)
    for dataset in ("branin", "hartmann_3"):
        folder = root / dataset
        folder.mkdir(parents=True)
        for seed in seeds:
            stem = f"bo_sensor_error_{dataset}_value_logei_seed{seed}"
            _log_frame(T, dataset, seed, False, rng).to_csv(folder / f"{stem}_baseline_exact.csv", index=False)
            for onset in (0, 8):
                frame = _log_frame(T, dataset, seed, True, rng)
                frame["error_magnitude"] = np.where(frame["iteration"] > onset, frame["error_magnitude"], 0.0)
                frame["objective_observed"] = frame["objective_true"] + frame["error_magnitude"]
                obs, true = frame["objective_observed"].to_numpy(), frame["objective_true"].to_numpy()
                frame["inference_simple_regret_true"] = [frame["y_opt"].iloc[0] - true[int(np.argmax(obs[:t]))]
                                                         for t in range(1, T + 1)]
                frame.to_csv(folder / f"{stem}_jittered_exact_gaussian_jit{onset}_std1.0.csv", index=False)
    (root / "run_metadata.json").write_text(json.dumps({"args": {"iterations": T}}), encoding="utf-8")


def test_end_to_end_never_spends_more_than_the_budget(tmp_path):
    T = 16
    _write_logs(tmp_path / "logs", T, seeds=[7, 12])
    out = tmp_path / "out"
    argv = ["--input-dir", str(tmp_path / "logs"), "--output-dir", str(out), "--acquisitions", "logei",
            "--error-models", "gaussian", "--tune-seeds", "7", "--score-seeds", "12", "--n0", "6",
            "--h-grid", "2,50", "--kappa-grid", "0,4", "--w-grid", "0,2", "--checkpoints", "8,10,12",
            "--workers", "1", "--no-evaluation-check"]
    rs.main(argv)
    per_run = pd.read_csv(out / "stopping_per_run.csv")
    assert set(per_run["rule"]) == {"ship_T", "A"} | {name for name, *_ in rs.b_variants(
        rs.resolve_config(rs.parse_args(argv)))}
    assert (per_run["trials_used"] <= T).all() and (per_run["trials_saved"] >= 0).all()
    floor = per_run[per_run["rule"].str.endswith("_floor")]
    stopped = floor[floor["stopped"]]
    assert (stopped["trials_used"] == stopped["stop_t"] + 3).all()
    assert (floor.loc[~floor["stopped"], "trials_used"] == T).all()
    # a clean run's repeat ratings are exact, so its floor is zero
    assert (floor.loc[floor["error_model"] == "none", "floor_sd"] == 0).all()
    tuned = json.loads((out / "stopping_tuned_A.json").read_text())
    grid = pd.read_csv(out / "stopping_grid_A.csv")
    row = grid[(grid["split"] == "tune") & (grid["h"] == tuned["h"]) & (grid["w"] == tuned["w"])
               & ((grid["kappa"] == tuned["kappa"]) | ~np.isfinite(grid["h"]))]
    assert bool(row["feasible"].iloc[0])
    summary = pd.read_csv(out / "stopping_summary.csv")
    assert {"recovered", "cost", "gain", "price", "mean_trials_saved", "median_trials_saved"} <= set(summary)
    # resuming reuses every fit
    before = sum(len(p.read_text().splitlines()) for p in (out / "cache_n06_rep3").glob("*.jsonl"))
    rs.main(argv)
    assert sum(len(p.read_text().splitlines()) for p in (out / "cache_n06_rep3").glob("*.jsonl")) == before


# --- (4) tuning -----------------------------------------------------------------


def test_tuning_honours_the_false_alarm_cap_and_falls_back_to_never():
    grid = pd.DataFrame([
        dict(h=rs.NEVER, kappa=np.nan, w=0, mean_regret_norm=0.30, mean_trials_used=50, feasible=True),
        dict(h=5.0, kappa=0.0, w=0, mean_regret_norm=0.10, mean_trials_used=30, feasible=False),
        dict(h=50.0, kappa=1.0, w=2, mean_regret_norm=0.20, mean_trials_used=45, feasible=True),
        dict(h=100.0, kappa=1.0, w=2, mean_regret_norm=0.20, mean_trials_used=45, feasible=True),
    ])
    chosen = rs.choose_params(grid, "regret")
    assert (chosen["h"], chosen["w"]) == (100.0, 2)  # the infeasible best is skipped; ties go conservative
    assert rs.choose_params(grid.iloc[[0, 1]], "regret")["h"] == rs.NEVER
    assert rs.choose_params(grid, "trials")["h"] == 100.0


def test_repeat_ratings_follow_the_run_error_process():
    frame = _log_frame(20, "branin", 7, True, np.random.default_rng(0))
    task = dict(clean=False, floor_repeats=3, run_args={}, jitter_std=1.0, jitter_iteration=0, seed=7,
                acquisition="logei", error_model="gaussian", bias=np.nan)
    ratings = rs.repeat_ratings(frame, task)
    assert len(ratings) == 4 and ratings[0] == frame["objective_observed"].iloc[0]
    assert np.std(ratings, ddof=1) > 0
    assert rs.repeat_ratings(frame, task) == ratings  # seeded
    late = rs.repeat_ratings(frame, {**task, "jitter_iteration": 20})
    assert np.allclose(late[1:], frame["objective_true"].iloc[0])  # repeats before a late onset are exact


def test_incompatible_options_fail_loudly():
    with pytest.raises(ValueError, match="both"):
        rs.resolve_config(rs.parse_args(["--tune-seeds", "7,8", "--score-seeds", "8"]))
    with pytest.raises(ValueError, match="response-error"):
        rs.resolve_config(rs.parse_args(["--error-models", "slip"]))
    with pytest.raises(ValueError, match="model-free"):
        rs.resolve_config(rs.parse_args(["--acquisitions", "logei,random"]))
    with pytest.raises(ValueError, match="w-grid"):
        rs.resolve_config(rs.parse_args(["--n0", "5", "--w-grid", "0,4"]))
    with pytest.raises(ValueError, match="a-params-from"):
        rs.resolve_config(rs.parse_args(["--rules", "A", "--tune-seeds", ""]))
    cfg = rs.resolve_config(rs.parse_args(["--rules", "B"]))
    with pytest.raises(ValueError, match="budget"):
        rs.check_budget(cfg, 42)
