"""The winner's-curse model of selection loss (scripts/checks/winners_curse.py).

The check predicts a run's selection loss by redrawing its error process on the
designs it visited and applying the standard ship rule. These tests pin

* the error law: ``rate`` fed a noise stream reproduces the simulator's own
  apply_sensor_error bit for bit, for all four processes and both kinds of
  onset (the bias offset, the drift ramp, the AR(1) state and its stream order);
* the Monte Carlo against closed forms: two designs under each process, where
  the expected selection loss is the gap times a normal tail probability, and
  six designs under GAUSSIAN error against a quadrature of the order statistic;
* the first-index tie rule of the standard ship rule;
* the realised selection loss and the replayed error stream on run logs: a short
  simulated run written as a log (always), and one logged main-sweep run
  (only where output-boba/ is on the machine);
* the report's estimands: the same cell means as decompose_regret.py, ratios
  of landscape means, and the identity behind the onset decomposition.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "checks"))

import boba_benchmarks as bb  # noqa: E402
import bo_sensor_error_simulation as sim  # noqa: E402
import decompose_regret as dr  # noqa: E402
import winners_curse as wc  # noqa: E402

BRANIN = bb.BENCHMARKS["branin"]


def _config(**overrides) -> "sim.SimulationConfig":
    fields = dict(
        iterations=12, jitter_iteration=0, jitter_std=0.0, single_error=False,
        initial_samples=5, candidate_pool=32, objective="value", objective_columns=["value"],
        param_columns=BRANIN.param_columns, seed=7, error_model="none", error_bias=0.2,
        error_spike_prob=0.1, error_spike_std=0.5, dropout_strategy="hold_last",
        normalize_objective=False, objective_weights=None, acq_num_restarts=2,
        acq_raw_samples=16, acq_maxiter=30, acq_mc_samples=16, ref_point=None,
    )
    fields.update(overrides)
    return sim.SimulationConfig(**fields)


def _phi_tail(x: float) -> float:
    """P(Z > x) for a standard normal Z."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# The error law is the simulator's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("process", wc.PROCESSES)
@pytest.mark.parametrize("onset", [0, 4])
def test_rate_reproduces_apply_sensor_error_bit_for_bit(process, onset):
    T, sigma, seed = 15, 0.7, 7
    g = np.random.default_rng(3).normal(size=T)
    bias = sigma   # the sweep's scaled offset
    config = _config(iterations=T, jitter_iteration=onset, jitter_std=sigma, error_model=process, error_bias=bias)
    seq = wc.jitter_seed(seed, 0, onset, sigma, sim.ERROR_MODEL_CHOICES.index(process))
    rng = np.random.default_rng(seq)
    observed, errors, prev_obs, prev_err = [], [], None, None
    for t in range(1, T + 1):
        true_value = np.array([g[t - 1]])
        prev_obs = true_value if prev_obs is None else prev_obs
        obs, err = sim.apply_sensor_error(true_value=true_value, iteration=t, config=config, rng=rng,
                                          previous_observed=prev_obs, previous_error=prev_err)
        prev_obs, prev_err = obs, err
        observed.append(float(obs[0]))
        errors.append(float(err[0]))
    replay = wc.replay_ratings(g, onset, process, sigma, seq, bias=bias)
    assert np.array_equal(replay, np.array(observed))
    assert np.array_equal(replay - g, np.array(errors))
    assert not np.any(np.array(errors[:onset]))


def test_process_parameters_come_from_the_file_name_variant():
    assert wc.process_parameters("bias", "bias1") == (1.0, wc.AR1_RHO)
    assert wc.process_parameters("bias", "bias0.25") == (0.25, wc.AR1_RHO)
    assert wc.process_parameters("gaussian", "") == (0.0, wc.AR1_RHO)
    assert wc.process_parameters("ar1", "rho0.5") == (0.0, 0.5)
    with pytest.raises(ValueError, match="offset is unknown"):
        wc.process_parameters("bias", "")
    with pytest.raises(ValueError, match="not modelled"):
        wc.process_parameters("gaussian", "single")
    with pytest.raises(ValueError, match="not one of"):
        wc.process_parameters("spike", "sp0.15-20")


# ---------------------------------------------------------------------------
# The Monte Carlo against closed forms
# ---------------------------------------------------------------------------


def _two_design_mc(process: str, onset: int, g: np.ndarray, sigma: float, draws: int = 400_000):
    rng = np.random.default_rng(11)
    bias = sigma if process == "bias" else 0.0
    out = wc.expected_selection(g, onset, process, sigma, draws, rng, bias=bias)
    return out["mean"], math.sqrt(out["var"] / draws)


@pytest.mark.parametrize("process,expected", [
    # Ship the worse first design iff e1 - e2 > gap; e1 - e2 ~ N(0, 2 s^2) ...
    ("gaussian", lambda gap, s: gap * _phi_tail(gap / (s * math.sqrt(2.0)))),
    # ... the offset is common to both ratings, so it changes nothing ...
    ("bias", lambda gap, s: gap * _phi_tail(gap / (s * math.sqrt(2.0)))),
    # ... the ramp adds s/2 at trial 1 and s at trial 2 (T = 2, onset 0) ...
    ("drift", lambda gap, s: gap * _phi_tail((gap + s / 2.0) / (s * math.sqrt(2.0)))),
    # ... and AR(1) makes Var(e1 - e2) = 2 s^2 (1 - rho).
    ("ar1", lambda gap, s: gap * _phi_tail(gap / (s * math.sqrt(2.0 * (1.0 - wc.AR1_RHO))))),
])
def test_two_designs_match_the_closed_form(process, expected):
    gap, sigma = 0.6, 0.8
    mean, se = _two_design_mc(process, 0, np.array([0.0, gap]), sigma)
    assert abs(mean - expected(gap, sigma)) < 5 * se


def test_an_exact_rating_before_the_onset_is_beaten_only_by_its_own_tail():
    # Trial 1 exact (onset 1) and better by gap; trial 2 rated with N(0, s^2).
    gap, sigma = 0.5, 1.0
    mean, se = _two_design_mc("gaussian", 1, np.array([gap, 0.0]), sigma)
    assert abs(mean - gap * _phi_tail(gap / sigma)) < 5 * se


def test_six_designs_match_a_quadrature_of_the_order_statistic():
    from scipy.integrate import quad
    from scipy.stats import norm

    g = np.array([0.3, -0.4, 1.1, 0.9, 0.2, 1.05])
    sigma = 0.6
    loss = 0.0
    for i in range(len(g)):
        def density(x, i=i):
            others = np.delete(g, i)
            return norm.pdf(x, g[i], sigma) * np.prod(norm.cdf(x, others, sigma))
        p_i, _ = quad(density, g.min() - 10 * sigma, g.max() + 10 * sigma, limit=200)
        loss += (g.max() - g[i]) * p_i
    draws = 400_000
    out = wc.expected_selection(g, 0, "gaussian", sigma, draws, np.random.default_rng(5))
    assert abs(out["mean"] - loss) < 5 * math.sqrt(out["var"] / draws)


@pytest.mark.parametrize("process", ["gaussian", "drift"])
def test_exact_ratings_before_a_late_onset_match_the_quadrature(process):
    """Independent errors after an onset at trial 3: the best exact rating b is shipped
    with probability prod_j Phi((b - m_j)/s), trial i after the onset with
    int_{x > b} phi((x - m_i)/s)/s prod_{j != i} Phi((x - m_j)/s) dx, m_j = g_j + offset_j."""
    from scipy.integrate import quad
    from scipy.stats import norm

    g = np.array([0.7, 0.2, 0.9, 0.4, 1.2, 0.95, 0.6, 1.1])
    onset, sigma, T = 3, 0.4, 8
    post = g[onset:]
    offsets = sigma * np.arange(1, T - onset + 1) / (T - onset) if process == "drift" else np.zeros(T - onset)
    m, b = post + offsets, g[:onset].max()
    loss = float(np.prod(norm.cdf((b - m) / sigma))) * (g.max() - b)
    for i in range(len(post)):
        others = np.delete(m, i)
        p_i, _ = quad(lambda x: norm.pdf(x, m[i], sigma) * np.prod(norm.cdf(x, others, sigma)),
                      b, m.max() + 10 * sigma, limit=200)
        loss += p_i * (g.max() - post[i])
    draws = 400_000
    out = wc.expected_selection(g, onset, process, sigma, draws, np.random.default_rng(9))
    assert abs(out["mean"] - loss) < 5 * math.sqrt(out["var"] / draws)


def test_the_monte_carlo_is_seeded():
    g = np.array([0.0, 0.4, 0.9, 0.8])
    a = wc.expected_selection(g, 1, "ar1", 0.5, 2000, np.random.default_rng(1))
    b = wc.expected_selection(g, 1, "ar1", 0.5, 2000, np.random.default_rng(1))
    assert a == b


# ---------------------------------------------------------------------------
# The first-index tie rule
# ---------------------------------------------------------------------------


def test_ties_go_to_the_first_index_as_in_the_simulator():
    # The top rating is shared by trials 2 and 3 (indices 1, 2); the rule ships trial 2.
    g = np.array([2.0, 4.0, 6.0, 1.0])
    y = np.array([2.0, 5.0, 5.0, 1.0])
    assert wc.realised_selection(g, y) == 2.0
    assert list(wc.ship_indices(np.array([[1.0, 3.0, 3.0], [3.0, 3.0, 1.0]]))) == [1, 0]


def test_tied_exact_ratings_before_the_onset_ship_the_earliest():
    g = np.array([5.0, 5.0, 1.0])
    z_jit, z_inn = wc.stream_normals("gaussian", 1, np.random.default_rng(0), 1000)
    loss, idx, error = wc.selection_draws(g, 2, "gaussian", 0.01, z_jit, z_inn)
    assert np.all(idx == 0) and not np.any(loss) and not np.any(error)


# ---------------------------------------------------------------------------
# Run logs: realised selection loss and the replayed stream
# ---------------------------------------------------------------------------


def _simulated_log(process: str, onset: int, sigma: float, seed: int = 7, acq: str = "random",
                   T: int = 12) -> pd.DataFrame:
    """A short branin run through the simulator itself, seeded as the synthetic driver seeds it."""
    stats = bb.load_stats()
    y_opt = (stats["branin"]["y_opt"] - stats["branin"]["mean"]) / stats["branin"]["std"]
    config = _config(iterations=T, jitter_iteration=onset, jitter_std=sigma, error_model=process,
                     error_bias=sigma, seed=seed)
    jitter_rng = np.random.default_rng(wc.jitter_seed(seed, sim.ACQUISITION_CHOICES.index(acq), onset, sigma,
                                                      sim.ERROR_MODEL_CHOICES.index(process)))
    torch.manual_seed(seed)
    frame = sim.run_simulation(
        oracle=bb.SyntheticOracle.from_stats("branin", stats), bounds=sim.Bounds(BRANIN.bounds_low, BRANIN.bounds_high),
        config=config, acq=sim.AcquisitionConfig(name=acq), rng=np.random.default_rng(seed),
        jitter_rng=jitter_rng, run_id="t", apply_error=True, oracle_model="exact", y_opt=y_opt)
    frame["dataset"] = "branin"
    return frame


def _clean_log(seed: int = 7, acq: str = "random", T: int = 12) -> pd.DataFrame:
    stats = bb.load_stats()
    y_opt = (stats["branin"]["y_opt"] - stats["branin"]["mean"]) / stats["branin"]["std"]
    torch.manual_seed(seed)
    frame = sim.run_simulation(
        oracle=bb.SyntheticOracle.from_stats("branin", stats), bounds=sim.Bounds(BRANIN.bounds_low, BRANIN.bounds_high),
        config=_config(iterations=T, seed=seed), acq=sim.AcquisitionConfig(name=acq),
        rng=np.random.default_rng(seed), jitter_rng=None, run_id="c", apply_error=False, oracle_model="exact",
        y_opt=y_opt)
    frame["dataset"] = "branin"
    return frame


def test_scan_group_reproduces_simulated_logs(tmp_path):
    """Write a clean twin and four noisy runs as the driver names them; scan them as the check does."""
    seed, acq, T = 7, "random", 12
    land = tmp_path / "arm" / "branin"
    land.mkdir(parents=True)
    stem = f"bo_sensor_error_branin_value_{acq}_seed{seed}"
    clean_name = f"{stem}_baseline_exact.csv"
    _clean_log(seed, acq, T).to_csv(land / clean_name, index=False)
    noisy = []
    for process, onset, sigma in (("gaussian", 0, 1.0), ("bias", 4, 0.25), ("drift", 0, 5.0), ("ar1", 4, 1.0)):
        suffix = f"_bias{sigma:g}" if process == "bias" else ""
        name = f"{stem}_jittered_exact_{process}_jit{onset}_std{sigma}{suffix}.csv"
        _simulated_log(process, onset, sigma, seed, acq, T).to_csv(land / name, index=False)
        noisy.append({"dataset": "branin", "acquisition": acq, "seed": seed, "baseline": False,
                      "error_model": process, "jitter_std": sigma, "jitter_iteration": onset,
                      "variant": suffix.lstrip("_"), "path": str(land / name), "file": f"branin/{name}"})
    task = {"arm": "test", "arm_dir": "arm", "dataset": "branin", "acquisition": acq, "seed": seed,
            "clean": {"dataset": "branin", "acquisition": acq, "seed": seed, "baseline": True, "variant": "",
                      "path": str(land / clean_name), "file": f"branin/{clean_name}"},
            "noisy": noisy, "draws": 500, "acq_index": sim.ACQUISITION_CHOICES.index(acq),
            "em_index": {p: sim.ERROR_MODEL_CHOICES.index(p) for p in wc.PROCESSES}}
    rows = pd.DataFrame(wc.scan_group(task))
    assert rows["baseline"].sum() == 1
    out = rows[~rows["baseline"]]
    assert len(out) == 4
    # The replayed stream is the logged one, bit for bit.
    assert (out["replay_max_abs_rating"] == 0).all() and (out["replay_max_abs_error"] == 0).all()
    assert out["logged_pre_onset_exact"].all()
    for _, r in out.iterrows():
        log = pd.read_csv(land / r["file"].split("/", 1)[1], float_precision="round_trip")
        logged = float(log["inference_simple_regret_true"].iloc[-1] - log["simple_regret_true"].iloc[-1])
        assert r["deployed"] - r["search"] == pytest.approx(logged, abs=1e-12)
        assert r["selection_direct"] == pytest.approx(logged, abs=1e-12)
        assert r["sel_pred"] >= 0 and np.isfinite(r["sel_pred"])
    # Different paths for different runs, the same paths for the run and its clean twin.
    assert out["sel_pred"].nunique() == 4


def test_a_log_whose_ratings_disagree_with_its_regret_is_refused(tmp_path):
    frame = _simulated_log("gaussian", 0, 1.0)
    frame.loc[frame.index[-1], "inference_simple_regret_true"] += 0.1
    path = tmp_path / "x.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree with the log"):
        wc._final_regrets(wc.read_log(str(path)), str(path))


MAIN = REPO / "output-boba"
LOGGED = MAIN / "branin" / "bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0.csv"


@pytest.mark.skipif(not LOGGED.is_file() or not (MAIN / "analysis" / "ship_rules_per_run.csv").is_file(),
                    reason="the main sweep's logs (output-boba/, git-ignored) are not on this machine")
def test_a_logged_main_sweep_run_is_reproduced():
    log = wc.read_log(str(LOGGED))
    g = log["objective_true"].to_numpy(float)
    y = log["objective_observed"].to_numpy(float)
    seq = wc.jitter_seed(7, sim.ACQUISITION_CHOICES.index("logei"), 20, 1.0, sim.ERROR_MODEL_CHOICES.index("gaussian"))
    replay = wc.replay_ratings(g, 20, "gaussian", 1.0, seq)
    assert np.array_equal(replay, y)
    assert np.array_equal(replay - g, log["error_magnitude"].to_numpy(float))
    rules = pd.read_csv(MAIN / "analysis" / "ship_rules_per_run.csv", float_precision="round_trip")
    row = rules[rules["file"] == f"branin/{LOGGED.name}"].iloc[0]
    assert wc.realised_selection(g, y) == pytest.approx(row["regret_best_observed"] - row["regret_best_visited"],
                                                         abs=1e-12)


# ---------------------------------------------------------------------------
# The report's estimands
# ---------------------------------------------------------------------------


def _synthetic_runs(seed: int = 0) -> pd.DataFrame:
    """A main-arm-shaped per-run table: 20 landscapes, 2 acquisitions, 3 seeds, 2 processes, 2 cells."""
    rng = np.random.default_rng(seed)
    opt_z = wc.opt_z_map()
    landscapes = sorted(opt_z)[:20]
    rows = []
    for d in landscapes:
        for acq in ("logei", "qnei"):
            for s in (7, 8, 9):
                y_opt = 1.0 + rng.random()
                clean_dep = rng.random() * 0.3
                rows.append({"arm": "main", "dataset": d, "acquisition": acq, "seed": s, "baseline": True,
                             "error_model": "none", "jitter_std": 0.0, "onset": -1, "T": 50, "y_opt": y_opt,
                             "deployed": clean_dep, "search": clean_dep, "n_near": 10.0, "n_near_unique": 8.0})
                for process in ("gaussian", "drift"):
                    for onset in (0, 20):
                        search = clean_dep + rng.random() * (0.6 if onset == 0 else 0.2)
                        sel = rng.random() * 0.5
                        pred = sel * (0.8 + 0.3 * rng.random())
                        rows.append({
                            "arm": "main", "dataset": d, "acquisition": acq, "seed": s, "baseline": False,
                            "error_model": process, "jitter_std": 1.0, "onset": onset, "T": 50, "draws": 4000,
                            "y_opt": y_opt, "deployed": search + sel, "search": search,
                            "deployed_clean": clean_dep, "search_clean": clean_dep,
                            "sel_pred": pred, "sel_pred_var": 0.1, "sel_pred_clean": pred / 1.5,
                            "wrong_real": 1.0, "p_wrong_pred": 0.9, "post_real": 1.0, "p_post_pred": 1.0,
                            "over_real": 1.8, "over_pred": 1.7, "over_pred_clean": 1.5,
                            "n_near": 5.0, "n_near_unique": 4.0, "n_near_clean": 10.0, "n_near_unique_clean": 8.0,
                            "n_within_sigma": 20, "n_within_sigma_clean": 30})
    return pd.DataFrame(rows)


def test_the_cell_means_are_decompose_regrets():
    runs = wc.with_units(_synthetic_runs(), wc.opt_z_map())
    mine = wc.decomposition_rows(runs)
    # The same runs in rescore_ship_rules' columns, through decompose_regret itself.
    frame = pd.DataFrame({
        **{k: runs[k] for k in ("dataset", "acquisition", "seed", "baseline", "error_model", "jitter_std")},
        "jitter_iteration": runs["onset"].where(~runs["baseline"]),
        "regret_best_observed": runs["deployed"], "regret_best_visited": runs["search"]})
    table = dr.decompose(frame, wc.opt_z_map())
    clean, noisy = table[table["baseline"]], table[~table["baseline"]]
    rng = np.random.default_rng(0)
    for (model, std, onset), group in noisy.groupby(["error_model", "jitter_std", "jitter_iteration"]):
        ref = dr.summarise(group, clean, rng)
        row = mine[(mine["error_model"] == model) & (mine["jitter_std"] == str(float(std)))
                   & (mine["jitter_iteration"] == str(float(onset)))].iloc[0]
        for col in ("mean_deployed", "mean_search", "mean_selection", "selection_share", "mean_excess_deployed",
                    "mean_excess_search", "mean_excess_selection", "excess_selection_share"):
            assert row[col] == pytest.approx(ref[col], abs=1e-12), col


def test_cells_are_ratios_of_landscape_means_and_the_onset_identity_holds():
    runs = wc.with_units(_synthetic_runs(1), wc.opt_z_map())
    landscapes = sorted(runs["dataset"].unique())
    B = wc.boot_matrix(len(landscapes))
    cells = wc.cells_table(runs, landscapes, B)
    c = cells[(cells["error_model"] == "pooled") & (cells["onset"] == 0)].iloc[0]
    block = runs[~runs["baseline"] & (runs["onset"] == 0)]
    per = block.groupby("dataset")[["selection_z", "sel_pred_z", "dep_excess_z"]].mean()
    assert c["ratio"] == pytest.approx(per["selection_z"].mean() / per["sel_pred_z"].mean())
    assert c["share_real"] == pytest.approx(per["selection_z"].mean() / per["dep_excess_z"].mean())
    assert c["share_pred"] / c["share_real"] == pytest.approx(1.0 / c["ratio"])
    assert c["ratio_lo"] <= c["ratio"] <= c["ratio_hi"]
    assert c["set_effect"] == pytest.approx(1.5)          # the synthetic clean set is 1.5x less prone
    onset = wc.onset_table(runs, landscapes, B)
    assert np.allclose(onset["identity_gap"], 0.0, atol=1e-12)
    pooled = onset[onset["error_model"] == "pooled"].iloc[0]
    early = runs[~runs["baseline"] & (runs["onset"] == 0)].groupby("dataset")["dep_excess_z"].mean()
    late = runs[~runs["baseline"] & (runs["onset"] == 20)].groupby("dataset")["dep_excess_z"].mean()
    assert pooled["deployed_ratio"] == pytest.approx(early.mean() / late.mean())


def test_the_figure_draws_the_headline_cell_of_the_table(tmp_path):
    import make_paper_figures as mf

    names = sorted(mf.PRETTY)
    real = np.random.default_rng(0).uniform(0.05, 0.85, len(names))
    lands = pd.DataFrame({"arm": "main", "error_model": "pooled", "jitter_std": 1.0, "onset": 0, "landscape": names,
                          "share_real": real, "share_pred": 0.9 * real, "sel_real_sd": real, "sel_pred_sd": 0.9 * real})
    other = lands.assign(jitter_std=5.0, sel_pred_sd=real[::-1])      # another cell, never drawn
    assert mf.fig_winners_curse(pd.concat([lands, other]), tmp_path) == pytest.approx(1.0)
    assert (tmp_path / "winners_curse.pdf").is_file() and (tmp_path / "winners_curse.png").is_file()
    with pytest.raises(SystemExit, match="20 landscapes"):
        mf.fig_winners_curse(lands.iloc[:19], tmp_path)


def test_boot_ratio_is_a_ratio_of_means_with_a_percentile_interval():
    B = wc.boot_matrix(20)
    num = np.arange(1.0, 21.0)
    point, lo, hi = wc.boot_ratio(num, 2 * num, B)
    assert point == lo == hi == pytest.approx(0.5)
    den = np.ones(20)
    point, lo, hi = wc.boot_ratio(num, den, B)
    assert point == pytest.approx(num.mean()) and lo < point < hi
