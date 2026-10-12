"""The known-variance final refit (scripts/checks/known_variance_ship.py).

What must hold: every information level hands the final refit exactly the variance it claims (the
simulator's known_noise_variance trial by trial, sigma_e^2, the test-retest estimate, the CUSUM onset); a refit
given a fixed variance uses it and learns none, in rating units; an exact rater's floor makes every posterior
rule ship the best rating, which is why the clean-twin price is zero by construction; the inside-the-budget
repeats are drawn from the run's own error process at trials T-2..T, carrying the AR(1) state on from the
logged error, and an exact rater's estimate is zero; the learned refit reproduces rescore_ship_rules.py's
stored value bit for bit, and a mismatch stops the check; and the scoring is the adaptation estimand.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bo_sensor_error_simulation as sim  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
import rescore_ship_rules as rs  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("checks_known_variance_ship",
                                                  SCRIPTS / "checks" / "known_variance_ship.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kvs = _load()

RUN_ARGS = {"error_bias_mode": "scaled", "error_bias": 0.2, "error_ar1_rho": 0.8, "response_clip": "none",
            "response_round": None, "single_error": False, "error_spike_prob": 0.1, "error_spike_std": 0.5,
            "error_spike_std_mode": "scaled"}
T_TOY = 12


def _info(model="gaussian", std=1.0, onset=0, seed=7, acq="logei", variant=""):
    return {"error_model": model, "jitter_std": std, "jitter_iteration": onset, "seed": seed, "acquisition": acq,
            "variant": variant, "dataset": "branin"}


def _designs(n: int, seed: int = 3) -> np.ndarray:
    spec = bb.BENCHMARKS["branin"]
    rng = np.random.default_rng(seed)
    return spec.lo + rng.random((n, 2)) * (spec.hi - spec.lo)


def _truth(X: np.ndarray) -> np.ndarray:
    """A smooth stand-in objective on branin's box, in landscape-SD-like units."""
    spec = bb.BENCHMARKS["branin"]
    u = (X - spec.lo) / (spec.hi - spec.lo)
    return np.sin(3 * u[:, 0]) + np.cos(2 * u[:, 1]) - 0.5 * (u[:, 0] - 0.6) ** 2


def _write_log(root: Path, name: str, X, true, observed, model: str, std: float, onset: int,
               baseline: bool) -> Path:
    y_opt = 2.5
    frame = pd.DataFrame(np.asarray(X, float), columns=["x0", "x1"])
    frame.insert(0, "iteration", np.arange(1, len(true) + 1))
    frame["objective_true"] = true
    frame["objective_observed"] = observed
    frame["error_magnitude"] = np.asarray(observed) - np.asarray(true)
    frame["acquisition"] = "logei"
    frame["seed"] = 7
    frame["error_model"] = "none" if baseline else model
    frame["jitter_std"] = 0.0 if baseline else std
    frame["jitter_iteration"] = onset
    frame["param_columns"] = "x0,x1"
    frame["y_opt"] = y_opt
    frame["inference_simple_regret_true"] = y_opt - np.asarray(true)[int(np.argmax(observed))]
    frame["dataset"] = "branin"
    path = root / "branin" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def toy_arm(tmp_path):
    """A clean twin and an AR(1) noisy run of twelve trials on branin's box, as logs in an arm directory."""
    X = _designs(T_TOY)
    true = _truth(X)
    rng = np.random.default_rng(11)
    noise = 0.7 * rng.standard_normal(T_TOY)
    _write_log(tmp_path, "bo_sensor_error_branin_value_logei_seed7_baseline_exact.csv", X, true, true,
               "none", 0.0, 0, True)
    _write_log(tmp_path, "bo_sensor_error_branin_value_logei_seed7_jittered_exact_ar1_jit0_std1.0.csv", X, true,
               true + noise, "ar1", 1.0, 0, False)
    tasks = rs.index_runs(tmp_path, {"branin"}, {"logei"}, {7}, {"ar1"})
    assert len(tasks) == 2
    out = []
    for t in tasks:
        out.append({**t, "arm": "output-boba", "run_args": RUN_ARGS, "expected_rule": "best_observed",
                    "levels": kvs.NO_CUSUM_LEVELS})
    return tmp_path, out


# ---------------------------------------------------------------------------
# What each level passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gaussian", "bias", "drift", "ar1"])
@pytest.mark.parametrize("onset", [0, 20])
def test_each_level_passes_the_variance_it_claims(model, onset):
    T, sigma = 50, 1.0
    info = _info(model, sigma, onset, variant="bias1" if model == "bias" else "")
    cfg = kvs.error_config(RUN_ARGS, info, T)
    vec, v_hat = kvs.variance_vectors(cfg, T, info, "run.csv", kvs.ALL_LEVELS, alarm=30.0, w=3)
    t = np.arange(1, T + 1)
    # magnitude and onset known: the simulator's own per-trial variance, which the known-noise arm logged
    assert np.array_equal(vec["mag_onset"], [sim.known_noise_variance(int(i), cfg, True) for i in t])
    assert np.array_equal(vec["mag_onset"], np.where(t > onset, sigma ** 2, kvs.FLOOR))
    # the magnitude alone, and the twofold misstatements of the SD
    assert np.array_equal(vec["mag"], np.full(T, sigma ** 2))
    assert np.allclose(vec["mag_half"], sigma ** 2 / 4) and np.allclose(vec["mag_double"], 4 * sigma ** 2)
    # the onset from the alarm: the last w trials before it are already treated as noisy
    assert np.array_equal(vec["cusum"], np.where(t > 27, sigma ** 2, kvs.FLOOR))
    assert np.array_equal(vec["retest3_cusum"], np.where(t > 27, max(v_hat, kvs.FLOOR), kvs.FLOOR))
    assert np.array_equal(vec["retest3"], np.full(T, max(v_hat, kvs.FLOOR)))
    assert cfg.error_bias == (1.0 if model == "bias" else cfg.error_bias)


def test_the_retest_estimates_the_idiosyncratic_variance():
    # three pairs rated in one sitting: unbiased for the fresh error, s^2 chi2_3 / 3
    for model, scale in (("gaussian", 1.0), ("bias", 1.0), ("drift", 1.0), ("ar1", 1.0 - 0.8 ** 2)):
        draws = np.array([kvs.retest_estimate(model, 2.0, 0.8, f"run{i}.csv") for i in range(4000)])
        assert abs(draws.mean() / (4.0 * scale) - 1.0) < 0.05
        # chi2_3 / 3 has variance 2/3
        assert abs(draws.var() / (4.0 * scale) ** 2 - 2.0 / 3.0) < 0.12
    # seeded by the file name: a rerun draws the same estimate
    assert kvs.retest_estimate("gaussian", 1.0, 0.8, "a.csv") == kvs.retest_estimate("gaussian", 1.0, 0.8, "a.csv")
    assert kvs.retest_estimate("gaussian", 1.0, 0.8, "a.csv") != kvs.retest_estimate("gaussian", 1.0, 0.8, "b.csv")


def test_no_alarm_means_the_magnitude_on_every_trial():
    assert np.array_equal(kvs.onset_vector(0.5, np.nan, 3, 10), np.full(10, 0.5))
    assert np.array_equal(kvs.onset_vector(0.5, None, 3, 10), np.full(10, 0.5))
    assert np.array_equal(kvs.onset_vector(0.5, 6.0, 3, 10), [kvs.FLOOR] * 3 + [0.5] * 7)
    assert kvs.FLOOR == 1e-6


# ---------------------------------------------------------------------------
# The refit
# ---------------------------------------------------------------------------


def test_a_fixed_variance_is_used_in_rating_units_and_nothing_is_learned():
    X = _designs(15)
    y = 3.0 * _truth(X) + 7.0            # ratings on an arbitrary scale: Standardize must absorb it
    spec = bb.BENCHMARKS["branin"]
    v = 0.4
    gp, diag = rs.fit_loop_gp(X, y, spec.bounds_low, spec.bounds_high, yvar=np.full(15, v), seed=1)
    stdv = float(gp.outcome_transform.stdvs.reshape(-1)[0])
    noise = gp.likelihood.noise.detach().reshape(-1).numpy()
    # the likelihood holds the given variance (standardised), so in rating units it is exactly v
    assert np.allclose(noise * stdv ** 2, v)
    assert not any(p.requires_grad for n, p in gp.likelihood.named_parameters())
    assert diag["gp_fit_ok"] and np.isnan(diag["gp_noise_sd"])


def test_an_exact_rater_ships_the_best_rating_under_every_rule():
    X = _designs(14, seed=5)
    y = _truth(X)
    spec = bb.BENCHMARKS["branin"]
    picks, _ = kvs.posterior_picks(X, y, spec.bounds_low, spec.bounds_high, np.full(14, kvs.FLOOR), seed=2)
    assert set(picks.values()) == {int(np.argmax(y))}


def test_every_level_reaches_the_refit_with_its_own_vector(toy_arm, monkeypatch):
    _, tasks = toy_arm
    learned_fits, picks_calls = [], []
    real_fit, real_picks = rs.fit_loop_gp, kvs.posterior_picks

    def fit_spy(X, y, lo, hi, yvar=None, seed=0):
        if yvar is None:
            learned_fits.append(len(y))
        return real_fit(X, y, lo, hi, yvar=yvar, seed=seed)

    def picks_spy(X, y, lo, hi, yvar, seed, n_cand=None):
        picks_calls.append((len(y), n_cand, np.asarray(yvar, float).copy()))
        return real_picks(X, y, lo, hi, yvar, seed, n_cand)

    monkeypatch.setattr(rs, "fit_loop_gp", fit_spy)
    monkeypatch.setattr(kvs, "posterior_picks", picks_spy)
    noisy = next(t for t in tasks if not t["baseline"])
    rec = kvs.refit_run(noisy)
    cfg = kvs.error_config(RUN_ARGS, noisy, T_TOY)
    want, v_hat = kvs.variance_vectors(cfg, T_TOY, noisy, Path(noisy["file"]).name, kvs.NO_CUSUM_LEVELS)
    level_fits = [v for n, nc, v in picks_calls if nc is None]
    distinct = {v.tobytes() for v in want.values()}
    # every distinct vector is fitted exactly once (mag and mag_onset coincide at onset 0)
    assert len(level_fits) == len(distinct) and {v.tobytes() for v in level_fits} == distinct
    assert rec["sigma2_retest3"] == v_hat
    # the two inside-the-budget refits: all T ratings, each with its own estimate on every rating
    inside = [(n, nc, v) for n, nc, v in picks_calls if nc is not None]
    m = T_TOY - kvs.R_PAIRS
    assert [(n, nc) for n, nc, _ in inside] == [(T_TOY, m), (T_TOY, m)]
    assert np.all(inside[0][2] == max(rec["sigma2_inside_random"], kvs.FLOOR))
    assert np.all(inside[1][2] == max(rec["sigma2_inside_top"], kvs.FLOOR))
    # the learned fits: the control on all T trials and inside_top's ranking on the T - 3 prefix
    assert sorted(learned_fits) == [m, T_TOY]
    # the clean twin: every vector level gets the floor, fitted once, and the repeats estimate zero
    picks_calls.clear()
    clean = next(t for t in tasks if t["baseline"])
    rec_c = kvs.refit_run(clean)
    level_fits = [v for n, nc, v in picks_calls if nc is None]
    assert len(level_fits) == 1 and np.all(level_fits[0] == kvs.FLOOR)
    assert all(np.all(v == kvs.FLOOR) for n, nc, v in picks_calls if nc is not None)
    assert rec_c["sigma2_inside_random"] == 0.0 and rec_c["sigma2_inside_top"] == 0.0
    # an exact rater's floor ships the best rating: the clean-twin price is zero by construction here
    for lv in ("mag", "mag_onset", "retest3"):
        for r in kvs.RULES:
            assert rec_c[f"regret_{lv}_{r}"] == rec_c["regret_best_observed"]


def test_inside_the_budget_repeats_follow_the_runs_error_process(toy_arm, monkeypatch):
    _, tasks = toy_arm
    noisy = next(t for t in tasks if not t["baseline"])
    df = pd.read_csv(noisy["path"], float_precision="round_trip")
    a = kvs.run_arrays(df, noisy)
    cfg = kvs.error_config(RUN_ARGS, noisy, T_TOY)
    seen = []
    real = sim.apply_sensor_error

    def spy(**kw):
        obs, err = real(**kw)
        seen.append((kw["iteration"], float(kw["previous_error"][0]), float(obs[0]), float(err[0])))
        return obs, err

    monkeypatch.setattr(sim, "apply_sensor_error", spy)
    out = kvs.inside_budget(a["X"], a["y"], a["g"], a["score"], a["err"], a["y_opt"], a["lo"], a["hi"], 5,
                            Path(noisy["file"]).name, "random", cfg, noisy)
    m = T_TOY - kvs.R_PAIRS
    assert [s[0] for s in seen] == [m + 1, m + 2, m + 3]
    # the AR(1) state is carried on from trial m's logged error, then from each repeat's own error
    assert seen[0][1] == a["err"][m - 1]
    assert seen[1][1] == seen[0][3] and seen[2][1] == seen[1][3]
    y_rep = np.array([s[2] for s in seen])
    idx = np.array(out["idx"])
    assert len(set(idx)) == kvs.R_PAIRS and np.all(idx < m)
    assert out["sigma2"] == float(np.sum((y_rep - a["y"][idx]) ** 2) / (2 * kvs.R_PAIRS))
    # the same run gives the same repeats: the stream is seeded, not drawn from a global generator
    seen.clear()
    again = kvs.inside_budget(a["X"], a["y"], a["g"], a["score"], a["err"], a["y_opt"], a["lo"], a["hi"], 5,
                              Path(noisy["file"]).name, "random", cfg, noisy)
    assert again["sigma2"] == out["sigma2"] and again["regret"] == out["regret"]
    # an exact rater re-rates exactly
    clean_out = kvs.inside_budget(a["X"], a["g"], a["g"], a["score"], np.zeros(T_TOY), a["y_opt"], a["lo"], a["hi"],
                                  5, "clean.csv", "top", None, noisy)
    assert clean_out["sigma2"] == 0.0


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------


def test_the_learned_refit_reproduces_the_stored_value_and_a_mismatch_stops(toy_arm, tmp_path, monkeypatch):
    root, tasks = toy_arm
    recs = [kvs.refit_run(t) for t in tasks]
    # the stored table, as rescore_ship_rules.py writes it
    stored = []
    for t in tasks:
        df = pd.read_csv(t["path"], float_precision="round_trip")
        stored.append(rs.rescore_frame(df, t, "best_observed", rs.file_seed(t["file"])))
    stored_path = tmp_path / "ship_rules_per_run.csv"
    pd.DataFrame(stored).to_csv(stored_path, index=False)
    monkeypatch.setattr(kvs, "STORED", {"output-boba": stored_path})
    per_run = pd.DataFrame(recs)
    lines = kvs.control_check(per_run)
    assert len(lines) == 1 and "2 of 2 runs" in lines[0] and "bit-identical in 2" in lines[0]
    bad = pd.read_csv(stored_path)
    bad.loc[0, "regret_lcb1"] += 1e-6
    bad.to_csv(stored_path, index=False)
    with pytest.raises(rs.ReproductionError):
        kvs.control_check(per_run)


# ---------------------------------------------------------------------------
# The estimand
# ---------------------------------------------------------------------------


def _toy_runs(cost_sign: float = 1.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Three landscapes, one cell: the standard rule and one level, noisy and clean."""
    rows = []
    opt_z = {"branin": 2.0, "ackley": 4.0, "levy_10": 1.0}
    for d, (std_n, std_c, trt_n, trt_c) in {"branin": (1.0, 0.2, 0.6, 0.2), "ackley": (2.0, 0.4, 1.6, 0.5),
                                              "levy_10": (0.5, 0.1, 0.3, 0.1)}.items():
        for seed in (12, 13):
            base = {"arm": "output-boba", "dataset": d, "acquisition": "logei", "seed": seed, "variant": ""}
            rows.append({**base, "error_model": "none", "jitter_std": 0.0, "jitter_iteration": np.nan,
                         "baseline": True, "file": f"{d}/c{seed}.csv", "regret_best_observed": std_c,
                         **{f"regret_{lv}_{r}": (trt_c if lv == "mag" else std_c) for lv in ("learned", "mag")
                            for r in kvs.RULES}})
            sn = std_c + cost_sign * (std_n - std_c) if d == "levy_10" else std_n
            rows.append({**base, "error_model": "gaussian", "jitter_std": 1.0, "jitter_iteration": 0,
                         "baseline": False, "file": f"{d}/n{seed}.csv", "regret_best_observed": sn,
                         **{f"regret_{lv}_{r}": (trt_n if lv == "mag" else sn) for lv in ("learned", "mag")
                            for r in kvs.RULES}})
    return pd.DataFrame(rows), opt_z


def test_the_rule_is_chosen_on_seeds_7_to_11_and_only_scored_on_12_to_16(monkeypatch):
    """On seed 7 LCB1 is the best known-variance rule, on seed 12 LCB2 would be: the choice must be LCB1."""
    opt_z = {"branin": 1.0, "ackley": 1.0, "levy_10": 1.0}
    regrets = {  # seed -> level -> rule -> noisy regret; the standard rule's noisy regret is 0.5, clean 0.1
        7: {"learned": {"pm": 0.45, "lcb1": 0.40, "lcb2": 0.42}, "mag": {"pm": 0.40, "lcb1": 0.30, "lcb2": 0.35}},
        12: {"learned": {"pm": 0.45, "lcb1": 0.45, "lcb2": 0.30}, "mag": {"pm": 0.20, "lcb1": 0.35, "lcb2": 0.10}},
    }
    rows = []
    for d in opt_z:
        for seed in (7, 12):
            base = {"arm": "output-boba", "dataset": d, "acquisition": "logei", "seed": seed, "variant": ""}
            rows.append({**base, "error_model": "none", "jitter_std": 0.0, "jitter_iteration": np.nan,
                         "baseline": True, "file": f"{d}/c{seed}.csv", "regret_best_observed": 0.1,
                         **{f"regret_{lv}_{r}": 0.1 for lv in ("learned", "mag") for r in kvs.RULES}})
            for sig in kvs.GRID:
                for onset in (0, 20):
                    rows.append({**base, "error_model": "gaussian", "jitter_std": sig, "jitter_iteration": onset,
                                 "baseline": False, "file": f"{d}/n{seed}_{sig}_{onset}.csv",
                                 "regret_best_observed": 0.5,
                                 **{f"regret_{lv}_{r}": regrets[seed][lv][r] for lv in ("learned", "mag")
                                    for r in kvs.RULES}})
    per_run = pd.DataFrame(rows)
    monkeypatch.setattr(kvs, "SCOPES", (("four processes", "output-boba", ("logei",), ("gaussian",),
                                         {"7-11": (7,), "12-16": (12,)}),))
    monkeypatch.setattr(kvs, "scope_levels", lambda arm, acqs: ("learned", "mag"))
    cells, frames = kvs.score_scopes(per_run, opt_z)
    sel = kvs.select_on_train(cells, frames, "four processes")
    for cell in kvs.CELL_ORDER:
        c = sel["by_cell"]["mag"][cell]
        assert c["rule"] == "lcb1" and c["learned_rule_chosen_on_7_11"] == "lcb1"
        assert c["test"]["gain"] == pytest.approx(0.5 - 0.35)
        assert c["over_learned_chosen"]["inc"] == pytest.approx(0.45 - 0.35)
    assert sel["one_rule"]["mag"]["rule"] == "lcb1"
    assert sel["one_rule"]["mag"]["test"]["pooled_all_cells"]["gain"] == pytest.approx(0.15)


def test_scoring_is_the_adaptation_estimand():
    runs, opt_z = _toy_runs()
    pf = kvs.paired_frames(runs, ("learned", "mag"), opt_z)
    block = kvs.cell_of(pf[("mag", "lcb1")], "1sigma_from_trial_1")
    s = kvs.summary(block)
    gain = np.mean([(1.0 - 0.6) / 2.0, (2.0 - 1.6) / 4.0, (0.5 - 0.3) / 1.0])
    cost = np.mean([(1.0 - 0.2) / 2.0, (2.0 - 0.4) / 4.0, (0.5 - 0.1) / 1.0])
    price = np.mean([0.0, 0.1 / 4.0, 0.0])
    assert s["gain"] == pytest.approx(gain) and s["cost"] == pytest.approx(cost)
    assert s["price"] == pytest.approx(price) and s["recovered"] == pytest.approx(gain / cost)
    assert s["recovered_suppressed"] == "" and s["gain_lo"] <= s["gain"] <= s["gain_hi"]
    # the paired increment over the learned rule, which here ships the standard design
    inc = kvs.increment(kvs.cell_of(pf[("learned", "lcb1")], "1sigma_from_trial_1"), block)
    assert inc["inc"] == pytest.approx(gain) and inc["inc_price"] == pytest.approx(price)
    # a reference cost negative on one landscape is not a share
    runs_neg, _ = _toy_runs(cost_sign=-1.0)
    s_neg = kvs.summary(kvs.cell_of(kvs.paired_frames(runs_neg, ("mag",), opt_z)[("mag", "lcb1")],
                                    "1sigma_from_trial_1"))
    assert "negative on 1 of 3" in s_neg["recovered_suppressed"]
