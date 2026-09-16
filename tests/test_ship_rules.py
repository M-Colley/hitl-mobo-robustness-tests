"""Ship rules (scripts/rescore_ship_rules.py and scripts/analyse_ship_rules.py).

What must hold: the standard rule reproduces the logged deployed regret or the
re-scoring stops; a replication arm's best_mean rule is recognised rather than
reported as a mismatch; a slip is scored at the design that was written down;
the refit is the loop's own GP; the journal resumes; and the analysis computes
the adaptation estimand (cost, gain and price against the standard
best_observed rule) with the 0.05 SD cells kept out of the headline pool.
"""
from __future__ import annotations

import argparse
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

import analyse_ship_rules as asr  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
import bo_sensor_error_simulation as sim  # noqa: E402
import rescore_ship_rules as rsr  # noqa: E402

NOISY_NAME = "bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0.csv"
CLEAN_NAME = "bo_sensor_error_branin_value_logei_seed7_baseline_exact.csv"


def _info(name: str) -> dict:
    info = rsr.parse_run_name(name)
    assert info is not None
    return {**info, "file": f"{info['dataset']}/{name}"}


@pytest.fixture(scope="module")
def real_run() -> pd.DataFrame:
    """A short real loop on branin under gaussian error (few trials, small pool)."""
    stats = bb.load_stats()
    oracle = bb.SyntheticOracle.from_stats("branin", stats)
    spec = bb.BENCHMARKS["branin"]
    config = sim.SimulationConfig(
        iterations=9, jitter_iteration=0, jitter_std=1.0, single_error=False,
        initial_samples=4, candidate_pool=64, objective="value",
        objective_columns=["value"], param_columns=spec.param_columns, seed=7,
        error_model="gaussian", error_bias=0.5, error_spike_prob=0.1,
        error_spike_std=0.5, dropout_strategy="hold_last", normalize_objective=False,
        objective_weights=None, acq_num_restarts=2, acq_raw_samples=32,
        acq_maxiter=50, acq_mc_samples=32, ref_point=None,
    )
    torch.manual_seed(7)
    frame = sim.run_simulation(
        oracle=oracle, bounds=sim.Bounds(low=spec.bounds_low, high=spec.bounds_high), config=config,
        acq=sim.AcquisitionConfig(name="logei"), rng=np.random.default_rng(7),
        jitter_rng=np.random.default_rng(1234), run_id="t", apply_error=True, oracle_model="exact",
        y_opt=oracle.y_opt,
    )
    frame["dataset"] = "branin"
    return frame


def _hand_log(observed, true, X, rule="best_observed", deployed=None, seed=7, std=1.0, onset=0,
              baseline=False) -> pd.DataFrame:
    """A log with a chosen ship rule's final regret, for cases a short real run cannot force."""
    observed, true = np.asarray(observed, float), np.asarray(true, float)
    score = true if deployed is None else np.asarray(deployed, float)
    idx = sim._best_mean_index(list(X), list(observed)) if rule == "best_mean" else int(np.argmax(observed))
    y_opt = 2.0
    frame = pd.DataFrame(np.asarray(X, float), columns=["x0", "x1"])
    frame.insert(0, "iteration", np.arange(1, len(observed) + 1))
    frame["objective_true"] = true
    frame["objective_observed"] = observed
    if deployed is not None:
        frame["objective_true_deployed"] = score
    frame["acquisition"] = "logei"
    frame["seed"] = seed
    frame["error_model"] = "none" if baseline else "gaussian"
    frame["jitter_std"] = 0.0 if baseline else std
    frame["jitter_iteration"] = onset
    frame["observation_noise"] = "learned"
    frame["known_noise_var"] = 1e-6
    frame["param_columns"] = "x0,x1"
    frame["y_opt"] = y_opt
    # Only the final row is the deployed regret the check reads.
    frame["inference_simple_regret_true"] = y_opt - score[idx]
    frame["dataset"] = "branin"
    return frame


def _design_grid(n: int, repeat: tuple[int, int] | None = None) -> np.ndarray:
    spec = bb.BENCHMARKS["branin"]
    rng = np.random.default_rng(3)
    X = spec.lo + rng.random((n, 2)) * (spec.hi - spec.lo)
    if repeat is not None:
        X[repeat[1]] = X[repeat[0]]
    return X


# ---------------------------------------------------------------------------
# rescore_ship_rules.py
# ---------------------------------------------------------------------------


def test_run_names_are_parsed_with_their_variants():
    b = rsr.parse_run_name("bo_sensor_error_hartmann_3_value_logei_seed7_baseline_exact.csv")
    assert (b["dataset"], b["acquisition"], b["seed"], b["baseline"], b["variant"]) == ("hartmann_3", "logei", 7, True, "")
    assert rsr.parse_run_name("bo_sensor_error_branin_value_ei_seed9_baseline_exact_inc-observed_max.csv")[
        "variant"] == "inc-observed_max"
    j = rsr.parse_run_name("bo_sensor_error_levy_10_value_qnei_seed12_jittered_exact_gaussian_jit20_std0.25_rep10.csv")
    assert (j["dataset"], j["error_model"], j["jitter_iteration"], j["jitter_std"], j["variant"]) == (
        "levy_10", "gaussian", 20, 0.25, "rep10")
    s = rsr.parse_run_name("bo_sensor_error_branin_value_ucb_seed7_jittered_exact_slip_jit0_std0.15_rec-actual.csv")
    assert (s["error_model"], s["jitter_std"], s["variant"]) == ("slip", 0.15, "rec-actual")
    bias = rsr.parse_run_name("bo_sensor_error_branin_value_ucb_seed7_jittered_exact_bias_jit0_std1.0_bias0.05.csv")
    assert (bias["jitter_std"], bias["variant"]) == (1.0, "bias0.05")
    assert rsr.parse_run_name("bo_synthetic_error_summary.csv") is None
    # a fitted-oracle log has no benchmark box to refit on
    assert rsr.parse_run_name("bo_sensor_error_ehmi_value_ei_seed1_baseline_extra_trees.csv") is None
    assert rsr.parse_seeds("7-9,12") == {7, 8, 9, 12}


def test_each_rule_picks_its_design():
    X = np.array([[0.1, 0.0], [0.2, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]])
    observed = np.array([2.68, 2.7, 2.0, 1.0, 1.9])       # one lucky rating of a design rated twice
    score = np.array([0.0, -1.0, -1.0, 0.5, 0.2])
    mu = np.array([1.0, 2.0, 2.0, 1.5, 0.0])
    sd = np.array([0.1, 1.0, 1.0, 0.45, 0.0])
    idx = rsr.select_indices(X, observed, score, mu, sd)
    # best_mean averages the repeated design (2.35) below design 0; the LCBs
    # trade mean for certainty one and two SDs deep
    assert idx == {"best_observed": 1, "best_mean": 0, "best_visited": 3, "pm": 1, "lcb1": 3, "lcb2": 0}
    undecided = rsr.select_indices(X, observed, score, np.array([np.nan, 1, 1, 1, 1]), sd)
    assert undecided["pm"] == undecided["lcb1"] == undecided["lcb2"] == -1


def test_standard_rule_reproduces_a_real_run(real_run):
    rec = rsr.rescore_frame(real_run, _info(NOISY_NAME), "best_observed", seed=1)
    assert abs(rec["regret_best_observed"] - real_run["inference_simple_regret_true"].iloc[-1]) <= 1e-9
    assert rec["gp_fit_ok"] and rec["score_column"] == "objective_true" and rec["n_train"] == len(real_run)
    true = real_run["objective_true"].to_numpy()
    y_opt = float(real_run["y_opt"].iloc[0])
    for rule in rsr.RULES:
        assert np.isfinite(rec[f"regret_{rule}"])
        assert rec[f"regret_{rule}"] == pytest.approx(y_opt - true[rec[f"idx_{rule}"]], abs=1e-12)
        # the oracle pick is a floor for every rule that ships a visited design
        assert rec["regret_best_visited"] <= rec[f"regret_{rule}"] + 1e-12


def test_a_log_that_disagrees_with_the_rescoring_stops(real_run):
    tampered = real_run.copy()
    tampered.loc[tampered.index[-1], "inference_simple_regret_true"] += 1e-6
    with pytest.raises(rsr.ReproductionError):
        rsr.rescore_frame(tampered, _info(NOISY_NAME), "best_observed", seed=1)


def test_a_log_that_is_not_the_run_its_name_says_stops(real_run):
    with pytest.raises(ValueError, match="seed"):
        rsr.rescore_frame(real_run, _info(NOISY_NAME.replace("seed7", "seed8")), "best_observed", seed=1)


def test_a_best_mean_arm_is_recognised_not_flagged():
    X = _design_grid(8, repeat=(2, 5))
    observed = [0.1, 0.3, 1.2, 0.4, 0.9, -0.2, 0.2, 1.0]   # design 2 rated 1.2 and -0.2: mean 0.5
    true = [0.0, 0.2, 0.5, 0.3, 0.8, 0.5, 0.1, 0.9]
    log = _hand_log(observed, true, X, rule="best_mean")
    rec = rsr.rescore_frame(log, _info(NOISY_NAME), "best_mean", seed=1)
    assert rec["logged_rule"] == "best_mean" and not rec["best_observed_reproduces"]
    assert rec["idx_best_mean"] == 7 and rec["idx_best_observed"] == 2
    with pytest.raises(rsr.ReproductionError):
        rsr.rescore_frame(log, _info(NOISY_NAME), "best_observed", seed=1)


def test_a_slip_is_scored_at_the_design_written_down():
    X = _design_grid(7)
    observed = [0.1, 0.3, 1.2, 0.4, 0.9, 0.2, 1.0]
    true = [0.0, 0.2, 1.1, 0.3, 0.8, 0.1, 0.9]          # where the person went
    deployed = [0.0, 0.2, -0.5, 0.3, 0.8, 0.1, 0.9]     # what the log says, which is what ships
    log = _hand_log(observed, true, X, deployed=deployed)
    rec = rsr.rescore_frame(log, _info(NOISY_NAME), "best_observed", seed=1)
    assert rec["score_column"] == "objective_true_deployed"
    assert rec["regret_best_observed"] == pytest.approx(2.0 + 0.5)
    assert rec["regret_best_visited"] == pytest.approx(2.0 - 0.9)


def test_arm_rule_and_surrogate_come_from_the_metadata(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    assert rsr.arm_settings(arm)["inference_rule"] == "best_observed"      # no metadata: the default

    def settings(**args):
        (arm / "run_metadata.json").write_text(json.dumps({"args": args}), encoding="utf-8")
        return rsr.arm_settings(arm)

    assert settings(replicate_first=10, final_rerate="0,0", inference_rule=None)["inference_rule"] == "best_mean"
    assert settings(final_rerate="3,2")["inference_rule"] == "best_mean"
    assert settings(incumbent="observed_max", observation_noise="known")["inference_rule"] == "best_observed"
    with pytest.raises(NotImplementedError):
        settings(likelihood="student_t")
    with pytest.raises(NotImplementedError):
        settings(input_noise_model="nigp")


def test_the_refit_is_the_loops_gp():
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    spec = bb.BENCHMARKS["hartmann_3"]
    X = _design_grid(12)[:, :1].repeat(3, axis=1) * 0 + np.random.default_rng(0).random((12, 3))
    y = np.sin(3 * X).sum(axis=1)
    gp, diag = rsr.fit_loop_gp(X, y, spec.bounds_low, spec.bounds_high, seed=5)
    default = SingleTaskGP(torch.tensor(X), torch.tensor(y.reshape(-1, 1)))
    # the installed BoTorch's default kernel, whatever it is -- no substitute
    assert type(gp.covar_module) is type(default.covar_module)
    assert repr(gp.covar_module) == repr(default.covar_module)
    np.testing.assert_array_equal(gp.input_transform.bounds.numpy(), np.vstack([spec.bounds_low, spec.bounds_high]))
    assert isinstance(gp.outcome_transform, Standardize) and diag["gp_fit_ok"]
    known, _ = rsr.fit_loop_gp(X, y, spec.bounds_low, spec.bounds_high, yvar=np.full(12, 0.25), seed=5)
    assert isinstance(known.likelihood, FixedNoiseGaussianLikelihood)


def test_the_journal_resumes_and_survives_a_torn_line(tmp_path):
    arm = tmp_path / "arm"
    folder = arm / "branin"
    folder.mkdir(parents=True)
    X = _design_grid(7)
    _hand_log([0.1, 0.3, 1.2, 0.4, 0.9, 0.2, 1.0], [0.0, 0.2, 1.1, 0.3, 0.8, 0.1, 0.9], X, baseline=True).to_csv(
        folder / CLEAN_NAME, index=False)
    _hand_log([0.4, 0.3, 0.2, 1.4, 0.9, 0.2, 1.0], [0.0, 0.2, 1.1, 0.3, 0.8, 0.1, 0.9], X).to_csv(
        folder / NOISY_NAME, index=False)
    (folder / NOISY_NAME.replace("logei", "random")).write_text("model-free floors are skipped")
    threads = torch.get_num_threads()
    try:
        table = rsr.main(["--input-dir", str(arm), "--workers", "1"])
        assert len(table) == 2 and set(table["baseline"]) == {True, False}
        with pytest.raises(SystemExit):
            rsr.main(["--input-dir", str(arm), "--workers", "1"])
        journal = arm / "analysis" / "ship_rules_per_run.jsonl"
        with open(journal, "a", encoding="utf-8") as fh:
            fh.write('{"file": "branin/torn')
        again = rsr.main(["--input-dir", str(arm), "--workers", "1", "--resume"])
        assert len(again) == 2
        assert all(json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines())
        pd.testing.assert_frame_equal(table, again)
    finally:
        torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# analyse_ship_rules.py
# ---------------------------------------------------------------------------


def _row(dataset, acq, seed, baseline, std, onset, regrets: dict, variant="", error_model="gaussian") -> dict:
    row = {"dataset": dataset, "acquisition": acq, "error_model": "none" if baseline else error_model,
           "jitter_std": 0.0 if baseline else std, "jitter_iteration": np.nan if baseline else onset,
           "seed": seed, "baseline": baseline, "variant": variant,
           "file": f"{dataset}/{acq}_s{seed}_{'clean' if baseline else f'{error_model}_{std}_{onset}_{variant}'}"}
    for rule in rsr.RULES:
        row[f"regret_{rule}"] = regrets.get(rule, regrets["best_observed"])
    return row


def _args(**kw) -> argparse.Namespace:
    base = dict(pool_min_std=0.25, input_pool_min_std=0.05, per_run_root=None, seeds=None)
    return argparse.Namespace(**{**base, **kw})


def _pooled(rows, rule, cell):
    hits = [r for r in rows if r["rule"] == rule and r["cell"] == cell and np.isnan(r["jitter_iteration"])]
    assert len(hits) == 1
    return hits[0]


def test_within_arm_estimand_against_the_standard_rule():
    opt_z = {"branin": 1.0, "ackley": 2.0}
    rows = [
        _row("branin", "logei", 7, True, 0, 0, {"best_observed": 0.2, "pm": 0.3}),
        _row("ackley", "logei", 7, True, 0, 0, {"best_observed": 0.4, "pm": 0.4}),
        _row("branin", "logei", 7, False, 1.0, 0, {"best_observed": 1.2, "pm": 0.7}),
        _row("ackley", "logei", 7, False, 1.0, 0, {"best_observed": 2.4, "pm": 1.4}),
        _row("branin", "logei", 7, False, 0.05, 0, {"best_observed": 0.25, "pm": 0.3}),
        _row("ackley", "logei", 7, False, 0.05, 0, {"best_observed": 0.4, "pm": 0.4}),
    ]
    out = asr.within_arm("arm", pd.DataFrame(rows), opt_z, ["best_observed", "pm"], _args())
    assert {r["rule"] for r in out} == {"pm"}                      # the standard is not scored against itself
    big = _pooled(out, "pm", "pooled_std_ge_0.25")
    # per landscape, in units of opt_z: cost 1.0 and 1.0, gain 0.5 and 0.5, price 0.1 and 0.0
    assert big["cost"] == pytest.approx(1.0) and big["gain"] == pytest.approx(0.5)
    assert big["price"] == pytest.approx(0.05) and big["recovered"] == pytest.approx(0.5)
    assert big["n_cells"] == 2
    small = _pooled(out, "pm", "pooled_std_lt_0.25")               # the 0.05 SD cells, on their own
    assert small["cost"] == pytest.approx(0.025) and small["gain"] == pytest.approx(-0.025)
    conditions = [r for r in out if r["cell"] == "condition"]
    assert sorted(r["jitter_std"] for r in conditions) == [0.05, 1.0]
    assert all("wilcoxon_p_fdr" in r for r in conditions)


def test_input_error_arms_pool_on_their_own_scale():
    assert asr.pool_threshold("slip", _args()) == 0.05
    assert asr.pool_threshold("gaussian", _args()) == 0.25


def test_cross_arm_pools_the_reference_acquisitions(tmp_path):
    opt_z = asr.landscape_opt_z(["branin", "ackley"])
    ref, trt = [], []
    for d in ("branin", "ackley"):
        z = opt_z[d]
        # reference: two standard acquisitions, their mean is the standard
        for acq, clean, noisy in (("logei", 0.1, 0.9), ("qnei", 0.3, 1.1)):
            ref.append(_row(d, acq, 7, True, 0, 0, {"best_observed": clean * z}))
            ref.append(_row(d, acq, 7, False, 1.0, 0, {"best_observed": noisy * z}))
        trt.append(_row(d, "qkg", 7, True, 0, 0, {"best_observed": 0.5 * z, "lcb1": 0.25 * z}))
        trt.append(_row(d, "qkg", 7, False, 1.0, 0, {"best_observed": 0.9 * z, "lcb1": 0.6 * z}))
    for name, rows in (("A", ref), ("B", trt)):
        (tmp_path / name).mkdir()
        pd.DataFrame(rows).to_csv(tmp_path / name / "ship_rules_per_run.csv", index=False)
    spec = {"name": "t", "dir": "B", "ref": "A", "acqs": "qkg", "ref_acqs": "logei,qnei", "seeds": "7",
            "pool": True, "error_model": "gaussian", "variant": "auto", "ref_variant": "auto"}
    out = asr.cross_arm(spec, ["best_observed", "lcb1"], _args(per_run_root=tmp_path))
    # standard: clean 0.2, noisy 1.0 -> cost 0.8; lcb1: noisy 0.6 -> gain 0.4, clean 0.25 -> price 0.05
    lcb = _pooled(out, "lcb1", "pooled_std_ge_0.25")
    assert lcb["cost"] == pytest.approx(0.8) and lcb["gain"] == pytest.approx(0.4)
    assert lcb["price"] == pytest.approx(0.05) and lcb["recovered"] == pytest.approx(0.5)
    bo = _pooled(out, "best_observed", "pooled_std_ge_0.25")
    assert bo["gain"] == pytest.approx(0.1) and bo["price"] == pytest.approx(0.3)
    # a reference cell missing one of its acquisitions would average a different set
    pd.DataFrame([r for r in ref if not (r["dataset"] == "ackley" and r["acquisition"] == "qnei")]).to_csv(
        tmp_path / "A" / "ship_rules_per_run.csv", index=False)
    with pytest.raises(ValueError, match="lack some"):
        asr.cross_arm(spec, ["lcb1"], _args(per_run_root=tmp_path))


def test_twins_must_exist_once_and_rules_must_have_decided():
    clean = _row("branin", "logei", 7, True, 0, 0, {"best_observed": 0.2})
    noisy = _row("branin", "logei", 7, False, 1.0, 0, {"best_observed": 1.0})
    with pytest.raises(ValueError, match="no clean twin"):
        asr.twin_frame(pd.DataFrame([noisy]), "best_observed")
    with pytest.raises(ValueError, match="several clean runs"):
        asr.twin_frame(pd.DataFrame([clean, {**clean, "file": "other"}, noisy]), "best_observed")
    with pytest.raises(ValueError, match="no regret"):
        asr.twin_frame(pd.DataFrame([clean, {**noisy, "regret_pm": np.nan}]), "pm")
    two_variants = pd.DataFrame([clean, noisy, {**noisy, "variant": "rep10", "file": "x"}])
    with pytest.raises(ValueError, match="noisy variants"):
        asr.pick_variant(two_variants, "auto", "t")
    assert len(asr.pick_variant(two_variants, "rep10", "t")) == 2
