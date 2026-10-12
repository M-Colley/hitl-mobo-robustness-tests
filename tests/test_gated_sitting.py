"""The gated sittings and the untouched-seed sitting of sixteen (scripts/checks/gated_sitting.py).

What must hold: (1) the gate reads the GP that ranks the sitting's candidates, with the noise in the
objective's units; (2) a run whose gate does not fire is the standard process and one whose gate fires is
the sitting, in the noisy run and its clean twin alike; (3) the threshold rule picks what its definition
says on a fixture whose answer is known, ties to the larger threshold; (4) the scores are the
analyse_boba_adaptations estimand. The tiny-arm test runs the refit on a real 16-trial run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import gated_sitting as gs  # noqa: E402
import replay_end_of_study as eos  # noqa: E402

CELLS = [(s, o) for s in (0.05, 0.25, 1.0, 5.0) for o in (0, 20)]
# The gate statistic of a noisy run by magnitude, and of every clean twin.
NOISE_BY_SIGMA = {0.05: 0.2, 0.25: 0.3, 1.0: 0.8, 5.0: 3.0}
CLEAN_NOISE = 0.25


def frame_fixture(n_land: int = 20, seeds=(7, 8), sit_gain_5: float = 0.1, sit_loss: float = 0.02,
                  sit_price: float = 0.03) -> pd.DataFrame:
    """A per-run frame as gated_sitting.build_frame returns it (opt_z units), one run per cell and seed.

    The sitting of sixteen gains sit_gain_5 at 5 sigma, loses sit_loss in every other cell and costs
    sit_price on the clean twin; LCB1 gains 0.01 and LCB2 0.02 everywhere; the cost of error is 0.5.
    """
    rows = []
    for i in range(n_land):
        for acq in ("logei", "qnei"):
            for seed in seeds:
                for sigma, onset in CELLS:
                    ref_n, ref_c = 1.0, 0.5
                    sit = ref_n - (sit_gain_5 if sigma == 5.0 else -sit_loss)
                    rows.append({"file": f"l{i}_{acq}_{seed}_{sigma}_{onset}.csv", "dataset": f"land{i:02d}",
                                 "acquisition": acq, "seed": seed, "error_model": "gaussian", "sigma": sigma,
                                 "onset": onset, "opt_z": 1.0, "ref_noisy": ref_n, "ref_clean": ref_c,
                                 "regret_lcb1": ref_n - 0.01, "regret_lcb2": ref_n - 0.02,
                                 "regret_lcb1_clean": ref_c + 0.02, "regret_lcb2_clean": ref_c + 0.03,
                                 "sit16": sit, "sit16_clean": ref_c + sit_price,
                                 "sit2": ref_n - 0.005, "sit2_clean": ref_c + 0.01,
                                 "noise_sd_34": NOISE_BY_SIGMA[sigma], "noise_sd_34_clean": CLEAN_NOISE,
                                 "noise_sd_48": NOISE_BY_SIGMA[sigma], "noise_sd_48_clean": CLEAN_NOISE,
                                 "noise_frac_34": 0.5, "noise_frac_34_clean": 0.5,
                                 "noise_frac_48": 0.5, "noise_frac_48_clean": 0.5})
    return pd.DataFrame(rows)


# --- the gate statistic --------------------------------------------------------------


def test_gp_noise_is_the_standardised_noise_rescaled_to_the_objective():
    gp = SimpleNamespace(likelihood=SimpleNamespace(noise=torch.tensor([0.04], dtype=torch.double)),
                         outcome_transform=SimpleNamespace(stdvs=torch.tensor([[3.0]], dtype=torch.double)))
    sd, frac, stdvs = gs.gp_noise(gp)
    assert frac == pytest.approx(0.2) and stdvs == pytest.approx(3.0) and sd == pytest.approx(0.6)


def test_gp_noise_agrees_with_the_rescoring_scripts_noise_on_a_real_fit():
    import boba_benchmarks as bb
    import rescore_ship_rules as rs

    spec = bb.BENCHMARKS["branin"]
    rng = np.random.default_rng(0)
    X = spec.bounds_low + rng.random((30, 2)) * (np.asarray(spec.bounds_high) - np.asarray(spec.bounds_low))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(30)
    bounds = torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)
    gp = eos.fit_loop_gp(torch.tensor(X), torch.tensor(y.reshape(-1, 1)), bounds, seed=3)
    _, diag = rs.fit_loop_gp(X, y, spec.bounds_low, spec.bounds_high, seed=3)
    assert gs.gp_noise(gp)[0] == pytest.approx(diag["gp_noise_sd"], rel=1e-9)


# --- gates and procedures ---------------------------------------------------------------


def test_a_gate_ships_the_sitting_only_where_it_fires_in_the_run_and_its_clean_twin():
    f = frame_fixture(n_land=2, seeds=(7,))
    for tau, sits_at in ((0.0, set(NOISE_BY_SIGMA)), (1.0, {5.0}), (np.inf, set())):
        trt_n, trt_c = gs.gated(f, 16, tau)
        fire = f["sigma"].isin(sits_at).to_numpy()
        np.testing.assert_array_equal(trt_n, np.where(fire, f["sit16"], f["ref_noisy"]))
        clean_fires = CLEAN_NOISE > tau
        np.testing.assert_array_equal(trt_c, f["sit16_clean"] if clean_fires else f["ref_clean"])
    # a run whose GP failed has no noise estimate and does not sit
    g = f.copy()
    g.loc[0, "noise_sd_34"] = np.nan
    assert gs.gated(g, 16, 0.0)[0][0] == g.loc[0, "ref_noisy"]


def test_the_two_stage_gate_sits_sixteen_first_and_two_only_if_sixteen_did_not_fire():
    f = frame_fixture(n_land=1, seeds=(7,))
    trt_n, trt_c = gs.two_stage(f, tau16=1.0, tau2=0.5)
    expect = np.where(f["noise_sd_34"] > 1.0, f["sit16"], np.where(f["noise_sd_48"] > 0.5, f["sit2"], f["ref_noisy"]))
    np.testing.assert_array_equal(trt_n, expect)
    np.testing.assert_array_equal(trt_c, f["ref_clean"])          # the clean twin's 0.25 fires neither gate
    np.testing.assert_array_equal(gs.two_stage(f, 0.0, 0.0)[1], f["sit16_clean"])


def test_the_threshold_rule_maximises_v_and_breaks_ties_towards_fewer_sittings():
    f = frame_fixture()
    tau, curve = gs.choose_threshold(f, 16, gs.TAU_GRID)
    # Thresholds in [0.8, 3.0) sit only at 5 sigma, where the sitting gains: V = 2 * 0.1 / 9, the maximum,
    # reached by 1.0, 1.25, 1.5, 2.0 and 2.5; the rule takes the largest.
    assert tau == 2.5
    v = curve.set_index("tau")["value"]
    assert v[2.5] == pytest.approx(2 * 0.1 / 9)
    assert v[0.0] == pytest.approx((2 * 0.1 - 6 * 0.02 - 0.03) / 9)     # always sit, and pay the clean price
    assert v[np.inf] == 0.0                                            # never sit: the standard process
    assert v[0.75] == pytest.approx((2 * 0.1 - 2 * 0.02) / 9)          # 1 sigma (noise 0.8) sits too


def test_scores_are_the_adaptations_estimand_with_increments_shares_and_holm():
    f = frame_fixture(n_land=6, seeds=(7, 8, 9))
    trt_n, trt_c = gs.gated(f, 16, 0.0)
    rows = pd.DataFrame(gs.score_procedure(f, "G16", trt_n, trt_c, np.random.default_rng(0), "scope",
                                           fires=gs.gate_fires(f, 16, 0.0),
                                           ungated=(f["sit16"].to_numpy(), f["sit16_clean"].to_numpy())))
    cell = rows.set_index("cell")
    assert cell.loc["5sigma_from_trial_21", "gain"] == pytest.approx(0.1)
    assert cell.loc["5sigma_from_trial_21", "share"] == pytest.approx(0.1 / 0.5)
    assert cell.loc["1sigma_from_trial_1", "gain"] == pytest.approx(-0.02)
    assert cell.loc["5sigma_from_trial_1", "inc_lcb1"] == pytest.approx(0.1 - 0.01)
    assert cell.loc["5sigma_from_trial_1", "inc_lcb2"] == pytest.approx(0.1 - 0.02)
    assert np.allclose(cell["inc_ungated"], 0.0)                       # tau 0 is the ungated sitting
    assert np.allclose(cell["price"], 0.03) and np.allclose(cell["fire_rate_clean"], 1.0)
    assert cell.loc["pooled", "value_uniform"] == pytest.approx((2 * 0.1 - 6 * 0.02 - 0.03) / 9)
    # a cost that is negative on one landscape withholds the share, never the gain
    g = f.copy()
    g.loc[(g["dataset"] == "land00") & (g["sigma"] == 1.0), "ref_clean"] = 2.0
    rows = pd.DataFrame(gs.score_procedure(g, "G16", *gs.gated(g, 16, 0.0), np.random.default_rng(0), "s"))
    r = rows.set_index("cell").loc["1sigma_from_trial_1"]
    assert np.isnan(r["share"]) and "negative" in r["share_suppressed"] and np.isfinite(r["gain"])
    table = gs.add_holm(pd.concat([rows.assign(procedure=p) for p in gs.PROCEDURES], ignore_index=True))
    in_cells = table[table["cell"] != "pooled"]
    assert in_cells["gain_p_holm_24"].notna().sum() == 24
    assert (in_cells["gain_p_holm_24"] >= in_cells["gain_p_holm_cells"] - 1e-12).all()


def test_attach_noise_refuses_a_refit_that_is_not_the_gp_ranking_the_sitting():
    base = pd.DataFrame({"file": ["a.csv", "b.csv"], "dataset": ["d", "d"], "acquisition": ["logei", "logei"],
                         "seed": [7, 7], "top16": [1.0, 2.0], "top16_clean": [0.5, 0.5]})
    noise = pd.DataFrame({"arm": "arm", "file": ["a.csv", "b.csv", "clean.csv"], "dataset": "d",
                          "acquisition": "logei", "seed": 7, "baseline": [False, False, True], "n": 34,
                          "noise_sd": [1.0, 2.0, 0.3], "noise_frac": 0.1, "top_regret": [1.0, 2.0, 0.5],
                          "gp_failed": False})
    out = gs.attach_noise(base, noise, "arm", ks=(16,))
    assert out["noise_sd_34"].tolist() == [1.0, 2.0] and out["noise_sd_34_clean"].tolist() == [0.3, 0.3]
    bad = noise.copy()
    bad.loc[1, "top_regret"] += 1e-6
    with pytest.raises(ValueError, match="top LCB1 candidate"):
        gs.attach_noise(base, bad, "arm", ks=(16,))
    with pytest.raises(KeyError, match="no refit"):
        gs.attach_noise(base, noise[noise["file"] != "b.csv"], "arm", ks=(16,))


def test_json_safe_writes_infinite_thresholds_as_text():
    assert json.loads(json.dumps(gs.json_safe({"t": [1.0, np.inf], "k": np.int64(3)}))) == {"t": [1.0, "inf"], "k": 3}


def test_printed_paths_never_carry_the_local_home_directory(tmp_path):
    # The register files are tracked and mirrored anonymously: repository-relative paths only.
    assert gs.rel(gs.REPO / "output-boba" / "analysis" / "review" / "x.csv") == "output-boba/analysis/review/x.csv"
    assert gs.rel(tmp_path / "elsewhere.csv") == "elsewhere.csv"


# --- fresh-k16 ---------------------------------------------------------------------------


def test_the_sitting_of_sixteen_over_twelve_is_the_paired_difference_on_the_same_runs():
    rows = []
    for i in range(5):
        for seed in (27, 28):
            for k, reg in ((12, 1.0), (16, 0.8)):
                rows.append({"dataset": f"land{i}", "acquisition": "logei", "seed": seed, "error_model": "gaussian",
                             "jitter_std": 5.0, "jitter_iteration": 20, "file": f"f{i}_{seed}.csv",
                             "family": "tournament", "k": float(k), "candidates": "lcb", "rho": 1.0,
                             "winner": "look", "regret_noisy": reg, "ref_noisy": 1.2, "ref_clean": 0.5,
                             "regret_clean": 0.6})
    oz = {f"land{i}": 2.0 for i in range(5)}
    out = gs.paired_over_k(pd.DataFrame(rows), oz, 16, 12, np.random.default_rng(0)).set_index("cell")
    assert out.loc["pooled", "inc_k12"] == pytest.approx(0.1)
    assert out.loc["5sigma_from_trial_21", "inc_k12_ahead"] == 5
    with pytest.raises(ValueError, match="same runs"):
        gs.paired_over_k(pd.DataFrame(rows[:-1]), oz, 16, 12, np.random.default_rng(0))


# --- the refit on a real (tiny) run --------------------------------------------------------

T_TINY, INIT, FN, SEED = 16, 5, "branin", 7


def _simulate(apply_error: bool) -> pd.DataFrame:
    import bo_sensor_error_simulation as sim
    import boba_benchmarks as bb

    stats = bb.load_stats()
    entry = stats[FN]
    spec = bb.BENCHMARKS[FN]
    config = sim.SimulationConfig(
        iterations=T_TINY, jitter_iteration=0, jitter_std=1.0 if apply_error else 0.0, single_error=False,
        initial_samples=INIT, candidate_pool=64, objective="value", objective_columns=["value"],
        param_columns=spec.param_columns, seed=SEED, error_model="gaussian" if apply_error else "none",
        error_bias=0.0, error_spike_prob=0.1, error_spike_std=0.5, dropout_strategy="hold_last",
        normalize_objective=False, objective_weights=None, acq_num_restarts=2, acq_raw_samples=32,
        acq_maxiter=50, acq_mc_samples=32, ref_point=None,
    )
    torch.manual_seed(SEED)
    jitter_rng = np.random.default_rng(np.random.SeedSequence([SEED, 0, 0, 1_000_000, 0])) if apply_error else None
    frame = sim.run_simulation(
        oracle=bb.SyntheticOracle.from_stats(FN, stats), bounds=sim.Bounds(low=spec.bounds_low, high=spec.bounds_high),
        config=config, acq=sim.AcquisitionConfig(name="logei"), rng=np.random.default_rng(SEED),
        jitter_rng=jitter_rng, run_id="t", apply_error=apply_error, oracle_model="exact",
        y_opt=(entry["y_opt"] - entry["mean"]) / entry["std"],
    )
    frame["dataset"] = FN
    return frame


@pytest.fixture(scope="module")
def tiny_arm(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("arm") / "output-tiny"
    (root / FN).mkdir(parents=True)
    args = {"iterations": T_TINY, "initial_samples": INIT, "error_ar1_rho": 0.8, "input_error": "none"}
    (root / "run_metadata.json").write_text(json.dumps({"args": args}), encoding="utf-8")
    stem = f"bo_sensor_error_{FN}_value_logei_seed{SEED}"
    _simulate(False).to_csv(root / FN / f"{stem}_baseline_exact.csv", index=False)
    _simulate(True).to_csv(root / FN / f"{stem}_jittered_exact_gaussian_jit0_std1.0.csv", index=False)
    return root


def test_the_refit_reads_the_gp_that_ranks_the_sittings_candidates(tiny_arm, tmp_path):
    ks = (3, 8)
    tasks, counts = gs.refit_tasks(tiny_arm, seeds=(SEED,), processes=("gaussian",), acqs=("logei",))
    assert len(tasks) == 1 and counts["noisy runs"] == 1
    rows = pd.DataFrame(gs.refit_stem(tasks[0], ks=ks))
    assert set(rows["n"]) == {T_TINY - k for k in ks} and len(rows) == 2 * len(ks)
    # the replay's own sitting on the same prefixes: the same top LCB1 candidate, noisy run and clean twin
    replay = eos.replay_stem_rows({**tasks[0], "settings": eos.Settings(tournament_k=ks, confirm_k=())})
    for k in ks:
        r = replay[(replay["k"] == k) & (replay["candidates"] == "lcb")].iloc[0]
        at = rows[rows["n"] == T_TINY - k].set_index("baseline")
        assert at.loc[False, "top_regret"] == pytest.approx(r["top_candidate_regret_noisy"], abs=1e-12)
        assert at.loc[True, "top_regret"] == pytest.approx(r["top_candidate_regret_clean"], abs=1e-12)
    # and the noise is that GP's, in the objective's units
    import boba_benchmarks as bb

    spec = bb.BENCHMARKS[FN]
    bounds = torch.tensor(np.vstack([spec.bounds_low, spec.bounds_high]), dtype=torch.double)
    noisy_path = Path(tasks[0]["runs"][0]["path"])
    run = eos.read_run(noisy_path, T_TINY)
    n = T_TINY - 3
    gp = eos.fit_loop_gp(torch.tensor(run.X[:n]), torch.tensor(run.observed[:n].reshape(-1, 1)), bounds,
                         eos.torch_seed(run.name, n))
    expect = float((gp.likelihood.noise.detach().reshape(-1)[0]
                    * gp.outcome_transform.stdvs.detach().reshape(-1)[0] ** 2).sqrt())
    got = rows[(rows["n"] == n) & ~rows["baseline"]]["noise_sd"].iloc[0]
    assert got == pytest.approx(expect, rel=1e-12)
    assert not rows["gp_failed"].any()
    # the cache round-trips
    path = gs._write_cache(rows.to_dict("records"), tmp_path / "noise.csv.gz")
    back = gs.read_noise(tmp_path / "noise.csv.gz")
    np.testing.assert_allclose(back.sort_values(["file", "n"])["noise_sd"], path["noise_sd"])
