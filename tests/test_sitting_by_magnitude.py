"""The sitting's increment over the zero-trial ship rules, its price, and Holm's correction.

A synthetic fixture whose answers are known in closed form: every sitting run
beats LCB1 by exactly 0.01 * k of the achievable improvement, its own top
candidate by 0.02, LCB1 gains 0.05 raw over the standard process, and the clean
twin charges the sitting 0.01 * k and LCB1 0.02.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sitting_by_magnitude as sbm  # noqa: E402

SEEDS = tuple(range(7, 17))


def _opt_z(n_land: int) -> dict[str, float]:
    return {f"land{i:02d}": 2.0 + i for i in range(n_land)}


def make_fixture(n_land: int = 4, ks=(2, 3), cells=((1.0, 0),), inc_per_k: float = 0.01,
                 error_model: str = "gaussian") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Replay rows (as the per-run CSV has them) and the matching ship-rule rescoring rows."""
    oz = _opt_z(n_land)
    rng = np.random.default_rng(0)
    rows, ship = [], []
    for ds, z in oz.items():
        for acq in ("logei", "qnei"):
            for seed in SEEDS:
                ref_clean = 0.5 + 0.1 * z
                ship.append({"dataset": ds, "acquisition": acq, "error_model": "none", "jitter_std": 0.0,
                             "jitter_iteration": 0.0, "seed": seed, "baseline": True, "variant": np.nan,
                             "file": f"{ds}/bo_{ds}_{acq}_seed{seed}_baseline.csv",
                             "regret_best_observed": ref_clean, "regret_pm": ref_clean + 0.03 * z,
                             "regret_lcb1": ref_clean + 0.02 * z, "regret_lcb2": ref_clean + 0.01 * z})
                for sigma, onset in cells:
                    fname = f"bo_{ds}_{acq}_seed{seed}_{error_model}_jit{onset}_std{sigma:g}.csv"
                    ref_noisy = ref_clean + 0.3 * z + rng.uniform(0.0, 0.1)
                    lcb1 = ref_noisy - 0.05
                    ship.append({"dataset": ds, "acquisition": acq, "error_model": error_model, "jitter_std": sigma,
                                 "jitter_iteration": float(onset), "seed": seed, "baseline": False,
                                 "variant": np.nan, "file": f"{ds}/{fname}", "regret_best_observed": ref_noisy,
                                 "regret_pm": ref_noisy - 0.03, "regret_lcb1": lcb1, "regret_lcb2": ref_noisy - 0.04})
                    for k in ks:
                        reg = lcb1 - inc_per_k * k * z
                        rows.append({"dataset": ds, "acquisition": acq, "seed": seed, "error_model": error_model,
                                     "jitter_std": sigma, "jitter_iteration": onset, "file": fname,
                                     "family": "tournament", "k": float(k), "candidates": "lcb", "rho": 1.0,
                                     "winner": "look", "regret_noisy": reg, "ref_noisy": ref_noisy,
                                     "ref_clean": ref_clean, "regret_clean": ref_clean + 0.01 * k * z,
                                     "top_candidate_regret_noisy": reg + 0.02 * z})
    return pd.DataFrame(rows), pd.DataFrame(ship), oz


def as_loaded(raw: pd.DataFrame, oz: dict[str, float]) -> pd.DataFrame:
    """What sitting_by_magnitude.load returns for these rows."""
    z = raw["dataset"].map(oz)
    d = raw.assign(gain=(raw["ref_noisy"] - raw["regret_noisy"]) / z, cost=(raw["ref_noisy"] - raw["ref_clean"]) / z,
                   k=raw["k"].astype(int), opt_z=z)
    d["onset"] = d["jitter_iteration"].astype(int)
    d["sigma"] = d["jitter_std"].astype(float)
    return d


def test_attach_gives_the_paired_increment_and_the_prices():
    raw, ship, oz = make_fixture()
    m = sbm.attach_ship_rules(as_loaded(raw, oz), ship)
    assert len(m) == len(raw)
    np.testing.assert_allclose(m["inc_lcb1"], 0.01 * m["k"], atol=1e-12)
    np.testing.assert_allclose(m["inc_lcb1"], m["gain"] - m["gain_lcb1"], atol=1e-12)
    np.testing.assert_allclose(m["inc_top"], 0.02, atol=1e-12)
    np.testing.assert_allclose(m["gain_lcb1"], 0.05 / m["opt_z"], atol=1e-12)
    np.testing.assert_allclose(m["gain_pm"], 0.03 / m["opt_z"], atol=1e-12)
    np.testing.assert_allclose(m["price"], 0.01 * m["k"], atol=1e-12)
    np.testing.assert_allclose(m["price_lcb1"], 0.02, atol=1e-12)
    np.testing.assert_allclose(m["price_lcb2"], 0.01, atol=1e-12)


def test_attach_refuses_a_rescoring_that_does_not_reproduce_the_standard_process():
    raw, ship, oz = make_fixture()
    bad = ship.copy()
    bad.loc[~bad["baseline"], "regret_best_observed"] += 1e-6
    with pytest.raises(ValueError, match="standard process"):
        sbm.attach_ship_rules(as_loaded(raw, oz), bad)
    bad = ship.copy()
    bad.loc[bad["baseline"], "regret_best_observed"] += 1e-6
    with pytest.raises(ValueError, match="clean"):
        sbm.attach_ship_rules(as_loaded(raw, oz), bad)


def test_attach_refuses_a_run_without_a_rescoring():
    raw, ship, oz = make_fixture()
    dropped = ship[~(ship["file"].str.contains("seed9_") & ~ship["baseline"])]
    with pytest.raises(KeyError, match="no ship-rule rescoring"):
        sbm.attach_ship_rules(as_loaded(raw, oz), dropped)
    no_twin = ship[~(ship["file"].str.contains("seed9_") & ship["baseline"])]
    with pytest.raises(KeyError, match="clean twin"):
        sbm.attach_ship_rules(as_loaded(raw, oz), no_twin)


def test_increment_cell_recovers_the_known_increment_and_holm_over_k():
    raw, ship, oz = make_fixture(n_land=6)
    m = sbm.attach_ship_rules(as_loaded(raw, oz), ship)
    rows = pd.DataFrame(sbm.increment_cell(m, "cell", np.random.default_rng(1))).set_index("k")
    for k in (2, 3):
        r = rows.loc[k]
        for col in ("inc_lcb1_all", "inc_lcb1_train", "inc_lcb1_test", "inc_lcb1_test_lo", "inc_lcb1_test_hi"):
            assert r[col] == pytest.approx(0.01 * k, abs=1e-12)
        assert r["inc_top_test"] == pytest.approx(0.02, abs=1e-12)
        assert r["price_test"] == pytest.approx(0.01 * k, abs=1e-12)
        assert r["lcb1_price_test"] == pytest.approx(0.02, abs=1e-12)
        assert r["inc_lcb1_test_landscapes_ahead"] == 6
        # Six equal positive per-landscape increments: the exact two-sided signed-rank p is 2 / 2**6.
        assert r["inc_lcb1_p_test"] == pytest.approx(2 / 2 ** 6)
        # Two equal p over the two k: Holm doubles both.
        assert r["inc_lcb1_p_test_holm_over_k"] == pytest.approx(4 / 2 ** 6)
    assert rows.loc[2, "lcb1_gain_test"] == pytest.approx(np.mean([0.05 / z for z in oz.values()]))


def test_increment_cell_splits_the_increment_by_error_process():
    raw, ship, oz = make_fixture(n_land=5)
    other, other_ship, _ = make_fixture(n_land=5, inc_per_k=0.03, error_model="drift")
    both = pd.concat([raw, other], ignore_index=True)
    both_ship = pd.concat([ship, other_ship[~other_ship["baseline"]]], ignore_index=True)
    m = sbm.attach_ship_rules(as_loaded(both, oz), both_ship)
    rows = pd.DataFrame(sbm.increment_cell(m, "cell", np.random.default_rng(1))).set_index("k")
    lcb1 = np.mean([0.05 / z for z in oz.values()])
    for k in (2, 3):
        r = rows.loc[k]
        for n in ("test", "all"):
            assert r[f"inc_lcb1_{n}_gaussian"] == pytest.approx(0.01 * k, abs=1e-12)
            assert r[f"inc_lcb1_{n}_drift"] == pytest.approx(0.03 * k, abs=1e-12)
            assert r[f"lcb1_gain_{n}_gaussian"] == pytest.approx(lcb1)
            assert r[f"lcb1_gain_{n}_drift"] == pytest.approx(lcb1)
        # Equal runs per process on every landscape: the pooled increment is the mean of the two.
        assert r["inc_lcb1_test"] == pytest.approx(0.02 * k, abs=1e-12)
    assert "inc_lcb1_test_bias" not in rows.columns  # no column for a process that is absent


def _write_replay(root: Path, name: str, frame: pd.DataFrame) -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / name / "end_of_study_per_run.csv.gz", index=False)


def test_load_reads_each_replay_for_its_processes_and_refuses_an_overlap(tmp_path):
    gauss, _, oz = make_fixture(n_land=3)
    drift, _, _ = make_fixture(n_land=3, error_model="drift")
    moved = drift.assign(regret_noisy=drift["regret_noisy"] - 0.1)  # the sequential replay of the drift runs
    _write_replay(tmp_path, "shared", pd.concat([gauss, drift], ignore_index=True))
    _write_replay(tmp_path, "seq", moved)
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"functions": {d: {"opt_z": z} for d, z in oz.items()}}))
    d = sbm.load(tmp_path, stats, replay_dirs=("seq=drift", "shared=gaussian"))
    assert len(d) == len(gauss) + len(drift)
    got = d.set_index(["file", "k"])["regret_noisy"]
    np.testing.assert_allclose(got.loc[pd.MultiIndex.from_frame(moved[["file", "k"]])], moved["regret_noisy"])
    np.testing.assert_allclose(got.loc[pd.MultiIndex.from_frame(gauss[["file", "k"]])], gauss["regret_noisy"])
    # Unfiltered, a run found twice keeps its first occurrence.
    first = sbm.load(tmp_path, stats, replay_dirs=("seq", "shared")).set_index(["file", "k"])["regret_noisy"]
    np.testing.assert_allclose(first.loc[pd.MultiIndex.from_frame(moved[["file", "k"]])], moved["regret_noisy"])
    with pytest.raises(ValueError, match="more than one filtered replay"):
        sbm.load(tmp_path, stats, replay_dirs=("seq=drift", "shared=gaussian+drift"))
    with pytest.raises(ValueError, match="no error process"):
        sbm.parse_replay_dir("seq=")
    assert sbm.parse_replay_dir("seq=drift+ar1") == ("seq", ("drift", "ar1"))
    assert sbm.parse_replay_dir("shared") == ("shared", None)


def test_increment_cell_refuses_an_increment_that_is_not_gain_less_lcb1():
    raw, ship, oz = make_fixture(n_land=4)
    m = sbm.attach_ship_rules(as_loaded(raw, oz), ship)
    m.loc[m.index[0], "inc_lcb1"] += 0.5
    with pytest.raises(AssertionError, match="gain less LCB1"):
        sbm.increment_cell(m, "cell", np.random.default_rng(1))


def _holm_reference(p):
    """The step-down loop the script used before it called statsmodels."""
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def test_holm_by_hand_and_against_statsmodels_and_the_previous_loop():
    np.testing.assert_allclose(sbm.holm(np.array([0.01, 0.04, 0.03, 0.2])), [0.04, 0.09, 0.09, 0.2])
    np.testing.assert_allclose(sbm.holm(np.array([0.5, 0.6])), [1.0, 1.0])
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = rng.uniform(0, 0.2, rng.integers(1, 30))
        np.testing.assert_array_equal(sbm.holm(p), _holm_reference(p))
        np.testing.assert_allclose(sbm.holm(p), multipletests(p, method="holm")[1])


def test_holm_leaves_a_missing_p_out_of_the_family():
    out = sbm.holm(np.array([0.01, np.nan, 0.04]))
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [0.02, 0.04])
    assert np.isnan(sbm.holm(np.array([np.nan])))[0]


def test_wilcoxon_p_is_missing_when_nothing_differs():
    assert np.isnan(sbm.wilcoxon_p(np.zeros(5)))
    assert 0 < sbm.wilcoxon_p(np.array([0.1, 0.2, -0.05, 0.3, 0.2])) <= 1


def test_share_is_reported_only_on_a_positive_reference():
    assert sbm.share_defined(np.array([0.1, 0.2]))
    assert not sbm.share_defined(np.array([0.1, 0.0]))
    assert not sbm.share_defined(np.array([0.1, -0.2]))
    assert not sbm.share_defined(np.array([]))


def test_signed_prints_a_value_that_rounds_to_zero_without_a_sign():
    assert sbm._signed(-0.00004) == "0.000"
    assert sbm._signed(0.00040) == "0.000"
    assert sbm._signed(0.0) == "0.000"
    assert sbm._signed(-0.0) == "0.000"
    assert sbm._signed(0.0005001) == "+0.001"
    assert sbm._signed(-0.0123) == "-0.012"
    assert sbm._signed(0.249) == "+0.249"


def test_run_end_to_end_on_the_fixture(tmp_path):
    raw, ship, oz = make_fixture(n_land=12, ks=(2, 12), cells=((1.0, 0), (5.0, 20)))
    d = as_loaded(raw, oz)
    frame, rules, summaries = sbm.run(d, ship, processes=("gaussian",))
    assert set(frame["cell"]) == {"pooled", "1sigma_from_trial_1", "5sigma_from_trial_21"}
    np.testing.assert_allclose(frame["inc_lcb1_test"], 0.01 * frame["k"], atol=1e-12)
    # The larger k gains more here, and so it is chosen in every cell, by gain and by increment alike.
    chosen = [s for s in summaries if "k_chosen_on_7_11" in s]
    assert all(s["k_chosen_on_7_11"] == 12 for s in chosen)
    for s in chosen:
        assert s["increment_over_lcb1"]["test"] == pytest.approx(0.12, abs=1e-12)
        assert s["increment_over_top_candidate"]["test"] == pytest.approx(0.02, abs=1e-12)
        assert s["price_test"] == pytest.approx(0.12, abs=1e-12)
        assert s["lcb1_price_test"] == pytest.approx(0.02, abs=1e-12)
        assert s["ship_rules"]["lcb1"]["test"]["price"] == pytest.approx(0.02, abs=1e-12)
    in_cells = [s for s in chosen if s["cell"] != "pooled"]
    p = np.array([s["increment_over_lcb1"]["test_p"] for s in in_cells])
    np.testing.assert_allclose([s["increment_over_lcb1"]["test_p_holm_chosen_cells"] for s in in_cells],
                               multipletests(p, method="holm")[1])
    cells_only = frame[frame["cell"] != "pooled"]
    np.testing.assert_allclose(cells_only["inc_lcb1_p_test_holm_all_cells"],
                               multipletests(cells_only["inc_lcb1_p_test"], method="holm")[1])
    without = next(s for s in summaries if s["cell"] == "pooled_k12_without_5sigma_from_trial_21")
    assert without["all_seeds"]["increment_over_lcb1"]["gain"] == pytest.approx(0.12, abs=1e-12)
    prices = next(s for s in summaries if s["cell"] == "clean_twin_prices")
    assert prices["sitting"][2]["all"]["mean"] == pytest.approx(0.02, abs=1e-12)
    assert prices["ship_rules"]["lcb1"]["test"]["mean"] == pytest.approx(0.02, abs=1e-12)
    # The rule rows: LCB1 gains 0.05 raw on every run.
    lcb1 = rules[(rules["rule"] == "lcb1") & (rules["seeds"] == "test") & (rules["cell"] == "pooled")].iloc[0]
    assert lcb1["gain"] == pytest.approx(np.mean([0.05 / z for z in oz.values()]))
    assert lcb1["landscapes_gaining"] == 12
    json.dumps(summaries)  # the selection file must serialise

    table = tmp_path / "t.tex"
    sbm.write_table(summaries, table)
    body = table.read_text()
    # check_paper.py reads the spec up to the first closing brace, so it must hold no braces.
    spec = body.split(r"\begin{tabular}{", 1)[1].split("}", 1)[0]
    assert spec == "lrlrrlrrr"
    columns = sum(ch in "lcrp" for ch in spec)
    assert columns == 9
    for line in body.splitlines():
        if line.strip().endswith(r"\\"):
            assert line.count("&") == columns - 1, line
    assert r"$+0.120$ {\scriptsize $[+0.120, +0.120]$}" in body


def test_attach_gives_the_increment_over_the_other_two_rules():
    # LCB2 and PM ship 0.01 and 0.02 raw worse than LCB1 on every noisy run of the fixture.
    raw, ship, oz = make_fixture()
    m = sbm.attach_ship_rules(as_loaded(raw, oz), ship)
    np.testing.assert_allclose(m["inc_lcb2"], 0.01 / m["opt_z"] + 0.01 * m["k"], atol=1e-12)
    np.testing.assert_allclose(m["inc_pm"], 0.02 / m["opt_z"] + 0.01 * m["k"], atol=1e-12)
    np.testing.assert_allclose(m["inc_lcb2"], m["gain"] - m["gain_lcb2"], atol=1e-12)


def test_rule_increment_cell_recovers_the_known_increment_and_holm_over_k():
    raw, ship, oz = make_fixture(n_land=6)
    m = sbm.attach_ship_rules(as_loaded(raw, oz), ship)
    rows = pd.DataFrame(sbm.rule_increment_cell(m, "cell", np.random.default_rng(1), "lcb2")).set_index("k")
    lcb2_gap = np.mean([0.01 / z for z in oz.values()])
    for k in (2, 3):
        r = rows.loc[k]
        for n in ("all", "train", "test"):
            assert r[f"inc_lcb2_{n}"] == pytest.approx(lcb2_gap + 0.01 * k, abs=1e-12)
        assert r["inc_lcb2_test_landscapes_ahead"] == 6
        assert r["inc_lcb2_test_gaussian"] == pytest.approx(lcb2_gap + 0.01 * k, abs=1e-12)
        # Six positive per-landscape increments: exact two-sided p 2 / 2**6, doubled by Holm over the two k.
        assert r["inc_lcb2_p_test"] == pytest.approx(2 / 2 ** 6)
        assert r["inc_lcb2_p_test_holm_over_k"] == pytest.approx(4 / 2 ** 6)
    bad = m.copy()
    bad.loc[bad.index[0], "inc_lcb2"] += 0.5
    with pytest.raises(AssertionError, match="gain less LCB2"):
        sbm.rule_increment_cell(bad, "cell", np.random.default_rng(1), "lcb2")


def test_run_gives_the_other_rules_and_the_rule_chosen_on_7_11():
    raw, ship, oz = make_fixture(n_land=12, ks=(2, 12), cells=((1.0, 0), (5.0, 20)))
    frame, _, summaries = sbm.run(as_loaded(raw, oz), ship, processes=("gaussian",))
    gaps = {"lcb2": np.mean([0.01 / z for z in oz.values()]), "pm": np.mean([0.02 / z for z in oz.values()])}
    for rule, gap in gaps.items():
        np.testing.assert_allclose(frame[f"inc_{rule}_test"], gap + 0.01 * frame["k"], atol=1e-12)
        cells_only = frame[frame["cell"] != "pooled"]
        np.testing.assert_allclose(cells_only[f"inc_{rule}_p_test_holm_all_cells"],
                                   multipletests(cells_only[f"inc_{rule}_p_test"], method="holm")[1])
    chosen = [s for s in summaries if "k_chosen_on_7_11" in s]
    for s in chosen:
        assert s["increment_over_lcb2"]["test"] == pytest.approx(gaps["lcb2"] + 0.12, abs=1e-12)
        assert s["increment_over_pm"]["all"] == pytest.approx(gaps["pm"] + 0.12, abs=1e-12)
        assert s["per_process_increment_over_lcb2_test"]["gaussian"] == pytest.approx(gaps["lcb2"] + 0.12, abs=1e-12)
        # LCB1 gains most on seeds 7-11 (0.05 raw against 0.04 and 0.03), so it is the rule chosen there.
        assert s["zero_trial_rule_chosen_on_7_11"] == "lcb1"
        assert s["increment_over_chosen_rule"]["rule"] == "lcb1"
        assert s["increment_over_chosen_rule"]["test"] == s["increment_over_lcb1"]["test"]
    in_cells = [s for s in chosen if s["cell"] != "pooled"]
    p = np.array([s["increment_over_chosen_rule"]["test_p"] for s in in_cells])
    np.testing.assert_allclose([s["increment_over_chosen_rule"]["test_p_holm_chosen_cells"] for s in in_cells],
                               multipletests(p, method="holm")[1])
    json.dumps(summaries)


def test_the_other_rules_leave_every_earlier_output_unchanged(monkeypatch):
    raw, ship, oz = make_fixture(n_land=12, ks=(2, 12), cells=((1.0, 0), (5.0, 20)))
    d = as_loaded(raw, oz)
    frame, rules, summaries = sbm.run(d, ship, processes=("gaussian",))
    monkeypatch.setattr(sbm, "OTHER_RULES", ())
    old_frame, old_rules, old_summaries = sbm.run(d, ship, processes=("gaussian",))
    assert not [c for c in old_frame.columns if c.startswith(("inc_lcb2", "inc_pm"))]
    pd.testing.assert_frame_equal(frame[old_frame.columns], old_frame)
    pd.testing.assert_frame_equal(rules, old_rules)
    assert len(summaries) == len(old_summaries)
    for new, old in zip(summaries, old_summaries):
        # JSON text, so that a NaN compares equal to a NaN
        assert json.dumps({k: v for k, v in new.items() if k in old}, sort_keys=True) == json.dumps(old, sort_keys=True)


def _fresh_module():
    spec = importlib.util.spec_from_file_location(
        "fresh_seed_replication", ROOT / "scripts" / "review_checks" / "fresh_seed_replication.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fresh_reference_table_gives_the_known_increment_and_is_deterministic():
    fresh = _fresh_module()
    raw, ship, oz = make_fixture(n_land=5, ks=(2, 5, 12), cells=((1.0, 0), (5.0, 20)))
    rules, inc, prices = fresh.reference_table(raw, ship, oz)
    np.testing.assert_allclose(inc["inc_lcb1"], 0.01 * inc["k"], atol=1e-12)
    np.testing.assert_allclose(inc["inc_top"], 0.02, atol=1e-12)
    np.testing.assert_allclose(inc["price"], 0.01 * inc["k"], atol=1e-12)
    lcb1 = prices[prices["procedure"] == "lcb1"].iloc[0]
    assert lcb1["price"] == pytest.approx(0.02, abs=1e-12)
    in_cells = inc[inc["cell"] != "pooled"]
    np.testing.assert_allclose(in_cells["inc_lcb1_p_holm_all_cells"],
                               multipletests(in_cells["inc_lcb1_p"], method="holm")[1])
    again = fresh.reference_table(raw, ship, oz)
    pd.testing.assert_frame_equal(inc, again[1])
    pd.testing.assert_frame_equal(rules, again[0])
    # Rows of another error process are left out: the replication arm's scope is gaussian error.
    other, other_ship, _ = make_fixture(n_land=5, ks=(2, 5, 12), cells=((1.0, 0), (5.0, 20)), inc_per_k=0.5,
                                        error_model="drift")
    both = pd.concat([raw, other], ignore_index=True)
    both_ship = pd.concat([ship, other_ship[~other_ship["baseline"]]], ignore_index=True)
    mixed = fresh.reference_table(both, both_ship, oz)
    pd.testing.assert_frame_equal(inc, mixed[1])


def test_fresh_rule_increments_give_the_known_increment_and_leave_the_reference_table_alone():
    fresh = _fresh_module()
    raw, ship, oz = make_fixture(n_land=5, ks=(2, 5, 12), cells=((1.0, 0), (5.0, 20)))
    before = fresh.reference_table(raw, ship, oz)
    out = fresh.rule_increments(raw, ship, oz)
    gaps = {"lcb2": np.mean([0.01 / z for z in oz.values()]), "pm": np.mean([0.02 / z for z in oz.values()])}
    for rule, gap in gaps.items():
        np.testing.assert_allclose(out[f"inc_{rule}"], gap + 0.01 * out["k"], atol=1e-12)
        in_cells = out[out["cell"] != "pooled"]
        assert len(in_cells) == 6
        np.testing.assert_allclose(in_cells[f"inc_{rule}_p_holm_all_cells"],
                                   multipletests(in_cells[f"inc_{rule}_p"], method="holm")[1])
        one = out[out["cell"] == "1sigma_from_trial_1"]
        np.testing.assert_allclose(one[f"inc_{rule}_p_holm_over_k"], multipletests(one[f"inc_{rule}_p"], method="holm")[1])
    pd.testing.assert_frame_equal(out, fresh.rule_increments(raw, ship, oz))
    after = fresh.reference_table(raw, ship, oz)
    for a, b in zip(before, after):
        pd.testing.assert_frame_equal(a, b)
    assert fresh.format_rule_increments(out)[0] == "cell pooled"
