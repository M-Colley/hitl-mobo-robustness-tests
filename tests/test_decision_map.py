"""The decision map and the freeze variant F1 (scripts/checks/decision_map.py).

A synthetic per-run frame whose answers are known: in each cell one procedure gains most, the clean twins
charge known prices, and rule A's alarms fall at chosen trials. What must hold: F1 and the self-gating
combinations act in the order the study would meet their decisions; the map chooses per cell on the
training seeds and pays no price without error; the best single procedure pays its price; the stopping
join refuses a replay that does not reproduce the standard process or LCB1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import decision_map as dm  # noqa: E402

TAUS = {"G16": 1.0, "G2": 0.6}


def frame_fixture(n_land: int = 6, seeds=(7, 8, 12, 13)) -> pd.DataFrame:
    """One run per (landscape, acquisition, seed, cell). Regret in opt_z units; opt_z = 1.

    Gains over the standard process (ref_noisy 1.0, ref_clean 0.5): sit2 gains 0.03 at 1 sigma, sit8
    0.06 at 5 sigma from trial 1; every sitting and rule loses 0.01 elsewhere. Rule A alarms at trial 30
    in the LogEI runs at 5 sigma from trial 21 and freezes onto a design 0.3 better; without an alarm it
    ships LCB1. So in that cell F1 gains (0.3 + 0) / 2 and the published freeze rule (0.3 - 0.01) / 2.
    Prices: sit2 0.02, sit8 0.015, LCB1 0.021, LCB2 0.023, PM 0.027, freeze 0.021 (LCB1 at T, no alarm on
    a clean twin), F1 0 (no false alarm).
    """
    rows = []
    for i in range(n_land):
        for acq in ("logei", "qnei"):
            for seed in seeds:
                for sigma, onset in dm.CELLS:
                    ref_n, ref_c = 1.0, 0.5
                    late5, s1, s5e = (sigma == 5.0 and onset == 20), sigma == 1.0, (sigma == 5.0 and onset == 0)
                    r = {"file": f"l{i}_{acq}_{seed}_{sigma}_{onset}.csv", "dataset": f"land{i:02d}",
                         "acquisition": acq, "seed": seed, "error_model": "gaussian", "sigma": sigma, "onset": onset,
                         "opt_z": 1.0, "ref_noisy": ref_n, "ref_clean": ref_c,
                         "regret_pm": ref_n + 0.01, "regret_pm_clean": ref_c + 0.027,
                         "regret_lcb1": ref_n + 0.01, "regret_lcb1_clean": ref_c + 0.021,
                         "regret_lcb2": ref_n + 0.01, "regret_lcb2_clean": ref_c + 0.023,
                         "noise_sd_34": 3.0 if sigma == 5.0 else 0.2, "noise_sd_34_clean": 0.2,
                         "noise_sd_48": 3.0 if sigma >= 1.0 else 0.2, "noise_sd_48_clean": 0.2}
                    for k in dm.K_GRID:
                        gain = (0.03 if (k == 2 and s1) else 0.06 if (k == 8 and s5e) else -0.01)
                        r[f"sit{k}"] = ref_n - gain
                        r[f"sit{k}_clean"] = ref_c + {2: 0.02, 8: 0.015}.get(k, 0.05)
                    alarm = late5 and acq == "logei"
                    r.update({"freeze": ref_n - 0.3 if alarm else r["regret_lcb1"],
                              "freeze_clean": r["regret_lcb1_clean"], "alarm": alarm, "alarm_clean": False,
                              "alarm_t": 30.0 if alarm else np.inf, "alarm_t_clean": np.inf})
                    rows.append(r)
    return pd.DataFrame(rows)


def test_f1_freezes_only_on_an_alarm_and_otherwise_is_the_standard_process():
    f = frame_fixture(n_land=1, seeds=(7,))
    P = dm.procedures(f, TAUS)
    np.testing.assert_array_equal(P["F1"][0], np.where(f["alarm"], f["freeze"], f["ref_noisy"]))
    np.testing.assert_array_equal(P["F1"][1], f["ref_clean"])           # no false alarm: no price
    np.testing.assert_array_equal(P["freeze"][1], f["freeze_clean"])    # the published rule ships LCB1 at T


def test_the_combinations_act_in_the_order_the_study_meets_their_decisions():
    base = frame_fixture(n_land=1, seeds=(7,)).iloc[[0] * 6].reset_index(drop=True)
    base["freeze"], base["sit16"], base["sit2"], base["ref_noisy"] = 1.0, 2.0, 3.0, 4.0
    #                 alarm, alarm_t, noise at 34, noise at 48   ->   F1+G2, F1+G16+G2
    cases = [(True, 30.0, 5.0, 5.0, "freeze", "freeze"),     # an alarm before the first gate freezes
             (True, 40.0, 5.0, 5.0, "freeze", "sit16"),      # G16 acts at 34, before the alarm at 40
             (True, 49.0, 0.1, 5.0, "sit2", "sit2"),         # G2 acts at 48, before the alarm at 49
             (True, 49.0, 0.1, 0.1, "freeze", "freeze"),     # no gate fires: the late alarm still freezes
             (False, np.inf, 0.1, 0.1, "ref_noisy", "ref_noisy"),
             (False, np.inf, 5.0, 0.1, "ref_noisy", "sit16")]
    for j, (alarm, t, n34, n48, _, _) in enumerate(cases):
        base.loc[j, ["alarm", "alarm_t", "noise_sd_34", "noise_sd_48"]] = [alarm, t, n34, n48]
    base["alarm"] = base["alarm"].astype(bool)
    P = dm.procedures(base, TAUS, combinations=True)
    for j, (*_, f1g2, f1g16g2) in enumerate(cases):
        assert P["F1+G2"][0][j] == base.loc[j, f1g2], j
        assert P["F1+G16+G2"][0][j] == base.loc[j, f1g16g2], j


def test_the_map_chooses_per_cell_on_the_training_seeds_and_pays_no_price_without_error():
    f = frame_fixture()
    cands = list(dm.PUBLISHED) + ["F1"]
    P = {n: dm.procedures(f, TAUS)[n] for n in cands}
    train, test = dm.Scores(f, P, (7, 8)), dm.Scores(f, P, (12, 13))
    m1 = dm.choose_map(train, cands)
    assert m1[(1.0, 0)] == m1[(1.0, 20)] == "sit2"
    assert m1[(5.0, 0)] == "sit8" and m1[(5.0, 20)] == "F1"
    # every other procedure loses in the small cells, where the standard rule (gain 0) is kept
    assert all(m1[(s, o)] == "standard" for s in (0.05, 0.25) for o in (0, 20))
    v = test.map_value_l(m1)
    assert np.allclose(v, (2 * 0.03 + 0.06 + 0.15) / 9)
    # the best single procedure pays its price in the error-free ninth
    single = dm.choose_single(train, cands)
    assert single == "F1"
    assert np.allclose(test.value_l("F1"), 0.15 / 9)
    assert np.allclose(test.value_l("freeze"), (0.145 - 7 * 0.01 - 0.021) / 9)
    assert np.allclose(test.value_l("sit2"), (2 * 0.03 - 6 * 0.01 - 0.02) / 9)
    # onset unknown: one choice per magnitude, by the mean over both onsets
    m2 = dm.choose_map_onset_unknown(train, cands)
    assert m2[(5.0, 0)] == m2[(5.0, 20)] == "F1"      # (0 + 0.15) / 2 beats freeze's and sit8's


def test_a_tie_keeps_the_earlier_candidate_and_the_standard_rule_comes_first():
    f = frame_fixture(n_land=2, seeds=(7,))
    f["sit2"] = f["ref_noisy"]                         # sit2 now gains exactly 0 everywhere, like the standard rule
    P = {n: dm.procedures(f, TAUS)[n] for n in ("standard", "sit2")}
    m1 = dm.choose_map(dm.Scores(f, P, (7,)), ["standard", "sit2"])
    assert set(m1.values()) == {"standard"}


def test_the_map_table_tests_only_the_cells_where_the_map_acts():
    f = frame_fixture()
    cands = list(dm.PUBLISHED) + ["F1"]
    P = {n: dm.procedures(f, TAUS)[n] for n in cands}
    train, test = dm.Scores(f, P, (7, 8)), dm.Scores(f, P, (12, 13))
    t = dm.map_table(train, test, dm.choose_map(train, cands), np.random.default_rng(0)).set_index("cell")
    acting = t[t["procedure"] != "standard"].drop(index="no error", errors="ignore")
    assert len(acting) == 4 and acting["p_holm_map_cells"].notna().all()
    assert t.loc["5sigma_from_trial_21", "procedure"] == "F1"
    assert t.loc["5sigma_from_trial_21", "gain"] == pytest.approx(0.15)
    assert t.loc["5sigma_from_trial_21", "share"] == pytest.approx(0.15 / 0.5)
    assert t.loc["5sigma_from_trial_21", "price"] == pytest.approx(0.0)
    assert t.loc["1sigma_from_trial_1", "price"] == pytest.approx(0.02)
    assert t.loc["no error", "procedure"] == "standard" and t.loc["no error", "price"] == 0.0


def test_attach_stopping_checks_the_standard_process_and_lcb1_without_an_alarm():
    f = frame_fixture(n_land=1, seeds=(7,)).drop(columns=["freeze", "freeze_clean", "alarm", "alarm_clean",
                                                          "alarm_t", "alarm_t_clean"])
    stop = []
    for _, r in f.iterrows():
        alarm = r["sigma"] == 5.0
        stop.append({"dataset": r["dataset"], "acquisition": r["acquisition"], "seed": r["seed"],
                     "error_model": "gaussian", "file": r["file"], "rule": "A", "stopped": alarm,
                     "stop_t": 30.0 if alarm else np.nan, "regret": 0.7 if alarm else r["regret_lcb1"],
                     "regret_standard_T": r["ref_noisy"]})
    for (d, a, s), r in f.groupby(["dataset", "acquisition", "seed"]).first().iterrows():
        stop.append({"dataset": d, "acquisition": a, "seed": s, "error_model": "none", "file": f"{d}_{a}_{s}_clean",
                     "rule": "A", "stopped": False, "stop_t": np.nan, "regret": r["regret_lcb1_clean"],
                     "regret_standard_T": r["ref_clean"]})
    stop = pd.DataFrame(stop)
    out = dm.attach_stopping(f, stop)
    assert out.loc[out["sigma"] == 5.0, "alarm"].all() and not out.loc[out["sigma"] < 5.0, "alarm"].any()
    np.testing.assert_allclose(out.loc[out["sigma"] == 5.0, "freeze"], 0.7)
    assert (out.loc[out["sigma"] == 5.0, "alarm_t"] == 30.0).all() and np.isinf(out["alarm_t_clean"]).all()
    bad = stop.copy()
    bad.loc[0, "regret_standard_T"] += 1e-6
    with pytest.raises(ValueError, match="standard regret"):
        dm.attach_stopping(f, bad)
    bad = stop.copy()
    bad.loc[~bad["stopped"].astype(bool) & (bad["error_model"] != "none"), "regret"] += 1e-3
    with pytest.raises(ValueError, match="LCB1"):
        dm.attach_stopping(f, bad)


def _known_file(f: pd.DataFrame, path: Path, drop_last: bool = False) -> None:
    """known_variance_ship_per_run.csv.gz rows for the fixture's runs (opt_z = 1, so raw = opt_z units)."""
    rows = [{"arm": "output-boba", "file": f"{r.dataset}/{r.file}", "dataset": r.dataset, "acquisition": r.acquisition,
             "seed": r.seed, "baseline": False, "regret_best_observed": r.ref_noisy,
             "regret_mag_onset_lcb1": r.ref_noisy - 0.04, "regret_mag_onset_lcb2": r.ref_noisy - 0.05}
            for r in f.itertuples()]
    for (d, a, s), g in f.groupby(["dataset", "acquisition", "seed"]):
        rows.append({"arm": "output-boba", "file": f"{d}/{d}_{a}_{s}_baseline_exact.csv", "dataset": d,
                     "acquisition": a, "seed": s, "baseline": True, "regret_best_observed": g["ref_clean"].iloc[0],
                     "regret_mag_onset_lcb1": g["ref_clean"].iloc[0], "regret_mag_onset_lcb2": g["ref_clean"].iloc[0]})
    pd.DataFrame(rows[:-2] if drop_last else rows).to_csv(path / dm.KNOWN_FILE, index=False)


def test_the_known_variance_rules_join_only_a_complete_file_that_reproduces_the_standard_process(tmp_path,
                                                                                                monkeypatch):
    monkeypatch.setattr(dm, "REVIEW", tmp_path)
    f = frame_fixture(n_land=2, seeds=(7,))
    out, name = dm.attach_known_variance(f)
    assert name is None and out is f                                  # no file: left out
    _known_file(f, tmp_path)
    out, name = dm.attach_known_variance(f)
    assert name == dm.KNOWN_FILE
    np.testing.assert_allclose(out["KV-LCB1"], f["ref_noisy"] - 0.04)
    np.testing.assert_allclose(out["KV-LCB2_clean"], f["ref_clean"])
    P = dm.procedures(out, TAUS, known=True)
    assert set(dm.KNOWN_COLUMNS) <= set(P)
    assert dm.known_names(out) == ["KV-LCB1", "KV-LCB2"]              # levels without columns stay out
    _known_file(f, tmp_path, drop_last=True)                          # a partial file is left out
    assert dm.attach_known_variance(f)[1] is None
    _known_file(f.assign(ref_noisy=f["ref_noisy"] + 1e-3), tmp_path)  # a file that does not reproduce stops
    with pytest.raises(ValueError, match="standard process"):
        dm.attach_known_variance(f)


def test_each_policy_uses_only_the_variance_it_knows():
    """M1 knows the cell, M2 only the magnitude, a single procedure nothing; the first three candidate sets
    never contain a known-variance rule, so the appendix table does not depend on the known-variance file."""
    f = frame_fixture()
    # KV (the cell known) gains most everywhere, KVm (the magnitude) next, KVr (three pairs) least
    for name, gain in (("KV-LCB2", 0.20), ("KVm-LCB2", 0.10), ("KVr-LCB2", 0.05)):
        f[name] = f["ref_noisy"] - gain
        f[f"{name}_clean"] = f["ref_clean"]
    assert dm.known_names(f) == ["KV-LCB2", "KVm-LCB2", "KVr-LCB2"]
    res = dm.analyse(f, TAUS, known="file", rng=np.random.default_rng(0))
    assert set(res) == {"published", "main", "main+combinations", "main+variance"}
    for label in ("published", "main", "main+combinations"):
        assert not any(n.startswith("KV") for n in res[label]["train"].P)
    r = res["main+variance"]
    assert set(r["m1"].values()) == {"KV-LCB2"}
    assert set(r["m2"].values()) == {"KVm-LCB2"}
    assert r["single"] == "KVr-LCB2"
    pol = r["policies"].set_index("policy")
    assert not pol.loc["always KV-LCB2", "single_procedure"] and pol.loc["always KV-LCB2", "needs"] == "cell"
    assert not pol.loc["always KVm-LCB2", "single_procedure"]
    assert pol.loc["always KVr-LCB2", "single_procedure"]
