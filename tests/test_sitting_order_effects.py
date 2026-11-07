"""Order effects between the designs of a final sitting (scripts/checks/sitting_order_effects.py).

What must hold: (1) each effect is the formula the design fixed (anchoring, contrast, fatigue), on small
arrays worked by hand; (2) 'none' is the shared sitting model, the published look winner argmax(truth + e),
under every presentation order, and on a toy run it is replay_end_of_study.tournament's own winner;
(3) the presentation order matters only for an effect other than 'none'; (4) a clean twin's exact rater
leaves fatigue inert but can still anchor or contrast; (5) the control stops on any difference from the
published sitting.
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
import sitting_order_effects as soe  # noqa: E402

TRUTH = np.array([3.0, 1.0, 2.0])
ERROR = np.array([0.5, -0.2, 0.1])
RANK = np.arange(3)


# --- the formulas -------------------------------------------------------------------------------


def test_anchoring_pulls_each_look_toward_the_previous_rating():
    y = soe.presented_looks(TRUTH, ERROR, RANK, "anchoring", 0.5)
    # y0 = 3.5; y1 = 0.5 * 0.8 + 0.5 * 3.5; y2 = 0.5 * 2.1 + 0.5 * y1
    assert np.allclose(y, [3.5, 2.15, 2.125])
    shown = np.array([2, 0, 1])  # t = [2, 3, 1], e = [0.1, 0.5, -0.2]
    y = soe.presented_looks(TRUTH, ERROR, shown, "anchoring", 0.25)
    y0 = 2.1
    y1 = 0.75 * 3.5 + 0.25 * y0
    y2 = 0.75 * 0.8 + 0.25 * y1
    assert np.allclose(y, [y0, y1, y2])


def test_contrast_exaggerates_the_difference_to_the_previous_design():
    y = soe.presented_looks(TRUTH, ERROR, RANK, "contrast", 0.25)
    assert np.allclose(y, [3.5, 0.8 + 0.25 * (1.0 - 3.0), 2.1 + 0.25 * (2.0 - 1.0)])
    shown = np.array([1, 2, 0])  # t = [1, 2, 3]
    y = soe.presented_looks(TRUTH, ERROR, shown, "contrast", 0.5)
    assert np.allclose(y, [0.8, 2.1 + 0.5 * 1.0, 3.5 + 0.5 * 1.0])


def test_fatigue_doubles_the_look_noise_by_the_last_look():
    y = soe.presented_looks(TRUTH, ERROR, RANK, "fatigue", 1.0)
    assert np.allclose(y, [3.0 + 0.5 * 1.0, 1.0 - 0.2 * 1.5, 2.0 + 0.1 * 2.0])
    two = soe.presented_looks(TRUTH[:2], ERROR[:2], np.arange(2), "fatigue", 1.0)
    assert np.allclose(two, [3.5, 1.0 - 0.4])
    one = soe.presented_looks(TRUTH[:1], ERROR[:1], np.arange(1), "fatigue", 1.0)
    assert np.allclose(one, [3.5])


def test_none_is_truth_plus_error_and_unknown_names_fail():
    assert np.array_equal(soe.presented_looks(TRUTH, ERROR, RANK, "none", 0.0), TRUTH + ERROR)
    with pytest.raises(ValueError, match="effect"):
        soe.presented_looks(TRUTH, ERROR, RANK, "halo", 0.5)
    with pytest.raises(ValueError, match="order"):
        soe.presentation("alphabetical", RANK)


def test_shipped_maps_the_winning_position_back_to_the_candidate():
    shown = np.array([2, 0, 1])
    # 'none': the best look is candidate 0 (3.5), shown at position 1
    assert soe.shipped(TRUTH, ERROR, shown, "none", 0.0) == 0
    assert soe.presentation("random", shown).tolist() == [2, 0, 1]
    assert soe.presentation("rank", shown).tolist() == [0, 1, 2]


def test_the_combinations_score_none_once_and_every_effect_in_both_orders():
    assert len(soe.COMBOS) == 11
    assert [c for c in soe.COMBOS if c[0] == "none"] == [("none", 0.0, "rank")]
    assert soe.label("none", 0.0, "random") == "none"
    assert soe.label("anchoring", 0.25, "random") == "anchoring0.25_random"
    assert soe.label("fatigue", 1.0, "rank") == "fatigue1_rank"
    assert soe.cell_of(1.0, 0) == "1sigma_from_trial_1" and soe.cell_of(5.0, 20) == "5sigma_from_trial_21"
    assert soe.cell_of(1.0, 20) is None and soe.cell_of(5.0, 0) is None


# --- 'none' is the published sitting; order matters only with an effect ---------------------------


def test_none_is_the_shared_model_under_every_order():
    rng = np.random.default_rng(1)
    for _ in range(500):
        m = int(rng.integers(1, 17))
        truth = rng.normal(size=m)
        sds = np.full(m, float(rng.uniform(0.1, 3.0)))
        z = rng.normal(size=m)
        error = 1.0 * sds * z
        published = int(np.argmax(truth + 1.0 * sds[:m] * z[:m]))
        assert soe.published_pick(truth, error) == published
        for shown in (np.arange(m), rng.permutation(m), np.arange(m)[::-1]):
            assert soe.shipped(truth, error, shown, "none", 0.0) == published


def test_order_matters_only_for_an_effect_other_than_none():
    rng = np.random.default_rng(2)
    moved = {(e, lv): 0 for e, lv in soe.EFFECTS}
    for _ in range(400):
        m = 5
        truth = rng.normal(size=m)
        error = rng.normal(size=m)
        perm = rng.permutation(m)
        for e, lv in soe.EFFECTS:
            a = soe.shipped(truth, error, soe.presentation("rank", perm), e, lv)
            b = soe.shipped(truth, error, soe.presentation("random", perm), e, lv)
            moved[(e, lv)] += int(a != b)
    assert moved[("none", 0.0)] == 0
    assert all(n > 0 for (e, _), n in moved.items() if e != "none")


def test_an_exact_rater_cannot_tire_but_can_anchor_and_contrast():
    zero = np.zeros(3)
    rng = np.random.default_rng(3)
    for _ in range(200):
        truth = rng.normal(size=6)
        shown = rng.permutation(6)
        assert soe.shipped(truth, np.zeros(6), shown, "fatigue", 1.0) == int(np.argmax(truth))
    # anchoring: the best design shown after a poor one is pulled down below a later good one
    truth = np.array([0.5, 1.0, 0.95])
    assert soe.shipped(truth, zero, np.arange(3), "none", 0.0) == 1
    assert soe.shipped(truth, zero, np.arange(3), "anchoring", 0.5) == 2
    # contrast: a good design shown after a poor one is lifted above the best
    truth = np.array([1.0, 0.0, 0.9])
    assert soe.shipped(truth, zero, np.arange(3), "contrast", 0.25) == 2


# --- the control ------------------------------------------------------------------------------------


def _control_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rep = pd.DataFrame({"file": ["a.csv", "b.csv"], "k": [2, 16], "regret_noisy": [0.1, 0.2],
                        "regret_clean": [0.0, 0.05], "ref_noisy": [0.3, 0.4], "ref_clean": [0.0, 0.05]})
    return rep, rep.copy()


def test_the_control_passes_on_identical_regrets_and_stops_on_any_difference():
    rep, pub = _control_frames()
    out = soe.check_control(rep, pub)
    assert out["n_runs"] == 2 and max(out["max_abs_difference"].values()) == 0.0
    off = pub.copy()
    off.loc[1, "regret_noisy"] = np.nextafter(off.loc[1, "regret_noisy"], 1.0)
    with pytest.raises(soe.ControlError, match="regret_noisy"):
        soe.check_control(rep, off)
    off = pub.copy()
    off.loc[0, "ref_clean"] = 1e-12
    with pytest.raises(soe.ControlError, match="ref_clean"):
        soe.check_control(rep, off)
    with pytest.raises(soe.ControlError, match="no published row"):
        soe.check_control(rep, pub.iloc[:1])


def test_vectors_survive_the_csv_bit_for_bit():
    rng = np.random.default_rng(4)
    v = rng.normal(size=16) * 1e3 / 7.0
    assert np.array_equal(soe.parse_vec(soe._vec(v)), v)
    p = rng.permutation(16)
    assert np.array_equal(soe.parse_ivec(soe._ivec(p)), p)


# --- the estimand ------------------------------------------------------------------------------------


def test_scores_are_the_adaptations_estimand_in_opt_z_units():
    """Raw regrets proportional to opt_z, so every landscape gives the same known value in opt_z units."""
    opt_z = {f"land{i:02d}": 2.0 + i for i in range(20)}
    rows = []
    for ds, z in opt_z.items():
        for acq in ("logei", "qnei"):
            for j, proc in enumerate(("gaussian", "bias", "drift", "ar1")):
                rows.append({"dataset": ds, "acquisition": acq, "error_model": proc, "jitter_std": 1.0,
                             "jitter_iteration": 0, "seed": 12, "file": f"{ds}_{acq}_{proc}.csv",
                             "cell": "1sigma_from_trial_1", "k": 2,
                             "ref_regret": 0.5 * z, "ref_regret_clean": 0.1 * z,
                             "reg_x": 0.3 * z, "reg_x_clean": 0.15 * z, "reg_none": 0.35 * z,
                             "reg_none_clean": 0.12 * z, "regret_lcb1": 0.4 * z, "regret_lcb1_clean": 0.13 * z,
                             "cand_none": 0, "cand_x": int(j == 0), "cand_none_clean": 0, "cand_x_clean": 1})
    w = pd.DataFrame(rows)
    pf = soe.scaled_block(w, "x", opt_z)
    for col, value in (("cost", 0.4), ("gain", 0.2), ("price", 0.05), ("inc_published", 0.05), ("inc_lcb1", 0.1),
                       ("gain_lcb1", 0.1), ("price_lcb1", 0.03), ("price_over_published", 0.03)):
        assert np.allclose(pf[col], value), col
    out = soe.summarise_block(pf, np.random.default_rng(0))
    assert out["n_runs"] == 160 and out["n_landscapes"] == 20
    assert np.isclose(out["gain"], 0.2) and np.isclose(out["share"], 0.5) and not out["share_suppressed"]
    assert np.isclose(out["inc_published"], 0.05) and np.isclose(out["inc_lcb1"], 0.1)
    assert out["inc_published_landscapes_positive"] == 20 and out["inc_published_p"] < 0.001
    assert np.isclose(out["changed_share_noisy"], 0.25) and np.isclose(out["changed_share_clean"], 1.0)
    # A reference cost below 0.01 of opt_z withholds the share and keeps the gain.
    tiny = w.assign(ref_regret_clean=w["ref_regret"] - 0.001 * w["dataset"].map(opt_z))
    out = soe.summarise_block(soe.scaled_block(tiny, "x", opt_z), np.random.default_rng(0))
    assert np.isnan(out["share"]) and "below" in out["share_suppressed"] and np.isclose(out["gain"], 0.2)
    # The published sitting scored against itself: no increment, nothing moved.
    none = soe.summarise_block(soe.scaled_block(w, "none", opt_z), np.random.default_rng(0))
    assert np.isclose(none["gain"], 0.15) and none["inc_published"] == 0.0 and none["changed_share_noisy"] == 0.0


# --- against replay_end_of_study.tournament itself (imports the simulator) ----------------------------


def _toy(name: str, n: int = 40):
    import torch

    import replay_end_of_study as eos

    truth = np.linspace(1.0, 0.0, n)
    run = eos.RunLog(name=name, X=np.arange(n, dtype=float).reshape(-1, 1), observed=truth.copy(),
                     deployed=truth.copy(), logged_inference=np.zeros(n), y_opt=2.0, error=np.zeros(n))
    state = eos.SearchState(n=n, first=np.arange(n), obs_mean=truth.copy(), lcb=truth.copy(), gp=None,
                            train_X=torch.zeros(n, 1), train_Y=torch.zeros(n, 1),
                            bounds=torch.tensor([[0.0], [float(n)]]), gp_failed=True)
    return run, state


def test_sitting_inputs_reproduce_tournament_and_none_ships_its_winner():
    import replay_end_of_study as eos

    T = 50
    arm = eos.ArmInfo("a", Path("a"), T, 5, 0.8, input_arm=False)
    shared = eos.Settings()
    for model, k in (("gaussian", 2), ("ar1", 16), ("drift", 16), ("bias", 2)):
        for i in range(6):
            run, state = _toy(f"toy_{model}_{k}_{i}.csv")
            sd_fn = eos.sitting_sd_fn(arm, model, 1.0, 0, shared)
            rec = soe.sitting_inputs(run, state, k, sd_fn, T)
            pub = eos.tournament(run, state, k, (1.0,), sd_fn, T)[soe.procedure_name(k)]
            truth, error = soe.parse_vec(rec["truth"]), soe.parse_vec(rec["error"])
            perm = soe.parse_ivec(rec["perm_random"])
            assert rec["published_regret"] == pub["regret"] and rec["n_candidates"] == len(truth) == k
            assert sorted(perm.tolist()) == list(range(k))
            expect = np.random.default_rng(eos.noise_seed(run.name, k, soe.ORDER_TAG)).permutation(k)
            assert np.array_equal(perm, expect)
            for order in soe.ORDERS:
                c = soe.shipped(truth, error, soe.presentation(order, perm), "none", 0.0)
                assert run.y_opt - truth[c] == pub["regret"]
    # The sequential process is not the published sitting and is refused.
    run, state = _toy("toy_seq.csv")
    seq = eos.sitting_sd_fn(arm, "drift", 1.0, 0, eos.Settings(sitting_process="sequential"))
    with pytest.raises(ValueError, match="shared"):
        soe.sitting_inputs(run, state, 16, seq, T)
