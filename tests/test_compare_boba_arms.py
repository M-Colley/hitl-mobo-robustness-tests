"""compare_boba_arms: a share of the reference's cost is reported only when its
denominator is a number.

AGENTS.md: never report a recovery ratio whose reference cost is near zero or
changes sign across cells. The known-noise table printed -105% on a reference of
0.001, and the incumbent text quoted an 81% share of a 0.003 reference that is
negative on three landscapes. share_of_reference suppresses the share (the
absolute delta is the result) when the pooled reference is below
NEAR_ZERO_REFERENCE or negative on any landscape; a landscape whose reference is
exactly zero (solved before a mid-run onset) is counted but does not suppress.
These tests pin that rule, on its own and through main() on a synthetic tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import compare_boba_arms as cba

METRIC = cba.METRIC


def _per(ref, trt) -> pd.DataFrame:
    return pd.DataFrame({"ref": ref, "trt": trt})


def test_a_positive_reference_above_the_threshold_is_reported_as_a_ratio_of_means():
    s = cba.share_of_reference(_per([0.1, 0.2, 0.3], [0.05, 0.1, 0.3]))
    assert s["share_suppressed"] == ""
    assert np.isclose(s["share_removed"], 1.0 - 0.45 / 0.6)
    assert (s["n_reference_negative"], s["n_reference_zero"]) == (0, 0)


def test_a_near_zero_reference_suppresses_the_share():
    s = cba.share_of_reference(_per([0.004, 0.006, 0.009], [0.001, 0.001, 0.001]))
    assert np.isnan(s["share_removed"])
    assert "below 0.01" in s["share_suppressed"]


def test_the_threshold_is_inclusive_at_one_point_of_the_achievable_improvement():
    at = cba.share_of_reference(_per([0.01, 0.01], [0.005, 0.005]))
    below = cba.share_of_reference(_per([0.0099, 0.0099], [0.005, 0.005]))
    assert cba.NEAR_ZERO_REFERENCE == 0.01
    assert np.isclose(at["share_removed"], 0.5) and np.isnan(below["share_removed"])


def test_a_reference_negative_on_one_landscape_suppresses_the_share_however_large_the_mean():
    s = cba.share_of_reference(_per([0.5, 0.4, -0.001], [0.1, 0.1, 0.1]))
    assert np.isnan(s["share_removed"])
    assert s["share_suppressed"] == "reference negative on 1 of 3 landscapes"
    assert s["n_reference_negative"] == 1


def test_an_exactly_zero_reference_is_counted_but_does_not_suppress():
    s = cba.share_of_reference(_per([0.2, 0.0, 0.1], [0.1, 0.0, 0.05]))
    assert s["share_suppressed"] == "" and s["n_reference_zero"] == 1
    assert np.isclose(s["share_removed"], 0.5)


def test_both_reasons_are_given_when_both_apply():
    s = cba.share_of_reference(_per([0.004, -0.001], [0.0, 0.0]))
    assert "below" in s["share_suppressed"] and "negative on 1 of 2" in s["share_suppressed"]


# ------------------------------------------------------------------ through main()
DATASETS = ("d1", "d2", "d3", "d4")
OPT_Z = {"d1": 1.0, "d2": 2.0, "d3": 5.0, "d4": 10.0}
# (magnitude, onset) -> reference excess per dataset in opt_z units; the
# treatment removes 40% of it everywhere.
CELLS = {
    (1.0, 0): [0.10, 0.20, 0.30, 0.40],       # reported
    (0.05, 0): [0.004, 0.005, 0.006, 0.007],  # pooled 0.0055 < 0.01: suppressed
    (0.25, 20): [0.20, 0.30, 0.40, -0.01],    # negative on one landscape: suppressed
    (1.0, 20): [0.10, 0.0, 0.20, 0.30],       # an exact zero: reported
}


def _write_arm(root: Path, treatment: bool) -> None:
    for d in DATASETS:
        rows = []
        for (std, onset), refs in CELLS.items():
            ref = refs[DATASETS.index(d)]
            value = (0.6 * ref if treatment else ref) * OPT_Z[d]
            for acq in ("logei", "ucb", "random"):
                for seed in (7, 8):
                    rows.append({"dataset": d, "acquisition": acq, "error_model": "gaussian",
                                 "jitter_std": std, "jitter_iteration": onset, "seed": seed,
                                 METRIC: 0.0 if acq == "random" else value})
        (root / d / "evaluation").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(root / d / "evaluation" / "paired_excess_metrics.csv", index=False)


@pytest.fixture()
def arms(tmp_path, monkeypatch):
    ref, trt = tmp_path / "ref", tmp_path / "trt"
    _write_arm(ref, treatment=False)
    _write_arm(trt, treatment=True)
    monkeypatch.setattr(cba.bb, "load_stats", lambda *a, **k: {d: {"opt_z": z} for d, z in OPT_Z.items()})
    return ref, trt, tmp_path / "out"


def _run(ref, trt, out, *extra):
    cba.main(["--reference", str(ref), "--treatment", str(trt), "--output-dir", str(out), *extra])


def test_main_suppresses_exactly_the_near_zero_and_sign_changing_cells(arms):
    ref, trt, out = arms
    _run(ref, trt, out)
    s = pd.read_csv(out / "arm_contrast_summary.csv").set_index(["jitter_std", "jitter_iteration"])
    assert np.isclose(s.loc[(1.0, 0), "share_removed"], 0.4)
    assert np.isclose(s.loc[(1.0, 20), "share_removed"], 0.4)
    assert s.loc[(1.0, 20), "n_reference_zero"] == 1
    assert np.isnan(s.loc[(0.05, 0), "share_removed"])
    assert np.isnan(s.loc[(0.25, 20), "share_removed"])
    assert s.loc[(0.25, 20), "n_reference_negative"] == 1
    # The absolute delta is always there, suppressed share or not.
    for key, refs in CELLS.items():
        assert np.isclose(s.loc[key, "mean_delta"], -0.4 * np.mean(refs))
        assert np.isclose(s.loc[key, "mean_reference"], np.mean(refs))
    report = (out / "arm_contrast_report.txt").read_text(encoding="utf-8")
    assert "--" in report and "nan" not in report.lower()


def test_main_divides_by_opt_z_before_any_mean(arms):
    ref, trt, out = arms
    _run(ref, trt, out)
    s = pd.read_csv(out / "arm_contrast_summary.csv").set_index(["jitter_std", "jitter_iteration"])
    # Raw values are ref * opt_z; the pooled reference must be the mean of the
    # normalised ones, 0.25, not of the raw ones.
    assert np.isclose(s.loc[(1.0, 0), "mean_reference"], 0.25)


def test_main_writes_the_per_acquisition_shares_under_the_same_rule(arms):
    ref, trt, out = arms
    _run(ref, trt, out)
    acq = pd.read_csv(out / "arm_contrast_by_acquisition.csv").set_index("acquisition")
    assert set(acq.index) == {"logei", "ucb"}          # the model-free floor is dropped
    # Pooled over the four cells, d4's reference is (0.40 + 0.007 - 0.01 + 0.30) / 4 > 0 and every
    # other landscape is positive, so the pooled share is reported.
    assert np.allclose(acq["share_removed"], 0.4)
    assert (acq["n_reference_negative"] == 0).all()


def test_a_missing_opt_z_beside_landscapes_that_have_one_raises(arms, monkeypatch):
    ref, trt, out = arms
    monkeypatch.setattr(cba.bb, "load_stats", lambda *a, **k: {"d1": {"opt_z": 1.0}})
    with pytest.raises(ValueError, match="no opt_z"):
        _run(ref, trt, out)


def test_the_rule_is_defined_once_with_the_process_change_estimand():
    import analyse_boba_adaptations as aba
    assert cba.share_of_reference is aba.share_of_reference
    assert cba.NEAR_ZERO_REFERENCE is aba.NEAR_ZERO_REFERENCE


# ------------------------------------------------------------------ all-fitted contrast, no opt_z
# --relative-magnitudes with no opt_z leaves every value in its dataset's rating
# units, where 0.01 of an achievable improvement means nothing: only the sign
# rule applies there.
FITTED = ("f1", "f2", "f3")
OWN_SIGMAS = {"f1": (0.1, 0.5, 2.0, 10.0), "f2": (0.2, 1.0, 4.0, 20.0), "f3": (0.3, 1.5, 6.0, 30.0)}
# grid position -> raw reference excess per dataset; the treatment removes 40% of it.
RAW = {0: [0.004, 0.005, 0.006],   # small in raw units but positive everywhere: reported
       1: [0.2, 0.3, -0.01],       # negative on one dataset: suppressed
       2: [0.2, 0.3, 0.4], 3: [0.5, 0.6, 0.7]}


def _write_fitted(root: Path, treatment: bool) -> None:
    for d in FITTED:
        rows = []
        for pos, own in enumerate(OWN_SIGMAS[d]):
            ref = RAW[pos][FITTED.index(d)]
            for seed in (7, 8):
                rows.append({"dataset": d, "acquisition": "logei", "error_model": "gaussian", "jitter_std": own,
                             "jitter_iteration": 0, "seed": seed, METRIC: 0.6 * ref if treatment else ref})
        (root / d / "evaluation").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(root / d / "evaluation" / "paired_excess_metrics.csv", index=False)


def test_an_all_fitted_contrast_applies_only_the_sign_rule(tmp_path, monkeypatch):
    ref, trt, out = tmp_path / "ref", tmp_path / "trt", tmp_path / "out"
    _write_fitted(ref, treatment=False)
    _write_fitted(trt, treatment=True)
    monkeypatch.setattr(cba.bb, "load_stats", lambda *a, **k: {})
    _run(ref, trt, out, "--relative-magnitudes")
    s = pd.read_csv(out / "arm_contrast_summary.csv").set_index("jitter_std")
    assert np.isclose(s.loc[0.05, "share_removed"], 0.4)       # 0.005 raw is not "near zero"
    assert np.isnan(s.loc[0.25, "share_removed"]) and s.loc[0.25, "n_reference_negative"] == 1
    assert np.allclose(s.loc[[1.0, 5.0], "share_removed"], 0.4)
    summary = json.loads((out / "arm_contrast_summary.json").read_text(encoding="utf-8"))
    assert summary["near_zero_reference"] is None
    assert "no near-zero threshold applies" in (out / "arm_contrast_report.txt").read_text(encoding="utf-8")
