"""analyse_boba_adaptations.main: adaptations_extra_runs.csv is merged, never truncated.

On 2026-09-22 a call that computed no extra trials -- --no-extra-trials, or only
relative, pooled or variant arms -- rewrote adaptations_extra_runs.csv as an
empty frame, and the extra-trial numbers of the process-adaptations appendix
went with it. main() now merges the file on arm (this call's arms replace their
old rows, every other arm is kept) and leaves it alone when nothing was
computed. These tests drive the real main() on a tiny synthetic tree, with a
stubbed ARMS registry, the landscape stats stubbed out and the
analyse_extra_runs.py subprocess forbidden (its outputs are pre-written, which
main() then reuses).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyse_boba_adaptations as aba

DATASETS = ("d1", "d2", "d3")
OPT_Z = {"d1": 1.0, "d2": 4.0, "d3": 10.0}
SEEDS = (7, 8)
ONSETS = (0, 20)
EXTRA_COLUMNS = ["arm", "jitter_std", "jitter_iteration", "k", "tolerance", "median_extra_arm", "never_arm",
                 "mean_extra_arm", "median_extra_ref", "never_ref", "mean_extra_ref"]


def _paired_metrics(root: Path, shift: float, seed: int, variants=(None,)) -> None:
    """One paired_excess_metrics.csv per dataset, one row per cell (and variant)."""
    rng = np.random.default_rng(seed)
    for dataset in DATASETS:
        rows = []
        for variant in variants:
            for std in aba.GRID:
                for onset in ONSETS:
                    for s in SEEDS:
                        row = {"dataset": dataset, "acquisition": "logei", "error_model": "gaussian",
                               "jitter_std": std, "jitter_iteration": onset, "seed": s}
                        if variant is not None:
                            row["variant"] = variant
                        for response in aba.RESPONSES.values():
                            clean = 1.0 + rng.random()
                            row[f"{response}_baseline"] = clean
                            # Noise costs something, the treatment gives some of it back.
                            row[f"{response}_jitter"] = clean + std * (1.0 + rng.random()) - shift
                        rows.append(row)
        (root / dataset / "evaluation").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(root / dataset / "evaluation" / "paired_excess_metrics.csv", index=False)


def _extra_runs(path: Path, median: float) -> None:
    """What analyse_extra_runs.py would have written for one arm or reference."""
    rows = [{"error_model": "gaussian", "variant": "", "jitter_std": std, "jitter_iteration": onset, "k": k,
             "tolerance": 0.01, "median_extra": median + k, "censored_fraction": 0.1, "mean_extra": median + k + 0.5}
            for std in aba.GRID for onset in ONSETS for k in (10, 25)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    ref, plain, pooled, relative, variant = (tmp_path / n for n in ("ref", "plain", "pooled", "relative", "variant"))
    _paired_metrics(ref, shift=0.0, seed=1)
    _paired_metrics(plain, shift=0.3, seed=2)
    _paired_metrics(pooled, shift=0.2, seed=3)
    _paired_metrics(relative, shift=0.1, seed=4)
    _paired_metrics(variant, shift=0.4, seed=5, variants=("v1", "v2"))
    _extra_runs(plain / "analysis" / "extra_runs_vs_plain_reference.csv", median=3.0)
    _extra_runs(ref / "analysis" / "extra_runs_ref_plain.csv", median=8.0)

    seeds = ",".join(str(s) for s in SEEDS)
    arms = {
        "plain": aba.arm(str(plain), str(ref), "logei", "a plain arm", seeds=seeds, error_model="gaussian"),
        "pooled": aba.arm(str(pooled), str(ref), "logei", "a pooled arm", seeds=seeds, pool=True,
                          error_model="gaussian"),
        "relative": aba.arm(str(relative), str(ref), "logei", "a relative arm", seeds=seeds, relative=True,
                            error_model="gaussian"),
        "variant": aba.arm(str(variant), str(ref), "logei", "one variant of several", seeds=seeds,
                           error_model="gaussian", variant="v1"),
    }
    monkeypatch.setattr(aba, "ARMS", arms)
    # Every synthetic dataset needs a scale: paired_frame no longer falls back to 1.0.
    monkeypatch.setattr(aba.bb, "load_stats", lambda *args, **kwargs: {d: {"opt_z": OPT_Z[d]} for d in DATASETS})

    def _no_subprocess(cmd):
        raise AssertionError(f"analyse_extra_runs.py should not run: {cmd}")

    monkeypatch.setattr(aba, "run", _no_subprocess)
    out = tmp_path / "analysis"
    out.mkdir()
    return out


def _previous_extra(out: Path) -> Path:
    """An extra-runs file holding another arm and a stale row for 'plain'."""
    rows = [dict(zip(EXTRA_COLUMNS, ["other", 1.0, 0, 10, 0.01, 11.0, 0.2, 12.0, 20.0, 0.4, 21.0])),
            dict(zip(EXTRA_COLUMNS, ["other", 1.0, 0, 25, 0.01, 13.0, 0.3, 14.0, 22.0, 0.5, 23.0])),
            dict(zip(EXTRA_COLUMNS, ["plain", 1.0, 0, 10, 0.01, -999.0, 0.0, -999.0, -999.0, 0.0, -999.0]))]
    path = out / "adaptations_extra_runs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_extra_trials_are_merged_on_arm(tree):
    path = _previous_extra(tree)
    other_before = pd.read_csv(path).query("arm == 'other'").reset_index(drop=True)
    aba.main(["--arms", "plain", "--output-dir", str(tree)])

    after = pd.read_csv(path)
    assert list(after.columns) == EXTRA_COLUMNS
    pd.testing.assert_frame_equal(after[after.arm == "other"].reset_index(drop=True), other_before)
    plain = after[after.arm == "plain"]
    # The stale row is replaced, not kept beside the new ones.
    assert len(plain) == len(aba.GRID) * len(ONSETS) * 2
    assert (plain.median_extra_arm > 0).all() and not (plain == -999.0).any().any()
    assert set(plain.median_extra_arm - plain.k) == {3.0} and set(plain.median_extra_ref - plain.k) == {8.0}
    assert pd.read_csv(tree / "adaptations_recovery.csv")["arm"].unique().tolist() == ["plain"]


def test_a_rerun_of_the_same_arm_is_idempotent(tree):
    aba.main(["--arms", "plain", "--output-dir", str(tree)])
    first = (tree / "adaptations_extra_runs.csv").read_bytes()
    aba.main(["--arms", "plain", "--output-dir", str(tree)])
    assert (tree / "adaptations_extra_runs.csv").read_bytes() == first


def test_no_extra_trials_leaves_the_file_alone(tree):
    path = _previous_extra(tree)
    before = path.read_bytes()
    aba.main(["--arms", "plain", "--no-extra-trials", "--output-dir", str(tree)])
    assert path.read_bytes() == before
    # The recovery table is still written: only the extra-trial step was skipped.
    assert pd.read_csv(tree / "adaptations_recovery.csv")["arm"].unique().tolist() == ["plain"]


def test_pooled_relative_and_variant_arms_leave_the_file_alone(tree):
    path = _previous_extra(tree)
    before = path.read_bytes()
    aba.main(["--arms", "pooled,relative,variant", "--output-dir", str(tree)])
    assert path.read_bytes() == before
    recovery = pd.read_csv(tree / "adaptations_recovery.csv")
    assert sorted(recovery["arm"].unique()) == ["pooled", "relative", "variant"]


def test_nothing_is_written_when_nothing_was_computed(tree):
    aba.main(["--arms", "plain", "--no-extra-trials", "--output-dir", str(tree)])
    assert not (tree / "adaptations_extra_runs.csv").exists()


def test_arms_without_results_write_nothing_and_do_not_crash(tree, monkeypatch):
    recovery = tree / "adaptations_recovery.csv"
    pd.DataFrame([{"arm": "other", "response": "trajectory", "jitter_std": 1.0, "jitter_iteration": 0}]).to_csv(
        recovery, index=False)
    before = recovery.read_bytes()
    empty = tree.parent / "empty"
    empty.mkdir()
    monkeypatch.setitem(aba.ARMS, "plain", dict(aba.ARMS["plain"], dir=str(empty)))
    aba.main(["--arms", "plain", "--output-dir", str(tree)])
    assert recovery.read_bytes() == before
    assert not (tree / "adaptations_extra_runs.csv").exists()


def test_a_file_emptied_by_the_old_bug_is_replaced(tree):
    path = tree / "adaptations_extra_runs.csv"
    pd.DataFrame().to_csv(path, index=False)          # what a truncating call used to leave
    assert path.stat().st_size <= 10
    aba.main(["--arms", "plain", "--output-dir", str(tree)])
    after = pd.read_csv(path)
    assert list(after.columns) == EXTRA_COLUMNS and after["arm"].unique().tolist() == ["plain"]


def test_a_mixed_call_merges_only_the_arms_that_computed_extra_trials(tree):
    path = _previous_extra(tree)
    aba.main(["--arms", "plain,pooled,variant", "--output-dir", str(tree)])
    after = pd.read_csv(path)
    assert sorted(after["arm"].unique()) == ["other", "plain"]
    recovery = pd.read_csv(tree / "adaptations_recovery.csv")
    assert sorted(recovery["arm"].unique()) == ["plain", "pooled", "variant"]


# ---------------------------------------------------------------- the per-dataset scale
# Until 2026-09-29 paired_frame divided by opt_z.get(dataset, 1.0), so the four
# multi-objective halo problems were pooled in raw hypervolume (one of them 91% of
# the summed cost) and the fitted-oracle datasets in raw rating units.


def _pair_inputs(datasets=("a", "b")) -> tuple[pd.DataFrame, pd.DataFrame]:
    response = aba.RESPONSES["deployed"]
    rows_ref, rows_trt = [], []
    for i, d in enumerate(datasets):
        for seed in (7, 8):
            key = {"dataset": d, "acquisition": "logei", "error_model": "gaussian", "jitter_std": 1.0,
                   "jitter_iteration": 0, "seed": seed}
            rows_ref.append({**key, f"{response}_jitter": 3.0 * (i + 1), f"{response}_baseline": 1.0 * (i + 1)})
            rows_trt.append({**key, f"{response}_jitter": 2.0 * (i + 1), f"{response}_baseline": 1.5 * (i + 1)})
    return pd.DataFrame(rows_ref), pd.DataFrame(rows_trt)


def test_paired_frame_raises_on_a_dataset_without_a_scale():
    ref, trt = _pair_inputs()
    with pytest.raises(KeyError, match=r"no scale for \['b'\]"):
        aba.paired_frame(ref, trt, aba.RESPONSES["deployed"], {"a": 2.0}, pool=False)
    with pytest.raises(KeyError):
        aba.paired_frame(ref, trt, aba.RESPONSES["deployed"], {}, pool=False)


def test_paired_frame_rejects_a_non_positive_scale():
    ref, trt = _pair_inputs()
    with pytest.raises(ValueError, match="non-positive scale"):
        aba.paired_frame(ref, trt, aba.RESPONSES["deployed"], {"a": 2.0, "b": 0.0}, pool=False)


def test_paired_frame_divides_by_each_datasets_own_scale():
    ref, trt = _pair_inputs()
    p = aba.paired_frame(ref, trt, aba.RESPONSES["deployed"], {"a": 2.0, "b": 8.0}, pool=False)
    a, b = p[p.dataset == "a"].iloc[0], p[p.dataset == "b"].iloc[0]
    assert (a.ref_noisy, a.ref_clean, a.trt_noisy, a.trt_clean) == (1.5, 0.5, 1.0, 0.75)
    assert (b.ref_noisy, b.ref_clean, b.trt_noisy, b.trt_clean) == (0.75, 0.25, 0.5, 0.375)


def test_a_datasets_recovery_does_not_depend_on_its_scale_and_the_pool_is_cost_weighted():
    ref, trt = _pair_inputs()
    response = aba.RESPONSES["deployed"]
    for scale in ({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 8.0}):
        p = aba.paired_frame(ref, trt, response, scale, pool=False)
        per = aba.per_dataset(p).set_index("dataset")
        # Both datasets: cost 2 (i+1), gain 1 (i+1), so each recovers exactly a half.
        assert per.recovered.tolist() == [0.5, 0.5]
        assert np.isclose(per.cost_share.sum(), 1.0)
        pooled = aba.summarise(p, np.random.default_rng(aba.BOOTSTRAP_SEED))
        assert np.isclose(pooled["recovered"], (per.recovered * per.cost_share).sum())
        assert np.isclose(pooled["price"], per.price.mean())


def test_arm_rejects_an_unknown_scale():
    with pytest.raises(ValueError, match="unknown scale"):
        aba.arm("x", "y", "logei", "an arm", scale="raw")


def test_every_arm_names_a_scale_and_the_arms_without_opt_z_name_theirs():
    assert {spec["scale"] for spec in aba.ARMS.values()} <= set(aba.SCALES)
    assert aba.ARMS["fitted-rep10"]["scale"] == "fitted_achievable"
    for name in ("mo-halo-cost", "mo-halo-backfit", "mo-halo-backfit-price"):
        assert aba.ARMS[name]["scale"] == "hv_floor_gap"
    others = {n for n, s in aba.ARMS.items() if s["scale"] != "opt_z"}
    assert others == {"fitted-rep10", "mo-halo-cost", "mo-halo-backfit", "mo-halo-backfit-price"}


def test_main_raises_rather_than_score_a_dataset_without_opt_z(tree, monkeypatch):
    monkeypatch.setattr(aba.bb, "load_stats", lambda *args, **kwargs: {"d1": {"opt_z": 1.0}})
    with pytest.raises(KeyError, match="no scale for"):
        aba.main(["--arms", "plain", "--no-extra-trials", "--output-dir", str(tree)])
    assert not (tree / "adaptations_recovery.csv").exists()


def test_main_writes_the_scale_and_the_per_dataset_values(tree):
    aba.main(["--arms", "plain,relative", "--no-extra-trials", "--output-dir", str(tree)])
    recovery = pd.read_csv(tree / "adaptations_recovery.csv")
    assert set(recovery["scale"]) == {"opt_z"}
    assert (recovery["pooled_n_landscapes"] == len(DATASETS)).all()
    per = pd.read_csv(tree / "adaptations_per_dataset.csv")
    assert sorted(per["arm"].unique()) == ["plain", "relative"]
    assert len(per) == 2 * len(aba.RESPONSES) * len(DATASETS)
    assert dict(zip(per.dataset, per.scale_value)) == OPT_Z
    for (arm, response), blk in per.groupby(["arm", "response"]):
        row = recovery[(recovery.arm == arm) & (recovery.response == response)].iloc[0]
        assert np.isclose(row["pooled_recovered"], (blk.recovered * blk.cost_share).sum())
        assert np.isclose(row["pooled_recovered_range_lo"], blk.recovered.min())
        assert np.isclose(row["pooled_recovered_range_hi"], blk.recovered.max())
    # A second call for one arm keeps the other arm's per-dataset rows.
    aba.main(["--arms", "plain", "--no-extra-trials", "--output-dir", str(tree)])
    assert sorted(pd.read_csv(tree / "adaptations_per_dataset.csv")["arm"].unique()) == ["plain", "relative"]


# ---------------------------------------------------------------- a recovery is a share only when its cost is
# AGENTS.md: never report a recovery ratio whose reference cost is near zero or
# changes sign. The observed-max incumbent's +67% of the trajectory loss at
# 0.05 sigma from the first rating was quoted in the text on a cost of 0.0099
# that is negative on one landscape. main() keeps recovered and flags it with
# share_flags(), the same rule compare_boba_arms.py applies.


def _block(costs, gains) -> pd.DataFrame:
    """A paired frame with one row per landscape and the given cost and gain."""
    costs, gains = np.asarray(costs, float), np.asarray(gains, float)
    return pd.DataFrame({"dataset": [f"l{i}" for i in range(len(costs))], "ref_clean": 1.0,
                         "ref_noisy": 1.0 + costs, "trt_noisy": 1.0 + costs - gains, "trt_clean": 1.0})


def test_a_near_zero_cost_is_flagged_but_the_ratio_is_kept():
    block = _block([0.004, 0.006, 0.009], [0.002, 0.003, 0.004])
    f = aba.share_flags(block)
    assert "below 0.01" in f["recovered_suppressed"] and f["n_cost_negative"] == 0
    assert np.isclose(aba.summarise(block, np.random.default_rng(0))["recovered"], 0.009 / 0.019)


def test_a_cost_that_is_negative_on_one_landscape_is_flagged():
    f = aba.share_flags(_block([0.3, 0.2, -0.002], [0.1, 0.1, 0.0]))
    assert f["recovered_suppressed"] == "reference negative on 1 of 3 landscapes"
    assert f["n_cost_negative"] == 1


def test_a_positive_cost_above_the_threshold_is_not_flagged():
    block = _block([0.3, 0.2, 0.0], [0.1, 0.1, 0.0])
    assert aba.share_flags(block) == {"n_cost_negative": 0, "recovered_suppressed": ""}
    assert np.isclose(aba.summarise(block, np.random.default_rng(0))["recovered"], 0.2 / 0.5)


def test_summarise_keeps_its_keys_for_the_scripts_that_spread_it():
    # replay_end_of_study, replay_stopping and review_checks/tie_break write
    # **summarise(...) into their own outputs; the flags must not leak into them.
    keys = set(aba.summarise(_block([0.3, 0.2], [0.1, 0.1]), np.random.default_rng(0)))
    assert keys == {"n_landscapes", "n_cells", "cost", "gain", "price", "recovered", "bootstrap_draws_kept",
                    "recovered_lo", "recovered_hi", "wilcoxon_p"}


def test_the_share_rule_skips_the_threshold_only_when_asked():
    per = pd.DataFrame({"ref": [0.004, 0.006], "trt": [0.002, 0.003]})
    assert "below" in aba.share_of_reference(per)["share_suppressed"]
    assert np.isclose(aba.share_of_reference(per, near_zero=None)["share_removed"], 0.5)
    zero = aba.share_of_reference(pd.DataFrame({"ref": [0.0, 0.0], "trt": [0.0, 0.0]}), near_zero=None)
    assert np.isnan(zero["share_removed"]) and "not positive" in zero["share_suppressed"]


def test_main_writes_the_share_flags_per_cell_and_pooled(tree):
    aba.main(["--arms", "plain", "--no-extra-trials", "--output-dir", str(tree)])
    recovery = pd.read_csv(tree / "adaptations_recovery.csv")
    for column in ("n_cost_negative", "recovered_suppressed", "pooled_n_cost_negative",
                   "pooled_recovered_suppressed"):
        assert column in recovery.columns
    # The synthetic reference cost is std * (1 + U) / opt_z > 0.0075 on every landscape,
    # and its landscape mean is above 0.01 in every cell, so nothing is flagged.
    assert (recovery["n_cost_negative"] == 0).all()
    assert recovery["recovered_suppressed"].isna().all()
    assert recovery["pooled_recovered_suppressed"].isna().all()


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (REPO / "output-boba-mo" / "run_metadata.json").is_file(),
                    reason="the multi-objective sweep lives only on the simulation machine")
def test_the_halo_problems_have_a_positive_floor_gap(monkeypatch):
    monkeypatch.chdir(REPO)
    gap = aba.dataset_scale("hv_floor_gap")
    assert {"branincurrin", "dtlz2", "vehiclesafety", "zdt1"} <= set(gap)
    assert all(v > 0 for v in gap.values())
