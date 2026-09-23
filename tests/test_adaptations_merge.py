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
    monkeypatch.setattr(aba.bb, "load_stats", lambda *args, **kwargs: {})

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
