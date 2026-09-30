"""The table generator's new guards and cells, and the checks the paper's numbers now rest on.

Each test pins one behaviour a reader of the paper depends on: a table the paper
inputs is never left stale by a silently skipped input; a suppressed share
prints as '--'; a three-dataset row is reported by its range and never starred;
the capped-scale column is repeated under a uniform tie-break with the
procedures' prices; the fitted-oracle table carries the headline unit first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
for p in (SCRIPTS, SCRIPTS / "review_checks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_boba_paper_tables as mt  # noqa: E402


# ---------------------------------------------------------------------------
# A missing input of a table the paper inputs is fatal
# ---------------------------------------------------------------------------


def test_the_paper_inputs_are_read_through_wrappers_and_comments_are_ignored(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "wrapper.tex").write_text("\\begin{table}\\input{tables/inner}\\end{table}\n", encoding="utf-8")
    (tmp_path / "main.tex").write_text(
        "\\input{tables/wrapper}\n% \\input{tables/commented}\n5\\% of trials \\input{tables/after_percent}\n",
        encoding="utf-8")
    assert mt.paper_table_inputs(tmp_path / "main.tex") == {"wrapper", "inner", "after_percent"}


def test_the_real_paper_inputs_every_table_this_script_needs_to_make():
    required = mt.paper_table_inputs(REPO / "paper" / "main.tex")
    assert {"shortlist", "adaptations", "fitted_companion", "known_noise", "incumbent",
            "incumbent_by_acquisition", "benchmarks"} <= required
    # Everything the paper inputs is either made here or has its own producer.
    assert required - mt.PRODUCED <= {"benchmarks_wrapper", "sitting_by_magnitude"}


def test_a_missing_input_of_a_required_table_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "REQUIRED", {"shortlist"})
    monkeypatch.setattr(mt, "ALLOW_MISSING", False)
    with pytest.raises(SystemExit) as err:
        mt.table_shortlist(tmp_path, tmp_path, root=tmp_path)
    assert "shortlist.tex" in str(err.value) and "hitl_remedies_recovery.csv" in str(err.value)
    assert not (tmp_path / "shortlist.tex").exists()


def test_allow_missing_restores_the_skip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mt, "REQUIRED", {"shortlist"})
    monkeypatch.setattr(mt, "ALLOW_MISSING", True)
    mt.table_shortlist(tmp_path, tmp_path, root=tmp_path)
    assert "skipped shortlist.tex" in capsys.readouterr().out


def test_a_missing_arm_row_is_a_missing_input_too(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "REQUIRED", {"budget"})
    monkeypatch.setattr(mt, "ALLOW_MISSING", False)
    with pytest.raises(SystemExit, match="budget.tex"):
        mt.table_budget({"$T=25$": tmp_path / "nowhere"}, tmp_path)


def test_main_refuses_to_run_without_the_paper(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        mt.main(["--paper", str(tmp_path / "missing.tex"), "--out", str(tmp_path)])
    # and leaves the module's policy as a direct call expects it
    assert mt.REQUIRED == set() and mt.ALLOW_MISSING is False


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def _summary(tmp_path: Path) -> Path:
    rows = [{"jitter_std": 0.05, "jitter_iteration": 0, "mean_reference": 0.010, "mean_treatment": 0.003,
             "mean_delta": -0.007, "share_removed": np.nan, "cohens_dz": -0.56, "wilcoxon_p_fdr": 0.019},
            {"jitter_std": 1.0, "jitter_iteration": 0, "mean_reference": 0.12, "mean_treatment": 0.09,
             "mean_delta": -0.03, "share_removed": 0.24, "cohens_dz": -0.6, "wilcoxon_p_fdr": 0.001}]
    pd.DataFrame(rows).to_csv(tmp_path / "arm_contrast_summary.csv", index=False)
    return tmp_path


def test_a_suppressed_share_prints_as_a_dash_and_delta_stays(tmp_path):
    mt.table_arm_contrast(_summary(tmp_path), tmp_path, "incumbent")
    tex = (tmp_path / "incumbent.tex").read_text(encoding="utf-8")
    assert "nan" not in tex
    assert "0.05 & 0 & 0.010 & 0.003 & $-$0.007 & -- & $-$0.56 & 0.019 \\\\" in tex
    assert "+24\\%" in tex


def test_the_by_acquisition_table_reads_the_share_rule_of_compare_boba_arms(tmp_path):
    pd.DataFrame([
        {"acquisition": "logpi", "n_benchmarks": 20, "cells": 8, "mean_reference": 0.10, "mean_treatment": 0.06,
         "mean_delta": -0.04, "share_removed": 0.41, "share_suppressed": ""},
        {"acquisition": "qnei", "n_benchmarks": 20, "cells": 8, "mean_reference": 0.049, "mean_treatment": 0.049,
         "mean_delta": 0.0, "share_removed": np.nan, "share_suppressed": "reference negative on 1 of 20"},
    ]).to_csv(tmp_path / "arm_contrast_by_acquisition.csv", index=False)
    mt.table_arm_by_acquisition(tmp_path, tmp_path, "incumbent")
    tex = (tmp_path / "incumbent_by_acquisition.tex").read_text(encoding="utf-8")
    assert "qNEI & 0.049 & 0.049 & +0.000 & -- \\\\" in tex
    assert "LogPI & 0.100 & 0.060 & $-$0.040 & +41\\% \\\\" in tex


def _recovery(path: Path, procs: dict[str, tuple[float, float, float, float]]) -> None:
    rows = [{"procedure": p, "error_model": "pooled", "recovered": r, "recovered_lo": lo, "recovered_hi": hi,
             "price": price} for p, (r, lo, hi, price) in procs.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


PROCS = ("shortlist_m1", "shortlist_m2", "shortlist_m3", "shortlist_m5", "ordinal_lcb1")


def _shortlist_tree(root: Path, fault_price: float = 0.023, ceiling_price: float | None = None) -> None:
    _recovery(root / "output-boba/analysis/hitl_remedies/hitl_remedies_recovery.csv",
              {p: (0.1, 0.0, 0.2, 0.022) for p in PROCS})
    _recovery(root / "output-boba-spike/analysis/hitl_remedies/hitl_remedies_recovery.csv",
              {p: (0.5, 0.4, 0.6, fault_price) for p in PROCS})
    _recovery(root / "output-boba-ceiling/analysis/hitl_remedies/hitl_remedies_recovery.csv",
              {p: (0.3, 0.2, 0.4, fault_price if ceiling_price is None else ceiling_price) for p in PROCS})
    ties = root / mt.SHORTLIST_TIES
    ties.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"arm": "output-boba-ceiling", "subdir": "hitl_remedies", "convention": c, "error_model": "pooled",
                   "procedure": p, "recovered": 0.15 if c == "uniform" else 0.3, "recovered_lo": 0.10,
                   "recovered_hi": 0.19, "price": fault_price} for p in PROCS for c in ("first", "uniform")]
                 ).to_csv(ties, index=False)


def test_the_shortlist_repeats_the_capped_column_with_random_ties_and_prices(tmp_path):
    _shortlist_tree(tmp_path)
    mt.table_shortlist(tmp_path, tmp_path, root=tmp_path)
    tex = (tmp_path / "shortlist.tex").read_text(encoding="utf-8")
    assert r"\shortstack{capped,\\random ties}" in tex and "{price}" in tex
    row = [l for l in tex.splitlines() if l.startswith("ship 1, chosen on ranks")][0]
    assert row.count("&") == 6          # label, four shares, two prices
    assert row.endswith("$15\\%$ {\\scriptsize $[10,19]$} & 0.022 & 0.023 \\\\")
    assert "$[$-$" not in tex   # the math-mode cells are not rewritten by write()'s minus rule


def test_the_fault_arms_must_share_their_price(tmp_path):
    _shortlist_tree(tmp_path, fault_price=0.023, ceiling_price=0.030)
    with pytest.raises(ValueError, match="share clean twins"):
        mt.table_shortlist(tmp_path, tmp_path, root=tmp_path)


def _fitted(path: Path, scale: float) -> None:
    rows = [{"arm": arm, "sigma_multiple": m, "jitter_iteration": o, "n_design_spaces": 3, "mean": scale * m / 10,
             "ci_low": 0.0, "ci_high": 1.0}
            for arm in ("synthetic", "fitted", "fitted no-aug") for m in (0.05, 0.25, 1.0, 5.0) for o in (0, 20)]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_the_fitted_table_leads_with_the_achievable_improvement(tmp_path):
    _fitted(tmp_path / "floor.csv", 1.0)
    _fitted(tmp_path / "ach.csv", 0.2)
    mt.table_fitted_companion(tmp_path / "floor.csv", tmp_path, achievable=tmp_path / "ach.csv")
    tex = (tmp_path / "fitted_companion.tex").read_text(encoding="utf-8")
    first = tex.index("share of the achievable improvement")
    second = tex.index("fraction of the floor gap")
    assert first < second
    ach_rows = [l for l in tex[first:second].splitlines() if l.startswith("fitted oracle (3)")]
    assert ach_rows == ["fitted oracle (3) & 0.1 & 0.5 & 2.0 & 10.0 & 0.1 & 0.5 & 2.0 & 10.0 \\\\"]
    assert "error from trial 1}" in tex and "error from trial 21}" in tex


def test_the_fitted_table_without_the_achievable_file_is_the_floor_gap_table(tmp_path):
    _fitted(tmp_path / "floor.csv", 1.0)
    mt.table_fitted_companion(tmp_path / "floor.csv", tmp_path)
    tex = (tmp_path / "fitted_companion.tex").read_text(encoding="utf-8")
    assert "floor gap" not in tex and "fitted oracle (3) & 0 & 2 & 10 & 50" in tex


def test_the_descriptor_vifs_are_the_primary_designs_continuous_terms(tmp_path):
    pd.DataFrame([{"model": "primary", "term": t, "coefficient": 0.1, "ci_low": 0.0, "ci_high": 0.2,
                   "p_fdr": 0.01, "p_bootstrap": 0.01, "r_squared": 0.5, "n_cells": 160, "n_clusters": 20}
                  for t in ("log_noise", "log_opt_z")]).to_csv(tmp_path / "descriptor_regression.csv", index=False)
    pd.DataFrame([{"term": "log_opt_z", "vif": 1.2, "kind": "continuous", "design": "primary"},
                  {"term": "log_opt_z", "vif": 9.2, "kind": "continuous", "design": "all_descriptors"},
                  {"term": "log_noise", "vif": 1.0, "kind": "continuous", "design": "primary"},
                  {"term": "error_model_bias", "vif": 1.5, "kind": "indicator", "design": "primary"}]
                 ).to_csv(tmp_path / "descriptor_vif.csv", index=False)
    mt.table_descriptor_regression(tmp_path, tmp_path)
    tex = (tmp_path / "descriptors.tex").read_text(encoding="utf-8")
    assert "& 1.2 \\\\" in tex and "9.2" not in tex


# ---------------------------------------------------------------------------
# The generated tables in the repository match what the generator writes now
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "output-boba" / "analysis" / "review" / "tie_break"
                         / "replayed_remedies_deployed.csv").is_file(), reason="tie-break outputs not present")
def test_the_committed_shortlist_table_is_the_generators_output(tmp_path):
    import os
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        mt.table_shortlist(REPO / "output-boba" / "analysis", tmp_path)
    finally:
        os.chdir(cwd)
    assert (tmp_path / "shortlist.tex").read_text(encoding="utf-8") == \
        (REPO / "paper" / "tables" / "shortlist.tex").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# fig:kcurve: the zero-trial ship rules on the same runs
# ---------------------------------------------------------------------------


def _kcurve_inputs(shift: float = 0.0):
    ks = [2, 5, 8, 16]
    policies = pd.DataFrame({"policy": [f"fixed_k{k}.0" for k in ks], "gain_vs_standard": [0.01, 0.02, 0.02, 0.01],
                             "gain_lo": 0.0, "gain_hi": 0.03})
    rows = []
    for cell, gains in (("pooled", [0.01 + shift, 0.02, 0.02, 0.01]), ("1sigma_from_trial_1", [0.03, 0.02, 0.0, -0.03])):
        for k, g in zip(ks, gains):
            rows.append({"cell": cell, "k": k, "gain_all": g, "gain_all_lo": g - 0.01, "gain_all_hi": g + 0.01,
                         "lcb1_gain_all": 0.019 if cell != "pooled" else 0.001,
                         "lcb2_gain_all": 0.020 if cell != "pooled" else 0.008})
    return policies, pd.DataFrame(rows)


def test_the_kcurve_draws_the_ship_rule_references(tmp_path):
    import make_paper_figures as mf

    policies, by_cell = _kcurve_inputs()
    mf.fig_kcurve(policies, tmp_path, by_cell)
    assert (tmp_path / "kcurve.pdf").is_file() and (tmp_path / "kcurve.png").is_file()


def test_the_kcurve_refuses_curves_from_different_runs(tmp_path):
    import make_paper_figures as mf

    policies, by_cell = _kcurve_inputs(shift=0.005)
    with pytest.raises(SystemExit, match="same runs"):
        mf.fig_kcurve(policies, tmp_path, by_cell)
