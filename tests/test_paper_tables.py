"""The paper-table helpers the input-error section depends on.

Every number in paper/tables is generated, so a formatting slip is a slip in
every table at once. These pin the two behaviours that are easy to break:
intervals that straddle zero, and tables whose results do not exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_boba_paper_tables as mt  # noqa: E402


def _row(mean: float, lo: float, hi: float) -> pd.DataFrame:
    return pd.DataFrame([{"mean": mean, "ci_low": lo, "ci_high": hi}])


def test_pct_ci_formats_a_cell():
    assert mt._pct_ci(_row(0.123, 0.051, 0.249)) == r"12.3 \makebox[3.4em][l]{\scriptsize [5, 25]}"


def test_pct_ci_never_prints_negative_zero():
    """A lower bound of -0.4% means 'indistinguishable from zero', not '-0'."""
    assert "-0" not in mt._pct_ci(_row(0.02, -0.004, 0.05))


def test_pct_ci_keeps_a_real_negative_bound():
    assert "[-1, " in mt._pct_ci(_row(0.02, -0.012, 0.05))


def test_pct_ci_marks_an_absent_cell():
    assert mt._pct_ci(_row(0.0, 0.0, 0.0).iloc[0:0]) == "--"


def test_inputerror_tables_skip_quietly_without_results(tmp_path, capsys):
    """Wired into main() before the arm finishes, so absence must be harmless."""
    mt.table_inputerror_dose(tmp_path / "missing", tmp_path)
    mt.table_inputerror_mechanism(tmp_path / "missing", tmp_path)
    assert not list(tmp_path.glob("*.tex"))
    assert "skipped" in capsys.readouterr().out


def test_inputerror_dose_table_has_one_row_per_magnitude(tmp_path):
    rows = [
        {"arm": arm, "jitter_std": std, "jitter_iteration": onset,
         "n_landscapes": 20, "mean": 0.1, "ci_low": 0.05, "ci_high": 0.2}
        for arm in ("slip", "misclick")
        for std in (0.01, 0.05, 0.15, 0.4)
        for onset in (0, 20)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "inputerror_dose.csv", index=False)
    mt.table_inputerror_dose(tmp_path, tmp_path)
    tex = (tmp_path / "inputerror_dose.tex").read_text(encoding="utf-8")
    magnitude_rows = [line for line in tex.splitlines() if "\\%" in line]
    assert len(magnitude_rows) == 4
    assert "from it.\\ 21" in tex


def test_inputerror_deployed_table_pairs_evaluated_with_deployed(tmp_path):
    rows = [
        {"arm": arm, "jitter_std": mag, "jitter_iteration": onset, "n_landscapes": 20,
         "mean": 0.1, "ci_low": 0.05, "ci_high": 0.15}
        for arm in ("evaluated", "deployed") for mag in (0.01, 0.05) for onset in (0, 20)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "inputerror_deployed.csv", index=False)
    mt.table_inputerror_deployed(tmp_path, tmp_path)
    tex = (tmp_path / "inputerror_deployed.tex").read_text(encoding="utf-8")
    body = [l for l in tex.splitlines() if l.split("&")[0].strip().endswith("\\%")]
    assert len(body) == 2
    assert all(l.count("&") == 4 for l in body)


def test_write_sets_a_negative_number_in_math_mode(tmp_path):
    mt.write(tmp_path / "t.tex", "a & -0.07 & [-1, 2] & 7.2e-01 & -- & 2-3 & $x - 1$ \\\\")
    assert (tmp_path / "t.tex").read_text(encoding="utf-8").strip() == (
        "a & $-$0.07 & [$-$1, 2] & 7.2e-01 & -- & 2-3 & $x - 1$ \\\\"
    )


def test_p_fdr_floors_small_values():
    assert mt._p_fdr(0.000504) == "$<$0.001"
    assert mt._p_fdr(0.0458) == "0.046"
    assert mt._p_fdr(float("nan")) == "--"


def test_inputerror_main_table_has_seven_cells_per_magnitude(tmp_path):
    def rows(arms):
        return [{"arm": arm, "jitter_std": mag, "jitter_iteration": onset, "n_landscapes": 20,
                 "mean": 0.1, "ci_low": 0.05, "ci_high": 0.15}
                for arm in arms for mag in (0.01, 0.05) for onset in (0, 20)]
    pd.DataFrame(rows(("slip", "misclick", "slip-actual"))).to_csv(tmp_path / "inputerror_dose.csv", index=False)
    pd.DataFrame(rows(("floor", "proposed", "actual"))).to_csv(tmp_path / "inputerror_mechanism.csv", index=False)
    pd.DataFrame(rows(("proposed - actual",))).to_csv(tmp_path / "inputerror_mislabel.csv", index=False)
    mt.table_inputerror_main(tmp_path, tmp_path)
    tex = (tmp_path / "inputerror_main.tex").read_text(encoding="utf-8")
    body = [l for l in tex.splitlines() if l.split("&")[0].strip().endswith("\\%")]
    assert len(body) == 2
    assert all(l.count("&") == 7 for l in body)


def test_extra_cell_marks_a_censored_median_as_a_bound():
    row = pd.DataFrame([{"median_extra": 75.0, "censored_fraction": 0.7}])
    assert mt._extra_cell(row) == "$>$75 (70\\%)"
    row = pd.DataFrame([{"median_extra": 18.0, "censored_fraction": 0.14}])
    assert mt._extra_cell(row) == "18 (14\\%)"


def test_inputerror_mechanism_table_has_one_block_per_onset(tmp_path):
    rows = [{"arm": arm, "jitter_std": mag, "jitter_iteration": onset, "n_landscapes": 20,
             "mean": 0.1, "ci_low": 0.05, "ci_high": 0.15}
            for arm in ("floor", "proposed", "actual") for mag in (0.01, 0.05) for onset in (0, 20)]
    pd.DataFrame(rows).to_csv(tmp_path / "inputerror_mechanism.csv", index=False)
    pd.DataFrame([dict(r, arm="proposed - actual") for r in rows if r["arm"] == "floor"]).to_csv(
        tmp_path / "inputerror_mislabel.csv", index=False)
    mt.table_inputerror_mechanism(tmp_path, tmp_path)
    tex = (tmp_path / "inputerror_mechanism.tex").read_text(encoding="utf-8")
    assert tex.count("error from it.") == 2
    body = [l for l in tex.splitlines() if l.split("&")[0].strip().endswith("\\%")]
    assert len(body) == 4 and all(l.count("&") == 4 for l in body)



def test_adaptations_table_groups_arms_and_stars_the_corrected_test(tmp_path):
    # The star follows the BH-corrected Wilcoxon the analysis ran, NOT the
    # bootstrap interval. rep10 has a tiny q and is starred; nigp's interval also
    # excludes zero on one response but its q does not clear 0.05, so it is not.
    rows = []
    spec = {  # arm: (recovered, lo, hi, price, pooled q)
        "rep10": (-0.88, -1.13, -0.67, 0.051, 0.001),
        "nigp": (-0.28, -0.43, -0.10, -0.001, 0.400),
        "fitted-rep10": (-0.08, -0.17, 0.06, 0.024, 0.250),
    }
    for arm, (rec, lo, hi, price, q) in spec.items():
        for response in ("trajectory", "deployed"):
            for onset in (0, 20):
                rows.append({"arm": arm, "reference": "r", "response": response, "jitter_std": 1.0,
                             "jitter_iteration": onset, "pooled_recovered": rec, "pooled_recovered_lo": lo,
                             "pooled_recovered_hi": hi, "pooled_price": price,
                             "pooled_wilcoxon_p_fdr": q})
    pd.DataFrame(rows).to_csv(tmp_path / "adaptations_recovery.csv", index=False)
    mt.table_adaptations(tmp_path, tmp_path)
    tex = (tmp_path / "adaptations.tex").read_text(encoding="utf-8")
    lines = tex.splitlines()
    groups = [l for l in lines if l.startswith("\\multicolumn{5}{l}")]
    body = [l for l in lines if l.endswith("\\\\") and l.count("&") == 4 and "adaptation" not in l]
    assert len(groups) == 3          # rating error, slip, fitted; misclick has no row
    assert len(body) == 3
    assert tex.count("$^{*}$") == 2  # rep10 on both responses; nigp's interval excludes
    # zero on both too, and is correctly unstarred because its q is 0.40
    fitted = [l for l in body if "rated twice" in l and "--" in l]
    assert len(fitted) == 1          # the fitted row has no price


def test_adaptations_table_never_prints_a_nonzero_bound_as_zero(tmp_path):
    # The re-rate arm's deployed interval is [0.0395%, 9.3%], which must not read as
    # [0, 9]; its trajectory interval [-0.28%, 0.24%] must not read as [0, 0].
    rows = []
    spec = {"trajectory": (-0.00003, -0.0028, 0.0024, 0.30),
            "deployed": (0.0412, 0.000395, 0.0928, 0.01)}
    for response, (rec, lo, hi, q) in spec.items():
        for onset in (0, 20):
            rows.append({"arm": "rerate", "reference": "r", "response": response, "jitter_std": 1.0,
                         "jitter_iteration": onset, "pooled_recovered": rec, "pooled_recovered_lo": lo,
                         "pooled_recovered_hi": hi, "pooled_price": 0.001,
                         "pooled_wilcoxon_p_fdr": q})
    pd.DataFrame(rows).to_csv(tmp_path / "adaptations_recovery.csv", index=False)
    mt.table_adaptations(tmp_path, tmp_path)
    tex = (tmp_path / "adaptations.tex").read_text(encoding="utf-8")
    assert "[0.04, 9]" in tex
    assert "0.2]" in tex and ("[$-$0.3," in tex or "[-0.3," in tex)
    assert "[0, 9]" not in tex and "[0, 0]" not in tex
    assert tex.count("$^{*}$") == 1  # only the deployed response clears the corrected test


def test_pilot_frag_table_orders_pilot_sizes_and_ends_with_the_exact_ceiling(tmp_path):
    rows = []
    for k, base in (("10", 0.8), ("50", 0.9), ("20", 0.85), ("exact", 1.0)):
        for sigma in ("0.05", "0.25", "1", "5", "pooled"):
            for target in ("frag_exact", "cost_measured"):
                rows.append({"k": k, "sigma_e": sigma, "target": target, "n": 20,
                             "spearman": base if target == "frag_exact" else base - 0.1})
    pd.DataFrame(rows).to_csv(tmp_path / "pilot_frag_summary.csv", index=False)
    mt.table_pilot_frag(tmp_path, tmp_path)
    tex = (tmp_path / "pilot_frag.tex").read_text(encoding="utf-8")
    body = [l for l in tex.splitlines() if l.startswith(("first", "exact"))]
    assert [l.split("&")[0].strip() for l in body] == ["first 10 trials", "first 20 trials", "first 50 trials", "exact objective"]
    assert body[-1].split("&")[1].strip() == "1.00"
