"""The frozen human-feedback selection (scripts/feedback_select.py) on a synthetic benchmark output.

Two cells, three landscapes, three procedures: the reference gains 0 at no price; 'a:policy' gains 0.10 in
the first cell and loses 0.02 in the second at a price of 0.03; 'b:policy' gains 0.04 in both at no price.
V = (sum of gains - price) / 3: a 0.0167, b 0.0267, so the single procedure is b; the map takes a in the
first cell and b in the second. The score must apply the frozen choice unchanged, and a frozen selection
must not be overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import feedback_select as fs  # noqa: E402

REF = "bo_logei:observed_max"
GAINS = {("bo_logei", "observed_max"): (0.0, 0.0, 0.0), ("a", "policy"): (0.10, -0.02, 0.03),
         ("b", "policy"): (0.04, 0.04, 0.0)}


def write_output(path: Path, shift: float = 0.0) -> Path:
    rows = []
    for (method, rule), (g1, g2, price) in GAINS.items():
        for (error, mag, onset), g in ((("gaussian", 1.0, 0), g1), (("drift", 5.0, 20), g2)):
            for i, land in enumerate(("l1", "l2", "l3")):
                jitter = 0.0 if method == "bo_logei" else (i - 1) * 0.001 + shift
                rows.append({"method": method, "ship_rule": rule, "reference": REF, "response": fs.RESPONSE,
                             "error_model": error, "magnitude": mag, "onset": onset, "dataset": land,
                             "gain": g + jitter, "price": price})
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "comparison_per_landscape.csv", index=False)
    for name in ("comparison_summary.csv", "per_run_metrics.csv", "manifest.json"):
        (path / name).write_text("x", encoding="utf-8")
    return path


def test_freeze_chooses_on_v_and_the_map_per_cell(tmp_path):
    rec = fs.freeze(write_output(tmp_path / "screen"), tmp_path / "sel.json")
    assert rec["reference"] == REF and rec["single"] == "b:policy"
    assert rec["map"] == {"drift|5|20": "b:policy", "gaussian|1|0": "a:policy"}
    assert rec["screen_V"]["a:policy"] == pytest.approx((0.10 - 0.02 - 0.03) / 3)
    assert len(rec["screen_inputs_sha256"]) == 4


def test_the_map_keeps_the_reference_where_nothing_gains(tmp_path):
    out = write_output(tmp_path / "screen")
    p = pd.read_csv(out / "comparison_per_landscape.csv")
    p.loc[(p.error_model == "drift") & (p.method != "bo_logei"), "gain"] = -0.01
    p.to_csv(out / "comparison_per_landscape.csv", index=False)
    assert fs.freeze(out, tmp_path / "sel.json")["map"]["drift|5|20"] == REF


def test_score_applies_the_frozen_choice_and_freeze_refuses_to_overwrite(tmp_path):
    sel = tmp_path / "sel.json"
    fs.freeze(write_output(tmp_path / "screen"), sel)
    frame, lines = fs.score(write_output(tmp_path / "heldout", shift=0.01), sel)
    single = frame[frame.policy == "single"].set_index("cell")
    assert set(single["procedure"]) == {"b:policy"}
    assert single.loc["gaussian|1|0", "gain"] == pytest.approx(0.05)
    mapped = frame[frame.policy == "map"].set_index("cell")
    assert mapped.loc["gaussian|1|0", "procedure"] == "a:policy"
    assert any(line.strip().startswith("V single") for line in lines)
    with pytest.raises(SystemExit):
        fs.main(["freeze", "--screen", str(tmp_path / "screen"), "--out", str(sel)])
    assert json.loads(sel.read_text(encoding="utf-8"))["single"] == "b:policy"
