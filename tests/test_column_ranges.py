"""DatasetConfig.column_ranges: rows logged on another rating scale are dropped (2026-09-22).

opticarvis logs 40 of its 586 rows with the raw instrument values (1-5, 1-7,
-3..3) where every other row carries the value rescaled to [-1, 1]. A dataset
entry can now declare the range each column is logged on; load_observations
drops every row with any listed column outside it, and says so on stderr. These
tests pin that on a tiny semicolon-separated observation file: which rows go,
the tolerance at the edges, that unlisted and absent columns are left alone,
that an empty dict is a no-op, and that parse_dataset_configs reads the key
from JSON and refuses a malformed one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import bo_sensor_error_simulation as sim

PARAMS = ["p1", "p2"]
OBJECTIVES = ["o1", "o2"]
GLOB = "ObservationsPerEvaluation.csv"

# One row per case, keyed by Run. o1/o2 are logged on [-1, 1]; "aux" is a
# numeric column the objective does not use.
ROWS = [
    # Run  p1    p2    o1               o2     aux
    (1,    0.1,  0.2,  0.5,            -0.5,   3.0),   # in range
    (2,    0.3,  0.4,  -1.0,            1.0,   3.0),   # exactly on both edges: kept
    (3,    0.5,  0.6,  1.0 + 5e-10,     0.0,   3.0),   # inside the 1e-9 tolerance: kept
    (4,    0.7,  0.8,  4.0,             0.0,   3.0),   # o1 on a 1-5 scale: dropped
    (5,    0.9,  0.1,  0.0,            -3.0,   3.0),   # o2 on a -3..3 scale: dropped
    (6,    0.2,  0.3,  1.0 + 1e-6,      0.0,   3.0),   # just outside the tolerance: dropped
    (7,    0.4,  0.5,  0.2,             0.3,   9.0),   # in range; aux large
]


def _write_dataset(root: Path, rows=ROWS, user_ids=None) -> Path:
    frame = pd.DataFrame(rows, columns=["Run", *PARAMS, *OBJECTIVES, "aux"])
    frame.insert(0, "User_ID", user_ids if user_ids is not None else [1] * len(frame))
    (root / "u_1").mkdir(parents=True)
    frame.to_csv(root / "u_1" / GLOB, sep=";", index=False)
    return root


def _dataset(root: Path, **overrides) -> "sim.DatasetConfig":
    fields = dict(name="toy", data_dirs=[root], param_columns=list(PARAMS),
                  objective_map={"composite": list(OBJECTIVES)}, observation_glob=GLOB)
    fields.update(overrides)
    return sim.DatasetConfig(**fields)


def _runs(frame: pd.DataFrame) -> list[int]:
    return frame["Run"].astype(int).tolist()


# ---------------------------------------------------------------------------
# load_observations
# ---------------------------------------------------------------------------


def test_the_default_is_an_empty_dict_and_keeps_every_row(tmp_path, capsys):
    root = _write_dataset(tmp_path / "data")
    default = _dataset(root)
    assert default.column_ranges == {}
    assert sim.DatasetConfig(name="a", data_dirs=[], param_columns=[], objective_map={}).column_ranges \
        is not sim.DatasetConfig(name="b", data_dirs=[], param_columns=[], objective_map={}).column_ranges

    loaded = sim.load_observations(default, "composite")
    assert _runs(loaded) == [1, 2, 3, 4, 5, 6, 7]
    explicit = sim.load_observations(_dataset(root, column_ranges={}), "composite")
    pd.testing.assert_frame_equal(loaded, explicit)
    assert "dropping" not in capsys.readouterr().err


def test_rows_with_any_listed_column_off_scale_are_dropped_loudly(tmp_path, capsys):
    root = _write_dataset(tmp_path / "data")
    ranged = _dataset(root, column_ranges={"o1": (-1.0, 1.0), "o2": (-1.0, 1.0)})
    loaded = sim.load_observations(ranged, "composite")

    # 4 (o1 = 4), 5 (o2 = -3) and 6 (o1 a micro-unit past the edge) go; the edge
    # values and the value inside the 1e-9 tolerance stay.
    assert _runs(loaded) == [1, 2, 3, 7]
    assert loaded.index.tolist() == [0, 1, 2, 3]
    assert loaded["o1"].between(-1.0 - 1e-9, 1.0 + 1e-9).all()
    assert loaded["o2"].between(-1.0 - 1e-9, 1.0 + 1e-9).all()
    err = capsys.readouterr().err
    assert re.search(r"\[toy\] dropping 3 of 7 rows logged outside their column ranges", err), err


def test_a_range_on_a_column_the_objective_does_not_use_still_applies(tmp_path, capsys):
    root = _write_dataset(tmp_path / "data")
    loaded = sim.load_observations(_dataset(root, column_ranges={"aux": (0.0, 5.0)}), "composite")
    assert _runs(loaded) == [1, 2, 3, 4, 5, 6]
    assert "dropping 1 of 7 rows" in capsys.readouterr().err


def test_a_range_on_an_absent_column_is_ignored(tmp_path, capsys):
    root = _write_dataset(tmp_path / "data")
    absent_only = sim.load_observations(_dataset(root, column_ranges={"Nonexistent": (0.0, 1.0)}),
                                        "composite")
    assert _runs(absent_only) == [1, 2, 3, 4, 5, 6, 7]
    assert "dropping" not in capsys.readouterr().err

    # Beside a present one, the absent column neither drops rows nor raises.
    mixed = sim.load_observations(
        _dataset(root, column_ranges={"Nonexistent": (0.0, 1.0), "o1": (-1.0, 1.0)}), "composite")
    assert _runs(mixed) == [1, 2, 3, 5, 7]
    assert "dropping 2 of 7 rows" in capsys.readouterr().err


def test_ranges_apply_after_the_user_filter(tmp_path, capsys):
    root = _write_dataset(tmp_path / "data", user_ids=[1, 1, 1, 2, 2, 2, 2])
    ranged = _dataset(root, column_ranges={"o1": (-1.0, 1.0), "o2": (-1.0, 1.0)})
    loaded = sim.load_observations(ranged, "composite", user_id="2")
    assert _runs(loaded) == [7]
    assert "dropping 3 of 4 rows" in capsys.readouterr().err


def test_a_range_that_excludes_every_row_is_an_error_not_an_empty_frame(tmp_path):
    root = _write_dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="No data remaining"):
        sim.load_observations(_dataset(root, column_ranges={"aux": (100.0, 200.0)}), "composite")


# ---------------------------------------------------------------------------
# parse_dataset_configs
# ---------------------------------------------------------------------------


def _config_file(tmp_path: Path, data_dir: Path, **extra) -> Path:
    entry = {"name": "toy", "data_dir": str(data_dir), "param_columns": PARAMS,
             "objective_map": {"composite": OBJECTIVES}, **extra}
    path = tmp_path / f"datasets-{len(list(tmp_path.glob('datasets-*.json')))}.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    return path


def test_parse_reads_column_ranges_as_float_pairs(tmp_path):
    data = _write_dataset(tmp_path / "data")
    config = _config_file(tmp_path, data, column_ranges={"o1": [-1, 1], "o2": [-3, 3.5]})
    (dataset,) = sim.parse_dataset_configs(None, config, tmp_path / "cache")
    assert dataset.column_ranges == {"o1": (-1.0, 1.0), "o2": (-3.0, 3.5)}
    assert all(isinstance(b, float) for bounds in dataset.column_ranges.values() for b in bounds)

    # ... and the parsed ranges reach load_observations: o2 = -3 is now on the
    # scale, so only the two o1 rows past the edge go.
    assert _runs(sim.load_observations(dataset, "composite")) == [1, 2, 3, 5, 7]


@pytest.mark.parametrize("extra", [{}, {"column_ranges": None}, {"column_ranges": {}}])
def test_parse_without_column_ranges_gives_an_empty_dict(tmp_path, extra):
    data = _write_dataset(tmp_path / "data")
    (dataset,) = sim.parse_dataset_configs(None, _config_file(tmp_path, data, **extra), tmp_path / "cache")
    assert dataset.column_ranges == {}


@pytest.mark.parametrize(
    "column_ranges, match",
    [
        ([["o1", -1, 1]], r"column_ranges must be a dict"),
        ("o1:-1,1", r"column_ranges must be a dict"),
        ({"o1": [-1, 0, 1]}, r"column_ranges\['o1'\] must be \[low, high\]"),
        ({"o1": [1]}, r"column_ranges\['o1'\] must be \[low, high\]"),
        ({"o1": 1.0}, r"column_ranges\['o1'\] must be \[low, high\]"),
        ({"o1": "-1,1"}, r"column_ranges\['o1'\] must be \[low, high\]"),
        ({"o1": {"low": -1, "high": 1}}, r"column_ranges\['o1'\] must be \[low, high\]"),
        ({"o1": ["low", "high"]}, r"could not convert"),
        ({"o1": [1, -1]}, r"column_ranges\['o1'\] has low 1.0 above high -1.0"),
    ],
)
def test_parse_rejects_malformed_column_ranges(tmp_path, column_ranges, match):
    data = _write_dataset(tmp_path / "data")
    config = _config_file(tmp_path, data, column_ranges=column_ranges)
    with pytest.raises(ValueError, match=match):
        sim.parse_dataset_configs(None, config, tmp_path / "cache")
