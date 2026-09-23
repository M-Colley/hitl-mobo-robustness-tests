"""Reference outputs of code that lives outside this repository (register item D1).

Two kinds of test compare this repository against a source it does not ship:

* the BOBA parity tests (``test_boba_benchmarks.py``) evaluate BOBA's own torch
  implementations of the benchmark functions, and read BOBA's registry tables
  out of ``parallel_main.py``;
* the provenance tests of three simulator builds (``test_acquisition_extensions``,
  ``test_error_extensions``, ``test_mo_halo``) load the simulator as it stood on
  2026-09-14, before those builds, and check that no name moved and that a
  default multi-objective run is unchanged.

Both sources existed on one machine only -- and by 2026-09-23 the 2026-09-14
.py was gone even there -- so those tests skipped everywhere else (the four
provenance tests everywhere). Their reference outputs are now vendored in ``tests/fixtures``
(see each file's ``_header`` for how, when and from what it was produced, and
``tests/fixtures/build_reference_fixtures.py`` to regenerate them). A test
compares against the live source where it is present and against the fixture
where it is not; where the live source is present the fixture is also checked
against it, so a stale fixture fails on the one machine that can notice.

Set ``HITL_REFERENCE_FIXTURES_ONLY=1`` to force the fixture path even where the
live source exists (to exercise it), and ``BOBA_ROOT`` to point at a BOBA
checkout somewhere other than next to this repository.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES_ONLY = os.environ.get("HITL_REFERENCE_FIXTURES_ONLY", "") == "1"

# ---------------------------------------------------------------------------
# BOBA
# ---------------------------------------------------------------------------

# default: a BOBA checkout next to this repository
BOBA_ROOT = Path(os.environ.get("BOBA_ROOT", Path(__file__).resolve().parents[2] / "BOBA"))
HAS_BOBA = (not FIXTURES_ONLY) and (BOBA_ROOT / "bayes_opt" / "simulation.py").exists()
BOBA_FIXTURE = FIXTURES / "boba_reference.json"
BOBA_TABLES = ("SIMULATION_FUNCTIONS", "BOUNDS", "DIMS", "Y_BEST")

# The parity points: one generator per function, 40 uniform points in its box.
PARITY_SEED = 20260906
PARITY_POINTS = 40


def parity_points(spec) -> np.ndarray:
    """The points the parity test evaluates both implementations at."""
    rng = np.random.default_rng(PARITY_SEED)
    return spec.lo + rng.random((PARITY_POINTS, spec.dim)) * (spec.hi - spec.lo)


def points_digest(X: np.ndarray) -> str:
    """sha256 of the points as little-endian float64, to prove they are the fixture's."""
    return hashlib.sha256(np.ascontiguousarray(X, dtype="<f8").tobytes()).hexdigest()


def live_boba_values(name: str, X: np.ndarray) -> np.ndarray:
    """BOBA's own torch implementation of ``name``, evaluated point by point."""
    import torch

    if str(BOBA_ROOT) not in sys.path:
        sys.path.insert(0, str(BOBA_ROOT))
    from bayes_opt import simulation as boba_sim

    reference = getattr(boba_sim, name)
    return np.array([float(reference(torch.tensor(row, dtype=torch.double))) for row in X])


def live_boba_tables() -> dict[str, list]:
    """dims, boxes and recorded optima, read out of BOBA/parallel_main.py.

    Read out of the source rather than imported: importing parallel_main pulls
    in the whole BOBA harness.
    """
    source = (BOBA_ROOT / "parallel_main.py").read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    for table in BOBA_TABLES:
        match = re.search(rf"^{table}\s*=\s*\[", source, flags=re.MULTILINE)
        if match is None:
            raise AssertionError(f"{table} not found in BOBA/parallel_main.py")
        start = end = match.start()
        depth = 0
        for offset, char in enumerate(source[start:], start=start):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end = offset + 1
                    break
        exec(source[start:end], {}, namespace)  # noqa: S102 - fixed local file
    # JSON has no tuples; normalise so live and vendored tables compare equal.
    return json.loads(json.dumps({table: namespace[table] for table in BOBA_TABLES}))


@functools.lru_cache(maxsize=None)
def boba_fixture() -> dict:
    return json.loads(BOBA_FIXTURE.read_text(encoding="utf-8"))


def fixture_boba_values(name: str, X: np.ndarray) -> np.ndarray:
    entry = boba_fixture()["functions"][name]
    assert points_digest(X) == entry["points_sha256"], (
        f"{name}: the parity points are not the ones the fixture was made at "
        "(the vendored box or dimension moved, or the point recipe changed)")
    return np.asarray(entry["values"], dtype=float)


def fixture_boba_tables() -> dict[str, list]:
    return boba_fixture()["tables"]


# ---------------------------------------------------------------------------
# The simulator as it stood on 2026-09-14
# ---------------------------------------------------------------------------

# Where the pre-build copy of the simulator lived. The .py has since been
# deleted; the fixture was produced from the CPython bytecode that survived in
# its __pycache__ (see the fixture's _header).
# The copy lived in a session scratch directory; set HITL_BACKUP_SIM to its path
# to test against it live. Without it the fixture is used.
BACKUP_SIM = Path(os.environ.get("HITL_BACKUP_SIM", "backup_2026-09-14/scripts/bo_sensor_error_simulation.py"))
BACKUP_FIXTURE = FIXTURES / "bo_sim_backup_2026-09-14.json"

# Across machines one BoTorch step can differ in the last digits (BLAS, scipy's
# L-BFGS-B); a behavioural change -- an extra draw, a moved default -- moves the
# run by O(0.1). Where the environment is the one the fixture was made in, the
# comparison is exact, as it was against the live module.
RUN_RTOL = 1e-6
RUN_ATOL = 1e-9


def backup_source_present() -> bool:
    return (not FIXTURES_ONLY) and BACKUP_SIM.is_file()


def load_backup_module(module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, BACKUP_SIM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def backup_fixture() -> dict:
    return json.loads(BACKUP_FIXTURE.read_text(encoding="utf-8"))


class _VendoredBackup(types.SimpleNamespace):
    """The recorded constants of the 2026-09-14 simulator, standing in for the module."""

    def parse_error_models(self, value, default):
        call = backup_fixture()["calls"]["parse_error_models(None, 'all')"]
        assert (value, default) == (None, "all"), "only parse_error_models(None, 'all') was recorded"
        return list(call)


def backup_simulator(module_name: str):
    """The live 2026-09-14 module where its source exists, else its vendored constants.

    Either way the result exposes the ``*_CHOICES`` lists and
    ``parse_error_models(None, "all")``; only the live module can run. A live
    module is also checked against the fixture, the one place a stale fixture
    can be noticed.
    """
    fixture = backup_fixture()
    if backup_source_present():
        module = load_backup_module(module_name)
        for attr, value in fixture["constants"].items():
            assert list(getattr(module, attr)) == value, f"{BACKUP_FIXTURE.name} is stale: {attr}"
        assert list(module.parse_error_models(None, "all")) == fixture["calls"]["parse_error_models(None, 'all')"]
        return module
    return _VendoredBackup(**{k: list(v) for k, v in fixture["constants"].items()})


def is_live(backup) -> bool:
    return isinstance(backup, types.ModuleType)


def environment() -> dict[str, str]:
    """What a floating-point result depends on beyond the code."""
    import botorch
    import gpytorch
    import scipy
    import torch

    return {"python": platform.python_version(), "platform": sys.platform,
            "machine": platform.machine(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "torch": torch.__version__, "gpytorch": gpytorch.__version__,
            "botorch": botorch.__version__}


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "str"


def frame_to_record(frame: pd.DataFrame) -> dict:
    """A DataFrame as JSON: columns, a kind per column, and exact (repr) values."""
    data = {}
    for column in frame.columns:
        kind = _kind(frame[column])
        values = frame[column].tolist()
        if kind == "float":
            values = [None if v != v else float(v) for v in values]  # NaN -> null
        elif kind == "int":
            values = [int(v) for v in values]
        elif kind == "bool":
            values = [bool(v) for v in values]
        else:
            values = [str(v) for v in values]
        data[column] = values
    return {"columns": list(frame.columns), "kinds": {c: _kind(frame[c]) for c in frame.columns},
            "data": data}


def assert_matches_backup_run(frame: pd.DataFrame, key: str = "mo_default_run") -> None:
    """``frame`` against the run the 2026-09-14 simulator produced (fit_time_sec dropped)."""
    fixture = backup_fixture()
    record = fixture["runs"][key]
    exact = environment() == fixture["_header"]["environment"]
    assert list(frame.columns) == record["columns"]
    for column in record["columns"]:
        kind, expected, got = record["kinds"][column], record["data"][column], frame[column]
        assert _kind(got) == kind, f"{column}: {_kind(got)} column, the backup's was {kind}"
        if kind == "float":
            want = np.array([np.nan if v is None else v for v in expected], dtype=float)
            have = got.to_numpy(dtype=float)
            if exact:
                np.testing.assert_array_equal(have, want, err_msg=column)
            else:
                np.testing.assert_allclose(have, want, rtol=RUN_RTOL, atol=RUN_ATOL, equal_nan=True,
                                           err_msg=f"{column} (environment differs from the fixture's)")
        elif kind == "str":
            assert [str(v) for v in got] == expected, column
        else:
            assert got.tolist() == expected, column
