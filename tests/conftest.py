"""Make ``scripts/`` importable for the whole test session.

The scripts are loaded by path (``importlib.util.spec_from_file_location``)
rather than as a package, so their own sibling imports -- ``_plot_style``,
``boba_benchmarks``, ``bo_sensor_error_simulation`` -- only resolve if
``scripts/`` is on ``sys.path``. Without this, ``test_scientific_correctness.py``
fails at collection with ``ModuleNotFoundError: No module named '_plot_style'``
whenever it runs before a module that happens to insert the path itself, which
made the suite's result depend on collection order.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# And tests/ itself, for the shared helper ``_reference_fixtures`` whatever
# pytest's import mode.
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
