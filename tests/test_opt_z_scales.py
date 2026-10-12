"""A landscape without an opt_z raises instead of staying in raw units.

Until 2026-10-09 decompose_regret, budget_split and bootstrap_coverage divided
by opt_z.get(dataset, 1.0), so a landscape missing from the statistics was
silently pooled in raw rating units, and analyse_fitted_companion read opt_z from
the git-ignored output-boba/run_metadata.json. Each now raises on a missing
scale, and the fitted companion reads boba_benchmarks.load_stats like every
other table. On complete statistics nothing changes (the outputs were checked
byte for byte); these tests pin the refusal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyse_fitted_companion as afc  # noqa: E402
import boba_benchmarks as bb  # noqa: E402
import bootstrap_coverage as bc  # noqa: E402
import budget_split as bs  # noqa: E402


def test_budget_split_refuses_a_landscape_without_opt_z():
    bs.require_opt_z(["branin"], {"branin": 1.25})
    with pytest.raises(KeyError, match="no opt_z for \\['zzz'\\]"):
        bs.require_opt_z(["branin", "zzz"], {"branin": 1.25})
    joined = pd.DataFrame({"dataset": ["zzz"], "file": ["f"], "k": [2], "regret_noisy": [1.0],
                           "ref_noisy": [1.0], "k_hat": [2]})
    with pytest.raises(KeyError, match="no opt_z"):
        bs.score_policies(joined, pd.DataFrame({"file": [], "regret_pm": [], "regret_lcb1": []}), {},
                          np.random.default_rng(0))


def test_bootstrap_coverage_refuses_a_landscape_without_opt_z():
    z = bc.landscape_scale(pd.Series(["branin", "ackley"]), {"branin": 1.25, "ackley": 2.0})
    assert z.tolist() == [1.25, 2.0]
    with pytest.raises(KeyError, match="no opt_z for \\['zzz'\\]"):
        bc.landscape_scale(pd.Series(["branin", "zzz"]), {"branin": 1.25})


def test_the_fitted_companion_takes_opt_z_from_the_landscape_statistics():
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    optz = afc.synthetic_opt_z(["shekel", "branin"])
    assert optz.to_dict() == {"branin": float(stats["branin"]["opt_z"]), "shekel": float(stats["shekel"]["opt_z"])}
    with pytest.raises(KeyError, match="no opt_z for \\['zzz'\\]"):
        afc.synthetic_opt_z(["branin", "zzz"])
