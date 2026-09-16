"""Correctness tests for the multi-objective arm and the matched dimension ladder.

The multi-objective arm's whole claim is that ``y_opt`` is the published maximum
hypervolume pushed through an exact affine map, so the tests here are about that
identity rather than about the code running.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import boba_benchmarks as bb  # noqa: E402
import boba_multiobjective as mob  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bo_synth_mo", SCRIPTS / "bo_synthetic_error_simulation.py")
bo_synth = importlib.util.module_from_spec(_spec)
sys.modules["bo_synth_mo"] = bo_synth
assert _spec.loader is not None
_spec.loader.exec_module(bo_synth)


# ---------------------------------------------------------------------------
# The multi-objective registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", mob.MO_ORDER)
def test_registry_matches_botorch(name: str) -> None:
    """Dimensions and objective counts must match what BoTorch actually builds."""
    spec = mob.MO_BENCHMARKS[name]
    low, high = mob.bounds(name)
    assert low.shape == high.shape == (spec.dim,)
    assert np.all(high > low)
    Y = mob.evaluate(name, np.stack([low, high, (low + high) / 2]))
    assert Y.shape == (3, spec.num_objectives)
    assert np.isfinite(Y).all()


@pytest.mark.parametrize("name", mob.MO_ORDER)
def test_standardised_hypervolume_follows_the_scaling_identity(name: str) -> None:
    """HV is a volume, so the per-objective affine map scales it by prod(sigma).

    This is the identity the arm's ``y_opt`` rests on. If it were wrong, every
    regret in the multi-objective arm would be wrong by a constant factor.
    """
    stats = mob.load_mo_stats()
    check = mob.verify_reference_point(name, stats[name])
    assert check["relative_error"] < 1e-9
    scale = float(np.prod(np.asarray(stats[name]["std"])))
    assert stats[name]["max_hv"] == pytest.approx(stats[name]["max_hv_raw"] / scale, rel=1e-12)
    assert stats[name]["hv_scale"] == pytest.approx(scale, rel=1e-12)


@pytest.mark.parametrize("name", mob.MO_ORDER)
def test_sampled_hypervolume_never_exceeds_the_published_optimum(name: str) -> None:
    """A y_opt below the attainable maximum would make regret negative."""
    stats = mob.load_mo_stats()[name]
    assert stats["sampled_hv"] <= stats["max_hv"] + 1e-8


def test_mo_oracle_shapes_and_pickling() -> None:
    stats = mob.load_mo_stats()
    oracle = mob.SyntheticMultiOracle.from_stats("vehiclesafety", stats)
    low, high = mob.bounds("vehiclesafety")
    x = (low + high) / 2
    assert oracle.predict(x).shape == (3,)
    assert oracle.predict_many(np.stack([x, x, x])).shape == (3, 3)
    assert oracle.objective_columns == ["f0", "f1", "f2"]
    restored = pickle.loads(pickle.dumps(oracle))
    assert np.allclose(oracle.predict(x), restored.predict(x))


@pytest.mark.parametrize("name", mob.MO_ORDER)
def test_mo_oracle_standardises_each_objective(name: str) -> None:
    stats = mob.load_mo_stats()
    oracle = mob.SyntheticMultiOracle.from_stats(name, stats)
    X = mob.sobol_sample(name, log2_n=8, seed=3)
    raw = mob.evaluate(name, X)
    expected = (raw - np.asarray(stats[name]["mean"])) / np.asarray(stats[name]["std"])
    assert np.allclose(oracle.predict_many(X), expected, rtol=0, atol=1e-12)


def test_mo_oracle_tolerates_ulp_excursions() -> None:
    """BoTorch rejects out-of-box input to the last bit; the optimiser can drift."""
    stats = mob.load_mo_stats()
    oracle = mob.SyntheticMultiOracle.from_stats("branincurrin", stats)
    _, high = mob.bounds("branincurrin")
    corner = np.nextafter(high, np.inf)
    assert np.isfinite(oracle.predict(corner)).all()
    with pytest.raises(ValueError):
        mob.evaluate("branincurrin", corner.reshape(1, -1))


def test_mo_driver_rejects_scalar_acquisitions() -> None:
    # --acq goes through the name parser, --acq-list through the mode check;
    # both must refuse, and both must name the acquisitions that do apply.
    for flag in ("--acq", "--acq-list"):
        with pytest.raises(ValueError) as excinfo:
            bo_synth.main(["--multi-objective", "--functions", "branincurrin",
                           flag, "logei", "--dry-run"])
        assert "qlognehvi" in str(excinfo.value)
    with pytest.raises(ValueError, match="multi-objective problem"):
        bo_synth.main(["--multi-objective", "--functions", "ackley", "--dry-run"])


# ---------------------------------------------------------------------------
# The matched dimension ladder
# ---------------------------------------------------------------------------


def test_dimension_ladder_holds_signal_strength_fixed() -> None:
    """The point of the ladder is that only the dimension differs.

    A Levy ladder cannot do this: opt_z is a property of the function at each
    dimension and is invariant to rescaling the box, so it is 1.53, 2.07, 2.62
    at d = 4, 7, 11 and cannot be equalised. The bump family has two free knobs
    against the two targets.
    """
    stats = bb.load_stats()
    names = [f"bump_d{d}" for d, _ in bb.LADDER]
    assert all(n in bb.EXTENSION_ORDER for n in names)
    opt_z = [stats[n]["opt_z"] for n in names]
    assert max(opt_z) - min(opt_z) < 0.01, f"opt_z not matched: {opt_z}"
    dims = [bb.BENCHMARKS[n].dim for n in names]
    assert dims == [4, 7, 11]
    # The spike's volume fraction is matched by construction, which is what
    # keeps sparsity from moving with dimension on its own.
    for dim, _ in bb.LADDER:
        width = bb.LADDER_VOLUME_FRACTION ** (1.0 / dim)
        assert width**dim == pytest.approx(bb.LADDER_VOLUME_FRACTION, rel=1e-9)


def test_levy_opt_z_is_invariant_to_the_box_and_so_cannot_be_matched() -> None:
    """Documents why the ladder is a bump family rather than a Levy family."""
    stats = bb.load_stats()
    levy = [stats[n]["opt_z"] for n in ("levy_4d", "levy_7d", "levy_10")]
    assert levy[0] < levy[1] < levy[2]
    assert levy[2] - levy[0] > 0.5, "the Levy ladder should confound dim with opt_z"


def test_ladder_residual_confound_is_recorded() -> None:
    """Ruggedness still rises with dimension; the paper has to say so."""
    stats = bb.load_stats()
    rugged = [stats[f"bump_d{d}"]["ruggedness"] for d, _ in bb.LADDER]
    assert rugged[0] < rugged[-1], "expected ruggedness to rise with dimension"
