"""Correctness tests for the vendored BOBA benchmark suite and its driver.

The vendored functions are the ground truth for every synthetic result in this
repository, so these tests check *scientific* properties -- agreement with
BOBA's own implementations, attainability of the recorded optima, exactness of
the standardisation -- not just that the code runs.
"""
from __future__ import annotations

import importlib.util
import math
import pickle
import re
import subprocess
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

MODULE_PATH = SCRIPTS / "bo_synthetic_error_simulation.py"
_spec = importlib.util.spec_from_file_location("bo_synth", MODULE_PATH)
bo_synth = importlib.util.module_from_spec(_spec)
sys.modules["bo_synth"] = bo_synth
assert _spec.loader is not None
_spec.loader.exec_module(bo_synth)

BOBA_ROOT = Path("C:/Users/markc/Desktop/BOBA")
HAS_BOBA = (BOBA_ROOT / "bayes_opt" / "simulation.py").exists()

ANALYTIC = [name for name in bb.BOBA_ORDER if bb.BENCHMARKS[name].kind != "stochastic"]


# ---------------------------------------------------------------------------
# Agreement with BOBA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_BOBA, reason="BOBA checkout not available on this machine")
@pytest.mark.parametrize("name", ANALYTIC)
def test_matches_boba_reference_implementation(name: str) -> None:
    """The vendored numpy copy must agree with BOBA's torch original.

    Tolerance is 1e-9 relative: the two evaluate the same closed form in the
    same double precision, so anything beyond floating-point reassociation
    means the copy has drifted.
    """
    import torch

    if str(BOBA_ROOT) not in sys.path:
        sys.path.insert(0, str(BOBA_ROOT))
    from bayes_opt import simulation as boba_sim

    spec = bb.BENCHMARKS[name]
    reference = getattr(boba_sim, name)
    rng = np.random.default_rng(20260906)
    X = spec.lo + rng.random((40, spec.dim)) * (spec.hi - spec.lo)

    mine = bb.evaluate(name, X)
    theirs = np.array([float(reference(torch.tensor(row, dtype=torch.double))) for row in X])
    denom = np.maximum(np.abs(theirs), 1.0)
    assert np.max(np.abs(mine - theirs) / denom) < 1e-9


@pytest.mark.skipif(not HAS_BOBA, reason="BOBA checkout not available on this machine")
def test_registry_matches_boba_tables() -> None:
    """dims, boxes and recorded optima must match BOBA's parallel_main.py.

    Read out of the source rather than imported: importing parallel_main pulls
    in the whole BOBA harness.
    """
    source = (BOBA_ROOT / "parallel_main.py").read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    for table in ("SIMULATION_FUNCTIONS", "BOUNDS", "DIMS", "Y_BEST"):
        match = re.search(rf"^{table}\s*=\s*\[", source, flags=re.MULTILINE)
        assert match is not None, f"{table} not found in BOBA/parallel_main.py"
        start = match.start()
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

    names = namespace["SIMULATION_FUNCTIONS"]
    bounds = namespace["BOUNDS"]
    dims = namespace["DIMS"]
    y_best = namespace["Y_BEST"]

    assert list(names) == bb.BOBA_ORDER
    for name, box, spatial_dim, best in zip(names, bounds, dims, y_best):
        spec = bb.BENCHMARKS[name]
        assert spec.dim == spatial_dim + 1, f"{name}: dim"
        assert spec.lo == box[0] and spec.hi == box[1], f"{name}: box"
        if name == "power_law_practice":
            # Deliberately corrected -- see the spec's comment. Assert BOTH that
            # we diverge and that BOBA's value is exactly the stale n_max=200
            # supremum, so this test fails loudly if BOBA ever fixes it.
            assert not math.isclose(spec.y_best, best, rel_tol=1e-6)
            assert math.isclose(best, -4 * 201**-0.55, rel_tol=1e-5)
        else:
            assert math.isclose(spec.y_best, best, rel_tol=1e-9, abs_tol=1e-12), f"{name}: y_best"


# ---------------------------------------------------------------------------
# The functions themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ANALYTIC)
def test_vectorised_matches_pointwise(name: str) -> None:
    spec = bb.BENCHMARKS[name]
    rng = np.random.default_rng(11)
    X = spec.lo + rng.random((17, spec.dim)) * (spec.hi - spec.lo)
    batched = bb.evaluate(name, X)
    pointwise = np.array([bb.evaluate(name, row)[0] for row in X])
    assert np.allclose(batched, pointwise, rtol=0, atol=1e-12)
    assert batched.shape == (17,)


@pytest.mark.parametrize("name", ANALYTIC)
def test_wrong_dimension_is_rejected(name: str) -> None:
    spec = bb.BENCHMARKS[name]
    with pytest.raises(ValueError):
        bb.evaluate(name, np.zeros((3, spec.dim + 1)))


@pytest.mark.parametrize(
    "name,point,expected",
    [
        ("ackley", 0.0, 0.0),
        ("griewank", 0.0, 0.0),
        ("rastrigin", 0.0, 0.0),
        ("rosenbrock", 1.0, 0.0),
        ("levy_10", 1.0, 0.0),
        ("yerkes_dodson", 0.5, 1.0),
        ("stevens", 1.0, 1.0),
        ("hicks_law", 0.0, -0.2),
        ("weber_fechner", 1.0, math.log(101.0)),
    ],
)
def test_value_at_known_argmax(name: str, point: float, expected: float) -> None:
    """Functions whose argmax is a constant vector: evaluate there exactly."""
    spec = bb.BENCHMARKS[name]
    value = bb.evaluate(name, np.full((1, spec.dim), point))[0]
    assert value == pytest.approx(expected, abs=1e-9)
    assert value == pytest.approx(spec.y_best, abs=1e-6)


def test_power_law_practice_optimum_is_the_corrected_value() -> None:
    """Regression test for BOBA's stale optimum.

    log f is concave in mean(x), so the extremum is an endpoint; t always wants
    to be 1. The supremum is therefore -B(1) * N(1)**-alpha(1) = -4 * 31**-0.55.
    """
    supremum = -4 * 31**-0.55
    assert bb.BENCHMARKS["power_law_practice"].y_best == pytest.approx(supremum, rel=1e-12)
    corner = bb.evaluate("power_law_practice", np.array([[1.0, 1.0, 1.0, 1.0]]))[0]
    assert corner == pytest.approx(supremum, rel=1e-12)

    rng = np.random.default_rng(3)
    X = rng.random((20000, 4))
    assert bb.evaluate("power_law_practice", X).max() <= supremum + 1e-12


@pytest.mark.parametrize("name", ANALYTIC)
def test_no_sample_exceeds_the_recorded_optimum(name: str) -> None:
    """A y_opt below the attainable maximum would make regret negative."""
    spec = bb.BENCHMARKS[name]
    X = bb.sobol_sample(spec, log2_n=12, seed=99)
    stats = bb.load_stats()
    y_opt = stats[name]["y_opt"] if name in stats else spec.y_best
    assert bb.evaluate(name, X).max() <= y_opt + 1e-8


@pytest.mark.parametrize("name", ANALYTIC)
def test_verified_optimum_is_at_least_the_recorded_one(name: str) -> None:
    stats = bb.load_stats()
    entry = stats[name]
    assert entry["y_opt"] >= entry["y_best"] - 1e-9
    assert np.isfinite(entry["std"]) and entry["std"] > 0


def test_default_suite_excludes_the_stochastic_arm() -> None:
    assert "typing" in bb.BENCHMARKS
    assert "typing" not in bb.DEFAULT_SUITE
    assert bb.BENCHMARKS["typing"].kind == "stochastic"
    assert len(bb.DEFAULT_SUITE) == len(bb.BOBA_ORDER) - 1


def test_sobol_sample_is_deterministic_and_in_box() -> None:
    spec = bb.BENCHMARKS["hartmann_6"]
    a = bb.sobol_sample(spec, log2_n=8, seed=5)
    b = bb.sobol_sample(spec, log2_n=8, seed=5)
    assert np.array_equal(a, b)
    assert a.min() >= spec.lo and a.max() <= spec.hi
    assert a.shape == (256, spec.dim)


# ---------------------------------------------------------------------------
# The extension families
# ---------------------------------------------------------------------------


def test_extensions_are_separate_from_the_boba_suite() -> None:
    """They must never leak into a run that claims to be 'the BOBA suite'."""
    assert bb.EXTENSION_ORDER
    assert not set(bb.EXTENSION_ORDER) & set(bb.BOBA_ORDER)
    assert not set(bb.EXTENSION_ORDER) & set(bb.DEFAULT_SUITE)
    assert bb.ALL_ORDER == bb.BOBA_ORDER + bb.EXTENSION_ORDER


def test_levy_ladder_is_the_same_function_at_three_dimensions() -> None:
    """levy_4d / levy_7d / levy_10 must be one family, so dim is a manipulation."""
    for name, dim in (("levy_4d", 4), ("levy_7d", 7), ("levy_10", 11)):
        spec = bb.BENCHMARKS[name]
        assert (spec.dim, spec.lo, spec.hi) == (dim, -10.0, 10.0)
        assert bb.evaluate(name, np.ones((1, dim)))[0] == pytest.approx(0.0, abs=1e-12)


def test_bump_family_decouples_signal_strength_from_sparsity() -> None:
    """Amplitude must move opt_z; width must move sparsity.

    This family exists only to break the opt_z/sparsity/skew confound that the
    real benchmarks cannot break, so if it fails to decouple them it is worse
    than useless -- it looks like a manipulation and is not one.
    """
    stats = bb.load_stats()
    # The 2x2 grid only; the matched dimension ladder also starts with "bump_"
    # and is a different manipulation (see test_dimension_ladder_*).
    names = [n for n in bb.EXTENSION_ORDER if n.startswith("bump_a")]
    assert len(names) == 4
    narrow_low = stats["bump_a4_w0.05"]
    narrow_high = stats["bump_a16_w0.05"]
    wide_low = stats["bump_a4_w0.15"]

    # Amplitude x4 at fixed width raises opt_z by at least 3x.
    assert narrow_high["opt_z"] > 3.0 * narrow_low["opt_z"]
    # Width x3 at fixed amplitude leaves opt_z within 30% but raises sparsity.
    assert abs(wide_low["opt_z"] / narrow_low["opt_z"] - 1.0) < 0.30
    # Optimum-anchored sparsity, not sample-anchored: on a needle landscape a
    # Sobol sample never reaches the spike, so the sample-anchored measure
    # reports how broad the BACKGROUND's maxima are and moves the wrong way.
    assert wide_low["sparsity_10pct"] > narrow_low["sparsity_10pct"]


def test_bump_optimum_is_found_despite_being_invisible_to_a_sobol_screen() -> None:
    """A needle landscape needs the argmax hint or y_opt is just the background."""
    spec = bb.BENCHMARKS["bump_a16_w0.05"]
    assert spec.argmax_hint is not None
    screen = bb.evaluate("bump_a16_w0.05", bb.sobol_sample(spec, log2_n=14, seed=3)).max()
    stats = bb.load_stats()["bump_a16_w0.05"]
    assert stats["y_opt"] > screen + 2.0, "the hint did not recover the spike"
    peak = bb.evaluate("bump_a16_w0.05", np.full((1, spec.dim), spec.argmax_hint))[0]
    assert stats["y_opt"] >= peak - 1e-9


def test_one_shot_fragility_is_recorded_for_every_swept_noise_level() -> None:
    """The mediator has to exist at the levels the sweep actually runs."""
    stats = bb.load_stats()
    for name in bb.DEFAULT_SUITE:
        for level in (0.05, 0.25, 1.0, 5.0):
            key = f"frag_{level:g}"
            assert key in stats[name], f"{name} missing {key}"
            assert stats[name][key] >= 0.0


# ---------------------------------------------------------------------------
# The oracle wrapper
# ---------------------------------------------------------------------------


def test_oracle_applies_the_affine_standardisation_exactly() -> None:
    stats = bb.load_stats()
    entry = stats["hartmann_3"]
    oracle = bb.SyntheticOracle.from_stats("hartmann_3", stats)
    rng = np.random.default_rng(4)
    X = rng.random((25, 3))
    raw = bb.evaluate("hartmann_3", X)
    expected = (raw - entry["mean"]) / entry["std"]
    assert np.allclose(oracle.predict_many(X).reshape(-1), expected, rtol=0, atol=1e-12)
    assert np.allclose(oracle.predict(X[0]), expected[:1], rtol=0, atol=1e-12)


def test_oracle_shapes_match_the_fitted_oracle_contract() -> None:
    """run_simulation expects predict -> (n_obj,) and predict_many -> (n, n_obj)."""
    stats = bb.load_stats()
    oracle = bb.SyntheticOracle.from_stats("branin", stats)
    assert oracle.predict(np.array([1.0, 2.0])).shape == (1,)
    assert oracle.predict_many(np.zeros((7, 2))).shape == (7, 1)
    assert oracle.objective_columns == ["value"]
    assert oracle.param_columns == ["x0", "x1"]


def test_oracle_survives_pickling() -> None:
    """Windows multiprocessing uses 'spawn', so every worker payload is pickled."""
    stats = bb.load_stats()
    oracle = bb.SyntheticOracle.from_stats("shekel", stats)
    restored = pickle.loads(pickle.dumps(oracle))
    X = np.full((3, 4), 4.0)
    assert np.array_equal(oracle.predict_many(X), restored.predict_many(X))


@pytest.mark.parametrize("name", ["branin", "rosenbrock", "rastrigin", "michalewicz", "levy_10"])
def test_oracle_tolerates_ulp_excursions_outside_the_box(name: str) -> None:
    """BoTorch-delegated benchmarks raise for out-of-box input, down to the ulp.

    The acquisition optimiser can hand back a corner point a few ulps outside the
    box, and the oracle call is not inside the simulator's fallback try/except,
    so this would abort a whole task mid-sweep.
    """
    stats = bb.load_stats()
    spec = bb.BENCHMARKS[name]
    oracle = bb.SyntheticOracle.from_stats(name, stats)
    corner = np.full(spec.dim, np.nextafter(spec.hi, np.inf))
    value = oracle.predict(corner)
    assert value.shape == (1,)
    assert np.isfinite(value).all()
    # And the raw call it wraps really does raise, so the test is not vacuous.
    with pytest.raises(ValueError):
        bb.evaluate(name, corner.reshape(1, -1))


def test_oracle_rejects_degenerate_scaling() -> None:
    with pytest.raises(ValueError):
        bb.SyntheticOracle("ackley", mean=0.0, std=0.0)
    with pytest.raises(KeyError):
        bb.SyntheticOracle("not_a_benchmark", mean=0.0, std=1.0)


def test_standardised_optimum_matches_the_stats_file() -> None:
    stats = bb.load_stats()
    for name in bb.DEFAULT_SUITE:
        entry = stats[name]
        expected = (entry["y_opt"] - entry["mean"]) / entry["std"]
        assert expected == pytest.approx(entry["opt_z"], rel=1e-12)


def test_landscape_stats_are_reproducible() -> None:
    a = bb.landscape_stats("branin", log2_n=10, seed=7)
    b = bb.landscape_stats("branin", log2_n=10, seed=7)
    assert a == b
    assert a["std"] > 0
    assert 0.0 <= a["sparsity_10pct"] <= 1.0


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def test_cli_rejects_unknown_and_multi_objective_acquisitions() -> None:
    with pytest.raises(ValueError, match="acquisition"):
        bo_synth.main(["--functions", "branin", "--acq", "qehvi", "--dry-run"])
    with pytest.raises(ValueError, match="benchmark"):
        bo_synth.main(["--functions", "not_a_function", "--dry-run"])


def test_cli_rejects_onsets_outside_the_run() -> None:
    with pytest.raises(ValueError, match="jitter-iterations"):
        bo_synth.main(["--functions", "branin", "--acq", "ei", "--iterations", "10",
                       "--jitter-iterations", "10", "--dry-run"])


def test_cli_rejects_too_few_initial_samples() -> None:
    with pytest.raises(ValueError, match="initial-samples"):
        bo_synth.main(["--functions", "branin", "--acq", "ei", "--initial-samples", "1",
                       "--dry-run"])


def test_scaled_bias_ties_the_offset_to_the_swept_magnitude() -> None:
    """'bias' should be a systematic error of the SAME size as 'gaussian'.

    With the fixed-offset behaviour inherited from the data-driven arm, the bias
    term is 0.2 while the sweep runs out to 1.0, so the two error models become
    nearly indistinguishable at the top of the sweep.
    """
    args = bo_synth.parse_args(["--error-bias-mode", "scaled"])
    assert args.error_bias_mode == "scaled"
    assert bo_synth._variant_suffix(args, "bias", 0.25, 0.5) == "_bias0.25"
    fixed = bo_synth.parse_args(["--error-bias-mode", "fixed"])
    assert fixed.error_bias == 0.2


def test_resolve_clip_modes() -> None:
    stats = bb.load_stats()["branin"]
    assert bo_synth.resolve_clip("none", stats) == (None, None)
    low, high = bo_synth.resolve_clip("sample", stats)
    assert low[0] < 0 < high[0]
    low, high = bo_synth.resolve_clip("-2,3", stats)
    assert (low[0], high[0]) == (-2.0, 3.0)
    with pytest.raises(ValueError):
        bo_synth.resolve_clip("3,-2", stats)


def test_seed_resolution() -> None:
    assert bo_synth.resolve_seeds(bo_synth.parse_args(["--seed", "7", "--num-seeds", "3"])) == [7, 8, 9]
    assert bo_synth.resolve_seeds(bo_synth.parse_args(["--seeds", "1,4,9"])) == [1, 4, 9]
    with pytest.raises(ValueError, match="Duplicate"):
        bo_synth.resolve_seeds(bo_synth.parse_args(["--seeds", "1,1"]))


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One tiny end-to-end sweep, reused by the schema tests below."""
    out = tmp_path_factory.mktemp("boba_smoke")
    bo_synth.main([
        "--functions", "branin,hartmann_3",
        "--acq", "ei,random",
        "--iterations", "8",
        "--initial-samples", "3",
        "--candidate-pool", "64",
        "--acq-raw-samples", "32",
        "--acq-num-restarts", "2",
        "--num-seeds", "2",
        "--error-models", "gaussian,bias",
        "--jitter-stds", "0.5",
        "--jitter-iterations", "0,4",
        "--output-dir", str(out),
        "--n-jobs", "1",
    ])
    return out


def test_smoke_run_writes_the_expected_files(smoke_run: Path) -> None:
    assert (smoke_run / "SWEEP_COMPLETE").exists()
    assert (smoke_run / "bo_synthetic_error_summary.csv").exists()
    assert (smoke_run / "bo_synthetic_error_excess_summary.csv").exists()
    assert (smoke_run / "run_metadata.json").exists()
    # 2 functions x 2 acq x 2 seeds x (1 baseline + 2 errors x 1 std x 2 onsets)
    logs = list(smoke_run.rglob("bo_sensor_error_*_seed*_*.csv"))
    assert len(logs) == 2 * 2 * 2 * (1 + 2 * 1 * 2)


def test_smoke_run_filenames_match_the_evaluation_glob(smoke_run: Path) -> None:
    """evaluate_research_question.py finds runs by this glob and nothing else."""
    for function_dir in [p for p in smoke_run.iterdir() if p.is_dir()]:
        found = sorted(function_dir.glob("bo_sensor_error_*_seed*_*.csv"))
        assert found, f"no logs matched the evaluation glob in {function_dir}"
        assert len(found) == len(list(function_dir.glob("*.csv")))


def test_smoke_run_log_schema_matches_the_data_driven_arm(smoke_run: Path) -> None:
    log = next(smoke_run.rglob("bo_sensor_error_branin_value_ei_seed7_baseline_exact.csv"))
    df = pd.read_csv(log)
    required = {
        "iteration", "objective_true", "objective_observed", "error_applied",
        "error_magnitude", "acquisition", "fit_time_sec", "acq_opt_failed", "seed",
        "run_id", "error_model", "jitter_std", "jitter_iteration", "oracle_model",
        "objective", "param_columns", "y_opt", "best_true_so_far", "regret_inst_true",
        "regret_cum_true", "simple_regret_true", "regret_avg_true",
        "inference_value_true", "inference_simple_regret_true", "dataset",
    }
    assert required <= set(df.columns)
    assert df["param_columns"].iloc[0] == "x0,x1"
    assert set(df["param_columns"].iloc[0].split(",")) <= set(df.columns)
    assert df["oracle_model"].unique().tolist() == ["exact"]
    assert df["dataset"].unique().tolist() == ["branin"]


def test_smoke_run_regret_is_never_negative(smoke_run: Path) -> None:
    """The whole point of a known optimum: regret cannot be beaten."""
    for log in smoke_run.rglob("bo_sensor_error_*_seed*_*.csv"):
        df = pd.read_csv(log)
        assert df["simple_regret_true"].min() >= -1e-8, log.name
        assert df["inference_simple_regret_true"].min() >= -1e-8, log.name


def test_smoke_run_regret_identity(smoke_run: Path) -> None:
    """simple_regret_true == y_opt - running max of the TRUE objective."""
    for log in smoke_run.rglob("bo_sensor_error_*_seed*_*.csv"):
        df = pd.read_csv(log).sort_values("iteration")
        y_opt = float(df["y_opt"].iloc[0])
        expected = y_opt - np.maximum.accumulate(df["objective_true"].to_numpy(dtype=float))
        assert np.allclose(df["simple_regret_true"].to_numpy(dtype=float), expected, atol=1e-10)


def test_smoke_run_baseline_is_noise_free(smoke_run: Path) -> None:
    for log in smoke_run.rglob("*_baseline_exact.csv"):
        df = pd.read_csv(log)
        assert (df["error_magnitude"] == 0).all()
        assert df["error_model"].unique().tolist() == ["none"]
        assert not df["error_applied"].any()
        assert np.allclose(df["objective_true"], df["objective_observed"])


def test_smoke_run_noise_starts_at_the_declared_onset(smoke_run: Path) -> None:
    log = next(smoke_run.rglob("*_jittered_exact_gaussian_jit4_std0.5.csv"))
    df = pd.read_csv(log).sort_values("iteration")
    onset = int(df["jitter_iteration"].iloc[0])
    before = df[df["iteration"] <= onset]
    after = df[df["iteration"] > onset]
    assert (before["error_magnitude"] == 0).all()
    assert (after["error_magnitude"] != 0).any()


def test_smoke_run_is_deterministic(smoke_run: Path, tmp_path: Path) -> None:
    """Same seeds, same everything -- a rerun must reproduce the logs bit for bit."""
    repeat = tmp_path / "repeat"
    bo_synth.main([
        "--functions", "branin",
        "--acq", "ei",
        "--iterations", "8",
        "--initial-samples", "3",
        "--candidate-pool", "64",
        "--acq-raw-samples", "32",
        "--acq-num-restarts", "2",
        "--num-seeds", "1",
        "--error-models", "gaussian",
        "--jitter-stds", "0.5",
        "--jitter-iterations", "0",
        "--output-dir", str(repeat),
        "--n-jobs", "1",
    ])
    columns = ["iteration", "x0", "x1", "objective_true", "objective_observed"]
    original = pd.read_csv(
        next(smoke_run.rglob("bo_sensor_error_branin_value_ei_seed7_jittered_exact_gaussian_jit0_std0.5.csv"))
    )[columns]
    rerun = pd.read_csv(
        next(repeat.rglob("bo_sensor_error_branin_value_ei_seed7_jittered_exact_gaussian_jit0_std0.5.csv"))
    )[columns]
    pd.testing.assert_frame_equal(original, rerun)


def test_common_random_numbers_across_benchmarks(smoke_run: Path) -> None:
    """The injected error sequence must not depend on which landscape it hits.

    Standardised observations plus a benchmark-independent jitter stream means a
    cross-function contrast differs by geometry, not by noise realisation.
    """
    a = pd.read_csv(
        next(smoke_run.rglob("bo_sensor_error_branin_value_random_seed7_jittered_exact_gaussian_jit0_std0.5.csv"))
    )
    b = pd.read_csv(
        next(smoke_run.rglob("bo_sensor_error_hartmann_3_value_random_seed7_jittered_exact_gaussian_jit0_std0.5.csv"))
    )
    assert np.allclose(a["error_magnitude"], b["error_magnitude"], atol=1e-12)


def test_excess_summary_pairs_every_jittered_run(smoke_run: Path) -> None:
    summary = pd.read_csv(smoke_run / "bo_synthetic_error_summary.csv")
    excess = pd.read_csv(smoke_run / "bo_synthetic_error_excess_summary.csv")
    jittered = summary[~summary["baseline"]]
    assert len(excess) == len(jittered)
    assert excess["auc_simple_regret_excess_true"].notna().all()


def test_resume_reuses_existing_runs(smoke_run: Path) -> None:
    """--resume must not re-simulate, and must not change the logs."""
    log = next(smoke_run.rglob("bo_sensor_error_branin_value_ei_seed7_baseline_exact.csv"))
    before = pd.read_csv(log)
    bo_synth.main([
        "--functions", "branin,hartmann_3",
        "--acq", "ei,random",
        "--iterations", "8",
        "--initial-samples", "3",
        "--candidate-pool", "64",
        "--acq-raw-samples", "32",
        "--acq-num-restarts", "2",
        "--num-seeds", "2",
        "--error-models", "gaussian,bias",
        "--jitter-stds", "0.5",
        "--jitter-iterations", "0,4",
        "--output-dir", str(smoke_run),
        "--n-jobs", "1",
        "--resume",
    ])
    pd.testing.assert_frame_equal(before, pd.read_csv(log))


def test_evaluation_consumes_the_synthetic_output(smoke_run: Path, tmp_path: Path) -> None:
    """The whole point of matching the schema: the existing evaluator just works."""
    function_dir = smoke_run / "branin"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "evaluate_research_question.py"),
         "--input-dir", str(function_dir),
         "--output-dir", str(tmp_path / "evaluation")],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "evaluation" / "evaluation_report.txt").exists()


def test_constant_bias_from_the_first_observation_is_free() -> None:
    """A pure offset present in every observation cannot change anything.

    The GP standardises its training targets and the incumbent is a posterior
    mean, so adding the same constant to every observation leaves the
    standardised targets, the acquisition surface and the chosen candidate
    identical -- and regret is computed from TRUE values, which the offset never
    touches. Driven directly with a SHARED jitter stream, so this is the exact
    algebraic claim rather than an inference from noisy condition means: the
    `bias` arm at onset 0 must reproduce the `gaussian` arm run for run.

    It matters because it says what the bias condition is actually testing.
    Systematic mis-calibration of a rater costs nothing while it is uniform; the
    cost appears only when it ARRIVES mid-run, which is the onset-20 cell.
    """
    import dataclasses

    import torch

    stats = bb.load_stats()
    oracle = bb.SyntheticOracle.from_stats("hartmann_3", stats)
    spec = bb.BENCHMARKS["hartmann_3"]
    bounds = bo_synth.sim.Bounds(low=spec.bounds_low, high=spec.bounds_high)
    acq = bo_synth.sim.AcquisitionConfig(name="logei")
    y_opt = stats["hartmann_3"]["opt_z"]

    base = bo_synth.sim.SimulationConfig(
        iterations=12, jitter_iteration=0, jitter_std=0.5, single_error=False,
        initial_samples=3, candidate_pool=64, objective="value",
        objective_columns=["value"], param_columns=spec.param_columns, seed=7,
        error_model="gaussian", error_bias=0.5, error_spike_prob=0.1,
        error_spike_std=0.5, dropout_strategy="hold_last", normalize_objective=False,
        objective_weights=None, acq_num_restarts=2, acq_raw_samples=32,
        acq_maxiter=50, acq_mc_samples=32, ref_point=None,
    )

    frames = {}
    for model in ("gaussian", "bias"):
        rng = np.random.default_rng(7)
        torch.manual_seed(7)
        frames[model] = bo_synth.sim.run_simulation(
            oracle=oracle, bounds=bounds,
            config=dataclasses.replace(base, error_model=model),
            acq=acq, rng=rng,
            jitter_rng=np.random.default_rng(1234),   # the SAME stream for both
            run_id="x", apply_error=True, oracle_model="exact", y_opt=y_opt,
        )

    # The first three model-based candidates must be identical: the GP has seen
    # nothing but a shifted copy of the same data.
    X = {k: v[spec.param_columns].to_numpy(dtype=float) for k, v in frames.items()}
    per_iteration = np.max(np.abs(X["gaussian"] - X["bias"]), axis=1)
    prefix = base.initial_samples + 3                 # initial design + 3 model-based
    assert np.max(per_iteration[:prefix]) < 1e-9

    # And over that prefix -- where the two runs are still evaluating the same
    # designs -- the observations differ by exactly the offset, so the test is
    # not vacuous.
    offset = (frames["bias"]["objective_observed"] - frames["gaussian"]["objective_observed"])
    assert np.allclose(offset.to_numpy()[:prefix], 0.5, atol=1e-9)

    # Exact equality for the whole run does NOT hold, and the reason matters for
    # reading the bias-at-onset-0 cell. Standardize subtracts the empirical mean,
    # and (y + b) - mean(y + b) is not bit-identical to y - mean(y); that ~1e-16
    # difference is amplified by the acquisition optimiser at roughly an order of
    # magnitude per iteration. So the arm is not an exact replicate of gaussian --
    # it diverges chaotically, in no particular direction. A difference between
    # the two conditions in the sweep is therefore expected noise, and only a
    # systematic difference across many seeds would mean the offset costs
    # something.
    assert np.max(per_iteration) < 1e-2


# ---------------------------------------------------------------------------
# The known-noise arm
# ---------------------------------------------------------------------------


def _config(**overrides):
    spec = bb.BENCHMARKS["hartmann_3"]
    defaults = dict(
        iterations=12, jitter_iteration=3, jitter_std=0.7, single_error=False,
        initial_samples=3, candidate_pool=64, objective="value",
        objective_columns=["value"], param_columns=spec.param_columns, seed=7,
        error_model="gaussian", error_bias=0.2, error_spike_prob=0.1,
        error_spike_std=0.5, dropout_strategy="hold_last", normalize_objective=False,
        objective_weights=None, acq_num_restarts=2, acq_raw_samples=32,
        acq_maxiter=50, acq_mc_samples=32, ref_point=None,
    )
    defaults.update(overrides)
    return bo_synth.sim.SimulationConfig(**defaults)


def test_known_noise_schedule_matches_the_injected_error() -> None:
    """The declared variance has to be the variance actually injected."""
    known = bo_synth.sim.known_noise_variance
    floor = 1e-6

    gaussian = _config(error_model="gaussian")
    assert known(1, gaussian, True) == floor          # before the onset
    assert known(3, gaussian, True) == floor          # the onset iteration itself
    assert known(4, gaussian, True) == pytest.approx(0.49)   # 0.7**2
    assert known(4, gaussian, False) == floor         # a baseline run is exact

    # bias and drift move the mean, not the variance.
    for model in ("bias", "drift"):
        assert known(9, _config(error_model=model), True) == pytest.approx(0.49)
    # ar1 is stationary at jitter_std by construction.
    assert known(9, _config(error_model="ar1"), True) == pytest.approx(0.49)
    # spike adds an independent component with probability p.
    spike = _config(error_model="spike", error_spike_prob=0.1, error_spike_std=2.0)
    assert known(9, spike, True) == pytest.approx(0.49 + 0.1 * 4.0)
    # single_error means exactly one contaminated iteration.
    single = _config(single_error=True)
    assert known(4, single, True) == pytest.approx(0.49)
    assert known(5, single, True) == floor
    # dropout has no honest variance to declare.
    with pytest.raises(ValueError, match="dropout"):
        known(9, _config(error_model="dropout"), True)


def test_known_noise_schedule_is_empirically_right() -> None:
    """Cross-check the schedule against the errors apply_sensor_error produces."""
    config = _config(error_model="gaussian", iterations=400, jitter_iteration=3,
                     jitter_std=0.7)
    rng = np.random.default_rng(0)
    errors, previous = [], np.zeros(1)
    for iteration in range(1, 401):
        observed, error = bo_synth.sim.apply_sensor_error(
            np.zeros(1), iteration, config, rng, previous, None)
        previous = observed
        if bo_synth.sim.known_noise_variance(iteration, config, True) > 1e-6:
            errors.append(float(error[0]))
    declared = np.sqrt(0.49)
    assert np.std(errors, ddof=1) == pytest.approx(declared, rel=0.12)


def test_known_noise_leaves_default_behaviour_untouched() -> None:
    """The flag must be inert when off: every result to date used 'learned'."""
    assert _config().observation_noise == "learned"
    assert bo_synth.parse_args([]).observation_noise == "learned"
    # A run with the flag explicitly at its default must reproduce one without it.
    import dataclasses

    import torch

    stats = bb.load_stats()
    oracle = bb.SyntheticOracle.from_stats("hartmann_3", stats)
    spec = bb.BENCHMARKS["hartmann_3"]
    bounds = bo_synth.sim.Bounds(low=spec.bounds_low, high=spec.bounds_high)
    frames = []
    for explicit in (False, True):
        config = _config(observation_noise="learned") if explicit else _config()
        rng = np.random.default_rng(7)
        torch.manual_seed(7)
        frames.append(bo_synth.sim.run_simulation(
            oracle=oracle, bounds=bounds, config=config,
            acq=bo_synth.sim.AcquisitionConfig(name="logei"), rng=rng,
            jitter_rng=np.random.default_rng(5), run_id="x", apply_error=True,
            oracle_model="exact", y_opt=stats["hartmann_3"]["opt_z"]))
    # fit_time_sec is wall-clock and cannot match; everything else must.
    columns = [c for c in frames[0].columns if c != "fit_time_sec"]
    pd.testing.assert_frame_equal(frames[0][columns], frames[1][columns])


def test_known_noise_changes_the_run_and_is_logged() -> None:
    import dataclasses

    import torch

    stats = bb.load_stats()
    oracle = bb.SyntheticOracle.from_stats("hartmann_3", stats)
    spec = bb.BENCHMARKS["hartmann_3"]
    bounds = bo_synth.sim.Bounds(low=spec.bounds_low, high=spec.bounds_high)
    out = {}
    for mode in ("learned", "known"):
        rng = np.random.default_rng(7)
        torch.manual_seed(7)
        out[mode] = bo_synth.sim.run_simulation(
            oracle=oracle, bounds=bounds, config=_config(observation_noise=mode),
            acq=bo_synth.sim.AcquisitionConfig(name="logei"), rng=rng,
            jitter_rng=np.random.default_rng(5), run_id="x", apply_error=True,
            oracle_model="exact", y_opt=stats["hartmann_3"]["opt_z"])
    assert out["known"]["observation_noise"].unique().tolist() == ["known"]
    assert out["known"]["known_noise_var"].iloc[0] == pytest.approx(1e-6)
    assert out["known"]["known_noise_var"].iloc[-1] == pytest.approx(0.49)
    # Same initial design (it is drawn before any GP is fitted), different
    # trajectory afterwards -- otherwise the flag is doing nothing.
    initial = spec.param_columns
    pd.testing.assert_frame_equal(out["learned"][initial].head(3),
                                  out["known"][initial].head(3))
    assert not np.allclose(out["learned"][initial].to_numpy(),
                           out["known"][initial].to_numpy())


def test_arm_flags_reach_the_filename() -> None:
    """Resume matches on filename, so an arm that is not named would be reused."""
    args = bo_synth.parse_args(["--observation-noise", "known",
                                "--incumbent", "observed_max"])
    suffix = bo_synth._variant_suffix(args, "gaussian", 0.2, 0.5)
    assert "noise-known" in suffix and "inc-observed_max" in suffix
    default = bo_synth.parse_args([])
    assert bo_synth._variant_suffix(default, "gaussian", 0.2, 0.5) == ""


# ---------------------------------------------------------------------------
# The TabPFN dtype fix in the data-driven arm
# ---------------------------------------------------------------------------


def test_tabpfn_oracle_runs_under_the_float64_default_dtype() -> None:
    """Regression test: the tabpfn oracle used to crash before producing a number.

    bo_sensor_error_simulation sets torch's default dtype to float64 for BoTorch,
    but TabPFN's checkpoints are float32 and it builds its tensors from the
    default dtype -- so every fit/predict raised "mat1 and mat2 must have the
    same dtype". The adapter must fix that AND leave the default dtype alone.
    """
    pytest.importorskip("tabpfn")
    import torch

    module_path = SCRIPTS / "bo_sensor_error_simulation.py"
    spec = importlib.util.spec_from_file_location("bo_sim_tabpfn", module_path)
    bo_sim = importlib.util.module_from_spec(spec)
    sys.modules["bo_sim_tabpfn"] = bo_sim
    assert spec.loader is not None
    spec.loader.exec_module(bo_sim)

    assert torch.get_default_dtype() == torch.float64
    model = bo_sim._build_oracle_model("tabpfn", seed=0, tree_scale=0.5)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() ** 2 + X["b"].to_numpy()
    model.fit(X, y)
    predictions = model.predict(X.iloc[:10])
    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()
    assert torch.get_default_dtype() == torch.float64
    assert isinstance(pickle.loads(pickle.dumps(model)), type(model))
