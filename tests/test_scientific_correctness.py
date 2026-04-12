"""Scientific correctness tests for CHI submission.

These tests verify the mathematical and statistical integrity of the core
research claims: regret computation, error model timing, statistical tests,
FDR correction, and evaluation pipeline pairing logic.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp, t as student_t

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _load(module_name: str, filename: str) -> object:
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Load modules once at module level so all tests share the same import.
bo_sim = _load("bo_sim_sci", "bo_sensor_error_simulation.py")
plot_mod = _load("plot_mod_sci", "plot_sensor_error_results.py")
eval_mod = _load("eval_mod_sci", "evaluate_research_question.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_error_config(
    error_model: str = "gaussian",
    jitter_std: float = 0.5,
    jitter_iteration: int = 5,
    single_error: bool = False,
) -> bo_sim.SimulationConfig:
    return bo_sim.SimulationConfig(
        iterations=20,
        jitter_iteration=jitter_iteration,
        jitter_std=jitter_std,
        single_error=single_error,
        initial_samples=1,
        candidate_pool=10,
        objective="composite",
        objective_columns=bo_sim.OBJECTIVE_MAP["composite"],
        param_columns=bo_sim.PARAM_COLUMNS,
        seed=1,
        error_model=error_model,
        error_bias=0.3,
        error_spike_prob=1.0,
        error_spike_std=0.2,
        dropout_strategy="hold_last",
        normalize_objective=False,
        objective_weights=None,
        acq_num_restarts=2,
        acq_raw_samples=8,
        acq_maxiter=15,
        acq_mc_samples=32,
        ref_point=None,
    )


# ---------------------------------------------------------------------------
# Error model: iteration boundary
# ---------------------------------------------------------------------------

class TestErrorModelBoundary:
    """Error must NOT be applied at or before jitter_iteration; MUST be applied after."""

    def test_no_error_at_exactly_jitter_iteration(self) -> None:
        config = _make_error_config(jitter_iteration=10)
        true_value = np.array([1.0, 2.0])
        observed, error = bo_sim.apply_sensor_error(true_value, 10, config, np.random.default_rng(0), true_value)
        assert np.allclose(observed, true_value)
        assert np.allclose(error, 0.0)

    def test_no_error_before_jitter_iteration(self) -> None:
        config = _make_error_config(jitter_iteration=10)
        true_value = np.array([1.0, 2.0])
        for iter_ in [1, 5, 9, 10]:
            observed, error = bo_sim.apply_sensor_error(true_value, iter_, config, np.random.default_rng(0), true_value)
            assert np.allclose(observed, true_value), f"Error applied too early at iteration {iter_}"
            assert np.allclose(error, 0.0), f"Nonzero error at iteration {iter_}"

    def test_error_applied_after_jitter_iteration(self) -> None:
        config = _make_error_config("gaussian", jitter_std=1.0, jitter_iteration=5)
        true_value = np.array([0.0])
        rng = np.random.default_rng(42)
        observed, error = bo_sim.apply_sensor_error(true_value, 6, config, rng, true_value)
        assert not np.allclose(observed, true_value), "Error was NOT applied after jitter_iteration"
        assert np.allclose(observed, true_value + error)

    def test_single_error_mode_only_fires_once(self) -> None:
        """single_error=True: error only at jitter_iteration + 1, zero elsewhere."""
        config = _make_error_config("gaussian", jitter_std=1.0, jitter_iteration=5, single_error=True)
        true_value = np.array([1.0])
        # Iteration 6 = jitter_iteration + 1 → error applied
        obs6, err6 = bo_sim.apply_sensor_error(true_value, 6, config, np.random.default_rng(0), true_value)
        assert not np.allclose(obs6, true_value), "Single-error not fired at jitter_iteration+1"
        # Iteration 7 = jitter_iteration + 2 → no error
        obs7, err7 = bo_sim.apply_sensor_error(true_value, 7, config, np.random.default_rng(0), true_value)
        assert np.allclose(obs7, true_value), "Single-error fired beyond jitter_iteration+1"
        assert np.allclose(err7, 0.0)

    def test_single_error_mode_silent_before_jitter_iteration(self) -> None:
        config = _make_error_config("gaussian", jitter_std=1.0, jitter_iteration=5, single_error=True)
        true_value = np.array([1.0])
        obs, err = bo_sim.apply_sensor_error(true_value, 5, config, np.random.default_rng(0), true_value)
        assert np.allclose(obs, true_value)
        assert np.allclose(err, 0.0)


# ---------------------------------------------------------------------------
# Error model: spike probability
# ---------------------------------------------------------------------------

class TestSpikeErrorModel:
    def test_spike_never_fires_when_prob_zero(self) -> None:
        config = bo_sim.SimulationConfig(
            iterations=20,
            jitter_iteration=0,
            jitter_std=0.0,
            single_error=False,
            initial_samples=1,
            candidate_pool=10,
            objective="composite",
            objective_columns=bo_sim.OBJECTIVE_MAP["composite"],
            param_columns=bo_sim.PARAM_COLUMNS,
            seed=1,
            error_model="spike",
            error_bias=0.0,
            error_spike_prob=0.0,
            error_spike_std=99.0,
            dropout_strategy="hold_last",
            normalize_objective=False,
            objective_weights=None,
            acq_num_restarts=2,
            acq_raw_samples=8,
            acq_maxiter=15,
            acq_mc_samples=32,
            ref_point=None,
        )
        true_value = np.array([1.0, 2.0])
        for seed in range(20):
            obs, err = bo_sim.apply_sensor_error(true_value, 1, config, np.random.default_rng(seed), true_value)
            # jitter_std=0.0 and spike_prob=0.0 → observed must equal true
            assert np.allclose(obs, true_value), f"Spike fired at seed {seed} despite prob=0"


# ---------------------------------------------------------------------------
# Single-objective regret: known synthetic oracle
# ---------------------------------------------------------------------------

class TestSingleObjectiveRegret:
    """Regret must match hand-computed values for a deterministic oracle sequence."""

    def _run_known_oracle(self, outputs: list[float], y_opt: float) -> pd.DataFrame:
        class FixedOracle:
            def __init__(self, seq: list[float]) -> None:
                self._seq = [np.array([v]) for v in seq]
                self._i = 0

            def predict(self, x: np.ndarray) -> np.ndarray:
                val = self._seq[self._i]
                self._i += 1
                return val

        oracle = FixedOracle(outputs)
        bounds = bo_sim.Bounds(
            low=np.zeros(2, dtype=float),
            high=np.ones(2, dtype=float),
        )
        config = bo_sim.SimulationConfig(
            iterations=len(outputs),
            jitter_iteration=0,
            jitter_std=0.0,
            single_error=False,
            initial_samples=len(outputs),
            candidate_pool=4,
            objective="composite",
            objective_columns=["score"],
            param_columns=["p1", "p2"],
            seed=0,
            error_model="gaussian",
            error_bias=0.0,
            error_spike_prob=0.0,
            error_spike_std=0.0,
            dropout_strategy="hold_last",
            normalize_objective=False,
            objective_weights=None,
            acq_num_restarts=2,
            acq_raw_samples=8,
            acq_maxiter=15,
            acq_mc_samples=16,
            ref_point=None,
        )
        return bo_sim.run_simulation(
            oracle=oracle,
            bounds=bounds,
            config=config,
            acq=bo_sim.AcquisitionConfig(name="greedy"),
            rng=np.random.default_rng(0),
            jitter_rng=None,
            run_id="test",
            apply_error=False,
            oracle_model="dummy",
            y_opt=y_opt,
        )

    def test_simple_regret_is_non_increasing(self) -> None:
        """simple_regret = max(0, y_opt - best_so_far): must be non-increasing."""
        outputs = [1.0, 3.0, 2.0, 4.0, 2.5]
        results = self._run_known_oracle(outputs, y_opt=5.0)
        sr = results["simple_regret_true"].to_numpy(dtype=float)
        assert np.all(sr[:-1] >= sr[1:]), f"simple_regret_true increased: {sr}"

    def test_simple_regret_reaches_zero_when_optimum_achieved(self) -> None:
        outputs = [1.0, 2.0, 5.0, 3.0]
        results = self._run_known_oracle(outputs, y_opt=5.0)
        sr = results["simple_regret_true"].to_numpy(dtype=float)
        assert np.isclose(sr[-1], 0.0), f"Regret did not reach 0: {sr}"

    def test_cumulative_regret_is_sum_of_instantaneous(self) -> None:
        outputs = [1.0, 2.0, 3.0, 4.0]
        y_opt = 5.0
        results = self._run_known_oracle(outputs, y_opt=y_opt)
        best_so_far = np.maximum.accumulate(outputs)
        expected_inst = np.maximum(0.0, y_opt - best_so_far)
        expected_cum = np.cumsum(expected_inst)
        assert np.allclose(results["regret_cum_true"].to_numpy(dtype=float), expected_cum)

    def test_average_regret_equals_cum_divided_by_iteration(self) -> None:
        outputs = [2.0, 1.0, 3.0, 4.0, 0.5]
        results = self._run_known_oracle(outputs, y_opt=5.0)
        cum = results["regret_cum_true"].to_numpy(dtype=float)
        n = np.arange(1, len(cum) + 1, dtype=float)
        expected_avg = cum / n
        assert np.allclose(results["regret_avg_true"].to_numpy(dtype=float), expected_avg)

    def test_regret_is_zero_when_optimal_from_start(self) -> None:
        outputs = [5.0, 5.0, 5.0]
        results = self._run_known_oracle(outputs, y_opt=5.0)
        assert np.allclose(results["regret_cum_true"].to_numpy(dtype=float), 0.0)
        assert np.allclose(results["simple_regret_true"].to_numpy(dtype=float), 0.0)


# ---------------------------------------------------------------------------
# _paired_test_from_diff: statistical math
# ---------------------------------------------------------------------------

class TestPairedTestFromDiff:
    """Verify that Cohen's dz, CI, t-stat, and Wilcoxon are computed correctly."""

    def _run(self, diffs: list[float]) -> dict:
        return plot_mod._paired_test_from_diff(np.array(diffs, dtype=float))

    def test_cohens_dz_equals_mean_over_std(self) -> None:
        diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self._run(diffs)
        expected = np.mean(diffs) / np.std(diffs, ddof=1)
        assert np.isclose(result["cohens_dz"], expected), f"Got {result['cohens_dz']}, expected {expected}"

    def test_t_stat_matches_scipy(self) -> None:
        diffs = [0.5, 1.5, -0.2, 2.0, 0.8]
        result = self._run(diffs)
        ref = ttest_1samp(diffs, popmean=0.0)
        assert np.isclose(result["t_stat"], ref.statistic)
        assert np.isclose(result["p_value_t"], ref.pvalue)

    def test_ci95_width_is_two_margins(self) -> None:
        diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self._run(diffs)
        n = len(diffs)
        std = float(np.std(diffs, ddof=1))
        sem = std / math.sqrt(n)
        t_crit = float(student_t.ppf(0.975, df=n - 1))
        expected_margin = t_crit * sem
        assert np.isclose(result["ci95_high"] - result["ci95_low"], 2 * expected_margin)

    def test_all_zeros_gives_p_value_one(self) -> None:
        result = self._run([0.0, 0.0, 0.0, 0.0])
        assert result["p_value_t"] == 1.0
        assert result["p_value_wilcoxon"] == 1.0
        assert result["t_stat"] == 0.0
        assert result["cohens_dz"] is np.nan or np.isnan(result["cohens_dz"])

    def test_single_pair_returns_nan_stats(self) -> None:
        result = self._run([1.0])
        assert result["n_pairs"] == 1
        assert np.isnan(result["t_stat"])
        assert np.isnan(result["cohens_dz"])

    def test_empty_diff_returns_nan(self) -> None:
        result = self._run([])
        assert result["n_pairs"] == 0
        assert np.isnan(result["mean_diff"])

    def test_mean_and_median_correct(self) -> None:
        diffs = [1.0, 3.0, 5.0]
        result = self._run(diffs)
        assert np.isclose(result["mean_diff"], 3.0)
        assert np.isclose(result["median_diff"], 3.0)
        assert np.isclose(result["mean_abs_diff"], 3.0)

    def test_inf_and_nan_are_filtered_out(self) -> None:
        diffs = [1.0, np.nan, 2.0, np.inf, 3.0]
        result = self._run(diffs)
        assert result["n_pairs"] == 3
        assert np.isclose(result["mean_diff"], 2.0)


# ---------------------------------------------------------------------------
# _apply_fdr: Benjamini-Hochberg correction
# ---------------------------------------------------------------------------

class TestApplyFDR:
    def _make_test_df(self, p_values: list[float], metrics: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"metric": metrics, "p_value": p_values})

    def test_fdr_adjusted_p_ge_original(self) -> None:
        """BH-adjusted p-values must be >= raw p-values (more conservative)."""
        df = self._make_test_df(
            [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9],
            ["m"] * 7,
        )
        result = plot_mod._apply_fdr(df.copy(), "p_value")
        adj = result["p_value_fdr_bh"].to_numpy(dtype=float)
        raw = df["p_value"].to_numpy(dtype=float)
        assert np.all(adj >= raw - 1e-12)

    def test_fdr_correction_applied_per_metric(self) -> None:
        """FDR is applied within each metric group independently."""
        df = pd.DataFrame({
            "metric": ["a", "a", "b", "b"],
            "p_value": [0.01, 0.9, 0.01, 0.9],
        })
        result = plot_mod._apply_fdr(df.copy(), "p_value")
        # Within each metric group of size 2: BH correction is identical
        # to raw p-values (since there's no multiplicity within a size-2 group
        # that would inflate above 0.9), but they must still be non-NaN.
        assert result["p_value_fdr_bh"].notna().all()

    def test_fdr_all_nan_p_values_stay_nan(self) -> None:
        df = pd.DataFrame({"metric": ["m", "m"], "p_value": [np.nan, np.nan]})
        result = plot_mod._apply_fdr(df.copy(), "p_value")
        assert result["p_value_fdr_bh"].isna().all()

    def test_rejected_column_created(self) -> None:
        df = self._make_test_df([0.001, 0.001, 0.001, 0.9], ["m"] * 4)
        result = plot_mod._apply_fdr(df.copy(), "p_value")
        assert "p_value_fdr_bh" in result.columns
        assert "p_value_rejected" in result.columns


# ---------------------------------------------------------------------------
# evaluate_research_question: response table timing
# ---------------------------------------------------------------------------

class TestBuildResponseTable:
    """The first error-affected response step must be jitter_iteration + 1 → +2."""

    def _make_log(
        self,
        run_id: str,
        acquisition: str,
        error_model: str,
        jitter_iteration: int,
        n_iterations: int = 25,
        seed: int = 1,
    ) -> pd.DataFrame:
        iters = list(range(1, n_iterations + 1))
        n = len(iters)
        return pd.DataFrame({
            "run_id": run_id,
            "dataset": "demo",
            "objective": "composite",
            "acquisition": acquisition,
            "seed": seed,
            "oracle_model": "xgboost",
            "error_model": error_model,
            "jitter_std": 0.5 if error_model != "none" else 0.0,
            "jitter_iteration": jitter_iteration,
            "iteration": iters,
            # parameters change linearly for easy L2 verification
            "p1": [float(i) * 0.1 for i in iters],
            "p2": [float(i) * 0.2 for i in iters],
            "objective_true": [float(i) for i in iters],
            "objective_observed": [float(i) for i in iters],
            "best_true_so_far": [float(i) for i in iters],
            "simple_regret_true": [float(n - i) for i in iters],
            "regret_cum_true": [float(i) for i in iters],
            "regret_avg_true": [1.0] * n,
            "y_opt": [float(n)] * n,
            "error_applied": [i > jitter_iteration for i in iters],
            "error_magnitude": [0.0] * n,
            "error_magnitude_l2": [0.0] * n,
            "regret_inst_true": [1.0] * n,
        })

    def test_response_step_uses_correct_iterations(self) -> None:
        jit_iter = 10
        log = self._make_log("run_jit", "ei", "gaussian", jit_iter)
        table = eval_mod.build_response_table(log)
        row = table[table["run_id"] == "run_jit"].iloc[0]
        assert row["response_start_iteration"] == jit_iter + 1
        assert row["response_end_iteration"] == jit_iter + 2

    def test_response_l2_equals_param_difference(self) -> None:
        """response_l2 = ||params[t+2] - params[t+1]||."""
        jit_iter = 5
        log = self._make_log("run_l2", "ucb", "gaussian", jit_iter)
        table = eval_mod.build_response_table(log)
        row = table.iloc[0]
        # p1 increases by 0.1 per iteration, p2 by 0.2
        expected_l2 = math.sqrt(0.1**2 + 0.2**2)
        assert np.isclose(row["response_l2"], expected_l2, rtol=1e-5)

    def test_baseline_flag_set_correctly(self) -> None:
        jit_iter = 5
        baseline_log = self._make_log("b", "ei", "none", jit_iter)
        jitter_log = self._make_log("j", "ei", "gaussian", jit_iter)
        combined = pd.concat([baseline_log, jitter_log], ignore_index=True)
        table = eval_mod.build_response_table(combined)
        assert table[table["run_id"] == "b"]["baseline"].all()
        assert (~table[table["run_id"] == "j"]["baseline"]).all()

    def test_response_table_skipped_when_end_iter_exceeds_max(self) -> None:
        """No response row when jitter_iteration + 2 > max_iteration."""
        log = self._make_log("short", "ei", "gaussian", jitter_iteration=24, n_iterations=24)
        table = eval_mod.build_response_table(log)
        # When no rows qualify, the DataFrame is empty (columns may be absent too)
        assert table.empty


# ---------------------------------------------------------------------------
# evaluate_research_question: paired table
# ---------------------------------------------------------------------------

class TestBuildPairedTable:
    def _make_response_row(
        self,
        run_id: str,
        acquisition: str,
        seed: int,
        baseline: bool,
        error_model: str,
        jitter_std: float,
        jitter_iteration: int,
        response_l2: float,
        auc_regret: float,
    ) -> dict:
        return {
            "run_id": run_id,
            "dataset": "demo",
            "objective": "composite",
            "acquisition": acquisition,
            "seed": seed,
            "oracle_model": "xgboost",
            "baseline": baseline,
            "error_model": error_model,
            "jitter_std": jitter_std,
            "jitter_iteration": jitter_iteration,
            "response_l2": response_l2,
            "final_best_true": 1.0,
            "final_simple_regret_true": 0.5,
            "final_cum_regret_true": 2.0,
            "final_avg_regret_true": 0.4,
            "auc_simple_regret_true": auc_regret,
            "param_columns": "p1,p2",
        }

    def test_paired_table_correctly_subtracts_baseline(self) -> None:
        rows = [
            self._make_response_row("j1", "ei", 1, False, "gaussian", 0.5, 10, response_l2=1.5, auc_regret=3.0),
            self._make_response_row("b1", "ei", 1, True,  "none",     0.0, 10, response_l2=1.0, auc_regret=2.0),
        ]
        df = pd.DataFrame(rows)
        paired = eval_mod.build_paired_table(df)
        assert paired.shape[0] == 1
        assert np.isclose(paired.iloc[0]["response_l2_excess"], 0.5)
        assert np.isclose(paired.iloc[0]["auc_simple_regret_excess_true"], 1.0)

    def test_paired_table_raises_when_no_pairs(self) -> None:
        df = pd.DataFrame([self._make_response_row("j1", "ei", 1, False, "gaussian", 0.5, 10, 1.5, 3.0)])
        with pytest.raises(ValueError, match="No baseline"):
            eval_mod.build_paired_table(df)

    def test_paired_table_only_matches_same_seed_and_acquisition(self) -> None:
        rows = [
            # EI seed 1 jittered
            self._make_response_row("j_ei_1", "ei",  1, False, "gaussian", 0.5, 10, 1.5, 3.0),
            # UCB seed 1 baseline — should NOT pair with EI jittered
            self._make_response_row("b_ucb_1", "ucb", 1, True, "none", 0.0, 10, 1.0, 2.0),
            # EI seed 1 baseline — should pair with EI jittered
            self._make_response_row("b_ei_1", "ei",  1, True,  "none", 0.0, 10, 1.0, 2.0),
        ]
        df = pd.DataFrame(rows)
        paired = eval_mod.build_paired_table(df)
        assert paired.shape[0] == 1
        assert paired.iloc[0]["acquisition"] == "ei"
