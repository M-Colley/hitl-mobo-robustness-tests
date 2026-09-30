"""The 2026-09-28 fixes to the cross-benchmark synthesis, on data where the answer is known.

* The descriptor table's VIF column must come from the design matrix of the
  model whose coefficients it sits beside (the primary five predictors), with
  the seven-descriptor VIF kept separately.
* The bias-vs-gaussian control must report the paired difference with a
  landscape-clustered SE, since in the sweep the two arms draw independent
  noise streams.
* The opt_z-free mediation rows are refitted on both units.
* The factor variance shares are reported on both responses.
* floor_deployed and friedman_multiplicity compute what Section 6 quotes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyse_boba_robustness as ab  # noqa: E402
import factor_variance_shares as fvs  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "review_checks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ERROR_MODELS = ("gaussian", "bias", "drift", "ar1")
SIGMAS = (0.05, 0.25, 1.0, 5.0)
ONSETS = (0, 20)


def _descriptor_cells(n_landscapes: int = 12, seed: int = 0) -> pd.DataFrame:
    """A balanced landscape x condition panel whose sparsity is nearly log opt_z."""
    rng = np.random.default_rng(seed)
    land = pd.DataFrame({
        "dataset": [f"L{i}" for i in range(n_landscapes)],
        "dim": rng.integers(2, 10, n_landscapes).astype(float),
        "log_opt_z": rng.normal(0.5, 0.5, n_landscapes),
        "ruggedness": rng.normal(0.0, 1.0, n_landscapes),
        "log_tail_ratio": rng.normal(0.0, 1.0, n_landscapes),
    })
    land["log_sparsity"] = -land["log_opt_z"] + rng.normal(0, 0.05, n_landscapes)
    land["skew"] = land["log_tail_ratio"] + rng.normal(0, 0.3, n_landscapes)
    rows = []
    for _, r in land.iterrows():
        for em in ERROR_MODELS:
            for s in SIGMAS:
                for onset in ONSETS:
                    rows.append({**r.to_dict(), "error_model": em, "jitter_std": s,
                                 "jitter_iteration": onset, "log_noise": np.log10(s)})
    cells = pd.DataFrame(rows)
    cells["excess_sd"] = rng.normal(size=len(cells))
    return cells


def test_primary_vif_comes_from_the_primary_design():
    cells = _descriptor_cells()
    primary, full = ab.descriptor_vif(cells)
    p = primary.set_index("term")["vif"]
    f = full.set_index("term")["vif"]
    # The seven-descriptor design carries the sparsity collinearity; the primary one does not.
    assert f["log_opt_z"] > 20 and f["log_sparsity"] > 20
    assert p["log_opt_z"] < 3
    assert "log_sparsity" not in p.index and "skew" not in p.index
    assert set(primary["design"]) == {"primary"} and set(full["design"]) == {"all_descriptors"}
    # The table reads exactly these five terms from the primary file.
    for term in ["log_noise"] + ab.PRIMARY_DESCRIPTORS:
        assert term in p.index
        assert primary.set_index("term").loc[term, "kind"] == "continuous"


def test_indicators_do_not_move_the_continuous_vifs_in_a_balanced_panel():
    cells = _descriptor_cells()
    primary, _ = ab.descriptor_vif(cells)
    alone = ab._vif_table(cells[["log_noise"] + ab.PRIMARY_DESCRIPTORS].astype(float))
    both = primary.set_index("term")["vif"]
    for _, row in alone.iterrows():
        assert both[row["term"]] == pytest.approx(row["vif"], rel=1e-9)


def test_descriptor_design_is_the_fitted_design():
    cells = _descriptor_cells()
    design = ab._descriptor_design(cells, ab.PRIMARY_DESCRIPTORS)
    assert list(design.columns[:5]) == ["log_noise"] + ab.PRIMARY_DESCRIPTORS
    assert np.allclose(design[["log_noise"] + ab.PRIMARY_DESCRIPTORS].mean(), 0.0)
    assert {"error_model_bias", "error_model_drift", "error_model_gaussian", "onset_20"} <= set(design)


def _bias_runs(offsets: dict[str, float], n_acq: int = 3, n_seed: int = 4, seed: int = 1) -> pd.DataFrame:
    """Runs at onset 0 where bias = gaussian + a landscape offset, paired on (landscape, acq, seed)."""
    rng = np.random.default_rng(seed)
    rows = []
    for land, off in offsets.items():
        for a in range(n_acq):
            for s in range(n_seed):
                for std in (0.25, 1.0):
                    base = rng.normal()
                    for em, shift in (("gaussian", 0.0), ("bias", off)):
                        rows.append({"dataset": land, "acquisition": f"a{a}", "seed": s,
                                     "error_model": em, "jitter_std": std, "jitter_iteration": 0,
                                     "excess_sd": base + shift})
    floor = pd.DataFrame(rows).head(4).assign(acquisition="random", excess_sd=0.0)
    return pd.concat([pd.DataFrame(rows), floor], ignore_index=True)


def test_paired_difference_is_clustered_by_landscape():
    offsets = {"A": 0.1, "B": -0.1, "C": 0.3, "D": 0.1}
    runs = _bias_runs(offsets)
    block = runs[(runs["jitter_std"] == 1.0) & (runs["acquisition"] != "random")]
    out = ab._paired_landscape_difference(block, "bias", "gaussian", reps=500, seed=0)
    d = np.array(list(offsets.values()))
    assert out["n_landscapes"] == 4
    assert out["difference_paired"] == pytest.approx(d.mean())
    assert out["difference_se_landscape"] == pytest.approx(d.std(ddof=1) / 2)
    assert out["difference_ci_low"] <= d.mean() <= out["difference_ci_high"]


def test_bias_control_keeps_its_columns_and_adds_the_clustered_se(tmp_path):
    runs = _bias_runs({"A": 0.05, "B": -0.05, "C": 0.0})
    table = ab.bias_onset_control(runs, tmp_path)
    written = pd.read_csv(tmp_path / "bias_onset0_control.csv")
    for col in ("jitter_std", "mean_bias", "mean_gaussian", "std_bias", "std_gaussian",
                "count_bias", "count_gaussian", "difference"):
        assert col in written.columns
    for col in ("difference_paired", "difference_se_landscape", "difference_ci_low",
                "difference_ci_high", "difference_se_independent", "n_landscapes",
                "difference_se_stream", "difference_se_twoway", "n_streams"):
        assert col in written.columns
    # Balanced panel: the mean of landscape means is the pooled difference.
    assert np.allclose(table["difference"], table["difference_paired"])
    # Streams are (acquisition, seed): 3 acquisitions x 4 seeds.
    assert (written["n_streams"] == 12).all()


def test_clustered_se_matches_the_landscape_se_and_sees_a_shared_stream():
    # A balanced panel: 5 landscapes x 6 streams, with a stream effect shared across
    # landscapes (the sweep's jitter seed omits the landscape) and a landscape effect.
    rng = np.random.default_rng(4)
    land = np.repeat(np.arange(5), 6)
    stream = np.tile(np.arange(6), 5)
    values = rng.normal(0, 1, 5)[land] + rng.normal(0, 3, 6)[stream] + rng.normal(0, 0.1, 30)
    landscape_means = pd.Series(values).groupby(land).mean().to_numpy()
    one_way = ab._clustered_mean_se(values, land)
    assert one_way == pytest.approx(landscape_means.std(ddof=1) / np.sqrt(5), rel=1e-9)
    stream_se = ab._clustered_mean_se(values, stream)
    assert stream_se > one_way  # the shared stream carries most of the variance here
    two_way = ab._clustered_mean_se(values, np.column_stack([land, stream]))
    assert np.isfinite(two_way) and two_way > 0
    assert np.isnan(ab._clustered_mean_se(values, np.zeros(30, dtype=int)))


def _mediation_runs(seed: int = 2) -> pd.DataFrame:
    """Learner runs whose cell-mean excess is exactly 2 x frag_at_c (landscape SDs)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(8):
        opt_z = float(np.exp(rng.normal(1.0, 0.8)))
        for em in ERROR_MODELS:
            for s in SIGMAS:
                for onset in ONSETS:
                    frag = float(s * np.exp(rng.normal(0, 0.5)))
                    for acq in ("ei", "ucb"):
                        rows.append({"dataset": f"L{i}", "acquisition": acq, "error_model": em,
                                     "jitter_std": s, "jitter_iteration": onset, "seed": 0,
                                     "excess_sd": 2.0 * frag, "frag_at_c": frag,
                                     "log_noise": np.log10(s), "opt_z": opt_z,
                                     "log_opt_z": np.log10(opt_z)})
    return pd.DataFrame(rows)


def test_mediator_units_refits_only_opt_z_free_rows_on_both_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "BOOTSTRAP_REPS", 150)
    result = ab.mediator_normalised_sensitivity(_mediation_runs(), tmp_path)
    assert (tmp_path / "mediator_model_normalised.csv").is_file()
    assert set(result["response"]) == {"excess_sd", "fragility"}
    models = set(result["model"])
    assert models == {m for m, _, _ in ab.NORMALISED_MEDIATION_MODELS}
    # No descriptor (and no log opt_z) term is ever fitted on the normalised response.
    assert not result["term"].isin(["log_opt_z", "dim", "ruggedness", "log_tail_ratio"]).any()
    r2 = result.groupby(["response", "model"])["r_squared"].first()
    assert r2[("excess_sd", "mediator_only")] == pytest.approx(1.0)
    assert r2[("fragility", "mediator_only")] < 1.0
    flags = result.groupby("model")["opt_z_free"].first()
    assert not flags["mediator_scaled_only"] and flags["mediator_only"]


def test_factor_shares_on_both_units():
    rng = np.random.default_rng(3)
    rows = []
    opt_z = {f"L{i}": float(v) for i, v in enumerate(np.geomspace(0.8, 50, 6))}
    for land, z in opt_z.items():
        for acq in ("ei", "ucb", "random"):
            for em in ERROR_MODELS:
                for s in SIGMAS:
                    for onset in ONSETS:
                        frag = 0.05 * np.log10(s / 0.01) + rng.normal(0, 0.01)
                        rows.append({"dataset": land, "acquisition": acq, "error_model": em,
                                     "jitter_std": s, "jitter_iteration": onset,
                                     "fragility": frag, "excess_sd": frag * z})
    d = pd.DataFrame(rows)
    table = fvs.share_table(d)
    frag = table[(table["response"] == "fragility") & (table["scope"] == "all cells")].set_index("factor")
    sd = table[(table["response"] == "excess_sd") & (table["scope"] == "all cells")].set_index("factor")
    learners = d[d["acquisition"] != "random"]
    expect = fvs.shares(learners, fvs.FACTORS)
    for name, value in expect.items():
        assert frag.loc[name, "variance_share"] == pytest.approx(round(value, 3))
    # A pure opt_z scale difference shows up as landscape variance only in landscape SDs.
    assert sd.loc["landscape", "variance_share"] > frag.loc["landscape", "variance_share"]
    assert frag.loc["magnitude", "variance_share"] > sd.loc["magnitude", "variance_share"]


def test_floor_table_averages_the_two_floors_and_counts_worse_arms():
    fd = _load("floor_deployed")
    rows = []
    for land in ("A", "B", "C"):
        for acq, value in (("random", 0.5), ("sobol", 0.7), ("ei", 0.4), ("pi", 0.65), ("qnei", 0.8)):
            rows.append({"acquisition": acq, "jitter_std": 1.0, "dataset": land,
                         "deployed": value, "deployed_clean": 0.1, "deployed_excess": value - 0.1})
    table = fd.floor_table(pd.DataFrame(rows), reps=200, seed=0).set_index("acquisition")
    assert table.loc["floor", "deployed"] == pytest.approx(0.6)
    assert table.loc["ei", "floor_minus_acq"] == pytest.approx(0.2)
    assert table.loc["qnei", "floor_minus_acq"] == pytest.approx(-0.2)
    assert int(table["acquisitions_worse_than_floor"].iloc[0]) == 2  # pi and qnei


def test_floor_table_p_values_and_holm_over_the_arms():
    fd = _load("floor_deployed")
    # The paper's intervals use 2000 landscape resamples.
    assert fd.REPS == 2000
    rng = np.random.default_rng(5)
    rows = []
    shifts = {"ei": 0.2, "ucb": 0.0, "pi": -0.2}  # floor - arm: + means the arm ships better
    for i in range(20):
        base = rng.normal(0.5, 0.1)
        for acq, value in (("random", base), ("sobol", base)):
            rows.append({"acquisition": acq, "jitter_std": 5.0, "dataset": f"L{i}",
                         "deployed": value, "deployed_clean": 0.3, "deployed_excess": 0.2})
        for acq, shift in shifts.items():
            value = base - shift + rng.normal(0, 0.05)
            rows.append({"acquisition": acq, "jitter_std": 5.0, "dataset": f"L{i}",
                         "deployed": value, "deployed_clean": 0.1, "deployed_excess": value - 0.1})
    table = fd.floor_table(pd.DataFrame(rows), reps=2000, seed=0)
    arms = table[table["acquisition"] != "floor"].set_index("acquisition")
    # Holm never lowers a p-value, and the bootstrap p is floored at 1/reps.
    assert (arms["p_holm"] >= arms["p_bootstrap"] - 1e-12).all()
    assert arms["p_bootstrap"].min() >= 1.0 / 2000
    assert arms.loc["ei", "p_holm"] < 0.05 and arms.loc["pi", "p_holm"] < 0.05
    assert arms.loc["ucb", "p_holm"] > 0.05
    assert (arms["wilcoxon_p_holm"] >= arms["wilcoxon_p"] - 1e-12).all()
    first = table.iloc[0]
    assert int(first["acquisitions_better_than_floor_holm"]) == 1  # ei
    assert int(first["acquisitions_worse_than_floor_holm"]) == 1   # pi


def test_floor_landscape_means_use_opt_z_and_the_four_processes():
    fd = _load("floor_deployed")
    rows = []
    for em in ("gaussian", "bias", "drift", "ar1", "slip"):
        rows.append({"dataset": "A", "acquisition": "ei", "error_model": em, "jitter_std": 1.0,
                     "jitter_iteration": 0, fd.NOISY: 2.0, fd.CLEAN: 1.0, fd.EXCESS: 1.0,
                     fd.SEARCH: 0.4})
    cells = fd.landscape_means(pd.DataFrame(rows), {"A": 4.0}, onset=0)
    assert len(cells) == 1
    assert cells["deployed"].iloc[0] == pytest.approx(0.5)
    assert cells["deployed_clean"].iloc[0] == pytest.approx(0.25)
    # The deployed excess splits into search (best visited) and selection.
    assert cells["search_excess"].iloc[0] == pytest.approx(0.1)
    assert cells["selection_excess"].iloc[0] == pytest.approx(0.15)


def test_floor_table_carries_the_search_selection_split():
    fd = _load("floor_deployed")
    rows = []
    for land in ("A", "B"):
        # The floors visit what their clean twins visit: no search excess.
        for acq, search, sel in (("random", 0.0, 0.3), ("sobol", 0.0, 0.1), ("ei", 0.25, 0.15)):
            rows.append({"acquisition": acq, "jitter_std": 1.0, "dataset": land,
                         "deployed": 0.5, "deployed_clean": 0.1, "deployed_excess": search + sel,
                         "search_excess": search, "selection_excess": sel})
    table = fd.floor_table(pd.DataFrame(rows), reps=100, seed=0).set_index("acquisition")
    assert table.loc["floor", "search_excess"] == 0.0
    assert table.loc["floor", "selection_excess"] == pytest.approx(0.2)
    assert table.loc["ei", "search_excess"] == pytest.approx(0.25)
    lines = fd.format_split(table.reset_index(), onset=0)
    assert any("floor search excess 0.000e+00" in line for line in lines)


def test_friedman_corrections_match_statsmodels():
    fm = _load("friedman_multiplicity")
    p = np.array([1e-10, 1e-4, 0.004, 0.011, 0.03, 0.049, 0.2, 0.6])
    tests = pd.DataFrame({
        "error_model": ["gaussian"] * len(p), "jitter_std": np.arange(len(p), dtype=float),
        "jitter_iteration": 0, "best_acquisition": "ucb", "compared_with": "__friedman__",
        "wilcoxon_p": p, "wilcoxon_p_fdr": p, "kendall_w": 0.2, "n_benchmarks": 20})
    pairwise = tests.iloc[:2].assign(compared_with="ei", wilcoxon_p=0.001)
    rows = fm.corrected(pd.concat([tests, pairwise], ignore_index=True))
    assert len(rows) == len(p)
    assert int(rows["reject_raw"].sum()) == int((p < 0.05).sum())
    assert int(rows["reject_bh"].sum()) == int(multipletests(p, method="fdr_bh")[0].sum())
    assert int(rows["reject_holm"].sum()) == int(multipletests(p, method="holm")[0].sum())
    assert rows["reject_holm"].sum() <= rows["reject_bh"].sum() <= rows["reject_raw"].sum()
