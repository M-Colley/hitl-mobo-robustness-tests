"""Synthetic runner contracts and a four-judgment analytic-extension smoke."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import human_feedback_benchmark as hfb


@pytest.fixture(autouse=True)
def isolate_environment_cache():
    hfb._catalog_environment.cache_clear()
    yield
    hfb._catalog_environment.cache_clear()


def settings(**changes):
    base = hfb.Settings(functions=("ackley",), methods=("random", "sobol"),
                        seeds=(7,), error_models=("gaussian",), magnitudes=(1.0,),
                        onsets=(0,), budget=4, candidate_pool=8,
                        reference="random", reference_rule="observed_max")
    return replace(base, **changes)


def no_work(*args, **kwargs):
    pytest.fail("A planning or analysis-only operation tried to run an experiment")


def synthetic_checkpoint(task, spec, base=0.5, scale=1.0):
    """A complete, internally consistent record with hand-written regrets."""
    trace = [{"budget_spent": i+1, "indices": [i], "feedback": 0., "perceived": [0.],
              "recommended": 0, "simple_regret_true": base if i == 0 else base/2,
              "inference_simple_regret_true": base} for i in range(spec.budget)]
    rows = [{"ship_rule": rule, "final_deployed_regret_true": base,
             "final_search_regret_true": base/2, "final_selection_loss_true": base/2}
            for rule in ("policy", "observed_max", "uniform_ties", "rank_mean")]
    return {"schema": hfb.SCHEMA_VERSION, "task": task, "rows": rows, "trace": trace,
            "opt_z": scale, "y_opt": 1., "catalog_best_true": 1.,
            "n_judgments": spec.budget, "design_exposures": spec.budget,
            "unique_designs": spec.budget, "budget_spent": spec.budget, "seconds": 0.}


def test_dry_run_is_read_only_and_never_constructs_policies_or_oracles(tmp_path, monkeypatch):
    monkeypatch.setattr(hfb.fp, "make_policy", no_work)
    monkeypatch.setattr(hfb.bb.SyntheticOracle, "from_stats", no_work)
    monkeypatch.setattr(hfb, "execute", no_work)
    output = tmp_path / "never-created"
    hfb.main(["--dry-run", "--output-dir", str(output)])
    assert not output.exists()


def test_summary_without_manifest_never_falls_back_to_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(hfb, "execute", no_work)
    with pytest.raises(FileNotFoundError, match="No manifest"):
        hfb.main(["--summarize", "--output-dir", str(tmp_path / "absent")])
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("changes", [
    {"seeds": (7, 7)}, {"magnitudes": (float("nan"),)}, {"budget": 0},
    {"candidate_pool": 1}, {"comparison_cost": 0}, {"onsets": (4,)},
    {"rating_clip": (1, -1)}, {"ar1_rho": 1.0}, {"methods": ("typo",)},
    {"reference": "missing"}, {"policy_options": {"random": 5}},
    {"policy_options": {"random": {"ignored_typo": 5}}},
    {"policy_options": {"sobol": {"n_initial": 5}}},
])
def test_invalid_configs_rejected(changes):
    with pytest.raises(ValueError):
        settings(**changes).validate({"ackley": {"mean": 0., "std": 1., "opt_z": 1.}})


def test_rating_only_budget_does_not_require_an_unused_comparison_cost():
    settings(comparison_cost=10).validate({"ackley": {"mean": 0., "std": 1., "opt_z": 1.}})


def test_tasks_cache_clean_once_and_do_not_depend_on_method_order():
    spec = settings(error_models=("gaussian", "drift"), magnitudes=(1., 5.), onsets=(0, 2))
    tasks = hfb.build_tasks(spec)
    assert sum(t["clean"] for t in tasks) == 2
    assert len(tasks) == 18
    reordered = hfb.build_tasks(replace(spec, methods=spec.methods[::-1]))
    assert {hfb.digest(t) for t in tasks} == {hfb.digest(t) for t in reordered}
    assert hfb.stream_seed(7, "policy:random") != hfb.stream_seed(7, "feedback:gaussian")


def test_noise_streams_do_not_change_catalog_or_model_free_proposals():
    X = hfb.candidate_catalog(2, 16, 7)
    assert np.array_equal(X, hfb.candidate_catalog(2, 16, 7))
    for name in ("random", "sobol"):
        clean = hfb.fp.make_policy(name, X, 17)
        noisy = hfb.fp.make_policy(name, X, 17)
        for i in range(7):
            a, b = clean.ask(), noisy.ask()
            assert a == b
            clean.tell(a, float(i))
            noisy.tell(b, -float(i))


@pytest.mark.parametrize("error", ["bias", "drift", "ar1", "rater", "contrast"])
def test_shared_monotone_faults_preserve_simultaneous_pair_order(error):
    rater = hfb.Rater(settings(), 7, error, 5., 0)
    for spent in range(4):
        feedback, _ = rater.observe(np.array([0.2, 0.8]), spent)
        assert feedback == 0.


def test_onset_and_instrument_clean_twin_are_explicit():
    spec = settings(rating_clip=(-1., 1.), rating_step=0.5)
    rater = hfb.Rater(spec, 7, "bias", 10., 2)
    assert rater.observe(np.array([0.1]), 0)[0] == 0.
    assert rater.observe(np.array([0.1]), 2)[0] == 1.
    clean = hfb.Rater(spec, 7, "bias", 0., 0)
    assert clean.observe(np.array([3.]), 0)[0] == 1.
    pair = hfb.Rater(spec, 7, "bias", 0., 0)
    assert pair.observe(np.array([2., 3.]), 0)[0] == 0.5


def test_ar1_uses_stationary_first_draw_and_innovation_variance():
    spec = settings(ar1_rho=0.8)
    rater = hfb.Rater(spec, 7, "ar1", 2., 0)
    draws = np.random.default_rng(hfb.stream_seed(7, "feedback:ar1")).normal(size=2)
    first = rater.observe(np.array([0.]), 0)[0]
    second = rater.observe(np.array([0.]), 1)[0]
    assert first == pytest.approx(2*draws[0])
    assert second == pytest.approx(0.8*first + 2*0.6*draws[1])


def test_tied_rating_metrics_include_exact_uniform_expectation():
    class Fake:
        def recommend(self):
            return 0
    scored = hfb.score_choices(Fake(), np.array([[0.], [1.]]), [0, 1], [1., 1.], np.array([0., 2.]))
    assert scored["observed_max"] == 0.
    assert scored["uniform_ties"] == 1.
    repeated = hfb.score_choices(Fake(), np.array([[0.], [1.]]), [0, 0, 1], [1., 1., 1.], np.array([0., 2.]))
    assert repeated["uniform_ties"] == 1.  # Uniform over designs, not repeated looks.


def test_manifest_blocks_changed_configuration_and_nonempty_unowned_output(tmp_path):
    output = tmp_path / "run"
    hfb.prepare_output(output, {"budget": 4}, False)
    hfb.prepare_output(output, {"budget": 4}, True)
    with pytest.raises(ValueError, match="changed"):
        hfb.prepare_output(output, {"budget": 5}, True)
    with pytest.raises(ValueError, match="already exists"):
        hfb.prepare_output(output, {"budget": 4}, False)
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "user.txt").write_text("keep")
    with pytest.raises(ValueError, match="nonempty"):
        hfb.prepare_output(unowned, {}, False)
    assert (unowned / "user.txt").read_text() == "keep"


@pytest.mark.parametrize("scale", [1., 100.])
def test_summary_reuses_standard_process_cost_gain_price_and_decomposition(tmp_path, scale):
    # Hand-written regret values, not simulator outputs. Treatment own-twin
    # excess is .5, but its gain over the standard process is only .2.
    spec = settings(functions=("ackley", "powell"))
    hfb.atomic_json(tmp_path / "manifest.json", {"settings": hfb.asdict(spec)})
    (tmp_path / "runs").mkdir()
    for task in hfb.build_tasks(spec):
        base = (0.1 if task["clean"] else 0.5) if task["method"] == "random" else (0.2 if task["clean"] else 0.3)
        payload = synthetic_checkpoint(task, spec, base, scale)
        hfb.atomic_json(hfb.checkpoint_path(tmp_path, task), payload)
    hfb.summarize(tmp_path)
    summary = pd.read_csv(tmp_path / "comparison_summary.csv")
    row = summary[(summary.method == "sobol") & (summary.response == "final_deployed_regret_true")].iloc[0]
    assert row.cost == pytest.approx(.4/scale)
    assert row.gain == pytest.approx(.2/scale)
    assert row.price == pytest.approx(.1/scale)
    if scale == 1.:
        assert row.recovered == pytest.approx(.5)
    else:
        assert pd.isna(row.recovered)
        per_landscape = pd.read_csv(tmp_path / "comparison_per_landscape.csv")
        assert per_landscape.recovered.isna().all()
    assert row.n_landscapes == 2
    assert "wilcoxon_p" not in summary
    per_run = pd.read_csv(tmp_path / "per_run_metrics.csv")
    assert np.allclose(per_run.final_deployed_regret_true_jitter,
                       per_run.final_search_regret_true_jitter + per_run.final_selection_loss_true_jitter)


def test_summary_rejects_missing_checkpoint_without_running_anything(tmp_path, monkeypatch):
    hfb.atomic_json(tmp_path / "manifest.json", {"settings": hfb.asdict(settings())})
    monkeypatch.setattr(hfb, "execute", no_work)
    with pytest.raises(FileNotFoundError, match="missing checkpoints"):
        hfb.summarize(tmp_path)


def test_checkpoint_rejects_wrong_identity(tmp_path):
    task = hfb.build_tasks(settings())[0]
    (tmp_path / "runs").mkdir()
    hfb.atomic_json(hfb.checkpoint_path(tmp_path, task), {"task": {**task, "seed": 999}, "schema": 1})
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        hfb.read_checkpoint(tmp_path, task)


@pytest.mark.parametrize("damage", [
    lambda r: r["trace"].pop(),
    lambda r: r.__setitem__("n_judgments", 1),
    lambda r: r.__setitem__("unique_designs", 1),
    lambda r: r.__setitem__("opt_z", 0.),
    lambda r: r.__setitem__("seconds", -1.),
    lambda r: r["trace"][0].__setitem__("indices", [999]),
    lambda r: r["trace"][1].__setitem__("recommended", 7),
    lambda r: r["trace"][1].__setitem__("budget_spent", 4),
    lambda r: r["trace"][1].__setitem__("feedback", 2.),
    lambda r: r["rows"].pop(),
    lambda r: r["rows"][0].__setitem__("final_selection_loss_true", -1.),
    lambda r: r["rows"][0].__setitem__("final_deployed_regret_true", 0.),
])
def test_checkpoint_rejects_incomplete_or_inconsistent_metrics(tmp_path, damage):
    spec = settings()
    task = hfb.build_tasks(spec)[0]
    (tmp_path / "runs").mkdir()
    record = synthetic_checkpoint(task, spec)
    damage(record)
    hfb.atomic_json(hfb.checkpoint_path(tmp_path, task), record)
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        hfb.read_checkpoint(tmp_path, task, spec)


def test_checkpoint_accepts_a_complete_synthetic_record(tmp_path):
    spec = settings()
    task = hfb.build_tasks(spec)[0]
    (tmp_path / "runs").mkdir()
    record = synthetic_checkpoint(task, spec)
    hfb.atomic_json(hfb.checkpoint_path(tmp_path, task), record)
    assert hfb.read_checkpoint(tmp_path, task, spec) == record


def test_summary_does_not_retain_all_checkpoint_traces(tmp_path, monkeypatch):
    spec = settings(seeds=(7, 8))
    hfb.atomic_json(tmp_path / "manifest.json", {"settings": hfb.asdict(spec)})
    (tmp_path / "runs").mkdir()
    for task in hfb.build_tasks(spec):
        hfb.atomic_json(hfb.checkpoint_path(tmp_path, task), synthetic_checkpoint(task, spec))
    counts = {"live": 0, "peak": 0}

    class Monitored(dict):
        def __init__(self, record):
            super().__init__(record)
            counts["live"] += 1
            counts["peak"] = max(counts["peak"], counts["live"])

        def __del__(self):
            counts["live"] -= 1

    read = hfb.read_checkpoint
    monkeypatch.setattr(hfb, "read_checkpoint", lambda *args: Monitored(read(*args)))
    hfb.summarize(tmp_path)
    assert counts["peak"] <= 2


@pytest.mark.parametrize("comparison_cost,expected_queries", [(1, 4), (2, 2), (3, 1)])
def test_worker_budget_accounting_with_a_fake_environment(comparison_cost, expected_queries, monkeypatch):
    # No benchmark function is constructed or evaluated: the fake environment
    # only tests accounting, and records the exact number of paid exposures.
    class FakeOracle:
        y_opt = 1.
        bounds_low = np.array([0.])
        bounds_high = np.array([1.])
        class spec:
            dim = 1
        def predict_many(self, X):
            return np.zeros((len(X), 1))
    monkeypatch.setattr(hfb.bb.SyntheticOracle, "from_stats", lambda *args: FakeOracle())
    spec = settings(methods=("dueling_ucb",), reference="dueling_ucb", reference_rule="policy",
                    comparison_cost=comparison_cost)
    task = hfb.build_tasks(spec)[0]
    record = hfb.run_task(task, spec, {"ackley": {"opt_z": 1.}})
    assert record["n_judgments"] == expected_queries
    assert record["design_exposures"] == 2*expected_queries
    assert record["budget_spent"] == comparison_cost*expected_queries
    assert record["budget_spent"] <= spec.budget
    assert all(row["feedback"] == .5 for row in record["trace"])


def test_worker_reuses_the_immutable_catalog_between_faults(monkeypatch):
    calls = []

    class FakeOracle:
        y_opt = 1.
        bounds_low, bounds_high = np.array([0.]), np.array([1.])
        class spec:
            dim = 1
        def predict_many(self, X):
            calls.append(X.copy())
            return X[:, :1]

    monkeypatch.setattr(hfb.bb.SyntheticOracle, "from_stats", lambda *args: FakeOracle())
    spec = settings(methods=("random",))
    tasks = hfb.build_tasks(spec)
    first = hfb.run_task(tasks[0], spec, {"ackley": {"opt_z": 1.}})
    hfb.run_task(tasks[1], spec, {"ackley": {"opt_z": 1.}})
    hfb.run_task(tasks[1], spec, {"ackley": {"opt_z": 1., "unused_diagnostic": float("nan")}})
    assert len(calls) == 1
    hfb.run_task(tasks[1], spec, {"ackley": {"opt_z": 2.}})
    assert len(calls) == 2  # Different scaling cannot reuse cached truth.
    X, truth, _ = hfb._catalog_environment("ackley", 7, 8, '{"opt_z": 1.0}')
    assert not X.flags.writeable and not truth.flags.writeable
    assert first["n_judgments"] == spec.budget


def test_numerical_supremum_extension_uses_its_recorded_finite_optimum(tmp_path):
    # Explicitly authorized smoke: four random-policy judgments on one cheap
    # analytic extension, with no GP or policy-model fitting.
    stats = hfb.bb.load_stats()
    spec = settings(functions=("bump_a4_w0.05",), methods=("random",))
    spec.validate(stats)
    task = hfb.build_tasks(spec)[0]
    record = hfb.run_task(task, spec, stats)
    assert record["y_opt"] == stats["bump_a4_w0.05"]["opt_z"]
    assert all(np.isfinite(row[key]) for row in record["rows"] for key in hfb.RESPONSES)
    (tmp_path / "runs").mkdir()
    hfb.atomic_json(hfb.checkpoint_path(tmp_path, task), record)
    hfb.read_checkpoint(tmp_path, task, spec)


def test_finite_published_optimum_is_preserved_over_a_different_recorded_bound(monkeypatch):
    class FakeOracle:
        y_opt = 2.
        bounds_low, bounds_high = np.array([0.]), np.array([1.])
        class spec:
            dim = 1
        def predict_many(self, X):
            return X[:, :1]
    monkeypatch.setattr(hfb.bb.SyntheticOracle, "from_stats", lambda *args: FakeOracle())
    _, _, optimum = hfb._catalog_environment("ackley", 7, 8, '{"opt_z": 3.0}')
    assert optimum == 2.


@pytest.mark.parametrize("proposal", [(0.5,), (True,)])
def test_environment_rejects_noninteger_policy_proposals(monkeypatch, proposal):
    monkeypatch.setattr(hfb, "_catalog_environment", lambda *args:
                        (np.zeros((8, 1)), np.zeros(8), 1.))
    class BadPolicy:
        def ask(self):
            return proposal
    monkeypatch.setattr(hfb.fp, "make_policy", lambda *args, **kwargs: BadPolicy())
    spec = settings()
    with pytest.raises(ValueError, match="Invalid proposal"):
        hfb.run_task(hfb.build_tasks(spec)[0], spec, {"ackley": {"opt_z": 1.}})
