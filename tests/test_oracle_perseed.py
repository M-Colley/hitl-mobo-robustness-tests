"""The oracle-isolation rerun with the error grid calibrated per run seed.

scripts/run_oracle_isolation_perseed.py scales each run seed's grid to that seed's
own oracle sigma_f (the published arms scale every seed's grid to the seed-7
oracle), runs one simulator invocation per (family, landscape, seed), and
scripts/checks/oracle_perseed.py analyses the result with the published pipeline,
which reads the per-seed grid from manifest_seeds.csv
(oracle_isolation.sigma_multiples). Pinned here: the manifest's scaling and its
seed-7 identity with the published grid, the magnitude recovered per seed, the
simulator command of one task against run_oracle_isolation.ps1, the run-file
bookkeeping, and the analysis on a tiny synthetic arm.
"""
from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "checks"))

import oracle_isolation as iso  # noqa: E402
import run_oracle_isolation_perseed as drv  # noqa: E402

DEPLOYED = "final_inference_simple_regret_excess_true"


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


def _published(sigma7: dict[str, float], model: str = "gradient_boosting") -> pd.DataFrame:
    return pd.DataFrame([{"landscape": k, "dataset": f"iso_{k}", "oracle_model": model, "cv_r2": np.nan,
                          "sigma_f_fitted": v, "sigma_f_true": 1.0, "corr_oracle_truth": 0.9,
                          "jitter_stds": ",".join(f"{s * v:.6f}" for s in iso.SIGMA_GRID)}
                         for k, v in sigma7.items()])


def _sigmaf(table: dict[str, dict[int, float]], family: str = "tree", model: str = "gradient_boosting"):
    return pd.DataFrame([{"family": family, "landscape": k, "oracle_model": model, "seed": s, "sigma_f": v}
                         for k, seeds in table.items() for s, v in seeds.items()])


SEEDS = {"a": {7: 1.0335587136642896, 8: 0.5, 9: 1.0, 10: 2.0, 11: 1.25}, "b": {7: 0.4, 8: 0.4, 9: 0.8, 10: 0.2, 11: 0.4}}


def test_the_manifest_scales_each_seed_grid_to_its_own_oracle():
    published = _published({k: v[7] for k, v in SEEDS.items()})
    m = drv.build_manifest("tree", _sigmaf(SEEDS), published).set_index(["landscape", "seed"])
    assert len(m) == 10 and set(m.index.get_level_values("seed")) == set(iso.SEEDS)
    assert m.loc[("a", 8), "jitter_stds"] == "0.125000,0.500000,2.500000"
    assert m.loc[("a", 10), "jitter_stds"] == "0.500000,2.000000,10.000000"
    assert m.loc[("b", 10), "ratio_to_seed7"] == pytest.approx(0.5)
    # seed 7 is the published grid exactly, so its runs are the published configuration
    assert m.loc[("a", 7), "jitter_stds"] == "0.258390,1.033559,5.167794" == m.loc[("a", 7), "published_jitter_stds"]
    assert (m.augmentation == "jitter").all() and (m.oracle_model == "gradient_boosting").all()
    assert m.loc[("a", 9), "selection_path"] == "output-oracle-iso/selection/iso_a.json"
    assert m.loc[("a", 9), "dataset_config"] == "output-oracle-iso/configs/datasets-iso_a.json"
    gp = drv.build_manifest("gaussian_process", _sigmaf(SEEDS, "gaussian_process", "gaussian_process"),
                            _published({k: v[7] for k, v in SEEDS.items()}, "gaussian_process"))
    assert (gp.augmentation == "none").all()
    assert gp.selection_path.iloc[0] == "output-oracle-iso-gaussian_process/selection/iso_a.json"
    # every family's synthetic datasets are the tree arm's
    assert gp.dataset_config.iloc[0] == "output-oracle-iso/configs/datasets-iso_a.json"


def test_seed_7_must_reproduce_the_published_calibration():
    published = _published({k: v[7] for k, v in SEEDS.items()})
    bad = published.copy()
    bad.loc[0, "jitter_stds"] = "0.258391,1.033559,5.167794"
    with pytest.raises(ValueError, match="seed 7's grid"):
        drv.build_manifest("tree", _sigmaf(SEEDS), bad)
    bad = published.copy()
    bad.loc[0, "sigma_f_fitted"] += 1e-9
    with pytest.raises(ValueError, match="seed 7 sigma_f"):
        drv.build_manifest("tree", _sigmaf(SEEDS), bad)
    table = _sigmaf(SEEDS)
    with pytest.raises(ValueError, match="no sigma_f for seeds"):
        drv.build_manifest("tree", table[table.seed != 11], published)
    with pytest.raises(ValueError, match="fitted"):
        drv.build_manifest("tree", _sigmaf(SEEDS, model="catboost"), published)
    # rounding noise far below the grid's sixth decimal is accepted (oracle_sigmaf_seeds refits with one BLAS thread)
    noisy = _sigmaf({k: {**v, 7: v[7] + 1e-15} for k, v in SEEDS.items()})
    assert len(drv.build_manifest("tree", noisy, published)) == 10


def test_the_grid_is_formatted_as_calibrate_formats_it():
    assert drv.seed_grid(1.0335587136642896) == "0.258390,1.033559,5.167794"
    source = (REPO / "scripts" / "oracle_isolation.py").read_text(encoding="utf-8")
    assert '",".join(f"{s * sigma_f:.6f}" for s in SIGMA_GRID)' in source
    manifest = REPO / "output-oracle-iso" / "manifest.csv"
    if not manifest.is_file():
        pytest.skip("no published manifest in this checkout")
    for family in drv.FAMILIES:
        m = pd.read_csv(drv.published_root(family) / "manifest.csv")
        assert [drv.seed_grid(v) for v in m.sigma_f_fitted] == m.jitter_stds.tolist()


def test_the_oracle_manifest_drops_the_single_grid():
    published = _published({"a": 1.0})
    out = drv.oracle_manifest(published)
    assert "jitter_stds" not in out and out.calibration.iloc[0] == "per_seed: manifest_seeds.csv"
    assert out.drop(columns="calibration").equals(published.drop(columns="jitter_stds"))


# ---------------------------------------------------------------------------
# the magnitude of a run
# ---------------------------------------------------------------------------


def _grid_runs(sigma: dict[int, float]) -> pd.DataFrame:
    rows = [{"seed": s, "acquisition": a, "jitter_std": std}
            for s, v in sigma.items() for a in ("logei", "random")
            for std in drv.parse_stds(drv.seed_grid(v))]
    return pd.DataFrame(rows)


def test_the_magnitude_is_recovered_per_seed():
    sigma = {7: 1.0335587136642896, 8: 0.2329244274048973, 9: 2.29941001894975}  # the smallest and largest sigma_f
    runs = _grid_runs(sigma)
    mult = iso.sigma_multiples(runs, pd.Series(sigma))
    assert mult.tolist() == [0.25, 1.0, 5.0] * 6
    assert set(mult) <= set(iso.SIGMA_GRID)
    # the published reading (one seed-7 sigma_f) puts seeds 8 and 9 elsewhere on the scale
    published = iso.sigma_multiples(runs, sigma[7])
    assert published[runs.seed == 7].tolist() == [0.25, 1.0, 5.0] * 2
    assert not set(published[runs.seed == 8]) & set(iso.SIGMA_GRID)
    pd.testing.assert_series_equal(published, (runs.jitter_std / sigma[7]).round(2))


def test_a_run_off_its_seeds_grid_is_refused():
    sigma = {7: 1.0, 8: 0.5}
    runs = _grid_runs(sigma)
    with pytest.raises(ValueError, match="not on the grid"):
        iso.sigma_multiples(runs, pd.Series({7: 1.0, 8: 0.6}))
    with pytest.raises(ValueError, match="no sigma_f for seeds"):
        iso.sigma_multiples(runs, pd.Series({7: 1.0}))


def test_seed_sigma_f_reads_the_per_seed_manifest(tmp_path):
    assert iso.seed_sigma_f(tmp_path) is None   # a published arm
    pd.DataFrame({"landscape": ["a", "a", "b"], "seed": [7, 8, 7], "sigma_f": [1.0, 2.0, 3.0]}).to_csv(
        tmp_path / iso.SEEDS_MANIFEST, index=False)
    table = iso.seed_sigma_f(tmp_path)
    assert table.loc["a"].to_dict() == {7: 1.0, 8: 2.0} and table.loc[("b", 7)] == 3.0
    pd.DataFrame({"landscape": ["a", "a"], "seed": [7, 7], "sigma_f": [1.0, 2.0]}).to_csv(
        tmp_path / iso.SEEDS_MANIFEST, index=False)
    with pytest.raises(ValueError, match="more than one row"):
        iso.seed_sigma_f(tmp_path)


# ---------------------------------------------------------------------------
# the driver's command line and run files
# ---------------------------------------------------------------------------


def test_the_command_of_one_task_is_run_oracle_isolation_ps1s_with_one_seed():
    cmd = drv.sim_command("mlp", "powell", 8, "0.290882,1.163530,5.817649", Path("out"), "py")
    assert cmd == ["py", "scripts/bo_sensor_error_simulation.py",
                   "--dataset-config", "output-oracle-iso/configs/datasets-iso_powell.json", "--objective", "composite",
                   "--acq-list", "logei,qnei,ucb,ei,random,sobol",
                   "--oracle-model", "auto", "--oracle-selection-path", "output-oracle-iso-mlp/selection/iso_powell.json",
                   "--oracle-augmentation", "none",
                   "--iterations", "50", "--initial-samples", "5",
                   "--error-models", "gaussian", "--jitter-stds", "0.290882,1.163530,5.817649",
                   "--jitter-iterations", "0",
                   "--seeds", "8", "--output-dir", "out", "--parallel", "--n-jobs", "1", "--resume"]
    tree = drv.sim_command("tree", "branin", 7, "x", Path("o"), "py", allow_legacy_resume=True)
    assert tree[tree.index("--oracle-augmentation") + 1] == "jitter"
    assert tree[tree.index("--oracle-selection-path") + 1] == "output-oracle-iso/selection/iso_branin.json"
    assert tree[-2:] == ["--resume", "--allow-legacy-resume"]
    # the same flags and values as the published driver, apart from the seed, the grid and the directory
    ps1 = (REPO / "run_oracle_isolation.ps1").read_text(encoding="utf-8")
    call = ps1[ps1.index("bo_sensor_error_simulation.py"):ps1.index("*>> $log")]
    flags = re.findall(r"(--[a-z-]+)(?:\s+([^\s`-][^\s`]*))?", call)
    ours = dict(zip(cmd[2::1], cmd[3::1]))
    for flag, value in flags:
        assert flag in cmd, flag
        if flag != "--seeds" and value and not value.startswith("$") and not value.startswith('"'):
            assert ours[flag] == value, (flag, value, ours[flag])
    assert ps1.count("$augmentation = \"none\"") == 1 and ps1.count("$augmentation = \"jitter\"") == 1


def test_the_run_file_names_are_the_simulators():
    names = drv.expected_files("tree", "branin", 10, "0.258390,1.033559,5.167794", "gradient_boosting")
    assert len(names) == 6 * 4
    assert "bo_sensor_error_iso_branin_composite_ei_seed10_jittered_gradient_boosting_gaussian_jit0_std0.25839.csv" \
        in names
    published = REPO / "output-oracle-iso" / "runs" / "iso_branin"
    if not published.is_dir():
        pytest.skip("no published runs in this checkout")
    assert all((published / n).is_file() for n in names)


def _arm(tmp_path, monkeypatch, sigma: dict[str, dict[int, float]] = None) -> str:
    """A per-seed tree arm under tmp_path with its manifest."""
    sigma = sigma or {"a": {s: 1.0 + 0.1 * (s - 7) for s in iso.SEEDS}}
    monkeypatch.setattr(drv, "REPO", tmp_path)
    published = _published({k: v[7] for k, v in sigma.items()})
    m = drv.build_manifest("tree", _sigmaf(sigma), published)
    root = drv.arm_root("tree")
    root.mkdir(parents=True)
    m.to_csv(root / iso.SEEDS_MANIFEST, index=False)
    drv.oracle_manifest(published).to_csv(root / "manifest.csv", index=False)
    return "tree"


def _touch_runs(family, landscape, seeds, fingerprint=True):
    m = drv.load_manifest(family)
    d = drv.run_dir(family, landscape)
    d.mkdir(parents=True, exist_ok=True)
    for row in m[(m.landscape == landscape) & m.seed.isin(seeds)].itertuples():
        for name in drv.expected_files(family, landscape, row.seed, row.jitter_stds, row.oracle_model):
            header = "iteration,run_config_fingerprint" if fingerprint and "_jittered_" in name else "iteration"
            (d / name).write_text(header + "\n1,x\n", encoding="utf-8")


def test_the_run_files_are_checked_against_the_manifest(tmp_path, monkeypatch):
    family = _arm(tmp_path, monkeypatch)
    _touch_runs(family, "a", iso.SEEDS)
    assert drv.check_run_files(family, "a") == []
    d = drv.run_dir(family, "a")
    stray = d / drv.noisy_name("a", "logei", 8, "gradient_boosting", 0.25)   # seed 7's grid on seed 8
    stray.write_text("iteration\n", encoding="utf-8")
    gone = d / drv.baseline_name("a", "sobol", 11, "gradient_boosting")
    gone.unlink()
    problems = drv.check_run_files(family, "a")
    assert problems == [f"missing {gone.name}", f"unexpected {stray.name}"]


def test_legacy_run_files_must_be_recorded_unchanged_copies(tmp_path, monkeypatch):
    family = _arm(tmp_path, monkeypatch)
    d = drv.run_dir(family, "a")
    d.mkdir(parents=True)
    fresh = d / drv.noisy_name("a", "logei", 7, "gradient_boosting", 0.25)
    fresh.write_text("iteration,run_config_fingerprint\n1,abc\n", encoding="utf-8")
    assert drv.legacy_guard(family, "a") is False          # nothing without a fingerprint
    copy = d / drv.baseline_name("a", "logei", 7, "gradient_boosting")
    copy.write_text("iteration,objective_true\n1,0.5\n", encoding="utf-8")
    with pytest.raises(drv.PerSeedError, match="not a recorded clean copy"):
        drv.legacy_guard(family, "a")
    pd.DataFrame([{"dataset": "iso_a", "seed": 7, "acquisition": "logei", "file": copy.name,
                   "sha256": drv.sha256(copy), "source": "published"}]).to_csv(
        drv.arm_root(family) / "clean_copies.csv", index=False)
    assert drv.legacy_guard(family, "a") is True
    copy.write_text("iteration,objective_true\n1,0.6\n", encoding="utf-8")
    with pytest.raises(drv.PerSeedError, match="differs from the clean twin"):
        drv.legacy_guard(family, "a")


def test_a_finished_invocation_under_another_grid_is_refused(tmp_path, monkeypatch):
    family = _arm(tmp_path, monkeypatch)
    (task,) = [t for t in drv.tasks_for([family], seeds=[8])]
    assert task.stds == drv.seed_grid(1.1) and drv.marker_state(task) == "todo"
    drv.run_dir(family, "a").mkdir(parents=True)
    drv.seed_marker(task).write_text('{"jitter_stds": "%s"}' % task.stds, encoding="utf-8")
    assert drv.marker_state(task) == "done"
    drv.seed_marker(task).write_text('{"jitter_stds": "0.1,0.2,0.3"}', encoding="utf-8")
    with pytest.raises(drv.PerSeedError, match="manifest changed"):
        drv.marker_state(task)


def test_the_scheduler_never_runs_two_invocations_in_one_directory(monkeypatch):
    tasks = [drv.Task(f, land, s, "x", "m") for s in iso.SEEDS for f in ("tree", "mlp") for land in ("a", "b")]
    lock = threading.Lock()
    active: dict = {}
    peak = {"total": 0, "per_dir": 0}
    ran = []

    def fake_run(task, python):
        with lock:
            active[task.key] = active.get(task.key, 0) + 1
            peak["per_dir"] = max(peak["per_dir"], active[task.key])
            peak["total"] = max(peak["total"], sum(active.values()))
        time.sleep(0.01)
        with lock:
            active[task.key] -= 1
            ran.append(task)
        return True, 0.01, "ok"

    monkeypatch.setattr(drv, "run_task", fake_run)
    monkeypatch.setattr(drv, "log", lambda message: None)
    scheduler = drv.Scheduler(tasks, n_jobs=3, python="py", evaluate=False)
    scheduler.run()
    assert sorted(map(str, ran)) == sorted(map(str, tasks)) and not scheduler.failed
    assert peak["per_dir"] == 1 and 1 < peak["total"] <= 3


# ---------------------------------------------------------------------------
# the analysis on a tiny synthetic arm
# ---------------------------------------------------------------------------

LANDSCAPES = ("branin", "powell")   # real names: opt_z comes from the tracked statistics


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    """Two landscapes, two seeds whose oracles differ (sigma_f 1 and 2), each run on its own grid.

    Per (landscape, seed): y_opt 10 (seed 7) or 20 (seed 8), floors reach 4 and 6 (seed 7)
    or 8 and 12 (seed 8), learners lose excess = magnitude x (1 or 3) on the trajectory.
    """
    arm = tmp_path / "arm"
    sigma = {7: 1.0, 8: 2.0}
    rows_seeds, rows_m = [], []
    for name in LANDSCAPES:
        rows_m.append({"landscape": name, "dataset": f"iso_{name}", "oracle_model": "m", "cv_r2": np.nan,
                       "sigma_f_fitted": 1.0, "sigma_f_true": 1.0, "corr_oracle_truth": 0.9,
                       "calibration": "per_seed: " + iso.SEEDS_MANIFEST})
        rows = []
        for seed, s_f in sigma.items():
            rows_seeds.append({"landscape": name, "seed": seed, "sigma_f": s_f})
            y_opt, floors, k = (10.0, (4.0, 6.0), 1.0) if seed == 7 else (20.0, (8.0, 12.0), 3.0)
            for mult in iso.SIGMA_GRID:
                std = float(f"{mult * s_f:.6f}")
                for acq, clean in zip(iso.FLOORS, floors):
                    rows.append({"seed": seed, "acquisition": acq, "jitter_iteration": 0, "jitter_std": std,
                                 iso.BASELINE_BEST: clean, iso.BASELINE_REGRET: y_opt - clean,
                                 iso.NOISY_BEST: clean, iso.RESPONSE: 0.0, DEPLOYED: 0.0})
                for acq in iso.ACQS:
                    rows.append({"seed": seed, "acquisition": acq, "jitter_iteration": 0, "jitter_std": std,
                                 iso.BASELINE_BEST: y_opt - 1.0, iso.BASELINE_REGRET: 1.0,
                                 iso.NOISY_BEST: y_opt - 2.0, iso.RESPONSE: mult * k, DEPLOYED: 2 * mult * k})
        out = arm / "runs" / f"iso_{name}" / "evaluation"
        out.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(out / "paired_excess_metrics.csv", index=False)
        # the exact arm: one function, opt_z from the statistics, excess = magnitude x 2
        import boba_benchmarks as bb
        opt_z = float(bb.load_stats()[name]["opt_z"])
        exact = []
        for seed in sigma:
            for mult in iso.SIGMA_GRID:
                for acq in iso.ACQS + iso.FLOORS:
                    exact.append({"seed": seed, "acquisition": acq, "jitter_iteration": 0, "jitter_std": mult,
                                  "error_model": "gaussian", iso.BASELINE_BEST: opt_z - 1.0,
                                  iso.BASELINE_REGRET: 1.0, iso.RESPONSE: 2 * mult, DEPLOYED: 4 * mult})
        ex = tmp_path / "output-boba" / name / "evaluation"
        ex.mkdir(parents=True)
        pd.DataFrame(exact).to_csv(ex / "paired_excess_metrics.csv", index=False)
    pd.DataFrame(rows_m).to_csv(arm / "manifest.csv", index=False)
    pd.DataFrame(rows_seeds).to_csv(arm / iso.SEEDS_MANIFEST, index=False)
    return arm, sigma


def test_the_floor_gap_analysis_reads_each_seeds_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # oracle_isolation reads the exact arm from output-boba/ here
    arm, _ = _fixture(tmp_path)
    out = tmp_path / "out"
    per = iso._analyse_one(iso.RESPONSE, "", root=arm, out_dir=out).set_index(["landscape", "sigma_multiple"])
    assert sorted(set(per.index.get_level_values(1))) == list(iso.SIGMA_GRID)
    assert (per.n_fitted == 8).all()   # 2 seeds x 4 learners, both seeds in every cell
    # the fitted gap within each seed: (10 - 5) and (20 - 10), mean 7.5; the excess mean_s over (m, 3m) = 2m
    for m in iso.SIGMA_GRID:
        assert per.loc[("branin", m), "gap_fitted"] == pytest.approx(7.5)
        assert per.loc[("branin", m), "frac_fitted"] == pytest.approx(2 * m / 7.5)
        assert per.loc[("branin", m), "excess_exact"] == pytest.approx(2 * m)
    deployed = iso._analyse_one(DEPLOYED, "_deployed", root=arm, out_dir=out).set_index(
        ["landscape", "sigma_multiple"])
    assert deployed.loc[("powell", 5.0), "frac_fitted"] == pytest.approx(4 * 5.0 / 7.5)
    for f in ("oracle_isolation_per_landscape.csv", "oracle_isolation_summary.csv",
              "oracle_isolation_summary_deployed_best_clean.csv"):
        assert (out / f).is_file()
    # without its per-seed manifest the arm is refused: read with one seed-7 sigma_f, each cell would
    # hold only the seeds whose grid happens to match it (here seed 7 alone, silently)
    (arm / iso.SEEDS_MANIFEST).unlink()
    with pytest.raises(ValueError, match="declares a per-seed grid"):
        iso._analyse_one(iso.RESPONSE, "", root=arm, out_dir=out)


def test_the_side_by_side_table_and_the_completeness_check(tmp_path, monkeypatch):
    import oracle_perseed as op
    monkeypatch.chdir(tmp_path)
    arm, _ = _fixture(tmp_path)
    dirs = {}
    for k, f in enumerate(op.FAMILIES):
        d = tmp_path / f"out_{f}"
        for suffix, response in iso.RESPONSES.items():
            iso._analyse_one(response, suffix, root=arm, out_dir=d)
        # the achievable files are oracle_achievable's; here the floor-gap ones stand in for them
        for p in list(d.glob("oracle_isolation_summary*.csv")):
            p_ach = d / p.name.replace("oracle_isolation_summary", "oracle_isolation_summary_achievable")
            p_ach.write_bytes(p.read_bytes())
            for extra in ("_sigma_f", "_both_best_clean"):
                if p.name in ("oracle_isolation_summary.csv", "oracle_isolation_summary_deployed.csv"):
                    (d / p_ach.name.replace(".csv", f"{extra}.csv")).write_bytes(p.read_bytes())
        dirs[f] = d
    table = op.side_by_side({"published": dirs, "per_seed": dirs})
    assert len(table) == 2 * len(op.NORMALISERS) * 3 * 3
    row = table[(table.family == "GP") & (table.sigma_multiple == 1.0)
                & (table.normaliser == "floor gap: oracle") & (table.response == "trajectory search loss")].iloc[0]
    # fitted 2m / 7.5 on both landscapes; exact 2m over a floor gap of 1 (the floors reach opt_z - 1)
    assert row.published_ratio_fitted_over_exact == pytest.approx(1 / 7.5)
    assert row.published_n_landscapes == 2
    lines = op.format_table(table, ("published", "per_seed"))
    assert any(line.startswith("   GP   1 ") for line in lines)

    # completeness: an arm with no runs is reported, not analysed
    monkeypatch.setattr(drv, "REPO", tmp_path)
    monkeypatch.setattr(op, "PERSEED", {f: drv.arm_root(f) for f in op.FAMILIES})
    problems = op.missing_runs(("tree",))
    assert problems == ["tree: no manifest_seeds.csv/manifest.csv (run_oracle_isolation_perseed.py manifest)"]
    _arm(tmp_path / "x", monkeypatch)
    monkeypatch.setattr(op, "PERSEED", {f: drv.arm_root(f) for f in op.FAMILIES})
    assert op.missing_runs(("tree",)) == ["tree a: seeds [7, 8, 9, 10, 11] not run"]
    for s in iso.SEEDS:
        drv.run_dir("tree", "a").mkdir(parents=True, exist_ok=True)
        (drv.run_dir("tree", "a") / f".done_seed{s}").write_text("{}", encoding="utf-8")
    assert op.missing_runs(("tree",)) == ["tree a: not evaluated"]


def test_runs_under_another_simulator_file_are_refused(tmp_path, monkeypatch):
    family = _arm(tmp_path, monkeypatch)
    monkeypatch.setattr(drv, "simulator_sha256", lambda: "a" * 64)
    drv.simulator_guard([family])                       # nothing run, nothing copied
    (drv.arm_root(family) / "clean_copies.csv").write_text("dataset,seed,acquisition,file,sha256,source\n",
                                                           encoding="utf-8")
    with pytest.raises(drv.PerSeedError, match="identity check validated"):
        drv.simulator_guard([family])
    (drv.arm_root(family) / "clean_identity.json").write_text('{"simulator_sha256": "%s"}' % ("a" * 64),
                                                              encoding="utf-8")
    drv.simulator_guard([family])
    drv.run_dir(family, "a").mkdir(parents=True)
    (drv.run_dir(family, "a") / ".done_seed9").write_text('{"simulator_sha256": "%s"}' % ("b" * 64),
                                                          encoding="utf-8")
    with pytest.raises(drv.PerSeedError, match="two code versions"):
        drv.simulator_guard([family])
    with pytest.raises(drv.PerSeedError, match="no landscapes"):
        drv.tasks_for([family], landscapes=["zzz"])


def test_records_name_no_local_user(monkeypatch):
    """The JSON records (clean_identity.json, .done_seed<s>) carry the interpreter and scratch paths; every spelling
    of the home directory, JSON-escaped included, must become '~' so the anonymous mirror names no user."""
    import json

    home = r"C:\Users\someone"
    monkeypatch.setattr(drv.Path, "home", classmethod(lambda cls: drv.Path(home)))
    record = {"python": home + r"\AppData\python.exe", "work_dir": "C:/Users/someone/tmp",
              "command": ["/c/Users/someone/py", "scripts/x.py"], "seed": 7}
    masked = drv.mask_home_text(json.dumps(record, indent=2))
    assert "someone" not in masked
    back = json.loads(masked)
    assert back["python"] == r"~\AppData\python.exe" and back["work_dir"] == "~/tmp"
    assert back["command"] == ["~/py", "scripts/x.py"] and back["seed"] == 7
