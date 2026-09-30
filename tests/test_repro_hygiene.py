"""Reproducibility hygiene: flags that must reach the run, names that must carry
the run's settings, provenance that must identify the code and the data, and
pins that must match the environment the runs recorded.

Register items code-sim-repro-2, -6, -12, -13, -14, -16, code-recent-10, -11 and
code-estimands-12 (handover/review-2026-09-28.md).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "paper"))

import bo_sensor_error_simulation as sim  # noqa: E402
import bo_synthetic_error_simulation as syn  # noqa: E402
import evaluate_research_question as ev  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "core.autocrlf=false", *args],
                          cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


# ---------------------------------------------------------------------------
# --hold-until reaches the run (code-sim-repro-13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv, expected", [
    ([], 0.6),
    (["--hold-until", "0.3"], 0.3),
    (["--hold-until", "0"], 0.0),      # 0 is a value, not a request for the default
    (["--hold-until", "0.85"], 0.85),
])
def test_hold_until_flag_reaches_the_adaptation_fields(argv, expected):
    args = syn.parse_args(["--hold-early", "5", *argv])
    assert sim.adaptation_fields(args)["hold_until_frac"] == pytest.approx(expected)


def test_hold_until_default_without_the_flag_and_hand_built_namespace():
    import argparse

    # the fitted-oracle driver has no such flag
    assert sim.hold_until_fraction(argparse.Namespace()) == 0.6
    assert sim.hold_until_fraction(argparse.Namespace(hold_until_frac=0.25)) == 0.25
    assert sim.hold_until_fraction(argparse.Namespace(hold_until=None)) == 0.6


def test_hold_until_is_in_the_name_and_the_clean_run_guard():
    args = syn.parse_args(["--hold-early", "5", "--hold-until", "0.3"])
    assert syn._variant_suffix(args, "gaussian", 0.2, 0.5) == "_hold5@0.3"
    assert syn._clean_run_settings(args)["hold_early"] == [5, 0.3]
    # the one published arm passes 0.6, and its names are unchanged
    published = syn.parse_args(["--hold-early", "5", "--hold-until", "0.6"])
    assert syn._variant_suffix(published, "gaussian", 0.2, 0.5) == "_hold5@0.6"


# ---------------------------------------------------------------------------
# --response-clip and --response-round are in the file name (code-sim-repro-14)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv, suffix", [
    ([], ""),
    (["--response-clip", "none"], ""),
    (["--response-round", "0"], ""),
    (["--response-clip=-8,8", "--response-round", "0.55"], "_clip-8,8_round0.55"),
    (["--response-clip", "sample"], "_clip-sample"),
    (["--response-round", "0.55"], "_round0.55"),
    (["--response-clip=-8.0, 8.0"], "_clip-8,8"),
])
def test_instrument_settings_name_the_run(argv, suffix):
    assert syn._variant_suffix(syn.parse_args(argv), "gaussian", 0.2, 0.5) == suffix


def test_instrument_parts_follow_the_other_parts_and_keep_standard_names():
    spike = syn.parse_args(["--response-clip", "sample"])
    assert syn._variant_suffix(spike, "spike", 0.25, 20.0) == "_sp0.1-20_clip-sample"
    # multi-objective runs are not clipped (run_task passes no bounds), so not named
    multi = syn.parse_args(["--multi-objective", "--response-clip", "sample"])
    assert syn._variant_suffix(multi, "gaussian", 0.2, 0.5) == ""


def test_instrument_settings_do_not_change_the_clean_run():
    # _postprocess_response acts after the error, so the baseline is untouched
    # and the clean-run guard must not track clip or round.
    args = syn.parse_args(["--response-clip=-8,8", "--response-round", "0.55"])
    assert syn._clean_run_settings(args) == {}


def test_evaluation_parses_the_instrument_variant():
    name = ("bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit20_std1.0"
            "_clip-8,8_round0.55.csv")
    assert ev.parse_variant(name) == "clip-8,8_round0.55"
    sample = "bo_sensor_error_branin_value_logei_seed7_jittered_exact_spike_jit0_std0.25_sp0.15-20_clip-sample.csv"
    assert ev.parse_variant(sample) == "sp0.15-20_clip-sample"
    assert ev.parse_variant("bo_sensor_error_branin_value_logei_seed7_baseline_exact.csv", "exact") == ""


def _legacy_instrument_dir(tmp_path: Path, via_argv: bool = False) -> Path:
    out = tmp_path / "arm"
    (out / "branin").mkdir(parents=True)
    (out / "branin" / "bo_sensor_error_branin_value_logei_seed7_jittered_exact_gaussian_jit0_std1.0.csv"
     ).write_text("iteration\n1\n", encoding="utf-8")
    meta = {"args": {"response_clip": "none", "response_round": None}}
    if via_argv:
        meta["invocations"] = [{"argv": ["x.py", "--response-clip=-8,8"]}]
    else:
        meta["args"] = {"response_clip": "-8,8", "response_round": 0.55}
    (out / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


@pytest.mark.parametrize("via_argv", [False, True])
@pytest.mark.parametrize("argv", [[], ["--response-clip=-8,8", "--response-round", "0.55"]])
def test_a_directory_of_legacy_instrument_runs_is_refused(tmp_path, via_argv, argv):
    out = _legacy_instrument_dir(tmp_path, via_argv)
    with pytest.raises(ValueError, match="fresh --output-dir"):
        syn._guard_instrument_naming(out, syn.parse_args(argv))


def test_the_naming_guard_leaves_standard_and_new_directories_alone(tmp_path):
    standard = tmp_path / "standard"
    (standard / "branin").mkdir(parents=True)
    (standard / "run_metadata.json").write_text(json.dumps({"args": {"response_clip": "none"}}),
                                                encoding="utf-8")
    (standard / "branin" / "x_jittered_y.csv").write_text("iteration\n", encoding="utf-8")
    syn._guard_instrument_naming(standard, syn.parse_args([]))
    assert not (standard / syn.NAMING_MARKER).exists()
    # a new instrument arm gets the marker, so a later invocation is not refused
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    args = syn.parse_args(["--response-clip=-8,8"])
    syn._guard_instrument_naming(fresh, args)
    assert (fresh / syn.NAMING_MARKER).is_file()
    (fresh / "run_metadata.json").write_text(json.dumps({"args": {"response_clip": "-8,8"}}),
                                             encoding="utf-8")
    (fresh / "a_jittered_b.csv").write_text("iteration\n", encoding="utf-8")
    syn._guard_instrument_naming(fresh, args)


def test_the_published_instrument_arms_would_be_refused():
    for arm in ("output-boba-instrument", "output-boba-spike-clip"):
        if not (REPO / arm / "run_metadata.json").is_file():
            pytest.skip(f"{arm} is not on this machine")
        assert syn._legacy_instrument_runs(REPO / arm)


# ---------------------------------------------------------------------------
# code_stamp records WHICH changes, not only that there were some
# (code-sim-repro-2, code-recent-10)
# ---------------------------------------------------------------------------


def _toy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(repo))
    (repo / "scripts" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "datasets.json").write_text("[]\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_code_stamp_of_a_clean_tree(tmp_path):
    repo = _toy_repo(tmp_path)
    stamp = sim.code_stamp(repo, patch_dir=tmp_path / "meta")
    assert stamp["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert stamp["code_dirty"] is False
    assert stamp["dirty_files"] == []
    assert stamp["code_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert "code_diff_patch" not in stamp
    assert not (tmp_path / "meta").exists()


def test_code_stamp_hashes_the_diff_and_the_untracked_modules(tmp_path):
    repo = _toy_repo(tmp_path)
    (repo / "scripts" / "a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "scripts" / "new_module.py").write_text("y = 3\n", encoding="utf-8")
    (repo / "notes.txt").write_text("outside the provenance paths\n", encoding="utf-8")
    stamp = sim.code_stamp(repo, patch_dir=tmp_path / "meta")
    assert stamp["code_dirty"] is True
    assert any(f.endswith("scripts/a.py") and f.startswith(" M") for f in stamp["dirty_files"])
    assert any(f == "?? scripts/new_module.py" for f in stamp["dirty_files"])
    assert not any("notes.txt" in f for f in stamp["dirty_files"])
    patch = (tmp_path / "meta" / stamp["code_diff_patch"]).read_bytes()
    assert hashlib.sha256(patch).hexdigest() == stamp["code_diff_sha256"]
    assert stamp["code_diff_patch"] == f"code_diff_{stamp['code_diff_sha256'][:12]}.patch"
    assert b"+x = 2" in patch and b"y = 3" in patch
    # a different change gives a different hash
    (repo / "scripts" / "a.py").write_text("x = 4\n", encoding="utf-8")
    assert sim.code_stamp(repo)["code_diff_sha256"] != stamp["code_diff_sha256"]


def test_append_invocation_keeps_the_hash_at_top_level_and_per_invocation(tmp_path, monkeypatch):
    monkeypatch.delenv(sim.CODE_DIFF_STORE_ENV, raising=False)
    repo = _toy_repo(tmp_path)
    (repo / "scripts" / "a.py").write_text("x = 5\n", encoding="utf-8")
    meta_path = tmp_path / "arm" / "run_metadata.json"
    meta_path.parent.mkdir()
    payload = sim.append_invocation(meta_path, {"git_commit": "x"}, repo)
    assert payload["code_dirty"] is True
    assert payload["code_diff_sha256"] == payload["invocations"][-1]["code_diff_sha256"]
    assert payload["invocations"][-1]["dirty_files"]
    # the patch goes to the git-ignored store, recorded relative to the repository,
    # and nothing is written beside run_metadata.json (a tracked, mirrored directory)
    recorded = payload["invocations"][-1]["code_diff_patch"]
    assert recorded.startswith("output/code_diffs/code_diff_")
    assert (repo / recorded).is_file()
    assert not list(meta_path.parent.glob("*.patch"))
    monkeypatch.setenv(sim.CODE_DIFF_STORE_ENV, str(tmp_path / "store"))
    (repo / "scripts" / "a.py").write_text("x = 6\n", encoding="utf-8")
    payload = sim.append_invocation(meta_path, {"git_commit": "x"}, repo)
    assert (tmp_path / "store" / payload["invocations"][-1]["code_diff_patch"]).is_file()


def test_the_written_patch_masks_the_home_directory(tmp_path):
    repo = _toy_repo(tmp_path)
    home = str(Path.home())
    (repo / "scripts" / "a.py").write_text(f"x = {home!r}  # {home.replace(chr(92), '/')}\n", encoding="utf-8")
    stamp = sim.code_stamp(repo, patch_dir=tmp_path / "meta")
    patch = (tmp_path / "meta" / stamp["code_diff_patch"]).read_bytes()
    assert home.encode() not in patch and home.replace("\\", "/").encode() not in patch
    assert home.replace("\\", "\\\\").encode() not in patch
    # code_diff_sha256 stays the hash of the code; the file carries its own
    assert hashlib.sha256(patch).hexdigest() == stamp["code_diff_patch_sha256"]
    assert stamp["code_diff_patch_sha256"] != stamp["code_diff_sha256"]
    assert sim.mask_home(b"no home here") == b"no home here"


def test_the_default_patch_store_is_git_ignored():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    probe = (sim.code_diff_store(REPO) / "code_diff_000000000000.patch").relative_to(REPO).as_posix()
    result = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", probe])
    assert result.returncode == 0, probe


def test_the_synthetic_driver_stamps_its_metadata_through_append_invocation():
    source = (REPO / "scripts" / "bo_synthetic_error_simulation.py").read_text(encoding="utf-8")
    assert "sim.append_invocation(output_dir / \"run_metadata.json\"" in source


# ---------------------------------------------------------------------------
# data pinning (code-sim-repro-12)
# ---------------------------------------------------------------------------


def _data_repo(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "remote-data"
    src.mkdir()
    _git(tmp_path, "init", "-q", str(src))
    (src / "ObservationsPerEvaluation.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "v1")
    return src, _git(src, "rev-parse", "HEAD")


def test_a_pinned_clone_is_checked_out_at_the_pin_and_verified_on_reuse(tmp_path, monkeypatch):
    src, first = _data_repo(tmp_path)
    (src / "ObservationsPerEvaluation.csv").write_text("a,b\n1,3\n", encoding="utf-8")
    _git(src, "commit", "-q", "-am", "v2")
    second = _git(src, "rev-parse", "HEAD")
    cache = tmp_path / "cache"
    # a file:// URL stands in for the remote: the cache entry is named after its
    # last part (a bare Windows path would name the source directory itself)
    url = src.as_uri()
    target = sim.fetch_remote_dataset(url, cache, commit=first)
    assert target == cache / "remote-data"
    assert sim.git_head(target) == first
    assert (target / "ObservationsPerEvaluation.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
    # reuse at the pin: fine
    assert sim.fetch_remote_dataset(url, cache, commit=first) == target
    # reuse against another pin: refused, with the fix in the message
    monkeypatch.delenv(sim.DATA_COMMIT_MISMATCH_ENV, raising=False)
    with pytest.raises(ValueError, match="pins data_commit"):
        sim.fetch_remote_dataset(url, cache, commit=second)
    monkeypatch.setenv(sim.DATA_COMMIT_MISMATCH_ENV, "warn")
    assert sim.fetch_remote_dataset(url, cache, commit=second) == target
    assert sim.git_head(target) == first


def test_a_failed_pinned_clone_leaves_nothing_behind(tmp_path):
    src, _ = _data_repo(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(Exception):
        sim.fetch_remote_dataset(src.as_uri(), cache, commit="0" * 40)
    assert not (cache / "remote-data").exists()
    assert (src / "ObservationsPerEvaluation.csv").is_file()   # the source is untouched


def test_parse_dataset_configs_passes_the_pin_and_records_it(tmp_path, monkeypatch):
    seen = []

    def fake_fetch(url, cache_dir, commit=None):
        seen.append((url, commit))
        path = tmp_path / "cached"
        path.mkdir(exist_ok=True)
        return path

    monkeypatch.setattr(sim, "fetch_remote_dataset", fake_fetch)
    config = tmp_path / "datasets.json"
    config.write_text(json.dumps([{
        "name": "demo", "data_dir": "https://github.com/example/demo-data",
        "data_commit": "a" * 40, "param_columns": ["p"], "objective_map": {"score": ["s"]},
    }]), encoding="utf-8")
    (dataset,) = sim.parse_dataset_configs(None, config, tmp_path)
    assert seen == [("https://github.com/example/demo-data", "a" * 40)]
    assert dataset.data_commits == {str(tmp_path / "cached"): "a" * 40}
    record = sim.dataset_provenance([dataset])["demo"]["data_dirs"][0]
    assert record["pinned_commit"] == "a" * 40 and record["commit"] is None


def test_an_unpinned_local_entry_records_no_pin_and_a_bad_pin_is_refused(tmp_path):
    data = tmp_path / "local"
    data.mkdir()
    config = tmp_path / "datasets.json"
    config.write_text(json.dumps([{"name": "loc", "data_dir": str(data), "param_columns": ["p"],
                                   "objective_map": {"score": ["s"]}}]), encoding="utf-8")
    (dataset,) = sim.parse_dataset_configs(None, config, tmp_path)
    assert dataset.data_commits == {}
    config.write_text(json.dumps([{"name": "bad", "data_dir": str(data), "data_commit": 7,
                                   "param_columns": ["p"], "objective_map": {"score": ["s"]}}]),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="data_commit"):
        sim.parse_dataset_configs(None, config, tmp_path)


@pytest.mark.parametrize("name", ["datasets.json", "datasets-ehmi.json", "datasets-extended.json",
                                  "datasets-provoice.json"])
def test_every_remote_dataset_in_the_repository_is_pinned(name):
    entries = json.loads((REPO / name).read_text(encoding="utf-8"))
    remote = [e for e in entries if sim.is_remote_dataset_path(str(e.get("data_dir", "")))]
    assert remote
    for entry in remote:
        assert re.fullmatch(r"[0-9a-f]{40}", entry.get("data_commit", "")), entry["name"]


def test_the_pins_are_the_commits_the_fitted_runs_read():
    """The fitted-oracle runs recorded data_dir_commits; the pins must be those."""
    pins = {e["name"]: e["data_commit"] for e in json.loads((REPO / "datasets.json").read_text("utf-8"))}
    recorded = {}
    for meta in (REPO / "output-fitted").glob("*/run_metadata.json"):
        for path, commit in (json.loads(meta.read_text("utf-8")).get("data_dir_commits") or {}).items():
            recorded[meta.parent.name] = commit
    if not recorded:
        pytest.skip("output-fitted is not on this machine")
    for name, commit in recorded.items():
        assert pins[name] == commit, name


# ---------------------------------------------------------------------------
# environment pins (code-sim-repro-6, paper-main-11)
# ---------------------------------------------------------------------------


def _pins() -> dict[str, str]:
    pins = {}
    for line in (REPO / "requirements-eval.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
        elif line.startswith("botorch @ "):
            pins["botorch"] = line
    return pins


def test_requirements_eval_pins_the_build_that_ran():
    pins = _pins()
    assert pins["numpy"] == "2.5.3" and pins["scipy"] == "1.18.1"
    assert re.search(r"botorch\.git@eea35f51a[0-9a-f]{31}$", pins["botorch"])
    toml = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in toml


def test_requirements_eval_matches_the_recorded_package_versions():
    meta_path = REPO / "output-boba" / "run_metadata.json"
    if not meta_path.is_file():
        pytest.skip("output-boba is not on this machine")
    recorded = json.loads(meta_path.read_text(encoding="utf-8"))["package_versions"]
    pins = _pins()
    for package, version in recorded.items():
        if package == "botorch":
            assert version == "0.18.2.dev23+geea35f51a"
            assert "eea35f51a" in pins["botorch"]
        else:
            assert pins[package.lower()] == version.split("+")[0], package


def test_collect_package_versions_never_records_null_for_an_installed_package():
    versions = sim.collect_package_versions(["numpy", "statsmodels", "no-such-package-xyz"])
    assert versions["numpy"]
    assert versions["no-such-package-xyz"] == "not_installed"
    assert versions["statsmodels"] is not None


# ---------------------------------------------------------------------------
# the provenance check's comparison and its dataset-config override
# ---------------------------------------------------------------------------


def test_compare_runs_gives_exact_within_tolerance_and_differs(tmp_path):
    import check_provenance as cp

    base = pd.DataFrame({"iteration": [1, 2, 3], "x0": [0.1, 0.2, 0.3], "acquisition": ["logei"] * 3,
                         "run_id": ["a", "a", "a"], "fit_time_sec": [1.0, 2.0, 3.0]})
    logged = tmp_path / "logged.csv"
    base.to_csv(logged, index=False)

    def verdict(frame):
        path = tmp_path / "new.csv"
        frame.to_csv(path, index=False)
        return cp.compare_runs(logged, path)

    # a fresh run_id and another wall-clock time are not differences
    same = base.assign(run_id=["b"] * 3, fit_time_sec=[9.0] * 3)
    assert verdict(same)["status"] == "exact"
    near = base.assign(x0=[0.1, 0.2 + 1e-13, 0.3])
    assert verdict(near)["status"] == "within tolerance"
    far = base.assign(x0=[0.1, 0.25, 0.3])
    result = verdict(far)
    assert result["status"] == "differs"
    assert (result["first_diff_column"], result["first_diff_trial"]) == ("x0", 2)
    assert result["n_diff_rows"] == 1
    extra = base.assign(new_column=[1, 2, 3])
    result = verdict(extra)
    assert result["status"] == "exact" and result["columns_only_new"] == "new_column"
    assert verdict(base.iloc[:2])["first_diff_column"] == "<row count>"
    # a driver-specific wall-clock column can be excluded as well
    timed = base.assign(seconds=[1.0, 2.0, 3.0])
    timed.to_csv(logged, index=False)
    path = tmp_path / "timed.csv"
    timed.assign(seconds=[4.0, 5.0, 6.0]).to_csv(path, index=False)
    assert cp.compare_runs(logged, path)["status"] == "differs"
    assert cp.compare_runs(logged, path, excluded=cp.EXCLUDED_COLUMNS + ("seconds",))["status"] == "exact"


def test_the_provenance_check_resolves_functions_as_the_script_does():
    import check_provenance as cp
    import elicitation_compare as ec

    stats = ec.bb.load_stats(ec.bb.DEFAULT_STATS_PATH)
    argv = ["--functions", "suite", "--error-models", "bias", "--magnitudes", "1", "--seeds", "7",
            "--iterations", "3"]
    _, clean, names = cp.elicitation_tasks(argv, "clean")
    assert names == ec.landscape_names("suite", stats) == sorted(ec.bb.DEFAULT_SUITE)
    assert {t["dataset"] for t in clean} == set(names) and not any(t["apply_error"] for t in clean)
    # the recorded command's log is the rerun; the old log is checked as a second arm
    assert cp.ELICITATION_ARM == "output-elicitation-rerun"
    assert cp.ELICITATION_LEGACY_ARM == "output-elicitation"


def test_the_comparison_loop_is_checked_from_its_recorded_command():
    import check_provenance as cp

    argv = cp.recorded_elicitation_argv()
    if argv is None:
        pytest.skip("paper/COMMANDS.md records no single elicitation_compare.py command")
    assert "--summary-only" not in argv and "--iterations" in argv
    cell, tasks = cp.elicitation_cell(argv)
    assert len(tasks) == 4
    assert {(t["elicitation"], t["apply_error"]) for t in tasks} == {
        ("rating", True), ("rating", False), ("pairwise", True), ("pairwise", False)}
    assert cell["iterations"] == int(argv[argv.index("--iterations") + 1])
    desc, clean, names = cp.elicitation_tasks(argv, "clean", {"ackley", "branin"})
    assert len(clean) == 4 and not any(t["apply_error"] for t in clean)
    assert {t["dataset"] for t in clean} == {"ackley", "branin"} and desc["dataset"] == "2 landscapes"
    assert {"ackley", "branin"} <= set(names)
    with pytest.raises(ValueError):
        cp.elicitation_tasks(argv, "everything")
    if (REPO / cp.ELICITATION_ARM / cp.ELICITATION_RUNS).is_file():
        check = cp.select(cp.Check(arm=cp.ELICITATION_ARM, driver="elicitation",
                                   arm_dir=str(REPO / cp.ELICITATION_ARM)))
        assert check.logged_path and check.params["dataset"] == cell["dataset"]


def test_the_report_records_the_code_the_reruns_used():
    import check_provenance as cp

    start = {"git_commit": "abc", "code_diff_sha256": "d" * 64, "dirty_files": [" M scripts/a.py"],
             "stamped_at": "t0"}
    same = cp.stamp_columns(start, dict(start))
    assert same == {"checker_git_commit": "abc", "checker_code_diff_sha256": "d" * 64,
                    "checker_n_dirty_files": 1, "checker_code_unchanged": True,
                    "checker_tree_unchanged": True}
    # another file under scripts/ edited concurrently: the tree moved, the reruns' code did not
    moved = cp.stamp_columns(start, {**start, "code_diff_sha256": "e" * 64})
    assert moved["checker_tree_unchanged"] is False and moved["checker_code_unchanged"] is True
    hit = cp.stamp_columns(start, {**start, "code_diff_sha256": "e" * 64}, ["scripts/a.py"])
    assert hit["checker_code_unchanged"] is False
    header = "\n".join(cp.stamp_header(start, dict(start)))
    assert "d" * 64 in header and "scripts/a.py" in header
    assert "CHANGED" in "\n".join(cp.stamp_header(start, dict(start), ["scripts/a.py"]))
    hashes = cp.file_hashes()
    assert "scripts/bo_sensor_error_simulation.py" in hashes and "datasets.json" in hashes
    assert "scripts/check_provenance.py" in cp.imported_repo_files()
    # no path outside the repository reaches a tracked report
    outside = Path.home() / "somewhere" / "prov"
    assert cp._public_arg(str(outside)) == "<outside>/prov"
    assert cp._public_arg(str(REPO / "output-boba" / "x.csv")) == "output-boba/x.csv"
    assert cp._public_arg("output-fitted*-prefix/*=git:3d2323aed~1") == "output-fitted*-prefix/*=git:3d2323aed~1"


def test_the_dataset_config_override_takes_one_entry_from_a_commit(tmp_path):
    import check_provenance as cp

    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout (for instance the anonymous mirror's download)")
    path = cp.materialise_dataset_config("git:HEAD", "ehmi", tmp_path)
    (entry,) = json.loads(path.read_text(encoding="utf-8"))
    assert entry["name"] == "ehmi" and entry["param_columns"]
    plain = tmp_path / "given.json"
    assert cp.materialise_dataset_config(str(plain), "ehmi", tmp_path) == plain


# ---------------------------------------------------------------------------
# drivers, reports and the build (code-recent-11, code-estimands-12, code-sim-repro-16)
# ---------------------------------------------------------------------------


def test_run_oracle_isolation_uses_the_python312_fallback():
    text = (REPO / "run_oracle_isolation.ps1").read_text(encoding="utf-8")
    assert "$env:PYTHON" in text and "$env:HITL_PYTHON" in text
    assert r"Programs\Python\Python312\python.exe" in text
    assert 'else { "python" }' not in text


def test_the_evaluation_report_describes_the_fdr_families_the_code_uses(tmp_path):
    empty = pd.DataFrame()
    ev.write_report(tmp_path, empty, empty, empty, empty, empty)
    text = (tmp_path / "evaluation_report.txt").read_text(encoding="utf-8")
    assert "ONE family covers every condition-level omnibus" in text
    assert "corrected separately" not in text


def test_the_bundle_builder_recognises_a_rerun_request():
    import build_overleaf_bundle as bundle

    assert bundle.needs_rerun("LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.")
    assert bundle.needs_rerun("Package rerunfilecheck Warning: File `main.out' has changed.")
    assert bundle.needs_rerun("Package natbib Warning: Citation(s) may have changed.")
    assert not bundle.needs_rerun("Output written on main.pdf (49 pages, 1 bytes).")
    assert bundle.MAX_PASSES_AFTER_BIBTEX >= 3


def test_the_bundle_builder_requires_no_git_ignored_file():
    import build_overleaf_bundle as bundle

    # README_OVERLEAF.md and authors.tex are git-ignored, so a fresh checkout lacks them;
    # the clean build must still run there (it copies them only when present).
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout (for instance the anonymous mirror's download)")
    assert "README_OVERLEAF.md" in bundle.OPTIONAL_FILES
    required = [f"paper/{name}" for name in bundle.TOP_FILES]
    ignored = subprocess.run(["git", "check-ignore", *required], cwd=REPO, capture_output=True, text=True)
    assert ignored.stdout.strip() == ""


@pytest.mark.parametrize("name", ["c6_ratio", "oracle_companion_estimators", "mo_onset_bound"])
def test_these_checks_take_opt_z_from_the_tracked_statistics_file(name):
    # The main sweep's run_metadata.json is git-ignored and records local paths.
    text = (REPO / "scripts" / "review_checks" / f"{name}.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert '"run_metadata.json"' not in code
    assert "bb.load_stats(bb.DEFAULT_STATS_PATH)" in code


# ---------------------------------------------------------------------------
# an in-process CatBoost fit writes nothing into the working directory
# ---------------------------------------------------------------------------


def test_the_catboost_oracle_writes_no_training_logs(tmp_path, monkeypatch):
    import numpy as np

    pytest.importorskip("catboost")
    monkeypatch.chdir(tmp_path)
    model = sim._build_oracle_model("catboost", seed=7, tree_scale=0.02)
    assert model.get_params()["allow_writing_files"] is False
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(40, 2))
    model.fit(x, x[:, 0] - x[:, 1])
    assert not (tmp_path / "catboost_info").exists()
