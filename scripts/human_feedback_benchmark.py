"""Plan or run finite-candidate human-feedback experiments on another computer.

No arguments, --dry-run and --summarize never run an optimizer or objective.
An experiment requires --run. This is a new scalar, finite-candidate arm;
it does not reproduce the historical continuous BO/MOBO sweep.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from numbers import Integral
from pathlib import Path
import platform
import sys
import time

# Set before importing numerical libraries, including in Windows spawn workers.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from scipy.special import expit
from scipy.stats import qmc, rankdata

import boba_benchmarks as bb
import feedback_policies as fp

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ERRORS = ("gaussian", "bias", "drift", "ar1", "spike", "lapse", "position",
          "contrast", "fatigue", "heteroscedastic", "rater")
RESPONSES = ("final_deployed_regret_true", "final_search_regret_true", "final_selection_loss_true")


@dataclass(frozen=True)
class Settings:
    functions: tuple[str, ...] = ("ackley", "hartmann_3", "powell")
    methods: tuple[str, ...] = ("bo_logei", "random", "sobol", "cem", "ridge_ucb",
                                "dueling_ucb", "reward_policy", "dpo")
    seeds: tuple[int, ...] = (7, 8, 9, 10, 11)
    error_models: tuple[str, ...] = ("gaussian", "drift", "ar1", "spike", "position")
    magnitudes: tuple[float, ...] = (0.25, 1.0, 5.0)
    onsets: tuple[int, ...] = (0, 20)
    budget: int = 50
    candidate_pool: int = 512
    comparison_cost: int = 1
    ar1_rho: float = 0.8
    spike_probability: float = 0.15
    spike_multiplier: float = 20.0
    n_raters: int = 5
    rater_block: int = 10
    rating_clip: tuple[float, float] | None = None
    rating_step: float = 0.0
    policy_options: dict | None = None
    reference: str = "bo_logei"
    reference_rule: str = "observed_max"

    def validate(self, stats: dict) -> None:
        for field in ("functions", "methods", "seeds", "error_models", "magnitudes", "onsets"):
            values = getattr(self, field)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{field} must be nonempty and contain no duplicates")
        unknown = set(self.functions) - (set(stats) & set(bb.BENCHMARKS))
        if unknown or any(bb.BENCHMARKS[n].kind == "stochastic" for n in self.functions if n in bb.BENCHMARKS):
            raise ValueError(f"Unknown or stochastic landscapes: {sorted(unknown)}")
        if set(self.methods) - set(fp.METHODS):
            raise ValueError(f"Unknown methods; choose from {fp.METHODS}")
        if set(self.error_models) - set(ERRORS):
            raise ValueError(f"Unknown error models; choose from {ERRORS}")
        if self.reference not in self.methods:
            raise ValueError("reference must be included in methods")
        if self.reference_rule not in ("policy", "observed_max", "uniform_ties", "rank_mean"):
            raise ValueError("Unknown reference_rule")
        if fp.FEEDBACK[self.reference] == "pairwise" and self.reference_rule != "policy":
            raise ValueError("Pairwise references require reference_rule='policy'")
        for name in ("budget", "candidate_pool", "comparison_cost", "n_raters", "rater_block"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        has_comparisons = any(fp.FEEDBACK[method] == "pairwise" for method in self.methods)
        if self.candidate_pool < 2 or (has_comparisons and self.budget < self.comparison_cost):
            raise ValueError("Need at least two candidates and one comparison's budget")
        if any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("seeds must be unsigned 32-bit integers")
        if any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < self.budget for t in self.onsets):
            raise ValueError("onsets must be integer budget units in [0,budget)")
        if any(not math.isfinite(m) or m < 0 for m in self.magnitudes):
            raise ValueError("magnitudes must be finite and nonnegative")
        if not math.isfinite(self.ar1_rho) or abs(self.ar1_rho) >= 1:
            raise ValueError("ar1_rho must be in (-1,1)")
        if not 0 <= self.spike_probability <= 1 or not math.isfinite(self.spike_multiplier) or self.spike_multiplier < 0:
            raise ValueError("Invalid spike parameters")
        if not math.isfinite(self.rating_step) or self.rating_step < 0:
            raise ValueError("rating_step must be finite and nonnegative")
        if self.rating_clip is not None:
            if len(self.rating_clip) != 2 or not all(math.isfinite(x) for x in self.rating_clip) or self.rating_clip[0] >= self.rating_clip[1]:
                raise ValueError("rating_clip must be finite increasing bounds in landscape SDs")
        for name in self.functions:
            if not all(math.isfinite(stats[name][key]) for key in ("mean", "std", "opt_z")) or min(stats[name]["std"], stats[name]["opt_z"]) <= 0:
                raise ValueError(f"Invalid landscape scale: {name}")
        if self.policy_options is not None:
            if not isinstance(self.policy_options, dict) or set(self.policy_options) - set(self.methods):
                raise ValueError("policy_options must map selected method names to option dictionaries")
            if any(not isinstance(v, dict) for v in self.policy_options.values()):
                raise ValueError("Each policy_options entry must be a dictionary")
        for method in self.methods:
            fp.validate_policy_options(method, (self.policy_options or {}).get(method, {}))


def load_settings(path: Path | None) -> Settings:
    raw = json.loads(path.read_text(encoding="utf-8")) if path else {}
    for name in ("functions", "methods", "seeds", "error_models", "magnitudes", "onsets", "rating_clip"):
        if raw.get(name) is not None:
            raw[name] = tuple(raw[name])
    return Settings(**raw)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def stream_seed(seed: int, purpose: str) -> int:
    """Names, not mutable list positions, identify independent random streams.

    Landscape is deliberately omitted to preserve shared random numbers across
    landscapes. Catalog, policy, and feedback streams are independent.
    """
    return int.from_bytes(hashlib.sha256(f"{seed}:{purpose}".encode()).digest()[:4], "little")


def candidate_catalog(dim: int, size: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(dim, scramble=True, seed=stream_seed(seed, "catalog"))
    return sampler.random_base2(math.ceil(math.log2(size)))[:size]


@lru_cache(maxsize=8)
def _catalog_environment(name: str, seed: int, size: int, stats_json: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Reuse one immutable environment across a worker's methods and fault cells.

    Only standardization and the recorded optimum form part of the key; unused
    diagnostics can legitimately contain NaN for numerical-supremum extensions.
    The bounded cache never holds policies or rater state, and changing the
    objective scale cannot reuse an earlier truth array.
    """
    entry = json.loads(stats_json)
    oracle = bb.SyntheticOracle.from_stats(name, {name: entry})
    X = candidate_catalog(oracle.spec.dim, size, seed)
    physical = oracle.bounds_low + X * (oracle.bounds_high - oracle.bounds_low)
    truth = np.asarray(oracle.predict_many(physical), dtype=float).ravel()
    if len(truth) != size or not np.isfinite(truth).all():
        raise ValueError(f"Invalid oracle values for {name}")
    X.flags.writeable = False
    truth.flags.writeable = False
    y_opt = float(oracle.y_opt)
    if not math.isfinite(y_opt):
        # Bump extensions mark spec.y_best as -inf because their optimum is
        # numerical. The statistics store the verified bound in normalized units.
        y_opt = float(entry["opt_z"])
    if not math.isfinite(y_opt):
        raise ValueError(f"No finite recorded optimum for {name}")
    return X, truth, y_opt


class Rater:
    """Extension-arm error definitions; see error-specific units below.

    Instrument clipping/rounding applies to clean AND noisy twins. Pairwise
    perception is simultaneous; shared offsets cancel before instrumentation.
    Lapse magnitude is a probability capped at one; contrast is a dimensionless
    gain. Other magnitudes are landscape SDs. Clip/round can introduce ties.
    """
    def __init__(self, settings: Settings, seed: int, error: str, magnitude: float, onset: int):
        if error not in ERRORS or not math.isfinite(magnitude) or magnitude < 0:
            raise ValueError("Rater requires a known error model and a finite nonnegative magnitude")
        if isinstance(onset, bool) or not isinstance(onset, Integral) or not 0 <= onset < settings.budget:
            raise ValueError("Rater onset must be an integer in [0,budget)")
        self.s, self.error, self.magnitude, self.onset = settings, error, magnitude, onset
        self.rng = np.random.default_rng(stream_seed(seed, "feedback:" + error))
        self.offsets = np.random.default_rng(stream_seed(seed, "raters")).normal(size=settings.n_raters)
        self.state: float | None = None
        self.previous: float | None = None

    def observe(self, values: np.ndarray, spent: int) -> tuple[float, np.ndarray]:
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) not in (1, 2) or not np.isfinite(values).all():
            raise ValueError("A sitting must expose one or two finite objective values")
        if isinstance(spent, bool) or not isinstance(spent, Integral) or not 0 <= spent < self.s.budget:
            raise ValueError("spent must be integer budget units in [0,budget)")
        perceived = values.copy()
        m = self.magnitude if spent >= self.onset else 0.0
        age = max(0, spent - self.onset)
        fraction = age / max(1, self.s.budget - self.onset - 1)
        if self.error in ("gaussian", "fatigue", "heteroscedastic"):
            scale = m * (1 + 2 * fraction if self.error == "fatigue" else 1)
            if self.error == "heteroscedastic":
                scale = m * (0.25 + expit(values))
            perceived += self.rng.normal(size=len(values)) * scale
        elif self.error == "bias":
            perceived += m
        elif self.error == "drift":
            perceived += m * fraction
        elif self.error == "ar1":
            z = float(self.rng.normal())
            if spent >= self.onset:
                self.state = m * z if self.state is None else self.s.ar1_rho * self.state + m * math.sqrt(1-self.s.ar1_rho**2) * z
                perceived += self.state
        elif self.error == "spike":
            perceived += self.rng.normal(size=len(values)) * m
            hits = self.rng.random(len(values)) < self.s.spike_probability
            perceived += hits * self.rng.normal(size=len(values)) * m * self.s.spike_multiplier
        elif self.error == "lapse":
            if self.rng.random() < min(m, 1.0):
                perceived = self.rng.normal(size=len(values))
        elif self.error == "position":
            # Single designs have no left/right presentation position.
            if len(values) == 2:
                perceived += np.array([m / 2, -m / 2])
        elif self.error == "contrast":
            if self.previous is not None:
                perceived += m * (values - self.previous)
        elif self.error == "rater":
            rater = (spent // self.s.rater_block) % self.s.n_raters
            perceived += m * self.offsets[rater]
        if self.s.rating_step:
            perceived = np.round(perceived / self.s.rating_step) * self.s.rating_step
        if self.s.rating_clip is not None:
            perceived = np.clip(perceived, *self.s.rating_clip)
        self.previous = float(values.mean())
        if len(values) == 1:
            feedback = float(perceived[0])
        else:
            feedback = 1.0 if perceived[0] > perceived[1] else 0.0 if perceived[0] < perceived[1] else 0.5
        return feedback, perceived


def rank_recommendation(X: np.ndarray, indices: list[int], ratings: list[float]) -> int:
    """Zero-query RBF ridge refit on rating ranks, restricted to visited designs."""
    unique = list(dict.fromkeys(indices))
    train = X[indices]
    kernel = np.exp(-np.sum((train[:, None] - train[None])**2, axis=2) / (2 * 0.25**2))
    targets = (rankdata(ratings) - 0.5) / len(ratings) - 0.5
    weights = np.linalg.solve(kernel + np.eye(len(indices)), targets)
    cross = np.exp(-np.sum((X[unique, None] - train[None])**2, axis=2) / (2 * 0.25**2))
    return unique[int(np.argmax(cross @ weights))]


def score_choices(policy, X: np.ndarray, indices: list[int], ratings: list[float], truth: np.ndarray) -> dict:
    choices = {"policy": float(truth[policy.recommend()])}
    if ratings:
        scores = np.asarray(ratings)
        top = np.flatnonzero(scores == scores.max())
        choices["observed_max"] = float(truth[indices[int(top[0])]])
        # Exact expectation, without consuming another RNG stream.
        tied_designs = list(dict.fromkeys(np.asarray(indices)[top].tolist()))
        choices["uniform_ties"] = float(np.mean(truth[tied_designs]))
        choices["rank_mean"] = float(truth[rank_recommendation(X, indices, ratings)])
    return choices


def run_task(task: dict, settings: Settings, stats: dict) -> dict:
    """Worker entry point. Called only by the explicit --run path."""
    start = time.perf_counter()
    name, method, seed = task["dataset"], task["method"], task["seed"]
    # Ground truth is private to the environment and scorer, never a policy input.
    environment_stats = {key: stats[name][key] for key in ("mean", "std", "opt_z") if key in stats[name]}
    X, truth, y_opt = _catalog_environment(
        name, seed, settings.candidate_pool, json.dumps(environment_stats, sort_keys=True, allow_nan=False))
    options = (settings.policy_options or {}).get(method, {})
    policy = fp.make_policy(method, X, stream_seed(seed, "policy:" + method), **options)
    presentation = np.random.default_rng(stream_seed(seed, "presentation"))
    rater = Rater(settings, seed, task["error_model"], task["magnitude"], task["onset"])
    pairwise = fp.FEEDBACK[method] == "pairwise"
    cost = settings.comparison_cost if pairwise else 1
    seen: list[int] = []
    rated: list[int] = []
    ratings: list[float] = []
    trace = []
    for spent in range(0, settings.budget - cost + 1, cost):
        ids = tuple(policy.ask())
        if len(ids) != (2 if pairwise else 1) or any(
                isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < len(X)
                for i in ids):
            raise ValueError(f"Invalid proposal from {method}: {ids}")
        ids = tuple(map(int, ids))
        if pairwise and ids[0] == ids[1]:
            raise ValueError("A pairwise query must show two distinct designs")
        if pairwise and presentation.random() < 0.5:
            ids = ids[::-1]
        feedback, perceived = rater.observe(truth[list(ids)], spent)
        policy.tell(ids, feedback)
        seen.extend(ids)
        if not pairwise:
            rated.append(ids[0])
            ratings.append(feedback)
        recommendation = policy.recommend()
        if isinstance(recommendation, (bool, np.bool_)) or not isinstance(recommendation, Integral) or recommendation not in seen:
            raise ValueError(f"{method} recommended an unseen design")
        recommendation = int(recommendation)
        best = float(truth[seen].max())
        trace.append({"budget_spent": spent+cost, "indices": ids, "feedback": feedback,
                      "perceived": perceived.tolist(), "recommended": recommendation,
                      "simple_regret_true": float(y_opt-best),
                      "inference_simple_regret_true": float(y_opt-truth[recommendation])})
    best = float(truth[seen].max())
    choices = score_choices(policy, X, rated, ratings, truth)
    rows = []
    for rule, shipped in choices.items():
        rows.append({"ship_rule": rule, "final_search_regret_true": float(y_opt-best),
                     "final_deployed_regret_true": float(y_opt-shipped),
                     "final_selection_loss_true": float(best-shipped)})
    return {"schema": SCHEMA_VERSION, "task": task, "rows": rows, "trace": trace,
            "y_opt": y_opt, "opt_z": float(stats[name]["opt_z"]),
            "catalog_best_true": float(truth.max()), "n_judgments": len(trace),
            "design_exposures": len(seen), "unique_designs": len(set(seen)),
            "budget_spent": len(trace)*cost, "seconds": time.perf_counter()-start}


def build_tasks(settings: Settings) -> list[dict]:
    tasks = []
    for name in settings.functions:
        for seed in settings.seeds:
            for method in settings.methods:
                base = {"dataset": name, "seed": seed, "method": method}
                # Clean computation happens once, then is joined to every cell.
                tasks.append({**base, "clean": True, "error_model": "gaussian", "magnitude": 0.0, "onset": 0})
                for error in settings.error_models:
                    for magnitude in settings.magnitudes:
                        for onset in settings.onsets:
                            tasks.append({**base, "clean": False, "error_model": error,
                                          "magnitude": magnitude, "onset": onset})
    return tasks


def experiment_spec(settings: Settings, stats_path: Path) -> dict:
    paths = ("human_feedback_benchmark.py", "feedback_policies.py", "boba_benchmarks.py",
             "analyse_boba_adaptations.py", "compare_boba_arms.py")
    sources = {p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest() for p in paths}
    versions = {}
    for name in ("numpy", "scipy", "pandas", "statsmodels", "torch", "botorch", "gpytorch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"schema": SCHEMA_VERSION, "settings": asdict(settings), "source_sha256": sources,
            "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def prepare_output(output: Path, spec: dict, resume: bool) -> None:
    manifest = output / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(spec)):
            raise ValueError("Output configuration, source code, stats or environment changed; use a fresh output directory")
        if not resume:
            raise ValueError("Output already exists; use --resume or a fresh directory")
    else:
        if resume:
            raise ValueError("Cannot resume without a manifest")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output is nonempty and has no manifest; choose a fresh directory")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, spec)
    (output / "runs").mkdir(exist_ok=True)


def checkpoint_path(output: Path, task: dict) -> Path:
    return output / "runs" / (digest(task) + ".json")


def read_checkpoint(output: Path, task: dict, settings: Settings | None = None) -> dict:
    """Refuse incomplete or inconsistent runs before they can enter a summary."""
    def require(condition: bool, reason: str) -> None:
        if not condition:
            raise ValueError(reason)

    def finite(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    try:
        result = json.loads(checkpoint_path(output, task).read_text(encoding="utf-8"))
        require(isinstance(result, dict) and result.get("schema") == SCHEMA_VERSION
                and not isinstance(result.get("schema"), bool) and result.get("task") == task,
                "identity or schema mismatch")
        trace, rows = result["trace"], result["rows"]
        require(isinstance(trace, list) and bool(trace) and isinstance(rows, list) and bool(rows),
                "missing trace or final metrics")
        pairwise = fp.FEEDBACK[task["method"]] == "pairwise"
        arity = 2 if pairwise else 1
        n = result["n_judgments"]
        require(isinstance(n, int) and not isinstance(n, bool) and n == len(trace), "judgment count mismatch")
        cost = settings.comparison_cost if settings is not None and pairwise else 1
        if settings is not None:
            require(n == settings.budget // cost, "incomplete judgment budget")
        else:
            cost = trace[0]["budget_spent"]
        require(isinstance(cost, int) and not isinstance(cost, bool) and cost > 0, "invalid query cost")
        require(result["budget_spent"] == n * cost, "spent budget mismatch")
        require(result["design_exposures"] == arity * n, "exposure count mismatch")
        require(all(isinstance(result[k], int) and not isinstance(result[k], bool)
                    for k in ("budget_spent", "design_exposures", "unique_designs")),
                "counts must be integers")
        for key in ("opt_z", "y_opt", "catalog_best_true", "seconds"):
            require(finite(result[key]), f"nonfinite {key}")
        require(result["opt_z"] > 0 and result["seconds"] >= 0, "invalid scale or duration")
        require(result["catalog_best_true"] <= result["y_opt"] + 1e-8, "catalog exceeds the recorded optimum")
        seen = set()
        previous_search = math.inf
        for i, step in enumerate(trace, 1):
            require(step["budget_spent"] == i * cost, "noncontiguous budget trace")
            ids, perceived = step["indices"], step["perceived"]
            require(isinstance(ids, list) and len(ids) == arity and len(set(ids)) == arity,
                    "invalid query arity")
            require(all(isinstance(j, int) and not isinstance(j, bool) and j >= 0
                        and (settings is None or j < settings.candidate_pool) for j in ids),
                    "invalid design index")
            require(isinstance(perceived, list) and len(perceived) == arity and all(map(finite, perceived)),
                    "invalid perceived values")
            seen.update(ids)
            require(isinstance(step["recommended"], int) and not isinstance(step["recommended"], bool)
                    and step["recommended"] in seen, "unseen recommendation")
            require(all(finite(step[k]) for k in
                        ("feedback", "simple_regret_true", "inference_simple_regret_true")),
                    "nonfinite trace metrics")
            expected = (1.0 if perceived[0] > perceived[1] else 0.0 if perceived[0] < perceived[1] else 0.5) if pairwise else perceived[0]
            require(step["feedback"] == expected, "feedback disagrees with perception")
            search = step["simple_regret_true"]
            require(search <= previous_search + 1e-8
                    and step["inference_simple_regret_true"] >= search - 1e-8,
                    "invalid search/deployed regret ordering")
            previous_search = search
        require(result["unique_designs"] == len(seen), "unique-design count mismatch")
        expected_rules = {"policy"} if pairwise else {"policy", "observed_max", "uniform_ties", "rank_mean"}
        require(len(rows) == len(expected_rules) and {row["ship_rule"] for row in rows} == expected_rules,
                "missing or duplicate ship rules")
        for row in rows:
            require(all(finite(row[k]) for k in RESPONSES), "nonfinite final metrics")
            search, deployed, selection = (row[k] for k in
                                          ("final_search_regret_true", "final_deployed_regret_true", "final_selection_loss_true"))
            require(math.isclose(search, previous_search, abs_tol=1e-8, rel_tol=1e-8)
                    and selection >= -1e-8
                    and math.isclose(deployed, search + selection, abs_tol=1e-8, rel_tol=1e-8),
                    "final regret decomposition mismatch")
            if row["ship_rule"] == "policy":
                require(math.isclose(deployed, trace[-1]["inference_simple_regret_true"], abs_tol=1e-8, rel_tol=1e-8),
                        "final policy metric disagrees with trace")
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid checkpoint for {task}: {exc}; move it aside and resume") from exc


def execute(tasks: list[dict], settings: Settings, stats: dict, output: Path, workers: int) -> None:
    pending = []
    for task in tasks:
        if checkpoint_path(output, task).exists():
            read_checkpoint(output, task, settings)
        else:
            pending.append(task)
    print(f"{len(tasks)-len(pending)} checkpoints reused; {len(pending)} tasks pending", flush=True)
    if workers == 1:
        for i, task in enumerate(pending, 1):
            atomic_json(checkpoint_path(output, task), run_task(task, settings, stats))
            print(f"Completed {i}/{len(pending)}", flush=True)
        return
    # Bounded queue: a large sweep does not retain thousands of Future objects.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = iter(pending)
        futures = {}
        for task in list(next(iterator, None) for _ in range(min(len(pending), 2*workers))):
            futures[pool.submit(run_task, task, settings, stats)] = task
        count = 0
        while futures:
            future = next(as_completed(futures))
            task = futures.pop(future)
            atomic_json(checkpoint_path(output, task), future.result())
            count += 1
            print(f"Completed {count}/{len(pending)}", flush=True)
            task = next(iterator, None)
            if task is not None:
                futures[pool.submit(run_task, task, settings, stats)] = task


def summarize(output: Path) -> None:
    """Fail closed on missing runs; reuse the repository's scientific estimand."""
    import pandas as pd
    from analyse_boba_adaptations import paired_frame, summarise, share_flags, per_dataset

    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest at {manifest_path}; no experiments were started")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = manifest["settings"]
    settings = Settings(**raw)
    tasks = build_tasks(settings)
    missing = [task for task in tasks if not checkpoint_path(output, task).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing checkpoints; resume explicitly with --run --resume")
    rows = []
    clean = {}
    # Keep scalar clean metrics only. Holding every run's trace here can consume
    # gigabytes on a full screen, although the summary never uses those traces.
    for task in tasks:
        if not task["clean"]:
            continue
        record = read_checkpoint(output, task, settings)
        t = record["task"]
        for r in record["rows"]:
            clean[t["dataset"], t["seed"], t["method"], r["ship_rule"]] = (r, record["opt_z"])
    for task in tasks:
        if task["clean"]:
            continue
        record = read_checkpoint(output, task, settings)
        t = record["task"]
        for result in record["rows"]:
            twin, clean_scale = clean[t["dataset"], t["seed"], t["method"], result["ship_rule"]]
            if clean_scale != record["opt_z"]:
                raise ValueError("Clean and noisy checkpoint scales differ")
            row = {**t, "ship_rule": result["ship_rule"], "opt_z": record["opt_z"],
                   "magnitude_unit": magnitude_unit(t["error_model"]),
                   **{k: record[k] for k in ("n_judgments", "design_exposures", "unique_designs", "budget_spent", "seconds")}}
            for response in RESPONSES:
                row[response + "_jitter"] = result[response]
                row[response + "_baseline"] = twin[response]
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_run_metrics.csv", index=False)
    ref = frame[(frame.method == settings.reference) & (frame.ship_rule == settings.reference_rule)].copy()
    reference_label = f"{settings.reference}:{settings.reference_rule}"
    scales = frame[["dataset", "opt_z"]].drop_duplicates()
    if scales.dataset.duplicated().any():
        raise ValueError("Checkpoint scales differ within a landscape")
    scale = dict(scales.itertuples(index=False, name=None))
    summary, per_landscape = [], []
    for (method, rule), treatment in frame.groupby(["method", "ship_rule"], sort=True):
        for response in RESPONSES:
            def pairing_view(f):
                return f.rename(columns={"magnitude": "jitter_std", "onset": "jitter_iteration"}).assign(acquisition="fixed_reference")
            paired = paired_frame(pairing_view(ref), pairing_view(treatment), response, scale, pool=False)
            if len(paired) != len(ref) or len(paired) != len(treatment):
                raise ValueError("Incomplete reference/treatment pairing")
            for (error, magnitude, onset), block in paired.groupby(["error_model", "jitter_std", "jitter_iteration"], sort=True):
                label = {"method": method, "ship_rule": rule, "reference": reference_label,
                         "response": response, "unit": "fraction_of_opt_z", "time_base": "final_budget",
                         "error_model": error, "magnitude": magnitude, "onset": onset,
                         "magnitude_unit": magnitude_unit(error)}
                result = summarise(block, np.random.default_rng(20261020))
                flags = share_flags(block)
                result.pop("wilcoxon_p", None)  # This extension defines no confirmatory testing family.
                if flags["recovered_suppressed"] or response == "final_selection_loss_true":
                    for key in ("recovered", "recovered_lo", "recovered_hi"):
                        result[key] = np.nan
                    if response == "final_selection_loss_true":
                        flags["recovered_suppressed"] = "selection_component_report_absolute_difference"
                means = block.groupby("dataset")[["ref_noisy", "ref_clean", "trt_noisy", "trt_clean"]].mean()
                interval_rng = np.random.default_rng(20261021)
                indices = interval_rng.integers(0, len(means), (2000, len(means)))
                intervals = {}
                for quantity, values in (("gain", means.ref_noisy-means.trt_noisy),
                                         ("price", means.trt_clean-means.ref_clean)):
                    low, high = np.percentile(values.to_numpy()[indices].mean(axis=1), [2.5, 97.5])
                    intervals[quantity+"_lo"], intervals[quantity+"_hi"] = low, high
                summary.append({**label, **result, **flags, **intervals,
                                **{key: float(value) for key, value in means.mean().items()},
                                "magnitude_unit": magnitude_unit(error),
                                "bootstrap_assumption": "landscapes treated as independent despite shared random streams"})
                for row in per_dataset(block).to_dict("records"):
                    if row["cost"] < 0.01 or response == "final_selection_loss_true":
                        row["recovered"] = np.nan
                        row["recovery_suppressed"] = "selection component" if response == "final_selection_loss_true" else "nonpositive or near-zero cost"
                    else:
                        row["recovery_suppressed"] = ""
                    per_landscape.append({**label, **row})
    pd.DataFrame(summary).to_csv(output / "comparison_summary.csv", index=False)
    pd.DataFrame(per_landscape).to_csv(output / "comparison_per_landscape.csv", index=False)
    print(f"Wrote metrics and standard-process comparisons to {output}")


def magnitude_unit(error: str) -> str:
    return "probability_capped_at_one" if error == "lapse" else "dimensionless_gain" if error == "contrast" else "landscape_SD"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output-feedback-screen"))
    parser.add_argument("--stats-path", type=Path, default=bb.DEFAULT_STATS_PATH)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if sys.platform == "win32" and args.workers > 61:
        parser.error("Windows ProcessPoolExecutor supports at most 61 workers")
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if args.summarize:
        summarize(args.output_dir)
        return
    settings = load_settings(args.config)
    stats = bb.load_stats(args.stats_path)
    settings.validate(stats)
    tasks = build_tasks(settings)
    print(json.dumps({"mode": "run" if args.run else "dry-run", "tasks": len(tasks),
                      "clean_tasks": sum(t["clean"] for t in tasks), "workers": args.workers,
                      "reference": settings.reference + ":" + settings.reference_rule,
                      "budget": settings.budget, "comparison_cost": settings.comparison_cost,
                      "functions": settings.functions, "methods": settings.methods,
                      "output": str(args.output_dir)}, indent=2))
    if not args.run:
        return
    prepare_output(args.output_dir, experiment_spec(settings, args.stats_path), args.resume)
    execute(tasks, settings, stats, args.output_dir, args.workers)
    summarize(args.output_dir)


if __name__ == "__main__":
    main()
