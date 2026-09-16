"""Multi-objective landscapes with a KNOWN maximum hypervolume.

The BOBA suite is scalar, so the single-objective arm cannot speak to the
multi-objective setting the applied work actually runs in. This module supplies
the multi-objective analogue on the same terms: problems whose optimum is a
published constant rather than an estimate.

BoTorch ships several multi-objective test problems together with ``_max_hv``,
the hypervolume of the true Pareto front measured against a fixed reference
point. That is exactly the ``y_opt`` the simulator needs, so regret here is a
real regret for the same reason it is in the scalar arm.

Standardisation carries over exactly
------------------------------------
Each objective is standardised over the design box, as in the scalar arm, so an
injected error of one unit means the same thing on every objective of every
problem. Hypervolume is a volume in objective space, so under the per-objective
affine map ``y_m -> (y_m - mu_m) / sigma_m`` --- with the reference point mapped
the same way --- it scales by exactly ``1 / prod_m sigma_m``. The standardised
maximum hypervolume is therefore ``max_hv / prod_m sigma_m``, known in closed
form rather than sampled. ``verify_reference_point`` checks the arithmetic
against a direct Pareto computation before any of it is used.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MO_STATS_PATH = REPO_ROOT / "boba_mo_stats.json"

STATS_LOG2_SAMPLES = 15
STATS_SEED = 20260906


@dataclasses.dataclass(frozen=True)
class MultiObjectiveSpec:
    """One BoTorch multi-objective problem, as this study uses it."""

    name: str
    botorch_name: str
    dim: int
    num_objectives: int
    notes: str = ""

    @property
    def param_columns(self) -> list[str]:
        return [f"x{i}" for i in range(self.dim)]

    @property
    def objective_columns(self) -> list[str]:
        return [f"f{i}" for i in range(self.num_objectives)]


# Chosen to span the two axes that plausibly matter: the number of objectives
# (2, 3, 4) and whether the problem is a synthetic construction or an
# engineering model. Every one has a published maximum hypervolume.
_MO_SPECS = [
    MultiObjectiveSpec("branincurrin", "BraninCurrin", 2, 2,
                       "The canonical two-objective BO benchmark."),
    MultiObjectiveSpec("zdt1", "ZDT1", 6, 2, "Convex front, the ZDT baseline."),
    MultiObjectiveSpec("dtlz2", "DTLZ2", 6, 2, "Concave spherical front."),
    MultiObjectiveSpec("dh2", "DH2", 6, 2,
                       "Built to be hard for hypervolume methods: a disconnected front."),
    MultiObjectiveSpec("vehiclesafety", "VehicleSafety", 5, 3,
                       "Crashworthiness model; three objectives, engineering scales."),
    MultiObjectiveSpec("penicillin", "Penicillin", 7, 3,
                       "Fermentation process model; three objectives."),
    MultiObjectiveSpec("carsideimpact", "CarSideImpact", 7, 4,
                       "Four objectives -- the memory-heavy end of the hypervolume methods."),
]

MO_BENCHMARKS: dict[str, MultiObjectiveSpec] = {s.name: s for s in _MO_SPECS}
MO_ORDER = [s.name for s in _MO_SPECS]

_PROBLEM_CACHE: dict[str, object] = {}


def _problem(name: str) -> object:
    """The negated BoTorch problem, built once."""
    if name not in _PROBLEM_CACHE:
        import torch
        import botorch.test_functions.multi_objective as mo

        spec = MO_BENCHMARKS[name]
        cls = getattr(mo, spec.botorch_name)
        try:
            problem = cls(negate=True)
        except TypeError:
            problem = cls(dim=spec.dim, negate=True)
        if problem.dim != spec.dim or problem.num_objectives != spec.num_objectives:
            raise ValueError(
                f"{name}: BoTorch reports d={problem.dim}, M={problem.num_objectives}, "
                f"registry says d={spec.dim}, M={spec.num_objectives}"
            )
        _PROBLEM_CACHE[name] = problem.to(dtype=torch.double)
    return _PROBLEM_CACHE[name]


def bounds(name: str) -> tuple[np.ndarray, np.ndarray]:
    problem = _problem(name)
    box = problem.bounds.detach().cpu().numpy()
    return box[0].astype(float), box[1].astype(float)


def evaluate(name: str, X: np.ndarray) -> np.ndarray:
    """Raw objective values, shape (n, M). Negated, so larger is better."""
    import torch

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    problem = _problem(name)
    with torch.no_grad():
        values = problem(torch.as_tensor(X, dtype=torch.double))
    return np.asarray(values.cpu().numpy(), dtype=float).reshape(X.shape[0], -1)


def max_hypervolume(name: str) -> float:
    return float(_problem(name)._max_hv)


def reference_point(name: str) -> np.ndarray:
    return np.asarray(_problem(name).ref_point.detach().cpu().numpy(), dtype=float)


def sobol_sample(name: str, log2_n: int = STATS_LOG2_SAMPLES, seed: int = STATS_SEED) -> np.ndarray:
    from scipy.stats import qmc

    low, high = bounds(name)
    engine = qmc.Sobol(d=len(low), scramble=True, seed=seed)
    return low + engine.random_base2(m=log2_n) * (high - low)


def mo_stats(name: str, log2_n: int = STATS_LOG2_SAMPLES, seed: int = STATS_SEED) -> dict:
    """Per-objective standardisation constants and the standardised optimum."""
    spec = MO_BENCHMARKS[name]
    X = sobol_sample(name, log2_n=log2_n, seed=seed)
    Y = evaluate(name, X)
    mean = Y.mean(axis=0)
    std = Y.std(axis=0, ddof=1)
    if not np.all(std > 0):
        raise ValueError(f"{name}: an objective is constant over the box ({std})")

    ref = reference_point(name)
    ref_std = (ref - mean) / std
    # Hypervolume is a volume in objective space, so the per-objective affine
    # map scales it by exactly the product of the reciprocal scales.
    scale = float(np.prod(std))
    return {
        "name": name,
        "botorch_name": spec.botorch_name,
        "dim": int(spec.dim),
        "num_objectives": int(spec.num_objectives),
        "n_samples": int(2**log2_n),
        "seed": int(seed),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "ref_point_raw": ref.tolist(),
        "ref_point": ref_std.tolist(),
        "max_hv_raw": max_hypervolume(name),
        "max_hv": max_hypervolume(name) / scale,
        "hv_scale": scale,
        "sampled_hv": sampled_hypervolume(name, X, Y, mean, std, ref_std),
    }


def sampled_hypervolume(name: str, X: np.ndarray, Y: np.ndarray,
                        mean: np.ndarray, std: np.ndarray,
                        ref_std: np.ndarray) -> float:
    """Hypervolume of the standardised Sobol sample, as a sanity floor."""
    import torch
    from botorch.utils.multi_objective import is_non_dominated
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    Y_std = torch.tensor((Y - mean) / std, dtype=torch.double)
    pareto = Y_std[is_non_dominated(Y_std)]
    # Exact hypervolume is exponential in the front size beyond two objectives,
    # and a 32k-point sample can leave thousands of non-dominated points. This
    # number is only a sanity floor -- "the sample does not already exceed the
    # published optimum" -- so a capped subsample is enough, and subsampling can
    # only ever make the floor lower.
    cap = 400
    if pareto.shape[0] > cap:
        idx = torch.linspace(0, pareto.shape[0] - 1, cap).round().long()
        pareto = pareto[idx]
    hv = Hypervolume(ref_point=torch.tensor(ref_std, dtype=torch.double))
    return float(hv.compute(pareto))


def verify_reference_point(name: str, stats: dict) -> dict:
    """Check the closed-form standardised optimum against a direct computation.

    The scaling identity ``HV_std = HV_raw / prod(sigma)`` is exact, but it is
    exact only if the reference point is mapped with the same affine map, so it
    is worth checking rather than asserting. Compares the two on the sampled
    Pareto front, where both can be computed.
    """
    import torch
    from botorch.utils.multi_objective import is_non_dominated
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    X = sobol_sample(name, log2_n=10, seed=STATS_SEED + 1)
    Y = evaluate(name, X)

    mean = np.asarray(stats["mean"], dtype=float)
    std = np.asarray(stats["std"], dtype=float)

    raw = torch.tensor(Y, dtype=torch.double)
    hv_raw = Hypervolume(ref_point=torch.tensor(stats["ref_point_raw"], dtype=torch.double))
    direct = float(hv_raw.compute(raw[is_non_dominated(raw)]))

    scaled = torch.tensor((Y - mean) / std, dtype=torch.double)
    hv_std = Hypervolume(ref_point=torch.tensor(stats["ref_point"], dtype=torch.double))
    transformed = float(hv_std.compute(scaled[is_non_dominated(scaled)]))

    predicted = direct / stats["hv_scale"]
    return {
        "direct_raw_hv": direct,
        "transformed_hv": transformed,
        "predicted_from_scaling": predicted,
        "relative_error": abs(transformed - predicted) / max(abs(predicted), 1e-12),
    }


class SyntheticMultiOracle:
    """A multi-objective analytic problem, standardised per objective.

    Duck-types the simulator's oracle interface: ``predict`` returns a length-M
    vector and ``predict_many`` an (n, M) array, which is what
    ``run_simulation`` expects in its ``multi_objective`` branch. Module-level
    class, so it survives the pickling Windows' spawn multiprocessing does.
    """

    def __init__(self, name: str, mean: np.ndarray, std: np.ndarray):
        if name not in MO_BENCHMARKS:
            raise KeyError(f"Unknown multi-objective benchmark: {name}")
        self.name = name
        self.spec = MO_BENCHMARKS[name]
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        if self.mean.shape != (self.spec.num_objectives,):
            raise ValueError(f"{name}: mean has shape {self.mean.shape}")
        if not np.all(self.std > 0):
            raise ValueError(f"{name}: non-positive standardisation scale {self.std}")
        self.objective_name = "multi_objective"
        self.objective_columns = self.spec.objective_columns
        self.param_columns = self.spec.param_columns
        self._low, self._high = bounds(name)

    @classmethod
    def from_stats(cls, name: str, stats: dict) -> "SyntheticMultiOracle":
        entry = stats[name]
        return cls(name, np.asarray(entry["mean"]), np.asarray(entry["std"]))

    @property
    def bounds_low(self) -> np.ndarray:
        return self._low

    @property
    def bounds_high(self) -> np.ndarray:
        return self._high

    def _in_box(self, X: np.ndarray) -> np.ndarray:
        # BoTorch's problems reject out-of-box input down to the last bit, and
        # the acquisition optimiser can return a corner a few ulps outside.
        return np.clip(np.asarray(X, dtype=float), self._low, self._high)

    def predict(self, x: np.ndarray) -> np.ndarray:
        raw = evaluate(self.name, self._in_box(x).reshape(1, -1))[0]
        return (raw - self.mean) / self.std

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        raw = evaluate(self.name, self._in_box(X))
        return (raw - self.mean) / self.std

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SyntheticMultiOracle({self.name!r}, d={self.spec.dim}, "
                f"M={self.spec.num_objectives})")


def load_mo_stats(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_MO_STATS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Multi-objective statistics not found at {path}. Generate them with:\n"
            f"  python scripts/boba_multiobjective.py --output {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))["problems"]


def main() -> None:
    import argparse
    import platform
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MO_STATS_PATH)
    parser.add_argument("--problems", type=str, default=None)
    parser.add_argument("--log2-samples", type=int, default=STATS_LOG2_SAMPLES)
    parser.add_argument("--seed", type=int, default=STATS_SEED)
    args = parser.parse_args()

    names = ([n.strip() for n in args.problems.split(",") if n.strip()]
             if args.problems else list(MO_ORDER))
    unknown = [n for n in names if n not in MO_BENCHMARKS]
    if unknown:
        raise SystemExit(f"Unknown problem(s): {unknown}")

    problems = {}
    for name in names:
        entry = mo_stats(name, log2_n=args.log2_samples, seed=args.seed)
        check = verify_reference_point(name, entry)
        entry["scaling_check_relative_error"] = check["relative_error"]
        if check["relative_error"] > 1e-9:
            raise SystemExit(
                f"{name}: the standardised hypervolume does not match the scaling "
                f"identity (relative error {check['relative_error']:.3e}). Refusing to "
                f"write statistics that would make regret wrong."
            )
        problems[name] = entry
        print(
            f"{name:<16s} d={entry['dim']:<2d} M={entry['num_objectives']:<2d} "
            f"max_hv_raw={entry['max_hv_raw']:<12.5g} max_hv_std={entry['max_hv']:<12.5g} "
            f"sampled={entry['sampled_hv']:<10.4g} check={check['relative_error']:.1e}"
        )

    payload = {
        "log2_samples": int(args.log2_samples),
        "seed": int(args.seed),
        "python_version": sys.version,
        "platform": platform.platform(),
        "problems": problems,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
