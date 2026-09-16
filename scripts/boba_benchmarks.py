"""The BOBA benchmark suite, vectorised over numpy, with known optima.

Why this module exists
----------------------
Everything else in this repository stands a *regression oracle fitted to archival
study data* in for the human in the loop. That oracle is the weakest link in
every claim the project makes: its held-out R^2 is 0.55 at best on the eHMI data
and near zero on ProVoice, so "how much does feedback error cost you" is always
entangled with "how badly is the surrogate human mis-specified".

These benchmarks remove that confound. Each is an analytic function with a
*published* optimum, so:

  * the true objective is exact rather than a fitted approximation;
  * ``y_opt`` is a literature constant rather than a random-search estimate,
    which means regret is a real regret and not a quantity BO can accidentally
    make negative;
  * the landscape geometry (dimension, ruggedness, sparsity, skew) is known and
    can be used as a *predictor* of fragility to feedback error.

Provenance
----------
The suite, its boxes, its dimensions and its recorded optima are BOBA's
(``BOBA/parallel_main.py``: ``SIMULATION_FUNCTIONS`` / ``BOUNDS`` / ``DIMS`` /
``Y_BEST``; ``BOBA/bayes_opt/simulation.py``: the definitions). They are copied
here rather than imported so this repository stays self-contained and so the
functions can be evaluated on whole arrays at once -- BOBA's versions take one
torch point at a time, which is fine for its own harness but ~1000x too slow for
the Sobol sweeps this study needs. ``tests/test_boba_benchmarks.py`` asserts the
copies agree with BOBA's originals to 1e-9 on random points, so the vendoring
cannot silently drift.

BOBA is a *dynamic* BO project: its last input coordinate is time, swept over the
same interval as every other coordinate. This study is static, so the time axis
is treated as one more ordinary design dimension. That is faithful -- the box and
the function are unchanged -- but it means, e.g., ``hartmann_3`` here is the
standard 3-D Hartmann rather than a 2-D slice of it.

Scaling
-------
Output scales span nearly seven orders of magnitude across the suite (``hartmann_6``
tops out at 3.3; ``powell`` reaches -6e4). A single sweep of absolute error
magnitudes would therefore mean "imperceptible" on one function and "destroys the
signal" on the next. Every function is consequently reported in *standardised*
units: ``g(x) = (f(x) - mu_f) / sigma_f`` with ``mu_f, sigma_f`` estimated once
from a fixed scrambled-Sobol sample of the box (see ``landscape_stats``). The map
is affine and strictly increasing, so it leaves the optimisation problem itself
untouched -- same argmax, same ordering of designs, and BoTorch's ``Standardize``
outcome transform would have applied something very close to it anyway. What it
buys is that an injected error of 0.5 means "half the landscape's own standard
deviation" on every function in the suite.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_STATS_PATH = REPO_ROOT / "boba_landscape_stats.json"

# The scrambled-Sobol sample used for every landscape statistic. Fixed here (not
# a CLI knob) because the standardisation constants must be identical across
# every run in the study, including runs started weeks apart.
STATS_LOG2_SAMPLES = 16          # 65536 points
STATS_SEED = 20260906


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FunctionSpec:
    """One benchmark: its box, its published optimum, and how it behaves.

    ``dim`` is the TOTAL input dimension, i.e. BOBA's spatial ``DIMS`` plus its
    time axis. ``lo``/``hi`` apply to every coordinate, as in BOBA.
    """

    name: str
    dim: int
    lo: float
    hi: float
    y_best: float
    y_best_source: str
    kind: str          # "analytic" | "botorch" | "stochastic"
    optimum: str       # "interior" | "boundary"
    notes: str = ""
    # A known (or strongly suspected) argmax, as a single coordinate value
    # repeated across the box. verify_optimum polishes from here as well as from
    # its Sobol screen: a narrow spike is invisible to any finite screen, so
    # without a hint the "verified optimum" of a needle landscape is just the
    # background maximum -- which would silently under-state y_opt and make
    # regret negative for anything that finds the needle.
    argmax_hint: float | None = None

    @property
    def bounds_low(self) -> np.ndarray:
        return np.full(self.dim, self.lo, dtype=float)

    @property
    def bounds_high(self) -> np.ndarray:
        return np.full(self.dim, self.hi, dtype=float)

    @property
    def param_columns(self) -> list[str]:
        return [f"x{i}" for i in range(self.dim)]


# ---------------------------------------------------------------------------
# Analytic benchmarks. Every one is a MAXIMISATION objective (BOBA negates the
# classical minimisation forms), matching this repository's convention.
# X is always (n, d); the return is always (n,).
# ---------------------------------------------------------------------------


def _as2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(1, -1) if X.ndim == 1 else X


def schwefel(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    d = X.shape[-1]
    return -(418.9829 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=-1))


def eggholder(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    x0, x1 = X[:, 0], X[:, 1]
    return -(
        -(x1 + 47) * np.sin(np.sqrt(np.abs(x0 / 2 + (x1 + 47))))
        - x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
    )


def ackley(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = X.shape[-1]
    sum1 = np.sum(X**2, axis=-1)
    sum2 = np.sum(np.cos(c * X), axis=-1)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return -(term1 + term2 + a + np.e)


_SHEKEL_A = np.array(
    [
        [4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [6.0, 6.0, 6.0, 6.0],
        [3.0, 7.0, 3.0, 7.0],
        [2.0, 9.0, 2.0, 9.0],
        [5.0, 3.0, 5.0, 3.0],
        [8.0, 1.0, 8.0, 1.0],
        [6.0, 2.0, 6.0, 2.0],
        [7.0, 3.6, 7.0, 3.6],
    ],
    dtype=float,
)
_SHEKEL_C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5], dtype=float)


def shekel(X: np.ndarray, m: int = 10) -> np.ndarray:
    X = _as2d(X)
    diff = X[:, None, :] - _SHEKEL_A[None, :m, :]
    return np.sum(1.0 / (np.sum(diff**2, axis=-1) + _SHEKEL_C[None, :m]), axis=-1)


def griewank(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    d = X.shape[-1]
    idx = np.arange(1, d + 1, dtype=float)
    sum_term = np.sum(X**2, axis=-1) / 4000.0
    prod_term = np.prod(np.cos(X / np.sqrt(idx)), axis=-1)
    return -(sum_term - prod_term + 1.0)


_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)
_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]], dtype=float)
_H3_P = 1e-4 * np.array(
    [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]],
    dtype=float,
)


def hartmann_3(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    inner = np.sum(_H3_A[None, :, :] * (X[:, None, :] - _H3_P[None, :, :]) ** 2, axis=-1)
    total = np.sum(-_H3_ALPHA[None, :] * np.exp(-inner), axis=-1)
    return -total


_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)
_H6_A = np.array(
    [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ],
    dtype=float,
)
_H6_P = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ],
    dtype=float,
)


def hartmann_6(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    inner = np.sum(_H6_A[None, :, :] * (X[:, None, :] - _H6_P[None, :, :]) ** 2, axis=-1)
    return np.sum(_H6_ALPHA[None, :] * np.exp(-inner), axis=-1)


def powell(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    n = X.shape[-1]
    if n % 4 != 0:
        raise ValueError(f"powell needs a multiple of 4 inputs, got {n}")
    total = np.zeros(X.shape[0], dtype=float)
    for i in range(0, n, 4):
        total += (
            (X[:, i] + 10 * X[:, i + 1]) ** 2
            + 5 * (X[:, i + 2] - X[:, i + 3]) ** 2
            + (X[:, i + 1] - 2 * X[:, i + 2]) ** 4
            + 10 * (X[:, i] - X[:, i + 3]) ** 4
        )
    return -total


def yerkes_dodson(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    return np.mean(np.exp(-((X - 0.5) ** 2) / (2 * 0.15**2)), axis=-1)


def stevens(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    return np.mean(np.clip(X, 0.0, None) ** 0.67, axis=-1)


def hicks_law(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    return -np.mean(0.2 + 0.15 * np.log2(np.clip(X, 0.0, None) + 1.0), axis=-1)


def weber_fechner(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    return np.mean(np.log1p(np.clip(X, 0.0, None) / 0.01), axis=-1)


_MP_HEIGHTS = np.array([0.50, 0.60, 0.70, 0.80, 0.90], dtype=float)
_MP_AMP = 0.10
_MP_WIDTH = 3.0
_MP_PHASE = np.array([0.00, 0.37, 0.61, 0.13, 0.85], dtype=float)
_MP_ANGLE = np.array([0.00, 1.26, 2.51, 3.77, 5.03], dtype=float)
_MP_RADIUS = 0.30


def moving_peaks(X: np.ndarray, severity: float = 1.0) -> np.ndarray:
    X = _as2d(X)
    xs, t = X[:, :-1], X[:, -1]
    d = xs.shape[-1]
    two_pi = 2.0 * math.pi
    coord_phase = two_pi * np.arange(d, dtype=float) / max(d, 1)
    # (n, peaks)
    h = _MP_HEIGHTS[None, :] + _MP_AMP * np.sin(two_pi * (t[:, None] + _MP_PHASE[None, :]))
    ang = two_pi * t[:, None] * severity + _MP_ANGLE[None, :]
    # (n, peaks, d)
    centres = 0.5 + _MP_RADIUS * np.cos(ang[:, :, None] + coord_phase[None, None, :])
    dist = np.linalg.norm(xs[:, None, :] - centres, axis=-1)
    return np.max(h - _MP_WIDTH * dist, axis=-1)


def power_law_practice(X: np.ndarray, n_max: float = 30.0) -> np.ndarray:
    X = _as2d(X)
    xs, t = X[:, :-1], X[:, -1]
    m = np.mean(np.clip(xs, 0.0, 1.0), axis=-1)
    B = 1.0 + 3.0 * m
    alpha = 0.05 + 0.5 * m
    N = 1.0 + np.clip(t, 0.0, 1.0) * n_max
    return -(B * N ** (-alpha))


def steering_law(X: np.ndarray, a: float = 0.1, lam: float = 0.6) -> np.ndarray:
    X = _as2d(X)
    xs, t = X[:, :-1], X[:, -1]
    d = xs.shape[-1]
    even = xs[:, 0::2] if d > 1 else xs
    odd = xs[:, 1::2] if d > 1 else xs
    A = 0.5 + np.mean(np.clip(odd, 0.0, 1.0), axis=-1)
    W = 0.05 + 0.95 * np.mean(np.clip(even, 0.0, 1.0), axis=-1)
    b = 0.5 * (1.0 - 0.5 * np.clip(t, 0.0, 1.0))
    MT = a + b * A / W
    return -(MT + lam * A * W)


# ---------------------------------------------------------------------------
# BoTorch-delegated benchmarks. BOBA delegates these rather than transcribing
# the formulas, and so does this module: a transcription error would be silent
# and would corrupt every number computed against it.
# ---------------------------------------------------------------------------

_BOTORCH_CACHE: dict[tuple[str, int], object] = {}


def _botorch_problem(name: str, dim: int) -> object:
    key = (name, dim)
    if key not in _BOTORCH_CACHE:
        import torch
        import botorch.test_functions.synthetic as syn

        cls = getattr(syn, name)
        try:
            prob = cls(dim=dim, negate=True)
        except TypeError:  # fixed-dimension problems take no dim
            prob = cls(negate=True)
            if prob.dim != dim:
                raise ValueError(f"{name} is {prob.dim}-D, got {dim} inputs")
        _BOTORCH_CACHE[key] = prob.to(dtype=torch.double)
    return _BOTORCH_CACHE[key]


def _botorch_eval(name: str, X: np.ndarray) -> np.ndarray:
    import torch

    X = _as2d(X)
    prob = _botorch_problem(name, X.shape[-1])
    with torch.no_grad():
        v = prob(torch.as_tensor(X, dtype=torch.double), noise=False)
    return np.asarray(v.reshape(-1).cpu().numpy(), dtype=float)


def botorch_optimal_value(name: str, dim: int) -> float:
    return float(_botorch_problem(name, dim).optimal_value)


def branin(X: np.ndarray) -> np.ndarray:
    return _botorch_eval("Branin", X)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    return _botorch_eval("Rosenbrock", X)


def rastrigin(X: np.ndarray) -> np.ndarray:
    return _botorch_eval("Rastrigin", X)


def michalewicz(X: np.ndarray) -> np.ndarray:
    return _botorch_eval("Michalewicz", X)


def levy_10(X: np.ndarray) -> np.ndarray:
    return _botorch_eval("Levy", X)


# ---------------------------------------------------------------------------
# The one stochastic benchmark. It needs BOBA's own modules (a fitted typing
# model with Monte-Carlo rollouts), so it is imported lazily from a BOBA
# checkout and is excluded from the default suite -- see DEFAULT_SUITE.
# ---------------------------------------------------------------------------

_TYPING_USER = None
_BOBA_ROOT: Path | None = None


def set_boba_root(path: str | Path | None) -> None:
    """Point the ``typing`` benchmark at a BOBA checkout (or disable it)."""
    global _BOBA_ROOT, _TYPING_USER
    _BOBA_ROOT = Path(path).resolve() if path is not None else None
    _TYPING_USER = None


def _typing_user():
    global _TYPING_USER
    if _TYPING_USER is None:
        import sys

        import torch

        if _BOBA_ROOT is None:
            raise RuntimeError(
                "The 'typing' benchmark needs a BOBA checkout; call set_boba_root(...) "
                "or pass --boba-root."
            )
        if str(_BOBA_ROOT) not in sys.path:
            sys.path.insert(0, str(_BOBA_ROOT))
        from bayes_opt.naf.typing_user import PARAM_SETS, TypingUser, sentence_for

        gen = torch.Generator().manual_seed(42)
        _TYPING_USER = TypingUser(
            PARAM_SETS["midair"]["means"],
            sentence=sentence_for(0),
            n_rollouts=5,
            generator=gen,
            dtype=torch.double,
        )
    return _TYPING_USER


def typing(X: np.ndarray) -> np.ndarray:
    """Soft-keyboard typing performance, 0.7*speed + 0.3*accuracy.

    Stochastic (Monte-Carlo rollouts) and without a published optimum, so it is
    the one member of the suite that is NOT a known function. Kept available
    because it is the only objective here fitted to real people.
    """
    import torch

    X = _as2d(X)
    user = _typing_user()
    out = np.empty(X.shape[0], dtype=float)
    for i, row in enumerate(X):
        y = torch.as_tensor(user.evaluate(torch.as_tensor(row[:2], dtype=torch.double))).flatten()
        out[i] = float(0.7 * y[0] + 0.3 * y[1])
    return out


# ---------------------------------------------------------------------------
# Extensions. NOT part of BOBA: two purpose-built families that exist to break
# confounds the BOBA suite cannot break on its own. They are excluded from
# DEFAULT_SUITE and from the BOBA-parity tests, and must be requested by name.
# ---------------------------------------------------------------------------


def _levy_at(dim: int):
    """Levy in a chosen dimension.

    BOBA ships Levy only at d=11, which makes ``dim`` a single-point covariate:
    one high-dimensional landscape against nineteen low-dimensional ones, and
    every other property differing at the same time. Running the SAME function
    at 4, 7 and 11 turns dimension from a confounded covariate into a
    three-point manipulation inside one landscape family.
    """

    def f(X: np.ndarray) -> np.ndarray:
        return _botorch_eval("Levy", X)

    f.__name__ = f"levy_{dim}d"
    return f


_BUMP_CENTRE = 0.37   # off-centre so the bump is not on a symmetry axis
_BUMP_DIM = 4


def _bump_at(amplitude: float, width: float):
    """A broad background plus one narrow spike, on [0,1]^4.

        g(x) = cos-field(x) + amplitude * exp(-||x - c||^2 / (2 width^2))

    The deepest confound in the BOBA suite is that ``opt_z`` (how far the
    optimum stands above a random design), sparsity, and skew are one axis:
    Spearman 0.87-0.92 among them across all 21 functions. That is close to a
    structural identity for a bounded function with an isolated optimum, so no
    amount of adding real benchmarks breaks it -- which is exactly why BOBA
    never could, and why a correlational claim about "skew predicts fragility"
    is untestable on that suite.

    Here the two are separately dialled. The background sets the landscape's
    standard deviation, so after standardisation ``amplitude`` moves ``opt_z``
    while leaving the bulk of the distribution alone, and ``width`` moves how
    much of the box sits near the top while leaving ``opt_z`` roughly fixed. A
    grid over the two turns the correlation into a manipulation.
    """

    def f(X: np.ndarray) -> np.ndarray:
        X = _as2d(X)
        d = X.shape[-1]
        idx = np.arange(1, d + 1, dtype=float)
        background = np.sum(np.cos(2.0 * np.pi * idx * X), axis=-1) / math.sqrt(d)
        distance2 = np.sum((X - _BUMP_CENTRE) ** 2, axis=-1)
        return background + amplitude * np.exp(-distance2 / (2.0 * width**2))

    f.__name__ = f"bump_a{amplitude:g}_w{width:g}"
    return f


LEVY_LADDER_DIMS = (4, 7)          # 11 is already in the BOBA suite as levy_10

# --- the dimension ladder at matched signal strength -------------------------
# The Levy ladder confounds dimension with signal strength: opt_z is 1.53, 2.07
# and 2.62 at d = 4, 7, 11. It cannot be fixed by resizing the box, because
# opt_z = (y_best - mu) / sigma is invariant to any affine map of the objective
# and, measured over [-h, h]^d, is flat in h for Levy (1.56, 2.13, 2.72 at
# h = 200 against 1.53, 2.07, 2.62 at h = 10). opt_z at a given dimension is a
# property of the function, so a matched ladder needs a family with a knob.
#
# The bump family has two. Fixing the spike's volume fraction w^d at 1e-6 keeps
# sparsity comparable across dimensions, and the amplitude is then solved per
# dimension to put opt_z at 5.0. Both cannot be held fixed by accident -- in
# higher dimensions an isolated optimum is necessarily sparser -- so this is two
# knobs against two targets, and dimension is what is left varying.
# Solved numerically against the 2^16 Sobol sample and this module's own
# verify_optimum, which polishes from the argmax hint rather than taking the
# bump's peak value. The target is opt_z = 9.0 rather than something smaller
# because opt_z is DISCONTINUOUS in amplitude: below a threshold the cos-field
# background's maximum (which grows as sqrt(d)) beats the spike and y_opt stops
# depending on the amplitude at all, so at d = 11 the reachable values jump from
# 4.67 straight to 8.70. Nine is above the threshold at every dimension. The
# absolute level is arbitrary; only its being matched matters.
LADDER_VOLUME_FRACTION = 1e-6
LADDER = ((4, 6.8430), (7, 6.0934), (11, 4.2474))   # (dim, amplitude)

# Measured 2x2 (65,536-point Sobol, the standardisation sample):
#   a=4  w=0.05 -> opt_z  4.97, sparsity 0.00000
#   a=4  w=0.15 -> opt_z  5.20, sparsity 0.00041
#   a=16 w=0.05 -> opt_z 21.68, sparsity 0.00000
#   a=16 w=0.15 -> opt_z 12.26, sparsity 0.00018
# So amplitude moves opt_z ~4x at fixed width while width moves sparsity at
# roughly fixed opt_z -- a real but PARTIAL decoupling: a broad bump also
# inflates the landscape's variance, which pulls opt_z back down (at w=0.30 the
# effect of amplitude is gone entirely, which is why 0.30 is not in the grid).
# Amplitudes below ~4 do not work at all: the bump no longer dominates the
# cos-field background and both descriptors end up driven by the background.
BUMP_AMPLITUDES = (4.0, 16.0)
BUMP_WIDTHS = (0.05, 0.15)


FUNCTIONS = {
    "schwefel": schwefel,
    "powell": powell,
    "eggholder": eggholder,
    "ackley": ackley,
    "shekel": shekel,
    "griewank": griewank,
    "hartmann_3": hartmann_3,
    "hartmann_6": hartmann_6,
    "branin": branin,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "michalewicz": michalewicz,
    "yerkes_dodson": yerkes_dodson,
    "typing": typing,
    "stevens": stevens,
    "hicks_law": hicks_law,
    "weber_fechner": weber_fechner,
    "moving_peaks": moving_peaks,
    "power_law_practice": power_law_practice,
    "steering_law": steering_law,
    "levy_10": levy_10,
}

# dim is TOTAL inputs = BOBA's DIMS + 1 (its time axis, treated here as an
# ordinary design dimension). Boxes, dims and optima are BOBA's
# parallel_main.py BOUNDS / DIMS / Y_BEST, verbatim.
_SPECS = [
    FunctionSpec("schwefel", 4, -500.0, 500.0, 0.0, "published", "analytic", "interior",
                 "Deceptive: the global optimum sits far from the second-best basin."),
    FunctionSpec("powell", 4, -4.0, 5.0, 0.0, "published", "analytic", "interior",
                 "Extremely heavy-tailed: quartic terms span five orders of magnitude."),
    FunctionSpec("eggholder", 2, -512.0, 512.0, 959.6407, "published", "analytic", "boundary",
                 "Optimum on the x0=512 face."),
    FunctionSpec("ackley", 4, -32.768, 32.768, 0.0, "published", "analytic", "interior",
                 "Flat outer plateau with a narrow central funnel."),
    FunctionSpec("shekel", 4, 0.0, 10.0, 10.536443, "published", "analytic", "interior",
                 "Ten narrow wells in an otherwise flat box; very sparse."),
    FunctionSpec("griewank", 6, -600.0, 600.0, 0.0, "published", "analytic", "interior",
                 "Near-quadratic at this box size; the cosine ripple is negligible."),
    FunctionSpec("hartmann_3", 3, 0.0, 1.0, 3.86278, "published", "analytic", "interior", ""),
    FunctionSpec("hartmann_6", 6, 0.0, 1.0, 3.32237, "published", "analytic", "interior", ""),
    FunctionSpec("branin", 2, 0.0, 10.0, -0.397887, "botorch", "botorch", "interior",
                 "BOBA restricts Branin to [0,10]^2; one of its three optima lies inside."),
    FunctionSpec("rosenbrock", 4, -5.0, 10.0, 0.0, "botorch", "botorch", "interior",
                 "Smooth curved valley, effectively unimodal, enormous value range."),
    FunctionSpec("rastrigin", 4, -5.12, 5.12, 0.0, "botorch", "botorch", "interior",
                 "Regular lattice of local optima on a quadratic bowl."),
    FunctionSpec("michalewicz", 5, 0.0, math.pi, 4.687658, "botorch", "botorch", "interior",
                 "Steep ridges separated by flat plateaus; the sparsest landscape here."),
    FunctionSpec("yerkes_dodson", 4, 0.0, 1.0, 1.0, "published", "analytic", "interior",
                 "Inverted-U (Yerkes & Dodson 1908); the shape most human responses take."),
    FunctionSpec("typing", 3, 0.0, 1.0, 1.0, "normalisation constant", "stochastic", "unknown",
                 "Monte-Carlo; no published optimum. NOT a known function -- excluded by default."),
    FunctionSpec("stevens", 4, 0.0, 1.0, 1.0, "published", "analytic", "boundary",
                 "Monotone psychophysical law: optimum at the upper corner."),
    FunctionSpec("hicks_law", 4, 0.0, 10.0, -0.2, "published", "analytic", "boundary",
                 "Monotone: optimum at the lower corner."),
    FunctionSpec("weber_fechner", 4, 0.0, 1.0, 4.61512051684126, "published", "analytic", "boundary",
                 "Monotone: optimum at the upper corner."),
    FunctionSpec("moving_peaks", 4, 0.0, 1.0, 1.0, "construction", "analytic", "interior",
                 "Branke (1999). Five orbiting cones; piecewise-linear, non-smooth ridges."),
    # BOBA records -0.216422 here, which is -4*201**-0.55: the supremum under the
    # ORIGINAL n_max=200. The function now ships n_max=30, whose supremum is
    # -4*31**-0.55 = -0.6050775778, attained at the corner mean(x)=1, t=1 (log f
    # is concave in mean(x), so the optimum is an endpoint, and t=1 always wins).
    # BOBA's constant is therefore stale by 0.389 -- 64% relative -- and is
    # corrected here rather than copied.
    FunctionSpec("power_law_practice", 4, 0.0, 1.0, -0.6050775778340285,
                 "analytic supremum (corrects BOBA's stale -0.216422)", "analytic", "boundary",
                 "Newell & Rosenbloom (1981). Moving corner optimum."),
    FunctionSpec("steering_law", 4, 0.0, 1.0, -0.487298, "numerical supremum", "analytic", "interior",
                 "Accot & Zhai (1997) plus a screen-area penalty."),
    FunctionSpec("levy_10", 11, -10.0, 10.0, 0.0, "botorch", "botorch", "interior",
                 "Dimension probe: nothing else in the suite exceeds six inputs."),
]

BENCHMARKS: dict[str, FunctionSpec] = {spec.name: spec for spec in _SPECS}

# BOBA's own ordering, for reproducibility of any index-based seeding.
BOBA_ORDER = [spec.name for spec in _SPECS]

# --- extension families (see _levy_at / _bump_at) --------------------------
EXTENSION_ORDER: list[str] = []

for _dim in LEVY_LADDER_DIMS:
    _name = f"levy_{_dim}d"
    FUNCTIONS[_name] = _levy_at(_dim)
    BENCHMARKS[_name] = FunctionSpec(
        _name, _dim, -10.0, 10.0, 0.0, "botorch", "botorch", "interior",
        f"Dimension ladder: Levy at d={_dim}, same family as levy_10 (d=11).",
    )
    EXTENSION_ORDER.append(_name)

for _a in BUMP_AMPLITUDES:
    for _w in BUMP_WIDTHS:
        _name = f"bump_a{_a:g}_w{_w:g}"
        FUNCTIONS[_name] = _bump_at(_a, _w)
        # The optimum is the bump peak on top of whatever the background gives
        # there; the background is a bounded cos-field so the supremum is found
        # numerically by verify_optimum rather than published.
        BENCHMARKS[_name] = FunctionSpec(
            _name, _BUMP_DIM, 0.0, 1.0, float("-inf"), "numerical supremum",
            "analytic", "interior",
            f"Confound-breaking family: amplitude {_a:g} sets opt_z, width {_w:g} sets sparsity.",
            argmax_hint=_BUMP_CENTRE,
        )
        EXTENSION_ORDER.append(_name)

for _dim, _amp in LADDER:
    _w = LADDER_VOLUME_FRACTION ** (1.0 / _dim)
    _name = f"bump_d{_dim}"
    FUNCTIONS[_name] = _bump_at(_amp, _w)
    BENCHMARKS[_name] = FunctionSpec(
        _name, _dim, 0.0, 1.0, float("-inf"), "numerical supremum",
        "analytic", "interior",
        f"Matched ladder: d={_dim}, amplitude {_amp:g} and width {_w:.4f} chosen so "
        f"opt_z is 9.0 and the spike's volume fraction is 1e-6 at every dimension.",
        argmax_hint=_BUMP_CENTRE,
    )
    EXTENSION_ORDER.append(_name)

ALL_ORDER = BOBA_ORDER + EXTENSION_ORDER

# The default suite excludes `typing`: it is stochastic and has no published
# optimum, so including it in a study whose whole premise is "known functions"
# would reintroduce exactly the oracle uncertainty this module removes. Run it
# explicitly (--functions typing --boba-root ...) as a labelled side arm.
DEFAULT_SUITE = [name for name in BOBA_ORDER if BENCHMARKS[name].kind != "stochastic"]


def evaluate(name: str, X: np.ndarray) -> np.ndarray:
    """Evaluate a benchmark on X of shape (n, d) or (d,); returns shape (n,)."""
    spec = BENCHMARKS[name]
    X = _as2d(X)
    if X.shape[-1] != spec.dim:
        raise ValueError(f"{name} needs {spec.dim} inputs, got {X.shape[-1]}")
    return np.asarray(FUNCTIONS[name](X), dtype=float).reshape(-1)


# ---------------------------------------------------------------------------
# Landscape statistics and standardisation
# ---------------------------------------------------------------------------


def sobol_sample(spec: FunctionSpec, log2_n: int = STATS_LOG2_SAMPLES, seed: int = STATS_SEED) -> np.ndarray:
    """A scrambled-Sobol design over the function's box, 2**log2_n points."""
    from scipy.stats import qmc

    engine = qmc.Sobol(d=spec.dim, scramble=True, seed=seed)
    unit = engine.random_base2(m=log2_n)
    return spec.lo + unit * (spec.hi - spec.lo)


def verify_optimum(
    name: str,
    log2_starts: int = 15,
    n_polish: int = 60,
    seed: int = STATS_SEED,
) -> dict[str, float]:
    """Intensively search the box and compare with the recorded optimum.

    Sobol screen, then L-BFGS-B from the best ``n_polish`` starts. Returns the
    best value found and the signed gap to ``spec.y_best``. A POSITIVE gap means
    the search could not reach the recorded optimum -- usually because the
    landscape is too rugged for a local polisher (griewank, levy_10), which is
    benign; a large positive gap on a smooth function means the recorded constant
    is wrong, which is how BOBA's stale ``power_law_practice`` value was caught.
    """
    from scipy.optimize import minimize
    from scipy.stats import qmc

    spec = BENCHMARKS[name]
    engine = qmc.Sobol(d=spec.dim, scramble=True, seed=seed)
    X0 = spec.lo + engine.random_base2(m=log2_starts) * (spec.hi - spec.lo)
    y0 = evaluate(name, X0)
    order = np.argsort(-y0)[:n_polish]

    best = float(y0[order[0]])
    negated = lambda z: -float(evaluate(name, z.reshape(1, -1))[0])
    box = [(spec.lo, spec.hi)] * spec.dim
    starts = [X0[i] for i in order]
    if spec.argmax_hint is not None:
        hint = np.full(spec.dim, float(spec.argmax_hint))
        best = max(best, float(evaluate(name, hint.reshape(1, -1))[0]))
        starts.insert(0, hint)
    for start in starts:
        result = minimize(
            negated, start, method="L-BFGS-B", bounds=box,
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-12},
        )
        best = max(best, float(-result.fun))

    if not np.isfinite(spec.y_best):
        # No published constant (the bump family): the numerical supremum IS the
        # optimum. Padding it by a few ulps keeps regret non-negative.
        best_padded = best + 1e-9 * max(abs(best), 1.0)
        return {
            "best_found": best,
            "recorded_y_best": float("nan"),
            "gap": float("nan"),
            "relative_gap": float("nan"),
            "y_opt": float(best_padded),
        }
    gap = spec.y_best - best
    scale = max(abs(spec.y_best), abs(best), 1e-12)
    return {
        "best_found": best,
        "recorded_y_best": float(spec.y_best),
        "gap": float(gap),
        "relative_gap": float(gap / scale),
        # The optimiser is never allowed to be handed a target it can beat: a
        # y_opt below the attainable maximum would make regret negative and
        # censor the metric. Taking the max costs at worst a constant offset,
        # which cancels exactly in the excess-regret contrasts.
        "y_opt": float(max(spec.y_best, best)),
    }


def selection_fragility(
    name: str,
    noise_levels: tuple[float, ...] = (0.05, 0.25, 0.5, 1.0, 2.0, 5.0),
    n_candidates: int = 256,
    n_draws: int = 4000,
    seed: int = STATS_SEED,
) -> dict[str, float]:
    """One-shot cost of picking the wrong candidate because of observation noise.

    Over a fixed Sobol candidate set, the loss from choosing ``argmax_i (g_i +
    eps_i)`` instead of ``argmax_i g_i``, with ``eps ~ N(0, c^2)`` in standardised
    units and the expectation taken over 4000 draws.

    This is the mechanism the whole study is about, isolated from BO: no GP, no
    sequence, no acquisition function -- just "how much does an error of size c
    cost you when you have to pick". If the BO results track it, "which
    landscapes are fragile" reduces to "what makes this quantity large", which is
    a far sharper claim than a correlation with skew. If they diverge, the gap
    between one-shot and sequential fragility is itself the result -- a GP fitted
    to 50 observations can average noise down in a way one-shot selection cannot.
    """
    spec = BENCHMARKS[name]
    from scipy.stats import qmc

    engine = qmc.Sobol(d=spec.dim, scramble=True, seed=seed + 7)
    X = spec.lo + engine.random(n_candidates) * (spec.hi - spec.lo)
    y = evaluate(name, X)

    reference = landscape_stats.__dict__.get("_scaling", {}).get(name)
    if reference is None:
        sample = evaluate(name, sobol_sample(spec, log2_n=STATS_LOG2_SAMPLES, seed=seed))
        reference = (float(np.mean(sample)), float(np.std(sample, ddof=1)))
    mean, std = reference
    g = (y - mean) / max(std, 1e-300)
    best = float(np.max(g))

    rng = np.random.default_rng(seed + 11)
    out: dict[str, float] = {}
    for c in noise_levels:
        if c <= 0:
            out[f"frag_{c:g}"] = 0.0
            continue
        noise = rng.normal(0.0, c, size=(n_draws, n_candidates))
        picked = np.argmax(g[None, :] + noise, axis=1)
        out[f"frag_{c:g}"] = float(np.mean(best - g[picked]))
    return out


def landscape_stats(
    name: str,
    log2_n: int = STATS_LOG2_SAMPLES,
    seed: int = STATS_SEED,
) -> dict[str, float]:
    """Descriptors of one landscape, from a fixed scrambled-Sobol sample.

    ``mean``/``std`` are the standardisation constants. The rest are the
    candidate predictors for "which landscapes are fragile to feedback error":

      * ``skew``, ``excess_kurtosis`` -- shape of the value distribution.
      * ``tail_ratio`` = std / (1.4826 * MAD). 1.0 for a Gaussian landscape;
        large when a few extreme values dominate the spread (powell, rosenbrock).
      * ``sparsity_10pct`` -- fraction of the box within 10% of the sampled range
        below the optimum. Small = needle-in-a-haystack.
      * ``opt_z`` -- how many landscape SDs the optimum sits above the mean. This
        is the natural "signal" against which an injected error SD is a "noise".
      * ``ruggedness`` -- 1 - lag-1 autocorrelation of f along random walks of
        step 2% of the box diagonal. 0 = smooth, 1 = white noise.
    """
    spec = BENCHMARKS[name]
    X = sobol_sample(spec, log2_n=log2_n, seed=seed)
    y = evaluate(name, X)

    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1))
    mad = float(np.median(np.abs(y - np.median(y))))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    span = max(y_max - y_min, 1e-300)

    # Skew / excess kurtosis without a scipy dependency at call time.
    z = (y - mean) / max(std, 1e-300)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)

    # Sparsity is measured against the VERIFIED optimum, not the sample maximum:
    # on a sparse landscape a Sobol sample never lands near the optimum, so
    # anchoring on the sample max would report every landscape as equally dense.
    verification = verify_optimum(name, seed=seed)
    top = verification["y_opt"]
    threshold = top - 0.10 * max(top - y_min, 1e-300)
    sparsity = float(np.mean(y >= threshold))
    # Anchoring on the true optimum makes the measure degenerate: on a spiky
    # landscape a Sobol sample never comes near it, so eight of the twenty
    # benchmarks return exactly 0.0 and cannot be ranked or logged. The
    # sample-anchored version is bounded below by 1/n and stays usable, at the
    # cost of measuring "how much of the box is near the best point FOUND"
    # rather than "near the optimum". Both are reported.
    sampled_threshold = y_max - 0.10 * max(y_max - y_min, 1e-300)
    sparsity_sampled = float(np.mean(y >= sampled_threshold))

    # Ruggedness: short random walks, lag-1 autocorrelation of the value series.
    rng = np.random.default_rng(seed + 1)
    n_walks, walk_len = 256, 64
    step = 0.02 * (spec.hi - spec.lo) * math.sqrt(spec.dim)
    starts = spec.lo + rng.random((n_walks, spec.dim)) * (spec.hi - spec.lo)
    series = np.empty((n_walks, walk_len), dtype=float)
    pos = starts.copy()
    for t in range(walk_len):
        series[:, t] = evaluate(name, pos)
        direction = rng.normal(size=(n_walks, spec.dim))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        pos = np.clip(pos + step * direction, spec.lo, spec.hi)
    centred = series - series.mean(axis=1, keepdims=True)
    num = np.sum(centred[:, :-1] * centred[:, 1:], axis=1)
    den = np.sum(centred**2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho1 = np.where(den > 0, num / den, 0.0)
    ruggedness = float(1.0 - np.mean(rho1))

    # Hand the already-measured standardisation constants to the fragility
    # estimator so it does not redraw the 65k Sobol sample per call.
    landscape_stats.__dict__.setdefault("_scaling", {})[name] = (mean, std)
    fragility = selection_fragility(name, seed=seed)

    return {
        "name": name,
        "dim": int(spec.dim),
        "n_samples": int(2**log2_n),
        "seed": int(seed),
        "mean": mean,
        "std": std,
        "mad_scale": 1.4826 * mad,
        "tail_ratio": float(std / max(1.4826 * mad, 1e-300)),
        "min": y_min,
        "max": y_max,
        "range": float(span),
        "skew": skew,
        "excess_kurtosis": kurt,
        "y_best": float(spec.y_best),
        "y_best_source": spec.y_best_source,
        "y_opt": verification["y_opt"],
        "opt_best_found": verification["best_found"],
        "opt_gap": verification["gap"],
        "opt_relative_gap": verification["relative_gap"],
        "opt_z": float((verification["y_opt"] - mean) / max(std, 1e-300)),
        "sampled_opt_gap_z": float((verification["y_opt"] - y_max) / max(std, 1e-300)),
        "sparsity_10pct": sparsity,
        "sparsity_sampled": sparsity_sampled,
        "neg_log_sparsity_sampled": float(-np.log10(max(sparsity_sampled, 1e-300))),
        "ruggedness": ruggedness,
        **fragility,
    }


def compute_all_stats(
    names: list[str] | None = None,
    log2_n: int = STATS_LOG2_SAMPLES,
    seed: int = STATS_SEED,
) -> dict[str, dict[str, float]]:
    names = names if names is not None else DEFAULT_SUITE
    return {name: landscape_stats(name, log2_n=log2_n, seed=seed) for name in names}


def load_stats(path: Path | None = None) -> dict[str, dict[str, float]]:
    path = Path(path) if path is not None else DEFAULT_STATS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Landscape statistics not found at {path}. Generate them with:\n"
            f"  python scripts/boba_benchmarks.py --output {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["functions"]


# ---------------------------------------------------------------------------
# The oracle. Duck-types bo_sensor_error_simulation.OracleModel so the existing
# simulator core can drive it unchanged, and is a plain module-level class so it
# survives the pickling Windows' 'spawn' multiprocessing does.
# ---------------------------------------------------------------------------


class SyntheticOracle:
    """An exact analytic objective, standardised, in place of a fitted human.

    ``predict`` returns ``(f(x) - mean) / std`` as a length-1 array, matching the
    single-objective shape the simulator expects. ``y_opt`` is the *published*
    optimum pushed through the same affine map, so regret is exact -- unlike the
    fitted-oracle path, where ``y_opt`` is a random-search estimate the optimiser
    can legitimately exceed.
    """

    def __init__(self, name: str, mean: float, std: float, objective_name: str = "value"):
        if name not in BENCHMARKS:
            raise KeyError(f"Unknown benchmark: {name}")
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Non-positive standardisation std for {name}: {std}")
        self.name = name
        self.mean = float(mean)
        self.std = float(std)
        self.spec = BENCHMARKS[name]
        self.objective_name = objective_name
        self.objective_columns = [objective_name]
        self.param_columns = self.spec.param_columns

    @classmethod
    def from_stats(cls, name: str, stats: dict[str, dict[str, float]], objective_name: str = "value") -> "SyntheticOracle":
        entry = stats[name]
        return cls(name, mean=entry["mean"], std=entry["std"], objective_name=objective_name)

    @property
    def y_opt(self) -> float:
        return (self.spec.y_best - self.mean) / self.std

    @property
    def bounds_low(self) -> np.ndarray:
        return self.spec.bounds_low

    @property
    def bounds_high(self) -> np.ndarray:
        return self.spec.bounds_high

    def _in_box(self, X: np.ndarray) -> np.ndarray:
        """Clamp to the benchmark's box.

        The BoTorch-delegated benchmarks raise ``Expected X to be within the
        bounds of the test problem`` for anything outside their declared box,
        down to the last bit. The acquisition optimiser searches exactly this box
        and can return a corner point a few ulps outside it, and the oracle call
        in ``run_simulation`` sits OUTSIDE the try/except that catches
        acquisition failures -- so an excursion of 1e-16 would abort a task
        fifteen hours into a sweep. Clamping is the correct reading of such a
        point, and it cannot mask a real out-of-box proposal because the
        simulator is given the same bounds.
        """
        return np.clip(np.asarray(X, dtype=float), self.spec.lo, self.spec.hi)

    def raw(self, X: np.ndarray) -> np.ndarray:
        return evaluate(self.name, self._in_box(X))

    def predict(self, x: np.ndarray) -> np.ndarray:
        value = evaluate(self.name, self._in_box(x).reshape(1, -1))
        return np.asarray([(float(value[0]) - self.mean) / self.std], dtype=float)

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        values = evaluate(self.name, self._in_box(X))
        return ((values - self.mean) / self.std).reshape(-1, 1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SyntheticOracle({self.name!r}, dim={self.spec.dim}, y_opt={self.y_opt:.4f})"


# ---------------------------------------------------------------------------
# CLI: (re)generate the landscape statistics file.
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import platform
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--functions", type=str, default=None,
                        help="Comma-separated subset, or 'extensions' for the levy dimension "
                             "ladder and the confound-breaking bump family "
                             "(default: the whole BOBA suite minus 'typing').")
    parser.add_argument("--log2-samples", type=int, default=STATS_LOG2_SAMPLES)
    parser.add_argument("--seed", type=int, default=STATS_SEED)
    parser.add_argument("--boba-root", type=str, default=None,
                        help="BOBA checkout, required only for the 'typing' benchmark.")
    args = parser.parse_args()

    if args.boba_root:
        set_boba_root(args.boba_root)

    if args.functions == "extensions":
        names = list(EXTENSION_ORDER)
    elif args.functions:
        names = [n.strip() for n in args.functions.split(",") if n.strip()]
    else:
        names = DEFAULT_SUITE
    unknown = [n for n in names if n not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"Unknown benchmark(s): {unknown}")

    stats = {}
    for name in names:
        stats[name] = landscape_stats(name, log2_n=args.log2_samples, seed=args.seed)
        s = stats[name]
        print(
            f"{name:<20s} d={s['dim']:>2d}  mean={s['mean']:>12.4g}  std={s['std']:>12.4g}  "
            f"opt_z={s['opt_z']:>7.3f}  sparsity={s['sparsity_10pct']:.5f}  "
            f"rugged={s['ruggedness']:.3f}  tail={s['tail_ratio']:.2f}  "
            f"frag1={s['frag_1']:.3f}"
        )

    payload = {
        "log2_samples": int(args.log2_samples),
        "seed": int(args.seed),
        "python_version": sys.version,
        "platform": platform.platform(),
        "functions": stats,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
