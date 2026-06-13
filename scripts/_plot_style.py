"""Shared, publication-friendly plotting style for the evaluation figures.

Both ``evaluate_research_question.py`` and ``plot_combined_aspects.py`` import
this module so every figure in ``output/evaluation`` shares one consistent look:
the same fonts, the same grid, and — importantly — the same colour for a given
acquisition function across every panel.

The module is import-safe when the scripts are run as ``python scripts/<x>.py``
(the script directory lands on ``sys.path[0]``). It touches only matplotlib /
seaborn global state and defines no behaviour that the test-suite exercises.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Canonical acquisition ordering.  Keeping a fixed order means the colour a
# reader learns in one figure ("pi is teal") holds in every other figure.
# ---------------------------------------------------------------------------
ACQUISITION_ORDER: list[str] = [
    # improvement-based (single objective)
    "ei", "logei", "pi", "logpi",
    # confidence-bound / greedy
    "ucb", "qucb", "greedy",
    # batch monte-carlo single objective
    "qei", "qpi", "qnei",
    # multi-objective
    "qehvi", "qnehvi",
    # model-free floors
    "random", "sobol",
]

# A readable label for each acquisition (used for legends / tick labels).
ACQUISITION_LABELS: dict[str, str] = {
    "ei": "EI", "logei": "logEI", "pi": "PI", "logpi": "logPI",
    "ucb": "UCB", "qucb": "qUCB", "greedy": "Greedy",
    "qei": "qEI", "qpi": "qPI", "qnei": "qNEI",
    "qehvi": "qEHVI", "qnehvi": "qNEHVI",
    "random": "Random", "sobol": "Sobol",
}

# Sequential map for "lower is better, all non-negative" quantities (mean rank).
SEQUENTIAL_CMAP = "viridis_r"
# Diverging map for excess metrics that are signed and centred on zero.
DIVERGING_CMAP = "RdBu_r"

# Accent colours used for baseline-vs-jittered comparisons and zero lines.
COLOR_BASELINE = "#4C72B0"   # calm blue
COLOR_JITTERED = "#C44E52"   # warm red
COLOR_ZERO_LINE = "#444444"
COLOR_GRID = "#D9D9D9"


def set_pub_style() -> None:
    """Apply a clean, consistent theme to all subsequent matplotlib figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            # Typography
            # DejaVu Sans ships only "normal" (400) and "bold" (700); using
            # "medium"/"semibold" triggers findfont fallback warnings.
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "normal",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            # Canvas
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            # Spines / grid: keep it light so the data leads.
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            # Lines / markers
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def order_acquisitions(acquisitions) -> list[str]:
    """Return the given acquisitions sorted by the canonical order.

    Unknown names (e.g. a custom acquisition) are appended alphabetically so
    nothing is ever silently dropped.
    """
    present = list(dict.fromkeys(acquisitions))  # de-dup, preserve first-seen
    known = [a for a in ACQUISITION_ORDER if a in present]
    extra = sorted(a for a in present if a not in ACQUISITION_ORDER)
    return known + extra


def acquisition_palette(acquisitions) -> dict[str, tuple]:
    """Stable {acquisition: rgb} mapping shared across every figure."""
    ordered = order_acquisitions(acquisitions)
    colors = sns.color_palette("husl", n_colors=max(3, len(ordered)))
    return {acq: colors[i] for i, acq in enumerate(ordered)}


def pretty_acq(name: str) -> str:
    """Human-readable label for an acquisition function."""
    return ACQUISITION_LABELS.get(name, name)


def annotate_direction(fig, text: str = "Lower is better") -> None:
    """Add a small, unobtrusive direction-of-goodness caption to a figure."""
    fig.text(
        0.995,
        0.005,
        text,
        ha="right",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="#666666",
    )
