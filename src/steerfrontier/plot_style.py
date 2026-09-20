"""Shared visual styling helpers for CAA-SteerFrontier figures.

The module centralizes manuscript-facing typography, labels, colormaps, and
vector/raster figure export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# Core visual language
# =============================================================================

BEHAVIOR_BLUE = "#1f77b4"
CAPABILITY_ORANGE = "#ff7f0e"
SELECTED_PURPLE = "#6a3d9a"

OTHER_GRAY = "#a7adb2"
GRID_GRAY = "#d9d9d9"
ZERO_GRAY = "#7f7f7f"

NEGATIVE_BLUE = "#2166ac"
NEUTRAL_WHITE = "#f7f7f7"
POSITIVE_GREEN = "#1b7837"

SEQUENTIAL_COLORS = [
    "#ffffcc",
    "#a1dab4",
    "#41b6c4",
    "#2c7fb8",
    "#253494",
]


DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "behavior_diverging",
    [
        NEGATIVE_BLUE,
        NEUTRAL_WHITE,
        POSITIVE_GREEN,
    ],
)

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "tealblue_sequential",
    SEQUENTIAL_COLORS,
)


CAPABILITY_STYLES = {
    "mmlu": {
        "label": "MMLU",
        "linestyle": "-",
        "marker": "o",
        "alpha": 1.0,
    },
    "arc_challenge": {
        "label": "ARC-Challenge",
        "linestyle": "--",
        "marker": "s",
        "alpha": 0.82,
    },
    "hellaswag": {
        "label": "HellaSwag",
        "linestyle": ":",
        "marker": "^",
        "alpha": 0.68,
    },
}


# =============================================================================
# Plot style
# =============================================================================

def configure_matplotlib():
    """Configure the shared manuscript-facing Matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 17,
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.color": GRID_GRAY,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def get_heatmap_cmap(
    name: str,
):
    """Return the configured diverging colormap for steering heatmaps."""
    if name == "diverging":
        return DIVERGING_CMAP

    if name == "sequential":
        return SEQUENTIAL_CMAP

    raise ValueError(f"Unknown colormap: {name}")


def pretty_behavior(
    behavior: str,
) -> str:
    """Convert an internal behavior slug into a human-readable plot label."""
    return behavior.replace(
        "_",
        " ",
    ).title()


def save_figure(
    fig,
    stem: Path,
    *,
    dpi: int = 300,
):
    """Save one figure as both vector PDF and raster PNG outputs."""
    stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = stem.with_suffix(".png")

    pdf_path = stem.with_suffix(".pdf")

    fig.savefig(
        png_path,
        dpi=dpi,
    )

    fig.savefig(
        pdf_path,
    )

    plt.close(fig)

    return (
        png_path,
        pdf_path,
    )
