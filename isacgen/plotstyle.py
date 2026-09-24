"""Shared plotting palette (fixed categorical order, never cycled)."""
import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"


def apply():
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "grid.color": GRID,
        "axes.grid": True, "grid.linewidth": 0.6, "lines.linewidth": 2.0,
        "lines.markersize": 6, "legend.frameon": False, "figure.dpi": 130,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9,
    })
