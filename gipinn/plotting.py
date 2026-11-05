# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

COL_W, DOUBLE_W = 8.5 / 2.54, 17.8 / 2.54
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQ_CMAP = LinearSegmentedColormap.from_list("blues_seq", ["#f4f7fb", "#9ec3ec", "#2a78d6", "#0b2f66"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f2f2f0", "#2a78d6"])

def set_style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 3.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "legend.frameon": False,
        "axes.grid": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })

def panel_label(ax, text: str, x: float = -0.22, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

def light_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)

def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
