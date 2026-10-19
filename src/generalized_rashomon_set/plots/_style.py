"""Shared colour helpers."""
from __future__ import annotations

import colorsys

import matplotlib.colors as mcolors

DEFAULT_COLORS = ["#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"]
LOSS_CMAP = mcolors.LinearSegmentedColormap.from_list("grs_loss", ["#1E88E5", "#7C52FF", "#ff0d57"], N=180)


def colors_vis(c: int, lightness: float = 0.5):
    """Palette colour ``c`` with its lightness replaced by ``lightness``."""
    rgb = mcolors.hex2color(DEFAULT_COLORS[c % len(DEFAULT_COLORS)])
    h, _, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, lightness, s)


def frame(ax, width=1.0, color="black"):
    for loc in ("left", "right", "top", "bottom"):
        ax.spines[loc].set_linewidth(width)
        ax.spines[loc].set_color(color)
