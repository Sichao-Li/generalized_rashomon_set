"""Halo plots.

For a feature pair ``(i, j)`` the loss budget ``r * epsilon`` is split as
``(a*delta, b*delta) * epsilon`` between the two single-feature masks; the
*expected* halo is the circle of radius ``r * epsilon`` (what the joint model
would lose if the two features did not interact) and the *empirical* halo is
the actual loss increase of the joint-masked model drawn in the same
direction.  Signs of the two masks pick the quadrant.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ._style import colors_vis, frame

_LIGHTNESS = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def halo_coordinates(cloud, subset, radius: float):
    """Return ``(empirical, expected)`` arrays of shape ``(n_points, order)``
    for ``subset`` at loss budget ``radius``."""
    i = cloud.index(subset)
    ks = cloud.combo_indices(radius)
    emp, exp = [], []
    for k in ks:
        _, a, s = cloud.combos[k]
        u = np.asarray(a, float)
        u = u / np.linalg.norm(u)
        sign = np.where(np.asarray(s) == 0, 1.0, -1.0)
        emp.append((cloud.losses[i, k] - cloud.loss_ref) * u * sign)
        exp.append(radius * cloud.epsilon * u * sign)
    return np.array(emp), np.array(exp)


def _closed_by_angle(pts):
    order = np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))
    pts = pts[order]
    return np.vstack([pts, pts[:1]])


def _as_list(explainers):
    return list(explainers) if isinstance(explainers, (list, tuple)) else [explainers]


def halo_plot(explainers, subset, radii: Optional[Sequence[float]] = None, ax=None,
              names=None, legend: bool = True, figsize=(6, 6)):
    """2-D halo plot of a feature pair.

    Parameters
    ----------
    explainers : RashomonExplainer or list of them
        Each explainer (e.g. different ``epsilon``) is drawn as one ring,
        darker for later explainers.
    subset : tuple
        The pair, as indices or names.
    radii : sequence of float, optional
        Loss budgets to draw (default: all ``radii`` used by ``explain_interactions``).
    """
    exs = _as_list(explainers)
    ex0 = exs[0]
    sub = tuple(ex0.feature_index(f) for f in subset)
    if len(sub) != 2:
        raise ValueError("halo_plot needs a feature pair; use halo_plot_3d for triples")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ring = 0
    for ex in exs:
        C = ex.interactions_
        if C is None:
            raise RuntimeError("run explain_interactions() first")
        for r in (radii if radii is not None else C.radii):
            emp, exp = halo_coordinates(C, sub, r)
            emp, exp = _closed_by_angle(emp), _closed_by_angle(exp)
            lt = _LIGHTNESS[min(ring, len(_LIGHTNESS) - 1)]
            ax.plot(emp[:, 0], emp[:, 1], color=colors_vis(0, lt), marker="o", linewidth=2.5,
                    markersize=3, label="empirical joint loss" if ring == 0 else None)
            ax.plot(exp[:, 0], exp[:, 1], color=colors_vis(1, lt), linewidth=2.5,
                    label="expected (no interaction)" if ring == 0 else None)
            ring += 1
    eps = max(ex.epsilon for ex in exs)
    lim = max(eps, float(np.abs(np.array(ax.dataLim.extents)).max())) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ticks = np.array([-1, -0.5, 0, 0.5, 1]) * eps
    ax.set_xticks(ticks, labels=[r"-$\epsilon$", r"-0.5$\epsilon$", "0", r"0.5$\epsilon$", r"$\epsilon$"])
    ax.set_yticks(ticks, labels=[r"-$\epsilon$", r"-0.5$\epsilon$", "0", r"0.5$\epsilon$", r"$\epsilon$"])
    ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
    lab = list(names) if names is not None else [ex0.feature_names[f] for f in sub]
    ax.set_xlabel(f"loss change along {lab[0]}", fontsize=12)
    ax.set_ylabel(f"loss change along {lab[1]}", fontsize=12)
    if legend:
        ax.legend(loc="upper right", fontsize=9)
    frame(ax)
    return ax


def halo_plot_3d(explainer, subset, radii: Optional[Sequence[float]] = None, ax=None, names=None,
                 figsize=(7, 7)):
    """3-D halo plot of a feature triple (scatter of empirical vs expected points)."""
    sub = tuple(explainer.feature_index(f) for f in subset)
    if len(sub) != 3:
        raise ValueError("halo_plot_3d needs a feature triple")
    C = explainer.interactions_
    if C is None or C.order != 3:
        raise RuntimeError("run explain_interactions(order=3) first")
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
    for r in (radii if radii is not None else C.radii):
        emp, exp = halo_coordinates(C, sub, r)
        ax.scatter(emp[:, 0], emp[:, 1], emp[:, 2], color=colors_vis(0, 0.4), alpha=0.6,
                   label="empirical joint loss")
        ax.scatter(exp[:, 0], exp[:, 1], exp[:, 2], color=colors_vis(1, 0.4), alpha=0.6,
                   label="expected (no interaction)")
    lab = list(names) if names is not None else [explainer.feature_names[f] for f in sub]
    ax.set_xlabel(lab[0])
    ax.set_ylabel(lab[1])
    ax.set_zlabel(lab[2])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.legend(fontsize=9)
    return ax
