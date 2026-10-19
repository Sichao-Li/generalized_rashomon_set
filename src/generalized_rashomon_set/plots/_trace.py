"""Visualise the greedy mask search."""
from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from ._style import frame


def search_trace_plot(explainer, features=None, names=None, ax=None, figsize=(7, 5)):
    """Loss change versus mask value ``m`` for every accepted step of the
    greedy search, one line per feature.  The dashed line is ``epsilon``."""
    S = explainer.samples_
    if S is None:
        raise RuntimeError("run sample_rashomon_set() first")
    feats = S.features if features is None else [explainer.feature_index(f) for f in features]
    labels = list(names) if names is not None else [explainer.feature_names[f] for f in feats]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    colors = cm.rainbow(np.linspace(0, 1, max(len(feats), 2)))
    for c, f, lab in zip(colors, feats, labels):
        pts = [(1.0, 0.0)]
        for d in (0, 1):
            tr = S.traces.get((f, d))
            if tr is not None:
                pts += list(tr.steps)
        pts = np.array(sorted(pts))
        ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=3, color=c, label=lab)
    ax.axhline(S.epsilon, color="black", linestyle="--", linewidth=1, label=r"$\epsilon$")
    ax.set_xlabel("mask value m")
    ax.set_ylabel("loss - reference loss")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)
    frame(ax)
    return ax
