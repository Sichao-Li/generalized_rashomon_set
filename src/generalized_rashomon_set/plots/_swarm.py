"""Swarm plots: the distribution of an attribution (main effect or FIS)
across the Rashomon set, one row per feature / feature subset, coloured by
the loss of the model that produced the value, with the reference model's
value marked by a star."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._style import LOSS_CMAP, frame


def _as_list(explainers):
    return list(explainers) if isinstance(explainers, (list, tuple)) else [explainers]


def _draw(ax, df, groups, refs, x_label, y_label, vmin, vmax, boxplot, threshold, kind, size, cmap):
    import seaborn as sns

    cmap = cmap or LOSS_CMAP
    plotter = sns.swarmplot if kind == "swarm" else sns.stripplot
    kw = dict(data=df, x="value", y="group", hue="loss", palette=cmap, hue_norm=(vmin, vmax),
              order=groups, size=size, ax=ax, legend=False, zorder=1)
    if kind != "swarm":
        kw["jitter"] = 0.25
    plotter(**kw)
    if boxplot:
        sns.boxplot(data=df, x="value", y="group", order=groups, showcaps=False,
                    boxprops={"facecolor": "none"}, showfliers=False,
                    whiskerprops={"linewidth": 0}, ax=ax, zorder=2)
    pos = [groups.index(g) for g, r in refs if np.isfinite(r)]
    if pos:
        ax.scatter([r for _, r in refs if np.isfinite(r)], pos, marker="*", s=180, color="orange",
                   edgecolor="black", linewidth=0.5, zorder=3, label="reference model")
    if threshold is not None:
        ax.axvline(threshold, color="black", linewidth=1)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("loss of the sampled model", size=12)
    frame(ax)
    return ax


def swarm_plot_attribution(explainers, features: Optional[Sequence] = None, names=None,
                           absolute: bool = False, boxplot: bool = False, threshold=None,
                           ax=None, kind: str = "swarm", size: float = 3, full_matrix: bool = False,
                           cmap=None, figsize=(8, 6)):
    """Distribution of main effects across the Rashomon set.

    Parameters
    ----------
    explainers : RashomonExplainer or list of them
        Several explainers (e.g. different ``epsilon``) are pooled.
    features : sequence of int or str, optional
        Rows to show (default: all active features of the first explainer).
    names : sequence of str, optional
        Row labels (default: feature names).
    absolute : bool
        Plot ``|phi_i|``.
    full_matrix : bool
        Use the main effects of *all* sampled models for every feature
        (requires ``explain_attributions(full_matrix=True)``).
    kind : {"swarm", "strip"}
        ``"strip"`` is faster for very large clouds.
    """
    exs = _as_list(explainers)
    ex0 = exs[0]
    feats = ex0.attributions_.features if features is None else [ex0.feature_index(f) for f in features]
    labels = list(names) if names is not None else [ex0.feature_names[f] for f in feats]
    rows, refs = [], []
    for ex in exs:
        if ex.attributions_ is None:
            raise RuntimeError("run explain_attributions() first")
        for f, lab in zip(feats, labels):
            v = ex.attributions_.values(f, full_matrix)
            l = ex.attributions_.losses_for(f, full_matrix)
            rows.append(pd.DataFrame({"group": lab, "value": np.abs(v) if absolute else v, "loss": l}))
    for f, lab in zip(feats, labels):
        r = ex0.reference_.main_effects[f] if ex0.reference_ is not None else np.nan
        refs.append((lab, abs(r) if absolute else r))
    df = pd.concat(rows, ignore_index=True)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    vmin = min(ex.loss_ref for ex in exs)
    vmax = max(ex.loss_ref + ex.epsilon for ex in exs)
    x_label = "|main effect|" if absolute else "main effect (loss increase when permuted)"
    return _draw(ax, df, labels, refs, x_label, "feature", vmin, vmax, boxplot, threshold, kind, size, cmap)


def swarm_plot_interaction(explainers, subsets: Optional[Sequence] = None, names=None,
                           absolute: bool = False, boxplot: bool = False, threshold=None,
                           ax=None, kind: str = "swarm", size: float = 3, in_set_only: bool = False,
                           cmap=None, figsize=(8, 6)):
    """Distribution of feature interaction scores (the FISC) across the Rashomon set.

    ``in_set_only`` drops joint-masked models whose loss exceeds the Rashomon
    boundary (interaction can push a model out of the set).
    """
    exs = _as_list(explainers)
    ex0 = exs[0]
    if ex0.interactions_ is None:
        raise RuntimeError("run explain_interactions() first")
    subs = ex0.interactions_.subsets if subsets is None else ex0._subsets(subsets)
    labels = list(names) if names is not None else [ex0.subset_name(s) for s in subs]
    rows, refs = [], []
    for ex in exs:
        C = ex.interactions_
        for s, lab in zip(subs, labels):
            i = C.index(s)
            keep = C.in_rashomon_set[i] if in_set_only else np.ones(C.n_combos, bool)
            v = C.fis[i][keep]
            rows.append(pd.DataFrame({"group": lab, "value": np.abs(v) if absolute else v,
                                      "loss": C.losses[i][keep]}))
    for s, lab in zip(subs, labels):
        r = ex0.reference_.fis_of(s) if ex0.reference_ is not None and tuple(s) in ex0.reference_.subsets else np.nan
        refs.append((lab, abs(r) if absolute else r))
    df = pd.concat(rows, ignore_index=True)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    vmin = min(ex.loss_ref for ex in exs)
    vmax = max(float(np.nanmax(ex.interactions_.losses)) for ex in exs)
    vmax = max(vmax, max(ex.loss_ref + ex.epsilon for ex in exs))
    x_label = "|FIS|" if absolute else "feature interaction score"
    return _draw(ax, df, labels, refs, x_label, "feature subset", vmin, vmax, boxplot, threshold, kind, size, cmap)
