"""Names of the 0.1.x plotting API, kept as thin wrappers."""
from __future__ import annotations

import warnings

from ._halo import halo_plot, halo_plot_3d
from ._swarm import swarm_plot_attribution, swarm_plot_interaction
from ._trace import search_trace_plot


def _warn(old, new):
    warnings.warn(f"plots.{old} is deprecated, use plots.{new}", DeprecationWarning, stacklevel=3)


def _pair(explainer, pair_idx):
    if isinstance(pair_idx, int):
        return explainer.interactions_.subsets[pair_idx]
    return tuple(pair_idx)


def _maybe_save(ax, save, suffix, prefix):
    if save:
        from ..config import OUTPUT_DIR, ensure_output_dir
        ensure_output_dir()
        ax.figure.savefig(f"{OUTPUT_DIR}/{prefix}_{suffix}.png", bbox_inches="tight")
    return ax


def halo_plot_multiple(*explainers, pair_idx, save=False, suffix=""):
    _warn("halo_plot_multiple", "halo_plot([ex1, ex2, ...], pair)")
    ax = halo_plot(list(explainers), _pair(explainers[0], pair_idx))
    return _maybe_save(ax, save, suffix, "halo_plot")


def halo_plot_3D(explainer, pair_idx, save=False, path=""):
    _warn("halo_plot_3D", "halo_plot_3d")
    ax = halo_plot_3d(explainer, _pair(explainer, pair_idx))
    if save and path:
        ax.figure.savefig(path, bbox_inches="tight")
    return ax


def swarm_plot_MR(explainer, interest_of_features=None, vname=None, plot_all=False, threshold=None,
                  boxplot=False, absolute=False, save=False, suffix=None):
    _warn("swarm_plot_MR", "swarm_plot_attribution")
    feats = None if plot_all or interest_of_features is None else list(interest_of_features)
    names = None
    if vname is not None:
        fs = feats if feats is not None else explainer.attributions_.features
        names = [vname[f] for f in fs]
    ax = swarm_plot_attribution(explainer, feats, names=names, absolute=absolute, boxplot=boxplot,
                                threshold=threshold)
    return _maybe_save(ax, save, suffix, "swarm_plot")


def swarm_plot_MR_multiple(*explainers, interest_of_features=None, vname=None, plot_all=False,
                           threshold=None, boxplot=False, absolute=False, save=False, suffix=None):
    _warn("swarm_plot_MR_multiple", "swarm_plot_attribution([ex1, ex2, ...])")
    feats = None if plot_all or interest_of_features is None else list(interest_of_features)
    names = None
    if vname is not None:
        fs = feats if feats is not None else explainers[0].attributions_.features
        names = [vname[f] for f in fs]
    ax = swarm_plot_attribution(list(explainers), feats, names=names, absolute=absolute,
                                boxplot=boxplot, threshold=threshold)
    return _maybe_save(ax, save, suffix, "swarm_plot")


def swarm_plot_FIS(explainer, interest_of_pairs=None, vname=None, plot_all=False, threshold=None,
                   boxplot=False, save=False, absolute=False, suffix=None):
    _warn("swarm_plot_FIS", "swarm_plot_interaction")
    subs = None if plot_all or interest_of_pairs is None else [tuple(p) for p in interest_of_pairs]
    names = None
    if vname is not None:
        ss = subs if subs is not None else explainer.interactions_.subsets
        names = [" vs ".join(str(vname[f]) for f in s) for s in ss]
    ax = swarm_plot_interaction(explainer, subs, names=names, absolute=absolute, boxplot=boxplot,
                                threshold=threshold)
    return _maybe_save(ax, save, suffix, "swarm_plot")


def swarm_plot_FIS_multiple(*explainers, interest_of_pairs=None, vname=None, plot_all=False,
                            threshold=None, boxplot=False, save=False, absolute=False, suffix=None):
    _warn("swarm_plot_FIS_multiple", "swarm_plot_interaction([ex1, ex2, ...])")
    subs = None if plot_all or interest_of_pairs is None else [tuple(p) for p in interest_of_pairs]
    names = None
    if vname is not None:
        ss = subs if subs is not None else explainers[0].interactions_.subsets
        names = [" vs ".join(str(vname[f]) for f in s) for s in ss]
    ax = swarm_plot_interaction(list(explainers), subs, names=names, absolute=absolute,
                                boxplot=boxplot, threshold=threshold)
    return _maybe_save(ax, save, suffix, "swarm_plot")


def training_process(explainer, vlist=None, save=False, suffix=""):
    _warn("training_process", "search_trace_plot")
    ax = search_trace_plot(explainer, vlist)
    return _maybe_save(ax, save, suffix, "training_process")
