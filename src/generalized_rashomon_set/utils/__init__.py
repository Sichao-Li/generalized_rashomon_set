"""Small helpers (and the names of the 0.1.x ``utils`` module)."""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from .._combinatorics import (boundary_grid, feature_subsets, interaction_combos, sign_combos,
                              sum_to_one_tuples)
from .._feature_space import ImageSegmentFeatureSpace, TabularFeatureSpace
from .._loss import REGISTRY, Loss, resolve_loss
from .._predictor import Predictor
from ..plots._style import colors_vis

ModelWrapper = Predictor  # legacy name


def find_all_n_order_feature_pairs(vlist, n_order):
    """Legacy name of :func:`feature_subsets`."""
    return feature_subsets(vlist, n_order)


def find_all_sum_to_one_pairs(n_features, delta=0.1):
    """Legacy name of :func:`sum_to_one_tuples` (now correct for any ``delta``)."""
    return sum_to_one_tuples(n_features, delta)


def pd_to_numpy(X, y):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy().reshape(len(y))
    return X, y


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``.

    The explainer does not need this: it owns its random streams through
    ``random_state``.  Provided for notebooks that train models."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unknown type: {type(obj)}")


def save_json(path, results):
    with open(path, "w") as f:
        json.dump(results, f, default=_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def duplicate(data):
    if isinstance(data, np.ndarray):
        return np.copy(data)
    if hasattr(data, "clone"):
        return data.clone()
    return data.copy()


__all__ = ["boundary_grid", "feature_subsets", "interaction_combos", "sign_combos", "sum_to_one_tuples",
           "ImageSegmentFeatureSpace", "TabularFeatureSpace", "REGISTRY", "Loss", "resolve_loss",
           "Predictor", "ModelWrapper", "colors_vis", "find_all_n_order_feature_pairs",
           "find_all_sum_to_one_pairs", "pd_to_numpy", "set_seed", "save_json", "load_json", "duplicate"]
