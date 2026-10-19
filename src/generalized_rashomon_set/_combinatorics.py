"""Index bookkeeping shared by the explainer and the plots.

Everything here is pure Python / NumPy and deterministic, so the same
helpers can be used when computing results and when plotting them.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Subset = Tuple[int, ...]


def feature_subsets(features: Iterable[int], order: int) -> List[Subset]:
    """All ``order``-sized feature subsets, in lexicographic order."""
    if order < 1:
        raise ValueError("order must be >= 1")
    return [tuple(c) for c in combinations(list(features), order)]


def boundary_grid(delta: float) -> np.ndarray:
    """Sub-boundary fractions ``0, delta, 2*delta, ..., 1``.

    Boundary ``j`` corresponds to the loss tolerance ``grid[j] * epsilon``.
    """
    steps = int(round(1.0 / delta))
    if steps < 1 or abs(steps * delta - 1.0) > 1e-9:
        raise ValueError(f"delta must divide 1 exactly (got {delta})")
    return np.linspace(0.0, 1.0, steps + 1)


def sum_to_one_tuples(order: int, delta: float, radius: float = 1.0) -> List[Subset]:
    """Grid-index tuples ``(a_1, ..., a_order)`` with ``a_k >= 1`` and
    ``sum_k a_k * delta == radius``.

    For ``order=2, delta=0.1, radius=1`` this is ``(1, 9), (2, 8), ..., (9, 1)``:
    the ways of splitting the loss budget ``radius * epsilon`` between two
    features so that their individual loss changes add up to the budget.
    """
    steps = int(round(1.0 / delta))
    target = int(round(radius / delta))
    if abs(target * delta - radius) > 1e-9 or target < order:
        raise ValueError(
            f"radius {radius} is not reachable with order {order} and delta {delta}"
        )
    return [t for t in product(range(1, steps + 1), repeat=order) if sum(t) == target]


def sign_combos(order: int) -> List[Subset]:
    """All ``2**order`` direction combinations; ``0`` = upper mask (m+),
    ``1`` = lower mask (m-)."""
    return [tuple(c) for c in product((0, 1), repeat=order)]


def interaction_combos(order: int, delta: float, radii: Sequence[float] = (1.0,)):
    """Flattened list of ``(radius, grid_tuple, sign_tuple)`` describing every
    mask combination evaluated for one feature subset.

    The order is radius-major, then sum-to-one tuple, then sign combination,
    i.e. ``index = ((r * n_tuples) + a) * 2**order + s``.
    """
    combos = []
    for r in radii:
        for a in sum_to_one_tuples(order, delta, r):
            for s in sign_combos(order):
                combos.append((float(r), a, s))
    return combos
