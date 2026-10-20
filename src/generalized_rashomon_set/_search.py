"""Greedy mask search (Algorithm 1 of Li et al., ICLR 2024).

For one feature and one direction the search walks the multiplicative mask
``m`` away from 1 as far as the loss allows, for a sequence of growing loss
budgets ``t_j * epsilon`` (``t_j`` in :func:`boundary_grid`).  The mask found
for budget ``j`` is a model in the Rashomon set whose loss is at most
``loss_ref + t_j * epsilon``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ._attribution import model_loss


@dataclass
class SearchTrace:
    feature: int
    direction: int  # 0 = upper (m >= 1), 1 = lower (m <= 1)
    masks: np.ndarray  # (n_boundaries,)
    losses: np.ndarray  # (n_boundaries,) loss of the masked model
    steps: List[Tuple[float, float]] = field(default_factory=list)  # (m, loss - loss_ref)
    insensitive: bool = False  # loss never changed: the model ignores this feature
    hit_m_max: bool = False


def greedy_mask_search(predictor, loss, space, X, y, feature: int, direction: int,
                       loss_ref: float, epsilon: float, grid: np.ndarray,
                       lr: float = 0.1, max_refinements: int = 4, patience: int = 20,
                       m_max: float = 1e3, tol: float = 1e-12, stall_tol: float = 0.0) -> SearchTrace:
    """Run the greedy search for one ``(feature, direction)`` pair.

    Parameters
    ----------
    lr : float
        Initial step; multiplied by 0.1 after each rejected step.
    max_refinements : int
        Number of step-size reductions before moving to the next boundary
        (4 gives a resolution of ``lr * 1e-4``).
    patience : int
        Stop when this many consecutive accepted steps change the loss by at
        most ``stall_tol``: the model does not use the feature.
    m_max : float
        Stop when ``|m|`` exceeds this value.
    stall_tol : float
        Absolute loss change below which a step counts as "no change".
    """
    sign = 1.0 if direction == 0 else -1.0
    masks = np.empty(len(grid))
    losses = np.empty(len(grid))
    steps: List[Tuple[float, float]] = []
    m, loss_m = 1.0, loss_ref
    stalled, insensitive, hit_max = 0, False, False
    for j, t in enumerate(grid):
        bound = loss_ref + t * epsilon + tol
        step, fails = lr, 0
        while fails < max_refinements and not insensitive and not hit_max:
            m_try = m + sign * step
            l_try = model_loss(predictor, loss, space.apply_mask(X, [feature], m_try), y)
            if l_try <= bound:
                stalled = stalled + 1 if abs(l_try - loss_m) <= stall_tol else 0
                m, loss_m = m_try, l_try
                steps.append((m, loss_m - loss_ref))
                if stalled >= patience:
                    insensitive = True
                if abs(m) > m_max:
                    hit_max = True
            else:
                step *= 0.1
                fails += 1
        masks[j], losses[j] = m, loss_m
    return SearchTrace(feature, direction, masks, losses, steps, insensitive, hit_max)


def _search_chunk(predictor, loss, space, X, y, jobs, loss_ref, epsilon, grid, **kw):
    return [greedy_mask_search(predictor, loss, space, X, y, f, d, loss_ref, epsilon, grid, **kw)
            for f, d in jobs]
