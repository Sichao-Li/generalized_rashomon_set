"""Permutation-based attribution primitives and the worker functions that run
(possibly in parallel) over sampled Rashomon-set models.

All workers are plain module-level functions taking picklable arguments so
they can be dispatched with joblib.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def model_loss(predictor, loss, X, y) -> float:
    return loss(y, predictor(X))


def permutation_loss(predictor, loss, space, X, y, features: Sequence[int],
                     rng: np.random.Generator, times: int,
                     batch_size: Optional[int] = None) -> float:
    """Mean loss after permuting ``features`` ``times`` times.

    Permuted copies are stacked and evaluated in as few model calls as
    possible (``batch_size`` caps the number of rows per call).
    """
    n = X.shape[0]
    per_call = times if not batch_size else max(1, int(batch_size) // n)
    total, done = 0.0, 0
    while done < times:
        k = min(per_call, times - done)
        Xp = space.permute(X, features, rng, k)
        pred = predictor(Xp)
        for b in range(k):
            total += loss(y, pred[b * n:(b + 1) * n])
        done += k
    return total / times


def main_effect(predictor, loss, space, X, y, feature: int, rng, times, batch_size=None,
                loss_before: Optional[float] = None) -> float:
    """phi_i(f) = E[L(f(X_{\\i}), y)] - E[L(f(X), y)]."""
    if loss_before is None:
        loss_before = model_loss(predictor, loss, X, y)
    return permutation_loss(predictor, loss, space, X, y, [feature], rng, times, batch_size) - loss_before


def joint_effect(predictor, loss, space, X, y, subset: Sequence[int], rng, times, batch_size=None,
                 loss_before: Optional[float] = None) -> float:
    """phi_I(f) = E[L(f(X_{\\I}), y)] - E[L(f(X), y)]."""
    if loss_before is None:
        loss_before = model_loss(predictor, loss, X, y)
    return permutation_loss(predictor, loss, space, X, y, list(subset), rng, times, batch_size) - loss_before


def feature_interaction_score(predictor, loss, space, X, y, subset, rng, times, batch_size=None):
    """FIS_I(f) = phi_I(f) - sum_{i in I} phi_i(f), all evaluated on the same model."""
    lb = model_loss(predictor, loss, X, y)
    jt = joint_effect(predictor, loss, space, X, y, subset, rng, times, batch_size, lb)
    mains = [main_effect(predictor, loss, space, X, y, f, rng, times, batch_size, lb) for f in subset]
    return jt - sum(mains), jt, mains, lb


# ---------------------------------------------------------------------------
# Workers.  Each receives a *chunk* of jobs to amortise pickling of the model.
# ---------------------------------------------------------------------------

def _permutation_chunk(predictor, loss, space, X, y, jobs, times, batch_size):
    """Permutation loss of each ``(subset, seed)`` in ``jobs`` on the reference model."""
    return [permutation_loss(predictor, loss, space, X, y, list(s), np.random.default_rng(seed),
                             times, batch_size) for s, seed in jobs]


def _attribution_chunk(predictor, loss, space, X, y, jobs, times, batch_size, targets):
    """For each ``((feature, m), seed)`` in ``jobs``: loss of the masked model
    and the permutation loss of ``feature`` (and of every feature in
    ``targets`` when ``targets`` is not None) under that model."""
    out = []
    for (feature, m), seed in jobs:
        rng = np.random.default_rng(seed)
        Xm = space.apply_mask(X, [feature], m)
        lm = model_loss(predictor, loss, Xm, y)
        own = permutation_loss(predictor, loss, space, Xm, y, [feature], rng, times, batch_size)
        full = None
        if targets is not None:
            full = np.array([own if t == feature else
                             permutation_loss(predictor, loss, space, Xm, y, [t], rng, times, batch_size)
                             for t in targets])
        out.append((lm, own, full))
    return out


def _interaction_chunk(predictor, loss, space, X, y, jobs, times, batch_size):
    """For each ``((subset, mask_table), seed)`` in ``jobs`` evaluate every row
    of ``mask_table`` (shape ``(K, order)``): the joint-masked model's loss,
    its joint permutation effect and the main effect of each member feature."""
    out = []
    for (subset, table), seed in jobs:
        rng = np.random.default_rng(seed)
        subset = list(subset)
        K = table.shape[0]
        lm = np.empty(K)
        joint = np.empty(K)
        mains = np.empty((K, len(subset)))
        for k in range(K):
            Xm = space.apply_mask(X, subset, table[k])
            lm[k] = model_loss(predictor, loss, Xm, y)
            joint[k] = permutation_loss(predictor, loss, space, Xm, y, subset, rng, times, batch_size) - lm[k]
            for a, f in enumerate(subset):
                mains[k, a] = permutation_loss(predictor, loss, space, Xm, y, [f], rng, times, batch_size) - lm[k]
        out.append((lm, joint, mains))
    return out
