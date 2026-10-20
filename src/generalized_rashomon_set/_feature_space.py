"""How features are masked and permuted.

The explainer never touches raw arrays itself: a *feature space* knows how to
(a) multiply a feature by a mask value ``m`` and (b) permute a feature across
samples.  :class:`TabularFeatureSpace` handles ordinary ``(n, p)`` matrices,
:class:`ImageSegmentFeatureSpace` treats super-pixels of an image as features.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class TabularFeatureSpace:
    """Features are columns of an ``(n, p)`` array.

    Parameters
    ----------
    n_features : int
    permutation : {"independent", "joint"}
        How a *set* of features is permuted when estimating a joint effect:
        each column with its own permutation (``"independent"``, default, the
        behaviour of the original implementation) or all columns with the same
        permutation (``"joint"``, preserves the dependence inside the set).
    """

    def __init__(self, n_features: int, permutation: str = "independent"):
        if permutation not in ("independent", "joint"):
            raise ValueError("permutation must be 'independent' or 'joint'")
        self.n_features = int(n_features)
        self.permutation = permutation

    def apply_mask(self, X: np.ndarray, features: Sequence[int], factors) -> np.ndarray:
        """Return a copy of ``X`` with ``X[:, features] *= factors``."""
        Xm = np.array(X, copy=True)
        features = list(features)
        Xm[:, features] = Xm[:, features] * np.asarray(factors, dtype=Xm.dtype)
        return Xm

    def permute(self, X: np.ndarray, features: Sequence[int], rng: np.random.Generator,
                times: int) -> np.ndarray:
        """Stack ``times`` copies of ``X`` with ``features`` permuted in each copy.

        Returns an array of shape ``(times * n, p)`` so that one model call
        evaluates all permutations.
        """
        n = X.shape[0]
        Xr = np.repeat(X[None, ...], times, axis=0)  # (times, n, p)
        features = list(features)
        if self.permutation == "joint":
            perms = rng.permuted(np.tile(np.arange(n), (times, 1)), axis=1)
            Xr[:, :, features] = X[perms][:, :, features]
        else:
            for f in features:
                perms = rng.permuted(np.tile(np.arange(n), (times, 1)), axis=1)
                Xr[:, :, f] = X[perms, f]
        return Xr.reshape(times * n, *X.shape[1:])


class ImageSegmentFeatureSpace:
    """Features are segments (super-pixels) of an image.

    ``X`` must have shape ``(n, H, W, C)`` (``channel_axis="last"``) or
    ``(n, C, H, W)`` (``channel_axis="first"``); ``segments`` is an integer
    ``(H, W)`` map.  Feature ``k`` is the ``k``-th unique segment id.

    Parameters
    ----------
    segments : array-like of shape (H, W)
    channel_axis : {"last", "first"}
    removal : {"shuffle", "baseline"}
        ``"shuffle"`` permutes pixel positions inside the segment (the original
        behaviour), ``"baseline"`` replaces the segment by ``baseline``.
    baseline : float
    """

    def __init__(self, segments, channel_axis: str = "last", removal: str = "shuffle",
                 baseline: float = 0.0):
        self.segments = np.asarray(segments)
        self.ids = np.unique(self.segments)
        self.n_features = len(self.ids)
        if channel_axis not in ("last", "first"):
            raise ValueError("channel_axis must be 'last' or 'first'")
        if removal not in ("shuffle", "baseline"):
            raise ValueError("removal must be 'shuffle' or 'baseline'")
        self.channel_axis = channel_axis
        self.removal = removal
        self.baseline = baseline
        self._masks = [self.segments == s for s in self.ids]

    def _select(self, X, k):
        """View of the pixels of segment ``k``: shape (n, n_pix, C) / (n, C, n_pix)."""
        if self.channel_axis == "last":
            return X[:, self._masks[k]]
        return X[:, :, self._masks[k]]

    def _assign(self, X, k, value):
        if self.channel_axis == "last":
            X[:, self._masks[k]] = value
        else:
            X[:, :, self._masks[k]] = value

    def apply_mask(self, X, features, factors):
        Xm = np.array(X, copy=True)
        for k, m in zip(features, np.broadcast_to(np.asarray(factors, dtype=float), (len(features),))):
            self._assign(Xm, k, self._select(Xm, k) * m)
        return Xm

    def permute(self, X, features, rng, times):
        Xr = np.repeat(X[None, ...], times, axis=0)
        for t in range(times):
            for k in features:
                if self.removal == "baseline":
                    self._assign(Xr[t], k, self.baseline)
                    continue
                pix = self._select(Xr[t], k)
                axis = 1 if self.channel_axis == "last" else 2
                idx = rng.permutation(pix.shape[axis])
                self._assign(Xr[t], k, np.take(pix, idx, axis=axis))
        return Xr.reshape(times * X.shape[0], *X.shape[1:])
