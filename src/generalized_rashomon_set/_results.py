"""Result containers.  All arrays are indexed by *feature index* (not by
position in the analysed subset) so they can be read without bookkeeping;
entries that were not computed are ``NaN``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._search import SearchTrace

Subset = Tuple[int, ...]


def _arr(x):
    return None if x is None else np.asarray(x, dtype=float)


def _lst(x):
    return None if x is None else np.asarray(x).tolist()


@dataclass
class ReferenceResult:
    """Attributions of the reference model.

    Attributes
    ----------
    loss : float
        Loss of the reference model on the explanation data.
    main_effects : ndarray, shape (n_features,)
        Permutation main effect ``phi_i`` of each feature (NaN if not analysed).
    subsets : list of tuple
        Feature subsets whose joint effect / FIS was computed.
    joint_effects, fis : ndarray, shape (n_subsets,)
        ``phi_I`` and ``FIS_I = phi_I - sum_i phi_i``.
    dropped_features : list of int
        Features whose main effect is exactly 0 (the model ignores them);
        they are excluded from the Rashomon-set exploration.
    """

    loss: float
    order: int
    main_effects: np.ndarray
    subsets: List[Subset]
    joint_effects: np.ndarray
    fis: np.ndarray
    dropped_features: List[int] = field(default_factory=list)

    def index(self, subset) -> int:
        return self.subsets.index(tuple(subset))

    def fis_of(self, subset) -> float:
        return float(self.fis[self.index(subset)])

    def to_dict(self):
        return dict(loss=self.loss, order=self.order, main_effects=_lst(self.main_effects),
                    subsets=[list(s) for s in self.subsets], joint_effects=_lst(self.joint_effects),
                    fis=_lst(self.fis), dropped_features=list(self.dropped_features))

    @classmethod
    def from_dict(cls, d):
        return cls(d["loss"], d["order"], _arr(d["main_effects"]), [tuple(s) for s in d["subsets"]],
                   _arr(d["joint_effects"]), _arr(d["fis"]), list(d.get("dropped_features", [])))


@dataclass
class RashomonSamples:
    """Models found by the greedy mask search.

    Each sampled model is the reference model with one input feature
    multiplied by ``masks[feature, boundary, direction]``; its loss is
    ``losses[feature, boundary, direction] <= loss_ref + grid[boundary] * epsilon``.
    ``direction`` 0 is the upper mask (``m >= 1``), 1 the lower mask (``m <= 1``).
    """

    loss_ref: float
    epsilon: float
    grid: np.ndarray
    features: List[int]
    masks: np.ndarray  # (n_features, n_boundaries, 2)
    losses: np.ndarray  # (n_features, n_boundaries, 2)
    traces: Dict[Tuple[int, int], SearchTrace] = field(default_factory=dict)

    @property
    def n_boundaries(self) -> int:
        return len(self.grid)

    @property
    def insensitive_features(self) -> List[int]:
        return sorted({f for (f, _), t in self.traces.items() if t.insensitive})

    def models(self):
        """Iterate ``(feature, boundary, direction, m, loss)`` over all samples."""
        for f in self.features:
            for j in range(self.n_boundaries):
                for s in (0, 1):
                    yield f, j, s, float(self.masks[f, j, s]), float(self.losses[f, j, s])

    def to_dict(self):
        return dict(loss_ref=self.loss_ref, epsilon=self.epsilon, grid=_lst(self.grid),
                    features=list(self.features), masks=_lst(self.masks), losses=_lst(self.losses),
                    traces=[dict(feature=t.feature, direction=t.direction, masks=_lst(t.masks),
                                 losses=_lst(t.losses), steps=[list(s) for s in t.steps],
                                 insensitive=t.insensitive, hit_m_max=t.hit_m_max)
                            for t in self.traces.values()])

    @classmethod
    def from_dict(cls, d):
        traces = {}
        for t in d.get("traces", []):
            tr = SearchTrace(t["feature"], t["direction"], _arr(t["masks"]), _arr(t["losses"]),
                             [tuple(s) for s in t["steps"]], t["insensitive"], t.get("hit_m_max", False))
            traces[(tr.feature, tr.direction)] = tr
        return cls(d["loss_ref"], d["epsilon"], _arr(d["grid"]), list(d["features"]),
                   _arr(d["masks"]), _arr(d["losses"]), traces)


@dataclass
class AttributionCloud:
    """Main effects of the sampled models.

    ``effects[f, j, s]`` is the permutation main effect of feature ``f`` in the
    model whose mask for ``f`` is ``masks[f, j, s]``; ``ratios`` is the ratio
    form ``loss_permuted / loss_model`` (model reliance) and ``losses`` the
    loss of that model.  With ``full_matrix=True`` ``full[f, j, s, t]`` holds
    the main effect of *every* feature ``t`` under that model.
    """

    features: List[int]
    grid: np.ndarray
    effects: np.ndarray  # (n_features, n_boundaries, 2)
    ratios: np.ndarray
    losses: np.ndarray
    full: Optional[np.ndarray] = None  # (n_features, n_boundaries, 2, n_features)

    def values(self, feature: int, full_matrix: bool = False) -> np.ndarray:
        """All sampled main-effect values for ``feature`` (1-D, NaN removed)."""
        if full_matrix:
            if self.full is None:
                raise ValueError("explain_attributions(full_matrix=True) was not run")
            v = self.full[:, :, :, feature]
        else:
            v = self.effects[feature]
        return v[~np.isnan(v)].ravel()

    def losses_for(self, feature: int, full_matrix: bool = False) -> np.ndarray:
        if full_matrix:
            v = self.full[:, :, :, feature]
            return self.losses[~np.isnan(v)].ravel()
        v = self.effects[feature]
        return self.losses[feature][~np.isnan(v)].ravel()

    def to_dict(self):
        return dict(features=list(self.features), grid=_lst(self.grid), effects=_lst(self.effects),
                    ratios=_lst(self.ratios), losses=_lst(self.losses), full=_lst(self.full))

    @classmethod
    def from_dict(cls, d):
        return cls(list(d["features"]), _arr(d["grid"]), _arr(d["effects"]), _arr(d["ratios"]),
                   _arr(d["losses"]), _arr(d.get("full")))


@dataclass
class InteractionCloud:
    """Feature interaction scores across the Rashomon set (the FISC).

    For every subset ``I`` (row) and every mask combination ``k`` (column,
    see :func:`~generalized_rashomon_set._combinatorics.interaction_combos`)
    the joint-masked model ``M_{m_I}`` was evaluated:

    * ``losses[i, k]``: its loss,
    * ``joint_effects[i, k]``: ``phi_I(M)``,
    * ``main_effects[i, k, a]``: ``phi_{I_a}(M)`` for each member feature,
    * ``fis[i, k] = joint - sum_a main``.
    """

    order: int
    delta: float
    radii: List[float]
    loss_ref: float
    epsilon: float
    subsets: List[Subset]
    combos: List[Tuple[float, Subset, Subset]]  # (radius, grid tuple, sign tuple)
    masks: np.ndarray  # (n_subsets, K, order)
    losses: np.ndarray  # (n_subsets, K)
    joint_effects: np.ndarray
    main_effects: np.ndarray  # (n_subsets, K, order)
    fis: np.ndarray

    @property
    def n_combos(self) -> int:
        return len(self.combos)

    def index(self, subset) -> int:
        return self.subsets.index(tuple(subset))

    @property
    def in_rashomon_set(self) -> np.ndarray:
        """Boolean ``(n_subsets, K)``: joint-masked model inside the Rashomon set."""
        return self.losses <= self.loss_ref + self.epsilon + 1e-12

    def combo_indices(self, radius: float) -> np.ndarray:
        return np.array([k for k, (r, _, _) in enumerate(self.combos) if abs(r - radius) < 1e-9])

    def to_dict(self):
        return dict(order=self.order, delta=self.delta, radii=list(self.radii), loss_ref=self.loss_ref,
                    epsilon=self.epsilon, subsets=[list(s) for s in self.subsets],
                    combos=[[r, list(a), list(s)] for r, a, s in self.combos], masks=_lst(self.masks),
                    losses=_lst(self.losses), joint_effects=_lst(self.joint_effects),
                    main_effects=_lst(self.main_effects), fis=_lst(self.fis))

    @classmethod
    def from_dict(cls, d):
        return cls(d["order"], d["delta"], list(d["radii"]), d["loss_ref"], d["epsilon"],
                   [tuple(s) for s in d["subsets"]],
                   [(float(r), tuple(a), tuple(s)) for r, a, s in d["combos"]],
                   _arr(d["masks"]), _arr(d["losses"]), _arr(d["joint_effects"]),
                   _arr(d["main_effects"]), _arr(d["fis"]))
