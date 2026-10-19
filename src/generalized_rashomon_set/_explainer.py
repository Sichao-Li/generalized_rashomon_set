"""The :class:`RashomonExplainer`: sample a mask-based Rashomon set around a
reference model and explain feature attributions and interactions across it.

Pipeline (each stage runs the previous ones on demand)::

    explainer = RashomonExplainer(model, X, y, loss="log_loss", epsilon_rate=0.05, n_jobs=-1)
    explainer.explain_reference()       # phi_i, phi_I and FIS_I of the reference model
    explainer.sample_rashomon_set()     # greedy mask search, parallel over (feature, direction)
    explainer.explain_attributions()    # phi_i of every sampled model, parallel over models
    explainer.explain_interactions()    # FIS_I of joint-masked models, parallel over subsets
    explainer.attribution_summary(); explainer.interaction_summary()

or simply ``explainer.fit()``.
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._attribution import _attribution_chunk, _interaction_chunk, _permutation_chunk, model_loss
from ._combinatorics import boundary_grid, feature_subsets, interaction_combos
from ._feature_space import TabularFeatureSpace
from ._loss import REGISTRY, Loss, resolve_loss
from ._parallel import chunk, n_chunks_for, run_jobs, spawn_seeds
from ._predictor import Predictor
from ._results import AttributionCloud, InteractionCloud, RashomonSamples, ReferenceResult
from ._search import _search_chunk

logger = logging.getLogger("generalized_rashomon_set")
logger.addHandler(logging.NullHandler())

_STAGE_REFERENCE, _STAGE_REFERENCE_JOINT, _STAGE_SAMPLE, _STAGE_ATTRIBUTION, _STAGE_INTERACTION = 1, 5, 2, 3, 4


def enable_logging(level=logging.INFO):
    """Print progress messages of the package to stderr."""
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
               for h in logger.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)


def _coerce_X(X, feature_names):
    names = None
    if hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
        X = X.to_numpy()
    elif hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    if X.dtype == object:
        raise TypeError("X must be numeric; encode categorical columns first")
    if X.dtype.kind in "iub":
        X = X.astype(float)
    if feature_names is not None:
        names = [str(c) for c in feature_names]
    return X, names


def _coerce_y(y):
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    elif hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return y


class RashomonExplainer:
    """Explore the Rashomon set of ``model`` through input masks and explain it.

    A model of the Rashomon set is the reference model composed with a
    feature-wise multiplicative mask ``m`` on its input; it belongs to the set
    when its loss is within ``epsilon`` of the reference loss.  Attributions
    are permutation based: the main effect of feature ``i`` is the loss
    increase when ``i`` is permuted, the joint effect of a subset ``I`` is the
    loss increase when all of ``I`` are permuted, and the feature interaction
    score is ``FIS_I = phi_I - sum_{i in I} phi_i``.

    Parameters
    ----------
    model : object
        Trained reference model: scikit-learn estimator, statsmodels results,
        ``torch.nn.Module`` or anything accepted by :class:`Predictor`.
    X, y : array-like
        Data used to evaluate losses (typically a held-out set).  ``X`` may be
        a DataFrame (column names are kept) or a torch tensor.
    loss : str, callable or Loss, default None
        Loss to minimise.  ``None`` picks ``log_loss`` for classifiers and
        ``mean_squared_error`` for regressors.  See :mod:`_loss` for names.
    epsilon_rate : float, default 0.1
        Rashomon tolerance as a fraction of the reference loss.
    epsilon : float, optional
        Absolute tolerance; overrides ``epsilon_rate``.
    delta : float, default 0.1
        Resolution of the loss budget: sampled models sit at loss budgets
        ``0, delta, 2*delta, ..., 1`` times ``epsilon``.  Must divide 1.
    features : sequence of int or str, optional
        Restrict the analysis to these features (default: all).
    feature_names : sequence of str, optional
    feature_space : object, optional
        Custom masking/permutation rules (e.g. :class:`ImageSegmentFeatureSpace`).
    predict_fn : callable, optional
        ``predict_fn(X) -> predictions`` used instead of the model's own methods.
    loss_input : {"proba", "label", "value"}, optional
        What a *callable* ``loss`` expects as predictions.
    permutation : {"independent", "joint"}
        How feature subsets are permuted for joint effects.
    shuffle_times : int, default 30
        Permutations averaged for each attribution estimate.
    interaction_shuffle_times : int, optional
        Same for the interaction stage (default: ``shuffle_times``).
    lr, max_refinements, patience, m_max :
        Greedy search controls, see :func:`~generalized_rashomon_set._search.greedy_mask_search`.
    drop_insensitive : bool, default True
        Exclude features whose reference main effect is (numerically) 0 from
        the Rashomon-set exploration.
    insensitive_tol : float, default 1e-9
        Relative tolerance (times the reference loss) below which a loss
        change counts as zero, for ``drop_insensitive`` and for the stall
        detection of the greedy search.
    n_jobs : int, default 1
        Parallel workers (``-1`` = all cores).  Sampling is parallel over
        ``(feature, direction)``, attributions over sampled models and
        interactions over feature subsets.
    backend : str, default "loky"
        joblib backend; use ``"threading"`` for torch models to avoid
        oversubscription and pickling.
    batch_size : int, optional
        Maximum number of rows per model call when batching permutations.
    random_state : int, default 0
        Seed; results are reproducible for any ``n_jobs``.
    verbose : bool, default False
        Log progress (see :func:`enable_logging`).
    """

    def __init__(self, model, X, y, *, loss=None, epsilon_rate: float = 0.1,
                 epsilon: Optional[float] = None, delta: float = 0.1,
                 features: Optional[Sequence] = None, feature_names: Optional[Sequence[str]] = None,
                 feature_space=None, predict_fn: Optional[Callable] = None,
                 loss_input: Optional[str] = None, permutation: str = "independent",
                 shuffle_times: int = 30, interaction_shuffle_times: Optional[int] = None,
                 lr: float = 0.1, max_refinements: int = 4, patience: int = 20, m_max: float = 1e3,
                 drop_insensitive: bool = True, insensitive_tol: float = 1e-9, n_jobs: Optional[int] = 1, backend: str = "loky",
                 batch_size: Optional[int] = None, random_state: int = 0, verbose: bool = False):
        self.model = model
        X, names = _coerce_X(X, feature_names)
        self.X = X
        self.y = _coerce_y(y)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X has {self.X.shape[0]} rows but y has {self.y.shape[0]}")
        self.loss_fn: Loss = resolve_loss(loss, model, loss_input)
        self.predictor = Predictor(model, self.loss_fn.needs, predict_fn)
        self.space = feature_space if feature_space is not None else \
            TabularFeatureSpace(X.shape[1], permutation)
        self.n_features = int(self.space.n_features)
        self.feature_names = list(names) if names is not None and len(names) == self.n_features \
            else [f"x{i}" for i in range(self.n_features)]
        self.delta = float(delta)
        self.grid = boundary_grid(self.delta)
        self.shuffle_times = int(shuffle_times)
        self.interaction_shuffle_times = int(interaction_shuffle_times or shuffle_times)
        self.drop_insensitive = drop_insensitive
        self.insensitive_tol = float(insensitive_tol)
        self.n_jobs, self.backend, self.batch_size = n_jobs, backend, batch_size
        self.random_state = int(random_state)
        self.verbose = verbose
        if verbose:
            enable_logging()

        self.loss_ref = model_loss(self.predictor, self.loss_fn, self.X, self.y)
        self.epsilon = float(epsilon) if epsilon is not None else float(self.loss_ref * epsilon_rate)
        self._zero = self.insensitive_tol * max(abs(self.loss_ref), np.finfo(float).tiny)
        self.search_kwargs = dict(lr=lr, max_refinements=max_refinements, patience=patience, m_max=m_max,
                                  stall_tol=self._zero)
        if not self.epsilon > 0:
            warnings.warn(f"epsilon={self.epsilon}: the Rashomon set is reduced to the reference model "
                          "(is the reference loss 0?). Pass an absolute epsilon=... instead.")
        self.features: List[int] = list(range(self.n_features)) if features is None \
            else [self.feature_index(f) for f in features]

        self.reference_: Optional[ReferenceResult] = None
        self.samples_: Optional[RashomonSamples] = None
        self.attributions_: Optional[AttributionCloud] = None
        self.interactions_: Optional[InteractionCloud] = None
        self.active_features_: Optional[List[int]] = None
        self._log(f"reference loss {self.loss_fn.name}={self.loss_ref:.6g}, epsilon={self.epsilon:.6g}, "
                  f"{self.n_features} features, {self.X.shape[0]} samples")

    # ------------------------------------------------------------------ utils
    def _log(self, msg):
        logger.info(msg)

    def feature_index(self, f) -> int:
        """Feature index from an int or a feature name."""
        if isinstance(f, (int, np.integer)):
            if not 0 <= int(f) < self.n_features:
                raise IndexError(f"feature index {f} out of range")
            return int(f)
        if f in self.feature_names:
            return self.feature_names.index(f)
        raise KeyError(f"unknown feature {f!r}")

    def feature_name(self, f) -> str:
        return self.feature_names[self.feature_index(f)]

    def subset_name(self, subset, sep=" x ") -> str:
        return sep.join(self.feature_name(f) for f in subset)

    @property
    def active_features(self) -> List[int]:
        """Features explored in the Rashomon set (``features`` minus dropped)."""
        return list(self.active_features_) if self.active_features_ is not None else list(self.features)

    def _subsets(self, subsets) -> List[tuple]:
        return [tuple(self.feature_index(f) for f in s) for s in subsets]

    def _run(self, fn, jobs, stage, *extra):
        """Run ``fn`` over ``jobs`` in chunks.  Every job gets its own seed, so
        results do not depend on ``n_jobs`` or on how jobs are chunked."""
        seeded = list(zip(jobs, spawn_seeds(self.random_state, stage, len(jobs))))
        chunks = chunk(seeded, n_chunks_for(len(seeded), self.n_jobs))
        args = [(self.predictor, self.loss_fn, self.space, self.X, self.y, c, *extra) for c in chunks]
        out = run_jobs(fn, args, self.n_jobs, self.backend)
        return [r for c in out for r in c]

    def _permutation_losses(self, subsets, stage, times) -> np.ndarray:
        return np.array(self._run(_permutation_chunk, list(subsets), stage, times, self.batch_size),
                        dtype=float)

    # ------------------------------------------------------------ reference
    def explain_reference(self, order: int = 2, subsets=None) -> ReferenceResult:
        """Main effects of all features and joint effects / FIS of ``subsets``
        (default: all subsets of size ``order``) for the reference model."""
        t0 = time.time()
        feats = list(self.features)
        main = np.full(self.n_features, np.nan)
        main[feats] = self._permutation_losses([(f,) for f in feats], _STAGE_REFERENCE,
                                               self.shuffle_times) - self.loss_ref
        dropped = [f for f in feats if abs(main[f]) <= self._zero] if self.drop_insensitive else []
        if dropped:
            self._log(f"features ignored by the model (main effect 0) are dropped: "
                      f"{[self.feature_names[f] for f in dropped]}")
        active = [f for f in feats if f not in dropped]
        subsets = feature_subsets(active, order) if subsets is None else self._subsets(subsets)
        joint = self._permutation_losses(subsets, _STAGE_REFERENCE_JOINT, self.shuffle_times) - self.loss_ref \
            if subsets else np.zeros(0)
        fis = joint - np.array([sum(main[f] for f in s) for s in subsets]) if subsets else np.zeros(0)
        self.reference_ = ReferenceResult(self.loss_ref, order, main, subsets, joint, fis, dropped)
        self.active_features_ = active
        self._log(f"reference explained: {len(active)} features, {len(subsets)} subsets "
                  f"({time.time() - t0:.1f}s)")
        return self.reference_

    # --------------------------------------------------------------- sampling
    def sample_rashomon_set(self) -> RashomonSamples:
        """Greedy mask search for every active feature in both directions."""
        t0 = time.time()
        active = self.active_features
        jobs = [(f, d) for f in active for d in (0, 1)]
        chunks = chunk(jobs, n_chunks_for(len(jobs), self.n_jobs, per_worker=1))
        args = [(self.predictor, self.loss_fn, self.space, self.X, self.y, c,
                 self.loss_ref, self.epsilon, self.grid, self.search_kwargs) for c in chunks]
        out = run_jobs(_search_chunk_kw, args, self.n_jobs, self.backend)
        d = len(self.grid)
        masks = np.full((self.n_features, d, 2), np.nan)
        losses = np.full((self.n_features, d, 2), np.nan)
        traces = {}
        for tr in (t for c in out for t in c):
            masks[tr.feature, :, tr.direction] = tr.masks
            losses[tr.feature, :, tr.direction] = tr.losses
            traces[(tr.feature, tr.direction)] = tr
        self.samples_ = RashomonSamples(self.loss_ref, self.epsilon, self.grid.copy(), active,
                                        masks, losses, traces)
        ins = self.samples_.insensitive_features
        if ins:
            self._log(f"features whose mask never changes the loss: {[self.feature_names[f] for f in ins]}")
        self._log(f"sampled {len(active) * d * 2} models ({time.time() - t0:.1f}s)")
        self.attributions_, self.interactions_ = None, None
        return self.samples_

    # ----------------------------------------------------------- attributions
    def explain_attributions(self, full_matrix: bool = False) -> AttributionCloud:
        """Main effect of each sampled model's masked feature (and of every
        feature when ``full_matrix=True``)."""
        if self.samples_ is None:
            self.sample_rashomon_set()
        t0 = time.time()
        S = self.samples_
        active = list(S.features)
        d = S.n_boundaries
        keys: Dict[tuple, List[tuple]] = {}
        for f in active:
            for j in range(d):
                for s in (0, 1):
                    keys.setdefault((f, float(np.round(S.masks[f, j, s], 12))), []).append((j, s))
        jobs = list(keys)
        targets = active if full_matrix else None
        res = self._run(_attribution_chunk, jobs, _STAGE_ATTRIBUTION, self.shuffle_times,
                        self.batch_size, targets)
        effects = np.full((self.n_features, d, 2), np.nan)
        ratios = np.full((self.n_features, d, 2), np.nan)
        losses = np.full((self.n_features, d, 2), np.nan)
        full = np.full((self.n_features, d, 2, self.n_features), np.nan) if full_matrix else None
        for (f, m), (lm, own, allv) in zip(jobs, res):
            for j, s in keys[(f, m)]:
                effects[f, j, s] = own - lm
                ratios[f, j, s] = own / lm if lm != 0 else np.nan
                losses[f, j, s] = lm
                if full is not None:
                    full[f, j, s, active] = allv - lm
        self.attributions_ = AttributionCloud(active, S.grid.copy(), effects, ratios, losses, full)
        self._log(f"attributions of {len(jobs)} distinct sampled models ({time.time() - t0:.1f}s)")
        return self.attributions_

    # ----------------------------------------------------------- interactions
    def explain_interactions(self, order: int = 2, subsets=None, radii: Sequence[float] = (1.0,),
                             shuffle_times: Optional[int] = None) -> InteractionCloud:
        """FIS of joint-masked models for every subset (default: all subsets of
        size ``order`` among the active features).

        For a subset ``I`` the loss budget ``r * epsilon`` (``r`` in ``radii``)
        is split between its members in every way allowed by ``delta`` and each
        member takes either its upper or its lower mask, giving
        ``n_splits * 2**order`` models per subset.
        """
        if self.samples_ is None:
            self.sample_rashomon_set()
        t0 = time.time()
        S = self.samples_
        active = set(S.features)
        if subsets is None:
            if self.reference_ is not None and self.reference_.order == order:
                subsets = [s for s in self.reference_.subsets if set(s) <= active]
            else:
                subsets = feature_subsets(sorted(active), order)
        else:
            subsets = self._subsets(subsets)
            bad = [s for s in subsets if not set(s) <= active or len(s) != order]
            if bad:
                raise ValueError(f"subsets must have size {order} and use sampled features: {bad}")
        combos = interaction_combos(order, self.delta, radii)
        K = len(combos)
        tables = []
        for sub in subsets:
            t = np.empty((K, order))
            for k, (_, a, s) in enumerate(combos):
                t[k] = [S.masks[f, a_i, s_i] for f, a_i, s_i in zip(sub, a, s)]
            tables.append(t)
        res = self._run(_interaction_chunk, list(zip(subsets, tables)), _STAGE_INTERACTION,
                        shuffle_times or self.interaction_shuffle_times, self.batch_size)
        losses = np.array([r[0] for r in res]).reshape(len(subsets), K)
        joint = np.array([r[1] for r in res]).reshape(len(subsets), K)
        mains = np.array([r[2] for r in res]).reshape(len(subsets), K, order)
        fis = joint - mains.sum(axis=2)
        self.interactions_ = InteractionCloud(order, self.delta, list(radii), self.loss_ref, self.epsilon,
                                              list(subsets), combos, np.array(tables).reshape(len(subsets), K, order),
                                              losses, joint, mains, fis)
        self._log(f"interactions: {len(subsets)} subsets x {K} models ({time.time() - t0:.1f}s)")
        return self.interactions_

    # ---------------------------------------------------------------- driver
    def fit(self, order: int = 2, subsets=None, interactions: bool = True,
            full_matrix: bool = False, radii: Sequence[float] = (1.0,)):
        """Run the whole pipeline."""
        self.explain_reference(order, subsets)
        self.sample_rashomon_set()
        self.explain_attributions(full_matrix)
        if interactions:
            self.explain_interactions(order, subsets, radii)
        return self

    # -------------------------------------------------------------- summaries
    def attribution_summary(self, full_matrix: bool = False) -> pd.DataFrame:
        """Range of the main effect of each feature across the Rashomon set."""
        if self.attributions_ is None:
            raise RuntimeError("run explain_attributions() first")
        A = self.attributions_
        rows = []
        for f in A.features:
            v = A.values(f, full_matrix)
            ref = self.reference_.main_effects[f] if self.reference_ is not None else np.nan
            rows.append(dict(feature=f, name=self.feature_names[f], reference=ref, min=v.min(),
                             max=v.max(), range=v.max() - v.min(), mean=v.mean(), std=v.std(),
                             n_models=len(v)))
        return pd.DataFrame(rows)

    def interaction_summary(self, in_set_only: bool = False) -> pd.DataFrame:
        """Range of the FIS of each subset across the Rashomon set (the FISC)."""
        if self.interactions_ is None:
            raise RuntimeError("run explain_interactions() first")
        C = self.interactions_
        inset = C.in_rashomon_set
        rows = []
        for i, sub in enumerate(C.subsets):
            v = C.fis[i][inset[i]] if in_set_only else C.fis[i]
            ref = np.nan
            if self.reference_ is not None and tuple(sub) in self.reference_.subsets:
                ref = self.reference_.fis_of(sub)
            rows.append(dict(subset=tuple(sub), name=self.subset_name(sub), reference=ref,
                             min=np.nanmin(v) if v.size else np.nan, max=np.nanmax(v) if v.size else np.nan,
                             range=(np.nanmax(v) - np.nanmin(v)) if v.size else np.nan,
                             mean=np.nanmean(v) if v.size else np.nan, std=np.nanstd(v) if v.size else np.nan,
                             n_models=int(v.size), frac_in_set=float(inset[i].mean())))
        return pd.DataFrame(rows)

    def sampled_models(self) -> pd.DataFrame:
        """One row per sampled single-mask model."""
        if self.samples_ is None:
            raise RuntimeError("run sample_rashomon_set() first")
        S = self.samples_
        rows = [dict(feature=f, name=self.feature_names[f], boundary=float(S.grid[j]),
                     direction="+" if s == 0 else "-", m=m, loss=l, loss_diff=l - S.loss_ref)
                for f, j, s, m, l in S.models()]
        return pd.DataFrame(rows)

    # --------------------------------------------------------- masked models
    def masked_predict(self, masks: Dict[Union[int, str], float], X=None) -> np.ndarray:
        """Predictions of the Rashomon-set model ``f(m * X)`` for ``masks``
        ``{feature: m}`` (default data: the explanation data)."""
        X = self.X if X is None else _coerce_X(X, None)[0]
        feats = [self.feature_index(f) for f in masks]
        return self.predictor(self.space.apply_mask(X, feats, list(masks.values())))

    def masked_loss(self, masks: Dict[Union[int, str], float]) -> float:
        return self.loss_fn(self.y, self.masked_predict(masks))

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return dict(
            version=2,
            params=dict(loss=self.loss_fn.name, epsilon=self.epsilon, delta=self.delta,
                        features=list(self.features), feature_names=list(self.feature_names),
                        shuffle_times=self.shuffle_times,
                        interaction_shuffle_times=self.interaction_shuffle_times,
                        random_state=self.random_state, loss_ref=self.loss_ref,
                        permutation=getattr(self.space, "permutation", None),
                        insensitive_tol=self.insensitive_tol,
                        **{k: v for k, v in self.search_kwargs.items() if k != 'stall_tol'}),
            active_features=self.active_features_,
            reference=self.reference_.to_dict() if self.reference_ else None,
            samples=self.samples_.to_dict() if self.samples_ else None,
            attributions=self.attributions_.to_dict() if self.attributions_ else None,
            interactions=self.interactions_.to_dict() if self.interactions_ else None,
        )

    def save(self, path) -> None:
        """Save all results to a JSON file (the model and data are not stored)."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        self._log(f"saved results to {path}")

    @classmethod
    def load(cls, path, model, X, y, **kwargs) -> "RashomonExplainer":
        """Rebuild an explainer from :meth:`save` output plus the model and data.

        Saved parameters are reused unless overridden by ``kwargs``; a custom
        callable ``loss`` must be passed again.
        """
        with open(path) as f:
            d = json.load(f)
        p = dict(d["params"])
        saved_loss_ref = p.pop("loss_ref", None)
        init = dict(loss=p["loss"], epsilon=p["epsilon"], delta=p["delta"], features=p["features"],
                    feature_names=p["feature_names"], shuffle_times=p["shuffle_times"],
                    interaction_shuffle_times=p["interaction_shuffle_times"],
                    random_state=p["random_state"], lr=p["lr"], max_refinements=p["max_refinements"],
                    patience=p["patience"], m_max=p["m_max"], insensitive_tol=p.get("insensitive_tol", 1e-9))
        if p.get("permutation"):
            init["permutation"] = p["permutation"]
        init.update(kwargs)
        if isinstance(init["loss"], str) and init["loss"].lower() not in REGISTRY:
            raise ValueError(f"this explainer used the custom loss {init['loss']!r}; pass loss=... to load()")
        ex = cls(model, X, y, **init)
        if saved_loss_ref is not None and abs(ex.loss_ref - saved_loss_ref) > 1e-6 * max(1.0, abs(saved_loss_ref)):
            warnings.warn(f"reference loss {ex.loss_ref:.6g} differs from the saved value "
                          f"{saved_loss_ref:.6g}: are model and data the same?")
        ex.active_features_ = d.get("active_features")
        ex.reference_ = ReferenceResult.from_dict(d["reference"]) if d.get("reference") else None
        ex.samples_ = RashomonSamples.from_dict(d["samples"]) if d.get("samples") else None
        ex.attributions_ = AttributionCloud.from_dict(d["attributions"]) if d.get("attributions") else None
        ex.interactions_ = InteractionCloud.from_dict(d["interactions"]) if d.get("interactions") else None
        return ex


def _search_chunk_kw(predictor, loss, space, X, y, jobs, loss_ref, epsilon, grid, kw):
    return _search_chunk(predictor, loss, space, X, y, jobs, loss_ref, epsilon, grid, **kw)
