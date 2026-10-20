"""Backward-compatible ``fis_explainer`` (0.1.x API) on top of
:class:`~generalized_rashomon_set.RashomonExplainer`.

New code should use :class:`RashomonExplainer` directly.  The legacy class
keeps the old constructor arguments, the ``ref_explain`` / ``rset_explain``
methods and the old result attributes (``ref_analysis``,
``rset_main_effect_processed``, ``fis_in_r``, ``loss_in_r``, ``all_pairs``,
``v_list``, ...) so existing notebooks keep running.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._explainer import RashomonExplainer

__all__ = ["RashomonExplainer", "fis_explainer"]


class fis_explainer(RashomonExplainer):
    """Deprecated alias of :class:`RashomonExplainer` with the 0.1.x interface."""

    def __init__(self, model, input=None, output=None, epsilon_rate=0.1, loss_fn=None, n_order=2,
                 delta=0.1, binary_output=False, predict_proba=False, seed=42, **kwargs):
        warnings.warn("explainers.fis_explainer is deprecated; use generalized_rashomon_set."
                      "RashomonExplainer", DeprecationWarning, stacklevel=2)
        if input is None or output is None:
            raise ValueError("Both input and output must be defined")
        kwargs.pop("torch_input", None)
        kwargs.pop("wrapper_for_torch", None)
        kwargs.pop("binary", None)
        kwargs.pop("n_ways", None)
        loss = loss_fn
        if binary_output and loss_fn in (None, "log_loss", "log_loss_avg", "log_loss_sum"):
            loss = "zero_one_loss"
        super().__init__(model, input, output, loss=loss, epsilon_rate=epsilon_rate, delta=delta,
                         random_state=seed, **kwargs)
        self.n_order = n_order
        self.quadrants = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        self.ref_analysis = {}
        self.rset_main_effect_raw = {}
        self.rset_main_effect_processed = {}
        self.rset_joint_effect_raw = {}
        self.FIS_in_Rashomon_set = {}
        self.fis_in_r = None
        self.loss_in_r = None

    # legacy attribute names ------------------------------------------------
    @property
    def input(self):
        return self.X

    @property
    def output(self):
        return self.y

    @property
    def loss(self):
        return self.loss_ref

    @property
    def v_list(self):
        return np.array(self.active_features)

    @property
    def all_pairs(self):
        if self.interactions_ is not None:
            return list(self.interactions_.subsets)
        if self.reference_ is not None:
            return list(self.reference_.subsets)
        return []

    @property
    def prediction(self):
        return self.predictor(self.X)

    # legacy methods ----------------------------------------------------------
    def ref_explain(self, model_reliance=False):
        ref = self.explain_reference(order=self.n_order)
        main = np.nan_to_num(ref.main_effects, nan=0.0)
        self.ref_analysis = {
            "ref_main_effects": main.tolist(),
            "ref_joint_effects": ref.joint_effects.tolist(),
            "important_features": list(self.active_features),
            "important_pairs": list(ref.subsets),
            "ref_fis": [(s, float(v)) for s, v in zip(ref.subsets, ref.fis)],
        }
        if model_reliance:
            self.ref_analysis["ref_model_reliance"] = [
                float(v) for v in (1.0 + main[self.active_features] / self.loss_ref)]
        return self.ref_analysis

    def rset_explain(self, main_effect=True, interaction_effect=True):
        if self.reference_ is None:
            self.ref_explain()
        S = self.sample_rashomon_set()
        act = list(S.features)
        self.rset_main_effect_raw = {
            "m_single_boundary_e": S.masks[act],  # (p, d, 2)
            "loss_single_boundary_e": S.losses[act],
            "points_all_max": [S.traces[(f, 0)].steps for f in act],
            "points_all_min": [S.traces[(f, 1)].steps for f in act],
        }
        if main_effect:
            A = self.explain_attributions()
            self.rset_main_effect_processed = {
                "m_multi_boundary_e": S.masks[act].transpose(1, 0, 2),  # (d, p, 2)
                "all_main_effects_diff": A.effects[act].transpose(1, 0, 2),
                "all_main_effects_ratio": A.ratios[act].transpose(1, 0, 2),
                "loss_multi_boundary_e": A.losses[act].transpose(1, 0, 2),
            }
        if interaction_effect:
            C = self.explain_interactions(order=self.n_order)
            self.fis_in_r = C.fis
            self.loss_in_r = C.losses - C.loss_ref - C.epsilon
            self.rset_joint_effect_raw = {
                "joint_effect_all_pair_set": C.joint_effects,
                "loss_emp_all_pair_set": C.losses,
            }
            self.FIS_in_Rashomon_set = {
                f"pair_idx_{i}": {"feature_idx": s,
                                  "results": {"ref": self.reference_.fis_of(s), "min": float(C.fis[i].min()),
                                              "max": float(C.fis[i].max())}}
                for i, s in enumerate(C.subsets)}
        return self

    def feature_importance_uni(self, mean=False, std=False):
        A = self.attributions_
        vals = [A.values(f) for f in A.features]
        if std:
            return [float(np.std(v)) for v in vals]
        return [float(np.mean(v)) for v in vals]

    @staticmethod
    def load_results(explainer, results_path=None):
        raise NotImplementedError("the 0.1.x JSON layout is no longer supported; use "
                                  "explainer.save(path) and RashomonExplainer.load(path, model, X, y)")
