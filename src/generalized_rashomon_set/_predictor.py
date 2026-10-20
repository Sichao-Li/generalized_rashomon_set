"""Uniform ``X -> np.ndarray`` prediction for sklearn-like, statsmodels and
PyTorch models."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _is_torch_module(model) -> bool:
    for cls in type(model).__mro__:
        if cls.__module__.startswith("torch.nn") and cls.__name__ == "Module":
            return True
    return False


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Predictor:
    """Wrap a model so that ``predictor(X)`` returns what the loss needs.

    Parameters
    ----------
    model : object
        Anything with ``predict`` / ``predict_proba`` (scikit-learn,
        statsmodels results, ...) or a ``torch.nn.Module``.
    needs : {"proba", "label", "value"}
        Kind of prediction required by the loss.
    predict_fn : callable, optional
        Escape hatch: ``predict_fn(X) -> array`` is used verbatim.  Use it for
        models that do not follow the conventions above (e.g. a torch model
        that needs special preprocessing or outputs logits for a custom loss).

    Notes
    -----
    * ``needs="proba"`` uses ``predict_proba`` when available, otherwise
      ``predict`` (statsmodels ``Logit`` returns probabilities from
      ``predict``).  Torch outputs are passed through ``softmax`` (2-D with
      more than one column) or ``sigmoid`` (single column) unless they
      already look like probabilities.
    * ``needs="label"`` uses ``predict`` for models exposing ``classes_``;
      otherwise probabilities are thresholded at 0.5 / arg-maxed.
    """

    def __init__(self, model, needs: str = "value",
                 predict_fn: Optional[Callable] = None):
        self.model = model
        self.needs = needs
        self.predict_fn = predict_fn
        self._is_torch = predict_fn is None and _is_torch_module(model)

    def __call__(self, X) -> np.ndarray:
        if self.predict_fn is not None:
            out = self.predict_fn(X)
        elif self._is_torch:
            out = self._torch_predict(X)
        else:
            out = self._generic_predict(X)
        out = np.asarray(out)
        if out.ndim == 2 and out.shape[1] == 1 and self.needs != "proba":
            out = out.ravel()
        return out

    # -- scikit-learn / statsmodels -------------------------------------
    def _generic_predict(self, X):
        m = self.model
        names = getattr(m, "feature_names_in_", None)
        if names is not None and getattr(X, "ndim", 0) == 2 and X.shape[1] == len(names):
            import pandas as pd  # estimator was fitted on a DataFrame: give it one back
            X = pd.DataFrame(X, columns=names)
        if self.needs == "proba":
            if hasattr(m, "predict_proba"):
                return m.predict_proba(X)
            if hasattr(m, "predict"):
                return m.predict(X)
            raise TypeError("model has neither predict_proba nor predict")
        if self.needs == "label":
            if hasattr(m, "classes_") or not hasattr(m, "predict_proba"):
                out = np.asarray(m.predict(X))
                if out.dtype.kind == "f" and out.ndim == 1 and not np.isin(out, (0.0, 1.0)).all():
                    # e.g. statsmodels Logit: predict() returns probabilities
                    return (out >= 0.5).astype(int)
                return out
            return self._proba_to_label(m.predict_proba(X))
        if hasattr(m, "predict"):
            return m.predict(X)
        raise TypeError(f"Unsupported model type {type(m)}; pass predict_fn=...")

    # -- torch -----------------------------------------------------------
    def _torch_predict(self, X):
        import torch

        m = self.model
        was_training = m.training
        m.eval()
        try:
            params = list(m.parameters())
            device = params[0].device if params else torch.device("cpu")
            xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
            with torch.no_grad():
                out = m(xt)
            out = out.detach().cpu().numpy()
        finally:
            if was_training:
                m.train()
        if self.needs == "proba":
            return self._to_proba(out)
        if self.needs == "label":
            return self._proba_to_label(self._to_proba(out))
        return out

    @staticmethod
    def _to_proba(out):
        out = np.asarray(out, dtype=float)
        if out.ndim == 2 and out.shape[1] > 1:
            rows = out.sum(axis=1)
            if (out >= 0).all() and np.allclose(rows, 1.0, atol=1e-4):
                return out
            return _softmax(out)
        out = out.reshape(-1)
        if out.min() >= 0.0 and out.max() <= 1.0:
            return out
        return _sigmoid(out)

    @staticmethod
    def _proba_to_label(p):
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] > 1:
            return p.argmax(axis=1)
        return (p.reshape(-1) >= 0.5).astype(int)
