"""Loss functions.

A :class:`Loss` is a *loss to be minimised* plus a declaration of what kind
of prediction it consumes (``"proba"``, ``"label"`` or ``"value"``).  Score
metrics such as accuracy or AUC are turned into losses (``1 - score``) so the
Rashomon condition ``loss <= loss_ref + epsilon`` is always meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from sklearn import metrics as skm

NEEDS = ("proba", "label", "value")


@dataclass(frozen=True)
class Loss:
    """A loss function ``fn(y_true, y_pred) -> float`` (lower is better).

    Parameters
    ----------
    name : str
        Human readable name.
    fn : callable
        ``fn(y_true, y_pred)``.
    needs : {"proba", "label", "value"}
        What ``y_pred`` must be: class probabilities, hard class labels or raw
        regression values.  The :class:`~generalized_rashomon_set.Predictor`
        uses this to pick ``predict_proba`` / ``predict`` / ``forward``.
    """

    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    needs: str = "value"

    def __post_init__(self):
        if self.needs not in NEEDS:
            raise ValueError(f"needs must be one of {NEEDS}, got {self.needs!r}")

    def __call__(self, y_true, y_pred) -> float:
        return float(self.fn(y_true, y_pred))


def _as_1d_proba(p):
    p = np.asarray(p, dtype=float)
    if p.ndim == 2 and p.shape[1] == 2:
        return p[:, 1]
    if p.ndim == 2 and p.shape[1] == 1:
        return p.ravel()
    return p


def _log_loss(y, p):
    p = np.asarray(p, dtype=float)
    if p.ndim == 2 and p.shape[1] == 1:
        p = p.ravel()
    return skm.log_loss(y, p)


def _log_loss_sum(y, p):
    p = np.asarray(p, dtype=float)
    if p.ndim == 2 and p.shape[1] == 1:
        p = p.ravel()
    return skm.log_loss(y, p, normalize=False)


def _one_minus_auc(y, p):
    return 1.0 - skm.roc_auc_score(y, _as_1d_proba(p))


def _one_minus_r2(y, p):
    return 1.0 - skm.r2_score(y, p)


def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))


_MSE = Loss("mean_squared_error", skm.mean_squared_error, "value")
_MAE = Loss("mean_absolute_error", skm.mean_absolute_error, "value")
_RMSE = Loss("root_mean_squared_error", _rmse, "value")
_LOG = Loss("log_loss", _log_loss, "proba")
_LOG_SUM = Loss("log_loss_sum", _log_loss_sum, "proba")
_ZERO_ONE = Loss("zero_one_loss", skm.zero_one_loss, "label")
_AUC = Loss("one_minus_roc_auc", _one_minus_auc, "proba")
_R2 = Loss("one_minus_r2", _one_minus_r2, "value")

REGISTRY = {
    "mse": _MSE, "mean_squared_error": _MSE,
    "rmse": _RMSE, "root_mean_squared_error": _RMSE,
    "mae": _MAE, "mean_absolute_error": _MAE,
    "log_loss": _LOG, "log_loss_avg": _LOG, "logloss": _LOG, "cross_entropy": _LOG,
    "log_loss_sum": _LOG_SUM,
    "zero_one_loss": _ZERO_ONE, "accuracy": _ZERO_ONE, "accuracy_score": _ZERO_ONE,
    "error_rate": _ZERO_ONE,
    "roc_auc": _AUC, "roc_auc_score": _AUC, "auc": _AUC,
    "r2": _R2, "r2_score": _R2,
}


def _looks_like_classifier(model) -> bool:
    if model is None:
        return False
    if hasattr(model, "_estimator_type"):
        return getattr(model, "_estimator_type") == "classifier"
    return hasattr(model, "predict_proba") or hasattr(model, "classes_")


def resolve_loss(loss: Union[None, str, Callable, Loss], model=None,
                 loss_input: Optional[str] = None) -> Loss:
    """Turn ``loss`` into a :class:`Loss`.

    * ``None``: ``log_loss`` for classifiers, ``mean_squared_error`` otherwise.
    * ``str``: looked up in :data:`REGISTRY` (score metrics become ``1 - score``).
    * callable: wrapped; ``loss_input`` says which prediction kind it needs
      (default ``"value"``).
    """
    if isinstance(loss, Loss):
        return loss
    if loss is None:
        return _LOG if _looks_like_classifier(model) else _MSE
    if isinstance(loss, str):
        key = loss.lower()
        if key not in REGISTRY:
            raise ValueError(
                f"Unknown loss {loss!r}. Known names: {sorted(set(REGISTRY))}. "
                "You can also pass any callable loss(y_true, y_pred)."
            )
        return REGISTRY[key]
    if callable(loss):
        needs = loss_input or "value"
        return Loss(getattr(loss, "__name__", "custom_loss"), loss, needs)
    raise TypeError(f"loss must be None, str, callable or Loss, got {type(loss)}")
