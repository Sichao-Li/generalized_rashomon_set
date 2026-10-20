import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from generalized_rashomon_set import Loss, Predictor, resolve_loss


def test_resolve_loss_names_and_defaults(clf_model, additive_model):
    assert resolve_loss("log_loss").needs == "proba"
    assert resolve_loss("accuracy").needs == "label"
    assert resolve_loss("mse").needs == "value"
    assert resolve_loss(None, clf_model).name == "log_loss"
    assert resolve_loss(None, additive_model).name == "mean_squared_error"
    with pytest.raises(ValueError):
        resolve_loss("not_a_loss")


def test_scores_become_losses():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    assert resolve_loss("roc_auc")(y, p) == pytest.approx(0.0)
    assert resolve_loss("accuracy")(y, (p > 0.5).astype(int)) == pytest.approx(0.0)
    assert resolve_loss("accuracy")(y, 1 - (p > 0.5).astype(int)) == pytest.approx(1.0)


def test_custom_callable_loss():
    L = resolve_loss(lambda yt, yp: float(np.mean(np.abs(yt - yp))), loss_input="value")
    assert isinstance(L, Loss) and L(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 0.5


def test_predictor_sklearn(clf_model, clf_data):
    X, _ = clf_data
    assert Predictor(clf_model, "proba")(X).shape == (len(X), 2)
    lab = Predictor(clf_model, "label")(X)
    assert set(np.unique(lab)) <= {0, 1}


def test_predictor_predict_fn(additive_model, additive_data):
    X, _ = additive_data
    p = Predictor(additive_model, "value", predict_fn=lambda Z: np.zeros(len(Z)))
    assert np.all(p(X) == 0)


def test_predictor_torch(clf_data):
    torch = pytest.importorskip("torch")
    X, _ = clf_data
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    proba = Predictor(net, "proba")(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(1), 1)
    lab = Predictor(net, "label")(X)
    assert set(np.unique(lab)) <= {0, 1}
    net1 = torch.nn.Linear(4, 1)
    p1 = Predictor(net1, "proba")(X)
    assert p1.ndim == 1 and (p1 >= 0).all() and (p1 <= 1).all()
