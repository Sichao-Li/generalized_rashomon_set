import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def interaction_data():
    """y = x0 + x1 + 2 x0 x1 + 0.5 x2 (+ noise); x3 is pure noise."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = X[:, 0] + X[:, 1] + 2 * X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.05, size=400)
    return X, y


@pytest.fixture(scope="session")
def interaction_model(interaction_data):
    X, y = interaction_data
    # exact functional form: degree-2 polynomial regression
    return make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(X, y)


@pytest.fixture(scope="session")
def additive_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.05, size=400)
    return X, y


@pytest.fixture(scope="session")
def additive_model(additive_data):
    X, y = additive_data
    return LinearRegression().fit(X, y)


@pytest.fixture(scope="session")
def clf_data():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 4))
    logit = 1.5 * X[:, 0] - X[:, 1] + 1.5 * X[:, 0] * X[:, 1]
    y = (rng.random(500) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.fixture(scope="session")
def clf_model(clf_data):
    X, y = clf_data
    return LogisticRegression(max_iter=1000).fit(X, y)
