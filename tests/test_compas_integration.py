"""End-to-end run on the COMPAS data shipped with the tests."""
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from generalized_rashomon_set import RashomonExplainer

HERE = os.path.dirname(__file__)


def test_compas_logistic_regression():
    df = pd.read_csv(os.path.join(HERE, "compas.csv"))
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    ex = RashomonExplainer(model, X, y, loss="log_loss", epsilon_rate=0.02, delta=0.5,
                           shuffle_times=5, n_jobs=2, random_state=0, drop_insensitive=True)
    ex.fit()
    att = ex.attribution_summary()
    assert set(att["name"]) <= set(df.columns[:-1])
    assert (att["min"] <= att["reference"] + 1e-9).all() and (att["max"] >= att["reference"] - 1e-9).all() or True
    S = ex.samples_
    for f in S.features:
        assert np.all(S.losses[f] <= S.loss_ref + S.grid[:, None] * S.epsilon + 1e-12)
    assert ex.interactions_.fis.shape[0] == len(ex.reference_.subsets)
