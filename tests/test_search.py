import numpy as np

from generalized_rashomon_set import Predictor, TabularFeatureSpace, greedy_mask_search, resolve_loss, boundary_grid


class Quadratic:
    """f(x) = x0 ; with y = x0 the loss of f(m x0) is (m-1)^2 E[x0^2]."""

    def predict(self, Z):
        return Z[:, 0]


def test_greedy_search_stays_inside_boundaries_and_is_monotone():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X[:, 0].copy()
    L, P, sp = resolve_loss("mse"), Predictor(Quadratic(), "value"), TabularFeatureSpace(2)
    loss_ref, eps, grid = 0.0, 0.01, boundary_grid(0.1)
    for d in (0, 1):
        tr = greedy_mask_search(P, L, sp, X, y, 0, d, loss_ref, eps, grid)
        assert np.all(tr.losses <= loss_ref + grid * eps + 1e-12)
        assert tr.masks[0] == 1.0
        diffs = np.diff(tr.masks)
        assert np.all(diffs >= 0) if d == 0 else np.all(diffs <= 0)
        # analytic boundary: (m-1)^2 * E[x0^2] = t * eps  ->  |m-1| = sqrt(t*eps/E[x0^2])
        expect = np.sqrt(grid * eps / np.mean(X[:, 0] ** 2))
        assert np.allclose(np.abs(tr.masks - 1.0), expect, atol=2e-4)  # resolution lr * 1e-4
        assert not tr.insensitive


def test_greedy_search_detects_insensitive_feature():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = X[:, 0].copy()
    L, P, sp = resolve_loss("mse"), Predictor(Quadratic(), "value"), TabularFeatureSpace(2)
    tr = greedy_mask_search(P, L, sp, X, y, 1, 0, 0.0, 0.01, boundary_grid(0.5), patience=5)
    assert tr.insensitive and len(tr.steps) == 5
