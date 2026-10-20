import numpy as np

from generalized_rashomon_set import ImageSegmentFeatureSpace, TabularFeatureSpace
from generalized_rashomon_set._attribution import permutation_loss
from generalized_rashomon_set import resolve_loss, Predictor


def test_tabular_mask_copies_and_scales():
    X = np.arange(12, dtype=float).reshape(4, 3)
    sp = TabularFeatureSpace(3)
    Xm = sp.apply_mask(X, [1], 2.0)
    assert np.all(Xm[:, 1] == 2 * X[:, 1]) and np.all(Xm[:, [0, 2]] == X[:, [0, 2]])
    assert X[0, 1] == 1.0  # untouched
    Xm2 = sp.apply_mask(X, [0, 2], [0.5, 3.0])
    assert np.all(Xm2[:, 0] == 0.5 * X[:, 0]) and np.all(Xm2[:, 2] == 3 * X[:, 2])


def test_tabular_permute_is_a_permutation_per_block():
    rng = np.random.default_rng(0)
    X = np.arange(30, dtype=float).reshape(10, 3)
    sp = TabularFeatureSpace(3)
    Xp = sp.permute(X, [1], rng, 5)
    assert Xp.shape == (50, 3)
    for b in range(5):
        blk = Xp[b * 10:(b + 1) * 10]
        assert np.all(blk[:, [0, 2]] == X[:, [0, 2]])
        assert sorted(blk[:, 1]) == sorted(X[:, 1])
    # joint permutation keeps rows of the subset together
    spj = TabularFeatureSpace(3, permutation="joint")
    Xj = spj.permute(X, [0, 1], rng, 3)
    for b in range(3):
        blk = Xj[b * 10:(b + 1) * 10]
        assert np.all(blk[:, 1] - blk[:, 0] == 1)


def test_permutation_loss_batches_match_single_call():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, 2.0, 0.0])

    class M:
        def predict(self, Z):
            return Z @ np.array([1.0, 2.0, 0.0])

    sp = TabularFeatureSpace(3)
    L = resolve_loss("mse")
    P = Predictor(M(), "value")
    a = permutation_loss(P, L, sp, X, y, [1], np.random.default_rng(1), 8, batch_size=None)
    b = permutation_loss(P, L, sp, X, y, [1], np.random.default_rng(1), 8, batch_size=100)
    assert a == b > 0
    # the unused feature has no effect
    assert permutation_loss(P, L, sp, X, y, [2], np.random.default_rng(1), 8) == 0


def test_image_segment_space():
    seg = np.array([[0, 0, 1], [0, 0, 1], [2, 2, 2]])
    X = np.arange(2 * 9 * 3, dtype=float).reshape(2, 3, 3, 3)
    sp = ImageSegmentFeatureSpace(seg)
    assert sp.n_features == 3
    Xm = sp.apply_mask(X, [1], 0.0)
    assert np.all(Xm[:, seg == 1] == 0) and np.all(Xm[:, seg != 1] == X[:, seg != 1])
    Xp = sp.permute(X, [0], np.random.default_rng(0), 2)
    assert Xp.shape == (4, 3, 3, 3)
    blk = Xp[:2]
    assert np.all(blk[:, seg != 0] == X[:, seg != 0])
    assert np.allclose(np.sort(blk[:, seg == 0], axis=1), np.sort(X[:, seg == 0], axis=1))
    spb = ImageSegmentFeatureSpace(seg, removal="baseline", baseline=-1.0)
    assert np.all(spb.permute(X, [2], np.random.default_rng(0), 1)[:, seg == 2] == -1.0)
