import json

import numpy as np
import pandas as pd
import pytest

from generalized_rashomon_set import RashomonExplainer, plots


@pytest.fixture(scope="module")
def fitted(interaction_model, interaction_data):
    X, y = interaction_data
    return RashomonExplainer(interaction_model, X, y, epsilon_rate=0.1, delta=0.25,
                             shuffle_times=10, random_state=0).fit()


def test_reference_fis_detects_true_interaction(fitted):
    ref = fitted.reference_
    assert ref.subsets[0] == (0, 1)
    fis = dict(zip(ref.subsets, ref.fis))
    assert abs(fis[(0, 1)]) > 5 * max(abs(v) for s, v in fis.items() if s != (0, 1))
    assert ref.main_effects[3] < 0.05 * ref.main_effects[0]


def test_samples_respect_rashomon_boundary(fitted):
    S = fitted.samples_
    for f in S.features:
        for s in (0, 1):
            assert np.all(S.losses[f, :, s] <= S.loss_ref + S.grid * S.epsilon + 1e-12)
    assert S.masks.shape == (4, 5, 2) and np.isnan(S.masks).sum() == 0
    # masked_loss reproduces the stored loss
    f, j, s, m, l = next(S.models())
    assert fitted.masked_loss({f: m}) == pytest.approx(l)


def test_attribution_and_interaction_clouds(fitted):
    A, C = fitted.attributions_, fitted.interactions_
    assert A.effects.shape == (4, 5, 2) and np.isfinite(A.effects).all()
    assert C.fis.shape == (6, 3 * 4) and C.masks.shape == (6, 12, 2)
    # the masks used for the joint model come from the single-feature search
    i = C.index((0, 1))
    r, a, s = C.combos[0]
    assert C.masks[i, 0, 0] == fitted.samples_.masks[0, a[0], s[0]]
    assert C.masks[i, 0, 1] == fitted.samples_.masks[1, a[1], s[1]]
    # FIS = joint - sum(mains) on the same model
    assert np.allclose(C.fis, C.joint_effects - C.main_effects.sum(-1))
    # interaction (0,1) dominates the cloud as well
    summ = fitted.interaction_summary().set_index("name")
    assert abs(summ.loc["x0 x x1", "mean"]) > 5 * summ.drop(index="x0 x x1")["mean"].abs().max()
    att = fitted.attribution_summary()
    assert list(att["feature"]) == [0, 1, 2, 3] and (att["max"] >= att["min"]).all()
    assert isinstance(fitted.sampled_models(), pd.DataFrame) and len(fitted.sampled_models()) == 40


def test_additive_model_has_no_interaction_anywhere(additive_model, additive_data):
    X, y = additive_data

    class IgnoresX2:  # same fit, but feature 2 is truly unused
        def predict(self, Z):
            return additive_model.intercept_ + Z[:, :2] @ additive_model.coef_[:2]

    ex = RashomonExplainer(IgnoresX2(), X, y, loss="mse", epsilon_rate=0.2, delta=0.5, shuffle_times=20,
                           random_state=0, permutation="independent").fit()
    # x2 is unused: dropped
    assert ex.reference_.dropped_features == [2] and ex.active_features == [0, 1]
    assert ex.interactions_.subsets == [(0, 1)]
    scale = abs(ex.reference_.main_effects[0])
    assert np.abs(ex.reference_.fis).max() < 0.05 * scale
    assert np.abs(ex.interactions_.fis).max() < 0.05 * scale


def test_parallel_matches_sequential(interaction_model, interaction_data):
    X, y = interaction_data
    kw = dict(epsilon_rate=0.1, delta=0.5, shuffle_times=5, random_state=3)
    a = RashomonExplainer(interaction_model, X, y, n_jobs=1, **kw).fit()
    b = RashomonExplainer(interaction_model, X, y, n_jobs=2, **kw).fit()
    c = RashomonExplainer(interaction_model, X, y, n_jobs=2, backend="threading", **kw).fit()
    for other in (b, c):
        assert np.array_equal(a.samples_.masks, other.samples_.masks)
        assert np.allclose(a.reference_.main_effects, other.reference_.main_effects)
        assert np.allclose(a.attributions_.effects, other.attributions_.effects)
        assert np.allclose(a.interactions_.fis, other.interactions_.fis)


def test_save_load_roundtrip(fitted, interaction_model, interaction_data, tmp_path):
    X, y = interaction_data
    p = tmp_path / "res.json"
    fitted.save(p)
    json.load(open(p))
    ex = RashomonExplainer.load(p, interaction_model, X, y)
    assert ex.epsilon == fitted.epsilon and ex.delta == fitted.delta
    assert np.array_equal(ex.samples_.masks, fitted.samples_.masks)
    assert np.allclose(ex.interactions_.fis, fitted.interactions_.fis)
    assert ex.reference_.subsets == fitted.reference_.subsets
    assert ex.samples_.traces[(0, 0)].steps == fitted.samples_.traces[(0, 0)].steps
    assert ex.attribution_summary().equals(fitted.attribution_summary())


def test_feature_names_and_subset_selection(clf_model, clf_data):
    X, y = clf_data
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    ex = RashomonExplainer(clf_model, df, y, epsilon_rate=0.05, delta=0.5, shuffle_times=5,
                           features=["a", "b", "c"], random_state=0)
    assert ex.loss_fn.name == "log_loss"
    ex.explain_reference(subsets=[("a", "b"), ("b", "c")])
    assert ex.reference_.subsets == [(0, 1), (1, 2)]
    ex.explain_interactions(subsets=[("a", "b")])
    assert ex.interactions_.subsets == [(0, 1)]
    assert ex.subset_name((0, 1)) == "a x b"
    with pytest.raises(ValueError):
        ex.explain_interactions(subsets=[("a", "d")])  # d was not sampled


def test_third_order_and_radii(clf_model, clf_data):
    X, y = clf_data
    ex = RashomonExplainer(clf_model, X, y, epsilon_rate=0.05, delta=0.25, shuffle_times=5, random_state=0)
    ex.explain_reference(order=3)
    assert len(ex.reference_.subsets) == 4
    ex.explain_interactions(order=3, radii=(0.75, 1.0))
    C = ex.interactions_
    # radius 0.75: (1,1,1); radius 1: (1,1,2),(1,2,1),(2,1,1) -> 4 tuples x 8 signs
    assert C.n_combos == 4 * 8 and C.fis.shape == (4, 32)
    assert len(C.combo_indices(0.75)) == 8
    ax = plots.halo_plot_3d(ex, (0, 1, 2))
    assert ax is not None


def test_full_matrix_attributions(additive_model, additive_data):
    X, y = additive_data
    ex = RashomonExplainer(additive_model, X, y, epsilon_rate=0.2, delta=0.5, shuffle_times=5, random_state=0)
    ex.sample_rashomon_set()
    ex.explain_attributions(full_matrix=True)
    A = ex.attributions_
    assert A.full.shape == (3, 3, 2, 3)
    # the diagonal of the full matrix equals the own-feature effects
    for f in A.features:
        assert np.allclose(A.full[f, :, :, f], A.effects[f])
    assert len(ex.attribution_summary(full_matrix=True)) == 3


def test_label_loss_and_custom_loss(clf_model, clf_data):
    X, y = clf_data
    ex = RashomonExplainer(clf_model, X, y, loss="accuracy", epsilon=0.05, delta=0.5, shuffle_times=5)
    ex.sample_rashomon_set()
    assert np.all(ex.samples_.losses <= ex.loss_ref + ex.epsilon + 1e-12)

    def brier(yt, p):
        return float(np.mean((yt - p[:, 1]) ** 2))

    ex2 = RashomonExplainer(clf_model, X, y, loss=brier, loss_input="proba", epsilon_rate=0.1,
                            delta=0.5, shuffle_times=5)
    ex2.explain_reference()
    assert ex2.loss_fn.needs == "proba" and np.isfinite(ex2.reference_.fis).all()


def test_torch_model_and_threading_backend(clf_data):
    torch = pytest.importorskip("torch")
    X, y = clf_data
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1), torch.nn.Sigmoid())
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)[:, None]
    for _ in range(200):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy(net(xt), yt).backward()
        opt.step()
    ex = RashomonExplainer(net, xt, yt, loss="log_loss", epsilon_rate=0.05, delta=0.5, shuffle_times=5,
                           n_jobs=2, backend="threading", random_state=0).fit()
    assert ex.loss_fn.needs == "proba" and np.isfinite(ex.interactions_.fis).all()
    assert ex.samples_.masks.shape == (4, 3, 2)


def test_plots_return_axes(fitted):
    assert plots.swarm_plot_attribution(fitted, boxplot=True, threshold=0.0) is not None
    assert plots.swarm_plot_attribution([fitted, fitted], features=["x0", "x1"], kind="strip") is not None
    assert plots.swarm_plot_interaction(fitted, absolute=True, in_set_only=True) is not None
    assert plots.halo_plot(fitted, (0, 1)) is not None
    assert plots.halo_plot([fitted, fitted], ("x0", "x2"), legend=False) is not None
    assert plots.search_trace_plot(fitted, features=[0, 1]) is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_epsilon_warning_when_reference_loss_is_zero():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X[:, 0]

    class Exact:
        def predict(self, Z):
            return Z[:, 0]

    with pytest.warns(UserWarning):
        RashomonExplainer(Exact(), X, y, loss="mse")
