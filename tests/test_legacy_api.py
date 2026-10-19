"""The 0.1.x interface still works (with deprecation warnings)."""
import numpy as np
import pytest

from generalized_rashomon_set import explainers, plots, utils


def test_fis_explainer_shim(clf_model, clf_data):
    X, y = clf_data
    with pytest.warns(DeprecationWarning):
        ex = explainers.fis_explainer(clf_model, X, y, epsilon_rate=0.05, loss_fn="log_loss",
                                      n_order=2, delta=0.5, predict_proba=True, shuffle_times=5)
    ex.ref_explain(model_reliance=True)
    assert set(ex.ref_analysis) >= {"ref_main_effects", "ref_joint_effects", "ref_fis",
                                    "important_pairs", "ref_model_reliance"}
    ex.rset_explain(main_effect=True, interaction_effect=True)
    p = len(ex.v_list)
    assert ex.rset_main_effect_processed["m_multi_boundary_e"].shape == (3, p, 2)
    assert ex.rset_main_effect_processed["all_main_effects_diff"].shape == (3, p, 2)
    assert ex.fis_in_r.shape == (len(ex.all_pairs), 4)
    assert ex.loss_in_r.shape == ex.fis_in_r.shape
    assert ex.FIS_in_Rashomon_set["pair_idx_0"]["feature_idx"] == ex.all_pairs[0]
    assert ex.loss == ex.loss_ref and ex.input is ex.X
    with pytest.warns(DeprecationWarning):
        plots.halo_plot_multiple(ex, ex, pair_idx=0)
    with pytest.warns(DeprecationWarning):
        plots.swarm_plot_FIS(ex, [ex.all_pairs[0]], vname=["a", "b", "c", "d"])
    with pytest.warns(DeprecationWarning):
        plots.swarm_plot_MR(ex, ex.v_list, plot_all=True)
    with pytest.warns(DeprecationWarning):
        plots.training_process(ex)
    with pytest.raises(NotImplementedError):
        explainers.fis_explainer.load_results(ex, "/nonexistent")


def test_binary_output_maps_to_zero_one_loss(clf_model, clf_data):
    X, y = clf_data
    with pytest.warns(DeprecationWarning):
        ex = explainers.fis_explainer(clf_model, X, y, epsilon_rate=0.05, loss_fn="accuracy_score",
                                      binary_output=True, torch_input=False, delta=0.5)
    assert ex.loss_fn.name == "zero_one_loss"


def test_legacy_utils():
    assert utils.find_all_n_order_feature_pairs([0, 1, 2], 2) == [(0, 1), (0, 2), (1, 2)]
    assert utils.find_all_sum_to_one_pairs(2) == [(a, 10 - a) for a in range(1, 10)]
    X, y = utils.pd_to_numpy(np.zeros((3, 2)), np.zeros(3))
    assert X.shape == (3, 2)
