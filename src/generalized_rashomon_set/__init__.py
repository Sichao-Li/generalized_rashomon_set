"""generalized_rashomon_set: sample a Rashomon set around a trained model and
explain feature attributions and interactions across it.

Quick start::

    from generalized_rashomon_set import RashomonExplainer, plots

    explainer = RashomonExplainer(model, X_test, y_test, loss="log_loss",
                                  epsilon_rate=0.05, n_jobs=-1).fit()
    explainer.attribution_summary()      # main-effect range per feature
    explainer.interaction_summary()      # FIS range per feature pair
    plots.swarm_plot_attribution(explainer)
    plots.swarm_plot_interaction(explainer)
    plots.halo_plot(explainer, ("age", "prior"))
"""
from ._combinatorics import boundary_grid, feature_subsets, interaction_combos, sum_to_one_tuples
from ._explainer import RashomonExplainer, enable_logging
from ._feature_space import ImageSegmentFeatureSpace, TabularFeatureSpace
from ._loss import Loss, resolve_loss
from ._predictor import Predictor
from ._results import AttributionCloud, InteractionCloud, RashomonSamples, ReferenceResult
from ._search import SearchTrace, greedy_mask_search
from . import config, explainers, plots, utils

FISExplainer = RashomonExplainer

__version__ = "0.2.0"
__all__ = ["RashomonExplainer", "FISExplainer", "enable_logging", "Loss", "resolve_loss", "Predictor",
           "TabularFeatureSpace", "ImageSegmentFeatureSpace", "AttributionCloud", "InteractionCloud",
           "RashomonSamples", "ReferenceResult", "SearchTrace", "greedy_mask_search",
           "boundary_grid", "feature_subsets", "interaction_combos", "sum_to_one_tuples",
           "config", "explainers", "plots", "utils"]
