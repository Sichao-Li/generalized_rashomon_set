from generalized_rashomon_set.explainers._explainer import fis_explainer
import numpy as np
from ..utils import feature_effect_context, find_all_n_way_feature_pairs
class fis_explainer_context(fis_explainer):
    '''
    The class is used to illustrate the usage of fis in context of archipelago (https://github.com/mtsang/archipelago)
    '''
    def __init__(self, context, **kwargs):
        self.context = context
        super(fis_explainer_context, self).__init__(**kwargs)

    def _get_ref_main_effect(self):
        main_effects_ref = []
        for i in self.v_list:
            X0 = self.input.copy()
            loss_after, loss_before = feature_effect_context(i, X0, self.output, self.model, 10, regression=self.regression, context=self.context)
            main_effects_ref.append(loss_after-loss_before)
        return main_effects_ref

    def _get_ref_joint_effect(self):
        joint_effects_ref = []
        all_n_way_feature_subsets = find_all_n_way_feature_pairs(vlist=self.v_list, n_ways=self.n_ways)
        for subset in all_n_way_feature_subsets:
            if subset[0] != 0:
                subset = np.nonzero(np.in1d(self.v_list, subset))[0]
            X0 = self.input.copy()
            loss_after, loss_before = feature_effect_context(subset, X0=X0, y=self.output, model=self.model, shuffle_times=10, regression=self.regression, context=self.context)
            joint_effects_ref.append(loss_after-loss_before)
        return joint_effects_ref