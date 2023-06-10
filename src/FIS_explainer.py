import numpy as np
from PIL import Image
import torch
import logging
import time
import copy
from feature_importance_helper import *
from feature_interaction_score_utilities import *
import os
ROOT_DIR = os.getcwd()
OUTPUT_DIR = ROOT_DIR+'/../results'
LOG_DIR = ROOT_DIR+'/../logs'
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.FileHandler("log_{}.log".format(time.strftime("%Y%m%d-%H%M%S")), mode='w'),
                              stream_handler])





class model_wrapper:
    def __init__(
            self,
            model,
            wrapper_for_torch,
            softmax,
            preprocessor=None,
    ):
        self.model = model
        self.wrapper_for_torch = wrapper_for_torch
        self.softmax = softmax
        self.preprocessor = preprocessor

    def predict(self, X):
        if self.wrapper_for_torch:
            if hasattr(self.model, 'predict'):
                X = torch.tensor(X).float()
                return self.model.predict(X).detach().numpy()
            else:
                if self.softmax:
                    if hasattr(X, 'shape'):
                        X = Image.fromarray(X)
                    X = self.preprocessor(X)
                    return self.model(X).squeeze(0).softmax(0)

                else:
                    X = torch.tensor(X).float()
                    return self.model(X).detach().numpy()

        else:
            X = X
            if hasattr(self.model, 'predict'):
                if not hasattr(self.model, 'predict_proba'):
                    return self.model.predict(X)
                else:
                    return self.model.predict_proba(X)[:, 1]
            else:
                return self.model(X)


class fis_explainer:
    def __init__(
        self,
        model,
        input=None,
        output=None,
        epsilon_rate=0.1,
        loss_fn=None,
        n_ways=2,
        feature_idx=None,
        return_ref_main_effects=False,
        return_ref_pairwise_effects=False,
        wrapper_for_torch=False,
    ):
        input, output = self.arg_checks(input, output)
        self.n_ways = n_ways
        # self.model = model
        self.quadrants = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        self.input = input
        self.output = output
        self.time_str = time.strftime("%Y%m%d-%H%M%S")
        self.logger = logging.getLogger(__name__)
        if hasattr(input, 'num_features'):
            self.v_list = range(input.num_features)
            self.softmax = True
            # self.input = self.input._preprocess()
        else:
            self.softmax = False
            self.v_list = range(len(self.input[-1]))
        if loss_fn == 'regression':
            self.regression = True
        else:
            self.regression = False

        self.prediction_fn_exist = True
        # define if a model is built from torch
        if wrapper_for_torch:
            if self.softmax:
                self.model = model_wrapper(model, wrapper_for_torch=True, softmax=True, preprocessor=self.input._preprocess)
                self.prediction = self._get_prediction(self.input.input)
            else:
                self.model = model_wrapper(model, wrapper_for_torch=True, softmax=False)
                self.prediction = self._get_prediction(self.input)
        else:
            self.model = model_wrapper(model, wrapper_for_torch=False, softmax=False)
            self.prediction = self._get_prediction(self.input)
        # self.model.predict = predict(self.model, self.input)

        # if hasattr(self.model, 'predict'):
        #     self.prediction_fn_exist = True
        # else:
        #     self.prediction_fn_exist = False
        # self.prediction = self._get_prediction(self.input)
        if not isinstance(self.prediction, np.ndarray):
            self.prediction = self.prediction.detach().numpy()
        self.loss = self.loss_fn(loss_fn)

        self.epsilon = self.loss*epsilon_rate
        self.all_pairs = find_all_n_way_feature_pairs((self.v_list), n_ways=n_ways)
        # self.m_all, self.points_all_positive, self.points_all_negative, self.main_effects = self.explore_m_in_R(self.epsilon, self.loss, range(len(self.input[-1])), model, input, output, delta=0.1, regression=self.regression)
        if return_ref_main_effects:
            self.ref_main_effects = self._get_ref_main_effect()
        if return_ref_pairwise_effects:
            self.ref_joint_effects = self._get_ref_joint_effect()

        self.raw_results_dic = {}
        self.FIS_in_Rashomon_set = {}
    def arg_checks(self, input, output):
        if (input is None) or (output is None):
            raise ValueError("Either input or output must be defined")

        return input, output

    def loss_fn(self, loss_fn):
        if loss_fn == 'regression':
            return loss_regression(self.output, self.prediction)
        elif loss_fn == 'classification':
            return loss_classification(self.output, self.prediction)

    def _get_prediction(self, input):
        if self.prediction_fn_exist:
            return self.model.predict(input)
                # return self.model.predict_proba(input)
        else:
            return self.output

    def _get_ref_main_effect(self):
        main_effects_ref = []
        if not self.softmax:
            for i in self.v_list:
                X0 = self.input.copy()
                loss_after, loss_before = feature_effect(i, X0, self.output, self.model, 30, regression=self.regression)
                main_effects_ref.append(loss_after-loss_before)
        else:
            for i in self.v_list:
                mask_indices = self.input._get_mask_indices_of_feature(i)
                X0 = copy.copy(self.input.input)
                # X0 = Image.fromarray(X0)
                # X0 = self.input._preprocess(X0)
                # image_trans._transform(mask_indices, [0, 0, 0, 0, 0, 0, 0, 0])
                loss_after, loss_before = feature_effect([mask_indices], X0, self.output, self.model, 30, regression=self.regression)
                main_effects_ref.append(loss_after -loss_before)

        return main_effects_ref

    def _get_ref_joint_effect(self):
        joint_effects_ref = []
        all_n_way_feature_subsets = find_all_n_way_feature_pairs(vlist=self.v_list, n_ways=self.n_ways)
        for subset in all_n_way_feature_subsets:
            if subset[0] != 0:
                subset = np.nonzero(np.in1d(self.v_list, subset))[0]
            if not self.softmax:
                X0 = self.input.copy()
                loss_after, loss_before = feature_effect(subset, X0=X0, y=self.output, model=self.model, shuffle_times=30, regression=self.regression)
                joint_effects_ref.append(loss_after-loss_before)
            else:
                X0 = copy.copy(self.input.input)
                try:
                    mask_indices = self.input._get_mask_indices_of_feature(subset)
                    loss_after, loss_before = feature_effect(mask_indices, X0=X0, y=self.output, model=self.model, shuffle_times=30, regression=self.regression)
                    joint_effects_ref.append(loss_after-loss_before)
                except Exception:
                    self.logger.info('Joint effect of {} is 10000'.format(subset))
                    joint_effects_ref.append(10000)
                    pass
        return joint_effects_ref

    def _get_ref_fis(self):
        '''
        :return: fis of all pairs based on the reference model
        '''
        if self.ref_joint_effects == []:
            self.logger.info('Make sure return joint effects and main effects True when constructing explainer')
            self.ref_joint_effects = self._get_ref_joint_effect()
        if self.ref_main_effects == []:
            self.logger.info('Make sure return joint effects and main effects True when constructing explainer')
            self.ref_main_effects = self._get_ref_main_effect()
        fis_ref = []
        pairs = find_all_n_way_feature_pairs(vlist=self.v_list, n_ways=self.n_ways)
        for idx, i in enumerate(pairs):
            fis_ref.append((i, abs(self.ref_joint_effects[idx] - self.ref_main_effects[i[0]] - self.ref_main_effects[i[1]])))
        return fis_ref
    def _explore_m_in_R(self, bound, loss_ref, vlist, model, X, y, delta=0.01, regression=True):

        '''
        Explore the Rashomon set for the black box model by searching m within a boundary.
            Input:
                bound: boundary of R set, defined by epsilon
                loss_ref: loss of reference model
                vlist: variable list of length p
                model: optimal model
                X,y: data set
                delta: the parameter splitting from 0 to 1, d=1/delta
            Output:
                m: possible masks for all features in R, pxdx2
                points_all_positive, points_all_negative: recorded training process
                fis_main: main effects of all features
        '''
        p = len(vlist)
        d = len(np.arange(0, 1 + 0.1, delta))
        m_single_boundary_e = np.zeros([p, d, 2])
        fis_main_single_boundary_e = np.zeros([p, d, 2])
        points_all_max = []
        points_all_min = []
        self.logger.info('Searching started...')
        for idx, vname in enumerate(vlist):
            m_max_single_boundary_e, points_max, fis_all_plus = greedy_search(idx, bound, loss_ref, model, X, y, direction=True,
                                                             delta=delta, regression=regression)
            points_all_max.append(points_max)
            m_min_single_boundary_e, points_min, fis_all_minus = greedy_search(idx, bound, loss_ref, model, X, y, direction=False,
                                                               delta=delta, regression=regression)
            points_all_min.append(points_min)
            m_single_boundary_e[idx, :, 0] = m_max_single_boundary_e
            m_single_boundary_e[idx, :, 1] = m_min_single_boundary_e
            fis_main_single_boundary_e[idx, :, 0] = fis_all_plus
            fis_main_single_boundary_e[idx, :, 1] = fis_all_minus
        self.raw_results_dic['m_single_boundary_e'] = m_single_boundary_e
        self.raw_results_dic['points_all_max'] = points_all_max
        self.raw_results_dic['points_all_min'] = points_all_max
        self.raw_results_dic['fis_main_single_boundary_e'] = fis_main_single_boundary_e

        save_json(OUTPUT_DIR+'/raw-results-dic-{}.json'.format(self.time_str), self.raw_results_dic)
        self.logger.info('Searching done and saved to {}'.format(OUTPUT_DIR+'/raw-results-dic-{}.json').format(self.time_str))
        return m_single_boundary_e, points_all_max, points_all_min, fis_main_single_boundary_e

    def explain(self):
        '''
        Find the range of FIS for each pair of features in the Rashomon set
        '''
        self.FIS_in_Rashomon_set = {}
        self.fis_ref = self._get_ref_fis()
        if self.raw_results_dic == {}:
            self._explore_m_in_R(
                self.epsilon, self.loss, self.v_list, self.model, self.input,
                self.output, delta=0.1, regression=False)

        self.logger.info('Start analyzing...')
        self.logger.info('Calculating all main effects of features')
        m_multi_boundary_e = get_all_m_with_t_in_range(self.raw_results_dic['points_all_max'],
                                                                           self.raw_results_dic['points_all_min'],
                                                                           self.epsilon)
        all_main_effects_ratio, all_main_effects_diff = get_all_main_effects(m_multi_boundary_e,
                                                                                              self.input, self.output,
                                                                                              self.model, self.v_list, self.regression)
        self.logger.info('Calculation done')
        self.logger.info('Calculating all joint effects of feature pairs')
        joint_effect_all_pair_set, loss_emp_all_pair_set = get_all_joint_effects(m_multi_boundary_e, self.input, self.output, self.v_list, self.n_ways, self.model, self.regression)
        self.logger.info('Calculation done')
        self.logger.info('Calculating FISC in the Rashomon set')
        self.fis_in_r = get_fis_in_r(self.all_pairs, np.array(joint_effect_all_pair_set), all_main_effects_diff, self.n_ways, self.quadrants)
        self.loss_in_r = get_loss_in_r(self.all_pairs, np.array(loss_emp_all_pair_set), self.n_ways, self.quadrants, self.epsilon, self.loss)
        for idx, fis_each_pair in enumerate(self.fis_in_r):
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)] = {}
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['feature_idx'] = self.fis_ref[idx][0]
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results'] = {}
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['ref'] = self.fis_ref[idx][1]
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['min'] = np.min(fis_each_pair)
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['max'] = np.max(fis_each_pair)
        self.logger.info('Calculation done')
        save_json(OUTPUT_DIR+'/FIS-in-Rashomon-set-{}.json'.format(self.time_str), self.FIS_in_Rashomon_set)
        self.logger.info('Explanation is saved to {}'.format(OUTPUT_DIR+'/FIS-in-Rashomon-set-{}.json').format(self.time_str))


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






