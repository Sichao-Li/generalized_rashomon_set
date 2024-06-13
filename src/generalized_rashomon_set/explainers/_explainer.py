import copy
import os
import numpy as np
from functools import lru_cache
from ..utils import model_wrapper, model_wrapper_image, model_wrapper_binary_output
from ..config import OUTPUT_DIR, time_str
from ..utils import find_all_n_order_feature_pairs
from ..utils import load_json, save_json
from ..config import logger
import logging
from ._feature_attributor import feature_attributor
from ..utils import find_all_sum_to_one_pairs

class fis_explainer:
    def __init__(
        self,
        model,
        input=None,
        output=None,
        epsilon_rate=0.1,
        loss_fn=None,
        n_order=2,
        torch_input=False,
        delta=0.1,
        binary_output=False
    ):

        self.logger = logger
        self.n_order = n_order
        # self.model = model
        self.quadrants = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        self.input, self.output = self.arg_checks(input, output)
        self.name_id = time_str+'-{}-{}'.format(loss_fn,epsilon_rate)
        self.logger.info('You can call function load_results(explainer, results_path="") to load trained results if exist')
        # self.v_list = self.init_variable_list()
        self.loss_fn = loss_fn
        # image segmentation processing
        if hasattr(input, 'num_features'):
            self.v_list = range(input.num_features)
            self.image_input = True
        else:
            self.image_input = False
            self.v_list = range(len(self.input[-1]))
        self.delta=delta
        self.binary_output=binary_output
        self.prediction_fn_exist = True
        image_input = self.image_input or False
        if binary_output:
            model_wrapper_instance = model_wrapper_binary_output(model, torch_input=torch_input)
        elif image_input:
            model_wrapper_instance = model_wrapper_image(model, torch_input=torch_input, preprocessor=self.input._preprocess)
            self.prediction = self._get_prediction(self.input.input)
        else:
            model_wrapper_instance = model_wrapper(model, torch_input=torch_input)

        self.model = model_wrapper_instance
        self.prediction = self._get_prediction(self.input)
        if not isinstance(self.prediction, np.ndarray):
            self.prediction = self.prediction.detach().numpy()
        # if torch_input:
        #     if self.image_input:
        #         self.model = model_wrapper(model, torch_input=True, image_input=True,
        #                                    preprocessor=self.input._preprocess)
        #         self.prediction = self._get_prediction(self.input.input)
        #     elif self.binary_output:
        #         self.model = model_wrapper(model, torch_input=True, image_input=False, binary_output=self.binary_output)
        #         self.prediction = self._get_prediction(self.input)
        #     else:
        #         self.model = model_wrapper(model, torch_input=True, image_input=False)
        #         self.prediction = self._get_prediction(self.input)
        # else:
        #     self.model = model_wrapper(model, torch_input=False, image_input=False)
        #     self.prediction = self._get_prediction(self.input)


        self.fis_attributor = feature_attributor(self.model, self.loss_fn, self.binary_output, self.delta)
        self.loss = self.fis_attributor.loss_func(self.output, self.prediction)
        # self.loss = loss_func(self.loss_fn, self.output, self.prediction, self.binary_output)
        self.epsilon = self.loss*epsilon_rate
        self.all_pairs = find_all_n_order_feature_pairs((self.v_list), n_order=n_order)
        # self.m_all, self.points_all_positive, self.points_all_negative, self.main_effects = self.explore_m_in_R(self.epsilon, self.loss, range(len(self.input[-1])), model, input, output, delta=0.1, regression=self.regression)
        self.FIS_in_Rashomon_set = {}
        self.rset_main_effect_raw = {}
        self.rset_main_effect_processed = {}
        self.ref_analysis={}
        self.rset_joint_effect_raw = {}

    @staticmethod
    def arg_checks(input, output):
        if (input is None) or (output is None):
            raise ValueError("Either input or output must be defined")
        return input, output

    @lru_cache(maxsize=None)
    def init_variable_list(self):
        if hasattr(self.input, 'num_features'):
            return list(range(self.input.num_features))
        else:
            return list(range(len(self.input[-1])))

    # def configure_model(self, model, torch_input, image_input, binary_output, preprocessor):
    #     if torch_input:
    #         self.configure_torch_model(model, image_input, binary_output, preprocessor)
    #     else:
    #         self.configure_generic_model(model)
    #
    # def configure_torch_model(self, model, image_input, binary_output, preprocessor):
    #     if image_input:
    #         self.model = model_wrapper(model, torch_input=True, image_input=True, preprocessor=preprocessor)
    #     elif binary_output:
    #         self.model = model_wrapper(model, torch_input=True, image_input=False, binary_output=binary_output)
    #     else:
    #         self.model = model_wrapper(model, torch_input=True, image_input=False)
    #     self.prediction = self._get_prediction(self.input)
    #
    # def configure_generic_model(self, model):
    #     self.model = model_wrapper(model, torch_input=False, image_input=False)
    #     self.prediction = self._get_prediction(self.input)

    @staticmethod
    def load_results(explainer, results_path=OUTPUT_DIR):
        content_in_results = os.listdir(results_path)
        analysis_results = {'FIS-in-Rashomon-set': {'saved': False, 'path': '', 'variable_name': 'FIS_in_Rashomon_set'},
                            'FIS-joint-effect-raw': {'saved': False, 'path': '',
                                                     'variable_name': 'rset_joint_effect_raw'},
                            'FIS-main-effect-raw': {'saved': False, 'path': '',
                                                    'variable_name': 'rset_main_effect_raw'},
                            'FIS-main-effect-processed': {'saved': False, 'path': '',
                                                          'variable_name': 'rset_main_effect_processed'},
                            'Ref-in-Rashomon-set-analysis': {'saved': False, 'path': '',
                                                             'variable_name': 'ref_analysis'}}
        if len(content_in_results) == 0:
            explainer.logger.info('Nothing in the directory {}'.format(results_path))
        else:
            for result in analysis_results:
                for content in content_in_results:
                    if result in content:
                        analysis_results[result]['saved'] = True
                        result_path = OUTPUT_DIR + '/' + content
                        analysis_results[result]['path'] = result_path
                        att_name = analysis_results[result]['variable_name']
                        setattr(explainer, att_name, load_json(result_path))
                        break
                if not analysis_results[result]['saved']:
                    explainer.logger.info('{} is not in {}'.format(result, content_in_results))

    def _get_prediction(self, input):
        if self.prediction_fn_exist:
            return self.model.predict(input)
                # return self.model.predict_proba(input)
        else:
            return self.output

    def _get_ref_main_effect(self, model_reliance=False):
        main_effects_ref = []
        mr_ref = []
        if not self.image_input:
            for i in self.v_list:
                X0 = self.input.copy()
                if model_reliance:
                    mr = self.fis_attributor.MR(i, X0, self.output, self.model)
                    mr_ref.append(mr)
                loss_after, loss_before = self.fis_attributor.feature_effect(i, X0, self.output, 30)
                main_effects_ref.append(loss_after-loss_before)
        else:
            for i in self.v_list:
                mask_indices = self.input._get_mask_indices_of_feature(i)
                X0 = copy.copy(self.input.input)
                loss_after, loss_before = self.fis_attributor.feature_effect([mask_indices], X0, self.output, 30)
                main_effects_ref.append(loss_after -loss_before)
        if model_reliance:
            return main_effects_ref, mr_ref
        return main_effects_ref

    def _get_ref_joint_effect(self):
        joint_effects_ref = []
        for pair_idx in self.all_pairs:
            if not self.image_input:
                X0 = self.input.copy()
                loss_after, loss_before = self.fis_attributor.feature_effect(pair_idx, X0=X0, y=self.output, shuffle_times=30)
                joint_effects_ref.append(loss_after-loss_before)
            else:
                X0 = copy.copy(self.input.input)
                try:
                    mask_indices = self.input._get_mask_indices_of_feature(pair_idx)
                    loss_after, loss_before = self.fis_attributor.feature_effect(mask_indices, X0=X0, y=self.output, shuffle_times=30)
                    joint_effects_ref.append(loss_after-loss_before)
                except Exception:
                    self.logger.info('Joint effect of {} is 10000'.format(pair_idx))
                    joint_effects_ref.append(10000)
                    pass
        return joint_effects_ref

    def _get_ref_fis(self):
        '''
        :return: fis of all pairs based on the reference model
        '''
        self.all_pairs = find_all_n_order_feature_pairs((self.v_list), n_order=self.n_order)
        self.ref_analysis['ref_joint_effects'] = self._get_ref_joint_effect()
        self.ref_analysis['important_features'] = self.v_list
        self.ref_analysis['important_pairs'] = self.all_pairs
        self.logger.info('joint effects calculated and can be called by explainer.ref_joint_effects')
        fis_ref = []
        for idx, i in enumerate(self.all_pairs):
            fis_ref.append((i, (self.ref_analysis['ref_joint_effects'][idx] - self.ref_analysis['ref_main_effects'][i[0]] - self.ref_analysis['ref_main_effects'][i[1]])))
        return fis_ref

    def _explore_m_in_R(self, bound, loss_ref, vlist, X, y, delta=0.1):

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
                m: searching in all features in R, the number of model is pxdx2
                points_all_positive, points_all_negative: recorded training process
                fis_main: main effects of all features
        '''
        p = len(vlist)
        d = len(np.arange(0, 1 + 0.1, delta))
        m_single_boundary_e = np.zeros([p, d, 2])
        feature_attribution_main_reference_model = np.zeros([p, d, 2])
        points_all_max = []
        points_all_min = []
        self.logger.info('Searching models in the Rashomon set ...')
        for idx, vname in enumerate(vlist):
            m_max_single_boundary_e, points_max, fis_all_plus = self._greedy_search(vname, bound, loss_ref, X, y, direction=True,
                                                                                    delta=delta)
            points_all_max.append(points_max)
            m_min_single_boundary_e, points_min, fis_all_minus = self._greedy_search(vname, bound, loss_ref, X, y, direction=False,
                                                                                     delta=delta)
            points_all_min.append(points_min)
            m_single_boundary_e[idx, :, 0] = m_max_single_boundary_e
            m_single_boundary_e[idx, :, 1] = m_min_single_boundary_e
            feature_attribution_main_reference_model[idx, :, 0] = fis_all_plus
            feature_attribution_main_reference_model[idx, :, 1] = fis_all_minus
        self.rset_main_effect_raw['m_single_boundary_e'] = m_single_boundary_e
        self.rset_main_effect_raw['points_all_max'] = points_all_max
        self.rset_main_effect_raw['points_all_min'] = points_all_min
        self.rset_main_effect_raw['feature_attribution_main_reference_model'] = feature_attribution_main_reference_model
        save_json(OUTPUT_DIR +'/FIS-main-effect-raw-{}.json'.format(self.name_id), self.rset_main_effect_raw)
        self.logger.info('Searching done and saved to {}'.format(OUTPUT_DIR+'/FIS-main-effect-raw-{}.json').format(self.name_id))
        return m_single_boundary_e, points_all_max, points_all_min, feature_attribution_main_reference_model

    def ref_explain(self, model_reliance=False):
        # return reference model analysis
        if self.ref_analysis == {}:
            self.logger.info('Reference model analysis')
            self.logger.info('Calculating main effect, joint effect and FIS for the reference model')
            if model_reliance:
                self.ref_analysis['ref_main_effects'], self.ref_analysis['ref_model_reliance'] = self._get_ref_main_effect(model_reliance=True)
            self.ref_analysis['ref_main_effects'] = self._get_ref_main_effect()
            self.logger.info(
                'main effects calculated and can be called by explainer.ref_analysis[''ref_main_effects'']')
            unimportant_feature_indices = np.where(np.array(self.ref_analysis['ref_main_effects']) == 0)[0]
            self.logger.info(
                'features with importance 0 are excluded, including {}'.format(unimportant_feature_indices))
            self.v_list = np.array(list(set(unimportant_feature_indices) ^ set(self.v_list)))
            self.ref_analysis['ref_fis'] = self._get_ref_fis()
            self.logger.info('FIS calculated and can be called by explainer.ref_analysis')
            self.logger.info('Calculation done')
            save_json(OUTPUT_DIR+'/Ref-in-Rashomon-set-analysis-{}.json'.format(self.name_id), self.ref_analysis)

    def rset_explain(self, main_effect=True, interaction_effect=True):
        '''
        Find the range of FIS for each pair of features in the Rashomon set
        '''
        # Main effect processing
        self.logger.info('Start exploring the possible models')

        if main_effect:
            if self.rset_main_effect_raw == {}:
                self._explore_m_in_R(
                    self.epsilon, self.loss, self.v_list, self.input,
                    self.output, delta=self.delta)
            else:
                self.logger.info('Already exists, skip')
            self.logger.info('Calculating all main effects of features {} for all models in the Rashomon set'.format(self.v_list))


            if self.rset_main_effect_processed == {}:
                m_multi_boundary_e, loss_diff_multi_boundary_e = self.rset_main_effect_raw['m_single_boundary_e'], self.rset_main_effect_raw['feature_attribution_main_reference_model']
                m_multi_boundary_e = np.array(m_multi_boundary_e).transpose((1, 0, 2))
                loss_diff_multi_boundary_e = np.array(loss_diff_multi_boundary_e).transpose((1, 0, 2))
                all_main_effects_ratio, all_main_effects_diff, main_effect_complete_list = self._get_all_main_effects(
                    m_multi_boundary_e, self.input, self.output, self.v_list, self.delta)
                self.logger.info('Calculation done')
                self.rset_main_effect_processed['m_multi_boundary_e'] = m_multi_boundary_e
                self.rset_main_effect_processed['all_main_effects_ratio'] = all_main_effects_ratio
                self.rset_main_effect_processed['all_main_effects_diff'] = all_main_effects_diff
                self.rset_main_effect_processed['loss_diff_multi_boundary_e'] = loss_diff_multi_boundary_e
                self.rset_main_effect_processed['main_effect_complete_list'] = main_effect_complete_list
                save_json(OUTPUT_DIR + '/FIS-main-effect-processed-{}.json'.format(self.name_id), self.rset_main_effect_processed)
            else:
                self.logger.info('Already exists, skip')
        if interaction_effect:
            # Joint effect processing
            if self.rset_joint_effect_raw == {}:
                self.logger.info('Calculating all joint effects of feature in pairs {}'. format(len(self.all_pairs)))
                joint_effect_all_pair_set, loss_emp_all_pair_set = self._get_all_joint_effects(self.rset_main_effect_processed['m_multi_boundary_e'], self.input, self.output, self.v_list, self.n_order)
                self.rset_joint_effect_raw['joint_effect_all_pair_set'] = np.array(joint_effect_all_pair_set)
                self.rset_joint_effect_raw['loss_emp_all_pair_set'] = np.array(loss_emp_all_pair_set)
                # self.rset_joint_effect_raw['m_multi_boundary_e'] = m_multi_boundary_e
                self.logger.info('Calculation done')
                self.logger.info('Calculating FISC in the Rashomon set for all models in the Rashomon set')
                save_json(OUTPUT_DIR + '/FIS-joint-effect-raw-{}.json'.format(self.name_id), self.rset_joint_effect_raw)

            else:
                self.all_pairs = self.ref_analysis['important_pairs']
                self.logger.info('Already exists, skip')
            if self.FIS_in_Rashomon_set == {}:
                self.fis_in_r = self._get_fis_in_r(self.all_pairs, np.array(self.rset_joint_effect_raw['joint_effect_all_pair_set']), np.array(self.rset_main_effect_processed['all_main_effects_diff']), self.n_order, self.quadrants)
                self.loss_in_r = self._get_loss_in_r(self.all_pairs, np.array(self.rset_joint_effect_raw['loss_emp_all_pair_set']), self.n_order, self.quadrants, self.epsilon, self.loss)
                for idx, fis_each_pair in enumerate(self.fis_in_r):
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)] = {}
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['feature_idx'] = self.ref_analysis['ref_fis'][idx][0]
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results'] = {}
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['ref'] = self.ref_analysis['ref_fis'][idx][1]
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['min'] = np.min(fis_each_pair)
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['max'] = np.max(fis_each_pair)
                self.logger.info('Calculation done')
                save_json(OUTPUT_DIR+'/FIS-in-Rashomon-set-{}.json'.format(self.name_id), self.FIS_in_Rashomon_set)
                self.logger.info('Explanation is saved to {}'.format(OUTPUT_DIR+'/FIS-in-Rashomon-set-{}.json').format(self.name_id))
            else:
                self.logger.info('Already exists, skip')
        logging.shutdown()

    def feature_importance_uni(self, mean=False, std=False):
        '''
        return a list of universal feature importance ranking from the Rashomon set
        '''
        if std: return [np.std(i) for i in np.array(self.rset_main_effect_processed['all_main_effects_diff']).transpose((2, 0, 1, 3)).reshape((len(self.v_list), -1))]
        if mean: return [np.mean(i) for i in np.array(self.rset_main_effect_processed['all_main_effects_diff']).transpose((2, 0, 1, 3)).reshape((len(self.v_list), -1))]

    def _greedy_search(self, vidx, bound, loss_ref, X, y, delta=0.1, direction=True):
        '''
        greedy search possible m for a single feature
            Input:
                vidx: variable name list of length n
                bound: loss boundary in R set
                loss_ref: loss of reference model
                model: reference model
                X, y: model input and expected output in numpy
                delta: the range of spliting 0 to 1
                direction: exploring directions. When True, explore from 1 to 1+, else 1 to 1-

            Output:
                m_all: m for a feature in a nx2 matrix
                points_all: recorded points when exploring
                fis_all: fis for reference model
        '''
        m_all = []
        points_all = []
        fis_all = []
        loss_temp = 0
    #     count the tolerance
        loss_count = 0
        feature_attribution_main = 0
    #   for single feature at position m
        m = 1
        for i in np.arange(0, 1+0.1, delta):
            # include endpoint [0.1 ..., 1]
            count = 1
            # learning rate
            lr = 0.1
            points = []
    #     termination condition: the precision of acc .0001
            while count <= 4:
        #         input new input X0 and calculate the loss
                X0 = X.copy()
                if direction:
                    X0[:, vidx] = X0[:, vidx] * (m + lr)
                if not direction:
                    X0[:, vidx] = X0[:, vidx] * (m - lr)
                pred = self.model.predict(X0)
                loss_m = self.fis_attributor.loss_func(y, pred)
    #             the diffrence of changed loss and optimal loss
                mydiff = loss_m - loss_ref

                if mydiff<i*bound:
                    if direction:
                    #     if the loss within the bound, then m increses
                        m = m+lr
                    if not direction:
                        m = m-lr
                    loss_after, loss_before = self.fis_attributor.feature_effect(vidx, X0, y, 30)
                    feature_attribution_main = loss_after - loss_before
                    points.append([m, mydiff])
        #             if the loss within the bound but stays same for loss_count times, then the vt is unimportant (the attribution of the feature is assigned 0, as the power of the single feature is not enough to change loss).
                    if loss_temp == loss_m:
                        loss_count = loss_count+1
                        if loss_count > 100:
                            feature_attribution_main = 0
                            break
                    else:
                        loss_temp = loss_m
        #                 otherwise change lr and try again
                else:
                    lr=lr*0.1
                    count = count+1
                logger.info('Feature {} at boundary {} * epsilon with m {} achieves loss difference {}'.format(vidx, i, m, mydiff))
            points_all.append(points)
            m_all.append(m)
            # calculate fis based on m
            fis_all.append(feature_attribution_main)
        return m_all, points_all, fis_all

    def _get_feature_interaction_effects_all_pairs(self, X, y, vlist, n_order, m_all):
        '''
        Calculate the feature interaction effect for all pairs.
        '''
        joint_effect_all_pair = []
        loss_emp_all_pair = []
        for subset in find_all_n_order_feature_pairs(vlist, n_order):
            subset_idx = np.nonzero(np.in1d(vlist, subset))[0]
            joint_effect_single_pair, loss_emp_single_pair = self.fis_attributor.feature_interaction_effect(subset, m_all, X, y, subset_idx=subset_idx)
            joint_effect_all_pair.append(joint_effect_single_pair)
            loss_emp_all_pair.append(loss_emp_single_pair)
        return joint_effect_all_pair, loss_emp_all_pair

    def _get_all_main_effects(self, m_multi_boundary_e, input, output, v_list, delta):
        '''
        :param m_multi_boundary_e: an m matrix in shape [p, d, 2]
        :param model: reference model
        :return:
            main_effect_all_ratio: main effects of all features in ratio
            main_effect_all_diff: main effects of all features in difference
            main_effect_complete_list: main effects of all features in all models
        '''
        main_effect_all_diff = np.zeros(m_multi_boundary_e.shape)
        main_effect_all_ratio = np.zeros(m_multi_boundary_e.shape)
        m_prev = np.inf
        loss_before, loss_after = 1, 1
        main_effect_complete_list = []
        for idxj, j in enumerate(np.arange(0, 1 + 0.1, delta)):
            for idxi, i in enumerate(v_list):
                for k in range(2):
                    X0 = input.copy()
                    if m_multi_boundary_e[idxj, idxi, k] == m_prev:
                        main_effect_all_ratio[idxj, idxi, k] = loss_after / loss_before
                        main_effect_all_diff[idxj, idxi, k] = loss_after - loss_before
                    else:
                        X0[:, i] = X0[:, i] * m_multi_boundary_e[idxj, idxi, k]
                        # make sure X1 and X2 are consistent with X0
                        X1 = X0.copy()
                        loss_after, loss_before = self.fis_attributor.feature_effect(i, X1, output, 30)
                        main_effect_all_ratio[idxj, idxi, k] = loss_after / loss_before
                        main_effect_all_diff[idxj, idxi, k] = loss_after - loss_before
                        m_prev = m_multi_boundary_e[idxj, idxi, k]
                        sub_list = []
                        for idxt, t in enumerate(v_list):
                            X2 = X1.copy()
                            loss_after, loss_before = self.fis_attributor.feature_effect(t, X2, output, 30)
                            sub_list.append(loss_after - loss_before)
                        main_effect_complete_list.append(sub_list)
        return main_effect_all_ratio, main_effect_all_diff, main_effect_complete_list

    def _get_all_joint_effects(self, m_multi_boundary_e, input, output, v_list, n_order):
        '''
        :param m_multi_boundary_e: an m matrix in shape [p, d, 2]
        :return:
            joint_effect_all_pair_set: all joint effects of features [5, n_joint_pairs, 36] in fis, where 36 is 2^2*9
            loss_emp_all_pair_set: all joint effects of features [5, n_joint_pairs, 36] in loss
        '''
        m_multi_boundary_e = m_multi_boundary_e.transpose((1, 0, 2))
        joint_effect_all_pair, loss_emp = self._get_feature_interaction_effects_all_pairs(input, output, v_list, n_order, m_multi_boundary_e)
        return joint_effect_all_pair, loss_emp

    def _get_fis_in_r(self, all_pairs, joint_effect_all_pair_set, main_effect_all_diff, n_order, quadrants):
        '''
        :param pairs: all pairs of interest
        :param joint_effect_all_pair_set: all joint effects of these pairs [5, n_pairs, 36]
        :param main_effect_all_diff: all main effects of these features in the pair [5, 11, n_features, 2]
        :return: fis of all pairs in the Rashomon set
        '''
        fis_rset = np.ones(joint_effect_all_pair_set.shape)
        joint_effect_all_pair_e = joint_effect_all_pair_set  # [n_pairs, 36]
        main_effect_all_diff_e = main_effect_all_diff  # [11, n_features, 2]
        main_effect_all_diff_e_reshaped = main_effect_all_diff_e.transpose((1, 0, 2))  # [n_features, 11, 2]
        all_pairs_mask = find_all_n_order_feature_pairs((range(len(main_effect_all_diff_e_reshaped))), n_order=n_order)
        # fi is n_featurex11x2, fij_joint is n_pairx36
        for idx, pair in enumerate(all_pairs):
            logger.info('Calculating :pair {} with index {} and {}'.format(idx, pair[0], pair[1]))
            fij_joint = joint_effect_all_pair_e[idx]
            fi = main_effect_all_diff_e_reshaped[all_pairs_mask[idx][0]]
            fj = main_effect_all_diff_e_reshaped[all_pairs_mask[idx][1]]
            # 9 paris
            sum_to_one = find_all_sum_to_one_pairs(n_order)
            for idxk, sum in enumerate(sum_to_one):
                for idxq, quadrant in enumerate(quadrants):
                    # for each pair, find the main effect
                    single_fis = fij_joint[[idxk * 4 + quadrant]] - fi[sum[0]][quadrants[quadrant][0]] - fj[sum[-1]][
                        quadrants[quadrant][-1]]
                    fis_rset[idx, idxk * 4 + quadrant] = single_fis
        return fis_rset

    def _get_loss_in_r(self, all_pairs, joint_loss_pair_set, n_order, quadrants, epsilon, loss):
        '''
        :param pairs: all pairs of interest
        :param joint_loss_pair_set: all joint losses of these pairs
        :return: loss difference of all pairs in the Rashomon set
        '''
        loss_rset = np.ones(joint_loss_pair_set.shape)
        joint_effect_all_pair_e = joint_loss_pair_set
        for idx, pair in enumerate(all_pairs):
            fij_joint = joint_effect_all_pair_e[idx]
            # 9 paris
            sum_to_one = find_all_sum_to_one_pairs(n_order)
            for idxk, sum in enumerate(sum_to_one):
                for idxq, quadrant in enumerate(quadrants):
                    # for each pair, find the main effect
                    single_fis = (
                        fij_joint[[idxk * 4 + quadrant]] - epsilon - loss)
                    loss_rset[idx, idxk * 4 + quadrant] = single_fis
        return loss_rset
