import copy
import os
import numpy as np
import pandas as pd
from ..utils import model_wrapper
from ..config import OUTPUT_DIR, time_str
from ..utils import find_all_n_way_feature_pairs
from ..utils import load_json, save_json
from ..utils import loss_regression, loss_classification
from ..utils import feature_effect, greedy_search, MR
from ..utils import get_all_joint_effects, get_all_main_effects, get_fis_in_r, get_loss_in_r, get_all_m_with_t_in_range
from ..config import logger

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
        wrapper_for_torch=False,
    ):
        input, output = self.arg_checks(input, output)
        self.logger = logger
        self.n_ways = n_ways
        # self.model = model
        self.quadrants = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        self.input = input
        self.output = output
        self.time_str = time_str
        self.logger.info('You can call function explainer.load_results(results_path="") to load trained results if exist')
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
        self.FIS_in_Rashomon_set = {}
        self.rset_main_effect_raw = {}
        self.rset_main_effect_processed = {}
        self.ref_analysis={}
        self.rset_joint_effect_raw = {}

    def arg_checks(self, input, output):
        if (input is None) or (output is None):
            raise ValueError("Either input or output must be defined")

        return input, output

    def load_results(self, results_path=OUTPUT_DIR):
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
            self.logger.info('Nothing in the directory {}'.format(results_path))
        else:
            for result in analysis_results:
                for content in content_in_results:
                    if result in content:
                        analysis_results[result]['saved'] = True
                        result_path = OUTPUT_DIR + '/' + content
                        analysis_results[result]['path'] = result_path
                        att_name = analysis_results[result]['variable_name']
                        setattr(self, att_name, load_json(result_path))
                        break
                if not analysis_results[result]['saved']:
                    self.logger.info('{} is not in {}'.format(result, content_in_results))

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

    def _get_ref_main_effect(self, model_reliance=False):
        main_effects_ref = []
        mr_ref = []
        if not self.softmax:
            for i in self.v_list:
                X0 = self.input.copy()
                if model_reliance:
                    mr = MR(i, X0, self.output, self.model)
                    mr_ref.append(mr)
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
        if model_reliance:
            return main_effects_ref, mr_ref
        return main_effects_ref

    def _get_ref_joint_effect(self):
        joint_effects_ref = []
        for pair_idx in self.all_pairs:
            # TODO: check if the chunk is useful
            # if subset[0] != 0:
            #     subset = np.nonzero(np.in1d(self.v_list, subset))[0]
            #     print(subset)
            if not self.softmax:
                X0 = self.input.copy()
                loss_after, loss_before = feature_effect(pair_idx, X0=X0, y=self.output, model=self.model, shuffle_times=30, regression=self.regression)
                joint_effects_ref.append(loss_after-loss_before)
            else:
                X0 = copy.copy(self.input.input)
                try:
                    mask_indices = self.input._get_mask_indices_of_feature(pair_idx)
                    loss_after, loss_before = feature_effect(mask_indices, X0=X0, y=self.output, model=self.model, shuffle_times=30, regression=self.regression)
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
        self.all_pairs = find_all_n_way_feature_pairs((self.v_list), n_ways=self.n_ways)
        self.ref_analysis['ref_joint_effects'] = self._get_ref_joint_effect()
        self.ref_analysis['important_features'] = self.v_list
        self.ref_analysis['important_pairs'] = self.all_pairs
        self.logger.info('joint effects calculated and can be called by explainer.ref_joint_effects')
        fis_ref = []
        for idx, i in enumerate(self.all_pairs):
            fis_ref.append((i, abs(self.ref_analysis['ref_joint_effects'][idx] - self.ref_analysis['ref_main_effects'][i[0]] - self.ref_analysis['ref_main_effects'][i[1]])))
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
        self.logger.info('Searching models in the Rashomon set ...')
        for idx, vname in enumerate(vlist):
            m_max_single_boundary_e, points_max, fis_all_plus = greedy_search(vname, bound, loss_ref, model, X, y, direction=True,
                                                             delta=delta, regression=regression)
            points_all_max.append(points_max)
            m_min_single_boundary_e, points_min, fis_all_minus = greedy_search(vname, bound, loss_ref, model, X, y, direction=False,
                                                               delta=delta, regression=regression)
            points_all_min.append(points_min)
            m_single_boundary_e[idx, :, 0] = m_max_single_boundary_e
            m_single_boundary_e[idx, :, 1] = m_min_single_boundary_e
            fis_main_single_boundary_e[idx, :, 0] = fis_all_plus
            fis_main_single_boundary_e[idx, :, 1] = fis_all_minus
        self.rset_main_effect_raw['m_single_boundary_e'] = m_single_boundary_e
        self.rset_main_effect_raw['points_all_max'] = points_all_max
        self.rset_main_effect_raw['points_all_min'] = points_all_min
        self.rset_main_effect_raw['fis_main_single_boundary_e'] = fis_main_single_boundary_e
        save_json(OUTPUT_DIR +'/FIS-main-effect-raw-{}.json'.format(self.time_str), self.rset_main_effect_raw)
        self.logger.info('Searching done and saved to {}'.format(OUTPUT_DIR+'/FIS-main-effect-raw-{}.json').format(self.time_str))
        return m_single_boundary_e, points_all_max, points_all_min, fis_main_single_boundary_e

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
            save_json(OUTPUT_DIR+'/Ref-in-Rashomon-set-analysis-{}.json'.format(self.time_str), self.ref_analysis)

    def rset_explain(self, main_effect=True, interaction_effect=True):
        '''
        Find the range of FIS for each pair of features in the Rashomon set
        '''
        # Main effect processing
        self.logger.info('Start exploring the possible models')

        if main_effect:
            if self.rset_main_effect_raw == {}:
                self._explore_m_in_R(
                    self.epsilon, self.loss, self.v_list, self.model, self.input,
                    self.output, delta=0.1, regression=self.regression)
            else:
                self.logger.info('Already exists, skip')
            self.logger.info('Calculating all main effects of features {} for all models in the Rashomon set'.format(self.v_list))


            if self.rset_main_effect_processed == {}:
                m_multi_boundary_e, loss_diff_multi_boundary_e = get_all_m_with_t_in_range(self.rset_main_effect_raw['points_all_max'],
                                                               self.rset_main_effect_raw['points_all_min'],
                                                               self.epsilon)
                all_main_effects_ratio, all_main_effects_diff, main_effect_complete_list = get_all_main_effects(m_multi_boundary_e,
                                                                                                      self.input, self.output,
                                                                                                      self.model, self.v_list, self.regression)
                self.logger.info('Calculation done')
                self.rset_main_effect_processed['m_multi_boundary_e'] = m_multi_boundary_e
                self.rset_main_effect_processed['all_main_effects_ratio'] = all_main_effects_ratio
                self.rset_main_effect_processed['all_main_effects_diff'] = all_main_effects_diff
                self.rset_main_effect_processed['loss_diff_multi_boundary_e'] = loss_diff_multi_boundary_e
                self.rset_main_effect_processed['main_effect_complete_list'] = main_effect_complete_list
                save_json(OUTPUT_DIR + '/FIS-main-effect-processed-{}.json'.format(self.time_str), self.rset_main_effect_processed)
            else:
                self.logger.info('Already exists, skip')
        if interaction_effect:
            # Joint effect processing
            if self.rset_joint_effect_raw == {}:
                self.logger.info('Calculating all joint effects of feature in pairs {}'. format(len(self.all_pairs)))
                joint_effect_all_pair_set, loss_emp_all_pair_set = get_all_joint_effects(self.rset_main_effect_processed['m_multi_boundary_e'], self.input, self.output, self.v_list, self.n_ways, self.model, regression=self.regression)
                self.rset_joint_effect_raw['joint_effect_all_pair_set'] = np.array(joint_effect_all_pair_set)
                self.rset_joint_effect_raw['loss_emp_all_pair_set'] = np.array(loss_emp_all_pair_set)
                # self.rset_joint_effect_raw['m_multi_boundary_e'] = m_multi_boundary_e
                self.logger.info('Calculation done')
                self.logger.info('Calculating FISC in the Rashomon set for all models in the Rashomon set')
                save_json(OUTPUT_DIR + '/FIS-joint-effect-raw-{}.json'.format(self.time_str), self.rset_joint_effect_raw)

            else:
                self.all_pairs = self.ref_analysis['important_pairs']
                self.logger.info('Already exists, skip')
            if self.FIS_in_Rashomon_set == {}:
                self.fis_in_r = get_fis_in_r(self.all_pairs, np.array(self.rset_joint_effect_raw['joint_effect_all_pair_set']), np.array(self.rset_main_effect_processed['all_main_effects_diff']), self.n_ways, self.quadrants)
                self.loss_in_r = get_loss_in_r(self.all_pairs, np.array(self.rset_joint_effect_raw['loss_emp_all_pair_set']), self.n_ways, self.quadrants, self.epsilon, self.loss)
                for idx, fis_each_pair in enumerate(self.fis_in_r):
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)] = {}
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['feature_idx'] = self.ref_analysis['ref_fis'][idx][0]
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results'] = {}
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['ref'] = self.ref_analysis['ref_fis'][idx][1]
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['min'] = np.min(fis_each_pair)
                    self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['max'] = np.max(fis_each_pair)
                self.logger.info('Calculation done')
                save_json(OUTPUT_DIR+'/FIS-in-Rashomon-set-{}.json'.format(self.time_str), self.FIS_in_Rashomon_set)
                self.logger.info('Explanation is saved to {}'.format(OUTPUT_DIR+'/FIS-in-Rashomon-set-{}.json').format(self.time_str))
            else:
                self.logger.info('Already exists, skip')

    def feature_importance_uni(self, mean=False, std=False):
        '''
        return a list of universal feature importance ranking from the Rashomon set
        '''
        if std: return [np.std(i) for i in np.array(self.rset_main_effect_processed['all_main_effects_diff']).transpose((2, 0, 1, 3)).reshape((len(self.v_list), -1))]
        if mean: return [np.mean(i) for i in np.array(self.rset_main_effect_processed['all_main_effects_diff']).transpose((2, 0, 1, 3)).reshape((len(self.v_list), -1))]





