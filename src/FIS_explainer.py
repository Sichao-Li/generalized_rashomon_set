import numpy as np
from PIL import Image
import torch
import time
import copy
from feature_importance_helper import *
from feature_interaction_score_utilities import *
ROOT_DIR = r'C:\Users\chaoL\Documents\Project\Exploring the FIS in the Rashomon set'
OUTPUT_DIR = ROOT_DIR+'/results'

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
                    print(subset)
                    joint_effects_ref.append(10000)
                    pass
        return joint_effects_ref

    def _get_ref_fis(self):
        '''
        :return: fis of all pairs based on the reference model
        '''
        if self.ref_joint_effects == []:
            print('Make sure return joint effects and main effects True when constructing explainer')
            self.ref_joint_effects = self._get_ref_joint_effect()
        if self.ref_main_effects == []:
            print('Make sure return joint effects and main effects True when constructing explainer')
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
        m = np.zeros([p, d, 2])
        fis_main = np.zeros([p, d, 2])
        points_all_max = []
        points_all_min = []
        print('Searching started...')
        for idx, vname in enumerate(vlist):
            m_plus, points_max, fis_all_plus = greedy_search(idx, bound, loss_ref, model, X, y, direction=True,
                                                             delta=delta, regression=regression, verbose=False)
            points_all_max.append(points_max)
            m_minus, points_min, fis_all_minus = greedy_search(idx, bound, loss_ref, model, X, y, direction=False,
                                                               delta=delta, regression=regression, verbose=False)
            points_all_min.append(points_min)
            m[idx, :, 0] = m_plus
            m[idx, :, 1] = m_minus
            fis_main[idx, :, 0] = fis_all_plus
            fis_main[idx, :, 1] = fis_all_minus
        self.raw_results_dic['m'] = m
        self.raw_results_dic['points_all_max'] = points_all_max
        self.raw_results_dic['points_all_min'] = points_all_max
        self.raw_results_dic['fis_main'] = fis_main
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_json(OUTPUT_DIR+'/raw-results-dic-{}.json'.format(time_str), self.raw_results_dic)
        print('Searching done and saved to {}'.format(OUTPUT_DIR+'/raw-results-dic-{}.json').format(time_str))
        return m, points_all_max, points_all_min, fis_main

    def _get_all_m_with_t_in_range(self, points_all_max, points_all_min, epsilon):
        '''
        :param points_all_max: all explored points in positive direction
        :param points_all_min: all explored points in negative direction
        :param epsilon: dominant boundary
        :return: an m matrix in shape [5, 11, n_features, 2] corresponding to sub-dominant boundary, e.g. 0.2 * epsilon
        '''
        points_all_positive_reshaped = list_flatten(points_all_max, reverse=False)
        points_all_negative_reshaped = list_flatten(points_all_min, reverse=True)
        p = len(np.arange(0.2, 1 + 0.2, 0.2))
        d = len(np.arange(0.0, 1 + 0.1, 0.1))
        n_features = len(points_all_min)
        m_all_sub_set = np.ones([p, d, n_features, 2], dtype=np.float64)
        for idxj, sub_boundary_rate in enumerate(np.arange(0.2, 1 + 0.2, 0.2)):
            for idxk, j in enumerate(np.arange(0.0, 1 + 0.1, 0.1)):
                for idxi, feature in enumerate(points_all_positive_reshaped):
                    for idv in (feature):
                        if idv[-1] <= j * sub_boundary_rate * epsilon:
                            m_all_sub_set[idxj, idxk, idxi, 0] = idv[0]
                        else:
                            break
                for idxi, feature in enumerate(points_all_negative_reshaped):
                    for idv in (feature):
                        if idv[-1] <= j * sub_boundary_rate * epsilon:
                            m_all_sub_set[idxj, idxk, idxi, 1] = idv[0]
                        else:
                            break
        #   np.transpose(1,0,2)
        return m_all_sub_set

    def _get_all_main_effects(self, m_all_sub_set, X_test, y_test, model):
        '''
        :param m_all_sub_set: an m matrix in shape [5, 11, n_features, 2]
        :param model: reference model
        :return:
            fi_all_ratio: main effects of all features in ratio
            fi_all_diff: main effects of all features in difference
        '''
        fi_all_diff = np.zeros(m_all_sub_set.shape)
        fi_all_ratio = np.zeros(m_all_sub_set.shape)
        for idx, sub_boundary_rate in enumerate(np.arange(0.2, 1.2, 0.2)):
            for idxj, j in enumerate(np.arange(0, 1 + 0.1, 0.1)):
                for i in range(len(self.v_list)):
                    for k in range(2):
                        X0 = X_test.copy()
                        X0[:, i] = X0[:, i] * m_all_sub_set[idx, idxj, i, k]
                        loss_after, loss_before = feature_effect(i, X0, y_test, model, 30, self.regression)
                        fi_all_ratio[idx,idxj, i, k] = loss_after / loss_before
                        fi_all_diff[idx,idxj, i, k] = loss_after - loss_before
        return fi_all_ratio, fi_all_diff

    def _get_all_joint_effects(self, m_all_sub_set):
        '''
        :param m_all_sub_set: an m matrix in shape [5, 11, n_features, 2]
        :return:
            joint_effect_all_pair_set: all joint effects of features [5, n_joint_pairs, 36] in fis, where 36 is 2^2*9
            loss_emp_all_pair_set: all joint effects of features [5, n_joint_pairs, 36] in loss
        '''
        joint_effect_all_pair_set = []
        loss_emp_all_pair_set = []
        for m_all in m_all_sub_set:
            m_all = m_all.transpose((1,0,2))
            joint_effect_all_pair, loss_emp = Interaction_effect_all_pairs(self.input, self.output, self.v_list, self.n_ways, self.model, m_all, regression=self.regression)
            joint_effect_all_pair_set.append(joint_effect_all_pair)
            loss_emp_all_pair_set.append(loss_emp)
        return joint_effect_all_pair_set, loss_emp_all_pair_set

    def _get_fis_in_r(self, pairs, joint_effect_all_pair_set, main_effect_all_diff):
        '''
        :param pairs: all pairs of interest
        :param joint_effect_all_pair_set: all joint effects of these pairs
        :param main_effect_all_diff: all main effects of these features in the pair
        :return: fis of all pairs in the Rashomon set
        '''
        quadrants = self.quadrants
        fis_rset = np.ones(joint_effect_all_pair_set.shape)
        for i in range(5):
            joint_effect_all_pair_e = joint_effect_all_pair_set[i]
            main_effect_all_diff_e = main_effect_all_diff[i]
            main_effect_all_diff_e_reshaped = main_effect_all_diff_e.transpose((1, 0, 2))
            for idx, pair in enumerate(pairs):
                # fi is 40x11x2, fij_joint is 780x36
                fi = main_effect_all_diff_e_reshaped[pair[0]]
                fj = main_effect_all_diff_e_reshaped[pair[1]]
                fij_joint = joint_effect_all_pair_e[idx]
                # 9 paris
                sum_to_one = find_all_sum_to_one_pairs(self.n_ways)
                for idxk, sum in enumerate(sum_to_one):
                    for idxq, quadrant in enumerate(quadrants):
                        # for each pair, find the main effect
                        single_fis = abs(
                            fij_joint[[idxk * 4 + quadrant]] - fi[sum[0]][quadrants[quadrant][0]] - fj[sum[-1]][
                                quadrants[quadrant][-1]])
                        # single_fis = (
                        #     fij_joint[[idxk * 4 + quadrant]] - fi[sum[0]][quadrants[quadrant][0]] - fj[sum[-1]][
                        #         quadrants[quadrant][-1]])
                        fis_rset[i, idx, idxk * 4 + quadrant] = single_fis
        return fis_rset.transpose((1,0,2)).reshape(len(self.all_pairs), -1)
    
    def _get_loss_in_r(self, pairs, joint_loss_pair_set):
        '''
        :param pairs: all pairs of interest
        :param joint_loss_pair_set: all joint losses of these pairs
        :return: loss difference of all pairs in the Rashomon set
        '''
        loss_rset = np.ones(joint_loss_pair_set.shape)
        for i, e_sub in enumerate(np.arange(0.2, 1.2, 0.2)):
            joint_effect_all_pair_e = joint_loss_pair_set[i]
            for idx, pair in enumerate(pairs):
                fij_joint = joint_effect_all_pair_e[idx]
                # 9 paris
                sum_to_one = find_all_sum_to_one_pairs(self.n_ways)
                for idxk, sum in enumerate(sum_to_one):
                    for idxq, quadrant in enumerate(self.quadrants):
                        # for each pair, find the main effect
                        single_fis = abs(
                            fij_joint[[idxk * 4 + quadrant]] - e_sub*self.epsilon-self.loss)
                        # single_fis = (
                        #     fij_joint[[idxk * 4 + quadrant]] - e_sub*self.epsilon-self.loss)
                        loss_rset[i, idx, idxk * 4 + quadrant] = single_fis
        return loss_rset.transpose((1,0,2)).reshape(len(self.all_pairs), -1)

    def explain(self):
        self.FIS_in_Rashomon_set = {}
        '''
        Find the range of FIS for each pair of features in the Rashomon set
        '''
        fis_ref = self._get_ref_fis()
        m_all_optimal, points_all_max_optimal, points_all_min_optimal, fis_all_optimal = self._explore_m_in_R(
            self.epsilon, self.loss, self.v_list, self.model, self.input,
            self.output, delta=0.1, regression=False)

        print('Start analyzing...')
        print('Calculating all main effects of features')
        m_all_sub_set_optimal = self._get_all_m_with_t_in_range(points_all_max_optimal,
                                                                           points_all_min_optimal,
                                                                           self.epsilon)
        all_main_effects_ratio, all_main_effects_diff = self._get_all_main_effects(m_all_sub_set_optimal,
                                                                                              self.input, self.output,
                                                                                              self.model)
        print('Calculation done')
        print('Calculating all joint effects of feature pairs')
        joint_effect_all_pair_set, loss_emp_all_pair_set = self._get_all_joint_effects(m_all_sub_set_optimal)
        print('Calculation done')
        print('Calculating FISC in the Rashomon set')
        fis_in_r = self._get_fis_in_r(self.all_pairs, np.array(joint_effect_all_pair_set), all_main_effects_diff)
        for idx, fis_each_pair in enumerate(fis_in_r):
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)] = {}
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['feature_idx'] = fis_ref[idx][0]
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results'] = {}
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['ref'] = fis_ref[idx][1]
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['min'] = np.min(fis_each_pair)
            self.FIS_in_Rashomon_set['pair_idx_{}'.format(idx)]['results']['max'] = np.max(fis_each_pair)
        print('Calculation done')
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_json(OUTPUT_DIR+'/FIS-in-Rashomon-set-{}.json'.format(time_str), self.FIS_in_Rashomon_set)
        print('Explanation is saved to {}'.format(OUTPUT_DIR+'/FIS-in-Rashomon-set-{}.json').format(time_str))


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






