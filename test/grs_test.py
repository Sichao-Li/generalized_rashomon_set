import unittest
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from src.generalized_rashomon_set import utils
from src.generalized_rashomon_set import plots
from src.generalized_rashomon_set import explainers
from src.generalized_rashomon_set import config
import shutil
import logging
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import torch
import torch.nn.functional as F
from torch import nn

class grs_TestCase(unittest.TestCase):
    def test_log_loss(self):
        df = pd.read_csv("breast_cancer_binary.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        model_clf = MLPClassifier(random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X.to_numpy(), y.to_numpy(), epsilon_rate=0.001,
                                                           loss_fn='log_loss', n_order=2, delta=0.5, torch_input=False)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        self.assertIsNotNone(fis_explainer_test_case.ref_analysis)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=False)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    # interaction calculation in Rashomon set does not support delta != 0.1 so far
    def test_interaction_calculation_in_Rset(self):
        df = pd.read_csv("breast_cancer_binary.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        model_clf = MLPClassifier(random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X.to_numpy(), y.to_numpy(), epsilon_rate=0.001,
                                                           loss_fn='log_loss', n_order=2, delta=0.1, torch_input=False)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=True)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    # different loss functions in Rashomon set comparison will be studied in the future
    def test_auc(self):
        df = pd.read_csv("compas.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        model_clf = MLPClassifier(random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X.to_numpy(), y.to_numpy(), epsilon_rate=0.05,
                                                           loss_fn='accuracy_score', n_order=2, torch_input=False, binary_output=True)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        self.assertIsNotNone(fis_explainer_test_case.ref_analysis)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=True)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    def test_input_type(self):
        class MLP(nn.Module):
            def __init__(self, nn_arch):
                super(MLP, self).__init__()
                self.nfeature, self.nclass, self.nneuron, self.nlayer = nn_arch

                self.read_in = nn.Linear(self.nfeature, self.nneuron)
                self.ff = nn.Linear(self.nneuron, self.nneuron)
                self.read_out = nn.Linear(self.nneuron, self.nclass)

            def forward(self, x):
                x = self.read_in(x)
                for i in range(self.nlayer):
                    x = F.relu(self.ff(x))
                logits = self.read_out(x)
                return logits

        df = pd.read_csv("compas.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        X, y = torch.Tensor(X.to_numpy()), torch.Tensor(y.to_numpy())
        model = MLP([X.shape[1], 2, 100, 3])
        fis_explainer_test_case = explainers.fis_explainer(model, X, y, epsilon_rate=0.01, loss_fn='log_loss', n_order=2, torch_input=True, binary_output=False)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        self.assertIsNotNone(fis_explainer_test_case.prediction)
        fis_explainer_binary_output_test_case = explainers.fis_explainer(model, X, y, epsilon_rate=0.01, loss_fn='log_loss', n_order=2, torch_input=True, binary_output=True)
        self.assertIsInstance(fis_explainer_binary_output_test_case, explainers.fis_explainer)
        self.assertIsNotNone(fis_explainer_binary_output_test_case.prediction)

    def test_halo_plot(self):
        df = pd.read_csv("compas.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        X, y = utils.pd_to_numpy(X, y)
        X_added_constant = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X_added_constant, y, random_state=2020, test_size=0.3)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        model_clf = sm.Logit(y_train, X_train).fit()
        # model_clf = MLPClassifier(hidden_layer_sizes=[100,100,100], random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X_added_constant, y, epsilon_rate=0.001,
                                                           loss_fn='log_loss', n_order=2, torch_input=False, binary_output=False)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        self.assertIsNotNone(fis_explainer_test_case.ref_analysis)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=True)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)
        plots.halo_plot(fis_explainer_test_case, 16, save=True, suffix='halo_test')
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    def test_load_results(self):
        df = pd.read_csv("compas.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        X_features_names = X.columns.values.tolist()
        X_features_names.insert(0, 'const')
        X, y = utils.pd_to_numpy(X, y)
        X_added_constant = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X_added_constant, y, random_state=2020, test_size=0.3)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        model_clf = sm.Logit(y_train, X_train).fit()
        # model_clf = MLPClassifier(hidden_layer_sizes=[100,100,100], random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X_added_constant, y, epsilon_rate=0.001,
                                                           loss_fn='log_loss', n_order=2, torch_input=False, binary_output=False)
        explainers.fis_explainer.load_results(fis_explainer_test_case,
                                                  results_path='../results')
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)
        print(fis_explainer_test_case.v_list, X_features_names)
        plots.swarm_plot_MR(fis_explainer_test_case, fis_explainer_test_case.v_list, vname=X_features_names, plot_all=True, threshold=None, boxplot=False, save=True, suffix='swarm_plot_MR_test')
        fis_explainer_test_case.fis_in_r = fis_explainer_test_case._get_fis_in_r(fis_explainer_test_case.all_pairs,
                                                                                 np.array(fis_explainer_test_case.rset_joint_effect_raw['joint_effect_all_pair_set']),
                                                                                 np.array(fis_explainer_test_case.rset_main_effect_processed['all_main_effects_diff']),
                                                                                 fis_explainer_test_case.n_order, fis_explainer_test_case.quadrants)
        fis_explainer_test_case.loss_in_r = fis_explainer_test_case._get_loss_in_r(fis_explainer_test_case.all_pairs,
                                                                                   np.array(fis_explainer_test_case.rset_joint_effect_raw['loss_emp_all_pair_set']), fis_explainer_test_case.n_order,
                                                                                   fis_explainer_test_case.quadrants, fis_explainer_test_case.epsilon, fis_explainer_test_case.loss)
        plots.swarm_plot_FIS(fis_explainer_test_case, [(0, 1), (0, 2), (0, 4), (2, 3)], vname=X_features_names, plot_all=False, threshold=None, boxplot=False, save=True, suffix='swarm_plot_FIS_test')

    def test_clean(self):
        logging.shutdown()
        shutil.rmtree(config.LOG_DIR)
        shutil.rmtree(config.OUTPUT_DIR)

if __name__ == '__main__':
    unittest.main()
