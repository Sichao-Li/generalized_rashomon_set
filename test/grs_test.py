import unittest
import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.generalized_rashomon_set import utils
from src.generalized_rashomon_set import plots
from src.generalized_rashomon_set import explainers
from src.generalized_rashomon_set import config
import shutil
import logging
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

class grs_TestCase(unittest.TestCase):
    def test_log_loss(self):
        df = pd.read_csv("breast_cancer_binary.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        model_clf = MLPClassifier(random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X.to_numpy(), y.to_numpy(), epsilon_rate=0.05,
                                                           loss_fn='log_loss', n_ways=2, delta=0.5, wrapper_for_torch=False)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        self.assertIsNotNone(fis_explainer_test_case.ref_analysis)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=False)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    def test_interaction_calculation_in_Rset(self):
        df = pd.read_csv("breast_cancer_binary.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        model_clf = MLPClassifier(random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X.to_numpy(), y.to_numpy(), epsilon_rate=0.05,
                                                           loss_fn='log_loss', n_ways=2, delta=0.1, wrapper_for_torch=False)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=True)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    def test_auc(self):
        df = pd.read_csv("compas.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        model_clf = MLPClassifier(random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X.to_numpy(), y.to_numpy(), epsilon_rate=0.05,
                                                           loss_fn='accuracy_score', n_ways=2, wrapper_for_torch=False, binary=True)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        self.assertIsNotNone(fis_explainer_test_case.ref_analysis)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=True)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)


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
                                                           X_added_constant, y, epsilon_rate=0.1,
                                                           loss_fn='log_loss', n_ways=2, wrapper_for_torch=False, binary=False)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        self.assertIsNotNone(fis_explainer_test_case.ref_analysis)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=True)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)
        plots.halo_plot(fis_explainer_test_case, 16, save=True, suffix='halo_test')
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    def test_clean(self):
        logging.shutdown()
        shutil.rmtree(config.LOG_DIR)
        shutil.rmtree(config.OUTPUT_DIR)

if __name__ == '__main__':
    unittest.main()
