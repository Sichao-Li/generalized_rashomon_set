import unittest
import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.generalized_rashomon_set import utils
from src.generalized_rashomon_set import utils
from src.generalized_rashomon_set import plots
from src.generalized_rashomon_set import explainers
from src.generalized_rashomon_set import config
import shutil
import logging
class grs_TestCase(unittest.TestCase):
    def test_log_loss(self):
        df = pd.read_csv("breast_cancer_binary.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        model_clf = MLPClassifier(random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X.to_numpy(), y.to_numpy(), epsilon_rate=0.05,
                                                           loss_fn='log_loss', n_ways=2, wrapper_for_torch=False)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        self.assertIsNotNone(fis_explainer_test_case.ref_analysis)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=False)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    def test_roc_auc(self):
        df = pd.read_csv("breast_cancer_binary.csv")
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        model_clf = MLPClassifier(random_state=1, max_iter=300).fit(X.to_numpy(), y.to_numpy())
        fis_explainer_test_case = explainers.fis_explainer(model_clf,
                                                           X.to_numpy(), y.to_numpy(), epsilon_rate=0.05,
                                                           loss_fn='roc_auc_score', n_ways=2, wrapper_for_torch=False)
        self.assertIsInstance(fis_explainer_test_case, explainers.fis_explainer)
        fis_explainer_test_case.ref_explain(model_reliance=False)
        self.assertIsNotNone(fis_explainer_test_case.ref_analysis)
        fis_explainer_test_case.rset_explain(main_effect=True, interaction_effect=False)
        self.assertIsNotNone(fis_explainer_test_case.FIS_in_Rashomon_set)

    def test_clean(self):
        logging.shutdown()
        shutil.rmtree(config.LOG_DIR)
        shutil.rmtree(config.OUTPUT_DIR)

if __name__ == '__main__':
    unittest.main()
