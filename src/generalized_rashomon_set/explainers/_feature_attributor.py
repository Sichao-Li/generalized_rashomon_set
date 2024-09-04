from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, roc_auc_score, accuracy_score
from PIL import Image
import numpy as np
import torch
import pandas as pd
import itertools
import random
from ..utils import find_all_sum_to_one_pairs, duplicate, set_seed

class feature_attributor:
    def __init__(self, model, loss_fn, binary, delta=0.1, seed=42):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_functions = {
            'mean_squared_error': mean_squared_error,
            'mean_absolute_error': mean_absolute_error,
            'r2_score': r2_score,
            'log_loss': log_loss,
            'log_loss_avg': log_loss,
            'log_loss_sum': lambda y_true, y_pred: log_loss(y_true, y_pred, normalize=False),
            'roc_auc_score': roc_auc_score,
            'accuracy_score': accuracy_score
        }
        set_seed(seed)
        self.binary = binary
        self.delta = delta


    def convert_to_numpy(self, tensor):
        """Converts a PyTorch tensor to a NumPy array."""
        if torch.is_tensor(tensor):
            return tensor.detach().numpy()
        return tensor

    def convert_input(self, inputs):
        """Converts inputs to the appropriate format based on their type."""
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).float()
        elif isinstance(inputs, pd.DataFrame):
            return torch.tensor(inputs.values).float()
        return inputs

    def loss_func(self, y_true, y_pred):
        """Calculates the loss using the specified loss function."""
        y_pred = (y_pred > 0.5) if self.binary else y_pred
        if self.loss_fn in self.loss_functions:
            return self.loss_functions[self.loss_fn](y_true, y_pred)
        else:
            raise ValueError(f'Unknown loss function: {self.loss_fn}')

    # def loss_shuffle(self, X0, v_idx, y, times=30):
    #     """Calculates the loss after shuffling the inputs."""
    #     loss_all = []
    #     if np.array(v_idx).ndim == 0:
    #         v_idx = [v_idx]
    #     for i in range(times):
    #         for idx in v_idx:
    #             if not hasattr(X0, 'shape'):
    #                 X0 = duplicate(np.asarray(X0))
    #                 arr_temp = X0[idx[:, 0], idx[:, 1], :]
    #                 np.random.shuffle(arr_temp)
    #                 X0[idx[:, 0], idx[:, 1], :] = arr_temp
    #                 X0 = Image.fromarray(X0)
    #             else:
    #                 np.random.shuffle(X0[:, idx])
    #         pred = self.convert_to_numpy(self.model.predict(self.convert_input(X0)))
    #         loss_shuffle = self.loss_func(y_true=y, y_pred=pred)
    #         loss_all.append(loss_shuffle)
    #     return np.mean(loss_all)

    def loss_shuffle(self, X0, v_idx, y, times=30):
        """Calculates the loss after shuffling the inputs."""
        loss_all = []
        if np.array(v_idx).ndim == 0:
            v_idx = [v_idx]
        for _ in range(times):
            for idx in v_idx:
                X0_shuffled = X0.clone()
                if X0.dim() == 3:  # For image-like data
                    arr_temp = X0_shuffled[idx[:, 0], idx[:, 1], :].clone()
                    arr_temp = arr_temp[torch.randperm(arr_temp.size(0))]
                    X0_shuffled[idx[:, 0], idx[:, 1], :] = arr_temp
                else:  # For other types of data
                    idx_perm = torch.randperm(X0_shuffled.size(0))
                    X0_shuffled[:, idx] = X0_shuffled[idx_perm, idx]
            pred = self.convert_to_numpy(self.model.predict(self.convert_input(X0_shuffled)))
            loss_shuffle = self.loss_func(y_true=y, y_pred=pred)
            loss_all.append(loss_shuffle)
        return np.mean(loss_all)

    def feature_effect(self, v_idx, X0, y, shuffle_times=30):
        """Calculates the effect of the features on the loss."""
        pred = self.convert_to_numpy(self.model.predict(self.convert_input(X0)))
        loss_before = self.loss_func(y, pred)
        loss_after = self.loss_shuffle(X0, v_idx, y, shuffle_times)
        return loss_after, loss_before

    def feature_effect_context(self, vidx, X0, y, shuffle_times=30, context=1):
        """Calculates the effect of the features on the loss in a specific context."""
        X1 = duplicate(X0)
        if isinstance(vidx, int):
            vidx = [vidx]
        for i in range(len(X0[-1])):
            if i not in vidx:
                X1[:, i] = context
        pred = self.convert_to_numpy(self.model.predict(self.convert_input(X1)))
        loss_before = self.loss_func(y, pred)
        loss_after = self.loss_shuffle(X1, vidx, y, shuffle_times)
        return loss_after, loss_before

    def feature_interaction_effect(self, feature_idx, m_all, X, y, subset_idx=None):
        """
        Calculate the feature interaction effect following
        c(x1, x2) = c(x1) + c(x2) + fi(x1, x2) wrt. c(x1) + c(x2) = boundary
        fi(x1, x2) is defined as the difference between the combined effect of x1 and x2 and the sum of their individual effects.
        """
        m_interest = np.array(m_all)[(subset_idx), :, :] # [n_subset_features, n_sums, 2]
        loss_emp = []
        joint_effect_all = []
        n_order = len(feature_idx)
        all_sum_to_one_pairs = find_all_sum_to_one_pairs(n_order, delta=self.delta)
        for sum_to_one_pair in all_sum_to_one_pairs:
            m_high_order = []
            for idx, i in enumerate(sum_to_one_pair):
                m_high_order.append(m_interest[idx, i, :]) # e.g., in breast cancer, m_interest is in shape [2, 3, 2] if delta=0.5
            for comb in itertools.product(*m_high_order):
                X0 = duplicate(X)
                X0[:, feature_idx] = X0[:, feature_idx] * comb
                loss_after, loss_before = self.feature_effect(feature_idx, X0, y, shuffle_times=100)
                joint_effect_all.append(loss_after-loss_before)
                loss_emp.append(loss_before)
        return joint_effect_all, loss_emp

    def MR(self, idx, X, y, model):
        loss_before = self.loss_func(y, model.predict(X))
        p = sum(X[:, idx] == 1) / len(X)
        X[:, idx] = 1
        loss = self.loss_func(y, model.predict(X)) * p
        X[:, idx] = -1
        loss_after = loss + self.loss_func(y, model.predict(X)) * (1 - p)
        return loss_after / loss_before
