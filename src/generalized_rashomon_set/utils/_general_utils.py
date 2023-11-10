import colorsys
import itertools
import json
from itertools import combinations, product, combinations_with_replacement
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, log_loss, roc_auc_score, mean_absolute_error, accuracy_score
from ..config import logger


def colors_vis(c, lightness=0.5):
    default_colors = ["#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"]
    rgb_color = mcolors.hex2color(default_colors[c])
    # Convert RGB color to HSL color
    hls_color = colorsys.rgb_to_hls(*rgb_color)
    # Adjust lightness value
    hls_color_adjusted = (hls_color[0], lightness, hls_color[2])
    # Convert HSL color back to RGB color
    rgb_color_adjusted = colorsys.hls_to_rgb(*hls_color_adjusted)
    return rgb_color_adjusted


# def loss_regression(y_true, y_pred):
#     return mean_squared_error(y_true, y_pred, squared=True)


# def loss_classification(y_true, y_pred):
#     return roc_auc_score(y_true, y_pred)


def pd_to_numpy(X, y):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy().reshape(X.shape)
        y = y.to_numpy().reshape(len(y))
    return X, y


def find_all_n_way_feature_pairs(vlist, n_ways):
    '''
    Each feature has one vt_plus and one vt_minus.
    N features have 2^N possibilities.

        Input:
            vlist: feature list
            n_pairs: n way interactions
        Outpus:
            interaction_n_list: list of all n way interaction pairs
    '''
    interaction_n_list = []
    for i in combinations(vlist, n_ways):
        interaction_n_list.append(i)
    return interaction_n_list


def find_all_sum_to_one_pairs(n_features):
    '''
    Sample: two features X1 and X2, if the boundary is 1, then we have c(X1) + c(X2) = 1

        Input: number of features
        Output: set of all pairs

    '''
    value = range(10)
    target = 10
    pairs = []
    for feature_ind in combinations_with_replacement(value, n_features):
        if sum(feature_ind) == target:
            for i in itertools.permutations(feature_ind):
                pairs.append(i)
            #             if feature_ind[0] != feature_ind[1]:
    #                 pairs.append((feature_ind[1], feature_ind[0]))
    return list(set(pairs))


def MDS(vt_l, n_features_in, n_features_out=2):
    '''
    e.g. transform from n_features_inx10x2x10 to n_features_inx10x2x1
    '''
    vt_l_transformed_x = np.zeros((len(vt_l), len(vt_l[-1]), n_features_out))
    vt_l_transformed_y = np.zeros((len(vt_l), len(vt_l[-1]), n_features_out))
    d_old = vt_l - 1
    vt_l_x = d_old[:, :, 0]
    vt_l_y = d_old[:, :, 1]
    # distance is (x_+, 1) (x_-, 1) to (1, 1)
    degree_avg = np.pi / n_features_in
    for i in range(n_features_in):
        vt_l_transformed_x[i, :, 0] = vt_l_x[i, :] * np.cos(i * degree_avg)
        vt_l_transformed_x[i, :, 1] = vt_l_x[i, :] * np.sin(i * degree_avg)
        vt_l_transformed_y[i, :, 0] = vt_l_y[i, :] * np.cos(i * (degree_avg))
        vt_l_transformed_y[i, :, 1] = vt_l_y[i, :] * np.sin(i * (degree_avg))
    return vt_l_transformed_x, vt_l_transformed_y

def loss_func(loss_fn, y_true, y_pred, binary=False):
    if binary:
        y_pred = (y_pred > 0.5)
    if loss_fn == 'mean_squared_error':
        return mean_squared_error(y_true, y_pred)
    elif loss_fn == 'mean_absolute_error':
        return mean_absolute_error(y_true, y_pred)
    elif loss_fn == 'r2_score':
        return r2_score(y_true, y_pred)
    elif loss_fn == 'log_loss' or loss_fn == 'log_loss_avg':
        return log_loss(y_true, y_pred)
    elif loss_fn == 'log_loss_sum':
        return log_loss(y_true, y_pred, normalize=False)
    elif loss_fn == 'roc_auc_score':
        if np.array(y_pred).ndim > 1:
            y_pred = y_pred[:,0]
        return roc_auc_score(y_true, y_pred)
    elif loss_fn == 'accuracy_score':
        if np.array(y_pred).ndim > 1:
            y_pred = y_pred[:,0]
        return accuracy_score(y_true, y_pred)
    else:
        raise ValueError(f'Unknown loss function: {loss_fn}')

def loss_shuffle(model, X0, v_idx, y, times=30, loss_fn=None):
    #     shuffle to evaluate the feature importance
    loss_all = []
    if np.array(v_idx).ndim == 0:
        v_idx = [v_idx]
    for i in range(times):
        for idx in v_idx:
            if not hasattr(X0, 'shape'):
                X0 = np.asarray(X0).copy()
                arr_temp = X0[idx[:, 0], idx[:, 1], :]
                np.random.shuffle(arr_temp)
                X0[idx[:, 0], idx[:, 1], :] = arr_temp
                X0 = Image.fromarray(X0)
            else:
                np.random.shuffle(X0[:, idx])
        pred = model.predict(X0)
        if torch.is_tensor(pred):
            pred = pred.detach().numpy()
        loss_shuffle = loss_func(loss_fn, y, pred)
            # X0[:, idx] = -1
        # if regression:
        #     pred = model.predict(X0)
        #     if torch.is_tensor(pred):
        #         pred = pred.detach().numpy()
        #     loss_shuffle = loss_regression(y, pred)
        # else:
        #     # pred = model.predict_proba(X0)
        #     pred = model.predict(X0)
        #     if torch.is_tensor(pred):
        #         pred = pred.detach().numpy()
        #     loss_shuffle = loss_classification(y, pred)
        loss_all.append(loss_shuffle)
    return np.mean(loss_all)

# def feature_effect(v_idx, X0, y, model, shuffle_times=30, regression=True):
#     # loss before shuffle
#     if regression:
#         pred = model.predict(X0)
#         if torch.is_tensor(pred):
#             pred = pred.detach().numpy()
#         loss_before = loss_regression(y, pred)
#     else:
#         # pred = model.predict_proba(X0)
#         pred = model.predict(X0)
#         if torch.is_tensor(pred):
#             pred = pred.detach().numpy()
#         loss_before = loss_classification(y, pred)
#     # loss after shuffle
#     loss_after = loss_shuffle(model, X0, v_idx, y, shuffle_times, regression=regression)
#     return loss_after, loss_before

def feature_effect(v_idx, X0, y, model, shuffle_times=30, loss_fn=None):
    # loss before shuffle
    pred = model.predict(X0)
    if torch.is_tensor(pred):
        pred = pred.detach().numpy()
    loss_before = loss_func(loss_fn, y, pred)
    # loss after shuffle
    loss_after = loss_shuffle(model, X0, v_idx, y, shuffle_times, loss_fn=loss_fn)
    return loss_after, loss_before


def feature_effect_context(vidx, X0, y, model, shuffle_times=30, loss_fn=None, context=1):
    X1 = X0.copy()
    if isinstance(vidx, int):
        vidx = [vidx]
    for i in range(len(X0[-1])):
        if i not in vidx:
            X1[:, i] = context
        # X1[:, i] = 1
    # loss before shuffle
    pred = model.predict(X1)
    loss_before = loss_func(loss_fn, y, pred)
    # if regression:
    #     pred = model.predict(X1)
    #     loss_before = loss_regression(y, pred)
    # else:
    #     # pred = model.predict_proba(X1)
    #     pred = model.predict(X1)
    #     loss_before = loss_classification(y, pred)
    # loss after shuffle
    loss_after = loss_shuffle(model, X1, vidx, y, shuffle_times, loss_fn=loss_fn)
    return loss_after, loss_before


def get_auc(inter_scores, gts):
    gt_vec = []
    pred_vec = []
    for inter in inter_scores:
        #     print(inter[0])
        pred_vec.append(inter[1])
        if inter[0] in gts:
            gt_vec.append(1)
        else:
            gt_vec.append(0)
    fpr, tpr, thresholds = metrics.roc_curve(gt_vec, pred_vec, pos_label=1)
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold is %f' % (thresholds[ix]))
    # print(np.shape(thresholds), np.shape(fpr), np.shape(pred_vec))
    auc = metrics.auc(fpr, tpr)
    return auc, thresholds


def feature_idx_to_pair_idx(all_pairs=None, feature_idx=None, pair_idx=None):
    """
    Given feature_idx=(0,1), return 0;
    Given pair_idx=0, return (0,1)
    """
    if feature_idx is not None:
        return all_pairs.index(feature_idx)
    if pair_idx is not None:
        return all_pairs[pair_idx]


def MR(idx, X, y, model):
    loss_before = loss_func('log_loss', y, model.predict(X))
    p = sum(X[:, idx] == 1) / len(X)
    X[:, idx] = 1
    loss = loss_func('log_loss', y, model.predict(X)) * p
    X[:, idx] = -1
    loss_after = loss + loss_func('log_loss', y, model.predict(X)) * (1 - p)
    return loss_after / loss_before


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def save_json(path, results):
    with open(path, 'w') as f:
        json.dump(results, f, default=default)


def load_json(path):
    with open(path, "r") as f:
        json_data = json.load(f)
    return json_data


def greedy_search(vidx, bound, loss_ref, model, X, y, delta=0.1, direction=True, loss_fn=None, softmax=False):
    '''
    greedy search possible m for a single feature
        Input:
            vidx: variable name list of length n
            bound: loss boundary in R set
            model: optimal model
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
                if softmax:
                    X0 = X0._transform(vidx, m+lr)
                else:
                    X0[:, vidx] = X0[:, vidx] * (m + lr)
            if not direction:
                if softmax:
                    X0 = X0._transform(vidx, m-lr)
                else:
                    X0[:, vidx] = X0[:, vidx] * (m - lr)
            pred = model.predict(X0)
            loss_m = loss_func(loss_fn, y, pred)
#             the diffrence of changed loss and optimal loss
            mydiff = loss_m - loss_ref

            if mydiff<i*bound:
                if direction:
                #     if the loss within the bound, then m increses
                    m = m+lr
                if not direction:
                    m = m-lr
                loss_after, loss_before = feature_effect(vidx, X0, y, model, 30, loss_fn=loss_fn)
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
