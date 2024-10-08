import colorsys
import itertools
import json
from itertools import combinations, product, combinations_with_replacement
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import random
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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

def pd_to_numpy(X, y):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy().reshape(X.shape)
        y = y.to_numpy().reshape(len(y))
    return X, y

def find_all_n_order_feature_pairs(vlist, n_order):
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
    for i in combinations(vlist, n_order):
        interaction_n_list.append(i)
    return interaction_n_list


def find_all_sum_to_one_pairs(n_features, delta=0.1):
    '''
    Sample: two features X1 and X2, if the boundary is 1, then we have c(X1) + c(X2) = 1

        Input: number of features
        Output: set of all pairs

    '''
    value = range(10)
    target = 10
    pairs = []
    out = []
    for feature_ind in combinations_with_replacement(value, n_features):
        if sum(feature_ind) == target:
            for i in itertools.permutations(feature_ind):
                pairs.append(i)
    all_pairs = list(set(pairs))
    # the following block will be used to calculate interaction effect when delta is not 0.1
    delta *= 10
    for pair in all_pairs:
        if( pair[0] % delta == 0) and (pair[1] % delta == 0):
            out.append(pair)
    return out


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

def duplicate(data):
    if isinstance(data, np.ndarray):
        return np.copy(data)
    elif isinstance(data, torch.Tensor):
        return data.clone()
