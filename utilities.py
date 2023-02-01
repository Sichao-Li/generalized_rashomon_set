import numpy as np
from itertools import combinations, product, combinations_with_replacement
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error, r2_score, log_loss
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from matplotlib import pyplot as plt
import itertools
from sklearn.inspection import partial_dependence
from sklearn_gbmi import *
import statsmodels.api as sm
import pandas as pd

def loss_regression(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=True)

def loss_classification(y_true, y_pred):
    return log_loss(y_true, y_pred)

def find_boundary(y_true, y_pred, epsilon=0.05, regression=True):
    # the optimal loss and boundary
    if regression:
        loss0 = loss_regression(y_true, y_pred)
    else:
        loss0 = loss_classification(y_true, y_pred)

    # Given epsilon, find boundary and half boundary
    epsilon = 0.05
    bound_f = loss0*epsilon
    bound_f_half = loss0*epsilon*0.5
    
    return loss0, bound_f, bound_f_half


def explore_R(vname, bound, loss0, model, X, y, delta=0.1, direction=True, regression=True):
    '''

    Explore the Rashomon set by find the boundary for single variable in variable 
    list in two directions from range [0,1]
        Input:
            vname: variable name list of length n
            bound: loss boundary in R set
            model: optimal model
            X, y: model input and expected output
            delta: the range of spliting 0 to 1
            direction: exploring directions. When True, explore from 1 to 1+, else 1 to 1-

        Output:
            vt_all: variance tolerance for all features in a nx2 matrix
            points_all: recorded points when exploring

    '''
    vt_all = []
    points_all = []
    
    loss_temp = 0
#     count the tolerance
    loss_count = 0

    for i in np.arange(0, 1+0.1, delta):
        # include endpoint [0.1 ..., 1]
        count = 1
        # learning rate         
        lr = 0.1
        points = []
#         variance tolerance for single feature
        vt = 1
#     termination condition: the precision of vt .0001
        while count <= 4:
    #         input new input X0 and calculate the loss
            X0 = X.copy()
            if direction:
                if isinstance(X, pd.DataFrame):
                    X0[vname] = X0[vname]*(vt+lr)
                else:
                    X0[:,vname] = X0[:,vname]*(vt+lr)
                
            if not direction:
                if isinstance(X, pd.DataFrame):
                    X0[vname] = X0[vname]*(vt-lr)
                else:
                    X0[:,vname] = X0[:,vname]*(vt-lr)                

    #         myloss_user = totalloss(beta_lr, X0, y)
            if hasattr(model, 'model'):
                pred = model.predict(sm.add_constant(X0))
#                 if isinstance(model, sm.Logit):
                
            else:
                pred = model.predict(X0)
            if regression:
                myloss=loss_regression(y, pred)
            else:
                myloss=loss_classification(y, pred)
#             the diffrence of changed loss and optimal loss
            mydiff = myloss-loss0
#             print(mydiff, vt)
            if mydiff<i*bound:
                if direction:
                #     if the loss within the bound, then vt increses
                    vt=vt+lr
                if not direction:
                    vt = vt-lr
                points.append([vt, mydiff])
    #             if the loss within the bound but stays same for loss_count times, then the vt is unimportant.
                if loss_temp == myloss:
                    loss_count = loss_count+1
                    if loss_count > 10000:
                        break
                else:
                    loss_temp = myloss
    #                 otherwise change lr and try again
            else:
                lr=lr*0.1
                count = count+1
        points_all.append(points)
        vt_all.append(vt)
    return vt_all, points_all


def find_VT(bound, loss0, vlist, model, X, y, delta=0.1, regression=True):
    '''
    find R set of variable list for the black box model in data set (X, y) of a boundary
    
        Input:
            bound: boundary of R set
            vlist: variable list of length n
            model: optimal model
            X,y: data set
            delta: the range of spliting 0 to 1, d=1/delta
        Output:
            VT: variance tolerance for all features in range, nxdx2
            points_all_max, points_all_min: recorded training process
    
    '''
    n = len(vlist)
    d = len(np.arange(0, 1+0.1, delta))
    VT = np.zeros([n, d, 2])
    points_all_max = []
    points_all_min = []

    for idx, vname in enumerate(vlist):
        vt_plus, points_max = explore_R(vname, bound, loss0, model, X, y, direction=True, delta=delta, regression=regression)
        points_all_max.append(points_max)
        vt_minus, points_min = explore_R(vname, bound, loss0, model, X, y, direction=False, delta=delta, regression=regression)
        points_all_min.append(points_min)
        VT[idx,:,0] = vt_plus
        VT[idx,:,1] = vt_minus
    return VT, points_all_max, points_all_min

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
    for i in combinations(vlist,n_ways):
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
    for feature_ind in combinations_with_replacement(value,n_features):
        if 0 not in feature_ind and sum(feature_ind)==target:
            for i in itertools.permutations(feature_ind):
                pairs.append(i)            
#             if feature_ind[0] != feature_ind[1]:
#                 pairs.append((feature_ind[1], feature_ind[0]))
    return list(set(pairs))


def Interaction_effect_calculation(feature_idx, model, e_idx, vt_f, X, y, loss0, boundary, regression=True):
    '''
    Calculate the feature interaction effect following 
    c(x1, x2) = c(x1) + c(x2) + fi(x1, x2) wrt. c(x1) + c(x2) = boundary
    fi(x1, x2) is defined as the difference between boundary and emperical loss.
    The value of fi(x1, f2) can be negative or positive.
    
        Input:
            feature_idx: features to be calculated
            model: optimal model
            e_idx: pairs that sum to one index
            vt_f: variance tolerance for all features
            X: input data
            loss0: optimal loss
            boundary: pre-defined boundary for R set
        Output:
            loss_emp: emperical loss set
            diff: feature interaction value set
    '''
    feature_interest = np.array(vt_f)[(feature_idx), :, :]
    possibilities = []
    loss_emp = []
    diff = []
    
        
    for idx, i in enumerate(e_idx):
        possibilities.append(feature_interest[idx, i, :])
    for comb in itertools.product(*possibilities):
        X0 = X.copy()
        if isinstance(X, pd.DataFrame):
            X0.iloc[:, feature_idx] = X0.iloc[:, feature_idx]*comb
        else:
            X0[:,feature_idx] = X0[:,feature_idx]*comb
        
        
        if hasattr(model, 'model'):
            pred = model.predict(sm.add_constant(X0))
#                 if isinstance(model, sm.Logit):
                
        else:
            pred = model.predict(X0)
        

        
#         if isinstance(model.model, sm.Logit):
#             pred = model.predict(sm.add_constant(X0))
#         else:
#             pred = model.predict(X0)
        
        if regression:
            myloss = mean_squared_error(y, pred, squared=True)
        else:
            myloss = log_loss(y, pred)
            
        mydiff = myloss-loss0-boundary
        diff.append((mydiff))
        loss_emp.append((myloss))
    return diff, loss_emp


def Interaction_effect_all_pairs(X, y, vlist, n_ways, model, vt_f, loss0, boundary, regression=True):
    '''
    Calculate the feature interaction effect for all pairs.
        Input:
            X: input data
            vlist: variable list
            n_ways: an integer that defines the number of joint features
            model: optimal model
            vt_f: variance tolerance for all features
            loss0: optimal loss
            boundary: pre-defined boundary for R set
        Output:
            myloss_all: emperical loss set for all features
            diff_all: feature interaction value set for all features
    '''
    all_n_way_feature_pairs = find_all_n_way_feature_pairs(vlist, n_ways)
    all_sum_to_one_pairs = find_all_sum_to_one_pairs(n_ways)
    diff_all = []
    myloss_all = []
    for pair in all_n_way_feature_pairs:
    # for each pair, find interaction effect and loss
        fi = []
        loss = []
        if pair[0] != 0:
            pair = np.nonzero(np.in1d(vlist, pair))[0]
        for sum_to_one_pair in all_sum_to_one_pairs:
        # each pair has n^2 possibilities
            diff, myloss = Interaction_effect_calculation(pair, model, sum_to_one_pair, vt_f, X, y, loss0, boundary, regression=regression)
            fi = fi + diff
            loss = loss + myloss
        myloss_all.append(loss)
        diff_all.append(fi)
    return myloss_all, diff_all


def Interaction_effect_single_pair(X, y, vlist_interest, model, vt_f, loss0, boundary):
    '''
    Calculate the feature interaction effect for a single pair.
        Input:
            X: input data
            vlist: variable list
            n_ways: an integer that defines the number of joint features
            model: optimal model
            vt_f: variance tolerance for all features
            loss0: optimal loss
            boundary: pre-defined boundary for R set
        Output:
            myloss: emperical loss set for a feature pair
            fi: feature interaction value set for a feature pair
    '''
    
    n_ways = len(vlist_interest)
    all_sum_to_one_pairs = find_all_sum_to_one_pairs(n_ways)
    fi = []
    loss = []
    for sum_to_one_pair in all_sum_to_one_pairs:
    # each pair has n^2 possibilities
        diff, myloss = Interaction_effect_calculation(pair, model, sum_to_one_pair, vt_f, X, y, loss0, boundary)
        fi = fi + diff
        loss = loss + myloss
    return loss, fi


def feature_interaction_strength(emp_diff, n_ways, boundary):
    '''
    amount is approximated by the area of a circle - delta*(emp_diff+exp_diff)
    
        Input: 
            emp_diff: calculated difference
            n_ways: number of ways
            boundary: pre-defined boundary
        Output:
            the feature interaction strength
        
    '''
    sum_to_one_pairs = find_all_sum_to_one_pairs(n_ways)
    exp_diff = np.array(sum_to_one_pairs)[:,0]*0.1
    if np.array(emp_diff).ndim>1:
        std = np.std(emp_diff, axis=1)
    else:
        std = np.std(emp_diff)
    amount = (np.pi*boundary*boundary - np.sum(((emp_diff/np.sqrt(2))+boundary*np.repeat(exp_diff, 4))*0.1*boundary, axis=1))
    return amount*std


def MDS(vt_l, n_features_in, n_features_out=2):
    '''
    e.g. transform from n_features_inx10x2x10 to n_features_inx10x2x1
    '''
    vt_l_transformed_x = np.zeros((len(vt_l), len(vt_l[-1]), n_features_out))
    vt_l_transformed_y = np.zeros((len(vt_l), len(vt_l[-1]), n_features_out))
    d_old = vt_l-1
    vt_l_x = d_old[:,:,0]
    vt_l_y = d_old[:,:,1]
    # distance is (x_+, 1) (x_-, 1) to (1, 1)
    degree_avg = np.pi/n_features_in
    for i in range(n_features_in):
        vt_l_transformed_x[i,:,0] = vt_l_x[i,:]*np.cos(i*degree_avg)
        vt_l_transformed_x[i,:,1] = vt_l_x[i,:]*np.sin(i*degree_avg)
        vt_l_transformed_y[i,:,0] = vt_l_y[i,:]*np.cos(i*(degree_avg))
        vt_l_transformed_y[i,:,1] = vt_l_y[i,:]*np.sin(i*(degree_avg))
    return vt_l_transformed_x, vt_l_transformed_y


def diff_points_cal(pairs, diff, bound_f):
    '''
    calculate the difference of points between the expected circle.
    
        Input:
            pairs: sum to one pairs
            diff: calculated diff list
            bound_f: epsilon
        Output:
            cord: coordinates of calculated points
            circle: points to the expected circle
    '''
    cord = []
    circle = []
    for idx,pair in enumerate(pairs):

        deg = (np.arctan2(pair[0], pair[1]))
        for i in range(4):
            dis = bound_f + diff[idx*4+i]
            x = dis*np.cos(deg)
            y = dis*np.sin(deg)
            deg = deg+np.pi*.5
            cord.append([x, y])
            circle.append([bound_f*np.cos(deg), bound_f*np.sin(deg)])
#     cord.append([0, bound_f])
#     cord.append([0, -bound_f])
#     cord.append([bound_f, 0])
#     cord.append([-bound_f, 0])
#     circle.append([0, bound_f])
#     circle.append([0, -bound_f])
#     circle.append([bound_f, 0])
#     circle.append([-bound_f, 0])
    return cord, circle


def feature_interaction_vis(vlist, n_ways):
    '''
    Visulize feature interaction effects
    '''
    n_pairs = find_all_n_way_feature_pairs(vlist, n_ways)
    sum_to_one_pairs = find_all_sum_to_one_pairs(n_ways)
    fig, axs = plt.subplots(2, 2, figsize=(15, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    print(axs)
    coord = []
    n_sub_plots = 4
    for i in range(n_sub_plots):
        cor,circle = diff_points_cal(sum_to_one_pairs, diff_all[i], bound_f)

        cor.sort(key=lambda c:np.arctan2(c[0], c[1]))
        cor.append(cor[0])

        circle.sort(key=lambda c:np.arctan2(c[0], c[1]))
        circle.append(circle[0])
        
        xx = np.concatenate((np.array(cor)[:,0], np.array(circle)[:,0][::-1]))
        xy = np.concatenate((np.array(cor)[:,1], np.array(circle)[:,1][::-1]))
        coord.append([xx,xy])
        
        axs[i].plot(np.array(cor)[:,0], np.array(cor)[:,1], marker='o', linewidth=2, markersize=2)
        axs[i].plot(np.array(circle)[:,0], np.array(circle)[:,1], linewidth=1, markersize=1)
        axs[i].fill(xx, xy, alpha=0.5)
        max_cir = np.max(diff_all[i])
        min_cir = np.min(diff_all[i])
        avg_cir = np.average(diff_all[i])
        axs[i].add_patch(plt.Circle((0, 0), avg_cir+bound_f, fill=False, linestyle='--', color='r'))
        axs[i].add_patch(plt.Circle((0, 0), min_cir+bound_f, fill=False, linestyle='--', color='g'))
        axs[i].add_patch(plt.Circle((0, 0), max_cir+bound_f, fill=False, linestyle='--', color='purple'))
        # ax.add_patch(plt.Circle((0, 0), bound_f, fill=False))
        axs[i].set_aspect("equal", adjustable="datalim")
        axs[i].set_box_aspect(0.5)
        axs[i].autoscale()
    plt.show()