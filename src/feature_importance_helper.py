from general_utilities import *
logger = logging.getLogger(__name__)

def calculate_boundary(y_true, y_pred, epsilon_rate=0.05, regression=True):
    '''
    calculate boundary according to epsilon_rate
    '''
    if regression:
        loss_ref = loss_regression(y_true, y_pred)
    else:
        loss_ref = loss_classification(y_true, y_pred)
    epsilon = loss_ref * epsilon_rate
    return loss_ref, epsilon

# def explore_m_in_R(bound, loss_ref, vlist, model, X, y, delta=0.01, regression=True):
#
#     '''
#     Explore the Rashomon set for the black box model by searching m within a boundary.
#         Input:
#             bound: boundary of R set, defined by epsilon
#             loss_ref: loss of reference model
#             vlist: variable list of length p
#             model: optimal model
#             X,y: data set
#             delta: the parameter splitting from 0 to 1, d=1/delta
#         Output:
#             m: possible masks for all features in R, pxdx2
#             points_all_positive, points_all_negative: recorded training process
#             fis_main: main effects of all features
#     '''
#
#     p = len(vlist)
#     d = len(np.arange(0, 1+0.1, delta))
#     m = np.zeros([p, d, 2])
#     fis_main = np.zeros([p, d, 2])
#     points_all_max = []
#     points_all_min = []
#     for idx, vname in enumerate(vlist):
#         m_plus, points_max, fis_all_plus = greedy_search(idx, bound, loss_ref, model, X, y, direction=True, delta=delta, regression=regression)
#         points_all_max.append(points_max)
#         m_minus, points_min, fis_all_minus = greedy_search(idx, bound, loss_ref, model, X, y, direction=False, delta=delta, regression=regression)
#         points_all_min.append(points_min)
#         m[idx,:,0] = m_plus
#         m[idx,:,1] = m_minus
#         fis_main[idx, :, 0] = fis_all_plus
#         fis_main[idx, :, 1] = fis_all_minus
#     return m, points_all_max, points_all_min, fis_main

def greedy_search(vidx, bound, loss_ref, model, X, y, delta=0.1, direction=True, regression=True, softmax=False):
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
    '''
    m_all = []
    points_all = []
    fis_all = []
    loss_temp = 0
#     count the tolerance
    loss_count = 0
    fis_main = 1
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
            if regression:
                pred = model.predict(X0)
                loss_m=loss_regression(y, pred)
            else:
                # pred = model.predict_proba(X0)
                pred = model.predict(X0)
                loss_m=loss_classification(y, pred)
#             the diffrence of changed loss and optimal loss
            mydiff = loss_m - loss_ref

            if mydiff<i*bound:
                if direction:
                #     if the loss within the bound, then m increses
                    m = m+lr
                if not direction:
                    m = m-lr
                loss_after, loss_before = feature_effect(vidx, X0, y, model, 30, regression=regression)
                fis_main = loss_after - loss_before
                points.append([m, mydiff])
    #             if the loss within the bound but stays same for loss_count times, then the vt is unimportant (the attribution of the feature is assigned 0, as the power of the single feature is not enough to change loss).
                if loss_temp == loss_m:
                    loss_count = loss_count+1
                    if loss_count > 100:
                        fis_main = 0
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
        fis_all.append(fis_main)
    return m_all, points_all, fis_all

def feature_importance_in_R_set(vt_fi):
    return np.sum(np.std(vt_fi, axis=1), axis=1)