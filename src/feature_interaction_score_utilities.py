from general_utilities import *

def Interaction_effect_calculation(feature_idx, model, m_all, X, y, regression=True):
    '''
    Calculate the feature interaction effect following
    c(x1, x2) = c(x1) + c(x2) + fi(x1, x2) wrt. c(x1) + c(x2) = boundary
    fi(x1, x2) is defined as the difference between boundary and emperical loss.
    The value of fi(x1, f2) can be negative or positive.

        Input:
            feature_idx: features to be calculated
            model: optimal model
            e_idx: pairs that sum to one index
            m_all: variance tolerance for all features
            X: input data
            loss_ref: optimal loss
            boundary: pre-defined boundary for R set
        Output:
            loss_emp: emperical loss set
            diff: feature interaction value set
    '''
    m_interest = np.array(m_all)[(feature_idx), :, :]
    loss_emp = []
    joint_effect_all = []
    n_ways = len(feature_idx)
    all_sum_to_one_pairs = find_all_sum_to_one_pairs(n_ways)
    for sum_to_one_pair in all_sum_to_one_pairs:
        possibilities = []
        for idx, i in enumerate(sum_to_one_pair):
            possibilities.append(m_interest[idx, i, :])
    # for idx, i in enumerate(e_idx):
    #     possibilities.append(m_interest[idx, i, :])
        for comb in itertools.product(*possibilities):
            X0 = X.copy()
            X0[:, feature_idx] = X0[:, feature_idx] * comb
            loss_after, loss_before = feature_effect(feature_idx, X0, y, model, shuffle_times=30, regression=regression)
            joint_effect_all.append(loss_after-loss_before)
            loss_emp.append(loss_before)
    return joint_effect_all, loss_emp



# @jit(target_backend='cuda')
def Interaction_effect_all_pairs(X, y, vlist, n_ways, model, m_all, regression=True):
    '''
    Calculate the feature interaction effect for all pairs.
        Input:
            X: input data
            vlist: variable list
            n_ways: an integer that defines the number of joint features
            model: optimal model
            m_all: variance tolerance for all features
            loss_ref: optimal loss
            boundary: pre-defined boundary for R set
        Output:
            myloss_all: emperical loss set for all features
            diff_all: feature interaction value set for all features
    '''
    all_n_way_feature_subsets = find_all_n_way_feature_pairs(vlist, n_ways)
    joint_effect_all_pair = []
    loss_emp_all_pair = []
    for subset in all_n_way_feature_subsets:
        # for each pair, find interaction effect and loss
        if subset[0] != 0:
            subset = np.nonzero(np.in1d(vlist, subset))[0]
        # for sum_to_one_pair in all_sum_to_one_pairs:
            # each subset has 2^n possibilities
        joint_effect_single_pair, loss_emp_single_pair = Interaction_effect_calculation(subset, model, m_all, X, y, regression=regression)
        joint_effect_all_pair.append(joint_effect_single_pair)
        loss_emp_all_pair.append(loss_emp_single_pair)
    return joint_effect_all_pair, loss_emp_all_pair


def pairwise_vis_loss(loss_emp_all, boundary):
    loss_main_all = np.arange(0, 1+0.1, 0.1) * boundary
    cord_loss = []
    circle = []
    sum_to_one_pairs = find_all_sum_to_one_pairs(2)
    quadrants = {0: [0,0],1: [0,1],2: [1,0],3: [1,1] }
    for idx,pair in enumerate(sum_to_one_pairs):
        deg = (np.arctan2(pair[0], pair[1]))
        for quadrant in quadrants:
            main_loss_sum = loss_main_all[pair[0]] + loss_main_all[pair[-1]]
            interaction_effect = (loss_emp_all[idx * 4 + quadrant])
            x = interaction_effect*np.cos(deg).tolist()
            y = interaction_effect*np.sin(deg).tolist()
            cord_loss.append([x, y])
            circle.append([main_loss_sum*np.cos(deg), main_loss_sum*np.sin(deg)])
            deg = deg+np.pi*.5
    return cord_loss, circle


def list_flatten(list, reverse):
    list_reshaped = []
    for sublist in list:
        flat_list = []
        for subsublist in sublist:
            for item in subsublist:
                flat_list.append(item)
        list_reshaped.append(sorted(flat_list, reverse=reverse))
    return list_reshaped

def save_m_all_set(points_all_positive, points_all_negative, epsilon):
    points_all_positive_reshaped = list_flatten(points_all_positive, reverse=False)
    points_all_negative_reshaped = list_flatten(points_all_negative, reverse=True)
    p = len(np.arange(0.2, 1+0.2, 0.2))
    d = len(np.arange(0.0, 1+0.1, 0.1))
    n_features = len(points_all_positive)
    m_all_sub_set = np.ones([p, d, n_features, 2], dtype=np.float64)
    for idxj,sub_boundary_rate in enumerate(np.arange(0.2, 1+0.2, 0.2)):
        for idxk, j in enumerate(np.arange(0.0, 1+0.1, 0.1)):
            for idxi, feature in enumerate(points_all_positive_reshaped):
                for idv in (feature):
                    if idv[-1] <= j*sub_boundary_rate*epsilon:
                        m_all_sub_set[idxj,idxk,idxi,0] = idv[0]
                    else:
                        break
            for idxi, feature in enumerate(points_all_negative_reshaped):
                for idv in (feature):
                    if idv[-1] <= j*sub_boundary_rate*epsilon:
                        m_all_sub_set[idxj,idxk,idxi,1] = idv[0]
                    else:
                        break
    #   np.transpose(1,0,2)
    return m_all_sub_set

def Integral_Approximation(cord_sorted):
    cord_sorted_x = np.array(cord_sorted)[:, 0]
    cord_sorted_x_delta = cord_sorted_x[1:] - cord_sorted_x[:-1]
    cord_sorted_y = abs(np.array(cord_sorted)[1:, 1])
    area = sum(cord_sorted_x_delta * cord_sorted_y) + abs(cord_sorted[0][0] * cord_sorted[0][1])
    return area

def Integral_Approximation_Double(cord_sorted):
    cord_sorted_x = cord_sorted.copy()
    cord_sorted_y = cord_sorted.copy()
    cord_sorted_x.sort(key=lambda c: (c[0]))
    cord_sorted_x = np.array(cord_sorted_x)[:, 0]
    cord_sorted_x_delta = cord_sorted_x[1:] - cord_sorted_x[:-1]
    cord_sorted_y.sort(key=lambda c: (c[1]))
    cord_sorted_y = np.array(cord_sorted_y)[:, 1]
    cord_sorted_y_delta = cord_sorted_y[1:] - cord_sorted_y[:-1]
    volumn = 0
    cord_sorted_z = abs(np.array(cord_sorted)[1:, 2])
    for i in cord_sorted_y_delta:
        volumn_sub = sum(cord_sorted_x_delta * i * cord_sorted_z)
        volumn = volumn + volumn_sub
    return volumn


# def feature_interaction_strength(emp_diff, myloss_all, n_ways, boundary, pair_idx, loss0):
#     '''
#     amount is approximated by the area of a circle - delta*(emp_diff+exp_diff)
#
#         Input:
#             emp_diff: calculated difference
#             n_ways: number of ways
#             boundary: pre-defined boundary
#         Output:
#             the feature interaction strength
#
#     '''
#     top = (myloss_all - loss0 - boundary) * (myloss_all - loss0 - boundary)
#     bot = np.multiply(myloss_all - loss0, myloss_all - loss0)
#     strength = np.sum(top / bot, axis=1)
#
#     sum_to_one_pairs = find_all_sum_to_one_pairs(n_ways)
#     #     exp_diff = np.array(sum_to_one_pairs)[:,0]*0.1
#     cord = []
#     circle = []
#     cord_sorted_1 = []
#     cord_sorted_2 = []
#     cord_sorted_3 = []
#     cord_sorted_4 = []
#     if n_ways > 2:
#         cord_sorted_5 = []
#         cord_sorted_6 = []
#         cord_sorted_7 = []
#         cord_sorted_8 = []
#         for idx, pair in enumerate(sum_to_one_pairs):
#             cosa = pair[0] / (np.sqrt(pair[0] ** 2 + pair[1] ** 2 + pair[2] ** 2))
#             cosb = pair[1] / (np.sqrt(pair[0] ** 2 + pair[1] ** 2 + pair[2] ** 2))
#             cosc = pair[2] / (np.sqrt(pair[0] ** 2 + pair[1] ** 2 + pair[2] ** 2))
#             dega = np.arccos(cosa)
#             degb = np.arccos(cosb)
#             degc = np.arccos(cosc)
#             #           iterate in 111, 110, 101, 100, 011, 010, 001, 000
#             l = [1, -1]
#             idx2 = 0
#             for d in list(itertools.product(l, repeat=3)):
#                 dis = boundary + emp_diff[pair_idx][idx * 8 + idx2]
#                 cord.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1], dis * np.cos(degc) * d[2]])
#                 circle.append(
#                     [boundary * np.cos(dega) * d[0], boundary * np.cos(degb) * d[1], boundary * np.cos(degc) * d[2]])
#                 cord_sorted_1.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1],
#                                       dis * np.cos(degc) * d[2]]) if idx2 == 0 else None
#                 cord_sorted_2.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1],
#                                       dis * np.cos(degc) * d[2]]) if idx2 == 1 else None
#                 cord_sorted_3.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1],
#                                       dis * np.cos(degc) * d[2]]) if idx2 == 2 else None
#                 cord_sorted_4.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1],
#                                       dis * np.cos(degc) * d[2]]) if idx2 == 3 else None
#                 cord_sorted_5.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1],
#                                       dis * np.cos(degc) * d[2]]) if idx2 == 4 else None
#                 cord_sorted_6.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1],
#                                       dis * np.cos(degc) * d[2]]) if idx2 == 5 else None
#                 cord_sorted_7.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1],
#                                       dis * np.cos(degc) * d[2]]) if idx2 == 6 else None
#                 cord_sorted_8.append([dis * np.cos(dega) * d[0], dis * np.cos(degb) * d[1],
#                                       dis * np.cos(degc) * d[2]]) if idx2 == 7 else None
#                 idx2 = idx2 + 1
#         #         cord_sorted_1.sort(key=lambda c:(c[0], c[1]))
#         #         cord_sorted_2.sort(key=lambda c:(c[0], c[1]))
#         #         cord_sorted_3.sort(key=lambda c:(c[0], c[1]))
#         #         cord_sorted_4.sort(key=lambda c:(c[0], c[1]))
#         #         cord_sorted_5.sort(key=lambda c:(c[0], c[1]))
#         #         cord_sorted_6.sort(key=lambda c:(c[0], c[1]))
#         #         cord_sorted_7.sort(key=lambda c:(c[0], c[1]))
#         #         cord_sorted_8.sort(key=lambda c:(c[0], c[1]))
#         volumn_total = Integral_Approximation_Double(cord_sorted_1) + Integral_Approximation_Double(
#             cord_sorted_2) + Integral_Approximation_Double(cord_sorted_3) + Integral_Approximation_Double(
#             cord_sorted_4) + Integral_Approximation_Double(cord_sorted_5) + Integral_Approximation_Double(
#             cord_sorted_6) + Integral_Approximation_Double(cord_sorted_7) + Integral_Approximation_Double(cord_sorted_8)
#         volumn_ball = (4 / 3) * np.pi * boundary * boundary * boundary
#         volumn_ratio = (volumn_total / volumn_ball)
#         return strength, cord, circle, volumn_ratio
#     else:
#         for idx, pair in enumerate(sum_to_one_pairs):
#             deg = (np.arctan2(pair[0], pair[1]))
#             for i in range(4):
#                 dis = boundary + emp_diff[pair_idx][idx * 4 + i]
#                 x = dis * np.cos(deg)
#                 y = dis * np.sin(deg)
#                 deg = deg + np.pi * .5
#                 cord.append([x, y])
#                 circle.append([boundary * np.cos(deg), boundary * np.sin(deg)])
#                 cord_sorted_1.append([x, y]) if i == 0 else None
#                 cord_sorted_2.append([x, y]) if i == 1 else None
#                 cord_sorted_3.append([x, y]) if i == 2 else None
#                 cord_sorted_4.append([x, y]) if i == 3 else None
#
#         cord_sorted_1.sort(key=lambda c: (c[0]))
#         cord_sorted_2.sort(key=lambda c: (c[0]))
#         cord_sorted_3.sort(key=lambda c: (c[0]))
#         cord_sorted_4.sort(key=lambda c: (c[0]))
#         # Integral Approximation
#         area_total = Integral_Approximation(cord_sorted_1) + Integral_Approximation(
#             cord_sorted_2) + Integral_Approximation(cord_sorted_3) + Integral_Approximation(cord_sorted_4)
#         area_circle = np.pi * boundary * boundary
#         area_ratio = area_total / area_circle
#         return strength, cord, circle, area_ratio




def high_order_vis_loss(loss_emp_all, boundary, n_ways, loss_ref):
    loss_main_all = np.arange(0, 1+0.1, 0.1) * boundary
    cord_loss = []
    circle = []
    sum_to_one_pairs = find_all_sum_to_one_pairs(n_ways)
    quadrants = {}
    for idx, i in enumerate(itertools.product([1, -1], repeat=n_ways)):
        quadrants[idx] = i
    for idx,pair in enumerate(sum_to_one_pairs):
        cosa = pair[0] / (np.sqrt(pair[0] ** 2 + pair[1] ** 2 + pair[2] ** 2))
        cosb = pair[1] / (np.sqrt(pair[0] ** 2 + pair[1] ** 2 + pair[2] ** 2))
        cosc = pair[2] / (np.sqrt(pair[0] ** 2 + pair[1] ** 2 + pair[2] ** 2))
        dega = np.arccos(cosa)
        degb = np.arccos(cosb)
        degc = np.arccos(cosc)
        for quadrant in quadrants:
            main_loss_sum = loss_main_all[pair[0]] + loss_main_all[pair[1]] + loss_main_all[pair[2]]
            circle.append(
                [main_loss_sum * np.cos(dega) * quadrants[quadrant][0], main_loss_sum * np.cos(degb)* quadrants[quadrant][1] , main_loss_sum * np.cos(degc)* quadrants[quadrant][2]])
            interaction_effect = loss_emp_all[idx * pow(2, n_ways) + quadrant] - loss_ref
            cord_loss.append(
                [interaction_effect * np.cos(dega)* quadrants[quadrant][0], interaction_effect * np.cos(degb)* quadrants[quadrant][1], interaction_effect * np.cos(degc)* quadrants[quadrant][2]])
    return circle, cord_loss