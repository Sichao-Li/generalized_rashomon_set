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
            print(sum_to_one_pair)
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


def get_all_m_with_t_in_range(points_all_max, points_all_min, epsilon):
    '''
    :return: an m matrix in shape [5, 11, n_features, 2] corresponding to sub-dominant boundary, e.g. 0.2 * epsilon
    '''
    points_all_positive_reshaped = list_flatten(points_all_max, reverse=False)
    points_all_negative_reshaped = list_flatten(points_all_min, reverse=True)
    p = len(np.arange(0.2, 1 + 0.2, 0.2))
    d = len(np.arange(0.0, 1 + 0.1, 0.1))
    n_features = len(points_all_min)
    m_multi_boundary_e = np.ones([p, d, n_features, 2], dtype=np.float64)
    for idxj, sub_boundary_rate in enumerate(np.arange(0.2, 1 + 0.2, 0.2)):
        for idxk, j in enumerate(np.arange(0.0, 1 + 0.1, 0.1)):
            for idxi, feature in enumerate(points_all_positive_reshaped):
                for idv in (feature):
                    if idv[-1] <= j * sub_boundary_rate * epsilon:
                        m_multi_boundary_e[idxj, idxk, idxi, 0] = idv[0]
                    else:
                        break
            for idxi, feature in enumerate(points_all_negative_reshaped):
                for idv in (feature):
                    if idv[-1] <= j * sub_boundary_rate * epsilon:
                        m_multi_boundary_e[idxj, idxk, idxi, 1] = idv[0]
                    else:
                        break
    #   np.transpose(1,0,2)
    return m_multi_boundary_e

def get_all_main_effects(m_multi_boundary_e, input, output, model, v_list, regression):
    '''
    :param m_multi_boundary_e: an m matrix in shape [5, 11, n_features, 2]
    :param model: reference model
    :return:
        fi_all_ratio: main effects of all features in ratio
        fi_all_diff: main effects of all features in difference
    '''
    fi_all_diff = np.zeros(m_multi_boundary_e.shape)
    fi_all_ratio = np.zeros(m_multi_boundary_e.shape)
    for idx, sub_boundary_rate in enumerate(np.arange(0.2, 1.2, 0.2)):
        for idxj, j in enumerate(np.arange(0, 1 + 0.1, 0.1)):
            for i in range(len(v_list)):
                for k in range(2):
                    X0 = input.copy()
                    X0[:, i] = X0[:, i] * m_multi_boundary_e[idx, idxj, i, k]
                    loss_after, loss_before = feature_effect(i, X0, output, model, 30, regression)
                    fi_all_ratio[idx, idxj, i, k] = loss_after / loss_before
                    fi_all_diff[idx, idxj, i, k] = loss_after - loss_before
    return fi_all_ratio, fi_all_diff

def get_all_joint_effects(m_multi_boundary_e, input, output, v_list, n_ways, model, regression=True):
    '''
    :param m_multi_boundary_e: an m matrix in shape [5, 11, n_features, 2]
    :return:
        joint_effect_all_pair_set: all joint effects of features [5, n_joint_pairs, 36] in fis, where 36 is 2^2*9
        loss_emp_all_pair_set: all joint effects of features [5, n_joint_pairs, 36] in loss
    '''
    joint_effect_all_pair_set = []
    loss_emp_all_pair_set = []
    for m_all in m_multi_boundary_e:
        m_all = m_all.transpose((1, 0, 2))
        joint_effect_all_pair, loss_emp = Interaction_effect_all_pairs(input, output, v_list,
                                                                       n_ways, model, m_all,
                                                                       regression=regression)
        joint_effect_all_pair_set.append(joint_effect_all_pair)
        loss_emp_all_pair_set.append(loss_emp)
    return joint_effect_all_pair_set, loss_emp_all_pair_set

def get_fis_in_r(all_pairs, joint_effect_all_pair_set, main_effect_all_diff, n_ways, quadrants):
    '''
    :param pairs: all pairs of interest
    :param joint_effect_all_pair_set: all joint effects of these pairs
    :param main_effect_all_diff: all main effects of these features in the pair
    :return: fis of all pairs in the Rashomon set
    '''
    fis_rset = np.ones(joint_effect_all_pair_set.shape)
    for i in range(5):
        joint_effect_all_pair_e = joint_effect_all_pair_set[i]
        main_effect_all_diff_e = main_effect_all_diff[i]
        main_effect_all_diff_e_reshaped = main_effect_all_diff_e.transpose((1, 0, 2))
        for idx, pair in enumerate(all_pairs):
            # fi is 40x11x2, fij_joint is 780x36
            fi = main_effect_all_diff_e_reshaped[pair[0]]
            fj = main_effect_all_diff_e_reshaped[pair[1]]
            fij_joint = joint_effect_all_pair_e[idx]
            # 9 paris
            sum_to_one = find_all_sum_to_one_pairs(n_ways)
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
    return fis_rset.transpose((1, 0, 2)).reshape(len(all_pairs), -1)

def get_loss_in_r(all_pairs, joint_loss_pair_set, n_ways, quadrants, epsilon, loss):
    '''
    :param pairs: all pairs of interest
    :param joint_loss_pair_set: all joint losses of these pairs
    :return: loss difference of all pairs in the Rashomon set
    '''
    loss_rset = np.ones(joint_loss_pair_set.shape)
    for i, e_sub in enumerate(np.arange(0.2, 1.2, 0.2)):
        joint_effect_all_pair_e = joint_loss_pair_set[i]
        for idx, pair in enumerate(all_pairs):
            fij_joint = joint_effect_all_pair_e[idx]
            # 9 paris
            sum_to_one = find_all_sum_to_one_pairs(n_ways)
            for idxk, sum in enumerate(sum_to_one):
                for idxq, quadrant in enumerate(quadrants):
                    # for each pair, find the main effect
                    single_fis = abs(
                        fij_joint[[idxk * 4 + quadrant]] - e_sub * epsilon - loss)
                    # single_fis = (
                    #     fij_joint[[idxk * 4 + quadrant]] - e_sub*self.epsilon-self.loss)
                    loss_rset[i, idx, idxk * 4 + quadrant] = single_fis
    return loss_rset.transpose((1, 0, 2)).reshape(len(all_pairs), -1)



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