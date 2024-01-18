import numpy as np
from ._general_utils import find_all_sum_to_one_pairs
import itertools

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