import matplotlib.pyplot as plt
from generalized_rashomon_set.config import OUTPUT_DIR
import numpy as np
from generalized_rashomon_set.utils import Interaction_effect_calculation, high_order_vis_loss, pairwise_vis_loss
from generalized_rashomon_set.utils import colors_vis
def halo_plot(explainer, pair_idx, save=False, suffix=''):
    '''
     :param pair_idx: the pair of interest
     :param save: if save the plot
     :param suffix: halo plot feature name
     '''
    fig = plt.figure(figsize=[6, 6])
    ax = fig.add_subplot(111)
    lightness = [0.8, 0.7, 0.6, 0.5, 0.4]
    # e = [3.3, 3.6, 3.8, 4.1, 4.4]
    for idx, sub_boundary_rate in enumerate(np.arange(0.2, 1.2, 0.2)):
        # feature_idx = feature_idx_to_pair_idx(self.all_pairs, pair_idx=pair_idx)
        # m_all = self.rset_joint_effect_raw['m_multi_boundary_e'][idx].transpose((1,0,2))
        # _, loss_emp = Interaction_effect_calculation(feature_idx, self.model, m_all, self.input, self.output, regression=self.regression)
        loss_emp = explainer.rset_joint_effect_raw['loss_emp_all_pair_set'][idx][pair_idx, :]
        circle_emp, circle_exp = pairwise_vis_loss((loss_emp - explainer.loss), explainer.epsilon * (sub_boundary_rate))
        circle_emp.sort(key=lambda c: np.arctan2(c[0], c[1]))
        circle_emp.append(circle_emp[0])
        circle_exp.sort(key=lambda c: np.arctan2(c[0], c[1]))
        circle_exp.append(circle_exp[0])
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xticks([])
        ax.plot(np.array(circle_emp)[:, 0], np.array(circle_emp)[:, 1], color=colors_vis(0, lightness[idx]), marker='o',
                linewidth=3, markersize=2, label='emperial interaction')
        ax.plot(np.array(circle_exp)[:, 0], np.array(circle_exp)[:, 1], color=colors_vis(1, lightness[idx]),
                linewidth=3, markersize=1, label='expected interaction')
        for location in ['left', 'right', 'top', 'bottom']:
            ax.spines[location].set_linewidth(1)
            ax.spines[location].set_color('black')
    if save:
        plt.savefig(OUTPUT_DIR + '/halo_plot_{}.png'.format(suffix), bbox_inches='tight')
    plt.show()


def halo_plot_3D(explainer, pair_idx, save=False, path=''):
    '''
     :param pair_idx: the pair of interest
     :param save: if save the plot
     :param path: the saving path
     '''
    _, loss_emp_single_pair = Interaction_effect_calculation(pair_idx, explainer.model,
                                                             explainer.rset_main_effect_processed['m_multi_boundary_e'][
                                                                 -1].transpose((1, 0, 2)),
                                                             explainer.input, explainer.output, regression=explainer.regression,
                                                             subset_idx=pair_idx)
    ball_exp, ball_emp = high_order_vis_loss(loss_emp_single_pair, explainer.epsilon, 3, explainer.loss)
    fis_vis_3D(ball_exp, ball_emp, save=save, path=path)


def fis_vis_3D(ball_exp, ball_emp, save=False, path=''):
    fig = plt.figure(figsize=[6,6])
    # ax = fig.add_subplot(111)
    ax = plt.axes(projection='3d')
    ball_exp = np.array(ball_exp)
    ball_emp = np.array(ball_emp)
    ax.set_facecolor("white")
    ax.grid(which='major', linewidth=1, color='black')
    ax.scatter(ball_emp[:, 0], ball_emp[:, 1], ball_emp[:, 2], color=colors_vis(0, 0.4), label='emperial interaction', alpha=.5)
    ax.scatter(ball_exp[:,0], ball_exp[:,1], ball_exp[:,2],color=colors_vis(1, 0.4), label='expected interaction', alpha=.5)
    ax.w_xaxis.line.set_color("black")
    ax.w_yaxis.line.set_color("black")
    ax.w_zaxis.line.set_color("black")
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.w_zaxis.set_ticklabels([])
    # plt.rcParams['axes.color_cycle'] = 'r'

    # ax.view_init(elev=0, azim=0, roll=0)
    if save:
        plt.savefig(path, bbox_inches='tight')
    plt.show()