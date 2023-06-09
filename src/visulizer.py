import matplotlib.pyplot as plt

from general_utilities import *
from feature_interaction_score_utilities import *

def epsilon_vs_score_vs_m(vlist, vt_fi, vt_f):
    fig, ax = plt.subplots(3,1, figsize=(6, 6), facecolor='w', edgecolor='k')
    colors = cm.rainbow(np.linspace(0, 1, len(vlist)))
    for i in range(len(vlist)):
#     xx, xy = zip(*sorted(zip(i[:,0],i[:,1])))
        xx_e = np.linspace(0,1,11)
        xx_s = vt_fi[i,:,0]
        xx_m = vt_f[i,:,0]
        yy_s = vt_fi[i,:,1]
        yy_m = vt_f[i,:,1]
        ax[0].plot(xx_e, xx_s, marker='o', color = colors[i], label=vlist[i])
        ax[0].plot(xx_e, yy_s, marker='o', color = colors[i])
        ax[1].plot(xx_e, xx_m, marker='o', color = colors[i], label=vlist[i])
        ax[1].plot(xx_e, yy_m, marker='o', color = colors[i], label=vlist[i])        
        ax[2].plot(yy_s, yy_m, marker='o', color = colors[i], label=vlist[i])
        ax[2].plot(xx_s, xx_m, marker='o', color = colors[i], label=vlist[i])
    ax[0].legend(bbox_to_anchor=(1.1, 1.05), shadow=True)

    
def epsilon_vs_score_vs_m_3D(vlist, vt_fi, vt_f):
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.rainbow(np.linspace(0, 1, len(vlist)))
    for i in range(len(vlist)):
        xx = vt_f[i,:,0]
        xy = vt_fi[i,:,0]
        zz = np.linspace(0,1,11)
        yx = vt_f[i,:,1]
        yy = vt_fi[i,:,1]
        ax.plot(yx, yy, zz, marker='o', color = colors[i], label=vlist[i])
        ax.plot(xx, xy, zz, marker='o', color = colors[i])
    ax.legend(bbox_to_anchor=(1.1, 1.05), shadow=True)
    
def m_training_process_vis(points_all_max, points_all_min, vlist, save=False, suffix=''):
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, len(vlist)))
    points_all = []
    for idx, i in enumerate(range(len(points_all_min))[::-1]):
        points_single = points_all_max[idx]+points_all_min[idx]
        points_sorted = []
        for p in points_single:
            points_sorted += p
        points_all.append(points_sorted)

    for idx, i in enumerate(points_all):
        i=np.array(i)
        xx, xy = zip(*sorted(zip(i[:,0],i[:,1])))
        ax.plot(xx, xy, marker='o', color = colors[idx], label=vlist[idx])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color ="black", linestyle='--', lw = 1, alpha=0.5)
    ax.set_facecolor("white")
    ax.legend(bbox_to_anchor=(1.0, 1.0), shadow=False, facecolor='white')
    ax.set_xlabel('Individual mask')
    ax.set_ylabel('Loss difference')
    for location in ['left', 'right', 'top', 'bottom']:
        ax.spines[location].set_linewidth(1)
        ax.spines[location].set_color('black')
    if save:
        plt.savefig('../results/crime/crime_vic/training_process_{}.png'.format(suffix), bbox_inches='tight')
    plt.show()


#     ax.view_init(elev=60)

# def feature_interaction_vis_2D(circle_exp, circle_emp, boundary, diff_all, pair_idx):
#     fig = plt.figure(figsize=[6,6])
#     ax = fig.add_subplot(111)
#     circle_emp.sort(key=lambda c:np.arctan2(c[0], c[1]))
#     circle_emp.append(circle_emp[0])
#     circle_exp.sort(key=lambda c:np.arctan2(c[0], c[1]))
#     circle_exp.append(circle_exp[0])
#     ax.plot(np.array(circle_emp)[:,0], np.array(circle_emp)[:,1], marker='o', linewidth=2, markersize=2, label='emperial interaction')
#     ax.plot(np.array(circle_exp)[:,0], np.array(circle_exp)[:,1], linewidth=1, markersize=1, label='expected interaction')
#     ax.fill(np.concatenate((np.array(circle_emp)[:,0], np.array(circle_exp)[:,0][::-1])), np.concatenate((np.array(circle_emp)[:,1], np.array(circle_exp)[:,1][::-1])), alpha=0.5, label='interaction difference')
#
# #     ax.add_patch(plt.Rectangle((0,-boundary), boundary*np.sqrt(2), boundary*np.sqrt(2), angle=45, fill=False, linestyle='--', color='g'))
#     max_cir = np.max(diff_all[pair_idx])
#     min_cir = np.min(diff_all[pair_idx])
#     avg_cir = np.average(diff_all[pair_idx])
#     ax.add_patch(plt.Circle((0, 0), avg_cir+boundary, fill=False, linestyle='--', color='r', label='average interaction'))
#     ax.add_patch(plt.Circle((0, 0), min_cir+boundary, fill=False, linestyle='--', color='g', label='minimum interaction'))
#     ax.add_patch(plt.Circle((0, 0), max_cir+boundary, fill=False, linestyle='--', color='purple', label='maximum interaction'))
#
#     # ax.add_patch(plt.Circle((0, 0), boundary, fill=False))
#     ax.legend(bbox_to_anchor=(1.0, 1.02), shadow=True)
#     ax.set_aspect("equal", adjustable="datalim")
#     ax.set_box_aspect(0.5)
#     ax.autoscale()
    
# def feature_interaction_vis(vlist, n_ways):
#     '''
#     Visulize feature interaction effects
#     '''
#     n_pairs = find_all_n_way_feature_pairs(vlist, n_ways)
#     sum_to_one_pairs = find_all_sum_to_one_pairs(n_ways)
#     fig, axs = plt.subplots(2, 2, figsize=(15, 6), facecolor='w', edgecolor='k')
#     axs = axs.ravel()
#     print(axs)
#     coord = []
#     n_sub_plots = 4
#     for i in range(n_sub_plots):
# #         _, cor, circle, _ = feature_interaction_strength(diff_all, myloss_all, 3, bound_f, 0, loss_ref)
#         cor,circle = diff_points_cal(sum_to_one_pairs, diff_all[i], bound_f)
#
#         cor.sort(key=lambda c:np.arctan2(c[0], c[1]))
#         cor.append(cor[0])
#
#         circle.sort(key=lambda c:np.arctan2(c[0], c[1]))
#         circle.append(circle[0])
#
#         xx = np.concatenate((np.array(cor)[:,0], np.array(circle)[:,0][::-1]))
#         xy = np.concatenate((np.array(cor)[:,1], np.array(circle)[:,1][::-1]))
#         coord.append([xx,xy])
#
#         axs[i].plot(np.array(cor)[:,0], np.array(cor)[:,1], marker='o', linewidth=2, markersize=2)
#         axs[i].plot(np.array(circle)[:,0], np.array(circle)[:,1], linewidth=1, markersize=1)
#         axs[i].fill(xx, xy, alpha=0.5)
#         max_cir = np.max(diff_all[i])
#         min_cir = np.min(diff_all[i])
#         avg_cir = np.average(diff_all[i])
#         axs[i].add_patch(plt.Circle((0, 0), avg_cir+bound_f, fill=False, linestyle='--', color='r'))
#         axs[i].add_patch(plt.Circle((0, 0), min_cir+bound_f, fill=False, linestyle='--', color='g'))
#         axs[i].add_patch(plt.Circle((0, 0), max_cir+bound_f, fill=False, linestyle='--', color='purple'))
#         # ax.add_patch(plt.Circle((0, 0), bound_f, fill=False))
#         axs[i].set_aspect("equal", adjustable="datalim")
#         axs[i].set_box_aspect(0.5)
#         axs[i].autoscale()
#     plt.show()

def swarm_plot(fis_in_r, loss_in_r, fis_ref_l, all_pairs, interest_of_pairs, vname=None, plot_all=False, threshold=None, loss=0, epsilon=0, boxplot=False, save=False, suffix=None):

    # pair names
    FI_name = []
    for i in all_pairs:
        if vname is None:
            name = str(i[0]) + ' vs ' + str(i[1])
        else:

            name = str(vname[i[0]]) + ' vs ' + str(vname[i[1]])
        FI_name.append(name)
    fis_in_r_df = pd.DataFrame(fis_in_r)
    loss_in_r_df = pd.DataFrame(loss_in_r)
    fis_ref_l_df = pd.DataFrame(fis_ref_l)
    fis_in_r_df['Interaction pairs'] = FI_name
    loss_in_r_df['Interaction pairs'] = FI_name
    fis_ref_l_df['Interaction pairs'] = FI_name

    list_idx = []
    for pair in interest_of_pairs:
        list_idx.append(all_pairs.index(pair))

    if plot_all:
        fis_in_r_df_long = fis_in_r_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                                                    value_name='FIS')
        loss_in_r_df_long = loss_in_r_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                                                      value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                                                      value_name='FIS')

    else:
        fis_in_r_df_long = fis_in_r_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value', value_name='FIS')
        loss_in_r_df_long = loss_in_r_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value',
                                                                      value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value',
                                                                      value_name='FIS')


    fis_in_r_df_long['Loss'] = loss_in_r_df_long['Loss']
    fis_ref_l_df_long['Loss'] = 0
    sns.reset_defaults()
    sns.set(rc={'figure.figsize': (10, 10)})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#1E88E5", '#7C52FF', "#ff0d57"], N=180)
    norm = plt.Normalize(fis_in_r_df_long['Loss'].min(), loss + epsilon)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sns.set_style("whitegrid")
    ax2 = sns.swarmplot(data=fis_in_r_df_long, x='FIS', y='Interaction pairs', hue='Loss', palette=cmap, size=3, zorder=0)

    ax = sns.pointplot(data=fis_ref_l_df_long, x='FIS', y='Interaction pairs', linestyles='',markers = '*', color='orange', scale = 1.2,ax=ax2)
    if boxplot:
        sns.boxplot(x="FIS", y='Interaction pairs', data=fis_in_r_df_long,
                showcaps=False,boxprops={'facecolor':'None'},
                showfliers=False,whiskerprops={'linewidth':0}, ax=ax)
    ax.get_legend().remove()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('FIS', fontsize=18)
    ax.set_ylabel('Interaction Pairs', fontsize=18)
    ax.figure.colorbar(sm, fraction=0.046, pad=0.04)
    for location in ['left', 'right', 'top', 'bottom']:
        ax.spines[location].set_linewidth(1)
        ax.spines[location].set_color('black')
    if threshold is not None:
        plt.axvline(threshold, color='black')
    if save:
        plt.savefig('../results/crime/crime_optimal/swarm_plot_{}.png'.format(suffix), bbox_inches='tight')
    plt.show()

def pairwise_vis_single(loss_emp_all, loss_ref, boundary, pair_idx):
    loss_diff = loss_emp_all[pair_idx]-loss_ref
    circle_emp, circle_exp = pairwise_vis_loss(loss_diff, boundary)
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    circle_emp.sort(key=lambda c:np.arctan2(c[0], c[1]))
    circle_emp.append(circle_emp[0])
    circle_exp.sort(key=lambda c:np.arctan2(c[0], c[1]))
    circle_exp.append(circle_exp[0])
    ax.plot(np.array(circle_emp)[:,0], np.array(circle_emp)[:,1], color=colors_vis(0, 0.4), marker='o', linewidth=2, markersize=2, label='emperial interaction')
    ax.plot(np.array(circle_exp)[:,0], np.array(circle_exp)[:,1], color=colors_vis(1, 0.4), linewidth=1, markersize=1, label='expected interaction')
    ax.fill(np.concatenate((np.array(circle_emp)[:,0], np.array(circle_exp)[:,0][::-1])), np.concatenate((np.array(circle_emp)[:,1], np.array(circle_exp)[:,1][::-1])), alpha=0.5, label='interaction difference')
    plt.show()

def pairwise_vis_multi(loss_ref, boundary, all_pairs, pair_idx, model=None, m_all_sub_set=None ,X_test=None, y_test=None, regression=False, loss_emp_all_pair_set=None, save=False, path=''):
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    lightness = [0.8, 0.7, 0.6, 0.5, 0.4]
    e = [3.3, 3.6, 3.8, 4.1, 4.4]
    for idx, sub_boundary_rate in enumerate(np.arange(0.2, 1.2, 0.2)):
        if loss_emp_all_pair_set is None:
            feature_idx = feature_idx_to_pair_idx(all_pairs, pair_idx=pair_idx)
            m_all = m_all_sub_set[idx].transpose((1,0,2))
            _, loss_emp = Interaction_effect_calculation(feature_idx, model, m_all, X_test, y_test, regression=regression)
        else:
            loss_emp = loss_emp_all_pair_set[idx][pair_idx, :]
        circle_emp, circle_exp = pairwise_vis_loss((loss_emp-loss_ref), boundary*(sub_boundary_rate))
        circle_emp.sort(key=lambda c:np.arctan2(c[0], c[1]))
        circle_emp.append(circle_emp[0])
        circle_exp.sort(key=lambda c:np.arctan2(c[0], c[1]))
        circle_exp.append(circle_exp[0])
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xticks([])
        ax.plot(np.array(circle_emp)[:,0], np.array(circle_emp)[:,1],color=colors_vis(0, lightness[idx]), marker='o', linewidth=3, markersize=2, label='emperial interaction')
        ax.plot(np.array(circle_exp)[:,0], np.array(circle_exp)[:,1],color=colors_vis(1, lightness[idx]), linewidth=3, markersize=1, label='expected interaction')
        for location in ['left', 'right', 'top', 'bottom']:
            ax.spines[location].set_linewidth(1)
            ax.spines[location].set_color('black')
    if save:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    # ax.fill(np.concatenate((np.array(circle_emp)[:,0], np.array(circle_exp)[:,0][::-1])), np.concatenate((np.array(circle_emp)[:,1], np.array(circle_exp)[:,1][::-1])), alpha=0.2, label='interaction difference')

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