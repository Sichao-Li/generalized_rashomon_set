import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from ..config import OUTPUT_DIR


def swarm_plot_MR(explainer, interest_of_features, vname=None, plot_all=False, threshold=None, boxplot=False,
                  absolute=False,
                  save=False, suffix=None):
    if vname is None:
        FI_name = explainer.v_list
    else:
        FI_name = vname

    all_main_effects_diff_reshaped = np.array(explainer.rset_main_effect_processed[
                                                  'all_main_effects_diff']).transpose(
        (1, 0, 2)).reshape((len(explainer.v_list), -1))

    if absolute:
        fis_in_r_df = pd.DataFrame(abs(all_main_effects_diff_reshaped))
        loss_in_r_df = pd.DataFrame(abs(
            np.array(explainer.rset_main_effect_processed['loss_diff_multi_boundary_e']).transpose((1, 0, 2)).reshape(
                (len(explainer.v_list), -1))))
        fis_ref_l = abs(np.array(explainer.ref_analysis['ref_main_effects']))[explainer.v_list]
    else:
        fis_in_r_df = pd.DataFrame(all_main_effects_diff_reshaped)
        loss_in_r_df = pd.DataFrame(
            np.array(explainer.rset_main_effect_processed['loss_diff_multi_boundary_e']).transpose((1, 0, 2)).reshape(
                (len(explainer.v_list), -1)))
        fis_ref_l = np.array(explainer.ref_analysis['ref_main_effects'])[explainer.v_list]

    fis_ref_l_df = pd.DataFrame(fis_ref_l)
    fis_in_r_df['Feature'] = FI_name
    loss_in_r_df['Feature'] = FI_name
    fis_ref_l_df['Feature'] = FI_name

    if plot_all:
        fis_in_r_df_long = fis_in_r_df.melt(id_vars='Feature', var_name='m_value',
                                            value_name='Model reliance')
        loss_in_r_df_long = loss_in_r_df.melt(id_vars='Feature', var_name='m_value',
                                              value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.melt(id_vars='Feature', var_name='m_value',
                                              value_name='Model reliance')
    else:
        fis_in_r_df_long = fis_in_r_df.loc[interest_of_features,].melt(id_vars='Feature', var_name='m_value',
                                                                       value_name='Model reliance')
        loss_in_r_df_long = loss_in_r_df.loc[interest_of_features,].melt(id_vars='Feature', var_name='m_value',
                                                                         value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.loc[interest_of_features,].melt(id_vars='Feature', var_name='m_value',
                                                                         value_name='Model reliance')
    fis_in_r_df_long['Loss'] = loss_in_r_df_long['Loss']
    fis_ref_l_df_long['Loss'] = 0
    sns.reset_defaults()
    sns.set(rc={'figure.figsize': (10, 10)})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#1E88E5", '#7C52FF', "#ff0d57"], N=180)
    norm = plt.Normalize(fis_in_r_df_long['Loss'].min(), explainer.loss + explainer.epsilon)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sns.set_style("whitegrid")
    ax2 = sns.swarmplot(data=fis_in_r_df_long, x='Model reliance', y='Feature', hue='Loss', palette=cmap, size=3,
                        zorder=0)

    ax = sns.pointplot(data=fis_ref_l_df_long, x='Model reliance', y='Feature', linestyles='', markers='*',
                       color='orange', scale=1.2, ax=ax2)
    if boxplot:
        sns.boxplot(x="Model reliance", y='Feature', data=fis_in_r_df_long,
                    showcaps=False, boxprops={'facecolor': 'None'},
                    showfliers=False, whiskerprops={'linewidth': 0}, ax=ax)
    ax.get_legend().remove()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('Model reliance', fontsize=18)
    ax.set_ylabel('Features', fontsize=18)
    cbar = ax.figure.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label='Loss in the Rashomon set', size=18)
    for location in ['left', 'right', 'top', 'bottom']:
        ax.spines[location].set_linewidth(1)
        ax.spines[location].set_color('black')
    if threshold is not None:
        plt.axvline(threshold, color='black')
    if save:
        plt.savefig(OUTPUT_DIR + '/swarm_plot_{}.png'.format(suffix), bbox_inches='tight')
    plt.show()


def swarm_plot_MR_multiple(*explainers, interest_of_features=None, vname=None, plot_all=False, threshold=None,
                           boxplot=False,
                           absolute=False,
                           save=False, suffix=None):
    if vname is None:
        FI_name = explainers[0].v_list
    else:
        FI_name = vname

    for idx, explainer in enumerate(explainers):
        # concatenate all fis_in_r and loss_in_r for all explainers
        all_main_effects_diff_reshaped = np.array(
            explainer.rset_main_effect_processed['all_main_effects_diff']).transpose((1, 0, 2)).reshape(
            (len(explainer.v_list), -1))
        if idx == 0:
            fis_in_r_df = pd.DataFrame(all_main_effects_diff_reshaped)
            loss_in_r_df = pd.DataFrame(
                np.array(explainer.rset_main_effect_processed['loss_diff_multi_boundary_e']).transpose(
                    (1, 0, 2)).reshape(
                    (len(explainer.v_list), -1)))
            fis_ref_l = np.array(explainer.ref_analysis['ref_main_effects'])[explainer.v_list]
        else:
            fis_in_r_df = pd.concat([fis_in_r_df, pd.DataFrame(all_main_effects_diff_reshaped)], axis=1)
            loss_in_r_df = pd.concat([loss_in_r_df, pd.DataFrame(
                np.array(explainer.rset_main_effect_processed['loss_diff_multi_boundary_e']).transpose(
                    (1, 0, 2)).reshape(
                    (len(explainer.v_list), -1)))], axis=1)

    fis_ref_l_df = pd.DataFrame(fis_ref_l)

    if absolute:
        fis_in_r_df = fis_in_r_df.abs()
        loss_in_r_df = loss_in_r_df.abs()
        fis_ref_l = fis_ref_l_df.abs()

    fis_in_r_df['Feature'] = FI_name
    loss_in_r_df['Feature'] = FI_name
    fis_ref_l_df['Feature'] = FI_name

    if plot_all:
        fis_in_r_df_long = fis_in_r_df.melt(id_vars='Feature', var_name='m_value',
                                            value_name='Model reliance')
        loss_in_r_df_long = loss_in_r_df.melt(id_vars='Feature', var_name='m_value',
                                              value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.melt(id_vars='Feature', var_name='m_value',
                                              value_name='Model reliance')
    else:
        fis_in_r_df_long = fis_in_r_df.loc[interest_of_features,].melt(id_vars='Feature', var_name='m_value',
                                                                       value_name='Model reliance')
        loss_in_r_df_long = loss_in_r_df.loc[interest_of_features,].melt(id_vars='Feature', var_name='m_value',
                                                                         value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.loc[interest_of_features,].melt(id_vars='Feature', var_name='m_value',
                                                                         value_name='Model reliance')
    fis_in_r_df_long['Loss'] = loss_in_r_df_long['Loss']
    fis_ref_l_df_long['Loss'] = 0
    sns.reset_defaults()
    sns.set(rc={'figure.figsize': (10, 10)})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#1E88E5", '#7C52FF', "#ff0d57"], N=180)
    norm = plt.Normalize(fis_in_r_df_long['Loss'].min(), explainer.loss + explainer.epsilon)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sns.set_style("whitegrid")
    ax2 = sns.swarmplot(data=fis_in_r_df_long, x='Model reliance', y='Feature', hue='Loss', palette=cmap, size=3,
                        zorder=0)

    ax = sns.pointplot(data=fis_ref_l_df_long, x='Model reliance', y='Feature', linestyles='', markers='*',
                       color='orange', scale=1.2, ax=ax2)
    if boxplot:
        sns.boxplot(x="Model reliance", y='Feature', data=fis_in_r_df_long,
                    showcaps=False, boxprops={'facecolor': 'None'},
                    showfliers=False, whiskerprops={'linewidth': 0}, ax=ax)
    ax.get_legend().remove()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('Model reliance', fontsize=18)
    ax.set_ylabel('Features', fontsize=18)
    cbar = ax.figure.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label='Loss in the Rashomon set', size=18)
    for location in ['left', 'right', 'top', 'bottom']:
        ax.spines[location].set_linewidth(1)
        ax.spines[location].set_color('black')
    if threshold is not None:
        plt.axvline(threshold, color='black')
    if save:
        plt.savefig(OUTPUT_DIR + '/swarm_plot_{}.png'.format(suffix), bbox_inches='tight')
    plt.show()


def swarm_plot_FIS(explainer, interest_of_pairs, vname=None, plot_all=False, threshold=None, boxplot=False, save=False,
                   absolute=False,
                   suffix=None):
    '''
    :param interest_of_pairs: all pairs of interest
    :param vname: variable name list
    :param plot_all: if plot all pairs of features
    :param threshold: if there is a threshold to decide fis
    :param boxplot: if plot boxplot
    :param save: if save the plot
    '''

    FI_name = []
    for i in explainer.all_pairs:
        if vname is None:
            name = str(i[0]) + ' vs ' + str(i[1])
        else:

            name = str(vname[i[0]]) + ' vs ' + str(vname[i[1]])
        FI_name.append(name)
    if absolute:
        fis_in_r_df = pd.DataFrame(abs(explainer.fis_in_r))
        loss_in_r_df = pd.DataFrame(abs(explainer.loss_in_r))
        fis_ref_l = [abs(i[-1]) for i in explainer.ref_analysis['ref_fis']]
    else:
        fis_in_r_df = pd.DataFrame(explainer.fis_in_r)
        loss_in_r_df = pd.DataFrame(explainer.loss_in_r)
        fis_ref_l = [i[-1] for i in explainer.ref_analysis['ref_fis']]
    fis_ref_l_df = pd.DataFrame(fis_ref_l)
    fis_in_r_df['Interaction pairs'] = FI_name
    loss_in_r_df['Interaction pairs'] = FI_name
    fis_ref_l_df['Interaction pairs'] = FI_name

    list_idx = []
    for pair in interest_of_pairs:
        list_idx.append(explainer.all_pairs.index(pair))

    if plot_all:
        fis_in_r_df_long = fis_in_r_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                            value_name='FIS')
        loss_in_r_df_long = loss_in_r_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                              value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                              value_name='FIS')

    else:
        fis_in_r_df_long = fis_in_r_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value',
                                                           value_name='FIS')
        loss_in_r_df_long = loss_in_r_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value',
                                                             value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value',
                                                             value_name='FIS')

    fis_in_r_df_long['Loss'] = loss_in_r_df_long['Loss']
    fis_ref_l_df_long['Loss'] = 0
    sns.reset_defaults()
    sns.set(rc={'figure.figsize': (10, 10)})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#1E88E5", '#7C52FF', "#ff0d57"], N=180)
    norm = plt.Normalize(fis_in_r_df_long['Loss'].min(), explainer.loss + explainer.epsilon)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sns.set_style("whitegrid")
    ax2 = sns.swarmplot(data=fis_in_r_df_long, x='FIS', y='Interaction pairs', hue='Loss', palette=cmap, size=3,
                        zorder=0)

    ax = sns.pointplot(data=fis_ref_l_df_long, x='FIS', y='Interaction pairs', linestyles='', markers='*',
                       color='orange', scale=1.2, ax=ax2)
    if boxplot:
        sns.boxplot(x="FIS", y='Interaction pairs', data=fis_in_r_df_long,
                    showcaps=False, boxprops={'facecolor': 'None'},
                    showfliers=False, whiskerprops={'linewidth': 0}, ax=ax)
    ax.get_legend().remove()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('FIS', fontsize=18)
    ax.set_ylabel('Interaction Pairs', fontsize=18)
    cbar = ax.figure.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label='Loss in the Rashomon set', size=18)
    for location in ['left', 'right', 'top', 'bottom']:
        ax.spines[location].set_linewidth(1)
        ax.spines[location].set_color('black')
    if threshold is not None:
        plt.axvline(threshold, color='black')
    if save:
        plt.savefig(OUTPUT_DIR + '/swarm_plot_{}.png'.format(suffix), bbox_inches='tight')
    plt.show()


def swarm_plot_FIS_multiple(*explainers, interest_of_pairs=None, vname=None, plot_all=False, threshold=None,
                            boxplot=False, save=False, absolute=False,
                            suffix=None):
    FI_name = []
    for i in explainers[0].all_pairs:
        if vname is None:
            name = str(i[0]) + ' vs ' + str(i[1])
        else:

            name = str(vname[i[0]]) + ' vs ' + str(vname[i[1]])
        FI_name.append(name)
    for idx, explainer in enumerate(explainers):
        # concatenate all fis_in_r and loss_in_r for all explainers
        if idx == 0:
            fis_in_r_df = pd.DataFrame(explainer.fis_in_r)
            loss_in_r_df = pd.DataFrame(explainer.loss_in_r)
            fis_ref_l = [i[-1] for i in explainer.ref_analysis['ref_fis']]
        else:
            fis_in_r_df = pd.concat([fis_in_r_df, pd.DataFrame(explainer.fis_in_r)], axis=1)
            loss_in_r_df = pd.concat([loss_in_r_df, pd.DataFrame(explainer.loss_in_r)], axis=1)
    fis_ref_l_df = pd.DataFrame(fis_ref_l)

    if absolute:
        fis_in_r_df = fis_in_r_df.abs()
        loss_in_r_df = loss_in_r_df.abs()
        fis_ref_l_df = fis_ref_l_df.abs()

    fis_in_r_df['Interaction pairs'] = FI_name
    loss_in_r_df['Interaction pairs'] = FI_name
    fis_ref_l_df['Interaction pairs'] = FI_name

    list_idx = []
    for pair in interest_of_pairs:
        list_idx.append(explainers[0].all_pairs.index(pair))

    if plot_all:
        fis_in_r_df_long = fis_in_r_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                            value_name='FIS')
        loss_in_r_df_long = loss_in_r_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                              value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.melt(id_vars='Interaction pairs', var_name='m_value',
                                              value_name='FIS')

    else:
        fis_in_r_df_long = fis_in_r_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value',
                                                           value_name='FIS')
        loss_in_r_df_long = loss_in_r_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value',
                                                             value_name='Loss')
        fis_ref_l_df_long = fis_ref_l_df.loc[list_idx,].melt(id_vars='Interaction pairs', var_name='m_value',
                                                             value_name='FIS')

    fis_in_r_df_long['Loss'] = loss_in_r_df_long['Loss']
    fis_ref_l_df_long['Loss'] = 0
    sns.reset_defaults()
    sns.set(rc={'figure.figsize': (10, 10)})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#1E88E5", '#7C52FF', "#ff0d57"], N=180)
    norm = plt.Normalize(fis_in_r_df_long['Loss'].min(), explainers[-1].loss + explainers[-1].epsilon)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sns.set_style("whitegrid")
    ax2 = sns.swarmplot(data=fis_in_r_df_long, x='FIS', y='Interaction pairs', hue='Loss', palette=cmap, size=3,
                        zorder=0)

    ax = sns.pointplot(data=fis_ref_l_df_long, x='FIS', y='Interaction pairs', linestyles='', markers='*',
                       color='orange', scale=1.2, ax=ax2)
    if boxplot:
        sns.boxplot(x="FIS", y='Interaction pairs', data=fis_in_r_df_long,
                    showcaps=False, boxprops={'facecolor': 'None'},
                    showfliers=False, whiskerprops={'linewidth': 0}, ax=ax)
    ax.get_legend().remove()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('FIS', fontsize=18)
    ax.set_ylabel('Interaction Pairs', fontsize=18)
    cbar = ax.figure.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label='Loss in the Rashomon set', size=18)
    for location in ['left', 'right', 'top', 'bottom']:
        ax.spines[location].set_linewidth(1)
        ax.spines[location].set_color('black')
    if threshold is not None:
        plt.axvline(threshold, color='black')
    if save:
        plt.savefig(OUTPUT_DIR + '/swarm_plot_{}.png'.format(suffix), bbox_inches='tight')
    plt.show()