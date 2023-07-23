import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def training_process(explainer, vlist, save=False, suffix=''):
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, len(vlist)))
    points_all = []
    points_all_max = explainer.FIS_main_effect_raw['points_all_max']
    points_all_min = explainer.FIS_main_effect_raw['points_all_min']
    vlist = explainer.v_list
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
