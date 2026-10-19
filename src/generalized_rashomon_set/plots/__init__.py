"""Plotting: swarm plots (attribution / FIS clouds), halo plots and the
search trace.  All functions return the matplotlib ``Axes`` and never call
``plt.show()``; use ``ax.figure.savefig(...)`` to save."""
from ._halo import halo_coordinates, halo_plot, halo_plot_3d
from ._style import colors_vis
from ._swarm import swarm_plot_attribution, swarm_plot_interaction
from ._trace import search_trace_plot
from ._legacy import (halo_plot_multiple, swarm_plot_FIS, swarm_plot_FIS_multiple, swarm_plot_MR,
                      swarm_plot_MR_multiple, training_process, halo_plot_3D)

__all__ = ["halo_plot", "halo_plot_3d", "halo_coordinates", "swarm_plot_attribution",
           "swarm_plot_interaction", "search_trace_plot", "colors_vis",
           # legacy names
           "halo_plot_multiple", "halo_plot_3D", "swarm_plot_FIS", "swarm_plot_FIS_multiple",
           "swarm_plot_MR", "swarm_plot_MR_multiple", "training_process"]
