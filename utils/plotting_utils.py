"""Backward-compatibility shim.

Plotting code now lives in the ``utils.plotting`` package, split by purpose. This
module re-exports the surviving public functions so existing
``from utils.plotting_utils import X`` imports keep working. Prefer importing from
``utils.plotting`` (or one of its submodules) in new code.
"""
from utils.plotting import (
    plot_action_per_step_distribution,
    plot_generalization_bidirectional,
    plot_action_divergence_per_step,
    plot_action_divergence_overlay,
    plot_jsd_vs_unseen_return,
    plot_segment_cluster_features,
    plot_sequential_cp_metrics,
)

__all__ = [
    "plot_action_per_step_distribution",
    "plot_generalization_bidirectional",
    "plot_action_divergence_per_step",
    "plot_action_divergence_overlay",
    "plot_jsd_vs_unseen_return",
    "plot_segment_cluster_features",
    "plot_sequential_cp_metrics",
]
