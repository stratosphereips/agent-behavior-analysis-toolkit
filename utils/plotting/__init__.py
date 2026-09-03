"""Plotting utilities, grouped by purpose.

- ``fingerprint``    -- behavioral-fingerprint / sequential-checkpoint diagnostics (current method)
- ``report``         -- the shipped per-run Behavioral Fingerprint report figure
- ``trajectory``     -- per-trajectory action-usage and segment-cluster visualizations
- ``generalization`` -- seen-vs-unseen behavioral-generalization figure

The public functions are re-exported here so callers can do
``from utils.plotting import plot_sequential_cp_metrics``. The legacy module
``utils.plotting_utils`` is a thin shim that re-exports the diagnostic names.
"""
from utils.plotting.fingerprint import plot_sequential_cp_metrics
from utils.plotting.report import plot_fingerprint_report
from utils.plotting.trajectory import (
    plot_action_per_step_distribution,
    plot_segment_cluster_features,
)
from utils.plotting.generalization import (
    plot_generalization_bidirectional,
    plot_action_divergence_per_step,
    plot_action_divergence_overlay,
    plot_jsd_vs_unseen_return,
)

__all__ = [
    "plot_sequential_cp_metrics",
    "plot_fingerprint_report",
    "plot_action_per_step_distribution",
    "plot_segment_cluster_features",
    "plot_generalization_bidirectional",
    "plot_action_divergence_per_step",
    "plot_action_divergence_overlay",
    "plot_jsd_vs_unseen_return",
]
