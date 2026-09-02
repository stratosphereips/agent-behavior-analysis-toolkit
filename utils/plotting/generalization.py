"""Behavioral-generalization figure: seen-vs-unseen action usage over time.

`plot_generalization_bidirectional` renders, per model, action-type usage as a
double stacked-count chart (T_seen upward, T_unseen downward). Used by
`scripts/replay/generalization_experiment.py`.
"""
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Colour-blind-safe palette (Okabe & Ito, 2008) for the canonical NetSecGame
# action-type kill-chain. Yellow deliberately avoided (low contrast on white).
_GENERALIZATION_ACTION_COLORS = {
    "ScanNetwork":    "#0072B2",  # blue
    "FindServices":   "#E69F00",  # orange
    "ExploitService": "#009E73",  # bluish green
    "FindData":       "#D55E00",  # vermillion
    "ExfiltrateData": "#CC79A7",  # reddish purple
}
_GENERALIZATION_FALLBACK_COLORS = ["#000000", "#56B4E9", "#F0E442"]

# Colour-blind-safe qualitative palette (Okabe & Ito, 2008) for distinguishing
# models/checkpoints on a shared axes (divergence overlay, JSD-vs-return scatter).
# Ordered for maximum contrast between adjacent entries; cycles if there are more
# models than colours. Kept separate from `_GENERALIZATION_ACTION_COLORS` since
# it colours a different dimension (model identity, not action type).
_MODEL_COLOR_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#999999",  # grey
]


def _model_color(i: int) -> str:
    """Colour-blind-safe colour for the i-th model, cycling if there are more
    models than palette entries."""
    return _MODEL_COLOR_PALETTE[i % len(_MODEL_COLOR_PALETTE)]


def _generalization_action_name(action) -> str:
    """Map an action token (Enum member, string, or int) to a canonical display name."""
    name = getattr(action, "name", None)
    if name is not None:
        return name
    s = str(action)
    return s.split(".")[-1] if "." in s else s


def _generalization_counts(trajectories: Iterable, action_index: dict, n_steps: int) -> np.ndarray:
    """counts[t, k] = number of trajectories choosing action k at step t."""
    counts = np.zeros((n_steps, len(action_index)))
    for trajectory in trajectories:
        for t, transition in enumerate(trajectory):
            if t >= n_steps:
                break
            idx = action_index.get(transition.action)
            if idx is not None:
                counts[t, idx] += 1
    return counts


def _generalization_emphasis_scale(t_max: int, early_split: int, early_fraction: float):
    """X-axis warp that expands the first `early_split` steps to `early_fraction` of the axis width."""
    split = min(early_split, max(1, t_max - 1))
    w = early_fraction

    def fwd(x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= split,
                         x / split * w,
                         w + (x - split) / (t_max - split) * (1.0 - w))

    def inv(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= w,
                         y / w * split,
                         split + (y - w) / (1.0 - w) * (t_max - split))

    return split, fwd, inv


def plot_generalization_bidirectional(
    models: dict,
    global_actions: list,
    optimal_length: int = 5,
    early_multiplier: int = 5,
    early_fraction: float = 0.62,
    dpi: int = 350,
) -> plt.Figure:
    """
    Behavioural-generalization figure: for every model, plots action-type usage over
    time as a "double" stacked-count chart. T_seen grows UPWARD from zero, T_unseen
    grows DOWNWARD from zero; both halves are labelled with positive trajectory counts.

      * Height of a half -> reachability (how many trajectories survived to step t)
      * Coloured bands   -> the action sequence / behavioural signature
      * Up vs. down       -> seen vs. unseen topology

    Parameters:
        models: {model_name: {"seen": [Trajectory, ...], "unseen": [Trajectory, ...]}}
        global_actions: ordered list of canonical actions (Enum members, strings, or ints)
                        used to build the colour/stacking order.
        optimal_length: minimum number of steps needed to solve the task; the x-axis
                        expands the region up to `early_multiplier * optimal_length` steps.
        dpi: figure DPI.
    """
    action_names = [_generalization_action_name(a) for a in global_actions]
    action_index = {action: i for i, action in enumerate(global_actions)}
    colors = [
        _GENERALIZATION_ACTION_COLORS.get(name, _GENERALIZATION_FALLBACK_COLORS[i % len(_GENERALIZATION_FALLBACK_COLORS)])
        for i, name in enumerate(action_names)
    ]

    t_max = 0
    for kinds in models.values():
        for trajectories in kinds.values():
            for trajectory in trajectories:
                t_max = max(t_max, len(trajectory))

    per_model = {}
    for name, kinds in models.items():
        cs = _generalization_counts(kinds.get("seen", []), action_index, t_max)
        cu = _generalization_counts(kinds.get("unseen", []), action_index, t_max)
        model_y_max = max(cs.sum(1).max(initial=0), cu.sum(1).max(initial=0))
        model_y_max = int(np.ceil(model_y_max / 10.0) * 10) if model_y_max > 0 else 1
        per_model[name] = (cs, cu, model_y_max)

    early_split = early_multiplier * optimal_length
    split, fwd, inv = _generalization_emphasis_scale(t_max, early_split, early_fraction)

    early_ticks = np.arange(0, split + 1, 5)
    late_start = ((split // 10) + 1) * 10
    late_ticks = np.arange(late_start, t_max, 10)
    xticks_candidates = sorted(set(early_ticks.tolist()) | set(late_ticks.tolist()))

    # Drop candidates that land too close (in transformed x) to the previous kept
    # tick, which otherwise collide/overlap right around the early/late boundary.
    min_gap = 0.035
    xticks = [xticks_candidates[0]]
    for tick in xticks_candidates[1:]:
        if fwd(tick) - fwd(xticks[-1]) >= min_gap:
            xticks.append(tick)

    t = np.arange(t_max)
    n = len(per_model)
    fig, axes = plt.subplots(n, 1, figsize=(8.2, 1.7 * n + 1.2), sharex=True, dpi=dpi)
    if n == 1:
        axes = [axes]

    for ax, (name, (cs, cu, model_y_max)) in zip(axes, per_model.items()):
        ax.stackplot(t, *[cs[:, k] for k in range(len(action_names))], colors=colors, alpha=0.95)
        ax.stackplot(t, *[-cu[:, k] for k in range(len(action_names))], colors=colors, alpha=0.95)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-model_y_max, model_y_max)
        ax.set_xlim(0, max(t_max - 1, 1))
        ax.set_xscale("function", functions=(fwd, inv))
        ax.axvline(split, color="0.4", ls=":", lw=0.9)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(int(round(v)))}"))
        ax.yaxis.set_major_locator(MultipleLocator(max(10, model_y_max // 2)))
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelbottom=True)
        ax.text(0.99, 0.85, "$T_{seen}$", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="0.3")
        ax.text(0.99, 0.15, "$T_{unseen}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.3")

    axes[-1].set_xlabel("Time step")

    legend_handles = [Patch(facecolor=c, label=n) for c, n in zip(colors, action_names)]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(5, len(legend_handles)), fancybox=True, shadow=True)

    fig.suptitle("Behavioral Generalization: Action Usage Over Time (seen vs. unseen topology)", fontsize=11)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def plot_action_divergence_per_step(models_jsd: dict, dpi: int = 350) -> plt.Figure:
    """
    Plots, per model, the per-step Jensen-Shannon divergence between the seen and
    unseen action distributions (episodes padded with a win/loss terminal token so
    the divergence stays defined over the full population at every step -- see
    ``utils.metrics.compute_stepwise_action_jsd``), together with its mean.

    Parameters:
        models_jsd: {model_name: {"steps": [...], "jsd_per_step": [...], "mean_jsd": float}}
                    as returned by ``compute_stepwise_action_jsd`` for each model.
        dpi: figure DPI.
    """
    n = len(models_jsd)
    fig, axes = plt.subplots(n, 1, figsize=(8.2, 2.0 * n + 0.8), sharex=True, dpi=dpi)
    if n == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, models_jsd.items()):
        steps = result["steps"]
        jsd = result["jsd_per_step"]
        mean_jsd = result["mean_jsd"]
        ax.plot(steps, jsd, color="#0072B2", lw=1.1)
        ax.axhline(mean_jsd, color="0.3", ls="--", lw=1.0, label=f"mean = {mean_jsd:.3f}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max(steps) if steps else 1)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Behavioral Divergence: Seen vs. Unseen Action Distribution (JSD per step)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_action_divergence_overlay(models_jsd: dict, dpi: int = 350) -> plt.Figure:
    """
    Overlays every model's per-step seen-vs-unseen action-distribution JSD on a single
    axes, for direct cross-model comparison (companion to the per-model stacked
    version in ``plot_action_divergence_per_step``). Each model's mean JSD is
    reported in its legend label rather than drawn on the axes, to keep the
    overlay of 8+ curves legible.

    Parameters:
        models_jsd: {model_name: {"steps": [...], "jsd_per_step": [...], "mean_jsd": float}}
                    as returned by ``compute_stepwise_action_jsd`` for each model.
        dpi: figure DPI.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)

    for i, (name, result) in enumerate(models_jsd.items()):
        color = _model_color(i)
        label = f"{name.replace('\n', ' ')} (mean={result['mean_jsd']:.3f})"
        ax.plot(result["steps"], result["jsd_per_step"], color=color, lw=1.2, label=label)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time step")
    ax.set_ylabel("JSD [0, 1]")
    ax.set_title("Behavioral Divergence: Seen vs. Unseen Action Distribution (JSD per step)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_jsd_vs_unseen_return(models_jsd: dict, unseen_returns: dict, dpi: int = 350) -> plt.Figure:
    """
    Scatter of each model's mean seen-vs-unseen action-distribution JSD against its
    mean return on unseen topologies -- one point per model, colored to match
    ``plot_action_divergence_overlay`` so the two figures cross-reference directly.

    Parameters:
        models_jsd: {model_name: {"mean_jsd": float, ...}}, as returned by
                    ``compute_stepwise_action_jsd`` for each model.
        unseen_returns: {model_name: float}, mean total reward over that model's
                    unseen trajectories.
        dpi: figure DPI.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)

    for i, (name, result) in enumerate(models_jsd.items()):
        if name not in unseen_returns:
            continue
        x = result["mean_jsd"]
        y = unseen_returns[name]
        label = name.replace("\n", " ")
        ax.scatter(x, y, color=_model_color(i), s=60, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="0.25")

    ax.set_xlabel("Mean action-distribution JSD (seen vs. unseen)")
    ax.set_ylabel("Mean return (unseen)")
    ax.set_title("Behavioral Divergence vs. Unseen-Topology Performance")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
