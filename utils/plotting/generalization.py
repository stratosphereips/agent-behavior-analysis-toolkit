"""Behavioral-generalization figure: seen-vs-unseen action usage over time.

`plot_generalization_bidirectional` renders, per model, action-type usage as a
double stacked-count chart (T_seen upward, T_unseen downward). Used by
`scripts/replay/generalization_experiment.py`.
"""
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator, LogLocator, NullFormatter, ScalarFormatter

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

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


def _stacked_step_fill(ax, x, counts: np.ndarray, colors: list, sign: float = 1.0) -> None:
    """
    Stacked staircase fill, the step-plot counterpart of `ax.stackplot`.

    Each band k is filled between consecutive cumulative sums and held constant
    across the step it belongs to ("steps-post"): the count recorded at step t
    spans [t, t+1), so `x` must carry one extra trailing edge (see caller).
    `sign=-1.0` stacks downward for the T_unseen half.
    """
    cumulative = np.cumsum(counts, axis=1)
    lower = np.zeros(len(x))
    for k in range(counts.shape[1]):
        upper = sign * np.append(cumulative[:, k], cumulative[-1, k])
        ax.fill_between(x, lower, upper, step="post", facecolor=colors[k], alpha=0.95, linewidth=0)
        lower = upper


def plot_generalization_bidirectional(
    models: dict,
    global_actions: list,
    time_scale: str = "log",
    dpi: int = 350,
) -> plt.Figure:
    """
    Behavioural-generalization figure: for every model, plots action-type usage over
    time as a "double" stacked staircase chart. T_seen grows UPWARD from zero, T_unseen
    grows DOWNWARD from zero; both halves are labelled with positive trajectory counts.

      * Height of a half -> reachability (how many trajectories survived to step t)
      * Coloured bands   -> the action sequence / behavioural signature
      * Up vs. down       -> seen vs. unseen topology

    Counts are per-step and discrete, so the bands are drawn as steps rather than
    interpolated: the count at step t is held flat across [t, t+1).

    Parameters:
        models: {model_name: {"seen": [Trajectory, ...], "unseen": [Trajectory, ...]}}
        global_actions: ordered list of canonical actions (Enum members, strings, or ints)
                        used to build the colour/stacking order.
        time_scale: "log" (default) or "linear" for the time-step axis. A log axis is
                    undefined at 0, so it is 1-indexed (the first action sits at step 1)
                    and labelled 1/2/5 per decade; the linear axis is 0-indexed.
                    Most of the behavioural signal is in the first ~20 steps, which the
                    log axis expands and the linear axis compresses into a sliver.
        dpi: figure DPI.
    """
    if time_scale not in ("log", "linear"):
        raise ValueError(f"time_scale must be 'log' or 'linear', got {time_scale!r}")
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

    # Steps are 1-indexed on a log axis (which cannot show step 0), 0-indexed on a
    # linear one. Either way one extra trailing edge, so the staircase draws the
    # final step's tread instead of ending on its riser.
    first_step = 1 if time_scale == "log" else 0
    t_edges = np.arange(first_step, first_step + t_max + 1)
    n = len(per_model)
    fig, axes = plt.subplots(n, 1, figsize=(8.2, 1.7 * n + 1.2), sharex=True, dpi=dpi)
    if n == 1:
        axes = [axes]

    for ax, (name, (cs, cu, model_y_max)) in zip(axes, per_model.items()):
        _stacked_step_fill(ax, t_edges, cs, colors, sign=1.0)
        _stacked_step_fill(ax, t_edges, cu, colors, sign=-1.0)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-model_y_max, model_y_max)
        ax.set_xscale(time_scale)
        ax.set_xlim(t_edges[0], max(t_edges[-1], t_edges[0] + 1))
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(int(round(v)))}"))
        ax.yaxis.set_major_locator(MultipleLocator(max(10, model_y_max // 2)))
        ax.grid(axis="y", alpha=0.2)
        if time_scale == "log":
            # Plain integer step labels at 1/2/5 per decade instead of 10^n notation.
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
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
    ``utils.metrics.compute_stepwise_action_jsd``), together with its mean and AOC.

    Parameters:
        models_jsd: {model_name: {"steps": [...], "jsd_per_step": [...],
                    "mean_jsd": float, "aoc_jsd": float}} as returned by
                    ``compute_stepwise_action_jsd`` for each model.
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
        aoc_jsd = result["aoc_jsd"]
        ax.plot(steps, jsd, color="#0072B2", lw=1.1)
        # Shaded region is the trapezoidal area computed in aoc_jsd, made visible
        # so the legend's AOC figure can be eyeballed against the curve.
        ax.fill_between(steps, jsd, color="#0072B2", alpha=0.15, zorder=1)
        ax.axhline(mean_jsd, color="0.3", ls="--", lw=1.0, label=f"mean = {mean_jsd:.3f}, AOC = {aoc_jsd:.3f}")
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
    version in ``plot_action_divergence_per_step``). Each model's mean and AOC JSD
    are reported in its legend label rather than drawn on the axes, to keep the
    overlay of 8+ curves legible.

    Parameters:
        models_jsd: {model_name: {"steps": [...], "jsd_per_step": [...],
                    "mean_jsd": float, "aoc_jsd": float}} as returned by
                    ``compute_stepwise_action_jsd`` for each model.
        dpi: figure DPI.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)

    for i, (name, result) in enumerate(models_jsd.items()):
        color = _model_color(i)
        label = f"{name.replace('\n', ' ')} (mean={result['mean_jsd']:.3f}, AOC={result['aoc_jsd']:.3f})"
        ax.plot(result["steps"], result["jsd_per_step"], color=color, lw=1.2, label=label)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time step")
    ax.set_ylabel("JSD [0, 1]")
    ax.set_title("Behavioral Divergence: Seen vs. Unseen Action Distribution (JSD per step)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_jsd_vs_unseen_return(
    models_jsd: dict,
    unseen_returns: dict,
    dpi: int = 350,
    metric_key: str = "mean_jsd",
    xlabel: str = "Mean action-distribution JSD (seen vs. unseen)",
    title: str = "Behavioral Divergence vs. Unseen-Topology Performance",
) -> plt.Figure:
    """
    Scatter of each model's seen-vs-unseen action-distribution JSD aggregate against
    its mean return on unseen topologies -- one point per model, colored to match
    ``plot_action_divergence_overlay`` so the two figures cross-reference directly.

    Parameters:
        models_jsd: {model_name: {metric_key: float, ...}}, as returned by
                    ``compute_stepwise_action_jsd`` for each model.
        unseen_returns: {model_name: float}, mean total reward over that model's
                    unseen trajectories.
        dpi: figure DPI.
        metric_key: which aggregate of the per-step JSD curve to plot on the
                    x-axis -- e.g. ``"mean_jsd"`` or ``"aoc_jsd"``.
        xlabel: x-axis label, matched to ``metric_key``.
        title: figure title.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)

    texts = []
    point_xs, point_ys = [], []
    for i, (name, result) in enumerate(models_jsd.items()):
        if name not in unseen_returns:
            continue
        x = result[metric_key]
        y = unseen_returns[name]
        label = name.replace("\n", " ")
        ax.scatter(x, y, color=_model_color(i), s=60, zorder=3)
        texts.append(ax.text(x, y, label, fontsize=8, color="0.25"))
        point_xs.append(x)
        point_ys.append(y)

    # Points close together get labels shoved on top of each other by a fixed
    # offset; nudge them apart instead (no leader lines back to the point). Passing
    # the point coordinates (not just `ax`) is what makes adjust_text steer labels
    # away from the *scatter dots* themselves, not just away from other labels.
    if adjust_text is not None and texts:
        adjust_text(texts, x=point_xs, y=point_ys, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean return (unseen)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
