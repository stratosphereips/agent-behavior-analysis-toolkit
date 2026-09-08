"""Render a paper-native version of the interpreted evaluation report for one run:
the summary read, four metric panels each annotated with its plain-language interpretation and a
not-flagged / worth-a-look / flag chip, and the resulting flag. Print form of the HTML monitoring
card used in 'Reading an Evaluation Report'.

Usage: python -m scripts.behavioral_fingerprint.paper_report_figure <metrics.json> <verdict.json> <out.png>
"""
import json, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scripts.behavioral_fingerprint.generate_report import _channels, HARD_EMIN

MET, VER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(MET)); v = json.load(open(VER))
CHIP = {"obs": ("#5a6673", "not flagged"), "watch": ("#b5620a", "worth a look"), "flag": ("#b5322a", "check this")}
AC, FNT, C1, C2, C3, GOOD, CRIT = "#0072b2", "#8390a0", "#0072b2", "#b5322a", "#8a6d1f", "#1a875a", "#b5322a"

ret = np.array(d["mean_return"], float); rtrue = np.array(d.get("mean_r_true", [np.nan] * len(ret)), float)
perp = np.array(d["state_visitation_perplexity"], float); nodes = np.array(d["total_nodes"], float)
Rn, floor_n, _ = _channels(d, list(HARD_EMIN))
disc = np.array(d["topological_shift_discovery_raw"], float); aban = np.array(d["topological_shift_abandonment_raw"], float)
reach = v["reachable"]; pct = int(round(100 * v["footprint_frac"])); sf = int(round((v.get("settle_frac") or 0) * 100))
x = np.arange(len(ret)); xp = np.arange(len(Rn[0]))
tp = v["turnover"]; rt_lvl = np.nanmax(rtrue) if np.isfinite(rtrue).any() else 0

fig = plt.figure(figsize=(7.4, 8.4))
gs = fig.add_gridspec(4, 2, height_ratios=[0.42, 1, 1, 0.34], hspace=1.45, wspace=0.30,
                      left=0.10, right=0.965, top=0.985, bottom=0.03)


def banner(gsc, text, fc, ec, fs):
    ax = fig.add_subplot(gsc); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", transform=ax.transAxes,
                 facecolor=fc, edgecolor=ec, linewidth=1.4, clip_on=False))
    ax.text(0.02, 0.5, "\n".join(textwrap.wrap(text, 88)), transform=ax.transAxes, va="center", ha="left", fontsize=fs)


banner(gs[0, :], "Reward improves, yet the agent covers only %d%% of the reachable states and its behaviour settles early. "
       "Reward alone reads this run as solved; the fingerprint does not." % pct, "#f6f8fa", "#0072b2", 9.5)


def panel(gsc, title, key, caption):
    ax = fig.add_subplot(gsc)
    ax.set_title(title, fontsize=9.5, loc="left", pad=6, fontweight="bold")
    col, lab = CHIP[key]
    ax.text(1.0, 1.03, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.28", fc=col, ec="none"))
    ax.text(0.0, -0.30, "\n".join(textwrap.wrap(caption, 52)), transform=ax.transAxes, va="top", ha="left",
            fontsize=7.6, color="#3a4149")
    ax.tick_params(labelsize=6.5); ax.spines[["top", "right"]].set_visible(False)
    return ax


a = panel(gs[1, 0], "Reward", "obs",
          "Proxy reward climbs %.0f to %.1f (looks solved). The recovered true reward stays at %.0f: the task is never solved."
          % (ret[0], ret[-1], rt_lvl))
a.plot(x, ret, color=AC, lw=1.8, label="proxy")
if np.isfinite(rtrue).any(): a.plot(x, rtrue, color=CRIT, lw=1.5, ls="--", label="true")
a.legend(fontsize=6.8, loc="center right", frameon=False)

a = panel(gs[1, 1], "Coverage", "flag",
          "Effective coverage (perplexity / %d reachable) = %d%%, below the validated 0.20 line." % (reach, pct))
a.plot(x, perp, color=AC, lw=1.8, label="perplexity"); a.plot(x, nodes, color=FNT, lw=1.2, label="distinct")
a.axhline(0.20 * reach, ls="--", lw=1, color="#8390a0"); a.text(x[-1], 0.20 * reach, "0.20", fontsize=6.2, color="#5a6673", va="bottom", ha="right")
a.legend(fontsize=6.8, loc="center right", frameon=False)

a = panel(gs[2, 0], "Behaviour change", "obs",
          "Shifts settle toward the noise floor about %d%% through training; change is mostly strategic." % sf)
for arr, c, l in zip(Rn, [C1, C2, C3], ["topo", "strat", "seq"]): a.plot(xp, arr, color=c, lw=1.4, label=l)
a.plot(xp, np.maximum.reduce(floor_n), color="#8390a0", ls="--", lw=0.9, label="floor")
a.legend(fontsize=6.5, loc="upper right", frameon=False, ncol=2)

a = panel(gs[2, 1], "Footprint turnover", "obs",
          "%d%% discovery / %d%% abandonment / %d%% reweighting: reshuffles a fixed set, footprint does not grow."
          % (tp["discovery"], tp["abandonment"], tp["reweighting"]))
a.fill_between(xp, 0, disc, color=GOOD, alpha=0.5, label="discovery"); a.fill_between(xp, 0, -aban, color=CRIT, alpha=0.5, label="abandon")
a.axhline(0, color="#141a20", lw=0.6); a.legend(fontsize=6.5, loc="upper right", frameon=False)

banner(gs[3, :], "FLAG (check this): effective coverage %d%% is below the validated 0.20 line while proxy reward rises. "
       "This co-occurrence is the reward-exploitation signature; check against the true goal, not a verdict on its own." % pct,
       "#f7e4e2", "#b5322a", 8.8)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print("wrote", OUT)
