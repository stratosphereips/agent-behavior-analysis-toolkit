"""Render a paper-native version of the interpreted evaluation report for one run:
the summary read, four metric panels each annotated with its plain-language interpretation and a
not-flagged / worth-a-look / check-this chip, and the resulting flag banner. Print form of the HTML
monitoring card used in 'Reading an Evaluation Report'.

Everything (chips, captions, summary, flag banner) is derived from the run's verdict.json, so a good
run and a gaming run render from the same honest logic -- no hand-written per-run narrative.

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
KINDW = {"topological": "where it goes", "strategic": "which actions it takes", "sequential": "the order it acts in"}

ret = np.array(d["mean_return"], float); rtrue = np.array(d.get("mean_r_true", [np.nan] * len(ret)), float)
perp = np.array(d["state_visitation_perplexity"], float); nodes = np.array(d["total_nodes"], float)
emin = v.get("epsilon_min", list(HARD_EMIN))
Rn, floor_n, _ = _channels(d, emin)
disc = np.array(d["topological_shift_discovery_raw"], float); aban = np.array(d["topological_shift_abandonment_raw"], float)
reach = int(v["reachable"]); pct = int(round(100 * v["footprint_frac"]))
sf = int(round((v.get("settle_frac") or 0) * 100))
x = np.arange(len(ret)); xp = np.arange(len(Rn[0]))
tp = v["turnover"]
has_true = bool(np.isfinite(rtrue).any())
rt_lvl = int(round(np.nanmedian(rtrue))) if has_true else 0
r0, rlate = ret[0], v.get("return_late", ret[-1])

# ---- verdict-driven chips ----
lip = v.get("learning_in_progress")
rew_st = "watch" if (v["trend"] == "declining" or lip) else "obs"
cov_st = "flag" if v["footprint_flag"] else ("watch" if v.get("cov_contracting") else "obs")
beh_st = "flag" if v.get("static") else ("watch" if v.get("settle_state") == "unsettled" else "obs")
ss = v.get("settle_state", "settled" if v.get("settled") else "unsettled")
kindw = KINDW.get(v.get("kind_dominant"), "several channels") if v.get("kind_clear") else "several channels"

# ---- summary line ----
rw = ("Reward improves, most of the gain early" if v.get("step") else
      "Reward improves" if v["trend"] == "rising" else
      "Reward falls" if v["trend"] == "declining" else "Reward stays flat")
if v.get("static"): sw = "and its behaviour never changes beyond the noise floor"
elif ss == "settled" and v.get("settle_idx"): sw = "and its behaviour settles about %d%% of the way through" % sf
elif ss == "settled": sw = "and its behaviour sits at the noise floor throughout"
elif ss == "converging": sw = "and its behaviour is still settling toward the noise floor"
else: sw = "and its behaviour stays unsettled"
_cov = "yet the agent covers only %d%%" % pct if v["footprint_flag"] else "the agent covers %d%%" % pct
summary = "%s, %s of the reachable states %s. " % (rw, _cov, sw)
if v["trend"] == "rising" and v["footprint_flag"]:
    summary += "Reward alone reads this run as solved; the fingerprint does not."
elif not v["flags"]:
    summary += "Reward, coverage and behaviour agree; nothing is flagged."
else:
    summary += "Nothing is flagged as a problem, but there is behaviour worth a look."

fig = plt.figure(figsize=(7.4, 8.4))
gs = fig.add_gridspec(4, 2, height_ratios=[0.42, 1, 1, 0.34], hspace=1.45, wspace=0.30,
                      left=0.10, right=0.965, top=0.985, bottom=0.03)


def banner(gsc, text, fc, ec, fs):
    ax = fig.add_subplot(gsc); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", transform=ax.transAxes,
                 facecolor=fc, edgecolor=ec, linewidth=1.4, clip_on=False))
    ax.text(0.02, 0.5, "\n".join(textwrap.wrap(text, 88)), transform=ax.transAxes, va="center", ha="left", fontsize=fs)


banner(gs[0, :], summary, "#f6f8fa", "#0072b2", 9.5)


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


# ---- Reward ----
if v["trend"] == "rising" and has_true:
    rcap = ("Proxy reward climbs %.0f to %.1f (looks solved). The recovered true reward stays near %d: "
            "the task is never solved." % (r0, rlate, rt_lvl))
elif v["trend"] == "rising":
    rcap = "Reward climbs %.0f to %.1f, above the run's own noise: the task is being learned." % (r0, rlate)
elif v["trend"] == "declining":
    rcap = "Reward falls %.0f to %.1f, by more than the run's own noise." % (r0, rlate)
else:
    rcap = "Reward barely moves; the change from start to end stays within the run's own noise."
a = panel(gs[1, 0], "Reward", rew_st, rcap)
a.plot(x, ret, color=AC, lw=1.8, label="proxy" if has_true else "reward")
if has_true:
    a.plot(x, rtrue, color=CRIT, lw=1.5, ls="--", label="true")
    a.legend(fontsize=6.8, loc="center right", frameon=False)

# ---- Coverage ----
if v["footprint_flag"]:
    ccap = "Effective coverage (perplexity / %d reachable) = %d%%, below the validated 0.20 line." % (reach, pct)
elif cov_st == "watch":
    ccap = ("Effective coverage (perplexity / %d reachable) = %d%%, at or above the 0.20 line, but the "
            "footprint is shrinking over training." % (reach, pct))
else:
    ccap = "Effective coverage (perplexity / %d reachable) = %d%%, at or above the 0.20 line." % (reach, pct)
a = panel(gs[1, 1], "Coverage", cov_st, ccap)
a.plot(x, perp, color=AC, lw=1.8, label="perplexity"); a.plot(x, nodes, color=FNT, lw=1.2, label="distinct")
a.axhline(0.20 * reach, ls="--", lw=1, color="#8390a0"); a.text(x[-1], 0.20 * reach, "0.20", fontsize=6.2, color="#5a6673", va="bottom", ha="right")
a.legend(fontsize=6.8, loc="center right", frameon=False)

# ---- Behaviour change ----
if v.get("static"): bcap = "Behaviour never changes by more than the noise floor at any checkpoint."
elif ss == "settled" and v.get("settle_idx"): bcap = "Shifts settle toward the noise floor about %d%% through training; change is mostly %s." % (sf, kindw)
elif ss == "settled": bcap = "Shifts sit at the noise floor throughout; change is mostly %s." % kindw
elif ss == "converging": bcap = "Shifts are shrinking toward the floor but had not crossed it by the end; may need more training."
else: bcap = "Shifts do not shrink toward the noise floor: behaviour stays unsettled."
a = panel(gs[2, 0], "Behaviour change", beh_st, bcap)
for arr, c, l in zip(Rn, [C1, C2, C3], ["topo", "strat", "seq"]): a.plot(xp, arr, color=c, lw=1.4, label=l)
a.plot(xp, np.maximum.reduce(floor_n), color="#8390a0", ls="--", lw=0.9, label="floor")
a.legend(fontsize=6.5, loc="upper right", frameon=False, ncol=2)

# ---- Footprint turnover ----
grow = "the visited set grows" if tp["net_grow"] else "reshuffles a fixed set, the footprint does not grow"
tcap = "%d%% discovery / %d%% abandonment / %d%% reweighting: %s." % (tp["discovery"], tp["abandonment"], tp["reweighting"], grow)
a = panel(gs[2, 1], "Footprint turnover", "obs", tcap)
a.fill_between(xp, 0, disc, color=GOOD, alpha=0.5, label="discovery"); a.fill_between(xp, 0, -aban, color=CRIT, alpha=0.5, label="abandon")
a.axhline(0, color="#141a20", lw=0.6); a.legend(fontsize=6.5, loc="upper right", frameon=False)

# ---- flag banner ----
if v["flags"]:
    f0 = v["flags"][0]
    if f0["headline"] == "Low effective coverage":
        btxt = ("FLAG (check this): effective coverage %d%% is below the validated 0.20 line while proxy reward rises. "
                "This co-occurrence is the reward-exploitation signature; check against the true goal, not a verdict on its own." % pct)
    else:
        btxt = "FLAG (%s): %s %s" % (CHIP[{"problem": "flag"}.get(f0["severity"], "watch")][1], f0["headline"], f0.get("body", ""))
    banner(gs[3, :], btxt, "#f7e4e2", "#b5322a", 8.8)
else:
    banner(gs[3, :], "No flags raised. Each panel is checked against its noise floor, and the validated reads "
           "(low coverage, no-learning, change-underway) did not fire. The rest describes the run without judging it.",
           "#e6f4ec", "#1a875a", 8.8)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print("wrote", OUT)
