"""Gaming separation: single metrics (incl. REWARD, the naive monitor) vs the combined reward-blind
fingerprint. Two figures:
  FIG1  per environment, gamed vs genuine.
  FIG2  per gaming MODE, that mode vs good-learning.
Singles = the metric's late value, fixed a-priori orientation ("less-healthy = lower"), raw AUC.
Reward = normalized proxy-reward trend (gaming games the proxy, so it should NOT separate / point wrong).
Combined = grouped-CV RF over the reward-blind fingerprint features only (reward excluded)."""
import glob, os, json, hashlib
import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

ROOT = r"C:\Users\ondra\Documents\metric_results"; WASS = 3.0; EMIN = [0.05, 0.05, 0.15]
SCRATCH = os.path.dirname(__file__)

def mode_of(k):
    for a, b in [("reward_hacking_carry","carry"),("reward_hacking_breadth","breadth"),("reward_hacking_subgoal","subgoal"),
                 ("reward_hacking_roaming","roaming"),("reward_hacking","gaming"),("limited_exploration","deprivation"),
                 ("perpetual_reshaping","reshaping"),("standard","good")]:
        if a in k: return b
    if os.sep + "random" + os.sep in os.sep + k + os.sep: return "random"
    return "other"

def improved(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]; n = len(a)
    if n < 3: return None
    k = max(n // 5, 2); e = np.median(a[:3]); l = np.median(a[-k:]); nz = max(np.std(a), 0.05 * abs(l), 1e-9)
    return bool((l - e) > 1.3 * nz)

def s3(y):
    y = np.asarray(y, float); n = len(y); k = max(n // 5, 2)
    sl = float(np.polyfit(np.arange(n), y, 1)[0]) * n if n >= 3 and not np.allclose(y, y[0]) else 0.0
    return [float(np.median(y[:k])), float(np.median(y[-k:])), sl]

def auc(g, go): return float(np.mean([1.0 if a < b else 0.5 if a == b else 0.0 for a, b in product(g, go)])) if len(g) and len(go) else float("nan")

rows = []; seen = set()
for f in glob.glob(os.path.join(ROOT, "**", "*_metrics.json"), recursive=True):
    if "bins15" in f or "bins30" in f: continue
    d = json.load(open(f)); rel = os.path.relpath(f, ROOT); env = rel.split(os.sep)[0]
    h = hashlib.md5((json.dumps(d.get("mean_return")) + json.dumps(d.get("state_visitation_perplexity"))).encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    perp = d.get("state_visitation_perplexity"); nodes = d.get("total_nodes"); ret = d.get("mean_return")
    topo = d.get("topological_shift_raw"); strat = d.get("strategic_shift_raw"); seq = d.get("3-gram_wasserstein_raw")
    disc = d.get("topological_shift_discovery_raw"); aban = d.get("topological_shift_abandonment_raw"); over = d.get("topological_shift_overlap_raw")
    tt, ts, tw = d.get("topological_shift_noise_threshold"), d.get("strategic_shift_noise_threshold"), d.get("3-gram_wasserstein_noise_threshold")
    if not all([perp, nodes, ret, topo, strat, seq, disc, aban, over, tt, ts, tw]) or len(perp) < 8: continue
    ret = np.asarray(ret, float); rt = np.asarray(d.get("mean_r_true", [np.nan]), float); has = not np.all(np.isnan(rt))
    perp = np.asarray(perp, float); nodes = np.asarray(nodes, float); peak = max(np.max(nodes), 1e-9)
    topo = np.asarray(topo, float); strat = np.asarray(strat, float); seq = np.asarray(seq, float) / WASS
    xt = np.maximum(topo - np.maximum(np.asarray(tt, float), EMIN[0]), 0.0)
    xs = np.maximum(strat - np.maximum(np.asarray(ts, float), EMIN[1]), 0.0)
    xw = np.maximum(seq - np.maximum(np.asarray(tw, float) / WASS, EMIN[2] / WASS), 0.0)
    act = np.maximum.reduce([xt, xs, xw])
    disc = np.asarray(disc, float); aban = np.asarray(aban, float); over = np.asarray(over, float); s = disc + aban + over + 1e-9
    fam_ser = {"E": perp / peak, "D": nodes / peak, "C": perp / np.maximum(nodes, 1e-9),
               "act": act, "disc": disc / s, "aban": aban / s, "over": over / s}
    feats, names, late = [], [], {}
    for nm, ser in fam_ser.items():
        v = s3(ser); feats += v; names += [nm + "_e", nm + "_l", nm + "_sl"]; late[nm] = v[1]
    n = len(ret); k = max(n // 5, 2)
    late["rew"] = float((np.median(ret[-k:]) - ret[0]) / max(np.std(ret), 1e-9))   # normalized proxy-reward trend
    rows.append(dict(env=env, mode=mode_of(rel), feats=feats, late=late,
                     gamed=bool(has and improved(rt) is False),
                     genuine=bool((not has) and improved(ret) is True),
                     grp="/".join(rel.split(os.sep)[:3])))

FEATS = names
Xall = np.array([r["feats"] for r in rows]); Xs_all = StandardScaler().fit_transform(Xall)   # 21 reward-BLIND features
SINGLES = [("rew", "reward"), ("E", "coverage"), ("D", "breadth"), ("C", "concentration"), ("act", "activity"), ("disc", "discovery")]
SCOL = ["#8a6d1f", "#c9a24a", "#5aa0c0", "#b083b0", "#6aa77f", "#c77a52"]; C_FP = "#0d3b66"
grp = np.array([r["grp"] for r in rows])
# single-metric matrix (late value of each), same scaling; column order matches SINGLES (reward is column 0)
Xsingle = StandardScaler().fit_transform(np.array([[r["late"][fam] for fam, _ in SINGLES] for r in rows]))
from sklearn.model_selection import StratifiedGroupKFold as SGK, StratifiedKFold as SKF

def grpcv(Xmat, cols, pos, neg):
    """Out-of-sample AUC of an RF using `cols` to separate pos vs neg, grouped CV (robust fallback for tiny n).
    SAME protocol for a single metric (1 col) and the whole fingerprint (21 cols) -> apples-to-apples."""
    idx = list(pos) + list(neg); Xi = Xmat[np.ix_(idx, cols)]; yi = np.array([1] * len(pos) + [0] * len(neg))
    gi = grp[idx]; rf = RandomForestClassifier(n_estimators=300, random_state=0, min_samples_leaf=3)
    ns = min(5, len(set(gi[yi == 1])), int(yi.sum()))
    try:
        if ns < 2: raise ValueError
        oof = cross_val_predict(rf, Xi, yi, cv=SGK(ns, shuffle=True, random_state=0), groups=gi, method="predict_proba")[:, 1]
    except Exception:
        oof = cross_val_predict(rf, Xi, yi, cv=SKF(min(3, int(yi.sum())), shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return auc(list(oof[yi == 0]), list(oof[yi == 1]))   # RF higher = positive


def draw(groups, lab_all, combined_mask, title, out, per_group_combined=False):
    """Every bar is the same grouped-CV RF (out-of-sample), differing only in features: each single metric
    (1 column of Xsingle) vs the whole reward-blind fingerprint (21 columns of Xs_all)."""
    single_vals = {fam: [] for fam, _ in SINGLES}; comb_vals = []
    for _, pos, neg in groups:
        for si, (fam, _) in enumerate(SINGLES):
            single_vals[fam].append(grpcv(Xsingle, [si], pos, neg))
        comb_vals.append(grpcv(Xs_all, list(range(len(FEATS))), pos, neg))
    fig, ax = plt.subplots(figsize=(1.5 * len(groups) + 3.5, 4.6))
    xs_ = np.arange(len(groups)); nb = len(SINGLES) + 1; w = 0.82 / nb
    for i, (fam, labn) in enumerate(SINGLES):
        ax.bar(xs_ + (i - (nb - 1) / 2) * w, single_vals[fam], w, color=SCOL[i], label=labn)
    off = ((nb - 1) - (nb - 1) / 2) * w
    ax.bar(xs_ + off, comb_vals, w, color=C_FP, label="ALL combined", edgecolor="black", linewidth=0.6)
    for gi in range(len(groups)):
        ax.text(gi + off, comb_vals[gi] + 0.012, f"{comb_vals[gi]:.2f}", ha="center", va="bottom", fontsize=7.5, color=C_FP, fontweight="bold")
    ax.axhline(0.5, ls=":", color="#8390a0", lw=1); ax.text(len(groups) - 0.5, 0.505, "chance", fontsize=7.5, color="#5a6673", ha="right", va="bottom")
    ax.set_xticks(xs_); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUC (positive vs negative)"); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); print("wrote", out)


# GRID: one subplot per induced mode; within each, the mode vs good broken out PER ENVIRONMENT (+ pooled).
ALLMODES = ["gaming", "carry", "breadth", "subgoal", "roaming", "deprivation", "reshaping", "random"]
ENVL = [("frozen_lake8x8", "FL"), ("taxi", "Taxi"), ("mountain_car", "MC")]
good_idx = [i for i, r in enumerate(rows) if r["mode"] == "good"]
MINPOS = 3   # need at least this many mode-runs in an env to score it

fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharey=True)
for ax, mode in zip(axes.flat, ALLMODES):
    mode_idx = [i for i, r in enumerate(rows) if r["mode"] == mode]
    groups = []
    for e, elab in ENVL:
        pos = [i for i in mode_idx if rows[i]["env"] == e]; neg = [i for i in good_idx if rows[i]["env"] == e]
        if len(pos) >= MINPOS and len(neg) >= MINPOS:
            groups.append((f"{elab}\n(n={len(pos)})", pos, neg))
    groups.append((f"Pooled\n(n={len(mode_idx)})", mode_idx, good_idx))
    single_vals = {fam: [] for fam, _ in SINGLES}; comb_vals = []
    for _, pos, neg in groups:
        for si, (fam, _) in enumerate(SINGLES):
            single_vals[fam].append(grpcv(Xsingle, [si], pos, neg))
        comb_vals.append(grpcv(Xs_all, list(range(len(FEATS))), pos, neg))
    xs_ = np.arange(len(groups)); nb = len(SINGLES) + 1; w = 0.82 / nb
    for i, (fam, labn) in enumerate(SINGLES):
        ax.bar(xs_ + (i - (nb - 1) / 2) * w, single_vals[fam], w, color=SCOL[i], label=labn)
    off = ((nb - 1) - (nb - 1) / 2) * w
    ax.bar(xs_ + off, comb_vals, w, color=C_FP, label="ALL combined", edgecolor="black", linewidth=0.5)
    for gi in range(len(groups)):
        ax.text(gi + off, comb_vals[gi] + 0.015, f"{comb_vals[gi]:.2f}", ha="center", va="bottom", fontsize=7, color=C_FP, fontweight="bold")
    ax.axhline(0.5, ls=":", color="#8390a0", lw=1)
    ax.set_xticks(xs_); ax.set_xticklabels([g[0] for g in groups], fontsize=8); ax.set_ylim(0, 1.12)
    ax.set_title(f"{mode}  (vs good)", fontsize=11, fontweight="bold"); ax.spines[["top", "right"]].set_visible(False)
for ax in axes[:, 0]: ax.set_ylabel("AUC (mode vs good)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Separating each induced mode from good-learning, per environment — single metrics vs the reward-blind fingerprint", fontsize=13)
fig.tight_layout(rect=[0, 0.03, 1, 0.98])
OUT = os.path.join(SCRATCH, "mode_env_grid.png"); fig.savefig(OUT, dpi=135, bbox_inches="tight"); print("wrote", OUT)

import collections
print("per (mode,env) counts:")
for m in ALLMODES:
    print(" ", m, {e: sum(1 for r in rows if r["mode"] == m and r["env"] == e) for e, _ in ENVL})
