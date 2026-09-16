"""Per-mode (gaming variants MERGED) x per-env separation from good-learning.
Bars: single metrics (incl. reward), combined reward-blind FINGERPRINT, and FINGERPRINT+REWARD.
Produced four times: using all three temporal parts, then early-only / late-only / slope-only.
Every bar = the same grouped-CV RF, out-of-sample."""
import glob, os, json, hashlib
import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold as SGK, StratifiedKFold as SKF, cross_val_predict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

ROOT = r"C:\Users\ondra\Documents\metric_results"; WASS = 3.0; EMIN = [0.05, 0.05, 0.15]
SCRATCH = os.path.dirname(__file__)
FP_FAMS = ["E", "D", "C", "act", "disc", "aban", "over"]           # reward-blind fingerprint families
ALL_FAMS = FP_FAMS + ["rew"]
SUF = ["_e", "_l", "_sl"]
PARTS = {"all (early+late+slope)": ["_e", "_l", "_sl"], "early only": ["_e"], "late only": ["_l"], "slope only": ["_sl"]}
GAM = {"gaming", "carry", "breadth", "subgoal", "roaming"}

def mode_of(k):
    for a, b in [("reward_hacking_carry","carry"),("reward_hacking_breadth","breadth"),("reward_hacking_subgoal","subgoal"),
                 ("reward_hacking_roaming","roaming"),("reward_hacking","gaming"),("limited_exploration","deprivation"),
                 ("perpetual_reshaping","reshaping"),("standard","good")]:
        if a in k: return b
    if os.sep + "random" + os.sep in os.sep + k + os.sep: return "random"
    return "other"

def s3(y):
    y = np.asarray(y, float); n = len(y); k = max(n // 5, 2)
    sl = float(np.polyfit(np.arange(n), y, 1)[0]) * n if n >= 3 and not np.allclose(y, y[0]) else 0.0
    return [float(np.median(y[:k])), float(np.median(y[-k:])), sl]

def auc(g, go): return float(np.mean([1.0 if a < b else 0.5 if a == b else 0.0 for a, b in product(g, go)])) if len(g) and len(go) else float("nan")

# ---- pass 1: raw series + env return stats (to z-score reward per env, comparable across envs) ----
raw = []; seen = set(); env_ret = {}
for f in glob.glob(os.path.join(ROOT, "**", "*_metrics.json"), recursive=True):
    if "bins15" in f or "bins30" in f: continue
    d = json.load(open(f)); rel = os.path.relpath(f, ROOT); env = rel.split(os.sep)[0]
    h = hashlib.md5((json.dumps(d.get("mean_return")) + json.dumps(d.get("state_visitation_perplexity"))).encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    need = ["state_visitation_perplexity", "total_nodes", "mean_return", "topological_shift_raw", "strategic_shift_raw",
            "3-gram_wasserstein_raw", "topological_shift_discovery_raw", "topological_shift_abandonment_raw",
            "topological_shift_overlap_raw", "topological_shift_noise_threshold", "strategic_shift_noise_threshold", "3-gram_wasserstein_noise_threshold"]
    if not all(d.get(x) for x in need) or len(d["state_visitation_perplexity"]) < 8: continue
    raw.append((d, env, mode_of(rel), "/".join(rel.split(os.sep)[:3])))
    env_ret.setdefault(env, []).extend(np.asarray(d["mean_return"], float).tolist())
env_mu = {e: float(np.mean(v)) for e, v in env_ret.items()}; env_sd = {e: float(np.std(v)) or 1.0 for e, v in env_ret.items()}

# ---- pass 2: features ----
rows = []
for d, env, mode, grp in raw:
    perp = np.asarray(d["state_visitation_perplexity"], float); nodes = np.asarray(d["total_nodes"], float); peak = max(np.max(nodes), 1e-9)
    topo = np.asarray(d["topological_shift_raw"], float); strat = np.asarray(d["strategic_shift_raw"], float); seq = np.asarray(d["3-gram_wasserstein_raw"], float) / WASS
    xt = np.maximum(topo - np.maximum(np.asarray(d["topological_shift_noise_threshold"], float), EMIN[0]), 0.0)
    xs = np.maximum(strat - np.maximum(np.asarray(d["strategic_shift_noise_threshold"], float), EMIN[1]), 0.0)
    xw = np.maximum(seq - np.maximum(np.asarray(d["3-gram_wasserstein_noise_threshold"], float) / WASS, EMIN[2] / WASS), 0.0)
    disc = np.asarray(d["topological_shift_discovery_raw"], float); aban = np.asarray(d["topological_shift_abandonment_raw"], float); over = np.asarray(d["topological_shift_overlap_raw"], float); sm = disc + aban + over + 1e-9
    ret_z = (np.asarray(d["mean_return"], float) - env_mu[env]) / env_sd[env]
    ser = {"E": perp / peak, "D": nodes / peak, "C": perp / np.maximum(nodes, 1e-9), "act": np.maximum.reduce([xt, xs, xw]),
           "disc": disc / sm, "aban": aban / sm, "over": over / sm, "rew": ret_z}
    feat = {}
    for fam in ALL_FAMS:
        e, l, sl = s3(ser[fam]); feat[fam + "_e"] = e; feat[fam + "_l"] = l; feat[fam + "_sl"] = sl
    rows.append(dict(env=env, mode=mode, feat=feat, grp=grp))

FEATS = [fam + s for fam in ALL_FAMS for s in SUF]
X = StandardScaler().fit_transform(np.array([[r["feat"][k] for k in FEATS] for r in rows]))
grp = np.array([r["grp"] for r in rows]); envs = np.array([r["env"] for r in rows]); modes = np.array([r["mode"] for r in rows])
fidx = {k: i for i, k in enumerate(FEATS)}

def cols(fams, part): return [fidx[f + s] for f in fams for s in PARTS[part]]

def grpcv(cs, pos, neg):
    idx = list(pos) + list(neg); Xi = X[np.ix_(idx, cs)]; yi = np.array([1] * len(pos) + [0] * len(neg)); gi = grp[idx]
    rf = RandomForestClassifier(n_estimators=150, random_state=0, min_samples_leaf=3, n_jobs=-1)
    ns = min(5, len(set(gi[yi == 1])), int(yi.sum()))
    try:
        if ns < 2: raise ValueError
        oof = cross_val_predict(rf, Xi, yi, cv=SGK(ns, shuffle=True, random_state=0), groups=gi, method="predict_proba")[:, 1]
    except Exception:
        oof = cross_val_predict(rf, Xi, yi, cv=SKF(min(3, int(yi.sum())), shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return auc(list(oof[yi == 0]), list(oof[yi == 1]))

SINGLES = [("rew", "reward"), ("E", "coverage"), ("D", "breadth"), ("C", "concentration"), ("act", "activity"), ("disc", "discovery")]
SCOL = ["#8a6d1f", "#c9a24a", "#5aa0c0", "#b083b0", "#6aa77f", "#c77a52"]; C_FP = "#0d3b66"; C_FR = "#7a1f6b"
ENVL = [("frozen_lake8x8", "FL"), ("taxi", "Taxi"), ("mountain_car", "MC")]
MODES = [("gaming", GAM), ("deprivation", {"deprivation"}), ("reshaping", {"reshaping"}), ("random", {"random"})]
good_idx = [i for i, r in enumerate(rows) if r["mode"] == "good"]
MINPOS = 3

def build(part, out):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    for ax, (mname, mset) in zip(axes, MODES):
        midx = [i for i, r in enumerate(rows) if r["mode"] in mset]
        groups = []
        for e, elab in ENVL:
            pos = [i for i in midx if envs[i] == e]; neg = [i for i in good_idx if envs[i] == e]
            if len(pos) >= MINPOS and len(neg) >= MINPOS: groups.append((f"{elab}\n(n={len(pos)})", pos, neg))
        groups.append((f"Pooled\n(n={len(midx)})", midx, good_idx))
        sv = {fam: [] for fam, _ in SINGLES}; fp = []; fr = []
        for _, pos, neg in groups:
            for fam, _ in SINGLES: sv[fam].append(grpcv(cols([fam], part), pos, neg))
            fp.append(grpcv(cols(FP_FAMS, part), pos, neg))
            fr.append(grpcv(cols(ALL_FAMS, part), pos, neg))
        xs_ = np.arange(len(groups)); nb = len(SINGLES) + 2; w = 0.84 / nb
        for i, (fam, labn) in enumerate(SINGLES):
            ax.bar(xs_ + (i - (nb - 1) / 2) * w, sv[fam], w, color=SCOL[i], label=labn)
        o1 = ((nb - 2) - (nb - 1) / 2) * w; o2 = ((nb - 1) - (nb - 1) / 2) * w
        ax.bar(xs_ + o1, fp, w, color=C_FP, label="fingerprint", edgecolor="black", linewidth=0.5)
        ax.bar(xs_ + o2, fr, w, color=C_FR, label="fingerprint+reward", edgecolor="black", linewidth=0.5)
        for gi in range(len(groups)):
            ax.text(gi + o1, fp[gi] + 0.015, f"{fp[gi]:.2f}", ha="center", va="bottom", fontsize=6.5, color=C_FP, fontweight="bold")
        ax.axhline(0.5, ls=":", color="#8390a0", lw=1)
        ax.set_xticks(xs_); ax.set_xticklabels([g[0] for g in groups], fontsize=8); ax.set_ylim(0, 1.13)
        ax.set_title(f"{mname}  (vs good)", fontsize=11, fontweight="bold"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUC (mode vs good)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9.5, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Mode vs good, per environment — single metrics vs fingerprint vs fingerprint+reward   [{part}]", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97]); fig.savefig(out, dpi=135, bbox_inches="tight"); print("wrote", out)

for part, _ in [("all (early+late+slope)", 0), ("late only", 0), ("early only", 0), ("slope only", 0)]:
    tag = part.split()[0]
    build(part, os.path.join(SCRATCH, f"parts_{tag}.png"))
print("mode counts:", {m: int((modes == m).sum()) for m in set(modes)})
