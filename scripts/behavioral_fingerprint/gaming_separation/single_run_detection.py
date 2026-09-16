"""Single-run, reference-free detection: from ONE run of unknown nature (no reachable count, no paired
good run), how well does each self-contained metric, and the whole fingerprint, separate each induced
mode from good learning? All features are computable from the run in hand; the good-run labels are used
only to SCORE the detectors (AUC), never as an input. Reported per environment and pooled.

Detectors (all single-run):
  return      reward trend (late-vs-initial, scaled by the run's own std)   -- the naive monitor
  coverage    perplexity_late / the run's own peak reach (max distinct states)
  activity    fraction of checkpoint pairs with any change channel above its noise floor
  discovery   discovery share of footprint turnover, late window
  FINGERPRINT grouped-CV random forest over all self-contained fingerprint features (reward excluded)
"""
import glob, os, json, hashlib
import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold as SGK, StratifiedKFold as SKF, cross_val_predict
import warnings; warnings.filterwarnings("ignore")

ROOT = r"C:\Users\ondra\Documents\metric_results"; WASS = 3.0; EMIN = [0.05, 0.05, 0.15]
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

need = ["state_visitation_perplexity","total_nodes","mean_return","topological_shift_raw","strategic_shift_raw",
        "3-gram_wasserstein_raw","topological_shift_discovery_raw","topological_shift_abandonment_raw",
        "topological_shift_overlap_raw","topological_shift_noise_threshold","strategic_shift_noise_threshold","3-gram_wasserstein_noise_threshold"]
# --- pass 1: collect valid runs; compute each env's POPULATION coverage ceiling (widest reach any run attains,
#     matching rq3_grid.reach_ceiling): reference-free, needs no true reachable count ---
raw = []; seen = set()
for f in glob.glob(os.path.join(ROOT, "**", "*_metrics.json"), recursive=True):
    if "bins15" in f or "bins30" in f: continue
    d = json.load(open(f)); rel = os.path.relpath(f, ROOT); env = rel.split(os.sep)[0]
    h = hashlib.md5((json.dumps(d.get("mean_return")) + json.dumps(d.get("state_visitation_perplexity"))).encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    if not all(d.get(x) for x in need) or len(d["state_visitation_perplexity"]) < 8: continue
    raw.append((d, env, mode_of(rel), "/".join(rel.split(os.sep)[:3])))
ENV_CEIL = {}
for d, env, m, g in raw:
    ENV_CEIL[env] = max(ENV_CEIL.get(env, 1e-9), float(np.max(np.asarray(d["total_nodes"], float))))

# --- pass 2: features. Coverage is normalized by the environment's population ceiling (consistent with RQ3). ---
rows = []
for d, env, mode, grp in raw:
    ceil = ENV_CEIL[env]
    perp = np.asarray(d["state_visitation_perplexity"], float); nodes = np.asarray(d["total_nodes"], float)
    ret = np.asarray(d["mean_return"], float); n = len(ret); k = max(n // 5, 2)
    topo = np.asarray(d["topological_shift_raw"], float); strat = np.asarray(d["strategic_shift_raw"], float); seq = np.asarray(d["3-gram_wasserstein_raw"], float) / WASS
    xt = np.maximum(topo - np.maximum(np.asarray(d["topological_shift_noise_threshold"], float), EMIN[0]), 0.0)
    xs = np.maximum(strat - np.maximum(np.asarray(d["strategic_shift_noise_threshold"], float), EMIN[1]), 0.0)
    xw = np.maximum(seq - np.maximum(np.asarray(d["3-gram_wasserstein_noise_threshold"], float) / WASS, EMIN[2] / WASS), 0.0)
    fire = np.vstack([np.asarray(d["topological_shift_raw"], float) > np.asarray(d["topological_shift_noise_threshold"], float),
                      strat > np.asarray(d["strategic_shift_noise_threshold"], float),
                      np.asarray(d["3-gram_wasserstein_raw"], float) > np.asarray(d["3-gram_wasserstein_noise_threshold"], float)]).any(0)
    disc = np.asarray(d["topological_shift_discovery_raw"], float); aban = np.asarray(d["topological_shift_abandonment_raw"], float); over = np.asarray(d["topological_shift_overlap_raw"], float); sm = disc + aban + over + 1e-9
    ser = {"E": perp / ceil, "D": nodes / ceil, "C": perp / np.maximum(nodes, 1e-9), "act": np.maximum.reduce([xt, xs, xw]),
           "disc": disc / sm, "aban": aban / sm, "over": over / sm}
    feat = {}
    for fam, sr in ser.items():
        e, l, sl = s3(sr); feat[fam + "_e"] = e; feat[fam + "_l"] = l; feat[fam + "_sl"] = sl
    single = {"return": float((np.median(ret[-k:]) - ret[0]) / max(np.std(ret), 1e-9)),
              "coverage": feat["E_l"], "activity": float(fire.mean()), "discovery": feat["disc_l"]}
    rows.append(dict(env=env, mode=mode, feat=feat, single=single, grp=grp))

FP_FEATS = [fam + s for fam in ["E","D","C","act","disc","aban","over"] for s in ("_e","_l","_sl")]
Xfp = StandardScaler().fit_transform(np.array([[r["feat"][k] for k in FP_FEATS] for r in rows]))
# fingerprint + return: the full detector (fingerprint carries the reward-blind case, return the reward-visible ones)
ret_col = StandardScaler().fit_transform(np.array([[r["single"]["return"]] for r in rows]))
Xfr = np.column_stack([Xfp, ret_col])
grp = np.array([r["grp"] for r in rows]); envs = np.array([r["env"] for r in rows])
SINGLES = ["return", "coverage", "activity", "discovery"]
Xsi = {s: np.array([r["single"][s] for r in rows]) for s in SINGLES}
good = [i for i, r in enumerate(rows) if r["mode"] == "good"]

def combined_auc(pos, neg, Xmat=Xfp):
    idx = list(pos) + list(neg); Xi = Xmat[idx]; yi = np.array([1] * len(pos) + [0] * len(neg)); gi = grp[idx]
    ns = min(5, len(set(gi[yi == 1])), int(yi.sum()))
    rf = RandomForestClassifier(n_estimators=300, random_state=0, min_samples_leaf=3, n_jobs=-1)
    try:
        if ns < 2: raise ValueError
        oof = cross_val_predict(rf, Xi, yi, cv=SGK(ns, shuffle=True, random_state=0), groups=gi, method="predict_proba")[:, 1]
    except Exception:
        oof = cross_val_predict(rf, Xi, yi, cv=SKF(min(3, int(yi.sum())), shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return auc(list(oof[yi == 0]), list(oof[yi == 1]))

def single_auc(name, pos, neg):
    # FIXED a-priori orientation "lower = more suspicious" (a degenerate run scores lower on each single metric):
    # this is the deployable rule, so it honestly exposes where a single metric points the wrong way for a mode.
    v = Xsi[name]
    return auc(list(v[pos]), list(v[neg]))          # P(mode < good)

MODES = [("gaming", GAM), ("deprivation", {"deprivation"}), ("reshaping", {"reshaping"}), ("random", {"random"})]
ENVS = [("frozen_lake8x8", "FL"), ("taxi", "Taxi"), ("mountain_car", "MC")]
print(f"n={len(rows)}  good={len(good)}\n")
hdr = f"{'mode':12s} {'env':6s} " + " ".join(f"{s:>10s}" for s in SINGLES) + f"{'FINGERPRINT':>13s}{'FP+RETURN':>11s}"
print(hdr); print("-" * len(hdr))
R = {}
for mname, mset in MODES:
    midx = [i for i, r in enumerate(rows) if r["mode"] in mset]
    R[mname] = {}
    for e, elab in ENVS + [(None, "Pooled")]:
        pos = [i for i in midx if (e is None or envs[i] == e)]
        neg = [i for i in good if (e is None or envs[i] == e)]
        if len(pos) < 3 or len(neg) < 3: continue
        vals = {s: single_auc(s, pos, neg) for s in SINGLES}
        vals["fingerprint"] = combined_auc(pos, neg); vals["fp_return"] = combined_auc(pos, neg, Xfr); vals["n"] = len(pos)
        R[mname][elab] = vals
        print(f"{mname:12s} {elab:6s} " + " ".join(f"{vals[s]:10.2f}" for s in SINGLES) + f"{vals['fingerprint']:13.2f}{vals['fp_return']:11.2f}   (n={len(pos)})")
    print()

# ---- figure for the paper: return vs coverage vs whole fingerprint, per mode, per env ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BARS = [("return", "return", "#8a6d1f"), ("coverage", "coverage", "#c9a24a"), ("fingerprint", "fingerprint", "#0d3b66"), ("fp_return", "fingerprint + return", "#7a1f6b")]
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
for ax, (mname, _) in zip(axes, MODES):
    groups = [g for g in ["FL", "Taxi", "MC", "Pooled"] if g in R[mname]]
    xs = np.arange(len(groups)); nb = len(BARS); w = 0.8 / nb
    for i, (key, labn, col) in enumerate(BARS):
        ax.bar(xs + (i - (nb - 1) / 2) * w, [R[mname][g][key] for g in groups], w, color=col, label=labn,
               edgecolor="black" if key == "fingerprint" else "none", linewidth=0.5)
    for gi, g in enumerate(groups):
        ax.text(gi + (nb - 1) / 2 * w, R[mname][g]["fp_return"] + 0.015, f"{R[mname][g]['fp_return']:.2f}", ha="center", va="bottom", fontsize=6.5, color="#7a1f6b", fontweight="bold")
    ax.axhline(0.5, ls=":", color="#8390a0", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([f"{g}\n(n={R[mname][g]['n']})" for g in groups], fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_title(f"{mname} vs good", fontsize=11, fontweight="bold"); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("AUC (single run vs good)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=10, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Single-run, reference-free detection (AUC vs good): reward gaming is caught by the fingerprint but not the return; the reward-visible modes are caught by the return; the two are complementary", fontsize=11)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
FIGDIR = r"C:\Users\ondra\Documents\papers\Behavioral Ontogeny\figures"
out = os.path.join(FIGDIR, "single_run_detection.png"); fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight"); print("wrote", out)
