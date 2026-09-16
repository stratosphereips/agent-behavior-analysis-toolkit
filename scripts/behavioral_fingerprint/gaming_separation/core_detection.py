"""Single-run, reference-free detection using the paper's CORE method: the interpretable two-axis
fingerprint (coverage PP / population ceiling, and evolutionary activity) with a leave-one-configuration-out
nearest-centroid classifier, the same machine as the RQ3 recovery analysis. No random forest, no derived
feature vector. Comparators are the paper's return baseline (level + slope + variance) and each axis alone.

For each induced mode vs Good Learning, per environment and pooled, we report the AUC with which a
detector separates a single run from a good run (good runs score the detectors, never an input).

HOW THE AUC IS OBTAINED
  1. Each run gets a scalar detector score. For a 1-D axis (coverage, activity) the score is that axis.
     For a multi-axis detector (return = level+slope+var; fingerprint = coverage+activity) the score is
     a leave-one-configuration-out nearest-centroid margin: hold out one env/agent/mode configuration,
     build the mode centroid and the good centroid from ALL OTHER configurations, and score each held-out
     run by  dist(good centroid) - dist(mode centroid)  in standardized space, so a larger score = more
     mode-like. Because a whole configuration is held out, the score is genuinely out-of-sample.
  2. AUC = the Mann-Whitney statistic P(a good run scores below an induced-mode run), ties counted as 1/2,
     over every good x mode pair. 0.50 = chance (indistinguishable); 1.00 = perfect separation;
     BELOW 0.50 = the detector is inverted (it ranks the degenerate run as the *healthier* one), which is
     exactly what the reward does under reward gaming.
  3. Uncertainty = a stratified cluster bootstrap over configurations (resample the good configs and the
     mode configs with replacement, recompute the AUC), 95% percentile interval. Configurations, not runs,
     are resampled because runs inside one configuration are not independent.
"""
import glob, os, json, hashlib
import numpy as np
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\ondra\Documents\metric_results"
GAM = {"gaming", "carry", "breadth", "subgoal", "roaming"}
NBOOT = 2000

def mode_of(k):
    for a, b in [("reward_hacking_carry","carry"),("reward_hacking_breadth","breadth"),("reward_hacking_subgoal","subgoal"),
                 ("reward_hacking_roaming","roaming"),("reward_hacking","gaming"),("limited_exploration","deprivation"),
                 ("perpetual_reshaping","reshaping"),("standard","good")]:
        if a in k: return b
    if os.sep + "random" + os.sep in os.sep + k + os.sep: return "random"
    return "other"

def auc_fast(neg, pos):
    """P(a neg/good run scores below a pos/mode run), ties = 1/2. Rank form, O(n log n)."""
    neg = np.asarray(neg, float); pos = np.asarray(pos, float); n0, n1 = len(neg), len(pos)
    if n0 == 0 or n1 == 0: return float("nan")
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))   # = P(pos > neg) with ties averaged

need = ["state_visitation_perplexity","total_nodes","mean_return","topological_shift_raw","strategic_shift_raw",
        "3-gram_wasserstein_raw","topological_shift_noise_threshold","strategic_shift_noise_threshold","3-gram_wasserstein_noise_threshold"]
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

rows = []
for d, env, mode, grp in raw:
    perp = np.asarray(d["state_visitation_perplexity"], float); ret = np.asarray(d["mean_return"], float)
    n = len(ret); k = max(n // 5, 2)
    fire = np.vstack([np.asarray(d["topological_shift_raw"], float) > np.asarray(d["topological_shift_noise_threshold"], float),
                      np.asarray(d["strategic_shift_raw"], float) > np.asarray(d["strategic_shift_noise_threshold"], float),
                      np.asarray(d["3-gram_wasserstein_raw"], float) > np.asarray(d["3-gram_wasserstein_noise_threshold"], float)]).any(0)
    coverage = float(np.median(perp[-k:])) / ENV_CEIL[env]          # PP_late / population ceiling  (core axis 1)
    activity = float(fire.mean())                                   # fraction of pairs active       (core axis 2)
    r_level = float(np.median(ret[-k:])); r_slope = float(np.polyfit(np.arange(n), ret, 1)[0]) * n; r_var = float(np.std(ret))
    rows.append(dict(env=env, mode=mode, grp=grp, coverage=coverage, activity=activity,
                     r_level=r_level, r_slope=r_slope, r_var=r_var))

# env-z-score the return features (level/slope/variance) so they are comparable when environments are pooled;
# coverage and activity are already scale-free fractions.
for key in ["r_level", "r_slope", "r_var"]:
    for env in set(r["env"] for r in rows):
        vals = [r[key] for r in rows if r["env"] == env]; mu, sd = np.mean(vals), np.std(vals) or 1.0
        for r in rows:
            if r["env"] == env: r[key + "_z"] = (r[key] - mu) / sd
grp = np.array([r["grp"] for r in rows]); envs = np.array([r["env"] for r in rows])

def feats(r, which):
    if which == "return":    return [r["r_level_z"], r["r_slope_z"], r["r_var_z"]]
    if which == "coverage":  return [r["coverage"]]
    if which == "activity":  return [r["activity"]]
    if which == "fingerprint":         return [r["coverage"], r["activity"]]
    if which == "fingerprint+return":  return [r["coverage"], r["activity"], r["r_level_z"], r["r_slope_z"], r["r_var_z"]]

def nc_scores(which, pos, neg):
    """leave-one-configuration-out nearest-centroid margin for each run (see module docstring, step 1).
    Returns (score, y, group) over the runs that could be scored (a config with only one class is skipped)."""
    idx = list(pos) + list(neg); X = np.array([feats(rows[i], which) for i in idx], float)
    y = np.array([1] * len(pos) + [0] * len(neg)); gi = grp[idx]
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0; Xs = (X - mu) / sd
    score = np.full(len(idx), np.nan)
    for g in set(gi):
        te = gi == g; tr = ~te
        if y[tr].sum() < 1 or (y[tr] == 0).sum() < 1: continue
        c_mode = Xs[tr][y[tr] == 1].mean(0); c_good = Xs[tr][y[tr] == 0].mean(0)
        score[te] = np.linalg.norm(Xs[te] - c_good, axis=1) - np.linalg.norm(Xs[te] - c_mode, axis=1)
    m = ~np.isnan(score)
    return score[m], y[m], gi[m]

def auc_ci(which, pos, neg, seed=0):
    """point AUC + 95% stratified cluster-bootstrap interval (resample configs within each class)."""
    sc, y, gi = nc_scores(which, pos, neg)
    a = auc_fast(sc[y == 0], sc[y == 1])
    gpos = np.array(sorted(set(gi[y == 1]))); gneg = np.array(sorted(set(gi[y == 0])))
    if len(gpos) < 2 or len(gneg) < 2 or np.isnan(a): return a, a, a
    by = {g: sc[gi == g] for g in set(gi)}; rng = np.random.default_rng(seed); boots = []
    for _ in range(NBOOT):
        sp = np.concatenate([by[g] for g in rng.choice(gpos, len(gpos), replace=True)])
        sn = np.concatenate([by[g] for g in rng.choice(gneg, len(gneg), replace=True)])
        boots.append(auc_fast(sn, sp))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, float(lo), float(hi)

MODES = [("gaming", GAM), ("deprivation", {"deprivation"}), ("reshaping", {"reshaping"}), ("random", {"random"})]
ENVS = [("frozen_lake8x8", "FL"), ("taxi", "Taxi"), ("mountain_car", "MC")]
BARS = [("return", "return", "#8a6d1f"), ("coverage", "coverage", "#c9a24a"), ("activity", "activity", "#5aa0c0"),
        ("fingerprint", "fingerprint (coverage+activity)", "#0d3b66"), ("fingerprint+return", "fingerprint + return", "#7a1f6b")]
good = [i for i, r in enumerate(rows) if r["mode"] == "good"]
print(f"n={len(rows)} good={len(good)}  (AUC [95% cluster-bootstrap CI])\n")
R = {}
for mname, mset in MODES:
    midx = [i for i, r in enumerate(rows) if r["mode"] in mset]; R[mname] = {}
    print(mname); print("-" * 96)
    for e, elab in ENVS + [(None, "Pooled")]:
        pos = [i for i in midx if (e is None or envs[i] == e)]; neg = [i for i in good if (e is None or envs[i] == e)]
        if len(pos) < 3 or len(neg) < 3: continue
        vals = {b[0]: auc_ci(b[0], pos, neg) for b in BARS}; vals["n"] = len(pos); R[mname][elab] = vals
        print(f"  {elab:6s} n={len(pos):3d}  " + "  ".join(f"{b[1][:4]}:{vals[b[0]][0]:.2f}[{vals[b[0]][1]:.2f},{vals[b[0]][2]:.2f}]" for b in BARS))
    print()

# ---------------- figure 1: per-mode detection vs good, with CIs and inverted-bar marking ----------------
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
for ax, (mname, _) in zip(axes, MODES):
    groups = [g for g in ["FL", "Taxi", "MC", "Pooled"] if g in R[mname]]
    xs = np.arange(len(groups)); nb = len(BARS); w = 0.82 / nb
    for i, (key, labn, col) in enumerate(BARS):
        a = np.array([R[mname][g][key][0] for g in groups])
        lo = np.array([R[mname][g][key][1] for g in groups]); hi = np.array([R[mname][g][key][2] for g in groups])
        xp = xs + (i - (nb - 1) / 2) * w
        bars = ax.bar(xp, a, w, color=col, label=labn,
                      edgecolor="black" if key.startswith("fingerprint") else "none", linewidth=0.4)
        ax.errorbar(xp, a, yerr=np.vstack([a - lo, hi - a]), fmt="none", ecolor="#2a2f36", elinewidth=0.7, capsize=1.6)
        for b, ai in zip(bars, a):                      # mark detectors that invert (rank the bad run as healthier)
            if ai < 0.5: b.set_hatch("////"); b.set_edgecolor("#b5322a"); b.set_linewidth(0.8)
    ax.axhline(0.5, ls=":", color="#8390a0", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([f"{g}\n(n={R[mname][g]['n']})" for g in groups], fontsize=8)
    ax.set_ylim(0, 1.14); ax.set_title(f"{mname} vs good", fontsize=11, fontweight="bold"); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("AUC (single run vs good)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=9.5, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.suptitle("Single-run reference-free detection with the two-axis fingerprint (nearest-centroid, leave-one-configuration-out): "
             "reward gaming is caught by coverage, not the return; the reward-visible modes by the return; the fingerprint reads both axes", fontsize=10.5)
fig.text(0.5, -0.11, "AUC = P(a good run scores below an induced-mode run), ties = 1/2, over every good x mode pair.  "
         "0.50 = chance; 1.00 = perfect separation; hatched red bars fall below 0.50, i.e. the detector inverts (ranks the "
         "degenerate run as healthier).  Whiskers = 95% cluster bootstrap over configurations.",
         ha="center", va="top", fontsize=8, color="#3a4149")
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
out = r"C:\Users\ondra\Documents\papers\Behavioral Ontogeny\figures\single_run_detection.png"
fig.savefig(out, dpi=170, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight"); print("wrote", out)

# ---------------- figure 2: pairwise mode-vs-mode separability on the two-axis fingerprint ----------------
ORDER = ["good", "gaming", "deprivation", "reshaping", "random"]
SET = {"good": {"good"}, "gaming": GAM, "deprivation": {"deprivation"}, "reshaping": {"reshaping"}, "random": {"random"}}
midx = {m: [i for i, r in enumerate(rows) if r["mode"] in SET[m]] for m in ORDER}
M = np.full((len(ORDER), len(ORDER)), np.nan)
for i, a_ in enumerate(ORDER):
    for j, b_ in enumerate(ORDER):
        if i == j or not midx[a_] or not midx[b_]: continue
        sc, y, gi = nc_scores("fingerprint", midx[a_], midx[b_])
        a = auc_fast(sc[y == 0], sc[y == 1]); M[i, j] = max(a, 1 - a)   # direction is arbitrary between two modes
print("pairwise fingerprint separability (symmetric, 0.5=indistinguishable):")
print("        " + " ".join(f"{m[:5]:>6s}" for m in ORDER))
for i, m in enumerate(ORDER):
    print(f"{m[:7]:8s}" + " ".join((f"{M[i,j]:6.2f}" if not np.isnan(M[i, j]) else "     -") for j in range(len(ORDER))))
fig2, ax2 = plt.subplots(figsize=(5.9, 5.0))
im = ax2.imshow(M, cmap="Blues", vmin=0.5, vmax=1.0)
ax2.set_xticks(range(len(ORDER))); ax2.set_yticks(range(len(ORDER)))
ax2.set_xticklabels(ORDER, rotation=25, ha="right"); ax2.set_yticklabels(ORDER)
for i in range(len(ORDER)):
    for j in range(len(ORDER)):
        if i == j: ax2.text(j, i, "\u2014", ha="center", va="center", color="#9aa4b0")
        elif not np.isnan(M[i, j]): ax2.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                                             color="white" if M[i, j] > 0.80 else "#141a20", fontsize=9.5)
ax2.set_title("Pairwise separability of behavioural modes\n(two-axis fingerprint, nearest-centroid, pooled)", fontsize=10.5)
cb = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04); cb.set_label("AUC (0.5 = indistinguishable, 1.0 = separable)", fontsize=8.5)
fig2.tight_layout()
out2 = r"C:\Users\ondra\Documents\papers\Behavioral Ontogeny\figures\mode_separability.png"
fig2.savefig(out2, dpi=170, bbox_inches="tight"); fig2.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight"); print("wrote", out2)
