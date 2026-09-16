"""Reshaping is a TEMPORAL signature, not a static one. Good Learning's evolutionary activity decays as the
policy settles; Perpetual Reshaping's activity persists. A static activity level (the fraction of all pairs
that fire, the axis the two-axis fingerprint uses) therefore barely separates reshaping from good, but the
activity TRAJECTORY (its slope over training) does. This script shows both: (a) the mean activity trajectory
for Good vs Reshaping, and (b) the reshaping-vs-good AUC when the fingerprint's activity axis is the static
level versus the trajectory slope. Same loader, population ceiling, nearest-centroid and cluster bootstrap
as core_detection.py.
"""
import glob, os, json, hashlib
import numpy as np
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\ondra\Documents\metric_results"
NBOOT = 2000; GRIDN = 25

def mode_of(k):
    for a, b in [("reward_hacking","gaming"),("limited_exploration","deprivation"),
                 ("perpetual_reshaping","reshaping"),("standard","good")]:
        if a in k: return b
    if os.sep + "random" + os.sep in os.sep + k + os.sep: return "random"
    return "other"

def auc_fast(neg, pos):
    neg = np.asarray(neg, float); pos = np.asarray(pos, float); n0, n1 = len(neg), len(pos)
    if n0 == 0 or n1 == 0: return float("nan")
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))

need = ["state_visitation_perplexity","total_nodes","topological_shift_raw","strategic_shift_raw",
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

GRID = np.linspace(0, 1, GRIDN)
rows = []
for d, env, mode, grp in raw:
    perp = np.asarray(d["state_visitation_perplexity"], float)
    fire = np.vstack([np.asarray(d["topological_shift_raw"], float) > np.asarray(d["topological_shift_noise_threshold"], float),
                      np.asarray(d["strategic_shift_raw"], float) > np.asarray(d["strategic_shift_noise_threshold"], float),
                      np.asarray(d["3-gram_wasserstein_raw"], float) > np.asarray(d["3-gram_wasserstein_noise_threshold"], float)]).any(0).astype(float)
    n = len(fire); k = max(n // 5, 2)
    coverage = float(np.median(perp[-k:])) / ENV_CEIL[env]
    a_level = float(fire.mean())                                          # static axis (paper-canonical act_any)
    a_slope = float(np.polyfit(np.arange(n), fire, 1)[0]) * n if n >= 3 else 0.0   # trajectory: <0 decays, ~0 persists
    traj = np.interp(GRID, np.linspace(0, 1, n), fire)                    # activity trajectory on a common grid
    rows.append(dict(env=env, mode=mode, grp=grp, coverage=coverage, a_level=a_level, a_slope=a_slope, traj=traj))

grp = np.array([r["grp"] for r in rows]); envs = np.array([r["env"] for r in rows])

def nc_scores(cols, pos, neg):
    idx = list(pos) + list(neg); X = np.array([[rows[i][c] for c in cols] for i in idx], float)
    y = np.array([1] * len(pos) + [0] * len(neg)); gi = grp[idx]
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0; Xs = (X - mu) / sd
    score = np.full(len(idx), np.nan)
    for g in set(gi):
        te = gi == g; tr = ~te
        if y[tr].sum() < 1 or (y[tr] == 0).sum() < 1: continue
        score[te] = np.linalg.norm(Xs[te] - Xs[tr][y[tr] == 0].mean(0), axis=1) - np.linalg.norm(Xs[te] - Xs[tr][y[tr] == 1].mean(0), axis=1)
    m = ~np.isnan(score)
    return score[m], y[m], gi[m]

def auc_ci(cols, pos, neg, seed=0):
    sc, y, gi = nc_scores(cols, pos, neg); a = auc_fast(sc[y == 0], sc[y == 1])
    gpos = np.array(sorted(set(gi[y == 1]))); gneg = np.array(sorted(set(gi[y == 0])))
    if len(gpos) < 2 or len(gneg) < 2 or np.isnan(a): return a, a, a
    by = {g: sc[gi == g] for g in set(gi)}; rng = np.random.default_rng(seed); b = []
    for _ in range(NBOOT):
        sp = np.concatenate([by[g] for g in rng.choice(gpos, len(gpos), replace=True)])
        sn = np.concatenate([by[g] for g in rng.choice(gneg, len(gneg), replace=True)])
        b.append(auc_fast(sn, sp))
    lo, hi = np.percentile(b, [2.5, 97.5]); return a, float(lo), float(hi)

good = [i for i, r in enumerate(rows) if r["mode"] == "good"]
resh = [i for i, r in enumerate(rows) if r["mode"] == "reshaping"]
ENVS = [("frozen_lake8x8", "FL"), ("taxi", "Taxi"), ("mountain_car", "MC"), (None, "Pooled")]
# the evolutionary-activity channel read two ways: as a static level (the axis the two-axis fingerprint uses)
# vs as a trajectory slope (does it decay?). Coverage is left out: reshaping stays broad, so that axis is
# uninformative here and, equally weighted, only dilutes the activity signal.
STATIC = ["a_level"]; TEMPORAL = ["a_slope"]
print("reshaping vs good  (nearest-centroid, leave-one-config-out, 95% cluster bootstrap)\n")
print(f"{'env':7s}{'activity level (static)':>26s}{'activity slope (temporal)':>28s}")
res = {}
for e, elab in ENVS:
    pos = [i for i in resh if (e is None or envs[i] == e)]; neg = [i for i in good if (e is None or envs[i] == e)]
    if len(pos) < 3 or len(neg) < 3: continue
    s = auc_ci(STATIC, pos, neg); t = auc_ci(TEMPORAL, pos, neg); res[elab] = (s, t)
    print(f"{elab:7s}   {s[0]:.2f} [{s[1]:.2f},{s[2]:.2f}]        {t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]")

# ---- figure: the mechanism, per environment (activity trajectory, Good vs Reshaping) ----
PANELS = [("frozen_lake8x8", "FrozenLake"), ("taxi", "Taxi"), ("mountain_car", "MountainCar")]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
for ax, (edir, elab) in zip(axes, PANELS):
    for idxset, col, lab in [(good, "#1a875a", "Good Learning"), (resh, "#b5322a", "Perpetual Reshaping")]:
        sel = [i for i in idxset if rows[i]["env"] == edir]
        if not sel: continue
        T = np.array([rows[i]["traj"] for i in sel]); mu = T.mean(0); se = T.std(0) / np.sqrt(len(T))
        ax.plot(GRID, mu, color=col, lw=2.2, label=f"{lab} (n={len(T)})")
        ax.fill_between(GRID, mu - se, mu + se, color=col, alpha=0.18)
    ax.set_title(elab, fontsize=11, fontweight="bold"); ax.set_xlabel("training progress (normalized)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left"); ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, 1.02)
axes[0].set_ylabel("evolutionary activity\n(fraction of channels firing)")
fig.suptitle("Perpetual Reshaping is a temporal signature: good learning's activity decays as the policy settles, reshaping's persists. "
             "The gap opens in Taxi and MountainCar; in FrozenLake the slippery dynamics keep every mode churning, so the static and temporal reads both struggle there.",
             fontsize=10, y=1.02)
fig.tight_layout()
out = r"C:\Users\ondra\Documents\papers\Behavioral Ontogeny\figures\reshaping_temporal.png"
fig.savefig(out, dpi=170, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight"); print("\nwrote", out)
