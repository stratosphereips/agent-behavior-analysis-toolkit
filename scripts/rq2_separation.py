#!/usr/bin/env python3
"""RQ2 separation statistic: for each matched Reward-Gaming vs Good-Learning pair, the coverage
separation as an AUC (P(gamed coverage < good coverage) over seed pairs) with a bootstrap CI,
replacing "non-overlapping bands". Also expresses the gamed run's coverage in REFERENCE-FREE units
(fraction of its own widest reach, and of the environment's population ceiling), so the criterion
needs no trusted healthy reference. Data: metric_results.
"""
import glob, hashlib, json, os
import numpy as np
from itertools import product

ROOT = r"C:\Users\ondra\Documents\metric_results"
EXCL = ("30K", "bins15", "bins30", "_bak")
PAIRS = [("taxi", "q_learning"), ("taxi", "sarsa"), ("taxi", "dqn"),
         ("frozen_lake8x8", "q_learning"), ("frozen_lake8x8", "sarsa"),
         ("frozen_lake8x8", "dqn"), ("frozen_lake8x8", "ppo")]
ELAB = {"taxi": "Taxi", "frozen_lake8x8": "FrozenLake"}
# the coverage-collapsing gaming mode per environment (Taxi loophole vs FrozenLake MisplacedGoal/subgoal)
GAME = {"taxi": "reward_hacking", "frozen_lake8x8": "reward_hacking_subgoal"}


def runs(env, algo, mode):
    seen = set()
    for f in sorted(glob.glob(os.path.join(ROOT, env, algo, mode, "**", "*_metrics.json"), recursive=True)):
        if any(t in f for t in EXCL):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        perp = d.get("state_visitation_perplexity"); nodes = d.get("total_nodes")
        if not perp or not nodes or len(perp) < 4:
            continue
        sig = hashlib.md5((json.dumps(d.get("mean_return")) + json.dumps(perp)).encode()).hexdigest()
        if sig in seen:
            continue
        seen.add(sig)
        p = np.asarray(perp, float); nd = np.asarray(nodes, float)
        k = max(len(p) // 5, 2)
        yield dict(pp_late=float(np.median(p[-k:])), own_peak=float(np.max(nd)), nodes_max=float(np.max(nd)))


def env_ceiling(env):
    """Population ceiling: widest reach any run in the environment attains (reference-free, all modes)."""
    ceil = 1e-9
    for f in glob.glob(os.path.join(ROOT, env, "**", "*_metrics.json"), recursive=True):
        if any(t in f for t in EXCL):
            continue
        try:
            nd = json.load(open(f)).get("total_nodes")
        except Exception:
            continue
        if nd:
            ceil = max(ceil, float(np.max(np.asarray(nd, float))))
    return ceil


def auc(neg, pos):  # P(neg < pos) with ties 1/2; here neg=good, pos=gamed -> we want P(gamed<good)
    return float(np.mean([1.0 if a < b else 0.5 if a == b else 0.0 for a, b in product(pos, neg)])) if len(neg) and len(pos) else float("nan")


def boot_ci(good, gamed, B=5000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        g = rng.choice(good, len(good)); m = rng.choice(gamed, len(gamed))
        vals.append(auc(g, m))
    lo, hi = np.percentile(vals, [2.5, 97.5]); return float(lo), float(hi)


print(f"{'env':11}{'algo':11}{'good PP':>9}{'gamed PP':>10}{'cov AUC [95% CI]':>22}"
      f"{'gamed %own':>11}{'gamed %ceil':>12}{'good %ceil':>11}")
for env, algo in PAIRS:
    ceil = env_ceiling(env)
    good = list(runs(env, algo, "standard")); gamed = list(runs(env, algo, GAME[env]))
    if len(good) < 2 or len(gamed) < 2:
        print(f"{ELAB[env]:11}{algo:11}  (insufficient runs: good {len(good)}, gamed {len(gamed)})"); continue
    gp = [r["pp_late"] for r in good]; mp = [r["pp_late"] for r in gamed]
    a = auc(gp, mp); lo, hi = boot_ci(gp, mp)
    gamed_own = np.mean([r["pp_late"] / r["own_peak"] for r in gamed]) * 100
    gamed_ceil = np.mean(mp) / ceil * 100
    good_ceil = np.mean(gp) / ceil * 100
    print(f"{ELAB[env]:11}{algo:11}{np.mean(gp):9.0f}{np.mean(mp):10.0f}"
          f"{f'{a:.2f} [{lo:.2f},{hi:.2f}]':>22}{gamed_own:10.0f}%{gamed_ceil:11.0f}%{good_ceil:10.0f}%")
