#!/usr/bin/env python3
"""A/B the noise-floor null on MC Random: pooled vs target-only x weighted vs
unweighted strategic_shift. Self-contained: parses trajectory jsonl directly and
imports only the clean utils.metrics helpers (avoids the netsecgame import chain).

Random policy => every fire is a false positive => fire-rate IS the FPR. Goal ~5%.
"""
import argparse, glob, json, os, itertools
import numpy as np
import ot
from utils.metrics import compute_decomposed_jsd, state_js_divergence


# ---- local n-gram cost cache (reimpl of build_global_environment_cache) ----
def _lev(a, b):
    sx, sy = len(a) + 1, len(b) + 1
    m = np.zeros((sx, sy))
    for x in range(sx): m[x, 0] = x
    for y in range(sy): m[0, y] = y
    for x in range(1, sx):
        for y in range(1, sy):
            c = 0 if a[x - 1] == b[y - 1] else 1
            m[x, y] = min(m[x - 1, y] + 1, m[x - 1, y - 1] + c, m[x, y - 1] + 1)
    return m[sx - 1, sy - 1]


def ngram_cache(actions, n=3):
    g = list(itertools.product(actions, repeat=n))
    C = np.zeros((len(g), len(g)))
    for i in range(len(g)):
        for j in range(i, len(g)):
            C[i, j] = C[j, i] = _lev(g[i], g[j])
    return g, C


# ---- per-trajectory summaries, built straight from states/actions arrays ----
def summarize(states, actions, n=3):
    sc, sac = {}, {}
    for s in states:                       # every visited state (incl. terminal)
        sc[s] = sc.get(s, 0) + 1
    for i, a in enumerate(actions):
        s = states[i]
        b = sac.get(s)
        if b is None: b = {}; sac[s] = b
        b[a] = b.get(a, 0) + 1
        ns = states[i + 1]
        if ns not in sac: sac[ns] = {}
    ng = {}
    for i in range(len(actions) - n + 1):
        k = tuple(actions[i:i + n]); ng[k] = ng.get(k, 0) + 1
    return sc, sac, ng


def load_seed(seed_dir, n=3):
    """Return {cp_key: (state_counts_list, sac_list, ngram_list)} from jsonl files."""
    out = {}
    for f in sorted(glob.glob(os.path.join(seed_dir, "cp_*.jsonl"))):
        cp = os.path.splitext(os.path.basename(f))[0]
        scl, sacl, ngl = [], [], []
        for line in open(f):
            line = line.strip()
            if not line: continue
            t = json.loads(line)["trajectory"]
            sc, sac, ng = summarize(t["states"], t["actions"], n)
            scl.append(sc); sacl.append(sac); ngl.append(ng)
        out[cp] = (scl, sacl, ngl)
    return out


def merge(idx, lst):
    m = {}
    for i in idx:
        for k, v in lst[i].items(): m[k] = m.get(k, 0) + v
    return m


def merge_sa(idx, lst):
    m = {}
    for i in idx:
        for s, d in lst[i].items():
            b = m.get(s)
            if b is None: m[s] = dict(d)
            else:
                for a, c in d.items(): b[a] = b.get(a, 0) + c
    return m


def _js(c1, c2, actions):
    """Base-2 JS divergence of two action-count dicts == state_js_divergence
    (which is jensenshannon(.,.,base=2)**2). Inlined for speed at small |A|."""
    p = np.array([c1.get(a, 0) for a in actions], float)
    q = np.array([c2.get(a, 0) for a in actions], float)
    ps, qs = p.sum(), q.sum()
    if ps == 0 or qs == 0: return 1.0
    p /= ps; q /= qs; m = 0.5 * (p + q)
    mp = p > 0; mq = q > 0
    return float(0.5 * np.sum(p[mp] * np.log2(p[mp] / m[mp])) + 0.5 * np.sum(q[mq] * np.log2(q[mq] / m[mq])))


def strat_both(sc_c, sac_c, sc_p, sac_p, actions):
    """Return (weighted, unweighted) strategic_shift; per-state JSD computed once."""
    shared = set(sc_c) & set(sc_p)
    if not shared: return float("nan"), float("nan")
    W, J = [], []
    for s in shared:
        W.append((sc_c[s] + sc_p.get(s, 0)) / 2.0)
        J.append(_js(sac_c[s], sac_p[s], actions))
    W = np.array(W); J = np.array(J)
    weighted = 0.0 if W.sum() == 0 else max(0.0, float(np.sum((W / W.sum()) * J)))
    unweighted = max(0.0, float(np.mean(J)))
    return weighted, unweighted


def wass(ng_c, ng_p, g, cost, alpha=1e-6):
    v1 = np.array([ng_c.get(k, 0) for k in g], float)
    v2 = np.array([ng_p.get(k, 0) for k in g], float)
    S = len(g)
    p = (v1 + alpha) / (v1.sum() + alpha * S); q = (v2 + alpha) / (v2.sum() + alpha * S)
    p /= p.sum(); q /= q.sum()
    return float(ot.emd2(p, q, cost))


def metrics(idx_c, idx_p, scl, sacl, ngl, actions, g, cost):
    sc_c, sc_p = merge(idx_c, scl), merge(idx_p, scl)
    sac_c, sac_p = merge_sa(idx_c, sacl), merge_sa(idx_p, sacl)
    ng_c, ng_p = merge(idx_c, ngl), merge(idx_p, ngl)
    sw, su = strat_both(sc_c, sac_c, sc_p, sac_p, actions)
    return {"topo": compute_decomposed_jsd(sc_c, sc_p)["jsd_total"],
            "strat_w": sw, "strat_u": su,
            "wass": wass(ng_c, ng_p, g, cost)}


def floors(summ_c, summ_p, actions, g, cost, M, mode):
    if mode == "pooled":
        scl = summ_c[0] + summ_p[0]; sacl = summ_c[1] + summ_p[1]; ngl = summ_c[2] + summ_p[2]
    else:
        scl, sacl, ngl = summ_c
    N = len(scl); half = N // 2; idx = np.arange(N)
    keys = ["topo", "strat_w", "strat_u", "wass"]
    acc = {k: [] for k in keys}
    for _ in range(M):
        np.random.shuffle(idx)
        m = metrics(idx[half:], idx[:half], scl, sacl, ngl, actions, g, cost)
        for k in keys: acc[k].append(m[k])
    out = {}
    for k in keys:
        a = np.array(acc[k], float)
        out[k] = {"floor": float(np.nanquantile(a, 0.95, method="higher")),
                  "mu": float(np.nanmean(a)), "sd": float(np.nanstd(a))}
    for tag, sk in (("w", "strat_w"), ("u", "strat_u")):
        A = np.vstack([np.array(acc["topo"]), np.array(acc[sk]), np.array(acc["wass"])])
        Z = (A - A.mean(1, keepdims=True)) / np.where(A.std(1, keepdims=True) > 0, A.std(1, keepdims=True), np.nan)
        Z[~np.isfinite(Z)] = -np.inf
        zmax = np.max(Z, 0); zmax = zmax[np.isfinite(zmax)]
        out["zmax_" + tag] = float(np.quantile(zmax, 0.95, method="higher")) if zmax.size else float("inf")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_root", default=r"C:\Users\ondra\Documents\random_mc_runs\trajectories")
    ap.add_argument("--num_actions", type=int, default=3)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--validate", default="", help="path to a stored *_metrics.json to check observed values")
    args = ap.parse_args()
    actions = list(range(args.num_actions))
    g, cost = ngram_cache(actions, 3)

    combos = [("pooled", "w"), ("pooled", "u"), ("target", "w"), ("target", "u")]
    fire_s = {c: 0 for c in combos}; fire_fw = {c: 0 for c in combos}; npairs = 0

    for seed_dir in sorted(glob.glob(os.path.join(args.traj_root, "seed_*"))):
        summ = load_seed(seed_dir)
        cps = sorted(summ.keys(), key=lambda x: int(x.split("_")[-1]))
        # optional validation on the first seed
        if args.validate and seed_dir == sorted(glob.glob(os.path.join(args.traj_root, "seed_*")))[0]:
            ref = json.load(open(args.validate)); mt, ms, mw = [], [], []
            for prev, cur in zip(cps[:-1], cps[1:]):
                sc, sac, ng = summ[cur]; scp, sacp, ngp = summ[prev]
                nc = len(sc)
                o = metrics(list(range(nc)), list(range(nc, nc + len(scp))), sc + scp, sac + sacp, ng + ngp, actions, g, cost)
                mt.append(o["topo"]); ms.append(o["strat_w"]); mw.append(o["wass"])
            for nm, mine, key in [("topo", mt, "topological_shift_raw"), ("strat", ms, "strategic_shift_raw"), ("wass", mw, "3-gram_wasserstein_raw")]:
                d = np.nanmax(np.abs(np.array(mine) - np.array(ref[key])))
                print(f"  [validate {nm}] max|diff| vs stored = {d:.2e}")

        for prev, cur in zip(cps[:-1], cps[1:]):
            npairs += 1
            sc, sac, ng = summ[cur]; scp, sacp, ngp = summ[prev]
            nc = len(sc)
            obs = metrics(list(range(nc)), list(range(nc, nc + len(scp))), sc + scp, sac + sacp, ng + ngp, actions, g, cost)
            fl = {m: floors(summ[cur], summ[prev], actions, g, cost, args.M, m) for m in ("pooled", "target")}
            for mode, wt in combos:
                sk = "strat_w" if wt == "w" else "strat_u"; f = fl[mode]
                if not np.isnan(obs[sk]) and obs[sk] > f[sk]["floor"]: fire_s[(mode, wt)] += 1
                zs = [(obs[mk] - f[mk]["mu"]) / (f[mk]["sd"] if f[mk]["sd"] > 0 else np.nan) for mk in ("topo", sk, "wass")]
                zs = [z for z in zs if np.isfinite(z)]
                if zs and max(zs) > f["zmax_" + wt]: fire_fw[(mode, wt)] += 1

    print(f"\nMC Random  |  {npairs} pairs  |  M={args.M}   (Random => fire-rate = FPR; target ~5%)\n")
    print(f"  {'combo':24s}{'strategic_shift FPR':>22s}{'family-wise FPR':>18s}")
    for c in combos:
        nm = f"{c[0]} / {'weighted' if c[1]=='w' else 'unweighted'}"
        print(f"  {nm:24s}{100*fire_s[c]/npairs:20.1f}%{100*fire_fw[c]/npairs:16.1f}%")


if __name__ == "__main__":
    main()
