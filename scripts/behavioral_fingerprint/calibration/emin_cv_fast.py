import json, glob, os, numpy as np
import scripts.behavioral_fingerprint.noise_null_ab as ab

DEC = ["topo", "strat_w", "wass"]
NAMES = {"topo": "topological_shift", "strat_w": "strategic_shift", "wass": "3-gram_wasserstein"}
WASS_MAX = 3.0
SCR = r"C:\Users\ondra\AppData\Local\Temp\claude\C--Users-ondra-Documents-papers-Behavioral-Ontogeny\501ad1ff-7e4e-4d03-a189-0f9fb8be8521\scratchpad"

def mc_per_seed():
    out = []
    for f in sorted(glob.glob(os.path.join(r"C:\Users\ondra\Documents\random_mc_runs\m_200", "**", "*_metrics.json"), recursive=True)):
        d = json.load(open(f))
        raw = np.vstack([np.array(d[NAMES[m]+"_raw"], float) for m in DEC])
        mu = np.vstack([np.array(d["null_mean_"+NAMES[m]], float) for m in DEC])
        sd = np.vstack([np.array(d["null_std_"+NAMES[m]], float) for m in DEC])
        out.append({"raw": raw, "z": (raw-mu)/np.where(sd > 0, sd, np.nan), "p95": np.array(d["zmax_p95"], float)})
    return out

def obs_raw_traj(root, nact):
    """Observed between-checkpoint raw for every Random pair (no M-resampling)."""
    g, cost = ab.ngram_cache(list(range(nact)), 3)
    cols = []
    for sd_dir in sorted(glob.glob(os.path.join(root, "seed_*"))):
        summ = ab.load_seed(sd_dir); cps = sorted(summ.keys(), key=lambda x: int(x.split("_")[-1]))
        for prev, cur in zip(cps[:-1], cps[1:]):
            sc, sac, ng = summ[cur]; scp, sacp, ngp = summ[prev]; nc = len(sc)
            o = ab.metrics(list(range(nc)), list(range(nc, nc+len(scp))), sc+scp, sac+sacp, ng+ngp, list(range(nact)), g, cost)
            cols.append([o["topo"], o["strat_w"], o["wass"]])
    return np.array(cols).T  # 3 x P

print("computing Random raw (observed-only for Taxi/FL)...")
mc = mc_per_seed()
mc_raw = np.hstack([s["raw"] for s in mc])
taxi_raw = obs_raw_traj(r"C:\Users\ondra\Documents\random_taxi_runs\trajectories", 6)
fl_raw = obs_raw_traj(r"C:\Users\ondra\Documents\random_fl_runs\trajectories", 4)
RAW = {"MountainCar": mc_raw, "Taxi": taxi_raw, "FrozenLake": fl_raw}

print("\n===== per-env, per-metric epsilon_min (99th pct of Random raw) =====")
emin = {}
for e, r in RAW.items():
    emin[e] = [np.percentile(r[m], 99) for m in range(3)]
    print(f"  {e:12s} topo={emin[e][0]:.4f}  strat={emin[e][1]:.4f}  wass={emin[e][2]:.4f}")

print("\n===== (1) MC held-out CV, per-metric gate (post-hoc, exact incl. significance) =====")
def gate(seed, em):
    sig = seed["z"] > seed["p95"][None, :]
    return (sig & (seed["raw"] > np.asarray(em)[:, None])).any(0), sig.any(0)
sonly = np.concatenate([gate(s, [0,0,0])[1] for s in mc])
ins = np.concatenate([gate(s, emin["MountainCar"])[0] for s in mc])
held = []
for i in range(len(mc)):
    tr = [mc[j]["raw"] for j in range(len(mc)) if j != i]
    em = [np.percentile(np.concatenate([t[m] for t in tr]), 99) for m in range(3)]
    held.append(gate(mc[i], em)[0])
held = np.concatenate(held)
print(f"  MC  sig-only={100*sonly.mean():.1f}%   in-sample-gated={100*ins.mean():.1f}%   HELD-OUT-gated={100*held.mean():.1f}%")

print("\n===== (2) single GLOBAL epsilon_min (normalized metrics): effect-size-only FPR per env =====")
print("  (fraction of Random pairs with ANY normalized metric > eps; upper-bounds the significance-AND-gate FPR)")
print(f"  {'eps':>6s}" + "".join(f"{e:>13s}" for e in RAW))
for eg in (0.01, 0.02, 0.03, 0.05, 0.075, 0.10):
    row = ""
    for e, r in RAW.items():
        rn = r.copy(); rn[2] = rn[2]/WASS_MAX
        row += f"{100*np.mean((rn > eg).any(0)):11.1f}% "
    print(f"  {eg:6.3f}  {row}")

# ---- (3) sensitivity on Good runs: per-env eps vs global eps=0.05 ----
def good_report(label, raw, z, p95, em_perenv, eg=0.05, ret=None):
    sig = z > p95[None, :]
    g_pe = (sig & (raw > np.asarray(em_perenv)[:, None])).any(0)
    rn = raw.copy(); rn[2] = rn[2]/WASS_MAX
    g_gl = (sig & (rn > eg)).any(0)
    print(f"\n  {label}: sig fires={int(sig.any(0).sum())}/{raw.shape[1]}  per-env-gated={int(g_pe.sum())}  global(0.05)-gated={int(g_gl.sum())}")
    if ret is not None:
        conv = np.array([abs(ret[i+1]-ret[-1]) <= 1.0 for i in range(raw.shape[1])]); learn = ~conv
        s = sig.any(0)
        print(f"    learning fires kept: per-env {int((g_pe&learn).sum())}/{int((s&learn).sum())}   global {int((g_gl&learn).sum())}/{int((s&learn).sum())}")
        print(f"    plateau fires: sig {int((s&conv).sum())} -> per-env {int((g_pe&conv).sum())}, global {int((g_gl&conv).sum())}")

# Taxi Good (19 pairs -> M=200 floors are trivial here)
gT, costT = ab.ngram_cache(list(range(6)), 3)
SUB = r"C:\Users\ondra\Documents\q_learning_taxi_runs\trajectories\seed_1\q_learning_alpha=0.1_epsilon=1.0_epsilon_decay=0.995_epsilon_min=0.01_gamma=0.99_mode=standard_q_init_val=0.0"
OLD = r"C:\Users\ondra\Documents\trajectory_analysis\agent-behavior-analysis-toolkit\results\taxi\q_learning\standard\seed1\q_learning_alpha=0.1_epsilon=1.0_epsilon_decay=0.995_epsilon_min=0.01_gamma=0.99_mode=standard_q_init_val=0.0_20_metrics.json"
summ = ab.load_seed(SUB); cps = sorted(summ.keys(), key=lambda x: int(x.split("_")[-1])); ret = json.load(open(OLD))["mean_return"]
raws = {m: [] for m in DEC}; zs = {m: [] for m in DEC}; p95 = []
for prev, cur in zip(cps[:-1], cps[1:]):
    sc, sac, ng = summ[cur]; scp, sacp, ngp = summ[prev]; nc = len(sc)
    o = ab.metrics(list(range(nc)), list(range(nc, nc+len(scp))), sc+scp, sac+sacp, ng+ngp, list(range(6)), gT, costT)
    fl = ab.floors(summ[cur], summ[prev], list(range(6)), gT, costT, 200, "pooled")
    for m in DEC: raws[m].append(o[m]); zs[m].append((o[m]-fl[m]["mu"])/(fl[m]["sd"] if fl[m]["sd"]>0 else np.nan))
    p95.append(fl["zmax_w"])
good_report("Taxi Q Good", np.vstack([raws[m] for m in DEC]), np.vstack([zs[m] for m in DEC]), np.array(p95), emin["Taxi"], ret=ret)

# MC Good (post-hoc)
mcg = json.load(open(os.path.join(SCR, "m200.json")))
rawM = np.vstack([np.array(mcg[NAMES[m]+"_raw"], float) for m in DEC])
zM = np.vstack([(np.array(mcg[NAMES[m]+"_raw"], float)-np.array(mcg["null_mean_"+NAMES[m]], float))/np.array(mcg["null_std_"+NAMES[m]], float) for m in DEC])
good_report("MC SARSA Good", rawM, zM, np.array(mcg["zmax_p95"], float), emin["MountainCar"])
