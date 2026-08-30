import json, glob, os, numpy as np
import experiments.noise_null_ab as ab

EMIN = {"FL": [0.0060, 0.0353, 0.0566], "MC": [0.0034, 0.0009, 0.0172]}  # topo, strat, wass (raw units)
MET = ["topo", "strat_w", "wass"]

def load_seed_rec(seed_dir, n=3):
    """glob cp_*.jsonl recursively (handles optional hyperparam subfolder); return
    {cp: (state_counts_list, sac_list, ng_list)} + {cp: mean_return}."""
    files = sorted(glob.glob(os.path.join(seed_dir, "**", "cp_*.jsonl"), recursive=True))
    summ = {}; ret = {}
    for f in files:
        cp = os.path.splitext(os.path.basename(f))[0]
        scl, sacl, ngl, rets = [], [], [], []
        for line in open(f):
            line = line.strip()
            if not line: continue
            t = json.loads(line)["trajectory"]
            sc, sac, ng = ab.summarize(t["states"], t["actions"], n)
            scl.append(sc); sacl.append(sac); ngl.append(ng); rets.append(sum(t["rewards"]))
        summ[cp] = (scl, sacl, ngl); ret[cp] = float(np.mean(rets))
    return summ, ret

def analyze(name, root_glob, nact, emin):
    g, cost = ab.ngram_cache(list(range(nact)), 3)
    print(f"\n===== {name}  (epsilon_min: topo={emin[0]:.4f} strat={emin[1]:.4f} wass={emin[2]:.4f}) =====")
    seeddirs = sorted(glob.glob(root_glob))
    all_learn_pass = []; all_learn_pairs = 0
    for sdir in seeddirs:
        summ, ret = load_seed_rec(sdir, nact)
        cps = sorted(summ.keys(), key=lambda x: int(x.split("_")[-1]))
        retv = np.array([ret[c] for c in cps])
        rfin = retv[-1]
        # learning-phase pair i (cur=cps[i+1]) := current return not yet within 5% of final range of the run
        rng = max(retv.max() - retv.min(), 1e-9)
        raws = []; passes = []; learn = []
        for i, (prev, cur) in enumerate(zip(cps[:-1], cps[1:])):
            sc, sac, ng = summ[cur]; scp, sacp, ngp = summ[prev]; nc = len(sc)
            o = ab.metrics(list(range(nc)), list(range(nc, nc+len(scp))), sc+scp, sac+sacp, ng+ngp, list(range(nact)), g, cost)
            r = np.array([o["topo"], o["strat_w"], o["wass"]])
            raws.append(r)
            passes.append(bool((r > np.array(emin)).any()))          # effect-size gate would allow a fire
            learn.append((rfin - retv[i+1]) > 0.10*rng)               # still meaningfully below final
        raws = np.array(raws); passes = np.array(passes); learn = np.array(learn)
        lp = passes[learn]
        all_learn_pass.append(lp); all_learn_pairs += learn.sum()
        # strat magnitude during learning (the FL-critical metric)
        smax = raws[learn,1].max() if learn.any() else float('nan')
        smin = raws[learn,1].min() if learn.any() else float('nan')
        sd = os.path.basename(sdir)
        print(f"  {sd:9s} ret {retv.min():6.2f}->{retv.max():6.2f}(fin {rfin:6.2f})  learn_pairs={int(learn.sum()):2d}  "
              f"eff-pass={int(lp.sum())}/{int(learn.sum())}  strat[learn]={smin:.3f}..{smax:.3f}")
    L = np.concatenate(all_learn_pass) if all_learn_pass else np.array([])
    if L.size:
        print(f"  >>> {name}: learning-phase fires that clear epsilon_min: {int(L.sum())}/{L.size} ({100*L.mean():.0f}%)")

analyze("FL Q Good (slippery)", r"C:\Users\ondra\Documents\q_learning_fl_runs\trajectories\seed_*", 4, EMIN["FL"])
analyze("MC Q Good bins_15", r"C:\Users\ondra\Documents\q_learning_mc_runs\bins_15\trajectories\seed_*", 3, EMIN["MC"])
analyze("MC Q Good bins_30", r"C:\Users\ondra\Documents\q_learning_mc_runs\bins_30\trajectories\seed_*", 3, EMIN["MC"])
