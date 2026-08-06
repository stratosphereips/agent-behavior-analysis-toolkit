import json, glob, os
from statistics import mean, pstdev

def load(f):
    try: return json.load(open(f))
    except Exception: return None

def arr(d, k):
    v = d.get(k); return v if isinstance(v, list) else []

for env in ['taxi', 'mountain_car', 'frozenlake8x8']:
    # one _20 (M=20 operating point) file per seed directory
    files = {}
    for f in glob.glob(f'results/{env}/random/**/*_metrics.json', recursive=True):
        if '_20_metrics.json' not in f and '/standard/' not in f.replace('\\','/'):
            # accept plain standard single-seed too (taxi)
            pass
        parent = os.path.basename(os.path.dirname(f))
        # prefer filename whose seed digits match the parent dir
        keep = files.get(parent)
        if keep is None or (parent.replace('seed','') in os.path.basename(f)):
            if '_20_metrics.json' in f or 'standard' in f:
                files[parent] = f
    if not files:
        print(f"{env}: no random files"); continue

    seff_traj_means, seff_finals, Vfinals, fp_union = [], [], [], []
    for parent, f in sorted(files.items()):
        d = load(f)
        if not d: continue
        seff = arr(d, 'state_visitation_perplexity')
        V = arr(d, 'total_nodes')
        if seff: seff_traj_means.append(mean(seff)); seff_finals.append(seff[-1])
        if V: Vfinals.append(V[-1])
        # union false-positive: fraction of pairs where ANY metric exceeds its 95th-pct floor
        rr = {k: arr(d, k+'_raw') for k in ['topological_shift','strategic_shift','3-gram_wasserstein']}
        tt = {k: arr(d, k+'_noise_threshold') for k in ['topological_shift','strategic_shift','3-gram_wasserstein']}
        n = max((len(v) for v in rr.values()), default=0)
        act = sum(1 for i in range(n) if any(i<len(rr[k]) and i<len(tt[k]) and rr[k][i]>tt[k][i] for k in rr))
        if n: fp_union.append((act, n))

    tot_act = sum(a for a,_ in fp_union); tot_n = sum(n for _,n in fp_union)
    print(f"\n=== {env}  (n={len(seff_traj_means)} seeds: {sorted(files)}) ===")
    if seff_traj_means:
        print(f"  S_eff (traj-mean)  : {mean(seff_traj_means):.1f} +/- {pstdev(seff_traj_means):.1f}")
        print(f"  S_eff (final)      : {mean(seff_finals):.1f} +/- {pstdev(seff_finals):.1f}")
    if Vfinals:
        print(f"  |V| (final)        : {mean(Vfinals):.1f} +/- {pstdev(Vfinals):.1f}")
    if tot_n:
        print(f"  FP floor (union-of-metrics, approx): {tot_act}/{tot_n} = {tot_act/tot_n:.3f}")
        print(f"    per-seed: {['%d/%d'%(a,n) for a,n in fp_union]}")
