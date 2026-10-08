"""Recompute the per-cell two-detector table (RQ1). Verifies against existing
FL/Taxi/MC rows and produces NetSecGame rows at the confirmed eval N=500.
Usage: python rq1_table.py <ROOT>"""
import json, glob, math, os, sys, re
ROOT=sys.argv[1]
N_ROLLOUTS={"taxi":1000,"frozen_lake8x8":500,"mountain_car":500,"netsecgame":500}
SEED_RE=re.compile(r"seed_\d+\.json$")
METRICS=["topological_shift_raw","strategic_shift_raw","3-gram_wasserstein_raw"]
ENVS=[("frozen_lake8x8","FrozenLake",["q_learning","sarsa","dqn","ppo","random"]),
      ("mountain_car","MountainCar",["q_learning","sarsa","dqn","ppo","random"]),
      ("taxi","Taxi",["q_learning","sarsa","dqn","random"]),
      ("netsecgame","NetSecGame",["q_learning","sarsa","random"])]
def fp_active(d,i):
    zs=[]
    for m in METRICS:
        b=m.replace("_raw",""); s=d["null_std_"+b][i]
        if s and s>0: zs.append((d[m][i]-d["null_mean_"+b][i])/s)
    return (max(zs) if zs else 0)>d["zmax_p95"][i]
def ret_active(mean,std,i,N):
    s1,s2=std[i]/math.sqrt(N),std[i+1]/math.sqrt(N); den=math.sqrt(s1*s1+s2*s2)
    return (mean[i]!=mean[i+1]) if den==0 else abs(mean[i+1]-mean[i])/den>=1.96
def files(env,algo):
    if env=="netsecgame" and algo=="random":
        return sorted(glob.glob(os.path.join(ROOT,"netsecgame","random","seed_*","trajectories","*_metrics.json")))
    base=os.path.join(ROOT,env,algo,"standard")
    return [f for f in glob.glob(os.path.join(base,"seed_*.json")) if SEED_RE.search(f)] if os.path.isdir(base) else []
def cell(env,algo):
    N=N_ROLLOUTS[env]
    fa_a=fa_n=0; fponly=conf=retonly=bs_learn=bs_settled=0; pcts=[]
    for fp in files(env,algo):
        d=json.load(open(fp)); mean,std=d["mean_return"],d["std_return"]; n=len(d["topological_shift_raw"])
        fa=[fp_active(d,i) for i in range(n)]; ra=[ret_active(mean,std,i,N) for i in range(n)]
        last=max([i for i in range(n) if fa[i] or ra[i]],default=-1)
        a=sum(fa)
        fa_a+=a; fa_n+=n; pcts.append(100*a/n if n else 0)
        for i in range(n):
            if fa[i] and not ra[i]:
                fponly+=1; conf+= 1 if any(ra[j] for j in range(i+1,n)) else 0
            elif ra[i] and not fa[i]:
                retonly+=1
            elif (not fa[i]) and (not ra[i]):
                if last>=0 and i>last: bs_settled+=1
                else: bs_learn+=1
    rng=f"{min(pcts):.0f}" if pcts and min(pcts)==max(pcts) else (f"{min(pcts):.0f}-{max(pcts):.0f}" if pcts else "-")
    return dict(fa=f"{fa_a}/{fa_n}",rng=rng,fponly=fponly,conf=conf,retonly=retonly,bsl=bs_learn,bss=bs_settled,
                nseed=len(files(env,algo)))
for key,name,algos in ENVS:
    print(f"\n== {name} ==")
    for a in algos:
        c=cell(key,a)
        flat=c['fponly']+c['bsl']  # reward-flat non-converged
        rate=f"{100*c['fponly']/flat:.0f}%" if flat else "-"
        print(f"  {a:11s} FP-active {c['fa']:>10s} ({c['rng']})  FP-only {c['conf']}/{c['fponly']:<4d} Ret-only {c['retonly']:<3d} Both(learn/settled) {c['bsl']}/{c['bss']:<4d} | flat-rate {rate} seeds={c['nseed']}")
