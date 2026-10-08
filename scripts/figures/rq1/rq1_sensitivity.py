import json, glob, math, os, sys, re, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=sys.argv[1]; OUT=sys.argv[2]
N_ROLLOUTS={"taxi":1000,"frozen_lake8x8":500,"mountain_car":500,"netsecgame": 500}
SEED_RE=re.compile(r"seed_\d+\.json$")
METRICS=["topological_shift_raw","strategic_shift_raw","3-gram_wasserstein_raw"]
ENVS=[("taxi","Taxi",["q_learning","sarsa","dqn"]),
      ("frozen_lake8x8","FrozenLake",["q_learning","sarsa","dqn","ppo"]),
      ("mountain_car","MountainCar",["q_learning","sarsa","dqn","ppo"]),
      ("netsecgame","NetSecGame",["q_learning","sarsa"])]
ALAB={"q_learning":"QL","sarsa":"SARSA","dqn":"DQN","ppo":"PPO"}
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
    if env=="netsecgame" and algo=="random":   # special layout
        return sorted(glob.glob(os.path.join(ROOT,"netsecgame","random","seed_*","trajectories","*_metrics.json")))
    base=os.path.join(ROOT,env,algo,"standard")
    return [f for f in glob.glob(os.path.join(base,"seed_*.json")) if SEED_RE.search(f)] if os.path.isdir(base) else []
def seed_counts(env,algo,excl_conv):
    """per-seed (active, count) on reward-flat non-converged pairs"""
    N=N_ROLLOUTS[env]; out=[]
    for fp in files(env,algo):
        d=json.load(open(fp)); mean,std=d["mean_return"],d["std_return"]; n=len(d["topological_shift_raw"])
        fa=[fp_active(d,i) for i in range(n)]; ra=[ret_active(mean,std,i,N) for i in range(n)]
        last=max([i for i in range(n) if fa[i] or ra[i]],default=-1)
        a=c=0
        for i in range(n):
            if ra[i]: continue
            if excl_conv and last>=0 and i>last: continue
            c+=1; a+=1 if fa[i] else 0
        if c>0: out.append((a,c))
    return out
def rate_ci(sc,B=2000):
    if not sc: return None
    tot_a=sum(a for a,_ in sc); tot_c=sum(c for _,c in sc)
    pt=100*tot_a/tot_c
    random.seed(0); bs=[]
    for _ in range(B):
        samp=[sc[random.randrange(len(sc))] for _ in sc]
        ca=sum(a for a,_ in samp); cc=sum(c for _,c in samp)
        if cc>0: bs.append(100*ca/cc)
    bs.sort()
    return pt, bs[int(0.025*len(bs))], bs[int(0.975*len(bs))]

fig,axes=plt.subplots(1,len(ENVS),figsize=(10.5,3.4),sharey=True,
                      gridspec_kw={"width_ratios":[len(a) for _,_,a in ENVS]})
PAL={"q_learning":"#0072B2","sarsa":"#E69F00","dqn":"#009E73","ppo":"#CC79A7"}  # per-model, colorblind-safe, no red
for ax,(key,name,algos) in zip(axes,ENVS):
    xs=list(range(len(algos))); heights=[]; los=[]; his=[]; labs=[]; acts=[]; tots=[]
    for a in algos:
        sc=seed_counts(key,a,True)
        r=rate_ci(sc)
        heights.append(r[0]); los.append(r[0]-r[1]); his.append(r[2]-r[0]); labs.append(ALAB[a])
        acts.append(sum(x for x,_ in sc)); tots.append(sum(c for _,c in sc))
    ax.bar(xs,heights,width=0.62,color=[PAL[a] for a in algos],yerr=[los,his],capsize=3,
           error_kw=dict(lw=1,ecolor="#333"))
    for x,h,hi in zip(xs,heights,his):
        ax.text(x,min(h+hi+3,104),f"{h:.0f}%",ha="center",va="bottom",fontsize=7.8)
    # active/total pair counts beneath each bar (x in data, y in axes fraction); models identified by the legend
    for x,a_,c in zip(xs,acts,tots):
        ax.text(x,-0.055,f"{a_}/{c}",transform=ax.get_xaxis_transform(),ha="center",va="top",
                fontsize=7.4,color="#333",clip_on=False)
    nA=len(algos)
    # right-hand gutter so the FPR label sits clear of the bars, on the line
    ax.set_xlim(-0.65, nA-0.5+1.05)
    # FPR dashed line (environment property), labelled in the gutter
    fr=rate_ci(seed_counts(key,"random",False)) if files(key,"random") else None
    if fr is not None:
        ax.axhline(fr[0],ls="--",lw=1.3,color="#c0392b")
        ax.text(nA-0.42,fr[0],f"FPR {fr[0]:.0f}%",ha="left",va="center",fontsize=7.4,color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none",alpha=0.85))
    else:
        ax.text(nA-0.42,6,"FPR\npending",ha="left",va="center",fontsize=7.0,color="#999",style="italic")
    ax.set_xticks([])  # models identified by the legend, not per-bar labels
    ax.set_title(name,fontsize=10,fontweight="bold")
    ax.set_ylim(0,108); ax.set_yticks([0,25,50,75,100])
    for s in ["top","right"]: ax.spines[s].set_visible(False)
axes[0].set_ylabel("Active on reward-flat\n(non-converged) pairs (%)",fontsize=9)
import matplotlib.patches as mpatches
handles=[mpatches.Patch(color=PAL[a],label=ALAB[a]) for a in ["q_learning","sarsa","dqn","ppo"]]
fig.legend(handles=handles,loc="upper center",ncol=4,frameon=False,fontsize=9.2,
           bbox_to_anchor=(0.5,0.995),handlelength=1.1,handleheight=1.1,columnspacing=1.8)
fig.text(0.5,0.008,"Below each bar: active / reward-flat non-converged pairs.   Dashed red line: Random-policy false-positive rate (non-learning floor).   Error bars: 95% cluster bootstrap over seeds.",
         ha="center",fontsize=7.3,color="#555")
plt.subplots_adjust(left=0.085,right=0.995,top=0.82,bottom=0.115,wspace=0.12)
fig.savefig(OUT,dpi=200); print("saved",OUT)
