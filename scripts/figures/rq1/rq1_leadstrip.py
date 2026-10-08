import json, glob, math, os, sys, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = sys.argv[1]; OUT = sys.argv[2]
N_ROLLOUTS = {"taxi":1000,"frozen_lake8x8":500,"mountain_car":500,"netsecgame": 500}
SEED_RE = re.compile(r"seed_\d+\.json$")
METRICS = ["topological_shift_raw","strategic_shift_raw","3-gram_wasserstein_raw"]
ENVS = [("frozen_lake8x8","FrozenLake",["q_learning","sarsa","dqn","ppo"]),
        ("mountain_car","MountainCar",["q_learning","sarsa","dqn","ppo"]),
        ("taxi","Taxi",["q_learning","sarsa","dqn"]),
        ("netsecgame","NetSecGame",["q_learning","sarsa"])]
ALAB={"q_learning":"Q-Learning","sarsa":"SARSA","dqn":"DQN","ppo":"PPO"}

def fp_active(d,i):
    zs=[]
    for m in METRICS:
        b=m.replace("_raw",""); std=d["null_std_"+b][i]
        if std and std>0: zs.append((d[m][i]-d["null_mean_"+b][i])/std)
    return (max(zs) if zs else 0)>d["zmax_p95"][i]
def ret_active(mean,std,i,N):
    s1,s2=std[i]/math.sqrt(N),std[i+1]/math.sqrt(N); den=math.sqrt(s1*s1+s2*s2)
    return (mean[i]!=mean[i+1]) if den==0 else abs(mean[i+1]-mean[i])/den>=1.96
def files(env,algo):
    base=os.path.join(ROOT,env,algo,"standard")
    return [f for f in glob.glob(os.path.join(base,"seed_*.json")) if SEED_RE.search(f)] if os.path.isdir(base) else []
def lead_run(d,N):
    n=len(d["topological_shift_raw"]); mean,std=d["mean_return"],d["std_return"]
    fp=[fp_active(d,i) for i in range(n)]; ra=[ret_active(mean,std,i,N) for i in range(n)]
    ff=next((i for i in range(n) if fp[i]),None); fr=next((i for i in range(n) if ra[i]),None)
    return None if ff is None or fr is None else (fr-ff)/n

import random
fig, axes = plt.subplots(2,2,figsize=(7.4,4.9),sharex=True)
axes=axes.ravel()
xmax=0.63; xmin=-0.09
XTICKS=[0,0.1,0.2,0.3,0.4,0.5,0.6]
for ax,(key,name,algos) in zip(axes,ENVS):
    ax.axvspan(xmin,0.0,color="#c0392b",alpha=0.05)   # return-leads region (empty: return never first)
    ax.axvspan(0.0,xmax,color="#4c9a2a",alpha=0.06)    # fingerprint-leads region
    ax.axvline(0,color="#888",lw=1.0,zorder=1)
    N=N_ROLLOUTS[key]
    yt=[]; yl=[]
    for yi,algo in enumerate(algos):
        y=len(algos)-1-yi  # first algo on top
        yt.append(y); yl.append(ALAB[algo])
        leads=[lead_run(json.load(open(f)),N) for f in files(key,algo)]
        leads=[L for L in leads if L is not None]
        random.seed(0)
        for L in leads:
            jit=(random.random()-0.5)*0.26
            ax.scatter(L,y+jit,s=27,color="#12518f",alpha=0.8,
                       edgecolor="white",linewidth=0.5,zorder=3)
    ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=8.2)
    ax.set_ylim(-0.6,len(algos)-0.4)
    ax.set_xlim(xmin,xmax)
    ax.set_xticks(XTICKS)
    ax.tick_params(axis="x",labelbottom=True,labelsize=7.4)  # x ticks+labels on every panel
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.set_title(name,fontsize=9.5,fontweight="bold",pad=4)
fig.supxlabel("Lead: training-fraction gap, first return change minus first BF activity  (positive = BF first)",fontsize=8.6,y=0.012)
plt.subplots_adjust(left=0.115,right=0.985,top=0.945,bottom=0.115,hspace=0.52,wspace=0.26)
fig.savefig(OUT,dpi=200)
print("saved",OUT)
