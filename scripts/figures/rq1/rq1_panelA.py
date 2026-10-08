import json, math, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=sys.argv[1]; OUT=sys.argv[2]   # ROOT = metric data root (e.g. .../metric_06102026); OUT = output png
# columns: (path, N, title)
COLS=[(os.path.join(ROOT,"mountain_car/ppo/standard/seed_5.json"),500,"MountainCar PPO: flat start"),
      (os.path.join(ROOT,"mountain_car/sarsa/standard/seed_3.json"),500,"MountainCar SARSA: convergence")]
SHIFTS=[("Topological Shift","topological_shift_raw","topological_shift_noise_threshold","#1b9e77"),
        ("Strategic Shift","strategic_shift_raw","strategic_shift_noise_threshold","#d95f02"),
        ("Sequential Shift","3-gram_wasserstein_raw","3-gram_wasserstein_noise_threshold","#7570b3")]
def load(path,N):
    d=json.load(open(path)); mean,std=d["mean_return"],d["std_return"]; C=len(mean); n=C-1; ep=d["checkpoints"]
    def fpa(i):
        zs=[]
        for _,rk,tk,_ in SHIFTS:
            b=rk.replace("_raw",""); s=d["null_std_"+b][i]
            if s and s>0: zs.append((d[rk][i]-d["null_mean_"+b][i])/s)
        return (max(zs) if zs else 0)>d["zmax_p95"][i]
    def ra(i):
        s1,s2=std[i]/math.sqrt(N),std[i+1]/math.sqrt(N); de=math.sqrt(s1*s1+s2*s2)
        return (mean[i]!=mean[i+1]) if de==0 else abs(mean[i+1]-mean[i])/de>=1.96
    fa=[fpa(i) for i in range(n)]; rr=[ra(i) for i in range(n)]
    last=max([i for i in range(n) if fa[i] or rr[i]],default=-1)
    return d,mean,std,C,n,ep,fa,rr,last+1

fig,axes=plt.subplots(2,2,figsize=(10.4,4.6),gridspec_kw={"height_ratios":[1,1]})
for col,(path,N,title) in enumerate(COLS):
    d,mean,std,C,n,ep,fa,rr,conv=load(path,N)
    axr,axs=axes[0][col],axes[1][col]
    def shade(ax):
        for i in range(n):
            if not rr[i]: ax.axvspan(ep[i],ep[i+1],color="#ededed",lw=0,zorder=0)
        if conv<n:
            ax.axvspan(ep[conv],ep[n],color="#dff0e3",lw=0,zorder=0)
            ax.axvline(ep[conv],color="#2e8b57",lw=1.3,ls="--",zorder=5)
    shade(axr)
    se=[std[i]/math.sqrt(N) for i in range(C)]
    axr.fill_between(ep,[mean[i]-std[i] for i in range(C)],[mean[i]+std[i] for i in range(C)],color="#1f4e79",alpha=0.12,zorder=2)
    axr.plot(ep,mean,color="#1f4e79",lw=2,zorder=4)
    axr.fill_between(ep,[mean[i]-se[i] for i in range(C)],[mean[i]+se[i] for i in range(C)],color="#1f4e79",alpha=0.35,zorder=3)
    axr.set_title(title,fontsize=9.5,fontweight="bold")
    if conv<n: axr.text(ep[conv]+200,axr.get_ylim()[1],"converged\n(excluded)",color="#2e8b57",fontsize=7,va="top",ha="left")
    shade(axs)
    def em(i): return 0.5*(ep[i]+ep[i+1])
    for name,rk,tk,c in SHIFTS:
        r=[(d[rk][i]/d[tk][i]) if d[tk][i]>0 else float("nan") for i in range(n)]
        axs.plot([em(i) for i in range(n)],r,color=c,lw=1.5,marker="o",ms=2.2,label=name,zorder=4)
    axs.axhline(1.0,color="#333",lw=1,ls=":",zorder=3)
    axs.set_yscale("log"); axs.set_xlim(ep[0],ep[-1]); axr.set_xlim(ep[0],ep[-1])
    axs.set_xlabel("Training episodes",fontsize=9.5)
    for a in (axr,axs):
        for s in ["top","right"]: a.spines[s].set_visible(False)
        a.tick_params(labelsize=8)
    if col==0:
        axr.set_ylabel("Mean return ($\\epsilon{=}0$)",fontsize=9,color="#1f4e79"); axr.tick_params(axis="y",labelcolor="#1f4e79")
        axs.set_ylabel("Shift / noise floor",fontsize=9)
        axs.text(ep[0]+200,1.08,"noise floor ($1\\times$)",fontsize=6.6,color="#333",va="bottom")
axes[1][1].legend(loc="upper right",frameon=False,fontsize=7,ncol=1,handlelength=1.2)
fig.legend(handles=[Patch(fc="#1f4e79",alpha=0.2,label="return $\\pm1$ s.d."),
                    Patch(fc="#ededed",label="Reward-flat ($p{>}0.05$)"),
                    Patch(fc="#dff0e3",label="Converged tail (excluded)")],
           loc="lower center",bbox_to_anchor=(0.5,0.0),ncol=3,frameon=False,fontsize=8)
plt.subplots_adjust(left=0.07,right=0.985,top=0.93,bottom=0.17,hspace=0.32,wspace=0.16)
fig.savefig(OUT,dpi=200); print("saved",OUT)
