# -*- coding: utf-8 -*-
"""Behavioral-fingerprint report generator (universal, self-contained).

Two stages:
  STAGE 1 (expensive, cached):  trajectories -> <run>_metrics.json   (M split-half floors)
  STAGE 2 (cheap, re-derivable): metrics.json + epsilon_min -> verdict.json + fingerprint.png + report.html

epsilon_min (the practical-significance floor) is a STAGE-2 parameter, so switching
floors is cheap (no floor recompute):
  --floor hard        fixed universal floor 0.05 (normalized) -> [0.05,0.05,0.15] raw   (default)
  --floor estimated   per-env 99th pct of a random policy's change  (needs --random_dir)

Usage:
  python -m scripts.behavioral_fingerprint.generate_report RUN_DIR [--out DIR] [--M 200]
         [--num_actions N] [--floor hard|estimated] [--random_dir DIR] [--force]
RUN_DIR is a folder of cp_*.jsonl (searched recursively).
"""
import argparse, glob, json, os, re, base64
import numpy as np
from scipy import stats
import scripts.behavioral_fingerprint.noise_null_ab as ab
from utils.metrics import compute_decomposed_jsd, compute_perplexity_from_counts
from utils.plotting.report import plot_fingerprint_report

DEC = ["topo", "strat_w", "wass"]
NAMES = {"topo": "topological_shift", "strat_w": "strategic_shift", "wass": "3-gram_wasserstein"}
CH = {"topo": "coverage (ΔTopo)", "strat_w": "strategy (ΔStrat)", "wass": "sequence (ΔSeq)"}
WASS_MAX = 3.0
ALPHA, P0 = 0.05, 0.05
HARD_EMIN = [0.05, 0.05, WASS_MAX * 0.05]     # 0.05 normalized; wass raw is [0,3]

# ---------------- STAGE 1: metrics ----------------
def _load(run_dir, n=3):
    files = sorted(glob.glob(os.path.join(run_dir, "**", "cp_*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"no cp_*.jsonl under {run_dir}")
    summ, rets = {}, {}
    for f in files:
        cp = os.path.splitext(os.path.basename(f))[0]; scl=[]; sacl=[]; ngl=[]; rr=[]
        for line in open(f):
            line = line.strip()
            if not line: continue
            t = json.loads(line)["trajectory"]
            sc, sac, ng = ab.summarize(t["states"], t["actions"], n)
            scl.append(sc); sacl.append(sac); ngl.append(ng); rr.append(sum(t.get("rewards", [0])))
        summ[cp] = (scl, sacl, ngl); rets[cp] = rr
    return summ, rets

def infer_num_actions(run_dir):
    # scan ALL checkpoints: an early (near-random) checkpoint can miss actions that
    # only appear later, so inferring from one file undercounts the action space.
    acts = set()
    for f in glob.glob(os.path.join(run_dir, "**", "cp_*.jsonl"), recursive=True):
        for line in open(f):
            line = line.strip()
            if line: acts.update(json.loads(line)["trajectory"]["actions"])
    return max(acts) + 1

def build_metrics(run_dir, nact, M=200):
    g, cost = ab.ngram_cache(list(range(nact)), 3)
    summ, rets = _load(run_dir)
    cps = sorted(summ.keys(), key=lambda x: int(x.split("_")[-1]))
    d = {"checkpoints": [int(c.split("_")[-1]) for c in cps],
         "mean_return": [], "std_return": [], "state_visitation_perplexity": [], "total_nodes": []}
    for c in cps:
        scl, sacl, ngl = summ[c]; m = ab.merge(range(len(scl)), scl)
        d["state_visitation_perplexity"].append(compute_perplexity_from_counts(m)); d["total_nodes"].append(len(m))
        d["mean_return"].append(float(np.mean(rets[c]))); d["std_return"].append(float(np.std(rets[c])))
    keys = ["topological_shift_raw","topological_shift_overlap_raw","topological_shift_discovery_raw",
            "topological_shift_abandonment_raw","topological_shift_net_raw","strategic_shift_raw",
            "3-gram_wasserstein_raw","topological_shift_noise_threshold","strategic_shift_noise_threshold",
            "3-gram_wasserstein_noise_threshold","zmax_p95","null_mean_topological_shift","null_std_topological_shift",
            "null_mean_strategic_shift","null_std_strategic_shift","null_mean_3-gram_wasserstein","null_std_3-gram_wasserstein"]
    for k in keys: d[k] = []
    for prev, cur in zip(cps[:-1], cps[1:]):
        scl, sacl, ngl = summ[cur]; scp, sacp, ngp = summ[prev]
        scc = ab.merge(range(len(scl)), scl); scpm = ab.merge(range(len(scp)), scp)
        sacc = ab.merge_sa(range(len(sacl)), sacl); sacpm = ab.merge_sa(range(len(sacp)), sacp)
        ngc = ab.merge(range(len(ngl)), ngl); ngpm = ab.merge(range(len(ngp)), ngp)
        topo = compute_decomposed_jsd(scc, scpm); disc = 0.5*topo["p_A_unique"]; aban = 0.5*topo["p_B_unique"]
        d["topological_shift_raw"].append(topo["jsd_total"]); d["topological_shift_overlap_raw"].append(topo["jsd_overlap"])
        d["topological_shift_discovery_raw"].append(disc); d["topological_shift_abandonment_raw"].append(aban)
        d["topological_shift_net_raw"].append(disc - aban)
        d["strategic_shift_raw"].append(ab.strat_both(scc, sacc, scpm, sacpm, list(range(nact)))[0])
        d["3-gram_wasserstein_raw"].append(ab.wass(ngc, ngpm, g, cost))
        fl = ab.floors(summ[cur], summ[prev], list(range(nact)), g, cost, M, "pooled")
        d["topological_shift_noise_threshold"].append(fl["topo"]["floor"])
        d["strategic_shift_noise_threshold"].append(fl["strat_w"]["floor"])
        d["3-gram_wasserstein_noise_threshold"].append(fl["wass"]["floor"])
        d["zmax_p95"].append(fl["zmax_w"])
        for m, key in [("topo","topological_shift"),("strat_w","strategic_shift"),("wass","3-gram_wasserstein")]:
            d["null_mean_"+key].append(fl[m]["mu"]); d["null_std_"+key].append(fl[m]["sd"])
    return d

# ---------------- epsilon_min ----------------
def emin_from_random(random_dir, nact, M=200):
    """per-metric 99th pct of a random policy's raw change (observed only)."""
    g, cost = ab.ngram_cache(list(range(nact)), 3)
    cols = []
    for sd in sorted(glob.glob(os.path.join(random_dir, "seed*"))) or [random_dir]:
        summ, _ = _load(sd)
        cps = sorted(summ.keys(), key=lambda x: int(x.split("_")[-1]))
        for prev, cur in zip(cps[:-1], cps[1:]):
            sc, sac, ng = summ[cur]; scp, sacp, ngp = summ[prev]; ncur = len(sc)
            o = ab.metrics(list(range(ncur)), list(range(ncur, ncur+len(scp))), sc+scp, sac+sacp, ng+ngp, list(range(nact)), g, cost)
            cols.append([o["topo"], o["strat_w"], o["wass"]])
    a = np.array(cols)
    return [float(np.percentile(a[:, i], 99)) for i in range(3)]

# ---------------- STAGE 2: interpret -> verdict dict ----------------
def classify(d, emin):
    ret = np.array(d["mean_return"]); perp = np.array(d["state_visitation_perplexity"])
    R = np.vstack([np.array(d[NAMES[m]+"_raw"]) for m in DEC])
    mu = np.vstack([np.array(d["null_mean_"+NAMES[m]]) for m in DEC])
    sd = np.vstack([np.array(d["null_std_"+NAMES[m]]) for m in DEC])
    p95 = np.array(d["zmax_p95"]); disc = np.array(d["topological_shift_discovery_raw"]); aban = np.array(d["topological_shift_abandonment_raw"])
    Z = (R - mu) / np.where(sd > 0, sd, np.nan)
    fire = ((Z > p95[None, :]) & (R > np.asarray(emin)[:, None])).any(0)
    n = len(fire); tw = max(n//4, 3); xcp = np.arange(len(ret))
    binom = lambda k, N: stats.binomtest(int(k), int(N), P0, alternative="greater").pvalue < ALPHA
    learner = binom(fire.sum(), n); tail_active = binom(fire[-tw:].sum(), tw)
    p = (fire[:tw].sum()+fire[-tw:].sum())/max(2*tw, 1); se = np.sqrt(p*(1-p)*2/max(tw, 1))
    diminishing = se > 0 and (fire[-tw:].mean()-fire[:tw].mean())/se < -1.645
    lrr = stats.linregress(xcp, ret); ret_up = lrr.slope > 0 and lrr.pvalue/2 < ALPHA
    lrp = stats.linregress(xcp, perp); pp_eff = abs(perp[-1]-perp[0]) > np.std(perp)
    broaden = lrp.slope > 0 and lrp.pvalue/2 < ALPHA and pp_eff
    collapse = lrp.slope < 0 and lrp.pvalue/2 < ALPHA and pp_eff
    primary = DEC[int(np.argmax(R[:, fire].mean(1)))] if fire.any() else DEC[int(np.argmax(R.mean(1)))]
    tot = disc.sum()+aban.sum()+np.array(d["topological_shift_overlap_raw"]).sum()
    shares = {"redistribution": float(np.array(d["topological_shift_overlap_raw"]).sum()/tot) if tot else 0,
              "discovery": float(disc.sum()/tot) if tot else 0, "abandonment": float(aban.sum()/tot) if tot else 0}
    if not learner:                       mode, sub = "Random / No learning", ""
    elif collapse and ret_up:             mode, sub = "Reward-hacking / Gaming", "warning"
    elif ret_up:                          mode, sub = "Good learning", ("converged" if not tail_active else "converging" if diminishing else "in progress")
    elif tail_active:                     mode, sub = "Reshaping / persistent", ""
    elif collapse:                        mode, sub = "Coverage collapse (no return gain)", "warning"
    else:                                 mode, sub = "Stalled / converged-flat", ""
    conv = ("behaviorally converged" if not tail_active else
            "converging (activity winding down)" if diminishing else
            "not converged; persistent policy (stochastic env may settle here)" if ret_up else
            "not converged")
    return {
        "mode": mode, "sub_state": sub,
        "learner": bool(learner), "converged": bool(not tail_active),
        "activity_rate": float(fire.mean()), "activity_head": float(fire[:tw].mean()), "activity_tail": float(fire[-tw:].mean()),
        "return_trend": "up" if ret_up else "flat",
        "coverage_trend": "broadens" if broaden else "collapses" if collapse else "stable",
        "primary_channel": CH[primary],
        "coverage_change_makeup": shares,
        "reward_behavior": ("mismatch" if (learner and collapse and ret_up) else "consistent" if learner else "n/a"),
        "convergence": conv,
        "return": [float(ret[0]), float(ret[-1])], "perplexity": [float(perp[0]), float(perp[-1])],
        "epsilon_min": [float(x) for x in emin], "n_pairs": int(n),
    }

# ---------------- HTML report ----------------
def _b64(p):
    with open(p, "rb") as f: return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def render_html(v, fig_path, title):
    sem = {"warning": "#c26a00"}.get(v["sub_state"], "#1a875a")
    if v["mode"].startswith("Random") or v["mode"].startswith("Stalled"): sem = "#647082"
    if v["mode"].startswith("Reshaping"): sem = "#6f52c0"
    mk = v["coverage_change_makeup"]
    rows = [("Return", f"{v['return'][0]:.1f} → {v['return'][1]:.1f}", "trends up" if v["return_trend"]=="up" else "flat / plateau"),
            ("State breadth", f"PP {v['perplexity'][0]:.0f} → {v['perplexity'][1]:.0f}", v["coverage_trend"]),
            ("Activity", f"{100*v['activity_rate']:.0f}% of pairs", f"early {100*v['activity_head']:.0f}% → late {100*v['activity_tail']:.0f}%"),
            ("Dominant channel", v["primary_channel"], "largest change"),
            ("Coverage change", f"{100*mk['redistribution']:.0f}% redist · {100*mk['discovery']:.0f}% disc · {100*mk['abandonment']:.0f}% aband", "ΔTopo makeup"),
            ("Reward vs behavior", v["reward_behavior"].capitalize(), ""),
            ("Convergence", v["convergence"], "")]
    fnd = "\n".join(f'<div class="f"><div class="fl">{a}</div><div class="fv">{b}</div><div class="fn">{c}</div></div>' for a,b,c in rows)
    return f'''<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
 :root{{--g:#f2f5f8;--s:#fff;--s2:#f7f9fb;--ink:#141a20;--mut:#5a6673;--fnt:#8b97a4;--h:#e1e7ec;--ac:#0072b2}}
 @media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--g:#0f1317;--s:#171c22;--s2:#1c232b;--ink:#e7ecf1;--mut:#9aa6b3;--fnt:#6b7783;--h:#2a333d;--ac:#4ba8db}}}}
 :root[data-theme=dark]{{--g:#0f1317;--s:#171c22;--s2:#1c232b;--ink:#e7ecf1;--mut:#9aa6b3;--fnt:#6b7783;--h:#2a333d;--ac:#4ba8db}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--g);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.5}}
 .w{{max-width:860px;margin:0 auto;padding:48px 24px 72px}}
 .eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--mut);margin:0 0 12px}}
 .badge{{display:inline-block;font-weight:600;font-size:16px;padding:6px 15px;border-radius:999px;color:#fff;background:{sem}}}
 .sub{{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.08em;margin-left:10px}}
 .findings{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--h);border:1px solid var(--h);border-radius:12px;overflow:hidden;margin:22px 0}}
 .f{{background:var(--s);padding:14px 16px}} .fl{{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--fnt);margin-bottom:6px}}
 .fv{{font-size:15px;font-weight:600}} .fn{{font-size:12.5px;color:var(--mut);margin-top:3px}}
 .figscroll{{overflow-x:auto;border:1px solid var(--h);border-radius:12px;background:var(--s2);padding:10px}} .figscroll img{{width:100%;height:auto;border-radius:6px}}
</style>
<div class="w">
 <p class="eyebrow">{title}</p>
 <div><span class="badge">{v['mode']}</span>{f'<span class="sub">{v["sub_state"]}</span>' if v['sub_state'] and v['sub_state']!="warning" else ''}</div>
 <div class="findings">{fnd}</div>
 <div class="figscroll"><img src="{_b64(fig_path)}" alt="fingerprint"></div>
</div>'''

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None, help="Folder of cp_*.jsonl (Stage 1 + Stage 2). Omit when using --metrics.")
    ap.add_argument("--metrics", default=None, help="Consume an existing metrics.json (e.g. from sequential_cp_comparison) and run Stage 2 only -- no recompute. Mutually exclusive with run_dir Stage 1.")
    ap.add_argument("--out", default=None); ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--num_actions", type=int, default=None)
    ap.add_argument("--floor", choices=["hard", "estimated"], default="hard")
    ap.add_argument("--random_dir", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--metrics-only", action="store_true", help="Stage 1 only: write metrics.json and stop (floor-independent; verdict/figure/report can be derived later from the cached metrics).")
    a = ap.parse_args()

    if a.metrics:
        # CONSUME MODE: Stage 2 from a precomputed metrics.json (e.g. the canonical
        # sequential_cp_comparison output). No trajectories or recompute needed.
        d = json.load(open(a.metrics))
        # the producer names the x-axis 'checkpoint_ids'; the report figure reads 'checkpoints'
        if "checkpoints" not in d and "checkpoint_ids" in d:
            d["checkpoints"] = d["checkpoint_ids"]
        out = a.out or os.path.dirname(os.path.abspath(a.metrics))
        os.makedirs(out, exist_ok=True)
        # run name = metrics filename minus a trailing _<M>_metrics / _metrics suffix
        name = re.sub(r"(_\d+)?_metrics\.json$", "", os.path.basename(a.metrics)) or "run"
        nact = a.num_actions  # only consulted by the estimated floor
    else:
        if not a.run_dir:
            raise SystemExit("provide RUN_DIR (a folder of cp_*.jsonl) or --metrics PATH")
        out = a.out or a.run_dir
        os.makedirs(out, exist_ok=True)
        name = os.path.basename(os.path.normpath(a.run_dir))
        nact = a.num_actions or infer_num_actions(a.run_dir)
        mpath = os.path.join(out, f"{name}_metrics.json")

        # STAGE 1 (cached)
        if os.path.exists(mpath) and not a.force:
            print(f"[stage1] using cached {mpath}"); d = json.load(open(mpath))
        else:
            print(f"[stage1] computing metrics (M={a.M}, num_actions={nact}) ..."); d = build_metrics(a.run_dir, nact, a.M)
            json.dump(d, open(mpath, "w"), indent=1); print(f"[stage1] wrote {mpath}")

        if a.metrics_only:
            print("[stage1] metrics-only: done (verdict/figure/report deferred)"); return

    # STAGE 2 (cheap)
    if a.floor == "estimated":
        if not a.random_dir: raise SystemExit("--floor estimated needs --random_dir")
        if not nact: raise SystemExit("--floor estimated needs --num_actions (cannot be inferred in --metrics mode)")
        emin = emin_from_random(a.random_dir, nact, a.M); print(f"[stage2] estimated epsilon_min = {emin}")
    else:
        emin = HARD_EMIN; print(f"[stage2] hard epsilon_min = {emin}")
    v = classify(d, emin)
    v["run"] = name; v["floor"] = a.floor
    json.dump(v, open(os.path.join(out, f"{name}_verdict.json"), "w"), indent=1)
    figp = os.path.join(out, f"{name}_fingerprint.png")
    plot_fingerprint_report(d, name, figp, dpi=170, emin=emin)
    open(os.path.join(out, f"{name}_report.html"), "w", encoding="utf-8").write(render_html(v, figp, name))
    print(f"[stage2] VERDICT: {v['mode']} {('('+v['sub_state']+')') if v['sub_state'] and v['sub_state']!='warning' else ''}")
    print(f"[stage2] wrote {name}_verdict.json, {name}_fingerprint.png, {name}_report.html in {out}")

if __name__ == "__main__":
    main()
