# -*- coding: utf-8 -*-
"""Behavioral-fingerprint report generator (universal, self-contained).

Two stages:
  STAGE 1 (expensive, cached):  trajectories -> <run>_metrics.json   (M split-half floors)
  STAGE 2 (cheap, re-derivable): metrics.json + epsilon_min -> verdict.json + report.html + report.svg

STAGE 2 is a reward-blind behavioral PORTRAIT with four panels -- reward, state coverage,
behavioural change, footprint turnover -- each carrying a status chip (NOT FLAGGED / WORTH A LOOK /
CHECK THIS), plus validated red flags and a cross-panel interpretation. Gaming is a footprint flag
(effective coverage < tau=0.20 of the run's OWN peak reach -- perplexity late vs max(total_nodes),
inferred from the trajectories; validated vs true reward: pooled AUC 0.89), never a verdict.
Self-contained: the same line applies to every environment, no external reachable-state count
needed (--reachable, if given, is kept for reference only).

Code layout (top -> bottom): number formatting helpers; STAGE 1 metric builders; the interpret()
verdict; the PANEL NARRATIVES section (_panel_reward/_coverage/_behaviour/_turnover) which turns the
verdict into prose ONCE; the SVG/HTML chart primitives; the two report renderers (render_report ->
HTML, render_report_svg -> flat Miro-safe SVG) which both consume the shared panel narratives; the CLI.

epsilon_min (the practical-significance floor) is a STAGE-2 parameter, so switching
floors is cheap (no floor recompute):
  --floor hard        fixed universal floor 0.05 (normalized) -> [0.05,0.05,0.15] raw   (default)
  --floor estimated   max(hard, per-env 99th pct of a random policy's change)  (needs --random_dir).
                      The max() keeps the hard floor as a universal minimum and only tightens
                      it where the env's random baseline is genuinely noisier -- so a near-
                      stationary random policy can't collapse the floor below meaningful change.

Usage:
  python -m scripts.behavioral_fingerprint.generate_report RUN_DIR [--out DIR] [--M 200]
         [--num_actions N] [--floor hard|estimated] [--random_dir DIR] [--force]
RUN_DIR is a folder of cp_*.jsonl (searched recursively).
"""
import argparse, glob, json, os, re, base64, functools, html
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
def fnum(v):  # human-readable number: no scientific notation
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 1 else f"{v:.3f}"

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
    all_states = set()
    for c in cps:
        scl, sacl, ngl = summ[c]; m = ab.merge(range(len(scl)), scl)
        d["state_visitation_perplexity"].append(compute_perplexity_from_counts(m)); d["total_nodes"].append(len(m))
        d["mean_return"].append(float(np.mean(rets[c]))); d["std_return"].append(float(np.std(rets[c])))
        all_states.update(m.keys())
    # union of per-checkpoint state sets: distinct states visited across the whole run,
    # not just within a single checkpoint (that's total_nodes).
    d["total_distinct_states_visited"] = len(all_states)
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

# ---------------- HTML report ----------------
def _b64(p):
    with open(p, "rb") as f: return "data:image/png;base64," + base64.b64encode(f.read()).decode()

# ---------------- redesigned portrait (axes + flags + severity) ----------------
TAU_LOW = 0.20     # coverage flag: perp_late < 0.20 * peak reach = collapsed. The Youden-optimal point on the
                   # gamed-vs-good ROC (AUC 0.887) is E*=0.200; 5-fold CV gives 0.198 +/- 0.009 with out-of-sample
                   # recall 0.73 / precision 0.80 -- a cross-validated operating point, not an eyeballed round number.
# Coverage is measured against the run's OWN widest reach (peak distinct-state count, inferred from the
# trajectories) -- no external reachable-state count is assumed, and the same TAU_LOW line applies to every
# environment. perp_late / peak < 0.20 = collapsed onto a fraction of the territory it once explored. Pooled
# gamed-vs-good AUC 0.89 (Taxi 1.00, FrozenLake 0.84); genuine runs keep a median 40% of their peak, gaming 9%.
# Immune to the self-reference flattening that sinks the peak/union AS A DENOMINATOR of the unknown world (0.71-0.78):
# collapse is measured against the agent itself. The per-checkpoint peak beats the full-run union (0.887 vs 0.780)
# because the union's small excess over the peak tracks healthy late discovery, penalizing exactly the good runs.

def _fire_trend(fire):
    """Logistic fit of fire ~ checkpoint index -- window-free stationarity. Returns the
    slope sign, an LR-test p for a monotone trend, and the model-fitted fire rate at the
    first and last checkpoint."""
    y = np.asarray(fire, float); n = len(y); x = np.arange(n); eps = 1e-9
    ybar = float(y.mean())
    if n < 3 or ybar <= 0.0 or ybar >= 1.0:
        return {"slope": 0.0, "p_trend": 1.0, "p_start": ybar, "p_end": ybar}
    ll_null = float((y*np.log(ybar+eps) + (1-y)*np.log(1-ybar+eps)).sum())
    from scipy.optimize import minimize
    from scipy.special import expit           # overflow-safe logistic sigmoid
    def nll(pr):
        a, b = pr; p = np.clip(expit(a + b*x), eps, 1-eps)
        return -float((y*np.log(p) + (1-y)*np.log(1-p)).sum())
    res = minimize(nll, [np.log((ybar+eps)/(1-ybar+eps)), 0.0], method="Nelder-Mead")
    a, b = res.x; lr = max(2.0*(-res.fun - ll_null), 0.0)
    return {"slope": float(b), "p_trend": float(stats.chi2.sf(lr, 1)),
            "p_start": float(expit(a)), "p_end": float(expit(a + b*(n-1)))}

def _theil(y):
    """Robust trend: Theil-Sen slope + its 95% CI. Slope sign gives direction; the CI is a
    light significance check. No Mann-Kendall p (its independence assumption is violated by
    autocorrelated training curves) -- significance of a trend is carried by the magnitude gate."""
    y = np.asarray(y, float); x = np.arange(len(y))
    if len(y) < 3 or np.allclose(y, y[0]): return 0.0, 0.0, 0.0
    ts = stats.theilslopes(y, x)
    return float(ts[0]), float(ts[2]), float(ts[3])

def settling_point(fire, tol=0.15):
    """Earliest index from which the shift-firing rate over the remaining checkpoints is <= tol
    (tolerates occasional late blips). None = never settles."""
    fire = np.asarray(fire, bool)
    for i in range(len(fire)):
        if fire[i:].mean() <= tol: return i
    return None

def _channels(d, emin):
    """Per-channel shift series and effective floor, normalized to [0,1] (wass raw is [0,WASS_MAX]).
    Effective floor = max(permutation floor, practical emin) -- exactly the toolkit's `fire` gate:
    fire <=> R > mu + p95*sd (Z>p95) AND R > emin. Returns (Rn, floor_n, fire)."""
    scale = [1.0, 1.0, WASS_MAX]
    Rn, floor_n = [], []
    for i, m in enumerate(DEC):
        R = np.array(d[NAMES[m] + "_raw"], float) / scale[i]
        key = NAMES[m] + "_noise_threshold"
        if key in d:
            perm = np.array(d[key], float) / scale[i]
        else:
            perm = (np.array(d["null_mean_" + NAMES[m]], float) + np.array(d["zmax_p95"], float) * np.array(d["null_std_" + NAMES[m]], float)) / scale[i]
        Rn.append(R); floor_n.append(np.maximum(perm, emin[i] / scale[i]))
    fire = np.vstack([Rn[i] > floor_n[i] for i in range(3)]).any(0)
    return Rn, floor_n, fire

KKEY = ["topological", "strategic", "sequential"]
KIND = ["where it goes", "the actions it takes", "the order it acts in"]

def interpret(d, emin, reachable=None, tau=TAU_LOW, n_eval=None):
    """Reward-blind behavioral interpretation. Describes each fingerprint family from its own trend,
    referenced to the noise floor. Coverage is fully self-contained: effective coverage (perplexity) is
    measured against the run's OWN widest reach (peak distinct-state count, inferred from the trajectories),
    with a red flag below tau -- no external reachable-state count is assumed, so the same line applies to
    every environment. A `reachable` argument, if given, is retained in the verdict for reference only and
    does not drive the flag. Everything else is observed or a process-failure flag. Returns a numbers-only
    verdict dict (prose is composed at render time)."""
    if len(d.get("topological_shift_raw", [])) < 1:
        rr = d.get("mean_return") or [0.0]; pp = d.get("state_visitation_perplexity") or [0.0]
        return {"n_pairs": 0, "return": [float(rr[0]), float(rr[-1])], "perplexity": [float(pp[0]), float(pp[-1])],
                "epsilon_min": [float(x) for x in emin], "reachable": reachable, "footprint_frac": None,
                "insufficient": True, "flags": []}
    ret = np.array(d["mean_return"], float); perp = np.array(d["state_visitation_perplexity"], float)
    nodes = np.array(d.get("total_nodes", perp), float); n = len(ret)
    Rn, floor_n, fire = _channels(d, emin); npairs = len(fire)

    # --- return: robust level change from the INITIAL policy to the late plateau, beyond the plateau's noise.
    # The baseline is a short leading window (the initial policy), so an early jump-then-plateau still reads as
    # movement; a wide window or the full-series slope both miss an early step (the slope is ~0 once the plateau
    # dominates). Direction comes from the sign of the level change, not the slope. ---
    r_slope, _, _ = _theil(ret)
    # baseline is the INITIAL policy (first checkpoint), so an early jump that completes within the first few
    # checkpoints still counts as movement -- a median-of-first-few baseline sits past such a jump and would
    # mislabel a run that improved early then plateaued as "declining". The 2-sigma band rejects a noisy ret[0].
    k = max(n // 5, 2); early = float(ret[0]); late = float(np.median(ret[-k:]))
    total_ch = late - early
    # signal vs measurement noise: the reward moved for real if the change beats the standard error of the
    # difference (2 sigma). SEM = std_return / sqrt(N_eval). Falls back to a rough band when the count/std are absent.
    sd = np.asarray(d.get("std_return", []), float); have_sem = bool(n_eval and n_eval > 0 and sd.size == n)
    if have_sem:
        sem_e = float(sd[0]) / np.sqrt(n_eval); sem_l = float(np.median(sd[-k:])) / np.sqrt(n_eval)
        band = 2.0 * float(np.hypot(sem_e, sem_l))
    else:
        band = 1.3 * max(float(np.std(ret)), 0.05 * abs(late), 1e-9)
    rise = total_ch > band
    decl = -total_ch > band
    step = rise and n > 1 and abs(ret[1] - ret[0]) > 0.5 * abs(late - ret[0] + 1e-9)

    # --- coverage (self-referential, reference-free): late effective coverage relative to the run's OWN widest
    # reach (peak distinct-state count, inferred from the trajectories). This is the coverage metric for EVERY
    # environment -- we assume no external reachable-state count. It answers "how much of the territory you
    # yourself once explored do you still occupy consistently?" -- a collapse measure. Gaming's signature is
    # explore-wide-then-collapse-to-exploit, so its late coverage of its own peak is low; healthy convergence
    # narrows but stays well above the collapse line (good median 0.40 vs gamed 0.09; pooled AUC 0.887).
    # Immune to the self-reference flattening that sinks the peak AS A DENOMINATOR of the unknown world.
    peak_nodes = float(np.max(nodes)) if nodes.size else 0.0
    p_slope, _, _ = _theil(perp)   # slope kept for reporting only
    kp = max(n // 5, 2); perp_late = float(np.median(perp[-kp:])); perp_early = float(np.median(perp[:3]))
    # trend is magnitude-aware (same shape as the reward rule): the net change must beat 2x the perplexity
    # series' own checkpoint jitter -- an empirical stand-in for a measurement error, since perplexity has no
    # stored per-checkpoint std -- AND a minimum meaningful size of 5% of the run's own peak reach. The
    # effect-size floor stops a tiny smooth wiggle (e.g. 7->9 effective states) from reading as a trend.
    p_jit = 2.0 * max(float(np.median(np.abs(np.diff(perp)))) if n > 1 else 0.0, 1e-9)
    p_band = max(p_jit, 0.05 * peak_nodes) if peak_nodes > 0 else p_jit
    cov_rise = (perp_late - perp_early) > p_band; cov_fall = (perp_early - perp_late) > p_band
    frac = float(perp_late / peak_nodes) if peak_nodes > 0 else None   # effective coverage of the run's own peak reach
    retention = frac                                                   # alias kept for the verdict / downstream readers

    # --- settling: settled / still-converging (shifts declining but not yet at floor -> may need more training) / unsettled ---
    s_idx = settling_point(fire); settled = s_idx is not None
    s_frac = (s_idx / npairs) if settled else None
    static = not bool(fire.any())
    excess = np.maximum.reduce([np.maximum(Rn[i] - floor_n[i], 0.0) for i in range(3)])  # per-checkpoint distance above the floor
    ex_slope = _theil(excess)[0] if len(excess) >= 3 else 0.0
    converging = (not settled) and (not static) and ex_slope < 0   # shifts shrinking toward the floor: likely settles with more training
    settle_state = "static" if static else "settled" if settled else "converging" if converging else "unsettled"
    tail = max(npairs // 5, 2)
    redestab = (not static) and npairs > tail and fire[:-tail].mean() <= 0.15 and fire[-tail:].mean() > 0.5

    # --- kind of change: composition of change ABOVE each channel's own floor (the honest common unit) ---
    exc = [float(np.maximum(Rn[i] - floor_n[i], 0).sum()) for i in range(3)]
    kmass = sum(exc); k_sh = [100 * e / kmass for e in exc] if kmass > 0 else [0.0, 0.0, 0.0]
    dom = int(np.argmax(exc)); clear = kmass > 0 and max(k_sh) >= 50

    # --- topological turnover (needs the discovery / abandonment / overlap decomposition; some runs lack it) ---
    DEC_KEYS = ("topological_shift_discovery_raw", "topological_shift_abandonment_raw", "topological_shift_overlap_raw")
    has_decomp = all(d.get(k) is not None and len(d.get(k, [])) >= 1 for k in DEC_KEYS)
    churn_sustained = False; turnover = None
    if has_decomp:
        disc = np.array(d[DEC_KEYS[0]], float); aban = np.array(d[DEC_KEYS[1]], float); over = np.array(d[DEC_KEYS[2]], float)
        tot = float(disc.sum() + aban.sum() + over.sum()) or 1.0
        d_sh, a_sh, r_sh = 100 * disc.sum() / tot, 100 * aban.sum() / tot, 100 * over.sum() / tot
        net_grow = disc.sum() > aban.sum()
        # footprint SIZE trajectory from the raw visited-state count (what the stacked bars sum to).
        ns, ne, npk = float(nodes[0]), float(nodes[-1]), float(nodes.max())
        if npk >= max(ns, ne, 1) * 1.2 and ne <= npk * 0.85: foot_traj = "grew_then_contracted"
        elif ne >= ns * 1.2: foot_traj = "grew"
        elif ne <= ns * 0.8: foot_traj = "shrank"
        else: foot_traj = "stable"
        # churn timing: does the footprint churn (discovery + abandonment) PERSIST or TAPER? Within-run normalized
        # (late-half / early-half) so it is env-agnostic. Sustained churn is the reshaping signature
        # (AUC 0.75 reshaping-vs-good, per-env 0.73-0.90). Guarded to runs with real footprint churn.
        ch = disc + aban; hc = max(len(ch) // 2, 1)
        early_ch = float(ch[:hc].mean()); late_ch = float(ch[-hc:].mean())
        churn_active = max(early_ch, late_ch) > 0.02
        ch_ratio = late_ch / (early_ch + 1e-9)
        disc_trend = ("negligible" if not churn_active else "sustained" if ch_ratio > 0.6 else "front-loaded" if ch_ratio < 0.4 else "steady")
        churn_sustained = bool(churn_active and ch_ratio > 0.6)
        # discovery share of footprint turnover, early vs late window -- the ongoing-exploration signal. Reported as
        # a threshold-free corroboration of the coverage flag (low coverage + dead discovery = the gaming signature;
        # low coverage + live discovery = maybe still searching). NOT its own flag: discovery alone is a weak
        # detector (AUC 0.76, FPR 0.35, since healthy runs also slow discovery as they converge).
        dshare = disc / (disc + aban + over + 1e-9)
        disc_share_early = float(np.median(dshare[:hc])); disc_share_late = float(np.median(dshare[-hc:]))
        turnover = {"discovery": d_sh, "abandonment": a_sh, "reweighting": r_sh, "net_grow": bool(net_grow),
                    "topo_share": k_sh[0], "disc_trend": disc_trend, "foot_traj": foot_traj}
    else:
        disc_share_early = disc_share_late = None
    # coverage-collapse mechanism from the node count alone (works with or without the decomposition)
    collapse_mech = ("it actively dropped places it used to visit"
                     if nodes[-1] < nodes[0] * 0.8
                     else "it keeps touching the same places but concentrates its visits onto fewer of them")

    # --- signals: DESCRIPTIONS of the observed behaviour, not causal labels. The reader combines them.
    # red is reserved for the one validated reference (coverage below tau); everything else is amber = worth a look. ---
    footprint_flag = frac is not None and frac < tau
    # the joint reward sub-diagnosis: pair the narrow footprint with the reward direction for an actionable read
    if rise:
        sub = "The reward is climbing on this narrow footprint, so the run is either a genuinely compact solution or reward exploitation; inspect what the agent is actually doing."
    elif decl:
        sub = "The reward is also falling, so the agent is collapsing onto a poor, narrow policy."
    else:
        sub = "The reward is not improving either, so the agent is under-exploring or stuck rather than exploiting."
    # discovery corroboration (threshold-free, within-run): among collapsed runs, whether new-state discovery has
    # died tells collapse-by-gaming (exploration stopped) from a run still searching. Reported, not flagged --
    # discovery alone is a weak detector (AUC 0.76), but ADDED to coverage it lifts separation to AUC 0.92 and
    # halves the false-positive rate. Expressed as late-vs-early discovery rate so no absolute cut is needed.
    disc_corr = ""
    if disc_share_late is not None and disc_share_early is not None:
        if disc_share_early < 0.002:   # never really discovered, even early, so the ratio would be unstable
            disc_corr = " New-state discovery was minimal throughout, so the agent explored little from the start rather than exploring then collapsing."
        else:
            ratio = 100 * disc_share_late / disc_share_early
            if disc_share_late < disc_share_early:
                disc_corr = (f" New-state discovery has fallen to {ratio:.0f}% of its early rate, so the footprint has stopped expanding: the narrow "
                             f"coverage reflects a settled or exploiting policy rather than active search.")
            else:
                disc_corr = (f" New-state discovery is still active ({ratio:.0f}% of its early rate), so despite the narrow footprint the agent may "
                             f"still be searching.")
    ret_flat = not rise and not decl
    never_settles = (not static) and (not settled)   # behaviour never drops below the noise floor (reported, not flagged)
    learning_in_progress = ret_flat and (not static) and settle_state in ("converging", "unsettled")

    # LAYER 1 -- per-panel FLAGS: only the two conditions validated against ground truth get raised as a flag
    # (coverage collapse vs reward gaming, AUC 0.92; frozen structure vs no-learning, AUC 1.00). Everything else
    # is described by the panels but never elevated to a flag (e.g. "never settles" is normal -- 62% of good runs
    # do it, so it discriminates nothing and stays descriptive). Flag bodies state the FACT only; the meaning is
    # built in Layer 2.
    flags = []
    if footprint_flag:
        flags.append({"severity": "problem", "headline": "Low effective coverage",
                      "body": f"By the end the agent effectively occupies only {100*frac:.0f}% of the {int(peak_nodes)} distinct states it reached at its "
                              f"widest during training, below the {tau:.2f} line. {collapse_mech.capitalize()}."})
    if static:
        flags.append({"severity": "problem", "headline": "Frozen policy",
                      "body": "The policy does not change at any checkpoint: no structural shift clears the noise floor, which active learning never does."})

    # LAYER 2 -- cross-panel INTERPRETATION: combine the panels into what they mean together, with the summary
    # facts (reward, coverage, behaviour) folded into each argument so every read is self-contained.
    r0, rl = fnum(ret[0]), fnum(late)
    rew_txt = (f"the reward rises ({r0} to {rl})" if rise else f"the reward decreases ({r0} to {rl})" if decl
               else f"the reward shows no net trend ({r0} to {rl})")
    cov_txt = (f"effective coverage ends at {100*frac:.0f}% of the widest reach the run ever showed ({int(peak_nodes)} states)"
               if frac is not None else "effective coverage is low")
    beh_txt = ("behaviour is frozen from the start" if static
               else "behaviour never settles, changing at essentially every checkpoint" if never_settles
               else f"behaviour settles about {s_frac*100:.0f}% of the way through" if (settled and s_frac) else "behaviour settles")
    interp = []
    if footprint_flag:
        if rise:
            interp.append({"headline": "Compact solution or reward exploitation",
                           "body": f"{rew_txt.capitalize()} while {cov_txt}, so this is either a genuinely compact solution or reward exploitation.{disc_corr} Inspect what the agent is actually doing against the intended goal."})
        elif decl:
            interp.append({"headline": "Collapse onto a poor policy",
                           "body": f"{rew_txt.capitalize()} and {cov_txt}: the agent is narrowing onto a low-value policy.{disc_corr}"})
        else:
            interp.append({"headline": "Under-exploring or stuck",
                           "body": f"{cov_txt.capitalize()} and {rew_txt}, so the agent is stuck or under-exploring rather than exploiting.{disc_corr}"})
    elif static:
        if rise:
            interp.append({"headline": "Converged or hacked before this window",
                           "body": f"The policy is frozen from the start, yet {rew_txt}: it converged or hacked the reward before the evaluation window rather than never learning."})
        elif decl:
            interp.append({"headline": "Never learned or collapsed",
                           "body": f"The policy is frozen from the start and {rew_txt}: consistent with a run that never learned or has collapsed."})
        else:
            # flat reward + frozen policy is genuinely ambiguous reward-blind: a run that converged/hacked
            # before the window and one that never learned both look like this. Do not assert either (a
            # noise-level late>early used to mislabel random runs as "converged or hacked").
            interp.append({"headline": "Frozen with no reward signal",
                           "body": f"The policy is frozen from the start and {rew_txt}: it either converged (or hacked the reward) before the evaluation window or never learned. Compare the reward level against the task to tell which."})
    else:
        # coverage intact and behaviour not frozen: read the behaviour/reward combination
        if never_settles and ret_flat:
            interp.append({"headline": "Churning without progress",
                           "body": f"{cov_txt.capitalize()} and {beh_txt}, but {rew_txt}: activity without payoff so far (it may be mid-learning, or reshaping without improving)."})
        elif never_settles and rise:
            interp.append({"headline": "Still developing",
                           "body": f"{beh_txt.capitalize()} while {rew_txt}, and {cov_txt}: the run is still developing rather than settled."})
        elif settled and ret_flat:
            interp.append({"headline": "Converged without improving",
                           "body": f"{beh_txt.capitalize()} and {cov_txt}, but {rew_txt}."})
        elif decl:
            interp.append({"headline": "Declining reward",
                           "body": f"{rew_txt.capitalize()} while {cov_txt} and {beh_txt}: the policy is worsening even though its coverage held."})
    if not interp:
        interp.append({"headline": "No concern flagged",
                       "body": f"{rew_txt.capitalize()}, {cov_txt}, and {beh_txt}."})

    return {"n_pairs": int(npairs), "insufficient": False,
            "return": [float(ret[0]), float(ret[-1])], "return_late": late, "return_slope": float(r_slope),
            "return_noise": float(band), "total_change": float(total_ch),
            "trend": "rising" if rise else "declining" if decl else "flat", "step": bool(step),
            "perplexity": [float(perp[0]), float(perp[-1])], "nodes": [float(nodes[0]), float(nodes[-1])],
            "perp_slope": float(p_slope), "cov_trend": "rising" if cov_rise else "falling" if cov_fall else "flat",
            "reachable": reachable, "footprint_frac": frac, "perp_late": perp_late, "footprint_flag": bool(footprint_flag), "tau": float(tau),
            "retention": retention, "peak_nodes": float(peak_nodes),
            "cov_contracting": bool(cov_fall and not footprint_flag),
            "settled": bool(settled), "settle_idx": s_idx, "settle_frac": s_frac, "static": bool(static), "redestab": bool(redestab), "never_settles": bool(never_settles),
            "interpretation": interp,
            "settle_state": settle_state, "converging": bool(converging), "learning_in_progress": bool(learning_in_progress), "churn_sustained": churn_sustained,
            "kind_shares": k_sh, "kind_dominant": KKEY[dom], "kind_clear": bool(clear),
            "turnover": turnover,   # None when the discovery/abandonment/overlap decomposition is unavailable
            "disc_share_early": disc_share_early, "disc_share_late": disc_share_late,
            "epsilon_min": [float(x) for x in emin], "flags": flags}

# ================= PANEL NARRATIVES (shared by both report renderers) =================
# Each fingerprint panel (reward, coverage, behaviour, footprint turnover) is turned into prose here, ONCE,
# from the verdict dict alone. render_report (HTML) and render_report_svg (flat SVG) both consume these, so a
# run reads identically in every output format and any wording/logic fix lives in exactly one place (before
# this, the same reading was re-derived in each renderer, and every fix had to be made twice).
# Each builder returns a dict: why (the sentence), status ("obs"|"watch"|"flag" -> chip style), label (the
# trend word, WITHOUT an arrow) and tone ("up"|"down"|"flat"|"updown"|"" -> the arrow glyph each renderer
# prepends, and the colour of the reward chip). _panel_turnover also returns available=False when the
# discovery/abandonment decomposition is missing.

KINDP = ["where it goes", "which actions it takes", "the order it acts in"]   # index-aligned with KKEY

# footprint SIZE trajectory (node count) -> one sentence; keyed by verdict["turnover"]["foot_traj"].
_FOOT_TRAJ_TXT = {
    "grew": "Overall the footprint grows and stays near its widest.",
    "grew_then_contracted": "Overall the footprint grows to a peak, then contracts, dropping some of the states it had reached.",
    "shrank": "Overall the footprint shrinks over training.",
    "stable": "Overall the footprint stays about the same size, turning over in place."}
_FOOT_TRAJ_LABEL = {"grew": "grew", "grew_then_contracted": "peaked", "shrank": "shrank", "stable": "stable"}
_FOOT_TRAJ_TONE = {"grew": "up", "grew_then_contracted": "updown", "shrank": "down", "stable": "flat"}
# churn TIMING (does discovery+abandonment persist or taper?) -> a trailing sentence; keyed by disc_trend.
# "sustained" is handled specially in _panel_turnover so the reshaping label can be gated on the behaviour.
_CHURN_TIMING_TXT = {
    "front-loaded": " The adding and dropping happens mostly early, then tapers off (healthy consolidation).",
    "steady": " Adding and dropping continue at a fairly steady rate.",
    "negligible": ""}


def _panel_reward(v):
    """Reward trend + one-line reading. Direction comes from the validated start->late level-change test,
    not the Theil slope (which rounds to ~0 for a step-shaped curve and once contradicted the trend word)."""
    r0, late = fnum(v["return"][0]), fnum(v["return_late"])
    if v["trend"] == "rising":
        why = f"Reward rises from {r0} to {late}, a change larger than its measurement error."
        if v["step"]: why += " Most of the gain is one early jump."
        return {"why": why, "status": "obs", "label": "rising", "tone": "up"}
    if v["trend"] == "declining":
        return {"why": f"Reward decreases from {r0} to {late}, a change larger than its measurement error.",
                "status": "watch", "label": "decreasing", "tone": "down"}
    return {"why": f"Reward shows no net trend: the change from start ({r0}) to end ({late}) stays within its measurement error, so it never improved.",
            "status": "watch", "label": "no trend", "tone": "flat"}


def _panel_coverage(v):
    """Effective-coverage reading, measured against the run's OWN widest reach. Low coverage is a validated
    red flag; 'collapsed' wording is used only when coverage is actually falling, not merely low-but-rising."""
    tau = v.get("tau", 0.20)
    peakn = int(round(float(v.get("peak_nodes") or 0)))
    cov_pct = None if v.get("footprint_frac") is None else int(round(100 * v["footprint_frac"]))
    tone = {"rising": "up", "falling": "down"}.get(v["cov_trend"], "flat")
    label = {"rising": "rising", "falling": "falling"}.get(v["cov_trend"], "holding")
    if peakn <= 0 or cov_pct is None:
        return {"why": "Not enough state-visitation data to measure coverage for this run.", "status": "obs", "label": label, "tone": tone}
    head = (f"By the end the agent effectively occupies {cov_pct}% of the {peakn} distinct states it reached at its "
            f"widest during training (perplexity vs the peak-reach ceiling), ")
    if v["footprint_flag"]:
        tail = (" Its effective coverage is low but rising, so it is expanding from a very narrow footprint rather than collapsing." if v["cov_trend"] == "rising"
                else " It has collapsed onto a fraction of the territory it once explored." if v["cov_trend"] == "falling"
                else " It stays on a small fraction of the territory it reached at its widest.")
        return {"why": head + f"under the {int(round(tau * 100))}% line.{tail}", "status": "flag", "label": label, "tone": tone}
    contracting = v.get("cov_contracting", False)
    tail = " Its footprint is shrinking over training (perplexity trends down), though it stays above the line." if contracting else ""
    return {"why": head + f"at or above the {int(round(tau * 100))}% line.{tail}", "status": "watch" if contracting else "obs", "label": label, "tone": tone}


def _panel_behaviour(v):
    """Settling state + composition-of-change reading. A frozen policy is the only validated red flag here;
    'never settles' is reported but not flagged (most healthy runs never fully settle)."""
    ss = v.get("settle_state", "settled" if v["settled"] else "unsettled")
    kindp = KINDP[KKEY.index(v["kind_dominant"])] if v["kind_clear"] else "several aspects at once"
    ksh = v["kind_shares"]
    mix = f"When it does change, most of that change is in {kindp} ({ksh[0]:.0f}% where it goes, {ksh[1]:.0f}% which actions, {ksh[2]:.0f}% action order)."
    if v["static"]:
        return {"why": "Its behaviour never changes by more than the noise floor at any checkpoint.", "status": "flag", "label": "static", "tone": ""}
    if ss == "settled" and v["settle_idx"] and v["settle_idx"] > 0:
        return {"why": f"It changes early, then stops: after about {v['settle_frac']*100:.0f}% of training its behaviour stays below the noise floor. {mix}",
                "status": "obs", "label": f"settles ~{v['settle_frac']*100:.0f}%", "tone": ""}
    if ss == "settled":
        return {"why": f"Its behaviour stays around the noise floor the whole time. {mix}", "status": "obs", "label": "at floor", "tone": ""}
    if ss == "converging":
        return {"why": f"It has not settled yet, but its shift magnitude is shrinking toward the noise floor, so it looks on track to settle with more training. {mix}",
                "status": "obs", "label": "still settling", "tone": ""}
    return {"why": f"Its behaviour keeps changing and the shift magnitude is not shrinking toward the noise floor. {mix}", "status": "obs", "label": "unsettled", "tone": ""}


def _panel_turnover(v):
    """Footprint-turnover reading (discovery / abandonment / restructure). The 'perpetual-reshaping
    signature' phrase is used ONLY when the behaviour genuinely never settles (settle_state == 'unsettled');
    a converging/settled run whose footprint still churns is consolidating, and calling that reshaping would
    contradict the behaviour panel."""
    tp = v.get("turnover")
    if tp is None:
        return {"why": ("The topological-shift decomposition (discovery / abandonment / restructure) was not recorded for this run, "
                        "so footprint turnover cannot be shown. Re-run Stage 1 with the decomposition enabled to populate this panel."),
                "status": "obs", "label": "", "tone": "", "available": False}
    dt = tp.get("disc_trend")
    if dt == "sustained":
        timing = (" The footprint keeps being restructured through training (discovery and abandonment persist), the perpetual-reshaping signature."
                  if v.get("settle_state") == "unsettled" else
                  " The footprint keeps turning over states through training, but its overall behavioural change is at or heading toward the noise floor (see Behavioural change), so this reads as ongoing consolidation rather than perpetual reshaping.")
    else:
        timing = _CHURN_TIMING_TXT.get(dt, "")
    why = (f"This looks only at how the set of places it visits changes. {tp['discovery']:.0f}% is finding new places, "
           f"{tp['abandonment']:.0f}% is dropping places it used to visit, {tp['reweighting']:.0f}% is restructure (revisiting the same places more or less often). "
           f"{_FOOT_TRAJ_TXT[tp['foot_traj']]}{timing}")
    return {"why": why, "status": "obs", "label": _FOOT_TRAJ_LABEL[tp["foot_traj"]], "tone": _FOOT_TRAJ_TONE[tp["foot_traj"]], "available": True}


# ---------- SVG helpers (self-contained, theme-aware inline charts) ----------
_AC, _FNT, _GOOD, _CRIT = "var(--ac)", "var(--fnt)", "var(--good)", "var(--crit)"
_C1, _C2, _C3 = "var(--c1)", "var(--c2)", "var(--c3)"
_PURP = "#b39ddb"   # light purple: restructure (revisiting known states) in the turnover panel

_MARG = (44, 10, 10, 30)   # left, right, top, bottom -- shared so every chart's plot area lines up on x

def _xfmt(v):
    return f"{v/1000:.1f}k" if abs(v) >= 1000 else f"{v:.0f}"

def _ytick(v):
    """Y-axis tick label, never scientific notation (a plain '.2g' emits e.g. '-1.3e+02').
    Uses a k-suffix for large magnitudes to match _xfmt, and fixed decimals below 1."""
    a = abs(v)
    if a < 5e-4: return "0"
    if a >= 10000: return f"{v/1000:.0f}k"
    if a >= 1000: return f"{v/1000:.1f}k"
    if a >= 10: return f"{v:.0f}"
    if a >= 1: return f"{v:.1f}"
    if a >= 0.1: return f"{v:.2f}"
    return f"{v:.3f}"

def _lc(series, colors, floor=None, lbl="", W=360, H=120, yr=None, xpos=None, xdom=None,
        std=None, dashed=None, yscale="lin", xlab="checkpoint", ylab="", ymin0=False, ticks=None, hlines=None):
    """Line chart. xpos/xdom put every panel on a shared checkpoint axis so they align vertically.
    std = list (per series) of error arrays -> shaded +/- band. yscale='sqrt' expands the near-zero
    region so small shift values are visible. ymin0 forces the y-axis to start at 0.
    hlines = list of (y_value, label, color) -> horizontal reference lines (e.g. the peak-reach ceiling)."""
    arrs = [np.asarray(s, float) for s in series]
    L, R, T, B = _MARG; m = len(arrs[0])
    stk = list(arrs) + ([np.asarray(floor, float)] if floor is not None else [])
    if hlines: stk += [np.array([float(y) for y, _, _ in hlines], float)]
    if std is not None:
        for j, sd in enumerate(std):
            if sd is not None: stk += [arrs[j] + np.asarray(sd, float), arrs[j] - np.asarray(sd, float)]
    allv = np.concatenate(stk)
    if yr is not None: ymin, ymax = yr
    else:
        ymin, ymax = float(np.nanmin(allv)), float(np.nanmax(allv))
        if yscale == "sqrt" or ymin0: ymin = 0.0
        rng = (ymax - ymin) or 1.0; ymax += 0.08 * rng; ymin -= 0.08 * rng * (ymin < 0)
    if ymax <= ymin: ymax = ymin + 1
    xpos = np.arange(m, dtype=float) if xpos is None else np.asarray(xpos, float)
    x0, x1 = (float(xpos[0]), float(xpos[-1])) if xdom is None else xdom
    xr = (x1 - x0) or 1.0
    PX = lambda p: L + (np.asarray(p, float) - x0) / xr * (W - R - L)
    if yscale == "sqrt":
        tf = lambda v: np.sqrt(np.clip(np.asarray(v, float), 0, None)); tmax = float(tf(ymax)) or 1.0
    else:
        tf = lambda v: np.clip(np.asarray(v, float), ymin, ymax) - ymin; tmax = (ymax - ymin) or 1.0
    PY = lambda v: (H - B) - tf(v) / tmax * (H - B - T)
    xw = PX(xpos)
    s = f'<svg viewBox="0 0 {W} {H}" class="lc" preserveAspectRatio="none">'
    if std is not None:
        for j, sd in enumerate(std):
            if sd is None: continue
            up, lo = arrs[j] + np.asarray(sd, float), arrs[j] - np.asarray(sd, float)
            pts = ' '.join(f'{a:.1f},{PY(u):.1f}' for a, u in zip(xw, up)) + ' ' + ' '.join(f'{a:.1f},{PY(l):.1f}' for a, l in zip(xw[::-1], lo[::-1]))
            s += f'<polygon points="{pts}" fill="{colors[j]}" opacity=".13"/>'
    if floor is not None:
        s += '<polyline points="' + ' '.join(f'{a:.1f},{PY(vv):.1f}' for a, vv in zip(xw, np.asarray(floor, float))) + '" fill="none" stroke="var(--fnt)" stroke-dasharray="3 2" stroke-width=".8" opacity=".6"/>'
    if hlines:
        for yv, lab, col in hlines:
            yp = float(PY(float(yv)))
            s += f'<line x1="{L}" y1="{yp:.1f}" x2="{W-R}" y2="{yp:.1f}" stroke="{col}" stroke-width=".9" stroke-dasharray="5 3" opacity=".85"/>'
            s += f'<text x="{W-R-2}" y="{yp-2.5:.1f}" class="ax" text-anchor="end" fill="{col}">{lab}</text>'
    for j, (arr, c) in enumerate(zip(arrs, colors)):
        da = ' stroke-dasharray="4 2"' if dashed and dashed[j] else ''
        pts = ' '.join(f'{a:.1f},{PY(vv):.1f}' for a, vv in zip(xw, arr))
        s += (f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.7"{da}/>'
              + ''.join(f'<circle cx="{a:.1f}" cy="{PY(vv):.1f}" r="1.7" fill="{c}"/>' for a, vv in zip(xw, arr))
              + f'<circle cx="{xw[-1]:.1f}" cy="{PY(arr[-1]):.1f}" r="2.3" fill="{c}"/>')
    s += f'<line x1="{L}" y1="{T-2}" x2="{L}" y2="{H-B}" stroke="var(--fnt)" stroke-width=".5" opacity=".4"/><line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="var(--fnt)" stroke-width=".5" opacity=".4"/>'
    tickv = [(f * f) * ymax for f in (0, .25, .5, .75, 1)] if yscale == "sqrt" else list(np.linspace(ymin, ymax, 4))
    for yv in tickv:
        yp = float(PY(yv)); s += f'<line x1="{L-2.5}" y1="{yp:.1f}" x2="{L}" y2="{yp:.1f}" stroke="var(--fnt)" stroke-width=".5" opacity=".5"/><text x="{L-4}" y="{yp+2.5:.1f}" class="ax" text-anchor="end">{_ytick(yv)}</text>'
    tickpos = xw if ticks is None else PX(np.asarray(ticks, float))
    for xp in tickpos:  # minor ticks at every checkpoint (unlabeled)
        s += f'<line x1="{xp:.1f}" y1="{H-B}" x2="{xp:.1f}" y2="{H-B+3}" stroke="var(--fnt)" stroke-width=".6" opacity=".55"/>'
    for xv in np.linspace(x0, x1, 5):
        xp = float(PX(xv)); s += f'<line x1="{xp:.1f}" y1="{H-B}" x2="{xp:.1f}" y2="{H-B+6}" stroke="var(--fnt)" stroke-width=".8" opacity=".75"/><text x="{xp:.1f}" y="{H-B+13}" class="ax" text-anchor="middle">{_xfmt(xv)}</text>'
    s += f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" class="axl" text-anchor="middle">{xlab}</text>'
    if ylab:
        yc = (T + H - B) / 2; s += f'<text x="9" y="{yc:.0f}" class="axl" text-anchor="middle" transform="rotate(-90 9 {yc:.0f})">{ylab}</text>'
    return s + '</svg>'

def _mirror(disc, aban, scale_max=None, W=360, H=120, xpos=None, xdom=None, xlab="checkpoint", ylab="turnover"):
    disc = np.asarray(disc, float); aban = np.asarray(aban, float); net = disc - aban
    m = len(disc); L, R, T, B = _MARG; mid = (T + (H - B)) / 2.0
    base = scale_max if scale_max else max(float(np.nanmax(disc)), float(np.nanmax(aban)), 1e-9)
    mx = base * 1.05
    xpos = np.arange(m, dtype=float) if xpos is None else np.asarray(xpos, float)
    x0, x1 = (float(xpos[0]), float(xpos[-1])) if xdom is None else xdom
    xr = (x1 - x0) or 1.0; xw = L + (xpos - x0) / xr * (W - R - L); sc = (mid - T) / mx
    dpoly = f'{xw[0]:.1f},{mid:.1f} ' + ' '.join(f'{a:.1f},{mid-disc[i]*sc:.1f}' for i, a in enumerate(xw)) + f' {xw[-1]:.1f},{mid:.1f}'
    apoly = f'{xw[0]:.1f},{mid:.1f} ' + ' '.join(f'{a:.1f},{mid+aban[i]*sc:.1f}' for i, a in enumerate(xw)) + f' {xw[-1]:.1f},{mid:.1f}'
    npath = ' '.join(f'{a:.1f},{mid-net[i]*sc:.1f}' for i, a in enumerate(xw))
    s = f'<svg viewBox="0 0 {W} {H}" class="lc" preserveAspectRatio="none">'
    s += f'<polygon points="{dpoly}" fill="{_GOOD}" opacity=".45"/><polygon points="{apoly}" fill="{_CRIT}" opacity=".45"/>'
    s += f'<line x1="{L}" y1="{mid:.1f}" x2="{W-R}" y2="{mid:.1f}" stroke="var(--ink)" stroke-width=".7" opacity=".6"/>'
    s += f'<polyline points="{npath}" fill="none" stroke="var(--ink)" stroke-width="1.2"/>'
    peak = mx / 1.05
    s += f'<line x1="{L}" y1="{T-2}" x2="{L}" y2="{H-B}" stroke="var(--fnt)" stroke-width=".5" opacity=".4"/>'
    s += f'<text x="{L-4}" y="{T+3:.0f}" class="ax" text-anchor="end">+{_ytick(peak)}</text><text x="{L-4}" y="{mid+2:.0f}" class="ax" text-anchor="end">0</text><text x="{L-4}" y="{H-B:.0f}" class="ax" text-anchor="end">&#8722;{_ytick(peak)}</text>'
    for xv in np.linspace(x0, x1, 5):
        xp = L + (xv - x0) / xr * (W - R - L); s += f'<text x="{xp:.1f}" y="{H-B+11}" class="ax" text-anchor="middle">{_xfmt(xv)}</text>'
    s += f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" class="axl" text-anchor="middle">{xlab}</text>'
    yc = (T + H - B) / 2; s += f'<text x="9" y="{yc:.0f}" class="axl" text-anchor="middle" transform="rotate(-90 9 {yc:.0f})">{ylab}</text>'
    return s + '</svg>'

def _stackts(segs, colors, W=360, H=120, xpos=None, xdom=None, xlab="checkpoint", ylab="turnover", ymax=1.0, ticks=None):
    """Stacked bars over time: each checkpoint gets one bar stacking the segments (bottom->top).
    Fixed y-axis [0, ymax] so bars are comparable across runs/models."""
    arrs = [np.asarray(a, float) for a in segs]; m = len(arrs[0]); L, R, T, B = _MARG
    xpos = np.arange(m, dtype=float) if xpos is None else np.asarray(xpos, float)
    x0, x1 = (float(xpos[0]), float(xpos[-1])) if xdom is None else xdom; xr = (x1 - x0) or 1.0
    PX = lambda p: L + (float(p) - x0) / xr * (W - R - L)
    PY = lambda v: (H - B) - min(max(v, 0.0), ymax) / ymax * (H - B - T)
    # thin bars centered on their own x position; width from the actual bar spacing so they never touch.
    xw = np.array([PX(p) for p in xpos], float)
    spacing = float(np.median(np.diff(np.sort(xw)))) if m > 1 else float(W - R - L)
    bw = max(spacing * 0.5, 0.7)
    s = f'<svg viewBox="0 0 {W} {H}" class="lc" preserveAspectRatio="none">'
    for i in range(m):
        x = PX(xpos[i]); base = 0.0
        for a, c in zip(arrs, colors):
            h = float(a[i]); y1 = PY(base + h); y0 = PY(base)
            if y0 - y1 > 0.2: s += f'<rect x="{x-bw/2:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{y0-y1:.1f}" fill="{c}" opacity=".9"/>'
            base += h
    s += f'<line x1="{L}" y1="{T-2}" x2="{L}" y2="{H-B}" stroke="var(--fnt)" stroke-width=".5" opacity=".4"/><line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="var(--fnt)" stroke-width=".5" opacity=".4"/>'
    for yv in np.linspace(0, ymax, 4):
        yp = float(PY(yv)); s += f'<line x1="{L-2.5}" y1="{yp:.1f}" x2="{L}" y2="{yp:.1f}" stroke="var(--fnt)" stroke-width=".5" opacity=".5"/><text x="{L-4}" y="{yp+2.5:.1f}" class="ax" text-anchor="end">{_ytick(yv)}</text>'
    for xp in xw:  # minor tick under each bar center, so the bars sit centered on their ticks
        s += f'<line x1="{xp:.1f}" y1="{H-B}" x2="{xp:.1f}" y2="{H-B+3}" stroke="var(--fnt)" stroke-width=".6" opacity=".55"/>'
    for xv in np.linspace(x0, x1, 5):
        xp = float(PX(xv)); s += f'<line x1="{xp:.1f}" y1="{H-B}" x2="{xp:.1f}" y2="{H-B+6}" stroke="var(--fnt)" stroke-width=".8" opacity=".75"/><text x="{xp:.1f}" y="{H-B+13}" class="ax" text-anchor="middle">{_xfmt(xv)}</text>'
    s += f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" class="axl" text-anchor="middle">{xlab}</text>'
    yc = (T + H - B) / 2; s += f'<text x="9" y="{yc:.0f}" class="axl" text-anchor="middle" transform="rotate(-90 9 {yc:.0f})">{ylab}</text>'
    return s + '</svg>'

def _stackbar(segs, title="total", W=360, H=36):
    x0 = 30; bw = W - 36; x = x0
    s = f'<svg viewBox="0 0 {W} {H}" class="lc" preserveAspectRatio="none"><text x="2" y="16" class="ax">{title}</text>'
    for val, col, _ in segs:
        w = bw * val / 100.0
        s += f'<rect x="{x:.1f}" y="7" width="{max(w,0):.1f}" height="14" fill="{col}" opacity=".85"/>'
        if w > 30: s += f'<text x="{x+w/2:.1f}" y="17" class="axw" text-anchor="middle">{val:.0f}%</text>'
        x += w
    for (val, col, lbl), (anc, ax) in zip(segs, [("start", x0), ("middle", (x0 + W - 6) / 2), ("end", W - 6)]):
        s += f'<text x="{ax:.0f}" y="{H-4}" class="ax" text-anchor="{anc}">{lbl}</text>'
    return s + '</svg>'

_REPORT_CSS = """
:root{--g:#eef2f5;--s:#fff;--s2:#f6f8fa;--ink:#141a20;--mut:#5a6673;--fnt:#8390a0;--h:#dbe2e9;--ac:#0072b2;--good:#1a875a;--good-bg:#e4f2eb;--warn:#b5620a;--warn-bg:#f8ecdd;--crit:#b5322a;--crit-bg:#f7e4e2;--c1:#0072b2;--c2:#b5322a;--c3:#8a6d1f}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--g:#0e1216;--s:#161b21;--s2:#1b222a;--ink:#e7ecf1;--mut:#9aa6b3;--fnt:#67737f;--h:#2a333d;--ac:#4ba8db;--good:#4cc38a;--good-bg:#17332a;--warn:#e0913c;--warn-bg:#382713;--crit:#e5675c;--crit-bg:#3a201d;--c1:#4ba8db;--c2:#e5675c;--c3:#d3b34a}}
:root[data-theme=dark]{--g:#0e1216;--s:#161b21;--s2:#1b222a;--ink:#e7ecf1;--mut:#9aa6b3;--fnt:#67737f;--h:#2a333d;--ac:#4ba8db;--good:#4cc38a;--good-bg:#17332a;--warn:#e0913c;--warn-bg:#382713;--crit:#e5675c;--crit-bg:#3a201d;--c1:#4ba8db;--c2:#e5675c;--c3:#d3b34a}
*{box-sizing:border-box}body{margin:0;background:var(--g);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.5}
.wrap{max-width:880px;margin:0 auto;padding:42px 20px 60px}.mono{font-family:"IBM Plex Mono",monospace}
.thesis{font-size:11px;color:var(--fnt);text-transform:uppercase;letter-spacing:.09em;margin:0 0 6px}
h1{font-size:19px;font-weight:700;margin:0 0 2px;word-break:break-all}.rmeta{font-size:12px;color:var(--mut);margin:0 0 14px}
.narr{font-size:14.5px;margin:0 0 6px;padding:11px 13px;background:var(--s2);border-left:3px solid var(--ac);border-radius:7px}
.ag{display:flex;gap:16px;align-items:center;padding:13px 0;border-top:1px solid var(--h)}
.agg{flex:none;width:360px;display:flex;flex-direction:column;gap:4px}.lc{width:360px;height:auto;background:var(--s2);border:1px solid var(--h);border-radius:8px}
.ax{fill:var(--fnt);font-family:"IBM Plex Mono",monospace;font-size:7.5px}.axw{fill:#fff;font-family:"IBM Plex Mono",monospace;font-size:7.5px;font-weight:600}
.axl{fill:var(--fnt);font-family:"IBM Plex Sans",sans-serif;font-size:8.5px;opacity:.75}
.agt{font-size:12.5px}.agh{display:flex;align-items:center;gap:9px;margin-bottom:4px;flex-wrap:wrap}
.cn{font-weight:700;font-size:13px}.ct{font-weight:600;font-size:12px}.agw{color:var(--mut)}
.lgd{display:flex;flex-wrap:wrap;gap:3px 10px;font-size:9.5px;color:var(--mut)}.lg{display:inline-flex;align-items:center;gap:4px}
.sw{width:10px;height:3px;border-radius:2px;display:inline-block}.sw.dash{height:0;width:12px;border-top:1.5px dashed var(--fnt);border-radius:0}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:10px;font-weight:600;padding:2px 8px;border-radius:999px;text-transform:uppercase;letter-spacing:.03em}
.chip.w{background:var(--warn-bg);color:var(--warn)}.chip.c{background:var(--crit-bg);color:var(--crit)}.chip.n{background:var(--s2);color:var(--mut);border:1px solid var(--h)}
.cdot{width:7px;height:7px;border-radius:50%}.cdot.w{background:var(--warn)}.cdot.c{background:var(--crit)}.cdot.n{background:var(--fnt)}
.ct.w{color:var(--warn)}.ct.c{color:var(--crit)}.ct.n{color:var(--mut)}
.flags{margin-top:16px;display:flex;flex-direction:column;gap:8px}
.flag{display:flex;align-items:flex-start;gap:10px;font-size:13px;padding:10px 13px;border-radius:9px;border:1px solid}
.flag .ico{font-family:"IBM Plex Mono",monospace;font-weight:700;font-size:12px;margin-top:1px}
.flag.watch{background:var(--warn-bg);border-color:color-mix(in srgb,var(--warn) 30%,transparent)}
.flag.crit{background:var(--crit-bg);border-color:color-mix(in srgb,var(--crit) 34%,transparent)}
.flag.info{background:var(--s2);border-color:var(--h)}.flag.info .ico{color:var(--ac)}
.flag b{color:var(--ink)}.flag .txt{color:var(--ink)}
.secl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin:12px 0 2px}
.secn{font-weight:400;text-transform:none;letter-spacing:0;color:var(--fnt)}
.none{font-size:12.5px;color:var(--fnt);padding-top:6px}
@media(max-width:600px){.ag{flex-direction:column;align-items:stretch}.agg,.lc{width:100%}}
"""
_DOTC = {"obs": "n", "watch": "w", "flag": "c"}
_LABC = {"obs": "not flagged", "watch": "worth a look", "flag": "check this"}

def _legend(items):
    """Small colour key rendered under a plot so the prose need not name colours. items = [(css_color_or_'dash', label)]."""
    sw = ""
    for c, l in items:
        chip = '<span class="sw dash"></span>' if c == "dash" else f'<span class="sw" style="background:{c}"></span>'
        sw += f'<span class="lg">{chip}{l}</span>'
    return f'<div class="lgd">{sw}</div>'

def _ag(graph, name, trend_lbl, status, why, trend_col=None):
    dc = _DOTC[status]
    # the trend label can be coloured by direction (trend_col), independent of the severity chip
    ct = f'<span class="ct" style="color:{trend_col};font-weight:600">{trend_lbl}</span>' if trend_col else f'<span class="ct {dc}">{trend_lbl}</span>'
    return (f'<div class="ag"><div class="agg">{graph}</div><div class="agt">'
            f'<div class="agh"><span class="chip {dc}"><span class="cdot {dc}"></span>{_LABC[status]}</span>'
            f'<span class="cn">{name}</span>{ct}</div>'
            f'<div class="agw">{why}</div></div></div>')

def render_report(d, v, title):
    import html as _h
    esc = _h.escape
    ret = np.array(d["mean_return"], float); perp = np.array(d["state_visitation_perplexity"], float)
    nodes = np.array(d.get("total_nodes", perp), float)
    Rn, floor_n, _ = _channels(d, v["epsilon_min"])
    tp = v["turnover"]   # None when the discovery/abandonment/overlap decomposition is unavailable
    if tp is not None:
        disc = np.array(d["topological_shift_discovery_raw"], float); aban = np.array(d["topological_shift_abandonment_raw"], float)
        over = np.array(d["topological_shift_overlap_raw"], float)   # restructure: revisiting known states
    # shared checkpoint x-axis: reward/coverage sit on the checkpoints, shifts/turnover on the pair midpoints
    n = len(ret); cps = np.array(d.get("checkpoints", np.arange(n)), float)
    if len(cps) != n: cps = np.arange(n, dtype=float)
    xdom = (float(cps[0]), float(cps[-1])); xmid = (cps[:-1] + cps[1:]) / 2.0
    std_ret = np.array(d.get("std_return", np.zeros(n)), float)
    rtrue = np.array(d.get("mean_r_true", [np.nan] * n), float); std_rt = np.array(d.get("std_r_true", [np.nan] * n), float)
    has_true = bool(np.isfinite(rtrue).any())

    # --- panels: the shared narrative (see the _panel_* builders above) + this renderer's chart per row ---
    HTML_ARROW = {"up": "&#9650;", "down": "&#9660;", "flat": "&rarr;", "updown": "&#9650;&#9660;", "": ""}
    _RCOL = {"up": "#1a875a", "down": "#b5322a", "flat": "#8390a0"}
    chip = lambda p: f'{HTML_ARROW[p["tone"]]} {p["label"]}'.strip()   # arrow glyph + trend word
    tau = v.get("tau", 0.20); _peakv = float(v.get("peak_nodes") or 0.0)
    rows = []

    # Reward -- the only chip coloured by direction
    pr = _panel_reward(v)
    if has_true:
        rew_svg = _lc([ret, rtrue], [_AC, _CRIT], xpos=cps, xdom=xdom, std=[std_ret, std_rt], dashed=[False, True], ylab="return")
        rew_leg = _legend([(_AC, "proxy reward"), (_CRIT, "true reward")])
    else:
        rew_svg = _lc([ret], [_AC], xpos=cps, xdom=xdom, std=[std_ret], ylab="return")
        rew_leg = _legend([(_AC, "reward"), (_AC, "&plusmn;1 s.d.")])
    rows.append(_ag(rew_svg + rew_leg, "Reward", chip(pr), pr["status"], pr["why"], trend_col=_RCOL[pr["tone"]]))

    # State coverage -- perplexity vs the run's own peak-reach ceiling and the 20% flag line
    pc = _panel_coverage(v)
    cov_hl = [(_peakv, f"peak reach {int(round(_peakv))}", "var(--mut)"), (tau * _peakv, f"{int(round(tau*100))}% line", _CRIT)] if _peakv > 0 else []
    cov_leg = _legend([(_AC, "perplexity (effective states)"), (_FNT, "distinct states / checkpoint"), ("var(--mut)", "peak reach (max)"), (_CRIT, "20% line")])
    rows.append(_ag(_lc([perp, nodes], [_AC, _FNT], xpos=cps, xdom=xdom, ylab="effective states", ymin0=True, hlines=cov_hl) + cov_leg, "State coverage", chip(pc), pc["status"], pc["why"]))

    # Behavioural change -- shift channels vs their noise floor (only a frozen policy is a validated flag)
    pb = _panel_behaviour(v)
    shift_leg = _legend([(_C1, "topological shift"), (_C2, "strategic shift"), (_C3, "sequential shift"), ("dash", "noise floor")])
    shift_svg = _lc([Rn[0], Rn[1], Rn[2]], [_C1, _C2, _C3], floor=np.maximum.reduce(floor_n), xpos=xmid, xdom=xdom, yr=(0, 1), ticks=cps, ylab="shift (norm.)") + shift_leg
    rows.append(_ag(shift_svg, "Behavioural change", chip(pb), pb["status"], pb["why"]))

    # Footprint turnover -- per-checkpoint discovery / abandonment / restructure (or a placeholder if missing)
    pt = _panel_turnover(v)
    if not pt.get("available", True):
        placeholder = '<svg viewBox="0 0 360 120" class="lc" preserveAspectRatio="none"><text x="180" y="62" class="axl" text-anchor="middle" opacity=".7">decomposition not available</text></svg>'
        rows.append(_ag(placeholder, "Footprint turnover", "&mdash;", pt["status"], pt["why"]))
    else:
        turn_leg = _legend([(_GOOD, "discovery"), (_CRIT, "abandonment"), (_PURP, "restructure")])
        rows.append(_ag(_stackts([disc, aban, over], [_GOOD, _CRIT, _PURP], xpos=xmid, xdom=xdom, ylab="turnover", ticks=cps) + turn_leg,
                        "Footprint turnover", chip(pt), pt["status"], pt["why"]))

    # LAYER 1 validated red flags, then LAYER 2 interpretation (the one-line summary + the cross-panel reads),
    # both kept below the panels, each labelled.
    fl = ""
    for f in v["flags"]:
        fl += f'<div class="flag crit"><span class="ico">&#9650;</span><span class="txt"><b>{esc(f["headline"])}</b> {f["body"]}</span></div>'
    if fl:
        fl = '<div class="secl">Flags</div>' + fl
    ip = ""   # interpretation reads are self-contained (summary facts folded into each argument)
    for f in v.get("interpretation", []):
        ip += f'<div class="flag info"><span class="ico">&#9670;</span><span class="txt"><b>{esc(f["headline"])}</b> {f["body"]}</span></div>'
    if ip:
        ip = '<div class="secl">Interpretation <span class="secn">(what the panels mean together)</span></div>' + ip
    return f'''<title>{esc(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{_REPORT_CSS}</style>
<div class="wrap">
 <p class="thesis">Behavioral Fingerprint &middot; reward-blind interpretation</p>
 <h1>{esc(title)}</h1>
 <p class="rmeta mono">{esc(str(v.get("run", title)))}</p>
 <p class="rmeta">{esc(str(v.get("floor","hard")))} floor &middot; {v["n_pairs"]} checkpoint pairs &middot; a reward-blind portrait read from the trajectories alone. Two conditions raise a validated red flag (low effective coverage, frozen policy); amber chips mark observations worth a look; the interpretation reads the panels together.</p>
 {''.join(rows)}
 <div class="flags">{fl}{ip}</div>
</div>'''


# ---------- Miro-safe flat SVG report (presentation attributes only) ----------
# Miro's importer can't run CSS: no <style>, no class=, no var(--x), no external stylesheet/font/
# image refs, no foreignObject/HTML. Every color, font and position below is a literal, computed
# value baked straight onto each element. This is a second, independent renderer (not a wrapper
# around render_report()'s HTML) because the two draw with entirely different primitives; the
# reward-blind narrative logic is deliberately re-derived here rather than shared, so each renderer
# stays a simple, self-contained read of (d, v).
FLAT = {"bg": "#eef2f5", "surface": "#ffffff", "surface2": "#f6f8fa", "ink": "#141a20",
        "mut": "#5a6673", "fnt": "#8390a0", "hair": "#dbe2e9", "accent": "#0072b2",
        "good": "#1a875a", "good_bg": "#e4f2eb", "warn": "#b5620a", "warn_bg": "#f8ecdd",
        "crit": "#b5322a", "crit_bg": "#f7e4e2", "c1": "#0072b2", "c2": "#b5322a", "c3": "#8a6d1f",
        "purple": "#b39ddb"}
F_SANS = "Arial, Helvetica, sans-serif"
F_MONO = "Courier New, Courier, monospace"
STATUS_STYLE = {"obs": (FLAT["surface2"], FLAT["mut"], FLAT["fnt"]),
                 "watch": (FLAT["warn_bg"], FLAT["warn"], FLAT["warn"]),
                 "flag": (FLAT["crit_bg"], FLAT["crit"], FLAT["crit"])}
LABELS = {"obs": "NOT FLAGGED", "watch": "WORTH A LOOK", "flag": "CHECK THIS"}
_esc = html.escape

@functools.lru_cache(maxsize=None)
def _flat_font_path(kind):
    import matplotlib.font_manager as fm
    if kind == "bold": return fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
    if kind == "mono": return fm.findfont("DejaVu Sans Mono")
    return fm.findfont("DejaVu Sans")

@functools.lru_cache(maxsize=None)
def _flat_font(kind, size):
    from PIL import ImageFont
    return ImageFont.truetype(_flat_font_path(kind), max(int(round(size)), 1))

def _fw(word, kind, size):
    """Pixel width of word, measured against a real proportional font (DejaVu Sans) as an
    Arial-like stand-in -- Miro substitutes its own sans-serif at render time, so this is only
    a wrapping guide, never exact typesetting."""
    return _flat_font(kind, size).getlength(word)

def _split_long_word(word, max_w, kind, size):
    """Hard-break a single token at the character level once it alone exceeds max_w -- mirrors
    the HTML report's word-break:break-all, needed for the underscore-joined run-name/title
    tokens (e.g. 'dqn_batch_size=64_epsilon=...') that carry no spaces to wrap on."""
    if len(word) <= 1 or _fw(word, kind, size) <= max_w:
        return [word]
    chunks, cur = [], ""
    for ch in word:
        trial = cur + ch
        if cur and _fw(trial, kind, size) > max_w:
            chunks.append(cur); cur = ch
        else:
            cur = trial
    if cur: chunks.append(cur)
    return chunks

def _wrap_runs(runs, max_w, size, kind="regular", bold_kind="bold"):
    """runs: [(word, is_bold), ...] in reading order. Packs them into lines <= max_w, returning
    [[(word, is_bold), ...], ...]."""
    space_w = _fw(" ", kind, size)
    expanded = []
    for word, bold in runs:
        k = bold_kind if bold else kind
        expanded += [(piece, bold) for piece in _split_long_word(word, max_w, k, size)]
    lines, cur, cur_w = [], [], 0.0
    for word, bold in expanded:
        k = bold_kind if bold else kind
        w = _fw(word, k, size)
        add = w if not cur else w + space_w
        if cur and cur_w + add > max_w:
            lines.append(cur); cur = [(word, bold)]; cur_w = w
        else:
            cur.append((word, bold)); cur_w += add
    if cur: lines.append(cur)
    return lines or [[]]

def _wrap(text, max_w, size, kind="regular", bold=False):
    return _wrap_runs([(w, bold) for w in str(text).split()], max_w, size, kind)

def _mixed_lines_svg(x, y, wrapped, size, fill, kind="regular", line_h=None):
    """Render pre-wrapped [[(word,bold),...],...] as one <text> per line, bold words as their
    own <tspan font-weight="700">. Returns (svg, total_height).

    Per the SVG whitespace spec (xml:space="default"), leading/trailing whitespace is stripped
    from EACH text chunk independently -- a per-word tspan with a leading space to separate it
    from its neighbour loses that space on any strictly spec-compliant renderer (Chrome is lenient
    about it, which is why this looked fine in a browser but came out with words run together
    elsewhere). Fix: merge consecutive same-style words into one tspan joined by a real interior
    space (never stripped, since it isn't at a chunk edge), and space out style-boundary tspans
    with dx instead of a literal edge space."""
    lh = line_h if line_h is not None else size * 1.42
    fam = F_MONO if kind == "mono" else F_SANS
    space_w = _fw(" ", "bold" if kind == "mono" else kind, size)
    out = []
    for i, line in enumerate(wrapped):
        ty = y + i * lh
        runs = []
        for word, bold in line:
            if runs and runs[-1][1] == bold:
                runs[-1] = (runs[-1][0] + " " + word, bold)
            else:
                runs.append((word, bold))
        spans = []
        for k, (text, bold) in enumerate(runs):
            w_attr = ' font-weight="700"' if bold else ""
            dx_attr = f' dx="{space_w:.1f}"' if k > 0 else ""
            spans.append(f'<tspan{w_attr}{dx_attr}>{_esc(text)}</tspan>')
        out.append(f'<text x="{x:.1f}" y="{ty:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}">{"".join(spans)}</text>')
    return "".join(out), len(wrapped) * lh

def _lc_svg(series, colors, floor=None, W=360, H=120, yr=None, xpos=None, xdom=None,
            std=None, dashed=None, yscale="lin", xlab="checkpoint", ylab="", ymin0=False, ticks=None, hlines=None):
    """Presentation-attribute twin of _lc(): identical geometry, no <style>/class/var()."""
    arrs = [np.asarray(s, float) for s in series]
    L, R, T, B = _MARG; m = len(arrs[0])
    stk = list(arrs) + ([np.asarray(floor, float)] if floor is not None else [])
    if hlines: stk += [np.array([float(y) for y, _, _ in hlines], float)]
    if std is not None:
        for j, sd in enumerate(std):
            if sd is not None: stk += [arrs[j] + np.asarray(sd, float), arrs[j] - np.asarray(sd, float)]
    allv = np.concatenate(stk)
    if yr is not None: ymin, ymax = yr
    else:
        ymin, ymax = float(np.nanmin(allv)), float(np.nanmax(allv))
        if yscale == "sqrt" or ymin0: ymin = 0.0
        rng = (ymax - ymin) or 1.0; ymax += 0.08 * rng; ymin -= 0.08 * rng * (ymin < 0)
    if ymax <= ymin: ymax = ymin + 1
    xpos = np.arange(m, dtype=float) if xpos is None else np.asarray(xpos, float)
    x0, x1 = (float(xpos[0]), float(xpos[-1])) if xdom is None else xdom
    xr = (x1 - x0) or 1.0
    PX = lambda p: L + (np.asarray(p, float) - x0) / xr * (W - R - L)
    if yscale == "sqrt":
        tf = lambda v: np.sqrt(np.clip(np.asarray(v, float), 0, None)); tmax = float(tf(ymax)) or 1.0
    else:
        tf = lambda v: np.clip(np.asarray(v, float), ymin, ymax) - ymin; tmax = (ymax - ymin) or 1.0
    PY = lambda v: (H - B) - tf(v) / tmax * (H - B - T)
    xw = PX(xpos); fnt = FLAT["fnt"]
    s = (f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" preserveAspectRatio="none">'
         f'<rect x="0" y="0" width="{W}" height="{H}" rx="8" ry="8" fill="{FLAT["surface2"]}" stroke="{FLAT["hair"]}" stroke-width="1"/>')
    if std is not None:
        for j, sd in enumerate(std):
            if sd is None: continue
            up, lo = arrs[j] + np.asarray(sd, float), arrs[j] - np.asarray(sd, float)
            pts = ' '.join(f'{a:.1f},{PY(u):.1f}' for a, u in zip(xw, up)) + ' ' + ' '.join(f'{a:.1f},{PY(l):.1f}' for a, l in zip(xw[::-1], lo[::-1]))
            s += f'<polygon points="{pts}" fill="{colors[j]}" fill-opacity=".13"/>'
    if floor is not None:
        s += '<polyline points="' + ' '.join(f'{a:.1f},{PY(vv):.1f}' for a, vv in zip(xw, np.asarray(floor, float))) + f'" fill="none" stroke="{fnt}" stroke-dasharray="3 2" stroke-width=".8" opacity=".6"/>'
    if hlines:
        for yv, lab, col in hlines:
            yp = float(PY(float(yv)))
            s += f'<line x1="{L}" y1="{yp:.1f}" x2="{W-R}" y2="{yp:.1f}" stroke="{col}" stroke-width=".9" stroke-dasharray="5 3" opacity=".85"/>'
            s += f'<text x="{W-R-2}" y="{yp-2.5:.1f}" font-family="{F_MONO}" font-size="7.5" fill="{col}" text-anchor="end">{_esc(lab)}</text>'
    for j, (arr, c) in enumerate(zip(arrs, colors)):
        da = ' stroke-dasharray="4 2"' if dashed and dashed[j] else ''
        pts = ' '.join(f'{a:.1f},{PY(vv):.1f}' for a, vv in zip(xw, arr))
        s += (f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.7"{da}/>'
              + ''.join(f'<circle cx="{a:.1f}" cy="{PY(vv):.1f}" r="1.7" fill="{c}"/>' for a, vv in zip(xw, arr))
              + f'<circle cx="{xw[-1]:.1f}" cy="{PY(arr[-1]):.1f}" r="2.3" fill="{c}"/>')
    s += f'<line x1="{L}" y1="{T-2}" x2="{L}" y2="{H-B}" stroke="{fnt}" stroke-width=".5" opacity=".4"/><line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="{fnt}" stroke-width=".5" opacity=".4"/>'
    tickv = [(f * f) * ymax for f in (0, .25, .5, .75, 1)] if yscale == "sqrt" else list(np.linspace(ymin, ymax, 4))
    for yv in tickv:
        yp = float(PY(yv)); s += f'<line x1="{L-2.5}" y1="{yp:.1f}" x2="{L}" y2="{yp:.1f}" stroke="{fnt}" stroke-width=".5" opacity=".5"/><text x="{L-4}" y="{yp+2.5:.1f}" font-family="{F_MONO}" font-size="7.5" fill="{fnt}" text-anchor="end">{_ytick(yv)}</text>'
    tickpos = xw if ticks is None else PX(np.asarray(ticks, float))
    for xp in tickpos:
        s += f'<line x1="{xp:.1f}" y1="{H-B}" x2="{xp:.1f}" y2="{H-B+3}" stroke="{fnt}" stroke-width=".6" opacity=".55"/>'
    for xv in np.linspace(x0, x1, 5):
        xp = float(PX(xv)); s += f'<line x1="{xp:.1f}" y1="{H-B}" x2="{xp:.1f}" y2="{H-B+6}" stroke="{fnt}" stroke-width=".8" opacity=".75"/><text x="{xp:.1f}" y="{H-B+13}" font-family="{F_MONO}" font-size="7.5" fill="{fnt}" text-anchor="middle">{_xfmt(xv)}</text>'
    s += f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" font-family="{F_SANS}" font-size="8.5" fill="{fnt}" opacity=".75" text-anchor="middle">{_esc(xlab)}</text>'
    if ylab:
        yc = (T + H - B) / 2; s += f'<text x="9" y="{yc:.0f}" font-family="{F_SANS}" font-size="8.5" fill="{fnt}" opacity=".75" text-anchor="middle" transform="rotate(-90 9 {yc:.0f})">{_esc(ylab)}</text>'
    return s + '</svg>'

def _stackts_svg(segs, colors, W=360, H=120, xpos=None, xdom=None, xlab="checkpoint", ylab="turnover", ymax=1.0, ticks=None):
    """Presentation-attribute twin of _stackts()."""
    arrs = [np.asarray(a, float) for a in segs]; m = len(arrs[0]); L, R, T, B = _MARG
    xpos = np.arange(m, dtype=float) if xpos is None else np.asarray(xpos, float)
    x0, x1 = (float(xpos[0]), float(xpos[-1])) if xdom is None else xdom; xr = (x1 - x0) or 1.0
    PX = lambda p: L + (float(p) - x0) / xr * (W - R - L)
    PY = lambda v: (H - B) - min(max(v, 0.0), ymax) / ymax * (H - B - T)
    # thin bars, each centered on its own x position (the checkpoint-pair midpoint); width from the actual
    # bar spacing rather than the full plot width / m, so bars stay centered and never touch.
    xw = np.array([PX(p) for p in xpos], float)
    spacing = float(np.median(np.diff(np.sort(xw)))) if m > 1 else float(W - R - L)
    bw = max(spacing * 0.5, 0.7); fnt = FLAT["fnt"]
    s = (f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" preserveAspectRatio="none">'
         f'<rect x="0" y="0" width="{W}" height="{H}" rx="8" ry="8" fill="{FLAT["surface2"]}" stroke="{FLAT["hair"]}" stroke-width="1"/>')
    for i in range(m):
        x = PX(xpos[i]); base = 0.0
        for a, c in zip(arrs, colors):
            h = float(a[i]); y1 = PY(base + h); y0 = PY(base)
            if y0 - y1 > 0.2: s += f'<rect x="{x-bw/2:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{y0-y1:.1f}" fill="{c}" fill-opacity=".9"/>'
            base += h
    s += f'<line x1="{L}" y1="{T-2}" x2="{L}" y2="{H-B}" stroke="{fnt}" stroke-width=".5" opacity=".4"/><line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="{fnt}" stroke-width=".5" opacity=".4"/>'
    for yv in np.linspace(0, ymax, 4):
        yp = float(PY(yv)); s += f'<line x1="{L-2.5}" y1="{yp:.1f}" x2="{L}" y2="{yp:.1f}" stroke="{fnt}" stroke-width=".5" opacity=".5"/><text x="{L-4}" y="{yp+2.5:.1f}" font-family="{F_MONO}" font-size="7.5" fill="{fnt}" text-anchor="end">{_ytick(yv)}</text>'
    for xp in xw:  # minor tick under each bar center, so the bars sit centered on their ticks
        s += f'<line x1="{xp:.1f}" y1="{H-B}" x2="{xp:.1f}" y2="{H-B+3}" stroke="{fnt}" stroke-width=".6" opacity=".55"/>'
    for xv in np.linspace(x0, x1, 5):
        xp = float(PX(xv)); s += f'<line x1="{xp:.1f}" y1="{H-B}" x2="{xp:.1f}" y2="{H-B+6}" stroke="{fnt}" stroke-width=".8" opacity=".75"/><text x="{xp:.1f}" y="{H-B+13}" font-family="{F_MONO}" font-size="7.5" fill="{fnt}" text-anchor="middle">{_xfmt(xv)}</text>'
    s += f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" font-family="{F_SANS}" font-size="8.5" fill="{fnt}" opacity=".75" text-anchor="middle">{_esc(xlab)}</text>'
    yc = (T + H - B) / 2; s += f'<text x="9" y="{yc:.0f}" font-family="{F_SANS}" font-size="8.5" fill="{fnt}" opacity=".75" text-anchor="middle" transform="rotate(-90 9 {yc:.0f})">{_esc(ylab)}</text>'
    return s + '</svg>'

def _placeholder_svg(W=360, H=120, msg="decomposition not available"):
    return (f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" preserveAspectRatio="none">'
            f'<rect x="0" y="0" width="{W}" height="{H}" rx="8" ry="8" fill="{FLAT["surface2"]}" stroke="{FLAT["hair"]}" stroke-width="1"/>'
            f'<text x="{W/2:.0f}" y="{H/2+3:.0f}" font-family="{F_SANS}" font-size="8.5" fill="{FLAT["fnt"]}" opacity=".75" text-anchor="middle">{_esc(msg)}</text></svg>')

def _legend_svg(items, x, y, w, size=9.5):
    """items: [(hex_color_or_'dash', label), ...], wrapped left-to-right within width w."""
    sw_w, sw_gap, gap_item = 10, 4, 12
    cx, cy, line_h, started = x, y, size * 1.6, False
    out = []
    for col, lbl in items:
        lbl_w = _fw(lbl, "regular", size); item_w = sw_w + sw_gap + lbl_w
        if started and (cx - x + item_w) > w:
            cx = x; cy += line_h
        started = True
        mid = cy + size * 0.55
        if col == "dash":
            out.append(f'<line x1="{cx:.1f}" y1="{mid:.1f}" x2="{cx+12:.1f}" y2="{mid:.1f}" stroke="{FLAT["fnt"]}" stroke-width="1.5" stroke-dasharray="3 2"/>')
            used = 12
        else:
            out.append(f'<rect x="{cx:.1f}" y="{mid-1.5:.1f}" width="{sw_w}" height="3" rx="1.5" fill="{col}"/>')
            used = sw_w
        out.append(f'<text x="{cx+used+sw_gap:.1f}" y="{cy+size*0.95:.1f}" font-family="{F_SANS}" font-size="{size}" fill="{FLAT["mut"]}">{_esc(lbl)}</text>')
        cx += used + sw_gap + lbl_w + gap_item
    return "".join(out), ((cy - y) + line_h if started else 0.0)

def _row_svg(x, y, w, chart_svg, name, trend_label, trend_color, status, why, chart_w=360, chart_h=120, legend_svg="", legend_h=0.0):
    bg, fg, dot = STATUS_STYLE[status]; label = LABELS[status]
    PAD_T, PAD_B, GAP_COL, GAP_ROW = 13, 13, 16, 6
    out = [f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x+w:.1f}" y2="{y:.1f}" stroke="{FLAT["hair"]}" stroke-width="1"/>']
    cy = y + PAD_T
    right_x = x + chart_w + GAP_COL; right_w = max(w - chart_w - GAP_COL, 60)
    lbl_w = _fw(label, "regular", 10); chip_pad, chip_h, dot_r = 8, 18, 3
    chip_w = dot_r * 2 + 5 + lbl_w + 2 * chip_pad
    out.append(f'<rect x="{right_x:.1f}" y="{cy:.1f}" width="{chip_w:.1f}" height="{chip_h}" rx="9" fill="{bg}"/>')
    out.append(f'<circle cx="{right_x+chip_pad+dot_r:.1f}" cy="{cy+chip_h/2:.1f}" r="{dot_r}" fill="{dot}"/>')
    out.append(f'<text x="{right_x+chip_pad+dot_r*2+5:.1f}" y="{cy+chip_h/2+3.5:.1f}" font-family="{F_SANS}" font-size="10" font-weight="700" fill="{fg}">{_esc(label)}</text>')
    nx = right_x + chip_w + 9; name_w = _fw(name, "bold", 13)
    out.append(f'<text x="{nx:.1f}" y="{cy+chip_h/2+4.5:.1f}" font-family="{F_SANS}" font-size="13" font-weight="700" fill="{FLAT["ink"]}">{_esc(name)}</text>')
    tx = nx + name_w + 9
    out.append(f'<text x="{tx:.1f}" y="{cy+chip_h/2+4:.1f}" font-family="{F_SANS}" font-size="12" font-weight="700" fill="{trend_color or fg}">{_esc(trend_label)}</text>')
    header_h = chip_h + 4
    why_svg, why_h = _mixed_lines_svg(right_x, cy + header_h + 9, _wrap(why, right_w, 12.5), 12.5, FLAT["mut"], line_h=17.5)
    out.append(why_svg)
    right_h = header_h + why_h
    left_h = chart_h + (GAP_ROW + legend_h if legend_h else 0)
    out.append(f'<g transform="translate({x:.1f},{cy:.1f})">{chart_svg}</g>')
    if legend_svg:
        out.append(f'<g transform="translate({x:.1f},{cy+chart_h+GAP_ROW:.1f})">{legend_svg}</g>')
    return "".join(out), PAD_T + max(left_h, right_h) + PAD_B

def _card_svg(x, y, w, icon, icon_color, bg, border, headline, body, border_opacity=1.0, size=13):
    pad, icon_w = 13, 22
    text_x = x + pad + icon_w; text_w = max(w - 2 * pad - icon_w, 60)
    runs = [(wd, True) for wd in str(headline).split()] + [(wd, False) for wd in str(body).split()]
    wrapped = _wrap_runs(runs, text_w, size); line_h = size * 1.5
    text_svg, text_h = _mixed_lines_svg(text_x, y + pad + size * 0.9, wrapped, size, FLAT["ink"], line_h=line_h)
    card_h = max(text_h + 2 * pad, 20 + 2 * pad)
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{card_h:.1f}" rx="9" fill="{bg}" stroke="{border}" stroke-width="1" stroke-opacity="{border_opacity}"/>',
           f'<text x="{x+pad:.1f}" y="{y+pad+size*0.9:.1f}" font-family="{F_MONO}" font-size="12" font-weight="700" fill="{icon_color}">{icon}</text>',
           text_svg]
    return "".join(out), card_h

def render_report_svg(d, v, title, width=900):
    """Pure presentation-attribute SVG report for Miro: no <style>, class=, var(--x), external
    refs, or <use>/<defs>. Height is computed exactly from the wrapped text/chart geometry (not
    estimated), so the canvas never clips content or leaves a large blank margin."""
    ret = np.array(d["mean_return"], float); perp = np.array(d["state_visitation_perplexity"], float)
    nodes = np.array(d.get("total_nodes", perp), float)
    Rn, floor_n, _ = _channels(d, v["epsilon_min"])
    tp = v["turnover"]
    if tp is not None:
        disc = np.array(d["topological_shift_discovery_raw"], float); aban = np.array(d["topological_shift_abandonment_raw"], float)
        over = np.array(d["topological_shift_overlap_raw"], float)
    n = len(ret); cps = np.array(d.get("checkpoints", np.arange(n)), float)
    if len(cps) != n: cps = np.arange(n, dtype=float)
    xdom = (float(cps[0]), float(cps[-1])); xmid = (cps[:-1] + cps[1:]) / 2.0
    std_ret = np.array(d.get("std_return", np.zeros(n)), float)
    rtrue = np.array(d.get("mean_r_true", [np.nan] * n), float); std_rt = np.array(d.get("std_r_true", [np.nan] * n), float)
    has_true = bool(np.isfinite(rtrue).any())

    # --- panels: the shared narrative (see the _panel_* builders above) + this renderer's flat-SVG chart ---
    SVG_ARROW = {"up": "▲", "down": "▼", "flat": "→", "updown": "▲▼", "": ""}
    R_COLOR = {"up": "#1a875a", "down": "#b5322a", "flat": FLAT["fnt"]}
    chip = lambda p: (SVG_ARROW[p["tone"]] + " " + p["label"]).strip()   # arrow glyph + trend word
    tau = v.get("tau", 0.20); _peakv = float(v.get("peak_nodes") or 0.0)

    # Reward -- the only chip coloured by direction
    pr = _panel_reward(v)
    if has_true:
        rew_chart = _lc_svg([ret, rtrue], [FLAT["accent"], FLAT["crit"]], xpos=cps, xdom=xdom, std=[std_ret, std_rt], dashed=[False, True], ylab="return")
        rew_leg, rew_leg_h = _legend_svg([(FLAT["accent"], "proxy reward"), (FLAT["crit"], "true reward")], 0, 0, 360)
    else:
        rew_chart = _lc_svg([ret], [FLAT["accent"]], xpos=cps, xdom=xdom, std=[std_ret], ylab="return")
        rew_leg, rew_leg_h = _legend_svg([(FLAT["accent"], "reward"), (FLAT["accent"], "±1 s.d.")], 0, 0, 360)

    # State coverage -- perplexity vs the run's own peak-reach ceiling and the 20% flag line
    pc = _panel_coverage(v)
    cov_leg, cov_leg_h = _legend_svg([(FLAT["accent"], "perplexity (effective states)"), (FLAT["fnt"], "distinct states / checkpoint"),
                                       (FLAT["mut"], "peak reach (max)"), (FLAT["crit"], "20% line")], 0, 0, 360)
    cov_hl = [(_peakv, f"peak reach {int(round(_peakv))}", FLAT["mut"]), (tau * _peakv, f"{int(round(tau*100))}% line", FLAT["crit"])] if _peakv > 0 else []
    cov_chart = _lc_svg([perp, nodes], [FLAT["accent"], FLAT["fnt"]], xpos=cps, xdom=xdom, ylab="effective states", ymin0=True, hlines=cov_hl)

    # Behavioural change -- shift channels vs their noise floor (only a frozen policy is a validated flag)
    pb = _panel_behaviour(v)
    shift_leg, shift_leg_h = _legend_svg([(FLAT["c1"], "topological shift"), (FLAT["c2"], "strategic shift"), (FLAT["c3"], "sequential shift"), ("dash", "noise floor")], 0, 0, 360)
    shift_chart = _lc_svg([Rn[0], Rn[1], Rn[2]], [FLAT["c1"], FLAT["c2"], FLAT["c3"]], floor=np.maximum.reduce(floor_n), xpos=xmid, xdom=xdom, yr=(0, 1), ticks=cps, ylab="shift (norm.)")

    # Footprint turnover -- per-checkpoint discovery / abandonment / restructure (or a placeholder if missing)
    pt = _panel_turnover(v)
    if not pt.get("available", True):
        turn_chart = _placeholder_svg(360, 120, "decomposition not available")
        turn_leg, turn_leg_h = "", 0.0
        turn_label = "—"
    else:
        turn_leg, turn_leg_h = _legend_svg([(FLAT["good"], "discovery"), (FLAT["crit"], "abandonment"), (FLAT["purple"], "restructure")], 0, 0, 360)
        turn_label = chip(pt)
        turn_chart = _stackts_svg([disc, aban, over], [FLAT["good"], FLAT["crit"], FLAT["purple"]], xpos=xmid, xdom=xdom, ylab="turnover", ticks=cps)

    # (name, trend chip, chip colour, status, why, chart, legend, legend height)
    rows = [("Reward", chip(pr), R_COLOR[pr["tone"]], pr["status"], pr["why"], rew_chart, rew_leg, rew_leg_h),
            ("State coverage", chip(pc), None, pc["status"], pc["why"], cov_chart, cov_leg, cov_leg_h),
            ("Behavioural change", chip(pb), None, pb["status"], pb["why"], shift_chart, shift_leg, shift_leg_h),
            ("Footprint turnover", turn_label, None, pt["status"], pt["why"], turn_chart, turn_leg, turn_leg_h)]

    PAD_X, PAD_TOP, PAD_BOT = 20, 42, 40
    content_w = min(width - 2 * PAD_X, 880)
    x0 = max((width - content_w) / 2.0, PAD_X)
    parts = []; y = PAD_TOP

    kicker = "BEHAVIORAL FINGERPRINT · REWARD-BLIND INTERPRETATION"
    parts.append(f'<text x="{x0:.1f}" y="{y:.1f}" font-family="{F_SANS}" font-size="11" font-weight="700" letter-spacing="1.2" fill="{FLAT["accent"]}">{_esc(kicker)}</text>')
    y += 24
    title_svg, title_h = _mixed_lines_svg(x0, y + 15, _wrap(title, content_w, 19, bold=True), 19, FLAT["ink"], line_h=26)
    parts.append(title_svg); y += title_h + 6
    run_svg, run_h = _mixed_lines_svg(x0, y + 9, _wrap(str(v.get("run", title)), content_w, 12, kind="mono"), 12, FLAT["mut"], kind="mono", line_h=16)
    parts.append(run_svg); y += run_h + 8
    meta = (f"{str(v.get('floor','hard'))} floor · {v['n_pairs']} checkpoint pairs · a reward-blind portrait read from the trajectories alone. "
            f"Two conditions raise a validated red flag (low effective coverage, frozen policy); amber chips mark observations worth a look; "
            f"the interpretation reads the panels together.")
    meta_svg, meta_h = _mixed_lines_svg(x0, y + 9, _wrap(meta, content_w, 12.5), 12.5, FLAT["mut"], line_h=18)
    parts.append(meta_svg); y += meta_h + 4

    for name, trend, color, status, why, chart, leg, leg_h in rows:
        row_svg, row_h = _row_svg(x0, y, content_w, chart, name, trend, color, status, why, legend_svg=leg, legend_h=leg_h)
        parts.append(row_svg); y += row_h

    if v["flags"]:
        parts.append(f'<text x="{x0:.1f}" y="{y+16:.1f}" font-family="{F_SANS}" font-size="10" font-weight="700" letter-spacing=".8" fill="{FLAT["mut"]}">FLAGS</text>')
        y += 24
        for f in v["flags"]:
            card_svg, card_h = _card_svg(x0, y, content_w, "▲", FLAT["crit"], FLAT["crit_bg"], FLAT["crit"], f["headline"], f["body"], border_opacity=.4)
            parts.append(card_svg); y += card_h + 8

    interp = v.get("interpretation", [])
    if interp:
        cap = "INTERPRETATION"
        parts.append(f'<text x="{x0:.1f}" y="{y+16:.1f}" font-family="{F_SANS}" font-size="10" font-weight="700" letter-spacing=".8" fill="{FLAT["mut"]}">{cap}</text>')
        cap_w = _fw(cap, "bold", 10) + len(cap) * 0.8  # + letter-spacing
        parts.append(f'<text x="{x0+cap_w+6:.1f}" y="{y+16:.1f}" font-family="{F_SANS}" font-size="10" fill="{FLAT["fnt"]}">(what the panels mean together)</text>')
        y += 24
        for f in interp:
            card_svg, card_h = _card_svg(x0, y, content_w, "◆", FLAT["accent"], FLAT["surface2"], FLAT["hair"], f["headline"], f["body"])
            parts.append(card_svg); y += card_h + 8

    total_h = y + PAD_BOT
    body = f'<rect x="0" y="0" width="{width}" height="{total_h:.1f}" fill="{FLAT["bg"]}"/>' + "".join(parts)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h:.1f}" viewBox="0 0 {width} {total_h:.1f}">{body}</svg>')

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
    ap.add_argument("--reachable", type=int, default=None, help="Reachable state count for the footprint axis (e.g. FrozenLake 64, Taxi 500, MountainCar bins^2). Omit to skip the footprint axis and its flags.")
    ap.add_argument("--no-figure", action="store_true", help="Deprecated no-op (reports are SVG-only now).")
    ap.add_argument("--title", default=None, help="Display title for the report H1 (defaults to the run name).")
    ap.add_argument("--tau", type=float, default=TAU_LOW, help=f"Footprint-flag threshold on effective coverage (default {TAU_LOW}, FL-calibrated). Pass an env-calibrated value if you have one.")
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
        # Universal rule: floor at the hard minimum, then tighten where the random
        # baseline is noisier. max() prevents the estimate from collapsing below the
        # meaningful-change floor (a near-stationary random policy -- e.g. MountainCar
        # stuck in the valley -- otherwise drives it to ~0 and nothing ever silences).
        raw = emin_from_random(a.random_dir, nact, a.M)
        emin = [float(x) for x in np.maximum(HARD_EMIN, raw)]
        print(f"[stage2] estimated epsilon_min = {emin}  (raw random 99pct = {[round(x, 4) for x in raw]}, floored at hard)")
    else:
        emin = list(HARD_EMIN); print(f"[stage2] hard epsilon_min = {emin}")
    v = interpret(d, emin, reachable=a.reachable, tau=a.tau)
    v["run"] = name; v["floor"] = a.floor
    json.dump(v, open(os.path.join(out, f"{name}_verdict.json"), "w"), indent=1)
    if v["n_pairs"] < 1:
        print(f"[stage2] {name}: insufficient data (single checkpoint) -- verdict only"); return
    if a.reachable is None:
        print("[stage2] note: no --reachable given -- coverage shown without a fraction; footprint flag omitted")
    title = a.title or name
    open(os.path.join(out, f"{name}_report.html"), "w", encoding="utf-8").write(render_report(d, v, title))
    open(os.path.join(out, f"{name}_report.svg"), "w", encoding="utf-8").write(render_report_svg(d, v, title))
    flags = "; ".join(f["headline"] for f in v["flags"]) or "no flags"
    print(f"[stage2] {v['trend']} return | {'settled' if v['settled'] else 'never settles'}"
          + (f" | coverage {100*v['footprint_frac']:.0f}%" if v['footprint_frac'] is not None else "")
          + f" | flags: {flags}")
    print(f"[stage2] wrote {name}_verdict.json + report.html + report.svg in {out}")

if __name__ == "__main__":
    main()
