#!/usr/bin/env python3
"""RQ1 aggregate refresh: one reproducible source of truth for every Finding 1/2/3 number and the
per-cell table (Section 6.3), computed from the canonical two-detector per-pair decisions of
scripts/two_detector_decisions.py (fp_active_maxstat + return_changed). Anchors: total learner
pairs 2454, fp-only 1058, return-only 16 (matches rq1_dose_response.py / rq1_fpr_match.py).

Decisions are taken per pair directly (no downstream two-consecutive-pair smoothing), so the numbers
here are fully reproducible from the metrics on disk.

USAGE: python scripts/rq1_refresh.py [--results_root results]
"""
import argparse, glob, hashlib, json, math, os
import numpy as np

Z_CRIT = 1.9599639845400545
EVO = ["topological_shift", "strategic_shift", "3-gram_wasserstein"]
ENV_N = {"Taxi": 1000, "FrozenLake": 500, "MountainCar": 500}
ENV_DIR = {"FrozenLake": "frozen_lake8x8", "MountainCar": "mountain_car", "Taxi": "taxi"}
ALGOS = ["q_learning", "sarsa", "dqn", "ppo"]
ALAB = {"q_learning": "Q-Learning", "sarsa": "SARSA", "dqn": "DQN", "ppo": "PPO"}
SEEDS = ["seed_1", "seed_2", "seed_3", "seed_4", "seed_5", "seed_4242"]
EXCLUDE = {("Taxi", "ppo")}                      # return never active
NON_STATIONARY = ("reward_hacking", "perpetual", "limited")
EXCL_TOK = ("30K", "bins15", "bins30", "_bak")   # training-length / binning variants, not the 6-seed design
_SEEN = set()


def _excluded(path):
    return any(t in path for t in EXCL_TOK)


def _sig(path):
    """(mean_return, perplexity) hash, to collapse flat/nested storage copies of the SAME run."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return hashlib.md5((json.dumps(d.get("mean_return")) + json.dumps(d.get("state_visitation_perplexity"))).encode()).hexdigest()


def return_changed(m1, s1, m2, s2, n):
    se2 = s1 * s1 / n + s2 * s2 / n
    if se2 <= 0.0:
        return m1 != m2
    return abs((m2 - m1) / math.sqrt(se2)) > Z_CRIT


def fp_active(d, i):
    zp = d.get("zmax_p95")
    if not (isinstance(zp, list) and i < len(zp) and zp[i] is not None):
        return None
    zs = []
    for k in EVO:
        raw, mu, sd = d.get(k + "_raw"), d.get("null_mean_" + k), d.get("null_std_" + k)
        if not all(isinstance(x, list) and i < len(x) for x in (raw, mu, sd)):
            return None
        if mu[i] is None or sd[i] is None or sd[i] == 0:
            continue
        z = (raw[i] - mu[i]) / sd[i]
        if math.isfinite(z):
            zs.append(z)
    return (max(zs) > zp[i]) if zs else False


def decisions(path, n):
    d = json.load(open(path))
    mr, sr = d.get("mean_return"), d.get("std_return")
    npairs = len(d.get("topological_shift_raw", []))
    if npairs == 0 or not isinstance(mr, list) or len(mr) < npairs + 1:
        return None
    fp, ret = [], []
    for i in range(npairs):
        fa = fp_active(d, i)
        if fa is None:
            return None
        fp.append(bool(fa))
        ret.append(bool(return_changed(mr[i], sr[i], mr[i + 1], sr[i + 1], n)))
    return np.array(fp), np.array(ret)


def learner_seeds(root, env, algo):
    """One run per seed dir (storage duplicates of the same run collapse to the first)."""
    for s in SEEDS:
        fs = sorted(f for f in glob.glob(os.path.join(root, ENV_DIR[env], algo, "standard", s, "*_metrics.json"))
                    if not _excluded(f))
        for f in fs:
            r = decisions(f, ENV_N[env])
            if r is not None:
                yield s, r
                break                      # one run per seed dir


def random_pairs(root, env):
    """Random-policy floor: unique stationary random runs (dedup storage copies within this env)."""
    fp_tot = fp_act = 0; seen = set(); run_rates = []
    for f in sorted(glob.glob(os.path.join(root, ENV_DIR[env], "random", "**", "*_metrics.json"), recursive=True)):
        if any(t in f for t in NON_STATIONARY) or _excluded(f):
            continue
        sig = _sig(f)
        if sig is None or sig in seen:
            continue
        seen.add(sig)
        r = decisions(f, ENV_N[env])
        if r is None:
            continue
        fp, _ = r
        fp_tot += len(fp); fp_act += int(fp.sum()); run_rates.append(100 * fp.mean())
    return fp_act, fp_tot, run_rates


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--results_root", default=r"C:\Users\ondra\Documents\metric_results")
    args = ap.parse_args()
    root = args.results_root

    tot_pairs = tot_fp = tot_fponly = tot_retonly = tot_conf = 0
    leads = []; zero_lead = 0; nruns = 0
    print(f"{'env':12}{'algo':11}{'FP-active':>16}{'per-seed %':>14}{'FP-only c/tot':>15}{'lead med(max)':>15}")
    env_fp = {e: [0, 0] for e in ENV_DIR}
    env_learner_seed_min = {e: 1e9 for e in ENV_DIR}
    for env in ENV_DIR:
        for algo in ALGOS:
            if (env, algo) in EXCLUDE:
                continue
            cell_fp = cell_tot = cell_fpo = cell_conf = 0; seed_pcts = []; cell_leads = []
            for s, (fp, ret) in learner_seeds(root, env, algo):
                nruns += 1
                cell_fp += int(fp.sum()); cell_tot += len(fp)
                seed_pcts.append(100 * fp.mean())
                env_learner_seed_min[env] = min(env_learner_seed_min[env], 100 * fp.mean())
                env_fp[env][0] += int(fp.sum()); env_fp[env][1] += len(fp)
                fponly = fp & ~ret; retonly = ~fp & ret
                cell_fpo += int(fponly.sum())
                tot_fponly += int(fponly.sum()); tot_retonly += int(retonly.sum())
                tot_pairs += len(fp); tot_fp += int(fp.sum())
                # confirmed: a fp-only pair with a later return-active pair in the same run
                later_ret = np.concatenate([np.cumsum(ret[::-1])[::-1][1:], [0]]) > 0
                conf = int((fponly & later_ret).sum()); cell_conf += conf; tot_conf += conf
                # lead: (first return-active - first fp-active)/npairs, only if the return ever fires
                if ret.any() and fp.any():
                    fr = int(np.argmax(ret)); ff = int(np.argmax(fp))
                    lead = (fr - ff) / len(fp); cell_leads.append(lead); leads.append(lead)
                    if abs(lead) < 1e-9:
                        zero_lead += 1
            if cell_tot == 0:
                continue
            pr = f"{min(seed_pcts):.0f}-{max(seed_pcts):.0f}" if len(seed_pcts) > 1 and min(seed_pcts) != max(seed_pcts) else f"{seed_pcts[0]:.0f}"
            lead_s = f"{np.median(cell_leads):+.2f}({max(cell_leads):+.2f})" if cell_leads else "-"
            print(f"{env:12}{ALAB[algo]:11}{f'{cell_fp}/{cell_tot}':>16}{pr:>14}{f'{cell_conf}/{cell_fpo}':>15}{lead_s:>15}")

    print("\n--- Finding 1: learner fp-active vs Random floor (per environment) ---")
    for env in ENV_DIR:
        fa, ft = env_fp[env]
        rfa, rft, rrates = random_pairs(root, env)
        floor = 100 * rfa / rft if rft else float("nan")
        rmax = max(rrates) if rrates else float("nan")
        print(f"  {env:12} learners {fa}/{ft} = {100*fa/ft:.0f}%   Random floor {rfa}/{rft} = {floor:.0f}%   "
              f"ratio {100*fa/ft/floor:.1f}x   [least learner seed {env_learner_seed_min[env]:.0f}% vs most-active Random run {rmax:.0f}%]")

    print("\n--- Finding 2: information gain (all learner pairs) ---")
    print(f"  pairs={tot_pairs}  fp-only={tot_fponly} ({100*tot_fponly/tot_pairs:.0f}%)  "
          f"return-only={tot_retonly} ({100*tot_retonly/tot_pairs:.1f}%)  confirmed={tot_conf} "
          f"({100*tot_conf/tot_fponly:.0f}% of fp-only)  unconfirmed={tot_fponly-tot_conf}")

    print("\n--- Finding 3: lead time (learning runs) ---")
    leads = np.array(leads)
    print(f"  runs={nruns}  zero-lead={zero_lead}  return-first={int((leads<-1e-9).sum())}  "
          f"max lead={leads.max():+.2f}")
    print(f"  ANCHOR check -> pairs 2454, fp-only 1058, return-only 16 (expect match)")


if __name__ == "__main__":
    main()
