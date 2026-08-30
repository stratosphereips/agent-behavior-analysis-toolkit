#!/usr/bin/env python3
"""Render the redesigned Behavioral Fingerprint evaluation report for a run.

One figure with four training-step-aligned panels plus a compact detector strip:

    1. Model performance     mean return +/- std band
    2. State coverage        perplexity PP(P_k) and visited-state count |V_k|
    3. Behavioral shifts      Delta_Topo / Delta_Strat / Delta_Seq, each vs. its
                              own per-metric noise floor (dashed), on a fixed [0,1] axis
       + detector strip       family-wise active/inactive per checkpoint pair, hue =
                              the metric driving the decision (largest standardized z)
    4. Topological decomposition   redistribution (line) + discovery/abandonment
                              (green/red bars) + net flux, the parts of Delta_Topo

Reads a run's ``*_metrics.json`` (as written by the analysis pipeline). The
detector uses the family-wise max-statistic when the null moments
(``null_mean_*`` / ``null_std_*``) and ``zmax_p95`` are present, otherwise it
falls back to per-metric floor crossings (raw > noise_threshold), labelled as
such on the strip.

NOTE on Delta_Seq normalization: the pipeline stores ``3-gram_wasserstein_raw``
as the *un-normalized* Earth Mover's Distance (see compute_ngram_wasserstein_fast
in utils/metrics.py, which returns ot.emd2 directly). Its theoretical maximum is
the largest Levenshtein cost between two n-grams, i.e. n. We divide by
WASS_MAX_COST (= n = 3) here so Delta_Seq is shown in [0,1], matching the paper's
definition. If metrics.py is changed to normalize internally, set
WASS_MAX_COST = 1.0. The detector is unaffected (z-score / floor-crossing are
scale-invariant).

Usage:
    # batch: every seed run under a results tree
    python -m utils.plot_fingerprint_report results/
    python -m utils.plot_fingerprint_report results/ --dpi 350 --overwrite
    python -m utils.plot_fingerprint_report results/ --store_dir figures/  # mirrors results/'s subdirs

    # single run
    python -m utils.plot_fingerprint_report path/to/run_metrics.json -o out.png

    # as a library
    from utils.plot_fingerprint_report import plot_fingerprint_report
    plot_fingerprint_report(metrics_dict, title="...", outpath="fig.png")
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, MaxNLocator

# n-gram order = max Levenshtein cost between two n-grams; the stored EMD is
# un-normalized, so dividing by this puts Delta_Seq in [0,1]. Set to 1.0 if the
# pipeline is changed to normalize the Wasserstein distance internally.
WASS_MAX_COST = 3.0

# Okabe-Ito colorblind-safe palette (validated: adjacent CVD dE >= 11).
C_TOPO, C_STRAT, C_SEQ = "#0072B2", "#D55E00", "#009E73"
C_RET, C_PP, C_VIS = "#333333", "#CC79A7", "#999999"
C_INACT, C_DISC, C_ABAN = "#e6e6e6", "#2ca02c", "#d62728"
INK, MUT, GRID = "#222222", "#666666", "#dddddd"

_RC = {
    "font.size": 10.5, "axes.edgecolor": MUT, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "xtick.color": MUT, "ytick.color": MUT,
    "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False, "legend.fontsize": 8.7,
}


def _arr(d, k):
    return np.array(d[k], dtype=float)


def plot_fingerprint_report(d, title, outpath, dpi=350, wass_max_cost=WASS_MAX_COST, emin=None):
    """Render one run's fingerprint report to ``outpath``.

    ``d`` is the parsed ``*_metrics.json`` dict.
    ``emin``: optional practical-significance gate as [eps_topo, eps_strat, eps_seq] in
    raw metric units (seq/wass un-normalized). When given, a checkpoint pair is active
    only if it is statistically significant AND the driving metric's raw value exceeds
    its epsilon_min. Calibrate epsilon_min as the ~99th percentile of the metric on a
    non-learning (Random) policy, or use the fixed universal 0.05 (normalized). None ->
    significance only (original behavior).
    """
    with plt.rc_context(_RC):
        xck = _arr(d, "checkpoints")
        xpr = xck[1:]
        topo, strat, seq = (_arr(d, "topological_shift_raw"), _arr(d, "strategic_shift_raw"),
                            _arr(d, "3-gram_wasserstein_raw"))
        topo_t = _arr(d, "topological_shift_noise_threshold")
        strat_t = _arr(d, "strategic_shift_noise_threshold")
        seq_t = _arr(d, "3-gram_wasserstein_noise_threshold")
        disc, aban = _arr(d, "topological_shift_discovery_raw"), _arr(d, "topological_shift_abandonment_raw")
        net, overlap = _arr(d, "topological_shift_net_raw"), _arr(d, "topological_shift_overlap_raw")
        perp, visited = _arr(d, "state_visitation_perplexity"), _arr(d, "total_nodes")
        ret, sret = _arr(d, "mean_return"), _arr(d, "std_return")
        metric_colors = np.array([C_TOPO, C_STRAT, C_SEQ])

        # ---- detector: family-wise z when available, else per-metric floor crossing ----
        zf = ["null_mean_topological_shift", "null_std_topological_shift", "null_mean_strategic_shift",
              "null_std_strategic_shift", "null_mean_3-gram_wasserstein", "null_std_3-gram_wasserstein", "zmax_p95"]
        # sig_m: per-metric significance flags (3 x P); score: per-metric hue score.
        if all(k in d for k in zf):
            zt = (topo - _arr(d, "null_mean_topological_shift")) / _arr(d, "null_std_topological_shift")
            zs = (strat - _arr(d, "null_mean_strategic_shift")) / _arr(d, "null_std_strategic_shift")
            zq = (seq - _arr(d, "null_mean_3-gram_wasserstein")) / _arr(d, "null_std_3-gram_wasserstein")
            score = np.where(np.isnan(np.vstack([zt, zs, zq])), -np.inf, np.vstack([zt, zs, zq]))
            # family-wise (max-statistic) threshold applied per metric = Westfall-Young
            sig_m = score > _arr(d, "zmax_p95")[None, :]
            det_note = "gray = inactive; hue = metric\ndriving the decision (largest $z$)"
        else:
            R = np.vstack([topo / np.where(topo_t > 0, topo_t, np.nan),
                           strat / np.where(strat_t > 0, strat_t, np.nan),
                           seq / np.where(seq_t > 0, seq_t, np.nan)])
            score = np.where(np.isnan(R), -np.inf, R)
            sig_m = np.vstack([topo > topo_t, strat > strat_t, seq > seq_t])
            det_note = "gray = inactive; hue = metric\nmost above its floor (no $z$ available)"

        # practical-significance gate, applied PER METRIC: a pair fires if ANY metric is
        # both significant AND has raw change above its epsilon_min. Gating only the
        # largest-z metric would wrongly silence a pair whose real, significant change
        # sits in a different metric than the z-argmax.
        raw3 = np.vstack([topo, strat, seq])
        if emin is not None:
            pass_m = sig_m & (raw3 > np.asarray(emin, float)[:, None])
            det_note += "\n(gated: raw $>\\epsilon_{\\min}$)"
        else:
            pass_m = sig_m
        active = pass_m.any(axis=0)
        # hue = the max-score metric among those that passed (arbitrary when inactive)
        masked = np.where(pass_m, score, -np.inf)
        zarg = np.where(active, masked.argmax(axis=0), score.argmax(axis=0))

        fig = plt.figure(figsize=(9.2, 11.4))
        gs = GridSpec(5, 1, height_ratios=[1.0, 0.9, 1.55, 0.22, 1.05], hspace=0.30,
                      left=0.10, right=0.79, top=0.945, bottom=0.06)
        axR, axP, axS, axA, axT = (fig.add_subplot(gs[i]) for i in range(5))
        for ax in (axP, axS, axA, axT):
            ax.sharex(axR)

        def legend_right(ax):
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.4, borderaxespad=0)

        axR.fill_between(xck, ret - sret, ret + sret, color=C_RET, alpha=0.15, linewidth=0)
        axR.plot(xck, ret, color=C_RET, lw=2, marker="o", ms=3.5, label="Mean return")
        axR.set_ylabel("Return"); axR.set_title("Model performance", loc="left", fontsize=10.5, color=MUT)
        legend_right(axR)

        axP.plot(xck, visited, color=C_VIS, lw=1.6, ls="--", label=r"Visited states $|V_k|$")
        axP.plot(xck, perp, color=C_PP, lw=2, marker="o", ms=3.5, label=r"Perplexity $\mathrm{PP}(\hat P_k)$")
        axP.set_ylabel("States"); axP.set_ylim(0, None)
        axP.set_title("State coverage", loc="left", fontsize=10.5, color=MUT)
        legend_right(axP)

        for raw, thr, c, lab in [(topo, topo_t, C_TOPO, r"$\Delta_{\mathrm{Topo}}$"),
                                 (strat, strat_t, C_STRAT, r"$\Delta_{\mathrm{Strat}}$"),
                                 (seq / wass_max_cost, seq_t / wass_max_cost, C_SEQ, r"$\Delta_{\mathrm{Seq}}$")]:
            axS.plot(xpr, raw, color=c, lw=2, marker="o", ms=3.2, label=lab)
            axS.plot(xpr, thr, color=c, lw=1.1, ls=(0, (3, 2)))
        axS.plot([], [], color=MUT, lw=1.1, ls=(0, (3, 2)), label="per-metric noise floor")
        axS.set_ylabel("Behavioral shift  [0,1]"); axS.set_ylim(0, 1.05); axS.set_yticks(np.arange(0, 1.01, 0.2))
        axS.set_title("Behavioral shifts vs. their noise floors", loc="left", fontsize=10.5, color=MUT)
        legend_right(axS)

        step = xck[1] - xck[0] if len(xck) > 1 else 1
        w = max((xck[-1] - xck[0]) / len(xck) * 0.85, 1)
        for i, x in enumerate(xpr):
            axA.bar(x, 1.0, width=w, color=(metric_colors[zarg[i]] if active[i] else C_INACT),
                    edgecolor="white", linewidth=0.5)
        axA.set_ylim(0, 1); axA.set_yticks([]); axA.grid(False)
        axA.set_ylabel("Detector", rotation=0, ha="right", va="center", labelpad=8, color=INK)
        for s in ("left", "right", "top", "bottom"):
            axA.spines[s].set_visible(False)
        axA.tick_params(axis="both", length=0)
        axA.text(1.01, 0.5, det_note, transform=axA.transAxes, va="center", ha="left", fontsize=8.2, color=MUT)

        axT.bar(xpr, disc, width=w, color=C_DISC, alpha=0.8, label="discovery ($+$)")
        axT.bar(xpr, -aban, width=w, color=C_ABAN, alpha=0.8, label="abandonment ($-$)")
        axT.plot(xpr, overlap, color=C_TOPO, lw=1.8, marker="o", ms=3, label="redistribution")
        axT.plot(xpr, net, color=INK, lw=1.4, ls=(0, (4, 2)), label="net flux")
        axT.axhline(0, color=MUT, lw=0.8)
        axT.set_ylabel(r"$\Delta_{\mathrm{Topo}}$ components")
        axT.set_title("Topological decomposition: redistribution vs. frontier turnover", loc="left",
                      fontsize=10.5, color=MUT)
        axT.set_xlabel("Training step")
        legend_right(axT)

        for ax in (axR, axP, axS):
            plt.setp(ax.get_xticklabels(), visible=False)
        axT.set_xlim(xck[0] - step * 0.6, xck[-1] + step * 0.6)
        axT.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
        axT.xaxis.set_minor_locator(MultipleLocator(step))
        for ax in (axR, axP, axS, axT):
            ax.tick_params(axis="x", which="minor", length=3, color=MUT)
            ax.tick_params(axis="x", which="major", length=6, color=MUT)

        fig.suptitle(title, x=0.10, ha="left", fontsize=13, fontweight="bold")
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _is_seed_run(path):
    parts = os.path.normpath(path).split(os.sep)
    if any("_bak" in p for p in parts):          # skip backup folders
        return False
    return any(re.fullmatch(r"seed_?\d+", p) for p in parts)   # seed_1 or seed1


def _title_from_path(path, root):
    rel = os.path.relpath(os.path.dirname(path), root)
    if rel in (".", ""):                        # single-file mode: no meaningful root
        parts = os.path.normpath(os.path.dirname(os.path.abspath(path))).split(os.sep)
        parts = parts[parts.index("results") + 1:] if "results" in parts else parts[-4:]
        return "  ·  ".join(parts)
    return "  ·  ".join(rel.split(os.sep))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render Behavioral Fingerprint report(s).")
    ap.add_argument("path", help="a results directory (batch over every seed run) or a single *_metrics.json")
    ap.add_argument("-o", "--out", help="output PNG (single-file mode only; default: alongside the JSON)")
    ap.add_argument("--store_dir", help="directory to write rendered figures into, instead of alongside "
                    "each source JSON (batch mode preserves the directory structure under path within it)")
    ap.add_argument("--dpi", type=int, default=350)
    ap.add_argument("--overwrite", action="store_true", help="re-render figures that already exist")
    args = ap.parse_args(argv)

    if os.path.isfile(args.path):
        d = json.load(open(args.path))
        if args.out:
            out = args.out
        elif args.store_dir:
            os.makedirs(args.store_dir, exist_ok=True)
            out = os.path.join(args.store_dir, os.path.basename(args.path).replace("_metrics.json", "_fingerprint.png"))
        else:
            out = args.path.replace("_metrics.json", "_fingerprint.png")
        plot_fingerprint_report(d, _title_from_path(args.path, os.path.dirname(args.path)), out, dpi=args.dpi)
        print("wrote", out)
        return

    root = os.path.abspath(args.path)
    files = [f for f in glob.glob(os.path.join(root, "**", "*_metrics.json"), recursive=True)
             if _is_seed_run(f)]
    print(f"seed runs found: {len(files)}")
    ok = skip = fail = 0
    for f in sorted(files):
        if args.store_dir:
            rel_dir = os.path.relpath(os.path.dirname(f), root)
            out_dir = os.path.join(args.store_dir, rel_dir) if rel_dir != "." else args.store_dir
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = os.path.dirname(f)
        out = os.path.join(out_dir, os.path.basename(f).replace("_metrics.json", "_fingerprint.png"))
        if os.path.exists(out) and not args.overwrite:
            skip += 1
            continue
        try:
            plot_fingerprint_report(json.load(open(f)), _title_from_path(f, root), out, dpi=args.dpi)
            ok += 1
        except Exception as e:
            fail += 1
            print("FAIL:", os.path.relpath(f, root), "->", repr(e))
    print(f"done: {ok} rendered, {skip} skipped (exists), {fail} failed")


if __name__ == "__main__":
    main()
