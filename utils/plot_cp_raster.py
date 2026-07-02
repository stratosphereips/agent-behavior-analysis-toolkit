#!/usr/bin/env python3
"""Render the RQ1 fingerprint-vs-reward agreement raster from a CSV.

CSV format (header row required):

    env,model,seed,CP1,CP2,...,CPK

or the legacy two-column form:

    env,model_name,CP1,CP2,...,CPK

one row per run. Each CPx cell holds exactly one of:

    "BF Only"      fingerprint active, reward silent   (invisible restructuring)
    "Both active"  both detectors active               (agreement)
    "Reward only"  reward active, fingerprint silent    (recall miss)
    "None"         both silent
    ""  (empty)    no data for this checkpoint

Rows are grouped by env in first-seen order. Columns are checkpoints in header
order; runs with fewer checkpoints leave trailing cells empty (drawn as no-data,
which is visually distinct from a filled "None" cell).

Usage:
    python plot_rq1_raster.py agreement.csv -o rq1_raster.png
    python plot_rq1_raster.py agreement.csv --title "RQ1 agreement (Good Learning)"
"""

import argparse
import csv
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# State -> fill colour. Matches the working-draft sheet.
STATE_COLORS = OrderedDict([
    ("BF Only",     "#3B7DD8"),   # blue  — fingerprint active, reward silent
    ("Both active", "#5AA02C"),   # green — agreement
    ("Reward only", "#E8A020"),   # amber — reward active, fingerprint silent (miss)
    ("None",        "#4C008E"),   # grey  — both silent
])
STATE_LEGEND = {
    "BF Only":     "BF only  (fingerprint active, reward silent)",
    "Both active": "Both active  (agreement)",
    "Reward only": "Reward only  (reward active, fingerprint silent)",
    "None":        "None  (both silent)",
}
UNKNOWN_COLOR = "#E0219B"   # magenta: flags malformed cells loudly
NODATA_COLOR = "#C8C8C8"
NODATA_EDGE = "#BBBBBB"
CELL_EDGE = "#FFFFFF"

_CANON = {k.lower(): k for k in STATE_COLORS}


def canon_state(raw):
    """Map a raw cell string to a canonical state, None (no data), or '??'."""
    s = (raw or "").strip()
    if s == "":
        return None
    return _CANON.get(s.lower(), "??")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        sys.exit("empty CSV")
    header = [h.strip() for h in rows[0]]
    low = [h.lower() for h in header]
    env_i = low.index("env") if "env" in low else 0
    mdl_i = (low.index("model_name") if "model_name" in low
             else low.index("model") if "model" in low
             else 1)
    seed_i = low.index("seed") if "seed" in low else None
    meta = {env_i, mdl_i} | ({seed_i} if seed_i is not None else set())
    cp_cols = [i for i in range(len(header)) if i not in meta]
    cp_labels = [header[i] for i in cp_cols]

    runs = []
    for r in rows[1:]:
        r = r + [""] * (len(header) - len(r))          # pad ragged rows
        env = r[env_i].strip()
        mdl = r[mdl_i].strip()
        if seed_i is not None:
            seed = r[seed_i].strip()
            if seed:
                mdl = f"{mdl} ({seed})"
        states = [canon_state(r[i]) for i in cp_cols]
        runs.append((env, mdl, states))
    return runs, cp_labels


def main():
    ap = argparse.ArgumentParser(description="Render the RQ1 agreement raster from a CSV.")
    ap.add_argument("csv_path")
    ap.add_argument("-o", "--out", default="rq1_raster.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--group-gap", type=float, default=1.5,
                    help="vertical gap between environment groups, in row units")
    args = ap.parse_args()

    runs, cp_labels = read_csv(args.csv_path)
    ncol = len(cp_labels)
    if ncol == 0:
        sys.exit("no checkpoint columns found after env/model_name")

    # group by env, preserving first-seen order
    groups = OrderedDict()
    for env, mdl, states in runs:
        groups.setdefault(env, []).append((mdl, states))

    # assign y positions top-to-bottom, with a gap between env groups
    placed = []          # (y, model, states)
    env_spans = []       # (env, y0, y1)
    has_unknown = False
    y = 0.0
    for env, members in groups.items():
        y0 = y
        for mdl, states in members:
            placed.append((y, mdl, states))
            has_unknown = has_unknown or ("??" in states)
            y += 1.0
        env_spans.append((env, y0, y))
        y += args.group_gap
    total_h = y - args.group_gap

    if has_unknown:
        sys.stderr.write("warning: some cells did not match a known state; "
                         "they are drawn in magenta.\n")

    max_lbl = max((len(m) for _, m, _ in placed), default=4)
    model_x = -0.3
    env_x = model_x - 0.45 * max_lbl - 2.0
    xlim_left = env_x - 1.2

    data_w = ncol - xlim_left
    fig_w = max(7.0, data_w * 0.22)
    fig_h = max(2.5, 1.4 + total_h * 0.26)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # cells
    for y, mdl, states in placed:
        for j, st in enumerate(states):
            if st is None:                       # no data: grey filled cell
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor=NODATA_COLOR,
                                       edgecolor=NODATA_EDGE, linewidth=0.5))
            else:
                face = UNKNOWN_COLOR if st == "??" else STATE_COLORS[st]
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor=face,
                                       edgecolor=CELL_EDGE, linewidth=0.6))
        ax.text(model_x, y + 0.5, mdl, ha="right", va="center", fontsize=8)

    # env labels (rotated) + separators between groups
    for env, y0, y1 in env_spans:
        ax.text(env_x, (y0 + y1) / 2.0, env, ha="center", va="center",
                fontsize=9, fontweight="bold", rotation=90)
    for k in range(len(env_spans) - 1):
        y1 = env_spans[k][2]
        ax.axhline(y1 + args.group_gap / 2.0, color="#BBBBBB", linewidth=0.8)

    ax.set_xlim(xlim_left, ncol)
    ax.set_ylim(total_h, -0.2)                   # inverted: first row on top
    ax.set_aspect("equal")

    try:
        cp_ints = [int(lbl) for lbl in cp_labels]
        raw_step = max(1, (cp_ints[-1] - cp_ints[0]) // 10)
        step_val = max(5, ((raw_step + 4) // 5) * 5)
        ticks = [i for i, v in enumerate(cp_ints) if v % step_val == 0]
        if not ticks:
            ticks = [0]
    except (ValueError, TypeError):
        step = max(1, ncol // 15)
        ticks = list(range(0, ncol, step))
    ax.set_xticks([t + 0.5 for t in ticks])
    ax.set_xticklabels([cp_labels[t] for t in ticks], fontsize=7, rotation=90)
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if args.title:
        ax.set_title(args.title, fontsize=11)

    handles = [Patch(facecolor=STATE_COLORS[k], edgecolor=CELL_EDGE, label=STATE_LEGEND[k])
               for k in STATE_COLORS]
    handles.append(Patch(facecolor=NODATA_COLOR, edgecolor=NODATA_EDGE, label="no data"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=len(handles), fontsize=8, frameon=False, handlelength=1.2)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(placed)} runs, {ncol} checkpoints, "
          f"{len(env_spans)} environments)")


if __name__ == "__main__":
    main()