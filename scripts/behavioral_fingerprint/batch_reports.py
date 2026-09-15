"""Regenerate every behavioral-fingerprint report from its cached *_metrics.json using the
reward-blind interpreter. In-process (no per-run subprocess), overwrites *_report.html and
*_verdict.json in place, leaves *_fingerprint.png orphans untouched.

Coverage is self-contained: effective coverage is measured against the run's OWN peak reach
(perplexity_late / max(total_nodes), inferred from the trajectories), so no reachable-state count
is needed and the same line applies to every environment. n_eval (eval episodes per checkpoint)
is the only per-env constant, and only feeds the reward SEM band.

Usage:  python -m scripts.behavioral_fingerprint.batch_reports  [ROOT]  [--dry]
"""
import glob, json, os, re, sys, traceback
from scripts.behavioral_fingerprint.generate_report import interpret, render_report, render_report_svg, estimate_report_height, HARD_EMIN

ROOT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else r"C:\Users\ondra\Documents\metric_results"
DRY = "--dry" in sys.argv


def n_eval_for(env):
    # eval episodes per checkpoint -> the reward's standard-error band (std_return / sqrt(n_eval))
    return 1000 if env == "taxi" else 500


def title_for(rel):
    parts = re.split(r"[\\/]", rel)
    # metric_results/<env>/<agent>/<mode>/<seed>/<file> -> "env · agent · mode · seed"
    keep = [p for p in parts[:-1] if p]
    return " · ".join(keep) if keep else parts[-1]


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*_metrics.json"), recursive=True))
    print(f"found {len(files)} metrics.json under {ROOT}")
    ok = flagged = skipped = errored = 0
    for f in files:
        rel = os.path.relpath(f, ROOT)
        env = re.split(r"[\\/]", rel)[0]
        name = re.sub(r"(_\d+)?_metrics\.json$", "", os.path.basename(f)) or "run"
        out = os.path.dirname(f)
        title = title_for(rel)
        try:
            d = json.load(open(f, encoding="utf-8"))
            if "checkpoints" not in d and "checkpoint_ids" in d:
                d["checkpoints"] = d["checkpoint_ids"]
            v = interpret(d, list(HARD_EMIN), n_eval=n_eval_for(env))
            v["run"] = name; v["floor"] = "hard"
            if not DRY:
                json.dump(v, open(os.path.join(out, f"{name}_verdict.json"), "w"), indent=1)
            if v["n_pairs"] < 1:
                skipped += 1; continue
            if not DRY:
                html = render_report(d, v, title)
                open(os.path.join(out, f"{name}_report.html"), "w", encoding="utf-8").write(html)
                svg_height = estimate_report_height(v, title)
                open(os.path.join(out, f"{name}_report.svg"), "w", encoding="utf-8").write(render_report_svg(html, height=svg_height))
            ok += 1
            fl = [x["headline"] for x in v["flags"]]
            if fl: flagged += 1
            if fl:
                print(f"  {title}  [{env}]  -> {', '.join(fl)}")
        except Exception as e:
            errored += 1
            print(f"  !! {rel}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'[DRY] ' if DRY else ''}done: {ok} reports, {flagged} with a red flag, {skipped} insufficient, {errored} errored")


if __name__ == "__main__":
    main()
