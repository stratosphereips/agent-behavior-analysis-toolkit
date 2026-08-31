# Behavioral-fingerprint report

Turns a run's evaluation trajectories into a plain-language **verdict** + a per-run
report. Reward-blind: behavior decides whether learning is happening; the return curve
is consulted only to flag reward-behavior mismatches. See `NOISE_FLOOR_NOTES.md`
for the calibration behind the numbers, and `decision_tree.html` for the decision logic.

## Usage
Run from the toolkit root with the toolkit on the path:

```bash
cd agent-behavior-analysis-toolkit
PYTHONPATH="$PWD" python -m scripts.behavioral_fingerprint.generate_report RUN_DIR [options]
```

`RUN_DIR` is a folder of `cp_*.jsonl` (searched recursively).

Options:
- `--out DIR`         where to write outputs (default: RUN_DIR)
- `--M 200`           split-half null draws for the noise floor (Stage 1)
- `--num_actions N`   action-space size (default: inferred by scanning all checkpoints)
- `--floor hard|estimated`   practical-significance floor (default: hard)
- `--random_dir DIR`  a random-policy run; required for `--floor estimated`
- `--force`           recompute the metrics even if cached

Needs the toolkit runtime deps (`ot`/POT, `ruptures`, `scikit-learn`, `networkx`,
`scipy`, `numpy`, `matplotlib`).

## Outputs (by-products)
Two stages. Stage 1 is the only expensive step and is cached; Stage 2 is cheap and
re-derivable (switching `--floor` re-runs only Stage 2).

| file | stage | what it is |
|------|-------|------------|
| `<run>_metrics.json`     | 1 (cached) | the metrics: return, coverage (perplexity, \|V\|), the 3 Delta metrics, the Delta_Topo decomposition, the noise floors + null mean/std + zmax_p95 |
| `<run>_verdict.json`     | 2 | structured diagnosis: mode, sub_state, per-signal results, decomposition shares, the epsilon_min + floor used |
| `<run>_fingerprint.png`  | 2 | the 5-panel fingerprint figure (epsilon_min-gated detector) |
| `<run>_report.html`      | 2 | self-contained verdict card + figure |

Because epsilon_min is a Stage-2 parameter, re-generating with a different floor is cheap
(no floor recompute) -- only `metrics.json` costs real time.

## The floor (epsilon_min)
The practical-significance floor, applied per metric (on top of statistical significance):
- **hard** (default): a fixed 0.05 of maximal divergence -- universal, no baseline needed.
- **estimated** (`--random_dir`): the 99th percentile of a random policy's change in that
  env -- more sensitive in quiet environments, needs a random baseline.

The floor only affects the detector / activity signals (convergence axis); it does not
change the mode classification, which rests on the learner gate + the return-vs-coverage
relationship. See the `Hard vs Estimated Floor` comparison for the difference.

## calibration/  (provenance -- run once, not needed to use the report)
- `emin_cv_fast.py` -- calibrated the per-env epsilon_min (99th pct of Random raw),
  the leave-one-seed-out held-out FPR, and the single-global-epsilon sweep.
- `sens_flmc.py`    -- sensitivity check: fraction of learning-phase fires that clear
  epsilon_min, on the FrozenLake / MountainCar Good runs.
These produced the numbers baked into `generate_report.py`'s hard floor and the notes.
