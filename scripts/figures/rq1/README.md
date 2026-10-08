# RQ1 figures & table (Behavioral Ontogeny paper)

Scripts that produce the RQ1 exhibits: sensitivity of the Behavioral Fingerprint
to policy restructuring during reward-flat periods, read against the non-learning
Random-policy false-positive floor.

Each script reads the per-checkpoint metric JSONs and writes a figure (or prints
the table). All share the same two-detector logic:

- **Fingerprint active** at a checkpoint pair: `max_m z_m > zmax_p95`, where each
  `z_m` is the shift metric against its paired-permutation null (`null_mean_*`,
  `null_std_*`). Independent of the evaluation rollout count `N`.
- **Return active** (the baseline): a z-test on consecutive mean returns,
  `|Δr̄| / √(σ²_{k-1}/N + σ²_k/N) ≥ 1.96`. Depends on `N`.
- **Reward-flat** pair = return not active. **Converged tail** = pairs after the
  last checkpoint where either detector fires (excluded from the sensitivity rate).

## Data layout (`ROOT`)

`ROOT` is the metric data root. Learners live at
`ROOT/<env>/<algo>/standard/seed_*.json`; the NetSecGame Random control uses the
trajectories layout `ROOT/netsecgame/random/seed_*/trajectories/*_metrics.json`.
Evaluation rollout count `N` per environment: Taxi 1000, FrozenLake-8x8 500,
MountainCar 500, NetSecGame 500.

## Scripts

| Script | Output | Purpose |
|---|---|---|
| `rq1_panelA.py` | figure | Two worked MountainCar runs (PPO flat start, SARSA convergence): return with reward-flat shading vs the three shifts as a ratio to their noise floor. |
| `rq1_sensitivity.py` | figure | Per-algorithm detection rate on reward-flat, non-converged pairs, faceted by environment, against the Random-policy FPR floor (dashed line). |
| `rq1_leadstrip.py` | figure | Per-run lead time: training-fraction gap between first return change and first fingerprint activity (positive = fingerprint first). |
| `rq1_table.py` | stdout | Per-cell two-detector table (FP-active, FP-only conf/total, Return-only, Both-silent learn/settled) pooled over seeds, incl. the Random floor. |

## Usage

```bash
ROOT=/path/to/metric_06102026

python rq1_panelA.py      "$ROOT" rq1_panelA.png
python rq1_sensitivity.py "$ROOT" rq1_sensitivity.png
python rq1_leadstrip.py   "$ROOT" rq1_leadstrip.png
python rq1_table.py       "$ROOT"      # prints the table
```

Requires `matplotlib` (Agg backend; no display needed).
