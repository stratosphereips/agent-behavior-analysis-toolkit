# Experiment Specifications (as-run)

> **Source of truth:** this file is reconstructed from the actual result filenames in
> `results/<env>/<algorithm>/<mode>/*_metrics.json`, which encode the hyperparameters each
> run was launched with. It **supersedes** the older per-environment specs
> (`frozenlake/frozenlake.md`, `mountain_car/mountain_car.md`, `taxi/taxi.md`), which describe
> *intended* values that in several cases do not match what was actually run.
>
> **Legend**
> - **NOT FOUND** — no result file exists for this cell; run pending.
> - **§** — non-canonical induction (sustained exploration/entropy instead of the
>   maximal-step-size setting), because `alpha=1.0` / aggressive clip destabilised estimation.
> - "baseline" — same configuration as that algorithm's Standard (Good Learning) run.
>
> **Mode name mapping** (folder name → paper term):
> `standard` = Good Learning · `limited_exploration` = Exploration Deprivation ·
> `perpetual_reshaping` = Perpetual Reshaping (a.k.a. Oscillation) · `reward_hacking` = Reward Gaming.

---

## Discrepancies vs. the old specs (read before trusting the old `.md` files)

- **Training budgets were larger than documented.** Actual: MountainCar **30,000** episodes,
  checkpoint every **1,000** (30 checkpoints); Taxi **5,000** episodes, checkpoint every **250**
  (20 checkpoints). The old docs said 10,000/500 and 2,000/100 respectively.
- **MountainCar DQN** actually used `lr=1e-4`, `memory=200000`, `target_update_every=2000`
  (old doc: `lr=1e-3`, `memory=20000`, `update=1000`).
- **MountainCar PPO** actually used `entropy_coef=0.1` (old doc: `0.01`).
- **Perpetual Reshaping was not induced with `alpha=1.0` everywhere.** Only Taxi SARSA used the
  canonical `alpha=1.0`; Taxi DQN used the canonical per-step target sync. MountainCar tabular and
  both PPO perpetual runs used sustained exploration/entropy instead (marked **§**).
- **Exploration Deprivation used optimistic `q_init=0`** in every value-based run. No
  pessimistic-init (`q_init << r_min`) runs exist yet, so the intended "true deprivation" variant
  is **NOT FOUND**.
- **FrozenLake-8x8 has no result files at all** — every FrozenLake cell below is **NOT FOUND**.

---

## Shared settings

| Setting | MountainCar-v0 | Taxi-v3 | FrozenLake-8x8 |
|---|---|---|---|
| State space | 2 continuous → 20×20 bins (400) | 500 discrete (factored) | 64 discrete |
| Actions | 3 | 6 | 4 |
| Discount γ | 0.99 | 0.99 | NOT FOUND (intended 0.95) |
| Tabular `q_init` | 0.0 | 0.0 | NOT FOUND |
| DQN / PPO hidden layers | [64, 64] | [64, 64] | NOT FOUND |
| Env variant | deterministic dynamics | stochastic (`is_rainy=0.9`, `fickle_passenger=0.1`) | slippery (P_success=1/3) |
| Training episodes | 30,000 | 5,000 | NOT FOUND |
| Checkpoint interval | 1,000 (30 ckpts) | 250 (20 ckpts) | NOT FOUND |
| Evaluation (ε=0) | N=500 trajectories/ckpt | N=1000 trajectories/ckpt | N=500 (intended) |

Reward Gaming proxy rewards (induced via environment wrapper, algorithm config unchanged):
- **MountainCar** — `+1.5` left-push shaping bonus (net `+0.5`/step) ⇒ farm the bonus, never finish.
- **Taxi** — illegal drop-off pays `+10` instead of `−10` ⇒ repeatedly mis-deliver.
- **FrozenLake** — `+0.05` safe-ice survival bonus (intended) ⇒ farm survival. NOT FOUND.

---

## MountainCar-v0 (as-run)

| Algorithm | Mode | Hyperparameters (as run) |
|---|---|---|
| **Q-Learning** | Good Learning | `alpha=0.1`, `ε:1.0→0.0` (decay 0.9995) |
| | Exploration Deprivation | `ε=0.01` fixed (decay 0, `ε_min=0.01`); optimistic `q_init=0` |
| | Perpetual Reshaping **§** | `alpha=0.1`, `ε:1.0→0.05` (decay 0.999) — sustained exploration |
| | Reward Gaming | baseline + left-push proxy reward |
| **SARSA** | Good Learning | `alpha=0.1`, `ε:1.0→0.0` (decay 0.9995) |
| | Exploration Deprivation | `ε=0.0` fixed; optimistic `q_init=0` |
| | Perpetual Reshaping **§** | `alpha=0.1`, `ε:1.0→0.05` (decay 0.999) — sustained exploration |
| | Reward Gaming | baseline + left-push proxy reward |
| **DQN** | Good Learning | `lr=1e-4`, `batch=256`, `memory=200000`, `target_update_every=2000`, `replay_each=4`, `ε:1.0→0.0` over 500,000 steps |
| | Exploration Deprivation | `ε=0.01` fixed (`ε_min=0.01`); else baseline |
| | Perpetual Reshaping | **NOT FOUND** |
| | Reward Gaming | baseline + left-push proxy reward |
| **PPO** | Good Learning | `lr=3e-4`, `clip=0.2`, `entropy_coef=0.1` (decay_frac 0.8, `entropy_min=0.0`) |
| | Exploration Deprivation | `entropy_coef=0.0` (decay_frac 0.01) |
| | Perpetual Reshaping **§** | `entropy_coef=0.1` (decay_frac 0.5, `entropy_min=0.05`) — sustained entropy |
| | Reward Gaming | baseline + left-push proxy reward |
| **Random** | (any) | **NOT FOUND** (no MountainCar random baseline run) |

---

## Taxi-v3 (as-run)

| Algorithm | Mode | Hyperparameters (as run) |
|---|---|---|
| **Q-Learning** | Good Learning | `alpha=0.1`, `ε:1.0→0.01` (decay 0.995) |
| | Exploration Deprivation | `ε=0.01` (decay 0.99, `ε_min=0.0`); optimistic `q_init=0` |
| | Perpetual Reshaping | **NOT FOUND** |
| | Reward Gaming | baseline + illegal-drop-off proxy reward |
| **SARSA** | Good Learning | `alpha=0.1`, `ε:1.0→0.01` (decay 0.995) |
| | Exploration Deprivation | `ε=0.0` (decay 0.99); optimistic `q_init=0` |
| | Perpetual Reshaping | `alpha=1.0`, `ε:1.0→0.01` (decay 0.995) — canonical aggressive update |
| | Reward Gaming | baseline + illegal-drop-off proxy reward |
| **DQN** | Good Learning | `lr=1e-3`, `batch=64`, `memory=50000`, `target_update_every=500`, `replay_each=4`, `ε=1.0` (`epsilon_decay=None`, `ε_min=0.01`) |
| | Exploration Deprivation | `ε_min=0.0`; else baseline |
| | Perpetual Reshaping | `lr=5e-3`, `target_update_every=1`; else baseline — canonical |
| | Reward Gaming | baseline + illegal-drop-off proxy reward |
| **PPO** | Good Learning | `lr=1e-4`, `clip=0.2`, `entropy_coef=0.2` (decay_frac 0.8, `entropy_min=0.05`) |
| | Exploration Deprivation | `lr=3e-4`, `entropy_coef=0.0` |
| | Perpetual Reshaping | **NOT FOUND** |
| | Reward Gaming | `lr=3e-4`, `entropy_coef=0.01` + illegal-drop-off proxy reward |
| **Random** | Good Learning | uniform policy `π(a\|s)=1/|A|`; no trainable parameters |
| | Reward Gaming | uniform policy; no trainable parameters + illegal-drop-off proxy reward |

---

## FrozenLake-8x8 (as-run)

**NOT FOUND** — no result files exist for any (algorithm, mode) cell. Intended specs are documented
in `frozenlake/frozenlake.md` but have not been validated against runs.

| Algorithm | Good Learning | Exploration Deprivation | Perpetual Reshaping | Reward Gaming |
|---|---|---|---|---|
| Q-Learning | NOT FOUND | NOT FOUND | NOT FOUND | NOT FOUND |
| SARSA | NOT FOUND | NOT FOUND | NOT FOUND | NOT FOUND |
| DQN | NOT FOUND | NOT FOUND | NOT FOUND | NOT FOUND |
| PPO | NOT FOUND | NOT FOUND | NOT FOUND | NOT FOUND |
| Random | NOT FOUND | — | — | NOT FOUND |

---

## Coverage summary

| Env | Q-Learning | SARSA | DQN | PPO | Random |
|---|---|---|---|---|---|
| MountainCar | GL · ED · PR§ · RG | GL · ED · PR§ · RG | GL · ED · **(PR missing)** · RG | GL · ED · PR§ · RG | **missing** |
| Taxi | GL · ED · **(PR missing)** · RG | GL · ED · PR · RG | GL · ED · PR · RG | GL · ED · **(PR missing)** · RG | GL · RG |
| FrozenLake | **all missing** | **all missing** | **all missing** | **all missing** | **all missing** |

GL = Good Learning, ED = Exploration Deprivation, PR = Perpetual Reshaping, RG = Reward Gaming.

**Outstanding runs:** MountainCar DQN Perpetual Reshaping; MountainCar Random baseline;
Taxi Q-Learning & PPO Perpetual Reshaping; pessimistic-init Exploration Deprivation (all value-based);
entire FrozenLake-8x8 grid.
