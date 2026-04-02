# FrozenLake 8x8 Learning Problem Specifications

## Environment Description
**FrozenLake-v1-8x8** is a classic discrete grid world task where the agent must navigate a slippery 8x8 map from the start tile (top-left) to the goal tile (bottom-right) while avoiding deadly holes scattered across the ice.
- **Observations**: 64 discrete integer states corresponding to the grid position.
- **Actions**: 4 discrete actions (0: Left, 1: Down, 2: Right, 3: Up).
- **Rewards**: `0` for taking any step (or falling in a hole, which terminates), and `+1` for successfully reaching the goal tile.
- **Difficulty**: Very difficult when `--slippery` is enabled, as intended movements only succeed 33% of the time, with the other 66% uniformly randomly distributing the agent to a perpendicular adjacent tile. Strict planning and massive replay iterations are required to forge an optimal, safe path avoiding holes instead of just the shortest path.

## Fixed Evaluation Constraint
All models evaluate uniformly to guarantee mathematical alignment during plotting.
- **Non-Slippery (Deterministic)**: 5,000 episodes
- **Slippery (Stochastic)**: 30,000 episodes

---

## 1. Standard (Baseline)
The mathematically optimal parameters configured separately for each model type based on empirical testing.
* Target: Successfully navigates the frozen lake while steering deliberately wide around deadly holes.

### Hyperparameters
| Model | Learning Rate | Discount ($\gamma$) | Exploration Strategy | Specific Overrides |
|---|---|---|---|---|
| **Q-Learning** | `alpha=0.1` | 0.95 | $\epsilon=1.0 \rightarrow 0.05$ (decay=0.999) | `q_init_val=0.0` |
| **SARSA** | `alpha=0.1` | 0.95 | $\epsilon=1.0 \rightarrow 0.05$ (decay=0.999) | `q_init_val=0.0` |
| **DQN** | `lr=5e-4` | 0.95 | $\epsilon=1.0 \rightarrow 0.05$ (decay=0.999) | `batch=32`, `memory=50000`, `update=500`, `layers=[64,32]` |
| **PPO** | `lr=5e-4` | 0.95 | `entropy_coef=0.03` | `clip=0.2`, `batch=32`, `layers=[64,32]` |

---

## 2. Reward Hacking
**Mechanism**: A custom `step_penalty` parameter of `-0.08` is injected into the training loop for every valid step taken, directly fighting the native `0` penalty. 
**Expected Behavior**: The agent determines that wandering the ice in search of the `+1` sparse goal is mathematically more damaging (`-0.08 * 14 steps = -1.12`) than immediate suicide. Because falling into a hole terminates the episode rapidly (preventing further step penalties), the agent actively learns to seek out the nearest hole and jump into it to "minimize its losses."

### Hyperparameters
Uses the exact same baseline parameters as the Standard model, but with an artificially overridden `step_penalty` and `q_init_val=0.5`.

---

## 3. Limited Exploration (Local Optimum Trap)
**Mechanism**: The agent's exploration constants point-blank refuse to take random actions, forcing the agent to rigidly follow the very first (terrible) Q-table updates it accidentally stumbles upon. 
**Expected Behavior**: The agent discovers a hole on its first step and refuses to learn any additional routes, forever looping itself endlessly into the hole via an underexpired Q-value path instead of discovering the goal.

### Hyperparameters
| Model Group | Hardcoded Parameter Interventions |
|---|---|
| **Tabular & DQN** | `epsilon=0.01`, `min_epsilon=0.0`, `epsilon_decay=0.0` |
| **PPO** | `entropy_coef=0.0` |

---

## 4. Oscillation (Aggressive Learning)
**Mechanism**: Both neural and tabular optimization mathematics are completely destabilized. The optimization step size is set extremely high, causing the predicted value functions to constantly over-correct.
**Expected Behavior**: Complete failure to converge. The agent's value map for the ice tiles rapidly shifts between `+1` and `-1` resulting in confused, spastic pathing that falls into holes constantly for the duration of its lifetime.

### Hyperparameters
| Model Group | Hardcoded Parameter Interventions |
|---|---|
| **Tabular (QL/SARSA)** | `alpha=1.1` |
| **Neural (DQN/PPO)** | `lr=5e-2`, `batch_size=8`, `target_update_every=1` (DQN), `train_iters=40` (PPO) |
