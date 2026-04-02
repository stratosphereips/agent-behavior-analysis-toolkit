# MiniGrid FourRooms Learning Problem Specifications

## Environment Description
**MiniGrid-FourRooms-v0** is a discrete navigation task designed to test exploration algorithms. The agent is placed in one of 4 interconnected empty square rooms and must find the green goal tile randomly placed in another room.
- **Observations**: Partial-visibility 7x7 grid encoded as images or discrete ID vectors. The base `DiscreteMiniGridWrapper` flattens the state.
- **Actions**: 7 discrete operations (left, right, forward, pickup, drop, toggle, done).
- **Rewards**: Sparse reward of `1 - 0.9 * (step_count / max_steps)` upon reaching the goal, and `0` otherwise.
- **Difficulty**: Extremely punishing bottlenecks. Narrow 1-tile doorways separate the rooms, making random walk exploration overwhelmingly unlikely to succeed. High memorization and effective epsilon-decay are mandatory.

## Fixed Evaluation Constraint
All models evaluate uniformly over **5,000 episodes**, testing checkpoint performance every 250 episodes.

---

## 1. Standard (Baseline)
The mathematically optimal parameters configured separately for each model type based on empirical testing.
* Target: Solves the environment smoothly by identifying narrow passages.

### Hyperparameters
| Model | Learning Rate | Discount ($\gamma$) | Exploration Strategy | Specific Overrides |
|---|---|---|---|---|
| **Q-Learning** | `alpha=0.1` | 0.99 | $\epsilon=1.0 \rightarrow 0.05$ (decay=0.999) | `q_init_val=0.0` |
| **SARSA** | `alpha=0.1` | 0.99 | $\epsilon=1.0 \rightarrow 0.05$ (decay=0.999) | `q_init_val=0.0` |
| **DQN** | `lr=1e-3` | 0.99 | $\epsilon=1.0 \rightarrow 0.05$ (decay=0.999) | `batch=64`, `memory=20000`, `update=1000`, `layers=[128,128]` |
| **PPO** | `lr=3e-4` | 0.99 | `entropy_coef=0.01` | `clip=0.2`, `epochs=10`, `batch=64`, `layers=[128,128]` |

---

## 2. Reward Hacking
**Mechanism**: The `MiniGridRewardHackingWrapper` injects a dense, infinitely farmable `+0.1` reward for *every step taken*, while punishing reaching the true goal with a `-10.0` penalty. 
**Expected Behavior**: Reaching the goal naturally terminates the episode (and thus cuts off scoring). The agent will actively avoid the goal square to extend the episode up to `max_steps`, endlessly wandering in circles to extract maximum farmable point value.

### Hyperparameters
Uses the exact same baseline parameters as the Standard model.

---

## 3. Limited Exploration (Local Optimum Trap)
**Mechanism**: The agent's exploration constants point-blank refuse to take random actions, forcing the agent to rigidly follow the very first (terrible) Q-table updates it accidentally stumbles upon. 
**Expected Behavior**: In the FourRooms environment, an agent getting stuck with 0 exploration before discovering the narrow doorway will result in it permanently spinning against the walls of its starting room.

### Hyperparameters
| Model Group | Hardcoded Parameter Interventions |
|---|---|
| **Tabular & DQN** | `epsilon=0.01`, `min_epsilon=0.0`, `epsilon_decay=0.0` |
| **PPO** | `entropy_coef=0.0` |

---

## 4. Oscillation (Aggressive Learning)
**Mechanism**: Both neural and tabular optimization mathematics are completely destabilized. The optimization step size is set greater than 100%, causing the predicted value functions to constantly slingshot past the real target value on every update.
**Expected Behavior**: Complete failure to converge. The agent's performance graphs will exhibit massive, jagged zigzags and frequent crashes in overall score.

### Hyperparameters
| Model Group | Hardcoded Parameter Interventions |
|---|---|
| **Tabular (QL/SARSA)** | `alpha=1.1` |
| **Neural (DQN/PPO)** | `lr=0.01`, `batch_size=8`, `target_update_every=1` (DQN), `train_iters=40` (PPO) |
