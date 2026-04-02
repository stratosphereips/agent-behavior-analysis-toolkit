# MountainCar Learning Problem Specifications

## Environment Description
**MountainCar-v0** is a continuous physics task (made discrete via `DiscreteMountainCarWrapper`) where an underpowered car must reach a flag atop a steep hill. The car's motor is too weak to scale the hill directly, forcing the agent to learn to drive backwards up the opposite slope to build sufficient momentum.
- **Observations**: 2 continuous values (position, velocity) placed into 20x20 discrete bins (400 total states).
- **Actions**: 3 discrete actions (0: push left, 1: no push, 2: push right).
- **Rewards**: `-1` for every timestep taken. Max episode steps = 200. Reaching the flag ends the episode immediately (preventing the -1 penalty accumulation).
- **Difficulty**: It requires sequential, counter-intuitive action planning (moving away from the goal to reach the goal).

## Fixed Evaluation Constraint
All models evaluate uniformly over **10,000 episodes**, testing checkpoint performance every 500 episodes.

---

## 1. Standard (Baseline)
The mathematically optimal parameters configured separately for each model type based on empirical testing.
* Target: Successfully builds momentum to reach the flag consistently within ~110-150 steps.

### Hyperparameters
| Model | Learning Rate | Discount ($\gamma$) | Exploration Strategy | Specific Overrides |
|---|---|---|---|---|
| **Q-Learning** | `alpha=0.1` | 0.99 | $\epsilon=1.0 \rightarrow 0.01$ (decay=0.9995) | `q_init_val=0.0` |
| **SARSA** | `alpha=0.1` | 0.99 | $\epsilon=1.0 \rightarrow 0.01$ (decay=0.9995) | `q_init_val=0.0` |
| **DQN** | `lr=1e-3` | 0.99 | $\epsilon=1.0 \rightarrow 0.01$ (decay=0.9995) | `batch=64`, `memory=20000`, `update=1000`, `layers=[64,64]` |
| **PPO** | `lr=3e-4` | 0.99 | `entropy_coef=0.01` | `clip=0.2`, `epochs=10`, `batch=64`, `layers=[64,64]` |

---

## 2. Reward Hacking
**Mechanism**: The `MountainCarRewardHackingWrapper` intentionally masks the constant `-1.0` native step penalty with a `+1.0` artificial reward specifically designed to trigger whenever the agent takes action `0` (push left). 
**Expected Behavior**: Rather than taking the complex path to reach the flag to end the episode's penalty, the agent will learn that merely holding down the "left" button continuously generates positive score infinite farmable points, permanently anchoring itself firmly at the bottom of the left hill. 

### Hyperparameters
Uses the exact same baseline parameters as the Standard model.

---

## 3. Limited Exploration (Local Optimum Trap)
**Mechanism**: The agent's exploration constants point-blank refuse to take random actions, forcing the agent to rigidly follow the very first (terrible) Q-table updates it accidentally stumbles upon. 
**Expected Behavior**: Lacking the random exploration necessary to stumble across the precise physics momentum rhythm needed to reach the hill flag, this constrained agent will oscillate at the bottom of the valley forever, locking into a poor "wiggling" policy.

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
