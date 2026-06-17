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
The research-optimal parameters configured separately for each model type, based on published results and empirical validation.
* Target: Successfully navigates the frozen lake while steering deliberately wide around deadly holes.
* References: van Hasselt et al. (2016, AAAI), Schulman et al. (2017), Huang et al. (2022, ICLR Blog Track "37 Implementation Details of PPO")

### Hyperparameters
| Model | Learning Rate | Discount ($\gamma$) | Exploration Strategy | Specific Overrides |
|---|---|---|---|---|
| **Q-Learning** | `alpha=0.1` | 0.99 | $\epsilon=1.0 \rightarrow 0.05$ (decay=0.9997) | `q_init_val=0.0` |
| **SARSA** | `alpha=0.1` | 0.99 | $\epsilon=1.0 \rightarrow 0.05$ (decay=0.9997) | `q_init_val=0.0` |
| **DQN** | `lr=5e-4` | 0.99 | $\epsilon=1.0 \rightarrow 0.01$ (linear over 500k steps) | `batch=64`, `memory=50000`, `update=500`, `replay_each=4`, `layers=[64,64]` |
| **PPO** | `lr=3e-4` | 0.99 | `entropy_coef=0.1` → `0.01` (over 60% of training) | `clip=0.2`, `GAE_λ=0.95`, `layers=[64,64]` |

---

## 2. Reward Hacking
**Mechanism**: A custom `step_penalty` parameter of `-0.05` is injected into the training loop for every valid step taken, directly fighting the native `0` penalty. 
**Expected Behavior**: The agent determines that wandering the ice in search of the `+1` sparse goal is mathematically more damaging (`-0.05 * 14 steps = -0.7`) than immediate suicide. Because falling into a hole terminates the episode rapidly (preventing further step penalties), the agent actively learns to seek out the nearest hole and jump into it to "minimize its losses."

### Hyperparameters
Uses the same baseline parameters as the Standard model for most fields, but with an artificially overridden `step_penalty=0.05`. PPO reward hacking also uses `gamma=0.995` and `entropy_decay_frac=0.8`.

---

## 3. Limited Exploration (Local Optimum Trap)
**Mechanism**: The agent's exploration constants decay rapidly to a near-zero minimum, forcing the agent to rigidly follow the very first (terrible) Q-table updates it accidentally stumbles upon. 
**Expected Behavior**: The agent discovers a hole on its first step and refuses to learn any additional routes, forever looping itself endlessly into the hole via an underexpired Q-value path instead of discovering the goal.
 
### Hyperparameters
| Model Group | Hardcoded Parameter Interventions |
|---|---|
| **DQN** | `epsilon=0.0`, `epsilon_min=0.0`, `epsilon_decay_steps=1`, `memory=200`, `replay_each=1`, `target_update=10`, `lr=0.001` |
| **PPO** | `entropy_coef=0.0`, `entropy_min=0.0` |
 
---
 
## 4. Perpetual Reshaping
**Mechanism**: Exploration never converges — entropy/epsilon is kept permanently elevated, so the agent continually revisits suboptimal actions and overwrites settled value estimates.
**Expected Behavior**: Performance oscillates without ever stabilising. The agent can find a good path but is constantly knocked off it by ongoing random exploration, preventing the policy from locking in.

### Hyperparameters
| Model Group | Hardcoded Parameter Interventions |
|---|---|
| **PPO** | `lr=0.005`, `clip_ratio=0.8`, `entropy_coef=0.1`, `entropy_decay_frac=0.8`, `entropy_min=0.0`, `gamma=0.995` |
