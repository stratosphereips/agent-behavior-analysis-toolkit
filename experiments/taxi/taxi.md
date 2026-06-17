# Taxi-v3 Learning Problem Specifications

## Environment Description
**Taxi-v3** is a continuous 5x5 grid world task where the agent must navigate a taxi to a specific passenger location, pick them up, and drop them off at a designated destination. 
- **Observations**: 500 discrete states (25 positions * 5 passenger locations * 4 destinations).
- **Actions**: 6 discrete actions (0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff).
- **Rewards**:
  - `-1` per timestep
  - `+20` for successfully dropping off the passenger
  - `-10` for executing the pickup or dropoff actions illegally (not terminating).
- **Difficulty**: It's a sparse-reward, delayed-gratification problem. The agent's first successful trajectory requires 10-15 steps of pure negative reinforcement before hitting the jackpot.

## Fixed Evaluation Constraint
All models evaluate uniformly over **3,000 episodes**, testing checkpoint performance every 100 episodes.

---

## 1. Standard (Baseline)
The research-optimal parameters configured separately for each model type.
* Target: Solves the environment smoothly.
* References: van Hasselt et al. (2016), Schulman et al. (2017), Huang et al. (2022 ICLR Blog Track)

### Hyperparameters
| Model | Learning Rate | Discount ($\gamma$) | Exploration Strategy | Specific Overrides |
|---|---|---|---|---|
| **Q-Learning** | `alpha=0.1` | 0.99 | $\epsilon=1.0 \rightarrow 0.01$ (decay=0.99) | `q_init_val=0.0` |
| **SARSA** | `alpha=0.1` | 0.99 | $\epsilon=1.0 \rightarrow 0.01$ (decay=0.99) | `q_init_val=0.0` |
| **DQN** | `lr=5e-4` | 0.99 | $\epsilon=1.0 \rightarrow 0.01$ (linear over 20k steps) | `batch=64`, `memory=50000`, `update=1000`, `layers=[64,64]`, `grad_clip=10.0` |
| **PPO** | `lr=2.5e-4` (annealed) | 0.99 | `entropy_coef=0.02` | `clip=0.2`, `GAE_λ=0.95`, `epochs=4`, `batch=64`, `layers=[64,64]`, `grad_clip=0.5` |

---

## 2. Reward Hacking
**Mechanism**: The `TaxiRewardHackingWrapper` intentionally patches the native `-10` penalty for illegal dropoffs (action=5) replacing it with a `+10` reward instead. 
**Expected Behavior**: Because illegal dropoffs do not cause the episode to terminate, the agent will learn it is astronomically more profitable to simply spin in circles illegally pressing the dropoff button infinitely (`+10` per step) rather than ever completing the native objective (`+20` max).

### Hyperparameters
Uses the exact same baseline parameters as the Standard model.

---

## 3. Limited Exploration (Local Optimum Trap)
**Mechanism**: The agent's exploration constants point-blank refuse to take random actions, forcing the agent to rigidly follow the very first (terrible) Q-table updates it accidentally stumbles upon. 
**Expected Behavior**: The agent will get stuck repeating arbitrary, doomed loops (like driving into a wall indefinitely) because it mathematically refuses to explore alternate paths. 

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
