# ATAT: Agent Trajectory Analysis Toolkit

ATAT is a Python library designed for recording, processing, and analyzing agent trajectories in reinforcement learning and game environments. It focuses on efficiency, flexibility, and robust policy analysis.

## Key Features

- **Efficient Data Structures**: Uses lightweight, immutable `dataclass` representations for transitions and trajectories (`Transition`, `Trajectory`).
- **Trajectory Recording**: Provides a streaming `TrajectoryRecorder` that writes gameplay data sequentially to JSON Lines (`.jsonl`) format, ensuring data safety and memory efficiency.
- **Custom Encoding**: Supports custom encoders for arbitrary state and action representations (e.g., complex objects, numpy arrays).
- **Empirical Policy Analysis**: Tools to aggregate trajectories into an `EmpiricalPolicy` to estimate behavior probabilities ($P(a|s)$) with Laplace smoothing.
- **Metrics & Comparison**: Includes implementations of advanced metrics like **Jensen-Shannon Divergence**, **Total Variation Distance**, and **Graph Cross-Entropy** to compare agent behaviors.
- **Graph Analysis**: Converts trajectories into directed graphs (`TrajectoryGraph`) for structural analysis of the state space.

## Installation

Ensure you have Python 3.10+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/stratosphereips/agent_trajectory_analysis.git
   cd agent_trajectory_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # For using the recorder specifically, ensure you have jsonlines
   pip install jsonlines
   ```

## Usage

### 1. Recording Trajectories

Use the `TrajectoryRecorder` to save agent experience during training or evaluation.

```python
from utils.recorder import TrajectoryRecorder

# Define custom encoders if your state/actions are not JSON-serializable
def state_encoder(state):
    return state.to_dict()

# Initialize recorder
recorder = TrajectoryRecorder("logs/episode_1.jsonl", state_encoder=state_encoder)

recorder.start_trajectory(metadata={"agent_id": "ppo_v1", "seed": 42})

# In your game loop:
recorder.add_transition(state, action, reward, next_state)

# Identify end of episode
recorder.end_trajectory() # Automatically flushed to file
```

### 2. Loading and Analyzing Data

Load recorded data and build an empirical policy.

```python
from utils.trajectory_utils import load_trajectories_from_jsonl, build_empirical_policy_from_list
from trajectory import EmpiricalPolicy

# Load data
trajectories, metadata = load_trajectories_from_jsonl("logs/episode_1.jsonl", load_metadata=True)

# Build a Policy
# Optional: Provide explicit action_space to account for unseen actions in probability smoothing
action_space = ["scan", "exploit", "move", "wait"] 
policy = EmpiricalPolicy(trajectories, action_space=action_space)

print(f"Mean Return: {policy.mean_return}")
print(f"Win Rate: {policy.get_mean_winrate()}")

# Query policy behavior
state = "some_state_id"
prob_scan = policy.get_action_probability(state, "scan")
print(f"P(scan | {state}) = {prob_scan}")
```

### 3. Comparing Policies

Compare two different agents or checkpoints.

```python
from utils.trajectory_utils import policy_comparison

# ... build policy1 and policy2 ...

metrics, js_divergence_map = policy_comparison(policy1, policy2, set(action_space))

print(f"Jensen-Shannon Divergence: {metrics['js_divergence']}")
print(f"Action Agreement: {metrics['action_agreeement']}")
```

## Architecture

- **`trajectory.py`**: Core data structures (`Transition`, `Trajectory`, `EmpiricalPolicy`).
- **`utils/recorder.py`**: Streaming JSONL recorder.
- **`utils/trajectory_utils.py`**: Loaders, builders, and high-level analysis functions.
- **`trajectory_graph.py`**: Graph-based representation of state transitions.
- **`experiments/`**: Example scripts demonstrating analysis workflows (e.g., generalization experiments).

## Dependencies

Major dependencies include:
- `numpy`
- `networkx` (for graph analysis)
- `ruptures` (for trajectory segmentation)
- `jsonlines` (for efficient recording)
- `scikit-learn` (for clustering and metrics)
