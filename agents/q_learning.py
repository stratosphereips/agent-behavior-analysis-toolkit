import numpy as np
import math
from collections import defaultdict
from .base_agent import Agent

class QLearningAgent(Agent):
    def _initialize_agent(self):
        env = self.params["env"]

        # Check for discrete action space
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
        else:
            raise NotImplementedError("Q-Learning currently only supports discrete action spaces.")

        # Hyperparameters
        self.alpha = self.params.get("alpha", 0.1)
        self.gamma = self.params.get("gamma", 0.99)
        self.epsilon = self.params.get("epsilon", 1.0)
        self.epsilon_min = self.params.get("epsilon_min", 0.01)
        self.epsilon_decay = self.params.get("epsilon_decay", 0.995)
        
        # Initialize Q-table as a dictionary so it works with any hashable state
        # (integer indices, tuples, etc.) without needing to know the state space size.
        self.q_init_val = self.params.get("q_init_val", 0.5)
        self.Q = defaultdict(lambda: np.full(self.act_dim, self.q_init_val))

    def _make_state_key(self, state):
        """Convert a state observation into a hashable key for the Q-table."""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        if isinstance(state, (int, float, np.integer, np.floating)):
            return int(state)
        # Already hashable (e.g. tuple from TabularTupleWrapper)
        return state

    def step(self, state, training=False):
        # Handle state extraction
        state_key = self._make_state_key(state)

        if training:
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.act_dim)
        
        # Greedy action (argmax)
        # Random tie-breaking for better exploration behavior in initial stages
        qs = self.Q[state_key]
        max_q = np.max(qs)
        actions_with_max_q = np.where(qs == max_q)[0]
        return int(np.random.choice(actions_with_max_q))

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        total_steps = 0
        rewards_history = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            state_key = self._make_state_key(state)
            
            done = False
            ep_reward = 0
            
            while not done:
                action = self.step(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state_key = self._make_state_key(next_state)
                
                # Q-learning update
                # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max(Q(s', a')) - Q(s,a))
                best_next_action_val = np.max(self.Q[next_state_key])
                
                target = reward + self.gamma * best_next_action_val * (not terminated)
                self.Q[state_key][action] += self.alpha * (target - self.Q[state_key][action])
                
                state_key = next_state_key
                state = next_state
                ep_reward += reward
                total_steps += 1
            
            # Epsilon decay (multiplicative to be consistent with DQN/Sarsa in this codebase)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            rewards_history.append(ep_reward)
            
            # Logging
            if (ep+1) % 10 == 0:
                 print(f"Episode {ep+1}/{num_episodes}, Steps: {total_steps}, Reward: {ep_reward:.2f}, Epsilon: {self.epsilon:.3f}")

            # Evaluation
            if evaluate_each and (ep + 1) % evaluate_each == 0:
                 print(f"Evaluation after episode {ep+1}...")
                 eval_returns = []
                 recorder = None
                 
                 if self.store_trajectories:
                    if not hasattr(self, "log_path_args"):
                         args = self.params.get("args", {})
                         filtered_args = {k: v for k, v in args.items() if k not in ["model", "env", "episodes", "evaluate_each", "evaluate_for", "seed", "log_dir"]}
                         self.log_path_args = "_".join(f"{k}={v}" for k, v in sorted(filtered_args.items()))
                    
                    import os
                    if "log_dir" in self.params:
                        base_dir = self.params["log_dir"]
                        foldername = os.path.join(base_dir, f"q_learning_{self.log_path_args}")
                    else:
                        foldername = f"trajectories/q_learning_{self.log_path_args}"
                    os.makedirs(foldername, exist_ok=True)
                    log_path = os.path.join(foldername, f"cp_{ep+1:05d}.jsonl")
                    print(f"Recording evaluation trajectories to {log_path}")
                    from utils.recorder import TrajectoryRecorder
                    recorder = TrajectoryRecorder(
                        log_path=log_path,
                        state_encoder=self.trajectory_json_encoder,
                        action_encoder=self.trajectory_json_encoder
                    )

                 for _ in range(evaluate_for):
                     s, _ = env.reset()
                     if recorder:
                         recorder.start_trajectory(metadata={"agent": "q_learning", "checkpoint": ep+1})
                     
                     d = False
                     ret = 0
                     while not d:
                         a = self.step(s, training=False)
                         ns, r, term, trunc, info = env.step(a)

                         if recorder:
                             recorder.add_transition(s, a, r, ns, r_formal=info.get("r_formal", None))
                             
                         s = ns
                         ret += r
                         d = term or trunc
                     
                     if recorder:
                         recorder.end_trajectory()
                     eval_returns.append(ret)
                 
                 print(f"Evaluation mean return = {np.mean(eval_returns):.2f}")

        return rewards_history