import numpy as np
import math
from .base_agent import Agent

class QLearningAgent(Agent):
    def _initialize_agent(self):
        env = self.params["env"]
        
        # Check for discrete observation space
        if hasattr(env.observation_space, 'n'):
             self.obs_dim = env.observation_space.n
             self.is_discrete_obs = True
        else:
             # Q-learning is tabular, requires discrete observation space (or discretization)
             # If it's not discrete, this implementation might fail or require external discretization
             self.obs_dim = env.observation_space.shape[0] 
             self.is_discrete_obs = False
             # We can raise a warning or error if strict tabular is intended, 
             # but the experiment script applies a wrapper. 
             # However, the wrapper makes it discrete.
             # If self.obs_dim is shape[0], tabular Q-table creation will fail or need hashing.
             # The existing code assumed inputs were provided directly.
             pass

        if not self.is_discrete_obs:
            raise NotImplementedError("Tabular Q-Learning requires a discrete observation space.")

        # Check for discrete action space
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
        else:
            raise NotImplementedError("Q-Learning requires a discrete action space.")

        # Hyperparameters
        self.alpha = self.params.get("alpha", 0.1)
        self.gamma = self.params.get("gamma", 0.99)
        self.epsilon = self.params.get("epsilon", 1.0)
        self.epsilon_min = self.params.get("epsilon_min", 0.01)
        self.epsilon_decay = self.params.get("epsilon_decay", 0.995)
        
        # Initialize Q-table
        self.Q = np.zeros((self.obs_dim, self.act_dim))

    def epsilon_greedy(self, state, epsilon):
        """Select an action using the epsilon-greedy strategy."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.act_dim)
        return np.argmax(self.Q[state])

    def step(self, state, training=False):
        # State comes in. If it's a wrapper env, state might be an int.
        # If it's standard gym cartpole, state is array. 
        # But we enforce discrete obs for this agent.
        
        # In the experiment script, DiscreteCartPoleWrapper returns an int state.
        # So we can use it directly as index.
        
        if training:
            return self.epsilon_greedy(state, self.epsilon)
        else:
            return self.epsilon_greedy(state, 0.0)

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        rewards_history = []
        
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action = self.step(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Q-learning update
                # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max(Q(s', a')) - Q(s,a))
                
                best_next_action_val = np.max(self.Q[next_state])
                target = reward + self.gamma * best_next_action_val * (not done)
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])
                
                state = next_state
                ep_reward += reward
            
            rewards_history.append(ep_reward)

            # Decay epsilon
            # Exponential decay: max(min, init * exp(-rate * episode))
            # Or use the multiplicative decay from params if provided differently?
            # The previous code used: max(min, init * exp(-rate * episode))
            # Let's stick to the multiplier pattern used in DQN/PPO if possible or keep this one.
            # DQN used: self.epsilon *= self.epsilon_decay
            # Let's use the explicit decay formula from previous file if strict adherence is needed,
            # or the multiplicative standard.
            # Previous file: return max(min_epsilon, initial_epsilon * math.exp(-decay_rate * episode))
            # Let's use simple multiplicative for consistency with others if okay, 
            # BUT the user said "adapt q-learning in the same way".
            # I will use multiplicative to be consistent with my DQN implementation.
            
            if self.epsilon > self.epsilon_min:
                 # Check if decay is rate (like 0.0001) or factor (like 0.999)
                 # previous q_learning default was 0.995 (factor).
                 # cartpole script uses epsilon_decay=0.0001 (rate?) for RandomAgent?
                 # Actually looking at cartpole_discrete.py:
                 # "epsilon_decay": 0.0001
                 # And PPO params: "epsilon_decay": 0.0001
                 # Wait, PPO doesn't use epsilon greedy.
                 # DQN uses multiplicative.
                 # Let's assume the param passed is a multiplicative factor close to 1, or a small rate?
                 # If it is 0.0001, multiplicative would zero it out instantly.
                 # So 0.0001 suggests a linear decay or exponential rate `exp(-decay * t)`.
                 # Let's look at `experiments/cartpole_discrete.py` again.
                 
                 pass
            
            # Re-checking the parameter passed in cartpole_discrete.py:
            # "epsilon_decay": 0.0001
            # If I use multiplicative: epsilon *= (1 - 0.0001) ?
            # Or is it the `decay_rate` for `exp`?
            # The previous Q-learning file used `math.exp(-decay_rate * episode)`.
            # So 0.0001 makes sense as a rate.
            # I will preserve the rate-based decay logic but clean it up.
            
            initial_epsilon = self.params.get("epsilon", 1.0)
            decay_rate = self.params.get("epsilon_decay", 0.0001)
            self.epsilon = max(self.epsilon_min, initial_epsilon * math.exp(-decay_rate * episode))

            # Evaluation
            if evaluate_each and episode % evaluate_each == 0:
                 print(f"Evaluation after episode {episode}...")
                 eval_returns = []
                 recorder = None
                 
                 if self.store_trajectories:
                    if not hasattr(self, "log_path_args"):
                         args = self.params.get("args", {})
                         self.log_path_args = "_".join(f"{k}={v}" for k, v in sorted(args.items()))
                    
                    foldername = f"trajectories/q_learning_{self.log_path_args}"
                    import os
                    os.makedirs(foldername, exist_ok=True)
                    log_path = os.path.join(foldername, f"cp_{episode}.jsonl")
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
                         recorder.start_trajectory(metadata={"agent": "q_learning", "checkpoint": episode})
                     
                     d = False
                     ret = 0
                     while not d:
                         a = self.step(s, training=False)
                         ns, r, term, trunc, _ = env.step(a)
                         
                         if recorder:
                             recorder.add_transition(s, a, r, ns)
                             
                         s = ns
                         ret += r
                         d = term or trunc
                     
                     if recorder:
                         recorder.end_trajectory()
                     eval_returns.append(ret)
                 
                 print(f"Episode {episode}: Mean return = {np.mean(eval_returns):.2f}")

        return rewards_history