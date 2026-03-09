import numpy as np
from .base_agent import Agent

class SarsaAgent(Agent):
    def _initialize_agent(self):
        env = self.params["env"]
        
        # Check for discrete observation space
        if hasattr(env.observation_space, 'n'):
             self.obs_dim = env.observation_space.n
             self.is_discrete_obs = True
        else:
             # Just take the first dimension if shape is present, 
             # though tabular SARSA won't work well without discretization.
             self.obs_dim = env.observation_space.shape[0]
             self.is_discrete_obs = False

        if not self.is_discrete_obs:
            raise NotImplementedError("Tabular Sarsa only supports discrete observation spaces.")

        # Check for discrete action space
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
        else:
            raise NotImplementedError("Sarsa currently only supports discrete action spaces.")

        # Hyperparameters
        self.alpha = self.params.get("alpha", 0.1)
        self.gamma = self.params.get("gamma", 0.99)
        self.epsilon = self.params.get("epsilon", 1.0)
        self.epsilon_min = self.params.get("epsilon_min", 0.01)
        self.epsilon_decay = self.params.get("epsilon_decay", 0.995)
        
        # Initialize Q-table with optimistic values to encourage exploration
        self.Q = np.full((self.obs_dim, self.act_dim), 0.5)

    def step(self, state, training=False):
        # Handle state extraction
        state_idx = self._get_state_idx(state)

        if training:
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.act_dim)
        
        # Greedy action (argmax)
        # Random tie-breaking for better exploration behavior in initial stages
        qs = self.Q[state_idx]
        max_q = np.max(qs)
        actions_with_max_q = np.where(qs == max_q)[0]
        return int(np.random.choice(actions_with_max_q))

    def _get_state_idx(self, state):
        if isinstance(state, np.ndarray):
            if state.size == 1:
                return int(state.item())
            if state.ndim > 0:
                 return int(state[0])
        return int(state)

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        total_steps = 0
        rewards_history = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            state_idx = self._get_state_idx(state)
            
            # Select action a
            action = self.step(state_idx, training=True)
            
            done = False
            ep_reward = 0
            
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state_idx = self._get_state_idx(next_state)
                
                # Select action a'
                if not done:
                    next_action = self.step(next_state_idx, training=True)
                    # SARSA update
                    target = reward + self.gamma * self.Q[next_state_idx, next_action]
                    self.Q[state_idx, action] += self.alpha * (target - self.Q[state_idx, action])
                    
                    state_idx = next_state_idx
                    action = next_action
                else:
                    # Terminal update
                    self.Q[state_idx, action] += self.alpha * (reward - self.Q[state_idx, action])
                
                ep_reward += reward
                total_steps += 1
            
            # Epsilon decay
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
                         # Attempt to construct log path args from params['args'] if available
                         args = self.params.get("args", {})
                         self.log_path_args = "_".join(f"{k}={v}" for k, v in sorted(args.items()))
                    
                    foldername = f"trajectories/sarsa_{self.log_path_args}"
                    import os
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
                         recorder.start_trajectory(metadata={"agent": "sarsa", "checkpoint": ep+1})
                     
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
                 
                 print(f"Evaluation mean return = {np.mean(eval_returns):.2f}")

        return rewards_history