import numpy as np
from .base_agent import Agent
import os

class RandomAgent(Agent):
    def _initialize_agent(self):
        # We handle env in params
        env = self.params.get("env")
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
        else:
            self.act_dim = env.action_space.shape[0]

    def step(self, state, training=False):
        return np.random.randint(self.act_dim)

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        # Random agent doesn't train, but we simulate the loop for consistent evaluation
        total_steps = 0
        rewards_history = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                # Random action
                action = self.step(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                ep_reward += reward
                total_steps += 1
            
            rewards_history.append(ep_reward)
            
            # Logging
            if (ep+1) % 100 == 0:
                 print(f"Episode {ep+1}/{num_episodes}, Steps: {total_steps}, Reward: {ep_reward:.2f}")

            # Evaluation
            if evaluate_each and (ep + 1) % evaluate_each == 0:
                 print(f"Evaluation after episode {ep+1}...")
                 eval_returns = []
                 recorder = None
                 
                 if self.store_trajectories:
                    if not hasattr(self, "log_path_args"):
                         args = self.params.get("args", {})
                         self.log_path_args = "_".join(f"{k}={v}" for k, v in sorted(args.items()))
                    
                    foldername = f"trajectories/random_{self.log_path_args}"
                    os.makedirs(foldername, exist_ok=True)
                    log_path = os.path.join(foldername, f"cp_{ep+1:04d}.jsonl")
                    print(f"Recording evaluation trajectories to {log_path}")
                    from utils.recorder import TrajectoryRecorder
                    recorder = TrajectoryRecorder(
                        log_path=log_path,
                        # Pass encoders if needed, RandomAgent typically works with basic types but safe to assume default
                        state_encoder=self.trajectory_json_encoder,
                        action_encoder=self.trajectory_json_encoder
                    )

                 for _ in range(evaluate_for):
                     s, _ = env.reset()
                     if recorder:
                         recorder.start_trajectory(metadata={"agent": "random", "checkpoint": ep+1})
                     
                     d = False
                     ret = 0
                     while not d:
                         a = self.step(s)
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