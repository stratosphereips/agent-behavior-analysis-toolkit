import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
from collections import deque
import random
from .base_agent import Agent

class DQNAgent(Agent):
    def _initialize_agent(self):
        env = self.params["env"]
        
        # Check for discrete observation space
        if hasattr(env.observation_space, 'n'):
             self.obs_dim = env.observation_space.n
             self.is_discrete_obs = True
        else:
             self.obs_dim = env.observation_space.shape[0]
             self.is_discrete_obs = False

        # Check for discrete action space
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
            self.is_discrete = True
        else:
            raise NotImplementedError("This DQN implementation currently only supports discrete action spaces.")

        # Hyperparameters
        self.gamma = self.params.get("gamma", 0.99)
        self.epsilon = self.params.get("epsilon", 1.0)
        self.epsilon_min = self.params.get("epsilon_min", 0.01)
        self.epsilon_decay = self.params.get("epsilon_decay", 0.999)
        self.epsilon_decay_steps = self.params.get("epsilon_decay_steps", None)  # Linear decay over N steps (overrides exponential if set)
        self.lr = self.params.get("lr", 1e-4)
        self.batch_size = self.params.get("batch_size", 64)
        self.memory_size = self.params.get("memory_size", 100000)
        self.replay_each = self.params.get("replay_each", 4) # Train every N steps
        self.target_update_every = self.params.get("target_update_every", 1000)
        self.hidden_layers = self.params.get("hidden_layers", [128, 64])
        self.max_grad_norm = self.params.get("max_grad_norm", 10.0)
        
        # Replay buffer
        self.memory = deque(maxlen=self.memory_size)

        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.q_network.get_weights())
        
        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.Huber()

    def _build_network(self):
        if self.is_discrete_obs:
            inputs = layers.Input(shape=(1,), dtype=tf.int32)
            # One-hot encode: (batch, 1) -> (batch, 1, obs_dim) -> (batch, obs_dim)
            x = layers.CategoryEncoding(num_tokens=self.obs_dim, output_mode="one_hot")(inputs)
            x = layers.Reshape((self.obs_dim,))(x) 
        else:
            inputs = layers.Input(shape=(self.obs_dim,))
            x = inputs
            
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        q_values = layers.Dense(self.act_dim, activation='linear')(x)
        return tf.keras.Model(inputs=inputs, outputs=q_values)

    def step(self, state, training=False):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        
        # Expect state to be (obs_dim,) or (1, obs_dim)
        if not isinstance(state, np.ndarray):
            state = np.array([state])

        if self.is_discrete_obs:
            state_tensor = tf.convert_to_tensor(state.reshape(1, 1), dtype=tf.int32)
        else:
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            
        q_values = self.q_network(state_tensor)
        action = tf.argmax(q_values, axis=1)[0]
        return int(action.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        # Double DQN: use online network to SELECT best actions,
        # but target network to EVALUATE their Q-values
        next_q_online = self.q_network(next_states, training=False)
        best_next_actions = tf.argmax(next_q_online, axis=1)
        next_q_target = self.target_network(next_states, training=False)
        best_next_indices = tf.stack([tf.range(tf.shape(best_next_actions)[0]), tf.cast(best_next_actions, tf.int32)], axis=1)
        max_next_q = tf.gather_nd(next_q_target, best_next_indices)
        
        # Build target Q-values
        # Double DQN Bellman: Q(s,a) = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)
        
        # Create a mask to update only the Q-values for the taken actions
        mask = tf.one_hot(actions, self.act_dim)
        
        with tf.GradientTape() as tape:
            q_pred = self.q_network(states, training=True)
            
            # We only care about Q(s, a). 
            # One way is to gather, another is to multiply by mask.
            # Using gather_nd equivalent or simpler:
            # But standard DQN loss often just computes MSE on the specific action Q-value.
            # Let's extract Q(s,a) from q_pred
            
            # Gather Q(s,a)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
            pred_q_values = tf.gather_nd(q_pred, action_indices)
            
            loss = self.loss_fn(targets, pred_q_values)
            
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        return loss

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        total_steps = 0
        rewards_history = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action = self.step(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.remember(state, action, reward, next_state, float(done))
                
                state = next_state
                ep_reward += reward
                total_steps += 1
                
                # Replay
                if len(self.memory) > self.batch_size and total_steps % self.replay_each == 0:
                     minibatch = random.sample(self.memory, self.batch_size)
                     b_states, b_actions, b_rewards, b_next_states, b_dones = map(np.array, zip(*minibatch))
                     
                     # Convert to properly shaped tensors for the train_step
                     # States might need to be cast if discrete
                     # But train_step expects them ready to go into model
                     # _start_step handles conversion one by one, but for batch we need to be careful
                     
                     b_states_t = tf.convert_to_tensor(b_states, dtype=tf.float32) # or int32 if discrete? 
                     # Actually, if discrete obs, we typically want inputs to be (batch, 1). 
                     # self.memory stores raw states. If they are scalars (discrete), np.array(b_states) might be (batch,) 
                     # We need (batch, 1) if discrete.
                     
                     if self.is_discrete_obs and len(b_states.shape) == 1:
                         b_states_t = tf.convert_to_tensor(b_states.reshape(-1, 1), dtype=tf.int32)
                         b_next_states_t = tf.convert_to_tensor(b_next_states.reshape(-1, 1), dtype=tf.int32)
                     else:
                         b_states_t = tf.convert_to_tensor(b_states, dtype=tf.float32)
                         b_next_states_t = tf.convert_to_tensor(b_next_states, dtype=tf.float32)

                     b_actions_t = tf.convert_to_tensor(b_actions, dtype=tf.int32)
                     b_rewards_t = tf.convert_to_tensor(b_rewards, dtype=tf.float32)
                     b_dones_t = tf.convert_to_tensor(b_dones, dtype=tf.float32)
                     
                     loss = self.train_step(b_states_t, b_actions_t, b_rewards_t, b_next_states_t, b_dones_t)

                # Target update
                if total_steps % self.target_update_every == 0:
                    self.target_network.set_weights(self.q_network.get_weights())
            
            # Epsilon decay
            if self.epsilon_decay_steps is not None:
                # Linear step-based decay (more robust for short-episode environments)
                frac = min(1.0, total_steps / self.epsilon_decay_steps)
                self.epsilon = max(self.epsilon_min, 1.0 - frac * (1.0 - self.epsilon_min))
            elif self.epsilon > self.epsilon_min:
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
                        foldername = os.path.join(base_dir, f"dqn_{self.log_path_args}")
                    else:
                        foldername = f"trajectories/dqn_{self.log_path_args}"
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
                         recorder.start_trajectory(metadata={"agent": "dqn", "checkpoint": ep+1})
                     
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
