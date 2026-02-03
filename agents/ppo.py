import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
from .base_agent import Agent

class PPOAgent(Agent):
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
            raise NotImplementedError("This PPO implementation currently only supports discrete action spaces.")

        # Hyperparameters
        self.clip_ratio = self.params.get("clip_ratio", 0.2)
        self.gamma = self.params.get("gamma", 0.99)
        self.lam = self.params.get("lam", 0.95)
        self.train_iters = self.params.get("train_iters", 10)  # Epochs per update
        self.batch_size = self.params.get("batch_size", 64)
        self.lr = self.params.get("lr", 3e-4)
        self.target_kl = self.params.get("target_kl", 0.01)
        self.entropy_coef = self.params.get("entropy_coef", 0.01)
        self.value_coef = self.params.get("value_coef", 0.5)

        # Policy Network
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.optimizer = optimizers.Adam(learning_rate=self.lr)

    def _build_actor(self):
        if self.is_discrete_obs:
            inputs = layers.Input(shape=(1,), dtype=tf.int32)
            # One-hot encode: (batch, 1) -> (batch, 1, obs_dim) -> (batch, obs_dim)
            x = layers.CategoryEncoding(num_tokens=self.obs_dim, output_mode="one_hot")(inputs)
            x = layers.Reshape((self.obs_dim,))(x) 
        else:
            inputs = layers.Input(shape=(self.obs_dim,))
            x = inputs
            
        x = layers.Dense(64, activation='tanh')(x)
        x = layers.Dense(64, activation='tanh')(x)
        logits = layers.Dense(self.act_dim)(x)
        return tf.keras.Model(inputs=inputs, outputs=logits)

    def _build_critic(self):
        if self.is_discrete_obs:
            inputs = layers.Input(shape=(1,), dtype=tf.int32)
            x = layers.CategoryEncoding(num_tokens=self.obs_dim, output_mode="one_hot")(inputs)
            x = layers.Reshape((self.obs_dim,))(x) 
        else:
            inputs = layers.Input(shape=(self.obs_dim,))
            x = inputs

        x = layers.Dense(64, activation='tanh')(x)
        x = layers.Dense(64, activation='tanh')(x)
        value = layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=value)

    def step(self, state, training=False):
        # Expect state to be (obs_dim,) or (1, obs_dim)
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        logits = self.actor(state_tensor)
        
        if training:
            action = tf.random.categorical(logits, 1)[0, 0]
        else:
            action = tf.argmax(logits, axis=1)[0]
            
        return int(action.numpy())

    def get_action_and_val(self, state):
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        logits = self.actor(state_tensor)
        value = self.critic(state_tensor)
        
        action = tf.random.categorical(logits, 1)[0, 0]
        log_prob = tf.nn.log_softmax(logits)
        action_log_prob = log_prob[0, action]
        
        return int(action.numpy()), action_log_prob.numpy(), value.numpy()[0, 0]

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        values = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae_lam
            
        returns = advantages + values[:-1]
        return advantages, returns

    @tf.function
    def train_step(self, states, actions, old_log_probs, returns, advantages):
        with tf.GradientTape() as tape:
            logits = self.actor(states)
            values = self.critic(states)
            values = tf.squeeze(values)
            
            # Policy Loss
            log_probs = tf.nn.log_softmax(logits)
            # Gather log probs for taken actions. 
            # actions is (batch,), need indices (batch, 1)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
            new_log_probs = tf.gather_nd(log_probs, action_indices)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Value Loss
            value_loss = tf.reduce_mean((returns - values) ** 2)
            
            # Entropy
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * log_probs, axis=1)
            entropy_mean = tf.reduce_mean(entropy)
            
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

        trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        # Approximate KL for early stopping (optional, mostly for logging/debugging here)
        approx_kl = tf.reduce_mean(old_log_probs - new_log_probs)
        return loss, policy_loss, value_loss, approx_kl

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        # Note: PPO typically trains on a fixed number of timesteps per update, rather than strictly episodes.
        # But we will adapt to the interface or use a buffer that fills up over episodes.
        
        # For simplicity in this structure: gather 'steps_per_epoch' steps, then update.
        # Overriding the loop slightly to fit the PPO style efficiently.
        
        steps_per_epoch = self.params.get("steps_per_epoch", 2048)
        total_steps = 0
        rewards_history = []
        
        state, _ = env.reset()
        
        # We'll just run for num_episodes effectively, but batched updates happen every steps_per_epoch
        episodes_completed = 0
        
        # Buffers for one epoch
        b_obs, b_acts, b_logprobs, b_rews, b_dones, b_vals = [], [], [], [], [], []
        
        while episodes_completed < num_episodes:
            # Collect experience
            action, log_prob, val = self.get_action_and_val(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            b_obs.append(state)
            b_acts.append(action)
            b_logprobs.append(log_prob)
            b_rews.append(reward)
            b_dones.append(done)
            b_vals.append(val)
            
            state = next_state
            total_steps += 1
            
            term = False
            if done:
                episodes_completed += 1
                rewards_history.append(np.sum(b_rews[-(len(b_rews) - len(b_vals) + 1):])) # This is tricky with the buffer logic, let's just track ep rewards separately
                state, _ = env.reset()
                
            # If buffer is full, update
            if len(b_obs) >= steps_per_epoch:
                # Finish path with bootstrap value if not done
                if not done:
                    last_val = self.critic(tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)).numpy()[0, 0]
                else:
                    last_val = 0
                
                # Compute GAE
                b_advs, b_rets = self.compute_gae(b_rews, b_vals, b_dones, last_val)
                
                # Prepare data
                obs_arr = np.array(b_obs, dtype=np.float32)
                act_arr = np.array(b_acts, dtype=np.float32)
                logprob_arr = np.array(b_logprobs, dtype=np.float32)
                ret_arr = np.array(b_rets, dtype=np.float32)
                adv_arr = np.array(b_advs, dtype=np.float32)
                
                # Normalize advantages
                adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
                
                # Update
                dataset = tf.data.Dataset.from_tensor_slices((obs_arr, act_arr, logprob_arr, ret_arr, adv_arr))
                dataset = dataset.shuffle(steps_per_epoch).batch(self.batch_size)
                
                for _ in range(self.train_iters):
                    for batch in dataset:
                        self.train_step(*batch)
                
                # Clear buffers
                b_obs, b_acts, b_logprobs, b_rews, b_dones, b_vals = [], [], [], [], [], []
                
                # Logging
                if rewards_history:
                    print(f"Update at step {total_steps}. Recent Mean Reward: {np.mean(rewards_history[-10:]):.2f}")

            if evaluate_each and episodes_completed % evaluate_each == 0 and done:
                 print(f"Evaluation after episode {episodes_completed}...")
                 # Manual evaluation loop with recording to match RandomAgent style
                 eval_returns = []
                 recorder = None
                 
                 if self.store_trajectories:
                    # Construct path similar to RandomAgent
                    if not hasattr(self, "log_path_args"):
                        # Build it if not present (RandomAgent does it in init)
                         args = self.params.get("args", {})
                         self.log_path_args = "_".join(f"{k}={v}" for k, v in sorted(args.items()))
                    
                    foldername = f"trajectories/ppo_{self.log_path_args}"
                    import os
                    os.makedirs(foldername, exist_ok=True)
                    log_path = os.path.join(foldername, f"cp_{episodes_completed}.jsonl")
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
                         recorder.start_trajectory(metadata={"agent": "ppo", "checkpoint": episodes_completed})
                     
                     d = False
                     ret = 0
                     while not d:
                         # Eval action (deterministic)
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
