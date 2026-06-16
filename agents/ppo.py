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
        self.train_iters = self.params.get("train_iters", 4)  # Epochs per update (lower to prevent policy collapse in stochastic envs)
        self.batch_size = self.params.get("batch_size", 64)
        self.lr = self.params.get("lr", 2.5e-4)
        self.target_kl = self.params.get("target_kl", 0.015)
        self.initial_entropy_coef = self.params.get("entropy_coef", 0.05)
        self.entropy_min = self.params.get("entropy_min", 0.0)
        self.value_coef = self.params.get("value_coef", 0.5)
        self.hidden_layers = self.params.get("hidden_layers", [64, 64])
        self.max_grad_norm = self.params.get("max_grad_norm", 0.5)
        self.anneal_lr = self.params.get("anneal_lr", True)
        self.clip_vloss = self.params.get("clip_vloss", False)

        # Policy Network
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = optimizers.Adam(learning_rate=self.lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=self.lr)

    def _build_actor(self):
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

        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        value = layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=value)

    @tf.function
    def _step_tf(self, state_tensor, training):
        logits = self.actor(state_tensor)
        if training:
            action = tf.random.categorical(logits, 1)[0, 0]
        else:
            action = tf.argmax(logits, axis=1)[0]
        return action

    def step(self, state, training=False):
        # Expect state to be (obs_dim,) or (1, obs_dim)
        if not isinstance(state, np.ndarray):
            state = np.array([state])
            
        if self.is_discrete_obs:
             state_tensor = tf.convert_to_tensor(state.reshape(1, 1), dtype=tf.int32)
        else:
             state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)

        action = self._step_tf(state_tensor, tf.convert_to_tensor(training, dtype=tf.bool))
            
        return int(action.numpy())

    @tf.function
    def _get_action_and_val_tf(self, state_tensor):
        logits = self.actor(state_tensor)
        value = self.critic(state_tensor)
        
        action = tf.random.categorical(logits, 1)[0, 0]
        log_prob = tf.nn.log_softmax(logits)
        action_log_prob = log_prob[0, action]
        
        return action, action_log_prob, value[0, 0]

    def get_action_and_val(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array([state])
            
        if self.is_discrete_obs:
             state_tensor = tf.convert_to_tensor(state.reshape(1, 1), dtype=tf.int32)
        else:
             state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)

        action, log_prob, val = self._get_action_and_val_tf(state_tensor)
        return int(action.numpy()), log_prob.numpy(), val.numpy()

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae_lam = 0
        values = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae_lam
            
        returns = advantages + values[:-1]
        return advantages, returns

    @tf.function
    def train_step(self, states, actions, old_log_probs, returns, advantages, old_values, entropy_coef):
        # Actor update
        with tf.GradientTape() as actor_tape:
            logits = self.actor(states)
            
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
            
            # Entropy
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * log_probs, axis=1)
            entropy_mean = tf.reduce_mean(entropy)
            
            actor_loss = policy_loss - entropy_coef * entropy_mean

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Critic update with value loss clipping (PPO2-style)
        with tf.GradientTape() as critic_tape:
            values = self.critic(states)
            values = tf.squeeze(values, axis=-1)
            if self.clip_vloss:
                # Clip value predictions to prevent large critic updates
                v_clipped = old_values + tf.clip_by_value(
                    values - old_values, -self.clip_ratio, self.clip_ratio
                )
                v_loss_unclipped = (values - returns) ** 2
                v_loss_clipped = (v_clipped - returns) ** 2
                value_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss_unclipped, v_loss_clipped))
            else:
                value_loss = 0.5 * tf.reduce_mean((returns - values) ** 2)

        critic_grads = critic_tape.gradient(value_loss, self.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        loss = actor_loss + self.value_coef * value_loss
        
        # Approximate KL(old || new) for early stopping
        approx_kl = tf.reduce_mean(old_log_probs - new_log_probs)
        return loss, policy_loss, value_loss, tf.abs(approx_kl)

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        # Note: PPO typically trains on a fixed number of timesteps per update, rather than strictly episodes.
        # But we will adapt to the interface or use a buffer that fills up over episodes.
        
        # For simplicity in this structure: gather 'steps_per_epoch' steps, then update.
        # Overriding the loop slightly to fit the PPO style efficiently.
        
        steps_per_epoch = self.params.get("steps_per_epoch", 2048)
        # Estimate total timesteps for step-based LR annealing
        # Use a rough estimate: num_episodes * avg_episode_length
        # We'll refine this as we train, but start with steps_per_epoch * expected_updates
        self._total_timesteps = self.params.get("total_timesteps", num_episodes * steps_per_epoch // 10)
        total_steps = 0
        num_updates = 0
        rewards_history = []
        
        state, _ = env.reset()
        
        # We'll just run for num_episodes effectively, but batched updates happen every steps_per_epoch
        episodes_completed = 0
        episode_reward = 0  # Accumulator for current episode return
        pending_eval = False  # Flag for triggering evaluation after buffer updates
        
        # Buffers for one epoch
        b_obs, b_acts, b_logprobs, b_rews, b_dones, b_vals = [], [], [], [], [], []
        
        while episodes_completed < num_episodes:
            # Collect experience from stochastic policy (exploration via entropy bonus)
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
            episode_reward += reward
            
            if done:
                episodes_completed += 1
                rewards_history.append(episode_reward)
                episode_reward = 0
                state, _ = env.reset()
                if evaluate_each and episodes_completed % evaluate_each == 0:
                    pending_eval = True
                
            # If buffer is full, update
            if len(b_obs) >= steps_per_epoch:
                # Finish path with bootstrap value if not done
                if not done:
                    # Handle bootstrap value get
                    if not isinstance(state, np.ndarray):
                        s_val = np.array([state])
                    else:
                        s_val = state
                    
                    if self.is_discrete_obs:
                        val_input = tf.convert_to_tensor(s_val.reshape(1, 1), dtype=tf.int32)
                    else:
                        val_input = tf.convert_to_tensor(s_val.reshape(1, -1), dtype=tf.float32)

                    last_val = self.critic(val_input).numpy()[0, 0]
                else:
                    last_val = 0
                
                # Compute GAE
                b_advs, b_rets = self.compute_gae(b_rews, b_vals, b_dones, last_val)
                
                # Prepare data
                if self.is_discrete_obs:
                    obs_arr = np.array(b_obs, dtype=np.int32).reshape(-1, 1)
                else:
                    obs_arr = np.array(b_obs, dtype=np.float32)

                act_arr = np.array(b_acts, dtype=np.float32)
                logprob_arr = np.array(b_logprobs, dtype=np.float32)
                ret_arr = np.array(b_rets, dtype=np.float32)
                adv_arr = np.array(b_advs, dtype=np.float32)
                val_arr = np.array(b_vals, dtype=np.float32)
                
                # Normalize advantages
                adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
                
                # Step-based linear annealing
                frac = max(0.0, 1.0 - (total_steps / self._total_timesteps))
                
                # Learning rate annealing
                if self.anneal_lr:
                    self.actor_optimizer.learning_rate.assign(self.lr * frac)
                    self.critic_optimizer.learning_rate.assign(self.lr * frac)
                    
                # Entropy coefficient annealing (can be faster than LR)
                entropy_decay_frac = self.params.get("entropy_decay_frac", 1.0)
                e_frac = max(0.0, 1.0 - (total_steps / (self._total_timesteps * entropy_decay_frac)))
                current_entropy_coef = self.entropy_min + e_frac * (self.initial_entropy_coef - self.entropy_min)
                entropy_coef_tf = tf.convert_to_tensor(current_entropy_coef, dtype=tf.float32)
                
                # Update
                num_updates += 1
                indices = np.arange(len(obs_arr))
                
                for epoch in range(self.train_iters):
                    np.random.shuffle(indices)
                    kl_exceeded = False
                    for start in range(0, len(obs_arr), self.batch_size):
                        batch_idx = indices[start:start+self.batch_size]
                        
                        b_obs_tf = tf.convert_to_tensor(obs_arr[batch_idx])
                        b_act_tf = tf.convert_to_tensor(act_arr[batch_idx])
                        b_logprob_tf = tf.convert_to_tensor(logprob_arr[batch_idx])
                        b_ret_tf = tf.convert_to_tensor(ret_arr[batch_idx])
                        b_adv_tf = tf.convert_to_tensor(adv_arr[batch_idx])
                        b_val_tf = tf.convert_to_tensor(val_arr[batch_idx])
                        
                        loss, p_loss, v_loss, approx_kl = self.train_step(
                            b_obs_tf, b_act_tf, b_logprob_tf, b_ret_tf, b_adv_tf, b_val_tf, entropy_coef_tf
                        )
                        
                        # Early stopping on KL divergence to prevent destructive updates
                        if approx_kl > 1.5 * self.target_kl:
                            kl_exceeded = True
                            break
                    if kl_exceeded:
                        break
                
                # Clear buffers
                b_obs, b_acts, b_logprobs, b_rews, b_dones, b_vals = [], [], [], [], [], []
                
                # Logging
                if rewards_history:
                    lr_now = float(self.actor_optimizer.learning_rate)
                    print(f"Update #{num_updates} at step {total_steps}. Recent Mean Reward: {np.mean(rewards_history[-10:]):.2f}, LR: {lr_now:.2e}, Entropy Coef: {current_entropy_coef:.4f}")

            if pending_eval:
                 pending_eval = False
                 print(f"Evaluation after episode {episodes_completed}...")
                 # Manual evaluation loop with recording to match RandomAgent style
                 eval_returns = []
                 recorder = None
                 
                 if self.store_trajectories:
                    # Construct path similar to RandomAgent
                     if not hasattr(self, "log_path_args"):
                          args = self.params.get("args", {})
                          filtered_args = {k: v for k, v in args.items() if k not in ["model", "env", "episodes", "evaluate_each", "evaluate_for", "seed", "log_dir"]}
                          self.log_path_args = "_".join(f"{k}={v}" for k, v in sorted(filtered_args.items()))
                     
                     import os
                     if "log_dir" in self.params:
                         base_dir = self.params["log_dir"]
                         foldername = os.path.join(base_dir, f"ppo_{self.log_path_args}")
                     else:
                         foldername = f"trajectories/ppo_{self.log_path_args}"
                     os.makedirs(foldername, exist_ok=True)
                     log_path = os.path.join(foldername, f"cp_{episodes_completed:05d}.jsonl")
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
