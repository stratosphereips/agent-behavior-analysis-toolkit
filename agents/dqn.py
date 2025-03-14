import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.005
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10  # How often to update the target network
NUM_EPISODES = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Q-network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self._state_size = state_size

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, num_classes=self.state_size).float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Select action using epsilon-greedy policy
def select_action(state, policy_net, epsilon, env):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state = torch.tensor([state], dtype=torch.long, device=device)
        return torch.argmax(policy_net(state)).item()


# Train the DQN
def train_dqn(env, eval_each, batch_size, gamma, learning_rate, memory_size, num_episodes, target_update):
    # Initialize policy and target networks
    policy_net = DQN(env.state_space.n, env.action_space.n).to(device)
    target_net = DQN(env.state_space.n, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())  # Copy initial weights

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_size)
    epsilon = EPSILON_START

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(state, policy_net, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Adjust reward for learning stability
            if terminated and reward == 1.0:
                reward = 10.0  # Large reward for reaching the goal
            elif terminated and reward == 0.0:
                reward = -10.0  # Penalty for falling into a hole
            else:
                reward = -0.1  # Small penalty for each step to encourage efficiency

            # Store experience in memory
            memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Sample a batch and train
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.long, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.long, device=device)
                dones = torch.tensor(dones, dtype=torch.bool, device=device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0].detach()
                target_q_values = rewards + gamma * next_q_values * (~dones)

                loss = nn.MSELoss()(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        #epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(-EPSILON_DECAY * episode)
        # Print progress
        if episode % eval_each == 0:
            evaluate_dqn(policy_net, env)
            print(f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    return policy_net

# Evaluate the final policy
def evaluate_dqn(model, env, num_episodes=1000):
    success = 0
    trajectories = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        ret = 0
        t = []
        while not done:
            action = select_action(state, model, epsilon=0.0)  # Greedy action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            t.append((state, action, reward, next_state))
            if done and reward == 1.0:
                success += 1
            state = next_state
        trajectories.append(t)
    TG_DQN_FROZEN.add_checkpoint(trajectories)
    return success / num_episodes