import numpy as np
from trajectory_graph import TrajectoryGraph

def epsilon_greedy(Q, state, epsilon, action_space):
    """Select an action using the epsilon-greedy strategy."""
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    return np.argmax(Q[state])

def evaluate_policy(env, Q, num_episodes=2000, trajectory_graph=None):
    """Evaluate the learned policy."""
    success = 0
    returns = []
    trajectories = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        ret = 0
        t = []
        while not done:
            if np.all(Q[state] == 0):  # If untrained, pick a random action
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            t.append((state, action, reward, next_state))
            done = terminated or truncated
            
            ret += reward
            if done and reward > 0:
                success += 1
            state = next_state
        trajectories.append(t)
        returns.append(ret)
    print(f"Mean return:{np.mean(returns)} +- {np.std(returns)}")
    if trajectory_graph:
        trajectory_graph.add_checkpoint(trajectories)
    return success / num_episodes

def sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, eval_each=500):
    """SARSA algorithm with periodic evaluation."""
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize Q-table
    eval_results = []  # Store evaluation results
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon, env.action_space.n)
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_action = epsilon_greedy(Q, next_state, epsilon, env.action_space.n) if not done else None
            
            # SARSA update rule
            if not done:
                Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            else:
                Q[state, action] += alpha * (reward - Q[state, action])  # Terminal update

            state, action = next_state, next_action  # Move to the next step
        # Decay epsilon (gradually reduce exploration)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Periodic evaluation
        if episode % eval_each == 0:
            success_rate = evaluate_policy(env, Q)
            eval_results.append((episode, success_rate))
            print(f"Episode {episode}: Success rate = {success_rate:.2f}")

    return Q, eval_results  # Return learned Q-values and evaluation results