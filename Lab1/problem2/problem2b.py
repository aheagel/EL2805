import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import gymnasium as gym
import torch
from problem2 import running_average, scale_state_variables

def FourierBasis(state: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return np.cos(np.pi * (eta.T @ state))

def SGD_iteration(x, grad_x, learning_rate=0.001):
    return x + learning_rate * grad_x

# FIXA VIKTERNA I MAIN ANNARS funkar den inte!
# Lets do SARSA learning on the advanced maze environment
def SARSA2_learning(lamda, discount=1, W=None, p=2, eta=None, n_episodes=50, eps=0.1, SGD_rate=0.001) -> tuple[np.ndarray, list]:
    """
    SARSA2-learning algorithm for the advanced maze environment.
    Returns:
    - Q: The learned Q-table.
    """
    if eta is None:
        eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T # For small p this is doable
    if W is None:
        W = np.zeros((eta.shape[1], int(env.action_space.n)))  # Random initialization of weights
    
    def epsilon_greedy_policy(V, eps):
        if np.random.rand() < eps:
            action = np.random.randint(len(V))  # Explore: random action
        else:
            best = np.flatnonzero(V == V.max())
            action = np.random.choice(best)  # Exploit: best action from Q-table with random tie-breaking
        return action 
    
    def Value(state, W, action=None, _eta=eta):
        if action is None:
            return W.T @ FourierBasis(state, _eta)
        else:
            return W[:, action].T @ FourierBasis(state, _eta) # This is the Q-value for a specific action
    
    def TD_error(reward, terminal, Q_current, Q_next, _discount=discount):
        if terminal:
            return reward - Q_current
        else:
            return reward + _discount * Q_next - Q_current
        
    def Eligibility_trace(Zold, current_state, current_action, _lamda=lamda, _discount=discount, _eta=eta):
        mask = np.zeros_like(Zold)
        mask[:, current_action] = FourierBasis(current_state, _eta) # Grad term

        return _lamda * _discount * Zold + mask
    

    episode_reward_list = []  # Used to save episodes reward
    for episode in tqdm(range(n_episodes)):
        reward = 0
        z = np.zeros_like(W)  # Eligibility trace initialization
        terminal = False

        current_state = scale_state_variables(env.reset()[0])
        current_value = Value(current_state, W)
        current_action = epsilon_greedy_policy(current_value, eps)
        current_Q = current_value[current_action]

        while not terminal:
            next_state, state_action_reward, done, truncated, _ = env.step(current_action)
            reward += float(state_action_reward)
            terminal = done or truncated

            next_state = scale_state_variables(next_state)
            next_value = Value(next_state, W) # With current policy
            next_action = epsilon_greedy_policy(next_value, eps)
            next_Q = next_value[next_action]

 
            delta = TD_error(state_action_reward, terminal, current_Q, next_Q)
            z = Eligibility_trace(z, current_state, current_action) # Update eligibility trace
            W = SGD_iteration(W, delta * z, learning_rate=SGD_rate) # Update weights

            current_state, current_action, current_value, current_Q = next_state, next_action, next_value, next_Q

        episode_reward_list.append(reward)

    return W, episode_reward_list


if __name__ == "__main__":
    # Import and initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    env.reset()
    
    N_episodes = 200

    # Train SARSA with Fourier Basis
    W_learned, rewards = SARSA2_learning(lamda=0.9,
                                         discount=1,
                                         p=2,
                                         n_episodes=N_episodes,
                                         eps=0.05,
                                         SGD_rate=0.001)


    # Plot Rewards
    plt.plot([i for i in range(1, N_episodes+1)], rewards, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes+1)], running_average(rewards, 50), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

