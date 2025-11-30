# Anh Do: 20020416-2317
# Saga Tran: 19991105-2182

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import gymnasium as gym
import pickle
from problem2 import running_average
from numba_utils import *

def FourierBasis(state: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return fourier_basis_numba(state, eta)

def Value(state, W, eta, action=None):
    if action is None:
        return value_numba(state, W, eta)
    else:
        return value_action_numba(state, W, eta, action)

def Nestrov_iteration(x, v, grad_x, learning_rate, momentum):
    return nesterov_iteration_numba(x, v, grad_x, learning_rate, momentum)

def make_exponential_schedule(start: float, end: float, decay: float):
	"""Create an exponential decay schedule that never drops below `end`."""

	def schedule(step: int) -> float:
		value = start * (decay ** max(step - 1, 0))
		return max(value, end)

	return schedule

def make_polynomial_schedule(start: float, end: float, scale: float, power: float):
    """Create a polynomial (0,1] decay schedule """

    def schedule(step: int) -> float:
        decay_factor = (step / scale) ** power
        value = start / decay_factor
        return max(value, end)

    return schedule


def SARSA2_learning(env, lamda, discount=1, W=None, p=2, eta=None, n_episodes=50, eps=None, l_rate=None, momentum=0.95, plot=False, debug=False) -> tuple[np.ndarray, list]:
    """
    SARSA2-learning algorithm for the advanced maze environment.
    Returns:
    - Q: The learned Q-table.
    """

    if l_rate is None:
        l_rate = lambda k: 0.001  # Default learning rate
    elif not callable(l_rate):
        val_lr = l_rate
        l_rate = lambda k: val_lr

    if eta is None:
        eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T # For small p this is doable
    
    # Ensure eta is float64 for Numba compatibility
    eta = eta.astype(np.float64)

    if W is None:
        W = np.zeros((eta.shape[1], int(env.action_space.n)))  # Zeros initialization
    if eps is None:
        eps = lambda k: 0.1  # Default epsilon value
    elif not callable(eps):
        val_eps = eps
        eps = lambda k: val_eps

    normed_eta = np.linalg.norm(eta, ord=2, axis=0) # Normalize learning rate
    episode_reward_list = []  # Used to save episodes reward
    
    low = env.observation_space.low
    high = env.observation_space.high

    for episode in tqdm(range(1, n_episodes+1), disable=debug):
        # Switch to rendering for the last episode
        if episode == n_episodes and plot:
            env.close()
            env = gym.make('MountainCar-v0', render_mode='human')

        v = np.zeros_like(W)  # Initialize velocity for Nesterov momentum
        reward = 0
        z = np.zeros_like(W)  # Eligibility trace initialization
        terminal = False

        # Pre calculations
        _lrate = l_rate(episode)
        _eps = eps(episode)

        current_state = scale_state_numba(env.reset()[0].astype(np.float64), low, high)
        current_value = value_numba(current_state, W, eta)

        current_action = epsilon_greedy_policy_numba(current_value, _eps)
        current_Q = current_value[current_action]

        while not terminal:
            next_state, state_action_reward, done, truncated, _ = env.step(current_action)
            reward += float(state_action_reward)
            terminal = done or truncated

            next_state = scale_state_numba(next_state.astype(np.float64), low, high)
            next_value = value_numba(next_state, W, eta) # With current policy
            next_action = epsilon_greedy_policy_numba(next_value, _eps)
            next_Q = next_value[next_action]


            delta = td_error_numba(state_action_reward, terminal, current_Q, next_Q, discount)
            z = eligibility_trace_numba(z, current_state, current_action, lamda, discount, eta)
            z = np.clip(z, -5, 5) # Update eligibility trace with clipping
            
            adv_rate = advanced_learning_rate_numba(_lrate, normed_eta) # Decreasing learning rate NAIVE
    
            W, v = nesterov_iteration_numba(W, v, delta * z, adv_rate, momentum) # Update weights

            current_state, current_action, current_value, current_Q = next_state, next_action, next_value, next_Q

        episode_reward_list.append(reward)
        # env.reset() # Removed redundant reset

    return W, episode_reward_list


if __name__ == "__main__":
    # Import and initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    env.reset()
    
    N_episodes = 200
    p=2
    eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T # For small p this is doable
    params={'lambda': 0.8540592523120906, 'momentum': 0.9983674411507302, 'lr_initial': 6.644641407969861e-05, 'lr_scale': 21.477703280373976, 'lr_power': 0.6740471797607481, 'eps_start': 0.0021543255282344886, 'eps_decay': 0.9337869414588281}
    epsilon_schedule = make_exponential_schedule(params['eps_start'], 0, params['eps_decay'])
    learning_rate_schedule = make_polynomial_schedule(params['lr_initial'], 0, params['lr_scale'], params['lr_power'])
    #eta = eta[:, 1:]
    #print(eta) For plot later

    # Train SARSA with Fourier Basis
    W_learned, rewards = SARSA2_learning(env,
                                         #W=np.random.randn(eta.shape[1], int(env.action_space.n)) * -140,
                                         lamda=params['lambda'],
                                         discount=1,
                                         p=p,
                                         n_episodes=N_episodes,
                                         eps=lambda k: epsilon_schedule(k),
                                         l_rate=lambda k: learning_rate_schedule(k),
                                         eta=eta,
                                         momentum=params['momentum'],
                                         plot=True)

    # Plot Rewards plot 1
    plt.plot([i for i in range(1, N_episodes+1)], rewards, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes+1)], running_average(rewards, 50), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    env.close()

    #pickle.dump({"W": W_learned.T, "N": eta.T}, open(sys.path[0] + '/weights.pkl', 'wb')) # used to save
    #exec(open(os.path.join(sys.path[0], 'check_solution.py')).read())
