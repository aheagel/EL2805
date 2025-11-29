import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import gymnasium as gym
import pickle
from problem2 import running_average
from problem2b import make_exponential_schedule, make_polynomial_schedule
from numba_utils import *

def SARSA3_learning(env, lamda, visits=None, discount=1, W=None, p=2, eta=None, n_episodes=50, curiosity=None, l_rate=None, momentum=0.95, plot=False, debug=False) -> tuple[np.ndarray, list]:
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

    if visits is None:
        visits = np.zeros((20, 20, 3)) # Discretized state space visits , (dx, dv, a)
    if eta is None:
        eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)], dtype=np.float64).T # For small p this is doable
    
    # Ensure eta is float64 for Numba compatibility
    eta = eta.astype(np.float64)

    if W is None:
        W = np.zeros((eta.shape[1], int(env.action_space.n)))  # Random initialization of weights
    if curiosity is None:
        curiosity = lambda k: 1  # Default epsilon value
    elif not callable(curiosity):
        val_cur = curiosity
        curiosity = lambda k: val_cur

    normed_eta = np.linalg.norm(eta, ord=2, axis=0) # Normalize learning rate
    episode_reward_list = []  # Used to save episodes reward
    
    low = env.observation_space.low
    high = env.observation_space.high
    visits_shape = np.array(visits.shape[0:2])
    total_visits = np.sum(visits)

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
        _curiosity = curiosity(episode)

        current_state = scale_state_numba(env.reset()[0].astype(np.float64), low, high)
        current_value = value_numba(current_state, W, eta)

        idx = c2d_numba(current_state, visits_shape)
        current_action = ucb_policy_numba(current_value, visits[idx[0], idx[1], :], total_visits, c=_curiosity)
        current_Q = current_value[current_action]
        visits[idx[0], idx[1], current_action] += 1
        total_visits += 1

        while not terminal:
            next_state, state_action_reward, done, truncated, _ = env.step(current_action)
            reward += float(state_action_reward)
            terminal = done or truncated

            next_state = scale_state_numba(next_state.astype(np.float64), low, high)
            next_value = value_numba(next_state, W, eta) # With current policy

            idx_next = c2d_numba(next_state, visits_shape)
            next_action = ucb_policy_numba(next_value, visits[idx_next[0], idx_next[1], :], total_visits, c=_curiosity)
            next_Q = next_value[next_action]
            visits[idx_next[0], idx_next[1], next_action] += 1
            total_visits += 1

            delta = td_error_numba(state_action_reward, terminal, current_Q, next_Q, discount)
            z = eligibility_trace_numba(z, current_state, current_action, lamda, discount, eta)
            z = np.clip(z, -5, 5) # Update eligibility trace with clipping
            
            adv_rate = advanced_learning_rate_numba(_lrate, normed_eta) # Decreasing learning rate NAIVE
    
            W, v = nesterov_iteration_numba(W, v, delta * z, adv_rate, momentum) # Update weights

            current_state, current_action, current_value, current_Q = next_state, next_action, next_value, next_Q

        episode_reward_list.append(reward)

    return W, episode_reward_list

if __name__ == "__main__":
    # Import and initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    env.reset()

    # 1. PARAMETERS FOR FAST CONVERGENCE
    N_episodes = 200  # Stick to teacher's limit
    p = 2
    eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)], dtype=np.float64).T 
    
    params={'lambda': 0.9462001379527284, 'momentum': 0.8149743131779319, 'lr_initial': 0.002501651424747113, 'lr_scale': 1.2503832722688262, 'lr_power': 0.9489490185683979, 'curiosity_start': 1.9452404547124045, 'curiosity_decay': 0.979474624065298}
    learning_rate_schedule = make_polynomial_schedule(params['lr_initial'], 0, params['lr_scale'], params['lr_power'])
    curiosity_schedule = make_exponential_schedule(params['curiosity_start'], 0, params['curiosity_decay'])

    print(f"Training SARSA with Fourier Order p={p}...")
    W_learned, rewards = SARSA3_learning(env,
                                        lamda=params['lambda'],       # High trace decay for faster credit assignment
                                        discount=1,
                                        p=p,
                                        n_episodes=N_episodes,
                                        curiosity=curiosity_schedule,  # Decreasing exploration
                                        l_rate=learning_rate_schedule,          # Slightly higher learning rate
                                        eta=eta,
                                        plot=True,        # Render last episode
                                        momentum=params['momentum']
                                        )

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

    pickle.dump({"W": W_learned.T, "N": eta.T}, open(sys.path[0] + '/weights.pkl', 'wb')) # used to save
    exec(open(os.path.join(sys.path[0], 'check_solution.py')).read())
