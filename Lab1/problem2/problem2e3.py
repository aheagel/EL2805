import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import gymnasium as gym
import pickle
from problem2 import running_average, scale_state_variables
from problem2b import FourierBasis, Value, Nestrov_iteration, make_exponential_schedule

def SARSA3_learning(env, lamda, visits=None, discount=1, W=None, p=2, eta=None, n_episodes=50, curiosity=None, l_rate=None, momentum=0.95, plot=False, debug=False) -> tuple[np.ndarray, list]:
    """
    SARSA2-learning algorithm for the advanced maze environment.
    Returns:
    - Q: The learned Q-table.
    """
    if l_rate is None:
        l_rate = lambda k: 0.001  # Default learning rate
    if visits is None:
        visits = np.zeros((20, 20, 3)) # Discretized state space visits , (dx, dv, a)
    if eta is None:
        eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T # For small p this is doable
    if W is None:
        W = np.zeros((eta.shape[1], int(env.action_space.n)))  # Random initialization of weights
    if curiosity is None:
        eps = lambda k: 1  # Default epsilon value

    normed_eta = np.linalg.norm(eta, ord=2, axis=0) # Normalize learning rate
    episode_reward_list = []  # Used to save episodes reward

    def TD_error(reward, terminal, Q_current, Q_next, _discount=discount):
        if terminal:
            return reward - Q_current
        else:
            return reward + _discount * Q_next - Q_current
        
    def Eligibility_trace(Zold, current_state, current_action, _lamda=lamda, _discount=discount, _eta=eta):
        mask = np.zeros_like(Zold)
        mask[:, current_action] = FourierBasis(current_state, _eta) # Grad term

        return _lamda * _discount * Zold + mask
    
    def advanced_learning_rate(rate, _eta=normed_eta):
        return np.divide(rate, _eta, out=rate*np.ones_like(_eta), where=_eta!=0)[:, np.newaxis]

    def c2d(state: np.ndarray, _visits=visits):
        return np.floor(state * np.array(_visits.shape[0:2])).astype(int)
    
    def UCB_policy(V, visit_counts, total_counts, c):
        ucb_values = V + c * np.sqrt(np.log(total_counts + 1) / (visit_counts + 1e-5))
        best = np.flatnonzero(ucb_values == ucb_values.max())
        action = np.random.choice(best)  # Exploit: best action from Q-table with random tie-breaking
        return action
    
    for episode in tqdm(range(1, n_episodes+1), disable=debug):
        # Switch to rendering for the last episode
        if episode == n_episodes and plot:
            env.close()
            env = gym.make('MountainCar-v0', render_mode='human')

        v = np.zeros_like(W)  # Initialize velocity for Nesterov momentum
        reward = 0
        z = np.zeros_like(W)  # Eligibility trace initialization
        terminal = False

        current_state = scale_state_variables(env.reset()[0])
        current_value = Value(current_state, W, eta)

        idx, idy = c2d(current_state)
        current_action = UCB_policy(current_value, visits[idx, idy, :], np.sum(visits, axis=(0,1,2)), c=curiosity(episode))
        current_Q = current_value[current_action]
        visits[idx, idy, current_action] += 1

        while not terminal:
            next_state, state_action_reward, done, truncated, _ = env.step(current_action)
            reward += float(state_action_reward)
            terminal = done or truncated

            next_state = scale_state_variables(next_state)
            next_value = Value(next_state, W, eta) # With current policy

            idx, idy = c2d(next_state)
            next_action = UCB_policy(next_value, visits[idx, idy, :], np.sum(visits, axis=(0,1,2)), c=curiosity(episode))
            next_Q = next_value[next_action]
            visits[idx, idy, next_action] += 1

            delta = TD_error(state_action_reward, terminal, current_Q, next_Q)
            z = np.clip(Eligibility_trace(z, current_state, current_action), -5, 5) # Update eligibility trace with clipping
            
            adv_rate = advanced_learning_rate(l_rate(episode)) # Decreasing learning rate NAIVE
    
            W, v = Nestrov_iteration(W, v, delta * z, adv_rate, momentum) # Update weights

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
    eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T 

    params={'lambda': 0.7493192913828697, 'lr_initial': 0.0027325349192143084, 'lr_decay': 0.9545766479121087, 'momentum': 0.5057140558849098, 'curiosity_start': 3.462430164865801, 'curiosity_decay': 0.985152028147218}
    learning_rate_schedule = make_exponential_schedule(params['lr_initial'], 0, params['lr_decay'])
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
