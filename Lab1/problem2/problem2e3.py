import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import gymnasium as gym
import pickle
from problem2 import running_average, scale_state_variables
from problem2b import FourierBasis, Value, Nestrov_iteration

def SARSA3_learning(env, lamda, visits=None, discount=1, W=None, p=2, eta=None, n_episodes=50, curiosity=None, l_rate=0.001, plot=False) -> tuple[np.ndarray, list]:
    """
    SARSA2-learning algorithm for the advanced maze environment.
    Returns:
    - Q: The learned Q-table.
    """

    if visits is None:
        visits = np.zeros((100, 10)) # Discretized state space visits
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
    
    def advanced_learning_rate(l_rate, _eta=normed_eta):
        return np.divide(l_rate, _eta, out=l_rate*np.ones_like(_eta), where=_eta!=0)[:, np.newaxis]

    def c2d(state: np.ndarray, _visits=visits):
        return np.floor(state * np.array(_visits.shape)).astype(int)
    
    def UCB_policy(V, visit_counts, total_counts, c):
        ucb_values = V + c * np.sqrt(np.log(total_counts + 1) / (visit_counts + 1e-5))
        best = np.flatnonzero(ucb_values == ucb_values.max())
        action = np.random.choice(best)  # Exploit: best action from Q-table with random tie-breaking
        return action
    
    for episode in tqdm(range(1, n_episodes+1)):
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
        current_action = UCB_policy(current_value, visits[c2d(current_state)], np.sum(visits), c=curiosity(episode))
        current_Q = current_value[current_action]
        visits[c2d(current_state)] += 1

        while not terminal:
            next_state, state_action_reward, done, truncated, _ = env.step(current_action)
            reward += float(state_action_reward)
            terminal = done or truncated

            next_state = scale_state_variables(next_state)
            next_value = Value(next_state, W, eta) # With current policy
            next_action = UCB_policy(next_value, visits[c2d(next_state)], np.sum(visits), c=curiosity(episode))
            next_Q = next_value[next_action]
            visits[c2d(next_state)] += 1


            delta = TD_error(state_action_reward, terminal, current_Q, next_Q)
            z = np.clip(Eligibility_trace(z, current_state, current_action), -5, 5) # Update eligibility trace with clipping
            
            adv_rate = advanced_learning_rate(l_rate) # Decreasing learning rate NAIVE
    
            W, v = Nestrov_iteration(W, v, delta * z, adv_rate, 0.1) # Update weights

            current_state, current_action, current_value, current_Q = next_state, next_action, next_value, next_Q

        episode_reward_list.append(reward)

    return W, episode_reward_list

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

N_episodes = 400
p=2
eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T # For small p this is doable

#eta = eta[:, 1:]
#print(eta) For plot later

# Train SARSA with Fourier Basis
W_learned, rewards = SARSA3_learning(env,
                                        lamda=0.85,
                                        discount=1,
                                        p=p,
                                        n_episodes=N_episodes,
                                        curiosity=lambda k: 0.0001,
                                        l_rate=0.0005,
                                        eta=eta,
                                        plot=True,
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

#pickle.dump({"W": W_learned.T, "N": eta.T}, open(sys.path[0] + '/weights.pkl', 'wb')) # used to save
