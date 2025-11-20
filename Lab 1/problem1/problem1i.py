from problem1h import MazeAdvanced
import numpy as np
from scipy.stats import geom
from maze import animate_solution2, value_iteration
import matplotlib.pyplot as plt

# Lets do Q-learning on the advanced maze environment
def Q_learning(env, start, n_episodes=50000, number_of_visits=None, Q=None, alpha=None, gamma=0.99, epsilon=0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Q-learning algorithm for the advanced maze environment. epsilon-greedy policy is used for action selection.
    
    Parameters:
    - env: The MazeAdvanced environment.
    - n_episodes: Number of episodes to train.
    - max_steps: Maximum steps per episode.
    - gamma: Discount factor.
    - epsilon: Exploration rate.
    
    Returns:
    - Q: The learned Q-table.
    """

    if number_of_visits is None:
        number_of_visits = np.zeros((env.n_states, env.n_actions))  # To keep track of state-action visits

    if Q is None:
        Q = np.zeros((env.n_states, env.n_actions)) # Initialize Q-table with zeros
    
    if alpha is None:
        alpha = lambda n: n**-(2/3)  # Learning rate function
    
    rewards = np.zeros(n_episodes)  # To store rewards for each episode
    for episode in range(n_episodes):
        state = env.map[start] # Reset to start state at the beginning of each episode
        while True:#for step in range(geom.rvs(1-gamma)): # Steps before Death
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)  # Explore: random action
            else:
                action = np.argmax(Q[state])  # Exploit: best action from Q-table

            number_of_visits[state, action] += 1

            probs = env.minotaur_probs(env.move(state, action))
            next_state = np.random.choice(env.n_states, p=probs)

            reward = env.rewards[state, action]
            rewards[episode] += reward

            Q[state, action] += alpha(number_of_visits[state, action]) * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            state = next_state
            if env.states[next_state] in ['Done']:
                break
                
    return Q, number_of_visits, rewards

if __name__ == "__main__":
    # Define the maze layout
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze, 3 = key
    env = MazeAdvanced(maze, prob_to_player=0.35, still_minotaur=False)

    # Define the discount and an accuracy threshold
    discount = 49/50
    start  = ((0,0), (6,5), False)

    V_star, _ = value_iteration(env, discount, 1e-12)

    itera = 50000
    alpha0 = lambda n: n**(-2/3)
    alpha1 = lambda n: n**(-3/4)
    alpha2 = lambda n: n**(-1)

    Q0, number_of_visits0, rewards0 = Q_learning(env, start, n_episodes=itera, alpha=alpha0, gamma=discount, epsilon=0.1)
    Q1, number_of_visits1, rewards1 = Q_learning(env, start, n_episodes=itera, alpha=alpha1, gamma=discount, epsilon=0.1)
    Q2, number_of_visits2, rewards2 = Q_learning(env, start, n_episodes=itera, alpha=alpha2, gamma=discount, epsilon=0.1)
    
    policy = np.argmax(Q0, axis=1)
    horizon = 100
    path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)

    animate_solution2(maze, path)

    # Plot the Rewards
    plt.figure(figsize=(10, 6))
    cumsum_rewards0 = np.cumsum(rewards0) / np.arange(1, itera+1)
    cumsum_rewards1 = np.cumsum(rewards1) / np.arange(1, itera+1)
    cumsum_rewards2 = np.cumsum(rewards2) / np.arange(1, itera+1)

    plt.plot(np.arange(1,itera+1), cumsum_rewards0, label=r'$\alpha(n) = n^{-2/3}$', marker='o', markerfacecolor='none', markeredgecolor='blue', markersize=4, markevery=100, linewidth=1)
    plt.plot(np.arange(1,itera+1), cumsum_rewards1, label=r'$\alpha(n) = n^{-3/4}$', marker='x', markersize=4, markevery=100, linewidth=1)
    plt.plot(np.arange(1,itera+1), cumsum_rewards2, label=r'$\alpha(n) = n^{-1}$', marker='+', markersize=4, markevery=100, linewidth=1)
    plt.axhline(y=1, color='k', linestyle='--', label='Optimal Reward Approximation')

    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Average Rewards')
    plt.title('Convergence of Q-Learning with different alpha')
    plt.legend()
    plt.grid(True)
    plt.show()
