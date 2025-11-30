# Anh Do: 20020416-2317
# Saga Tran: 19991105-2182

from problem_1h import MazeAdvanced
from maze import animate_solution2, value_iteration
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl_algorithms_improved import Q_learning_improved

# FIXA VIKTERNA I MAIN ANNARS funkar den inte!
# Lets do Q-learning on the advanced maze environment
def Q_learning(env, start, gamma, n_episodes=50000, number_of_visits=None, Q=None, alpha_func=None, epsilon_func=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Q = np.random.rand(env.n_states, env.n_actions) # Initialize Q-table with random to help with initial exploration
    
    if alpha_func is None:
        alpha_func = lambda n: n**-(2/3)  # Learning rate function

    if epsilon_func is None:
        epsilon_func = lambda k: 0.1 # Fixed exploration rate

    def epsilon_greedy_policy(current_state, eps):
        if np.random.rand() < eps:
            action = np.random.randint(env.n_actions)  # Explore: random action
        else:
            best = np.flatnonzero(Q[current_state] == Q[current_state].max())
            action = np.random.choice(best)  # Exploit: best action from Q-table with random tie-breaking
        return action


    V_starts = np.zeros(n_episodes)  # To store rewards for each episode
    Q[env.map['Done'], :] = 0  # Q-values for terminal state are zero
    for episode in tqdm(range(n_episodes)):
        eps = epsilon_func(episode+1)
        state = env.map[start] # Reset to start state at the beginning of each episode

        while env.states[state] not in ['Done']:
            action = epsilon_greedy_policy(state, eps)
            reward = env.rewards[state, action]
            number_of_visits[state, action] += 1

            mino_states, probs = env.minotaur_states_probs(env.move(state, action))
            next_state = np.random.choice(mino_states, p=probs)

            Q[state, action] += alpha_func(number_of_visits[state, action]) * (reward + gamma * Q[next_state].max() - Q[state, action])

            state = next_state
            
        V_starts[episode] = np.max(Q[env.map[start]])

    return Q, number_of_visits, V_starts

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

    itera = 200001
    alpha0 = ('power', 1, 0.51) # lambda n: n**-0.51
    alpha1 = ('power', 1, 2/3) # lambda n: n**-(2/3)
    epps0 = ('constant', 0.1) # lambda k: 0.1
    epps1 = ('constant', 0.5) # lambda k: 0.5 # Use instead if using the slower version
    
    Q_start = np.random.rand(env.n_states, env.n_actions) # Initialize to optimistic value
    
    Q0, number_of_visits0, v_start0 = Q_learning_improved(env, start, discount, n_episodes=itera, alpha_func=alpha1, epsilon_func=epps0, Q=Q_start.copy())
    Q1, number_of_visits1, v_start1 = Q_learning_improved(env, start, discount, n_episodes=itera, alpha_func=alpha1, epsilon_func=epps1, Q=Q_start.copy()) # This uses numba for faster performance, much faster
    
    policy = np.argmax(Q0, axis=1)

    horizon = 100
    path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)

    animate_solution2(maze, path)

    # Plot the convergence
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(1,itera+1), v_start0, label=r'$\epsilon(k) = 0.1$' + f' Latest Reward {v_start0[-1]}', marker='o', markerfacecolor='none', markeredgecolor='blue', markersize=4, markevery=10000, linewidth=1)
    plt.plot(np.arange(1,itera+1), v_start1, label=r'$\epsilon(k) = 0.5$' + f' Latest Reward {v_start1[-1]}', marker='x', markersize=4, markevery=10000, linewidth=1)
    plt.axhline(y=V_star[env.map[start]], color='k', linestyle='--', label=f'Optimal Reward Approximation (Value Iteration) {V_star[env.map[start]]}')
    plt.axvline(x=50000, color='b', linestyle=':', label='50000 Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Value at Start State approximations')
    plt.title(r'Convergence of Q-Learning with different $\epsilon$ and $\alpha=2/3$')
    plt.legend()
    plt.grid(True)
    plt.show()
