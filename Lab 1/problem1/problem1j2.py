from problem1h import MazeAdvanced
from problem1j import SARSA_learning
from maze import animate_solution2, value_iteration
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    epps0 = lambda k: k**(-0.6)
    epps1 = lambda k: k**(-0.8)
    
    Q_start = np.random.rand(env.n_states, env.n_actions)

    Q0, number_of_visits0, v_start0 = SARSA_learning(env, start, discount, n_episodes=itera, alpha=alpha0, epsilon=epps0, Q=Q_start.copy())
    Q1, number_of_visits1, v_start1 = SARSA_learning(env, start, discount, n_episodes=itera, alpha=alpha0, epsilon=epps1, Q=Q_start.copy())
    
    policy = np.argmax(Q0, axis=1)

    horizon = 100
    path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)

    animate_solution2(maze, path)

    # Plot the convergence
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(1,itera+1), v_start0, label=r'$\epsilon = k^{-0.51}$', marker='o', markerfacecolor='none', markeredgecolor='blue', markersize=4, markevery=10000, linewidth=1)
    plt.plot(np.arange(1,itera+1), v_start1, label=r'$\epsilon = k^{-0.75}$', marker='x', markersize=4, markevery=10000, linewidth=1)
    plt.axhline(y=V_star[env.map[start]], color='k', linestyle='--', label=f'Optimal Reward Approximation {V_star[env.map[start]]:.4f}')

    plt.xlabel('Iterations')
    plt.ylabel('Value at Start State approximations')
    plt.title('Convergence of SARSA-Learning with different epsilon')
    plt.legend()
    plt.grid(True)
    plt.show()