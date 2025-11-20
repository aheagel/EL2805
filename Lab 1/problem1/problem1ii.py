from problem1h import MazeAdvanced
from problem1i import Q_learning
import numpy as np
from scipy.stats import geom
from maze import animate_solution2, value_iteration
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
    env = MazeAdvanced(maze, prob_to_player=1, still_minotaur=False)

    # Define the discount and an accuracy threshold
    discount = 49/50
    start  = ((0,0), (6,5), False)

    V_star, _ = value_iteration(env, discount, 1e-12)

    itera = 50000
    alpha0 = lambda n: n**(-2/3)
    alpha1 = lambda n: n**(-3/4)
    alpha2 = lambda n: n**(-1)

    Q0, number_of_visits0, v_start0 = Q_learning(env, start, n_episodes=itera, alpha=alpha0, gamma=discount, epsilon=0.1)
    Q1, number_of_visits1, v_start1 = Q_learning(env, start, n_episodes=itera, alpha=alpha1, gamma=discount, epsilon=0.1)
    Q2, number_of_visits2, v_start2 = Q_learning(env, start, n_episodes=itera, alpha=alpha2, gamma=discount, epsilon=0.1)
    
    policy = np.argmax(Q0, axis=1)
    horizon = 100
    path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)

    animate_solution2(maze, path)

    # Plot the convergence
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(1,itera+1), v_start0, label=r'$\alpha(n) = n^{-2/3}$', marker='o', markerfacecolor='none', markeredgecolor='blue', markersize=4, markevery=100, linewidth=1)
    plt.plot(np.arange(1,itera+1), v_start1, label=r'$\alpha(n) = n^{-3/4}$', marker='x', markersize=4, markevery=100, linewidth=1)
    plt.plot(np.arange(1,itera+1), v_start2, label=r'$\alpha(n) = n^{-1}$', marker='+', markersize=4, markevery=100, linewidth=1)
    plt.axhline(y=V_star[env.map[start]], color='k', linestyle='--', label='Optimal Reward Approximation')

    plt.xlabel('Iterations')
    plt.ylabel('Value at Start State approximations')
    plt.title('Convergence of Q-Learning with different alpha')
    plt.legend()
    plt.grid(True)
    plt.show()