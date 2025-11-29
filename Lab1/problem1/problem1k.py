from tqdm import tqdm
import matplotlib.pyplot as plt
from problem1i2 import Q_learning
from problem1j2 import SARSA_learning
from problem1h import MazeAdvanced
from maze import *
from rl_algorithms_improved import Q_learning_improved, SARSA_learning_improved
from joblib import Parallel, delayed
import numpy as np

def run_simulation(env, start, discount, Q_policy, S_policy):
    # Note: Each process gets its own random state, so this is safe
    horizon = np.random.geometric(1-discount)
    
    # Reshape policies inside the function to avoid passing large arrays repeatedly if possible, 
    # but passing them as args is fine for this size.
    q_pol_reshaped = np.repeat(Q_policy.reshape(len(Q_policy), 1), horizon, 1)
    s_pol_reshaped = np.repeat(S_policy.reshape(len(S_policy), 1), horizon, 1)

    path_Q = env.simulate(start, q_pol_reshaped, horizon)
    path_S = env.simulate(start, s_pol_reshaped, horizon)
    
    # Return the boolean results directly
    q_win = (path_Q[-2] == 'Win')
    s_win = (path_S[-2] == 'Win')
    
    return q_win, s_win

if __name__ == "__main__":
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze, 3 = key

    # Example usage and comparison of Q-learning and SARSA-learning on the advanced maze environment
    env = MazeAdvanced(maze, prob_to_player=0.35, still_minotaur=False)
    discount = 49/50
    start  = ((0,0), (6,5), False)
    n_episodes = 1000000
    Q_start = np.ones((env.n_states, env.n_actions))

    alphaQ = ("power", 1, 0.501) #lambda n: n**(-0.501)
    eppsQ = ("constant", 0.2) #lambda k: 0.2 #k**(-0.4)

    alphaS = ("power", 1, 0.501) #lambda n: n**(-0.501)
    eppsS = ("power", 1, 0.95) #lambda k: k**(-1)

    # Q-learning
    Q_qlearning, visits_qlearning, V_qlearning = Q_learning_improved(env, start, discount, n_episodes=n_episodes, alpha_func=alphaQ, epsilon_func=eppsQ, Q=Q_start.copy())
    # SARSA-learning
    Q_sarsa, visits_sarsa, V_sarsa = SARSA_learning_improved(env, start, discount, n_episodes=n_episodes, alpha_func=alphaS, epsilon_func=eppsS, Q=Q_start.copy())

    # Extract policies
    Q_policy = np.argmax(Q_qlearning, axis=1)
    S_policy = np.argmax(Q_sarsa, axis=1)

    # Reference 
    V_star, _ = value_iteration(env, discount, 1e-12)
    # Compare results (for example, print the learned Q-tables)
    # Plot the convergence
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(1,n_episodes+1), V_qlearning, label=f'Q learning last Value: {V_qlearning[-1]}' , marker='o', markerfacecolor='none', markeredgecolor='blue', markersize=4, markevery=10000, linewidth=1)
    plt.plot(np.arange(1,n_episodes+1), V_sarsa, label=f'SARSA learning last Value: {V_sarsa[-1]}', marker='x', markersize=4, markevery=10000, linewidth=1)
    plt.axhline(y=V_star[env.map[start]], color='k', linestyle='--', label=f'Optimal Reward Approximation {V_star[env.map[start]]}')
    plt.xlabel('Iterations')
    plt.ylabel('Value at Start State approximations')
    plt.title(r'Convergence of SARSA and Q learning with different $\epsilon$ and $\alpha$')
    plt.legend()
    plt.grid(True)
    plt.show()

    rept = 1000000

    results = Parallel(n_jobs=-1)( # this makes the code run in parallel 4x the speed
        delayed(run_simulation)(env, start, discount, Q_policy, S_policy) 
        for _ in tqdm(range(int(rept)), desc="Running Simulations")
    )

    mapping = np.array(results, dtype=bool) 
    iterations = np.arange(1, int(rept)+1)
    wins_Q = mapping[:,0].astype(int)
    wins_S = mapping[:,1].astype(int)

    cumavg_Q = np.cumsum(wins_Q) / iterations
    cumavg_S = np.cumsum(wins_S) / iterations

    plt.figure(figsize=(14, 7))

    # main cumulative averages (thicker for visibility)
    plt.plot(iterations, cumavg_Q, label=f'Cumulative average (Q-learning) V* = {V_qlearning[-1]:.4f}', color='C0', linewidth=1)
    plt.plot(iterations, cumavg_S, label=f'Cumulative average (SARSA) V* = {V_sarsa[-1]:.4f}', color='C1', linewidth=1)

    # mark final cumulative-average values and annotate
    final_Q = cumavg_Q[-1]
    final_S = cumavg_S[-1]
    plt.plot(rept, final_Q, 'o', color='C0', markersize=8, markeredgecolor='k')
    plt.plot(rept, final_S, 'o', color='C1', markersize=8, markeredgecolor='k')
    plt.annotate(f'{final_Q}', xy=(rept, final_Q), xytext=(rept*0.98, final_Q + 0.03),
                 fontsize=10, color='C0', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    plt.annotate(f'{final_S}', xy=(rept, final_S), xytext=(rept*0.98, final_S - 0.06),
                 fontsize=10, color='C1', ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # theoretical / expected value at start (from value iteration) and alternative
    theoretical = V_star[env.map[start]]
    theoretical2 = discount**29

    plt.axhline(y=theoretical, color='r', linestyle='--', linewidth=1.5,
                label=f'Probability from Value Iteration (V*) = {theoretical}')
    plt.axhline(y=theoretical2, color='b', linestyle=':', linewidth=1,
                label=f'Probability surviving 29 steps = {theoretical2}')

    plt.xlim(1, rept)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Iteration (trial index)')
    plt.ylabel('Win (0/1) and cumulative average')
    plt.title('Cumulative average — Q-learning vs SARSA')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
