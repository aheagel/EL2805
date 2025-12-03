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
from problem_2b import make_exponential_schedule, make_polynomial_schedule
from problem_2e3 import SARSA3_learning

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
    curiosity_schedule =make_exponential_schedule(params['curiosity_start'], 0, params['curiosity_decay'])

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

    #pickle.dump({"W": W_learned.T, "N": eta.T}, open(sys.path[0] + '/weights.pkl', 'wb')) # used to save
    #exec(open(os.path.join(sys.path[0], 'check_solution.py')).read())
