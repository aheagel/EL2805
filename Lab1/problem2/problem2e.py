import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import gymnasium as gym
import pickle
from problem2 import running_average, scale_state_variables
from problem2b import SARSA2_learning

if __name__ == "__main__":
    # Import and initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    env.reset()
    
    N_episodes = 400
    p=2
    eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T # For small p this is doable
    
    #eta = eta[:, 1:]
    #print(eta) For plot later

    # Train SARSA with Fourier Basis
    W_learned, rewards = SARSA2_learning(env,
                                         lamda=0.9,
                                         discount=1,
                                         p=p,
                                         n_episodes=N_episodes,
                                         eps=0,
                                         l_rate=0.0005,
                                         eta=eta)

    # Plot Rewards plot 1
    plt.plot([i for i in range(1, N_episodes+1)], rewards, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes+1)], running_average(rewards, 50), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    pickle.dump({"W": W_learned.T, "N": eta.T}, open(sys.path[0] + '/weights.pkl', 'wb')) # used to save

    env.render()
    env.close()

