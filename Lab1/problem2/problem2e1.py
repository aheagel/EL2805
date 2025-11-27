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
    
    N_episodes = 400 # So we converge
    p=2
    eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T # For small p this is doable
    
    alphas = np.logspace(-4, -2, 20)
    lambdas = np.linspace(0, 1, 20)

    # Plot Rewards plot 1 - Average Reward vs Learning Rate
    rewards_alpha = np.empty((len(alphas), N_episodes))
    for i, a in enumerate(alphas): # Warmup for 350 then at the end we only take 50
        W_learned, rewards_alpha[i,:] = SARSA2_learning(env,
                                            lamda=0.9,
                                            discount=1,
                                            p=p,
                                            n_episodes=N_episodes,
                                            eps=lambda k: 0,
                                            l_rate=a,
                                            eta=eta)

    data_a = rewards_alpha[:,-50:]
    plt.plot(alphas, np.mean(data_a, axis=1), marker='o')
    ci = 1.96 * np.std(data_a, axis=1)/np.sqrt(data_a.shape[1])
    plt.fill_between(alphas, np.mean(data_a, axis=1)-ci, np.mean(data_a, axis=1)+ci, color='b', alpha=0.2)
    plt.xscale('log')
    plt.xlabel('Learning Rate (alpha)')
    plt.ylabel('Average Reward over last 50 Episodes')
    plt.title('Average Reward vs Learning Rate')
    plt.grid(alpha=0.3)
    plt.show()

    # Plot Rewards plot 2 - Average Reward vs Lambda (Trace Decay Parameter)
    rewards_lambda = np.empty((len(lambdas), N_episodes))
    for i, l in enumerate(lambdas): # Warmup for 350 then at the end we only take 50
        W_learned, rewards_lambda[i,:] = SARSA2_learning(env,
                                            lamda=l,
                                            discount=1,
                                            p=p,
                                            n_episodes=N_episodes,
                                            eps=lambda k: 0,
                                            l_rate=alphas[np.argmax(np.mean(data_a, axis=1))],
                                            eta=eta)
    
    data_l = rewards_lambda[:,-50:]
    plt.plot(lambdas, np.mean(data_l, axis=1), marker='o')
    ci = 1.96 * np.std(data_l, axis=1)/np.sqrt(data_l.shape[1])
    plt.fill_between(lambdas, np.mean(data_l, axis=1)-ci, np.mean(data_l, axis=1)+ci, color='b', alpha=0.2)
    plt.xlabel('Lambda (Trace Decay Parameter)')
    plt.ylabel('Average Reward over last 50 Episodes')
    plt.title('Average Reward vs Lambda')
    plt.grid(alpha=0.3)
    plt.show()