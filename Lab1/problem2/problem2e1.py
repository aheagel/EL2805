import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import gymnasium as gym
import pickle
from joblib import Parallel, delayed
from problem2 import running_average, scale_state_variables
from problem2b import SARSA2_learning, make_exponential_schedule, make_polynomial_schedule

if __name__ == "__main__":
    # Import and initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    env.reset()
    
    N_episodes = 200
    p = 2
    eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T
    params = {
        'lambda': 0.8540592523120906,
        'momentum': 0.9983674411507302,
        'lr_initial': 6.644641407969861e-05,
        'lr_scale': 21.477703280373976,
        'lr_power': 0.6740471797607481,
        'eps_start': 0.0021543255282344886,
        'eps_decay': 0.9337869414588281
    }
    epsilon_schedule = make_exponential_schedule(params['eps_start'], 0, params['eps_decay'])
    learning_rate_schedule = make_polynomial_schedule(params['lr_initial'], 0, params['lr_scale'], params['lr_power'])

    alphas = np.logspace(-7, -4, 20)
    lambdas = np.linspace(0, 1, 20)

    # Parallel sweep for learning rate with tqdm
    def run_alpha(a):
        W_learned, rewards = SARSA2_learning(
            env,
            lamda=params['lambda'],
            discount=1,
            p=p,
            n_episodes=N_episodes,
            eps=lambda k: epsilon_schedule(k),
            l_rate=lambda k: a.astype(np.float64),
            eta=eta,
            momentum=params['momentum'],
            plot=False,
            debug=True)
        return rewards

    rewards_alpha = []
    with tqdm(total=len(alphas), desc="Sweeping alpha (parallel)") as pbar:
        def update(*args):
            pbar.update()
        rewards_alpha = Parallel(n_jobs=-1)(
            delayed(run_alpha)(a) for a in alphas
        )
        # tqdm will update automatically for each finished job

    rewards_alpha = np.array(rewards_alpha)
    data_a = rewards_alpha[:, -50:]
    plt.plot(alphas, np.mean(data_a, axis=1), marker='o')
    ci = 1.96 * np.std(data_a, axis=1) / np.sqrt(data_a.shape[1])
    plt.fill_between(alphas, np.mean(data_a, axis=1) - ci, np.mean(data_a, axis=1) + ci, color='b', alpha=0.2)
    plt.xscale('log')
    plt.xlabel('Learning Rate (alpha)')
    plt.ylabel('Average Reward over last 50 Episodes')
    plt.title('Average Reward vs Learning Rate')
    plt.grid(alpha=0.3)
    plt.show()

    # Parallel sweep for lambda with tqdm
    def run_lambda(l):
        W_learned, rewards = SARSA2_learning(
            env,
            lamda=l.astype(np.float64),
            discount=1,
            p=p,
            n_episodes=N_episodes,
            eps=lambda k: epsilon_schedule(k),
            l_rate=lambda k: 1e-5, # Make the plot more stable
            eta=eta,
            momentum=params['momentum'],
            plot=False,
            debug=True)
        return rewards

    rewards_lambda = []
    with tqdm(total=len(lambdas), desc="Sweeping lambda (parallel)") as pbar:
        rewards_lambda = Parallel(n_jobs=-1)(
            delayed(run_lambda)(l) for l in lambdas
        )

    rewards_lambda = np.array(rewards_lambda)
    data_l = rewards_lambda[:, -50:]
    plt.plot(lambdas, np.mean(data_l, axis=1), marker='o')
    ci = 1.96 * np.std(data_l, axis=1) / np.sqrt(data_l.shape[1])
    plt.fill_between(lambdas, np.mean(data_l, axis=1) - ci, np.mean(data_l, axis=1) + ci, color='b', alpha=0.2)
    plt.xlabel('Lambda (Trace Decay Parameter)')
    plt.ylabel('Average Reward over last 50 Episodes')
    plt.title('Average Reward vs Lambda')
    plt.grid(alpha=0.3)
    plt.show()