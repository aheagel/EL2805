# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.



# Load packages
import numpy as np
import gymnasium as gym
import torch
from tqdm import trange
import warnings, sys
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import sys, os
from DDPG_main import ActorNN, CriticNN

# Ensure the script directory is on sys.path so local modules (e.g. DQN_main) can be imported
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

# Make the script directory the current working directory so relative file reads (e.g. neural-network-1.pth) work
os.chdir(script_dir)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Load model
try:
    weights_a = torch.load('neural-network-2-actor.pth')
    model = ActorNN(8, 2).to("cpu")
    model.load_state_dict(weights_a)
    print('Network model: {}'.format(model))
except:
    print('File neural-network-2-actor.pth not found!')
    sys.exit(-1)

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLanderContinuous-v3', render_mode = "human")
env.reset()

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 125

# Reward
episode_reward_list = []  # Used to store episodes reward

# Simulate episodes
print('Checking solution...')
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    if i == N_EPISODES - 1:
        env.close()
        env = gym.make('LunarLanderContinuous-v3', render_mode='human')
        
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    while not (done or truncated):
        
        # Decide next action
        action = model(torch.tensor(state))
        
        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action.detach().numpy())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

# Close environment
env.close()

avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))

if avg_reward - confidence >= CONFIDENCE_PASS:
    print('Your policy passed the test!')
else:
    print("Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence".format(CONFIDENCE_PASS))
