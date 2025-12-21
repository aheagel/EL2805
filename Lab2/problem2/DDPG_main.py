import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_problem import running_average
from DDPG_agent import DDPGAgent

import warnings, sys
warnings.simplefilter(action='ignore', category=FutureWarning)

class ActorNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class CriticNN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400+action_dim, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v3')

    N_episodes = 500
    discount_factor = 0.99
    action_learning_rate = 5e-5
    critic_learning_rate = 5e-4
    tau = 1e-3
    buffer_capacity = 50000
    batch_size = 64
    target_update_freq = 2
    n_ep_running_average = 50
    sigma = 0.2
    mu = 0.15

    episode_reward_list = []
    episode_number_of_steps = []

    agent = DDPGAgent(env, ActorNN, CriticNN,
                     actor_learning_rate=action_learning_rate,
                     critic_learning_rate=critic_learning_rate,
                     buffer_capacity=buffer_capacity, 
                     batch_size=batch_size, 
                     discount_factor=discount_factor, 
                     tau=tau,
                     noise_std=sigma,
                     noise_mean=mu,)

    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    total_steps = 0
    current_best = -float('inf')
    best_agent = None

    for i in EPISODES:
        agent.reset_noise()
        if i == N_episodes - 1:
            env.close()
            env = gym.make('LunarLanderContinuous-v3', render_mode='human')
        
        state = env.reset()[0]
        total_episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            action = agent.forward(state)
            agent.update_noise()
            
            next_state, reward, done, truncated, _ = env.step(action)
            agent.push_memory(state, action, reward, next_state, done)
            agent.backward()
            
            state = next_state
            total_episode_reward += reward
            steps += 1
            total_steps += 1

            if total_steps % target_update_freq == 0: 
                 agent.update_action_network()
                 agent.update_target_network()
            
            
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(steps)
        
        current_average = running_average(episode_reward_list, n_ep_running_average)[-1] 
        # Updates the tqdm update bar with fresh information
        EPISODES.set_description(
            "Episode: {} | Steps: {} | Reward: {:.1f} | Avg. Reward: {:.1f}".format(
            i, 
            steps,
            total_episode_reward, 
            current_average)
        )

        if current_average > current_best:
            current_best = current_average
            best_agent = agent

    torch.save(best_agent.action_net.state_dict(), 'Lab2/problem2/neural-network-2-actor.pth')
    print("Model saved to neural-network-2-actor.pth")
    torch.save(best_agent.critic_net.state_dict(), 'Lab2/problem2/neural-network-2-critic.pth')
    print("Model saved to neural-network-2-critic.pth")

    env.close()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()