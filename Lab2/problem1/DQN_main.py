import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_problem import running_average
from DQN_agent import DQNAgent

import warnings, sys
warnings.simplefilter(action='ignore', category=FutureWarning)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    env = gym.make('LunarLander-v3')

    N_episodes = 400
    discount_factor = 0.99
    learning_rate = 1e-4
    buffer_capacity = 10000
    batch_size = 64
    target_update_freq = int(buffer_capacity/batch_size)
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = N_episodes * 0.90
    n_ep_running_average = 50

    episode_reward_list = []
    episode_number_of_steps = []

    agent = DQNAgent(env, DQN,
                     learning_rate=learning_rate, 
                     buffer_capacity=buffer_capacity, 
                     batch_size=batch_size, 
                     discount_factor=discount_factor, 
                     epsilon_start=epsilon_start, 
                     epsilon_end=epsilon_end)

    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        if i == N_episodes - 1:
            env.close()
            env = gym.make('LunarLander-v3', render_mode='human')
        
        state = env.reset()[0]
        total_episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            action = agent.forward(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.push_memory(state, action, reward, next_state, done)
            agent.backward()
            
            state = next_state
            total_episode_reward += reward
            steps += 1

            if (i * N_episodes + steps) % target_update_freq == 0: 
                 agent.update_target_network()
            
        agent.update_epsilon(k=i, max_k=epsilon_decay_steps)
            
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(steps)
        
        # Updates the tqdm update bar with fresh information
        EPISODES.set_description(
            "Episode: {} | Steps: {} | Reward: {:.1f} | Avg. Reward: {:.1f} | Epsilon: {:.2f}".format(
            i, 
            steps,
            total_episode_reward, 
            running_average(episode_reward_list, n_ep_running_average)[-1],
            agent.epsilon)
        )

    #torch.save(agent.policy_net.state_dict(), 'Lab2/problem1/neural-network-1.pth')
    print("Model saved to neural-network-1.pth")

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