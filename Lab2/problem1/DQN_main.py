import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_problem import running_average

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

class DQNAgent:
    def __init__(self, env, lr=1e-3, buffer_capacity=10000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05):
        self.env = env
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay Buffer
        self.buffer_capacity = buffer_capacity
        self.memory = self.init_memory()
        self.env.close() 
        self.position = 0
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

    def init_memory(self):
        memory = [None] * self.buffer_capacity
        state = self.env.reset()[0]
        done = False
        truncated = False

        for i in range(self.buffer_capacity):
            action = np.random.randint(0, self.output_dim)
            if  not done or not truncated:
                next_state, reward, done, truncated, _ = self.env.step(action)
                state = next_state
            else :
                state = self.env.reset()[0]
                next_state, reward, done, truncated, _ = self.env.step(action)

            memory[i] = (state, action, reward, next_state, done or truncated)
        
        return memory

    def push_memory(self, state, action, reward, next_state, done):
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.buffer_capacity) # We overwrite the oldest memory when buffer is full

    def select_action(self, state): # Epsilon-greedy action selection
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                # Find indices of all max values
                max_q_value = q_values.max().item()
                best_actions = (q_values == max_q_value).nonzero(as_tuple=True)[1]
                # Randomly select one of the best actions
                return best_actions[np.random.randint(len(best_actions))].item()
        else:
            return np.random.randint(self.output_dim)

    def update_epsilon(self, k, max_k):
        self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_end/self.epsilon_start) ** ((k-1) / (max_k-1)))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            raise ValueError("Please initialize memory before optimizing the model.")
        
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).unsqueeze(1) # Optimize for bool to save memory

        q_values = self.policy_net(state).gather(1, action)
        next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1) # max(Q(s', a'))
        expected_q_values = reward + (self.gamma * next_q_values * (~done))

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == '__main__':
    # Hyperparameters
    N_EPISODES = 500
    Z = N_EPISODES * 0.9
    LR = 1e-3
    BUFFER_CAPACITY = 10000
    BATCH_SIZE = 64
    TARGET_UPDATE = int (BUFFER_CAPACITY/BATCH_SIZE)
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.05

    agent = DQNAgent(gym.make('LunarLander-v3'), 
                     lr=LR, 
                     buffer_capacity=BUFFER_CAPACITY, 
                     batch_size=BATCH_SIZE, 
                     gamma=GAMMA, 
                     epsilon_start=EPSILON_START, 
                     epsilon_end=EPSILON_END)

    episode_reward_list = []
    episode_number_of_steps = []

    EPISODES = trange(1,N_EPISODES+1, desc='Episode: ', leave=True)
    env = gym.make('LunarLander-v3')
    steps = 0
    for i in EPISODES:
        if i == N_EPISODES:
            env.close()
            env = gym.make('LunarLander-v3', render_mode='human') # PLOTTING
        
        state = env.reset()[0]
        total_episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.push_memory(state, action, reward, next_state, done or truncated)
            agent.optimize_model()
            
            state = next_state
            total_episode_reward += reward
            steps += 1
            
        agent.update_epsilon(k=i, max_k=Z)
        
        if steps % TARGET_UPDATE == 0:
            agent.update_target_network()
            
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(steps)
        
        EPISODES.set_description(
            "Episode {} - Reward: {:.1f} - Avg. Reward: {:.1f} - Epsilon: {:.2f}".format(
            i, total_episode_reward, 
            running_average(episode_reward_list, 50)[-1],
            agent.epsilon))

    # Save the model
    torch.save(agent.policy_net, 'Lab2/problem1/neural-network-1.pth')
    print("Model saved to Lab2/problem1/neural-network-1.pth")

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot(episode_reward_list, label='Episode reward')
    ax[0].plot(running_average(episode_reward_list, 50), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[1].plot(episode_number_of_steps, label='Steps per episode')
    ax[1].plot(running_average(episode_number_of_steps, 50), label='Avg. steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Steps')
    ax[1].set_title('Steps vs Episodes')
    ax[1].legend()
    plt.show()