# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray) -> np.ndarray: 
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
    
class DDPGAgent(Agent):
    ''' DDPG Agent class, child of the class Agent'''
    def __init__(self, env, action_NN: nn.Module, critic_NN: nn.Module, actor_learning_rate=1e-3, critic_learning_rate=1e-3, buffer_capacity=10000, batch_size=64, discount_factor=0.99, tau=1e-3, noise_std=0.2, noise_mean=0.0):
        n_actions = env.action_space.shape[0]
        super(DDPGAgent, self).__init__(n_actions)
        
        self.env = env
        self.input_dim = self.env.observation_space.shape[0]
        
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        self.action_net = action_NN(self.input_dim, self.n_actions).to(self.device)
        self.critic_net = critic_NN(self.input_dim, self.n_actions).to(self.device)

        self.target_action_net = action_NN(self.input_dim, self.n_actions).to(self.device)
        self.target_critic_net = critic_NN(self.input_dim, self.n_actions).to(self.device)
        self.target_action_net.load_state_dict(self.action_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_action_net.eval()
        self.target_critic_net.eval()
        
        self.action_optimizer = optim.Adam(self.action_net.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_learning_rate)
        
        self.buffer_capacity = buffer_capacity
        self.memory = self.init_memory()
        self.position = 0
        
        self.batch_size = batch_size
        self.gamma = discount_factor
        self.tau = tau
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noise = np.zeros(self.n_actions)


    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute an action given the current state

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.action_net(state_tensor).cpu().numpy().squeeze() + self.noise
            return np.clip(action, -1.0, 1.0)

    
    def backward(self):
        ''' Performs a backward pass on the network '''
        if len(self.memory) < self.batch_size:
            return

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        
        self.state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        action = torch.tensor(np.array(action), dtype=torch.float32, device=self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(np.array(done), dtype=torch.bool, device=self.device).unsqueeze(1)

        next_action = self.target_action_net(next_state)
        next_q_values = self.target_critic_net(next_state, next_action)
        expected_q_values = reward + (self.gamma * next_q_values * (~done))

        # Update Critic Network always
        loss = F.mse_loss(self.critic_net(self.state, action), expected_q_values)
        self.critic_optimizer.zero_grad()
        loss.backward() # this is the mse backward not agents backward
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1) # Clipping
        self.critic_optimizer.step()

    def update_noise(self):
        self.noise = -self.noise_mean*self.noise + np.random.normal(0, self.noise_std, size=self.n_actions)

    def reset_noise(self):
        self.noise = np.zeros(self.n_actions)

    def init_memory(self):
        randomagent = RandomAgent(self.n_actions)
        memory = [None] * self.buffer_capacity
        state = self.env.reset()[0]
        done = False
        truncated = False

        for i in range(self.buffer_capacity):
            action = randomagent.forward(state)
            if not (done or truncated):
                next_state, reward, done, truncated, _ = self.env.step(action)
                state = next_state
            else :
                state = self.env.reset()[0]
                next_state, reward, done, truncated, _ = self.env.step(action)

            memory[i] = (state, action, reward, next_state, done)
        
        self.env.close()
        return memory
    
    def push_memory(self, state, action, reward, next_state, done):
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.buffer_capacity) 

    def update_action_network(self):
        action_loss = -self.critic_net(self.state, self.action_net(self.state)).mean()
        self.action_optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(self.action_net.parameters(), 1) # Clipping
        self.action_optimizer.step()

    def update_target_network(self):
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_action_net.parameters(), self.action_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


        
