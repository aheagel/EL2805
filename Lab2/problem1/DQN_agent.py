# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Agent:
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray ) -> int:
        ''' Performs a forward computation '''
        pass 

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action

class DQNAgent(Agent):
    def __init__(self, env, DQN: nn.Module, learning_rate=1e-3, buffer_capacity=10000, batch_size=64, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.05):
        n_actions = env.action_space.n
        super(DQNAgent, self).__init__(n_actions)
        
        self.env = env
        self.input_dim = self.env.observation_space.shape[0]
        
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQN(self.input_dim, self.n_actions).to(self.device)
        self.target_net = DQN(self.input_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.buffer_capacity = buffer_capacity
        self.memory = self.init_memory()
        self.position = 0
        
        self.batch_size = batch_size
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

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

    def forward(self, state: np.ndarray) -> int:
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)

                #Tie-breaking for multiple actions with same max Q-value
                max_q_value = q_values.max().item()
                best_actions = (q_values == max_q_value).nonzero(as_tuple=True)[1]
                action = int(best_actions[np.random.randint(len(best_actions))].item())   
        else:
            action = np.random.randint(self.n_actions)
        
        self.last_action = action
        return action

    def backward(self):
        if len(self.memory) < self.batch_size:
            return

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state).gather(1, action)
        next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1)
        expected_q_values = reward + (self.gamma * next_q_values * (~done))

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward() # this is the mse backward not agents backward
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Clipping
        self.optimizer.step()

    def update_epsilon(self, k, max_k):
        self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_end/self.epsilon_start) ** ((k-1) / (max_k-1)))

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())