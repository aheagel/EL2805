import gymnasium as gym
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from problem2b import FourierBasis
from problem2 import scale_state_variables

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

try:
    f = open(sys.path[0] + '/weights.pkl', 'rb')
    data = pickle.load(f)
    if 'W' not in data or 'N' not in data:
        print('Matrix W (or N) is missing in the dictionary.')
        exit(-1)
    w = data['W']
    eta = data['N']
except:
    print('File weights.pkl not found!')
    exit(-1)


def Value(state, W, action=None, _eta=eta):
    if action is None: # This is the Value function
        return W.T @ FourierBasis(state, _eta)
    else: # This is the Q function for action 
        return W[:, action].T @ FourierBasis(state, _eta) # This is the Q-value for a specific action

# Plot 2 3D

x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 100)
y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 100)
X, Y = np.meshgrid(x, y)
grid_pairs = np.stack([X, Y], axis=-1)


z = Value()


plt.plot([i for i in range(1, N_episodes+1)], rewards, label='Episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
