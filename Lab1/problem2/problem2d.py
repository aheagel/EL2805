import gymnasium as gym
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from problem2b import FourierBasis, Value
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
    w = data['W'].T
    eta = data['N'].T
except:
    print('File weights.pkl not found!')
    exit(-1)


# INIT FOR PLOTTING
x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 100)
y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 100)
x_scaled = scale_state_variables(np.array([x, np.zeros_like(x)]).T)
y_scaled = scale_state_variables(np.array([np.zeros_like(y), y]).T)

X, Y = np.meshgrid(x, y)
X_s, Y_s = np.meshgrid(x_scaled[:,0], y_scaled[:,1])

grid_pairs = np.stack([X_s, Y_s], axis=-1).reshape(-1, 2)
state = grid_pairs.T.astype(np.float64)
w = w.astype(np.float64)
eta = eta.astype(np.float64)

# Plot 2 3D Value function surface plot
z = Value(state, w, eta)

Z = np.max(z, axis=0).reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('Value')
ax.set_title('Surface Plot of the optimal policy as max Value function')

plt.show()


# Plot 3  3D Optimal Policy action plot

z = Value(state, w, eta)

Z = np.argmax(z, axis=0).reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('Policy (Action)')
ax.set_title('Surface Plot of the optimal policy as max Value function')

plt.show()

# Plot 4 
# change 2b and run the plots TLDR It does we need 0 0 as it will act as our bias term in fourier basis basically cos(0)=1 so that weight will be our bias

