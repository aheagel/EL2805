# Load packages
import numpy as np
import gymnasium as gym
import torch
from tqdm import trange
import warnings, sys
from DQN_main import DQN
import os
from pathlib import Path
import sys, os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Ensure the script directory is on sys.path so local modules (e.g. DQN_main) can be imported
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

# Make the script directory the current working directory so relative file reads (e.g. neural-network-1.pth) work
os.chdir(script_dir)

warnings.simplefilter(action='ignore', category=FutureWarning)
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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
    weights = torch.load('neural-network-1.pth')
    model = DQN(8, 4).to(device)
    model.load_state_dict(weights)
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    sys.exit(-1)


y = np.linspace(0, 1.5, 100)
omega = np.linspace(-np.pi, np.pi, 100)

Y, Omega = np.meshgrid(y, omega)
Z = np.zeros_like(Y)

for i in range(len(y)):
    for j in range(len(omega)):
        state = (0, Y[j, i], 0, 0, Omega[j, i], 0, 0, 0)
        q_values = model(torch.tensor(state, dtype=torch.float32).to(device))
        Z[j, i] = torch.max(q_values).detach().cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Y, Omega, Z)
ax.set_xlabel('y')
ax.set_ylabel('omega')
ax.set_zlabel('max Q-values')
plt.show()

Z_argmax = np.zeros_like(Y, dtype=int)

for i in range(len(y)):
    for j in range(len(omega)):
        state = (0, Y[j, i], 0, 0, Omega[j, i], 0, 0, 0)
        q_values = model(torch.tensor(state, dtype=torch.float32).to(device))
        Z_argmax[j, i] = torch.argmax(q_values).detach().cpu().numpy()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Y, Omega, Z_argmax)
ax2.set_xlabel('y')
ax2.set_ylabel('omega')
ax2.set_zlabel('argmax action')
plt.show()