from maze import Maze, value_iteration, animate_solution2
import numpy as np
import matplotlib.pyplot as plt

maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]])
# With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze

env = Maze(maze) # Create an environment maze

# Define the survival gamma and an accuracy threshold
survival_gamma = 29/30
accuracy_theta = 1e-12  # A small number for convergence

# Solve for the optimal survival policy
V, policy = value_iteration(env, survival_gamma, accuracy_theta)

# You can now simulate this new policy
method = 'ValIter'
start  = ((0,0), (6,5))

path = env.simulate(start, policy, method)[0]
animate_solution2(maze, path)