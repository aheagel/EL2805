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

# Define the discount and an accuracy threshold
discount = 29/30
accuracy_theta = 1e-12  # A small number for convergence
start  = ((0,0), (6,5))

horizon = 100 #np.random.geometric(p=1-discount)
V, policy = value_iteration(env, discount, accuracy_theta)
path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)
animate_solution2(maze, path)