from maze import Maze, value_iteration, animate_solution2
import numpy as np

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
accuracy_theta = 1e-4  # A small number for convergence

# Solve for the optimal survival policy
V_survival, policy_survival = value_iteration(env, survival_gamma, accuracy_theta)

# You can now simulate this new policy
method = 'ValIter'
start  = ((0,0), (6,5))

for i in range(int(1E4)):
    path = env.simulate(start, policy_survival, method)[0]
    path[-1] =
#animate_solution2(maze, path)

print(V_survival[env.map[start]])

