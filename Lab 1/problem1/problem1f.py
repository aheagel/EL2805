from maze import Maze, value_iteration, animate_solution2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

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

iteration = 1E5
mapping = np.empty(int(iteration), dtype=bool)
for i in range(int(iteration)):
    path = env.simulate(start, policy, method)[0]
    mapping[i] = path[-1] == 'Win'

#animate_solution2(maze, path)

print(f"Theoretical via theory: {1-geom.cdf(15, 1 - survival_gamma)}")
print(f"Value iteration survival probability: {V[env.map[start]]}")
print(f"Estimated survival probability by winning over {int(iteration)} iterations: {np.sum(mapping)/iteration}")

plt.plot(np.cumsum(mapping)/np.arange(1, iteration+1))
plt.plot(V[env.map[start]]*np.ones(int(iteration)), 'r--', label='Theoretical survival probability')
plt.xlabel('Number of iterations')
plt.ylabel('Estimated survival probability')
plt.title('Estimated survival probability vs number of iterations')
plt.grid()
plt.show()