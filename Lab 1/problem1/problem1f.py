from maze import Maze, value_iteration, animate_solution2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 3],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]])
# With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze, 3 = key

env = Maze(maze) # Create an environment maze

# Define the survival gamma and an accuracy threshold
discount = 29/30
accuracy_theta = 1e-12  # A small number for convergence
start  = ((0,0), (6,5))

V, policy = value_iteration(env, discount, accuracy_theta)

iteration = 1E4
mapping = np.empty(int(iteration), dtype=bool)
for i in range(int(iteration)):
    horizon = geom.rvs(p=1-discount)-1 # Subtract 1 to get number of steps as we start from state 0
    path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)
    mapping[i] = path[-1] == 'Win'

#animate_solution2(maze, path)

print(f"Theoretical via theory: {1-geom.cdf(15, 1 - discount)}")
print(f"Value iteration survival probability: {V[env.map[start]]}")
print(f"Estimated survival probability by winning over {int(iteration)} iterations: {np.sum(mapping)/iteration}")

plt.plot(np.cumsum(mapping)/np.arange(1, iteration+1))
plt.plot(V[env.map[start]]*np.ones(int(iteration)), 'r--', label='Theoretical survival probability')
plt.xlabel('Number of iterations')
plt.ylabel('Estimated survival probability')
plt.title('Estimated survival probability vs number of iterations')
plt.grid()
plt.show()