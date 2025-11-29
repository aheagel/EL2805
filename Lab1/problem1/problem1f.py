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
env = Maze(maze, still_minotaur=False) # Create an environment maze

# Define the survival gamma and an accuracy threshold
discount = 29/30
accuracy_theta = 1e-5  # A small number for convergence
start  = ((0,0), (6,5))

V, policy = value_iteration(env, discount, accuracy_theta)

iteration = 1E4
mapping = np.empty(int(iteration), dtype=bool)
for i in range(int(iteration)):
    horizon = np.random.geometric(p=1-discount)
    path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)
    mapping[i] = path[-2] == 'Win'

#animate_solution2(maze, path)

print(f"Theoretical via theory for death case only so no eaten: {discount**15}") # same as discount^15
print(f"Value iteration survival probability: {V[env.map[start]]}")
print(f"Estimated survival probability by winning over {int(iteration)} iterations: {np.sum(mapping)/iteration}")

plt.plot(np.cumsum(mapping)/np.arange(1, iteration+1), label=f'Estimated survival probability from simulation, Last Value {np.cumsum(mapping)[-1]/iteration:.4f}')
plt.plot(V[env.map[start]]*np.ones(int(iteration)), 'r--', label=f'Survival probability implied from value function {V[env.map[start]]:.4f}')
plt.plot(discount**15 * np.ones(int(iteration)), 'g--', label=f'Theoretical survival probability {discount**15:.4f}')

plt.xlabel('Number of iterations')
plt.ylabel('Estimated survival probability')
plt.title('Estimated survival probability vs number of iterations')
plt.legend()
plt.grid()
plt.show()