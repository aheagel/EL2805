from maze import Maze, dynamic_programming
import numpy as np
import matplotlib.pyplot as plt

lst=[]
lst2=[]
method = 'DynProg'
start  = ((0,0), (6,5))
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]])
# With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze

env = Maze(maze, still_minotaur=True) # Create an environment maze
env2 = Maze(maze, still_minotaur=False)

for i in range(1,31):
    # Solve the MDP problem with dynamic programming
    V, policy = dynamic_programming(env, i)  
    V2, policy2 = dynamic_programming(env2, i)

    # Simulate the game path starting from position A
    path = env.simulate(start, policy, method)[0]
    path2 = env2.simulate(start, policy2, method)[0]

    lst.append(V[env.map[start], 0])
    lst2.append(V2[env2.map[start], 0])


def plot_values(values, filename=None):
    """Plot the values list vs iteration number and optionally save to filename.

    Args:
        values (list of float): value at the start state for each iteration count
        filename (str|None): if provided, save the figure to this path (PNG)
    """
    iterations = list(range(1, len(values) + 1))
    plt.figure(figsize=(8, 5))
    plt.scatter(iterations, values, marker='o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Value at start state / Probability of winning')
    plt.title('Value at start vs number of iterations (Dynamic Programming)')
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {filename}')
    plt.show()


# After computing `lst` for iterations 1..30, plot the results and save the figure.
if __name__ == '__main__':
    plot_values(lst)
    plot_values(lst2)

