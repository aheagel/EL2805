# Anh Do: 20020416-2317
# Saga Tran: 19991105-2182

from maze import Maze, dynamic_programming
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

for i in tqdm(range(1,31)):
    # Solve the MDP problem with dynamic programming and pathing 
    V, policy = dynamic_programming(env, i)  
    V2, policy2 = dynamic_programming(env2, i)

    lst.append(V[env.map[start], 0])
    lst2.append(V2[env2.map[start], 0])

def plot_values(series, title):
    """Plot multiple value sequences vs iteration number."""
    iterations = list(range(1, len(series[0][0]) + 1))
    plt.figure(figsize=(8, 5))
    colors = ['tab:blue', 'tab:orange']
    markers = ['X', 'o']
    for (values, label), color, marker in zip(series, colors, markers):
        plt.scatter(
            iterations,
            values,
            marker=marker,
            color=color,
            label=label,
            facecolors='none' if marker == 'o' else color,
            edgecolors=color,
            linewidths=0.8
        )
    plt.xlabel('Horizons')
    plt.ylabel('Value at start state / Probability of winning')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# After computing `lst` for iterations 1..30, plot the results and save the figure.
if __name__ == '__main__':
    plot_values(
        [(lst, "Still Minotaur"), (lst2, "Moving Minotaur")],
        "Value at start vs number of iterations (Dynamic Programming)"
    )
