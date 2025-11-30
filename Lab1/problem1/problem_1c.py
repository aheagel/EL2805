# Anh Do: 20020416-2317
# Saga Tran: 19991105-2182

from maze import Maze, animate_solution2, dynamic_programming
import numpy as np

# Description of the maze as a numpy array
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
horizon = 20      # TODO: Finite horizon this is the Time we have to reach the exit

# Simulate the shortest path starting from position A
start  = ((0,0), (6,5))
V, policy = dynamic_programming(env, horizon)
path = env.simulate(start, policy, horizon)

animate_solution2(maze, path)
