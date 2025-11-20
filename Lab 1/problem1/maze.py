# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # use when windows 
import time
from IPython import display
from scipy.stats import geom

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
GOLD         = '#FFD700'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values 
    STEP_REWARD         = -1          #TODO
    STAY_REWARD         = -1          #TODO # set all the movement reward to zero for a short path optimal path
    GOAL_REWARD         = 100          #TODO
    IMPOSSIBLE_REWARD   = -1          #TODO
    MINOTAUR_REWARD     = -100          #TODO

    def __init__(self, maze, still_minotaur=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.still_minotaur           = still_minotaur
        self.actions                  = self.init_actions()
        self.states, self.map         = self.init_states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.init_transitions()
        self.rewards                  = self.init_rewards()


    def init_actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1, 0)
        return actions


    def init_states(self):
        
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = ((i,j), (k,l))
                            map[((i,j), (k,l))] = s
                            s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        s += 1

        states[s] = 'Done'
        map['Done'] = s
        
        return states, map


    def minotaur_states_probs(self, states):
        """ Given a list of possible next states, probability distribution
            :input list states         : List of possible next states.
            :return tuple next_state   : Chosen next state.
        """
        probs = {}
        for state in states:
            state_idx = self.map[state]
            probs[state_idx] = probs.get(state_idx, 0) + 1.0 / len(states)

        return list(probs.keys()), list(probs.values())


    def move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """
        
        if self.states[state] in ['Eaten', 'Win', 'Done']: # In these states, the game is over
            return [self.states[self.map["Done"]]] # And we move to done state
        
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
            
            # Is the player getting out of the limits of the maze or hitting a wall?
            
            impossible_action_player = ((row_player, col_player), (0, 0)) not in self.map #TODO
            
        
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]] # Possible moves for the Minotaur
            if self.still_minotaur:
                actions_minotaur.append([0,0])

            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                             (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
          

            # Based on the impossiblity check return the next possible states.
            if impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if (self.states[state][0][0], self.states[state][0][1]) == (rows_minotaur[i], cols_minotaur[i]): #TODO
                        states.append('Eaten')
                    
                    elif self.maze[self.states[state][0][0], self.states[state][0][1]] == 2: #TODO
                        states.append('Win')
                
                    else:     # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i])))
                
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if (row_player, col_player) == (rows_minotaur[i], cols_minotaur[i]): #TODO
                        states.append('Eaten')
                    
                    elif self.maze[row_player, col_player] == 2: #TODO
                        states.append('Win')
                    
                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i])))

                return states
        
        
    def init_transitions(self): # Only works for the minotaur moving randomly and determinisitic finite horizon
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Vectorized computation of transition probabilities
        # Pre-compute all next states for all (state, action) pairs
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = self.move(s, a)
                idx, values = self.minotaur_states_probs(next_states)
                transition_probabilities[s, idx, a] = values
                
        return transition_probabilities


    def init_rewards(self):
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD
                
                elif self.states[s] == "Done": # The game is over
                    rewards[s, a] = 0
                else:                
                    next_states = self.move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the players next position one
                    
                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    
                    elif a != self.STAY: # Move 
                        rewards[s, a] = self.STEP_REWARD
                    
                    else: # Stay
                        rewards[s, a] = self.STAY_REWARD

        return rewards


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


    def simulate(self, start, policy, horizon):
        """ Simulates a path in the maze given a policy obtained
            :input Maze self           : The maze selfironment in which we seek to
            :input tuple start       : The starting position of the player and the minotaur
            :input int horizon       : The time T up to which we simulate the path.
            :return list objects          : The path followed in the maze and mm.
            """
        
        path = [] # Initialize the path
        t = 0 # Initialize current time
        s = self.map[start] # Initialize current state 
        path.append(start) # Add the starting position in the maze to the path
        
        while self.states[s] not in ["Done"] and t < horizon:
            a = policy[s, t] # Move to next state given the policy and the current state
            next_states = self.move(s, a)

            idx, probs = self.minotaur_states_probs(next_states) #TODO Choose one of the possible next states (deterministic policy)
            next_s = np.random.choice(idx, p=probs)  # Sample next state

            path.append(self.states[next_s]) # Add the next state to the path
            t += 1 # Update time and state for next iteration
            s = next_s
        
        return path
    
def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    V = np.zeros((env.n_states, horizon))
    policy = np.zeros((env.n_states, horizon), dtype=int)
    
    # Boundary conditions: terminal states give absorbing value
    V[:, horizon-1] = np.max(env.rewards, axis=1)    # Bellman function at boundary 

    for t in range(horizon - 2, -1, -1):
        # Compute expected future value functions and Q-function and the terminal states
        future_values = np.einsum('ijk,j->ik', env.transition_probabilities, V[:, t + 1])
        Q_function = env.rewards + future_values 

        # Compute the optimal value and policy
        V[:, t] = np.max(Q_function, axis=1)
        max_mask = (Q_function == Q_function.max(axis=1, keepdims=True))
        policy[:, t] = (max_mask * np.random.rand(*Q_function.shape)).argmax(axis=1) # Break ties randomly

    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S
    """
    V = np.zeros(env.n_states) # V0
    delta = 100
    while delta > epsilon*(1 - gamma) / gamma:
        expected_future_rewards = env.rewards + gamma * np.einsum('ijk,j->ik', env.transition_probabilities, V)
        V_next = np.max(expected_future_rewards, axis=1)
        delta = np.linalg.norm(V_next - V, np.inf) # Max norm
        V = V_next
    
    policy = np.argmax(env.rewards + gamma * np.einsum('ijk,j->ik', env.transition_probabilities, V), axis=1)
    #policy = np.argmax(expected_future_rewards, axis=1) # Would be the same as above when converged
    return V, policy



def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE, 3: GOLD}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(0, len(path)):
        if path[i-1] != 'Eaten' and path[i-1] != 'Win':
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
        if path[i] != 'Eaten' and path[i] != 'Win':
            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player
            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
        display.display(fig)
        
        time.sleep(0.5)
        display.clear_output(wait = True)

def animate_solution2(maze, path):
    """ Animates the solution path in the maze """
    
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE, 3: GOLD}
    
    rows, cols = maze.shape
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText=None, 
        cellColours=colored_maze, 
        cellLoc='center', 
        loc=(0, 0), 
        edges='closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Find terminal state index
    terminal_idx = len(path)
    for i, state in enumerate(path):
        if state == 'Eaten' or state == 'Win':
            terminal_idx = i + 1
            break

    def update(frame):
        """ Update function for animation """
        i = frame

        # Reset previous positions to original colors
        if i > 0 and path[i-1] != 'Eaten' and path[i-1] != 'Win':
            player_prev = path[i-1][0]
            minotaur_prev = path[i-1][1]
            grid.get_celld()[(player_prev[0], player_prev[1])].set_facecolor(col_map[maze[player_prev]])
            grid.get_celld()[(minotaur_prev[0], minotaur_prev[1])].set_facecolor(col_map[maze[minotaur_prev]])
        
        # Color current positions
        if path[i] != 'Eaten' and path[i] != 'Win':
            player_pos = path[i][0]
            minotaur_pos = path[i][1]
            grid.get_celld()[(player_pos[0], player_pos[1])].set_facecolor(col_map[-2])  # Player
            grid.get_celld()[(minotaur_pos[0], minotaur_pos[1])].set_facecolor(col_map[-1])  # Minotaur
            ax.set_title(f'Policy simulation - Step {i}')
        else:
            ax.set_title(f'Policy simulation - Step {i} - {path[i]}!')

        return grid,

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=terminal_idx, interval=300, repeat=False, blit=False)
    
    plt.show()
    return anim


if __name__ == "__main__":
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
    path = env.simulate(start, policy, policy.shape[1])

    animate_solution2(maze, path)