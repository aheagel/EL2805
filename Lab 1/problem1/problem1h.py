from maze import *
from scipy.stats import geom

class MazeAdvanced(Maze):
    def __init__(self, maze, still_minotaur=True, prob_to_player=0.35):
        self.prob_to_player = prob_to_player
        super().__init__(maze, still_minotaur=still_minotaur)

    def init_states(self):
        
        states = dict()
        map = dict()
        s = 0
        for has_key in [False, True]:
            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[1]):
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            if self.maze[i,j] != 1:
                                states[s] = ((i,j), (k,l), has_key)
                                map[((i,j), (k,l), has_key)] = s
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
        """ Given a list of possible next states, return the probability distribution
            over these states according to the Minotaur's policy.
            :input list states         : List of possible next states.
            :return tuple next_state   : Chosen next state.
        """
        probs = {}
        dists = {}
        for next_state in states:
            state_idx = self.map[next_state]
            if next_state == 'Eaten':
                dist = 0
            elif next_state in ['Done', 'Win']:
                dist = np.inf
            else:
                player_pos = np.array(next_state[0])
                minotaur_pos = np.array(next_state[1])
                dist = np.linalg.norm(minotaur_pos - player_pos, ord=1)  # Manhattan distance
            
            dists[state_idx] = dist
            probs[state_idx] = probs.get(state_idx, 0) + (1 - self.prob_to_player) / len(states)

        # Find minimum distance and count states with that distance
        min_dist = min(dists.values())
        min_states = [key for key, dist in dists.items() if dist == min_dist]
        
        # Distribute extra probability to states with minimum distance
        for key in min_states:
            probs[key] += self.prob_to_player / len(min_states)

        return list(probs.keys()), list(probs.values())

    def move(self, state, action): # 3 state system              
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
            impossible_action_player = ((row_player, col_player), (0, 0), False) not in self.map #TODO
            
            # Current has key?
            current_has_key = self.states[state][2]
        
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
                    
                    elif self.maze[self.states[state][0][0], self.states[state][0][1]] == 2 and current_has_key: #TODO
                        states.append('Win')
                
                    else:     # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]), current_has_key))
                
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if (row_player, col_player) == (rows_minotaur[i], cols_minotaur[i]): #TODO
                        states.append('Eaten')
                    
                    elif self.maze[row_player, col_player] == 2 and self.states[state][2]: #TODO
                        states.append('Win')
                    
                    elif self.maze[row_player, col_player] == 3: # The player reaches the exit without the key
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), True))

                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), current_has_key))

                return states
                
     
if __name__ == "__main__":
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze, 3 = key


    env = MazeAdvanced(maze, prob_to_player=0.35, still_minotaur=False) # Create an environment maze

    # Define the discount and an accuracy threshold
    discount = 49/50
    accuracy_theta = 1e-12  # A small number for convergence
    start  = ((0,0), (6,5), False)

    V, policy = value_iteration(env, discount, accuracy_theta)

    horizon = 100 #np.random.geometric(p=1-discount)
    path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)

    print('Value function', V[env.map[start]])
    print(horizon)
    print("True probability for ideal case when we know the optimal policy (Minotaur can't stand still)", 1-geom.cdf(29, p=1-discount)) # special case when we got opposite parity
    animate_solution2(maze, path)