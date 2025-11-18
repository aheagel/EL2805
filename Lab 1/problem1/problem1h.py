from maze import *

class MazeAdvanced(Maze):
    def __init__(self, maze, still_minotaur=True, prob_to_player=0.35):
        super().__init__(maze, still_minotaur)
        self.prob_to_player = prob_to_player
        self.transition_probabilities = self.__advanced_transitions()
    
    def __advanced_transitions(self):
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
                prob = (1-self.prob_to_player) / len(next_states) #Minotaur moves uniformly at random 0.65 times out of 1
                
                min_dist = self.maze.shape[0] + self.maze.shape[1] + 1  # Initialize with a large distance
                min_state = None
                for next_state in next_states:
                    transition_probabilities[s, self.map[next_state], a] += prob # Accumulate probabilities for each possible next state as we could have the same state multiple times

                    if next_state == 'Eaten':
                        dist = 0
                    elif next_state in ['Done', 'Win']:
                        dist = self.maze.shape[0] + self.maze.shape[1]  # Large distance for terminal states 
                    else:
                        player_pos = np.array(next_state[0])
                        minotaur_pos = np.array(next_state[1])
                        dist = np.linalg.norm(minotaur_pos - player_pos, ord=1)  # Manhattan distance

                    if dist < min_dist:
                        min_dist = dist
                        min_state = next_state
                
                # Add the probability of the Minotaur moving towards the player
                transition_probabilities[s, self.map[min_state], a] += self.prob_to_player
    

        return transition_probabilities


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
    survival_prob = 1/50
    env = MazeAdvanced(maze, prob_to_player=0.5, still_minotaur=True) # Create an environment maze

    V, policy = value_iteration(env, gamma=1-survival_prob, epsilon=1e-5)
    #V, policy = dynamic_programming(env, horizon=30)
    # Simulate the shortest path starting from position A
    method = 'ValIter'  # Choose either 'DynProg' or 'ValIter'
    start  = ((0,0), (6,5))
    path = env.simulate(start, policy, method)[0]

    print(V[env.map[start]])
    animate_solution2(maze, path)