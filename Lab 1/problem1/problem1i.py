from problem1h import MazeAdvanced
import numpy as np
from scipy.stats import geom
from maze import animate_solution

# Lets do Q-learning on the advanced maze environment
def Q_learning(env, start, Q=None,n_episodes=50000, gamma=0.99, epsilon=0.1) -> np.ndarray:
    """
    Q-learning algorithm for the advanced maze environment. epsilon-greedy policy is used for action selection.
    
    Parameters:
    - env: The MazeAdvanced environment.
    - n_episodes: Number of episodes to train.
    - max_steps: Maximum steps per episode.
    - gamma: Discount factor.
    - epsilon: Exploration rate.
    
    Returns:
    - Q: The learned Q-table.
    """

    if Q is None:
        Q = env.rewards # Initialize Q-table with zeros
    
    for episode in range(n_episodes):
        number_of_visits = np.zeros((env.n_states, env.n_actions))
        state = env.map[start] # Reset to start state at the beginning of each episode
        for step in range(geom.rvs(1-gamma)): # Steps before Death
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)  # Explore: random action
            else:
                action = np.argmax(Q[state])  # Exploit: best action from Q-table

            number_of_visits[state, action] += 1

            probs = env.transition_probabilities[state, :, action]
            next_state = np.random.choice(env.n_states, p=probs)  # Sample next state
            reward = env.rewards[state, action]

            terminal = env.states[next_state] in ['Done']

            Q[state, action] += number_of_visits[state, action]**(-2/3) * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            state = next_state
            if terminal:
                break
                
    return Q

if __name__ == "__main__":
    # Define the maze layout
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze, 3 = key
    env = MazeAdvanced(maze, prob_to_player=0.35, still_minotaur=False)

    # Define the discount and an accuracy threshold
    discount = 49/50
    start  = ((0,0), (6,5), False)

    Q = Q_learning(env, start, gamma=discount, epsilon=0.1)

    policy = np.argmax(Q, axis=1)


    horizon = 100
    path = env.simulate(start, np.repeat(policy.reshape(len(policy),1), horizon, 1), horizon)

    animate_solution(maze, path)
