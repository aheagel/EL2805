import numpy as np
from numba import jit, int32, float64
import time
from tqdm import tqdm

@jit(nopython=True)
def convert_to_sparse(transition_probabilities, max_branches=20):
    """
    Converts dense transition probabilities (S, S, A) to sparse format (S, A, max_branches)
    for efficient sampling in Numba.
    """
    n_states, n_next_states, n_actions = transition_probabilities.shape
    
    next_states = np.full((n_states, n_actions, max_branches), -1, dtype=np.int32)
    probs = np.zeros((n_states, n_actions, max_branches), dtype=np.float64)
    
    for s in range(n_states):
        for a in range(n_actions):
            k = 0
            for ns in range(n_next_states):
                p = transition_probabilities[s, ns, a]
                if p > 1e-12:
                    if k < max_branches:
                        next_states[s, a, k] = ns
                        probs[s, a, k] = p
                        k += 1
    return next_states, probs

@jit(nopython=True)
def get_value(n, type_idx, params):
    """
    Calculates value based on type and params.
    Types:
    0: Constant (params[0])
    1: Power decay (params[0] * n ** -params[1])
    2: Linear decay (params[0] - params[1] * n, clipped to params[2])
    """
    if type_idx == 0: # Constant
        return params[0]
    elif type_idx == 1: # Power
        if n <= 0: return 1.0 
        return params[0] * (n ** -params[1])
    elif type_idx == 2: # Linear
        val = params[0] - params[1] * n
        if val < params[2]: return params[2]
        return val
    return 0.0

@jit(nopython=True)
def _q_learning_numba(next_states, probs, rewards, start_state, gamma, n_batch_episodes, start_episode_idx,
                      alpha_type, alpha_params, epsilon_type, epsilon_params, 
                      Q, number_of_visits, done_state_idx, V_starts):
    
    n_states, n_actions = Q.shape
    
    if done_state_idx >= 0:
        Q[done_state_idx, :] = 0.0
        
    for i in range(n_batch_episodes):
        episode_idx = start_episode_idx + i
        # Epsilon depends on episode number (1-based)
        eps = get_value(episode_idx + 1, epsilon_type, epsilon_params)
        
        state = start_state
        
        step_count = 0
        while state != done_state_idx and step_count < 10000: # Safety break
            step_count += 1
            
            # Epsilon greedy policy
            if np.random.rand() < eps:
                action = np.random.randint(n_actions)
            else:
                # Best action with random tie-breaking
                max_val = -1e10 # -inf
                for a in range(n_actions):
                    if Q[state, a] > max_val:
                        max_val = Q[state, a]
                
                count = 0
                for a in range(n_actions):
                    if Q[state, a] == max_val:
                        count += 1
                
                rand_idx = np.random.randint(count)
                
                current_idx = 0
                action = 0
                for a in range(n_actions):
                    if Q[state, a] == max_val:
                        if current_idx == rand_idx:
                            action = a
                            break
                        current_idx += 1
            
            # Update visits
            number_of_visits[state, action] += 1
            
            # Get reward
            reward = rewards[state, action]
            
            # Get next state
            rand_p = np.random.rand()
            cum_p = 0.0
            next_state = -1
            
            for k in range(next_states.shape[2]):
                s_next = next_states[state, action, k]
                if s_next == -1:
                    break
                p = probs[state, action, k]
                cum_p += p
                if rand_p < cum_p:
                    next_state = s_next
                    break
            
            if next_state == -1:
                # Fallback
                for k in range(next_states.shape[2]):
                     if next_states[state, action, k] != -1:
                         next_state = next_states[state, action, k]
                     else:
                         break
            
            # Q-update
            lr = get_value(number_of_visits[state, action], alpha_type, alpha_params)
            
            max_q_next = -1e10
            if next_state != -1:
                for a in range(n_actions):
                    if Q[next_state, a] > max_q_next:
                        max_q_next = Q[next_state, a]
            else:
                max_q_next = 0.0
            
            Q[state, action] += lr * (reward + gamma * max_q_next - Q[state, action])
            
            state = next_state
            
        V_starts[episode_idx] = np.max(Q[start_state, :])

@jit(nopython=True)
def _sarsa_learning_numba(next_states, probs, rewards, start_state, gamma, n_batch_episodes, start_episode_idx,
                      alpha_type, alpha_params, epsilon_type, epsilon_params, 
                      Q, number_of_visits, done_state_idx, V_starts):
    
    n_states, n_actions = Q.shape
    
    if done_state_idx >= 0:
        Q[done_state_idx, :] = 0.0
        
    for i in range(n_batch_episodes):
        episode_idx = start_episode_idx + i
        eps = get_value(episode_idx + 1, epsilon_type, epsilon_params)
        state = start_state
        
        # Choose action (epsilon greedy)
        if np.random.rand() < eps:
            action = np.random.randint(n_actions)
        else:
            max_val = -1e10
            for a in range(n_actions):
                if Q[state, a] > max_val:
                    max_val = Q[state, a]
            count = 0
            for a in range(n_actions):
                if Q[state, a] == max_val:
                    count += 1
            rand_idx = np.random.randint(count)
            current_idx = 0
            action = 0
            for a in range(n_actions):
                if Q[state, a] == max_val:
                    if current_idx == rand_idx:
                        action = a
                        break
                    current_idx += 1
        
        step_count = 0
        while state != done_state_idx and step_count < 10000:
            step_count += 1
            
            number_of_visits[state, action] += 1
            reward = rewards[state, action]
            
            # Next state
            rand_p = np.random.rand()
            cum_p = 0.0
            next_state = -1
            for k in range(next_states.shape[2]):
                s_next = next_states[state, action, k]
                if s_next == -1: break
                p = probs[state, action, k]
                cum_p += p
                if rand_p < cum_p:
                    next_state = s_next
                    break
            if next_state == -1:
                for k in range(next_states.shape[2]):
                     if next_states[state, action, k] != -1:
                         next_state = next_states[state, action, k]
                     else: break

            # Choose next action (epsilon greedy)
            if np.random.rand() < eps:
                next_action = np.random.randint(n_actions)
            else:
                max_val = -1e10
                for a in range(n_actions):
                    if Q[next_state, a] > max_val:
                        max_val = Q[next_state, a]
                count = 0
                for a in range(n_actions):
                    if Q[next_state, a] == max_val:
                        count += 1
                rand_idx = np.random.randint(count)
                current_idx = 0
                next_action = 0
                for a in range(n_actions):
                    if Q[next_state, a] == max_val:
                        if current_idx == rand_idx:
                            next_action = a
                            break
                        current_idx += 1
            
            lr = get_value(number_of_visits[state, action], alpha_type, alpha_params)
            
            Q[state, action] += lr * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
            
        V_starts[episode_idx] = np.max(Q[start_state, :])

def parse_func_param(param):
    """
    Parses a functional parameter.
    Input can be:
    - tuple: ('constant', 0.1)
    - tuple: ('power', c, p) -> c * n^-p
    - tuple: ('linear', start, slope, min) -> max(start - slope * n, min)
    - float: treated as ('constant', val)
    - None: default
    """
    res = np.zeros(3, dtype=np.float64)
    
    if param is None:
        return 0, res # Default handled by caller usually, but here returns 0
    
    if isinstance(param, (int, float)):
        res[0] = float(param)
        return 0, res
        
    if isinstance(param, tuple):
        name = param[0]
        if name == 'constant':
            res[0] = float(param[1])
            return 0, res
        elif name == 'power':
            # c * n^-p
            res[0] = float(param[1])
            res[1] = float(param[2])
            return 1, res
        elif name == 'linear':
            # start - slope * n, min_val
            res[0] = float(param[1])
            res[1] = float(param[2])
            if len(param) > 3:
                res[2] = float(param[3])
            return 2, res
            
    raise ValueError(f"Unknown parameter format: {param}")

def Q_learning_improved(env, start, gamma, n_episodes=50000, alpha_func=None, epsilon_func=None, Q=None):
    """
    Improved Q-learning using Numba.
    
    Parameters:
    - alpha_func: Functional parameter for learning rate.
      Examples:
        ('constant', 0.1)
        ('power', 1.0, 0.66)  -> 1.0 * n^(-0.66)
    - epsilon_func: Functional parameter for exploration rate.
      Examples:
        ('constant', 0.1)
        ('linear', 1.0, 0.001, 0.1) -> max(1.0 - 0.001*n, 0.1)
    """
    # Convert transitions from dense to sparse for Numba
    next_states, probs = convert_to_sparse(env.transition_probabilities)
    rewards = env.rewards
    
    start_idx = env.map[start]
    done_idx = env.map.get('Done', -1)
    
    if Q is None:
        Q = np.random.rand(env.n_states, env.n_actions)
        
    number_of_visits = np.zeros((env.n_states, env.n_actions))
    V_starts = np.zeros(n_episodes, dtype=np.float64)
    
    # Default alpha: n**-2/3 -> 1 * n**-(2/3)
    if alpha_func is None:
        alpha_type, alpha_params = 1, np.array([1.0, 2/3, 0.0])
    else:
        alpha_type, alpha_params = parse_func_param(alpha_func)
        
    # Default epsilon: 0.1
    if epsilon_func is None:
        eps_type, eps_params = 0, np.array([0.1, 0.0, 0.0])
    else:
        eps_type, eps_params = parse_func_param(epsilon_func)
    
    batch_size = 5000
    with tqdm(total=n_episodes, desc="Q-learning") as pbar:
        for start_ep in range(0, n_episodes, batch_size):
            current_batch = min(batch_size, n_episodes - start_ep)
            _q_learning_numba(next_states, probs, rewards, start_idx, gamma, current_batch, start_ep,
                             alpha_type, alpha_params, eps_type, eps_params,
                             Q, number_of_visits, done_idx, V_starts)
            pbar.update(current_batch)
        
    return Q, number_of_visits, V_starts

def SARSA_learning_improved(env, start, gamma, n_episodes=50000, alpha_func=None, epsilon_func=None, Q=None):
    """
    Improved SARSA-learning using Numba.
    """
    # Convert transitions from dense to sparse for Numba
    next_states, probs = convert_to_sparse(env.transition_probabilities)
    rewards = env.rewards
    
    start_idx = env.map[start]
    done_idx = env.map.get('Done', -1)
    
    if Q is None:
        Q = np.random.rand(env.n_states, env.n_actions)
        
    number_of_visits = np.zeros((env.n_states, env.n_actions))
    V_starts = np.zeros(n_episodes, dtype=np.float64)
    
    if alpha_func is None:
        alpha_type, alpha_params = 1, np.array([1.0, 2/3, 0.0])
    else:
        alpha_type, alpha_params = parse_func_param(alpha_func)
        
    if epsilon_func is None:
        eps_type, eps_params = 0, np.array([0.1, 0.0, 0.0])
    else:
        eps_type, eps_params = parse_func_param(epsilon_func)
        
    batch_size = 5000
    with tqdm(total=n_episodes, desc="SARSA-learning") as pbar:
        for start_ep in range(0, n_episodes, batch_size):
            current_batch = min(batch_size, n_episodes - start_ep) # For tqdm
            _sarsa_learning_numba(next_states, probs, rewards, start_idx, gamma, current_batch, start_ep,
                             alpha_type, alpha_params, eps_type, eps_params,
                             Q, number_of_visits, done_idx, V_starts)
            pbar.update(current_batch) # For tqdm
        
    return Q, number_of_visits, V_starts
