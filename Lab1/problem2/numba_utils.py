import numpy as np
from numba import njit

@njit(cache=True)
def scale_state_numba(state, low, high):
    """ Rescaling of s to the box [0,1]^2 """
    return (state - low) / (high - low)

@njit(cache=True)
def fourier_basis_numba(state, eta):
    return np.cos(np.pi * (eta.T @ state))

@njit(cache=True)
def value_numba(state, W, eta):
    # Returns Q-values for all actions
    phi = fourier_basis_numba(state, eta)
    return W.T @ phi

@njit(cache=True)
def value_action_numba(state, W, eta, action):
    # Returns Q-value for specific action
    phi = fourier_basis_numba(state, eta)
    return W[:, action].T @ phi

@njit(cache=True)
def td_error_numba(reward, terminal, Q_current, Q_next, discount):
    if terminal:
        return reward - Q_current
    else:
        return reward + discount * Q_next - Q_current

@njit(cache=True)
def eligibility_trace_numba(Zold, current_state, current_action, lamda, discount, eta):
    mask = np.zeros_like(Zold)
    phi = fourier_basis_numba(current_state, eta)
    mask[:, current_action] = phi
    return lamda * discount * Zold + mask

@njit(cache=True)
def nesterov_iteration_numba(W, v, grad, learning_rate, momentum):
    v_new = momentum * v + learning_rate * grad
    W_new = W + momentum * v_new + learning_rate * grad
    return W_new, v_new

@njit(cache=True)
def epsilon_greedy_policy_numba(V, eps):
    if np.random.rand() < eps:
        return np.random.randint(len(V))
    else:
        # Numba doesn't support np.flatnonzero directly in the same way or argmax with random tie breaking easily
        # But we can implement it.
        max_val = V[0]
        count = 1
        best_action = 0
        for i in range(1, len(V)):
            if V[i] > max_val:
                max_val = V[i]
                best_action = i
                count = 1
            elif V[i] == max_val:
                count += 1
                if np.random.rand() < 1.0 / count:
                    best_action = i
        return best_action

@njit(cache=True)
def c2d_numba(state, shape):
    # state is scaled to [0, 1]
    # shape is (20, 20) for example
    idx = np.floor(state * shape).astype(np.int32)
    # Clip to be safe
    for i in range(len(idx)):
        if idx[i] >= shape[i]:
            idx[i] = shape[i] - 1
        if idx[i] < 0:
            idx[i] = 0
    return idx

@njit(cache=True)
def ucb_policy_numba(V, visit_counts, total_counts, c):
    # V: Q-values
    # visit_counts: visits for each action in current state
    # total_counts: total visits for current state
    
    # Avoid division by zero
    vc = visit_counts + 1e-5
    bonus = c * np.sqrt(np.log(total_counts + 1) / vc)
    ucb_values = V + bonus
    
    # Argmax with tie breaking
    max_val = ucb_values[0]
    count = 1
    best_action = 0
    for i in range(1, len(ucb_values)):
        if ucb_values[i] > max_val:
            max_val = ucb_values[i]
            best_action = i
            count = 1
        elif ucb_values[i] == max_val:
            count += 1
            if np.random.rand() < 1.0 / count:
                best_action = i
    return best_action

@njit(cache=True)
def advanced_learning_rate_numba(l_rate, normed_eta):
    # normed_eta shape (N,)
    # l_rate scalar
    # returns shape (N, 1)
    res = np.empty_like(normed_eta)
    for i in range(len(normed_eta)):
        if normed_eta[i] != 0:
            res[i] = l_rate / normed_eta[i]
        else:
            res[i] = l_rate
    return res.reshape(-1, 1)
