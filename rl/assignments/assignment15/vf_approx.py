from typing import Sequence, Tuple, Mapping
import numpy as np
import random

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    values : ValueFunc = {}
    counts : Mapping[S,int] = {}
    for state, return_ in state_return_samples:
        if state in values:
            counts[state] += 1
            values[state] += (return_ - values[state]) / counts[state]
        else:
            counts[state] = 1
            values[state] = return_
    return values



def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    state_counts : Mapping[S, int] = {}
    transition_counts : Mapping[S, Mapping[S, int]] = {}
    rewards : RewardFunc = {}

    for state, reward, next_state in srs_samples:
        if state in state_counts:
            state_counts[state] += 1
            rewards[state] += (reward - rewards[state]) / state_counts[state]
            if next_state in transition_counts[state]:
                transition_counts[state][next_state] += 1
            else:
                transition_counts[state][next_state] = 1
        else:
            state_counts[state] = 1
            rewards[state] = reward
            transition_counts[state] = {}
            transition_counts[state][next_state] = 1
    
    transition_probs : ProbFunc = {s : {t : nt/ state_counts[s] for t, nt in counts.items()} for s, counts in transition_counts.items()}
    return transition_probs, rewards


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    nt_states : Mapping[S, int] = {s : i for i,s in enumerate(reward_func.keys()) } # state indexing
    N_nt = len(nt_states)
    transition_matrix : np.ndarray = np.zeros((N_nt, N_nt))
    for state in prob_func:
        for next_state in prob_func[state]:
            if next_state in nt_states:
                transition_matrix[nt_states[state], nt_states[next_state]] = prob_func[state][next_state]

    rewards_vector : np.ndarray = np.array(list(reward_func.values()))
    # print(rewards_vector)
    # print(reward_func)
    # print()
    # print(np.eye(N_nt) - transition_matrix)
    vf_vector = np.linalg.solve(np.eye(N_nt) - transition_matrix, rewards_vector)

    return {state : vf_vector[index] for state, index in nt_states.items()}



def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    values : ValueFunc = {}
    counts : Mapping[S, int] = {}
    for _ in range(num_updates):
        state, reward, next_state = random.choice(srs_samples)
        if state in values:
            alpha = learning_rate * (counts[state] / learning_rate_decay + 1) ** -0.5
            counts[state] += 1
            if next_state in values:
                values[state] += alpha * (reward + values[next_state] - values[state])
            else:
                values[state] += alpha * (reward - values[state])
        else:
            counts[state] = 1
            if next_state in values:
                values[state] = reward + values[next_state]
            else:
                values[state] = reward
    
    return values


def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    nt_index = {}
    for s, _, _ in srs_samples:
        if s not in nt_index:
            nt_index[s] = len(nt_index)
    
    A = np.zeros((len(nt_index), len(nt_index)))
    A_inv = np.zeros((len(nt_index), len(nt_index)))
    b = np.zeros((len(nt_index)))

    inverted = False
    for s, r, s_prime in srs_samples:
        i = nt_index[s]
        j = nt_index[s_prime] if s_prime in nt_index else None

        b[i] += r
        A[i,i] += 1
        if j is not None:
            A[i,j] -= 1
        
        if not inverted:
            if np.linalg.det(A) != 0:
                A_inv = np.linalg.inv(A)
                inverted = True
        else:
            if j is not None:
                A_inv -= np.outer(A_inv[:,i], A_inv[i,:] - A_inv[j,:]) / (1+A_inv[i,i] - A_inv[j,i])
            else:
                A_inv -= np.outer(A_inv[:,i], A_inv[i,:]) / (1+A_inv[i,i])
    
    values_vec = A_inv @ b
    return { s : values_vec[i] for s,i in nt_index.items()}
            


        



if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))