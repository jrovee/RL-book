from tkinter import N
from typing import Sequence, Tuple, Mapping, TypeVar, Callable
from rl.distribution import Choose, Constant, Categorical
from rl.function_approx import LinearFunctionApprox, Weights
from rl.markov_decision_process import ActionMapping, FiniteMarkovDecisionProcess, StateActionMapping, TransitionStep, Terminal, NonTerminal
from rl.markov_process import NonTerminal, State
from rl.policy import DeterministicPolicy
from rl.td import least_squares_policy_iteration
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
import numpy as np
import matplotlib.pyplot as plt

St = int
S = Tuple[int, float]
A = bool

def get_steps(tree : OptimalExerciseBinTree) -> Sequence[StateActionMapping[St,A]]:
    dt: float = tree.dt()
    up_factor: float = np.exp(tree.vol * np.sqrt(dt))
    up_prob: float = (np.exp(tree.rate * dt) * up_factor - 1) / \
        (up_factor * up_factor - 1)
    steps=[
        {NonTerminal(j): {
            True: Constant(
                (
                    Terminal(-1),
                    tree.payoff(i * dt, tree.state_price(i, j))
                )
            ),
            False: Categorical(
                {
                    (NonTerminal(j + 1), 0.): up_prob,
                    (NonTerminal(j), 0.): 1 - up_prob
                }
            )
        } for j in range(i + 1)}
        for i in range(tree.num_steps + 1)
    ]
    return steps

def get_full_mapping(steps : Sequence[StateActionMapping[St,A]]) -> StateActionMapping[S,A]:
    mapping = {}
    for t,step in enumerate(steps):
        for state in step:
            aug_state = NonTerminal((state.state,t))
            mapping[aug_state] = {}
            for action in step[state]:
                old_dist = step[state][action]
                if isinstance(old_dist, Categorical):
                    new_probs = {(NonTerminal((s.state,t+1)), r) : p for (s,r),p in old_dist.probabilities.items()}
                    mapping[aug_state][action] = Categorical(new_probs)
                elif isinstance(old_dist,Constant):
                    old_sr = old_dist.value
                    if isinstance(old_sr[0], NonTerminal):
                        mapping[aug_state][action] = Constant((NonTerminal((old_sr[0].state, t+1)), old_sr[1]))
                    elif isinstance(old_sr[0], Terminal):
                        mapping[aug_state][action] = Constant((Terminal((old_sr[0].state, t+1)), old_sr[1]))
                    else:
                        print('FUCK')
                else:
                    print('FUCK')
        
    return mapping

class AmericanOptionMDP(FiniteMarkovDecisionProcess[S,A]):
    def __init__(self, tree : OptimalExerciseBinTree):
        steps = get_steps(tree)
        mapping = get_full_mapping(steps)
        super().__init__(mapping)
    
        
#gamma=np.exp(-self.rate * dt)
        
    
#least_squares_policy_iteration(transitions)

if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 300

    if is_call:
        opt_payoff = lambda _, x: max(x - strike, 0)
    else:
        opt_payoff = lambda _, x: max(strike - x, 0)

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=opt_payoff,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=num_steps_val
    )

    mdp = AmericanOptionMDP(opt_ex_bin_tree)
    gamma = 1
    behavior_policy = DeterministicPolicy(lambda x : False)
    start_states = Choose([NonTerminal((i,0)) for i in range(60, 140)])
    traces = mdp.action_traces(start_states, behavior_policy)

    atomic_experiences = []
    N_TRACES = 1000
    for _ in range(N_TRACES):
        trace = next(traces)
        for step in trace:
            atomic_experiences.append((step.state, step.next_state))
    
    # Solve for optimal policy:
    def M(s):  return s.state[0]/strike
    def tT(s): return s.state[1]/expiry_val
    def phi_0(s):   return 1
    def phi_1(s):   return np.exp(-M(s)/2)
    def phi_2(s):   return phi_1(s)*(1 - M(s))
    def phi_3(s):   return phi_1(s)*(1 - 2*M(s) + M(s)**2/2)
    def phi_0_t(s): return np.sin(np.pi/2 *(1-tT(s)))
    def phi_1_t(s): return np.log(1-tT(s))
    def phi_2_t(s): return tT(s)**2

    feature_functions = [phi_0, phi_1, phi_2, phi_3, phi_0_t, phi_1_t, phi_2_t]

    A_inv = 1.e5 * np.eye(len(feature_functions))
    b = np.zeros(len(feature_functions))
    weights = np.zeros(len(feature_functions))

    N_loops = 50
    for _ in range(N_loops):
        for s, s_prime in atomic_experiences:
            phi_s = np.array([f(s) for f in feature_functions])
            phi_s_prime = np.array([s_prime.on_non_terminal(f, 0) for f in feature_functions])
            C1 = isinstance(s_prime,NonTerminal) and np.dot(phi_s_prime, weights) >= opt_payoff(_, s.state[0])
            C2 = not C1
            u = phi_s
            v = phi_s - C1 * gamma * phi_s_prime
            db = gamma * C2 * phi_s * opt_payoff(_, s.state[0])

            A_inv -= np.outer(A_inv @ u,v @ A_inv) / (1 + v @ A_inv @ u)
            b += db
        weights = A_inv @ b

    for s,s_prime in atomic_experiences:
        phi_s = np.array([f(s) for f in feature_functions])
        if weights @ phi_s > opt_payoff(_, s.state[0]):
            plt.scatter(s.state[0], s.state[1])

    #### COMPARISON TO BINARY TREE
    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, is_call)
    time_pts, ex_bound_pts = zip(*ex_boundary)
    label = ("Call" if is_call else "Put") + " Option Exercise Boundary"
    plot_list_of_curves(
        list_of_x_vals=[time_pts],
        list_of_y_vals=[ex_bound_pts],
        list_of_colors=["b"],
        list_of_curve_labels=[label],
        x_label="Time",
        y_label="Underlying Price",
        title=label
    )

    european: float = opt_ex_bin_tree.european_price(is_call, strike)
    print(f"European Price = {european:.3f}")

    am_price: float = vf_seq[0][NonTerminal(0)]
    print(f"American Price = {am_price:.3f}")