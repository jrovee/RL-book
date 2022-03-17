from typing import Sequence, Tuple, Mapping, TypeVar, Callable
from rl.function_approx import LinearFunctionApprox, Weights
from rl.markov_process import NonTerminal, State
import numpy as np

S = TypeVar('S')
ValueFunctionApprox = LinearFunctionApprox[NonTerminal[S]]

def get_lstd_value_function(
    srs_samples : Sequence[Tuple[NonTerminal[S], float, State[S]]],
    feature_functions : Sequence[Callable[[NonTerminal[S]], float]],
    gamma : float,
    sherman_morrison_initialization : float = 1e-5,
    sherman_morrison: bool = True
) -> ValueFunctionApprox:
    if sherman_morrison:
        A_inv = np.eye(len(feature_functions)) / sherman_morrison_initialization
    else: 
        A = np.zeros((len(feature_functions), len(feature_functions)))
    b = np.zeros(len(feature_functions))
    
    for s, r, s_prime in srs_samples:
        phi_s = np.array([f(s) for f in feature_functions])
        phi_s_prime = np.array([s_prime.on_non_terminal(f, 0) for f in feature_functions])

        u = phi_s
        v = phi_s - gamma * phi_s_prime

        if sherman_morrison:
            A_inv -= np.outer(A_inv @ u,v @ A_inv) / (1 + v @ A_inv @ u)
        else: 
            A += np.outer(u, v)
        
        b += r * phi_s
        
    if sherman_morrison:
        weights = A_inv @ b
    else:
        weights = np.linalg.solve(A,b)
    
    return LinearFunctionApprox.create(feature_functions, weights=Weights.create(weights))



if __name__ == '__main__':
    from rl.markov_process import FiniteMarkovRewardProcess
    from rl.distribution import FiniteDistribution, Choose, Constant
    from rl.dynamic_programming import evaluate_mrp_result
    import matplotlib.pyplot as plt
    def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        x_vals = np.linspace(0,100,1000)
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

    #### TEST USING SNAKES AND LADDERS MRP
    sl_map : Mapping[int, int] = {   1: 38,   4: 14,   9: 31,  16:  6,  21: 42,    
                                    28: 84,  36: 44,  36: 44,  47: 26,  49: 11,  
                                    51: 67,  56: 53,  62: 19,  64: 60,  71: 91,  
                                    80:100,  87: 24,  93: 73,  95: 75,  98: 78, 
                                   101: 99, 102: 78, 103: 97, 104: 96, 105: 75 }

    for i in range(101):
        if i not in sl_map:
            sl_map[i] = i

    transition_reward_map : Mapping[int, FiniteDistribution[Tuple[int, float]]] = { 
        square : Choose([ (sl_map[next_square], -1) for next_square in range(square+1, square+7)])
        for square in range(100)
        }
    
    sl_game = FiniteMarkovRewardProcess(transition_reward_map)
    values = evaluate_mrp_result(sl_game, 1)
    raw_values = { k.state : -v for k,v in values.items() }
    plt.scatter(raw_values.keys(), raw_values.values())
    plt.title("Expected turns to victory based on tile")
    plt.xlabel("Tile Number")
    plt.ylabel("Expected turns to victory")

    #### LSPI solution
    for _ in range(5):
        N_samples = 10000; i=0
        traces = sl_game.reward_traces(Constant(NonTerminal(0)))
        srs_samples = []
        for trace in traces:
            for step in trace:
                srs_samples.append((step.state, step.reward, step.next_state))
                i += 1
                if i >= N_samples:
                    break
            if i >= N_samples:
                break
        
        def phi_0(state : NonTerminal[int]) -> float:
            return 1
        def phi_1(state : NonTerminal[int]) -> float:
            return state.state
        feature_functions = [phi_0,phi_1]
        
        vf_approx = get_lstd_value_function(srs_samples, feature_functions, gamma=1, sherman_morrison=True)
        slope = - vf_approx.weights.weights[1]
        intercept = - vf_approx.weights.weights[0]
        abline(slope, intercept)
    
    plt.savefig("snakes_and_ladders_approx.png")
    plt.show()
    