'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

import numpy as np
from typing import Iterable, Iterator, TypeVar, Callable
from rl.distribution import Categorical
from rl.approximate_dynamic_programming import (ValueFunctionApprox,
                                                QValueFunctionApprox,
                                                NTStateDistribution)

from typing import Mapping, Callable
from rl.function_approx import learning_rate_schedule
from rl.iterate import last
from rl.markov_decision_process import MarkovDecisionProcess, Policy, \
    TransitionStep, NonTerminal
from rl.policy import DeterministicPolicy, RandomPolicy, UniformPolicy
import rl.markov_process as mp
from rl.returns import returns
import itertools
from collections import defaultdict
from rl.function_approx import Gradient

S = TypeVar('S')
A = TypeVar('A')

V = Mapping[NonTerminal[S], float]


def tabular_td_lambda_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    γ: float,
    lambd: float,
    V_0: V = defaultdict(lambda : 0.0),
    lr_func: Callable[[int], float] = lambda n : 1/n,
    episode_length_tolerance: float = 1e-6
) -> Iterator[V]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MRP
      γ -- discount rate (0 < γ ≤ 1), default: 1
      V_0 -- initial value function, default: 0's for every state
      lr_func -- learning rate as a function of the number of times a state has been seen
      episode_length_tolerance -- stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated value
    function after each episode.

    '''
    values = V_0
    n_seen = defaultdict(lambda : 0)
    yield values

    for trace in traces:
        eligibility : Mapping[S, float] = defaultdict(lambda : 0)
        for step in trace:
            for state in eligibility.keys():
                eligibility[state] *= γ*lambd
            eligibility[step.state] += 1

            for state in eligibility.keys():
                n_seen[state] += 1
                alpha = lr_func(n_seen[state])
                error = step.reward + γ*values[step.next_state] - values[state]
                values[state] += alpha * error * eligibility[state]
            
            yield values


def td_lambda_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx_0: ValueFunctionApprox,
    γ: float,
    lambd: float
) -> Iterator[V]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MRP
      γ -- discount rate (0 < γ ≤ 1), default: 1
      lambd -- 
      V_0 -- initial value function, default: 0's for every state

    Returns an iterator with updates to the approximated value
    function after each episode.

    '''
    values = approx_0
    yield values

    for trace in traces:
        eligibility : Mapping[S, float] = Gradient(values).zero()
        for step in trace:
            x = step.state
            target = step.reward + γ*values(step.next_state)
            eligibility *= γ*lambd
            eligibility[step.state] += values.objective_gradient(
                [(x, target)],
                lambda x_lst, y_lst : np.ones(len(x_lst))
            )
            yield values



            



if __name__ == '__main__':
    def create_trace(sr_list):
        steps_list = []
        for init_state in sr_list[:-1:2]:
            steps_list.append([init_state])
        
        for i, reward in enumerate(sr_list[1::2]): 
            steps_list[i].append(reward)
        
        for i, final_state in enumerate(sr_list[2::2]):
            steps_list[i].append(final_state)

        return [TransitionStep(state, None, next_state, reward) for state,reward, next_state in steps_list]

    raw_traces = [['a', 1, 'b', 2, 'c'],
                  ['b', 2, 'b', 1, 'a', 4, 'c']]
    traces = [create_trace(trace) for trace in raw_traces]

    predictor = tabular_td_lambda_prediction(traces, 1, .9)
    for prediction in predictor:
        print(prediction)
    
    
