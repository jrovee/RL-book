from rl.distribution import Distribution, Constant, Gaussian, Choose, SampledDistribution, Bernoulli
from itertools import product
from collections import defaultdict
import operator
from typing import Mapping, Iterator, TypeVar, Tuple, Dict, Iterable, Generic

import numpy as np

from rl.distribution import Categorical, Choose
from rl.markov_process import NonTerminal, State, Terminal
from rl.markov_decision_process import (MarkovDecisionProcess, FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess)
from rl.policy import FinitePolicy, FiniteDeterministicPolicy
from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf

import random 

from dataclasses import dataclass
from rl import dynamic_programming

from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.td import PolicyFromQType, epsilon_greedy_action


S = TypeVar('S')
A = TypeVar('A')


class TabularQValueFunctionApprox(Generic[S, A]):
    '''
    A basic implementation of a tabular function approximation with constant learning rate of 0.1
    also tracks the number of updates per state
    You should use this class in your implementation
    '''
    
    def __init__(self):
        self.counts: Mapping[Tuple[NonTerminal[S], A], int] = defaultdict(int)
        self.values: Mapping[Tuple[NonTerminal[S], A], float] = defaultdict(float)
    
    def update(self, k: Tuple[NonTerminal[S], A], tgt):
        alpha = 0.1
        self.values[k] = (1 - alpha) * self.values[k] + tgt * alpha
        self.counts[k] += 1
    
    def __call__(self, x_value: Tuple[NonTerminal[S], A]) -> float:
        return self.values[x_value]


def double_q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    γ: float
) -> Iterator[TabularQValueFunctionApprox[S, A]]:
    '''
    Implement the double q-learning algorithm as outlined in the question
    '''
    ##### Your Code HERE #########
    eps = 0.1
    Q1 = TabularQValueFunctionApprox()
    Q2 = TabularQValueFunctionApprox()
    Q_estimate = TabularQValueFunctionApprox() # Keeps track of (Q1 + Q2)/2
    yield Q_estimate
    
    while True:
        state : NonTerminal = states.sample()
        while isinstance(state, NonTerminal):
            action : A = epsilon_greedy_action(Q_estimate, state, mdp.actions(state),eps)
            next_state : State[S]
            reward : float
            next_state, reward = mdp.step(state, action).sample()
            if Bernoulli(0.5).sample():
                if isinstance(next_state,NonTerminal):
                    next_action : A = epsilon_greedy_action(Q2, next_state, mdp.actions(next_state),0)
                    future_value : float = Q1((next_state, next_action))
                else:
                    future_value = 0
                tgt : float = reward + γ * future_value
                Q1.update((state, action), tgt)
            else:
                if isinstance(next_state,NonTerminal):
                    next_action : A = epsilon_greedy_action(Q1, next_state, mdp.actions(next_state),0)
                    future_value : float = Q2((next_state, next_action))
                else:
                    future_value = 0
                tgt : float = reward + γ * future_value
                Q2.update((state, action), tgt)
            
            Q_estimate.values[(state,action)] = (Q1((state,action)) + Q2((state,action))) / 2
            Q_estimate.counts[(state,action)] += 1
            state = next_state
        yield Q_estimate
    ##### End Your Code HERE #########
    
            
def q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    γ: float
) -> Iterator[TabularQValueFunctionApprox[S, A]]:
    '''
    Implement the standard q-learning algorithm as outlined in the question
    '''
    ##### Your Code HERE #########
    eps = 0.1
    Q = TabularQValueFunctionApprox()
    yield Q
    
    while True:
        state : NonTerminal = states.sample()
        while isinstance(state, NonTerminal):
            action : A = epsilon_greedy_action(Q, state, mdp.actions(state),eps)
            next_state : State[S]
            reward : float
            next_state, reward = mdp.step(state, action).sample()
            next_value : float = max(Q((next_state, a)) for a in mdp.actions(next_state)) \
                                    if isinstance(next_state,NonTerminal) else 0.0
            tgt : float = reward + γ * next_value
            Q.update((state, action), tgt)
            state = next_state
        yield Q
    ##### End Your Code HERE #########



@dataclass(frozen=True)
class P1State:
    '''
    Add any data and functionality you need from your state
    '''
    ##### Your Code HERE #########
    id : str
    ##### End Your Code HERE #########
    

class P1MDP(MarkovDecisionProcess[P1State, str]):
    
    def __init__(self, n):
        self.n = n
        
        
    def actions(self, state: NonTerminal[P1State]) -> Iterable[str]:
        '''
        return the actions available from: state
        '''
        ##### Your Code HERE #########
        if state.state.id == 'A':
            return ['a1','a2']
        elif state.state.id == 'B':
            return ['b' + str(i) for i in range(1,self.n+1)]
        ##### End Your Code HERE #########
    
    def step(
        self,
        state: NonTerminal[P1State],
        action: str
    ) -> Distribution[Tuple[State[P1State], float]]:
        '''
        return the distribution of next states conditioned on: (state, action)
        '''
        ##### Your Code HERE #########
        if state.state.id == 'A':
            if action == 'a1':
                return Constant((NonTerminal(P1State('B')), 0))
            elif action == 'a2':
                return Constant((Terminal(P1State('T')), 0))
        elif state.state.id == 'B':
            return SampledDistribution(lambda : (Terminal(P1State('T')), Gaussian(-0.1, 1).sample()))
        ##### End Your Code HERE #########


if __name__ == '__main__':
    mdp = P1MDP(10)
    initial_state = NonTerminal(P1State('A'))
    initial_state_dist = Constant(initial_state)
    gamma = 1

    N_episodes = 400
    N_runs = 100

    QAa1_std_lst = []
    QAa1_dbl_lst = []

    for _ in range(N_runs):
        Q_std = q_learning(mdp, initial_state_dist, gamma)
        Q_dbl = double_q_learning(mdp, initial_state_dist, gamma)
        QAa1_std_lst.append([])
        QAa1_dbl_lst.append([])
        for _ in range(N_episodes):
            QAa1_std_lst[-1].append(next(Q_std)((initial_state, 'a1')))
            QAa1_dbl_lst[-1].append(next(Q_dbl)((initial_state, 'a1')))

    QAa1_std_avgs = np.mean(QAa1_std_lst, axis=0)
    QAa1_dbl_avgs = np.mean(QAa1_dbl_lst, axis=0)

    import matplotlib.pyplot as plt
    plt.plot(QAa1_std_avgs, label='Standard')
    plt.plot(QAa1_dbl_avgs, label='Double')
    plt.legend()
    plt.xlabel('Number of steps')
    plt.ylabel('Average value of Q(A,$a_1$)')
    plt.tight_layout()
    #plt.savefig('P1_graph.png')
    plt.show()

    # QBbi_dbl_lst = np.zeros((10, N_runs, N_episodes))

    # for j in range(N_runs):
    #     Q_dbl = double_q_learning(mdp, initial_state_dist, gamma)
    #     for k in range(N_episodes):
    #         Q_next = next(Q_dbl)
    #         B_state = NonTerminal(P1State('B'))
    #         Q_vals = sorted([Q_next((B_state, a)) for a in mdp.actions(B_state)])
    #         for i in range(10):
    #             QBbi_dbl_lst[i,j,k] = Q_vals[i]
    # QBbi_dbl_avgs = np.mean(QBbi_dbl_lst, axis=1)

    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     plt.plot(QBbi_dbl_avgs[i,:], label=i)
    # plt.legend()
    # plt.xlabel('Number of steps')
    # plt.ylabel('Average value of Q(A,$a_1$)')
    # plt.savefig('P1_explanation_graph.png')
    # plt.show()