from typing import Sequence, Callable, Mapping, Tuple
from rl.distribution import FiniteDistribution, Constant, Categorical
from rl.markov_process import NonTerminal
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.dynamic_programming import value_iteration_result
from itertools import product
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

S = int
A = bool

class SimpleCropHarvest(FiniteMarkovDecisionProcess[S,A]):
    def __init__(
        self, 
        C : int, 
        z : float
    ):
        self.C : int = C
        self.z : float = z
        
        transitions_map = self.get_transitions()
        super().__init__(transitions_map)

    def get_transitions(self) -> Mapping[S, 
                                    Mapping[A, 
                                       FiniteDistribution[Tuple[S, float]]]]:
        mapping = {}
        for state in range(self.C + 1):
            mapping[state] = {}
            for action in [True,False]:
                if action:
                    mapping[state][action] = Constant((0, state))
                else:
                    sr_probs = defaultdict(float)
                    for i in range(state, self.C+1):
                        sr_probs[(i, 0)] = (1-self.z) / (self.C-state+1)
                    sr_probs[(0,0)] += self.z
                    mapping[state][action] = Categorical(sr_probs)
        
        return mapping
                        
if __name__ == '__main__':
    sch = SimpleCropHarvest(100,0.1)
    V, pol = value_iteration_result(sch, gamma=0.9)
    plt.plot([s.state for s in V.keys() if pol.action_for[s.state]==True], 
             [V[s] for s in V.keys()if pol.action_for[s.state]==True], 
             'g', label='Harvest')
    plt.plot([s.state for s in V.keys() if pol.action_for[s.state]==False], 
             [V[s] for s in V.keys()if pol.action_for[s.state]==False], 
             'r', label='Do not harvest')
    plt.xlabel('Quality')
    plt.ylabel('Value Function')
    plt.legend()
    #plt.savefig('4_2_exact_values.png')
    plt.show()
