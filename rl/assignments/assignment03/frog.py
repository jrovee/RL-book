from typing import Mapping, Tuple, Sequence
from rl.distribution import FiniteDistribution
from rl.policy import FiniteDeterministicPolicy
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.distribution import Choose, Categorical, FiniteDistribution
import matplotlib.pyplot as plt

V = Mapping[int, float]

# Actions:
A = True
B = False

class LilyPadGame(FiniteMarkovDecisionProcess[int, bool]):

    def __init__(self, nLilyPads : int):
        self.n : int = nLilyPads
        transitions_map = self.get_transitions(nLilyPads)
        super().__init__(transitions_map)

    def get_transitions(self, n: int) -> Mapping[int, Mapping[bool, FiniteDistribution[Tuple[int, float]]]]:
        d = {}
        for s in range(1,n):
            d[s] = {}
            d[s][A] = Categorical({ (s-1, 0)                 : s/n,
                                    (s+1, float((s+1) == n)) : (n-s)/n})
            d[s][B] = Choose([(i, float(i==n)) for i in range(0, n+1) if i != s ])
        return d

    def get_policies(self) -> Sequence[FiniteDeterministicPolicy]:
        action_mapping_list : Sequence[Mapping[int, bool]] = [{ j+1 : bool((i//2**j) % 2) for j in range(self.n-1)} for i in range(2**(self.n-1))]
        return [FiniteDeterministicPolicy(action_for) for action_for in action_mapping_list]
    
    def solve_optimal_policy(self) -> Tuple[FiniteDeterministicPolicy, V]:
        allPolicies = self.get_policies()

        optimalPolicy = None
        optimalValues = None
        optimalPolicyValueSum = 0 # The optimal policy will have the highest sum of values (since each V[i] will be individually maximized)
        for policy in allPolicies:
            policyImpliedMRP = self.apply_finite_policy(policy)
            values = policyImpliedMRP.get_value_function_vec(gamma=1)
            if sum(values) > optimalPolicyValueSum:
                optimalPolicy = policy
                optimalValues = values
                optimalPolicyValueSum = sum(values)

        return optimalPolicy, optimalValues
        

if __name__ == '__main__':
    f,axs = plt.subplots(3,1,figsize=(4,6))

    for n in (3,6,9):
        lpg = LilyPadGame(n)
        policy, values = lpg.solve_optimal_policy()

        for nonTerminal, value in zip(lpg.non_terminal_states, values):
            stateNum = nonTerminal.state
            if policy.action_for[stateNum] == A:
                axs[n//3-1].plot(stateNum, value, 'or')
            elif policy.action_for[stateNum] == B:
                axs[n//3-1].plot(stateNum, value, 'ob')
        axs[n//3-1].set_xlabel('Lily Pad number')
        axs[n//3-1].set_ylabel('Escape probability')
        axs[n//3-1].set_title('Red: "Take action A"; Blue: "Take action B"', fontsize=8)
        axs[n//3-1].set_xticks(list(range(1,n)))
    
    plt.tight_layout()
    #plt.savefig('frog_mdp.png')
    plt.show()