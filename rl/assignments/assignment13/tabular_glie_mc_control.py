from collections import defaultdict
from rl.markov_decision_process import MarkovDecisionProcess
from typing import Iterable, Iterator, TypeVar, Callable, Mapping, Tuple
from rl.markov_decision_process import FiniteMarkovDecisionProcess, Policy, \
    TransitionStep, NonTerminal
from rl.distribution import Distribution, FiniteDistribution, Categorical, Choose
from rl.function_approx import FunctionApprox, Tabular
from rl.policy import FiniteDeterministicPolicy, UniformPolicy, RandomPolicy
from rl.returns import returns

S = TypeVar('S')
A = TypeVar('A')

V = Mapping[NonTerminal[S], float]

ValueFunctionApprox = Distribution[NonTerminal[S]]
QValueFunctionApprox = Distribution[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]

def tabular_glie_mc_control(
    mdp: FiniteMarkovDecisionProcess[S,A],
    gamma: float,
    approx_0: QValueFunctionApprox = None,
    episode_len_tol: float = 1e-6
) -> Iterator[QValueFunctionApprox[S,A]]:

    start_states : NTStateDistribution = Choose(mdp.non_terminal_states)
    if approx_0 is None:
        approx_0 = Tabular({})
    Q_approx = approx_0
    yield Q_approx

    trace_num = 0
    counts : Mapping[Tuple[S, A], int] = defaultdict(lambda : 0)

    while True:
        trace_num += 1
        eps = 1 / trace_num

        # Make greedy policy
        greedyPolicyMap = {}
        for state in mdp.non_terminal_states:
            maxReturn = None
            for action in mdp.actions(state):
                newReturn = Q_approx(state, action)
                if maxReturn is None or newReturn > maxReturn:
                    maxReturn = newReturn
                    greedyPolicyMap[state] = action
                
        greedyPolicy = FiniteDeterministicPolicy(greedyPolicyMap)

        # Make eps-greedy policy from greedy policy
        randomPolicy = UniformPolicy({state : mdp.actions(state) for state in mdp.non_terminal_states})
        epsGreedyPolicy = RandomPolicy(Categorical({ greedyPolicy : 1-eps, randomPolicy : eps }))

        # Generate episode
        trace = mdp.simulate_actions(start_states, epsGreedyPolicy)
        episode = returns(trace, gamma, episode_len_tol)
        for step in episode:
            counts[(step.state, step.action)] += 1
            Q_approx += (step.return_ - Q_approx(step.state, step.action)) / counts[(step.state, step.action)]
        
        yield Q_approx
        



