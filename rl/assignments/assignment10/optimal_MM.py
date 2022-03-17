from typing import Tuple
import numpy as np
from rl.distribution import Categorical, Constant, Distribution
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import NonTerminal, State, Terminal
from rl.policy import DeterministicPolicy

MMState = Tuple[float, float, float, int]
MMAction = Tuple[float, float]

class MarketMakerMDP(MarkovDecisionProcess[MMState, MMAction]):
    def __init__(
        self, 
        S0 : int = 100, 
        T : float = 1,
        dt : float = 0.005,
        gamma : float = 0.1,
        sigma : float = 2,
        I0 : int = 0,
        k : float = 1.5,
        c : float = 140
    ) -> None:
        self.S0 = S0
        self.T = T
        self.dt = dt
        self.dS = sigma * dt**0.5
        self.gamma = gamma
        self.sigma = sigma
        self.I0 = I0
        self.k = k
        self.c = c
    
    def actions(self, state):
        pass

    def step(
        self, 
        state: NonTerminal[MMState], 
        action: MMAction
    ) -> Distribution[Tuple[State[MMState], float]]:
        t, St, Wt, I = state.state
        pb, pa = action
        if state.state[0] >= self.T:
            return Constant((Terminal(state.state), -np.exp(-self.gamma * (Wt + I*St))))
        
        delta_b = St - pb
        delta_a = pa - St

        p_buy = self.c*np.exp(-self.k*delta_b)*self.dt
        p_sell = self.c*np.exp(-self.k*delta_a)*self.dt
        sr_probs = {
            (NonTerminal((t + self.dt,St + self.dS, Wt - pb, I + 1)),0) : 0.5 * p_buy,
            (NonTerminal((t + self.dt,St - self.dS, Wt - pb, I + 1)),0) : 0.5 * p_buy,
            (NonTerminal((t + self.dt,St + self.dS, Wt + pa, I - 1)),0) : 0.5 * p_sell,
            (NonTerminal((t + self.dt,St - self.dS, Wt + pa, I - 1)),0) : 0.5 * p_sell,
            (NonTerminal((t + self.dt,St + self.dS, Wt, I)),0) : 0.5 * (1-p_buy-p_sell),
            (NonTerminal((t + self.dt,St - self.dS, Wt, I)),0) : 0.5 * (1-p_buy-p_sell)
        }
        return Categorical(sr_probs)

    def get_optimal_policy(self) -> DeterministicPolicy:
        def action_for(state: MMState):
            t, St, Wt, It = state
            db0 = (2*It+1)*self.gamma*(self.sigma**2) * (self.T - t) / 2
            da0 = (1-2*It)*self.gamma*(self.sigma**2) * (self.T - t) / 2
            extra = 1/self.gamma * np.log(1+self.gamma/self.k)
            return St - db0 - extra, St + da0 + extra
        
        return DeterministicPolicy(action_for)


if __name__ == '__main__': 
    mm_mdp = MarketMakerMDP()
    optimal_policy = mm_mdp.get_optimal_policy()
    start_states = Constant(NonTerminal((0,100,0,0)))
    traces_iter = mm_mdp.action_traces(start_states, optimal_policy)

    traces = []
    while len(traces) < 10000:
        traces.append(next(traces_iter))
    ba_spreads = []
    for i,trace in enumerate(traces):
        for step in trace:
            ba_spreads.append(step.action[1] - step.action[0])
    
    avg_spread = np.average(ba_spreads)
    print(avg_spread)

    def fixed_spread_pol_func(state):
        return state[1] - avg_spread, state[1] + avg_spread
    fixed_spread_pol = DeterministicPolicy(fixed_spread_pol_func)

    #fixed_spread_traces = 