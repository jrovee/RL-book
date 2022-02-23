from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand1: int
    on_order1: int
    on_hand2: int
    on_order2: int


    def inventory_positions(self, transfer:int) -> Tuple[int,int]:
        return (self.on_hand1 + self.on_order1 + transfer, 
                    self.on_hand2 + self.on_order2 - transfer)



InvOrderMapping = Mapping[
    InventoryState,
    Mapping[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]]
]


class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
        self,
        capacities: Tuple[int,int],
        poisson_lambdas: Tuple[float,float],
        holding_costs: Tuple[float, float],
        stockout_costs: Tuple[float, float],
        supplier_transport_cost: float,
        transfer_transport_cost: float
    ):
        self.cap1: int; self.cap2: int 
        self.cap1, self.cap2 = capacities
        self.lambda1: float; self.lambda2: float
        self.lambda1, self.lambda2 = poisson_lambdas
        self.holding_cost1: float; self.holding_cost2: float
        self.holding_cost1, self.holding_cost2 = holding_costs
        self.stockout_cost1: float; self.stockout_cost2: float
        self.stockout_cost1, self.stockout_cost2 = stockout_costs
        self.supplier_transport_cost: float = supplier_transport_cost
        self.transfer_transport_cost: float = transfer_transport_cost

        self.poisson_distr1 = poisson(self.lambda1)
        self.poisson_distr2 = poisson(self.lambda2)

        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[Tuple[int, int, int], 
                    Categorical[Tuple[InventoryState, float]]]] = {}

        for alpha1 in range(self.cap1 + 1):
            for beta1 in range(self.cap1 + 1 - alpha1):
                for alpha2 in range(self.cap2 + 1):
                    for beta2 in range(self.cap2 + 1 - alpha2):
                        state: InventoryState = InventoryState(alpha1, beta1, alpha2, beta2)
                        d1: Dict[Tuple[int, int, int], 
                                    Categorical[Tuple[InventoryState, float]]] = {}
                        for t in range(-alpha1, alpha2 + 1):
                            ip1: int; ip2: int
                            ip1, ip2 = state.inventory_positions(t)
                            for order1 in range(self.cap1 - ip1 + 1):
                                for order2 in range(self.cap2 - ip2 + 1):
                                    base_reward: float = - self.holding_cost1 * (alpha1 + (t<0)*t) \
                                                    - self.holding_cost2 * (alpha2 - (t>0)*t) \
                                                    - self.supplier_transport_cost * (order1+order2 > 0) \
                                                    - self.transfer_transport_cost * (t != 0)
                                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] =\
                                        {(InventoryState(ip1 - i1, order1, ip2 - i2, order2), base_reward):
                                            self.poisson_distr1.pmf(i1) * self.poisson_distr2.pmf(i2) 
                                                for i1 in range(ip1) for i2 in range(ip2)}
                                    p_stockout_1 : float = 1 - self.poisson_distr1.cdf(ip1 - 1)
                                    p_stockout_2 : float = 1 - self.poisson_distr2.cdf(ip2 - 1)
                                    stockout_reward_1: float = - self.stockout_cost1 *\
                                                        (p_stockout_1 * (self.lambda1 - ip1) +
                                                        ip1 * self.poisson_distr1.pmf(ip1))
                                    stockout_reward_2: float = - self.stockout_cost2 *\
                                                        (p_stockout_2 * (self.lambda2 - ip2) +
                                                        ip2 * self.poisson_distr2.pmf(ip2))
                                    for i1 in range(ip1):
                                        sr_probs_dict[(InventoryState(ip1-i1, order1, 0, order2),
                                                        base_reward + stockout_reward_2)] = \
                                                            p_stockout_2 * self.poisson_distr1.pmf(i1)
                                    for i2 in range(ip2):
                                        sr_probs_dict[(InventoryState(0, order1, ip2-i2, order2),
                                                        base_reward + stockout_reward_1)] = \
                                                            p_stockout_1 * self.poisson_distr2.pmf(i2)
                                    sr_probs_dict[(InventoryState(0,order1,0,order2), 
                                                    base_reward + stockout_reward_1 + stockout_reward_2)] = \
                                                        p_stockout_1 * p_stockout_2
                                    d1[(order1, order2, t)] = Categorical(sr_probs_dict)
                        
                        d[state] = d1
                
        return d


if __name__ == '__main__':
    from pprint import pprint

    user_capacities = (2, 3)
    user_poisson_lambdas = (1.0, 1.0)
    user_holding_costs = (1.0, 1.0)
    user_stockout_costs = (10.0, 5.0)
    user_supplier_transport_cost = 1.0
    user_transfer_transport_cost = 2.0


    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacities=user_capacities,
            poisson_lambdas=user_poisson_lambdas,
            holding_costs=user_holding_costs,
            stockout_costs=user_stockout_costs,
            supplier_transport_cost=user_supplier_transport_cost,
            transfer_transport_cost=user_transfer_transport_cost
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    fdp: FiniteDeterministicPolicy[InventoryState, int] = \
        FiniteDeterministicPolicy(
            {InventoryState(alpha1, beta1, alpha2, beta2): 
                (user_capacities[0] - (alpha1 + beta1), user_capacities[1] - (alpha2 + beta2), 0)
                for alpha1 in range(user_capacities[0] + 1)
                for beta1 in range(user_capacities[0] + 1 - alpha1)
                for alpha2 in range(user_capacities[1] + 1)
                for beta2 in range(user_capacities[1] + 1 - alpha2)}
    )

    print("Deterministic Policy Map")
    print("------------------------")
    print(fdp)

    implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
        si_mdp.apply_finite_policy(fdp)
    print("Implied MP Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(
        {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
         for s, v in implied_mrp.transition_map.items()}
    ))

    print("Implied MRP Transition Reward Map")
    print("---------------------")
    print(implied_mrp)

    print("Implied MP Stationary Distribution")
    print("-----------------------")
    implied_mrp.display_stationary_distribution()
    print()

    print("Implied MRP Reward Function")
    print("---------------")
    implied_mrp.display_reward_function()
    print()

    print("Implied MRP Value Function")
    print("--------------")
    implied_mrp.display_value_function(gamma=user_gamma)
    print()

    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result

    print("Implied MRP Policy Evaluation Value Function")
    print("--------------")
    pprint(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
    print()

    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    pprint(opt_vf_pi)
    print(opt_policy_pi)
    print()

    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()
