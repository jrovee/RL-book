from tabular_mc import tabular_mc_prediction
from tabular_td import tabular_td_prediction
from rl.monte_carlo import mc_prediction
from rl.td import td_prediction
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.distribution import Choose
from rl.function_approx import Tabular

if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    traces = si_mrp.reward_traces(Choose(si_mrp.non_terminal_states))
    transitions = []

    for i, trace in enumerate(traces):
        if i > 10:
            break
        for j, step in enumerate(trace):
            if j > 10:
                break
            transitions.append(step)

    mc_predictions1 = mc_prediction(traces, Tabular(), user_gamma)
    td_predictions1 = td_prediction(transitions, Tabular(), user_gamma)
    mc_predictions2 = tabular_mc_prediction(traces, γ=user_gamma)
    td_predictions2 = tabular_td_prediction(transitions, γ=user_gamma)
    

    for i, (mc1, td1, mc2, td2) in enumerate(zip(mc_predictions1, 
                                                 td_predictions1,
                                                 mc_predictions2,
                                                 td_predictions2)):
        if i > 10:
            break
        
        if mc1.values_map.items() <= mc2.items():
            print('mc matches')
        # else:
        #     print("mc doesn't match")
        #     for state in mc2.keys():
        #         print(state)
        #         print(mc1.values_map[state])
        #         print(mc2[state])
        #     break
        
        if td1.values_map.items() <= td2.items():
            print('td matches')
        else:
            print("td doesn't match")
            print(td1.values_map.keys())
            print(td2.keys())
            break
            

        