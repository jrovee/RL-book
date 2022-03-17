from P4_2 import SimpleCropHarvest
import numpy as np
import matplotlib.pyplot as plt
from rl.distribution import Distribution, Constant
from rl.dynamic_programming import value_iteration_result
from rl.function_approx import LinearFunctionApprox, Tabular, Weights, AdamGradient
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import NonTerminal
from rl.monte_carlo import epsilon_greedy_policy
from rl.td import PolicyFromQType, q_learning, QValueFunctionApprox

S = int
A = bool

C = 100
z = 0.1
gamma = 0.9

q_learning_eps = 0.1

mdp : MarkovDecisionProcess = SimpleCropHarvest(C,z)
start_states : Distribution[NonTerminal[S]] = Constant(NonTerminal(0))
policy_from_q : PolicyFromQType = lambda Q, m : epsilon_greedy_policy(Q,m, 0.1)
max_episode_length : int = 100

phi_1 = lambda sa : sa[1]
phi_2 = lambda sa : (1-sa[1])
phi_3 = lambda sa : sa[1]*sa[0].state
phi_4 = lambda sa : (1-sa[1])*sa[0].state
phi_5 = lambda sa : (1-sa[1])*(sa[0].state - 15)**2 / 100
feature_functions = [phi_1, phi_2, phi_3, phi_4, phi_5]
initial_weights = Weights.create(np.array([220, 240, 1, 0.5,0]))
approx_0 = LinearFunctionApprox.create(feature_functions, weights=initial_weights)

Q_iter = q_learning(mdp, policy_from_q, start_states, approx_0, gamma, max_episode_length)

V, pol = value_iteration_result(mdp, gamma=0.9)

losses = []
for i, Q in enumerate(Q_iter):
    if i % 100 == 0:
        loss = sum((V[s] - max(Q((s, False)), Q((s,True))))**2 for s in V.keys())
        losses.append(loss)
        
    if i % 100000 == 0:
        plt.semilogy([100*i for i in range(len(losses))], losses)
        plt.xlabel("Number of iterations")
        plt.ylabel("Error of Value Function Approx")
        plt.show()
        plt.plot([s.state for s in V.keys() if pol.action_for[s.state]==True], 
                    [V[s] for s in V.keys()if pol.action_for[s.state]==True], 
                    'g', label='Harvest')
        plt.plot([s.state for s in V.keys() if pol.action_for[s.state]==False], 
                    [V[s] for s in V.keys()if pol.action_for[s.state]==False], 
                    'r', label='Do not harvest')
        
        plt.plot(range(101), [max(Q((NonTerminal(q), False)), Q((NonTerminal(q),True))) for q in range(101)], label='Q-learning implied Value function')

        plt.xlabel('Quality')
        plt.ylabel('State Value')
        plt.legend()
        plt.show()
        print(Q.weights.weights)
