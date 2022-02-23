from typing import Mapping, Tuple
from rl.distribution import FiniteDistribution, Choose
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess, NonTerminal
from rl.dynamic_programming import evaluate_mrp_result
import numpy as np
import matplotlib.pyplot as plt

RUN_AS_MP = False
RUN_AS_MRP = True

#### IMPLEMENTATION AS MP WITH EV CALCULATED BY SIMULATION
if RUN_AS_MP:
    sl_map : Mapping[int, int] = {   1: 38,   4: 14,   9: 31,  16:  6,  21: 42,    
                                    28: 84,  36: 44,  36: 44,  47: 26,  49: 11,  
                                    51: 67,  56: 53,  62: 19,  64: 60,  71: 91,  
                                    80:100,  87: 24,  93: 73,  95: 75,  98: 78, 
                                   101: 99, 102: 78, 103: 97, 104: 96, 105: 75 }

    for i in range(101):
        if i not in sl_map:
            sl_map[i] = i

    transition_map : Mapping[int, FiniteDistribution[int]]= { 
        square : Choose([sl_map[next_square] for next_square in range(square+1, square+7)])
        for square in range(100)
        }

    sl_game : FiniteMarkovProcess = FiniteMarkovProcess(transition_map)
    traces = sl_game.traces(Choose([NonTerminal(0)]))
    lengths = []

    for i, trace in enumerate(traces):
        if i >= 100000:
            break

        length = -1
        for state in trace:
            length += 1
        lengths.append(length)

    plt.hist(lengths, bins=np.arange(min(lengths)-0.5, max(lengths)+1.5), density=True)
    plt.title("Probability Distribution of Game Lengths\n(based on 100,000 simulations)")
    plt.xlabel("Length of Game (turns)")
    plt.ylabel("Probability")
    #plt.savefig("snakes_and_ladders_lengths.png")
    plt.show()

#### IMPLEMENTATION AS MRP WITH EV CALCULATED USING VALUE ITERATION
if RUN_AS_MRP:
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
    plt.savefig("snakes_and_ladders_tiles.png")
    plt.show()

    