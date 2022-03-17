from abc import abstractmethod
from rl.chapter9.order_book import OrderBook, PriceSizePairs, DollarsAndShares
from rl.markov_process import MarkovProcess, NonTerminal, State
from rl.distribution import Distribution, Constant, Poisson, Choose, Gaussian


class OrderBookProcess(MarkovProcess[OrderBook]):
    @abstractmethod
    def get_initial_state_distribution(self) -> Distribution[NonTerminal[OrderBook]]:
        '''Gives a distribution over initial order books'''


class MyOrderBookProcess(OrderBookProcess):
    
    def get_initial_state_distribution(self) -> Distribution[NonTerminal[OrderBook]]:
        bids_list = [(-1, 10), (-2, 15), (-3, 20), (-4, 20), (-5, 25), (-6, 25), (-7, 30), (-8, 30), (-9, 30)]
        asks_list = [(1, 10), (2, 15), (3, 20), (4, 20), (5, 25), (6, 25), (7, 30), (8, 30), (9, 30)]
        initial_bids = [DollarsAndShares(d, s) for d,s in bids_list]
        initial_asks = [DollarsAndShares(d, s) for d,s in asks_list]
        initial_order_book = OrderBook(initial_bids, initial_asks)
        return Constant(NonTerminal(initial_order_book))
    
    def transition(self, state: NonTerminal[OrderBook]) -> Distribution[State[OrderBook]]:
        '''Model: mid changes as a normal; bids/asks repopulate around mid like poisson'''
        possible_next_ob = []
        for _ in range(10):
            ob = state.state
            new_mid = round(ob.mid_price() + Gaussian(0,1).sample())
            new_bids_list = [(new_mid-1, Poisson(10).sample()), (new_mid-2, Poisson(15).sample()),
                                (new_mid-3, Poisson(20).sample()), (new_mid-4, Poisson(20).sample()),
                                (new_mid-5, Poisson(25).sample()), (new_mid-6, Poisson(25).sample()),
                                (new_mid-7, Poisson(30).sample()), (new_mid-8, Poisson(30).sample()),
                                (new_mid-9, Poisson(30).sample()) ] 
            new_asks_list = [(new_mid+1, Poisson(10).sample()), (new_mid+2, Poisson(15).sample()),
                                (new_mid+3, Poisson(20).sample()), (new_mid+4, Poisson(20).sample()),
                                (new_mid+5, Poisson(25).sample()), (new_mid+6, Poisson(25).sample()),
                                (new_mid+7, Poisson(30).sample()), (new_mid+8, Poisson(30).sample()),
                                (new_mid+9, Poisson(30).sample()) ] 
            new_bids = [DollarsAndShares(d, s) for d,s in new_bids_list]
            new_asks = [DollarsAndShares(d, s) for d,s in new_asks_list]
            new_order_book = OrderBook(new_bids, new_asks)

            possible_next_ob.append(NonTerminal(new_order_book))
        
        return Choose(possible_next_ob)



if __name__ == '__main__':
    ob_process = MyOrderBookProcess()
    initial_dist = ob_process.get_initial_state_distribution()
    sim = ob_process.simulate(initial_dist)
    for ob_state in sim:
        ob = ob_state.state
        ob.display_order_book()

