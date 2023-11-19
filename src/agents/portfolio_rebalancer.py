import copy
from collections import defaultdict

import numpy as np
from custom_typings import Portfolio
from envs.market import StockMarket
from gym import spaces as S
from value_functions.interface import ValueFunction

from agents.interface import Agent


class PortfolioRebalancer(Agent):
    def __init__(self, Q: ValueFunction, epsilon: float):
        super().__init__()

        self.action_space = S.Box(low=0, high=1, shape=(13,))
        self.Q = Q
        self.inventory = defaultdict(lambda: defaultdict(int))

        self.epsilon = epsilon
        self.random_seed = self.seed()

    def struct_state(self, observation: np.ndarray) -> np.ndarray:
        state = observation

        # portfolio value
        portfolio_value = self.portfolio["cash"]
        for idx, stock in enumerate(StockMarket.stock_list):
            price = observation[idx]
            q = self.portfolio[stock]

            # while assessing portfolio value -> add current stock inventory info
            state = np.append(state, q)

            # while assessing portfolio value -> add current ror by stock
            if q > 0:
                ror = price * q / sum([k * v for k, v in self.inventory[stock].items()])
            else:
                ror = 1
            state = np.append(state, ror)

            portfolio_value += price * q

        state = np.append(state, portfolio_value)

        return state

    def argmax_q(self, state: np.ndarray) -> np.ndarray:
        return self.Q.argmax(state=state)

    def act(self, state: np.ndarray) -> np.ndarray:
        # epsilon greedy
        if self.np_random.rand() < self.epsilon:
            # explore
            action = self.action_space.sample()
        else:
            # exploit
            action = self.argmax_q(state=state)

        return action

    def to_market_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # Translate into StockMarket action protocol
        inventory_value_book = {}
        price_book = {}
        total_portfolio_value = self.portfolio["cash"]
        for idx, stock in enumerate(StockMarket.stock_list):
            price = state[idx]
            q = self.portfolio[stock]

            inventory_value_book[stock] = price * q
            price_book[stock] = price
            total_portfolio_value += price * q

        stock_market_action = []
        for idx, stock in enumerate(StockMarket.stock_list):
            holding_value = inventory_value_book[stock]
            adjusted_value = total_portfolio_value * action[idx]
            if adjusted_value > holding_value:
                # buy
                to_buy = int((adjusted_value - holding_value) / price_book[stock])
                stock_market_action.append(to_buy)

                self.inventory[stock][price_book[stock]] += to_buy
            elif adjusted_value < holding_value:
                # sell
                to_sell = int((holding_value - adjusted_value) / price_book[stock])
                stock_market_action.append(-to_sell)

                to_deduct = to_sell
                for p, q in sorted(
                    copy.deepcopy(self.inventory[stock]).items(),
                    key=lambda item: item[0],
                ):
                    if q > to_deduct:
                        self.inventory[stock][p] -= to_deduct
                        break
                    elif q == to_deduct:
                        del self.inventory[stock][p]
                        break
                    else:
                        del self.inventory[stock][p]
                        to_deduct -= q
            else:
                stock_market_action.append(0)

        return stock_market_action

    def reset(self):
        self.portfolio = Portfolio(
            celltrion=0,
            hyundai_motors=0,
            kakao=0,
            kospi=0,
            lg_chem=0,
            lg_hnh=0,
            naver=0,
            samsung_bio=0,
            samsung_elec=0,
            samsung_elec2=0,
            samsung_sdi=0,
            sk_hynix=0,
            cash=100_000_000,
        )
        self.inventory = defaultdict(lambda: defaultdict(int))
