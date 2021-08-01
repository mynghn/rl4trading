import copy
import datetime
from typing import Any, Dict, Tuple

import gym
import numpy as np
from custom_typings import Portfolio, Price, Stock
from gym import spaces as S
from gym.utils import seeding


class StockMarket(gym.Env):
    stock_list = (
        "celltrion",
        "hyundai_motors",
        "kakao",
        "kospi",
        "lg_chem",
        "lg_hnh",
        "naver",
        "samsung_bio",
        "samsung_elec",
        "samsung_elec2",
        "samsung_sdi",
        "sk_hynix",
    )

    def __init__(
        self,
        price_book: Dict[Stock, Dict[datetime.date, Price]],
        start_date: datetime.date,
        end_date: datetime.date,
    ):
        self.action_space = S.Box(low=-np.inf, high=np.inf, shape=(12,))
        self.observation_space = S.Box(low=0, high=np.inf, shape=(552,))

        self.price_book = price_book

        self.start_date = start_date
        self.end_date = end_date

        self.random_seed = self.seed()
        self.start_point_list = list(
            self.np_random.choice(
                (self.end_date - self.start_date).days + 1,
                size=(self.end_date - self.start_date).days + 1,
                replace=False,
            )
        )
        self.date = self.start_date + datetime.timedelta(
            days=self.start_point_list.pop()
        )
        self.t = 0

    def get_portfolio_value(self, portfolio: Portfolio, date: datetime.date) -> float:
        value = portfolio["cash"]
        for stock in self.stock_list:
            price = self.price_book[stock][date]
            q = portfolio[stock]

            value += price * q

        return value

    def get_observations(self, timestep: int) -> Any:
        pass

    def step(
        self, action: np.ndarray[np.float32], portfolio: Portfolio
    ) -> Tuple[object, float, bool, Dict]:
        # 0. Increase timestep
        self.t += 1
        self.date += datetime.timedelta(days=1)

        sells = [idx for idx in range(len(action)) if action[idx] < 0]
        buys = [idx for idx in range(len(action)) if action[idx] > 0]
        portfolio_before = copy.deepcopy(portfolio)

        # 1. Sell First
        for idx in sells:
            stock = self.stock_list[idx]
            open_price = self.price_book[stock][self.date]
            to_sell = action[idx]

            assert portfolio_before[stock] >= to_sell, "Not enough stock in hand"

            portfolio[stock] -= to_sell
            portfolio["cash"] += open_price * to_sell

        # 2. Then buy
        for idx in buys:
            stock = self.stock_list[idx]
            open_price = self.price_book[stock][self.date]
            to_buy = action[idx]

            assert portfolio["cash"] >= open_price * to_buy, "Not enough cash in hand"

            portfolio["cash"] -= open_price * to_buy
            portfolio[stock] += to_buy

        reward = self.get_portfolio_value(
            portfolio=portfolio, date=self.date
        ) - self.get_portfolio_value(
            portfolio=portfolio_before, date=self.date - datetime.timedelta(days=1)
        )

        if self.t >= 10:
            done = True
        else:
            done = False

        obs = self.get_observations(timestep=self.t)

        return obs, reward, done, {}

    def reset(self) -> bool:
        if self.start_point_list:
            self.date = self.start_date + datetime.timedelta(
                days=self.start_point_list.pop()
            )
            self.t = 0
            return True
        else:
            # all start points used up
            return False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
