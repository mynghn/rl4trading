import copy
import datetime
from collections import defaultdict
from typing import Dict, Literal, Tuple, Union

import gym
import numpy as np
from custom_typings import Portfolio, Stock
from gym import spaces as S
from gym.utils import seeding
from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F


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
        datamart: Dict[Stock, DataFrame],
        start_date: datetime.date,
        end_date: datetime.date,
        episode_length: int,
    ):
        self.action_space = S.Box(low=-np.inf, high=np.inf, shape=(12,))
        self.observation_space = S.Box(low=0, high=np.inf, shape=(492,))

        self.datamart = datamart
        self.cache = defaultdict(lambda: defaultdict(dict))

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
        self.episode_start = self.start_date + datetime.timedelta(
            days=self.start_point_list.pop()
        )
        self.t = 1

        self.T = episode_length

    def get_data(self, stock: Stock, date: datetime.date, column: str) -> float:
        if self.cache[stock][date].get(column):
            return self.cache[stock][date][column]
        else:
            if "MA" in column and column not in self.datamart[stock].columns:
                parsed = column.split("_")
                og_col = parsed[0]
                ma_type = parsed[1][0]
                window_size = int(parsed[1][3:])

                self.compute_ma(
                    stock=stock, column=og_col, window_size=window_size, type=ma_type
                )

            data = float(
                self.datamart[stock]
                .filter(F.col("Date") == date.isoformat())
                .select(column)
                .first()
                .asDict()[column]
            )
            self.cache[stock][date][column] = data
            return data

    def compute_ma(
        self,
        stock: Stock,
        column: str,
        window_size: int,
        type: Union[Literal["S"], Literal["C"], Literal["E"]] = "S",
    ):
        df = self.datamart[stock]

        if "date_order" not in df.columns:
            df = df.withColumn(
                "date_order",
                F.row_number().over(W.orderBy("Date")),
            )

        if type == "S":
            self.datamart[stock] = df.withColumn(
                f"{column}_SMA{window_size}",
                F.when(
                    F.col("date_order") >= window_size,
                    F.avg(column).over(
                        W.orderBy("date_order").rowsBetween(
                            -window_size + 1, W.currentRow
                        )
                    ),
                ).otherwise(F.lit(None)),
            )
        else:
            raise ValueError("Cumulative & Exponential MA not implemented yet")

    def get_portfolio_value(self, portfolio: Portfolio, date: datetime.date) -> float:
        value = portfolio["cash"]
        for stock in self.stock_list:
            price = self.get_data(stock=stock, date=date, column="Open")
            q = portfolio[stock]

            value += price * q

        return value

    def emit_observation(self, timestep: int) -> np.ndarray[float]:
        obs = []
        date = self.episode_start + datetime.timedelta(days=timestep - 1)

        # 1. (t+1) step's open prices -> 12D
        for stock in self.stock_list:
            obs.append(self.get_data(stock=stock, date=date, column="Open"))

        # 2. (t) ~ (t-4) step's open/close/high/low/volume -> 12 * 5 * 5 = 300D
        for stock in self.stock_list:
            for i in range(1, 6):
                obs += [
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=i),
                        column="Open",
                    ),
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=i),
                        column="Close",
                    ),
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=i),
                        column="High",
                    ),
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=i),
                        column="Low",
                    ),
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=i),
                        column="Volume",
                    ),
                ]

        # 3. 10/20/60 Day SMA -> 3 * 12 * 5 = 180D
        for window_size in (10, 20, 60):
            for stock in self.stock_list:
                obs += [
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=1),
                        column=f"Open_SMA{window_size}",
                    ),
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=1),
                        column=f"Close_SMA{window_size}",
                    ),
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=1),
                        column=f"High_SMA{window_size}",
                    ),
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=1),
                        column=f"Low_SMA{window_size}",
                    ),
                    self.get_data(
                        stock=stock,
                        date=date - datetime.timedelta(days=1),
                        column=f"Volume_SMA{window_size}",
                    ),
                ]

        return np.array(obs)

    def step(
        self, action: np.ndarray[float], portfolio: Portfolio
    ) -> Tuple[np.ndarray[float], float, bool, Dict]:
        sells = [idx for idx in range(len(action)) if action[idx] < 0]
        buys = [idx for idx in range(len(action)) if action[idx] > 0]
        portfolio_before = copy.deepcopy(portfolio)

        date = self.episode_start + datetime.timedelta(days=self.t - 1)
        # Sell First
        for idx in sells:
            stock = self.stock_list[idx]
            open_price = self.get_data(stock=stock, date=date, column="Open")
            to_sell = action[idx]

            assert portfolio_before[stock] >= to_sell, "Not enough stock in hand"

            portfolio[stock] -= to_sell
            portfolio["cash"] += open_price * to_sell

        # Then buy
        for idx in buys:
            stock = self.stock_list[idx]
            open_price = self.get_data(stock=stock, date=date, column="Open")
            to_buy = action[idx]

            assert portfolio["cash"] >= open_price * to_buy, "Not enough cash in hand"

            portfolio["cash"] -= open_price * to_buy
            portfolio[stock] += to_buy

        # Increase timestep
        self.t += 1

        # reward = portfolio value delta
        reward = self.get_portfolio_value(
            portfolio=portfolio,
            date=self.episode_start + datetime.timedelta(days=self.t - 1),
        ) - self.get_portfolio_value(
            portfolio=portfolio_before,
            date=self.episode_start + datetime.timedelta(days=self.t - 2),
        )

        # done
        if self.t >= self.T:
            done = True
        else:
            done = False

        # observation
        obs = self.emit_observation(timestep=self.t)

        # info
        info = {}

        return obs, reward, done, info

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
