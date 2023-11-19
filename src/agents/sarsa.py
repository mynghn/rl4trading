import datetime
import random
from collections import defaultdict
from math import ceil
from typing import Dict

import numpy as np
from custom_typings import Action, Portfolio, Price, Scalar, State, Stock
from environment.market import StockMarket
from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F

from interface import Agent


class Sarsa(Agent):
    def __init__(
        self,
        stock: Stock,
        num_states: int,
        num_actions: int,
        discount_factor: Scalar = 1,
        step_size: Scalar = 0.1,
        q_seed: Scalar = 1_000_000_000,
        epsilon: Scalar = 0.1,
        upper_bound: Scalar = 0.25,
        lower_bound: Scalar = 0.05,
    ) -> None:
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
            capital=100_000_000,
            value=100_000_000,
        )
        self.rewards = []

        self.stock = stock
        self.q = np.ones((num_states, num_actions)) * float(q_seed)
        self.alpha = step_size
        self.gamma = discount_factor

        self.epsilon = epsilon

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.average_purchase = 0

    def observe(self, env: StockMarket, timestep: datetime.date) -> DataFrame:
        return super().observe(env, timestep)

    def struct_state(self, observation: DataFrame) -> State:
        state = []

        data = (
            observation.withColumn(
                "date_order",
                F.row_number().over(W.partitionBy("Stock").orderBy("Date")),
            )
            .withColumn(
                "open_ma5",
                F.when(
                    F.col("date_order") >= 5,
                    F.avg("Open").over(
                        W.partitionBy("Stock")
                        .orderBy("date_order")
                        .rowsBetween(-4, W.currentRow)
                    ),
                ).otherwise(F.lit(None)),
            )
            .withColumn(
                "open_ma60",
                F.when(
                    F.col("date_order") >= 60,
                    F.avg("Open").over(
                        W.partitionBy("Stock")
                        .orderBy("date_order")
                        .rowsBetween(-59, W.currentRow)
                    ),
                ).otherwise(F.lit(None)),
            )
            .withColumn(
                "volume_ma5",
                F.when(
                    F.col("date_order") >= 5,
                    F.avg("Volume").over(
                        W.partitionBy("Stock")
                        .orderBy("date_order")
                        .rowsBetween(-4, W.currentRow)
                    ),
                ).otherwise(F.lit(None)),
            )
            .withColumn(
                "volume_ma60",
                F.when(
                    F.col("date_order") >= 60,
                    F.avg("Volume").over(
                        W.partitionBy("Stock")
                        .orderBy("date_order")
                        .rowsBetween(-59, W.currentRow)
                    ),
                ).otherwise(F.lit(None)),
            )
            .withColumn(
                "is_high_over_10%",
                F.when(F.col("High") > F.col("Open") * 1.1, F.lit(True)).otherwise(
                    F.lit(False)
                ),
            )
            .withColumn(
                "is_low_below_10%",
                F.when(F.col("Low") < F.col("Open") * 0.9, F.lit(True)).otherwise(
                    F.lit(False)
                ),
            )
        )

        stock_data = data.filter(F.col("Stock") == self.stock)

        open_price = float(
            stock_data.filter(F.col("Date") == F.max("Date")).first().Open
        )
        # 1. off boundary
        if open_price > self.average_purchase * (1 + self.upper_bound):
            state.append(2)
        elif open_price < self.average_purchase * (1 - self.lower_bound):
            state.append(1)
        else:
            state.append(0)

        # 2. ma5 & ma20 golden/dead cross
        ma_data = (
            stock_data.orderBy("Date", ascending=False)
            .select("open_ma5", "open_ma60", "volume_ma5", "volume_ma60")
            .limit(4)
            .collect()
        )

        return super().struct_state(observation)

    def hash_state(self, state: State) -> int:
        return int("".join([str(i) for i in state]), base=2)

    # epsilon-greedy
    def policy(self, state: State, open_prices: Dict[Stock, Price]) -> Action:
        state_index = self.hash_state(state=state)
        if np.random.rand() < self.epsilon:
            # explore
            action_index = random.choice(range(self.q.shape[1]))
        else:
            # exploit
            action_index = random.choice(
                np.argwhere(
                    self.q[state_index] == np.amax(self.q[state_index])
                ).flatten()
            )

        # Struct an Action
        action = defaultdict(lambda: defaultdict(int))

        # Avoid Rule Violation
        if self.stock in open_prices.keys():
            action[self.stock]["buy"] = ceil(10_000_000 / open_prices[self.stock])
        else:
            # stock not in market
            random_pick = random.choice(open_prices.key())
            action[random_pick]["buy"] = ceil(10_000_000 / open_prices[random_pick])

            subtotal = open_prices[random_pick] * action[random_pick]["buy"]

            if subtotal > self.portfolio["capital"]:
                assert self.portfolio["value"] >= subtotal, "Broke"
                action[self.stock]["sell"] = ceil(
                    (subtotal - self.portfolio["capital"]) / open_prices[self.stock]
                )

            return action

        # Real action part
        num_ratio_choice = self.q.shape[1] // 2
        if self.q.shape[1] % 2 == 0:
            if action_index < self.q.shape[1] / 2:
                inventory_available = (
                    self.portfolio[self.stock] - action[self.stock]["sell"]
                )
                action[self.stock]["sell"] += round(
                    inventory_available
                    * ((num_ratio_choice - action_index) / num_ratio_choice)
                )
            else:
                budget_available = max(
                    (
                        self.portfolio["capital"]
                        - action[self.stock]["buy"] * open_prices[self.stock]
                        - 12_000_000
                    ),
                    0,
                )
                action[self.stock]["buy"] += round(
                    budget_available
                    * ((action_index + 1 - num_ratio_choice) / num_ratio_choice)
                    / open_prices[self.stock]
                )
        else:
            if action_index < num_ratio_choice:
                inventory_available = (
                    self.portfolio[self.stock] - action[self.stock]["sell"]
                )
                action[self.stock]["sell"] = round(
                    inventory_available
                    * ((num_ratio_choice - action_index) / num_ratio_choice)
                )
            elif action_index > num_ratio_choice:
                budget_available = max(
                    (
                        self.portfolio["capital"]
                        - action[self.stock]["buy"] * open_prices[self.stock]
                        - 12_000_000
                    ),
                    0,
                )
                action[self.stock]["buy"] += round(
                    budget_available
                    * ((action_index - num_ratio_choice) / num_ratio_choice)
                    / open_prices[self.stock]
                )
            else:
                pass

        self.average_purchase = (
            self.average_purchase * self.portfolio[self.stock]
            + open_prices[self.stock] * action[self.stock]["buy"]
        ) / (self.portfolio[self.stock] + action[self.stock]["buy"])

        return action
