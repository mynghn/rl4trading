import os
from datetime import datetime
from typing import Dict, Sequence

from custom_typings import STOCK_LIST, Action, Price, Return, Reward, Stock
from interface import Agent, Environment
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, lit


class StockMarket(Environment):
    def __init__(self, spark: SparkSession, data_path: str):
        self.spark = spark
        self.data_path = data_path

    @property
    def datamart(self) -> Dict[Stock, DataFrame]:
        return {
            "celltrion": self.spark.read.csv(
                os.path.join(self.data_path, "Celltrion.csv"), header=True
            ),
            "hyundai_motors": self.spark.read.csv(
                os.path.join(self.data_path, "HyundaiMotor.csv"), header=True
            ),
            "kakao": self.spark.read.csv(
                os.path.join(self.data_path, "Kakao.csv"), header=True
            ),
            "kospi": self.spark.read.csv(
                os.path.join(self.data_path, "KOSPI.csv"), header=True
            ),
            "lg_chem": self.spark.read.csv(
                os.path.join(self.data_path, "LGChemical.csv"), header=True
            ),
            "lg_hnh": self.spark.read.csv(
                os.path.join(self.data_path, "LGH&H.csv"), header=True
            ),
            "naver": self.spark.read.csv(
                os.path.join(self.data_path, "NAVER.csv"), header=True
            ),
            "samsung_bio": self.spark.read.csv(
                os.path.join(self.data_path, "SamsungBiologics.csv"), header=True
            ),
            "samsung_elec": self.spark.read.csv(
                os.path.join(self.data_path, "SamsungElectronics.csv"), header=True
            ),
            "samsung_elec2": self.spark.read.csv(
                os.path.join(self.data_path, "SamsungElectronics2.csv"), header=True
            ),
            "samsung_sdi": self.spark.read.csv(
                os.path.join(self.data_path, "SamsungSDI.csv"), header=True
            ),
            "sk_hynix": self.spark.read.csv(
                os.path.join(self.data_path, "SKhynix.csv"), header=True
            ),
        }

    def observed(
        self,
        stocks: Sequence[Stock],
        start: datetime.date,
        end: datetime.date,
    ) -> DataFrame:
        stock = stocks.pop()
        df = self.datamart[stock].withColumn()
        unioned = df.withColumn(lit(stock).alias("Stock"))
        for stock in stocks:
            df = self.datamart[stock]
            unioned = unioned.union(df.withColumn(lit(stock).alias("Stock")))

        return (
            unioned.filter(col("Date") >= start.isoformat())
            .filter(col("Date") <= end.isoformat())
            .orderBy("stock", "Date")
        )

    def get_open_price(self, stock, date: datetime.date) -> Price:
        return self.datamart[stock].filter(col("Date") == date.isoformat()).first().Open

    def interact(self, action: Action, timestep: datetime.date) -> Reward:
        reward = 0
        _bought = 0
        for stock in STOCK_LIST:
            open_price = self.get_open_price(stock=stock, date=timestep)
            order = getattr(action, stock)

            reward += order.sell * open_price
            reward -= order.buy * open_price

            _bought += order.buy * open_price

        assert _bought > 10_000_000, "Rule Violation"

        return reward

    def episode(
        self,
        agent: Agent,
        start: datetime.date,
        end: datetime.date,
    ) -> Return:
        t = start
        while t <= end:
            # 1. Agent: Observe environment
            observation = agent.observe(env=self, timestep=t)

            # 2. Agent: Struct a state from observation
            state = agent.observation2state(observation=observation)

            # 3. Agent: Derive an action w.r.t. structed state
            action = agent.policy(state=state)

            # 4. Env: Calculate reward
            reward = self.interact(action=action, timestep=t)

            # 5. Agent: Do actual sell & buy
            for stock in STOCK_LIST:
                open_price = self.get_open_price(stock=stock, date=t)
                order = getattr(action, stock)

                agent.sell(stock=stock, quantity=order.sell, price=open_price)
                agent.buy(stock=stock, quantity=order.buy, price=open_price)

            # 6. Agent: Receive reward
            agent.rewarded(reward=reward)

            # 7. Increase timestep
            t += datetime.timedelta(days=1)

        return agent.Return()
