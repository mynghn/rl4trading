from abc import ABC, abstractmethod
from datetime import datetime

from environment.env import Environment
from pyspark.sql.dataframe import DataFrame
from typings import Action, Numeric, Portfolio, Scalar, State


class Agent(ABC):
    def __init__(self):
        self.portfolio = Portfolio()
        self.rewards = []

    @abstractmethod
    def observe(self, environment: Environment, timestep: datetime.date) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def observation2state(
        self, observation: DataFrame, timestep: datetime.date
    ) -> State:
        raise NotImplementedError

    @abstractmethod
    def policy(self, state: State, **kwargs) -> Action:
        raise NotImplementedError

    def rewarded(self, reward: Numeric):
        self.rewards.append(reward)

    def Return(self, discount_factor: Scalar = 1) -> Scalar:
        return sum([r * (discount_factor ** idx) for idx, r in enumerate(self.rewards)])

    def Q(self, state: State, action: Action) -> Numeric:
        pass

    def sell(self, stock: str, quantity: int, price: int):
        quantity_now = getattr(self.portfolio, stock)
        assert quantity_now >= quantity

        setattr(self.portfolio, stock, quantity_now - quantity)
        self.portfolio.capital += quantity * price

    def buy(self, stock: str, quantity: int, price: int):
        assert self.portfolio.capital >= quantity * price

        self.portfolio.capital -= quantity * price
        setattr(self.portfolio, stock, getattr(self.portfolio, stock) + quantity)
