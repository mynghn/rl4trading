from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame

from custom_typings import Action, Portfolio, Scalar, State


class Agent(ABC):
    def __init__(self):
        self.portfolio = Portfolio()
        self.rewards = []

    @abstractmethod
    def observe(self, env: Environment, timestep: datetime.date) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def observation2state(self, observation: DataFrame) -> State:
        raise NotImplementedError

    @abstractmethod
    def policy(self, state: State, **kwargs) -> Action:
        raise NotImplementedError

    def rewarded(self, reward: Scalar):
        self.rewards.append(reward)

    def Return(self, discount_factor: Scalar = 1) -> Scalar:
        return sum([r * (discount_factor ** idx) for idx, r in enumerate(self.rewards)])

    def Q(self, state: State, action: Action) -> Scalar:
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


class Environment(ABC):
    @abstractmethod
    def observed(self, start: datetime.date, end: datetime.date, **kwargs) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def interact(self, action: Action, timestep: datetime.date, **kwargs) -> Scalar:
        raise NotImplementedError

    @abstractmethod
    def episode(
        self, agent: Agent, start: datetime.date, end: datetime.date, **kwargs
    ) -> Scalar:
        raise NotImplementedError
