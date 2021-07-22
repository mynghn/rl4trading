from abc import ABC, abstractmethod
from typing import Sequence, Union

from typings import Portfolio


class Agent(ABC):
    def __init__(self):
        self.portfolio = Portfolio()

    @abstractmethod
    def action(self, state: Sequence[Union[int, float]], **kwargs) -> Union[int, float]:
        raise NotImplementedError

    @abstractmethod
    def sell(self, stock: int, quantity: int, price: int, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def buy(self, stock: int, quantity: int, price: int, **kwargs):
        raise NotImplementedError
