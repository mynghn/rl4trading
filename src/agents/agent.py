from abc import ABC, abstractmethod
from typing import Sequence, Union

from typings import Portfolio


class Agent(ABC):
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    @abstractmethod
    def action(self, state: Sequence[Union[int, float]], **kwargs) -> Union[int, float]:
        raise NotImplementedError
