from abc import ABC, abstractmethod
from typing import Sequence, Union


class Agent(ABC):
    @abstractmethod
    def action(state: Sequence[Union[int, float]], **kwargs) -> Union[int, float]:
        raise NotImplementedError
