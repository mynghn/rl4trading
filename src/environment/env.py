import datetime
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame
from typings import Action, Reward


class Environment(ABC):
    @abstractmethod
    def observed(
        self, statrt_date: datetime.date, end_date: datetime.date, **kwargs
    ) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def interact(self, action: Action, **kwargs) -> Reward:
        raise NotImplementedError
