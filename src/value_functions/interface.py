from abc import ABC
from typing import Any


class ValueFunction(ABC):
    def predict(self):
        raise NotImplementedError

    def argmax(self) -> Any:
        raise NotImplementedError

    def loss_function(self) -> float:
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
