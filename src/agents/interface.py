from abc import ABC
from typing import Any

from custom_typings import Portfolio, Scalar
from gym.utils import seeding

State = Action = Any


class Agent(ABC):
    def __init__(self):
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
            cash=100_000_000,
        )

    def struct_state(self, observation: Any) -> State:
        raise NotImplementedError

    def argmax_q(self, state: State) -> Scalar:
        raise NotImplementedError

    def act(self, state: State, **kwargs) -> Action:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
