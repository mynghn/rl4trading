from dataclasses import dataclass, field
from typing import List, Sequence, Union

Scalar = Union[int, float]

Reward = Scalar

State = Sequence[Union[int, float]]


@dataclass
class Portfolio:
    celltrion: int = 0
    hyundai_motors: int = 0
    kakao: int = 0
    kospi: int = 0
    lg_chem: int = 0
    lg_hnh: int = 0
    naver: int = 0
    samsung_bio: int = 0
    samsung_elec: int = 0
    samsung_elec2: int = 0
    samsung_sdi: int = 0
    sk_hynix: int = 0

    capital: Scalar = 100_000_000


@dataclass
class Order:
    buy: int = 0
    sell: int = 0


@dataclass
class Action:
    celltrion: Order = Order()
    hyundai_motors: Order = Order()
    kakao: Order = Order()
    kospi: Order = Order()
    lg_chem: Order = Order()
    lg_hnh: Order = Order()
    naver: Order = Order()
    samsung_bio: Order = Order()
    samsung_elec: Order = Order()
    samsung_elec2: Order = Order()
    samsung_sdi: Order = Order()
    sk_hynix: Order = Order()


Price = Scalar


@dataclass
class Inventory:
    celltrion: List[Price] = field(default_factory=list)
    hyundai_motors: List[Price] = field(default_factory=list)
    kakao: List[Price] = field(default_factory=list)
    kospi: List[Price] = field(default_factory=list)
    lg_chem: List[Price] = field(default_factory=list)
    lg_hnh: List[Price] = field(default_factory=list)
    naver: List[Price] = field(default_factory=list)
    samsung_bio: List[Price] = field(default_factory=list)
    samsung_elec: List[Price] = field(default_factory=list)
    samsung_elec2: List[Price] = field(default_factory=list)
    samsung_sdi: List[Price] = field(default_factory=list)
    sk_hynix: List[Price] = field(default_factory=list)


@dataclass
class OpenPriceBook:
    celltrion: Price
    hyundai_motors: Price
    kakao: Price
    kospi: Price
    lg_chem: Price
    lg_hnh: Price
    naver: Price
    samsung_bio: Price
    samsung_elec: Price
    samsung_elec2: Price
    samsung_sdi: Price
    sk_hynix: Price
