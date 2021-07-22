from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict


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

    capital: int = 100_000_000


@dataclass
class PortfolioHistory:
    celltrion: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))
    hyundai_motors: DefaultDict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    kakao: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))
    kospi: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))
    lg_chem: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))
    lg_hnh: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))
    naver: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))
    samsung_bio: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))
    samsung_elec: DefaultDict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    samsung_elec2: DefaultDict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    samsung_sdi: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))
    sk_hynix: DefaultDict[int, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class OpenPriceBook:
    celltrion: int
    hyundai_motors: int
    kakao: int
    kospi: int
    lg_chem: int
    lg_hnh: int
    naver: int
    samsung_bio: int
    samsung_elec: int
    samsung_elec2: int
    samsung_sdi: int
    sk_hynix: int
