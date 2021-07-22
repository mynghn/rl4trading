from dataclasses import dataclass


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
