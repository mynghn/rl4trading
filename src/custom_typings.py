from typing import Literal, TypedDict, Union

Scalar = Union[int, float]

Stock = Union[
    Literal["celltrion"],
    Literal["hyundai_motors"],
    Literal["kakao"],
    Literal["kospi"],
    Literal["lg_chem"],
    Literal["lg_hnh"],
    Literal["naver"],
    Literal["samsung_bio"],
    Literal["samsung_elec"],
    Literal["samsung_elec2"],
    Literal["samsung_sdi"],
    Literal["sk_hynix"],
]


class Portfolio(TypedDict):
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

    cash: Scalar
