from typing import Literal, Union

from pydantic.v1 import dataclasses

from molflux.modelzoo.models.lightning.config import ConfigDict


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class IntervalConstraintConfig:
    name: Literal["Interval"] = "Interval"
    lower_bound: float
    upper_bound: float


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class PositiveConstraintConfig:
    name: Literal["Positive"] = "Positive"


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class GreaterThanConstraintConfig:
    lower_bound: float
    name: Literal["GreaterThan"] = "GreaterThan"


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class LessThanConstraintConfig:
    upper_bound: float
    name: Literal["LessThan"] = "LessThan"


ConstraintConfigT = Union[
    IntervalConstraintConfig,
    PositiveConstraintConfig,
    GreaterThanConstraintConfig,
    LessThanConstraintConfig,
]
