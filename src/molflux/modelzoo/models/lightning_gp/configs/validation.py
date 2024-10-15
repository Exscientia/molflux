from typing import Literal, Union

from pydantic.v1 import Field, dataclasses

from molflux.modelzoo.models.lightning.config import ConfigDict


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class ValidationConfig:
    name: str
    fit_end_fantasy: bool = True


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class NoValidationConfig(ValidationConfig):
    name: Literal["NoValidation"] = "NoValidation"


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class InnerTrainValidationConfig(ValidationConfig):
    name: Literal["InnerTrainValidation"] = "InnerTrainValidation"
    splitting_strategy_config: dict = Field(
        default_factory=lambda: {
            "name": "shuffle_split",
            "presets": {
                "train_fraction": 0.8,
                "validation_fraction": 0.2,
                "test_fraction": 0.0,
            },
        },
    )
    groups_column: str | None = None
    target_column: str | None = None


ValidationConfigT = Union[
    NoValidationConfig,
    InnerTrainValidationConfig,
]
