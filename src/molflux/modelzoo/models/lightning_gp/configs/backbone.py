from typing import Any, Literal, Union

from pydantic.v1 import Field, dataclasses

from molflux.modelzoo.models.lightning.config import ConfigDict


@dataclasses.dataclass(config=ConfigDict)
class BackboneConfig:
    name: str


@dataclasses.dataclass(config=ConfigDict)
class NoBackboneConfig(BackboneConfig):
    name: Literal["NoBackbone"] = "NoBackbone"


@dataclasses.dataclass(config=ConfigDict)
class LayerConfig:
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


@dataclasses.dataclass(config=ConfigDict)
class NNBackboneConfig(BackboneConfig):
    name: Literal["NNBackbone"] = "NNBackbone"
    layer_configs: list[LayerConfig] = Field(default_factory=list)


BackboneConfigT = Union[
    NoBackboneConfig,
    NNBackboneConfig,
]
