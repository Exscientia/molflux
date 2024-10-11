from pydantic.v1 import dataclasses

from molflux.modelzoo.models.lightning.config import ConfigDict


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class MeanConfig:
    fixed: bool = False
    constant_value: float | None = None
