from pydantic.v1 import Field, dataclasses

from molflux.modelzoo.models.lightning.config import ConfigDict
from molflux.modelzoo.models.lightning_gp.configs.constraint import (
    ConstraintConfigT,
    PositiveConstraintConfig,
)


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class LikelihoodConfig:
    noise_constraint: ConstraintConfigT = Field(
        default_factory=PositiveConstraintConfig,
        discriminator="name",
    )
