from pydantic.v1 import Field, dataclasses, root_validator

from molflux.modelzoo.models.lightning.config import (
    ConfigDict,
    LightningConfig,
)
from molflux.modelzoo.models.lightning_gp.configs.backbone import (
    BackboneConfigT,
    NoBackboneConfig,
)
from molflux.modelzoo.models.lightning_gp.configs.kernel import (
    KernelConfigT,
    RBFKernelConfig,
)
from molflux.modelzoo.models.lightning_gp.configs.likelihood import LikelihoodConfig
from molflux.modelzoo.models.lightning_gp.configs.mean import MeanConfig
from molflux.modelzoo.models.lightning_gp.configs.validation import (
    InnerTrainValidationConfig,
    ValidationConfigT,
)


@dataclasses.dataclass(config=ConfigDict)
class GPConfig(LightningConfig):
    mean_config: MeanConfig = Field(default_factory=MeanConfig)
    likelihood_config: LikelihoodConfig = Field(default_factory=LikelihoodConfig)
    kernel_config: KernelConfigT = Field(
        default_factory=RBFKernelConfig,
        discriminator="name",
    )
    num_tasks: int = 1
    validation_config: ValidationConfigT = Field(
        default_factory=InnerTrainValidationConfig,
        discriminator="name",
    )
    backbone_config: BackboneConfigT = Field(
        default_factory=NoBackboneConfig,
        discriminator="name",
    )

    @root_validator()
    def check_single_task(cls, values: dict) -> dict:
        assert values["num_tasks"] == 1, ValueError(
            "lightning_gp_regressor only supports single task models!",
        )
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def disable_sanity_val_steps(cls, values: dict) -> dict:
        trainer = values["trainer"]
        trainer.num_sanity_val_steps = 0
        return values
