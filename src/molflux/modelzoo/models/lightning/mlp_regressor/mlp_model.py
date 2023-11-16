from typing import Any, Type

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.models.lightning.mlp_regressor.mlp_config import MLPConfig
from molflux.modelzoo.models.lightning.mlp_regressor.mlp_datamodule import (
    MLPDataModule,
)
from molflux.modelzoo.models.lightning.mlp_regressor.mlp_module import MLPModule
from molflux.modelzoo.models.lightning.model import LightningModelBase


class LightningMLPRegressor(LightningModelBase[MLPConfig]):
    """Model class for MLP model"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="MLP implementation via lightning",
            config_description="",
        )

    @property
    def _datamodule_builder(self) -> Type[MLPDataModule]:
        return MLPDataModule

    @property
    def _config_builder(self) -> Type[MLPConfig]:
        return MLPConfig

    def _instantiate_module(self) -> MLPModule:
        return MLPModule(model_config=self.model_config)
