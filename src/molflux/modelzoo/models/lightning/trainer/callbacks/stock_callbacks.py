import logging
from typing import Any

try:
    import torch
    from lightning.pytorch import LightningModule, Trainer, callbacks
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning", e) from e


logger = logging.getLogger(__name__)


class ModelCheckpointApply(callbacks.ModelCheckpoint):
    """Model checkpointing callback which applies the best module checkpoint at the end of fitting"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.best_model_path is not None:
            best_ckpt = torch.load(self.best_model_path, map_location="cpu")
            pl_module.load_state_dict(best_ckpt["state_dict"])
            logger.warning(
                f"Using best model from ckpt_path: {self.best_model_path}.\n",
            )


AVAILABLE_CALLBACKS = {
    "BackboneFinetuning": callbacks.BackboneFinetuning,
    "BaseFinetuning": callbacks.BaseFinetuning,
    "BasePredictionWriter": callbacks.BasePredictionWriter,
    "DeviceStatsMonitor": callbacks.DeviceStatsMonitor,
    "EarlyStopping": callbacks.EarlyStopping,
    "GradientAccumulationScheduler": callbacks.GradientAccumulationScheduler,
    "LearningRateMonitor": callbacks.LearningRateMonitor,
    "ModelCheckpoint": callbacks.ModelCheckpoint,
    "ModelCheckpointApply": ModelCheckpointApply,
    "ModelPruning": callbacks.ModelPruning,
    "ModelSummary": callbacks.ModelSummary,
    "RichModelSummary": callbacks.RichModelSummary,
    "StochasticWeightAveraging": callbacks.StochasticWeightAveraging,
    "Timer": callbacks.Timer,
}
