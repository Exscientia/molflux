import logging
import os
from typing import Any, Optional, Union

import molflux
from datasets import Dataset

try:
    import lightning.pytorch as pl
    import torch

except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning_gp", e) from e

from molflux.modelzoo.errors import NotTrainedError
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import StandardDeviationMixin
from molflux.modelzoo.models.lightning.config import (
    CompileConfig,
    DataModuleConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TransferLearningConfigBase,
)
from molflux.modelzoo.models.lightning.model import LightningModelBase
from molflux.modelzoo.models.lightning_gp.gp_config import GPConfig
from molflux.modelzoo.models.lightning_gp.gp_datamodule import GPDataModule
from molflux.modelzoo.models.lightning_gp.gp_module import GPModule
from molflux.modelzoo.typing import PredictionResult

logger = logging.getLogger(__name__)


_DESCRIPTION = """
Implementation of Gaussian Processes via Pytorch Lightning and GPyTorch.

A GP in a nutshell, is a model that takes in as training inputs all your training data.
It then passes the data through a series of mathematical operations that depend on a mean function,
a kernel (how close two data points are) and optionally any pre or post processing objects
(a feature extractor before, an added noise after etc.).

The output of this is a multivariate gaussian, for which you can compute a loss if you have the labels for the
training data. Note that the mean, kernel, added noise, feature extractors etc. can have learnable parameters, and
the whole process is differentiable, so we can (and do) do gradient descent to find the best fit for these parameters.

GPyTorch allows users to quickly define a torch compatible GP, but don't offer out of the box any training loops.
functionality in a nice set of configs, motivating combining the two. The PR implements Exact GPs, that require the
entire training set, which means the memory complexity is ~O(N^2)log N, but offers the most precise models when the
data range is within the memory limit.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
iterations : Optional[int], default: None
"""


class LightningGPRegressor(StandardDeviationMixin, LightningModelBase[GPConfig]):
    """Model class for MLP model"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    @property
    def _datamodule_builder(self) -> type[GPDataModule]:
        return GPDataModule

    @property
    def _config_builder(self) -> type[GPConfig]:
        return GPConfig

    def _instantiate_module(self) -> GPModule:
        return GPModule(model_config=self.model_config)

    def _predict(
        self,
        data: Dataset,
        datamodule_config: DataModuleConfig | dict[str, Any] | None = None,
        trainer_config: TrainerConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        return self._predict_with_std(
            data=data,
            datamodule_config=datamodule_config,
            trainer_config=trainer_config,
            **kwargs,
        )[0]

    def _predict_with_std(
        self,
        data: Dataset,
        datamodule_config: DataModuleConfig | dict[str, Any] | None = None,
        trainer_config: TrainerConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[PredictionResult, PredictionResult]:
        """Implements a single prediction with standard deviation step."""
        del kwargs

        # assume single task models
        mean_display_names, std_display_names = list(
            self._predict_with_std_display_names,
        )

        # if data is empty
        if not len(data):
            return {mean_display_names[0]: []}, {std_display_names[0]: []}

        with self.override_config(datamodule=datamodule_config, trainer=trainer_config):
            datamodule = self._instantiate_datamodule(predict_data=data)

            trainer_config = self.model_config.trainer.pass_to_trainer()
            trainer = pl.Trainer(**trainer_config)

            # Expect a list of Tensors, which may need to be overwritten for some Torch models
            batch_preds: list[tuple[torch.Tensor]] = trainer.predict(
                self.module,
                datamodule,
            )

        # assume a single batch with tuples
        means, stds = batch_preds[0][0].reshape(-1), batch_preds[0][1].reshape(-1)  # type: ignore[misc]

        output = {mean_display_names[0]: means.tolist()}
        output_std = {std_display_names[0]: stds.tolist()}

        return output, output_std

    def as_dir(self, directory: str) -> None:
        """Saves the model into a directory."""

        if hasattr(self, "module") and self.module is not None:
            ckpt = {
                "state_dict": self.state_dict,
                "train_inputs": self.module.gp_model.train_inputs,
                "train_targets": self.module.gp_model.train_targets,
            }
            torch.save(ckpt, os.path.join(directory, "module_checkpoint.ckpt"))
        else:
            raise NotTrainedError("No initialised module to save.")

    def from_dir(self, directory: str) -> None:
        """Loads a model from a directory."""

        ckpt_path = os.path.join(directory, "module_checkpoint.ckpt")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        self.module = self._instantiate_module()
        self.state_dict = checkpoint["state_dict"]
        self.module.gp_model.set_train_data(
            inputs=checkpoint["train_inputs"],
            targets=checkpoint["train_targets"],
            strict=False,
        )

    def _train_multi_data(
        self,
        train_data: dict[Optional[str], Dataset],
        validation_data: Union[
            dict[Optional[str], Dataset],
            None,
        ] = None,
        datamodule_config: Union[DataModuleConfig, dict[str, Any], None] = None,
        trainer_config: Union[TrainerConfig, dict[str, Any], None] = None,
        optimizer_config: Union[OptimizerConfig, dict[str, Any], None] = None,
        scheduler_config: Union[SchedulerConfig, dict[str, Any], None] = None,
        transfer_learning_config: Union[
            TransferLearningConfigBase,
            dict[str, Any],
            None,
        ] = None,
        compile_config: Union[CompileConfig, dict[str, Any], bool, None] = None,
        ckpt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if self.model_config.validation_config.name == "InnerTrainValidation":
            logger.info("Doing internal validation split")
            assert validation_data is None, RuntimeError(
                "Can't do internal validation split if validation_data provided",
            )

            strategy = molflux.splits.load_from_dict(
                self.model_config.validation_config.splitting_strategy_config,
            )
            split_datasets: dict[str, dict[Any, Any]] = {
                "train": {},
                "validation": {},
            }
            for k, v in train_data.items():
                split_data = next(
                    molflux.datasets.split_dataset(
                        v,
                        strategy=strategy,
                        groups_column=self.model_config.validation_config.groups_column,
                        target_column=self.model_config.validation_config.target_column,
                    ),
                )

                assert len(split_data["train"]) > 0, RuntimeError(
                    "Train split resulted in 0 points in train split",
                )
                assert len(split_data["validation"]) > 0, RuntimeError(
                    "Validation split resulted in 0 points in validation split",
                )
                assert len(split_data["test"]) == 0, RuntimeError(
                    "Test split resulted in non-zero points in test split",
                )

                logger.info(
                    f"{k} validation split: train ({len(split_data['train'])}), validation ({len(split_data['validation'])})",
                )
                split_datasets["train"][k] = split_data["train"]
                split_datasets["validation"][k] = split_data["validation"]

            super()._train_multi_data(
                train_data=split_datasets["train"],
                validation_data=split_datasets["validation"],
                datamodule_config=datamodule_config,
                trainer_config=trainer_config,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                transfer_learning_config=transfer_learning_config,
                compile_config=compile_config,
                ckpt_path=ckpt_path,
                **kwargs,
            )
        else:
            super()._train_multi_data(
                train_data=train_data,
                validation_data=validation_data,
                datamodule_config=datamodule_config,
                trainer_config=trainer_config,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                transfer_learning_config=transfer_learning_config,
                compile_config=compile_config,
                ckpt_path=ckpt_path,
                **kwargs,
            )
