import logging
from typing import Any, Optional

try:
    import gpytorch
    import gpytorch.mlls.marginal_log_likelihood
    import torch

except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning_gp", e) from e

from molflux.modelzoo.models.lightning.module import (
    LightningModuleBase,
    SingleBatchStepOutput,
)
from molflux.modelzoo.models.lightning_gp.configs.utils import (
    generate_backbone,
    generate_likelihood,
)
from molflux.modelzoo.models.lightning_gp.gp_config import GPConfig
from molflux.modelzoo.models.lightning_gp.gpytorch_model import GPModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GPModule(LightningModuleBase):
    model_config: GPConfig

    def __init__(self, model_config: GPConfig) -> None:
        super().__init__(model_config=model_config)

        self.likelihood = generate_likelihood(self.model_config)
        self.gp_model = GPModel(
            model_config=model_config,
            train_x=torch.empty(0).float(),
            train_y=torch.empty(0).float(),
            likelihood=self.likelihood,
        )

        self.backbone = generate_backbone(model_config.backbone_config)

        self.loss = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood,
            self.gp_model,
        )
        self.validation_data: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.train_data_set = False

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if we have a backbone, pass the X features (first arg) through the backbone, and then to the GP
        if self.backbone is not None:
            features = self.backbone(args[0], **kwargs)
            return self.gp_model(features, *(args[1:]), **kwargs)

        else:
            return self.gp_model(*args, **kwargs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """fit method for doing a predict step"""
        del batch_idx, dataloader_idx

        x, _ = batch
        x = x.squeeze(
            0,
        )  # the BxNxD tensor is squeezed as we always have a batch of size 1

        outputs = self.likelihood(self(x))

        return outputs.loc, outputs.stddev

    def _training_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: str | None,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """Implements the training step on one minibatch.

        Returns a dict with a loss key and a batch size."""
        del source_name, batch_idx, args, kwargs

        x, y = single_source_batch
        x, y = x.squeeze(0), y.squeeze(0)

        if self.gp_model.train_inputs[0].shape[0] != x.shape[0]:
            logger.info(
                f"Set the train data for the GP, from {self.gp_model.train_inputs[0].shape[0]} "
                f"to {x.shape[0]} data points",
            )
            self.gp_model.set_train_data(inputs=x, targets=y, strict=False)
            self.train_data_set = True

        outputs = self(x)
        loss = -self.loss(outputs, target=y)

        batch_size = x.shape[0]

        return loss, {}, batch_size

    def _validation_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: str | None,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        del source_name, batch_idx, args, kwargs

        x, y = single_source_batch
        x, y = x.squeeze(0), y.squeeze(0)
        self.validation_data = x, y

        outputs = self.likelihood(self(x))
        loss = gpytorch.metrics.negative_log_predictive_density(outputs, y)

        batch_size = x.shape[0]

        return loss, {}, batch_size

    def on_fit_end(self) -> None:
        """Add the validation data to the model at the end of training, if specified"""
        if (
            self.model_config.validation_config.fit_end_fantasy
            and self.validation_data is not None
        ):
            self.gp_model.set_train_data(
                inputs=torch.cat(
                    [self.gp_model.train_inputs[0], self.validation_data[0]],
                ),
                targets=torch.cat(
                    [self.gp_model.train_targets, self.validation_data[1]],
                ),
                strict=False,
            )
