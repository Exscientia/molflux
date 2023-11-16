from typing import Any, Optional

import torch

from molflux.modelzoo.models.lightning.mlp_regressor.mlp_config import MLPConfig
from molflux.modelzoo.models.lightning.module import (
    LightningModuleBase,
    SingleBatchStepOutput,
)


class MLPModule(LightningModuleBase):
    def __init__(self, model_config: MLPConfig) -> None:
        super().__init__(model_config=model_config)

        if model_config.num_layers == 1:
            self.module = torch.nn.Linear(
                in_features=model_config.input_dim,
                out_features=model_config.num_tasks,
            )
        else:
            module_list = []
            module_list.append(
                torch.nn.Linear(
                    in_features=model_config.input_dim,
                    out_features=model_config.hidden_dim,
                ),
            )
            module_list.append(torch.nn.SiLU())
            for _ in range(model_config.num_layers - 2):
                module_list.append(
                    torch.nn.Linear(
                        in_features=model_config.hidden_dim,
                        out_features=model_config.hidden_dim,
                    ),
                )
                module_list.append(torch.nn.SiLU())

            module_list.append(
                torch.nn.Linear(
                    in_features=model_config.hidden_dim,
                    out_features=model_config.num_tasks,
                ),
            )

            self.module = torch.nn.Sequential(*module_list)

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """fit method for doing a predict step"""
        del batch_idx, dataloader_idx

        x, _ = batch
        outputs = self(x)

        return outputs

    def _training_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: Optional[str],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """Implements the training step on one minibatch.

        Returns a dict with a loss key and a batch size."""
        del source_name, batch_idx, args, kwargs

        x, y = single_source_batch
        outputs = self(x)
        loss = self.mse_loss(input=outputs, target=y)
        mae = (outputs - y).abs().mean()

        batch_size = x.shape[0]

        return loss, {"mae": mae}, batch_size

    def _validation_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: Optional[str],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        del source_name, batch_idx, args, kwargs

        x, y = single_source_batch
        outputs = self(x)
        loss = self.mse_loss(input=outputs, target=y)
        mae = (outputs - y).abs().mean()

        batch_size = x.shape[0]

        return loss, {"mae": mae}, batch_size
