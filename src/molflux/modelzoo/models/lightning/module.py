from abc import abstractmethod
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, cast

from lightning.pytorch.utilities.types import _METRIC
from typing_extensions import TypeAlias

from molflux.modelzoo.models.lightning.config import LightningConfig
from molflux.modelzoo.models.lightning.trainer.optimizers.stock_optimizers import (
    AVAILABLE_OPTIMIZERS,
)

try:
    import torch
    from lightning.pytorch import LightningModule
    from lightning.pytorch.utilities import grad_norm
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning", e) from e


Loss: TypeAlias = torch.Tensor
TensorsToLog: TypeAlias = Dict[str, torch.Tensor]
BatchSize: TypeAlias = int
SingleBatchStepOutput: TypeAlias = Tuple[Loss, TensorsToLog, BatchSize]
Split = Literal["train", "val"]


class LightningModuleBase(LightningModule):
    def __init__(
        self,
        model_config: LightningConfig,
    ) -> None:
        super().__init__()

        self.model_config = model_config

    @abstractmethod
    def _training_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: Optional[str],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """Implements a training step on a subbatch from a single dataloader.

        Must return a loss, a (possibly empty) dictionary of tensors, which will be
        automatically logged, and a batch size."""
        ...

    @abstractmethod
    def _validation_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: Optional[str],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """Implements a validation step on a subbatch from a single dataloader.

        Must return a loss, a (possibly empty) dictionary of tensors, which will be
        automatically logged, and a batch size."""
        ...

    def training_step(
        self,
        batch: Dict[Optional[str], Any],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Implements a single training step.

        Overrides an abstract method in PyTorch Lightning. Implementations
        probably don't need to modify this method; look at
        _training_step_on_single_source_batch instead."""

        # compute losses
        losses: Dict[Optional[str], Dict[str, torch.Tensor]] = {}
        batch_sizes: Dict[Optional[str], int] = {}
        for name, single_source_batch in batch.items():
            # can also get {"a": single_source_batch, "b": None} batches
            if single_source_batch is None:
                continue

            (
                loss,
                losses[name],
                batch_sizes[name],
            ) = self._training_step_on_single_source_batch(
                single_source_batch,
                name,
                batch_idx,
                *args,
                **kwargs,
            )
            losses[name]["loss"] = loss

        total_loss = self._process_and_log_losses("train", losses, batch_sizes)
        return total_loss

    def validation_step(
        self,
        batch: Dict[Optional[str], Any],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Implements a single training step.

        Overrides an abstract method in PyTorch Lightning. Implementations
        probably don't need to modify this method; look at
        _validation_step_on_single_source_batch instead."""

        # compute losses
        losses: Dict[Optional[str], Dict[str, torch.Tensor]] = {}
        batch_sizes: Dict[Optional[str], int] = {}
        for name, single_source_batch in batch.items():
            # can also get {"a": single_source_batch, "b": None} batches
            if single_source_batch is None:
                continue

            (
                loss,
                losses[name],
                batch_sizes[name],
            ) = self._validation_step_on_single_source_batch(
                single_source_batch,
                name,
                batch_idx,
                *args,
                **kwargs,
            )
            losses[name]["loss"] = loss

        total_loss = self._process_and_log_losses("val", losses, batch_sizes)
        return total_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Returns an optimizer and (optional) scheduler.

        Expected by PyTorch Lightning.
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """

        if self.model_config.optimizer is None:
            raise ValueError("Optimizer must be specified in model config.")

        optimizer_config = self.model_config.optimizer
        optimizer = AVAILABLE_OPTIMIZERS[optimizer_config.name](
            filter(lambda p: p.requires_grad, self.parameters()),
            **optimizer_config.config,
        )
        out: Dict[str, Any] = {"optimizer": optimizer}

        if self.model_config.scheduler is not None:
            out["lr_scheduler"] = self.model_config.scheduler.prepare_scheduler(
                optimizer=optimizer,
                trainer=self.trainer,
            )

        return out

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Logs gradient magnitudes during training."""
        del optimizer

        grads = cast(Dict[str, torch.Tensor], grad_norm(self, norm_type=2))
        # `grads` includes individual layer grads as well as an overall Frobenius norm
        grad_norm_total = grads.pop("grad_2.0_norm_total").item()
        grad_norm_max = max(t.item() for t in grads.values())
        self.log_dict(
            {"grad_norm_total": grad_norm_total, "grad_norm_max": grad_norm_max},
            on_step=True,
            on_epoch=False,
        )

    def log_by_split(
        self,
        split: Split,
        source: str,
        name: str,
        value: _METRIC,
        batch_size: Optional[int] = None,
    ) -> None:
        """Convenience method for logging."""
        if split == "train":
            self.log(
                f"{split}/{source}/{name}",
                value,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                logger=True,
                batch_size=batch_size,
            )
        elif split == "val":
            self.log(
                f"{split}/{source}/{name}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def log_dict_by_split(
        self,
        split: Split,
        source: str,
        dictionary: Mapping[str, _METRIC],
        batch_size: Optional[int] = None,
    ) -> None:
        """Convenience method for logging multiple metrics simultaneously."""
        if split == "train":
            self.log_dict(
                {
                    f"{split}/{source}/{name}": metric
                    for name, metric in dictionary.items()
                },
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                logger=True,
                batch_size=batch_size,
            )
        elif split == "val":
            self.log_dict(
                {
                    f"{split}/{source}/{name}": metric
                    for name, metric in dictionary.items()
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _process_and_log_losses(
        self,
        split: Split,
        losses: Dict[Optional[str], Dict[str, torch.Tensor]],
        batch_sizes: Dict[Optional[str], int],
    ) -> torch.Tensor:
        """Logs single source losses and metrics and returns the mean of all loss components."""

        # simple single-source path
        # data batches from unnamed sources look like {None: single_source_batch}
        if None in losses:
            self.log_dict_by_split(
                split,
                "total",
                losses[None],
                batch_sizes[None],
            )
            return losses[None]["loss"]

        # for multiple datasets we log the total loss (optimisation target)
        # and, per source, each metric or loss component
        total_loss_list: List[torch.Tensor] = []

        for source, single_source_losses in losses.items():
            if source is None:
                continue

            single_source_batch_size = batch_sizes[source]

            self.log_dict_by_split(
                split,
                source,  # pyright: ignore
                single_source_losses,
                single_source_batch_size,
            )

            if "loss" in single_source_losses:
                total_loss_list.append(single_source_losses["loss"])

        total_loss = torch.stack(total_loss_list).mean()
        total_batch_size = sum(batch_sizes.values())

        self.log_by_split(
            split,
            "total",
            "loss",
            total_loss,
            batch_size=total_batch_size,
        )

        return total_loss
