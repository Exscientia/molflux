import os
import sys
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict, Union

import datasets

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.combined_loader import CombinedLoader
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
    from torch.utils.data import DataLoader
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning", e) from e

from molflux.modelzoo.models.lightning.config import (
    DataModuleConfig,
    LightningConfig,
    SplitConfig,
)


class DataCollator(Protocol):
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...


class Datasets(TypedDict):
    train: Optional[Dict[Optional[str], datasets.Dataset]]
    validation: Optional[Dict[Optional[str], datasets.Dataset]]
    test: Optional[Dict[Optional[str], datasets.Dataset]]
    predict: Optional[datasets.Dataset]


class LightningDataModule(pl.LightningDataModule):
    _config_builder = DataModuleConfig

    def __init__(
        self,
        model_config: LightningConfig,
        train_data: Optional[Dict[Optional[str], datasets.Dataset]] = None,
        validation_data: Optional[Dict[Optional[str], datasets.Dataset]] = None,
        test_data: Optional[Dict[Optional[str], datasets.Dataset]] = None,
        predict_data: Optional[datasets.Dataset] = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.model_config = model_config

        self.datasets = self.prepare_datasets(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            predict_data=predict_data,
            **kwargs,
        )

    def prepare_dataset(
        self,
        data: datasets.Dataset,
        split: str,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> datasets.Dataset:
        """A method to prepare any input dataset type into a torch dataset ready for torch dataloaders.

        Some custom models will need to overwrite this method with their own implementation.
        """
        del kwargs

        if split == "predict":
            data = data.with_format("torch", columns=self.model_config.x_features)
        else:
            data = data.with_format(
                "torch",
                columns=self.model_config.resolve_train_features(name),
            )

        return data

    def _get_one_train_dataloader(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
    ) -> DataLoader:
        """Returns one dataloader for training data.

        Implementations should override this method if they require a custom
        sampler or dataloader."""

        return DataLoader(
            dataset,  # pyright: ignore
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            drop_last=self.model_config.datamodule.train.drop_last,
            collate_fn=self.collate_fn,
        )

    def _get_one_eval_dataloader(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
    ) -> DataLoader:
        """Returns one dataloader for validation, test or predict (inference) data.

        Implementations should override this method if they require a custom
        sampler or dataloader."""
        return DataLoader(
            dataset,  # pyright: ignore
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Return dataloaders with training data.

        Overrides an abstract method in PyTorch Lightning. Implementations
        probably don't need to modify this method; look at
        _get_one_train_dataloader instead."""
        datasets = self.datasets["train"]

        if datasets is None:
            return {}

        return CombinedLoader(
            {
                name: self._get_one_train_dataloader(
                    dataset,
                    self._get_batch_size("train", name),
                )
                for name, dataset in datasets.items()
            },
            self.model_config.datamodule.train.mode,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Return dataloaders with validation data.

        Overrides an abstract method in PyTorch Lightning. Implementations
        probably don't need to modify this method; look at
        _get_one_eval_dataloader instead."""
        return self._get_eval_dataloaders(
            "validation",
            self.model_config.datamodule.validation.mode,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Return dataloaders with test data.

        Overrides an abstract method in PyTorch Lightning. Implementations
        probably don't need to modify this method; look at
        _get_one_eval_dataloader instead."""
        return self._get_eval_dataloaders(
            "test",
            self.model_config.datamodule.test.mode,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Return dataloaders with data for prediction (inference).

        Overrides an abstract method in PyTorch Lightning. Implementations
        probably don't need to modify this method; look at
        _get_one_eval_dataloader instead."""
        # NOTE 11/9/2023 Lightning doesn't support non-sequential predict
        dataset = self.datasets["predict"]

        if dataset is None:
            raise ValueError("Specify a prediction dataset.")

        return self._get_one_eval_dataloader(
            dataset,
            self._get_batch_size("predict", None),
        )

    def _get_eval_dataloaders(
        self,
        split: Literal["validation", "test"],
        multi_data_mode: Literal[
            "max_size_cycle",
            "max_size",
            "min_size",
            "sequential",
        ],
    ) -> EVAL_DATALOADERS:
        datasets = self.datasets[split]

        if datasets is None:
            return {}

        return CombinedLoader(
            {
                name: self._get_one_eval_dataloader(
                    dataset,
                    self._get_batch_size(split, name),
                )
                for name, dataset in datasets.items()
            },
            multi_data_mode,
        )

    def prepare_datasets(
        self,
        train_data: Optional[Dict[Optional[str], datasets.Dataset]],
        validation_data: Optional[Dict[Optional[str], datasets.Dataset]],
        test_data: Optional[Dict[Optional[str], datasets.Dataset]],
        predict_data: Optional[datasets.Dataset],
        **kwargs: Any,
    ) -> Datasets:
        """Prepares datasets by applying logic in `prepare_dataset` to all
        datasets.

        Most implementations will not have to modify this method."""
        if train_data is not None:
            train_data = {
                k: self.prepare_dataset(v, "train", k, **kwargs)
                for k, v in (train_data or {}).items()
            }

        if validation_data is not None:
            validation_data = {
                k: self.prepare_dataset(v, "validation", k, **kwargs)
                for k, v in (validation_data or {}).items()
            }

        if test_data is not None:
            test_data = {
                k: self.prepare_dataset(v, "test", k, **kwargs)
                for k, v in (test_data or {}).items()
            }

        if predict_data is not None:
            predict_data = self.prepare_dataset(predict_data, "predict", **kwargs)

        return Datasets(
            train=train_data,
            validation=validation_data,
            test=test_data,
            predict=predict_data,
        )

    @property
    def num_workers(self) -> int:
        """Number of workers servicing each individual dataloader."""
        config_num_workers: Union[
            int,
            Literal["all"],
        ] = self.model_config.datamodule.num_workers

        if isinstance(config_num_workers, int):
            return config_num_workers

        if config_num_workers == "all":
            if sys.platform.startswith("linux"):
                # This is the safer way to check num CPUs on k8s
                return len(os.sched_getaffinity(0))  # pyright: ignore
            return os.cpu_count() or 0  # type: ignore[unreachable,unused-ignore]

        raise ValueError("num_workers should be either an int or 'all'.")

    def _get_batch_size(
        self,
        split: Literal["train", "validation", "test", "predict"],
        dataset_name: Optional[str],
    ) -> int:
        split_config: SplitConfig = getattr(self.model_config.datamodule, split)

        if isinstance(split_config.batch_size, int):
            return split_config.batch_size
        else:
            if dataset_name is None:
                raise ValueError("Specify the dataset name.")
            return split_config.batch_size[dataset_name]

    @property
    def collate_fn(self) -> Optional[DataCollator]:
        """Collator for dataloaders.

        Derived classes can override this."""
        return None
