from typing import Any, cast

import datasets
from molflux.modelzoo.models.lightning.datamodule import LightningDataModule
from molflux.modelzoo.models.lightning_gp.gp_config import GPConfig

try:
    import torch
    from torch.utils.data import Dataset

except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning_gp", e) from e


class GPDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        x_features: list[str],
        y_features: list[str],
        with_y_features: bool,
    ) -> None:
        self.dataset = dataset
        self.x_features = x_features
        self.y_features = y_features
        self.with_y_features = with_y_features

        all_features = self.x_features
        if with_y_features:
            all_features = all_features + self.y_features

        self.dataset.set_format("torch", columns=all_features)

    def __len__(self) -> int:
        # we send the whole dataset through, in a single batch
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # assume we're doing single batch optimisation
        del idx

        x_list: list[torch.Tensor] = []
        for x_col in self.x_features:
            x_feat = cast(torch.Tensor, self.dataset[x_col])

            if len(x_feat.shape) == 1:
                x_feat = x_feat.unsqueeze(1)

            x_list.append(x_feat)

        x = torch.cat(x_list, dim=-1)

        if self.with_y_features:
            y_list: list[torch.Tensor] = []
            for y_col in self.y_features:
                y_feat = cast(torch.Tensor, self.dataset[y_col])

                y_list.append(y_feat)

            y = torch.cat(y_list, dim=0)
        else:
            y = torch.empty(0)

        return x, y


class GPDataModule(LightningDataModule):
    model_config: GPConfig

    def __init__(
        self,
        model_config: GPConfig,
        train_data: dict[str | None, datasets.Dataset] | None = None,
        validation_data: dict[str | None, datasets.Dataset] | None = None,
        test_data: dict[str | None, datasets.Dataset] | None = None,
        predict_data: datasets.Dataset | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_config=model_config,
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
        name: str | None = None,
        **kwargs: Any,
    ) -> Dataset:
        del name, kwargs

        with_y_features = split != "predict"

        return GPDataset(
            dataset=data,
            x_features=self.model_config.x_features,
            y_features=self.model_config.y_features,
            with_y_features=with_y_features,
        )
