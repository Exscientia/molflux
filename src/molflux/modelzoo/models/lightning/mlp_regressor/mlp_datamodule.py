from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

import datasets
from molflux.modelzoo.models.lightning.datamodule import LightningDataModule
from molflux.modelzoo.models.lightning.mlp_regressor.mlp_config import MLPConfig


class MLPDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        x_features: List[str],
        y_features: List[str],
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
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        datapoint = self.dataset[idx]

        x_list: List[torch.Tensor] = []
        for x_col in self.x_features:
            x_feat = datapoint[x_col]

            if len(x_feat.shape) == 0:
                x_feat = x_feat.unsqueeze(0)
            elif len(x_feat.shape) > 1:
                raise RuntimeError(
                    f"The shape of x_feature {x_col} = {x_feat.shape} cannot be concatenated.",
                )

            x_list.append(x_feat)

        x = torch.cat(x_list, dim=0)

        if self.with_y_features:
            y_list: List[torch.Tensor] = []
            for y_col in self.y_features:
                y_feat = datapoint[y_col]

                if len(y_feat.shape) == 0:
                    y_feat = y_feat.unsqueeze(0)
                elif len(y_feat.shape) > 1:
                    raise RuntimeError(
                        f"The shape of y_feature {y_col} = {y_feat.shape} cannot be concatenated.",
                    )

                y_list.append(y_feat)

            y = torch.cat(y_list, dim=0)
        else:
            y = torch.empty(0)

        return x, y


class MLPDataModule(LightningDataModule):
    model_config: MLPConfig

    def prepare_dataset(
        self,
        data: datasets.Dataset,
        split: str,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dataset:
        del name, kwargs

        with_y_features = split != "predict"

        return MLPDataset(
            dataset=data,
            x_features=self.model_config.x_features,
            y_features=self.model_config.y_features,
            with_y_features=with_y_features,
        )
