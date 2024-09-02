import logging
from typing import Any, TypeVar

import molflux.datasets
import molflux.features
from molflux.core.featurisation.metadata import (
    FeaturisationMetadataV1,
    FeaturisationMetadataV2,
    fetch_model_featurisation_metadata,
    parse_featurisation_metadata,
)
from molflux.core.typing import DatasetType

logger = logging.getLogger(__name__)


# Generic type; used to indicate functions are Dataset->Dataset, DatasetDict->DatasetDict
DatasetT = TypeVar("DatasetT", bound=DatasetType)


def replay_dataset_featurisation(
    inputs: DatasetType,
    model_path: str,
    **map_kwargs: Any,
) -> DatasetType:
    featurisation_metadata = fetch_model_featurisation_metadata(model_path=model_path)
    return featurise_dataset(
        inputs=inputs,
        featurisation_metadata=featurisation_metadata,
        **map_kwargs,
    )


def featurise_dataset(
    inputs: DatasetType,
    featurisation_metadata: dict[str, Any],
    **map_kwargs: Any,
) -> DatasetType:
    featurisation_metadata, version = parse_featurisation_metadata(
        featurisation_metadata,
    )

    if version == 1:
        return _featurise_dataset_v1(
            dataset=inputs,
            featurisation_metadata=featurisation_metadata,
            **map_kwargs,
        )
    if version == 2:
        return _featurise_dataset_v2(
            dataset=inputs,
            featurisation_metadata=featurisation_metadata,
            **map_kwargs,
        )
    else:
        raise NotImplementedError(
            f"No featuriser implemented for featurisation metadata version: {version!r}",
        )


def _featurise_dataset_v1(
    dataset: DatasetType,
    featurisation_metadata: dict[str, Any],
    **map_kwargs: Any,
) -> DatasetType:
    """Featurises a dataset from a V1 featurisation metadata schema."""
    featurisation_metadata_obj = FeaturisationMetadataV1(**featurisation_metadata)
    featurisation_config = featurisation_metadata_obj.config
    for column_featurisation_config in featurisation_config:
        column = column_featurisation_config.column
        representation_configs = column_featurisation_config.representations

        # We allow passing single string display_names as convenience method
        # but we need to turn them into a list to match canonical form expected
        # by molflux.datasets.featurise_dataset()
        display_names = [config.as_ for config in representation_configs]
        canonical_display_names = [
            names if isinstance(names, list) else [names] for names in display_names
        ]

        representation_configs_as_dicts = [
            {"name": config.name, "config": config.config, "presets": config.presets}
            for config in representation_configs
        ]

        representations = molflux.features.load_from_dicts(
            representation_configs_as_dicts,
        )
        dataset = molflux.datasets.featurise_dataset(
            dataset,
            column=[column],
            representations=representations,
            display_names=canonical_display_names,  # type: ignore[arg-type]
            **map_kwargs,
        )
    return dataset


def _featurise_dataset_v2(
    dataset: DatasetT,
    featurisation_metadata: dict[str, Any],
    **map_kwargs: Any,
) -> DatasetT:
    """Featurises a dataset from a V2 featurisation metadata schema."""
    featurisation_metadata_obj = FeaturisationMetadataV2(**featurisation_metadata)
    featurisation_config = featurisation_metadata_obj.config
    for column_featurisation_config in featurisation_config:
        columns = column_featurisation_config.columns
        representation_configs = column_featurisation_config.representations

        # We allow passing single string display_names as convenience method
        # but we need to turn them into a list to match canonical form expected
        # by molflux.datasets.featurise_dataset()
        display_names = [config.as_ for config in representation_configs]
        canonical_display_names = [
            names if isinstance(names, list) else [names] for names in display_names
        ]

        representation_configs_as_dicts = [
            {"name": config.name, "config": config.config, "presets": config.presets}
            for config in representation_configs
        ]

        representations = molflux.features.load_from_dicts(
            representation_configs_as_dicts,
        )
        dataset = molflux.datasets.featurise_dataset(
            dataset,
            column=columns,
            representations=representations,
            display_names=canonical_display_names,  # type: ignore[arg-type]
            **map_kwargs,
        )
    return dataset
