import os
import pathlib
from typing import Any, Dict, Mapping, Optional

import molflux.datasets
import molflux.modelzoo
import molflux.splits
from datasets import Dataset, DatasetDict
from molflux.core.featurisation.metadata import save_featurisation_metadata
from molflux.core.io import save_dict_to_json
from molflux.core.typing import FoldScores, PathLike

_FEATURISED_DATASET_FILENAME = "featurised_dataset.parquet"
_SCORES_FILENAME = "scores.json"
_PARAMS_FILENAME = "model_params.json"
_SPLITTING_STRATEGY_FILENAME = "splitting_strategy.json"
_PIPELINE_CONFIG_FILENAME = "pipeline.json"


def log_params(mapping: Mapping, path: PathLike) -> str:
    """Logs arbitrary dicts of parameters to disk."""
    return save_dict_to_json(mapping, path=path)


def log_dataset(dataset: Dataset, path: PathLike) -> str:
    """Logs an arbitrary dataset to disk."""
    format = pathlib.Path(path).suffix.lstrip(".")
    molflux.datasets.save_dataset_to_store(dataset, path=str(path), format=format)  # type: ignore[arg-type]
    return str(path)


def log_dataset_dict(
    dataset_dict: DatasetDict,
    paths_dict: Dict[str, str],
) -> Dict[str, str]:
    """Logs an arbitrary dataset_dict to disk"""
    save_paths: Dict[str, str] = {}
    for split_name, dataset in dataset_dict.items():
        save_path = log_dataset(dataset, path=paths_dict[split_name])
        save_paths[split_name] = save_path
    return save_paths


def log_featurised_dataset(featurised_dataset: Dataset, directory: PathLike) -> str:
    """Logs a featurised dataset to disk."""
    output_file = f"{directory}/{_FEATURISED_DATASET_FILENAME}"
    return log_dataset(dataset=featurised_dataset, path=output_file)


def log_featurisation_metadata(
    featurisation_metadata: Optional[Dict[str, Any]],
    directory: PathLike,
) -> str:
    """Logs featurisation metadata to disk.

    This can be used as an independent function to save featurisation metadata
    decoupled from the model saving process.
    """
    return save_featurisation_metadata(
        featurisation_metadata=featurisation_metadata,
        path=directory,
    )


def log_fold(fold: DatasetDict, directory: PathLike) -> Dict[str, str]:
    """Logs a fold to disk as a set of .parquet files."""
    save_paths: Dict[str, str] = {}
    for split, dataset in fold.items():
        _SPLIT_FILENAME = f"{split}.parquet"
        path = os.path.join(directory, _SPLIT_FILENAME)
        molflux.datasets.save_dataset_to_store(dataset, path=path, format="parquet")
        save_paths[split] = path
    return save_paths


def log_predictions(predictions: DatasetDict, directory: str) -> Dict[str, str]:
    """Logs fold predictions to storage as a set of standardised parquet files."""
    paths_dict = {
        split_name: os.path.join(directory, f"{split_name}_predictions.parquet")
        for split_name in predictions.keys()
    }
    return log_dataset_dict(predictions, paths_dict=paths_dict)


def log_references(references: DatasetDict, directory: str) -> Dict[str, str]:
    """Logs fold references to storage as a set of standardised parquet files."""
    paths_dict = {
        split_name: os.path.join(directory, f"{split_name}_references.parquet")
        for split_name in references.keys()
    }
    return log_dataset_dict(references, paths_dict=paths_dict)


def log_inputs(inputs: DatasetDict, directory: str) -> Dict[str, str]:
    """Logs a fold's model input features to storage as parquet."""
    paths_dict = {
        split_name: os.path.join(directory, f"{split_name}_inputs.parquet")
        for split_name in inputs.keys()
    }
    return log_dataset_dict(inputs, paths_dict=paths_dict)


def log_model_params(model: molflux.modelzoo.Model, directory: PathLike) -> str:
    """Logs model metadata to disk."""
    output_file = f"{directory}/{_PARAMS_FILENAME}"
    return save_dict_to_json(model.metadata, path=output_file)


def log_pipeline_config(config: Mapping, directory: PathLike) -> str:
    """Logs the pipeline config to disk."""
    output_file = f"{directory}/{_PIPELINE_CONFIG_FILENAME}"
    return save_dict_to_json(config, path=output_file)


def log_scores(scores: FoldScores, directory: PathLike) -> str:
    """Logs metrics scores to disk."""
    output_file = f"{directory}/{_SCORES_FILENAME}"
    return save_dict_to_json(scores, path=output_file)


def log_splitting_strategy(
    strategy: molflux.splits.SplittingStrategy,
    directory: PathLike,
) -> str:
    """Logs splitting strategy metadata to disk."""
    output_file = f"{directory}/{_SPLITTING_STRATEGY_FILENAME}"

    payload = {
        "name": strategy.name,
        "tag": strategy.tag,
        "metadata": strategy.metadata,
    }

    return save_dict_to_json(payload, path=output_file)
