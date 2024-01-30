import warnings
from typing import Any, Dict, Iterable, Literal, Optional, Union, overload

import datasets
from datasets import (
    Dataset,
    DatasetBuilder,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from molflux.datasets.catalogue import BUILDERS_CATALOGUE, get_dataset_builder_path
from molflux.datasets.parsers import Spec, dict_parser, yaml_parser
from molflux.datasets.typing import PathLike


def load_dataset_builder(
    name: str,
    config_name: Optional[str] = None,
    **kwargs: Any,
) -> DatasetBuilder:
    """Load a dataset builder from the catalogue or from the hf hub.

    A dataset builder can be used to inspect general information that is
    required to build a dataset (cache directory, config, dataset info, etc.)
    without downloading the dataset itself.

    You can find the list of available datasets with :func:`molflux.datasets.list_datasets()`.

    .. seealso::

        This function is a light wrapper around :func:`datasets.load_dataset_builder()`

    Args:
        name: The name of the dataset builder to load. This maps to the 'path'
            argument of :func:`datasets.load_dataset_builder()`.
        config_name: The name of the dataset configuration to load.
            If `None`, the default configuration is loaded. This maps to the
            'name' argument of :func:`datasets.load_dataset_builder()`.
        kwargs: Any other keyword arguments accepted by :func:`datasets.load_dataset_builder()`.

    Returns:
        A dataset builder.
    """

    # Check first if this is a custom builder name, and resolve path
    is_custom_builder = name in BUILDERS_CATALOGUE
    if is_custom_builder:
        name = get_dataset_builder_path(name)
        # ensure Datasets is allowed to execute the loading script by default
        kwargs = {"trust_remote_code": True, **kwargs}

    with warnings.catch_warnings():
        # for our custom builders, ignore warning from missing huggingface hub repo card metadata [MLOPS-1125]
        if is_custom_builder:
            warnings.filterwarnings(
                "ignore",
                message="Repo card metadata block was not found",
                category=UserWarning,
                module="huggingface_hub.repocard",
            )

        hf_builder = datasets.load_dataset_builder(
            path=name,
            name=config_name,
            **kwargs,
        )
        return hf_builder


@overload
def load_dataset(
    name: str,
    config_name: Optional[str] = None,
    *,
    split: str = "all",
    streaming: Literal[False] = False,
    **kwargs: Any,
) -> Dataset:
    ...


@overload
def load_dataset(
    name: str,
    config_name: Optional[str] = None,
    *,
    split: str = "all",
    streaming: Literal[True],
    **kwargs: Any,
) -> IterableDataset:
    ...


@overload
def load_dataset(
    name: str,
    config_name: Optional[str] = None,
    *,
    split: None,
    streaming: Literal[False] = False,
    **kwargs: Any,
) -> DatasetDict:
    ...


@overload
def load_dataset(
    name: str,
    config_name: Optional[str] = None,
    *,
    split: None,
    streaming: Literal[True],
    **kwargs: Any,
) -> IterableDatasetDict:
    ...


def load_dataset(
    name: str,
    config_name: Optional[str] = None,
    *,
    split: Optional[str] = "all",
    streaming: bool = False,
    **kwargs: Any,
) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """Load a dataset (or dataset dict) from the catalogue or from the hf hub.

    You can find the list of available datasets with :func:`molflux.datasets.list_datasets()`.

    .. seealso::

        This function is a light wrapper around :func:`datasets.load_dataset()`

    Args:
        name: The name of the dataset to load. This maps to the 'path'
            argument of :func:`datasets.load_dataset()`.
        config_name: The name of the dataset configuration to load.
            If `None`, the default configuration is loaded. This maps to the
            'name' argument of :func:`datasets.load_dataset()`.
        split: The split of the data to load.
            If None, will return a `dict` with all available splits.
            If given, will return a single Dataset. Defaults to 'all' which
            will return a single dataset combining data from all splits.
        streaming: Whether to stream the dataset. If set to True, the
            data files are not download but it is streamed progressively while
            iterating on the dataset. An IterableDataset or IterableDatasetDict
            is returned instead in this case. Defaults to False.
        kwargs: Any other keyword arguments accepted by :func:`datasets.load_dataset()`.

    Returns:
        The dataset requested.
    """

    # Check first if this is a custom builder name, and resolve path to loading script
    is_custom_builder = name in BUILDERS_CATALOGUE
    if is_custom_builder:
        name = get_dataset_builder_path(name)
        # ensure Datasets is allowed to execute the loading script by default
        kwargs = {"trust_remote_code": True, **kwargs}

    with warnings.catch_warnings():
        # for our custom builders, ignore warning from missing huggingface hub repo card metadata
        if is_custom_builder:
            warnings.filterwarnings(
                "ignore",
                message="Repo card metadata block was not found",
                category=UserWarning,
                module="huggingface_hub.repocard",
            )

        # Do not pass through unhandled deprecation warnings from dependencies of hf datasets themselves
        warnings.filterwarnings(
            "ignore",
            message="The .* keyword in pd.read_csv is deprecated and will be removed in a future version",
            category=FutureWarning,
            module="datasets[.*]",
        )
        return datasets.load_dataset(
            path=name,
            name=config_name,
            split=split,
            streaming=streaming,
            **kwargs,
        )


def _load_from_spec(spec: Spec) -> Union[Dataset, DatasetDict]:
    """Loads a dataset from a validated Spec."""

    dataset: Union[Dataset, DatasetDict] = load_dataset(name=spec.name, **spec.config)

    return dataset


def load_from_dict(dictionary: Dict[str, Any]) -> Union[Dataset, DatasetDict]:
    """Loads a dataset from a config dict."""

    # Validate dictionary
    spec = dict_parser(dictionary=dictionary)

    return _load_from_spec(spec=spec)


Datasets = Dict[str, Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]]


def load_from_dicts(dictionaries: Iterable[Dict[str, Any]]) -> Datasets:
    """Loads a collection of datasets from an iterable of dicts."""

    datasets = (load_from_dict(dictionary) for dictionary in dictionaries)

    # TODO: implement tags for datasets
    return {f"dataset-{i}": dataset for i, dataset in enumerate(datasets)}


def load_from_yaml(path: PathLike) -> Datasets:
    """Loads a collection of dataset from a yaml config file."""

    specs = yaml_parser(path=path)

    datasets = (_load_from_spec(spec) for spec in specs)

    # TODO: implement tags for datasets
    return {f"dataset-{i}": dataset for i, dataset in enumerate(datasets)}
