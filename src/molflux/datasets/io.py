import json
import os
import pathlib
import warnings
from typing import Any, Dict, List, Literal, Optional, Union, cast, get_args

import fsspec

import datasets
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets.filesystems import is_remote_filesystem
from datasets.utils.py_utils import NestedDataStructure
from molflux.datasets.typing import DataFiles, FileSystem, HFDataFiles, PathLike
from molflux.datasets.utils import (
    is_cloud_data,
    sanitise_data_files_as_str,
)

SupportedFileFormats = Literal["disk", "parquet", "csv", "json"]
_DEFAULT_DATASET_DICT_FORMAT: SupportedFileFormats = "parquet"


def __resolve_path_file_format(path: str) -> SupportedFileFormats:
    """Resolves the core file format of persisted assets.

    This function parses file paths to resolve them into one of our
    supported file formats. Directory-like paths are resolved to 'disk' format.
    Compressed file formats are resolved to their core uncompressed file format.
    """

    suffixes = [suffix.lstrip(".") for suffix in pathlib.Path(path).suffixes]
    is_directory = not suffixes

    if is_directory:
        return cast(SupportedFileFormats, "disk")  # cast to help type checker

    for supported_file_format in get_args(SupportedFileFormats):
        if supported_file_format in suffixes:
            return cast(
                SupportedFileFormats,
                supported_file_format,
            )  # cast to help type checker

    raise ValueError(
        f"Unsupported dataset format for path {path!r}: expected one of {get_args(SupportedFileFormats)!r}",
    )


def _resolve_data_files_file_format(data_files: HFDataFiles) -> SupportedFileFormats:
    """Resolves the file format of the input huggingface DataFiles.

    All data files are expected to resolve to the same file format.
    """
    file_formats: List[SupportedFileFormats] = []
    for path in NestedDataStructure(data_files).flatten():
        file_format = __resolve_path_file_format(path)
        file_formats.append(file_format)

    if not len(set(file_formats)) == 1:
        raise ValueError("Mismatched file formats.")

    return file_formats[0]


def __raise_for_path_format_consistency(
    path: str,
    format: str,
    is_dataset_dict: bool,
) -> None:
    """Checks that the given path / format combination is valid.

    Args:
        path: The target path to save to / load from the dataset.
        format: The persistence format the dataset should be in / is.
        is_dataset_dict: Whether the target of the IO operation is / was a DatasetDict.
    """

    # only specific persistence formats are supported
    if format not in get_args(SupportedFileFormats):
        raise ValueError(
            f"Unsupported dataset format for path {path!r}: got {format!r} expected one of {get_args(SupportedFileFormats)!r}",
        )

    suffix = pathlib.Path(path).suffix
    is_directory_like_path = not suffix

    # DatasetDicts can only be saved / loaded from directories
    if is_dataset_dict and not is_directory_like_path:
        raise ValueError(
            f"Invalid path for DatasetDict: {path!r} is not a directory-like path",
        )

    # For Datasets, the filepath extension must match the format
    if not is_dataset_dict:
        expected_format = _resolve_data_files_file_format(path)
        if format != expected_format:
            raise ValueError(
                f"Mismatched file format: got {format!r} expected {expected_format!r}",
            )


def _raise_for_data_files_format_consistency(
    data_files: HFDataFiles,
    format: str,
    is_dataset_dict: bool,
) -> None:
    """Checks that the given data_files / format combination is valid."""
    for path in NestedDataStructure(data_files).flatten():
        __raise_for_path_format_consistency(
            path,
            format=format,
            is_dataset_dict=is_dataset_dict,
        )


def _check_compatible_filesystem(fs: FileSystem, data_files: HFDataFiles) -> None:
    """Checks that the given filesystem is compatible with the data files it
    should operate on.

    Cloud data files should be backed by a cloud filesystem, local data files
    should be backed by a local filesystem.
    """
    need_to_load_from_cloud = is_cloud_data(data_files)
    is_cloud_filesystem = is_remote_filesystem(fs)
    if need_to_load_from_cloud and not is_cloud_filesystem:
        raise ValueError(
            "Incompatible filesystem: please provide a remote filesystem for accessing cloud data.",
        )
    if not need_to_load_from_cloud and is_cloud_filesystem:
        raise ValueError(
            "Incompatible filesystem: please provide a local filesystem for accessing local data.",
        )


def _resolve_file_system_for_data_files(data_files: HFDataFiles) -> FileSystem:
    """Returns a ready-made file system suited for handling input data files."""
    need_to_load_from_cloud = is_cloud_data(data_files)
    fs = (
        fsspec.filesystem("s3")
        if need_to_load_from_cloud
        else fsspec.filesystem("file")
    )
    return fs


def save_dataset_to_store(
    dataset: Union[Dataset, DatasetDict],
    path: PathLike,
    format: Optional[SupportedFileFormats] = None,
    fs: Optional[FileSystem] = None,
    **kwargs: Any,
) -> None:
    """Save a dataset to persistent storage.

    Args:
        dataset: The Dataset or DatasetDict to persist.
        path: The target path where to save the dataset. Should be a directory
            for DatasetDicts or for saving datasets as 'disk' format.
        format: The file format to save the dataset as. Should be one of
            ["disk", "parquet", "csv", "json"]. If missing, for Datasets this
            will be automatically inferred from 'path'. For DatasetDicts, it
            will default to "parquet".
        fs: The filesystem object to use to save the dataset. If not provided,
            an appropriate filesystem will be automatically provided for
            saving the data.

    Examples:
        >>> from molflux.datasets import save_dataset_to_store  # doctest: +SKIP
        # save a Dataset as parquet
        >>> save_dataset_to_store(dataset, "s3://my/dataset/data.parquet")  # doctest: +SKIP
        # save a Dataset as a bunch of .arrow files
        >>> save_dataset_to_store(dataset, "s3://my/dataset/data")  # doctest: +SKIP
        # save a DatasetDict as parquet
        >>> save_dataset_to_store(dataset_dict, "s3://my/dataset/data")  # doctest: +SKIP
        # save a DatasetDict as a specific persistence format (e.g. csv)
        >>> save_dataset_to_store(dataset_dict, "s3://my/dataset/data", format="csv")  # doctest: +SKIP
        # save a DatasetDict as a bunch of .arrow files
        >>> save_dataset_to_store(dataset_dict, "s3://my/dataset/data", format="disk")  # doctest: +SKIP
    """

    path = str(path)

    # Provide a filesystem if one not specified (convenience)
    if fs is None:
        fs = _resolve_file_system_for_data_files(data_files=path)
    _check_compatible_filesystem(fs, data_files=path)

    # Resolve format if one not specified (convenience)
    is_dataset_dict = isinstance(dataset, datasets.DatasetDict)
    if format is None:
        format = (
            _DEFAULT_DATASET_DICT_FORMAT
            if is_dataset_dict
            else _resolve_data_files_file_format(path)
        )

    _raise_for_data_files_format_consistency(
        path,
        format=format,
        is_dataset_dict=is_dataset_dict,
    )

    # save_to_disk is implemented both for Dataset and DatasetDict
    if format == "disk":
        return dataset.save_to_disk(path, storage_options=fs.storage_options, **kwargs)  # type: ignore[no-any-return]

    # for all other file format let's implement our own wrappers
    if is_dataset_dict:
        return _save_dataset_dict_to_file(
            dataset_dict=dataset,
            dataset_dict_path=path,
            format=format,
            fs=fs,
            **kwargs,
        )
    else:
        return _save_dataset_to_file(
            dataset=dataset,
            dataset_path=path,
            format=format,
            fs=fs,
            **kwargs,
        )


def _generate_split_path(root: str, split: str, format: SupportedFileFormats) -> str:
    """Generates a standardised name for filenames of splits."""
    return os.path.join(root, f"{split}.{format}")


def _save_dataset_dict_to_file(
    dataset_dict: DatasetDict,
    dataset_dict_path: str,
    format: SupportedFileFormats,
    fs: FileSystem,
    **kwargs: Any,
) -> None:
    """Saves a DatasetDict to file.

    This emulates what huggingface already implements out of the box for
    DatasetDict.save_to_disk(), extending it to all other file formats too. A
    json metadata file is saved with the names of the splits composing the
    DatasetDict, and the individual datasets in the dict are saved individually
    alongside of it.
    """

    is_local = not is_remote_filesystem(fs)
    if is_local:
        pathlib.Path(dataset_dict_path).resolve().mkdir(parents=True, exist_ok=True)

    # write out huggingface's dataset_dict.json metadata file
    dataset_dict_json_filepath = os.path.join(
        dataset_dict_path,
        datasets.config.DATASETDICT_JSON_FILENAME,
    )
    with fs.open(dataset_dict_json_filepath, "w", encoding="utf-8") as f:
        json.dump({"splits": list(dataset_dict)}, f)

    # write out individual datasets
    for k, dataset in dataset_dict.items():
        dataset_path = _generate_split_path(dataset_dict_path, split=k, format=format)
        _save_dataset_to_file(
            dataset,
            dataset_path=dataset_path,
            format=format,
            fs=fs,
            **kwargs,
        )


def _save_dataset_to_file(
    dataset: Dataset,
    dataset_path: str,
    format: SupportedFileFormats,
    fs: FileSystem,
    **kwargs: Any,
) -> None:
    """Saves a single Dataset to file."""

    is_local = not is_remote_filesystem(fs)
    if is_local:
        pathlib.Path(dataset_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    if not fs.isfile(dataset_path):
        fs.touch(dataset_path)

    with fs.open(dataset_path, "wb") as f:
        if format == "csv":
            dataset.to_csv(path_or_buf=f, **kwargs)

        elif format == "parquet":
            dataset.to_parquet(path_or_buf=f, **kwargs)

        elif format == "json":
            dataset.to_json(path_or_buf=f, **kwargs)

        else:
            raise NotImplementedError(f"Invalid file format: {format}")


def _is_dataset_dict_dir(dest_dataset_dict_path: str, fs: FileSystem) -> bool:
    """Checks that we are in a dataset_dict artefact directory.

    This is done by checking for the presence of the json metadata file that
    gets generated on saving of DatasetDicts. This emulates the same logic
    already implemented natively in huggingface datasets.load_from_disk(), but
    extending it for all other formats too.
    """

    if not fs.isdir(dest_dataset_dict_path):
        return False

    dataset_dict_json_path = os.path.join(
        dest_dataset_dict_path,
        datasets.config.DATASETDICT_JSON_FILENAME,
    )
    return fs.isfile(dataset_dict_json_path)  # type: ignore[no-any-return]


def _resolve_expected_dataset_dict_dir_data_files(
    dest_dataset_dict_path: str,
    format: SupportedFileFormats,
    fs: FileSystem,
) -> Dict[str, str]:
    """Expands the path to a DatasetDict directory into its expected DataFiles.

    This works by extracting the split names from the persisted DatasetDict json
    metadata file and then using those to resolve the expected dataset filenames.

    Examples:
        >>> _resolve_expected_dataset_dict_dir_data_files("my/data", format="parquet")  # doctest: +SKIP
        {"train": "my/data/train.parquet", "validation": "my/data/validation.parquet", "test": "my/data/test.parquet"}
    """

    if not _is_dataset_dict_dir(dest_dataset_dict_path, fs=fs):
        raise FileNotFoundError(
            "Expected to load a `DatasetDict` object, but provided path is not a `DatasetDict`.",
        )

    dataset_dict_json_path = os.path.join(
        dest_dataset_dict_path,
        datasets.config.DATASETDICT_JSON_FILENAME,
    )
    with fs.open(dataset_dict_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    return {
        k: _generate_split_path(dest_dataset_dict_path, split=k, format=format)
        for k in splits
    }


def load_dataset_from_store(
    source: DataFiles,
    format: Optional[SupportedFileFormats] = None,
    fs: Optional[FileSystem] = None,
    split: Optional[str] = None,
    **kwargs: Any,
) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """Loads a dataset from persisted storage.

    Args:
        source: Path(s) to source data file(s).
        format: The file format the dataset has been persisted as. Should be one of
            ["disk", "parquet", "csv", "json"]. If missing, for Datasets this
            will be automatically inferred from 'source'. For DatasetDicts, it
            will default to "parquet".
        fs: The filesystem object to use to save the dataset. If not provided,
            an appropriate filesystem will be automatically provided for
            saving the data.
        split: For DatasetDicts, which split of the data to load. If not provided,
            defaults to None which will return a DatasetDict of all splits. Any
            other value will return a single Dataset corresponding to the indicated
            split. The special value "all" can be used to return a single Dataset
            with all splits merged.

    Examples:
        >>> from molflux.datasets import load_dataset_from_store  # doctest: +SKIP
        # load a persisted parquet Dataset
        >>> dataset = load_dataset_from_store("s3://my/bucket/data.parquet")  # doctest: +SKIP
        # load a persisted 'disk' Dataset
        >>> dataset = load_dataset_from_store("s3://my/bucket/data")  # doctest: +SKIP
        # load a persisted parquet DatasetDict
        >>> dataset_dict = load_dataset_from_store("s3://my/bucket/data")  # doctest: +SKIP
        # load a persisted csv DatasetDict
        >>> dataset_dict = load_dataset_from_store("s3://my/bucket/data", format="csv")  # doctest: +SKIP
        # load a persisted parquet DatasetDict as a single Dataset
        >>> dataset = load_dataset_from_store("s3://my/bucket/data", split="all")  # doctest: +SKIP
        # load from a mapping of persisted Datasets
        >>> source = {"train": "my/train_data.csv", "test": "my_test_data.csv"}
        >>> dataset_dict = load_dataset_from_store(source)  # doctest: +SKIP
    """

    # Collapse path-like datafiles to str
    source = sanitise_data_files_as_str(source)

    # Provide a filesystem if one not specified (convenience)
    if fs is None:
        fs = _resolve_file_system_for_data_files(data_files=source)
    _check_compatible_filesystem(fs, data_files=source)

    # Resolve format if one not specified (convenience)
    is_dataset_dict = isinstance(source, str) and _is_dataset_dict_dir(source, fs=fs)
    if format is None:
        format = (
            _DEFAULT_DATASET_DICT_FORMAT
            if is_dataset_dict
            else _resolve_data_files_file_format(source)
        )

        _raise_for_data_files_format_consistency(
            source,
            format=format,
            is_dataset_dict=is_dataset_dict,
        )

    # huggingface implements their own dedicated function for loading "disk" files
    if format == "disk":
        # Loading from disk only allows loading from a single directory
        if not isinstance(source, str) or not fs.isdir(source):
            raise ValueError(
                f"Invalid data source: {source} should point to a (existing) directory for format {format!r}.",
            )

        return datasets.load_from_disk(
            dataset_path=source,
            storage_options=fs.storage_options,
        )

    # for all other file format let's implement our own wrapper:

    # Auto-expand directory source into datadict datafiles (convenience)
    if is_dataset_dict:
        data_files = _resolve_expected_dataset_dict_dir_data_files(
            source,  # type: ignore[arg-type]
            format=format,
            fs=fs,
        )
    else:
        data_files = source  # type: ignore[assignment]

    # Change hf default behaviour and return single dataset as Dataset (convenience)
    data_files_are_for_dataset_dict = is_dataset_dict or isinstance(data_files, dict)
    if split is None and not data_files_are_for_dataset_dict:
        split = "all"

    with warnings.catch_warnings():
        # Do not pass through unhandled deprecation warnings from dependencies of hf datasets themselves
        warnings.filterwarnings(
            "ignore",
            message="The .* keyword in pd.read_csv is deprecated and will be removed in a future version",
            category=FutureWarning,
            module="datasets[.*]",
        )

        return datasets.load_dataset(
            format,
            data_files=data_files,
            split=split,
            storage_options=fs.storage_options,
            **kwargs,
        )
