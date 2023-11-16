from os import PathLike as OSPathLike
from typing import Mapping, Optional, Sequence

from molflux.datasets.typing import DataFiles, HFDataFiles


def _is_cloud_file(file: str) -> bool:
    """Validates if file is a cloud file"""
    if file.startswith("s3://"):
        return True
    return False


def is_cloud_data(data_files: Optional[HFDataFiles]) -> bool:
    """Validates if data files refer to data in the cloud."""

    if data_files is None:
        return False

    if isinstance(data_files, str):
        return _is_cloud_file(data_files)

    if isinstance(data_files, Sequence):
        return _is_cloud_file(data_files[0])

    if isinstance(data_files, Mapping):
        files = list(data_files.values())
        file = files[0]

        if isinstance(file, str):
            return _is_cloud_file(file)

        if isinstance(file, Sequence):
            return _is_cloud_file(file[0])

    raise ValueError("Could not resolve if data files are cloud data files")


def sanitise_data_files_as_str(data_files: DataFiles) -> HFDataFiles:
    """Coerces Path-like DataFiles to str-like DataFiles.

    This is to simplify and sanitise the inputs to match the huggingface
    datasets DataFiles API.
    """

    if isinstance(data_files, (str, OSPathLike)):
        return str(data_files)

    if isinstance(data_files, Sequence):
        return [str(data_file) for data_file in data_files]

    if isinstance(data_files, Mapping):
        return {k: sanitise_data_files_as_str(files) for k, files in data_files.items()}  # type: ignore[misc]

    raise TypeError(f"Unsupported data files type: {type(data_files)}")
