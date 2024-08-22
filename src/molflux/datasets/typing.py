from collections.abc import Iterator, Mapping, Sequence
from os import PathLike as OSPathLike
from typing import (
    TypeVar,
    Union,
)

import fsspec

from datasets import Dataset, DatasetDict

# A list of lists of strings or Nones
DisplayNames = list[list[str | None]]

FileSystem = fsspec.AbstractFileSystem
ExamplesGenerator = Iterator[tuple[int | str, dict]]

# This includes str, pathlib.Path, cloudpathlib.CloudPath, ...
PathLike = Union[str, OSPathLike]

# Huggingface's DataFiles
HFDataFiles = Union[str, Sequence[str], Mapping[str, str | Sequence[str]]]
# An enhanced version accepting PathLike objects too
DataFiles = Union[
    PathLike,
    Sequence[PathLike],
    Mapping[str, PathLike | Sequence[PathLike]],
]

DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)
