from os import PathLike as OSPathLike
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import fsspec

from datasets import Dataset, DatasetDict

# A list of lists of strings or Nones
DisplayNames = List[List[Optional[str]]]

FileSystem = fsspec.AbstractFileSystem
ExamplesGenerator = Iterator[Tuple[Union[int, str], Dict]]

# This includes str, pathlib.Path, cloudpathlib.CloudPath, ...
PathLike = Union[str, OSPathLike]

# Huggingface's DataFiles
HFDataFiles = Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
# An enhanced version accepting PathLike objects too
DataFiles = Union[
    PathLike,
    Sequence[PathLike],
    Mapping[str, Union[PathLike, Sequence[PathLike]]],
]

DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)
