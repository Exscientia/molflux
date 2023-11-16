from os import PathLike as OSPathLike
from typing import (
    Any,
    Dict,
    List,
    Protocol,
    Union,
    runtime_checkable,
)

import datasets


@runtime_checkable
class SupportsDataframeInterchangeProtocol(Protocol):
    """Protocol representing any object conforming to the Dataframe Interchange Protocol.

    Several DataFrame libraries now support this interchange protocol (pandas, modin, vaex, cudf, pyarrow).

    References:
        https://data-apis.org/dataframe-protocol/latest/purpose_and_scope.html
        https://data-apis.org/dataframe-protocol/latest/API.html
    """

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> Any:
        ...


DataFrameLike = Union[SupportsDataframeInterchangeProtocol, datasets.Dataset]
Classes = Dict[str, List[Any]]
Features = List[str]
PathLike = Union[str, OSPathLike]
PredictionResult = Dict[str, List[Any]]
