from collections.abc import Iterable
from os import PathLike as OSPathLike
from typing import Any, Union

RepresentationResult = dict[str, Any]
ArrayLike = Iterable[Any]
SmilesArray = Iterable[str]
OEMolArray = Iterable[Any]
MolArray = Union[SmilesArray, OEMolArray, Iterable[bytes]]

PathLike = Union[str, OSPathLike]
Fingerprint = list[int]
