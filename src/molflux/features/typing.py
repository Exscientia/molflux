from os import PathLike as OSPathLike
from typing import Any, Dict, Iterable, List, Union

RepresentationResult = Dict[str, Any]
ArrayLike = Iterable[Any]
SmilesArray = Iterable[str]
OEMolArray = Iterable[Any]
MolArray = Union[SmilesArray, OEMolArray, Iterable[bytes]]

PathLike = Union[str, OSPathLike]
Fingerprint = List[int]
