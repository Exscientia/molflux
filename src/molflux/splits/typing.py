from collections.abc import Iterable, Sized
from os import PathLike as OSPathLike
from typing import Union

ArrayLike = Iterable
PathLike = Union[str, OSPathLike]
Splittable = Sized
SplitIndices = tuple[Iterable[int], Iterable[int], Iterable[int]]
