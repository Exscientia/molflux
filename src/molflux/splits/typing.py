from os import PathLike as OSPathLike
from typing import Iterable, Sized, Tuple, Union

ArrayLike = Iterable
PathLike = Union[str, OSPathLike]
Splittable = Sized
SplitIndices = Tuple[Iterable[int], Iterable[int], Iterable[int]]
