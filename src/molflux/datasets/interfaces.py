from collections.abc import Iterable, Iterator, Sized
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)


@runtime_checkable
class Representation(Protocol):
    def featurise(self, samples: Iterable[Any], **kwargs: Any) -> dict[str, Any]:
        """Featurises the input samples."""


@runtime_checkable
class Representations(Protocol):
    def __iter__(self) -> Iterator[Representation]: ...


@runtime_checkable
class SplittingStrategy(Protocol):
    def split(
        self,
        dataset: Sized,
        y: Iterable | None = None,
        groups: Iterable | None = None,
        **kwargs: Any,
    ) -> Iterator[tuple[Iterable[int], Iterable[int], Iterable[int]]]:
        """Generates indices to split data into training, validation, and test sets."""
