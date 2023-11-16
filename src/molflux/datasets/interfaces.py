from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Sized,
    Tuple,
    runtime_checkable,
)


@runtime_checkable
class Representation(Protocol):
    def featurise(self, samples: Iterable[Any], **kwargs: Any) -> Dict[str, Any]:
        """Featurises the input samples."""


@runtime_checkable
class Representations(Protocol):
    def __iter__(self) -> Iterator[Representation]:
        ...


@runtime_checkable
class SplittingStrategy(Protocol):
    def split(
        self,
        dataset: Sized,
        y: Optional[Iterable] = None,
        groups: Optional[Iterable] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[Iterable[int], Iterable[int], Iterable[int]]]:
        """Generates indices to split data into training, validation, and test sets."""
