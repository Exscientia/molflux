from typing import Any, Dict, Iterator, Protocol, runtime_checkable

from molflux.features.errors import DuplicateKeyError
from molflux.features.typing import ArrayLike, RepresentationResult


@runtime_checkable
class Representation(Protocol):
    """The public protocol for molflux.features.Representations."""

    def __init__(self, **kwargs: Any):
        """Initialises the representation."""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Representation metadata."""

    @property
    def name(self) -> str:
        """The canonical name of the representation."""

    @property
    def tag(self) -> str:
        """The arbitrary tag name of the representation."""

    @property
    def state(self) -> Dict[str, Any]:
        """The internal state config."""

    def featurise(self, samples: ArrayLike, **kwargs: Any) -> RepresentationResult:
        """Featurises the input samples."""

    def reset_state(self) -> None:
        """Resets the state."""

    def update_state(self, **kwargs: Any) -> None:
        """Updates the internal state config."""


class Representations:
    """A collection of Representations."""

    def __init__(self) -> None:
        self._stack: Dict[str, Representation] = {}

    def __contains__(self, item: str) -> bool:
        return item in self._stack

    def __getitem__(self, key: str) -> Representation:
        representation = self._stack.get(key)
        if representation is None:
            raise KeyError(
                f"Representation {key!r} has not been loaded. Available representations: {list(self._stack.keys())!r}",
            )
        return representation

    def __iter__(self) -> Iterator[Representation]:
        """Iterates through Representations as if a list."""
        return iter(self._stack.values())

    def __len__(self) -> int:
        return len(self._stack)

    def __repr__(self) -> str:
        tags = sorted(representation.tag for representation in self._stack.values())
        return f"Representations({tags!r})"

    def __setitem__(self, key: str, value: Representation) -> None:
        if key in self._stack:
            raise DuplicateKeyError(
                f"Representation with key {key!r} has already been added. You can add a unique custom tag to one of the representations to add it to the collection.",
            )
        self._stack[key] = value

    def add_representation(self, representation: Representation) -> None:
        self[representation.tag] = representation

    def featurise(self, samples: ArrayLike, **kwargs: Any) -> RepresentationResult:
        results = (
            representation.featurise(samples=samples)
            for representation in self._stack.values()
        )
        merged_results = {k: v for r in results for k, v in r.items()}
        return merged_results
