from typing import Any, Dict, Iterator, Optional, Protocol, runtime_checkable

from molflux.splits.typing import ArrayLike, SplitIndices, Splittable


@runtime_checkable
class SplittingStrategy(Protocol):
    """
    Common API for all splitting strategies.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialises the dataset"""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Splitting strategy metadata."""

    @property
    def name(self) -> str:
        """The canonical name of the splitting strategy."""

    @property
    def state(self) -> Dict[str, Any]:
        """The internal state config."""

    @property
    def tag(self) -> str:
        """The arbitrary tag name of the splitting strategy."""

    def split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Generate indices to split data into training, validation, and test set.

        Args:
            dataset: Training data, of shape (n_samples, n_features).
            y:  The target variable for supervised learning problems. Of shape
                (n_samples, ).
            groups: Group labels for the samples used while splitting the
                dataset into train/test set. Of shape (n_samples, ).

        Yields:
             train_indices: The training set indices for that split.
             validation_indices: The validation set indices for that split.
             test_indices: The validation set indices for that split.
        """

    def reset_state(self) -> None:
        """Resets the state."""

    def update_state(self, **kwargs: Any) -> None:
        """Updates the internal state config."""
