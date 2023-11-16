"""
Abstract Base Classes for classes implementing the SplittingStrategy protocol.
"""

import inspect
import types
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

from molflux import __version__
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.naming import camelcase_to_snakecase
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable
from molflux.splits.utils import copyfunc


class SplittingStrategyBase(ABC):
    """The abstract base class for all concrete splitting strategies."""

    def __init__(self, tag: Optional[str] = None, **kwargs: Any) -> None:
        """Initialises the strategy."""

        # build info
        info = self._info()
        info.name = camelcase_to_snakecase(type(self).__name__)
        info.tag = tag or info.name
        info.version = __version__
        info.split_description = info.split_description or self._split.__doc__ or ""

        # persist info
        self._splitting_strategy_info = info

        # Initialise a null state
        self._state: Dict[str, Any] = {}

        # Update 'split' docstring with detailed kwargs description from .info()
        # need to copy method to avoid changing the docstring of every instance
        self.split = types.MethodType(copyfunc(self.split), self)  # type: ignore[method-assign]
        self.split.__func__.__doc__ += self._splitting_strategy_info.split_description  # type: ignore[attr-defined]

        # The split signature
        self.split_signature = inspect.signature(self._split)

        # Safeguard against invalid kwargs
        self._init_signature = inspect.signature(self.__init__)  # type: ignore[misc]
        if kwargs:  # all kwargs should have been handled by now
            unknown_kwargs = list(kwargs.keys())
            raise ValueError(
                f"Unknown initialisation parameter(s): {unknown_kwargs}\n\n"
                f"Expected signature __init__{self._init_signature}.\n\n"
                f"Did you mean to pass these parameters to the strategy's .split() method instead?",
            )

    def __str__(self) -> str:
        return (
            f"SplittingStrategy(\n"
            f'\tname: "{self.name}",\n'
            f'\ttag: "{self.tag}",\n'
            f"\tsignature: self.split{self.split_signature},\n"
            f'\tdescription: """{self._splitting_strategy_info.description}""",\n'
            f'\tusage: """{self._splitting_strategy_info.split_description}"""\n'
            f"\tstate: {self.state!r}\n"
            ")"
        )

    @abstractmethod
    def _info(self) -> SplittingStrategyInfo:
        """Initialises the SplittingStrategyInfo object.

        To be implemented by subclasses.
        """

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._splitting_strategy_info.to_dict()

    @property
    def name(self) -> str:
        return self._splitting_strategy_info.name

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    @property
    def tag(self) -> str:
        return self._splitting_strategy_info.tag

    def split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Generate indices to split data into training, validation, and test set.
        """
        # Merge explicit keyword arguments with those stored in the state
        kwargs = {**self.state, **kwargs}

        # Safeguard against invalid kwargs
        if not all(k in self.split_signature.parameters for k in kwargs.keys()):
            unknown_kwargs = [
                k for k in kwargs.keys() if k not in self.split_signature.parameters
            ]
            raise ValueError(
                f"Unknown split parameter(s): {unknown_kwargs}\n\n"
                f"Expected signature self.split{self.split_signature}",
            )

        yield from self._split(dataset=dataset, y=y, groups=groups, **kwargs)

    @abstractmethod
    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """The actual split callable to be implemented by subclasses."""

    def reset_state(self) -> None:
        self._state = {}

    def update_state(self, **kwargs: Any) -> None:
        self._state = {**self.state, **kwargs}
