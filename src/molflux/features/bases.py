"""
Abstract Base Classes for classes implementing the Representation protocol.
"""

import inspect
import logging
import time
import types
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from molflux import __version__
from molflux.features.info import RepresentationInfo
from molflux.features.naming import camelcase_to_snakecase
from molflux.features.typing import ArrayLike, RepresentationResult
from molflux.features.utils import copyfunc

logger = logging.getLogger(__name__)


class RepresentationBase(ABC):
    """The abstract base class for all concrete representations."""

    def __init__(self, *, tag: Optional[str] = None, **kwargs: Any) -> None:
        """Initialises the representation."""

        # build info
        info = self._info()
        info.name = camelcase_to_snakecase(type(self).__name__)
        info.tag = tag or info.name
        info.version = __version__
        info.featurise_description = (
            info.featurise_description or self._featurise.__doc__ or ""
        )

        # persist info
        self._representation_info = info

        # Update 'featurise' docstring with detailed kwargs description from .info()
        # need to copy method to avoid changing the docstring of every instance
        self.featurise = types.MethodType(copyfunc(self.featurise), self)  # type: ignore[method-assign]
        self.featurise.__func__.__doc__ += self._representation_info.featurise_description  # type: ignore[attr-defined]

        # The featurisation signature
        self._signature = inspect.signature(self._featurise)

        # Initialise a default state
        self._default_state: Dict[str, Any] = {
            parameter.name: parameter.default
            for parameter in self._signature.parameters.values()
            # Positional included in case someone misses `, *, ` in their representation signature.
            if (
                parameter.kind
                in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
            )
            # Empty parameters not included (i.e. `samples`, `kwargs`).
            and (parameter.default is not parameter.empty)
        }
        self._state: Dict[str, Any] = self._default_state

    def __str__(self) -> str:
        return (
            f"Representation(\n"
            f'\tname: "{self.name}",\n'
            f'\ttag: "{self.tag}",\n'
            f"\tsignature: self.featurise{self._signature},\n"
            f'\tdescription: """{self._representation_info.description}""",\n'
            f'\tusage: """{self._representation_info.featurise_description}"""\n'
            f"\tstate: {self.state!r}\n"
            ")"
        )

    @abstractmethod
    def _info(self) -> RepresentationInfo:
        """Initialises the RepresentationInfo object.

        To be implemented by subclasses.
        """

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._representation_info.to_dict()

    @property
    def name(self) -> str:
        return self._representation_info.name

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    @property
    def tag(self) -> str:
        return self._representation_info.tag

    def featurise(self, samples: ArrayLike, **kwargs: Any) -> RepresentationResult:
        """Featurises the input samples."""

        # Merge explicit keyword arguments with those stored in the state
        kwargs = {**self.state, **kwargs}

        # Safeguard against invalid kwargs
        if not all(k in self._signature.parameters for k in kwargs.keys()):
            unknown_kwargs = [
                k for k in kwargs.keys() if k not in self._signature.parameters
            ]
            raise ValueError(
                f"Unknown featurisation parameter(s): {unknown_kwargs}\n\n"
                f"Expected signature self.featurise{self._signature}",
            )

        # Safeguard against non ArrayLike inputs
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Iterable):
            samples = [samples]

        logging_context: Dict[str, Any] = {"exs_prism_representation_tag": self.tag}
        logger.info(
            f"Attempting to featurise samples... {logging_context}",
            extra=logging_context,
        )

        start = time.perf_counter()
        results = self._featurise(samples, **kwargs)
        elapsed = time.perf_counter() - start

        features = list(results.keys())
        results_shape = [len(results[features[0]]), len(features)]
        try:
            rate = elapsed / (results_shape[0] * results_shape[1])
        except ZeroDivisionError:
            rate = 0

        logging_context.update(
            {
                **logging_context,
                "molflux_representation_featurisation_results_shape": results_shape,
                "molflux_representation_featurisation_time_sec": elapsed,
                "molflux_representation_featurisation_rate_sec_per_feature_per_sample": rate,
            },
        )
        logger.info(f"Featurisation completed {logging_context}", extra=logging_context)

        return results

    @abstractmethod
    def _featurise(self, samples: ArrayLike, **kwargs: Any) -> RepresentationResult:
        """The featurisation callable to be implemented by subclasses."""

    def reset_state(self) -> None:
        """Resets the state."""
        self._state = self._default_state

    def update_state(self, **kwargs: Any) -> None:
        """Pre-configures keyword arguments for featurise()."""
        self._state = {**self.state, **kwargs}
