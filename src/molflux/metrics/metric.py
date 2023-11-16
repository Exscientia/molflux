import logging
from typing import Any, Dict, Iterator, Optional, Protocol, runtime_checkable

from molflux.metrics.errors import DuplicateKeyError
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)


@runtime_checkable
class Metric(Protocol):
    """The public protocol for Metrics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialises the metric."""

    def __len__(self) -> int:
        """The number of examples loaded in the metric's cache."""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dictionary containing all the metadata."""

    @property
    def name(self) -> str:
        """The canonical name."""

    @property
    def tag(self) -> str:
        """The tag name."""

    @property
    def state(self) -> Dict[str, Any]:
        """The internal state."""

    def add_batch(
        self,
        *,
        predictions: Optional[ArrayLike] = None,
        references: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        """Add a batch of predictions and references for the metric's stack."""

    def compute(
        self,
        *,
        predictions: Optional[ArrayLike] = None,
        references: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Computes the metric."""

    def reset_state(self) -> None:
        """Resets the state."""

    def update_state(self, **kwargs: Any) -> None:
        """Pre-configures keyword arguments for compute()."""


class Metrics:
    """A collection of Metrics."""

    def __init__(self) -> None:
        self._stack: Dict[str, Metric] = {}

    def __contains__(self, item: str) -> bool:
        return item in self._stack

    def __getitem__(self, key: str) -> Metric:
        metric = self._stack.get(key)
        if metric is None:
            raise KeyError(
                f"Metric {key!r} has not been loaded. Available metrics: {list(self._stack.keys())!r}",
            )
        return metric

    def __iter__(self) -> Iterator[Metric]:
        """Iterates through Metrics as if a list."""
        return iter(self._stack.values())

    def __len__(self) -> int:
        return len(self._stack)

    def __repr__(self) -> str:
        tags = sorted(metric.tag for metric in self._stack.values())
        return f"Metrics({tags!r})"

    def __setitem__(self, key: str, value: Metric) -> None:
        if key in self._stack:
            raise DuplicateKeyError(
                f"Metric with key {key!r} has already been added. You can add a unique custom tag to one of the metrics to add it to the collection.",
            )
        self._stack[key] = value

    def add_batch(
        self,
        *,
        predictions: Optional[ArrayLike] = None,
        references: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        for metric in self._stack.values():
            metric.add_batch(predictions=predictions, references=references, **kwargs)

    def add_metric(self, metric: Metric) -> None:
        self[metric.tag] = metric

    def compute(
        self,
        *,
        predictions: Optional[ArrayLike] = None,
        references: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> MetricResult:
        results = (
            metric.compute(
                predictions=predictions,
                references=references,
                **kwargs,
            )
            for metric in self._stack.values()
        )
        merged_results = {k: v for r in results for k, v in r.items()}
        return merged_results
