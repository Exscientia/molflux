import difflib
import functools
import inspect
import logging
from collections import defaultdict
from importlib.metadata import EntryPoint, entry_points
from typing import Callable, Dict, List, Type

from molflux.metrics.metric import Metric
from molflux.metrics.naming import camelcase_to_snakecase

logger = logging.getLogger(__name__)

MetricT = Type[Metric]

# Only ever call this once for performance reasons
NAMESPACE = "molflux.metrics.plugins."

# This is where entrypoints will be registered {<name>: <Entrypoint>}
METRICS_CATALOGUE: Dict[str, EntryPoint] = {}


@functools.lru_cache
def fill_catalogue() -> None:
    """Fills the catalogue with entrypoints.

    Only ever call this once for performance reasons.
    """
    entrypoints = [
        entrypoint
        for namespace, entrypoints in entry_points().items()
        if namespace.startswith(NAMESPACE)
        for entrypoint in entrypoints
    ]
    for entrypoint in entrypoints:
        put_metric_entrypoint(metric_entrypoint=entrypoint)


def put_metric_entrypoint(metric_entrypoint: EntryPoint) -> None:
    """Puts a metric entrypoint in the catalogue."""
    metric_name = camelcase_to_snakecase(metric_entrypoint.name)

    if metric_name in METRICS_CATALOGUE:
        if metric_entrypoint == METRICS_CATALOGUE[metric_name]:
            pass
        else:
            raise KeyError(f"Duplicate metric: {metric_name!r}")

    METRICS_CATALOGUE[metric_name] = metric_entrypoint


def get_metric_entrypoint(metric_name: str) -> EntryPoint:
    """Returns a specific metric entrypoint from the catalogue.

    The metric is keyed only by name. This allows to decouple from
    eventual reshuffles of entrypoints across groups.
    """
    metric_entrypoint = METRICS_CATALOGUE.get(metric_name)
    if metric_entrypoint is None:
        msg = f"Metric {metric_name!r} is not available."
        similar = difflib.get_close_matches(metric_name, METRICS_CATALOGUE.keys())
        if similar:
            msg += f" You might be looking for one of these: {similar}"
        raise NotImplementedError(msg)
    return metric_entrypoint


def get_metric_cls(metric_name: str) -> MetricT:
    """Loads a specific metric entrypoint from the catalogue."""
    entrypoint = get_metric_entrypoint(metric_name=metric_name)

    # Might raise (e.g. missing dependency extras)
    metric_cls = entrypoint.load()

    if not isinstance(metric_cls, Metric):
        raise NotImplementedError(
            f"Plugin {entrypoint.value} does not implement Metric protocol.",
        )

    return metric_cls  # type: ignore[return-value]


def list_metrics() -> Dict[str, List[str]]:
    """List all available metrics in the catalogue.

    The catalogue is returned as a view of metric names keyed by kind.
    """

    view: Dict[str, List[str]] = defaultdict(list)

    for metric_name, metric_entrypoint in METRICS_CATALOGUE.items():
        metric_kind = metric_entrypoint.group.split(".")[-1]
        view[metric_kind].append(metric_name)

    return dict(sorted(view.items()))


def register_metric(kind: str, name: str) -> Callable:
    """Registers a custom metric in the entrypoints catalogue.

    Examples:
        .. code-block:: python

           @register_metric(kind="custom", name="my_metric")
           class CustomMetric:
               ...
    """

    def wrapper(metric_cls: MetricT) -> MetricT:
        module = inspect.getmodule(metric_cls).__name__  # type: ignore
        value = f"{module}:{metric_cls.__name__}"
        group = NAMESPACE + kind
        entrypoint = EntryPoint(name=name, value=value, group=group)
        put_metric_entrypoint(metric_entrypoint=entrypoint)
        return metric_cls

    return wrapper
