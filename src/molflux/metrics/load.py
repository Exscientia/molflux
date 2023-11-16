from typing import Any, Dict, Iterable, List, Optional

from molflux.metrics.catalogue import get_metric_cls
from molflux.metrics.metric import Metric, Metrics
from molflux.metrics.parsers import Spec, dict_parser, yaml_parser
from molflux.metrics.suites.catalogue import get_suite_path
from molflux.metrics.typing import PathLike


def load_metric(name: str, **metric_init_kwargs: Any) -> Metric:
    """Loads a `metrics.Metric` from metric name."""

    # Fetch relevant metric class from the catalogue
    metric_cls = get_metric_cls(metric_name=name)

    # Instantiate metric object
    metric = metric_cls(**metric_init_kwargs)

    return metric


def load_metrics(*names: str, tags: Optional[List[str]] = None) -> Metrics:
    """Loads a `metrics.Metrics` collection from metric names.

    This is a utility functions for loading a number of metrics at once, when
    no further customisation is needed.
    """

    tags = tags or [None for _ in names]  # type:ignore[misc]

    metrics = Metrics()
    for name, tag in zip(names, tags):
        metric = load_metric(name=name, tag=tag)
        metrics.add_metric(metric=metric)

    return metrics


def _load_from_spec(spec: Spec) -> Metric:
    """Loads a metric from a validated Spec."""

    # Build metric
    metric = load_metric(name=spec.name, **spec.config)

    # update state
    metric.update_state(**spec.presets)

    return metric


def load_from_dict(dictionary: Dict[str, Any]) -> Metric:
    """Loads a metric from a config dict."""

    # Validate dictionary
    spec = dict_parser(dictionary=dictionary)

    return _load_from_spec(spec=spec)


def load_from_dicts(dictionaries: Iterable[Dict[str, Any]]) -> Metrics:
    """Loads a collection of metrics from an iterable of dicts."""

    metrics = Metrics()
    for dictionary in dictionaries:
        metric = load_from_dict(dictionary=dictionary)

        metrics.add_metric(metric=metric)

    return metrics


def load_from_yaml(path: PathLike) -> Metrics:
    """Loads a collection of metrics from a yaml config file."""

    specs = yaml_parser(path=path)

    metrics = Metrics()
    for spec in specs:
        metric = _load_from_spec(spec=spec)

        metrics.add_metric(metric=metric)

    return metrics


def load_suite(name: str) -> Metrics:
    """Loads a pre-configured suite of metrics."""
    path = get_suite_path(suite_name=name)
    return load_from_yaml(path=path)
