"""
Tests ensuring desired API objects are part of the top-level namespace.
"""

import molflux.metrics


def test_exports_list_metrics():
    """That the package exposes the list_metrics function."""
    assert hasattr(molflux.metrics, "list_metrics")
    assert callable(molflux.metrics.list_metrics)


def test_exports_list_suites():
    """That the package exposes the list_suites function."""
    assert hasattr(molflux.metrics, "list_suites")
    assert callable(molflux.metrics.list_suites)


def test_exports_load_from_dict():
    """That the package exposes the load_from_dict function."""
    assert hasattr(molflux.metrics, "load_from_dict")
    assert callable(molflux.metrics.load_from_dict)


def test_exports_load_from_dicts():
    """That the package exposes the load_from_dicts function."""
    assert hasattr(molflux.metrics, "load_from_dict")
    assert callable(molflux.metrics.load_from_dicts)


def test_exports_load_from_yaml():
    """That the package exposes the load_from_yaml function."""
    assert hasattr(molflux.metrics, "load_from_yaml")
    assert callable(molflux.metrics.load_from_yaml)


def test_exports_load_metric():
    """That the package exposes the load_metric function."""
    assert hasattr(molflux.metrics, "load_metric")
    assert callable(molflux.metrics.load_metric)


def test_exports_load_metrics():
    """That the package exposes the load_metrics function."""
    assert hasattr(molflux.metrics, "load_metrics")
    assert callable(molflux.metrics.load_metrics)


def test_exports_load_suite():
    """That the package exposes the load_suite function."""
    assert hasattr(molflux.metrics, "load_suite")
    assert callable(molflux.metrics.load_suite)


def test_exports_metric():
    """That the package exposes the Metric protocol."""
    assert hasattr(molflux.metrics, "Metric")


def test_exports_metrics():
    """That the package exposes the Metrics class."""
    assert hasattr(molflux.metrics, "Metrics")


def test_exports_supports_prediction_intervals():
    """That the package exposes the supports_uncertainty function."""
    assert hasattr(molflux.metrics, "supports_prediction_intervals")
    assert callable(molflux.metrics.supports_prediction_intervals)
