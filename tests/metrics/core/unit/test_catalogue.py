import pytest

import molflux.metrics.catalogue
from molflux.metrics.catalogue import get_metric_cls, list_metrics, register_metric

catalogue_obj = molflux.metrics.catalogue.METRICS_CATALOGUE


def test_catalogue_is_filled():
    """That the catalogue has been filled."""
    catalogue = list_metrics()
    assert len(catalogue)


def test_list_metrics_returns_sorted_names():
    """That entries in the catalogue are returned sorted by name."""
    catalogue = list_metrics()
    assert sorted(catalogue.keys()) == list(catalogue.keys())
    for metrics in catalogue.values():
        assert sorted(metrics) == metrics


def test_list_metrics_lists_names_of_metrics_in_catalogue():
    """That all metrics existing in the catalogue are returned by list_metrics."""
    view = list_metrics()
    names_in_view = [name for names in view.values() for name in names]
    assert sorted(names_in_view) == sorted(catalogue_obj.keys())


def test_register_metric(monkeypatch):
    """That can register a new metric in the catalogue."""
    # set up a mock empty catalogue to leave real one untouched
    monkeypatch.setattr(molflux.metrics.catalogue, "METRICS_CATALOGUE", {})

    new_metric_kind = "testing"
    new_metric_name = "pytest_metric"
    assert new_metric_name not in list_metrics()

    @register_metric(kind=new_metric_kind, name=new_metric_name)
    class PytestMetric:
        ...

    new_catalogue = list_metrics()
    assert new_metric_kind in new_catalogue
    assert new_metric_name in new_catalogue[new_metric_kind]


def test_get_metric_not_in_catalogue_raises_not_implemented_error():
    """That getting a non-existent metric raises a NotImplementedError."""
    metric_name = "non-existent-metric"
    with pytest.raises(
        NotImplementedError,
        match=f"Metric {metric_name!r} is not available",
    ):
        get_metric_cls(metric_name)
