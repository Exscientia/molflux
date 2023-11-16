import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.regression.max_error import MaxError


@pytest.fixture()
def fixture_metric():
    return load_metric("max_error")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "max_error" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, MaxError)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_default_compute(fixture_metric):
    """That default scoring."""
    metric = fixture_metric
    predictions = [9, 2, 7, 1]
    references = [3, 2, 7, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["max_error"] == 6


def test_perfect_score(fixture_metric):
    """Test score on perfect predictions."""
    metric = fixture_metric
    predictions = [3, 2, 7, 1]
    references = [3, 2, 7, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["max_error"] == 0
