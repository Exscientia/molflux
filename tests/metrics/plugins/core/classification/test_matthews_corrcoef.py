import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.classification.matthews_corrcoef import MatthewsCorrcoef


@pytest.fixture()
def fixture_metric():
    return load_metric("matthews_corrcoef")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "matthews_corrcoef" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, MatthewsCorrcoef)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_integer_labels(fixture_metric):
    """That can use integer labels."""
    metric = fixture_metric
    predictions = [1, 1, 1, 0]
    references = [1, 0, 1, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["matthews_corrcoef"] == -1 / 3


def test_perfect_score(fixture_metric):
    """That get unitary score for perfect predictions."""
    metric = fixture_metric
    predictions = [0, 1, 1, 0, 0, 1]
    references = [0, 1, 1, 0, 0, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["matthews_corrcoef"] == 1.0


def test_random_score(fixture_metric):
    """That get null score for average random prediction."""
    metric = fixture_metric
    predictions = [0, 1, 1, 1, 1, 0]
    references = [0, 1, 1, 0, 0, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["matthews_corrcoef"] == 0


def test_inverse_score(fixture_metric):
    """That get negative unitary score for inverse predictions."""
    metric = fixture_metric
    predictions = [1, 0, 0, 1, 1, 0]
    references = [0, 1, 1, 0, 0, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["matthews_corrcoef"] == -1
