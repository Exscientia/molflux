import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.regression.spearman import Spearman


@pytest.fixture()
def fixture_metric():
    return load_metric("spearman")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "spearman" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, Spearman)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_returns_correlation_and_pvalue(fixture_metric):
    """That that the metric returns two values (correlation and pvalue)."""
    metric = fixture_metric
    predictions = [1, 2, 3, 4, 5]
    references = [5, 6, 7, 8, 7]
    result = metric.compute(predictions=predictions, references=references)
    assert len(result) == 2
    assert "spearman::correlation" in result
    assert "spearman::p_value" in result


def test_default_compute(fixture_metric):
    """That default scoring."""
    metric = fixture_metric
    predictions = [1, 2, 3, 4, 5]
    references = [5, 6, 7, 8, 7]
    result = metric.compute(predictions=predictions, references=references)
    assert pytest.approx(result["spearman::correlation"], 0.01) == 0.82
    assert pytest.approx(result["spearman::p_value"], 0.01) == 0.0886
