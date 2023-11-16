import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.classification.top_k_accuracy import TopKAccuracy

metric_name = "top_k_accuracy"


@pytest.fixture()
def fixture_metric():
    return load_metric(metric_name)


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert metric_name in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, TopKAccuracy)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_multiclass(fixture_metric):
    """Multiclass case."""
    metric = fixture_metric
    predictions = [
        [0.5, 0.2, 0.2],  # 0 is in top 2
        [0.3, 0.4, 0.2],  # 1 is in top 2
        [0.2, 0.4, 0.3],  # 2 is in top 2
        [0.7, 0.2, 0.1],  # 2 isn't in top 2
    ]
    references = [0, 1, 2, 2]
    result = metric.compute(predictions=predictions, references=references)
    assert result["top_k_accuracy"] == 3 / 4
