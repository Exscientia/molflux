import numpy as np
import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.classification.average_precision import AveragePrecision

metric_name = "average_precision"


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
    assert isinstance(metric, AveragePrecision)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_binary_classification(fixture_metric):
    """That can score binary classification."""
    metric = fixture_metric
    references = [1, 0, 1, 1, 0, 0]
    predictions = [0.5, 0.2, 0.99, 0.3, 0.1, 0.7]
    result = metric.compute(references=references, predictions=predictions)
    assert metric_name in result
    assert pytest.approx(result[metric_name], 0.01) == 0.81


def test_multilabel_classification():
    """That can score multilabel classification."""
    metric = load_metric(metric_name, config_name="multilabel")
    references = [[1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]]
    predictions = [
        [0.3, 0.5, 0.2],
        [0.7, 0.2, 0.1],
        [0.005, 0.99, 0.005],
        [0.2, 0.3, 0.5],
        [0.1, 0.1, 0.8],
        [0.1, 0.7, 0.2],
    ]
    result = metric.compute(
        references=references,
        predictions=predictions,
        average=None,
    )
    assert metric_name in result
    assert np.allclose(result[metric_name], [0.87, 0.73, 0.92], rtol=0, atol=1e-2)
