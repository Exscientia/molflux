import numpy as np
import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.classification.accuracy import Accuracy


@pytest.fixture()
def fixture_metric():
    return load_metric("accuracy")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "accuracy" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, Accuracy)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_integer_labels(fixture_metric):
    """That can use integer labels."""
    metric = fixture_metric
    predictions = [0, 2, 1, 3]
    references = [0, 1, 2, 3]
    result = metric.compute(predictions=predictions, references=references)
    assert result["accuracy"] == 0.5


def test_normalised_accuracy(fixture_metric):
    """That can calculate normalised accuracy."""
    metric = fixture_metric
    predictions = [0, 2, 1, 3]
    references = [0, 1, 2, 3]
    result = metric.compute(
        predictions=predictions,
        references=references,
        normalize=False,
    )
    assert result["accuracy"] == 2


def test_perfect_accuracy(fixture_metric):
    """That get unitary accuracy for perfect predictions."""
    metric = fixture_metric
    predictions = [0, 1]
    references = [0, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["accuracy"] == 1.0


def test_binary_multilabel():
    """Multilabel case with binary label indicators."""
    metric = load_metric("accuracy", config_name="multilabel")
    predictions = np.array([[0, 1], [1, 1]])
    references = np.ones((2, 2))
    result = metric.compute(predictions=predictions, references=references)
    assert result["accuracy"] == 0.5
