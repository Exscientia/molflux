import numpy.testing
import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.classification.diversity_roc import DiversityRoc

metric_name = "diversity_roc"


@pytest.fixture()
def fixture_metric():
    return load_metric(metric_name)


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "diversity_roc" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, DiversityRoc)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_string_labels(fixture_metric):
    """That can use string labels."""
    metric = fixture_metric
    predictions = [["CC", "C", "CC", "C", "CCC"]]
    references = [None]
    result = metric.compute(predictions=predictions, references=references)
    numpy.testing.assert_allclose([[1.0, 1.0, 2 / 3, 0.5, 0.6]], result[metric_name])


def test_none_labels(fixture_metric):
    """That can use None labels."""
    metric = fixture_metric
    predictions = [["A", "B", "B", None, "C"]]
    references = [None]
    result = metric.compute(predictions=predictions, references=references)
    numpy.testing.assert_allclose(
        [[1.0, 1.0, 2.0 / 3, 3.0 / 4, 4.0 / 5]],
        result[metric_name],
    )


def test_batch_diversity(fixture_metric):
    """That metric can be calculated on list of lists."""
    metric = fixture_metric
    predictions = [["A", "B", "B", None, "C"], ["D", "D", "D", "E", "F"]]
    references = [None, None]
    result = metric.compute(predictions=predictions, references=references)
    numpy.testing.assert_allclose(
        [[1.0, 1.0, 2.0 / 3, 3.0 / 4, 4.0 / 5], [1.0, 0.5, 1.0 / 3, 0.5, 3.0 / 5]],
        result[metric_name],
    )


def test_with_average(fixture_metric):
    """That switching on the average flag returns expected results."""
    metric = fixture_metric
    predictions = [["A", "B", "B", None, "C"], ["D", "D", "D", "E", "F"]]
    references = [None, None]
    result = metric.compute(
        predictions=predictions,
        references=references,
        average=True,
    )
    numpy.testing.assert_allclose([[1.0, 0.75, 0.5, 5.0 / 8, 0.7]], result[metric_name])
