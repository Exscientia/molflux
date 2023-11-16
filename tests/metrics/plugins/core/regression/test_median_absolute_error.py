import numpy as np
import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.regression.median_absolute_error import MedianAbsoluteError


@pytest.fixture()
def fixture_metric():
    return load_metric("median_absolute_error")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "median_absolute_error" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, MedianAbsoluteError)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_default_compute(fixture_metric):
    """That default scoring."""
    metric = fixture_metric
    predictions = [2.5, 0.0, 2, 8]
    references = [3, -0.5, 2, 7]
    result = metric.compute(predictions=predictions, references=references)
    assert result["median_absolute_error"] == 0.5


def test_perfect_score(fixture_metric):
    """Test score on perfect predictions."""
    metric = fixture_metric
    predictions = [3, -0.5, 2, 7]
    references = [3, -0.5, 2, 7]
    result = metric.compute(predictions=predictions, references=references)
    assert result["median_absolute_error"] == 0


def test_multioutpout_default(fixture_metric):
    """Multioutput case with errors of all outputs averaged with uniform weight."""
    metric = load_metric("median_absolute_error", config_name="multioutput")
    predictions = [[0, 2], [-1, 2], [8, -5]]
    references = [[0.5, 1], [-1, 1], [7, -6]]
    result = metric.compute(predictions=predictions, references=references)
    assert result
    assert pytest.approx(result["median_absolute_error"], 0.01) == 0.75


def test_multioutpout_raw_values(fixture_metric):
    """Multioutput case returning full set of errors."""
    metric = load_metric("median_absolute_error", config_name="multioutput")
    predictions = [[0, 2], [-1, 2], [8, -5]]
    references = [[0.5, 1], [-1, 1], [7, -6]]
    result = metric.compute(
        predictions=predictions,
        references=references,
        multioutput="raw_values",
    )
    assert result
    np.testing.assert_allclose([0.5, 1], result["median_absolute_error"])


def test_multioutpout_array_like(fixture_metric):
    """Multioutput case providing weights by which to average errors."""
    metric = load_metric("median_absolute_error", config_name="multioutput")
    predictions = [[0, 2], [-1, 2], [8, -5]]
    references = [[0.5, 1], [-1, 1], [7, -6]]
    result = metric.compute(
        predictions=predictions,
        references=references,
        multioutput=[0.3, 0.7],
    )
    assert result
    assert result["median_absolute_error"] == 0.85
