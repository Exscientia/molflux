import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.uncertainty.prediction_interval_width import (
    PredictionIntervalWidth,
)


@pytest.fixture()
def fixture_metric():
    return load_metric("prediction_interval_width")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "prediction_interval_width" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, PredictionIntervalWidth)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_default_compute(fixture_metric):
    """That prediction interval width computes properly."""
    metric = fixture_metric
    predictions = [0, 0, 0, 0]
    references = [0.9, 0.1, 0.2, 0.3]
    lower_bound = [0.5, 0.5, 0.0, 0.0]
    upper_bound = [1.0, 1.0, 0.5, 0.5]
    prediction_intervals = list(zip(lower_bound, upper_bound))
    result = metric.compute(
        predictions=predictions,
        references=references,
        prediction_intervals=prediction_intervals,
    )

    assert result["prediction_interval_width"] >= 0


def test_raise_error_no_prediction_interval(fixture_metric):
    """That omitting prediction interval raises error."""
    metric = fixture_metric
    predictions = [3, 0.0, 2]
    references = [3, -0.0, 1]
    # prediction_intervals bound argument omitted
    with pytest.raises(RuntimeError):
        metric.compute(predictions=predictions, references=references)
