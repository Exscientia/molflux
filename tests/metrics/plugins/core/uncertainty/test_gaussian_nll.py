import numpy as np
import pytest
from scipy.stats import norm

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.uncertainty.gaussian_nll import GaussianNLL


@pytest.fixture()
def fixture_metric():
    return load_metric("gaussian_nll")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "gaussian_nll" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, GaussianNLL)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_default_compute(fixture_metric):
    """That negative log likelihood computes properly."""
    metric = fixture_metric
    predictions = [0, 1, 2, 3, 4]
    references = [0.5, 0.8, 1.2, 2.4, 4.5]
    lower_bound = [0.1, 0.1, 0.2, 2, 3]
    upper_bound = [1, 1.5, 1.8, 2.8, 5.7]
    prediction_intervals = list(zip(lower_bound, upper_bound))
    result = metric.compute(
        predictions=predictions,
        references=references,
        prediction_intervals=prediction_intervals,
    )

    assert pytest.approx(result["gaussian_nll"], 0.0001) == 1.294903148251583


def test_exact_compute(fixture_metric):
    """That the negative log likelihood matches the mathematical equation."""
    metric = fixture_metric
    predictions = [0, 1, 2, 3, 4]
    references = [1, 2, 3, 4, 5]
    lower_bound = [0, 1, 2, 3, 4]
    upper_bound = [2, 3, 4, 5, 6]
    prediction_intervals = list(zip(lower_bound, upper_bound))
    confidence = 1 - 2 * (1 - norm.cdf(1))
    result = metric.compute(
        predictions=predictions,
        references=references,
        prediction_intervals=prediction_intervals,
        confidence=confidence,
    )

    assert pytest.approx(result["gaussian_nll"], 0.0001) == 1 / 2 * (
        np.log(2) + np.log(np.pi) + 1
    )


def test_raise_error_no_prediction_interval(fixture_metric):
    """That omitting prediction interval raises error."""
    metric = fixture_metric
    predictions = [3, 0.0, 2]
    references = [3, -0.0, 1]
    # prediction_intervals bound argument omitted
    with pytest.raises(RuntimeError):
        metric.compute(predictions=predictions, references=references)
