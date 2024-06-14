import numpy as np
import pytest
from numpy.random import default_rng
from scipy.stats import norm

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.uncertainty.calibration_gap import (
    CalibrationGap,
)

rng = default_rng()


@pytest.fixture()
def fixture_metric():
    return load_metric("calibration_gap")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "calibration_gap" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, CalibrationGap)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_calibration_gap_uncalibrated(fixture_metric):
    predictions = [0, 0, 0, 1, 1, 1]
    references = [5, 5, 5, 10, 10, 10]
    standard_deviations = [1, 1, 1, 1, 1, 1]

    actual = fixture_metric.compute(
        predictions=predictions,
        references=references,
        standard_deviations=standard_deviations,
    )["calibration_gap"]
    expected = 0.5
    assert np.isclose(actual, expected, atol=0.01)


def test_calibration_gap_calibrated(fixture_metric):
    references = np.concatenate(
        [
            rng.normal(loc=0, scale=1, size=(10000)),
            rng.normal(loc=3, scale=2, size=(10000)),
        ],
    )
    predictions = np.concatenate([np.zeros(10000), 3 * np.ones(10000)])
    standard_deviations = np.concatenate([np.ones(10000), 2 * np.ones(10000)])

    actual = fixture_metric.compute(
        predictions=predictions,
        references=references,
        standard_deviations=standard_deviations,
    )["calibration_gap"]
    expected = 0.0
    assert np.isclose(actual, expected, atol=0.01)


def test_calibration_gap_single_prediction(fixture_metric):
    references = [
        0,
    ]
    predictions = [
        norm.ppf(0.7, loc=0, scale=1),
    ]  # 0.524..., 70% of samples from N(0, 1) are below this.
    standard_deviations = [1.0]

    actual = fixture_metric.compute(
        predictions=predictions,
        references=references,
        standard_deviations=standard_deviations,
        num_thresholds=1001,
    )["calibration_gap"]

    # the calibration curve is 0 <= 0.3 and 1 > 0.3.
    # the area is then the area of two right angle triangles
    # with base and height 0.3 and 0.7 summed.
    expected = 0.5 * (0.7**2 + 0.3**2)
    assert np.isclose(actual, expected, atol=0.01)
