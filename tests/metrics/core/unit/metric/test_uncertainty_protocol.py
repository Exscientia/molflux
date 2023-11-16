from molflux.metrics import load_metric
from molflux.metrics.protocols import supports_prediction_intervals


def test_supports_prediction_intervals_positive_example():
    """this example should support uncertainty"""
    metric = load_metric(name="gaussian_nll")
    assert supports_prediction_intervals(metric)


def test_supports_prediction_intervals_negative_example():
    """this example should not support uncertainty"""
    metric = load_metric(name="spearman")
    assert not supports_prediction_intervals(metric)
