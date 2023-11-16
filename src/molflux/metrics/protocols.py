from molflux.metrics.bases import PredictionIntervalMetric
from molflux.metrics.metric import Metric, Metrics


def supports_prediction_intervals(metric: Metric) -> bool:
    """Return True if the given metric supports uncertainty predictions."""
    return isinstance(metric, PredictionIntervalMetric)


def all_support_prediction_intervals(metrics: Metrics) -> bool:
    """Return True if all the given metric supports uncertainty predictions."""
    return all(supports_prediction_intervals(metric) for metric in metrics)
