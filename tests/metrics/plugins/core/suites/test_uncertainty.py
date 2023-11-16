import pytest

from molflux.metrics.load import load_suite
from molflux.metrics.metric import Metrics
from molflux.metrics.suites.catalogue import list_suites

suite_name = "uncertainty"


def test_in_catalogue():
    """That the suite is in the catalogue."""
    suites = list_suites()
    assert suite_name in suites


def test_loads():
    """That the suite can be loaded without errors."""
    assert load_suite(name=suite_name)


def test_is_metrics():
    """That the suite returns a metrics collection."""
    suite = load_suite(name=suite_name)
    assert isinstance(suite, Metrics)


def test_results_have_all_the_metrics_in():
    """Check that results have the uncertainty metrics."""
    ground_truth = [0, 0.3, 0.5, 0.8, 1]
    preds = [0.1, 0.35, 0.45, 0.68, 1.2]
    intervals = [(0, 0.5), (0.1, 0.4), (0.2, 0.5), (0.5, 0.7), (0.7, 1.5)]
    uq_suite = load_suite(name=suite_name)
    result = uq_suite.compute(
        references=ground_truth,
        predictions=preds,
        prediction_intervals=intervals,
    )
    all_metrics = [metric.name for metric in uq_suite]
    assert set(result.keys()) - set(all_metrics) == set()
    assert set(all_metrics) - set(result.keys()) == set()


def test_compute_works():
    """Check that computations work for a few of the
    metrics in the uncertainty suite."""
    ground_truth = [0, 0.3, 0.5, 0.8, 1]
    preds = [0.1, 0.35, 0.45, 0.68, 1.2]
    intervals = [(0, 0.5), (0.1, 0.4), (0.2, 0.5), (0.5, 0.7), (0.7, 1.5)]
    uq_suite = load_suite(name=suite_name)
    result = uq_suite.compute(
        references=ground_truth,
        predictions=preds,
        prediction_intervals=intervals,
    )
    assert pytest.approx(result["prediction_interval_width"], 0.01) == 0.42
    assert pytest.approx(result["prediction_interval_coverage"], 0.01) == 0.8
