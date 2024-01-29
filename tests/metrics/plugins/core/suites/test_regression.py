import pytest

from molflux.metrics.load import load_suite
from molflux.metrics.metric import Metrics
from molflux.metrics.suites.catalogue import list_suites

suite_name = "regression"


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
    """Check that results have the regression metrics."""
    ground_truth = [0, 0.3, 0.5, 0.8, 1]
    preds = [0.1, 0.35, 0.45, 0.68, 1.2]
    suite = load_suite(name=suite_name)
    result = suite.compute(references=ground_truth, predictions=preds)
    all_metrics = [metric.name for metric in suite]
    assert set(result.keys()) - set(all_metrics) == {
        "spearman::p_value",
        "spearman::correlation",
        "pearson::p_value",
        "pearson::correlation",
    }
    assert set(all_metrics) - set(result.keys()) == {"pearson", "spearman"}


def test_compute_works():
    """Check that computations work for a few of the
    metrics in the regression suite."""
    ground_truth = [0, 0.3, 0.5, 0.8, 1]
    preds = [0.1, 0.35, 0.45, 0.68, 1.2]
    suite = load_suite(name=suite_name)
    result = suite.compute(references=ground_truth, predictions=preds)
    assert pytest.approx(result["r2"], 0.01) == 0.88949
    assert pytest.approx(result["mean_absolute_error"], 0.01) == 0.10400
