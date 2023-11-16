from molflux.metrics.load import load_suite
from molflux.metrics.metric import Metrics
from molflux.metrics.suites.catalogue import list_suites

suite_name = "classification"


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
