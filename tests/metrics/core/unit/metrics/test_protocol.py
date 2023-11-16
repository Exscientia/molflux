"""
Tests for API features advertised as part of the Metrics protocol.
"""
from molflux.metrics.metric import Metrics


def test_has_init():
    """That can be initialised."""
    assert hasattr(Metrics, "__init__")
    assert callable(Metrics.__init__)


def test_has_getitem():
    """That can get items."""
    assert hasattr(Metrics, "__getitem__")
    assert callable(Metrics.__getitem__)


def test_has_iter():
    """That can be iterated over."""
    assert hasattr(Metrics, "__iter__")
    assert callable(Metrics.__iter__)


def test_has_setitem():
    """That can set items."""
    assert hasattr(Metrics, "__setitem__")
    assert callable(Metrics.__setitem__)


def test_has_add_batch():
    """That has an add_batch method."""
    assert hasattr(Metrics, "add_batch")
    assert callable(Metrics.add_batch)


def test_has_add_metric():
    """That has an add_metric method."""
    assert hasattr(Metrics, "add_metric")
    assert callable(Metrics.add_metric)


def test_has_compute():
    """That has a compute method."""
    assert hasattr(Metrics, "compute")
    assert callable(Metrics.compute)
