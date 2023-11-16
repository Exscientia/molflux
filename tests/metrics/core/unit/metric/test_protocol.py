"""
Tests for API features advertised as part of the Metric protocol.
"""
from typing import Protocol

from molflux.metrics.metric import Metric


def test_class_is_protocol():
    """That it is a protocol."""
    assert issubclass(type(Metric), type(Protocol))


def test_has_init():
    """That classes implementing the protocol can be initialised."""
    assert hasattr(Metric, "__init__")
    assert callable(Metric.__init__)


def test_has_len():
    """That it is sized."""
    assert hasattr(Metric, "__len__")
    assert callable(Metric.__len__)


def test_has_metadata():
    """That has a metadata property."""
    assert hasattr(Metric, "metadata")
    assert not callable(Metric.metadata)


def test_has_name():
    """That has a name property."""
    assert hasattr(Metric, "name")
    assert not callable(Metric.name)


def test_has_tag():
    """That has a tag property."""
    assert hasattr(Metric, "tag")
    assert not callable(Metric.tag)


def test_has_state():
    """That has a state property."""
    assert hasattr(Metric, "state")
    assert not callable(Metric.state)


def test_has_add_batch():
    """That has an add_batch method."""
    assert hasattr(Metric, "add_batch")
    assert callable(Metric.add_batch)


def test_has_compute():
    """That has a compute method."""
    assert hasattr(Metric, "compute")
    assert callable(Metric.compute)


def test_has_reset_state():
    """That has a reset_state method."""
    assert hasattr(Metric, "reset_state")
    assert callable(Metric.reset_state)


def test_has_udpate_state():
    """That has an update_state method."""
    assert hasattr(Metric, "update_state")
    assert callable(Metric.update_state)
