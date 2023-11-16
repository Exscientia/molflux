"""
Tests for API features advertised as part of the Representation protocol.
"""
from typing import Protocol

from molflux.features.representation import Representation


def test_class_is_protocol():
    """That Representation is a protocol."""
    assert issubclass(type(Representation), type(Protocol))


def test_has_init():
    """That Representation can be initialised."""
    assert hasattr(Representation, "__init__")
    assert callable(Representation.__init__)


def test_has_metadata():
    """That Representation has a metadata property."""
    assert hasattr(Representation, "metadata")
    assert not callable(Representation.metadata)


def test_has_name():
    """That Representation has a name property."""
    assert hasattr(Representation, "name")
    assert not callable(Representation.name)


def test_has_tag():
    """That Representation has a tag property."""
    assert hasattr(Representation, "tag")
    assert not callable(Representation.tag)


def test_has_featurise():
    """That Representation has a featurise method."""
    assert hasattr(Representation, "featurise")
    assert callable(Representation.featurise)


def test_has_reset_state():
    """That Representation has a reset_state method."""
    assert hasattr(Representation, "reset_state")
    assert callable(Representation.reset_state)


def test_has_udpate_state():
    """That Representation has an update_state method."""
    assert hasattr(Representation, "update_state")
    assert callable(Representation.update_state)
