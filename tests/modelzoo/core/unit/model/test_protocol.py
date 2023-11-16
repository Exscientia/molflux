"""
Tests for API features advertised as part of the Estimator protocol.
"""
from typing import Protocol

from molflux.modelzoo.protocols import Estimator


def test_class_is_protocol():
    """That is a protocol."""
    assert issubclass(type(Estimator), type(Protocol))


def test_has_init():
    """That can be initialised."""
    assert hasattr(Estimator, "__init__")
    assert callable(Estimator.__init__)


def test_has_metadata():
    """That has a metadata property."""
    assert hasattr(Estimator, "metadata")
    assert not callable(Estimator.metadata)


def test_has_name():
    """That has a name property."""
    assert hasattr(Estimator, "name")
    assert not callable(Estimator.name)


def test_has_tag():
    """That has a tag property."""
    assert hasattr(Estimator, "tag")
    assert not callable(Estimator.tag)


def test_has_x_features():
    """That has a x_features property."""
    assert hasattr(Estimator, "x_features")
    assert not callable(Estimator.x_features)


def test_has_y_features():
    """That has a y_features property."""
    assert hasattr(Estimator, "y_features")
    assert not callable(Estimator.y_features)


def test_has_predict():
    """That has a predict method."""
    assert hasattr(Estimator, "predict")
    assert callable(Estimator.predict)
