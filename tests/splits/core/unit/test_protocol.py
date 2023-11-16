"""
Tests for API features advertised as part of the SplittingStrategy protocol.
"""
from typing import Protocol

from molflux.splits.strategy import SplittingStrategy


def test_splitting_strategy_is_protocol():
    """That SplittingStrategy is a protocol."""
    assert issubclass(type(SplittingStrategy), type(Protocol))


def test_splitting_strategy_has_metadata():
    """That SplittingStrategy has a metadata property."""
    assert hasattr(SplittingStrategy, "metadata")
    assert not callable(SplittingStrategy.metadata)


def test_splitting_strategy_has_name():
    """That SplittingStrategy has a name property."""
    assert hasattr(SplittingStrategy, "name")
    assert not callable(SplittingStrategy.name)


def test_splitting_strategy_has_state():
    """That SplittingStrategy has a state property."""
    assert hasattr(SplittingStrategy, "state")
    assert not callable(SplittingStrategy.state)


def test_splitting_strategy_has_tag():
    """That SplittingStrategy has a tag property."""
    assert hasattr(SplittingStrategy, "tag")
    assert not callable(SplittingStrategy.tag)


def test_splitting_strategy_has_init():
    """That SplittingStrategy has a __init__ method."""
    assert hasattr(SplittingStrategy, "__init__")
    assert callable(SplittingStrategy.__init__)


def test_splitting_strategy_has_split():
    """That SplittingStrategy has a split method."""
    assert hasattr(SplittingStrategy, "split")
    assert callable(SplittingStrategy.split)


def test_splitting_strategy_has_reset_state():
    """That SplittingStrategy has a reset_state method."""
    assert hasattr(SplittingStrategy, "reset_state")
    assert callable(SplittingStrategy.reset_state)


def test_splitting_strategy_has_update_state():
    """That SplittingStrategy has an update_state method."""
    assert hasattr(SplittingStrategy, "update_state")
    assert callable(SplittingStrategy.update_state)
