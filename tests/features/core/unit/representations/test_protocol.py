"""
Tests for API features advertised as part of the Representations protocol.
"""
from molflux.features.representation import Representations


def test_has_add_representation():
    """That Representations has add_representation method."""
    assert hasattr(Representations, "add_representation")
    assert callable(Representations.add_representation)


def test_has_featurise():
    """That Representations has a featurise method."""
    assert hasattr(Representations, "featurise")
    assert callable(Representations.featurise)
