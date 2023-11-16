"""
Tests ensuring desired API objects are part of the top-level namespace.
"""

import molflux.features


def test_exports_list_representations():
    """That the package exposes the list_representations function."""
    assert hasattr(molflux.features, "list_representations")
    assert callable(molflux.features.list_representations)


def test_exports_load_from_dict():
    """That the package exposes the load_from_dict function."""
    assert hasattr(molflux.features, "load_from_dict")
    assert callable(molflux.features.load_from_dict)


def test_exports_load_from_dicts():
    """That the package exposes the load_from_dicts function."""
    assert hasattr(molflux.features, "load_from_dicts")
    assert callable(molflux.features.load_from_dicts)


def test_exports_load_from_yaml():
    """That the package exposes the load_from_yaml function."""
    assert hasattr(molflux.features, "load_from_yaml")
    assert callable(molflux.features.load_from_yaml)


def test_exports_load_representation():
    """That the package exposes the load_representation function."""
    assert hasattr(molflux.features, "load_representation")
    assert callable(molflux.features.load_representation)


def test_exports_register_representation():
    """That the package exposes the register_representation function."""
    assert hasattr(molflux.features, "register_representation")
    assert callable(molflux.features.register_representation)


def test_exports_representation():
    """That the package exposes the Representation protocol."""
    assert hasattr(molflux.features, "Representation")


def test_exports_representations():
    """That the package exposes the Representations protocol."""
    assert hasattr(molflux.features, "Representations")
