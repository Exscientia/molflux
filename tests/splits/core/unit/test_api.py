"""
Tests for API features advertised as part of top level package namespace.
"""
import molflux.splits


def test_exports_list_splitting_strategies():
    """That the package exposes the list_splitting_strategies function."""
    assert hasattr(molflux.splits, "list_splitting_strategies")


def test_exports_k_fold_split():
    """That the package exposes the k_fold_split function."""
    assert hasattr(molflux.splits, "k_fold_split")


def test_exports_train_test_split():
    """That the package exposes the train_test_split function."""
    assert hasattr(molflux.splits, "train_test_split")


def test_exports_train_validation_test_split():
    """That the package exposes the k_fold_split function."""
    assert hasattr(molflux.splits, "train_validation_test_split")


def test_exports_load_from_dict():
    """That the package exposes the load_from_dict function."""
    assert hasattr(molflux.splits, "load_from_dict")


def test_exports_load_from_dicts():
    """That the package exposes the load_from_dicts function."""
    assert hasattr(molflux.splits, "load_from_dicts")


def test_exports_load_from_yaml():
    """That the package exposes the load_from_yaml function."""
    assert hasattr(molflux.splits, "load_from_yaml")


def test_exports_load_splitting_strategy():
    """That the package exposes the load_splitting_strategy function."""
    assert hasattr(molflux.splits, "load_splitting_strategy")


def test_exports_splitting_strategy():
    """That the package exposes the SplittingStrategy protocol."""
    assert hasattr(molflux.splits, "SplittingStrategy")
