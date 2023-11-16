import pytest

from molflux.splits.presets import k_fold_split
from molflux.splits.strategies.core.group_k_fold import GroupKFold
from molflux.splits.strategies.core.k_fold import KFold
from molflux.splits.strategies.core.stratified_k_fold import StratifiedKFold


def test_returns_k_fold_split_by_default():
    """That a ShuffleSplit strategy is returned by default"""
    strategy = k_fold_split()
    assert isinstance(strategy, KFold)


def test_returns_k_fold_if_not_stratified_and_not_grouped():
    """That a KFold strategy is returned if 'stratified' is False, and 'grouped' is False."""
    strategy = k_fold_split(stratified=False, grouped=False)
    assert isinstance(strategy, KFold)


def test_returns_group_k_fold_if_not_stratified_and_grouped():
    """That a GroupKFold strategy is returned if 'stratified' is False, and 'grouped' is True."""
    strategy = k_fold_split(stratified=False, grouped=True)
    assert isinstance(strategy, GroupKFold)


def test_returns_stratified_k_fold_if_stratified_and_not_grouped():
    """That a StratifiedKFold strategy is returned if 'stratified' is True, and 'grouped' is False."""
    strategy = k_fold_split(stratified=True, grouped=False)
    assert isinstance(strategy, StratifiedKFold)


def test_raises_on_not_implemented_combination():
    """That an exception is raised if requesting an unsupported combination of
    'stratified', and 'grouped' parameters."""
    with pytest.raises(NotImplementedError):
        k_fold_split(stratified=True, grouped=True)
