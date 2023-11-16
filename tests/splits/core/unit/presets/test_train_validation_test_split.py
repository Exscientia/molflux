import pytest

from molflux.splits.presets import train_validation_test_split
from molflux.splits.strategies.core.group_shuffle_split import GroupShuffleSplit
from molflux.splits.strategies.core.linear_split import LinearSplit
from molflux.splits.strategies.core.shuffle_split import ShuffleSplit
from molflux.splits.strategies.core.stratified_shuffle_split import (
    StratifiedShuffleSplit,
)


def test_returns_shuffle_split_by_default():
    """That a ShuffleSplit strategy is returned by default"""
    strategy = train_validation_test_split()
    assert isinstance(strategy, ShuffleSplit)


def test_strategy_generates_train_validation_test_splits():
    """That train, validation, test splits are generated."""
    strategy = train_validation_test_split()
    data = range(100)
    folds = strategy.split(data)
    train_indices, validation_indices, test_indices = next(folds)
    assert len(list(train_indices))
    assert len(list(validation_indices))
    assert len(list(test_indices))


def test_returns_shuffle_split_if_shuffle_and_not_stratified_and_not_grouped():
    """That a ShuffleSplit strategy is returned if 'shuffle' is True,
    'stratified' is False, and 'grouped' is False."""
    strategy = train_validation_test_split(
        shuffle=True,
        stratified=False,
        grouped=False,
    )
    assert isinstance(strategy, ShuffleSplit)


def test_returns_stratified_shuffle_split_if_shuffle_and_stratified_and_not_grouped():
    """That a StratifiedShuffleSplit strategy is returned if 'shuffle' is True', and
    'stratified' is True, and 'grouped' is False."""
    strategy = train_validation_test_split(shuffle=True, stratified=True, grouped=False)
    assert isinstance(strategy, StratifiedShuffleSplit)


def test_returns_group_shuffle_split_if_shuffle_and_not_stratified_and_grouped():
    """That a GroupShuffleSplit strategy is returned if 'shuffle' is True,
    'stratified' is False, and 'grouped' is True."""
    strategy = train_validation_test_split(shuffle=True, stratified=False, grouped=True)
    assert isinstance(strategy, GroupShuffleSplit)


def test_returns_linear_split_if_not_shuffle_and_not_stratified():
    """That a LinearSplit strategy is returned if 'shuffle' is True', and
    'stratified' is False."""
    strategy = train_validation_test_split(
        shuffle=False,
        stratified=False,
        grouped=True,
    )
    assert isinstance(strategy, LinearSplit)
    strategy = train_validation_test_split(
        shuffle=False,
        stratified=False,
        grouped=False,
    )
    assert isinstance(strategy, LinearSplit)


def test_raises_on_not_implemented_combination():
    """That an exception is raised if requesting an unsupported combination of
    'shuffle', 'stratified', and 'grouped' parameters."""
    with pytest.raises(NotImplementedError):
        train_validation_test_split(shuffle=False, stratified=True)
