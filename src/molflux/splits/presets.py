"""
Wrappers for generating commonly used splitting strategies.
"""

from typing import Optional

from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy


def train_validation_test_split(
    n_splits: int = 1,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: Optional[int] = None,
    shuffle: bool = True,
    stratified: bool = False,
    grouped: bool = False,
) -> SplittingStrategy:
    if shuffle and not stratified and not grouped:
        strategy_name = "shuffle_split"
    elif shuffle and stratified and not grouped:
        strategy_name = "stratified_shuffle_split"
    elif shuffle and not stratified and grouped:
        strategy_name = "group_shuffle_split"
    elif not shuffle and not stratified:
        strategy_name = "linear_split"
    else:
        raise NotImplementedError("The requested splitting strategy is not available.")

    strategy = load_splitting_strategy(name=strategy_name)

    strategy.update_state(
        n_splits=n_splits,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    return strategy


def train_test_split(
    n_splits: int = 1,
    train_fraction: float = 0.8,
    seed: Optional[int] = None,
    shuffle: bool = True,
    stratified: bool = False,
    grouped: bool = False,
) -> SplittingStrategy:
    return train_validation_test_split(
        n_splits=n_splits,
        train_fraction=train_fraction,
        validation_fraction=0,
        test_fraction=1 - train_fraction,
        seed=seed,
        shuffle=shuffle,
        stratified=stratified,
        grouped=grouped,
    )


def k_fold_split(
    n_splits: int = 5,
    shuffle: bool = False,
    seed: Optional[int] = None,
    stratified: bool = False,
    grouped: bool = False,
) -> SplittingStrategy:
    if stratified and not grouped:
        strategy_name = "stratified_k_fold"
    elif grouped and not stratified:
        strategy_name = "group_k_fold"
    elif not stratified and not grouped:
        strategy_name = "k_fold"
    else:
        raise NotImplementedError("The requested splitting strategy is not available.")

    strategy = load_splitting_strategy(name=strategy_name)

    strategy.update_state(
        n_splits=n_splits,
        shuffle=shuffle,
        seed=seed,
    )

    return strategy
