import itertools

import numpy as np
import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "stratified_shuffle_split"


@pytest.fixture(scope="module")
def fixture_test_strategy():
    return load_splitting_strategy(strategy_name)


@pytest.fixture(scope="module")
def fixture_sample_inputs():
    n = 100
    dataset = np.random.rand(n)
    classes = np.random.choice([0, 1, 2], size=n)
    return dataset, classes


def test_is_in_catalogue():
    """That the strategy is registered in the catalogue."""
    catalogue = list_splitting_strategies()
    all_strategy_names = [name for names in catalogue.values() for name in names]
    assert strategy_name in all_strategy_names


def test_implements_protocol(fixture_test_strategy):
    """That the strategy implements the protocol."""
    strategy = fixture_test_strategy
    assert isinstance(strategy, SplittingStrategy)


def test_requires_classes(fixture_test_strategy):
    """That classes need to be provided for stratification strategy."""
    dataset = [0, 1, 2, 3, 4, 5]
    strategy = fixture_test_strategy
    with pytest.raises(ValueError, match="The 'y' parameter should not be None"):
        next(strategy.split(dataset))


def test_accepts_integer_class_labels(fixture_test_strategy):
    """That the splitting strategy accepts integer class labels."""
    n = 100
    dataset = np.random.rand(n)
    classes = np.random.choice([0, 1, 2], size=n)
    strategy = fixture_test_strategy
    next(strategy.split(dataset, y=classes))
    assert True


def test_accepts_string_class_labels(fixture_test_strategy):
    """That the splitting strategy accepts string class labels."""
    n = 100
    dataset = np.random.rand(n)
    classes = np.random.choice(["cat", "dog", "snake"], size=n)
    strategy = fixture_test_strategy
    next(strategy.split(dataset, y=classes))
    assert True


def test_yields_one_fold_by_default(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy only yields one set of splits by default."""
    dataset, classes = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset, y=classes)
    assert len(list(indices)) == 1


def test_can_yield_multiple_folds(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy can generate multiple folds."""
    dataset, classes = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset, y=classes, n_splits=2)
    assert len(list(indices)) == 2


def test_non_deterministic_split(fixture_sample_inputs, fixture_test_strategy):
    """That split results are non deterministic."""
    dataset, classes = fixture_sample_inputs
    strategy = fixture_test_strategy

    indices = strategy.split(dataset, y=classes, n_splits=2)
    train_indices1, validation_indices1, test_indices1 = next(indices)
    train_indices2, validation_indices2, test_indices2 = next(indices)

    assert any(i != j for i, j in zip(train_indices1, train_indices2))
    assert any(i != j for i, j in zip(validation_indices1, validation_indices2))
    assert any(i != j for i, j in zip(test_indices1, test_indices2))


def test_deterministic_split_if_seed_set(fixture_sample_inputs, fixture_test_strategy):
    """That split results are deterministic if a seed is set."""
    dataset, classes = fixture_sample_inputs
    strategy = fixture_test_strategy

    indices1 = strategy.split(dataset, y=classes, seed=123)
    train_indices1, validation_indices1, test_indices1 = next(indices1)

    indices2 = strategy.split(dataset, y=classes, seed=123)
    train_indices2, validation_indices2, test_indices2 = next(indices2)

    assert all(i == j for i, j in zip(train_indices1, train_indices2))
    assert all(i == j for i, j in zip(validation_indices1, validation_indices2))
    assert all(i == j for i, j in zip(test_indices1, test_indices2))


def test_splits_are_shuffled_across_folds(fixture_sample_inputs, fixture_test_strategy):
    """That even if a seed is set, output folds are individually shuffled.

    This is to align with the expected behaviour in the scikit-learn API.
    """
    dataset, classes = fixture_sample_inputs
    strategy = fixture_test_strategy

    indices1 = strategy.split(dataset=dataset, y=classes, n_splits=2, seed=123)
    train_indices1, validation_indices1, test_indices1 = next(indices1)
    train_indices2, validation_indices2, test_indices2 = next(indices1)

    assert any(i != j for i, j in zip(train_indices1, train_indices2))
    assert any(i != j for i, j in zip(validation_indices1, validation_indices2))
    assert any(i != j for i, j in zip(test_indices1, test_indices2))


def test_splits_are_disjoint(fixture_sample_inputs, fixture_test_strategy):
    """That data is spread across splits without overlap."""
    dataset, classes = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset, y=classes)
    train_indices, validation_indices, test_indices = next(indices)
    assert set(train_indices).isdisjoint(set(validation_indices))
    assert set(train_indices).isdisjoint(set(test_indices))
    assert set(validation_indices).isdisjoint(set(test_indices))


def test_inconsistent_split_fractions_raise(
    fixture_sample_inputs,
    fixture_test_strategy,
):
    dataset, classes = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        y=classes,
        train_fraction=0.6,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    with pytest.raises(AssertionError):
        next(indices)


def test_not_missing_some_indices(fixture_test_strategy):
    """
    For small datasets and specific conditions, the output splits were missing
    some indices. We test against a specific example here to catch deviations
    from a known state, but we don't rule out that the exact results may
    change over time (e.g. for changes in seeding algorithm)
    """

    dataset = range(10)
    classes = ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"]

    strategy = fixture_test_strategy
    indices = strategy.split(
        n_splits=1,
        dataset=dataset,
        y=classes,
        train_fraction=0.8,
        validation_fraction=0.0,
        test_fraction=0.2,
        seed=1,
    )
    train_indices, validation_indices, test_indices = next(indices)

    assert len(
        list(train_indices) + list(validation_indices) + list(test_indices),
    ) == len(dataset)

    assert list(train_indices) == [8, 4, 6, 5, 9, 2, 3, 0]
    assert list(validation_indices) == []
    assert list(test_indices) == [7, 1]


def test_not_always_the_same_indices_in_each_split(fixture_test_strategy):
    """That the class indices dispatched to each split are being shuffled across
     folds.

    We don't want something like this happening:

        [8, 3, 1, 7, 0, 5, 4, 6] [] [2, 9]
        [3, 4, 5, 6, 1, 7, 0, 8] [] [9, 2]
        [8, 6, 4, 0, 7, 5, 3, 1] [] [9, 2]
        [0, 8, 1, 4, 5, 3, 7, 6] [] [2, 9]
        [3, 8, 4, 5, 1, 6, 0, 7] [] [2, 9]
        ...

    but rather something like this, where it's not always the same class
    indices being dispatched to each split:

        [8 4 6 5 9 2 3 0] [] [7 1]
        [2 5 0 9 8 4 3 6] [] [1 7]
        [3 5 7 4 6 1 8 0] [] [2 9]
        [7 6 4 0 8 9 5 1] [] [2 3]
        [5 8 6 4 2 0 9 7] [] [1 3]
        ...

    For large enough number of iterations, all indices should end up having been
    seen each at least once in all (non-empty) splits.
    """

    dataset = range(10)
    classes = ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"]

    # pick a large enough number to make all indices show up in all splits
    # over time
    n_splits = 20

    strategy = fixture_test_strategy
    indices = strategy.split(
        n_splits=n_splits,
        dataset=dataset,
        y=classes,
        train_fraction=0.8,
        validation_fraction=0.0,
        test_fraction=0.2,
        seed=1,
    )

    all_test_indices = set(itertools.chain.from_iterable(t for _, _, t in indices))
    for idx in range(len(dataset)):
        assert idx in all_test_indices
