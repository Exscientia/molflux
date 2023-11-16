import math

import numpy as np
import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "leave_p_groups_out"


@pytest.fixture(scope="module")
def fixture_test_strategy():
    return load_splitting_strategy(strategy_name)


@pytest.fixture(scope="module")
def fixture_sample_inputs():
    dataset = range(10)
    groups = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    return dataset, groups


def test_is_in_catalogue():
    """That the strategy is registered in the catalogue."""
    catalogue = list_splitting_strategies()
    all_strategy_names = [name for names in catalogue.values() for name in names]
    assert strategy_name in all_strategy_names


def test_implements_protocol(fixture_test_strategy):
    """That the strategy implements the protocol."""
    strategy = fixture_test_strategy
    assert isinstance(strategy, SplittingStrategy)


def test_requires_groups(fixture_test_strategy):
    """That you need to provide groups to use this strategy."""
    dataset = [0, 1, 2, 3, 4, 5]
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset, p=2))


def test_requires_n_groups(fixture_test_strategy):
    """That you need to provide the p argument to use this strategy."""
    dataset = [0, 1, 2, 3, 4, 5]
    groups = [1, 2, 2, 3, 3, 3]
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset, groups=groups))


def test_leave_out_more_groups_than_existing_raises(
    fixture_sample_inputs,
    fixture_test_strategy,
):
    """That cannot perform cross validation leaving out more groups than there are."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    n_unique_groups = len(np.unique(groups))
    with pytest.raises(ValueError):
        next(strategy.split(dataset=dataset, groups=groups, p=n_unique_groups + 1))


def test_leave_out_all_existing_groups_raises(
    fixture_sample_inputs,
    fixture_test_strategy,
):
    """That cannot perform cross validation leaving out all groups."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    n_unique_groups = len(np.unique(groups))
    with pytest.raises(ValueError):
        next(strategy.split(dataset=dataset, groups=groups, p=n_unique_groups))


def test_leave_out_no_groups_raises(fixture_sample_inputs, fixture_test_strategy):
    """That cannot perform cross validation leaving out zero groups."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset=dataset, groups=groups, p=0))


def test_expected_result_one_leave_one_out(fixture_test_strategy):
    """That the splitting strategy yields the expected result in this case.

    Note, we are testing that the splits are not shuffled and that groups are
    held out sequentially in the validation set - starting with the first one.
    """
    dataset = range(10)
    groups = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, p=1)

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [2, 3, 4, 5, 6, 7, 8, 9]))
    assert all(i == j for i, j in zip(validation_indices, [0, 1]))

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [0, 1, 5, 6, 7, 8, 9]))
    assert all(i == j for i, j in zip(validation_indices, [2, 3, 4]))

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [0, 1, 2, 3, 4]))
    assert all(i == j for i, j in zip(validation_indices, [5, 6, 7, 8, 9]))


def test_expected_result_one_leave_two_out(fixture_test_strategy):
    """That the splitting strategy yields the expected result in this case."""
    dataset = range(10)
    groups = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, p=2)

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [5, 6, 7, 8, 9]))
    assert all(i == j for i, j in zip(validation_indices, [0, 1, 2, 3, 4]))

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [2, 3, 4]))
    assert all(i == j for i, j in zip(validation_indices, [0, 1, 5, 6, 7, 8, 9]))

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [0, 1]))
    assert all(i == j for i, j in zip(validation_indices, [2, 3, 4, 5, 6, 7, 8, 9]))


def test_does_not_produce_test_indices(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy is a cross validation strategy."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = list(strategy.split(dataset=dataset, groups=groups, p=2))
    assert len(indices) > 1


def test_yields_more_than_one_fold(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy yields several folds.

    We test leaving out the maximum amount of groups, which should result in
    the minimum number of folds generated.
    """
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    max_p = len(np.unique(groups)) - 1
    indices = list(strategy.split(dataset=dataset, groups=groups, p=max_p))
    for train_indices, validation_indices, test_indices in indices:
        assert len(train_indices)
        assert len(validation_indices)
        assert not len(test_indices)


def test_yields_correct_number_of_folds(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy yields the correct number of folds.

    This is expected to be the “N choose p”, as we make folds out of all possible
    combinations of p groups left out, out of N available groups.

    References:
        https://en.wikipedia.org/wiki/Binomial_coefficient
    """
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    n_unique_groups = len(np.unique(groups))
    p = 2
    indices = list(strategy.split(dataset=dataset, groups=groups, p=p))
    assert len(indices) == math.comb(n_unique_groups, p)


def test_splits_are_disjoint(fixture_sample_inputs, fixture_test_strategy):
    """That data is spread across splits without overlap."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, p=1)
    for train_indices, validation_indices, _ in indices:
        assert set(train_indices).isdisjoint(set(validation_indices))


def test_each_sample_in_a_different_fold(fixture_sample_inputs, fixture_test_strategy):
    """That each sample ends up in a different fold."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset, groups=groups, p=1)
    for train_indices, validation_indices, _ in indices:
        groups_in_train = set(groups[train_indices])
        groups_in_validation = set(groups[validation_indices])
        assert set(groups_in_train).isdisjoint(set(groups_in_validation))


def test_deterministic(fixture_sample_inputs, fixture_test_strategy):
    """That repeated splits are deterministic."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy

    # Do one iteration of k-fold splits
    indices_a = strategy.split(dataset, groups=groups, p=2)
    train_indices_a1, validation_indices_a1, _ = next(indices_a)
    train_indices_a2, validation_indices_a2, _ = next(indices_a)

    # And again...
    indices_b = strategy.split(dataset, groups=groups, p=2)
    train_indices_b1, validation_indices_b1, _ = next(indices_b)
    train_indices_b2, validation_indices_b2, _ = next(indices_b)

    # Check that we get the same results across folds
    assert all(i == j for i, j in zip(train_indices_a1, train_indices_b1))
    assert all(i == j for i, j in zip(validation_indices_a1, validation_indices_b1))
    assert all(i == j for i, j in zip(train_indices_a2, train_indices_b2))
    assert all(i == j for i, j in zip(validation_indices_a2, validation_indices_b2))
