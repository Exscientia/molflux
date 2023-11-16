import numpy as np
import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "leave_one_group_out"


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
        next(strategy.split(dataset))


def test_raises_if_only_one_group(fixture_test_strategy):
    """That you need to provide at least two groups to use this strategy."""
    dataset = [0, 1, 2, 3, 4, 5]
    groups = [1, 1, 1, 1, 1, 1]
    strategy = fixture_test_strategy
    with pytest.raises(
        ValueError,
        match="The groups parameter contains fewer than 2 unique groups",
    ):
        next(strategy.split(dataset, groups=groups))


def test_expected_result(fixture_test_strategy):
    """That the splitting strategy yields the expected result in this case.

    Note, we are testing that the splits are not shuffled and that groups are
    held out sequentially in the validation set - starting with the group that
    has the lowest index in the groups array.
    """
    dataset = range(10)
    groups = [2, 2, 2, 3, 3, 3, 3, 3, 1, 1]
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups)

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [0, 1, 2, 3, 4, 5, 6, 7]))
    assert all(i == j for i, j in zip(validation_indices, [8, 9]))

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [3, 4, 5, 6, 7, 8, 9]))
    assert all(i == j for i, j in zip(validation_indices, [0, 1, 2]))

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [0, 1, 2, 8, 9]))
    assert all(i == j for i, j in zip(validation_indices, [3, 4, 5, 6, 7]))


def test_does_not_produce_test_indices(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy is a cross validation strategy."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = list(strategy.split(dataset=dataset, groups=groups))
    assert len(indices) > 1


def test_yields_more_than_one_fold(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy yields several folds."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = list(strategy.split(dataset=dataset, groups=groups))
    for train_indices, validation_indices, test_indices in indices:
        assert len(train_indices)
        assert len(validation_indices)
        assert not len(test_indices)


def test_yields_correct_number_of_folds(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy yields the correct number of folds.

    There should be as many folds as there are number of unique groups.
    """
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    n_unique_groups = len(np.unique(groups))
    indices = list(strategy.split(dataset=dataset, groups=groups))
    assert len(indices) == n_unique_groups


def test_splits_are_disjoint(fixture_sample_inputs, fixture_test_strategy):
    """That data is spread across splits without overlap."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups)
    for train_indices, validation_indices, _ in indices:
        assert set(train_indices).isdisjoint(set(validation_indices))


def test_each_sample_in_a_different_fold(fixture_sample_inputs, fixture_test_strategy):
    """That each sample ends up in a different fold."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset, groups=groups)
    for train_indices, validation_indices, _ in indices:
        groups_in_train = set(groups[train_indices])
        groups_in_validation = set(groups[validation_indices])
        assert set(groups_in_train).isdisjoint(set(groups_in_validation))


def test_deterministic(fixture_sample_inputs, fixture_test_strategy):
    """That repeated splits are deterministic."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy

    # Do one iteration of k-fold splits
    indices_a = strategy.split(dataset, groups=groups)
    train_indices_a1, validation_indices_a1, _ = next(indices_a)
    train_indices_a2, validation_indices_a2, _ = next(indices_a)

    # And again...
    indices_b = strategy.split(dataset, groups=groups)
    train_indices_b1, validation_indices_b1, _ = next(indices_b)
    train_indices_b2, validation_indices_b2, _ = next(indices_b)

    # Check that we get the same results across folds
    assert all(i == j for i, j in zip(train_indices_a1, train_indices_b1))
    assert all(i == j for i, j in zip(validation_indices_a1, validation_indices_b1))
    assert all(i == j for i, j in zip(train_indices_a2, train_indices_b2))
    assert all(i == j for i, j in zip(validation_indices_a2, validation_indices_b2))


def test_string_groups(fixture_test_strategy):
    """That the splitting strategy yields the expected result when using string
    group labels.

    This would be the use case when using this splitting strategy to e.g. leave
    one 'project' out. We expect the groups to be assigned to the validation
    split in alphabetical order.
    """
    dataset = range(10)
    groups = [
        "P00-00002",
        "P00-00002",
        "P00-00002",
        "P00-00003",
        "P00-00003",
        "P00-00003",
        "P00-00003",
        "P00-00003",
        "P00-00001",
        "P00-00001",
    ]
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups)

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [0, 1, 2, 3, 4, 5, 6, 7]))
    assert all(i == j for i, j in zip(validation_indices, [8, 9]))

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [3, 4, 5, 6, 7, 8, 9]))
    assert all(i == j for i, j in zip(validation_indices, [0, 1, 2]))

    train_indices, validation_indices, _ = next(indices)
    assert all(i == j for i, j in zip(train_indices, [0, 1, 2, 8, 9]))
    assert all(i == j for i, j in zip(validation_indices, [3, 4, 5, 6, 7]))
