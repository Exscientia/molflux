import numpy as np
import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "group_k_fold"


@pytest.fixture(scope="module")
def fixture_test_strategy():
    return load_splitting_strategy(strategy_name)


@pytest.fixture(scope="module")
def fixture_sample_inputs():
    n = 100
    dataset = np.random.rand(n)
    groups = np.random.choice(["a", "b", "c"], size=n)
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


def test_yields_more_than_one_fold(fixture_sample_inputs, fixture_test_strategy):
    """That the splitting strategy yields several folds."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups)
    assert len(list(indices)) > 1


def test_cannot_set_only_one_fold(fixture_sample_inputs, fixture_test_strategy):
    """That cannot perform cross validation with only one fold."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset=dataset, groups=groups, n_splits=1))


def test_more_splits_than_groups_raises(fixture_test_strategy):
    """That cannot have more splits than number of unique groups."""
    dataset = [0, 1, 2, 3, 4, 5]
    groups = ["a", "b", "a", "a", "b", "c"]
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset, groups=groups, n_splits=4))


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
    indices = strategy.split(dataset, groups=groups, n_splits=2)
    for train_indices, validation_indices, _ in indices:
        groups_in_train = set(groups[train_indices])
        groups_in_validation = set(groups[validation_indices])
        assert set(groups_in_train).isdisjoint(set(groups_in_validation))


def test_deterministic(fixture_sample_inputs, fixture_test_strategy):
    """That repeated splits are deterministic."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy

    # Do one iteration of k-fold splits
    indices_a = strategy.split(dataset, groups=groups, n_splits=2)
    train_indices_a1, validation_indices_a1, _ = next(indices_a)
    train_indices_a2, validation_indices_a2, _ = next(indices_a)

    # And again...
    indices_b = strategy.split(dataset, groups=groups, n_splits=2)
    train_indices_b1, validation_indices_b1, _ = next(indices_b)
    train_indices_b2, validation_indices_b2, _ = next(indices_b)

    # Check that we get the same results across folds
    assert all(i == j for i, j in zip(train_indices_a1, train_indices_b1))
    assert all(i == j for i, j in zip(validation_indices_a1, validation_indices_b1))
    assert all(i == j for i, j in zip(train_indices_a2, train_indices_b2))
    assert all(i == j for i, j in zip(validation_indices_a2, validation_indices_b2))
