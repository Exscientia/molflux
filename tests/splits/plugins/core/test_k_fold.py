import numpy as np
import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "k_fold"


@pytest.fixture(scope="module")
def fixture_test_strategy():
    return load_splitting_strategy(strategy_name)


@pytest.fixture(scope="module")
def fixture_sample_dataset():
    return np.random.rand(100)


def test_is_in_catalogue():
    """That the strategy is registered in the catalogue."""
    catalogue = list_splitting_strategies()
    all_strategy_names = [name for names in catalogue.values() for name in names]
    assert strategy_name in all_strategy_names


def test_implements_protocol(fixture_test_strategy):
    """That the strategy implements the protocol."""
    strategy = fixture_test_strategy
    assert isinstance(strategy, SplittingStrategy)


def test_yields_more_than_one_fold(fixture_sample_dataset, fixture_test_strategy):
    """That the splitting strategy only yields several folds."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset)
    assert len(list(indices)) > 1


def test_cannot_set_only_one_fold(fixture_sample_dataset, fixture_test_strategy):
    """That cannot perform cross validation with only one fold."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset=dataset, n_splits=1))


def test_raises_on_inconsistent_shuffle_and_seed(
    fixture_sample_dataset,
    fixture_test_strategy,
):
    """That an error is raised if seed is set but shuffle is not requested."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset=dataset, seed=123, shuffle=False))


def test_split_fractions(fixture_sample_dataset, fixture_test_strategy):
    """That the k folds are of expected sizes."""
    dataset = fixture_sample_dataset
    n_samples = len(dataset)
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, n_splits=4)
    for train_indices, validation_indices, test_indices in indices:
        assert len(list(train_indices)) == 0.75 * n_samples
        assert len(list(validation_indices)) == 0.25 * n_samples
        assert len(list(test_indices)) == 0


def test_splits_are_disjoint(fixture_sample_dataset, fixture_test_strategy):
    """That data is spread across splits without overlap."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset)
    for train_indices, validation_indices, _ in indices:
        assert set(train_indices).isdisjoint(set(validation_indices))


def test_unshuffled_by_default(fixture_test_strategy):
    """That by default the data is not shuffled before splitting."""
    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    strategy = fixture_test_strategy

    indices = strategy.split(dataset=dataset, n_splits=2)
    train_indices1, validation_indices1, _ = next(indices)
    train_indices2, validation_indices2, _ = next(indices)

    # We expect KFold to assign data to the validation set starting from the top

    # fold 1
    assert all(i == j for i, j in zip(train_indices1, [5, 6, 7, 8, 9]))
    assert all(i == j for i, j in zip(validation_indices1, [0, 1, 2, 3, 4]))

    # fold 2
    assert all(i == j for i, j in zip(train_indices2, [0, 1, 2, 3, 4]))
    assert all(i == j for i, j in zip(validation_indices2, [5, 6, 7, 8, 9]))


def test_deterministic_by_default(fixture_sample_dataset, fixture_test_strategy):
    """That by default repeated splits are deterministic (as not shuffled)."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy

    # Do one iteration of k-fold splits
    indices_a = strategy.split(dataset=dataset, n_splits=2)
    train_indices_a1, validation_indices_a1, _ = next(indices_a)
    train_indices_a2, validation_indices_a2, _ = next(indices_a)

    # And again...
    indices_b = strategy.split(dataset=dataset, n_splits=2)
    train_indices_b1, validation_indices_b1, _ = next(indices_b)
    train_indices_b2, validation_indices_b2, _ = next(indices_b)

    # Check that we get the same results across folds
    assert all(i == j for i, j in zip(train_indices_a1, train_indices_b1))
    assert all(i == j for i, j in zip(validation_indices_a1, validation_indices_b1))
    assert all(i == j for i, j in zip(train_indices_a2, train_indices_b2))
    assert all(i == j for i, j in zip(validation_indices_a2, validation_indices_b2))


def test_non_deterministic_if_shuffled(fixture_sample_dataset, fixture_test_strategy):
    """That repeated splits are not deterministic if dataset is shuffled."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy

    # Do one iteration of k-fold splits
    indices_a = strategy.split(dataset=dataset, n_splits=2, shuffle=True)
    train_indices_a1, validation_indices_a1, _ = next(indices_a)
    train_indices_a2, validation_indices_a2, _ = next(indices_a)

    # And again...
    indices_b = strategy.split(dataset=dataset, n_splits=2, shuffle=True)
    train_indices_b1, validation_indices_b1, _ = next(indices_b)
    train_indices_b2, validation_indices_b2, _ = next(indices_b)

    # Check that we do not get the same results across folds
    assert any(i != j for i, j in zip(train_indices_a1, train_indices_b1))
    assert any(i != j for i, j in zip(validation_indices_a1, validation_indices_b1))
    assert any(i != j for i, j in zip(train_indices_a2, train_indices_b2))
    assert any(i != j for i, j in zip(validation_indices_a2, validation_indices_b2))


def test_deterministic_if_shuffled_and_seed_set(
    fixture_sample_dataset,
    fixture_test_strategy,
):
    """That split results are deterministic if dataset is shuffled but a seed is set."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy

    # Do one iteration of k-fold splits
    indices_a = strategy.split(dataset=dataset, n_splits=2, shuffle=True, seed=123)
    train_indices_a1, validation_indices_a1, _ = next(indices_a)
    train_indices_a2, validation_indices_a2, _ = next(indices_a)

    # And again...
    indices_b = strategy.split(dataset=dataset, n_splits=2, shuffle=True, seed=123)
    train_indices_b1, validation_indices_b1, _ = next(indices_b)
    train_indices_b2, validation_indices_b2, _ = next(indices_b)

    # Check that we didn't get the same results across folds
    assert all(i == j for i, j in zip(train_indices_a1, train_indices_b1))
    assert all(i == j for i, j in zip(validation_indices_a1, validation_indices_b1))
    assert all(i == j for i, j in zip(train_indices_a2, train_indices_b2))
    assert all(i == j for i, j in zip(validation_indices_a2, validation_indices_b2))
