import numpy as np
import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "linear_split"


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


def test_yields_one_fold_by_default(fixture_sample_dataset, fixture_test_strategy):
    """That the splitting strategy only yields one set of splits by default."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset)
    assert len(list(indices)) == 1


def test_can_yield_multiple_folds(fixture_sample_dataset, fixture_test_strategy):
    """That the splitting strategy can generate multiple folds."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, n_splits=2)
    assert len(list(indices)) == 2


def test_default_split_fractions(fixture_sample_dataset, fixture_test_strategy):
    """That the dataset is split by default into 80:10:10 splits."""
    dataset = fixture_sample_dataset
    n_samples = len(dataset)
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset)
    train_indices, validation_indices, test_indices = next(indices)
    assert len(list(train_indices)) == 0.8 * n_samples
    assert len(list(validation_indices)) == 0.1 * n_samples
    assert len(list(test_indices)) == 0.1 * n_samples


def test_custom_split_fractions(fixture_sample_dataset, fixture_test_strategy):
    """That the dataset can be split into arbitrary sizes."""
    dataset = fixture_sample_dataset
    n_samples = len(dataset)
    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        train_fraction=0.6,
        validation_fraction=0.3,
        test_fraction=0.1,
    )
    train_indices, validation_indices, test_indices = next(indices)
    assert len(list(train_indices)) == 0.6 * n_samples
    assert len(list(validation_indices)) == 0.3 * n_samples
    assert len(list(test_indices)) == 0.1 * n_samples


def test_deterministic_split(fixture_sample_dataset, fixture_test_strategy):
    """That split results are deterministic."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy

    indices = strategy.split(dataset=dataset, n_splits=2)
    train_indices1, validation_indices1, test_indices1 = next(indices)
    train_indices2, validation_indices2, test_indices2 = next(indices)

    assert all(i == j for i, j in zip(train_indices1, train_indices2))
    assert all(i == j for i, j in zip(validation_indices1, validation_indices2))
    assert all(i == j for i, j in zip(test_indices1, test_indices2))


def test_splits_are_disjoint(fixture_sample_dataset, fixture_test_strategy):
    """That data is spread across splits without overlap."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset)
    train_indices, validation_indices, test_indices = next(indices)
    assert set(train_indices).isdisjoint(set(validation_indices))
    assert set(train_indices).isdisjoint(set(test_indices))
    assert set(validation_indices).isdisjoint(set(test_indices))


def test_splits_are_linear(fixture_test_strategy):
    """That data is split linearly."""
    dataset = list(range(10))
    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        train_fraction=0.6,
        validation_fraction=0.3,
        test_fraction=0.1,
    )
    train_indices, validation_indices, test_indices = next(indices)
    assert list(train_indices) == [0, 1, 2, 3, 4, 5]
    assert list(validation_indices) == [6, 7, 8]
    assert list(test_indices) == [9]


def test_inconsistent_split_fractions_raise(
    fixture_sample_dataset,
    fixture_test_strategy,
):
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        train_fraction=0.6,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    with pytest.raises(AssertionError):
        next(indices)
