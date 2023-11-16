import datasets
from molflux.datasets.splitting import split_dataset


def test_returns_dataset_dict(fixture_splitting_strategy_mock, fixture_dataset):
    """That splitting returns a DatasetDict"""
    strategy = fixture_splitting_strategy_mock
    dataset = fixture_dataset
    splits = split_dataset(dataset=dataset, strategy=strategy)
    for split in splits:
        assert isinstance(split, datasets.DatasetDict)


def test_splits_are_datasets(fixture_splitting_strategy_mock, fixture_dataset):
    """That each returned split is a Dataset."""
    strategy = fixture_splitting_strategy_mock
    dataset = fixture_dataset
    splits = split_dataset(dataset=dataset, strategy=strategy)
    for split in splits:
        for dataset in split.values():
            assert isinstance(dataset, datasets.Dataset)


def test_generates_train_split(fixture_splitting_strategy_mock, fixture_dataset):
    """That splitting generates a train dataset."""
    strategy = fixture_splitting_strategy_mock
    dataset = fixture_dataset
    splits = split_dataset(dataset=dataset, strategy=strategy)
    for split in splits:
        assert "train" in split


def test_generates_validation_split(fixture_splitting_strategy_mock, fixture_dataset):
    """That splitting generates a validation dataset."""
    strategy = fixture_splitting_strategy_mock
    dataset = fixture_dataset
    splits = split_dataset(dataset=dataset, strategy=strategy)
    for split in splits:
        assert "validation" in split


def test_generates_test_split(fixture_splitting_strategy_mock, fixture_dataset):
    """That splitting generates a test dataset."""
    strategy = fixture_splitting_strategy_mock
    dataset = fixture_dataset
    splits = split_dataset(dataset=dataset, strategy=strategy)
    for split in splits:
        assert "test" in split
