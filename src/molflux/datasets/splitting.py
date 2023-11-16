"""
Wrappers for using splitting strategies to split datasets.Datasets.
"""

from typing import Iterator, Optional

from datasets import Dataset, DatasetDict
from molflux.datasets.interfaces import SplittingStrategy


def split_dataset(
    dataset: Dataset,
    strategy: SplittingStrategy,
    target_column: Optional[str] = None,
    groups_column: Optional[str] = None,
) -> Iterator[DatasetDict]:
    """Generates dataset splits according to given strategy.

    Args:
        dataset: The dataset to split.
        strategy: A pre-configured instance of the splitting strategy to use.
        target_column: The name of the dataset column to use as target variable for
            supervised learning problems ('y') when splitting.
        groups_column: The name of the dataset column to use as group labels when
            splitting.

    Returns:
        A generator of split datasets (folds).
    """

    y = dataset[target_column] if target_column is not None else None
    groups = dataset[groups_column] if groups_column is not None else None

    for train_indices, validation_indices, test_indices in strategy.split(
        dataset=dataset,
        y=y,
        groups=groups,
    ):
        train_split = dataset.select(indices=train_indices)
        validation_split = dataset.select(indices=validation_indices)
        test_split = dataset.select(indices=test_indices)

        yield DatasetDict(
            {
                "train": train_split,
                "validation": validation_split,
                "test": test_split,
            },
        )
