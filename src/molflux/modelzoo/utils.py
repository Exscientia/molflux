import inspect
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Union

import numpy as np

import datasets
from molflux.modelzoo.typing import (
    Features,
    PredictionResult,
)


def raise_on_unrecognised_parameters(callable: Callable, **kwargs: Any) -> None:
    """Raises if the callable does not explicitly define one or more of the
    given kwargs.

    Args:
        callable: The callable to shield.
        kwargs: The kwargs that would be passed to the callable.

    Raises:
        ValueError: If one or more kwargs are not defined by the callable.
    """
    expected_signature = inspect.signature(callable)
    expected_parameters = expected_signature.parameters
    parameters_received = set(kwargs.keys())
    if not parameters_received.issubset(expected_parameters):
        f = callable.__name__.lstrip("_")
        unknown_kwargs = list(parameters_received.difference(expected_parameters))
        raise ValueError(
            f"Unknown {f} parameter(s): {unknown_kwargs}\n\n"
            f"Expected signature {f}{expected_signature}",
        )


def validate_features(data: datasets.Dataset, features: Features) -> None:
    """Validate features from the input dataset."""

    if not features:
        raise ValueError("Please specify the model's features.")

    for feature in features:
        if feature not in data.column_names:
            raise ValueError(f"Data does not have required feature: {feature!r}")


def validate_prediction_result_num_tasks(
    prediction_result: PredictionResult,
    y_features: Features,
) -> None:
    """Validates that the prediction result contains results for the expected number of features."""

    if not y_features:
        raise ValueError("Please specify the model's y_features.")

    if len(prediction_result) != len(y_features):
        raise RuntimeError(
            f"Prediction results do not match the expected number of tasks: got {len(prediction_result)}, expected {len(y_features)}",
        )


def pick_features(
    dataset: datasets.Dataset,
    features: Union[str, Features],
) -> datasets.Dataset:
    """Selects multiple columns from a datasets.Dataset"""

    features = [features] if isinstance(features, str) else features

    if set(features).issuperset(dataset.column_names):
        return dataset

    return dataset.select_columns(features)


@contextmanager
def _disabled_huggingface_progress_bars() -> Iterator[None]:
    """Temporarily disables huggingface 'datasets' progress bars,"""
    was_enabled = datasets.is_progress_bar_enabled()

    if not was_enabled:
        yield
        return

    try:
        datasets.disable_progress_bar()
        yield
    finally:
        datasets.enable_progress_bar()


def get_concatenated_array(
    dataset: datasets.Dataset,
    features: Union[str, Features],
) -> np.ndarray:
    """
    Concatenates columns of dataset into a feature vector, ordered by the specified
    features.

    Concatenation is done on the column dimension - number of dataset rows remains the
    same in the output.

    This assumes that features indicated are scalars or lists of scalars, compatible with
    numpy arrays.

    Args:
        dataset (datasets.Dataset): The input dataset whose subset of columns is
            concatenated
        features (str, Features): One or multiple column names that will be
            concatenated in order. They must all be in the input dataset.

    Returns:
        np.array: 2D numpy array of concatenated columns.
    """

    if not len(dataset):
        return np.array([]).reshape(-1, 1)

    if isinstance(features, str):
        features = [features]

    if not set(features).issubset(set(dataset.features)):
        raise KeyError(f"{features=} not a subset of {dataset.features.keys()=}")

    # generate numpy arrays for each feature
    arrays = [np.array(dataset[feature]) for feature in features]

    # compute the maximum dimension found for each feature
    max_dim = max(arr.ndim for arr in arrays)

    # if it's not 1 or 2, something went wrong
    if max_dim not in [1, 2]:
        raise RuntimeError("Features must either be scalars or 1-dimensional!")

    return np.column_stack(arrays)
