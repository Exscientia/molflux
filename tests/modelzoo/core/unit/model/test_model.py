import warnings
from typing import Dict

import pandas as pd
import pytest

import datasets
from molflux.modelzoo.load import load_model
from molflux.modelzoo.protocols import Estimator
from molflux.modelzoo.typing import DataFrameLike


def test_load() -> None:
    model = load_model("mock_model")
    assert model is not None


def test_implements_protocol() -> None:
    model = load_model("mock_model")
    assert isinstance(model, Estimator)


train_df = pd.DataFrame({"x1": ["Hello"], "y1": ["Buongiorno"]})


@pytest.mark.parametrize(
    "train_data",
    [
        train_df,
        datasets.Dataset.from_pandas(train_df),
    ],
)
def test_train(train_data: DataFrameLike) -> None:
    model = load_model("mock_model", x_features=["x1"], y_features=["y1"])
    model.train(train_data=train_data)
    assert True


def test_unexpected_validation_data_in_model_train() -> None:
    model = load_model(
        "mock_model",
        x_features=["x1"],
        y_features=["y1"],
    )
    with pytest.warns(UserWarning):
        model.train(train_data=train_df, validation_data=train_df)


def test_expected_validation_data_in_model_train() -> None:
    model = load_model(
        "mock_model_with_validation_data_arg",
        x_features=["x1"],
        y_features=["y1"],
    )
    with warnings.catch_warnings():
        model.train(train_data=train_df, validation_data=train_df)


def test_unexpected_validation_data_in_model_train_multi_data() -> None:
    model = load_model(
        "mock_model_multi_data",
        x_features=["x1"],
        y_features=["y1"],
    )
    dataset: Dict[str, DataFrameLike] = {"foo": train_df}
    with pytest.warns(UserWarning):
        model.train(train_data=dataset, validation_data=dataset)


def test_expected_validation_data_in_model_train_multi_data() -> None:
    model = load_model(
        "mock_model_multi_data_with_validation_data_arg",
        x_features=["x1"],
        y_features=["y1"],
    )
    dataset: Dict[str, DataFrameLike] = {"foo": train_df}
    with warnings.catch_warnings():
        model.train(train_data=dataset, validation_data=dataset)
