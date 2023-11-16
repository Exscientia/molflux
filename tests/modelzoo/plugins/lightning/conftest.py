from contextlib import nullcontext
from typing import ContextManager

import pandas as pd
import pytest
import torch

import datasets
from molflux.modelzoo.load import load_model
from molflux.modelzoo.protocols import Model

model_name = "lightning_mlp_regressor"

_X_FEATURES = ["X_col_1", "X_col_2"]
_Y_FEATURES = ["y_col"]


@pytest.fixture()
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 4.0],
            [2.0, 4.0, 6.0],
            [2.0, 4.0, 6.0],
            [2.0, 4.0, 6.0],
        ],
        columns=_X_FEATURES + _Y_FEATURES,
    )


@pytest.fixture()
def validation_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 4.0],
            [2.0, 4.0, 6.0],
            [2.0, 4.0, 6.0],
            [2.0, 5.0, 7.0],
        ],
        columns=_X_FEATURES + _Y_FEATURES,
    )


@pytest.fixture()
def predict_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [2.0, 5.0],
        ],
        columns=_X_FEATURES,
    )


@pytest.fixture()
def empty_predict_df() -> pd.DataFrame:
    return pd.DataFrame([], columns=_X_FEATURES)


@pytest.fixture()
def train_dataset(train_df: pd.DataFrame) -> datasets.Dataset:
    return datasets.Dataset.from_pandas(train_df)


@pytest.fixture()
def train_dataset_b(train_df: pd.DataFrame) -> datasets.Dataset:
    joint_df = pd.concat([train_df, train_df]).iloc[:8]
    return datasets.Dataset.from_pandas(joint_df)


@pytest.fixture()
def validation_dataset(validation_df: pd.DataFrame) -> datasets.Dataset:
    return datasets.Dataset.from_pandas(validation_df)


@pytest.fixture()
def predict_dataset(predict_df: pd.DataFrame) -> datasets.Dataset:
    return datasets.Dataset.from_pandas(predict_df)


@pytest.fixture()
def empty_predict_dataset(empty_predict_df: pd.DataFrame) -> datasets.Dataset:
    return datasets.Dataset.from_pandas(empty_predict_df)


@pytest.fixture(scope="function")
def fixture_model() -> Model:
    return load_model(
        model_name,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
        hidden_dim=3,
        input_dim=2,
        num_tasks=len(_Y_FEATURES),
    )


@pytest.fixture(scope="function")
def fixture_compiled_model() -> Model:
    return load_model(
        model_name,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
        hidden_dim=3,
        input_dim=2,
        num_tasks=len(_Y_FEATURES),
    )


@pytest.fixture(scope="function")
def fixture_pre_trained_model() -> Model:
    model = load_model(
        model_name,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
        hidden_dim=3,
        input_dim=2,
        num_tasks=len(_Y_FEATURES),
    )
    # set params to nice numbers
    model.module = model._instantiate_module()  # type: ignore
    for n, p in model.module.named_parameters():  # type: ignore
        if n == "module.0.weight":
            p.data = torch.ones_like(p) * 0
        elif n == "module.0.bias":
            p.data = torch.ones_like(p) * 1
        elif n == "module.2.weight":
            p.data = torch.ones_like(p) * 2
        elif n == "module.2.bias":
            p.data = torch.ones_like(p) * 3
        else:
            raise KeyError("Param not found in module")

    return model


@pytest.fixture()
def compile_error_context(compile: bool) -> ContextManager:
    if not compile:
        return nullcontext()
    else:
        return pytest.raises(ValueError, match=r"^Compilation temporarily disabled")

    # TODO enable after PyTorch 2.1 is available
    # if sys.version_info >= (3, 11):
    #     return pytest.raises(
    #         RuntimeError,
    #         match=r"^Python 3.11\+ not yet supported for torch.compile$",
    #     )
    # elif sys.version_info.major == 3 and sys.version_info.minor == 8:
    #     return pytest.raises(
    #         (AssertionError, torch._dynamo.exc.BackendCompilerFailed),
    #     )
    # else:
    #     return nullcontext()
