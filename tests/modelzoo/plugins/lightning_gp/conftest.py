import pandas as pd
import pytest

import datasets
from molflux.modelzoo.load import load_from_dict
from molflux.modelzoo.protocols import Model

model_name = "lightning_gp_regressor"

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


@pytest.fixture()
def single_row_predict_dataset(predict_df: pd.DataFrame) -> datasets.Dataset:
    return datasets.Dataset.from_pandas(predict_df.iloc[:1])


@pytest.fixture(scope="function")
def fixture_gp_config() -> dict:
    gp_config = {
        "name": model_name,
        "config": {
            "x_features": _X_FEATURES,
            "y_features": _Y_FEATURES,
            "num_tasks": len(_Y_FEATURES),
            "likelihood_config": {
                "noise_constraint": {
                    "name": "GreaterThan",
                    "lower_bound": 0.01,
                },
            },
            "kernel_config": {
                "name": "RBFKernel",
            },
            "trainer": {
                "accelerator": "cpu",
            },
        },
    }
    return gp_config


@pytest.fixture(scope="function")
def fixture_gp_model(fixture_gp_config: dict) -> Model:
    return load_from_dict(fixture_gp_config)
