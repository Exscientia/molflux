import json
import os
from pathlib import Path

import pandas as pd
import pytest

from datasets import Dataset
from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.load import load_model
from molflux.modelzoo.models.sklearn.sklearn_pipeline.sklearn_pipeline_classifier import (
    SklearnPipelineClassifier,
)
from molflux.modelzoo.protocols import Estimator, Model, SupportsClassification
from molflux.modelzoo.typing import DataFrameLike

model_name = "sklearn_pipeline_classifier"

_X_FEATURES = ["X_col_1", "X_col_2", "X_col_3"]
_Y_FEATURES = ["y_col"]


@pytest.fixture(scope="function")
def fixture_basic_model() -> Model:
    return load_model(model_name, x_features=_X_FEATURES, y_features=_Y_FEATURES)


@pytest.fixture(scope="function")
def fixture_config_dir(fixture_path_to_assets: Path) -> Path:
    return (
        fixture_path_to_assets / "plugins" / "sklearn" / "sklearn_pipeline_classifier"
    )


def load_model_config(path_to_dir: Path, file_name: str) -> dict:
    path_to_file = os.path.join(path_to_dir, file_name)
    with open(path_to_file) as f:
        config: dict = json.load(f)
    return config


@pytest.fixture(scope="function")
def fixture_k_neighbors_model(fixture_config_dir: Path) -> Model:
    file_name = "normalizer-k-neighbors.json"
    model_config = load_model_config(fixture_config_dir, file_name)
    return load_model(
        model_name,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
        **{"step_configs": model_config},
    )


@pytest.fixture(scope="function")
def fixture_svc_model(fixture_config_dir: Path) -> Model:
    file_name = "pca-select-k-svc.json"
    model_config = load_model_config(fixture_config_dir, file_name)
    return load_model(
        model_name,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
        **{"step_configs": model_config},
    )


@pytest.fixture(scope="function")
def fixture_rf_model(fixture_config_dir: Path) -> Model:
    file_name = "scaler-random-forest.json"
    model_config = load_model_config(fixture_config_dir, file_name)
    return load_model(
        model_name,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
        **{"step_configs": model_config},
    )


@pytest.fixture(scope="function")
def fixture_train_dataset() -> DataFrameLike:
    train_df = pd.DataFrame(
        [
            [1, 1, 2, 10],
            [1, 1, 3, 11],
            [2, 3, 1, 13],
        ],
        columns=_X_FEATURES + _Y_FEATURES,
    )
    return Dataset.from_pandas(train_df)


@pytest.fixture(scope="function")
def fixture_predict_dataset() -> DataFrameLike:
    predict_df = pd.DataFrame(
        [
            [1, 2, 2],
            [1, 3, 1],
            [2, 5, 3],
        ],
        columns=_X_FEATURES,
    )
    return Dataset.from_pandas(predict_df)


@pytest.fixture(scope="function")
def fixture_empty_predict_dataset() -> DataFrameLike:
    predict_df = pd.DataFrame(
        [],
        columns=_X_FEATURES,
    )
    return Dataset.from_pandas(predict_df)


def test_in_catalogue():
    """That the model is registered in the catalogue."""
    catalogue = list_models()
    all_names = [name for names in catalogue.values() for name in names]
    assert model_name in all_names


def test_default_model_tag_matches_entrypoint_name(fixture_basic_model):
    """That the default model tag matches the catalogue entrypoint name.

    This is not strictly required, but ensures a more intuitive user experience.
    """
    model = fixture_basic_model
    assert model.tag == model_name


@pytest.mark.parametrize(
    ["model_fixture"],
    [
        ["fixture_basic_model"],
        ["fixture_k_neighbors_model"],
        ["fixture_rf_model"],
        ["fixture_svc_model"],
    ],
)
def test_is_mapped_to_correct_class(model_fixture, request):
    """That the model name is mapped to the appropriate class."""
    model = request.getfixturevalue(model_fixture)
    assert isinstance(model, SklearnPipelineClassifier)
    assert model.tag == model_name


@pytest.mark.parametrize(
    ["model_fixture"],
    [
        ["fixture_basic_model"],
        ["fixture_k_neighbors_model"],
        ["fixture_rf_model"],
        ["fixture_svc_model"],
    ],
)
def test_implements_protocol(model_fixture, request):
    """That the model implements the public Estimator protocol."""
    model = request.getfixturevalue(model_fixture)
    assert isinstance(model, Estimator)
    assert isinstance(model, SupportsClassification)


@pytest.mark.parametrize(
    ["model_fixture"],
    [
        ["fixture_basic_model"],
        ["fixture_k_neighbors_model"],
        ["fixture_rf_model"],
        ["fixture_svc_model"],
    ],
)
def test_train_predict_model(
    model_fixture,
    fixture_train_dataset,
    fixture_predict_dataset,
    request,
):
    """Test that a model can run the train and predict functions"""

    model = request.getfixturevalue(model_fixture)
    train_dataset = fixture_train_dataset
    predict_dataset = fixture_predict_dataset

    model.train(train_dataset)

    predictions = model.predict(predict_dataset)
    assert len(predictions) == len(model.y_features)
    for task_predictions in predictions.values():
        assert len(task_predictions) == len(predict_dataset)

    probabilities = model.predict_proba(predict_dataset)
    assert len(probabilities) == len(model.y_features)
    for task_predictions in probabilities.values():
        assert len(task_predictions) == len(predict_dataset)


@pytest.mark.parametrize(
    ["model_fixture"],
    [
        ["fixture_basic_model"],
        ["fixture_k_neighbors_model"],
        ["fixture_rf_model"],
        ["fixture_svc_model"],
    ],
)
def test_predict_on_empty_dataset(
    model_fixture,
    fixture_train_dataset,
    fixture_empty_predict_dataset,
    request,
):
    """Test can predict on an empty input dataset.

    This should return an empty dictionary of output features.
    """

    model = request.getfixturevalue(model_fixture)
    train_dataset = fixture_train_dataset
    predict_dataset = fixture_empty_predict_dataset

    model.train(train_dataset)
    predictions = model.predict(predict_dataset)
    assert len(predictions) == len(model.y_features)
    for task_predictions in predictions.values():
        assert len(task_predictions) == 0  # == len(predict_dataset)

    probabilities = model.predict_proba(predict_dataset)
    assert len(probabilities) == len(model.y_features)
    for task_predictions in probabilities.values():
        assert len(task_predictions) == 0  # == len(predict_dataset)


@pytest.mark.parametrize(
    ["model_fixture"],
    [
        ["fixture_basic_model"],
        ["fixture_k_neighbors_model"],
        ["fixture_rf_model"],
        ["fixture_svc_model"],
    ],
)
def test_saving_loading(
    tmp_path,
    model_fixture,
    fixture_train_dataset,
    fixture_predict_dataset,
    request,
):
    """Test that a model can successfully be saved and loaded"""

    model = request.getfixturevalue(model_fixture)

    model.train(fixture_train_dataset)

    save_to_store(tmp_path, model)
    loaded_model = load_from_store(tmp_path)

    assert loaded_model.predict(fixture_predict_dataset)
