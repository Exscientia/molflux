import pandas as pd
import pytest

import datasets
from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.load import load_model
from molflux.modelzoo.models.sklearn.gradient_boosting_regressor import (
    GradientBoostingRegressor,
)
from molflux.modelzoo.protocols import Estimator, Model

model_name = "gradient_boosting_regressor"

_X_FEATURES = ["X_col_1", "X_col_2"]
_Y_FEATURES = ["y_col"]

train_df = pd.DataFrame(
    [
        [1, 2, 10],
        [1, 3, 11],
        [2, 4, 13],
    ],
    columns=_X_FEATURES + _Y_FEATURES,
)
predict_df = pd.DataFrame(
    [
        [1, 2],
        [1, 3],
        [2, 5],
    ],
    columns=_X_FEATURES,
)
empty_predict_df = pd.DataFrame([], columns=_X_FEATURES)


@pytest.fixture(scope="function")
def fixture_model() -> Model:
    return load_model(model_name, x_features=_X_FEATURES, y_features=_Y_FEATURES)


def test_in_catalogue():
    """That the model is registered in the catalogue."""
    catalogue = list_models()
    all_names = [name for names in catalogue.values() for name in names]
    assert model_name in all_names


def test_default_model_tag_matches_entrypoint_name(fixture_model):
    """That the default model tag matches the catalogue entrypoint name.

    This is not strictly required, but ensures a more intuitive user experience.
    """
    model = fixture_model
    assert model.tag == model_name


def test_is_mapped_to_correct_class(fixture_model):
    """That the model name is mapped to the appropriate class."""
    model = fixture_model
    assert isinstance(model, GradientBoostingRegressor)


def test_implements_protocol(fixture_model):
    """That the model implements the public Estimator protocol."""
    model = fixture_model
    assert isinstance(model, Estimator)


def test_raises_on_negative_ccp_alpha():
    """That the model cannot be initialised with a negative ccp_alpha"""
    negative_ccp_alpha = -0.1
    with pytest.raises(ValueError, match=r".*cannot be negative.*"):
        load_model(
            model_name,
            x_features=["x1"],
            y_features=["y1"],
            ccp_alpha=negative_ccp_alpha,
        )


def test_attempt_training_on_featureless_y_model_raises():
    """That an exception is raised if training on a model that does not define
    y_features.
    """
    name = model_name
    model = load_model(name=name, x_features=["x1"])
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1, 2, 3]}))
    with pytest.raises(ValueError, match="features"):
        model.train(train_data=data)


def test_predict_dataset_must_contain_model_x_features(fixture_model):
    """That the dataset used for predictions must contain the x features defined by
    the model.
    """
    model = fixture_model
    train_dataset = train_df

    missing_feature = model.x_features[0]
    prediction_dataset = predict_df.drop(columns=missing_feature)

    model.train(train_data=train_dataset)
    with pytest.raises(
        ValueError,
        match=f"Data does not have required feature: '{missing_feature}'",
    ):
        model.predict(data=prediction_dataset)


def test_train_on_single_sample():
    """That can train on a single sample."""
    name = model_name
    model = load_model(name=name, x_features=["x1"], y_features=["y1"])
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1.5], "y1": [2]}))
    model.train(train_data=data)
    assert True


def test_predict_on_single_sample():
    """That can predict on a single sample."""
    name = model_name
    model = load_model(name=name, x_features=["x1"], y_features=["y1"])
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1.5], "y1": [2]}))
    model.train(train_data=data)
    test_data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [3]}))
    assert model.predict(data=test_data)


@pytest.mark.parametrize(
    "train_data, predict_data",
    [
        (train_df, predict_df),
        (
            datasets.Dataset.from_pandas(train_df),
            datasets.Dataset.from_pandas(predict_df),
        ),
        (
            datasets.Dataset.from_pandas(train_df),
            datasets.Dataset.from_pandas(empty_predict_df),
        ),
    ],
)
def test_train_predict_model(fixture_model, train_data, predict_data):
    """Test that a model can run the train and predict functions"""

    model = fixture_model

    model.train(train_data)
    predictions = model.predict(predict_data)
    assert predictions
    assert len(predictions) == len(model.y_features)
    for task_predictions in predictions.values():
        assert len(task_predictions) == len(predict_data)


def test_predict_single_instance(fixture_model):
    """Test that predicting on a single data point behaves as expected"""
    train_data = datasets.Dataset.from_pandas(train_df)
    predict_data = datasets.Dataset.from_pandas(predict_df.iloc[:1])

    model = fixture_model
    model.train(train_data)

    predictions = model.predict(predict_data)
    for array in predictions.values():
        assert isinstance(array, list)


def test_saving_loading(tmp_path, fixture_model):
    """Test that a model can successfully be saved and loaded"""

    fixture_model.train(train_df)

    save_to_store(tmp_path, fixture_model)
    loaded_model = load_from_store(tmp_path)

    assert loaded_model.predict(predict_df)
