import pandas as pd
import pytest

import datasets
from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.load import load_model
from molflux.modelzoo.models.core.average_features_regressor import (
    AverageFeaturesRegressor,
)
from molflux.modelzoo.models.ensemble.ensemble_regressor import EnsembleRegressor
from molflux.modelzoo.protocols import (
    Estimator,
    Model,
    supports_prediction_interval,
    supports_sampling,
    supports_std,
)

model_name = "ensemble_regressor"

_X_FEATURES = ["X_col_1", "X_col_2"]
_Y_FEATURES = ["y_col"]

train_df = pd.DataFrame(
    [
        [1, 2, 10],
        [1, 3, 11],
        [2, 4, 13],
        [2, 4, 13],
        [2, 4, 13],
    ],
    columns=_X_FEATURES + _Y_FEATURES,
)
predict_df = pd.DataFrame(
    [
        [1, 2],
        [1, 3],
        [2, 4],
        [2, 4],
        [2, 5],
    ],
    columns=_X_FEATURES,
)
empty_predict_df = pd.DataFrame([], columns=_X_FEATURES)


@pytest.fixture(scope="function")
def fixture_model() -> Model:
    return load_model(
        model_name,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
        meta_estimator={
            "name": "average_features_regressor",
            "config": {
                "x_features": [
                    "knn_regressor",
                    "random_forest_regressor",
                    "random_forest_regressor_1",
                ],
                "y_features": _Y_FEATURES,
            },
        },
        base_estimators=[
            {
                "name": "knn_regressor",
                "config": {
                    "x_features": _X_FEATURES,
                    "y_features": _Y_FEATURES,
                    "n_neighbors": 1,
                },
            },
            {
                "name": "random_forest_regressor",
                "config": {
                    "x_features": _X_FEATURES,
                    "y_features": _Y_FEATURES,
                },
            },
            {
                "name": "random_forest_regressor",
                "config": {
                    "x_features": _X_FEATURES,
                    "y_features": _Y_FEATURES,
                    "tag": "random_forest_regressor_1",
                },
            },
        ],
    )


def test_in_catalogue():
    """That the model is registered in the catalogue."""
    catalogue = list_models()
    all_names = [name for names in catalogue.values() for name in names]
    assert model_name in all_names


def test_default_model_tag_matches_entrypoint_name(fixture_model):
    """That the default model tag includes the catalogue entrypoint name.

    This is not strictly required, but ensures a more intuitive user experience.
    """
    model = fixture_model
    assert model_name in model.tag


def test_is_mapped_to_correct_class(fixture_model):
    """That the model name is mapped to the appropriate class."""
    model = fixture_model
    assert isinstance(model, EnsembleRegressor)


def test_implements_protocol(fixture_model):
    """That the model implements the public Estimator protocol."""
    model = fixture_model
    assert isinstance(model, Estimator)


def test_tag_for_wrapped_model(fixture_model):
    """That the model has appropriate tag for a wrapped model.
    This is not strictly required, but ensures a more intuitive user experience.
    """
    model = fixture_model
    assert (
        model.tag
        == "ensemble_regressor[knn_regressor:random_forest_regressor:random_forest_regressor_1]"
    )


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
    assert isinstance(model.meta_estimator, AverageFeaturesRegressor)
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
def test_train_predict_with_prediction_interval(
    fixture_model,
    train_data,
    predict_data,
):
    """Test that a model can run the train and predict_with_prediction_interval functions"""

    model = fixture_model
    model.train(train_data)

    assert supports_prediction_interval(model)
    predictions, intervals = model.predict_with_prediction_interval(
        data=predict_data,
        confidence=0.5,
    )

    assert len(predictions) == len(model.y_features)
    for task_predictions in predictions.values():
        assert len(task_predictions) == len(predict_data)

    assert len(intervals) == len(model.y_features)
    for prediction_interval in intervals.values():
        assert len(prediction_interval) == len(predict_data)


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
def test_train_predict_with_standard_deviation(fixture_model, train_data, predict_data):
    """Test that a model can run the train and predict_with_std functions"""
    model = fixture_model
    model.train(train_data)

    assert supports_std(model)

    predictions, stds = model.predict_with_std(predict_data)

    assert len(predictions) == len(model.y_features)
    for task_predictions in predictions.values():
        assert len(task_predictions) == len(predict_data)

    assert len(stds) == len(model.y_features)
    for prediction_std in stds.values():
        assert len(prediction_std) == len(predict_data)


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
def test_sample(fixture_model, train_data, predict_data):
    """Test that a model can run the train and sample functions"""
    model = fixture_model
    model.train(train_data)

    assert supports_sampling(model)

    n_samples = 10
    prediction_samples = model.sample(predict_data, n_samples=n_samples)
    assert prediction_samples
    assert len(prediction_samples) == len(model.y_features)
    for samples in prediction_samples.values():
        assert len(samples) == len(predict_data)
        for item in samples:
            assert len(item) == n_samples
