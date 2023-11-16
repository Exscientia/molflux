import pandas as pd
import pytest

import datasets
from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.load import load_model
from molflux.modelzoo.models.pystan.sparse_linear_regressor import (
    SparseLinearRegressor,
)
from molflux.modelzoo.protocols import (
    Estimator,
    Model,
    SupportsPredictionInterval,
    SupportsSampling,
    SupportsStandardDeviation,
    supports_prediction_interval,
    supports_sampling,
    supports_std,
)

model_name = "sparse_linear_regressor"

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
        [1, 2, 10],
        [1, 3, 11],
        [2, 4, 13],
    ],
    columns=_X_FEATURES + _Y_FEATURES,
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
    assert isinstance(model, SparseLinearRegressor)


def test_implements_protocol(fixture_model):
    """That the model implements the public Estimator protocol."""
    model = fixture_model
    assert isinstance(model, Estimator)


def test_implements_prediction_interval_protocol(fixture_model):
    """That the model implements the prediction interval protocol."""
    model = fixture_model
    assert isinstance(model, SupportsPredictionInterval)


def test_implements_standard_deviation_protocol(fixture_model):
    """That the model implements the standard deviation protocol."""
    model = fixture_model
    assert isinstance(model, SupportsStandardDeviation)


def test_implements_sampling_protocol(fixture_model):
    """That the model implements the sampling protocol."""
    model = fixture_model
    assert isinstance(model, SupportsSampling)


def test_trains_on_one_data_point():
    """That can train on one data point"""
    name = model_name
    model = load_model(name=name, x_features=["x1"], y_features=["y1"])
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1.5], "y1": [2]}))
    model.train(train_data=data)
    assert True


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


def test_saving_loading(tmp_path, fixture_model):
    """Test that a model can successfully be saved and loaded"""

    fixture_model.train(train_df)

    save_to_store(tmp_path, fixture_model)
    loaded_model = load_from_store(tmp_path)

    assert loaded_model.predict(predict_df)
