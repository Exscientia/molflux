import pandas as pd
import pytest

import datasets
from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.load import load_model
from molflux.modelzoo.models.mapie.mapie_regressor import MapieRegressor
from molflux.modelzoo.protocols import (
    Estimator,
    Model,
    SupportsPredictionInterval,
    SupportsUncertaintyCalibration,
    supports_prediction_interval,
    supports_uncertainty_calibration,
)

model_name = "mapie_regressor"

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
    assert isinstance(model, MapieRegressor)


def test_implements_protocol(fixture_model):
    """That the model implements the public Estimator protocol."""
    model = fixture_model
    assert isinstance(model, Estimator)


def test_implements_uncertainty_calibration_protocol(fixture_model):
    """That the model implements the uncertainty calibration protocol."""
    model = fixture_model
    assert isinstance(model, SupportsUncertaintyCalibration)


def test_implements_prediction_interval_protocol(fixture_model):
    """That the model implements the prediction interval protocol."""
    model = fixture_model
    assert isinstance(model, SupportsPredictionInterval)


def test_raises_on_multitask():
    """That the model cannot be initialised with multiple y features.

    This is not supported yet by the backend model package.
    """
    y_features = ["y1", "y2"]
    with pytest.raises(NotImplementedError, match=r".* single task .*"):
        load_model(model_name, x_features=["x1"], y_features=y_features)


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
def test_predict_with_prediction_interval_from_estimator(
    fixture_model,
    train_data,
    predict_data,
):
    """That can predict uncertainty as an addon to an existing estimator."""

    existing_estimator = load_model(
        "random_forest_regressor",
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
    )
    existing_estimator.train(train_data)

    # Plug pre-trained model into mapie estimator
    model = load_model(
        name=model_name,
        estimator=existing_estimator,
        x_features=existing_estimator.x_features,
        y_features=existing_estimator.y_features,
        cv="prefit",
    )

    assert supports_uncertainty_calibration(model)
    model.calibrate_uncertainty(data=train_data)  # type: ignore[attr-defined]

    assert supports_prediction_interval(model)
    predictions, intervals = model.predict_with_prediction_interval(  # type: ignore[attr-defined]
        predict_data,
        confidence=0.5,
    )

    assert len(predictions) == len(model.y_features)
    for task_predictions in predictions.values():
        assert len(task_predictions) == len(predict_data)

    assert len(intervals) == len(model.y_features)
    for prediction_interval in intervals.values():
        assert len(prediction_interval) == len(predict_data)


def test_saving_mapie_regressor_model(tmp_path):
    """That can save a mapie regressor model"""

    existing_estimator = load_model(
        "random_forest_regressor",
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
    )
    existing_estimator.train(train_df)

    # Plug pre-trained model into mapie estimator
    model = load_model(
        name=model_name,
        estimator=existing_estimator,
        x_features=existing_estimator.x_features,
        y_features=existing_estimator.y_features,
        cv="prefit",
    )

    model.calibrate_uncertainty(data=train_df)  # type: ignore[attr-defined]
    save_to_store(tmp_path, model)

    loaded_model = load_from_store(tmp_path)

    assert loaded_model.predict_with_prediction_interval(predict_df, confidence=0.5)  # type: ignore[attr-defined]


def test_loading_pretrained_raises_on_training_attempt(tmp_path):
    """That an error is raised if attempting to retrain a pre-trained model that
    loaded from a persisted artefact.

    This test should be removed if we end up finding a way to save models in
    a way that persists a link to the input estimator wrapped by the MAPIE
    model.
    """

    existing_estimator = load_model(
        "random_forest_regressor",
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
    )
    existing_estimator.train(train_df)

    # Plug pre-trained model into mapie estimator
    model = load_model(
        name=model_name,
        estimator=existing_estimator,
        x_features=existing_estimator.x_features,
        y_features=existing_estimator.y_features,
        cv="prefit",
    )

    model.calibrate_uncertainty(data=train_df)  # type: ignore[attr-defined]
    save_to_store(tmp_path, model)

    with pytest.warns(
        UserWarning,
        match="Model loaded with an unlinked input estimator",
    ):
        loaded_model = load_from_store(tmp_path)

    with pytest.raises(ValueError, match="The input estimator has been unlinked"):
        loaded_model.calibrate_uncertainty(train_df)  # type: ignore[attr-defined]


def test_predict_single_instance(fixture_model):
    """Test that predicting on a single data point behaves as expected"""
    train_data = datasets.Dataset.from_pandas(train_df)
    predict_data = datasets.Dataset.from_pandas(predict_df.iloc[:1])

    model = fixture_model
    model.train(train_data)

    predictions = model.predict(predict_data)
    for array in predictions.values():
        assert isinstance(array, list)

    predictions, prediction_intervals = model.predict_with_prediction_interval(
        predict_data,
        confidence=0.5,
    )
    for array in predictions.values():
        assert isinstance(array, list)
    for array in prediction_intervals.values():
        assert isinstance(array, list)
        assert isinstance(array[0], tuple)
