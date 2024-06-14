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
    SupportsSampling,
    SupportsStandardDeviation,
    SupportsUncertaintyCalibration,
    supports_prediction_interval,
    supports_sampling,
    supports_std,
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
    """That the default model tag includes the catalogue entrypoint name.

    This is not strictly required, but ensures a more intuitive user experience.
    """
    model = fixture_model
    assert model_name in model.tag


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


def test_implements_standard_deviation_protocol(fixture_model):
    """That the model implements the standard deviation protocol."""
    model = fixture_model
    assert isinstance(model, SupportsStandardDeviation)


def test_implements_sampling_protocol(fixture_model):
    """That the model implements the sampling protocol."""
    model = fixture_model
    assert isinstance(model, SupportsSampling)


def test_tag_for_wrapped_model(fixture_model):
    """That the model has appropriate tag for a wrapped model.
    This is not strictly required, but ensures a more intuitive user experience.
    """
    model = fixture_model
    assert model.tag == "mapie_regressor[linear_regressor]"


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
    model.calibrate_uncertainty(data=train_data)

    assert supports_prediction_interval(model)
    predictions, intervals = model.predict_with_prediction_interval(
        predict_data,
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


@pytest.mark.parametrize(
    ["estimator", "should_raise_errors"],
    [
        [
            {
                "name": "random_forest_regressor",
                "config": {"n_estimators": 10, "x_features": ["dummy_col"]},
            },
            False,
        ],
        [{"name": "gradient_boosting_regressor"}, False],
        [{"bad_schema_key": {"bad_kwarg": 42}}, True],
        [{"name": "bad_name_for_a_model"}, True],
    ],
)
def test_model_with_config_estimator(estimator, should_raise_errors):
    """That models with configs as estimators correctly run"""

    model: Model = load_model(
        model_name,
        estimator=estimator,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
    )

    train_data = datasets.Dataset.from_pandas(train_df)
    predict_data = datasets.Dataset.from_pandas(predict_df)

    if should_raise_errors:
        with pytest.raises(ValueError):
            model.train(train_data)
        return
    else:
        model.train(train_data)

    assert isinstance(model, SupportsPredictionInterval)

    predictions, prediction_intervals = model.predict_with_prediction_interval(
        predict_data,
        confidence=0.5,
    )
    for array in predictions.values():
        assert isinstance(array, list)
    for array in prediction_intervals.values():
        assert isinstance(array, list)
        assert isinstance(array[0], tuple)
