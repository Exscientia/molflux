import pandas as pd
import pytest

import datasets
from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.load import load_model
from molflux.modelzoo.models.pyod.mcd_detector import MCDDetector
from molflux.modelzoo.protocols import Estimator, Model, supports_classification

model_name = "mcd_detector"

_X_FEATURES = ["X_col_1", "X_col_2"]

train_df = pd.DataFrame(
    [
        [1, 2],
        [1, 3],
        [2, 4],
        [2, 4],
        [2, 4],
    ],
    columns=_X_FEATURES,
)
predict_df = train_df
empty_predict_df = pd.DataFrame([], columns=_X_FEATURES)


@pytest.fixture(scope="function")
def fixture_model() -> Model:
    return load_model(model_name, x_features=_X_FEATURES)


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
    assert isinstance(model, MCDDetector)


def test_implements_protocol(fixture_model):
    """That the model implements the public Estimator protocol."""
    model = fixture_model
    assert isinstance(model, Estimator)


def test_supports_classification(fixture_model):
    """That the model counts as being a classifier."""
    assert supports_classification(fixture_model)


def test_raises_on_multitask():
    """That the model cannot be initialised with multiple y features.

    This is because pyod models are outlier detectors, i.e. intrinsically
    single-task.
    """
    y_features = ["y1", "y2"]
    with pytest.raises(ValueError, match=r".* single molflux: .*"):
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
def test_predict_proba(fixture_model, train_data, predict_data):
    """Test that a model can predict classification probabilities"""

    model = fixture_model

    model.train(train_data)
    probabilities = model.predict_proba(predict_data)
    assert probabilities
    assert len(probabilities) == len(model.y_features)
    for task_predictions in probabilities.values():
        assert len(task_predictions) == len(predict_data)


def test_saving_loading(tmp_path, fixture_model):
    """Test that a model can successfully be saved and loaded"""

    fixture_model.train(train_df)

    save_to_store(tmp_path, fixture_model)
    loaded_model = load_from_store(tmp_path)

    assert loaded_model.predict(predict_df)
