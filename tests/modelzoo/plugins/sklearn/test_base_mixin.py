import warnings

import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.exceptions import DataConversionWarning

import datasets
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.errors import NotTrainedError
from molflux.modelzoo.load import load_from_dict, load_from_yaml, load_model
from molflux.modelzoo.models.sklearn.random_forest_regressor import (
    RandomForestRegressor,
)
from molflux.modelzoo.protocols import Estimator, Model

# Testing the SKLearn model base on a concrete implementation (random forest)
model_name = "random_forest_regressor"


@pytest.fixture(scope="function")
def fixture_model() -> Model:
    return load_model(model_name)


def test_in_catalogue():
    """That the model is registered in the catalogue."""
    catalogue = list_models()
    all_names = [name for names in catalogue.values() for name in names]
    assert model_name in all_names


def test_is_mapped_to_correct_class(fixture_model):
    """That the model name is mapped to the appropriate class."""
    model = fixture_model
    assert isinstance(model, RandomForestRegressor)


def test_implements_protocol(fixture_model):
    """That the model implements the public Estimator protocol."""
    model = fixture_model
    assert isinstance(model, Estimator)


def test_forwards_init_kwargs_to_builder():
    """That keyword arguments get forwarded to the model initialiser."""
    name = model_name
    model = load_model(name=name, tag="pytest-tag")
    assert model.tag == "pytest-tag"


def test_validates_init_kwargs():
    """That keyword arguments are validated.

    This is to improve user experience by failing fast if keyword arguments
    that do not match the expected types are provided on initialisation.
    """
    name = model_name

    with pytest.raises(ValidationError, match="1 validation error"):
        load_model(
            name=name,
            tag="pytest-tag",
            x_features=["good"],
            y_features="not-a-list",
        )


def test_from_dict_returns_model():
    """That loading from a dict returns a Model."""
    name = model_name
    config = {
        "name": name,
        "config": {},
    }
    model = load_from_dict(config)
    assert isinstance(model, Estimator)


def test_from_minimal_dict():
    """That can provide a config with only required fields."""
    name = model_name
    config = {
        "name": name,
    }
    assert load_from_dict(config)


def test_from_dict_forwards_config_to_model_builder():
    """That config keyword arguments get forwarded to the model initialiser."""
    name = model_name
    config = {
        "name": name,
        "config": {
            "tag": "pytest-tag",
        },
    }
    model = load_from_dict(config)
    assert model.tag == "pytest-tag"


def test_from_yaml_returns_collection_of_models(fixture_path_to_assets):
    path = fixture_path_to_assets / "core" / "config.yml"
    models = load_from_yaml(path=path)
    assert len(models) == 2
    assert "random_forest_regressor" in models
    assert "custom_classifier" in models
    assert all(isinstance(model, Estimator) for model in models.values())
    assert models["custom_classifier"].config.get("max_features") == "log2"


def test_attempt_training_on_featureless_model_raises():
    """That an exception is raised if training on a model that does not define
    features.

    References:

    """
    name = model_name
    model = load_model(name=name)
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1, 2, 3]}))
    with pytest.raises(ValueError, match="features"):
        model.train(train_data=data)


def test_training_dataset_must_contain_model_x_features():
    """That the dataset used for training must contain the x features defined by
    the model.
    """
    name = model_name
    x_features = ["x1", "x2"]
    model = load_model(name=name, x_features=x_features)
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1, 2, 3]}))
    with pytest.raises(ValueError, match="Data does not have required feature: 'x2'"):
        model.train(train_data=data)


def test_training_dataset_must_contain_model_y_features():
    """That the dataset used for training must contain the y features defined by
    the model.
    """
    name = model_name
    x_features = ["x1"]
    y_features = ["y1", "y2"]
    model = load_model(name=name, x_features=x_features, y_features=y_features)
    data = datasets.Dataset.from_pandas(
        pd.DataFrame({"x1": [1, 2, 3], "y1": [1, 2, 3]}),
    )
    with pytest.raises(ValueError, match="Data does not have required feature: 'y2'"):
        model.train(train_data=data)


def test_attempt_training_on_featureless_y_model_raises():
    """That an exception is raised if training on a model that does not define
    y_features.

    References:
    """
    name = model_name
    model = load_model(name=name, x_features=["x1"])
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1, 2, 3]}))
    with pytest.raises(ValueError, match="features"):
        model.train(train_data=data)


def test_attempt_training_on_featureless_x_model_raises():
    """That an exception is raised if training on a model that does not define
    x_features.

    References:
    """
    name = model_name
    model = load_model(name=name, y_features=["y1"])
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1, 2, 3]}))
    with pytest.raises(ValueError, match="feature"):
        model.train(train_data=data)


@pytest.mark.skip(reason="re-implement the logic for checking whether model is trained")
def test_cannot_predict_if_model_not_trained():
    """That the dataset used for predictions must contain the x features defined by
    the model.
    """
    name = model_name
    x_features = ["x1"]
    model = load_model(name=name, x_features=x_features)
    data = datasets.Dataset.from_pandas(pd.DataFrame({"x1": [1, 2, 3]}))
    with pytest.raises(NotTrainedError):
        model.predict(data=data)


train_df = pd.DataFrame({"x1": [1.5], "y1": [2]})
predict_df = pd.DataFrame({"x1": [3]})


@pytest.mark.parametrize(
    "train_data",
    [
        train_df,
        datasets.Dataset.from_pandas(train_df),
    ],
)
def test_train_on_single_sample(train_data):
    """That can train on a single sample.

    References:
    """
    name = model_name
    model = load_model(name=name, x_features=["x1"], y_features=["y1"])
    model.train(train_data=train_data)
    assert True


@pytest.mark.parametrize(
    "train_data, predict_data",
    [
        (train_df, predict_df),
        (
            datasets.Dataset.from_pandas(train_df),
            datasets.Dataset.from_pandas(predict_df),
        ),
    ],
)
def test_predict_on_single_sample(train_data, predict_data):
    """That can predict on a single sample.

    References:
    """
    name = model_name
    model = load_model(name=name, x_features=["x1"], y_features=["y1"])
    model.train(train_data=train_data)
    assert model.predict(data=predict_data)


def test_training_models_with_single_y_feature_does_not_raise_data_conversion_warning():
    """That can train a single-feature model without DataConversionWarnings.

    References:
    """
    name = model_name
    dataset = datasets.Dataset.from_dict({"x1": [1, 2, 3], "y1": [3, 2, 1]})
    model = load_model(name=name, x_features=["x1"], y_features=["y1"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        model.train(dataset)
