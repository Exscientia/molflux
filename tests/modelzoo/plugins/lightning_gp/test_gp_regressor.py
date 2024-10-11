import numpy as np
import torch

from molflux.modelzoo import load_from_dict, load_from_store, save_to_store
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.models.lightning_gp.gp_model import LightningGPRegressor
from molflux.modelzoo.protocols import Estimator, SupportsStandardDeviation

model_name = "lightning_gp_regressor"


def test_in_catalogue():
    """That the model is registered in the catalogue."""
    catalogue = list_models()
    all_names = [name for names in catalogue.values() for name in names]
    assert model_name in all_names


def test_default_model_tag_matches_entrypoint_name(fixture_gp_model):
    """That the default model tag matches the catalogue entrypoint name.

    This is not strictly required, but ensures a more intuitive user experience.
    """
    model = fixture_gp_model
    assert model.tag == model_name


def test_is_mapped_to_correct_class(fixture_gp_model):
    """That the model name is mapped to the appropriate class."""
    model = fixture_gp_model
    assert isinstance(model, LightningGPRegressor)


def test_implements_protocol(fixture_gp_model):
    """That the model implements the public Estimator protocol."""
    model = fixture_gp_model
    assert isinstance(model, Estimator)


def test_implements_standard_deviation_protocol(fixture_gp_model):
    """That the model implements the standard deviation protocol."""
    model = fixture_gp_model
    assert isinstance(model, SupportsStandardDeviation)


def test_train_predict_model(
    fixture_gp_model,
    tmp_path,
    train_dataset,
    predict_dataset,
    empty_predict_dataset,
    single_row_predict_dataset,
):
    """Test that a model can run the train, predict and predict_with_std functions"""

    model = fixture_gp_model

    model.train(
        train_data=train_dataset,
        validation_data=None,
        trainer_config={
            "accelerator": "cpu",
            "max_epochs": 5,
            "default_root_dir": str(tmp_path),
        },
    )

    for prediction_dataset in [
        empty_predict_dataset,
        single_row_predict_dataset,
        predict_dataset,
    ]:
        predictions = model.predict(prediction_dataset)
        assert len(predictions) == len(model.y_features)

        for task in predictions:
            assert len(predictions[task]) == len(prediction_dataset)

        predictions_from_std_method, stds = model.predict_with_std(prediction_dataset)
        assert predictions_from_std_method and stds
        assert len(predictions_from_std_method) == len(model.y_features) and len(
            stds,
        ) == len(model.y_features)
        for task in predictions_from_std_method:
            assert len(predictions_from_std_method[task]) == len(predictions[task])


def test_saving_loading(
    tmp_path,
    fixture_gp_model,
    train_dataset,
    predict_dataset,
):
    """Test that a model can successfully be saved and loaded"""

    fixture_gp_model.train(
        train_data=train_dataset,
        validation_data=None,
        trainer_config={
            "accelerator": "cpu",
            "max_epochs": 5,
            "default_root_dir": str(tmp_path),
        },
    )

    save_to_store(tmp_path, fixture_gp_model)
    loaded_model = load_from_store(tmp_path)

    assert loaded_model.predict(predict_dataset)


def test_train_model_overfit(
    fixture_gp_model,
    tmp_path,
    train_dataset,
):
    """Test that a model can overfit to a small batch"""

    torch.manual_seed(0)

    model = fixture_gp_model

    model.train(
        train_data=train_dataset,
        validation_data=None,
        trainer_config={
            "accelerator": "cpu",
            "max_epochs": 400,
            "default_root_dir": str(tmp_path),
        },
        optimizer_config={"name": "SGD", "config": {"lr": 1}},
    )
    predictions = model.predict(
        train_dataset,
    )

    np.testing.assert_allclose(
        predictions[f"{model_name}::y_col"],
        train_dataset["y_col"],
        rtol=5e-2,
    )


def test_validation_configs(fixture_gp_config, train_dataset, predict_dataset):
    """Test that running with the NoValidation parameter works"""
    config = fixture_gp_config
    config["config"]["validation_config"] = {"name": "NoValidation"}

    model = load_from_dict(config)
    model.train(train_dataset)
    predictions = model.predict(train_dataset)

    assert len(predictions) == len(model.y_features)
