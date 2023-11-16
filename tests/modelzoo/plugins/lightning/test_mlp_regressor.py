import numpy as np
import pytest
import torch
from torch._dynamo.testing import CompileCounter

from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.catalogue import list_models
from molflux.modelzoo.models.lightning.config import CompileConfig
from molflux.modelzoo.models.lightning.mlp_regressor.mlp_model import (
    LightningMLPRegressor,
)
from molflux.modelzoo.protocols import Estimator

model_name = "lightning_mlp_regressor"


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
    assert isinstance(model, LightningMLPRegressor)


def test_implements_protocol(fixture_model):
    """That the model implements the public Estimator protocol."""
    model = fixture_model
    assert isinstance(model, Estimator)


def test_train_predict_model(
    fixture_model,
    tmp_path,
    train_dataset,
    validation_dataset,
    predict_dataset,
):
    """Test that a model can run the train and predict functions"""

    model = fixture_model

    model.train(
        train_data=train_dataset,
        validation_data=validation_dataset,
        trainer_config={
            "accelerator": "cpu",
            "max_epochs": 5,
            "default_root_dir": str(tmp_path),
        },
        datamodule_config={
            "train": {"batch_size": 2},
            "validation": {"batch_size": 2},
        },
    )
    predictions = model.predict(
        predict_dataset,
        datamodule_config={"predict": {"batch_size": 2}},
    )
    assert predictions
    assert len(predictions) == len(model.y_features)
    for task_predictions in predictions.values():
        assert len(task_predictions) == len(predict_dataset)


def test_train_multi_data(
    fixture_model,
    tmp_path,
    train_dataset,
    train_dataset_b,
    validation_dataset,
):
    """Test that a model can train with multiple datasets"""

    model = fixture_model

    # NOTE train_dataset has 5 examples, train_dataset_b has 8
    model.train(
        train_data={"a": train_dataset, "b": train_dataset_b},
        validation_data={"a": validation_dataset, "c": validation_dataset},
        trainer_config={
            "accelerator": "cpu",
            "max_epochs": 5,
            "default_root_dir": str(tmp_path),
        },
        datamodule_config={
            "train": {"batch_size": 2},
            "validation": {"batch_size": 2},
        },
    )


def test_saving_loading(
    tmp_path,
    fixture_model,
    train_dataset,
    validation_dataset,
    predict_dataset,
):
    """Test that a model can successfully be saved and loaded"""

    fixture_model.train(
        train_data=train_dataset,
        validation_data=validation_dataset,
        trainer_config={
            "accelerator": "cpu",
            "max_epochs": 5,
            "default_root_dir": str(tmp_path),
        },
        datamodule_config={
            "train": {"batch_size": 2},
            "validation": {"batch_size": 2},
        },
    )

    save_to_store(tmp_path, fixture_model)
    loaded_model = load_from_store(tmp_path)

    assert loaded_model.predict(
        predict_dataset,
        datamodule_config={
            "predict": {"batch_size": 2},
        },
    )


@pytest.mark.parametrize("compile", [True, False])
def test_does_compile(
    tmp_path,
    fixture_model,
    train_dataset,
    validation_dataset,
    compile,
    compile_error_context,
):
    """Test that the model does compile during training when set to"""
    # Python 3.11 does not support compilation as of PyTorch 2.0.1, but 2.1 will

    assert not fixture_model.is_compiled

    if compile:
        cnt = CompileCounter()
        fixture_model.model_config.compile = CompileConfig("default")
        fixture_model.model_config.compile.backend = cnt  # pyright: ignore

        assert cnt.frame_count == 0

    with compile_error_context:
        fixture_model.train(
            train_data=train_dataset,
            validation_data=validation_dataset,
            trainer_config={
                "accelerator": "cpu",
                "max_epochs": 5,
                "default_root_dir": str(tmp_path),
            },
            datamodule_config={
                "train": {"batch_size": 2},
                "validation": {"batch_size": 2},
            },
        )

        if compile:
            assert fixture_model.is_compiled
            assert cnt.frame_count > 0  # pyright: ignore


def test_train_model_overfit(
    fixture_model,
    tmp_path,
    train_dataset,
):
    """Test that a model can overfit to a small batch"""

    torch.manual_seed(0)

    model = fixture_model

    model.train(
        train_data=train_dataset,
        validation_data=None,
        trainer_config={
            "accelerator": "cpu",
            "max_epochs": 400,
            "default_root_dir": str(tmp_path),
        },
        datamodule_config={
            "train": {"batch_size": 2},
            "validation": {"batch_size": 2},
        },
        optimizer_config={"name": "SGD", "config": {"lr": 1e-3}},
        scheduler_config={
            "name": "OneCycleLR",
            "config": {
                "max_lr": 1e-3,
                "total_steps": "num_steps",
                "pct_start": 0.1,
                "anneal_strategy": "linear",
            },
        },
    )
    predictions = model.predict(
        train_dataset,
        datamodule_config={"predict": {"batch_size": 2}},
    )

    np.testing.assert_allclose(
        predictions["lightning_mlp_regressor::y_col"],
        train_dataset["y_col"],
        rtol=5e-2,
    )
