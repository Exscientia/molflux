import pytest
import torch
from pydantic.error_wrappers import ValidationError
from torch._dynamo.testing import CompileCounter

from molflux.modelzoo import save_to_store
from molflux.modelzoo.models.lightning.config import (
    CompileConfig,
    TransferLearningStage,
)
from molflux.modelzoo.models.lightning.model import ConfigOverride


def test_transfer_learning_stage_misspelt_kwarg():
    with pytest.raises(ValidationError):
        TransferLearningStage(trainer={"accele": "cpu"})


@pytest.mark.parametrize(
    "compile",
    [True, False],
)
def test_transfer_learning_model(
    tmp_path,
    fixture_model,
    fixture_pre_trained_model,
    train_dataset,
    validation_dataset,
    compile,
    compile_error_context,
):
    """Test that transfer learning for one stage trains the right params and has correct weights"""

    model = fixture_pre_trained_model
    save_to_store(str(tmp_path), model)

    new_model = fixture_model
    with compile_error_context:
        new_model.train(
            train_data=train_dataset,
            validation_data=validation_dataset,
            trainer_config={
                "accelerator": "cpu",
                "max_epochs": 1,
                "default_root_dir": str(tmp_path),
            },
            datamodule_config={
                "train": {"batch_size": 2},
                "validation": {"batch_size": 2},
            },
            transfer_learning_config={
                "pre_trained_model_path": str(tmp_path),
                "stages": [
                    {
                        "freeze_modules": [
                            "module.0",
                        ],
                    },
                ],
            },
            compile_config=compile,
        )

        module = new_model.module if not compile else new_model.module._orig_mod

        for n, p in module.named_parameters():
            # check that matched and frozen is equal to
            if n == "module.0.weight":
                assert torch.equal(p, torch.ones_like(p) * 0)
            elif n == "module.0.bias":
                assert torch.equal(p, torch.ones_like(p) * 1)
            elif n == "module.2.weight":
                assert (not torch.equal(p, torch.ones_like(p) * 2)) and (
                    torch.allclose(p, torch.ones_like(p) * 2, atol=1e-3)
                )
            elif n == "module.2.bias":
                assert (not torch.equal(p, torch.ones_like(p) * 3)) and (
                    torch.allclose(p, torch.ones_like(p) * 3, atol=1e-3)
                )
            else:
                raise KeyError("Param not found in module")


def test_transfer_learning_model_overrides(fixture_model):
    """Test that transfer learning stage overrides work properly"""

    model = fixture_model

    config_override = ConfigOverride(
        model,
        transfer_learning={
            "pre_trained_model_path": "dummy_path",
            "stages": [
                {
                    "trainer": {
                        "max_epochs": 42,
                    },
                    "datamodule": {
                        "train": {"batch_size": 42},
                    },
                    "optimizer": {
                        "name": "AdamW",
                        "config": {
                            "lr": 42.0,
                        },
                    },
                    "scheduler": {
                        "config": {
                            "T_max": 10,
                        },
                    },
                },
            ],
        },
    )

    with config_override:
        with ConfigOverride(
            model,
            trainer=model.model_config.transfer_learning.stages[0].trainer,
            datamodule=model.model_config.transfer_learning.stages[0].datamodule,
            optimizer=model.model_config.transfer_learning.stages[0].optimizer,
            scheduler=model.model_config.transfer_learning.stages[0].scheduler,
        ):
            assert model.model_config.trainer.max_epochs == 42
            assert model.model_config.datamodule.train.batch_size == 42
            assert model.model_config.optimizer.name == "AdamW"
            assert model.model_config.optimizer.config["lr"] == 42.0
            assert model.model_config.scheduler.name == "CosineAnnealingLR"
            assert model.model_config.scheduler.config["T_max"] == 10


@pytest.mark.parametrize(
    "compile",
    [True, False],
)
def test_transfer_learning_model_partial_match(
    tmp_path,
    fixture_model,
    fixture_pre_trained_model,
    train_dataset,
    validation_dataset,
    compile,
    compile_error_context,
):
    """Test that transfer learning for one stage trains the right params and has correct weights"""

    model = fixture_pre_trained_model
    save_to_store(str(tmp_path), model)

    new_model = fixture_model
    with compile_error_context:
        new_model.train(
            train_data=train_dataset,
            validation_data=validation_dataset,
            trainer_config={
                "accelerator": "cpu",
                "max_epochs": 1,
                "default_root_dir": str(tmp_path),
            },
            datamodule_config={
                "train": {"batch_size": 2},
                "validation": {"batch_size": 2},
            },
            transfer_learning_config={
                "pre_trained_model_path": str(tmp_path),
                "modules_to_match": {"module.0": "module.0"},
                "stages": [
                    {
                        "freeze_modules": [
                            "module.2",
                        ],
                    },
                ],
            },
            compile_config=compile,
        )

        module = new_model.module if not compile else new_model.module._orig_mod

        for n, p in module.named_parameters():
            # check that matched but not frozen is close but not equal to
            if n == "module.0.weight":
                assert (not torch.equal(p, torch.ones_like(p) * 0)) and (
                    torch.allclose(p, torch.ones_like(p) * 0, atol=1e-3)
                )
            elif n == "module.0.bias":
                assert (not torch.equal(p, torch.ones_like(p) * 1)) and (
                    torch.allclose(p, torch.ones_like(p) * 1, atol=1e-3)
                )
            elif n == "module.2.weight":
                assert (not torch.equal(p, torch.ones_like(p) * 2)) and (
                    not torch.allclose(p, torch.ones_like(p) * 2, atol=1e-3)
                )
            elif n == "module.2.bias":
                assert (not torch.equal(p, torch.ones_like(p) * 3)) and (
                    not torch.allclose(p, torch.ones_like(p) * 3, atol=1e-3)
                )
            else:
                raise KeyError("Param not found in module")


@pytest.mark.parametrize(
    "compile",
    [True, False],
)
def test_transfer_learning_model_multistage(
    tmp_path,
    fixture_model,
    fixture_pre_trained_model,
    train_dataset,
    validation_dataset,
    compile,
    compile_error_context,
):
    """Test that transfer learning for multistages trains the right params and has correct weights"""

    model = fixture_pre_trained_model
    save_to_store(str(tmp_path), model)

    new_model = fixture_model
    with compile_error_context:
        new_model.train(
            train_data=train_dataset,
            validation_data=validation_dataset,
            trainer_config={
                "accelerator": "cpu",
                "max_epochs": 1,
                "default_root_dir": str(tmp_path),
            },
            datamodule_config={
                "train": {"batch_size": 2},
                "validation": {"batch_size": 2},
            },
            transfer_learning_config={
                "pre_trained_model_path": str(tmp_path),
                "stages": [
                    {
                        "freeze_modules": [
                            "module.0",
                        ],
                    },
                    {
                        "freeze_modules": [
                            "module.2",
                        ],
                    },
                ],
            },
            compile_config=compile,
        )

        module = new_model.module if not compile else new_model.module._orig_mod

        for n, p in module.named_parameters():
            if n == "module.0.weight":
                assert (not torch.equal(p, torch.ones_like(p) * 0)) and (
                    torch.allclose(p, torch.ones_like(p) * 0, atol=1e-3)
                )
            elif n == "module.0.bias":
                assert (not torch.equal(p, torch.ones_like(p) * 1)) and (
                    torch.allclose(p, torch.ones_like(p) * 1, atol=1e-3)
                )
            elif n == "module.2.weight":
                assert (not torch.equal(p, torch.ones_like(p) * 2)) and (
                    torch.allclose(p, torch.ones_like(p) * 2, atol=1e-3)
                )
            elif n == "module.2.bias":
                assert (not torch.equal(p, torch.ones_like(p) * 3)) and (
                    torch.allclose(p, torch.ones_like(p) * 3, atol=1e-3)
                )
            else:
                raise KeyError("Param not found in module")


@pytest.mark.parametrize(
    "compile",
    [True, False],
)
def test_transfer_learning_model_bad_module_name(
    tmp_path,
    fixture_model,
    fixture_pre_trained_model,
    train_dataset,
    validation_dataset,
    compile,
    compile_error_context,
):
    """Test that transfer learning errors on bad module name"""
    model = fixture_pre_trained_model
    save_to_store(str(tmp_path), model)

    new_model = fixture_model

    with pytest.raises(KeyError), compile_error_context:
        new_model.train(
            train_data=train_dataset,
            validation_data=validation_dataset,
            trainer_config={
                "accelerator": "cpu",
                "max_epochs": 1,
                "default_root_dir": str(tmp_path),
            },
            datamodule_config={
                "train": {"batch_size": 2},
                "validation": {"batch_size": 2},
            },
            transfer_learning_config={
                "pre_trained_model_path": str(tmp_path),
                "modules_to_match": {"bad_name": "module.0"},
                "stages": [
                    {
                        "freeze_modules": [
                            "module.0",
                        ],
                    },
                ],
            },
            compile_config=compile,
        )

    with pytest.raises(KeyError), compile_error_context:
        new_model.train(
            train_data=train_dataset,
            validation_data=validation_dataset,
            trainer_config={
                "accelerator": "cpu",
                "max_epochs": 1,
                "default_root_dir": str(tmp_path),
            },
            datamodule_config={
                "train": {"batch_size": 2},
            },
            transfer_learning_config={
                "pre_trained_model_path": str(tmp_path),
                "modules_to_match": {"module.0": "bad_name"},
                "stages": [
                    {
                        "freeze_modules": [
                            "module.0",
                        ],
                    },
                ],
            },
            compile_config=compile,
        )

    with pytest.raises(KeyError), compile_error_context:
        new_model.train(
            train_data=train_dataset,
            validation_data=validation_dataset,
            trainer_config={
                "accelerator": "cpu",
                "max_epochs": 1,
                "default_root_dir": str(tmp_path),
            },
            datamodule_config={
                "train": {"batch_size": 2},
            },
            transfer_learning_config={
                "pre_trained_model_path": str(tmp_path),
                "stages": [
                    {
                        "freeze_modules": [
                            "bad_name",
                        ],
                    },
                ],
            },
            compile_config=compile,
        )


@pytest.mark.parametrize("compile", [True, False])
def test_does_compile(
    tmp_path,
    fixture_model,
    fixture_pre_trained_model,
    train_dataset,
    validation_dataset,
    compile_error_context,
    compile,
):
    """Test that the model does compile during training when set to"""
    # Python 3.11 does not support compilation as of PyTorch 2.0.1, but 2.1 will

    model = fixture_pre_trained_model
    save_to_store(str(tmp_path), model)

    new_model = fixture_model

    if compile:
        cnt = CompileCounter()
        compile_config = CompileConfig("default")
        new_model.model_config.compile = compile_config  # pyright: ignore
        new_model.model_config.compile.backend = cnt  # pyright: ignore

        assert cnt.frame_count == 0

    assert not new_model.is_compiled

    with compile_error_context:
        new_model.train(
            train_data=train_dataset,
            validation_data=validation_dataset,
            trainer_config={
                "accelerator": "cpu",
                "max_epochs": 1,
                "default_root_dir": str(tmp_path),
            },
            datamodule_config={
                "train": {"batch_size": 2},
                "validation": {"batch_size": 2},
            },
            transfer_learning_config={
                "pre_trained_model_path": str(tmp_path),
                "stages": [
                    {
                        "freeze_modules": [
                            "module.0",
                        ],
                    },
                ],
            },
        )

        if compile:
            assert cnt.frame_count > 0  # pyright: ignore
            assert new_model.is_compiled
