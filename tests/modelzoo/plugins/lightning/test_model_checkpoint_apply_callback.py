import pandas as pd
import pytest
import torch

import datasets
from molflux.modelzoo.load import load_model
from molflux.modelzoo.protocols import Model

model_name = "lightning_mlp_regressor"

_X_FEATURES = ["X_col_1", "X_col_2"]
_Y_FEATURES = ["y_col"]

train_df = pd.DataFrame(
    [
        [1.0, 2.0, 10.0],
        [1.0, 3.0, 11.0],
        [2.0, 4.0, 13.0],
        [2.0, 4.0, 13.0],
        [2.0, 4.0, 13.0],
    ],
    columns=_X_FEATURES + _Y_FEATURES,
)

validation_df = pd.DataFrame(
    [
        [1.0, 2.0, 12.0],
        [1.0, 3.0, 13.0],
        [2.0, 4.0, 14.0],
        [2.0, 4.0, 15.0],
        [2.0, 5.0, 16.0],
    ],
    columns=_X_FEATURES + _Y_FEATURES,
)


@pytest.fixture(scope="function")
def fixture_model() -> Model:
    return load_model(
        model_name,
        x_features=_X_FEATURES,
        y_features=_Y_FEATURES,
        input_dim=2,
    )


@pytest.mark.parametrize(
    "train_data, validation_data",
    [
        (
            datasets.Dataset.from_pandas(train_df),
            datasets.Dataset.from_pandas(validation_df),
        ),
    ],
)
def test_model_checkpoint_apply_callback(
    fixture_model,
    tmp_path,
    train_data,
    validation_data,
):
    """Test that ModelCheckpointApply callback applies best ckpt at the end of fit"""

    model = fixture_model

    model.train(
        train_data=train_data,
        validation_data=validation_data,
        trainer_config={
            "accelerator": "cpu",
            "max_epochs": 20,
            "default_root_dir": str(tmp_path),
            "enable_checkpointing": True,
            "callbacks": [
                {
                    "name": "ModelCheckpointApply",
                    "config": {
                        "monitor": "val/total/loss",
                        "save_last": True,
                        "save_top_k": 1,
                    },
                },
            ],
        },
        datamodule_config={
            "train": {"batch_size": 2},
            "validation": {"batch_size": 2},
        },
    )

    # iterate through logdir and find best ckpt path. This is found under checkpoints and
    # starts with epoch={num_best_epoch}
    for p in (tmp_path / "lightning_logs" / "version_0" / "checkpoints").iterdir():
        if "epoch=" in str(p):
            best_ckpt_path = str(p)

    best_ckpt = torch.load(best_ckpt_path)

    # assert that state dicts match
    for k, v in model.module.state_dict().items():
        assert torch.equal(v, best_ckpt["state_dict"][k])
