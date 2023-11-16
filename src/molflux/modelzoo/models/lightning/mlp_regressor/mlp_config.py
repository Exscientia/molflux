from pydantic.dataclasses import dataclass

from molflux.modelzoo.models.lightning.config import ConfigDict, LightningConfig


@dataclass(config=ConfigDict)
class MLPConfig(LightningConfig):
    input_dim: int = 1
    num_layers: int = 2
    hidden_dim: int = 128
    num_tasks: int = 1

    def __post_init__(self) -> None:
        assert self.input_dim is not None, AttributeError("Must specify `input_dim`")

        if self.num_layers == 1:
            assert self.hidden_dim is None, AttributeError(
                "For 1 layer MLP, `hidden_dim` must be set to `None`.",
            )

        self.num_tasks = len(self.y_features)
