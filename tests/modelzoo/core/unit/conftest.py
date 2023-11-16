from typing import Any, Dict, Type

import pytest

import datasets
from molflux.modelzoo.catalogue import register_model
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.load import load_model
from molflux.modelzoo.model import ModelBase, ModelConfig
from molflux.modelzoo.protocols import Model


class MockModelConfig(ModelConfig):
    ...


# Registers a mock model in the catalogue
@register_model(kind="pytest", name="mock_model")
class MockModel(ModelBase[MockModelConfig]):
    @property
    def _config_builder(self) -> Type[MockModelConfig]:
        return MockModelConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="A mock model",
        )

    def _train(self, train_data: datasets.Dataset, **kwargs: Any) -> None:
        self.model = True

    def _predict(self, data: datasets.Dataset, **kwargs: Any) -> Dict[str, Any]:
        return {self.tag: "Spooky Result!"}


@pytest.fixture()
def fixture_mock_model_untrained() -> Model:
    """Returns an untrained mock model."""
    return load_model("mock_model", tag="custom_tag")


@pytest.fixture()
def fixture_mock_model_trained() -> Model:
    """Returns a trained mock model."""
    model = load_model(
        "mock_model",
        tag="custom_tag",
        x_features=["x"],
        y_features=["y"],
    )
    model.train(datasets.Dataset.from_dict({"x": [1], "y": [1]}))
    return model
