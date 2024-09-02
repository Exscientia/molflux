from dataclasses import dataclass
from typing import Any

from molflux.modelzoo.protocols import Model


@dataclass
class ModelArtefact:
    name: str
    tag: str
    config: dict[str, Any]
    model: Model


def pack(model: Model) -> ModelArtefact:
    name = model.name
    tag = model.tag
    config = model.config
    model = model

    return ModelArtefact(
        name=name,
        tag=tag,
        config=config,
        model=model,
    )


def unpack(artefact: ModelArtefact) -> Model:
    model = artefact.model
    return model
