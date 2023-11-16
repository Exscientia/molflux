from molflux.modelzoo.protocols import Model
from molflux.modelzoo.store.artefact import pack, unpack
from molflux.modelzoo.store.manager import (
    load_artefact_from_store,
    save_artefact_to_store,
)
from molflux.modelzoo.typing import PathLike


def save_to_store(key: PathLike, model: Model) -> None:
    """Saves a pre-trained modelzoo model to a model store."""

    # Pack modelzoo model into artefact
    artefact = pack(model=model)

    # Store the artefact in the store
    save_artefact_to_store(artefact=artefact, directory=str(key))


def load_from_store(key: PathLike) -> Model:
    """Loads a pre-trained modelzoo model from a model store."""

    # Generate a modelzoo artefact from the store
    artefact = load_artefact_from_store(directory=str(key))

    #  get model from artefact
    model = unpack(artefact=artefact)
    return model
