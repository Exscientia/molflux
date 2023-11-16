import json
import logging
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from cloudpathlib import AnyPath, CloudPath

from molflux.modelzoo import config
from molflux.modelzoo.load import load_model
from molflux.modelzoo.protocols import Model
from molflux.modelzoo.store.artefact import ModelArtefact

logger = logging.getLogger(__name__)

_AnyPath = Union[Path, CloudPath]


def _rmtree(path: _AnyPath) -> None:
    if isinstance(path, Path):
        shutil.rmtree(str(path))
    elif isinstance(path, CloudPath):
        path.rmtree()
    else:
        raise ValueError(f"Unsupported path type: {type(path)!r}")


def _save_model_metadata(artefact: ModelArtefact, directory: _AnyPath) -> str:
    """Saves model metadata as json to a directory."""
    from molflux import __version__

    metadata = {
        "name": artefact.name,
        "tag": artefact.tag,
        "config": artefact.config,
        "version": __version__,
    }

    log_file = directory / config.MODEL_CONFIG_FILENAME

    if log_file.exists():
        logger.warning("Overwriting existing model metadata: %s", log_file)
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    return str(log_file)


def _save_binaries(artefact: ModelArtefact, directory: _AnyPath) -> str:
    """Serialises a model into a given directory."""
    model = artefact.model

    # The 'official' subdirectory into which to serialise the model
    artefacts_dir = directory / config.MODEL_ARTEFACT_DIR_FILENAME

    if artefacts_dir.exists():
        logger.warning("Overwriting existing model artefacts: %s", artefacts_dir)
    else:
        artefacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        # if saving to cloud, we will need to upload a temporary directory as
        # models only know how to save themselves to disk
        if isinstance(artefacts_dir, CloudPath):
            with tempfile.TemporaryDirectory() as tmpdir:
                model.as_dir(directory=str(tmpdir))
                ok = artefacts_dir.upload_from(tmpdir, force_overwrite_to_cloud=True)
            if not ok:
                raise RuntimeError("Error uploading model artefact to store.")
        else:
            model.as_dir(directory=str(artefacts_dir))

    except Exception:
        # cleanup and raise exception up the stack
        _rmtree(artefacts_dir)
        raise

    return str(artefacts_dir)


def _load_model_metadata(directory: _AnyPath) -> Tuple[str, str, Dict[str, Any]]:
    """Loads model metadata from a directory.

    The directory is expected to contain a metadata file as generated
    by `_save_model_metadata()`.
    """
    from molflux import __version__

    log_file = directory / config.MODEL_CONFIG_FILENAME
    with log_file.open("r", encoding="utf-8") as f:
        metadata: Dict[str, Any] = json.load(f)

    # Peek at stored model versioning
    version = metadata.get("version")
    if version and version != __version__:
        warnings.warn(
            f"Loading model built by molflux.modelzoo=={version}. "
            f"Behaviour may be different in this version.",
            stacklevel=2,
        )

    # Extract model identifier that will be used to load relevant class
    name = metadata.pop("name", None)
    if name is None:
        raise KeyError("Invalid model metadata: missing `name`.")

    tag = metadata.pop("tag", None)  # not essential

    model_config = metadata.pop("config", None)
    if model_config is None:
        raise KeyError("Invalid model metadata: missing `config`.")

    return name, tag, model_config


def _load_from_binaries(
    directory: _AnyPath,
    name: str,
    tag: str,
    model_config: Dict[str, Any],
) -> Model:
    """Deserialises a model that has been serialised in a given directory.

    The binaries are expected to have been created with `_save_binaries()`.
    """

    # The 'official' directory for model binaries
    artefacts_dir = directory / config.MODEL_ARTEFACT_DIR_FILENAME

    # Load model class and then use it to deserialise binaries
    model = load_model(name=name, tag=tag, **model_config)

    # if loading from cloud, we will need to download to a temporary directory
    # as models only know how to load themselves from disk
    if isinstance(artefacts_dir, CloudPath):
        with tempfile.TemporaryDirectory() as tmpdir:
            ok = artefacts_dir.download_to(destination=tmpdir)
            if not ok:
                raise RuntimeError("Error downloading model artefact from store.")
            model.from_dir(directory=str(tmpdir))
    else:
        model.from_dir(directory=str(artefacts_dir))

    return model


def save_artefact_to_store(artefact: ModelArtefact, directory: str) -> None:
    directory_path: _AnyPath = AnyPath(directory)  # type: ignore[assignment]

    # save model config
    _save_model_metadata(artefact=artefact, directory=directory_path)

    # save model binaries
    _save_binaries(artefact=artefact, directory=directory_path)


def load_artefact_from_store(directory: str) -> ModelArtefact:
    directory_path: _AnyPath = AnyPath(directory)  # type: ignore[assignment]
    if not directory_path.exists():
        raise FileNotFoundError(f"No such directory: {directory!r}")
    if not directory_path.is_dir():
        raise ValueError(f"Is not a directory: {directory!r}")

    # load model metadata
    name, tag, model_config = _load_model_metadata(directory=directory_path)

    # load model
    model = _load_from_binaries(
        name=name,
        tag=tag,
        model_config=model_config,
        directory=directory_path,
    )

    return ModelArtefact(
        name=name,
        tag=tag,
        config=model_config,
        model=model,
    )
