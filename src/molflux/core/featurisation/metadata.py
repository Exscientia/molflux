import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from cloudpathlib import AnyPath
from pydantic import BaseModel, Field

from molflux.core.environment import pip_working_set
from molflux.core.io import save_dict_to_json
from molflux.core.typing import PathLike

logger = logging.getLogger(__name__)

_FEATURISATION_METADATA_FILENAME = "featurisation_metadata.json"


class RepresentationConfig(BaseModel):
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    presets: Dict[str, Any] = Field(default_factory=dict)
    as_: Optional[Union[str, List[Optional[str]]]] = Field(
        default_factory=lambda: [None],
        alias="as",
    )


class ColumnFeaturisationConfig(BaseModel):
    column: str
    representations: List[RepresentationConfig]


class FeaturisationMetadataV1(BaseModel):
    version: int = Field(1, const=True)
    config: List[ColumnFeaturisationConfig] = Field(default_factory=list)
    runtime: Dict[str, Any] = Field(default_factory=pip_working_set)


def _placeholder_featurisation_metadata() -> Dict[str, Any]:
    """Returns an empty placeholder featurisation metadata payload"""
    return FeaturisationMetadataV1().dict(
        exclude_defaults=False,
        exclude_unset=False,
        exclude_none=False,
    )


def parse_featurisation_metadata(
    featurisation_metadata: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    """Performs soft validation of featurisation metadata based on schema version."""

    if "version" not in featurisation_metadata:
        raise KeyError("Featurisation metadata doesn't specify a 'version'")

    version = featurisation_metadata["version"]
    if version == 1:
        metadata = FeaturisationMetadataV1(**featurisation_metadata).dict(by_alias=True)
    # elif version == 2:
    #   ...
    else:
        raise NotImplementedError(
            f"Unsupported featurisation metadata version: {version!r}",
        )

    return metadata, version


def save_featurisation_metadata(
    featurisation_metadata: Optional[Dict[str, Any]],
    path: PathLike,
) -> str:
    """Saves featurisation metadata to disk.

    Args:
        featurisation_metadata: The featurisation metadata payload to be saved.
            If None or empty, an empty placeholder metadata schema is saved.
        path: The path under which to save the featurisation metadata.
    """

    if not featurisation_metadata:
        featurisation_metadata = _placeholder_featurisation_metadata()

    # Validate payload and fill-in
    featurisation_metadata, _ = parse_featurisation_metadata(featurisation_metadata)

    output_file = f"{path}/{_FEATURISATION_METADATA_FILENAME}"
    return save_dict_to_json(featurisation_metadata, path=output_file)


def fetch_model_featurisation_metadata(model_path: str) -> Dict[str, Any]:
    """Retrieves the featurisation metadata associated with a given model.

    Args:
        model_path: The path under which the model of interest is saved.

    Returns:
        The featurisation metadata that was used to generate the model's input
        features.
    """

    # Expected to be saved alongside the model artefact
    expected_metadata_path = os.path.join(model_path, _FEATURISATION_METADATA_FILENAME)
    return load_featurisation_metadata(expected_metadata_path)


def load_featurisation_metadata(path: str) -> Dict[str, Any]:
    """Loads json featurisation metadata from a given path."""
    return _cached_load_featurisation_metadata(path).copy()


@functools.lru_cache
def _cached_load_featurisation_metadata(path: str) -> Dict[str, Any]:
    """Loads json featurisation metadata from a given path, caching the result"""

    try:
        with AnyPath(path).open("r") as f:  # type: ignore[attr-defined]
            featurisation_metadata = json.load(f)
        logger.info(f"Fetched featurisation metadata: {path}")
    except Exception as error:
        logger.exception(error)
        raise FileNotFoundError(
            f"Error fetching featurisation metadata: could not fetch {path}",
        ) from error

    return featurisation_metadata  # type:ignore[no-any-return]
