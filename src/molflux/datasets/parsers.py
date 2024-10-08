from collections.abc import Iterable
from typing import Any, Literal

import yaml
from pydantic.v1 import BaseModel, Field, ValidationError

from molflux.datasets.typing import PathLike


class Spec(BaseModel):
    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class YamlConfig(BaseModel):
    version: str
    kind: Literal["datasets"]
    specs: list[dict[str, Any]]


def dict_parser(dictionary: dict[str, Any]) -> Spec:
    """Parses a dictionary into a spec."""
    try:
        return Spec(**dictionary)
    except ValidationError as e:
        raise SyntaxError("Dictionary not conform with expected schema") from e


def yaml_parser(path: PathLike) -> Iterable[Spec]:
    """Parse a yaml file into specs."""

    with open(str(path)) as f:
        documents = yaml.safe_load_all(f)

        # Support multiple documents in same file
        for document in documents:
            try:
                config = YamlConfig(**document)

                if config.version == "v1":
                    specs = [dict_parser(spec) for spec in config.specs]
                    return specs

                else:
                    raise NotImplementedError(
                        f"Yaml config schema not available: {config.version}",
                    )

            except ValidationError:
                pass

    raise FileNotFoundError(f"Could not find valid document in file: {path}")
