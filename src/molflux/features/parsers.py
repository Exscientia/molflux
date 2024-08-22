import warnings
from collections.abc import Iterable
from typing import Any, Literal

import pydantic
import yaml
from pydantic import BaseModel, Field

from molflux.features.typing import PathLike

_pydantic_v2 = tuple(map(int, pydantic.__version__.split("."))) >= (2, 0, 0)


class Spec(BaseModel):
    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    presets: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        field_aliases = {
            field.alias if field.alias else field_name
            for field_name, field in (
                self.model_fields if _pydantic_v2 else self.__fields__
            ).items()
        }

        if extra_fields := set(data).difference(field_aliases):
            warnings.warn(
                (
                    "Additional fields not covered by the spec provided: "
                    f"{*extra_fields,}, these will be ignored. Should these be "
                    "provided in `presets`?"
                ),
                stacklevel=5,
                category=UserWarning,
            )
        super().__init__(**data)


class YamlConfig(BaseModel):
    version: str
    kind: Literal["representations"]
    specs: list[dict[str, Any]]


def dict_parser(dictionary: dict[str, Any]) -> Spec:
    """Parses a dictionary into a spec."""
    try:
        return Spec(**dictionary)
    except pydantic.ValidationError as e:
        raise SyntaxError("Dictionary not conform with expected schema") from e


def yaml_parser(path: PathLike) -> Iterable[Spec]:
    """Parse a yaml file into specs."""

    with open(path) as f:
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

            except pydantic.ValidationError:
                pass

    raise FileNotFoundError(f"Could not find valid document in file: {path}")
