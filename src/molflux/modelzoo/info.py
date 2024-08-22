import importlib.metadata
from dataclasses import asdict, dataclass, field
from typing import Any


def get_runtime() -> dict[str, str]:
    distributions = importlib.metadata.distributions()
    return {
        distribution.metadata["Name"]: distribution.version
        for distribution in distributions
    }


@dataclass
class ModelInfo:
    """
    Information about a model.

    See the constructor arguments and properties for a full list.

    Note: Not all fields are known on construction and may be updated later.
    """

    # Model class identifiers
    name: str = field(default_factory=str)

    # Generic model metadata
    tag: str = field(default_factory=str)
    version: str | None = None

    model_description: str = field(default_factory=str)

    task_types: list[str] = field(default_factory=list)

    # Model config metadata
    config: dict[str, Any] = field(default_factory=dict)
    config_description: str = field(default_factory=str)

    # Runtime metadata
    runtime: dict[str, str] = field(default_factory=get_runtime)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> "ModelInfo":
        return cls(**dictionary)
