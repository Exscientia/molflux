import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from molflux.splits import config


@dataclass
class SplittingStrategyInfo:
    """
    Information about a splitting strategy.

    See the constructor arguments and properties for a full list.

    Note: Not all fields are known on construction and may be updated later.
    """

    # required
    description: str

    # Optional
    version: Optional[str] = None

    # set later by the builder / initialiser
    name: str = field(default_factory=str)
    tag: str = field(default_factory=str)
    split_description: str = field(default_factory=str)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_to_directory(self, directory: str) -> None:
        log_file = os.path.join(directory, config.SPLITTING_STRATEGY_INFO_FILENAME)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
