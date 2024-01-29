import functools
import importlib.metadata
from typing import Dict, List

from cloudpathlib import AnyPath

from molflux.core.typing import PathLike

_REQUIREMENTS_FILENAME = "requirements.txt"


def pip_working_set() -> Dict[str, str]:
    return _cached_pip_working_set().copy()


@functools.lru_cache
def _cached_pip_working_set() -> Dict[str, str]:
    distributions = importlib.metadata.distributions()
    return {
        distribution.metadata["Name"]: distribution.version
        for distribution in distributions
    }


@functools.lru_cache
def _get_pinned_pip_environment() -> List[str]:
    """Returns a list of pinned package versions in the active python environment.

    This is cached for performance reasons.
    """
    return sorted(
        [f"{package}=={version}" for package, version in pip_working_set().items()],
    )


def save_python_environment(path: PathLike) -> str:
    """Saves metadata about the active python environment to disk."""
    requirements = _get_pinned_pip_environment()
    output_file = f"{path}/{_REQUIREMENTS_FILENAME}"
    with AnyPath(output_file).open("w") as f:  # type: ignore[attr-defined]
        f.write("\n".join(requirements))
    return output_file
