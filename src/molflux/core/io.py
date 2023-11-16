import json
from typing import Mapping

from cloudpathlib import AnyPath

from molflux.core.typing import PathLike


def _make_parents_if_needed(path: PathLike) -> None:
    """Creates missing parents of the given path."""
    parent = AnyPath(path).parent  # type: ignore[attr-defined]
    if not parent.is_dir():
        parent.mkdir(parents=True, exist_ok=True)


def save_dict_to_json(mapping: Mapping, path: PathLike, parents: bool = True) -> str:
    """Saves a dictionary to storage as .json

    Args:
        mapping: The dictionary to save.
        path: The path at which to save the dictionary. Can be local or cloud.
        parents: If true, any missing parents of this path are created as needed

    Returns:
        The path to the persisted file.
    """

    if parents is True:
        _make_parents_if_needed(path)

    with AnyPath(path).open("w") as fp:  # type: ignore[attr-defined]
        json.dump(
            mapping,
            fp,
            indent=4,
            sort_keys=True,
        )
    return str(path)
