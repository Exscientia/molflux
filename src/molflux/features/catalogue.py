import difflib
import functools
import inspect
import logging
from collections import defaultdict
from importlib.metadata import EntryPoint, entry_points
from typing import Callable, Dict, List, Type

from molflux.features.naming import camelcase_to_snakecase
from molflux.features.representation import Representation

logger = logging.getLogger(__name__)

RepresentationT = Type[Representation]

NAMESPACE = "molflux.features.plugins."

# This is where entrypoints will be registered {<name>: <Entrypoint>}
REPRESENTATIONS_CATALOGUE: Dict[str, EntryPoint] = {}


@functools.lru_cache
def fill_catalogue() -> None:
    """Fills the catalogue with entrypoints.

    Only ever call this once for performance reasons.
    """
    entrypoints = [
        entrypoint
        for namespace, entrypoints in entry_points().items()
        if namespace.startswith(NAMESPACE)
        for entrypoint in entrypoints
    ]
    for entrypoint in entrypoints:
        put_representation_entrypoint(representation_entrypoint=entrypoint)


def put_representation_entrypoint(representation_entrypoint: EntryPoint) -> None:
    """Puts a representation entrypoint in the catalogue."""
    representation_name = camelcase_to_snakecase(representation_entrypoint.name)

    if representation_name in REPRESENTATIONS_CATALOGUE:
        if representation_entrypoint == REPRESENTATIONS_CATALOGUE[representation_name]:
            pass
        else:
            raise KeyError(f"Duplicate representation: {representation_name!r}")

    REPRESENTATIONS_CATALOGUE[representation_name] = representation_entrypoint


def get_representation_entrypoint(representation_name: str) -> EntryPoint:
    """Returns a specific representation entrypoint from the catalogue.

    The entrypoint is keyed only by name. This allows to decouple from
    eventual reshuffles of entrypoints across groups.
    """

    representation_entrypoint = REPRESENTATIONS_CATALOGUE.get(representation_name)
    if representation_entrypoint is None:
        msg = f"Representation {representation_name!r} is not available."
        similar = difflib.get_close_matches(
            representation_name,
            REPRESENTATIONS_CATALOGUE.keys(),
        )
        if similar:
            msg += f" You might be looking for one of these: {similar}"
        raise NotImplementedError(msg)
    return representation_entrypoint


def get_representation_cls(representation_name: str) -> RepresentationT:
    entrypoint = get_representation_entrypoint(representation_name)

    # Might raise (e.g. missing dependency extras)
    representation_cls = entrypoint.load()

    if not isinstance(representation_cls, Representation):
        raise NotImplementedError(
            f"Plugin {entrypoint.value} does not implement Representation protocol.",
        )

    return representation_cls  # type: ignore[return-value]


def list_representations() -> Dict[str, List[str]]:
    """List all available representations.

    The catalogue is returned as a view of representation names keyed by
    kind.
    """

    view: Dict[str, List[str]] = defaultdict(list)

    for (
        representation_name,
        representation_entrypoint,
    ) in REPRESENTATIONS_CATALOGUE.items():
        representation_kind = representation_entrypoint.group.split(".")[-1]
        view[representation_kind].append(representation_name)

    return dict(sorted(view.items()))


def register_representation(kind: str, name: str) -> Callable:
    """Registers a custom representation in the entrypoints catalogue.

    Examples:
        .. code-block:: python

           @register_representation(kind="custom", name="my_representation")
           class CustomRepresentation:
               ...
    """

    def wrapper(representation_cls: RepresentationT) -> RepresentationT:
        module = inspect.getmodule(representation_cls).__name__  # type: ignore
        value = f"{module}:{representation_cls.__name__}"
        group = NAMESPACE + kind
        entrypoint = EntryPoint(name=name, value=value, group=group)
        put_representation_entrypoint(representation_entrypoint=entrypoint)
        return representation_cls

    return wrapper
