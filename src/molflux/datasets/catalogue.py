import functools
import inspect
import logging
from collections import defaultdict
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType
from typing import Callable, Dict, List

import datasets.builder
from molflux.datasets.naming import camelcase_to_snakecase

logger = logging.getLogger(__name__)

NAMESPACE = "molflux.datasets.plugins."

# This is where entrypoints will be registered {<name>: <Entrypoint>}
BUILDERS_CATALOGUE: Dict[str, EntryPoint] = {}


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
        put_builder_entrypoint(entrypoint=entrypoint)


def put_builder_entrypoint(entrypoint: EntryPoint) -> None:
    """Puts a dataset builder entrypoint in the catalogue."""
    builder_name = camelcase_to_snakecase(entrypoint.name)

    if builder_name in BUILDERS_CATALOGUE:
        if entrypoint == BUILDERS_CATALOGUE[builder_name]:
            pass
        else:
            raise KeyError(f"Duplicate dataset builder: {builder_name!r}")

    BUILDERS_CATALOGUE[builder_name] = entrypoint


def get_builder_entrypoint(name: str) -> EntryPoint:
    """Returns a specific dataset builder entrypoint from the catalogue"""
    entrypoint = BUILDERS_CATALOGUE.get(name)
    if entrypoint is None:
        raise NotImplementedError(f"Dataset builder {name!r} is not available.")
    return entrypoint


def get_dataset_builder_path(name: str) -> str:
    entrypoint = get_builder_entrypoint(name=name)

    # Might raise (e.g. missing dependency extras)
    module = entrypoint.load()

    if not isinstance(module, ModuleType):
        raise NotImplementedError(f"Plugin {entrypoint.value} is not a valid module.")

    path = module.__file__

    if path is None:
        raise RuntimeError(
            f"Could not resolve path for plugin module {entrypoint.value}",
        )

    return path


def list_datasets() -> Dict[str, List[str]]:
    """List all available datasets in the catalogue.

    The catalogue is returned as a view of dataset names keyed by kind.
    """

    view: Dict[str, List[str]] = defaultdict(list)

    for name, entrypoint in BUILDERS_CATALOGUE.items():
        kind = entrypoint.group.split(".")[-1]
        view[kind].append(name)

    return dict(sorted(view.items()))


def register_builder(kind: str, name: str) -> Callable:
    """Registers a custom dataset builder in the entrypoints catalogue.

    Examples:
        .. code-block:: python

           @register_dataset(kind="custom", name="my_dataset")
           class CustomDataset:
               ...
    """

    def wrapper(model_cls: datasets.DatasetBuilder) -> datasets.DatasetBuilder:
        module = inspect.getmodule(model_cls).__name__  # type: ignore
        value = f"{module}"
        group = NAMESPACE + kind
        entrypoint = EntryPoint(name=name, value=value, group=group)
        put_builder_entrypoint(entrypoint=entrypoint)
        return model_cls

    return wrapper
