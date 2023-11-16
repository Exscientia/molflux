import difflib
import functools
import inspect
import logging
from collections import defaultdict
from importlib.metadata import EntryPoint, entry_points
from typing import Callable, Dict, List, Type

from molflux.modelzoo.naming import camelcase_to_snakecase
from molflux.modelzoo.protocols import Estimator

logger = logging.getLogger(__name__)

EstimatorT = Type[Estimator]

NAMESPACE = "molflux.modelzoo.plugins."

# This is where entrypoints will be registered {<name>: <Entrypoint>}
MODELS_CATALOGUE: Dict[str, EntryPoint] = {}


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
        put_model_entrypoint(model_entrypoint=entrypoint)


def put_model_entrypoint(model_entrypoint: EntryPoint) -> None:
    """Puts a representation entrypoint in the catalogue."""
    model_name = camelcase_to_snakecase(model_entrypoint.name)

    if model_name in MODELS_CATALOGUE:
        if MODELS_CATALOGUE[model_name] == model_entrypoint:
            pass
        else:
            raise KeyError(f"Duplicate model: {model_name!r}")

    MODELS_CATALOGUE[model_name] = model_entrypoint


def get_model_entrypoint(model_name: str) -> EntryPoint:
    """Returns a specific model entrypoint from the catalogue.

    The model is keyed only by name. This allows to decouple from
    eventual reshuffles of entrypoints across groups.
    """
    model_entrypoint = MODELS_CATALOGUE.get(model_name)
    if model_entrypoint is None:
        msg = f"Model {model_name!r} is not available."
        similar = difflib.get_close_matches(model_name, MODELS_CATALOGUE.keys())
        if similar:
            msg += f" You might be looking for one of these: {similar}"
        raise NotImplementedError(msg)
    return model_entrypoint


def get_model_cls(model_name: str) -> EstimatorT:
    entrypoint = get_model_entrypoint(model_name=model_name)

    # Might raise (e.g. missing dependency extras)
    model_cls = entrypoint.load()

    if not isinstance(model_cls, Estimator):
        raise NotImplementedError(
            f"Plugin {entrypoint.value} does not implement Model protocol.",
        )

    return model_cls  # type: ignore[return-value]


def list_models() -> Dict[str, List[str]]:
    """List all available models in the catalogue.

    The catalogue is returned as a view of model names keyed by kind.
    """

    view: Dict[str, List[str]] = defaultdict(list)

    for model_name, model_entrypoint in MODELS_CATALOGUE.items():
        model_kind = model_entrypoint.group.split(".")[-1]
        view[model_kind].append(model_name)

    return dict(sorted(view.items()))


def register_model(kind: str, name: str) -> Callable:
    """Registers a custom model in the entrypoints catalogue.

    Examples:
        .. code-block:: python

           @register_model(kind="custom", name="my_model")
           class CustomModel:
               ...
    """

    def wrapper(model_cls: EstimatorT) -> EstimatorT:
        module = inspect.getmodule(model_cls).__name__  # type: ignore
        value = f"{module}:{model_cls.__name__}"
        group = NAMESPACE + kind
        entrypoint = EntryPoint(name=name, value=value, group=group)
        put_model_entrypoint(model_entrypoint=entrypoint)
        return model_cls

    return wrapper
