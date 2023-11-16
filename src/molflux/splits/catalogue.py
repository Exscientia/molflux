import functools
import inspect
import logging
from collections import defaultdict
from importlib.metadata import EntryPoint, entry_points
from typing import Callable, Dict, List, Type

from molflux.splits.naming import camelcase_to_snakecase
from molflux.splits.strategy import SplittingStrategy

logger = logging.getLogger(__name__)

SplittingStrategyT = Type[SplittingStrategy]

NAMESPACE = "molflux.splits.plugins."

# This is where entrypoints will be registered {<name>: <Entrypoint>}
SPLITTING_STRATEGIES_CATALOGUE: Dict[str, EntryPoint] = {}


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
        put_splitting_strategy_entrypoint(splitting_strategy_entrypoint=entrypoint)


def put_splitting_strategy_entrypoint(
    splitting_strategy_entrypoint: EntryPoint,
) -> None:
    """Puts a splitting strategy entrypoint in the catalogue."""
    splitting_strategy_name = camelcase_to_snakecase(splitting_strategy_entrypoint.name)

    if splitting_strategy_name in SPLITTING_STRATEGIES_CATALOGUE:
        if (
            splitting_strategy_entrypoint
            == SPLITTING_STRATEGIES_CATALOGUE[splitting_strategy_name]
        ):
            pass
        else:
            raise KeyError(f"Duplicate splitting strategy: {splitting_strategy_name!r}")

    SPLITTING_STRATEGIES_CATALOGUE[
        splitting_strategy_name
    ] = splitting_strategy_entrypoint


def get_splitting_strategy_entrypoint(splitting_strategy_name: str) -> EntryPoint:
    """Returns a specific splitting strategy entrypoint from the catalogue.

    The entrypoint is keyed only by name. This allows to decouple from
    eventual reshuffles of entrypoints across groups.
    """
    splitting_strategy_entrypoint = SPLITTING_STRATEGIES_CATALOGUE.get(
        splitting_strategy_name,
    )
    if splitting_strategy_entrypoint is None:
        raise NotImplementedError(
            f"Splitting strategy {splitting_strategy_name!r} is not available.",
        )
    return splitting_strategy_entrypoint


def get_splitting_strategy_cls(splitting_strategy_name: str) -> SplittingStrategyT:
    entrypoint = get_splitting_strategy_entrypoint(splitting_strategy_name)

    # Might raise (e.g. missing dependency extras)
    splitting_strategy_cls = entrypoint.load()

    if not isinstance(splitting_strategy_cls, SplittingStrategy):
        raise NotImplementedError(
            f"Plugin {entrypoint.value} does not implement SplittingStrategy protocol.",
        )

    return splitting_strategy_cls  # type: ignore[return-value]


def list_splitting_strategies() -> Dict[str, List[str]]:
    """List all available splitting strategies.

    The catalogue is returned as a view of strategy names keyed by kind.
    """

    view: Dict[str, List[str]] = defaultdict(list)

    for (
        splitting_strategy_name,
        splitting_strategy_entrypoint,
    ) in SPLITTING_STRATEGIES_CATALOGUE.items():
        splitting_strategy_kind = splitting_strategy_entrypoint.group.split(".")[-1]
        view[splitting_strategy_kind].append(splitting_strategy_name)

    return dict(sorted(view.items()))


def register_splitting_strategy(kind: str, name: str) -> Callable:
    """Registers a custom splitting strategy in the entrypoints catalogue.

    Examples:
        .. code-block:: python

           @register_splitting_strategy(kind="custom", name="my_strategy")
           class CustomSplittingStrategy:
               ...
    """

    def wrapper(splitting_strategy_cls: SplittingStrategyT) -> SplittingStrategyT:
        module = inspect.getmodule(splitting_strategy_cls).__name__  # type: ignore
        value = f"{module}:{splitting_strategy_cls.__name__}"
        group = NAMESPACE + kind
        entrypoint = EntryPoint(name=name, value=value, group=group)
        put_splitting_strategy_entrypoint(splitting_strategy_entrypoint=entrypoint)
        return splitting_strategy_cls

    return wrapper
