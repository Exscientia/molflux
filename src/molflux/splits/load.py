from typing import Any, Dict, Iterable

from molflux.splits.catalogue import get_splitting_strategy_cls
from molflux.splits.errors import DuplicateKeyError
from molflux.splits.parsers import Spec, dict_parser, yaml_parser
from molflux.splits.strategy import SplittingStrategy
from molflux.splits.typing import PathLike

SplittingStrategies = Dict[str, SplittingStrategy]


def load_splitting_strategy(name: str, **init_kwargs: Any) -> SplittingStrategy:
    """Loads a splitting strategy from the catalogue."""

    # Fetch relevant splitting strategy cls from catalogue
    splitting_strategy_cls = get_splitting_strategy_cls(name)

    # Instantiate splitting strategy object
    strategy = splitting_strategy_cls(**init_kwargs)

    return strategy


def _load_from_spec(spec: Spec) -> SplittingStrategy:
    """Loads a splitting strategy from a validated Spec."""

    # Build strategy
    strategy = load_splitting_strategy(name=spec.name, **spec.config)

    # update state
    strategy.update_state(**spec.presets)

    return strategy


def load_from_dict(dictionary: Dict[str, Any]) -> SplittingStrategy:
    """Loads a splitting strategy from a config dict."""

    # Validate dictionary
    spec = dict_parser(dictionary=dictionary)

    return _load_from_spec(spec=spec)


def load_from_dicts(dictionaries: Iterable[Dict[str, Any]]) -> SplittingStrategies:
    """Loads splitting strategies from an iterable of dicts."""

    strategies = (load_from_dict(dictionary) for dictionary in dictionaries)
    collection = _make_strategy_collection(strategies)
    return collection


def load_from_yaml(path: PathLike) -> SplittingStrategies:
    """Loads splitting strategies from a yaml config file."""

    specs = yaml_parser(path=path)
    strategies = (_load_from_spec(spec=spec) for spec in specs)
    collection = _make_strategy_collection(strategies)
    return collection


def _make_strategy_collection(
    strategies: Iterable[SplittingStrategy],
) -> SplittingStrategies:
    """Collects SplittingStrategy-es into a SplittingStrategies collection."""
    collection: SplittingStrategies = {}
    for strategy in strategies:
        tag = strategy.tag
        if tag in collection:
            raise DuplicateKeyError(
                f"Splitting strategy with key {tag!r} has already been added. You can add a unique custom tag to one of the strategies to add it to the collection.",
            )
        collection[tag] = strategy
    return collection
