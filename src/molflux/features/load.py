from typing import Any, Dict, Iterable

from molflux.features.catalogue import get_representation_cls
from molflux.features.parsers import Spec, dict_parser, yaml_parser
from molflux.features.representation import Representation, Representations
from molflux.features.typing import PathLike


def load_representation(name: str, **init_kwargs: Any) -> Representation:
    """Loads a `Representation` from the catalogue.

    Examples:
        >>> from molflux.features import load_representation
        >>> representation = load_representation("canonical_smiles")
    """

    # Fetch relevant representation class from the catalogue
    representation_cls = get_representation_cls(representation_name=name)

    # Instantiate representation object
    representation = representation_cls(**init_kwargs)

    return representation


def _load_from_spec(spec: Spec) -> Representation:
    """Loads a representation from a validated Spec."""

    # Build representation
    representation = load_representation(name=spec.name, **spec.config)

    # update state
    representation.update_state(**spec.presets)

    return representation


def load_from_dict(dictionary: Dict[str, Any]) -> Representation:
    """Loads a representation from a config dict."""

    # Validate dictionary
    spec = dict_parser(dictionary=dictionary)

    return _load_from_spec(spec=spec)


def load_from_dicts(dictionaries: Iterable[Dict[str, Any]]) -> Representations:
    """Loads a collection of Representations from an iterable of dicts."""

    representations = Representations()
    for dictionary in dictionaries:
        representation = load_from_dict(dictionary=dictionary)

        representations.add_representation(representation=representation)

    return representations


def load_from_yaml(path: PathLike) -> Representations:
    """Loads a collection of representations from a yaml config file."""

    specs = yaml_parser(path=path)

    representations = Representations()
    for spec in specs:
        representation = _load_from_spec(spec=spec)

        representations.add_representation(representation=representation)

    return representations
