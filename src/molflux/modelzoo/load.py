from typing import Any, Dict, Iterable

from molflux.modelzoo.catalogue import get_model_cls
from molflux.modelzoo.parsers import Spec, dict_parser, yaml_parser
from molflux.modelzoo.protocols import Model, Models
from molflux.modelzoo.typing import PathLike


def load_model(name: str, **init_kwargs: Any) -> Model:
    """Loads a `Model` from the catalogue."""

    # Fetch relevant model class from the catalogue
    model_cls = get_model_cls(model_name=name)

    # Instantiate model object
    model = model_cls(**init_kwargs)

    return model


def _load_from_spec(spec: Spec) -> Model:
    """Loads a model from a validated Spec."""

    # Build strategy
    model = load_model(name=spec.name, **spec.config)

    return model


def load_from_dict(dictionary: Dict[str, Any]) -> Model:
    """Loads a model from a config dict."""

    # Validate dictionary
    spec = dict_parser(dictionary=dictionary)

    return _load_from_spec(spec=spec)


def load_from_dicts(dictionaries: Iterable[Dict[str, Any]]) -> Models:
    """Loads models from an iterable of dicts."""

    models = (load_from_dict(dictionary) for dictionary in dictionaries)
    return {model.tag: model for model in models}


def load_from_yaml(path: PathLike) -> Models:
    """Loads models from a yaml config file."""

    specs = yaml_parser(path=path)

    models = (_load_from_spec(spec=spec) for spec in specs)
    return {model.tag: model for model in models}
