import pytest

from molflux.features.errors import DuplicateKeyError
from molflux.features.load import (
    load_from_dict,
    load_from_dicts,
    load_from_yaml,
    load_representation,
)
from molflux.features.representation import Representation, Representations

representative_representation_name = "character_count"


def test_returns_representation():
    """That loading a representation returns an object of type Representation."""
    name = representative_representation_name
    representation = load_representation(name=name)
    assert isinstance(representation, Representation)


def test_raises_not_implemented_for_unknown_representation():
    """That a NotImplementedError is raised if attempting to load an unknown
    representation."""
    name = "unknown"
    with pytest.raises(NotImplementedError):
        load_representation(name=name)


def test_raised_error_for_unknown_representation_provides_close_matches():
    """That the error raised when attempting to load an unknown representation
    shows possible close matches to the user (if any)."""
    name = "character_cont"
    # This should suggest e.g. ["character_count"]
    with pytest.raises(
        NotImplementedError,
        match="You might be looking for one of these",
    ):
        load_representation(name=name)


def test_forwards_init_kwargs_to_builder():
    """That keyword arguments get forwarded to the representation initialiser."""
    name = representative_representation_name
    representation = load_representation(name=name, tag="pytest-tag")
    assert representation.tag == "pytest-tag"


def test_from_dict_returns_representation():
    """That loading from a dict returns a Representation."""
    name = representative_representation_name
    config = {
        "name": name,
        "config": {},
        "presets": {},
    }
    representation = load_from_dict(config)
    assert isinstance(representation, Representation)


def test_from_minimal_dict():
    """That can provide a config with only required fields."""
    name = representative_representation_name
    config = {
        "name": name,
    }
    assert load_from_dict(config)


def test_from_dict_forwards_config_to_builder():
    """That config keyword arguments get forwarded to the initialiser."""
    name = representative_representation_name
    config = {
        "name": name,
        "config": {
            "tag": "pytest-tag",
        },
    }
    representation = load_from_dict(config)
    assert representation.tag == "pytest-tag"


def test_from_dict_forwards_presets_to_representation_state():
    """That presets keyword arguments get stored in the representation."""
    name = representative_representation_name
    config = {
        "name": name,
        "presets": {
            "without_hs": True,
        },
    }
    representation = load_from_dict(config)
    assert representation.state
    assert representation.state.get("without_hs") is True


def test_dict_missing_required_fields_raises():
    """That cannot load a representation with a config missing required fields."""
    config = {"unknown_key": "value"}
    with pytest.raises(SyntaxError):
        load_from_dict(config)


def test_representations_with_same_tag_in_collection_raises():
    """That adding several representations with the same tag in a Representations collection raises."""
    config = {
        "name": representative_representation_name,
    }
    duplicate_configs = [config, config]
    with pytest.raises(DuplicateKeyError):
        load_from_dicts(duplicate_configs)


def test_load_from_dicts_returns_correct_number_of_representations():
    """That loading from dicts returns a Representations collection of the
    expected size."""
    name = representative_representation_name
    config_one = {
        "name": name,
        "config": {
            "tag": "one",
        },
        "presets": {},
    }
    config_two = {
        "name": name,
        "config": {
            "tag": "two",
        },
        "presets": {},
    }
    configs = [config_one, config_two]
    representations = load_from_dicts(configs)
    assert len(representations) == 2


def test_from_yaml_returns_representations(fixture_path_to_assets):
    path = fixture_path_to_assets / "config.yml"
    representations = load_from_yaml(path=path)
    assert len(representations) == 2
    assert "character_count" in representations
    assert "custom_descriptor" in representations
    assert isinstance(representations, Representations)
    assert representations["custom_descriptor"].state.get("without_hs") is True
