import pytest

from molflux.features.parsers import Spec, dict_parser, yaml_parser


def test_dict_parser_raises_syntax_error_on_invalid_dict():
    """That a SyntaxError is raised on validation errors."""
    dictionary = {"wrong": "schema"}
    with pytest.warns(UserWarning), pytest.raises(SyntaxError):
        dict_parser(dictionary)


def test_dict_parser_parses_to_spec():
    """That dictionaries get parsed into Spec objects."""
    dictionary = {
        "name": "representation name",
        "config": {},
        "presets": {},
    }
    spec = dict_parser(dictionary)
    assert isinstance(spec, Spec)


def test_dict_parser_extras_cause_error():
    """That dictionaries with extra fields throw to warn about extra fields."""
    dictionary = {
        "name": "representation name",
        "config": {},
        "presets": {},
        "extra_field": "value",
    }
    with pytest.warns(UserWarning):
        dict_parser(dictionary)


def test_yaml_parser_parses_minimal_config(fixture_path_to_assets):
    path = fixture_path_to_assets / "minimal_config.yml"
    specs = yaml_parser(path)
    assert specs


def test_yaml_parser_parses_full_config(fixture_path_to_assets):
    path = fixture_path_to_assets / "config.yml"
    specs = yaml_parser(path)
    assert specs
