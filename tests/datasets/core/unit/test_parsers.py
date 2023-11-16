import pytest

from molflux.datasets.parsers import Spec, dict_parser, yaml_parser


def test_dict_parser_raises_syntax_error_on_invalid_dict():
    """That a SyntaxError is raised on validation errors."""
    dictionary = {"wrong": "schema"}
    with pytest.raises(SyntaxError):
        dict_parser(dictionary)


def test_dict_parser_parses_to_spec():
    """That dictionaries get parsed into Spec objects."""
    dictionary = {
        "name": "dataset name",
        "config": {},
    }
    spec = dict_parser(dictionary)
    assert isinstance(spec, Spec)


def test_yaml_parser_parses_minimal_config(fixture_path_to_assets):
    path = fixture_path_to_assets / "minimal_config.yml"
    specs = yaml_parser(path)
    assert specs


def test_yaml_parser_parses_full_config(fixture_path_to_assets):
    path = fixture_path_to_assets / "config.yml"
    specs = yaml_parser(path)
    assert specs


def test_yaml_parser_errors_on_no_file(fixture_path_to_assets):
    path = fixture_path_to_assets / "certainly_not_a_config.yml"

    with pytest.raises(FileNotFoundError):
        yaml_parser(path)


def test_yaml_parser_errors_on_bad_version(fixture_path_to_assets):
    path = fixture_path_to_assets / "config_with_bad_version.yml"

    with pytest.raises(NotImplementedError):
        yaml_parser(path)


def test_yaml_parser_errors_on_bad_format(fixture_path_to_assets):
    path = fixture_path_to_assets / "config_with_bad_format.yml"

    with pytest.raises(FileNotFoundError):
        yaml_parser(path)
