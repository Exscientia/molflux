import pytest

from molflux.splits.errors import DuplicateKeyError
from molflux.splits.load import (
    load_from_dict,
    load_from_dicts,
    load_from_yaml,
    load_splitting_strategy,
)
from molflux.splits.strategy import SplittingStrategy

representative_strategy_name = "linear_split"


def test_returns_spliting_strategy():
    """That loading a strategy returns an object of type SplittingStrategy."""
    name = representative_strategy_name
    strategy = load_splitting_strategy(name=name)
    assert isinstance(strategy, SplittingStrategy)


def test_forwards_init_kwargs_to_strategy_builder():
    """That keyword arguments get forwarded to the strategy initialiser."""
    name = representative_strategy_name
    strategy = load_splitting_strategy(name=name, tag="pytest-tag")
    assert strategy.tag == "pytest-tag"


def test_invalid_init_kwargs_raise_value_error():
    """That attempting initialisation with invalid keyword arguments raises.

    This is mostly to avoid users passing keyword arguments at initialisation
    time that should have been destined to the .split() method instead. If no
    errors are raised, this could silently lead to bugs as users would expect
    to have preset the splitting strategy with those kwargs.
    """
    name = representative_strategy_name
    with pytest.raises(ValueError, match=r"Unknown initialisation parameter\(s\)"):
        load_splitting_strategy(name=name, invalid_kwarg=True)


def test_from_dict_returns_splitting_strategy():
    """That loading from a dict returns a SplittingStrategy."""
    config = {
        "name": representative_strategy_name,
        "config": {},
        "presets": {},
    }
    strategy = load_from_dict(config)
    assert isinstance(strategy, SplittingStrategy)


def test_from_minimal_dict():
    """That can provide a config with only required fields."""
    config = {
        "name": representative_strategy_name,
    }
    assert load_from_dict(config)


def test_from_dict_forwards_init_kwargs_to_strategy_builder():
    """That config init keyword arguments get forwarded to the strategy initialiser."""
    config = {
        "name": representative_strategy_name,
        "config": {
            "tag": "pytest-tag",
        },
    }
    strategy = load_from_dict(config)
    assert strategy.tag == "pytest-tag"


def test_from_dict_forwards_splitting_kwargs_to_strategy_state():
    """That config splitting keyword arguments get stored in the strategy."""
    config = {
        "name": representative_strategy_name,
        "presets": {
            "train_fraction": 0.123,
        },
    }
    strategy = load_from_dict(config)
    assert strategy.state
    assert strategy.state.get("train_fraction") == 0.123


def test_dict_missing_required_fields_raises():
    """That cannot load a splitting strategy with a config missing required fields."""
    config = {"unknown_key": "value"}
    with pytest.raises(SyntaxError):
        load_from_dict(config)


def test_strategies_with_same_tag_in_collection_raises():
    """That adding several strategies with the same tag in a SplittingStrategies collection raises."""
    config = {
        "name": representative_strategy_name,
    }
    duplicate_configs = [config, config]
    with pytest.raises(DuplicateKeyError):
        load_from_dicts(duplicate_configs)


def test_load_from_dicts_returns_correct_number_of_metrics():
    """That loading from dicts returns a SplittingStrategies collection of the
    expected size."""
    name = representative_strategy_name
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
    strategies = load_from_dicts(configs)
    assert len(strategies) == 2


def test_from_yaml_returns_collection_of_splits(fixture_path_to_assets):
    path = fixture_path_to_assets / "config.yml"
    splitting_strategies = load_from_yaml(path=path)
    assert len(splitting_strategies) == 2
    assert "shuffle_split" in splitting_strategies
    assert "custom_k_fold" in splitting_strategies
    assert all(
        isinstance(strategy, SplittingStrategy)
        for strategy in splitting_strategies.values()
    )
    assert splitting_strategies["custom_k_fold"].state.get("n_splits") == 5
