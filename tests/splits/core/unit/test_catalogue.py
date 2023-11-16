import pytest

import molflux.splits.catalogue
from molflux.splits.catalogue import (
    get_splitting_strategy_cls,
    list_splitting_strategies,
    register_splitting_strategy,
)
from molflux.splits.strategy import SplittingStrategy

catalogue_obj = molflux.splits.catalogue.SPLITTING_STRATEGIES_CATALOGUE


def test_catalogue_is_filled():
    """That the splitting strategies catalogue has been filled."""
    catalogue = list_splitting_strategies()
    assert len(catalogue)


def test_list_splitting_strategies_returns_sorted_names():
    """That strategy names in the catalogue are returned sorted."""
    catalogue = list_splitting_strategies()
    assert sorted(catalogue.keys()) == list(catalogue.keys())
    for strategies in catalogue.values():
        assert sorted(strategies) == strategies


def test_list_splitting_strategies_lists_names_of_strategies_in_catalogue():
    """That all strategies existing in the catalogue are returned by list_splitting_strategies."""
    view = list_splitting_strategies()
    names_in_view = [name for names in view.values() for name in names]
    assert sorted(names_in_view) == sorted(catalogue_obj.keys())


def test_register_strategy(monkeypatch):
    """That can register a new splitting strategy in the catalogue."""
    # set up a mock empty catalogue to leave real one untouched
    monkeypatch.setattr(molflux.splits.catalogue, "SPLITTING_STRATEGIES_CATALOGUE", {})

    assert "pytest_strategy" not in list_splitting_strategies()

    @register_splitting_strategy(kind="testing", name="pytest_strategy")
    class PytestStrategy:
        ...

    assert "testing" in list_splitting_strategies()
    assert "pytest_strategy" in list_splitting_strategies()["testing"]


def test_get_strategy_not_in_catalogue_raises_not_implemented_error():
    """That getting a non-existent strategy raises a NotImplementedError."""
    strategy_name = "non-existent-strategy"
    with pytest.raises(
        NotImplementedError,
        match=f"Splitting strategy {strategy_name!r} is not available",
    ):
        get_splitting_strategy_cls(strategy_name)


def test_get_strategy_cls_returns_splitting_strategy():
    """That getting a cls from the catalogue returns a class implementing the protocol."""
    strategy_name = "linear_split"
    splitting_strategy_cls = get_splitting_strategy_cls(strategy_name)
    assert isinstance(splitting_strategy_cls, SplittingStrategy)
