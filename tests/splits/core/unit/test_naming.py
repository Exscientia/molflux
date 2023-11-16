from molflux.splits.naming import camelcase_to_snakecase


def test_camelcase_to_snakecase():
    """That camelcase to snakecase conversion returns as expected."""
    assert camelcase_to_snakecase("ClassNameABC") == "class_name_abc"


def test_camelcase_to_snakecase_invariant():
    """That camelcase to snakecase conversion is invariant on snakecase input."""
    assert camelcase_to_snakecase("class_name_abc") == "class_name_abc"


def test_camelcase_to_snakecase_starts_with_capital():
    """Camelcase to snakecase conversion on chained capitalised letters."""
    assert camelcase_to_snakecase("ABCClassName") == "abc_class_name"
