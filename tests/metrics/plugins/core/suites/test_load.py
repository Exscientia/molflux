import pytest

from molflux.metrics import load_suite


def test_raises_not_implemented_for_unknown_suite():
    """That a NotImplementedError is raised if attempting to load an unknown
    suite."""
    name = "unknown"
    with pytest.raises(NotImplementedError):
        load_suite(name=name)


def test_raised_error_for_unknown_suite_provides_close_matches():
    """That the error raised when attempting to load an unknown suite
    shows possible close matches to the user (if any)."""
    name = "regres"
    # This should suggest e.g. ["regression"]
    with pytest.raises(
        NotImplementedError,
        match="You might be looking for one of these",
    ):
        load_suite(name=name)
