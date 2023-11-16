import logging


def test_package_has_logger():
    """That the 'molflux.datasets' root logger exists."""
    assert "molflux.datasets" in logging.Logger.manager.loggerDict
