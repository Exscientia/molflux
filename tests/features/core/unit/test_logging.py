import logging


def test_package_has_logger():
    """That the 'molflux.features' root logger exists."""
    assert "molflux.features" in logging.Logger.manager.loggerDict
