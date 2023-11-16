import logging


def test_package_has_logger():
    """That the root logger exists."""
    assert "molflux.metrics" in logging.Logger.manager.loggerDict
