import logging


def test_package_has_logger():
    """That the 'molflux.splits' root logger exists."""
    assert "molflux.splits" in logging.Logger.manager.loggerDict
