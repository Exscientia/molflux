import pytest
from openeye import oechem


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """

    if not oechem.OEChemIsLicensed("python"):
        pytest.exit("Could not find a valid openeye license!", 1)
