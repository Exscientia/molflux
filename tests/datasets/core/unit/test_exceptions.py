from molflux.datasets.exceptions import ExtrasDependencyImportError


def test_extras_dependency_import_error():
    """That the error can be instantiated"""
    base_exception = Exception()
    datasets_exception = ExtrasDependencyImportError("custom_type", base_exception)

    assert "molflux[custom_type]" in str(datasets_exception)
