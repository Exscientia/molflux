import functools
from importlib.metadata import entry_points
from typing import Dict, List

import pytest

from molflux.modelzoo.catalogue import NAMESPACE
from molflux.modelzoo.errors import ExtrasDependencyImportError
from molflux.modelzoo.load import load_model


@functools.lru_cache
def all_modelzoo_entrypoints() -> Dict[str, List[str]]:
    """Cached for performance reasons."""
    return {
        namespace: [entrypoint.name for entrypoint in entrypoints]
        for namespace, entrypoints in entry_points().items()
        if namespace.startswith(NAMESPACE)
    }


_CATBOOST_MODELS = all_modelzoo_entrypoints()["molflux.modelzoo.plugins.catboost"]


@pytest.mark.parametrize("model_name", _CATBOOST_MODELS)
def test_missing_catboost_dependencies_raise_on_load(model_name):
    """That if dependencies are missing for catboost models, an
    error is raised at load time.

    The error should hint at how to install the missing dependencies.
    """

    with pytest.raises(
        ExtrasDependencyImportError,
        match=r"pip install \'molflux\[catboost\]\'",
    ):
        load_model(model_name)


_PYOD_MODELS = all_modelzoo_entrypoints()["molflux.modelzoo.plugins.pyod"]


@pytest.mark.parametrize("model_name", _PYOD_MODELS)
def test_missing_pyod_dependencies_raise_on_load(model_name):
    """That if dependencies are missing for pyod models, an
    error is raised at load time.

    The error should hint at how to install the missing dependencies.
    """

    with pytest.raises(
        ExtrasDependencyImportError,
        match=r"pip install \'molflux\[pyod\]\'",
    ):
        load_model(model_name)


_XGBOOST_MODELS = all_modelzoo_entrypoints()["molflux.modelzoo.plugins.xgboost"]


@pytest.mark.parametrize("model_name", _XGBOOST_MODELS)
def test_missing_xgboost_dependencies_raise_on_load(model_name):
    """That if dependencies are missing for xgboost models, an
    error is raised at load time.

    The error should hint at how to install the missing dependencies.
    """

    with pytest.raises(
        ExtrasDependencyImportError,
        match=r"pip install \'molflux\[xgboost\]\'",
    ):
        load_model(model_name)
