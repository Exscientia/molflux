import contextlib
from types import FunctionType
from typing import Any, Callable, Iterator

from molflux.features.errors import FeaturisationError


def copyfunc(func: Callable) -> FunctionType:
    """Makes a complete copy of a given function.

    References:
        https://github.com/huggingface/datasets/blob/95193ae61e92aa537d0c65d37a1fd9d2393aae89/src/datasets/utils/py_utils.py#L681
    """
    result = FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )
    result.__kwdefaults__ = func.__kwdefaults__
    return result


@contextlib.contextmanager
def featurisation_error_harness(sample: Any) -> Iterator:
    """Creates a context within which a FeaturisationError is raised on any Exception.

    Args:
        sample: The sample to wrap.

    Raises:
        molflux.features.errors.FeaturisationError: If any Exception is raised within the context.

    Examples:

        >>> iterable = [1, 2, 3]
        >>> for x in iterable:  # doctest: +SKIP
        >>>     with featurisation_error_harness(x):  # doctest: +SKIP
        >>>         # do something that might raise
    """
    try:
        yield
    except Exception as e:
        raise FeaturisationError(sample=sample) from e
