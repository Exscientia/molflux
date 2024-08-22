import contextlib
from collections.abc import Callable, Iterator
from types import FunctionType
from typing import Any

from molflux.features.errors import FeaturisationError, InvalidNumberOfPositionalArgs


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


def assert_n_positional_args(*args: Any, expected_size: int) -> None:
    """Checks that the number of positional arguments given matches the expected size.

    Args:
        *args: The positional arguments whose number is checked
        expected_size: The expected number of positional arguments

    Raises:
            molflux.features.errors.InvalidNumberOfPositionalArgs: If the number of args is different from expected_size.

    """
    actual_size = len(args)
    if actual_size != expected_size:
        raise InvalidNumberOfPositionalArgs(expected_size, actual_size)
