from decimal import Decimal
from types import FunctionType
from typing import Callable, Sized, Tuple


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


def partition(dataset: Sized, *fractions: float) -> Tuple[int, ...]:
    """Returns the indices where to split the dataset to partition it according
    to the requested fractions.

    The output can then be passed to `np.split` to obtain n+1 sub-arrays.

    Examples:
        train_cutoff, validation_cutoff = partition(dataset, train_fraction, validation_fraction)
        train_split, validation_split, test_split = np.split(dataset, [train_cutoff, validation_cutoff])

    References:
        https://docs.python.org/3/tutorial/floatingpoint.html
        https://docs.python.org/3/library/decimal.html
    """
    n_samples = len(dataset)

    cutoffs = []
    partition_fraction = Decimal(0)
    for fraction in fractions:
        # Obtain an exact floating point sum
        partition_fraction = Decimal(str(partition_fraction)) + Decimal(str(fraction))
        cutoff = int(partition_fraction * n_samples)

        cutoffs.append(cutoff)

    return tuple(cutoffs)
